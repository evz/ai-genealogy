import tempfile
import uuid
from unittest.mock import Mock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from genealogy.models import Document, DocumentPage
from genealogy.tasks import process_document_ocr, process_page_ocr


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class OCRTaskTests(TestCase):
    """Test OCR Celery tasks - mock external dependencies"""

    def setUp(self):
        """Create test document and page"""
        self.document = Document.objects.create(
            title="Test Document",
            languages="eng+nld",
        )
        self.test_file = SimpleUploadedFile(
            "test.pdf",
            b"fake pdf content",
            content_type="application/pdf",
        )
        self.page = DocumentPage.objects.create(
            document=self.document,
            page_number=1,
            image_file=self.test_file,
        )

    @patch("genealogy.tasks.SmallModelProcessor")
    @patch("genealogy.tasks.DeepSeekOCRProcessor")
    @patch("genealogy.tasks.RotationDetector")
    @patch("genealogy.tasks.Image.open")
    @patch("genealogy.tasks.os.path.exists")
    def test_process_page_ocr_success(
        self, mock_exists, mock_image_open, mock_rotation_class, mock_ocr_class, mock_small_model_class
    ):
        """process_page_ocr should complete successfully and update page"""
        # Mock file exists
        mock_exists.return_value = True

        # Mock image
        mock_image = Mock()
        mock_image_open.return_value = mock_image

        # Mock rotation detector
        mock_rotation_detector = Mock()
        mock_rotation_detector.detect_and_correct.return_value = (mock_image, 0.5)
        mock_rotation_class.return_value = mock_rotation_detector

        # Mock OCR processor (DeepSeek returns only text, no confidence)
        mock_ocr_processor = Mock()
        mock_ocr_processor.process_page.return_value = "Extracted text content"
        mock_ocr_class.return_value = mock_ocr_processor

        # Mock small model processor
        mock_small_model = Mock()
        mock_small_model.clean_ocr_genealogy_ids.return_value = ["Extracted text content"]
        mock_small_model_class.return_value = mock_small_model

        # Run task
        result = process_page_ocr(str(self.page.id))

        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "Extracted text content")
        self.assertNotIn("confidence", result)  # DeepSeek doesn't provide confidence
        self.assertEqual(result["rotation_applied"], 0.5)

        # Check page was updated
        self.page.refresh_from_db()
        self.assertTrue(self.page.ocr_completed)
        self.assertEqual(self.page.ocr_text, "Extracted text content")
        self.assertIsNone(self.page.ocr_confidence)  # DeepSeek doesn't provide confidence
        self.assertEqual(self.page.rotation_applied, 0.5)

        # Check document status was updated
        self.document.refresh_from_db()
        self.assertTrue(self.document.ocr_completed)

    def test_process_page_ocr_invalid_uuid(self):
        """process_page_ocr should handle invalid UUID format"""
        result = process_page_ocr("invalid-uuid-format")

        self.assertFalse(result["success"])
        self.assertIn("Invalid UUID format", result["error"])

    def test_process_page_ocr_nonexistent_page(self):
        """process_page_ocr should handle valid UUID that doesn't exist"""
        fake_uuid = str(uuid.uuid4())
        result = process_page_ocr(fake_uuid)

        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    def test_process_page_ocr_already_completed(self):
        """process_page_ocr should skip already completed pages"""
        # Mark page as completed
        self.page.ocr_completed = True
        self.page.ocr_text = "Existing text"
        self.page.ocr_confidence = None
        self.page.save()

        result = process_page_ocr(str(self.page.id))

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Already processed")
        self.assertEqual(result["text"], "Existing text")
        self.assertEqual(result["confidence"], None)

    @patch("genealogy.tasks.os.path.exists")
    def test_process_page_ocr_file_not_found(self, mock_exists):
        """process_page_ocr should handle missing image files"""
        # Mock file doesn't exist
        mock_exists.return_value = False

        result = process_page_ocr(str(self.page.id))

        self.assertFalse(result["success"])
        self.assertIn("Image file not found", result["error"])

    @patch("genealogy.tasks.RotationDetector")
    @patch("genealogy.tasks.Image.open")
    @patch("genealogy.tasks.os.path.exists")
    def test_process_page_ocr_processing_failure(self, mock_exists, mock_image_open, mock_rotation_class):
        """process_page_ocr should handle OCR processing failures"""
        # Mock file exists
        mock_exists.return_value = True

        # Mock image
        mock_image = Mock()
        mock_image_open.return_value = mock_image

        # Mock rotation detector to raise exception
        mock_rotation_detector = Mock()
        mock_rotation_detector.detect_and_correct.side_effect = Exception("OCR processing failed")
        mock_rotation_class.return_value = mock_rotation_detector

        result = process_page_ocr(str(self.page.id))

        self.assertFalse(result["success"])
        self.assertIn("OCR processing failed", result["error"])

        # Page should not be marked as completed
        self.page.refresh_from_db()
        self.assertFalse(self.page.ocr_completed)

    @patch("genealogy.tasks.process_page_ocr.delay")
    def test_process_document_ocr_success(self, mock_page_task):
        """process_document_ocr should start tasks for all unprocessed pages"""
        # Add another unprocessed page
        test_file2 = SimpleUploadedFile("test2.pdf", b"fake pdf content", content_type="application/pdf")
        DocumentPage.objects.create(
            document=self.document,
            page_number=2,
            image_file=test_file2,
        )

        # Mock task delay
        mock_task = Mock()
        mock_task.id = "task-123"
        mock_page_task.return_value = mock_task

        result = process_document_ocr(str(self.document.id))

        self.assertTrue(result["success"])
        self.assertEqual(result["pages_processed"], 2)
        self.assertIn("task_ids", result)

        # Should have called page task twice
        self.assertEqual(mock_page_task.call_count, 2)

    def test_process_document_ocr_no_pages(self):
        """process_document_ocr should handle documents with no unprocessed pages"""
        # Mark the page as completed
        self.page.ocr_completed = True
        self.page.save()

        result = process_document_ocr(str(self.document.id))

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "No pages to process")
        self.assertEqual(result["pages_processed"], 0)

    def test_process_document_ocr_invalid_uuid(self):
        """process_document_ocr should handle invalid UUID format"""
        result = process_document_ocr("invalid-uuid-format")

        self.assertFalse(result["success"])
        self.assertIn("Invalid UUID format", result["error"])

    def test_process_document_ocr_nonexistent_document(self):
        """process_document_ocr should handle valid UUID that doesn't exist"""
        fake_uuid = str(uuid.uuid4())
        result = process_document_ocr(fake_uuid)

        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])
