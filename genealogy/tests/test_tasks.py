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

    @patch("genealogy.tasks.ocr.process_page_ocr.delay")
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
