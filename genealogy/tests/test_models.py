import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from genealogy.models import Document, DocumentPage


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class DocumentModelTests(TestCase):
    """Test Document model business logic"""

    def setUp(self):
        """Create test document"""
        self.document = Document.objects.create(
            title="Test Document",
            languages="eng+nld",
        )

    def test_can_process_ocr_with_pages(self):
        """Document with unprocessed pages should be ready for OCR"""
        test_file = SimpleUploadedFile(
            "page1.pdf", b"fake pdf content", content_type="application/pdf"
        )
        DocumentPage.objects.create(
            document=self.document,
            page_number=1,
            image_file=test_file,
        )

        self.assertTrue(self.document.can_process_ocr())

    def test_cannot_process_ocr_without_pages(self):
        """Document without pages should not be ready for OCR"""
        self.assertFalse(self.document.can_process_ocr())

    def test_cannot_process_ocr_if_already_completed(self):
        """Document with completed OCR should not be processable again"""
        self.document.ocr_completed = True
        self.document.save()
        self.assertFalse(self.document.can_process_ocr())

    def test_update_ocr_status_all_completed(self):
        """update_ocr_status should mark complete if all pages done"""
        test_file1 = SimpleUploadedFile(
            "page1.pdf", b"fake pdf content", content_type="application/pdf"
        )
        test_file2 = SimpleUploadedFile(
            "page2.pdf", b"fake pdf content", content_type="application/pdf"
        )

        DocumentPage.objects.create(
            document=self.document,
            page_number=1,
            image_file=test_file1,
            ocr_completed=True,
        )
        DocumentPage.objects.create(
            document=self.document,
            page_number=2,
            image_file=test_file2,
            ocr_completed=True,
        )

        self.document.update_ocr_status()
        self.assertTrue(self.document.ocr_completed)

    def test_ocr_progress_with_pages(self):
        """ocr_progress should return progress dict"""
        test_file1 = SimpleUploadedFile(
            "page1.pdf", b"fake pdf content", content_type="application/pdf"
        )
        test_file2 = SimpleUploadedFile(
            "page2.pdf", b"fake pdf content", content_type="application/pdf"
        )
        test_file3 = SimpleUploadedFile(
            "page3.pdf", b"fake pdf content", content_type="application/pdf"
        )

        DocumentPage.objects.create(
            document=self.document,
            page_number=1,
            image_file=test_file1,
            ocr_completed=True,
        )
        DocumentPage.objects.create(
            document=self.document,
            page_number=2,
            image_file=test_file2,
            ocr_completed=True,
        )
        DocumentPage.objects.create(
            document=self.document,
            page_number=3,
            image_file=test_file3,
            ocr_completed=False,
        )

        progress = self.document.ocr_progress
        self.assertEqual(progress["completed"], 2)
        self.assertEqual(progress["total"], 3)
        self.assertAlmostEqual(progress["percentage"], 66.67, places=1)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class DocumentPageModelTests(TestCase):
    """Test DocumentPage model business logic"""

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

    def test_can_process_ocr_with_file(self):
        """Page with file should be ready for OCR"""
        self.assertTrue(self.page.can_process_ocr())

    def test_cannot_process_ocr_without_file(self):
        """Page without file should not be ready for OCR"""
        page_no_file = DocumentPage.objects.create(
            document=self.document,
            page_number=2,
        )
        self.assertFalse(page_no_file.can_process_ocr())

    def test_validate_for_ocr_success(self):
        """validate_for_ocr should pass for valid page"""
        self.page.validate_for_ocr()  # Should not raise exception

    def test_validate_for_ocr_no_file(self):
        """validate_for_ocr should raise ValueError without file"""
        page_no_file = DocumentPage.objects.create(
            document=self.document,
            page_number=2,
        )
        with self.assertRaises(ValueError) as cm:
            page_no_file.validate_for_ocr()
        self.assertIn("No image file attached", str(cm.exception))

    def test_validate_for_ocr_already_completed(self):
        """validate_for_ocr should raise ValueError if already completed"""
        self.page.ocr_completed = True
        self.page.save()
        with self.assertRaises(ValueError) as cm:
            self.page.validate_for_ocr()
        self.assertIn("OCR already completed", str(cm.exception))
