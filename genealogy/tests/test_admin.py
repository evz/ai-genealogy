import tempfile
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from genealogy.models import Document, DocumentPage


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class AdminBatchUploadTests(TestCase):
    """Test batch upload functionality in admin"""

    def setUp(self):
        """Create admin user and client"""
        self.user = User.objects.create_superuser(
            username="admin",
            email="admin@test.com",
            password="testpass123",
        )
        self.client = Client()
        self.client.login(username="admin", password="testpass123")

    def test_batch_upload_view_loads(self):
        """Batch upload view should load without errors"""
        url = reverse("admin:genealogy_document_batch_upload")
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Batch Upload Documents")
        self.assertContains(response, "Select multiple files")

    def test_page_number_extraction_and_ordering(self):
        """Business logic: Extract page numbers from filenames and order correctly"""
        # Create test files with page numbers in filenames (uploaded out of order)
        file3 = SimpleUploadedFile(
            "document_003.pdf", b"content3", content_type="application/pdf"
        )
        file1 = SimpleUploadedFile(
            "document_001.pdf", b"content1", content_type="application/pdf"
        )
        file2 = SimpleUploadedFile(
            "document_002.pdf", b"content2", content_type="application/pdf"
        )

        url = reverse("admin:genealogy_document_batch_upload")
        self.client.post(
            url,
            {
                "files": [file3, file1, file2],  # Uploaded out of order
                "language": "eng",
                "upload_mode": "single_document",
                "document_title": "Test Document",
            },
        )

        # Business logic test: Pages ordered by extracted page number, not upload order
        document = Document.objects.first()
        assert document is not None
        pages = list(document.pages.order_by("page_number"))

        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[0].original_filename, "document_001.pdf")
        self.assertEqual(pages[1].page_number, 2)
        self.assertEqual(pages[1].original_filename, "document_002.pdf")
        self.assertEqual(pages[2].page_number, 3)
        self.assertEqual(pages[2].original_filename, "document_003.pdf")

    def test_upload_mode_creates_correct_structure(self):
        """Business logic: Different upload modes create different structures"""
        file1 = SimpleUploadedFile(
            "family_history_001.pdf", b"content1", content_type="application/pdf"
        )
        file2 = SimpleUploadedFile(
            "family_history_002.pdf", b"content2", content_type="application/pdf"
        )

        url = reverse("admin:genealogy_document_batch_upload")

        # Test multiple documents mode
        self.client.post(
            url,
            {
                "files": [file1, file2],
                "language": "eng",
                "upload_mode": "multiple_documents",
            },
        )

        # Business logic: Should create separate documents
        self.assertEqual(Document.objects.count(), 2)
        self.assertEqual(DocumentPage.objects.count(), 2)

        # Each document should have 1 page
        for doc in Document.objects.all():
            self.assertEqual(doc.page_count, 1)

    def test_batch_upload_multiple_files(self):
        """Should create multiple documents from multiple files"""
        # Create test files
        file1 = SimpleUploadedFile(
            "doc1.pdf", b"content1", content_type="application/pdf"
        )
        file2 = SimpleUploadedFile("doc2.jpg", b"content2", content_type="image/jpeg")

        url = reverse("admin:genealogy_document_batch_upload")
        response = self.client.post(
            url,
            {
                "files": [file1, file2],
                "language": "nld",
                "upload_mode": "multiple_documents",
            },
        )

        # Should create 2 documents and 2 pages
        self.assertEqual(Document.objects.count(), 2)
        self.assertEqual(DocumentPage.objects.count(), 2)

        # Check documents have correct language
        for doc in Document.objects.all():
            self.assertEqual(doc.languages, "nld")

    def test_batch_upload_invalid_file_type(self):
        """Should skip unsupported file types"""
        # Create test files - one valid, one invalid
        valid_file = SimpleUploadedFile(
            "doc.pdf", b"content", content_type="application/pdf"
        )
        invalid_file = SimpleUploadedFile(
            "doc.txt", b"content", content_type="text/plain"
        )

        url = reverse("admin:genealogy_document_batch_upload")
        response = self.client.post(
            url,
            {
                "files": [valid_file, invalid_file],
                "language": "eng",
                "upload_mode": "multiple_documents",
            },
        )

        # Should only create 1 document (skip .txt file)
        self.assertEqual(Document.objects.count(), 1)
        self.assertEqual(DocumentPage.objects.count(), 1)

    def test_batch_upload_no_files(self):
        """Should show error when no files selected"""
        url = reverse("admin:genealogy_document_batch_upload")
        response = self.client.post(
            url,
            {
                "language": "eng",
            },
        )

        # Should not create any documents
        self.assertEqual(Document.objects.count(), 0)

    def test_document_title_from_filename(self):
        """Should generate clean titles from filenames"""
        test_cases = [
            ("family_tree_1950.pdf", "Family Tree 1950"),
            ("birth-certificate.jpg", "Birth Certificate"),
            ("MARRIAGE_RECORD.PDF", "Marriage Record"),
        ]

        for filename, expected_title in test_cases:
            with self.subTest(filename=filename):
                test_file = SimpleUploadedFile(
                    filename, b"content", content_type="application/pdf"
                )

                url = reverse("admin:genealogy_document_batch_upload")
                self.client.post(
                    url,
                    {
                        "files": [test_file],
                        "language": "eng",
                        "upload_mode": "multiple_documents",
                    },
                )

                document = Document.objects.latest("upload_date")
                self.assertEqual(document.title, expected_title)

                # Clean up for next test
                Document.objects.all().delete()
                DocumentPage.objects.all().delete()

    @patch("genealogy.admin.process_page_ocr.delay")
    def test_batch_upload_auto_starts_ocr(self, mock_task_delay):
        """Should automatically start OCR processing for uploaded files"""
        # Mock the Celery task
        mock_task_delay.return_value.id = "test-task-id"

        # Create test files
        file1 = SimpleUploadedFile(
            "page1.pdf", b"content1", content_type="application/pdf"
        )
        file2 = SimpleUploadedFile(
            "page2.pdf", b"content2", content_type="application/pdf"
        )

        url = reverse("admin:genealogy_document_batch_upload")
        response = self.client.post(
            url,
            {
                "files": [file1, file2],
                "language": "eng",
                "upload_mode": "single_document",
                "document_title": "Auto OCR Test",
            },
        )

        # Should create 1 document with 2 pages
        self.assertEqual(Document.objects.count(), 1)
        self.assertEqual(DocumentPage.objects.count(), 2)

        # Should have called OCR task for both pages
        self.assertEqual(mock_task_delay.call_count, 2)

        # Verify task was called with correct page IDs
        page_ids = {str(page.id) for page in DocumentPage.objects.all()}
        called_page_ids = {call[0][0] for call in mock_task_delay.call_args_list}
        self.assertEqual(page_ids, called_page_ids)

    @patch("genealogy.admin.process_page_ocr.delay")
    def test_document_page_ocr_action(self, mock_task_delay):
        """Should process OCR for selected document pages"""
        # Mock the Celery task
        mock_task_delay.return_value.id = "test-task-id"

        # Create test document and pages
        document = Document.objects.create(title="Test Doc", languages="eng")
        page1 = DocumentPage.objects.create(
            document=document,
            page_number=1,
            image_file=SimpleUploadedFile(
                "page1.pdf", b"content1", content_type="application/pdf"
            ),
            original_filename="page1.pdf",
        )
        page2 = DocumentPage.objects.create(
            document=document,
            page_number=2,
            image_file=SimpleUploadedFile(
                "page2.pdf", b"content2", content_type="application/pdf"
            ),
            original_filename="page2.pdf",
            ocr_completed=True,  # Already processed
        )

        # Simulate admin action on document pages
        from django.contrib.admin.sites import AdminSite

        from genealogy.admin import DocumentPageAdmin

        admin = DocumentPageAdmin(DocumentPage, AdminSite())
        request = self.client.request().wsgi_request
        queryset = DocumentPage.objects.filter(id__in=[page1.id, page2.id])

        admin.process_ocr(request, queryset)

        # Should only process unprocessed page
        mock_task_delay.assert_called_once_with(str(page1.id))

    @patch("genealogy.admin.process_page_ocr.delay")
    def test_document_page_reprocess_ocr_action(self, mock_task_delay):
        """Should reprocess OCR for selected document pages including completed ones"""
        # Mock the Celery task
        mock_task_delay.return_value.id = "test-task-id"

        # Create test document and page that's already processed
        document = Document.objects.create(title="Test Doc", languages="eng")
        page = DocumentPage.objects.create(
            document=document,
            page_number=1,
            image_file=SimpleUploadedFile(
                "page1.pdf", b"content1", content_type="application/pdf"
            ),
            original_filename="page1.pdf",
            ocr_completed=True,
            ocr_text="Existing text",
            ocr_confidence=95.0,
        )

        # Simulate admin reprocess action
        from django.contrib.admin.sites import AdminSite

        from genealogy.admin import DocumentPageAdmin

        admin = DocumentPageAdmin(DocumentPage, AdminSite())
        request = self.client.request().wsgi_request
        queryset = DocumentPage.objects.filter(id=page.id)

        admin.reprocess_ocr(request, queryset)

        # Should reset page and start reprocessing
        page.refresh_from_db()
        self.assertFalse(page.ocr_completed)
        self.assertEqual(page.ocr_text, "")
        self.assertIsNone(page.ocr_confidence)
        self.assertEqual(page.rotation_applied, 0.0)

        mock_task_delay.assert_called_once_with(str(page.id))
