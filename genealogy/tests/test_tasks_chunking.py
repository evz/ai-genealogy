"""Tests for chunking tasks"""
import uuid
from unittest.mock import Mock, patch

from django.test import TestCase

from genealogy.chunking.models import ChunkType
from genealogy.models import BookSection, Document, DocumentPage, TextChunk
from genealogy.tasks.chunking import create_document_chunks

from .helpers import create_ocr_text


class ChunkingTaskTests(TestCase):
    """Test chunking Celery task"""

    def setUp(self):
        """Create test document with pages"""
        self.document = Document.objects.create(
            title="Test Genealogy",
            languages="nld",
            ocr_completed=True,
        )

        # Create OCR text for pages
        self.page1_text = create_ocr_text([
            {'content': 'TWEEDE GENERATIE', 'element_type': 'sub_title'},
            {'content': 'II.1. Kinderen van Jan Jansen:', 'element_type': 'sub_title'},
        ])

        self.page2_text = create_ocr_text([
            {'content': 'a. Pieter Jansen, geboren 1850.', 'element_type': 'text'},
        ])

        self.page1 = DocumentPage.objects.create(
            document=self.document,
            page_number=1,
            ocr_completed=True,
            ocr_text=self.page1_text,
        )

        self.page2 = DocumentPage.objects.create(
            document=self.document,
            page_number=2,
            ocr_completed=True,
            ocr_text=self.page2_text,
        )

    def test_chunks_document_with_book_sections(self):
        """Should chunk document successfully when book sections are defined"""
        # Create a book section
        section = BookSection.objects.create(
            document=self.document,
            title="Descendant Genealogy",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=2,
        )

        result = create_document_chunks(str(self.document.id))

        # Verify result structure
        self.assertTrue(result['success'])
        self.assertEqual(result['chunks_created'], 3)  # generation header + family group + individual entry
        self.assertEqual(result['pages_processed'], 2)
        self.assertIn('Descendant Genealogy', result['sections_processed'])
        self.assertEqual(result['sections_processed']['Descendant Genealogy']['status'], 'success')
        self.assertEqual(result['sections_processed']['Descendant Genealogy']['chunks'], 3)

        # Verify actual chunks were created in DB
        chunks = TextChunk.objects.filter(document=self.document).order_by('sequence_number')
        self.assertEqual(chunks.count(), 3)

        # Verify chunk types match enum values
        self.assertEqual(chunks[0].chunk_type, "generation_header")
        self.assertEqual(chunks[1].chunk_type, "family_group_header")
        self.assertEqual(chunks[2].chunk_type, "individual_entry")

    def test_ocr_not_completed(self):
        """Should return error if OCR is not completed"""
        self.document.ocr_completed = False
        self.document.save()

        BookSection.objects.create(
            document=self.document,
            title="Test Section",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=2,
        )

        result = create_document_chunks(str(self.document.id))

        self.assertFalse(result['success'])
        self.assertIn("OCR must be completed", result['error'])

    def test_no_book_sections(self):
        """Should return error if no book sections are defined"""
        result = create_document_chunks(str(self.document.id))

        self.assertFalse(result['success'])
        self.assertIn("No BookSections defined", result['error'])

    def test_section_skipped_by_strategy(self):
        """Should skip sections that strategy doesn't want to process"""
        # Create a SKIP section type
        section = BookSection.objects.create(
            document=self.document,
            title="Front Matter",
            section_type="SKIP",
            start_page=1,
            end_page=2,
        )

        result = create_document_chunks(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['chunks_created'], 0)
        self.assertIn('Front Matter', result['sections_processed'])
        self.assertEqual(result['sections_processed']['Front Matter']['status'], 'skipped')

    def test_no_ocr_completed_pages_in_section(self):
        """Should handle sections where no pages have completed OCR"""
        # Mark pages as not OCR completed
        self.page1.ocr_completed = False
        self.page1.save()
        self.page2.ocr_completed = False
        self.page2.save()

        section = BookSection.objects.create(
            document=self.document,
            title="Test Section",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=2,
        )

        result = create_document_chunks(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['chunks_created'], 0)
        self.assertIn('Test Section', result['sections_processed'])
        self.assertEqual(result['sections_processed']['Test Section']['status'], 'no_pages')

    def test_skips_pages_with_empty_ocr_text(self):
        """Should skip pages that have no OCR text"""
        # Set one page to have empty OCR text
        self.page2.ocr_text = ""
        self.page2.save()

        section = BookSection.objects.create(
            document=self.document,
            title="Test Section",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=2,
        )

        result = create_document_chunks(str(self.document.id))

        self.assertTrue(result['success'])
        # Should only process 1 page (page1), creating 2 chunks (generation + family group)
        self.assertEqual(result['pages_processed'], 1)
        self.assertEqual(result['chunks_created'], 2)

    def test_section_chunking_failure(self):
        """Should handle failures during section chunking"""
        section = BookSection.objects.create(
            document=self.document,
            title="Test Section",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=2,
        )

        # Mock the chunking service to return failure
        with patch('genealogy.tasks.chunking.ChunkingService') as MockService:
            mock_service = Mock()
            mock_service.should_process_section.return_value = True
            mock_service.chunk_section.return_value = {
                'success': False,
                'error': 'Chunking strategy failed'
            }
            MockService.return_value = mock_service

            result = create_document_chunks(str(self.document.id))

        self.assertTrue(result['success'])  # Task succeeds but section fails
        self.assertEqual(result['chunks_created'], 0)
        self.assertEqual(result['sections_processed']['Test Section']['status'], 'error')
        self.assertIn('Chunking strategy failed', result['sections_processed']['Test Section']['error'])

    def test_invalid_uuid_format(self):
        """Should handle invalid UUID format"""
        result = create_document_chunks("not-a-uuid")

        self.assertFalse(result['success'])
        self.assertIn("Invalid UUID format", result['error'])

    def test_document_not_found(self):
        """Should handle non-existent document"""
        fake_uuid = str(uuid.uuid4())
        result = create_document_chunks(fake_uuid)

        self.assertFalse(result['success'])
        self.assertIn("not found", result['error'])

    def test_clears_existing_chunks(self):
        """Should delete existing chunks before creating new ones"""
        section = BookSection.objects.create(
            document=self.document,
            title="Test Section",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=2,
        )

        # Create some existing chunks
        old_chunk = TextChunk.objects.create(
            document=self.document,
            chunk_type="generation_header",
            text_content="Old chunk",
            sequence_number=1,
            start_page=1,
            end_page=1,
        )

        self.assertEqual(TextChunk.objects.filter(document=self.document).count(), 1)

        result = create_document_chunks(str(self.document.id))

        self.assertTrue(result['success'])

        # Old chunks should be deleted, new chunks created
        chunks = TextChunk.objects.filter(document=self.document)
        self.assertEqual(chunks.count(), 3)  # 3 new chunks
        self.assertFalse(chunks.filter(text_content="Old chunk").exists())

    def test_processes_multiple_sections(self):
        """Should process multiple book sections in order"""
        section1 = BookSection.objects.create(
            document=self.document,
            title="Section 1",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=1,
        )

        section2 = BookSection.objects.create(
            document=self.document,
            title="Section 2",
            section_type="DESCENDANT_GENEALOGY",
            start_page=2,
            end_page=2,
        )

        result = create_document_chunks(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertIn('Section 1', result['sections_processed'])
        self.assertIn('Section 2', result['sections_processed'])
        self.assertEqual(len(result['sections_processed']), 2)

        # Section 1 should have 2 chunks (generation + family group)
        self.assertEqual(result['sections_processed']['Section 1']['chunks'], 2)
        # Section 2 should have 1 chunk (individual entry)
        self.assertEqual(result['sections_processed']['Section 2']['chunks'], 1)
        # Total: 3 chunks
        self.assertEqual(result['chunks_created'], 3)

    def test_general_exception_handling(self):
        """Should handle unexpected exceptions gracefully"""
        section = BookSection.objects.create(
            document=self.document,
            title="Test Section",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=2,
        )

        # Mock the service to raise an unexpected exception
        with patch('genealogy.tasks.chunking.ChunkingService') as MockService:
            MockService.side_effect = RuntimeError("Unexpected error")

            result = create_document_chunks(str(self.document.id))

        self.assertFalse(result['success'])
        self.assertIn("Unexpected error", result['error'])
