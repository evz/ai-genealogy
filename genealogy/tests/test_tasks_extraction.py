"""Tests for extraction tasks"""
import uuid
from unittest.mock import Mock, patch

from django.test import TestCase

from genealogy.models import BookSection, Document, TextChunk
from genealogy.tasks.extraction import extract_entities_from_chunks


class ExtractionTaskTests(TestCase):
    """Test extraction Celery task"""

    def setUp(self):
        """Create test document with chunks"""
        self.document = Document.objects.create(
            title="Test Genealogy",
            languages="nld",
        )

        self.section = BookSection.objects.create(
            document=self.document,
            title="Descendant Genealogy",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=2,
        )

        # Create chunks that need extraction
        self.chunk1 = TextChunk.objects.create(
            document=self.document,
            chunk_type="individual_entry",
            text_content="a. Pieter Jansen, geboren 1850.",
            sequence_number=1,
            start_page=1,
            end_page=1,
            entities_extracted=False,
        )

        self.chunk2 = TextChunk.objects.create(
            document=self.document,
            chunk_type="individual_entry",
            text_content="b. Maria Jansen, geboren 1852.",
            sequence_number=2,
            start_page=2,
            end_page=2,
            entities_extracted=False,
        )

    @patch("genealogy.tasks.extraction.OllamaClient")
    @patch("genealogy.tasks.extraction.ExtractionService")
    def test_extracts_from_document(self, MockService, MockOllama):
        """Should extract entities from document chunks"""
        # Setup mocks
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        MockOllama.return_value = mock_ollama

        mock_service = Mock()
        mock_service.extract_from_chunks_in_section.return_value = {
            'processed': 2,
            'failed': 0,
        }
        MockService.return_value = mock_service

        result = extract_entities_from_chunks(str(self.document.id))

        # Verify success
        self.assertTrue(result['success'])
        self.assertEqual(result['chunks_processed'], 2)
        self.assertEqual(result['chunks_failed'], 0)
        self.assertIn('Descendant Genealogy', result['sections_processed'])

        # Verify document was updated
        self.document.refresh_from_db()
        self.assertTrue(self.document.extraction_completed)

        # Verify extraction service was called
        mock_service.extract_from_chunks_in_section.assert_called_once()

    @patch("genealogy.tasks.extraction.OllamaClient")
    def test_no_book_sections(self, MockOllama):
        """Should return error if no book sections"""
        self.section.delete()

        result = extract_entities_from_chunks(str(self.document.id))

        self.assertFalse(result['success'])
        self.assertIn("No BookSections defined", result['error'])

    @patch("genealogy.tasks.extraction.OllamaClient")
    def test_ollama_not_available(self, MockOllama):
        """Should fail if Ollama is not available"""
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = False
        MockOllama.return_value = mock_ollama

        result = extract_entities_from_chunks(str(self.document.id))

        self.assertFalse(result['success'])
        self.assertIn("Ollama server not available", result['error'])

    @patch("genealogy.tasks.extraction.get_strategy")
    @patch("genealogy.tasks.extraction.OllamaClient")
    def test_unknown_section_type(self, MockOllama, mock_get_strategy):
        """Should skip sections with unknown strategy"""
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        MockOllama.return_value = mock_ollama

        mock_get_strategy.side_effect = KeyError("Unknown type")

        result = extract_entities_from_chunks(str(self.document.id))

        # Should still succeed but skip the section
        self.assertTrue(result['success'])
        self.assertEqual(result['chunks_processed'], 0)

    @patch("genealogy.tasks.extraction.OllamaClient")
    @patch("genealogy.tasks.extraction.ExtractionService")
    def test_no_unprocessed_chunks(self, MockService, MockOllama):
        """Should handle sections with no unprocessed chunks"""
        # Mark all chunks as extracted
        self.chunk1.entities_extracted = True
        self.chunk1.save()
        self.chunk2.entities_extracted = True
        self.chunk2.save()

        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        MockOllama.return_value = mock_ollama

        result = extract_entities_from_chunks(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['chunks_processed'], 0)
        self.assertEqual(result['sections_processed']['Descendant Genealogy']['processed'], 0)

    @patch("genealogy.tasks.extraction.OllamaClient")
    @patch("genealogy.tasks.extraction.ExtractionService")
    def test_uses_document_model(self, MockService, MockOllama):
        """Should use document's configured model if set"""
        self.document.llm_model_used = "custom-model"
        self.document.save()

        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        MockOllama.return_value = mock_ollama

        mock_service = Mock()
        mock_service.extract_from_chunks_in_section.return_value = {
            'processed': 2,
            'failed': 0,
        }
        MockService.return_value = mock_service

        result = extract_entities_from_chunks(str(self.document.id))

        # Verify custom model was used
        call_kwargs = mock_service.extract_from_chunks_in_section.call_args[1]
        self.assertEqual(call_kwargs['model'], "custom-model")

    @patch("genealogy.tasks.extraction.OllamaClient")
    @patch("genealogy.tasks.extraction.ExtractionService")
    @patch("genealogy.tasks.extraction.get_default_models")
    def test_uses_default_model_if_not_configured(self, mock_get_defaults, MockService, MockOllama):
        """Should use default model if document has none configured"""
        mock_get_defaults.return_value = {
            "llm_model": "default-model",
            "embedding_model": "default-embedding-model"
        }

        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        MockOllama.return_value = mock_ollama

        mock_service = Mock()
        mock_service.extract_from_chunks_in_section.return_value = {
            'processed': 2,
            'failed': 0,
        }
        MockService.return_value = mock_service

        result = extract_entities_from_chunks(str(self.document.id))

        # Verify default model was used
        call_kwargs = mock_service.extract_from_chunks_in_section.call_args[1]
        self.assertEqual(call_kwargs['model'], "default-model")

    @patch("genealogy.tasks.extraction.OllamaClient")
    @patch("genealogy.tasks.extraction.ExtractionService")
    def test_handles_partial_failures(self, MockService, MockOllama):
        """Should handle sections with partial failures"""
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        MockOllama.return_value = mock_ollama

        mock_service = Mock()
        mock_service.extract_from_chunks_in_section.return_value = {
            'processed': 1,
            'failed': 1,
        }
        MockService.return_value = mock_service

        result = extract_entities_from_chunks(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['chunks_processed'], 1)
        self.assertEqual(result['chunks_failed'], 1)

    @patch("genealogy.tasks.extraction.OllamaClient")
    @patch("genealogy.tasks.extraction.ExtractionService")
    def test_processes_multiple_sections(self, MockService, MockOllama):
        """Should process multiple sections"""
        # Add another section
        section2 = BookSection.objects.create(
            document=self.document,
            title="Section 2",
            section_type="DESCENDANT_GENEALOGY",
            start_page=3,
            end_page=4,
        )

        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        MockOllama.return_value = mock_ollama

        mock_service = Mock()
        mock_service.extract_from_chunks_in_section.return_value = {
            'processed': 1,
            'failed': 0,
        }
        MockService.return_value = mock_service

        result = extract_entities_from_chunks(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(len(result['sections_processed']), 2)
        self.assertIn('Descendant Genealogy', result['sections_processed'])
        self.assertIn('Section 2', result['sections_processed'])

    def test_invalid_uuid_format(self):
        """Should handle invalid UUID format"""
        result = extract_entities_from_chunks("not-a-uuid")

        self.assertFalse(result['success'])
        self.assertIn("Invalid UUID format", result['error'])

    def test_document_not_found(self):
        """Should handle non-existent document"""
        fake_uuid = str(uuid.uuid4())
        result = extract_entities_from_chunks(fake_uuid)

        self.assertFalse(result['success'])
        self.assertIn("not found", result['error'])

    @patch("genealogy.tasks.extraction.OllamaClient")
    def test_general_exception_handling(self, MockOllama):
        """Should handle unexpected exceptions"""
        MockOllama.side_effect = RuntimeError("Unexpected error")

        result = extract_entities_from_chunks(str(self.document.id))

        self.assertFalse(result['success'])
        self.assertIn("Unexpected error", result['error'])

    @patch("genealogy.tasks.extraction.OllamaClient")
    @patch("genealogy.tasks.extraction.ExtractionService")
    def test_filters_chunks_by_page_range(self, MockService, MockOllama):
        """Should only process chunks within section page range"""
        # Create chunk outside page range
        out_of_range_chunk = TextChunk.objects.create(
            document=self.document,
            chunk_type="individual_entry",
            text_content="Outside range",
            sequence_number=3,
            start_page=10,  # Outside section range (1-2)
            end_page=10,
            entities_extracted=False,
        )

        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        MockOllama.return_value = mock_ollama

        mock_service = Mock()
        mock_service.extract_from_chunks_in_section.return_value = {
            'processed': 2,
            'failed': 0,
        }
        MockService.return_value = mock_service

        result = extract_entities_from_chunks(str(self.document.id))

        # Should only process 2 chunks (not the out-of-range one)
        call_args = mock_service.extract_from_chunks_in_section.call_args
        chunks_passed = call_args[1]['chunks']
        self.assertEqual(chunks_passed.count(), 2)
