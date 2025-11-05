"""Tests for genealogy.services module

This module tests the pure business logic services (ChunkingService and ExtractionService)
that orchestrate the chunking and extraction processes.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from genealogy.services import ChunkingService, ExtractionService
from genealogy.chunking.models import ChunkType


@pytest.mark.unit
class TestChunkingService:
    """Test ChunkingService orchestration logic"""

    def test_chunk_section_success(self):
        """Successfully chunk a section and save to database"""
        service = ChunkingService()

        # Mock dependencies
        mock_document = Mock()
        mock_document.id = 1

        section_text = "## Tweede generatie\n\nII.1. Kinderen van Pieter en Maria"
        page_map = [{'page_number': 1, 'start': 0, 'end': 100}]

        with patch('genealogy.services.chunking_service.get_chunking_strategy') as mock_get_strategy, \
             patch('genealogy.services.chunking_service.save_chunks_to_db') as mock_save:

            # Setup mocks
            mock_strategy = Mock()
            mock_strategy.strategy_name = 'DescendantGenealogy'
            mock_strategy.chunk_section.return_value = ['chunk1', 'chunk2']
            mock_get_strategy.return_value = mock_strategy

            mock_save.return_value = ['saved_chunk1', 'saved_chunk2']

            # Execute
            result = service.chunk_section(
                section_type='DESCENDANT_GENEALOGY',
                section_text=section_text,
                document=mock_document,
                page_map=page_map,
                start_sequence=1
            )

            # Verify
            assert result['success'] is True
            assert result['chunks_created'] == 2
            assert len(result['saved_chunks']) == 2

            mock_get_strategy.assert_called_once_with('DESCENDANT_GENEALOGY')
            mock_strategy.chunk_section.assert_called_once_with(section_text, mock_document, page_map)
            mock_save.assert_called_once_with(
                ['chunk1', 'chunk2'],
                mock_document,
                page_map,
                section_text,
                start_sequence=1
            )

    def test_chunk_section_unknown_section_type(self):
        """Handle unknown section type gracefully"""
        service = ChunkingService()

        with patch('genealogy.services.chunking_service.get_chunking_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = KeyError("UNKNOWN_TYPE")

            result = service.chunk_section(
                section_type='UNKNOWN_TYPE',
                section_text='text',
                document=Mock(),
                page_map=[],
            )

            assert result['success'] is False
            assert 'Unknown section type' in result['error']
            assert result['chunks_created'] == 0

    def test_chunk_section_strategy_failure(self):
        """Handle strategy failure gracefully"""
        service = ChunkingService()

        with patch('genealogy.services.chunking_service.get_chunking_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.chunk_section.side_effect = Exception("Strategy error")
            mock_get_strategy.return_value = mock_strategy

            result = service.chunk_section(
                section_type='DESCENDANT_GENEALOGY',
                section_text='text',
                document=Mock(),
                page_map=[],
            )

            assert result['success'] is False
            assert 'Chunking failed' in result['error']
            assert result['chunks_created'] == 0

    def test_chunk_section_persistence_failure(self):
        """Handle database persistence failure gracefully"""
        service = ChunkingService()

        with patch('genealogy.services.chunking_service.get_chunking_strategy') as mock_get_strategy, \
             patch('genealogy.services.chunking_service.save_chunks_to_db') as mock_save:

            mock_strategy = Mock()
            mock_strategy.chunk_section.return_value = ['chunk1']
            mock_get_strategy.return_value = mock_strategy

            mock_save.side_effect = Exception("Database error")

            result = service.chunk_section(
                section_type='DESCENDANT_GENEALOGY',
                section_text='text',
                document=Mock(),
                page_map=[],
            )

            assert result['success'] is False
            assert 'Chunking failed' in result['error']

    def test_should_process_section_true(self):
        """Check if section should be processed - returns True"""
        service = ChunkingService()
        mock_section = Mock()
        mock_section.processed_at = None

        with patch('genealogy.services.chunking_service.get_chunking_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.should_process.return_value = True
            mock_get_strategy.return_value = mock_strategy

            result = service.should_process_section('DESCENDANT_GENEALOGY', mock_section)

            assert result is True
            mock_strategy.should_process.assert_called_once_with(mock_section)

    def test_should_process_section_false(self):
        """Check if section should be processed - returns False"""
        service = ChunkingService()
        mock_section = Mock()

        with patch('genealogy.services.chunking_service.get_chunking_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.should_process.return_value = False
            mock_get_strategy.return_value = mock_strategy

            result = service.should_process_section('DESCENDANT_GENEALOGY', mock_section)

            assert result is False

    def test_should_process_section_unknown_type(self):
        """Check unknown section type - returns False"""
        service = ChunkingService()

        with patch('genealogy.services.chunking_service.get_chunking_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = KeyError("UNKNOWN")

            result = service.should_process_section('UNKNOWN', Mock())

            assert result is False


@pytest.mark.unit
class TestExtractionService:
    """Test ExtractionService orchestration logic"""

    def test_extract_from_chunk_success(self):
        """Successfully extract entities from a chunk"""
        mock_ollama = Mock()
        service = ExtractionService(mock_ollama)

        mock_chunk = Mock()
        mock_chunk.chunk_type = ChunkType.INDIVIDUAL_ENTRY
        mock_chunk.sequence_number = 42

        with patch('genealogy.services.extraction_service.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.strategy_name = 'DescendantGenealogy'
            mock_strategy.should_process.return_value = True
            mock_strategy.extract.return_value = {
                'success': True,
                'people_count': 2,
                'events_count': 5
            }
            mock_get_strategy.return_value = mock_strategy

            result = service.extract_from_chunk(
                chunk=mock_chunk,
                section_type='DESCENDANT_GENEALOGY',
                model='llama3.2:3b'
            )

            assert result['success'] is True
            assert result['people_count'] == 2
            assert result['events_count'] == 5

            mock_get_strategy.assert_called_once_with('DESCENDANT_GENEALOGY')
            mock_strategy.should_process.assert_called_once_with(mock_chunk)
            mock_strategy.extract.assert_called_once_with(mock_chunk, mock_ollama, 'llama3.2:3b')

    def test_extract_from_chunk_strategy_refuses(self):
        """Strategy refuses to process chunk type"""
        mock_ollama = Mock()
        service = ExtractionService(mock_ollama)

        mock_chunk = Mock()
        mock_chunk.chunk_type = ChunkType.GENERATION_HEADER

        with patch('genealogy.services.extraction_service.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.strategy_name = 'DescendantGenealogy'
            mock_strategy.should_process.return_value = False
            mock_get_strategy.return_value = mock_strategy

            result = service.extract_from_chunk(
                chunk=mock_chunk,
                section_type='DESCENDANT_GENEALOGY',
                model='llama3.2:3b'
            )

            assert result['success'] is False
            assert 'cannot process chunk type' in result['error']
            mock_strategy.extract.assert_not_called()

    def test_extract_from_chunk_unknown_section_type(self):
        """Handle unknown section type gracefully"""
        mock_ollama = Mock()
        service = ExtractionService(mock_ollama)

        with patch('genealogy.services.extraction_service.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = KeyError("UNKNOWN")

            result = service.extract_from_chunk(
                chunk=Mock(),
                section_type='UNKNOWN',
                model='llama3.2:3b'
            )

            assert result['success'] is False
            assert 'Unknown section type' in result['error']

    def test_extract_from_chunk_extraction_failure(self):
        """Handle extraction failure gracefully"""
        mock_ollama = Mock()
        service = ExtractionService(mock_ollama)

        with patch('genealogy.services.extraction_service.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.should_process.return_value = True
            mock_strategy.extract.side_effect = Exception("Extraction error")
            mock_get_strategy.return_value = mock_strategy

            result = service.extract_from_chunk(
                chunk=Mock(),
                section_type='DESCENDANT_GENEALOGY',
                model='llama3.2:3b'
            )

            assert result['success'] is False
            assert 'Extraction failed' in result['error']

    def test_extract_from_chunks_in_section_all_success(self):
        """Extract from multiple chunks - all succeed"""
        mock_ollama = Mock()
        service = ExtractionService(mock_ollama)

        chunks = [Mock(sequence_number=i) for i in range(1, 4)]

        with patch('genealogy.services.extraction_service.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.should_process.return_value = True
            mock_strategy.extract.return_value = {'success': True}
            mock_get_strategy.return_value = mock_strategy

            result = service.extract_from_chunks_in_section(
                chunks=chunks,
                section_type='DESCENDANT_GENEALOGY',
                model='llama3.2:3b'
            )

            assert result['success'] is True
            assert result['processed'] == 3
            assert result['failed'] == 0
            assert len(result['errors']) == 0

    def test_extract_from_chunks_in_section_partial_failure(self):
        """Extract from multiple chunks - some fail"""
        mock_ollama = Mock()
        service = ExtractionService(mock_ollama)

        chunks = [Mock(sequence_number=i) for i in range(1, 4)]

        with patch('genealogy.services.extraction_service.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.should_process.return_value = True

            # First chunk succeeds, second fails, third succeeds
            mock_strategy.extract.side_effect = [
                {'success': True},
                {'success': False, 'error': 'LLM timeout'},
                {'success': True}
            ]
            mock_get_strategy.return_value = mock_strategy

            result = service.extract_from_chunks_in_section(
                chunks=chunks,
                section_type='DESCENDANT_GENEALOGY',
                model='llama3.2:3b'
            )

            assert result['success'] is False
            assert result['processed'] == 2
            assert result['failed'] == 1
            assert len(result['errors']) == 1
            assert 'Chunk 2' in result['errors'][0]
            assert 'LLM timeout' in result['errors'][0]

    def test_extract_from_chunks_in_section_all_fail(self):
        """Extract from multiple chunks - all fail"""
        mock_ollama = Mock()
        service = ExtractionService(mock_ollama)

        chunks = [Mock(sequence_number=i) for i in range(1, 3)]

        with patch('genealogy.services.extraction_service.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.should_process.return_value = True
            mock_strategy.extract.return_value = {'success': False, 'error': 'Failed'}
            mock_get_strategy.return_value = mock_strategy

            result = service.extract_from_chunks_in_section(
                chunks=chunks,
                section_type='DESCENDANT_GENEALOGY',
                model='llama3.2:3b'
            )

            assert result['success'] is False
            assert result['processed'] == 0
            assert result['failed'] == 2
            assert len(result['errors']) == 2

    def test_extract_from_chunks_in_section_empty_list(self):
        """Extract from empty chunk list"""
        mock_ollama = Mock()
        service = ExtractionService(mock_ollama)

        result = service.extract_from_chunks_in_section(
            chunks=[],
            section_type='DESCENDANT_GENEALOGY',
            model='llama3.2:3b'
        )

        assert result['success'] is True
        assert result['processed'] == 0
        assert result['failed'] == 0
        assert len(result['errors']) == 0
