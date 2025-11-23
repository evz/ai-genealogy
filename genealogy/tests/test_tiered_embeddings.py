"""
Tests for tiered embedding generation.
"""

import pytest
from unittest.mock import Mock, patch

from genealogy.models import TextChunk, Document
from genealogy.services.chunk_enrichment import ChunkEnrichmentService


@pytest.mark.django_db
class TestTieredEmbeddings:
    """Test that embeddings are only generated for narrative tier chunks"""

    def setup_method(self):
        """Create a test document for all tests"""
        self.document = Document.objects.create(
            title="Test Document",
            languages="eng+nld"
        )

    def test_metadata_tier_skips_embedding(self):
        """Test that metadata tier chunks skip embedding generation"""
        # Create a metadata tier chunk
        chunk = TextChunk.objects.create(
            document=self.document,
            text_content="a. Short stub entry.",
            chunk_type='individual_entry',
            search_tier='metadata',
            start_page=1,
            end_page=1,
            sequence_number=1
        )

        # Mock Ollama client
        mock_ollama = Mock()
        service = ChunkEnrichmentService(mock_ollama)

        # Try to enrich the chunk
        result = service.enrich_chunk(
            chunk,
            embedding_model="test-model",
            generate_embedding=True,
            generate_dm_codes=False
        )

        # Should succeed but not generate embedding
        assert result['success'] is True
        assert result['embedding_generated'] is False

        # Ollama embed should NOT have been called
        mock_ollama.embed.assert_not_called()

        # Chunk should still have no embedding
        chunk.refresh_from_db()
        assert chunk.embedding is None

    def test_narrative_tier_generates_embedding(self):
        """Test that narrative tier chunks do generate embeddings"""
        # Create a narrative tier chunk
        chunk = TextChunk.objects.create(
            document=self.document,
            text_content="a. Long entry with biographical content about occupation and residence that exceeds 100 characters.",
            chunk_type='individual_entry',
            search_tier='narrative',
            start_page=1,
            end_page=1,
            sequence_number=1
        )

        # Mock Ollama client to return a fake embedding
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024  # Fake 1024-dim embedding
        service = ChunkEnrichmentService(mock_ollama)

        # Try to enrich the chunk
        result = service.enrich_chunk(
            chunk,
            embedding_model="test-model",
            generate_embedding=True,
            generate_dm_codes=False
        )

        # Should succeed and generate embedding
        assert result['success'] is True
        assert result['embedding_generated'] is True

        # Ollama embed should have been called
        mock_ollama.embed.assert_called_once()

        # Chunk should now have an embedding
        chunk.refresh_from_db()
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 1024

    def test_force_flag_still_skips_metadata(self):
        """Test that force=True still skips metadata tier chunks"""
        # Create a metadata tier chunk with an existing embedding
        chunk = TextChunk.objects.create(
            document=self.document,
            text_content="a. Short stub.",
            chunk_type='individual_entry',
            search_tier='metadata',
            embedding=[0.5] * 1024,  # Has an existing embedding
            start_page=1,
            end_page=1,
            sequence_number=1
        )

        # Mock Ollama client
        mock_ollama = Mock()
        service = ChunkEnrichmentService(mock_ollama)

        # Try to enrich with force=True
        result = service.enrich_chunk(
            chunk,
            embedding_model="test-model",
            generate_embedding=True,
            generate_dm_codes=False,
            force=True
        )

        # Should succeed but not regenerate embedding
        assert result['success'] is True
        assert result['embedding_generated'] is False

        # Ollama embed should NOT have been called
        mock_ollama.embed.assert_not_called()

    def test_narrative_tier_biographical_text(self):
        """Test that biographical_text chunks (always narrative) get embeddings"""
        # Create a biographical_text chunk
        chunk = TextChunk.objects.create(
            document=self.document,
            text_content="Short biographical note.",
            chunk_type='biographical_text',
            search_tier='narrative',  # Should be narrative per classification logic
            start_page=1,
            end_page=1,
            sequence_number=1
        )

        # Mock Ollama client
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.2] * 1024
        service = ChunkEnrichmentService(mock_ollama)

        # Enrich
        result = service.enrich_chunk(
            chunk,
            embedding_model="test-model",
            generate_embedding=True,
            generate_dm_codes=False
        )

        # Should generate embedding
        assert result['success'] is True
        assert result['embedding_generated'] is True
        mock_ollama.embed.assert_called_once()
