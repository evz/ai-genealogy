"""
Tests for semantic query expansion.
"""

import pytest
from unittest.mock import Mock, patch

from genealogy.retrieval import HybridRetriever


@pytest.mark.django_db
class TestQueryExpansion:
    """Test that semantic queries are automatically expanded"""

    def test_query_expansion_is_called_for_semantic_queries(self):
        """Test that _expand_semantic_query is called for semantic queries (no DM codes)"""
        retriever = HybridRetriever()

        # Mock database check and search
        with patch('genealogy.retrieval.TextChunk') as mock_chunk:
            mock_chunk.objects.exclude.return_value.count.return_value = 1  # Has embeddings

            with patch.object(retriever, '_expand_semantic_query', return_value='expanded terms') as mock_expand:
                with patch.object(retriever, '_hybrid_search', return_value=[]):
                    # Semantic query (no capitalized names)
                    retriever.retrieve(query="who served in the military", top_k=10)

                    # Should have called expansion
                    mock_expand.assert_called_once_with("who served in the military")

    def test_query_expansion_not_called_for_name_queries(self):
        """Test that expansion is NOT called for name queries (has capitalized names)"""
        retriever = HybridRetriever()

        # Mock database check and search
        with patch('genealogy.retrieval.TextChunk') as mock_chunk:
            mock_chunk.objects.exclude.return_value.count.return_value = 1  # Has embeddings

            with patch.object(retriever, '_expand_semantic_query') as mock_expand:
                with patch.object(retriever, '_hybrid_search', return_value=[]):
                    # Name query (has capitalized name)
                    retriever.retrieve(query="Pieter van Zanten", top_k=10)

                    # Should NOT have called expansion
                    mock_expand.assert_not_called()

    def test_expansion_generates_additional_terms(self):
        """Test that expansion actually generates additional search terms"""
        retriever = HybridRetriever()

        # Mock ollama to return expanded terms
        mock_ollama = Mock()
        mock_ollama.generate.return_value = "militie, soldaat, ruiter, cavalerie, infanterie"
        retriever.ollama = mock_ollama

        expanded = retriever._expand_semantic_query("military service")

        # Should have called ollama
        mock_ollama.generate.assert_called_once()

        # Should return expanded terms
        assert "militie" in expanded
        assert "soldaat" in expanded
        assert "cavalerie" in expanded

    def test_expansion_fallback_on_error(self):
        """Test that expansion falls back to original query on error"""
        retriever = HybridRetriever()

        # Mock ollama to raise an error
        mock_ollama = Mock()
        mock_ollama.generate.side_effect = Exception("Connection failed")
        retriever.ollama = mock_ollama

        expanded = retriever._expand_semantic_query("military service")

        # Should return original query
        assert expanded == "military service"

    def test_detect_query_type_semantic(self):
        """Test that queries without capitalized names are detected as semantic"""
        retriever = HybridRetriever()

        query_type = retriever._detect_query_type("who served in the military", [])

        assert query_type == 'semantic'

    def test_detect_query_type_name(self):
        """Test that queries with DM codes are detected as name queries"""
        retriever = HybridRetriever()

        query_type = retriever._detect_query_type("Pieter van Zanten", ['536000'])

        assert query_type == 'name'
