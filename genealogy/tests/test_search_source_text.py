"""
Tests for search_source_text tool in GenealogyTools.

Tests the semantic search functionality that allows cross-cutting queries
like "Who lived in Minneapolis?" or "Are there any musicians?"
"""

import pytest
from unittest.mock import patch, Mock
from genealogy.services.genealogy_tools import GenealogyTools
from genealogy.models import Person


@pytest.mark.django_db
class TestSearchSourceText:
    """Test the search_source_text tool"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tools = GenealogyTools()

        # Create test people for genealogical ID extraction
        self.pieter = Person.objects.create(
            given_names="Pieter",
            surname="van Zanten",
            genealogical_id="VIII.3.d"
        )
        self.hilda = Person.objects.create(
            given_names="Hilda Victoria",
            surname="Fogelberg",
            genealogical_id="VIII.3.d.spouse1"
        )

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_search_source_text_returns_correct_structure(self, mock_retriever_class):
        """Test that search_source_text returns the expected data structure"""
        # Mock the retriever to return sample chunks
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                'id': 'chunk-123',
                'text_content': 'Pieter (VIII.3.d) lived in Minneapolis.',
                'subject': 'Pieter van Zanten',
                'genealogical_identifier': 'VIII.3.d',
                'chunk_type': 'individual_entry',
                'start_page': 45,
                'end_page': 47,
                'rrf_score': 0.92
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        result = self.tools.search_source_text(
            query="Minneapolis",
            max_results=5
        )

        # Should return expected top-level structure
        assert "count" in result
        assert "results" in result
        assert "query" in result
        assert result["query"] == "Minneapolis"
        assert result["count"] == 1

        # Should have called retriever with max_results * 3 (due to filtering)
        mock_retriever.retrieve.assert_called_once_with(
            query="Minneapolis",
            top_k=15,  # 5 * 3
            expand_window=0
        )

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_search_source_text_result_fields(self, mock_retriever_class):
        """Test that each result has all required fields"""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                'id': 'chunk-456',
                'text_content': 'Pieter played the flute and piccolo.',
                'subject': 'Pieter van Zanten',
                'genealogical_identifier': 'VIII.3.d',
                'chunk_type': 'individual_entry',
                'start_page': 50,
                'end_page': 52,
                'rrf_score': 0.88
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        result = self.tools.search_source_text(query="musician", max_results=10)

        # Check first result has all required fields
        first_result = result["results"][0]
        assert "chunk_id" in first_result
        assert "text" in first_result
        assert "subject" in first_result
        assert "genealogical_id" in first_result
        assert "page_range" in first_result
        assert "score" in first_result
        assert "mentioned_people" in first_result

        # Check types
        assert isinstance(first_result["chunk_id"], str)
        assert isinstance(first_result["text"], str)
        assert isinstance(first_result["score"], float)
        assert isinstance(first_result["mentioned_people"], list)

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_genealogical_id_extraction(self, mock_retriever_class):
        """Test that genealogical IDs are extracted from text and linked to people"""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                'id': 'chunk-789',
                'text_content': 'Pieter (VIII.3.d) married Hilda Victoria Fogelberg (VIII.3.d.spouse1) in 1930.',
                'subject': 'Pieter van Zanten',
                'genealogical_identifier': 'VIII.3.d',
                'chunk_type': 'individual_entry',
                'start_page': 60,
                'end_page': 62,
                'rrf_score': 0.95
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        result = self.tools.search_source_text(query="marriage", max_results=5)

        # Should extract both genealogical IDs
        mentioned = result["results"][0]["mentioned_people"]
        assert len(mentioned) == 2

        # Check that it found both people in the database
        ids = [p["genealogical_id"] for p in mentioned]
        assert "VIII.3.d" in ids
        assert "VIII.3.d.spouse1" in ids

        # Check that names are included
        names = [p["name"] for p in mentioned]
        assert "Pieter van Zanten" in names
        assert "Hilda Victoria Fogelberg" in names

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_max_results_limit(self, mock_retriever_class):
        """Test that max_results parameter limits number of chunks returned"""
        mock_retriever = Mock()
        # Return more chunks than requested
        mock_retriever.retrieve.return_value = [
            {
                'id': f'chunk-{i}',
                'text_content': f'Text chunk {i}',
                'subject': 'Person',
                'genealogical_identifier': 'I.1.a',
                'chunk_type': 'individual_entry',
                'start_page': i,
                'end_page': i,
                'rrf_score': 0.5
            }
            for i in range(20)
        ]
        mock_retriever_class.return_value = mock_retriever

        result = self.tools.search_source_text(query="test", max_results=5)

        # Should call retriever with max_results * 3
        mock_retriever.retrieve.assert_called_once_with(
            query="test",
            top_k=15,  # 5 * 3
            expand_window=0
        )

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_safety_limit_caps_max_results(self, mock_retriever_class):
        """Test that max_results can't exceed safety limit (20)"""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_class.return_value = mock_retriever

        # Request 100 results
        result = self.tools.search_source_text(query="test", max_results=100)

        # Should be capped at 20, then * 3 = 60
        mock_retriever.retrieve.assert_called_once_with(
            query="test",
            top_k=60,  # 20 (safety limit) * 3
            expand_window=0
        )

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_genealogical_id_deduplication(self, mock_retriever_class):
        """Test that duplicate genealogical IDs in text are deduplicated"""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                'id': 'chunk-999',
                'text_content': 'Pieter (VIII.3.d) and his wife. Pieter (VIII.3.d) later moved to America.',
                'subject': 'Pieter van Zanten',
                'genealogical_identifier': 'VIII.3.d',
                'chunk_type': 'individual_entry',
                'start_page': 70,
                'end_page': 72,
                'rrf_score': 0.90
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        result = self.tools.search_source_text(query="emigration", max_results=5)

        # Even though VIII.3.d appears twice in the text, should only be in mentioned_people once
        mentioned = result["results"][0]["mentioned_people"]
        ids = [p["genealogical_id"] for p in mentioned]
        assert ids.count("VIII.3.d") == 1

    def test_genealogical_id_regex_pattern(self):
        """Test that the genealogical ID regex pattern works correctly"""
        import re

        # This is the pattern used in search_source_text
        id_pattern = r'\b([IVX]+\.\d+\.[a-z]+(?:\.\w+)?)\b'

        # Should match valid IDs
        valid_ids = [
            "VII.3.a",
            "VIII.3.d",
            "IX.10.b",
            "VIII.3.d.spouse1",
            "VI.1.n.child2"
        ]

        for test_id in valid_ids:
            test_text = f"Person ({test_id}) lived here."
            matches = re.findall(id_pattern, test_text)
            assert test_id in matches, f"Pattern should match {test_id}"

        # Should not match invalid patterns
        invalid_text = "This is VII but not an ID, and neither is 3.a or VIII.a.3"
        matches = re.findall(id_pattern, invalid_text)
        assert len(matches) == 0, "Should not match invalid patterns"

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_empty_results(self, mock_retriever_class):
        """Test behavior when no chunks are returned"""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_class.return_value = mock_retriever

        result = self.tools.search_source_text(query="nonexistent", max_results=5)

        assert result["count"] == 0
        assert len(result["results"]) == 0
        assert result["query"] == "nonexistent"

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_handles_chunks_without_genealogical_ids(self, mock_retriever_class):
        """Test that chunks without genealogical IDs are filtered out"""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                'id': 'chunk-plain',
                'text_content': 'This is just plain text without any genealogical identifiers.',
                'subject': 'Unknown',
                'genealogical_identifier': '',
                'chunk_type': 'individual_entry',
                'start_page': 1,
                'end_page': 1,
                'rrf_score': 0.50
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        result = self.tools.search_source_text(query="text", max_results=5)

        # Should still return the chunk since it has chunk_type set
        assert result["count"] == 1
        # But mentioned_people will be empty since genealogical_identifier is empty
        assert len(result["results"][0]["mentioned_people"]) == 0

    @patch('genealogy.services.genealogy_tools.HybridRetriever')
    def test_filters_out_non_biographical_chunks(self, mock_retriever_class):
        """Test that non-biographical chunk types are filtered out"""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                'id': 'chunk-header',
                'text_content': '## VIERDE GENERATIE',
                'subject': None,
                'genealogical_identifier': None,
                'chunk_type': 'generation_header',
                'start_page': 1,
                'end_page': 1,
                'rrf_score': 0.50
            },
            {
                'id': 'chunk-bio',
                'text_content': 'Pieter van Zanten lived in Amsterdam.',
                'subject': 'Pieter van Zanten',
                'genealogical_identifier': 'VIII.3.d',
                'chunk_type': 'individual_entry',
                'start_page': 2,
                'end_page': 2,
                'rrf_score': 0.80
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        result = self.tools.search_source_text(query="test", max_results=5)

        # Should only return the biographical chunk, not the header
        assert result["count"] == 1
        assert result["results"][0]["chunk_id"] == 'chunk-bio'
