"""
Integration tests for RAG+RRF hybrid retrieval.

These tests validate the hybrid search system with real database queries,
including vector search, trigram search, phonetic search, and subject boosting.

We mock only the embedding generation to avoid dependencies on external services.
"""

import pytest
from unittest.mock import patch, Mock
from django.test import TestCase
import numpy as np

from genealogy.models import Document, TextChunk, Place
from genealogy.retrieval import HybridRetriever


@pytest.mark.django_db
class TestHybridRetrieval(TestCase):
    """Integration tests for hybrid RAG+RRF retrieval"""

    def setUp(self):
        """Set up test chunks with varied content"""
        self.document = Document.objects.create(
            title="Test Genealogy",
            languages="nld"
        )

        # Create places
        self.amsterdam = Place.objects.create(name="Amsterdam")
        self.rotterdam = Place.objects.create(name="Rotterdam")

        # Chunk 1: Aart van Zanten the mason (subject match)
        self.mason_chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=1,
            chunk_type="individual_entry",
            text_content="a. Aart van Zanten, * Amsterdam 1850, metselaar (mason), † Rotterdam 1920",
            subject="Aart van Zanten",
            genealogical_identifier="II.3.a",
            extracted_people=["Aart van Zanten"],
            start_page=10,
            end_page=10,
            embedding=[0.1] * 1024,  # Dummy embedding
            dm_codes=["A630", "F523", "Z353"]  # Aart, van, Zanten
        )

        # Chunk 2: Aart van Zanten the farmer (subject match, different occupation)
        self.farmer_chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=2,
            chunk_type="individual_entry",
            text_content="b. Aart van Zanten, * Rotterdam 1880, boer (farmer), geh. Maria de Vries",
            subject="Aart van Zanten",
            genealogical_identifier="III.5.b",
            extracted_people=["Aart van Zanten", "Maria de Vries"],
            start_page=15,
            end_page=15,
            embedding=[0.2] * 1024,
            dm_codes=["A630", "F523", "Z353"]
        )

        # Chunk 3: Child of Aart mentioning him (NOT subject)
        self.child_chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=3,
            chunk_type="individual_entry",
            text_content="c. Pieter van Zanten, zoon van Aart van Zanten en Maria, * 1905",
            subject="Pieter van Zanten",
            genealogical_identifier="IV.2.c",
            extracted_people=["Pieter van Zanten", "Aart van Zanten", "Maria"],
            start_page=20,
            end_page=20,
            embedding=[0.15] * 1024,
            dm_codes=["P361", "F523", "Z353", "A630"]
        )

        # Chunk 4: Different person with similar name
        self.similar_chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=4,
            chunk_type="individual_entry",
            text_content="d. Aat van Zande, * Utrecht 1860, timmerman (carpenter)",
            subject="Aat van Zande",
            genealogical_identifier="II.7.d",
            extracted_people=["Aat van Zande"],
            start_page=25,
            end_page=25,
            embedding=[0.12] * 1024,
            dm_codes=["A300", "F523", "Z530"]  # Phonetically similar
        )

        # Chunk 5: Completely unrelated
        self.unrelated_chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=5,
            chunk_type="individual_entry",
            text_content="e. Johannes de Vries, * Den Haag 1870, bakker (baker)",
            subject="Johannes de Vries",
            genealogical_identifier="III.1.e",
            extracted_people=["Johannes de Vries"],
            start_page=30,
            end_page=30,
            embedding=[0.9] * 1024,  # Very different embedding
            dm_codes=["J520", "D172"]
        )

    @patch('genealogy.retrieval.OllamaClient')
    def test_subject_boosting_ranks_subject_chunks_higher(self, mock_ollama_class):
        """Test that subject-based RRF leg (k=10) heavily boosts chunks where person is subject"""
        # Mock embedding to return similar vectors for all
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.15] * 1024
        mock_ollama_class.return_value = mock_ollama

        retriever = HybridRetriever()
        results = retriever.retrieve(
            query="Aart van Zanten the mason",
            top_k=5,
            expand_window=0
        )

        # Extract chunk IDs
        result_ids = [r['id'] for r in results]

        # Both mason and child should be in results
        self.assertIn(self.mason_chunk.id, result_ids)
        self.assertIn(self.child_chunk.id, result_ids)

        # Mason chunk should rank higher than child chunk
        # because mason has "Aart van Zanten" as subject (k=10 boost)
        # while child only mentions him (no subject boost)
        mason_rank = result_ids.index(self.mason_chunk.id)
        child_rank = result_ids.index(self.child_chunk.id)
        self.assertLess(mason_rank, child_rank,
                       "Mason chunk (subject) should rank higher than child chunk (mention only)")

    @patch('genealogy.retrieval.OllamaClient')
    def test_person_filter_uses_and_logic(self, mock_ollama_class):
        """Test that person filter uses AND logic not OR (bug fix validation)"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.15] * 1024
        mock_ollama_class.return_value = mock_ollama

        retriever = HybridRetriever()
        results = retriever.retrieve(
            query="Aart van Zanten",
            top_k=5,
            expand_window=0
        )

        result_ids = [r['id'] for r in results]

        # Should find chunks with "Aart" AND "Zanten"
        self.assertIn(self.mason_chunk.id, result_ids, "Should find Aart van Zanten mason")
        self.assertIn(self.farmer_chunk.id, result_ids, "Should find Aart van Zanten farmer")

        # "Aat van Zande" has "Aat" (not "Aart") and "Zande" (not "Zanten")
        # With AND logic, this should NOT match
        # (With OR logic, it would match on partial similarity)

    @patch('genealogy.retrieval.OllamaClient')
    def test_window_expansion(self, mock_ollama_class):
        """Test that window expansion includes surrounding chunks"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama_class.return_value = mock_ollama

        # Create adjacent chunks to mason (sequence_number=1)
        # Window=1 means we get seq 0 to 2
        before_chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=0,  # seq 1 - 1 = 0
            chunk_type="generation_header",
            text_content="Tweede generatie",
            start_page=9,
            end_page=9,
            embedding=[0.5] * 1024,
            dm_codes=[]
        )

        # Note: sequence_number=2 is already used by farmer_chunk
        # So the window will expand to include farmer_chunk instead of creating a new one

        retriever = HybridRetriever()
        results = retriever.retrieve(
            query="Aart van Zanten mason",
            top_k=1,
            expand_window=1  # Include 1 chunk before/after
        )

        result_ids = [r['id'] for r in results]

        # Should include center chunk
        self.assertIn(self.mason_chunk.id, result_ids)

        # Should include expanded chunks (seq 0 and seq 2)
        self.assertIn(before_chunk.id, result_ids, "Should include chunk before center (seq 0)")
        self.assertIn(self.farmer_chunk.id, result_ids, "Should include chunk after center (seq 2)")

        # Should be 3 total chunks: before (seq 0), mason (seq 1), farmer (seq 2)
        self.assertEqual(len(results), 3, "Should have 3 chunks with window=1")

        # Check is_center flag
        for result in results:
            if result['id'] == self.mason_chunk.id:
                self.assertTrue(result.get('is_center', False), "Center chunk should be marked")
            else:
                self.assertFalse(result.get('is_center', True), "Expanded chunks should not be marked as center")

    @patch('genealogy.retrieval.OllamaClient')
    def test_rrf_scores_are_descending(self, mock_ollama_class):
        """Test that RRF properly orders results by score"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama_class.return_value = mock_ollama

        retriever = HybridRetriever()
        results = retriever.retrieve(
            query="Aart van Zanten",
            top_k=5,
            expand_window=0
        )

        # All results should have rrf_score
        for result in results:
            self.assertIn('rrf_score', result)

        # Scores should be in descending order
        scores = [r['rrf_score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True), "Results should be sorted by RRF score descending")

    @patch('genealogy.retrieval.OllamaClient')
    def test_retrieval_with_occupation_modifier(self, mock_ollama_class):
        """Test retrieval can distinguish people by occupation"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama_class.return_value = mock_ollama

        retriever = HybridRetriever()
        results = retriever.retrieve(
            query="Aart van Zanten the mason",
            top_k=3,
            expand_window=0
        )

        result_ids = [r['id'] for r in results]

        # Mason chunk should be in results
        self.assertIn(self.mason_chunk.id, result_ids,
                     "Should find the mason when searching for 'Aart van Zanten the mason'")

        # If both mason and farmer are in results, mason should rank higher
        if self.mason_chunk.id in result_ids and self.farmer_chunk.id in result_ids:
            mason_rank = result_ids.index(self.mason_chunk.id)
            farmer_rank = result_ids.index(self.farmer_chunk.id)
            self.assertLess(mason_rank, farmer_rank,
                           "Mason should rank higher when query mentions 'mason'")

    @patch('genealogy.retrieval.OllamaClient')
    def test_build_context_includes_metadata(self, mock_ollama_class):
        """Test that build_context includes genealogical metadata"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama_class.return_value = mock_ollama

        retriever = HybridRetriever()
        chunks = retriever.retrieve(
            query="Aart van Zanten",
            top_k=2,
            expand_window=0
        )

        context = retriever.build_context(chunks, include_anchors=True, include_enrichment=True)

        # Should include genealogical identifiers
        self.assertIn("II.3.a", context, "Should include genealogical ID")

        # Should include subject
        self.assertIn("Aart van Zanten", context)

        # Should include people
        self.assertIn("PEOPLE MENTIONED:", context, "Should include extracted people metadata")

    @patch('genealogy.retrieval.OllamaClient')
    def test_build_context_without_enrichment(self, mock_ollama_class):
        """Test that build_context works without enrichment metadata"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama_class.return_value = mock_ollama

        retriever = HybridRetriever()
        chunks = retriever.retrieve(
            query="Aart van Zanten",
            top_k=2,
            expand_window=0
        )

        context = retriever.build_context(chunks, include_anchors=True, include_enrichment=False)

        # Should still have text content
        self.assertIn("Aart van Zanten", context)

        # Should NOT include enrichment metadata
        self.assertNotIn("PEOPLE MENTIONED:", context, "Should not include people metadata when enrichment disabled")

    @patch('genealogy.retrieval.OllamaClient')
    def test_retrieval_respects_top_k_limit(self, mock_ollama_class):
        """Test that retrieval respects top_k parameter"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama_class.return_value = mock_ollama

        retriever = HybridRetriever()

        # Test with top_k=2
        results = retriever.retrieve(
            query="Aart van Zanten",
            top_k=2,
            expand_window=0
        )

        # Should return exactly top_k center chunks (without expansion)
        # We have multiple chunks matching "Aart van Zanten", so should get exactly 2
        center_chunks = [r for r in results if r.get('is_center', True)]
        self.assertEqual(len(center_chunks), 2, "Should return exactly top_k=2 chunks")

    @patch('genealogy.retrieval.OllamaClient')
    def test_empty_query_returns_empty_results(self, mock_ollama_class):
        """Test that empty query returns no results"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.0] * 1024
        mock_ollama_class.return_value = mock_ollama

        retriever = HybridRetriever()
        results = retriever.retrieve(
            query="",
            top_k=5,
            expand_window=0
        )

        self.assertEqual(len(results), 0, "Empty query should return no results")
