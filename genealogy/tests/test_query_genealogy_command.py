"""
Integration tests for the query_genealogy management command.

Tests the CLI interface for both regular RAG mode and agent mode.
Mocks only the LLM calls to avoid dependencies on external services.
"""

import pytest
from io import StringIO
from unittest.mock import patch, Mock
from django.core.management import call_command
from django.test import TestCase

from genealogy.models import Document, TextChunk, Identity, PersonMention, MentionToIdentity, Event, Place


@pytest.mark.django_db
class TestQueryGenealogyCommand(TestCase):
    """Integration tests for query_genealogy CLI command"""

    def setUp(self):
        """Set up test data"""
        self.document = Document.objects.create(
            title="Test Genealogy",
            languages="nld"
        )

        # Create test chunks
        self.amsterdam = Place.objects.create(name="Amsterdam")

        self.chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=1,
            chunk_type="individual_entry",
            text_content="a. Pieter van Zanten, * Amsterdam 1850, timmerman",
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            extracted_people=["Pieter van Zanten"],
            start_page=10,
            end_page=10,
            embedding=[0.1] * 1024,
            dm_codes=["P361", "F523", "Z353"]
        )

        # Create identity for agent tests
        self.identity = Identity.objects.create(
            display_name="Pieter van Zanten",
            genealogical_identifier="II.3.a"
        )

        self.mention = PersonMention.objects.create(
            given_names="Pieter",
            surname="van Zanten",
            genealogical_id="II.3.a"
        )

        MentionToIdentity.objects.create(
            mention=self.mention,
            identity=self.identity,
            mapped_by="test"
        )

        Event.objects.create(
            mention=self.mention,
            event_type="BIRT",
            date="1850-01-15",
            place=self.amsterdam
        )

    @patch('genealogy.retrieval.OllamaClient')
    def test_basic_rag_query(self, mock_ollama_class):
        """Test basic RAG query without agent mode"""
        # Mock OllamaClient (used by HybridRetriever)
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama.generate.return_value = "Pieter van Zanten was a carpenter born in Amsterdam in 1850."
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Who is Pieter van Zanten?',
            stdout=out
        )

        output = out.getvalue()

        # Should show question
        self.assertIn("QUESTION: Who is Pieter van Zanten?", output)

        # Should show retrieval
        self.assertIn("Searching using hybrid RAG+RRF retrieval", output)
        self.assertIn("Retrieved 1 chunks", output)

        # Should show answer
        self.assertIn("ANSWER:", output)
        self.assertIn("Pieter van Zanten was a carpenter", output)

    @patch('genealogy.services.agent_executor.OllamaClient')
    @patch('genealogy.retrieval.OllamaClient')
    def test_agent_mode_query(self, mock_retrieval_ollama_class, mock_agent_ollama_class):
        """Test query with --agent flag"""
        # Mock retrieval OllamaClient
        mock_retrieval_ollama = Mock()
        mock_retrieval_ollama.embed.return_value = [0.1] * 1024
        mock_retrieval_ollama_class.return_value = mock_retrieval_ollama

        # Mock agent OllamaClient
        mock_agent_ollama = Mock()
        mock_agent_ollama.generate.side_effect = [
            'TOOL_CALL: search_person_by_name\nARGUMENTS: {"name": "Pieter van Zanten"}\nREASONING: Search for person',
            'ANSWER: Pieter van Zanten was born on January 15, 1850 in Amsterdam.'
        ]
        mock_agent_ollama_class.return_value = mock_agent_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Who is Pieter van Zanten?',
            '--agent',
            stdout=out
        )

        output = out.getvalue()

        # Should show agent mode
        self.assertIn("Using agentic workflow", output)

        # Should show tool calls
        self.assertIn("TOOL CALLS:", output)
        self.assertIn("search_person_by_name", output)

        # Should show answer
        self.assertIn("ANSWER:", output)
        self.assertIn("Pieter van Zanten", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_context_only_flag(self, mock_ollama_class):
        """Test --context-only flag shows context without generating answer"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Who is Pieter van Zanten?',
            '--context-only',
            stdout=out
        )

        output = out.getvalue()

        # Should show context
        self.assertIn("FULL CONTEXT:", output)
        self.assertIn("Pieter van Zanten", output)

        # Should NOT generate answer (no LLM call for generation)
        self.assertNotIn("Generating answer", output)
        # Should not have the answer section with double lines
        self.assertEqual(output.count("ANSWER:"), 0)

    @patch('genealogy.retrieval.OllamaClient')
    def test_show_scores_flag(self, mock_ollama_class):
        """Test --show-scores flag displays RRF scores"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama.generate.return_value = "Pieter was a carpenter."
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Pieter van Zanten',
            '--show-scores',
            stdout=out
        )

        output = out.getvalue()

        # Should show RRF scores
        self.assertIn("RETRIEVED CHUNKS (with RRF scores):", output)
        self.assertIn("Score:", output)
        self.assertIn("Seq:", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_top_k_parameter(self, mock_ollama_class):
        """Test --top-k parameter limits results"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama.generate.return_value = "Answer"
        mock_ollama_class.return_value = mock_ollama

        # Create additional chunks with van Zanten
        for i in range(5):
            TextChunk.objects.create(
                document=self.document,
                sequence_number=i + 10,
                chunk_type="individual_entry",
                text_content=f"Person {i} van Zanten, * Rotterdam 1900",
                subject=f"Person {i} van Zanten",
                start_page=i + 20,
                end_page=i + 20,
                embedding=[0.1] * 1024,
                dm_codes=["P361", "F523", "Z353"]
            )

        out = StringIO()
        call_command(
            'query_genealogy',
            'van Zanten',
            '--top-k', '2',
            '--expand-window', '0',  # Disable window expansion
            '--show-scores',
            stdout=out
        )

        output = out.getvalue()

        # Should retrieve exactly 2 chunks
        self.assertIn("Retrieved 2 chunks", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_language_detection_dutch(self, mock_ollama_class):
        """Test language detection for Dutch query"""
        mock_ollama = Mock()
        # Use same embedding as chunk to ensure match
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama.generate.return_value = "Pieter van Zanten was een timmerman."
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Pieter van Zanten',  # Simpler query that should match chunk
            stdout=out
        )

        output = out.getvalue()

        # Should detect Dutch (from query containing Dutch indicators like "van")
        self.assertIn("Detected language: Dutch", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_language_detection_english(self, mock_ollama_class):
        """Test language detection for English query"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama.generate.return_value = "Pieter van Zanten was a carpenter."
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Who was Pieter van Zanten?',  # English question
            stdout=out
        )

        output = out.getvalue()

        # Should detect English
        self.assertIn("Detected language: English", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_no_results_found(self, mock_ollama_class):
        """Test handling when no chunks are found"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.9] * 1024  # Very different embedding
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Napoleon Bonaparte',
            stdout=out
        )

        output = out.getvalue()

        # Should show warning about no results
        self.assertIn("No relevant chunks found", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_expand_window_parameter(self, mock_ollama_class):
        """Test --expand-window parameter"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama.generate.return_value = "Answer"
        mock_ollama_class.return_value = mock_ollama

        # Create adjacent chunks
        TextChunk.objects.create(
            document=self.document,
            sequence_number=0,
            chunk_type="generation_header",
            text_content="Tweede generatie",
            start_page=9,
            end_page=9,
            embedding=[0.1] * 1024,
            dm_codes=[]
        )

        TextChunk.objects.create(
            document=self.document,
            sequence_number=2,
            chunk_type="individual_entry",
            text_content="b. Another person",
            start_page=11,
            end_page=11,
            embedding=[0.1] * 1024,
            dm_codes=[]
        )

        out = StringIO()
        call_command(
            'query_genealogy',
            'Pieter van Zanten',
            '--expand-window', '1',
            stdout=out
        )

        output = out.getvalue()

        # Should retrieve center + expanded chunks
        # With window=1 and center at seq 1, should get seq 0, 1, 2 = 3 chunks
        self.assertIn("Retrieved 3 chunks", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_show_enrichment_flag(self, mock_ollama_class):
        """Test --show-enrichment flag includes metadata in context"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama.generate.return_value = "Answer"
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Pieter van Zanten',
            '--show-enrichment',
            '--context-only',
            stdout=out
        )

        output = out.getvalue()

        # Should include enrichment metadata
        self.assertIn("PEOPLE MENTIONED:", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_without_enrichment_flag(self, mock_ollama_class):
        """Test that enrichment is not shown by default"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.1] * 1024
        mock_ollama.generate.return_value = "Answer"
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'Pieter van Zanten',
            '--context-only',
            stdout=out
        )

        output = out.getvalue()

        # Should NOT include enrichment metadata
        self.assertNotIn("PEOPLE MENTIONED:", output)

    @patch('genealogy.services.agent_executor.OllamaClient')
    @patch('genealogy.retrieval.OllamaClient')
    def test_agent_mode_with_no_tool_calls(self, mock_retrieval_ollama_class, mock_agent_ollama_class):
        """Test agent mode when LLM answers immediately without tool calls"""
        # Mock retrieval OllamaClient
        mock_retrieval_ollama = Mock()
        mock_retrieval_ollama.embed.return_value = [0.1] * 1024
        mock_retrieval_ollama_class.return_value = mock_retrieval_ollama

        # Mock agent OllamaClient - immediate answer
        mock_agent_ollama = Mock()
        mock_agent_ollama.generate.return_value = 'ANSWER: Based on the context, Pieter van Zanten was born in 1850.'
        mock_agent_ollama_class.return_value = mock_agent_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            'When was Pieter born?',
            '--agent',
            stdout=out
        )

        output = out.getvalue()

        # Should show agent mode
        self.assertIn("Using agentic workflow", output)

        # Should NOT show tool calls section (no tools were called)
        self.assertNotIn("TOOL CALLS:", output)

        # Should show answer
        self.assertIn("ANSWER:", output)
        self.assertIn("1850", output)

    @patch('genealogy.retrieval.OllamaClient')
    def test_empty_query_returns_error(self, mock_ollama_class):
        """Test that empty query is handled gracefully"""
        mock_ollama = Mock()
        mock_ollama.embed.return_value = [0.0] * 1024
        mock_ollama_class.return_value = mock_ollama

        out = StringIO()
        call_command(
            'query_genealogy',
            '',  # Empty query
            stdout=out
        )

        output = out.getvalue()

        # Should show no results found
        self.assertIn("No relevant chunks found", output)
