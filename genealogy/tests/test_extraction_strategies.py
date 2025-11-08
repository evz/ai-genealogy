"""Tests for extraction strategies"""
from unittest.mock import Mock, patch

from django.test import TestCase

from genealogy.extraction_strategies.descendant_genealogy import \
    DescendantGenealogyStrategy
from genealogy.models import Document, TextChunk


class TestDescendantGenealogyStrategy(TestCase):
    """Test the descendant genealogy extraction strategy"""

    def setUp(self):
        """Create test document and chunk"""
        self.document = Document.objects.create(
            title="Test Genealogy",
            languages="nld",
        )

        self.chunk = TextChunk.objects.create(
            document=self.document,
            chunk_type="individual_entry",
            text_content="a. Pieter Jansen, geboren 1850 Amsterdam, timmerman.",
            sequence_number=1,
            start_page=1,
            end_page=1,
            generation_number=2,
            family_groups=["II.1. Kinderen van Jan en Maria"],
            extracted_people=["Pieter Jansen"],  # Phase 1 data
            extracted_relationships=[{"person1": "Jan", "relationship_type": "child", "person2": "Pieter Jansen"}],
        )

        self.strategy = DescendantGenealogyStrategy()

    def test_strategy_name(self):
        """Should return correct strategy name"""
        self.assertEqual(self.strategy.strategy_name, "Descendant Genealogy Extraction")

    def test_should_process_individual_entry(self):
        """Should process individual_entry chunks"""
        # Update to use new chunk type
        self.assertTrue(self.strategy.should_process(self.chunk))

    def test_should_not_process_other_types(self):
        """Should not process non-entry chunks"""
        self.chunk.chunk_type = "generation_header"
        self.chunk.save()

        self.assertFalse(self.strategy.should_process(self.chunk))

    def test_get_chunk_filter(self):
        """Should return correct filter"""
        filter_dict = self.strategy.get_chunk_filter()

        # Old filter value needs updating
        self.assertIn("chunk_type", filter_dict)

    @patch("genealogy.extraction_strategies.descendant_genealogy.load_examples")
    @patch("genealogy.extraction_strategies.descendant_genealogy.build_extraction_prompt")
    @patch("genealogy.extraction_strategies.descendant_genealogy.parse_extraction_output")
    def test_extract_success(self, mock_parse, mock_build_prompt, mock_load_examples):
        """Should extract entities and merge with Phase 1 data"""
        # Setup mocks
        mock_load_examples.return_value = "test examples"
        mock_build_prompt.return_value = "test prompt"
        mock_parse.return_value = {
            'people': ["Maria Pietersen"],  # New person from Phase 2
            'parent_child': [{"person1": "Maria", "relationship_type": "child", "person2": "Pieter Jansen"}],
            'partnerships': [{"person1": "Pieter Jansen", "relationship_type": "spouse", "person2": "Anna de Vries"}],
            'events': [
                {"person": "Pieter Jansen", "event_type": "BIRT", "date": "1850", "place": "Amsterdam"},
                {"person": "Pieter Jansen", "event_type": "OCCU", "date": "", "place": "timmerman"},
            ]
        }

        mock_ollama = Mock()
        mock_ollama.generate.return_value = "LLM response text"

        result = self.strategy.extract(self.chunk, mock_ollama, "test-model")

        # Verify success
        self.assertTrue(result['success'])
        self.assertEqual(result['people_count'], 2)  # Phase 1 (Pieter) + Phase 2 (Maria)
        self.assertEqual(result['relationships_count'], 3)  # 1 Phase 1 + 2 Phase 2
        self.assertEqual(result['events_count'], 2)
        self.assertEqual(result['phase1_people'], 1)
        self.assertEqual(result['phase2_people_added'], 1)
        self.assertEqual(result['phase1_relationships'], 1)
        self.assertEqual(result['phase2_relationships_added'], 2)

        # Verify chunk was updated
        self.chunk.refresh_from_db()
        self.assertTrue(self.chunk.entities_extracted)
        self.assertEqual(len(self.chunk.extracted_people), 2)
        self.assertIn("Pieter Jansen", self.chunk.extracted_people)
        self.assertIn("Maria Pietersen", self.chunk.extracted_people)
        self.assertEqual(len(self.chunk.extracted_relationships), 3)
        self.assertEqual(len(self.chunk.extracted_events), 2)

        # Verify LLM was called
        mock_ollama.generate.assert_called_once()
        call_kwargs = mock_ollama.generate.call_args[1]
        self.assertEqual(call_kwargs['model'], "test-model")
        self.assertEqual(call_kwargs['prompt'], "test prompt")
        self.assertEqual(call_kwargs['options']['temperature'], 0.0)

    @patch("genealogy.extraction_strategies.descendant_genealogy.load_examples")
    @patch("genealogy.extraction_strategies.descendant_genealogy.build_extraction_prompt")
    @patch("genealogy.extraction_strategies.descendant_genealogy.parse_extraction_output")
    def test_deduplicates_people(self, mock_parse, mock_build_prompt, mock_load_examples):
        """Should not add duplicate people from Phase 2"""
        mock_load_examples.return_value = "test"
        mock_build_prompt.return_value = "test"
        mock_parse.return_value = {
            'people': ["Pieter Jansen"],  # Same as Phase 1
            'parent_child': [],
            'partnerships': [],
            'events': []
        }

        mock_ollama = Mock()
        mock_ollama.generate.return_value = "response"

        result = self.strategy.extract(self.chunk, mock_ollama, "model")

        self.assertEqual(result['people_count'], 1)  # Should not duplicate
        self.assertEqual(result['phase2_people_added'], 0)

    @patch("genealogy.extraction_strategies.descendant_genealogy.load_examples")
    @patch("genealogy.extraction_strategies.descendant_genealogy.build_extraction_prompt")
    @patch("genealogy.extraction_strategies.descendant_genealogy.parse_extraction_output")
    def test_deduplicates_relationships(self, mock_parse, mock_build_prompt, mock_load_examples):
        """Should not add duplicate relationships from Phase 2"""
        mock_load_examples.return_value = "test"
        mock_build_prompt.return_value = "test"
        mock_parse.return_value = {
            'people': [],
            'parent_child': [{"person1": "Jan", "relationship_type": "child", "person2": "Pieter Jansen"}],  # Duplicate
            'partnerships': [],
            'events': []
        }

        mock_ollama = Mock()
        mock_ollama.generate.return_value = "response"

        result = self.strategy.extract(self.chunk, mock_ollama, "model")

        self.assertEqual(result['relationships_count'], 1)  # Should not duplicate
        self.assertEqual(result['phase2_relationships_added'], 0)

    @patch("genealogy.extraction_strategies.descendant_genealogy.load_examples")
    @patch("genealogy.extraction_strategies.descendant_genealogy.build_extraction_prompt")
    def test_handles_no_llm_response(self, mock_build_prompt, mock_load_examples):
        """Should handle when LLM returns no response"""
        mock_load_examples.return_value = "test"
        mock_build_prompt.return_value = "test"

        mock_ollama = Mock()
        mock_ollama.generate.return_value = None

        result = self.strategy.extract(self.chunk, mock_ollama, "model")

        self.assertFalse(result['success'])
        self.assertIn("No response from LLM", result['error'])

    @patch("genealogy.extraction_strategies.descendant_genealogy.load_examples")
    @patch("genealogy.extraction_strategies.descendant_genealogy.build_extraction_prompt")
    def test_handles_llm_exception(self, mock_build_prompt, mock_load_examples):
        """Should handle exceptions during extraction"""
        mock_load_examples.return_value = "test"
        mock_build_prompt.return_value = "test"

        mock_ollama = Mock()
        mock_ollama.generate.side_effect = RuntimeError("LLM connection failed")

        result = self.strategy.extract(self.chunk, mock_ollama, "model")

        self.assertFalse(result['success'])
        self.assertIn("LLM connection failed", result['error'])

    @patch("genealogy.extraction_strategies.descendant_genealogy.load_examples")
    @patch("genealogy.extraction_strategies.descendant_genealogy.build_extraction_prompt")
    @patch("genealogy.extraction_strategies.descendant_genealogy.parse_extraction_output")
    def test_handles_missing_phase1_data(self, mock_parse, mock_build_prompt, mock_load_examples):
        """Should handle chunks with empty Phase 1 data"""
        self.chunk.extracted_people = []
        self.chunk.extracted_relationships = []
        self.chunk.save()

        mock_load_examples.return_value = "test"
        mock_build_prompt.return_value = "test"
        mock_parse.return_value = {
            'people': ["Pieter Jansen"],
            'parent_child': [],
            'partnerships': [],
            'events': []
        }

        mock_ollama = Mock()
        mock_ollama.generate.return_value = "response"

        result = self.strategy.extract(self.chunk, mock_ollama, "model")

        self.assertTrue(result['success'])
        self.assertEqual(result['people_count'], 1)
        self.assertEqual(result['phase1_people'], 0)

    @patch("genealogy.extraction_strategies.descendant_genealogy.load_examples")
    @patch("genealogy.extraction_strategies.descendant_genealogy.build_extraction_prompt")
    @patch("genealogy.extraction_strategies.descendant_genealogy.parse_extraction_output")
    def test_calculates_context_window(self, mock_parse, mock_build_prompt, mock_load_examples):
        """Should calculate appropriate context window for chunk size"""
        mock_load_examples.return_value = "test"
        mock_build_prompt.return_value = "test"
        mock_parse.return_value = {'people': [], 'parent_child': [], 'partnerships': [], 'events': []}

        # Create a large chunk
        self.chunk.text_content = "x" * 20000  # Large content
        self.chunk.save()

        mock_ollama = Mock()
        mock_ollama.generate.return_value = "response"

        self.strategy.extract(self.chunk, mock_ollama, "model")

        # Verify num_ctx was set appropriately
        call_kwargs = mock_ollama.generate.call_args[1]
        self.assertIn('num_ctx', call_kwargs['options'])
        self.assertGreater(call_kwargs['options']['num_ctx'], 8192)  # Should use larger window

    @patch("genealogy.extraction_strategies.descendant_genealogy.load_examples")
    def test_caches_examples(self, mock_load_examples):
        """Should load examples once and cache them"""
        mock_load_examples.return_value = "cached examples"

        # Access _examples property multiple times
        self.strategy._examples = None

        mock_ollama = Mock()
        mock_ollama.generate.return_value = "response"

        with patch("genealogy.extraction_strategies.descendant_genealogy.build_extraction_prompt") as mock_build, \
             patch("genealogy.extraction_strategies.descendant_genealogy.parse_extraction_output") as mock_parse:
            mock_build.return_value = "prompt"
            mock_parse.return_value = {'people': [], 'parent_child': [], 'partnerships': [], 'events': []}

            # First call should load examples
            self.strategy.extract(self.chunk, mock_ollama, "model")
            self.assertEqual(mock_load_examples.call_count, 1)

            # Second call should use cached examples
            self.strategy.extract(self.chunk, mock_ollama, "model")
            self.assertEqual(mock_load_examples.call_count, 1)  # Still 1, not 2
