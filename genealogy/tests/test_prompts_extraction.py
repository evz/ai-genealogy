"""Tests for extraction prompts and parsers"""
from unittest.mock import Mock, mock_open, patch

from django.test import TestCase

from genealogy.models import Document, TextChunk
from genealogy.prompts.extraction import (DUTCH_ABBREVIATIONS,
                                          EVENT_TYPE_CODES,
                                          build_extraction_prompt,
                                          load_examples,
                                          parse_extraction_output)


class TestLoadExamples(TestCase):
    """Test loading extraction examples"""

    @patch("builtins.open", new_callable=mock_open, read_data="Example extraction text")
    def test_loads_examples_from_file(self, mock_file):
        """Should load examples from examples_extraction.txt"""
        result = load_examples()

        self.assertEqual(result, "Example extraction text")
        mock_file.assert_called_once()
        # Verify it's trying to open the right file
        call_args = mock_file.call_args[0][0]
        self.assertIn("examples_extraction.txt", call_args)


class TestBuildExtractionPrompt(TestCase):
    """Test prompt building"""

    def setUp(self):
        """Create test document and chunk"""
        self.document = Document.objects.create(
            title="Test Doc",
            languages="nld",
        )

        self.chunk = TextChunk.objects.create(
            document=self.document,
            chunk_type="individual_entry",
            text_content="a. Pieter Jansen, geboren 1850 Amsterdam.",
            sequence_number=1,
            start_page=1,
            end_page=1,
            generation_number=2,
            family_groups=["II.1. Kinderen van Jan en Maria"],
            extracted_people=["Pieter Jansen"],
            extracted_relationships=[{"parent": "Jan", "child": "Pieter"}],
        )

    def test_builds_prompt_with_context(self):
        """Should include generation and family group context"""
        examples = "Example text"
        prompt = build_extraction_prompt(self.chunk, examples=examples)

        # Check structure
        self.assertIn(DUTCH_ABBREVIATIONS, prompt)
        self.assertIn(EVENT_TYPE_CODES, prompt)
        self.assertIn("Example text", prompt)
        self.assertIn("Generation 2", prompt)
        self.assertIn("II.1. Kinderen van Jan en Maria", prompt)
        self.assertIn("Pieter Jansen", prompt)
        self.assertIn("1 parent-child relationships", prompt)
        self.assertIn(self.chunk.text_content, prompt)

    def test_handles_missing_generation(self):
        """Should handle chunks without generation number"""
        self.chunk.generation_number = None
        self.chunk.save()

        prompt = build_extraction_prompt(self.chunk, examples="test")

        self.assertIn("Generation: None", prompt)

    def test_handles_missing_family_group(self):
        """Should handle chunks without family group"""
        self.chunk.family_groups = []
        self.chunk.save()

        prompt = build_extraction_prompt(self.chunk, examples="test")

        self.assertIn("Family Group: None", prompt)

    def test_handles_no_phase1_people(self):
        """Should handle chunks with no extracted people"""
        self.chunk.extracted_people = []
        self.chunk.save()

        prompt = build_extraction_prompt(self.chunk, examples="test")

        self.assertIn("People: None", prompt)

    def test_handles_no_phase1_relationships(self):
        """Should handle chunks with no extracted relationships"""
        self.chunk.extracted_relationships = []
        self.chunk.save()

        prompt = build_extraction_prompt(self.chunk, examples="test")

        self.assertIn("Relationships: None", prompt)

    @patch("genealogy.prompts.extraction.load_examples")
    def test_loads_examples_if_not_provided(self, mock_load):
        """Should load examples from file if not provided"""
        mock_load.return_value = "Loaded examples"

        prompt = build_extraction_prompt(self.chunk, examples=None)

        mock_load.assert_called_once()
        self.assertIn("Loaded examples", prompt)

    def test_includes_instructions(self):
        """Should include extraction instructions"""
        prompt = build_extraction_prompt(self.chunk, examples="test")

        self.assertIn("Extract ONLY information explicitly stated", prompt)
        self.assertIn("Life events", prompt)
        self.assertIn("Occupations", prompt)
        self.assertIn("OUTPUT FORMAT", prompt)


class TestParseExtractionOutput(TestCase):
    """Test parsing LLM output"""

    def test_parses_people_section(self):
        """Should parse people list"""
        output = """PEOPLE:
- Pieter Jansen
- Maria Pietersen

PARENT_CHILD:
None

PARTNERSHIPS:
None

EVENTS:
None"""

        result = parse_extraction_output(output)

        self.assertEqual(len(result['people']), 2)
        self.assertIn("Pieter Jansen", result['people'])
        self.assertIn("Maria Pietersen", result['people'])

    def test_parses_parent_child_relationships(self):
        """Should parse parent-child relationships"""
        output = """PEOPLE:
None

PARENT_CHILD:
- Jan Jansen|child|Pieter Jansen
- Maria Pietersen|child|Pieter Jansen

PARTNERSHIPS:
None

EVENTS:
None"""

        result = parse_extraction_output(output)

        self.assertEqual(len(result['parent_child']), 2)
        self.assertEqual(result['parent_child'][0]['person1'], "Jan Jansen")
        self.assertEqual(result['parent_child'][0]['relationship_type'], "child")
        self.assertEqual(result['parent_child'][0]['person2'], "Pieter Jansen")

    def test_parses_partnerships(self):
        """Should parse partnership relationships"""
        output = """PEOPLE:
None

PARENT_CHILD:
None

PARTNERSHIPS:
- Jan Jansen|spouse|Maria Pietersen

EVENTS:
None"""

        result = parse_extraction_output(output)

        self.assertEqual(len(result['partnerships']), 1)
        self.assertEqual(result['partnerships'][0]['person1'], "Jan Jansen")
        self.assertEqual(result['partnerships'][0]['relationship_type'], "spouse")
        self.assertEqual(result['partnerships'][0]['person2'], "Maria Pietersen")

    def test_parses_events(self):
        """Should parse events with all fields"""
        output = """PEOPLE:
None

PARENT_CHILD:
None

PARTNERSHIPS:
None

EVENTS:
- Pieter Jansen|BIRT|1850-01-15|Amsterdam
- Pieter Jansen|DEAT|1920-05-20|Rotterdam"""

        result = parse_extraction_output(output)

        self.assertEqual(len(result['events']), 2)
        self.assertEqual(result['events'][0]['person'], "Pieter Jansen")
        self.assertEqual(result['events'][0]['event_type'], "BIRT")
        self.assertEqual(result['events'][0]['date'], "1850-01-15")
        self.assertEqual(result['events'][0]['place'], "Amsterdam")

    def test_parses_events_with_missing_fields(self):
        """Should handle events with missing date or place"""
        output = """EVENTS:
- Pieter Jansen|BIRT||Amsterdam
- Maria|DEAT|1920-05-20|"""

        result = parse_extraction_output(output)

        self.assertEqual(len(result['events']), 2)
        self.assertEqual(result['events'][0]['date'], "")
        self.assertEqual(result['events'][1]['place'], "")

    def test_skips_none_entries(self):
        """Should skip 'None' entries"""
        output = """PEOPLE:
- None

PARENT_CHILD:
None

PARTNERSHIPS:
- none

EVENTS:
- NONE"""

        result = parse_extraction_output(output)

        self.assertEqual(len(result['people']), 0)
        self.assertEqual(len(result['parent_child']), 0)
        self.assertEqual(len(result['partnerships']), 0)
        self.assertEqual(len(result['events']), 0)

    def test_ignores_interpretation_section(self):
        """Should stop parsing at INTERPRETATION section"""
        output = """PEOPLE:
- Pieter Jansen

INTERPRETATION:
This is just commentary that should be ignored
- Not a person"""

        result = parse_extraction_output(output)

        self.assertEqual(len(result['people']), 1)
        self.assertNotIn("Not a person", result['people'])

    def test_handles_empty_output(self):
        """Should handle empty output gracefully"""
        result = parse_extraction_output("")

        self.assertEqual(len(result['people']), 0)
        self.assertEqual(len(result['parent_child']), 0)
        self.assertEqual(len(result['partnerships']), 0)
        self.assertEqual(len(result['events']), 0)

    def test_handles_malformed_relationships(self):
        """Should skip malformed relationship lines"""
        output = """PARENT_CHILD:
- Invalid line without pipes
- Jan|child|Pieter

PARTNERSHIPS:
- Also invalid
- Person1|spouse|Person2"""

        result = parse_extraction_output(output)

        # Should only parse the valid lines
        self.assertEqual(len(result['parent_child']), 1)
        self.assertEqual(len(result['partnerships']), 1)

    def test_strips_whitespace(self):
        """Should strip whitespace from parsed values"""
        output = """PEOPLE:
-   Pieter Jansen

EVENTS:
-  Pieter Jansen | BIRT | 1850 | Amsterdam """

        result = parse_extraction_output(output)

        self.assertEqual(result['people'][0], "Pieter Jansen")
        self.assertEqual(result['events'][0]['person'], "Pieter Jansen")
        self.assertEqual(result['events'][0]['event_type'], "BIRT")
        self.assertEqual(result['events'][0]['date'], "1850")
        self.assertEqual(result['events'][0]['place'], "Amsterdam")

    def test_handles_lines_without_dashes(self):
        """Should ignore lines that don't start with dashes"""
        output = """PEOPLE:
Pieter Jansen (without dash)
- Maria Pietersen (with dash)

EVENTS:
Some random text
- Pieter|BIRT|1850|Amsterdam"""

        result = parse_extraction_output(output)

        # Should only parse lines with dashes
        self.assertEqual(len(result['people']), 1)
        self.assertEqual(result['people'][0], "Maria Pietersen (with dash)")
        self.assertEqual(len(result['events']), 1)
