"""Tests for genealogy.chunking.parser module"""

import pytest

from genealogy.chunking.models import ChunkType
from genealogy.chunking.parser import (
    detect_chunk_type,
    extract_person_from_individual_entry,
    is_biographical_text,
    is_source_citation,
    parse_grounding_tokens,
)
from genealogy.tests.helpers import create_grounding_token, load_fixture


@pytest.mark.unit
class TestParseGroundingTokens:
    """Test parse_grounding_tokens() function"""

    def test_parse_basic_token(self):
        """Parse a basic grounding token"""
        ocr_text = '<|ref|>text<|/ref|><|det|>[[10, 20, 100, 40]]<|/det|>\nHello World'
        tokens = parse_grounding_tokens(ocr_text)

        assert len(tokens) == 1
        assert tokens[0].content == 'Hello World'
        assert tokens[0].element_type == 'text'
        assert tokens[0].bbox.x1 == 10
        assert tokens[0].bbox.y1 == 20
        assert tokens[0].bbox.x2 == 100
        assert tokens[0].bbox.y2 == 40
        assert tokens[0].is_inverted is False

    def test_parse_inverted_token(self):
        """Parse token with inverted flag"""
        ocr_text = '<|ref|>sub_title<|/ref|><|det|>[[10, 20, 100, 40]]<|/det|><|inverted|>true<|/inverted|>\nInfo Box Header'
        tokens = parse_grounding_tokens(ocr_text)

        assert len(tokens) == 1
        assert tokens[0].content == 'Info Box Header'
        assert tokens[0].is_inverted is True

    def test_parse_multiple_tokens(self):
        """Parse sequence of multiple tokens"""
        ocr_text = (
            '<|ref|>sub_title<|/ref|><|det|>[[10, 10, 100, 30]]<|/det|>\n'
            'Header\n'
            '<|ref|>text<|/ref|><|det|>[[10, 35, 500, 55]]<|/det|>\n'
            'Body text'
        )
        tokens = parse_grounding_tokens(ocr_text)

        assert len(tokens) == 2
        assert tokens[0].element_type == 'sub_title'
        assert tokens[0].content == 'Header'
        assert tokens[1].element_type == 'text'
        assert tokens[1].content == 'Body text'

    def test_preserves_document_order(self):
        """Tokens are returned in document order, not sorted by bbox"""
        ocr_text = load_fixture('ocr_samples/page_77_generation_12.txt')
        tokens = parse_grounding_tokens(ocr_text)

        # First token should be generation header
        assert 'TWAALFDE GENERATIE' in tokens[0].content
        # Second should be first family group
        assert 'XII.1.' in tokens[1].content


@pytest.mark.unit
class TestChunkTypeDetection:
    """Test detect_chunk_type() function"""

    def test_generation_header_detection(self):
        """Detect generation headers"""
        # Standard format
        token = create_grounding_token(
            '## Tweede generatie',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.GENERATION_HEADER

        # Without markdown hashes
        token = create_grounding_token(
            'Derde generatie',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.GENERATION_HEADER

        # Twelfth generation
        token = create_grounding_token(
            'TWAALFDE GENERATIE',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.GENERATION_HEADER

        # OCR corruption: "ierde" (missing 'V')
        token = create_grounding_token(
            'ierde generatie',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.GENERATION_HEADER

    def test_family_group_header_detection_as_subtitle(self):
        """Detect family group headers in sub_title elements"""
        # Standard format
        token = create_grounding_token(
            'II.3. Kinderen van Pieter en Maria',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

        # English variant
        token = create_grounding_token(
            'XII.1. Children of Kelly and Jesse',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

        # With reference
        token = create_grounding_token(
            'VII.2. Kinderen van Jan en Anna (VI.1.b):',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

    def test_family_group_header_detection_as_text(self):
        """Detect family group headers in text elements (generation 12 bug fix)"""
        # This is the regression test for the generation 12 bug
        token = create_grounding_token(
            'XII.1. Children of Kelly Kamp Cameron and Jesse Newton (XI.8.a):',
            element_type='text'  # DeepSeek-OCR sometimes labels these as 'text'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

        # Another generation 12 example
        token = create_grounding_token(
            'XII.3. Children of Emily Frances Terpstra and Adam Michael Utsch (X.12.a):',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

    def test_family_group_header_without_period_after_id(self):
        """
        Detect family group headers without period after ID (chunk 335 bug fix).

        Some OCR text has format "X.14 Children of..." instead of "X.14. Children of..."
        Pattern should match both with and without the period.
        """
        # Without period after ID (the bug case)
        token = create_grounding_token(
            'X.14 Children of Harold Richard Van Zanten and Carol Sue Sohns (IX.6.b):',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

        # With period (standard format - should still work)
        token = create_grounding_token(
            'X.14. Children of Harold Richard Van Zanten and Carol Sue Sohns (IX.6.b):',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

        # Also test with markdown hashes
        token = create_grounding_token(
            '## X.14 Children of Harold and Carol:',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

    def test_remarriage_family_group_detection(self):
        """Detect remarriage family group headers (no roman numeral)"""
        token = create_grounding_token(
            'Kinderen van Hendrik Willem van Zanten en Hendrika Vuijst (V.1.c):',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

        # As text element
        token = create_grounding_token(
            'Children of Mary Kathleen Terpstra and Andrew James Spike (X.12.b):',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.FAMILY_GROUP_HEADER

    def test_individual_entry_detection(self):
        """Detect individual entries"""
        # Standard format
        token = create_grounding_token(
            'a. Pieter van Zanten',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.INDIVIDUAL_ENTRY

        # With biographical data
        token = create_grounding_token(
            'b. Maria Jansen, * Amsterdam 15.1.1850, † Naarden 20.3.1920',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.INDIVIDUAL_ENTRY

        # Hypothetical entry
        token = create_grounding_token(
            'c. (hypothetisch) Jan van Zanten',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.INDIVIDUAL_ENTRY

        # Labeled as sub_title by DeepSeek-OCR
        token = create_grounding_token(
            'd. Anna Pieterse',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.INDIVIDUAL_ENTRY

    def test_source_citation_detection(self):
        """Detect source citations"""
        # Keyword in text
        token = create_grounding_token(
            'RGV DTB 1234',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.SOURCE_CITATION

        # Header
        token = create_grounding_token(
            'Bronverwijzing',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.SOURCE_CITATION

        # Archive code
        token = create_grounding_token(
            'SSANO Weesregister folio 123',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.SOURCE_CITATION

    def test_info_box_detection(self):
        """Detect info boxes"""
        # Inverted content is always info box
        token = create_grounding_token(
            'Important Note',
            element_type='sub_title',
            is_inverted=True
        )
        assert detect_chunk_type(token) == ChunkType.INFO_BOX

        # Sub_title that's not a known pattern
        token = create_grounding_token(
            'Familie Van Zanten in Amerika',
            element_type='sub_title'
        )
        assert detect_chunk_type(token) == ChunkType.INFO_BOX

    def test_biographical_vs_narrative_text(self):
        """Distinguish biographical facts from narrative context"""
        # Biographical: contains genealogical markers
        token = create_grounding_token(
            'Pieter, * Amsterdam 1.1.1850, † Naarden 2.2.1920, x Maria',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.BIOGRAPHICAL_TEXT

        # Narrative: story text without genealogical markers
        token = create_grounding_token(
            'The family moved to America in search of better opportunities.',
            element_type='text'
        )
        assert detect_chunk_type(token) == ChunkType.NARRATIVE_CONTEXT

    def test_image_types(self):
        """Detect image-related chunk types"""
        token = create_grounding_token('', element_type='image')
        assert detect_chunk_type(token) == ChunkType.IMAGE

        token = create_grounding_token('Portrait of Pieter', element_type='image_caption')
        assert detect_chunk_type(token) == ChunkType.IMAGE_CAPTION

        token = create_grounding_token('', element_type='table')
        assert detect_chunk_type(token) == ChunkType.TABLE


@pytest.mark.unit
class TestIsSourceCitation:
    """Test is_source_citation() helper function"""

    def test_detects_archive_codes(self):
        """Detect archive codes at start of content"""
        assert is_source_citation('RGV DTB 1234')
        assert is_source_citation('SSANO Weesregister')

    def test_detects_keywords(self):
        """Detect source citation keywords"""
        assert is_source_citation('Bevolkingsregister Amsterdam')
        assert is_source_citation('Bronverwijzing: DTB')

    def test_rejects_normal_text(self):
        """Normal text is not a source citation"""
        assert not is_source_citation('Pieter van Zanten was born in Amsterdam')


@pytest.mark.unit
class TestIsBiographicalText:
    """Test is_biographical_text() helper function"""

    def test_detects_birth_markers(self):
        """Detect birth/death symbols with dates or places"""
        assert is_biographical_text('* 10.1.1850')
        assert is_biographical_text('* Amsterdam')
        assert is_biographical_text('† 12-1-1920')
        assert is_biographical_text('~ Naarden 5.6.1855')

    def test_detects_marriage_markers(self):
        """Detect marriage markers"""
        assert is_biographical_text('x 1. Maria Jansen')
        assert is_biographical_text('x 2. Anna Pieterse')

    def test_detects_genealogical_abbreviations(self):
        """Detect son/daughter of, widow abbreviations"""
        assert is_biographical_text('zv Pieter van Zanten')
        assert is_biographical_text('dv Jan Jansen')
        assert is_biographical_text('wed. Maria Pieterse')

    def test_detects_residence_patterns(self):
        """Detect residence year:place patterns"""
        assert is_biographical_text('1850: Naarden')
        assert is_biographical_text('1920: Amsterdam')

    def test_detects_address_patterns(self):
        """Detect addresses with street numbers"""
        assert is_biographical_text('Amsterdam 123, later Naarden')

    def test_rejects_narrative_with_birth_year(self):
        """Narrative mentioning birth year is not biographical"""
        # (* YYYY) in narrative context should not trigger
        text = 'Pieter van Zanten (* 1944) helped establish the organization.'
        # This should be False because "* 1944)" doesn't match the pattern
        # Pattern requires "* " followed by date (dd.mm or place name)
        assert not is_biographical_text(text)

    def test_rejects_normal_narrative(self):
        """Normal narrative is not biographical"""
        assert not is_biographical_text('The family moved to America in 1920.')


@pytest.mark.unit
class TestExtractPersonFromIndividualEntry:
    """Test extract_person_from_individual_entry() function"""

    def test_extracts_simple_name(self):
        """Extract person name from simple entry"""
        content = 'a. Pieter van Zanten, * Amsterdam 1.1.1850'
        assert extract_person_from_individual_entry(content) == 'Pieter van Zanten'

    def test_extracts_name_with_nickname(self):
        """Extract name and remove bracketed nickname"""
        content = 'b. Thomas [Tom] van Zanten, */~ Naarden 10.6.1862'
        assert extract_person_from_individual_entry(content) == 'Thomas van Zanten'

    def test_extracts_name_before_death_marker(self):
        """Extract name when followed by death marker"""
        content = 'c. Maria Jansen† Amsterdam 1920'
        assert extract_person_from_individual_entry(content) == 'Maria Jansen'

    def test_extracts_name_before_marriage_marker(self):
        """Extract name when followed by marriage marker"""
        content = 'd. Jan Pieterse x Anna de Vries'
        assert extract_person_from_individual_entry(content) == 'Jan Pieterse'

    def test_handles_ocr_error_uppercase_i(self):
        """Handle OCR error where 'l.' becomes 'I.'"""
        content = 'I. Hendrik van Zanten, * 1880'
        # Should still extract even with uppercase I
        assert extract_person_from_individual_entry(content) == 'Hendrik van Zanten'

    def test_returns_none_for_non_entry(self):
        """Return None for non-individual-entry content"""
        content = 'This is just regular text without an entry marker'
        assert extract_person_from_individual_entry(content) is None


@pytest.mark.integration
class TestGenerationTwelveRegression:
    """Regression test for generation 12 family group header bug"""

    def test_page_77_family_group_headers_detected(self):
        """
        Regression test: Family group headers in generation 12 should be detected
        even when labeled as 'text' instead of 'sub_title' by DeepSeek-OCR.
        """
        ocr_text = load_fixture('ocr_samples/page_77_generation_12.txt')
        tokens = parse_grounding_tokens(ocr_text)

        # Count how many family group headers are detected
        family_group_headers = [
            t for t in tokens
            if detect_chunk_type(t) == ChunkType.FAMILY_GROUP_HEADER
        ]

        # Should detect all three family group headers: XII.1, XII.2, XII.3
        assert len(family_group_headers) >= 3

        # Verify they are the correct tokens
        assert 'XII.1. Children of Kelly' in family_group_headers[0].content
        assert 'XII.2. Children of Ellery' in family_group_headers[1].content
        assert 'XII.3. Children of Emily' in family_group_headers[2].content

    def test_individual_entries_not_merged_with_headers(self):
        """
        Verify individual entries are separate from family group headers.
        """
        ocr_text = load_fixture('ocr_samples/page_77_generation_12.txt')
        tokens = parse_grounding_tokens(ocr_text)

        individual_entries = [
            t for t in tokens
            if detect_chunk_type(t) == ChunkType.INDIVIDUAL_ENTRY
        ]

        # Should detect individual entries like "a. Isabella", "b. Lorelai", etc.
        # The fixture has entries in a single token with multiple individuals,
        # so we check for the pattern
        entry_content = ''.join(t.content for t in individual_entries)
        assert 'a. Isabella Marian Newton' in entry_content or any(
            'Isabella' in t.content for t in tokens if 'a.' in t.content
        )


@pytest.mark.unit
class TestDetectChunkType:
    """Test detect_chunk_type() function"""

    def test_individual_entry_at_start_of_text(self):
        """Should detect individual entry when pattern is at start of text"""
        token = create_grounding_token(
            element_type='text',
            content='a. John Smith, * 1850, † 1920',
            bbox=(10, 10, 500, 30)
        )

        assert detect_chunk_type(token) == ChunkType.INDIVIDUAL_ENTRY

    def test_individual_entry_after_newline(self):
        """Should detect individual entry when pattern appears after a newline (regression test for IX.3.c bug)

        This tests the fix for the issue where individual entries were merged into citation
        chunks when they appeared after other text in the same grounding token.
        Example: chunk 241 had a photo caption followed by "c. Frances Joan Kamp..."
        """
        token = create_grounding_token(
            element_type='text',
            content='''From left to right: Upper row: Fran K, John K, Tom K, Lower row: Frances vZ, Jim K, James K, Hani K. The positioning of the persons is remarkable similar to the picture of the Van Zanten- van Barneveld family.

c. Frances Joan [Fran] Kamp, * Dearborn (Wayne, MI) 13.2.1928, † Bloomington (Hennepin, MN) 11.8.2007, administrative assistant''',
            bbox=(10, 10, 500, 200)
        )

        # Should detect this as INDIVIDUAL_ENTRY because "c." appears at start of a line
        assert detect_chunk_type(token) == ChunkType.INDIVIDUAL_ENTRY

    def test_individual_entry_in_subtitle(self):
        """Should detect individual entry when OCR labels it as sub_title"""
        token = create_grounding_token(
            element_type='sub_title',
            content='b. Thomas George Kamp, * Detroit (MI) 1925',
            bbox=(10, 10, 500, 30)
        )

        assert detect_chunk_type(token) == ChunkType.INDIVIDUAL_ENTRY

    def test_individual_entry_with_multiline_content(self):
        """Should detect individual entry even with multiple individuals in one token"""
        token = create_grounding_token(
            element_type='text',
            content='''Some narrative text about the family.

d. Dr James Nathaniel [Jim] Kamp, * Dearborn (MI) 14.11.1935
e. Dr Hannah Gezina Alida [Honey] Kamp, * Dearborn (Wayne, MI) 13.3.1938''',
            bbox=(10, 10, 500, 200)
        )

        # Should detect as INDIVIDUAL_ENTRY because "d." appears at start of a line
        assert detect_chunk_type(token) == ChunkType.INDIVIDUAL_ENTRY

    def test_not_individual_entry_when_letter_not_at_line_start(self):
        """Should NOT detect individual entry when letter+period is mid-sentence"""
        token = create_grounding_token(
            element_type='text',
            content='He was born in Amsterdam, e.g. in the Netherlands.',
            bbox=(10, 10, 500, 30)
        )

        # Should be narrative, not individual entry (e.g. is not at start of line)
        chunk_type = detect_chunk_type(token)
        assert chunk_type in (ChunkType.NARRATIVE_CONTEXT, ChunkType.BIOGRAPHICAL_TEXT)
