"""
Tests for chunk classification logic.
"""

import pytest

from genealogy.utils.chunk_classification import classify_chunk_tier


class TestChunkClassification:
    """Test the classify_chunk_tier function"""

    def test_generation_header_is_metadata(self):
        """Test that generation headers are always metadata tier"""
        result = classify_chunk_tier(
            text_content="## VIERDE GENERATIE",
            chunk_type='generation_header'
        )
        assert result == 'metadata'

    def test_family_group_header_is_metadata(self):
        """Test that family group headers are always metadata tier"""
        result = classify_chunk_tier(
            text_content="## II.9. Children of Jan van Zanten",
            chunk_type='family_group_header'
        )
        assert result == 'metadata'

    def test_short_individual_entry_is_metadata(self):
        """Test that short individual entries (< 100 chars) are metadata tier"""
        # 50 chars
        result = classify_chunk_tier(
            text_content="a. Thomas Lucas Borst, * Zwolle 24.3.2016.",
            chunk_type='individual_entry'
        )
        assert result == 'metadata'

    def test_long_individual_entry_is_narrative(self):
        """Test that long individual entries (>= 100 chars) are narrative tier"""
        # 173 chars with military content
        result = classify_chunk_tier(
            text_content="a. Leonardus Johannes van Zanten, * Watergraafsmeer 26.9.1866, † Breda 22.11.1895, sergeant der infanterie (1895), Begraven op de Nieuwe Ooster Begraafplaats.",
            chunk_type='individual_entry'
        )
        assert result == 'narrative'

    def test_exactly_100_chars_is_narrative(self):
        """Test that exactly 100 chars is narrative tier (boundary test)"""
        # Create a string that's exactly 100 chars
        text = "a" * 100
        result = classify_chunk_tier(
            text_content=text,
            chunk_type='individual_entry'
        )
        assert result == 'narrative'

    def test_99_chars_is_metadata(self):
        """Test that 99 chars is metadata tier (boundary test)"""
        # Create a string that's exactly 99 chars
        text = "a" * 99
        result = classify_chunk_tier(
            text_content=text,
            chunk_type='individual_entry'
        )
        assert result == 'metadata'

    def test_biographical_text_is_narrative(self):
        """Test that biographical_text chunks are always narrative tier"""
        result = classify_chunk_tier(
            text_content="Pieter emigrated to America in 1920.",
            chunk_type='biographical_text'
        )
        assert result == 'narrative'

    def test_narrative_context_is_narrative(self):
        """Test that narrative_context chunks are always narrative tier"""
        result = classify_chunk_tier(
            text_content="Gerrit van Santen, x (hypothetisch) Lijsbet ....",
            chunk_type='narrative_context'
        )
        assert result == 'narrative'

    def test_biographical_text_regardless_of_length(self):
        """Test that biographical_text is narrative even if short"""
        result = classify_chunk_tier(
            text_content="Short bio.",
            chunk_type='biographical_text'
        )
        assert result == 'narrative'

    def test_whitespace_is_stripped(self):
        """Test that whitespace is stripped when calculating length"""
        # 90 visible chars + 20 spaces = 110 total, but 90 after strip
        result = classify_chunk_tier(
            text_content="          " + "a" * 90 + "          ",
            chunk_type='individual_entry'
        )
        # 90 chars after strip < 100, so should be metadata
        assert result == 'metadata'

    def test_unknown_chunk_type_defaults_to_metadata(self):
        """Test that unknown chunk types default to metadata tier"""
        result = classify_chunk_tier(
            text_content="Some text content that's long enough to be narrative if it were individual_entry type",
            chunk_type='unknown'
        )
        assert result == 'metadata'

    def test_with_occupation_content(self):
        """Test that entries with occupations are narrative (assuming >= 100 chars)"""
        # 110 chars with occupation info
        result = classify_chunk_tier(
            text_content="a. Peter Bernardus van Zanten, * Bussum 19.1.1953, automonteur, garage eigenaar, Laarderweg 36, Bussum",
            chunk_type='individual_entry'
        )
        assert result == 'narrative'
