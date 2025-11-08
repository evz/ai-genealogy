"""Tests for genealogy.chunking.handlers module

Tests focus on edge cases and special logic in handlers, including:
- Remarriage handling (inheriting family group IDs)
- Individual entry consolidation (stopping at structural boundaries)
- Source citation boundaries
- Reading order preservation (not using spatial heuristics)
- Phase 1 entity extraction
"""

import pytest

from genealogy.chunking.handlers import (CHUNK_HANDLERS, DefaultChunkHandler,
                                         FamilyGroupHeaderHandler,
                                         GenerationHeaderHandler,
                                         IndividualEntryHandler,
                                         StandaloneSourceCitationHandler)
from genealogy.chunking.models import ChunkType
from genealogy.tests.helpers import create_token_sequence


@pytest.mark.unit
class TestGenerationHeaderHandler:
    """Test GenerationHeaderHandler"""

    def test_extracts_generation_number(self):
        """Extract generation text from header"""
        handler = GenerationHeaderHandler()
        tokens = create_token_sequence([
            {'content': '## Derde generatie', 'element_type': 'sub_title'},
        ])

        context = {
            'generation': None,
            'family_group': None,
            'family_group_id': None,
            'parents': None,
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.GENERATION_HEADER,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert chunk.chunk_type == ChunkType.GENERATION_HEADER
        assert chunk.generation == 'Derde generatie'
        assert context['generation'] == 'Derde generatie'
        assert next_idx == 1

    def test_handles_markdown_hashes(self):
        """Handle markdown hashes in generation headers"""
        handler = GenerationHeaderHandler()
        tokens = create_token_sequence([
            {'content': '## TWAALFDE GENERATIE', 'element_type': 'sub_title'},
        ])

        context = {}
        chunk, _ = handler.create_chunk(
            ChunkType.GENERATION_HEADER,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        # Should extract the full matched text including case
        assert chunk.generation == 'TWAALFDE GENERATIE'

    def test_handles_ocr_corruption(self):
        """Handle OCR corruptions in generation text"""
        handler = GenerationHeaderHandler()
        tokens = create_token_sequence([
            {'content': 'ierde generatie', 'element_type': 'sub_title'},  # Missing 'V'
        ])

        context = {}
        chunk, _ = handler.create_chunk(
            ChunkType.GENERATION_HEADER,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert chunk.generation == 'ierde generatie'


@pytest.mark.unit
class TestFamilyGroupHeaderHandler:
    """Test FamilyGroupHeaderHandler with focus on remarriage edge case"""

    def test_standard_family_group_with_roman_numeral(self):
        """Extract family group with roman numeral ID"""
        handler = FamilyGroupHeaderHandler()
        tokens = create_token_sequence([
            {'content': 'II.3. Kinderen van Pieter van Zanten en Maria Jansen', 'element_type': 'sub_title'},
        ])

        context = {'generation': 'Tweede generatie'}
        chunk, next_idx = handler.create_chunk(
            ChunkType.FAMILY_GROUP_HEADER,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert chunk.chunk_type == ChunkType.FAMILY_GROUP_HEADER
        assert chunk.family_group_id == 'II.3'
        assert chunk.parents == ('Pieter van Zanten', 'Maria Jansen')
        assert context['family_group_id'] == 'II.3'
        assert context['parents'] == ('Pieter van Zanten', 'Maria Jansen')
        assert next_idx == 1

    def test_remarriage_inherits_family_group_id(self):
        """
        EDGE CASE: Remarriage headers (no roman numeral) inherit family_group_id
        from previous marriage but update parents.
        """
        handler = FamilyGroupHeaderHandler()

        # First marriage: sets family_group_id
        tokens1 = create_token_sequence([
            {'content': 'VII.1. Kinderen van Hendrik van Zanten en Jannetje Pieterse', 'element_type': 'sub_title'},
        ])
        context = {'generation': 'Zevende generatie'}
        chunk1, _ = handler.create_chunk(
            ChunkType.FAMILY_GROUP_HEADER,
            tokens1[0],
            tokens1,
            0,
            context,
            []
        )

        assert context['family_group_id'] == 'VII.1'
        assert context['parents'] == ('Hendrik van Zanten', 'Jannetje Pieterse')

        # Second marriage (remarriage): inherits VII.1 but updates parents
        tokens2 = create_token_sequence([
            {'content': 'Kinderen van Hendrik Willem van Zanten en Hendrika Vuijst (V.1.c):', 'element_type': 'sub_title'},
        ])
        chunk2, _ = handler.create_chunk(
            ChunkType.FAMILY_GROUP_HEADER,
            tokens2[0],
            tokens2,
            0,
            context,
            []
        )

        # Family group ID should remain VII.1 (inherited)
        assert context['family_group_id'] == 'VII.1'
        # But parents should be updated
        assert context['parents'] == ('Hendrik Willem van Zanten', 'Hendrika Vuijst')
        assert chunk2.family_group_id == 'VII.1'
        assert chunk2.parents == ('Hendrik Willem van Zanten', 'Hendrika Vuijst')

    def test_single_parent_family_group(self):
        """Handle family groups with only one parent"""
        handler = FamilyGroupHeaderHandler()
        tokens = create_token_sequence([
            {'content': 'III.5. Kinderen van Maria Jansen', 'element_type': 'sub_title'},
        ])

        context = {'generation': 'Derde generatie'}
        chunk, _ = handler.create_chunk(
            ChunkType.FAMILY_GROUP_HEADER,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert chunk.parents == ('Maria Jansen',)

    def test_english_variant(self):
        """Handle English 'Children of' variant"""
        handler = FamilyGroupHeaderHandler()
        tokens = create_token_sequence([
            {'content': 'XII.1. Children of Kelly Cameron and Jesse Newton (XI.8.a):', 'element_type': 'text'},
        ])

        context = {'generation': 'Twaalfde generatie'}
        chunk, _ = handler.create_chunk(
            ChunkType.FAMILY_GROUP_HEADER,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert chunk.family_group_id == 'XII.1'
        assert chunk.parents == ('Kelly Cameron', 'Jesse Newton')


@pytest.mark.unit
class TestIndividualEntryHandler:
    """Test IndividualEntryHandler with focus on consolidation logic"""

    def test_basic_individual_entry(self):
        """Create basic individual entry chunk"""
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'a. Pieter van Zanten, * Amsterdam 1.1.1850', 'element_type': 'text'},
        ])

        context = {
            'generation': 'Tweede generatie',
            'family_group': 'II.1. Kinderen van...',
            'family_group_id': 'II.1',
            'parents': ('Jan van Zanten', 'Maria Pieterse')
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert chunk.chunk_type == ChunkType.INDIVIDUAL_ENTRY
        assert chunk.individual_marker == 'a.'
        assert 'Pieter van Zanten' in chunk.content
        assert next_idx == 1

    def test_consolidates_biographical_text(self):
        """
        EDGE CASE: Individual entries consolidate following biographical text
        until hitting a structural boundary.
        """
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'a. Thomas van Zanten, */~ Naarden 10.6.1862', 'element_type': 'text'},
            {'content': 'x 1. Alida Langelaar, * Amsterdam 1865', 'element_type': 'text'},
            {'content': '1890: Naarden, later Amsterdam', 'element_type': 'text'},
            {'content': 'b. Maria van Zanten, * 1864', 'element_type': 'text'},  # Next entry - STOP
        ])

        context = {
            'generation': 'Vierde generatie',
            'family_group': 'IV.2. Kinderen van Hendrik van Zanten',
            'family_group_id': 'IV.2',
            'parents': ('Hendrik van Zanten',)
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        # Should include first 3 tokens, stop at token 3 (next individual entry)
        assert len(chunk.grounding_tokens) == 3
        assert 'Thomas van Zanten' in chunk.content
        assert 'Alida Langelaar' in chunk.content
        assert '1890: Naarden' in chunk.content
        assert 'Maria van Zanten' not in chunk.content  # Stopped before this
        assert next_idx == 3  # Points to 'b. Maria'

    def test_stops_at_family_group_header(self):
        """
        EDGE CASE: Individual entry stops at family group header boundary.
        """
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'a. Pieter van Zanten, * 1850', 'element_type': 'text'},
            {'content': 'Biographical text about Pieter', 'element_type': 'text'},
            {'content': 'III.2. Kinderen van Anna en Jan', 'element_type': 'sub_title'},  # STOP
        ])

        context = {
            'generation': 'Derde generatie',
            'family_group': 'III.1. Kinderen van Pieter en Maria',
            'family_group_id': 'III.1',
            'parents': ('Pieter', 'Maria')
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert len(chunk.grounding_tokens) == 2
        assert 'Pieter van Zanten' in chunk.content
        assert 'Biographical text' in chunk.content
        assert 'III.2' not in chunk.content
        assert next_idx == 2

    def test_includes_inline_source_citations(self):
        """
        EDGE CASE: Individual entries include inline source citations (text level)
        but stop at source citation section headers (sub_title level).
        """
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'a. Pieter van Zanten, * 1850', 'element_type': 'text'},
            {'content': 'RGV DTB 1234', 'element_type': 'text'},  # Inline citation - INCLUDE
            {'content': 'Bronverwijzing', 'element_type': 'sub_title'},  # Section header - STOP
        ])

        context = {
            'generation': 'Tweede generatie',
            'family_group': 'II.1. Kinderen van Pieter en Maria',
            'family_group_id': 'II.1',
            'parents': ('Pieter', 'Maria')
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert len(chunk.grounding_tokens) == 2
        assert 'RGV DTB 1234' in chunk.content  # Inline citation included
        assert 'Bronverwijzing' not in chunk.content  # Section header excluded
        assert next_idx == 2

    def test_includes_narrative_context(self):
        """Individual entries include narrative context text"""
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'a. Pieter van Zanten, * 1850', 'element_type': 'text'},
            {'content': 'The family moved to America in 1920.', 'element_type': 'text'},
            {'content': 'b. Jan van Zanten', 'element_type': 'text'},  # STOP
        ])

        context = {
            'generation': 'Derde generatie',
            'family_group': 'III.1. Kinderen van Pieter en Maria',
            'family_group_id': 'III.1',
            'parents': ('Pieter', 'Maria')
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert len(chunk.grounding_tokens) == 2
        assert 'moved to America' in chunk.content
        assert next_idx == 2

    def test_includes_info_box_content(self):
        """
        EDGE CASE: Individual entries include info box content (sub-sections)
        that provide additional context.
        """
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'a. Pieter van Zanten, * 1850', 'element_type': 'text'},
            {'content': 'Familie in Amerika', 'element_type': 'sub_title'},  # Info box header
            {'content': 'Details about the American branch...', 'element_type': 'text'},
            {'content': 'b. Maria van Zanten', 'element_type': 'text'},  # STOP
        ])

        context = {
            'generation': 'Vierde generatie',
            'family_group': 'IV.1. Kinderen van Hendrik en Anna',
            'family_group_id': 'IV.1',
            'parents': ('Hendrik', 'Anna')
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert len(chunk.grounding_tokens) == 3
        assert 'Familie in Amerika' in chunk.content
        assert 'American branch' in chunk.content
        assert next_idx == 3

    def test_normalizes_ocr_error_uppercase_i(self):
        """
        EDGE CASE: OCR sometimes mistakes lowercase 'l' for uppercase 'I'.
        Normalize 'I.' to 'l.' for individual markers.
        """
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'I. Hendrik van Zanten, * 1880', 'element_type': 'text'},
        ])

        context = {
            'generation': 'Vijfde generatie',
            'family_group': 'V.1. Kinderen van Pieter en Maria',
            'family_group_id': 'V.1',
            'parents': ('Pieter', 'Maria')
        }
        chunk, _ = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        # 'I.' should be normalized to 'l.'
        assert chunk.individual_marker == 'l.'

    def test_phase1_entity_extraction(self):
        """
        EDGE CASE: Phase 1 extraction creates people and parent-child relationships.
        """
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'a. Pieter van Zanten, * Amsterdam 1.1.1850', 'element_type': 'text'},
        ])

        context = {
            'generation': 'Tweede generatie',
            'family_group': 'II.3. Kinderen van Jan van Zanten en Maria Pieterse',
            'family_group_id': 'II.3',
            'parents': ('Jan van Zanten', 'Maria Pieterse')
        }
        chunk, _ = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        # Should extract person name
        assert 'Pieter van Zanten' in chunk.extracted_people

        # Should extract parents
        assert 'Jan van Zanten' in chunk.extracted_people
        assert 'Maria Pieterse' in chunk.extracted_people

        # Should create parent-child relationships
        assert len(chunk.extracted_relationships) == 2
        assert {
            'person1': 'Jan van Zanten',
            'relationship_type': 'parent',
            'person2': 'Pieter van Zanten'
        } in chunk.extracted_relationships
        assert {
            'person1': 'Maria Pieterse',
            'relationship_type': 'parent',
            'person2': 'Pieter van Zanten'
        } in chunk.extracted_relationships

        # Should set the subject field
        assert chunk.subject == 'Pieter van Zanten'

    def test_no_phase1_extraction_without_generation(self):
        """Phase 1 extraction only runs if generation context exists"""
        handler = IndividualEntryHandler()
        tokens = create_token_sequence([
            {'content': 'a. Pieter van Zanten, * 1850', 'element_type': 'text'},
        ])

        context = {}  # No generation
        chunk, _ = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        # No extraction without generation context
        assert len(chunk.extracted_people) == 0
        assert len(chunk.extracted_relationships) == 0


@pytest.mark.unit
class TestStandaloneSourceCitationHandler:
    """Test StandaloneSourceCitationHandler"""

    def test_basic_source_citation(self):
        """Create basic source citation chunk"""
        handler = StandaloneSourceCitationHandler()
        tokens = create_token_sequence([
            {'content': 'Bronverwijzing', 'element_type': 'sub_title'},
            {'content': 'RGV DTB 1234', 'element_type': 'text'},
        ])

        context = {
            'generation': 'Derde generatie',
            'family_group': 'III.1. Kinderen van Pieter en Maria',
            'family_group_id': 'III.1',
            'parents': ('Pieter', 'Maria')
        }
        chunks = [None]  # One existing chunk
        chunk, next_idx = handler.create_chunk(
            ChunkType.SOURCE_CITATION,
            tokens[0],
            tokens,
            0,
            context,
            chunks
        )

        assert chunk.chunk_type == ChunkType.SOURCE_CITATION
        assert len(chunk.grounding_tokens) == 2
        assert 'RGV DTB 1234' in chunk.content
        assert chunk.supports_chunk_index == 0  # References previous chunk
        assert next_idx == 2

    def test_stops_at_structural_boundaries(self):
        """
        EDGE CASE: Source citations stop at structural boundaries
        (generation headers, family groups, individual entries).
        """
        handler = StandaloneSourceCitationHandler()
        tokens = create_token_sequence([
            {'content': 'Bronverwijzing', 'element_type': 'sub_title'},
            {'content': 'RGV DTB 1234', 'element_type': 'text'},
            {'content': 'SSANO Weesregister', 'element_type': 'text'},
            {'content': 'a. Next person entry', 'element_type': 'text'},  # STOP
        ])

        context = {
            'generation': 'Tweede generatie',
            'family_group': 'II.1. Kinderen van Pieter en Maria',
            'family_group_id': 'II.1',
            'parents': ('Pieter', 'Maria')
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.SOURCE_CITATION,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert len(chunk.grounding_tokens) == 3
        assert 'RGV DTB 1234' in chunk.content
        assert 'SSANO' in chunk.content
        assert 'Next person' not in chunk.content
        assert next_idx == 3

    def test_includes_info_boxes_within_citations(self):
        """
        EDGE CASE: Newspaper headers and other info boxes within citations
        are labeled as INFO_BOX but should be included.
        """
        handler = StandaloneSourceCitationHandler()
        tokens = create_token_sequence([
            {'content': 'Bronverwijzing', 'element_type': 'sub_title'},
            {'content': 'De Telegraaf', 'element_type': 'sub_title'},  # Newspaper header (INFO_BOX)
            {'content': 'Article text from newspaper...', 'element_type': 'text'},
        ])

        context = {
            'generation': 'Vierde generatie',
            'family_group': 'IV.1. Kinderen van Hendrik en Anna',
            'family_group_id': 'IV.1',
            'parents': ('Hendrik', 'Anna')
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.SOURCE_CITATION,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert len(chunk.grounding_tokens) == 3
        assert 'De Telegraaf' in chunk.content
        assert 'Article text' in chunk.content
        assert next_idx == 3

    def test_stops_at_next_citation_section(self):
        """
        EDGE CASE: Stop at next source citation section header,
        but include inline citations.
        """
        handler = StandaloneSourceCitationHandler()
        tokens = create_token_sequence([
            {'content': 'Bronverwijzing', 'element_type': 'sub_title'},
            {'content': 'RGV DTB 1234', 'element_type': 'text'},
            {'content': 'Literatuur', 'element_type': 'sub_title'},  # Next section - STOP
        ])

        # Empty context - tests that handler doesn't crash without genealogical context
        context = {
            'generation': None,
            'family_group': None,
            'family_group_id': None,
            'parents': None
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.SOURCE_CITATION,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert len(chunk.grounding_tokens) == 2
        assert 'RGV DTB 1234' in chunk.content
        assert 'Literatuur' not in chunk.content
        assert next_idx == 2


@pytest.mark.unit
class TestDefaultChunkHandler:
    """Test DefaultChunkHandler (fallback)"""

    def test_handles_any_chunk_type(self):
        """Default handler creates simple single-token chunk"""
        handler = DefaultChunkHandler()
        tokens = create_token_sequence([
            {'content': 'Some unrecognized content', 'element_type': 'text'},
        ])

        context = {
            'generation': 'Tweede generatie',
            'family_group': 'II.1. Kinderen van Pieter en Maria',
            'family_group_id': 'II.1',
            'parents': ('Pieter', 'Maria')
        }
        chunk, next_idx = handler.create_chunk(
            ChunkType.UNKNOWN,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        assert chunk.chunk_type == ChunkType.UNKNOWN
        assert chunk.content == 'Some unrecognized content'
        assert len(chunk.grounding_tokens) == 1
        assert next_idx == 1


@pytest.mark.integration
class TestHandlerRegistry:
    """Test handler registry and selection logic"""

    def test_handlers_registered_in_correct_order(self):
        """Handlers should be ordered from most specific to most general"""
        assert isinstance(CHUNK_HANDLERS[0], GenerationHeaderHandler)
        assert isinstance(CHUNK_HANDLERS[1], FamilyGroupHeaderHandler)
        assert isinstance(CHUNK_HANDLERS[2], IndividualEntryHandler)
        assert isinstance(CHUNK_HANDLERS[3], StandaloneSourceCitationHandler)
        assert isinstance(CHUNK_HANDLERS[4], DefaultChunkHandler)

    def test_handler_selection(self):
        """Correct handler is selected for each chunk type"""
        tokens = create_token_sequence([
            {'content': 'Tweede generatie', 'element_type': 'sub_title'},
        ])

        # Find handler for generation header
        selected_handler = None
        for handler in CHUNK_HANDLERS:
            if handler.can_handle(ChunkType.GENERATION_HEADER, tokens[0]):
                selected_handler = handler
                break

        assert isinstance(selected_handler, GenerationHeaderHandler)

    def test_default_handler_always_handles(self):
        """Default handler accepts any chunk type"""
        default_handler = DefaultChunkHandler()
        tokens = create_token_sequence([
            {'content': 'Any content', 'element_type': 'text'},
        ])

        assert default_handler.can_handle(ChunkType.UNKNOWN, tokens[0])
        assert default_handler.can_handle(ChunkType.GENERATION_HEADER, tokens[0])
        assert default_handler.can_handle(ChunkType.IMAGE, tokens[0])


@pytest.mark.integration
class TestComplexScenarios:
    """Test complex multi-handler scenarios"""

    def test_remarriage_sequence(self):
        """
        INTEGRATION TEST: Full remarriage sequence showing family group ID inheritance.
        This tests the edge case where a parent remarries and children from both
        marriages should share the same family group ID.
        """
        tokens = create_token_sequence([
            # First marriage
            {'content': 'VII.1. Kinderen van Hendrik van Zanten en Jannetje Pieterse', 'element_type': 'sub_title'},
            {'content': 'a. Pieter van Zanten, * 1880', 'element_type': 'text'},
            {'content': 'b. Maria van Zanten, * 1882', 'element_type': 'text'},
            # Second marriage (remarriage)
            {'content': 'Kinderen van Hendrik van Zanten en Anna de Vries:', 'element_type': 'sub_title'},
            {'content': 'c. Jan van Zanten, * 1890', 'element_type': 'text'},
            {'content': 'd. Truus van Zanten, * 1892', 'element_type': 'text'},
        ])

        context = {
            'generation': 'Zevende generatie',
            'family_group': None,
            'family_group_id': None,
            'parents': None
        }
        chunks = []

        # Process first family group header
        handler = FamilyGroupHeaderHandler()
        chunk1, idx = handler.create_chunk(
            ChunkType.FAMILY_GROUP_HEADER, tokens[0], tokens, 0, context, chunks
        )
        chunks.append(chunk1)
        assert context['family_group_id'] == 'VII.1'
        assert context['parents'] == ('Hendrik van Zanten', 'Jannetje Pieterse')

        # Process first marriage children
        handler = IndividualEntryHandler()
        chunk2, idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY, tokens[1], tokens, 1, context, chunks
        )
        chunks.append(chunk2)
        assert chunk2.family_group_id == 'VII.1'
        assert chunk2.parents == ('Hendrik van Zanten', 'Jannetje Pieterse')

        chunk3, idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY, tokens[2], tokens, 2, context, chunks
        )
        chunks.append(chunk3)
        assert chunk3.family_group_id == 'VII.1'

        # Process remarriage header
        handler = FamilyGroupHeaderHandler()
        chunk4, idx = handler.create_chunk(
            ChunkType.FAMILY_GROUP_HEADER, tokens[3], tokens, 3, context, chunks
        )
        chunks.append(chunk4)
        # Family group ID should STILL be VII.1
        assert context['family_group_id'] == 'VII.1'
        # But parents should be updated
        assert context['parents'] == ('Hendrik van Zanten', 'Anna de Vries')

        # Process second marriage children
        chunk5, idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY, tokens[4], tokens, 4, context, chunks
        )
        chunks.append(chunk5)
        # Should inherit same family group ID but with new parents
        assert chunk5.family_group_id == 'VII.1'
        assert chunk5.parents == ('Hendrik van Zanten', 'Anna de Vries')

    def test_reading_order_not_spatial(self):
        """
        INTEGRATION TEST: Verify that handlers trust reading order from OCR,
        not spatial heuristics. This tests the design decision documented in
        IndividualEntryHandler lines 187-189.

        The comment says: "We don't stop on column shifts because DeepSeek-OCR
        preserves reading order. In two-column layouts, the narrative naturally
        flows from left to right column."
        """
        # Create tokens that simulate a two-column layout where bounding boxes
        # shift horizontally, but reading order is preserved by OCR
        tokens = create_token_sequence([
            # Individual entry starts in left column
            {'content': 'a. Pieter van Zanten, * 1850', 'element_type': 'text', 'bbox': (50, 100, 300, 120)},
            # Biographical text continues in left column
            {'content': 'x Maria Jansen, * 1852', 'element_type': 'text', 'bbox': (50, 125, 300, 145)},
            # Text continues in RIGHT column (x-coordinate shifts significantly)
            {'content': '1880: Amsterdam, later moved to Naarden', 'element_type': 'text', 'bbox': (350, 100, 600, 120)},
            # Next individual entry (STOP)
            {'content': 'b. Jan van Zanten, * 1854', 'element_type': 'text', 'bbox': (350, 125, 600, 145)},
        ])

        handler = IndividualEntryHandler()
        context = {
            'generation': 'Derde generatie',
            'family_group': 'III.1. Kinderen van Pieter en Maria',
            'family_group_id': 'III.1',
            'parents': ('Pieter', 'Maria')
        }

        chunk, next_idx = handler.create_chunk(
            ChunkType.INDIVIDUAL_ENTRY,
            tokens[0],
            tokens,
            0,
            context,
            []
        )

        # Should include all 3 tokens, even though bbox x-coordinate shifts
        assert len(chunk.grounding_tokens) == 3
        assert 'Pieter van Zanten' in chunk.content
        assert 'Maria Jansen' in chunk.content
        assert '1880: Amsterdam' in chunk.content  # From right column!
        assert 'b. Jan' not in chunk.content
        assert next_idx == 3
