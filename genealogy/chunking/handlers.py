"""Chunk handlers for processing different chunk types

Each handler encapsulates the logic for processing a specific chunk type,
making _chunk_main_flow much simpler and more maintainable.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .models import ChunkType, GroundingToken, TextChunk
from .parser import FAMILY_GROUP_PATTERN, GENERATION_PATTERN, INDIVIDUAL_ENTRY_PATTERN, REMARRIAGE_FAMILY_GROUP_PATTERN


class ChunkHandler(ABC):
    """Base class for handling specific chunk types"""

    @abstractmethod
    def can_handle(self, chunk_type: ChunkType, token: GroundingToken) -> bool:
        """Check if this handler can process the given chunk type"""
        pass

    @abstractmethod
    def create_chunk(
        self,
        chunk_type: ChunkType,
        token: GroundingToken,
        tokens: List[GroundingToken],
        index: int,
        context: dict,
        chunks: List[TextChunk],
    ) -> Tuple[TextChunk, int]:
        """
        Create a chunk from the token and advance the index.

        Args:
            chunk_type: The detected chunk type
            token: Current token to process
            tokens: All tokens in the flow
            index: Current index in tokens
            context: Current genealogical context (generation, family_group, etc.)
            chunks: List of chunks created so far (for back-references)

        Returns:
            (created_chunk, new_index) - the chunk and the next index to process
        """
        pass


class GenerationHeaderHandler(ChunkHandler):
    """Handles generation headers like 'Tweede generatie'"""

    def can_handle(self, chunk_type: ChunkType, token: GroundingToken) -> bool:
        return chunk_type == ChunkType.GENERATION_HEADER

    def create_chunk(
        self,
        chunk_type: ChunkType,
        token: GroundingToken,
        tokens: List[GroundingToken],
        index: int,
        context: dict,
        chunks: List[TextChunk],
    ) -> Tuple[TextChunk, int]:
        match = GENERATION_PATTERN.search(token.content)
        if match:
            # Extract the matched generation text (e.g., "ZEVENDE GENERATIE")
            context['generation'] = match.group(0)

        chunk = TextChunk(
            chunk_type=ChunkType.GENERATION_HEADER,
            content=token.content,
            grounding_tokens=[token],
            generation=context['generation'],
        )

        return chunk, index + 1


class FamilyGroupHeaderHandler(ChunkHandler):
    """Handles family group headers like 'II.2 Kinderen van X en Y'"""

    def can_handle(self, chunk_type: ChunkType, token: GroundingToken) -> bool:
        return chunk_type == ChunkType.FAMILY_GROUP_HEADER

    def create_chunk(
        self,
        chunk_type: ChunkType,
        token: GroundingToken,
        tokens: List[GroundingToken],
        index: int,
        context: dict,
        chunks: List[TextChunk],
    ) -> Tuple[TextChunk, int]:
        # Strip markdown hashes for parsing
        content_stripped = token.content.lstrip('#').strip()

        # Try standard pattern first (with roman numeral)
        match = FAMILY_GROUP_PATTERN.match(content_stripped)
        if match:
            # New family group with roman numeral
            context['family_group_id'] = match.group(1)
            father = match.group(2).strip() if match.group(2) else None
            mother = match.group(3).strip() if match.group(3) else None
            context['parents'] = (father, mother) if father and mother else (father,) if father else None
            context['family_group'] = content_stripped
        else:
            # Try remarriage pattern (no roman numeral - inherit family group ID)
            remarriage_match = REMARRIAGE_FAMILY_GROUP_PATTERN.match(content_stripped)
            if remarriage_match:
                # Keep same family_group_id (e.g., "VII.1"), but update parents
                father = remarriage_match.group(1).strip() if remarriage_match.group(1) else None
                mother = remarriage_match.group(2).strip() if remarriage_match.group(2) else None
                context['parents'] = (father, mother) if father and mother else (father,) if father else None
                # Don't update family_group_id - it inherits from previous marriage
                context['family_group'] = content_stripped

        chunk = TextChunk(
            chunk_type=ChunkType.FAMILY_GROUP_HEADER,
            content=token.content,
            grounding_tokens=[token],
            generation=context['generation'],
            family_group=context['family_group'],
            family_group_id=context['family_group_id'],
            parents=context['parents'],
        )

        return chunk, index + 1


class SourceCitationSectionHandler(ChunkHandler):
    """Handles source citation sections (Bronverwijzing / Source indication)"""

    def can_handle(self, chunk_type: ChunkType, token: GroundingToken) -> bool:
        return chunk_type == ChunkType.SOURCE_CITATION and token.element_type == 'sub_title'

    def create_chunk(
        self,
        chunk_type: ChunkType,
        token: GroundingToken,
        tokens: List[GroundingToken],
        index: int,
        context: dict,
        chunks: List[TextChunk],
    ) -> Tuple[TextChunk, int]:
        from .parser import detect_chunk_type

        # Collect all tokens in the citation section (until next structural boundary)
        citation_section_tokens = [token]  # Include the header
        j = index + 1
        while j < len(tokens):
            next_token = tokens[j]
            next_type = detect_chunk_type(next_token)

            # Stop at next structural element
            if next_token.element_type == 'sub_title':
                break

            # Stop at individual entries (they're not part of the citation)
            if next_type == ChunkType.INDIVIDUAL_ENTRY:
                break

            # Stop at family group or generation headers
            if next_type in (ChunkType.FAMILY_GROUP_HEADER, ChunkType.GENERATION_HEADER):
                break

            # Stop at biographical text that looks like an individual entry
            if next_type == ChunkType.BIOGRAPHICAL_TEXT:
                if re.match(r'^[A-Z]\.', next_token.content):
                    break

            # Include all text content
            if next_token.element_type == 'text':
                citation_section_tokens.append(next_token)
                j += 1
            else:
                break

        # Create one chunk for the entire citation section
        citation_content = '\n\n'.join(t.content for t in citation_section_tokens)

        # Find the most recent individual entry to link to
        supports_index = None
        for k in range(len(chunks) - 1, -1, -1):
            if chunks[k].chunk_type == ChunkType.INDIVIDUAL_ENTRY:
                supports_index = k
                break

        chunk = TextChunk(
            chunk_type=ChunkType.SOURCE_CITATION,
            content=citation_content,
            grounding_tokens=citation_section_tokens,
            generation=context['generation'],
            family_group=context['family_group'],
            family_group_id=context['family_group_id'],
            parents=None,  # Don't include parents for source citations
            supports_chunk_index=supports_index,
        )

        return chunk, j


class IndividualEntryHandler(ChunkHandler):
    """Handles individual entries like 'a. Name, *date, †date'"""

    def can_handle(self, chunk_type: ChunkType, token: GroundingToken) -> bool:
        return chunk_type == ChunkType.INDIVIDUAL_ENTRY

    def create_chunk(
        self,
        chunk_type: ChunkType,
        token: GroundingToken,
        tokens: List[GroundingToken],
        index: int,
        context: dict,
        chunks: List[TextChunk],
    ) -> Tuple[TextChunk, int]:
        from .parser import detect_chunk_type, extract_person_from_individual_entry

        match = INDIVIDUAL_ENTRY_PATTERN.match(token.content)
        if match:
            marker = match.group(1)
            # Normalize OCR error: "I" -> "l"
            if marker == 'I':
                marker = 'l'
            individual_marker = marker + '.'
        else:
            individual_marker = None

        # Track the baseline x1 coordinate for this entry
        baseline_x1 = token.bbox.x1

        # Collect following biographical text and sources
        entry_tokens = [token]
        j = index + 1
        while j < len(tokens):
            next_token = tokens[j]
            next_type = detect_chunk_type(next_token)

            # Stop at next individual entry or family group
            if next_type in (ChunkType.INDIVIDUAL_ENTRY, ChunkType.FAMILY_GROUP_HEADER,
                            ChunkType.GENERATION_HEADER):
                break

            # Stop at source citation sections (sub_title)
            if next_type == ChunkType.SOURCE_CITATION and next_token.element_type == 'sub_title':
                break

            # Stop if we hit text from a significantly different column
            if next_token.element_type == 'text':
                x1_diff = abs(next_token.bbox.x1 - baseline_x1)
                if x1_diff > 100:  # Different column (coordinates are in 1000 bins)
                    break

            # Include biographical text, narrative context, info box headers (sub-sections),
            # and inline source citations (text-level only)
            if next_type in (ChunkType.BIOGRAPHICAL_TEXT, ChunkType.NARRATIVE_CONTEXT, ChunkType.INFO_BOX):
                entry_tokens.append(next_token)
                j += 1
            elif next_type == ChunkType.SOURCE_CITATION and next_token.element_type == 'text':
                entry_tokens.append(next_token)
                j += 1
            else:
                break

        entry_content = '\n\n'.join(t.content for t in entry_tokens)

        chunk = TextChunk(
            chunk_type=ChunkType.INDIVIDUAL_ENTRY,
            content=entry_content,
            grounding_tokens=entry_tokens,
            generation=context['generation'],
            family_group=context['family_group'],
            family_group_id=context['family_group_id'],
            parents=context['parents'],
            individual_marker=individual_marker,
        )

        # Extract entities (Phase 1: deterministic)
        if chunk.generation:
            person_name = extract_person_from_individual_entry(chunk.content)
            if person_name:
                chunk.extracted_people.append(person_name)

                # Also extract parent names and create parent-child relationships
                if chunk.parents:
                    for parent in chunk.parents:
                        if parent:
                            chunk.extracted_people.append(parent)
                            chunk.extracted_relationships.append({
                                "person1": parent,
                                "relationship_type": "parent",
                                "person2": person_name
                            })

        return chunk, j


class StandaloneSourceCitationHandler(ChunkHandler):
    """Handles standalone source citations (not part of individual entries)"""

    def can_handle(self, chunk_type: ChunkType, token: GroundingToken) -> bool:
        return chunk_type == ChunkType.SOURCE_CITATION

    def create_chunk(
        self,
        chunk_type: ChunkType,
        token: GroundingToken,
        tokens: List[GroundingToken],
        index: int,
        context: dict,
        chunks: List[TextChunk],
    ) -> Tuple[TextChunk, int]:
        chunk = TextChunk(
            chunk_type=ChunkType.SOURCE_CITATION,
            content=token.content,
            grounding_tokens=[token],
            generation=context['generation'],
            family_group=context['family_group'],
            family_group_id=context['family_group_id'],
            parents=context['parents'],
            supports_chunk_index=len(chunks) - 1 if chunks else None,
        )

        return chunk, index + 1


class DefaultChunkHandler(ChunkHandler):
    """Handles all other chunk types (fallback)"""

    def can_handle(self, chunk_type: ChunkType, token: GroundingToken) -> bool:
        return True  # Always handles as fallback

    def create_chunk(
        self,
        chunk_type: ChunkType,
        token: GroundingToken,
        tokens: List[GroundingToken],
        index: int,
        context: dict,
        chunks: List[TextChunk],
    ) -> Tuple[TextChunk, int]:
        chunk = TextChunk(
            chunk_type=chunk_type,
            content=token.content,
            grounding_tokens=[token],
            generation=context['generation'],
            family_group=context['family_group'],
            family_group_id=context['family_group_id'],
            parents=context['parents'],
        )

        return chunk, index + 1


# Handler registry - order matters! More specific handlers first
CHUNK_HANDLERS = [
    GenerationHeaderHandler(),
    FamilyGroupHeaderHandler(),
    SourceCitationSectionHandler(),
    IndividualEntryHandler(),
    StandaloneSourceCitationHandler(),
    DefaultChunkHandler(),  # Must be last (fallback)
]
