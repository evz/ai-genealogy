"""Database persistence for text chunks"""

import logging
from typing import List, Tuple

from django.db import models, transaction

from genealogy.utils import parse_name

from .models import ChunkType, GroundingToken, TextChunk

logger = logging.getLogger(__name__)


def _get_page_numbers_from_tokens(
    grounding_tokens: List[GroundingToken],
    book_text: str,
    page_map: List[dict]
) -> Tuple[int, int]:
    """
    Determine start and end page numbers for a chunk based on its grounding tokens.

    Args:
        grounding_tokens: List of grounding tokens in the chunk
        book_text: Full concatenated book text
        page_map: List of dicts with page_number, start_char, end_char

    Returns:
        Tuple of (start_page, end_page)
    """
    if not grounding_tokens or not page_map:
        return (1, 1)  # Default fallback

    # Find character positions of first and last tokens in book_text
    first_token = grounding_tokens[0]
    last_token = grounding_tokens[-1]

    # Search for the token content in book_text
    # Note: This is approximate since token content might appear multiple times
    # A better approach would be to track character offsets during tokenization
    first_pos = book_text.find(first_token.content)
    last_pos = book_text.find(last_token.content)

    if first_pos == -1 or last_pos == -1:
        return (1, 1)  # Fallback if not found

    # Find which pages these positions correspond to
    start_page = 1
    end_page = 1

    for page_info in page_map:
        if page_info['start_char'] <= first_pos < page_info['end_char']:
            start_page = page_info['page_number']
        if page_info['start_char'] <= last_pos < page_info['end_char']:
            end_page = page_info['page_number']

    return (start_page, end_page)


def save_chunks_to_db(
    chunks: List[TextChunk],
    document,
    page_map: List[dict],
    book_text: str,
    start_sequence=None
):
    """
    Save TextChunk dataclass instances to the database.

    Args:
        chunks: List of TextChunk dataclass instances from chunking strategies
        document: Document model instance
        page_map: List of dicts mapping character positions to pages
        book_text: Full book text (needed to map grounding tokens to character positions)
        start_sequence: Starting sequence number (if None, will auto-calculate from existing chunks)

    Returns:
        List of saved TextChunk database model instances
    """
    from ..models import TextChunk as TextChunkModel

    # Calculate starting sequence number if not provided
    if start_sequence is None:
        max_seq = TextChunkModel.objects.filter(document=document).aggregate(
            max_seq=models.Max('sequence_number')
        )['max_seq']
        start_sequence = (max_seq or 0) + 1

    saved_chunks = []

    # Filter out empty chunks (no content) and create index mapping
    # Map old indices (in chunks list) to new indices (in non_empty_chunks list)
    old_to_new_index = {}
    non_empty_chunks = []
    for old_idx, chunk in enumerate(chunks):
        if chunk.content and chunk.content.strip():
            new_idx = len(non_empty_chunks)
            old_to_new_index[old_idx] = new_idx
            non_empty_chunks.append(chunk)

    if len(chunks) != len(non_empty_chunks):
        logger.info(f"Filtered out {len(chunks) - len(non_empty_chunks)} empty chunks")

    # Update supports_chunk_index to use new indices after filtering
    for chunk in non_empty_chunks:
        if chunk.supports_chunk_index is not None:
            # Map old index to new index
            if chunk.supports_chunk_index in old_to_new_index:
                chunk.supports_chunk_index = old_to_new_index[chunk.supports_chunk_index]
            else:
                # The chunk it pointed to was filtered out (empty), so clear the link
                chunk.supports_chunk_index = None

    # Dutch generation names to Roman numerals
    GENERATION_MAPPING = {
        "Eerste generatie": 1,
        "Tweede generatie": 2,
        "Derde generatie": 3,
        "DERDE generatie": 3,
        "Vierde generatie": 4,
        "ierde generatie": 4,  # OCR corruption: missing 'V' at start
        "Vijfde generatie": 5,
        "Zesde generatie": 6,
        "Zevende generatie": 7,
        "Achtste generatie": 8,
        "Negende generatie": 9,
        "Tiende generatie": 10,
        "Elfde generatie": 11,
        "Twaalfde generatie": 12,
    }

    for i, chunk in enumerate(non_empty_chunks, start=start_sequence):
        # Use enum value directly as chunk type
        db_chunk_type = chunk.chunk_type.value

        # Extract generation header text if this is a generation header
        generation_header = ""
        if chunk.chunk_type == ChunkType.GENERATION_HEADER:
            generation_header = chunk.content

        # Convert generation text to number (case-insensitive)
        generation_number = None
        if chunk.generation:
            # Try case-insensitive lookup
            generation_number = GENERATION_MAPPING.get(chunk.generation.capitalize())

        # Build family groups list
        # Include for family group headers AND individual entries (they inherit from context)
        family_groups = []
        if chunk.family_group:
            family_groups = [chunk.family_group]

        # Determine page numbers from grounding tokens
        start_page, end_page = _get_page_numbers_from_tokens(chunk.grounding_tokens, book_text, page_map)

        # Determine related_genealogy_entry link for source citations
        related_entry = None
        if chunk.chunk_type == ChunkType.SOURCE_CITATION and chunk.supports_chunk_index is not None:
            # The supports_chunk_index is relative to the non_empty_chunks list
            # Find the corresponding saved chunk
            if 0 <= chunk.supports_chunk_index < len(saved_chunks):
                related_entry = saved_chunks[chunk.supports_chunk_index]

        # Build genealogical identifier for INDIVIDUAL_ENTRY chunks
        # Format: generation.family_group_id.individual_marker (e.g., "II.2.a")
        genealogical_identifier = None
        if chunk.chunk_type == ChunkType.INDIVIDUAL_ENTRY:
            if chunk.family_group_id and chunk.individual_marker:
                # Extract the letter from the marker (e.g., "a." -> "a")
                marker_letter = chunk.individual_marker.rstrip('.')
                genealogical_identifier = f"{chunk.family_group_id}.{marker_letter}"

        # Create PersonMention for the subject if this is an individual entry
        # ONLY create if we have a valid genealogical_identifier (genealogical_id is NOT NULL in DB)
        primary_person_mention = None
        if chunk.chunk_type == ChunkType.INDIVIDUAL_ENTRY and chunk.subject and genealogical_identifier:
            from ..models import PersonMention as PersonMentionModel

            # Parse name into given_names and surname
            given_names, surname = parse_name(chunk.subject)

            # Validate name quality:
            # 1. MUST have both given_names and surname, OR
            # 2. If only given_names, MUST have genealogical_identifier (provides context)
            has_both_names = given_names.strip() and surname.strip()
            has_contextual_given_name = given_names.strip() and genealogical_identifier

            if has_both_names or has_contextual_given_name:
                # Create PersonMention with genealogical_id
                primary_person_mention = PersonMentionModel.objects.create(
                    given_names=given_names,
                    surname=surname,
                    generation=generation_number,
                    genealogical_id=genealogical_identifier,
                )
                # Link to document (will link to chunk after chunk is created)
                primary_person_mention.source_documents.add(document)

                logger.debug(f"Created PersonMention for {chunk.subject} with genealogical_id={genealogical_identifier}")
            else:
                logger.warning(
                    f"Skipping PersonMention creation for chunk {i}: subject '{chunk.subject}' "
                    f"lacks sufficient name information (given_names='{given_names}', surname='{surname}')"
                )

        # Create database chunk
        db_chunk = TextChunkModel.objects.create(
            document=document,
            text_content=chunk.content,
            chunk_type=db_chunk_type,
            start_page=start_page,
            end_page=end_page,
            sequence_number=i,
            generation_number=generation_number,
            generation_header=generation_header,
            family_groups=family_groups,
            related_genealogy_entry=related_entry,
            extraction_method="regex",
            # Save extracted entities from Phase 1 (deterministic extraction)
            extracted_people=chunk.extracted_people,
            extracted_relationships=chunk.extracted_relationships,
            extracted_events=chunk.extracted_events,
            # Keep entities_extracted=False so Phase 2 (LLM) will still process this chunk
            entities_extracted=False,
            # Subject tracking (for INDIVIDUAL_ENTRY chunks)
            subject=chunk.subject,
            genealogical_identifier=genealogical_identifier,
            primary_person_mention=primary_person_mention,
        )
        saved_chunks.append(db_chunk)

        # Link PersonMention back to chunk
        if primary_person_mention:
            primary_person_mention.source_chunks.add(db_chunk)

    return saved_chunks
