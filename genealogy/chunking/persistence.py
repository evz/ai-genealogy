"""Database persistence for text chunks"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from django.db import models, transaction

from genealogy.models import TextChunk as TextChunkModel

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

    SIMPLIFIED VERSION: Only creates TextChunk records with metadata.
    Entity creation (Person, Relationship, etc.) will be done by build_genealogy_graph task.

    Args:
        chunks: List of TextChunk dataclass instances from chunking strategies
        document: Document model instance
        page_map: List of dicts mapping character positions to pages
        book_text: Full book text (needed to map grounding tokens to character positions)
        start_sequence: Starting sequence number (if None, will auto-calculate from existing chunks)

    Returns:
        List of saved TextChunk database model instances
    """
    # Calculate starting sequence number if not provided
    if start_sequence is None:
        max_seq = TextChunkModel.objects.filter(document=document).aggregate(
            max_seq=models.Max('sequence_number')
        )['max_seq']
        start_sequence = (max_seq or 0) + 1

    saved_chunks = []

    # Filter out empty chunks (no content) and create index mapping
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
            if chunk.supports_chunk_index in old_to_new_index:
                chunk.supports_chunk_index = old_to_new_index[chunk.supports_chunk_index]
            else:
                chunk.supports_chunk_index = None

    # Dutch generation names to Roman numerals
    GENERATION_MAPPING = {
        "Eerste generatie": 1,
        "Tweede generatie": 2,
        "Derde generatie": 3,
        "DERDE generatie": 3,
        "Vierde generatie": 4,
        "ierde generatie": 4,
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
        db_chunk_type = chunk.chunk_type.value

        generation_header = ""
        if chunk.chunk_type == ChunkType.GENERATION_HEADER:
            generation_header = chunk.content

        generation_number = None
        if chunk.generation:
            generation_number = GENERATION_MAPPING.get(chunk.generation.capitalize())

        family_groups = []
        if chunk.family_group:
            family_groups = [chunk.family_group]

        start_page, end_page = _get_page_numbers_from_tokens(chunk.grounding_tokens, book_text, page_map)

        related_entry = None
        if chunk.chunk_type == ChunkType.SOURCE_CITATION and chunk.supports_chunk_index is not None:
            if 0 <= chunk.supports_chunk_index < len(saved_chunks):
                related_entry = saved_chunks[chunk.supports_chunk_index]

        genealogical_identifier = None
        if chunk.chunk_type == ChunkType.INDIVIDUAL_ENTRY:
            if chunk.family_group_id and chunk.individual_marker:
                marker_letter = chunk.individual_marker.rstrip('.')
                genealogical_identifier = f"{chunk.family_group_id}.{marker_letter}"

        # NO ENTITY CREATION - just store metadata in TextChunk
        # Person/Relationship/Event creation done by build_genealogy_graph and persist_entities tasks

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
            # No primary_person link yet - will be set by build_genealogy_graph
        )
        saved_chunks.append(db_chunk)

    return saved_chunks


def save_chunk_enrichment(
    chunk: TextChunkModel,
    embedding: Optional[List[float]] = None,
    dm_codes: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Save enrichment data (embeddings and/or DM codes) to a chunk.

    Args:
        chunk: TextChunk model instance to update
        embedding: Vector embedding to save (or None to skip)
        dm_codes: List of DM codes to save (or None to skip)

    Returns:
        dict with:
            - embedding_saved: bool
            - dm_codes_saved: bool
    """
    result = {
        "embedding_saved": False,
        "dm_codes_saved": False,
    }

    update_fields = []

    if embedding is not None:
        chunk.embedding = embedding
        update_fields.append("embedding")
        result["embedding_saved"] = True

    if dm_codes is not None:
        chunk.dm_codes = dm_codes
        update_fields.append("dm_codes")
        result["dm_codes_saved"] = True

    if update_fields:
        chunk.save(update_fields=update_fields)

    return result


@transaction.atomic
def save_chunk_enrichments_batch(
    chunks_with_enrichments: List[Tuple[TextChunkModel, Optional[List[float]], Optional[List[str]]]]
) -> Dict[str, int]:
    """
    Save enrichments for multiple chunks in a transaction.

    Args:
        chunks_with_enrichments: List of tuples (chunk, embedding, dm_codes)

    Returns:
        dict with:
            - embeddings_saved: int
            - dm_codes_saved: int
            - total_chunks: int
    """
    embeddings_saved = 0
    dm_codes_saved = 0

    for chunk, embedding, dm_codes in chunks_with_enrichments:
        result = save_chunk_enrichment(chunk, embedding, dm_codes)
        if result["embedding_saved"]:
            embeddings_saved += 1
        if result["dm_codes_saved"]:
            dm_codes_saved += 1

    return {
        "embeddings_saved": embeddings_saved,
        "dm_codes_saved": dm_codes_saved,
        "total_chunks": len(chunks_with_enrichments),
    }
