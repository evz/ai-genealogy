"""Database persistence for text chunks"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from django.db import models, transaction

from genealogy.models import (Identity, MentionToIdentity,
                              PartnershipMention, RelationshipMention,
                              PersonMention as PersonMentionModel,
                              TextChunk as TextChunkModel)
from genealogy.utils import parse_family_group_header, parse_name

from .models import ChunkType, GroundingToken, TextChunk

logger = logging.getLogger(__name__)


def _create_person_mention_with_identity(
    given_names: str,
    surname: str,
    generation: Optional[int],
    genealogical_id: Optional[str],
    document
) -> Tuple[Optional[PersonMentionModel], Optional[Identity]]:
    """
    Create a PersonMention and its associated Identity.

    Returns:
        Tuple of (PersonMention, Identity) or (None, None) if validation fails
    """
    # Validate name quality
    has_both_names = given_names.strip() and surname.strip()
    has_contextual_given_name = given_names.strip() and genealogical_id

    if not (has_both_names or has_contextual_given_name):
        logger.warning(
            f"Skipping PersonMention creation: insufficient name information "
            f"(given_names='{given_names}', surname='{surname}', genealogical_id={genealogical_id})"
        )
        return None, None

    # Create PersonMention
    person_mention = PersonMentionModel.objects.create(
        given_names=given_names,
        surname=surname,
        generation=generation,
        genealogical_id=genealogical_id,
    )
    person_mention.source_documents.add(document)

    # Create singleton Identity
    identity = Identity.objects.create(
        display_name=person_mention.full_name,
        genealogical_identifier=genealogical_id,
        notes=f"Auto-created during chunking for {person_mention.full_name}"
    )

    # Create mapping
    MentionToIdentity.objects.create(
        mention=person_mention,
        identity=identity,
        mapped_by="CHUNKING"
    )

    logger.debug(f"Created PersonMention + Identity for {person_mention.full_name} (genealogical_id={genealogical_id})")

    return person_mention, identity


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

        # Create PersonMention + Identity for the subject if this is an individual entry
        # ONLY create if we have a valid genealogical_identifier (genealogical_id is NOT NULL in DB)
        primary_person_mention = None
        if chunk.chunk_type == ChunkType.INDIVIDUAL_ENTRY and chunk.subject and genealogical_identifier:
            # Parse name into given_names and surname
            given_names, surname = parse_name(chunk.subject)

            # Create PersonMention + Identity for subject
            primary_person_mention, _ = _create_person_mention_with_identity(
                given_names=given_names,
                surname=surname,
                generation=generation_number,
                genealogical_id=genealogical_identifier,
                document=document
            )

            # If subject was created, also create parents, partnership, and relationships
            if primary_person_mention and family_groups:
                parent_names, first_parent_gen_id = parse_family_group_header(family_groups)

                if len(parent_names) >= 2:
                    parent_mentions = []
                    parent1 = None
                    parent2 = None

                    # Process first parent
                    parent1_given, parent1_surname = parse_name(parent_names[0])
                    parent_generation = generation_number - 1 if generation_number else None

                    # If first parent has genealogical_id, find existing PersonMention
                    if first_parent_gen_id:
                        parent1 = PersonMentionModel.objects.filter(
                            genealogical_id=first_parent_gen_id
                        ).first()

                        if parent1:
                            # Link existing parent to this document
                            parent1.source_documents.add(document)
                            logger.debug(f"Found existing parent1: {parent1.full_name} ({first_parent_gen_id})")

                    # If not found, create new PersonMention + Identity
                    if not parent1:
                        parent1, _ = _create_person_mention_with_identity(
                            given_names=parent1_given,
                            surname=parent1_surname,
                            generation=parent_generation,
                            genealogical_id=first_parent_gen_id,
                            document=document
                        )

                    if parent1:
                        parent_mentions.append(parent1)

                    # Process second parent
                    parent2_given, parent2_surname = parse_name(parent_names[1])

                    # Try to find second parent in existing partnership with parent1
                    if first_parent_gen_id:
                        # Find partnerships involving parent1
                        existing_partnerships = PartnershipMention.objects.filter(
                            partners=parent1
                        )

                        # Look for a partner with matching name
                        for partnership in existing_partnerships:
                            potential_parent2 = partnership.partners.exclude(id=parent1.id).filter(
                                given_names=parent2_given,
                                surname=parent2_surname
                            ).first()

                            if potential_parent2:
                                parent2 = potential_parent2
                                parent2.source_documents.add(document)
                                logger.debug(f"Found existing parent2 via partnership: {parent2.full_name}")
                                break

                    # If not found in partnership, create new PersonMention + Identity
                    if not parent2:
                        parent2, _ = _create_person_mention_with_identity(
                            given_names=parent2_given,
                            surname=parent2_surname,
                            generation=parent_generation,
                            genealogical_id=None,  # Second parent doesn't have genealogical_id
                            document=document
                        )

                    if parent2:
                        parent_mentions.append(parent2)

                    # If we successfully created both parents, create partnership and relationships
                    if len(parent_mentions) == 2:
                        parent1, parent2 = parent_mentions

                        # Create PartnershipMention between parents
                        # Check if partnership already exists
                        existing_partnership = PartnershipMention.objects.filter(
                            partners=parent1
                        ).filter(
                            partners=parent2
                        ).first()

                        if not existing_partnership:
                            partnership = PartnershipMention.objects.create(
                                partnership_type='MARRIAGE'
                            )
                            partnership.partners.add(parent1, parent2)
                            partnership.source_documents.add(document)
                            logger.debug(f"Created PartnershipMention: {parent1.full_name} & {parent2.full_name}")

                        # Create RelationshipMentions (parent -> child)
                        for parent_mention in parent_mentions:
                            RelationshipMention.objects.get_or_create(
                                child_mention=primary_person_mention,
                                parent_mention=parent_mention,
                                defaults={'relationship_type': 'BIOLOGICAL'}
                            )
                            logger.debug(f"Created RelationshipMention: {parent_mention.full_name} -> {primary_person_mention.full_name}")

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
