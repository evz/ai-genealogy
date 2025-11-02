"""Descendant genealogy extraction strategy

Two-phase extraction approach:
- Phase 1 (during chunking): Deterministic extraction of people and parent-child relationships
- Phase 2 (LLM): Extract events, partnerships, and additional people
"""

import logging
from typing import Dict, Any

from genealogy.prompts.extraction import (
    build_extraction_prompt,
    load_examples,
    parse_extraction_output,
)
from .base import ExtractionStrategy

logger = logging.getLogger(__name__)


class DescendantGenealogyStrategy(ExtractionStrategy):
    """
    Strategy for descendant genealogy sections.

    Extracts genealogical facts from formatted genealogical entries using a
    complementary two-phase approach.
    """

    def __init__(self):
        """Initialize strategy and load examples"""
        self._examples = None

    @property
    def strategy_name(self) -> str:
        return "Descendant Genealogy Extraction"

    def should_process(self, chunk) -> bool:
        """Process GENEALOGY_ENTRY chunks that have genealogical context"""
        return chunk.chunk_type == "GENEALOGY_ENTRY"

    def get_chunk_filter(self) -> Dict[str, Any]:
        """Filter for GENEALOGY_ENTRY chunks only"""
        return {"chunk_type": "GENEALOGY_ENTRY"}

    def extract(self, chunk, ollama, model: str) -> Dict[str, Any]:
        """
        Extract entities from a descendant genealogy chunk using LLM.

        Phase 1 data (people, parent-child relationships) is already in the chunk.
        Phase 2 (this method) extracts events, partnerships, and additional people.
        """
        try:
            logger.info(f"Extracting from chunk {chunk.sequence_number} (Descendant Genealogy)")

            # Load examples if not already loaded
            if self._examples is None:
                self._examples = load_examples()

            # Build extraction prompt using shared utility
            prompt = build_extraction_prompt(chunk, self._examples)

            # Calculate required context window based on chunk size
            estimated_tokens = len(chunk.text_content) // 4 + 4000
            num_ctx = max(4096, min(131072, 2 ** (estimated_tokens - 1).bit_length()))

            if num_ctx > 8192:
                logger.info(f"Large chunk: using context window {num_ctx} tokens")

            # Query LLM
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'num_ctx': num_ctx,
                    'temperature': 0.0,
                }
            )

            if not response:
                logger.error(f"No response from LLM for chunk {chunk.sequence_number}")
                return {
                    'success': False,
                    'error': 'No response from LLM'
                }

            # Parse output using shared utility
            extracted_data = parse_extraction_output(response)

            # MERGE Phase 2 (LLM) data WITH Phase 1 (deterministic) data
            phase1_people = chunk.extracted_people or []
            phase2_people = extracted_data['people']
            all_people = phase1_people.copy()
            for person in phase2_people:
                if person not in all_people:
                    all_people.append(person)

            # Merge relationships
            phase1_relationships = chunk.extracted_relationships or []
            phase2_parent_child = extracted_data['parent_child']
            phase2_partnerships = extracted_data['partnerships']

            all_relationships = phase1_relationships.copy()
            for rel in phase2_parent_child + phase2_partnerships:
                # Check for duplicates
                is_duplicate = any(
                    existing['person1'] == rel['person1'] and
                    existing['relationship_type'] == rel['relationship_type'] and
                    existing['person2'] == rel['person2']
                    for existing in all_relationships
                )
                if not is_duplicate:
                    all_relationships.append(rel)

            # Events are new (Phase 1 doesn't extract events)
            all_events = extracted_data['events']

            # Save merged data
            chunk.extracted_people = all_people
            chunk.extracted_relationships = all_relationships
            chunk.extracted_events = all_events
            chunk.entities_extracted = True
            chunk.save(update_fields=[
                'extracted_people',
                'extracted_relationships',
                'extracted_events',
                'entities_extracted'
            ])

            result = {
                'success': True,
                'people_count': len(all_people),
                'relationships_count': len(all_relationships),
                'events_count': len(all_events),
                'phase1_people': len(phase1_people),
                'phase2_people_added': len(all_people) - len(phase1_people),
                'phase1_relationships': len(phase1_relationships),
                'phase2_relationships_added': len(all_relationships) - len(phase1_relationships),
            }

            logger.info(
                f"Chunk {chunk.sequence_number}: "
                f"{result['people_count']} people (P1:{result['phase1_people']} + P2:{result['phase2_people_added']}), "
                f"{result['relationships_count']} rels (P1:{result['phase1_relationships']} + P2:{result['phase2_relationships_added']}), "
                f"{result['events_count']} events"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk.sequence_number}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
