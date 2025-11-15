"""Descendant genealogy extraction strategy

Two-phase extraction approach:
- Phase 1 (during chunking): Deterministic extraction of people and relationships from genealogical IDs
- Phase 2 (LLM): Extract ONLY events (people/relationships already done in Phase 1)
"""

import logging
from typing import Any, Dict

from genealogy.prompts.extraction import (build_extraction_prompt,
                                          parse_extraction_output)

from .base import ExtractionStrategy

logger = logging.getLogger(__name__)


class DescendantGenealogyStrategy(ExtractionStrategy):
    """
    Strategy for descendant genealogy sections.

    Uses a two-phase approach:
    - Phase 1 (chunking): Extracts people and relationships from genealogical IDs
    - Phase 2 (LLM): Extracts ONLY life events from text content
    """

    def __init__(self):
        """Initialize strategy"""
        pass

    @property
    def strategy_name(self) -> str:
        return "Descendant Genealogy Extraction"

    def should_process(self, chunk) -> bool:
        """Process individual_entry chunks that have genealogical context"""
        return chunk.chunk_type == "individual_entry"

    def get_chunk_filter(self) -> Dict[str, Any]:
        """Filter for individual_entry chunks only"""
        return {"chunk_type": "individual_entry"}

    def extract(self, chunk, ollama, model: str) -> Dict[str, Any]:
        """
        Extract life events from a descendant genealogy chunk using LLM.

        Phase 1 data (people, relationships) is already in the chunk from genealogical IDs.
        Phase 2 (this method) extracts ONLY life events - we don't need people/relationships from LLM.
        """
        try:
            logger.info(f"Extracting from chunk {chunk.sequence_number} (Descendant Genealogy)")

            # Build extraction prompt using shared utility
            prompt = build_extraction_prompt(chunk)

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

            # Phase 1 (chunking) already extracted people and relationships from genealogical IDs
            # Phase 2 (LLM) only extracts events - no merging needed for people/relationships
            phase1_people = chunk.extracted_people or []
            phase1_relationships = chunk.extracted_relationships or []
            extracted_events = extracted_data['events']

            # Save event data (people and relationships remain unchanged from Phase 1)
            chunk.extracted_events = extracted_events
            chunk.entities_extracted = True
            chunk.save(update_fields=[
                'extracted_events',
                'entities_extracted'
            ])

            result = {
                'success': True,
                'people_count': len(phase1_people),
                'relationships_count': len(phase1_relationships),
                'events_count': len(extracted_events),
            }

            logger.info(
                f"Chunk {chunk.sequence_number}: "
                f"{result['people_count']} people (from gen IDs), "
                f"{result['relationships_count']} relationships (from gen IDs), "
                f"{result['events_count']} events (from LLM)"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk.sequence_number}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
