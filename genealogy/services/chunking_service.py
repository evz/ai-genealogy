"""Chunking service - pure business logic (no Django/Celery dependencies)

This service encapsulates the business logic for chunking OCR text into
semantic units. It's independent of Django models and Celery tasks, making
it easier to test and reuse.
"""

import logging
from typing import List, Dict, Any

from ..chunking_strategies import get_chunking_strategy
from ..chunking.persistence import save_chunks_to_db

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for chunking OCR pages into text chunks"""

    def chunk_section(
        self,
        section_type: str,
        section_text: str,
        document,
        page_map: List[Dict[str, Any]],
        start_sequence: int = 1
    ) -> Dict[str, Any]:
        """Chunk a book section using appropriate strategy

        Args:
            section_type: Type of section (DESCENDANT_GENEALOGY, etc.)
            section_text: Concatenated OCR text for the section
            document: Document model instance
            page_map: List of dicts mapping character positions to pages
            start_sequence: Starting sequence number for chunks

        Returns:
            dict with:
                - success: bool
                - chunks_created: int (if successful)
                - saved_chunks: list (if successful)
                - error: str (if failed)
        """
        try:
            # Get the chunking strategy for this section type
            strategy = get_chunking_strategy(section_type)
            logger.info(f"Using chunking strategy: {strategy.strategy_name}")

            # Chunk using the strategy
            chunks = strategy.chunk_section(section_text, document, page_map)

            # Save chunks to database
            saved_chunks = save_chunks_to_db(
                chunks,
                document,
                page_map,
                section_text,
                start_sequence=start_sequence
            )

            return {
                'success': True,
                'chunks_created': len(saved_chunks),
                'saved_chunks': saved_chunks,
            }

        except KeyError as e:
            error_msg = f"Unknown section type: {section_type}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'chunks_created': 0,
            }

        except Exception as e:
            error_msg = f"Chunking failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'chunks_created': 0,
            }

    def should_process_section(self, section_type: str, section) -> bool:
        """Check if a section should be processed

        Args:
            section_type: Type of section
            section: BookSection instance

        Returns:
            bool: True if section should be processed
        """
        try:
            strategy = get_chunking_strategy(section_type)
            return strategy.should_process(section)
        except KeyError:
            logger.error(f"Unknown section type: {section_type}")
            return False
