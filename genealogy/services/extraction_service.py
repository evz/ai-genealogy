"""Extraction service - pure business logic (no Django/Celery dependencies)

This service encapsulates the business logic for extracting genealogical
entities from text chunks. It's independent of Django models and Celery tasks,
making it easier to test and reuse.
"""

import logging
from typing import Any, Dict

from ..extraction_strategies import get_strategy

logger = logging.getLogger(__name__)


class ExtractionService:
    """Service for extracting entities from text chunks"""

    def __init__(self, ollama_client):
        """Initialize with Ollama client

        Args:
            ollama_client: OllamaClient instance for LLM calls
        """
        self.ollama = ollama_client

    def extract_from_chunk(
        self,
        chunk,
        section_type: str,
        model: str
    ) -> Dict[str, Any]:
        """Extract entities from a single chunk

        Args:
            chunk: TextChunk model instance to extract from
            section_type: Type of section (DESCENDANT_GENEALOGY, etc.)
            model: LLM model to use

        Returns:
            dict with:
                - success: bool
                - error: str (if failed)
                - Additional fields depend on strategy
        """
        try:
            # Get the extraction strategy for this section type
            strategy = get_strategy(section_type)
            logger.debug(f"Using extraction strategy: {strategy.strategy_name}")

            # Check if strategy wants to process this chunk
            if not strategy.should_process(chunk):
                return {
                    'success': False,
                    'error': f'Strategy {strategy.strategy_name} cannot process chunk type {chunk.chunk_type}'
                }

            # Extract using the strategy
            result = strategy.extract(chunk, self.ollama, model)

            return result

        except KeyError as e:
            error_msg = f"Unknown section type: {section_type}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
            }

        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
            }

    def extract_from_chunks_in_section(
        self,
        chunks,
        section_type: str,
        model: str
    ) -> Dict[str, Any]:
        """Extract entities from multiple chunks in a section

        Args:
            chunks: QuerySet or list of TextChunk instances
            section_type: Type of section
            model: LLM model to use

        Returns:
            dict with:
                - success: bool
                - processed: int
                - failed: int
                - errors: list of error messages
        """
        processed = 0
        failed = 0
        errors = []

        for chunk in chunks:
            result = self.extract_from_chunk(chunk, section_type, model)

            if result['success']:
                processed += 1
            else:
                failed += 1
                errors.append(f"Chunk {chunk.sequence_number}: {result.get('error')}")

        return {
            'success': failed == 0,
            'processed': processed,
            'failed': failed,
            'errors': errors,
        }
