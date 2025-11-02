"""Skip chunking strategy for sections that don't need chunking"""
import logging
from typing import List

from .base import ChunkingStrategy

logger = logging.getLogger(__name__)


class SkipChunkingStrategy(ChunkingStrategy):
    """
    Strategy for sections that should not be chunked.

    Used for:
    - FRONT_MATTER: Title pages, table of contents, etc.
    - KWARTIERSTATEN: Ancestor charts (need different chunking approach)
    - APPENDIX_NARRATIVE: Appendices (need different chunking approach)
    - GLOSSARY: Term definitions (need different chunking approach)
    - INDEX: Name/place indexes (need different chunking approach)
    """

    @property
    def strategy_name(self) -> str:
        return "Skip Chunking"

    def chunk_section(self, section_text: str, document, page_map: List[dict]) -> List:
        """
        Skip chunking for this section.

        Returns:
            Empty list (no chunks created)
        """
        logger.info(f"Skipping chunking for this section ({len(section_text)} characters)")
        return []

    def should_process(self, section) -> bool:
        """Don't process sections with this strategy"""
        return False
