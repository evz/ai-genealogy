"""Base class for section-specific chunking strategies"""
from abc import ABC, abstractmethod
from typing import List


class ChunkingStrategy(ABC):
    """Base class for chunking strategies - different sections need different chunking approaches"""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name for this strategy"""
        pass

    @abstractmethod
    def chunk_section(self, section_text: str, document, page_map: List[dict]) -> List:
        """
        Chunk the text from a specific book section.

        Args:
            section_text: OCR text from all pages in this section, concatenated
            document: Document model instance (for context like BookSections)
            page_map: List of dicts mapping character positions to page numbers

        Returns:
            List of TextChunk objects (not yet saved to database)
        """
        pass

    def should_process(self, section) -> bool:
        """
        Check if this strategy should process the given section.

        Default: always process. Override to add section-specific logic.

        Args:
            section: BookSection model instance

        Returns:
            True if this strategy should chunk this section
        """
        return True
