"""Base extraction strategy for section-specific extraction"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ExtractionStrategy(ABC):
    """Base class for section-specific extraction strategies"""

    @abstractmethod
    def should_process(self, chunk) -> bool:
        """
        Determine if this chunk should be processed.

        Args:
            chunk: TextChunk model instance

        Returns:
            True if the chunk should be processed by this strategy
        """
        pass

    @abstractmethod
    def build_prompt(self, chunk) -> str:
        """
        Build the extraction prompt for this chunk.

        Args:
            chunk: TextChunk model instance

        Returns:
            Prompt string for LLM
        """
        pass

    @abstractmethod
    def parse_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parse the LLM output into structured data.

        Args:
            output_text: Raw text output from LLM

        Returns:
            Dictionary with parsed extraction data (format depends on strategy)
        """
        pass

    @abstractmethod
    def extract(self, chunk, ollama, model: str) -> Dict[str, Any]:
        """
        Extract entities from the chunk using this strategy.

        Args:
            chunk: TextChunk model instance
            ollama: OllamaClient instance
            model: Model name to use for extraction

        Returns:
            dict with extraction results (format depends on strategy)
        """
        pass

    @abstractmethod
    def get_chunk_filter(self) -> Dict[str, Any]:
        """
        Get Django ORM filter kwargs for chunks to process.

        Returns:
            Dictionary of filter kwargs for TextChunk.objects.filter()
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name for this strategy"""
        pass
