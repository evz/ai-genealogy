"""Skip extraction strategy for sections that don't need processing"""

from typing import Any, Dict

from .base import ExtractionStrategy


class SkipExtractionStrategy(ExtractionStrategy):
    """
    Strategy for sections that should not be processed.

    Used for: FRONT_MATTER, GLOSSARY
    """

    @property
    def strategy_name(self) -> str:
        return "Skip Extraction"

    def should_process(self, chunk) -> bool:
        """Never process chunks with this strategy"""
        return False

    def build_prompt(self, chunk) -> str:
        """Should never be called"""
        raise NotImplementedError("SkipExtractionStrategy does not build prompts")

    def parse_output(self, output_text: str) -> Dict[str, Any]:
        """Should never be called"""
        raise NotImplementedError("SkipExtractionStrategy does not parse output")

    def extract(self, chunk, ollama, model: str) -> Dict[str, Any]:
        """Should never be called"""
        raise NotImplementedError("SkipExtractionStrategy should not extract - should_process() returns False")

    def get_chunk_filter(self) -> Dict[str, Any]:
        """Return filter that matches no chunks"""
        # Use an impossible condition to match no chunks
        return {"id": None}
