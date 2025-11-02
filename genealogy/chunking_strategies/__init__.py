"""
Chunking strategies for different book sections.

Each section type (DESCENDANT_GENEALOGY, KWARTIERSTATEN, etc.) can have
its own chunking strategy.
"""

from .base import ChunkingStrategy
from .descendant_genealogy import DescendantGenealogyChunkingStrategy
from .skip import SkipChunkingStrategy

# Strategy registry - maps section types to chunking strategies
CHUNKING_STRATEGY_REGISTRY = {
    'FRONT_MATTER': SkipChunkingStrategy(),
    'DESCENDANT_GENEALOGY': DescendantGenealogyChunkingStrategy(),
    'KWARTIERSTATEN': SkipChunkingStrategy(),
    'APPENDIX_NARRATIVE': SkipChunkingStrategy(),
    'GLOSSARY': SkipChunkingStrategy(),
    'INDEX': SkipChunkingStrategy(),
}


def get_chunking_strategy(section_type: str) -> ChunkingStrategy:
    """
    Get the appropriate chunking strategy for a section type.

    Args:
        section_type: The BookSection.section_type value

    Returns:
        ChunkingStrategy instance

    Raises:
        KeyError: If section_type is not in the registry
    """
    if section_type not in CHUNKING_STRATEGY_REGISTRY:
        raise KeyError(f"No chunking strategy registered for section type: {section_type}")

    return CHUNKING_STRATEGY_REGISTRY[section_type]


__all__ = [
    'ChunkingStrategy',
    'DescendantGenealogyChunkingStrategy',
    'SkipChunkingStrategy',
    'get_chunking_strategy',
    'CHUNKING_STRATEGY_REGISTRY',
]
