"""Extraction strategies for different book section types

Each strategy encapsulates the extraction logic (prompt building, parsing, processing)
for a specific type of genealogical content.
"""

from .base import ExtractionStrategy
from .descendant_genealogy import DescendantGenealogyStrategy
from .skip import SkipExtractionStrategy

# Strategy registry mapping section types to strategies
STRATEGY_REGISTRY = {
    'FRONT_MATTER': SkipExtractionStrategy(),
    'DESCENDANT_GENEALOGY': DescendantGenealogyStrategy(),
    'KWARTIERSTATEN': SkipExtractionStrategy(),  # TODO: Implement AncestorTableStrategy
    'APPENDIX_NARRATIVE': SkipExtractionStrategy(),  # TODO: Implement NarrativeExtractionStrategy
    'GLOSSARY': SkipExtractionStrategy(),
    'INDEX': SkipExtractionStrategy(),  # TODO: Implement IndexExtractionStrategy
}


def get_strategy(section_type: str) -> ExtractionStrategy:
    """
    Get the appropriate extraction strategy for a section type.

    Args:
        section_type: BookSection.section_type value

    Returns:
        ExtractionStrategy instance for that section type

    Raises:
        KeyError: If section_type is not recognized
    """
    if section_type not in STRATEGY_REGISTRY:
        raise KeyError(f"No extraction strategy defined for section type: {section_type}")

    return STRATEGY_REGISTRY[section_type]


__all__ = [
    'ExtractionStrategy',
    'DescendantGenealogyStrategy',
    'SkipExtractionStrategy',
    'get_strategy',
    'STRATEGY_REGISTRY',
]
