"""
Utilities for classifying text chunks into search tiers.
"""

from django.db.models import Count

from genealogy.models import TextChunk


def classify_chunk_tier(
    text_content: str,
    chunk_type: str,
    extracted_events: list = None,
    extracted_relationships: list = None
) -> str:
    """
    Determine if a chunk should be metadata-only or narrative-tier.

    Args:
        text_content: The chunk's text content
        chunk_type: The chunk type (individual_entry, generation_header, etc.)
        extracted_events: List of extracted events (optional, for heuristics)
        extracted_relationships: List of extracted relationships (optional)

    Returns:
        'metadata' or 'narrative'

    Logic:
    - Headers always -> metadata
    - individual_entry < 200 chars -> metadata
    - individual_entry >= 200 chars -> narrative
    - biographical_text, narrative_context -> narrative

    Future enhancements could check for:
    - Presence of occupation/military/education keywords
    - Number of events extracted
    - Presence of narrative sentences (not just vital stats)
    """

    # Headers and structural chunks -> metadata tier
    if chunk_type in ['generation_header', 'family_group_header']:
        return 'metadata'

    # Length-based classification for individual entries
    if chunk_type == 'individual_entry':
        text_length = len(text_content.strip())

        # Short entries are just vital statistics -> metadata tier
        if text_length < 100:
            return 'metadata'

        # Long entries likely have biographical narrative -> narrative tier
        return 'narrative'

    # Explicit biographical/narrative chunks -> narrative tier
    if chunk_type in ['biographical_text', 'narrative_context']:
        return 'narrative'

    # Default to metadata tier for safety
    return 'metadata'


def get_tier_statistics(chunk_queryset=None):
    """
    Get statistics about chunk tier distribution.

    Args:
        chunk_queryset: Optional queryset to analyze. If None, uses all TextChunks.

    Returns:
        dict with tier distribution statistics
    """
    if chunk_queryset is None:
        chunk_queryset = TextChunk.objects.all()

    stats = chunk_queryset.values('search_tier').annotate(count=Count('id'))

    total = chunk_queryset.count()

    result = {
        'total': total,
        'by_tier': {},
        'percentages': {}
    }

    for stat in stats:
        tier = stat['search_tier']
        count = stat['count']
        result['by_tier'][tier] = count
        result['percentages'][tier] = (count / total * 100) if total > 0 else 0

    return result
