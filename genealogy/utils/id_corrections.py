"""
Genealogical ID correction utilities.

This module provides mappings for known typos in genealogical IDs found in source documents.
These corrections are applied during graph building to ensure correct parent-child relationships.
"""

# Known genealogical ID typos in the Jan van Bulhuis Book
# Maps: incorrect_id -> correct_id
GENEALOGICAL_ID_CORRECTIONS = {
    # Generation IX/X confusion (systematic issue in the book)
    'IX.6.c': 'X.6.c',   # Thomas Gregory Kamp MSc - affects XI.10.a/b/c and XII.2.a
    'IX.6.d': 'X.6.d',   # Nancy Suzanne Kamp BSc - affects XI.11.a/b/c

    # Generation X numbering error
    'X.3.c': 'X.4.c',    # Ewout van der Reijden - affects XI.6.a/b
}


def correct_genealogical_id(genealogical_id: str) -> str:
    """
    Apply known corrections to a genealogical ID.

    Args:
        genealogical_id: The genealogical ID as it appears in the source

    Returns:
        The corrected genealogical ID, or the original if no correction is needed

    Example:
        >>> correct_genealogical_id('IX.6.c')
        'X.6.c'
        >>> correct_genealogical_id('II.3.a')
        'II.3.a'
    """
    if not genealogical_id:
        return genealogical_id

    return GENEALOGICAL_ID_CORRECTIONS.get(genealogical_id, genealogical_id)


def extract_parent_id_from_family_group(family_group_header: str) -> str | None:
    """
    Extract and correct the parent genealogical ID from a family group header.

    Handles formats like:
    - "XI.10. Children of Thomas Gregory Kamp jr. and Cheri Lynn Hofstad (IX.6.c):"
    - "XI.6. Kinderen van Ewout van der Reijden (X.3.c):"

    Args:
        family_group_header: The family group header text

    Returns:
        The corrected parent genealogical ID, or None if not found

    Example:
        >>> extract_parent_id_from_family_group("XI.10. Children of ... (IX.6.c):")
        'X.6.c'
    """
    import re

    if not family_group_header:
        return None

    # Match pattern: (ROMAN.NUMBER.LETTER) or (ROMAN.NUMBER.LETTER):
    # Handles both with and without trailing colon
    pattern = r'\(([IVX]+\.\d+\.[a-z])\):?'
    match = re.search(pattern, family_group_header)

    if match:
        parent_id = match.group(1)
        # Apply corrections
        return correct_genealogical_id(parent_id)

    return None


def get_correction_info(genealogical_id: str) -> dict | None:
    """
    Get information about a genealogical ID correction.

    Args:
        genealogical_id: The genealogical ID to check

    Returns:
        Dict with correction info, or None if no correction needed
        Format: {'original': 'IX.6.c', 'corrected': 'X.6.c', 'reason': '...'}
    """
    if genealogical_id in GENEALOGICAL_ID_CORRECTIONS:
        corrected = GENEALOGICAL_ID_CORRECTIONS[genealogical_id]

        # Provide context about the correction
        reasons = {
            'IX.6.c': 'Generation IX/X confusion - should be X.6.c (Thomas Gregory Kamp MSc)',
            'IX.6.d': 'Generation IX/X confusion - should be X.6.d (Nancy Suzanne Kamp BSc)',
            'X.3.c': 'Numbering error - should be X.4.c (Ewout van der Reijden)',
        }

        return {
            'original': genealogical_id,
            'corrected': corrected,
            'reason': reasons.get(genealogical_id, 'Known typo in source document'),
        }

    return None
