"""Utilities for parsing family group headers"""

import re
from typing import List, Optional, Tuple

from .id_corrections import correct_genealogical_id


def parse_family_group_header(family_groups: List[str]) -> Tuple[List[str], Optional[str]]:
    """
    Extract parent names and genealogical ID from family group header.

    The genealogical ID at the end of the header refers to the FIRST parent mentioned.

    **Automatic Correction**: This function applies known corrections to genealogical IDs
    to fix typos in the source document. For example, "(IX.6.c)" will be corrected to
    "X.6.c". See genealogy.utils.id_corrections for the full list of corrections.

    Args:
        family_groups: List of family group header strings

    Returns:
        Tuple of (parent_names, first_parent_genealogical_id)
        - parent_names: List of parent names (usually 2, but could be 0-2)
        - first_parent_genealogical_id: CORRECTED genealogical ID of first parent

    Examples:
        >>> parse_family_group_header(["XII.14. Kinderen van Marinus Borsten en Alieke Zwerver"])
        (['Marinus Borsten', 'Alieke Zwerver'], None)
        >>> parse_family_group_header(["II.3. Kinderen van Jan van Zanten en Maria Pieterse (II.1.a)"])
        (['Jan van Zanten', 'Maria Pieterse'], 'II.1.a')
        >>> parse_family_group_header(["XI.10. Children of Thomas Kamp and Cheri Hofstad (IX.6.c):"])
        (['Thomas Kamp', 'Cheri Hofstad'], 'X.6.c')  # Corrected from IX.6.c to X.6.c
    """
    if not family_groups:
        return [], None

    # Example: "XII.14. Kinderen van Marinus Wilhelmus Borsten en Alieke Zwerver"
    # Example: "II.3. Kinderen van Jan van Zanten en Maria Pieterse (II.1.a)"
    # Example: "XII.10. Children of Jamie Hall and Joshua Abercrombie (X1.16.b)"
    family_group = family_groups[0]

    # First, extract genealogical ID if present at the end
    # Pattern: (Roman.Number.letter) OR (Roman) at the end of the string
    # May have optional colon and/or whitespace after closing paren
    genealogical_id = None
    # Try full format first: I.1.a or II.3.b
    id_pattern = r'\(([IVX]+\.\d+\.[a-z])\)[:\s]*$'
    id_match = re.search(id_pattern, family_group)
    if id_match:
        genealogical_id = id_match.group(1)
        # Apply known corrections for typos in the source document
        genealogical_id = correct_genealogical_id(genealogical_id)
        # Remove the ID from the string for cleaner name parsing
        family_group = re.sub(id_pattern, '', family_group).strip()
    else:
        # Try simple Roman numeral format: I, II, III, etc.
        id_pattern_simple = r'\(([IVX]+)\)[:\s]*$'
        id_match = re.search(id_pattern_simple, family_group)
        if id_match:
            genealogical_id = id_match.group(1)
            # Apply known corrections
            genealogical_id = correct_genealogical_id(genealogical_id)
            # Remove the ID from the string for cleaner name parsing
            family_group = re.sub(id_pattern_simple, '', family_group).strip()

    # Pattern to extract everything after "Kinderen van" or "Children of"
    pattern = r'(?:Kinderen\s+van|Children\s+of)\s+(.+)'
    match = re.search(pattern, family_group, re.IGNORECASE)

    if not match:
        return [], genealogical_id

    names_part = match.group(1).strip()

    # Split on "en" or "and" to get both parent names
    parent_names = re.split(r'\s+(?:en|and)\s+', names_part, flags=re.IGNORECASE)
    parent_names = [p.strip() for p in parent_names if p.strip()]

    return parent_names, genealogical_id
