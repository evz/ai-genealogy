"""Utilities for parsing person names, particularly Dutch names with prefixes"""

import re


def parse_name(full_name: str) -> tuple[str, str]:
    """
    Parse full name into given_names and surname.

    Handles Dutch name prefixes like "van", "de", "van der", "van den", etc.

    Args:
        full_name: Full name string to parse

    Returns:
        Tuple of (given_names, surname)

    Examples:
        >>> parse_name("Pieter van Zanten")
        ('Pieter', 'van Zanten')
        >>> parse_name("Jan van der Meer")
        ('Jan', 'van der Meer')
        >>> parse_name("Thomas [Tom] de Jong")
        ('Thomas', 'de Jong')
    """
    # Remove brackets: "Pieter [Peter]" -> "Pieter"
    full_name = re.sub(r'\[.*?\]', '', full_name).strip()

    # Clean up parenthetical notes that are not part of the name
    # e.g. "Bessel van Zanten (son of Pieter)" -> "Bessel van Zanten"
    full_name = re.sub(r'\s*\([^)]*\)\s*$', '', full_name).strip()

    if not full_name:
        return '', ''

    parts = full_name.split()

    # Handle Dutch name prefixes (van, de, van de, van der, van den, etc.)
    if len(parts) >= 2:
        # Check for two-word prefixes: "van de", "van der", "van den"
        if len(parts) >= 3 and parts[-3].lower() == 'van' and parts[-2].lower() in ['de', 'der', 'den']:
            surname = ' '.join(parts[-3:])
            given_names = ' '.join(parts[:-3])
        # Check for single-word prefixes: "van", "de", "der", "den"
        elif parts[-2].lower() in ['van', 'de', 'der', 'den']:
            surname = ' '.join(parts[-2:])
            given_names = ' '.join(parts[:-2])
        else:
            # No prefix - last word is surname
            surname = parts[-1]
            given_names = ' '.join(parts[:-1])
    elif len(parts) == 1:
        # Single word - treat as surname
        surname = parts[0]
        given_names = ''
    else:
        return '', ''

    return given_names, surname
