"""Utilities for parsing dates from various formats"""
import logging
import re
from datetime import date
from typing import Optional, Tuple

from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


def parse_genealogical_date(date_str: str) -> Tuple[Optional[date], bool]:
    """
    Parse a date string from genealogical records into a date object.

    Handles various formats:
    - ISO format: 1845-03-12, 1845-03, 1845
    - European format: 15.3.1850, 3/15/1850
    - Partial dates: "1845" (year only), "March 1845"
    - Approximate indicators: "ca. 1845", "~1850", "about 1845"

    Args:
        date_str: Date string to parse

    Returns:
        Tuple of (parsed_date, is_approximate)
        - parsed_date: date object or None if unparseable
        - is_approximate: True if date has approximation indicators
    """
    if not date_str or not isinstance(date_str, str):
        return None, False

    original = date_str
    date_str = date_str.strip()

    # Check for approximation indicators
    is_approximate = False
    approx_indicators = ['ca.', 'ca', 'circa', '~', 'about', 'abt', 'est', 'estimated', 'approx']
    for indicator in approx_indicators:
        if indicator in date_str.lower():
            is_approximate = True
            # Remove the indicator for parsing
            date_str = re.sub(rf'\b{re.escape(indicator)}\b\.?', '', date_str, flags=re.IGNORECASE)
            date_str = date_str.replace('~', '').strip()

    # Try to parse with dateutil
    try:
        # dateutil is very flexible and handles most formats
        parsed = date_parser.parse(date_str, fuzzy=True, default=date(1, 1, 1))

        # If only a year was provided, dateutil defaults to Jan 1
        # Check if the original string was just a year
        if re.match(r'^\d{4}$', date_str.strip()):
            # Just a year - use Jan 1 of that year
            return date(parsed.year, 1, 1), is_approximate

        # parsed might be datetime or date object
        parsed_date = parsed.date() if hasattr(parsed, 'date') and callable(parsed.date) else parsed
        return parsed_date, is_approximate

    except (ValueError, TypeError, date_parser.ParserError) as e:
        logger.debug(f"Could not parse date '{original}': {e}")
        return None, is_approximate
