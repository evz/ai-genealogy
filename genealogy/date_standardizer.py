"""
Date standardization utilities for genealogical data

Reuses extraction patterns from generate_ner_training_data.py but adds standardization
to convert Dutch dates to ISO format (YYYY-MM-DD).
"""

import logging
import re

logger = logging.getLogger(__name__)

# Dutch month names to numbers (from existing patterns)
DUTCH_MONTHS = {
    # Full Dutch names
    "januari": 1,
    "februari": 2,
    "maart": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "augustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
    # Full English names
    "january": 1,
    "february": 2,
    "march": 3,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "october": 10,
    # Common abbreviations
    "jan": 1,
    "feb": 2,
    "mrt": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "okt": 10,
    "nov": 11,
    "dec": 12,
}


class DateStandardizer:
    """Standardizes Dutch genealogical dates to ISO format"""

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile date parsing patterns (reusing logic from training data generator)"""

        # Complex dates with day/month/year - matches existing training pattern
        self.complex_date_pattern = re.compile(
            r"\b(?:"
            r"(?:circa|ca\.?|around|about|omstreeks)?\s*"  # Optional circa
            r"(?:"
            r"(?:(\d{1,2})[-/\s]*)?"  # Optional day (capture group 1)
            r"(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december|"
            r"january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mrt|apr|mei|jun|jul|aug|sep|okt|nov|dec)[-/\s]*"  # Month names
            r")?\s*"
            r"(1[5-9]\d{2}|20[0-2]\d)"  # Years 1500-2029 (capture group 3)
            r")\b",
            re.IGNORECASE,
        )

        # Simple year pattern
        self.year_pattern = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")

        # Dot format dates: DD.MM.YYYY
        self.dot_date_pattern = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")

    def standardize_date(self, date_text: str) -> str | None:
        """
        Standardize a single date string to ISO format

        Args:
            date_text: Raw date string like "15 maart 1654"

        Returns:
            ISO date string like "1654-03-15" or None if unparseable
        """
        if not date_text or not date_text.strip():
            return None

        date_text = date_text.strip()

        # Try dot format first: DD.MM.YYYY
        dot_match = self.dot_date_pattern.search(date_text)
        if dot_match:
            return self._parse_dot_format(dot_match)

        # Try complex date pattern: "15 maart 1654", "maart 1654", "1654"
        complex_match = self.complex_date_pattern.search(date_text)
        if complex_match:
            return self._parse_complex_date(complex_match)

        # Try simple year pattern
        year_match = self.year_pattern.search(date_text)
        if year_match:
            return self._parse_year_only(year_match)

        logger.debug(f"Could not standardize date: '{date_text}'")
        return None

    def _parse_dot_format(self, match) -> str | None:
        """Parse DD.MM.YYYY format"""
        day_str, month_str, year_str = match.groups()

        try:
            day = int(day_str)
            month = int(month_str)
            year = int(year_str)

            # Basic validation
            if not (1 <= day <= 31 and 1 <= month <= 12 and 1500 <= year <= 2100):
                return None

            return f"{year:04d}-{month:02d}-{day:02d}"

        except ValueError:
            return None

    def _parse_complex_date(self, match) -> str | None:
        """Parse complex date patterns with Dutch/English month names"""
        day_str, month_str, year_str = match.groups()

        try:
            year = int(year_str)

            if not (1500 <= year <= 2100):
                return None

            # Handle month
            if month_str:
                month_num = DUTCH_MONTHS.get(month_str.lower())
                if not month_num:
                    return None
            else:
                month_num = 1  # Default to January if no month

            # Handle day
            if day_str:
                day = int(day_str)
                if not (1 <= day <= 31):
                    return None
            else:
                day = 1  # Default to 1st if no day

            return f"{year:04d}-{month_num:02d}-{day:02d}"

        except ValueError:
            return None

    def _parse_year_only(self, match) -> str | None:
        """Parse year-only dates"""
        year_str = match.group(1)

        try:
            year = int(year_str)

            if not (1500 <= year <= 2100):
                return None

            # Use January 1st for year-only dates
            return f"{year:04d}-01-01"

        except ValueError:
            return None

    def standardize_date_list(self, date_list: list[str]) -> list[str]:
        """
        Standardize a list of date strings

        Args:
            date_list: List of raw date strings

        Returns:
            List of standardized ISO date strings (unparseable dates filtered out)
        """
        standardized = []

        for date_text in date_list:
            standardized_date = self.standardize_date(date_text)
            if standardized_date:
                standardized.append(standardized_date)
                logger.debug(f"Standardized '{date_text}' -> '{standardized_date}'")
            else:
                logger.warning(f"Failed to standardize date: '{date_text}'")

        return standardized


def standardize_date(date_text: str) -> str | None:
    """
    Convenience function to standardize a single date

    Args:
        date_text: Raw date string

    Returns:
        ISO date string (YYYY-MM-DD) or None if unparseable
    """
    standardizer = DateStandardizer()
    return standardizer.standardize_date(date_text)


def standardize_dates(date_list: list[str]) -> list[str]:
    """
    Convenience function to standardize a list of dates

    Args:
        date_list: List of raw date strings

    Returns:
        List of standardized ISO date strings
    """
    standardizer = DateStandardizer()
    return standardizer.standardize_date_list(date_list)
