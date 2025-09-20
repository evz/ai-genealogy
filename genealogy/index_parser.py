"""
Index parser for genealogical 6-column indices.

Parses OCR output from genealogical index pages with the format:
surname | given_names | ID | surname | given_names | ID

Creates lookup tables for validation and correction of genealogical extractions.
"""

import logging
import re
from dataclasses import dataclass

from .training_tools.ocr_corrections import OCRCorrections

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    """Single entry from the genealogical index"""

    surname: str
    given_names: str
    genealogical_id: str
    tussenvoegsel: str = ""  # Dutch prepositions (van, de, etc.)


class IndexParser:
    """Parser for 6-column genealogical indices"""

    def __init__(self):
        self.genealogical_id_pattern = r"\b[IVX]+\.\d+\.[a-z]+"
        self.tussenvoegsel_pattern = r"\b(van|de|der|den|het|in|op|over|onder|bij|tot|te)\b"
        self.cross_ref_pattern = r"\bzie:?\s+([^,\n]+)"
        self.date_pattern = r"\([0-9]{4}\)"
        self.ocr_corrector = OCRCorrections()

    def parse_index_line(self, line: str) -> list[IndexEntry]:
        """
        Parse a single line from the index into genealogical entries.

        Tesseract PSM 6 adds | characters to mark column boundaries in tables.
        Expected format: surname given_names ID | surname given_names ID

        Args:
            line: OCR text line from index

        Returns:
            List of IndexEntry objects (0-2 entries per line)
        """
        line = line.strip()
        if not line:
            return []

        entries = []

        # Handle lines that start with | (continuation from previous line)
        if line.startswith("|"):
            line = line[1:].strip()

        # Split on | to separate the column groups (Tesseract table detection)
        if "|" in line:
            parts = line.split("|")
        else:
            # Fallback: treat as single entry if no pipe separators
            parts = [line]

        for part in parts:
            part = part.strip()
            if not part:
                continue

            entry = self._parse_single_entry(part)
            if entry:
                entries.append(entry)

        return entries

    def _parse_single_entry(self, text: str) -> IndexEntry | None:
        """
        Parse a single entry: 'surname given_names tussenvoegsel ID[,additional_refs]'

        Args:
            text: Single entry text

        Returns:
            IndexEntry or None if parsing fails
        """
        text = text.strip()
        if not text:
            return None

        # Apply OCR corrections to fix common errors (e.g., X1 -> XI)
        text = self.ocr_corrector.apply_text_corrections(text)

        # Check for cross-references (zie: other name)
        cross_ref_match = re.search(self.cross_ref_pattern, text, re.IGNORECASE)
        if cross_ref_match:
            logger.debug(f"Cross-reference found: {text}")
            return None  # Skip cross-references for now

        # Find genealogical ID (e.g., VII.5.c, VII.1e,g)
        id_matches = list(re.finditer(self.genealogical_id_pattern, text))
        if not id_matches:
            logger.debug(f"No genealogical ID found in: {text}")
            return None

        # Use the first ID match (primary location, ignore references)
        id_match = id_matches[0]
        full_id = id_match.group()

        # Extract just the primary ID (before any commas)
        primary_id = full_id.split(",")[0].strip()

        # Extract text before the ID
        name_part = text[: id_match.start()].strip()

        # Remove date disambiguation if present: "v (1901)" -> "v"
        name_part = re.sub(self.date_pattern, "", name_part).strip()

        # Parse name part into surname, given_names, tussenvoegsel
        surname, given_names, tussenvoegsel = self._parse_name_part(name_part)

        if not surname:
            logger.debug(f"Could not extract surname from: {name_part}")
            return None

        return IndexEntry(
            surname=surname, given_names=given_names, genealogical_id=primary_id, tussenvoegsel=tussenvoegsel
        )

    def _parse_name_part(self, name_part: str) -> tuple[str, str, str]:
        """
        Parse name part into surname, given_names, and tussenvoegsel.

        Args:
            name_part: Text before the genealogical ID

        Returns:
            Tuple of (surname, given_names, tussenvoegsel)
        """
        if not name_part:
            return "", "", ""

        words = name_part.split()
        if not words:
            return "", "", ""

        # First word is typically the surname
        surname = words[0]

        # Find tussenvoegsel (Dutch prepositions)
        tussenvoegsel_words = []
        name_words = []

        for word in words[1:]:
            if re.match(self.tussenvoegsel_pattern, word, re.IGNORECASE):
                tussenvoegsel_words.append(word)
            else:
                name_words.append(word)

        given_names = " ".join(name_words)
        tussenvoegsel = " ".join(tussenvoegsel_words)

        return surname, given_names, tussenvoegsel

    def create_lookup_table(self, entries: list[IndexEntry]) -> dict[str, list[IndexEntry]]:
        """
        Create a lookup table from index entries for fast searching.

        Args:
            entries: List of IndexEntry objects

        Returns:
            Dictionary mapping surnames to lists of entries
        """
        lookup = {}

        for entry in entries:
            surname_key = entry.surname.lower()
            if surname_key not in lookup:
                lookup[surname_key] = []
            lookup[surname_key].append(entry)

        return lookup

    def parse_index_page(self, ocr_text: str, skip_lines: int = 0) -> list[IndexEntry]:
        """
        Parse an entire index page. Since we only want primary IDs,
        we ignore continuation lines with additional reference IDs.

        Args:
            ocr_text: OCR text from the page
            skip_lines: Number of lines to skip at beginning (for explanatory text)

        Returns:
            List of IndexEntry objects from the page
        """
        lines = ocr_text.split("\n")
        entries = []

        for line in lines[skip_lines:]:
            line = line.strip()
            if not line:
                continue

            line_entries = self.parse_index_line(line)
            entries.extend(line_entries)

        logger.info(f"Parsed {len(entries)} entries from index page")
        return entries
