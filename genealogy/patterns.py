"""
Centralized regex patterns for genealogical entity extraction.

This module consolidates all regex patterns used across the genealogy
extraction system to eliminate duplication and provide a clean interface.
"""

import re


class GenealogyPatterns:
    """Compiled regex patterns for genealogical entity extraction"""

    # Genealogical IDs: II.1.a, XII.5.b, etc.
    # Handles standard format, OCR corrupted formats, and spaced variants
    GENEALOGY_ID = re.compile(
        r"\b(?:"
        r"([IVXLCDMilvxlcdm]+)\.(\d+)\.([a-zA-Z])"
        r"|"
        r"([A-Z]{1,4}[LI]+)\.(\d+)\.([a-zA-Z])"
        r"|"
        r"([IVXLCDMilvxlcdm]+)\.\s*(\d+)\.\s*([a-zA-Z])"
        r")\b"
    )

    # Family group headers: "X.9. Children of John & Mary"
    # Handles English/Dutch variants and line breaks
    FAMILY_GROUP = re.compile(
        r"(?:"
        r"\b([IVXLCDMilvxlcdm]+|[A-Z]{1,4}[LI]*)\.(\d+)\.\s+(?:Children\s+of|Kinderen\s+van|children\s+of|kinderen\s+van)(?:\s+[A-Z][a-zA-Z\s&,\-\.]*)?"
        r"|"
        r"\b([IVXLCDMilvxlcdm]+|[A-Z]{1,4}[LI]*)\.(\d+)\.\s+(?:Children|Kinderen|children|kinderen)(?!\s+of)"
        r"|"
        r"\b([IVXLCDMilvxlcdm]+|[A-Z]{1,4}[LI]*)\.(\d+)\.\s*\n?\s*(?:Children\s+of|Kinderen\s+van|children\s+of|kinderen\s+van)"
        r")",
        re.IGNORECASE,
    )

    # Generation headers: "eerste generatie", "tweede generatie", etc.
    GENERATION_HEADER = re.compile(
        r"\b(eerste|tweede|derde|vierde|vijfde|zesde|zevende|achtste|negende|tiende|elfde|twaalfde)\s+generatie\b",
        re.IGNORECASE,
    )

    # Complex date patterns with optional circa and month names
    # Handles Dutch and English month names
    DATE_COMPREHENSIVE = re.compile(
        r"\b(?:"
        r"(?:circa|ca\.?|around|about|omstreeks)?\s*"
        r"(?:"
        r"(?:\d{1,2}[-/\s]*)?"
        r"(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december|"
        r"january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mrt|apr|mei|jun|jul|aug|sep|okt|nov|dec)[-/\s]*"
        r")?\s*"
        r"(?:1[5-9]\d{2}|20[0-2]\d)"
        r")\b",
        re.IGNORECASE,
    )

    # Simple year pattern: 1500-2029
    YEAR = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")

    # Dutch place names including major cities and geographical indicators
    PLACE = re.compile(
        r"\b(?:"
        r"(?:Amsterdam|Utrecht|Rotterdam|Den Haag|s-Gravenhage|Eindhoven|Tilburg|"
        r"Groningen|Almere|Breda|Nijmegen|Enschede|Haarlem|Arnhem|Amersfoort|Zaanstad|Apeldoorn|"
        r"s-Hertogenbosch|Hoofddorp|Maastricht|Leiden|Dordrecht|Zoetermeer|Zwolle|"
        r"Deventer|Delft|Alkmaar|Leeuwarden|Westland|Hilversum|Venlo|Roosendaal|"
        r"Ede|Helmond|Purmerend|Leidschendam|Alphen|Gouda|Spijkenisse|Vlaardingen)"
        r"|"
        r"(?:Nederland|Holland|Friesland|Gelderland|Noord-Holland|Zuid-Holland|"
        r"Noord-Brabant|Limburg|Zeeland|Utrecht|Overijssel|Flevoland|Drenthe|Groningen)"
        r")\b",
        re.IGNORECASE,
    )

    # Dutch person names with particles and various formats
    # Handles full names with particles, initials, and simple first/last combinations
    # Also handles line breaks within names
    PERSON_NAME = re.compile(
        r"\b(?:"
        r"(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\s+(?:van|de|der|den|te|ten|tot|op|aan|in|bij|d\'|du|le|la|des)\s*\n?\s*[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)"
        r"|"
        r"(?:[A-Z]\.(?:\s*[A-Z]\.)*\s+(?:van|de|der|den|te|ten|tot|op|aan|in|bij|d\'|du|le|la|des)\s*\n?\s*[A-Z][a-z]{2,})"
        r"|"
        r"(?:[A-Z][a-z]{3,}\s*\n?\s*(?:van|de|der|den|te|ten|tot|op|aan|in|bij|d\'|du|le|la|des)\s*\n?\s*[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{3,})?)"
        r"|"
        r"(?:[A-Z][a-z]{3,}\s*\n?\s*[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})?)"
        r"|"
        r"(?:[A-Z]\.(?:\s*[A-Z]\.)*\s*\n?\s*[A-Z][a-z]{3,})"
        r")\b"
    )

    @classmethod
    def get_all_patterns(cls) -> dict[str, re.Pattern]:
        """Get all compiled patterns as a dictionary"""
        return {
            "GENEALOGY_ID": cls.GENEALOGY_ID,
            "FAMILY_GROUP": cls.FAMILY_GROUP,
            "GENERATION_HEADER": cls.GENERATION_HEADER,
            "DATE_COMPREHENSIVE": cls.DATE_COMPREHENSIVE,
            "YEAR": cls.YEAR,
            "PLACE": cls.PLACE,
            "PERSON_NAME": cls.PERSON_NAME,
        }

    @classmethod
    def get_entity_types(cls) -> list[str]:
        """Get list of entity types supported by these patterns"""
        return [
            "GENEALOGY_ID",
            "FAMILY_GROUP",
            "GENERATION_HEADER",
            "DATE",
            "PLACE",
            "PERSON_NAME",
        ]
