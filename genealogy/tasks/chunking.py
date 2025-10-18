"""Text chunking logic for genealogy documents"""
import logging
import re

import spacy

from ..models import Document
from ..patterns import GenealogyPatterns

logger = logging.getLogger(__name__)


class GenealogyChunker:
    """Helper class for chunking documents based on genealogical structure"""

    # Dutch generation names to Roman numerals
    GENERATION_MAPPING = {
        "eerste generatie": 1,
        "tweede generatie": 2,
        "derde generatie": 3,
        "vierde generatie": 4,
        "ierde generatie": 4,  # OCR corruption: missing 'V' at start
        "vijfde generatie": 5,
        "zesde generatie": 6,
        "zevende generatie": 7,
        "achtste generatie": 8,
        "negende generatie": 9,
        "tiende generatie": 10,
        "elfde generatie": 11,
        "twaalfde generatie": 12,
    }

    # Common OCR corrections based on analysis
    OCR_CORRECTIONS = {
        "IL": "II",
        "XIL": "XII",
        "VIL": "VII",
        "VIIL": "VIII",
        "Vil": "VII",
        "VIl": "VII",
        "VIlI": "VIII",
        "VIll": "VIII",
        "Vill": "VIII",
        "VIIl": "VIII",
        "/": "IV",  # OCR corruption: "/" mistaken for "IV"
        "HL": "II",
        "W": "VI",
        "WV": "XV",
        "'V": "IV",
        "zy": "zv",
    }

    BURIAL_SYMBOL_PATTERNS = [
        (r", n ([A-Z][a-zA-Z]+)", r", [buried] \1"),
        (r", o ([A-Z][a-zA-Z]+)", r", [buried] \1"),
        (r" n ([A-Z][a-zA-Z]+) \d", r" [buried] \1 "),
        (r" o ([A-Z][a-zA-Z]+) \d", r" [buried] \1 "),
    ]

    DATE_OCR_PATTERNS = [
        (r"\.4 (\d{3})", r".1\1"),
        (r"4 0\.", r"10."),
    ]

    def __init__(self):
        self.current_generation = 1

    def chunk_document_text(self, document: Document) -> list[dict]:
        """
        Chunk document text based on genealogical structure with enhanced line classification

        Returns:
            List of dicts with keys: text_content, start_page, end_page, chunk_type,
            generation_number, family_groups, extraction_method
        """
        full_text = document.get_combined_ocr_text()
        if not full_text.strip():
            return []

        chunks = []
        lines = full_text.split("\n")

        logger.info(f"Chunking {len(lines)} lines using person-level boundaries")

        current_chunk_lines = []
        current_start_page = 1
        current_end_page = 1
        current_generation = 1
        current_family_context = None  # Track current family group (e.g., "X.9")
        current_genealogy_entry = None  # Track current genealogy entry for linking narrative chunks
        in_special_section = None  # Track if we're in KWARTIERSTATEN or APPENDIX section

        i = 0
        while i < len(lines):
            line = lines[i]
            # Track page numbers
            page_match = re.match(r"=== Page (\d+) ===", line)
            if page_match:
                current_end_page = int(page_match.group(1))
                if not current_chunk_lines:
                    current_start_page = current_end_page
                i += 1
                continue

            # Check for KWARTIERSTATEN section headers
            if self._is_kwartierstaat_header(line):
                # Finish current chunk if it has content
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        genealogical_data = self._extract_genealogical_data(
                            chunk_text, current_generation, current_family_context
                        )
                        # Determine chunk type based on context
                        chunk_type = self._determine_chunk_type(in_special_section, current_genealogy_entry)
                        chunk_dict = self._create_chunk_dict(
                            text_content=chunk_text,
                            chunk_type=chunk_type,
                            start_page=current_start_page,
                            end_page=current_end_page,
                            generation_number=current_generation,
                            genealogical_data=genealogical_data,
                        )
                        chunks.append(chunk_dict)

                # KWARTIERSTATEN section - clear family context
                # These are ancestor charts that don't follow normal genealogy structure
                current_family_context = None
                current_genealogy_entry = None
                in_special_section = "KWARTIERSTATEN"

                # Create header chunk for KWARTIERSTATEN section
                header_chunk = self._create_chunk_dict(
                    text_content=line.strip(),
                    chunk_type="KWARTIERSTATEN_HEADER",
                    start_page=current_end_page,
                    end_page=current_end_page,
                    generation_number=current_generation,
                    genealogical_data={"family_groups": [], "extraction_method": "regex"}
                )
                chunks.append(header_chunk)

                # Reset for new chunk
                current_chunk_lines = []
                current_start_page = current_end_page
                i += 1
                continue

            # Check for appendix/register section headers (Bijlage 1, Bijlage 2, Register)
            if self._is_appendix_header(line):
                # Finish current chunk if it has content
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        genealogical_data = self._extract_genealogical_data(
                            chunk_text, current_generation, current_family_context
                        )
                        # Determine chunk type based on context
                        chunk_type = self._determine_chunk_type(in_special_section, current_genealogy_entry)
                        chunk_dict = self._create_chunk_dict(
                            text_content=chunk_text,
                            chunk_type=chunk_type,
                            start_page=current_start_page,
                            end_page=current_end_page,
                            generation_number=current_generation,
                            genealogical_data=genealogical_data,
                        )
                        chunks.append(chunk_dict)

                # Appendix/Register section - clear family context
                # These are narrative stories, glossaries, or indices, not genealogy
                current_family_context = None
                current_genealogy_entry = None
                in_special_section = "APPENDIX"

                # Create header chunk for appendix/register section
                header_chunk = self._create_chunk_dict(
                    text_content=line.strip(),
                    chunk_type="APPENDIX_HEADER",
                    start_page=current_end_page,
                    end_page=current_end_page,
                    generation_number=current_generation,
                    genealogical_data={"family_groups": [], "extraction_method": "regex"}
                )
                chunks.append(header_chunk)

                # Reset for new chunk
                current_chunk_lines = []
                current_start_page = current_end_page
                i += 1
                continue

            # Check for generation headers
            generation_num = self._extract_generation_number(line)
            if generation_num:
                # Finish current chunk if it has content
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        genealogical_data = self._extract_genealogical_data(
                            chunk_text, current_generation, current_family_context
                        )
                        chunk_dict = self._create_chunk_dict(
                            text_content=chunk_text,
                            chunk_type="CONTENT",
                            start_page=current_start_page,
                            end_page=current_end_page,
                            generation_number=current_generation,
                            genealogical_data=genealogical_data,
                        )
                        chunks.append(chunk_dict)

                # Create header chunk
                header_chunk = self._create_chunk_dict(
                    text_content=line.strip(),
                    chunk_type="HEADER",
                    start_page=current_end_page,
                    end_page=current_end_page,
                    generation_number=generation_num,
                )
                chunks.append(header_chunk)

                # Reset for new chunk
                current_chunk_lines = []
                current_start_page = current_end_page
                current_generation = generation_num
                in_special_section = None  # Back to normal genealogy
                i += 1
                continue

            # Check for family group headers
            if self._is_family_group_header(line):
                # Finish current chunk if it has content
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        genealogical_data = self._extract_genealogical_data(
                            chunk_text, current_generation, current_family_context
                        )
                        chunk_dict = self._create_chunk_dict(
                            text_content=chunk_text,
                            chunk_type="CONTENT",
                            start_page=current_start_page,
                            end_page=current_end_page,
                            generation_number=current_generation,
                            genealogical_data=genealogical_data,
                        )
                        chunks.append(chunk_dict)

                # Create family group header chunk with family_groups populated
                family_header_text = line.strip()
                # Remove trailing colon if present
                if family_header_text.endswith(':'):
                    family_header_text = family_header_text[:-1]

                header_chunk = self._create_chunk_dict(
                    text_content=line.strip(),
                    chunk_type="HEADER",
                    start_page=current_end_page,
                    end_page=current_end_page,
                    generation_number=current_generation,
                    genealogical_data={
                        "family_groups": [family_header_text],  # Store the full header text
                        "extraction_method": "regex",
                    }
                )
                chunks.append(header_chunk)

                # Update family context for subsequent individual entries
                # Store the full header text (not just the ID) so it provides meaningful context
                current_family_context = family_header_text

                # Reset for new chunk
                current_chunk_lines = []
                current_start_page = current_end_page
                i += 1
                continue

            # Check for genealogical entries (dense biographical entries)
            # Skip if we're in KWARTIERSTATEN section (no family context)
            if self._is_genealogy_entry(line) and current_family_context is not None:
                # Finish current chunk if it has content
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        genealogical_data = self._extract_genealogical_data(
                            chunk_text, current_generation, current_family_context
                        )

                        # Determine chunk type for previous content
                        chunk_type = self._determine_chunk_type(in_special_section, current_genealogy_entry)

                        chunk_dict = self._create_chunk_dict(
                            text_content=chunk_text,
                            chunk_type=chunk_type,
                            start_page=current_start_page,
                            end_page=current_end_page,
                            generation_number=current_generation,
                            genealogical_data=genealogical_data,
                            related_genealogy_entry_index=current_genealogy_entry,
                        )
                        chunks.append(chunk_dict)

                # Start new genealogy entry - collect ALL lines for this person
                # including narrative context, until we hit the next person or 25K chars
                genealogy_lines = [line]

                # Look ahead to collect complete person entry (biographical + narrative)
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]

                    # Stop if we hit another genealogy entry, generation header, or family group
                    # (Allow chunks to span multiple pages)
                    if (
                        self._is_genealogy_entry(next_line)
                        or self._extract_generation_number(next_line)
                        or self._is_family_group_header(next_line)
                    ):
                        break

                    # Stop if chunk is getting too large (25K chars)
                    current_chunk = "\n".join(genealogy_lines)
                    if len(current_chunk) > 25000:
                        break

                    # Include ALL content (biographical and narrative) for this person
                    if next_line.strip():
                        genealogy_lines.append(next_line)
                    j += 1

                # Create genealogy entry chunk
                genealogy_text = "\n".join(genealogy_lines).strip()
                genealogical_data = self._extract_genealogical_data(
                    genealogy_text, current_generation, current_family_context
                )

                genealogy_chunk = self._create_chunk_dict(
                    text_content=genealogy_text,
                    chunk_type="GENEALOGY_ENTRY",
                    start_page=current_start_page,
                    end_page=current_end_page,
                    generation_number=current_generation,
                    genealogical_data=genealogical_data,
                    related_genealogy_entry_index=None,
                )

                chunks.append(genealogy_chunk)
                current_genealogy_entry = len(chunks) - 1  # Index of this genealogy entry

                # Reset for new chunk
                current_chunk_lines = []
                current_start_page = current_end_page

                # Skip the lines we already processed
                i = j
                continue

            # Regular content line
            if line.strip():
                # Check if this line establishes a new family context
                family_context = self._extract_family_context(line, current_generation)
                if family_context:
                    current_family_context = family_context

                current_chunk_lines.append(line)

                # Only split if chunk gets very large (>25000 chars for person-level chunks)
                chunk_text = "\n".join(current_chunk_lines)
                if len(chunk_text) > 25000:
                    genealogical_data = self._extract_genealogical_data(
                        chunk_text, current_generation, current_family_context
                    )

                    # Determine chunk type based on context
                    chunk_type = self._determine_chunk_type(in_special_section, current_genealogy_entry)

                    chunk_dict = self._create_chunk_dict(
                        text_content=chunk_text,
                        chunk_type=chunk_type,
                        start_page=current_start_page,
                        end_page=current_end_page,
                        generation_number=current_generation,
                        genealogical_data=genealogical_data,
                        related_genealogy_entry_index=current_genealogy_entry,
                    )
                    chunks.append(chunk_dict)
                    current_chunk_lines = []
                    current_start_page = current_end_page

            # Move to next line
            i += 1

        # Handle final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                genealogical_data = self._extract_genealogical_data(
                    chunk_text, current_generation, current_family_context
                )

                # Determine chunk type based on content
                chunk_type = self._determine_chunk_type(in_special_section, current_genealogy_entry)

                chunk_dict = self._create_chunk_dict(
                    text_content=chunk_text,
                    chunk_type=chunk_type,
                    start_page=current_start_page,
                    end_page=current_end_page,
                    generation_number=current_generation,
                    genealogical_data=genealogical_data,
                    related_genealogy_entry_index=current_genealogy_entry,
                )
                chunks.append(chunk_dict)

        return chunks

    def _determine_chunk_type(
        self,
        in_special_section: str | None,
        current_genealogy_entry: int | None,
    ) -> str:
        """
        Determine the appropriate chunk type based on current context.

        Args:
            in_special_section: Current special section type ("KWARTIERSTATEN" or "APPENDIX") or None
            current_genealogy_entry: Index of current genealogy entry or None

        Returns:
            Chunk type string: "GENEALOGY_ENTRY", "KWARTIERSTATEN", "APPENDIX", or "CONTENT"
        """
        if current_genealogy_entry is not None:
            return "GENEALOGY_ENTRY"
        elif in_special_section == "KWARTIERSTATEN":
            return "KWARTIERSTATEN"
        elif in_special_section == "APPENDIX":
            return "APPENDIX"
        else:
            return "CONTENT"

    def _create_chunk_dict(
        self,
        text_content: str,
        chunk_type: str,
        start_page: int,
        end_page: int,
        generation_number: int,
        genealogical_data: dict | None = None,
        related_genealogy_entry_index: int | None = None,
    ) -> dict:
        """Create a standardized chunk dictionary with all required fields"""

        # Use empty genealogical data if not provided
        if genealogical_data is None:
            genealogical_data = {
                "family_groups": [],
                "extraction_method": "regex",
            }

        return {
            "text_content": text_content,
            "start_page": start_page,
            "end_page": end_page,
            "chunk_type": chunk_type,
            "generation_number": generation_number,
            "family_groups": genealogical_data["family_groups"],
            "extraction_method": genealogical_data["extraction_method"],
            "related_genealogy_entry_index": related_genealogy_entry_index,
        }

    def _is_genealogy_entry(self, text: str) -> bool:
        """Detect if text line starts a genealogical entry (e.g., 'n. Bessel van Zanten, * ...')"""
        import re

        text = text.strip()

        # Standard pattern: letter + name + genealogical markers
        standard_pattern = r"^[a-z]\.\s+[A-Z][a-zA-Z\s]+.*[*+~,]"

        # Kind entries: "a. Kind, o place date"
        kind_pattern = r"^[a-z]\.\s+Kind[,\s]"

        # OCR-corrupted pattern: missing letter but has genealogical markers near name
        # Must have ~ (baptism) or * (birth) within first part of line
        missing_letter_pattern = r"^\.\s+[A-Z][a-zA-Z\s]+[,\s]*[~*][^,]{0,50}[,.]"

        return bool(
            re.match(standard_pattern, text) or re.match(kind_pattern, text) or re.match(missing_letter_pattern, text)
        )

    def _is_narrative_content(self, text: str) -> bool:
        """Use spaCy NLP to detect if text contains narrative prose vs biographical fragments"""
        text = text.strip()

        if not text:
            return False

        # Strong indicators this is biographical data, not narrative
        biographical_indicators = [
            r"\*.*\d{4}",
            r"\+.*\d{4}",
            r"~.*\d{4}",
            r"x \d+\.",
            r"dv ",
            r"zv ",
            r"wed ",
            r"\d{4}[-–]\d{4}:",
            r"[A-Z][a-z]+(straat|gracht|kade|plein|laan)",
            r"[a-z]+\s*\(\d{4}\)",
        ]

        # If text has strong biographical markers, bias against narrative classification
        bio_marker_count = sum(1 for pattern in biographical_indicators if re.search(pattern, text, re.IGNORECASE))

        if bio_marker_count >= 2:
            return False

        # Load appropriate spaCy model based on content
        english_indicators = ["the ", "and ", "of ", "to ", "in ", "a ", "is ", "was ", "he ", "she "]
        is_likely_english = any(indicator in text.lower() for indicator in english_indicators)

        if is_likely_english:
            if not hasattr(self, "_nlp_en"):
                self._nlp_en = spacy.load("en_core_web_sm")
            nlp = self._nlp_en
        else:
            if not hasattr(self, "_nlp_nl"):
                self._nlp_nl = spacy.load("nl_core_news_sm")
            nlp = self._nlp_nl

        # Analyze with spaCy
        doc = nlp(text)
        sentences = list(doc.sents)

        # Multiple sentences strongly indicate narrative
        if len(sentences) > 1:
            return True

        # Single sentence analysis
        if len(sentences) == 1:
            sent = sentences[0]

            # Look for complete sentence structure with subject and verb
            has_subject = any(token.dep_ in ["nsubj", "nsubj:pass"] for token in sent)
            has_verb = any(token.pos_ == "VERB" for token in sent)

            # Look for narrative linguistic markers
            has_pronouns = any(token.pos_ == "PRON" for token in sent)
            has_past_tense = any(token.pos_ == "VERB" and "Past" in str(token.morph) for token in sent)

            # For single biographical markers, be more conservative
            if bio_marker_count >= 1:
                return has_subject and has_verb and has_pronouns and has_past_tense

            # Standard narrative detection for non-biographical content
            if has_subject and has_verb and (has_pronouns or has_past_tense):
                return True

        return False

    def _extract_generation_number(self, line: str) -> int | None:
        """Extract generation number from a line"""
        line_lower = line.lower().strip()

        # Only consider lines that are short and likely actual headers
        if len(line.strip()) > 30:
            return None

        for dutch_name, number in self.GENERATION_MAPPING.items():
            # Check if the line is exactly or nearly exactly the generation name
            if dutch_name == line_lower or line_lower == dutch_name.upper():
                return number
        return None

    def _is_kwartierstaat_header(self, line: str) -> bool:
        """Detect KWARTIERSTATEN section headers"""
        line = line.strip()

        # Main KWARTIERSTATEN header (all caps)
        if line.upper() == "KWARTIERSTATEN":
            return True

        # Individual chart headers: "1. Kwartierstaat van ..." or numbered variants
        kwartierstaat_pattern = r"^\d+\.\s+Kwartierstaat\s+van\s+"
        if re.match(kwartierstaat_pattern, line, re.IGNORECASE):
            return True

        return False

    def _is_appendix_header(self, line: str) -> bool:
        """Detect appendix/register section headers (Bijlage 1, Bijlage 2, Register)"""
        line = line.strip()

        # Bijlage headers: "Bijlage 1", "Bijlage 2", etc.
        bijlage_pattern = r"^Bijlage\s+\d+$"
        if re.match(bijlage_pattern, line, re.IGNORECASE):
            return True

        # Register header
        if line.upper() == "REGISTER":
            return True

        return False

    def _is_family_group_header(self, line: str) -> bool:
        """Detect family group headers like 'II.1. Kinderen van...' or '1. Kinderen van...' (OCR corrupted)"""
        line = line.strip()

        # Standard pattern: Roman numeral + number + "Children of"/"Kinderen van"
        # II.1. Kinderen van Gerrit van Santen en Lijsbet
        standard_pattern = r"\b[IVXLCDMilvxlcdm]+\.\d+\.\s+(?:Kinderen\s+van|Children\s+of)"

        # OCR-corrupted pattern: Missing Roman numeral, just number + "Children of"/"Kinderen van"
        # 1. Kinderen van Gerrit van Santen en Lijsbet (missing "II.")
        corrupted_pattern = r"^\d+\.\s+(?:Kinderen\s+van|Children\s+of)"

        # OCR-corrupted pattern: "/" instead of Roman numeral
        # /.1. Kinderen van... (should be IV.1. Kinderen van...)
        slash_pattern = r"^/\.\d+\.\s+(?:Kinderen\s+van|Children\s+of)"

        # OCR-corrupted pattern: Roman numeral with digit appended (missing dot)
        # X1.1. Children of... (should be XI.1. Children of...)
        # X11.1. Children of... (should be XII.1. Children of...)
        roman_digit_pattern = r"^[IVXLCDMilvxlcdm]+\d+\.\d+\.\s+(?:Kinderen\s+van|Children\s+of)"

        # OCR-corrupted pattern: Missing dot after number
        # X.14 Children of... (should be X.14. Children of...)
        missing_dot_pattern = r"^[IVXLCDMilvxlcdm]+\.\d+\s+(?:Kinderen\s+van|Children\s+of)"

        # Second marriage pattern: No prefix, just "Children of"/"Kinderen van" at start
        # Children of Harold Richard Van Zanten and Mary Jane Sanko (IX.6.b):
        # This occurs for second marriages as sub-groups
        second_marriage_pattern = r"^(?:Kinderen\s+van|Children\s+of)\s+[A-Z]"

        return bool(
            re.search(standard_pattern, line, re.IGNORECASE) or
            re.match(corrupted_pattern, line, re.IGNORECASE) or
            re.match(slash_pattern, line, re.IGNORECASE) or
            re.match(roman_digit_pattern, line, re.IGNORECASE) or
            re.match(missing_dot_pattern, line, re.IGNORECASE) or
            re.match(second_marriage_pattern, line, re.IGNORECASE)
        )

    def _extract_family_context(self, line: str, current_generation: int) -> str | None:
        """Extract family context from family group headers"""
        # Pattern for family group headers: "X.9. Children of..." or
        # "X.9. Kinderen van..." or "/.9. Kinderen van..." (OCR corrupted)
        family_pattern = r"\b([IVXLCDMilvxlcdm/]+|[A-Z/]+)\.(\d+)\.\s+(?:Children\s+of|Kinderen\s+van)"
        match = re.search(family_pattern, line, re.IGNORECASE)

        if match:
            roman_part, number_part = match.groups()
            # Apply OCR corrections
            corrected_roman = self.OCR_CORRECTIONS.get(roman_part, roman_part)
            if not self._is_valid_roman(corrected_roman):
                corrected_roman = self._number_to_roman(current_generation)
            return f"{corrected_roman}.{number_part}"

        # OCR-corrupted pattern: Roman numeral with digit appended (missing dot)
        # X1.1. Children of... (should be XI.1. Children of...)
        roman_digit_pattern = r"^([IVXLCDMilvxlcdm]+)(\d+)\.(\d+)\.\s+(?:Children\s+of|Kinderen\s+van)"
        match = re.match(roman_digit_pattern, line, re.IGNORECASE)

        if match:
            roman_part, appended_digit, number_part = match.groups()
            # Try to reconstruct the correct Roman numeral
            # X1 -> XI, X11 -> XII, etc.
            reconstructed = roman_part + "I" * int(appended_digit)
            corrected_roman = self.OCR_CORRECTIONS.get(reconstructed, reconstructed)
            if not self._is_valid_roman(corrected_roman):
                corrected_roman = self._number_to_roman(current_generation)
            return f"{corrected_roman}.{number_part}"

        # OCR-corrupted pattern: Missing dot after number
        # X.14 Children of... (should be X.14. Children of...)
        missing_dot_pattern = r"^([IVXLCDMilvxlcdm]+)\.(\d+)\s+(?:Children\s+of|Kinderen\s+van)"
        match = re.match(missing_dot_pattern, line, re.IGNORECASE)

        if match:
            roman_part, number_part = match.groups()
            corrected_roman = self.OCR_CORRECTIONS.get(roman_part, roman_part)
            if not self._is_valid_roman(corrected_roman):
                corrected_roman = self._number_to_roman(current_generation)
            return f"{corrected_roman}.{number_part}"

        # Second marriage pattern: No prefix, use full text as context
        # Children of Harold Richard Van Zanten and Mary Jane Sanko (IX.6.b):
        second_marriage_pattern = r"^(?:Kinderen\s+van|Children\s+of)\s+[A-Z]"
        if re.match(second_marriage_pattern, line, re.IGNORECASE):
            # Return the full line (without trailing colon if present) as the context
            return line.rstrip(':').strip()

        return None

    def _extract_genealogy_ids(
        self,
        text: str,
        current_generation: int,
        current_family_context: str | None = None,
    ) -> tuple[list[str], str]:
        """Extract and correct genealogical IDs from text using regex patterns only

        Returns:
            tuple: (corrected_ids, extraction_method)
        """
        corrected_ids = []
        extraction_method = "regex"

        # Extract explicit genealogical IDs using centralized patterns
        explicit_matches = GenealogyPatterns.GENEALOGY_ID.findall(text)

        for match_tuple in explicit_matches:
            # Pattern has 3 alternatives with 3 groups each = 9 total groups
            # Find the non-empty group set (only one alternative will match)
            roman_part = match_tuple[0] or match_tuple[3] or match_tuple[6]
            number_part = match_tuple[1] or match_tuple[4] or match_tuple[7]
            letter_part = match_tuple[2] or match_tuple[5] or match_tuple[8]

            if roman_part and number_part and letter_part:
                corrected_id = self._correct_genealogy_id(
                    roman_part,
                    number_part,
                    letter_part,
                    current_generation,
                )
                if corrected_id:
                    corrected_ids.append(corrected_id)

        # Extract family group IDs using centralized patterns
        family_matches = GenealogyPatterns.FAMILY_GROUP.findall(text)
        for match_tuple in family_matches:
            # Pattern has 3 alternatives with 2 groups each = 6 total groups
            roman_part = match_tuple[0] or match_tuple[2] or match_tuple[4]
            number_part = match_tuple[1] or match_tuple[3] or match_tuple[5]

            if roman_part and number_part:
                corrected_roman = self.OCR_CORRECTIONS.get(roman_part, roman_part)
                if not self._is_valid_roman(corrected_roman):
                    corrected_roman = self._number_to_roman(current_generation)
                family_id = f"{corrected_roman}.{number_part}"
                corrected_ids.append(family_id)

        # Add inferred IDs from family context
        if current_family_context:
            individual_pattern = r"\b([a-z])\.\s+([A-Z][a-zA-Z\s]+)"
            individual_matches = re.findall(individual_pattern, text)
            for letter, _name in individual_matches:
                inferred_id = f"{current_family_context}.{letter}"
                corrected_ids.append(inferred_id)

        return list(set(corrected_ids)), extraction_method  # Remove duplicates

    def _extract_genealogical_data(
        self,
        text: str,
        current_generation: int,
        current_family_context: str | None = None,
    ) -> dict:
        """Extract all genealogical data (IDs, dates, places, names, groups) from text

        Returns:
            dict: {
                'family_groups': list[str],
                'extraction_method': str
            }
        """
        result = {
            "family_groups": [],
            "extraction_method": "regex",  # Default
        }

        # Extract family groups (full text matches)
        family_group_matches = GenealogyPatterns.FAMILY_GROUP.finditer(text)
        family_groups = [match.group(0).strip() for match in family_group_matches]

        # If we have a current family context from a header, add it to family_groups
        # This ensures GENEALOGY_ENTRY chunks inherit the family group context
        if current_family_context and current_family_context not in family_groups:
            family_groups.append(current_family_context)

        result["family_groups"] = family_groups

        logger.debug(f"Extracted family groups: {family_groups}")

        return result

    def _correct_genealogy_id(
        self,
        roman_part: str,
        number_part: str,
        letter_part: str,
        current_generation: int,
    ) -> str | None:
        """Correct OCR errors in genealogical ID"""
        # Apply OCR corrections
        corrected_roman = self.OCR_CORRECTIONS.get(roman_part, roman_part)

        # Ensure letter is lowercase
        letter_part = letter_part.lower()

        # If roman numeral still looks wrong, use current generation context
        if not self._is_valid_roman(corrected_roman):
            corrected_roman = self._number_to_roman(current_generation)

        return f"{corrected_roman}.{number_part}.{letter_part}"

    def _is_valid_roman(self, roman: str) -> bool:
        """Check if string is a valid Roman numeral"""
        valid_romans = {
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
            "XII",
            "XIII",
            "XIV",
            "XV",
            "XVI",
            "XVII",
            "XVIII",
            "XIX",
            "XX",
        }
        return roman.upper() in valid_romans

    def _number_to_roman(self, number: int) -> str:
        """Convert number to Roman numeral"""
        mapping = [
            (20, "XX"),
            (19, "XIX"),
            (18, "XVIII"),
            (17, "XVII"),
            (16, "XVI"),
            (15, "XV"),
            (14, "XIV"),
            (13, "XIII"),
            (12, "XII"),
            (11, "XI"),
            (10, "X"),
            (9, "IX"),
            (8, "VIII"),
            (7, "VII"),
            (6, "VI"),
            (5, "V"),
            (4, "IV"),
            (3, "III"),
            (2, "II"),
            (1, "I"),
        ]

        for value, roman in mapping:
            if number >= value:
                return roman
        return "I"
