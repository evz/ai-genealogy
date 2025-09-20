import logging
import os
import re

from django.core.exceptions import ValidationError

import spacy
from celery import chain, shared_task
from pdf2image import convert_from_path
from PIL import Image

from .date_standardizer import standardize_dates
from .document_layout_detector import DocumentLayoutDetector
from .models import Document, DocumentPage, TextChunk
from .ner_extractor import get_default_ner_extractor
from .patterns import GenealogyPatterns
from .region_ocr_processor import RegionOCRProcessor
from .rotation_detector import RotationDetector

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def process_page_ocr(self, page_id: str):  # noqa: ARG001
    """
    Process OCR for a single DocumentPage

    Args:
        page_id: UUID string of the DocumentPage to process

    Returns:
        dict: Processing result with text, confidence, and status
    """
    try:
        # Get the page
        page = DocumentPage.objects.get(id=page_id)
        logger.info(f"Starting OCR processing for page {page}")

        # Check if already processed
        if page.ocr_completed:
            logger.info(f"Page {page_id} already processed, skipping")
            return {
                "success": True,
                "message": "Already processed",
                "text": page.ocr_text,
                "confidence": page.ocr_confidence,
            }

        # Get language from document
        language = page.document.languages

        # Initialize modular OCR components
        rotation_detector = RotationDetector()
        layout_detector = DocumentLayoutDetector()
        ocr_processor = RegionOCRProcessor(tesseract_language=language if language else "eng+nld")

        # Process the image file
        file_path = page.image_file.path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Load image (handle both PDF and image files)
        if file_path.lower().endswith(".pdf"):
            # Convert PDF to image (first page only)
            images = convert_from_path(file_path, first_page=1, last_page=1)
            if not images:
                raise ValueError(f"Could not convert PDF to image: {file_path}")
            image = images[0]
        else:
            # Load image file directly
            image = Image.open(file_path)

        # Step 1: Detect and correct rotation
        logger.info(f"Step 1: Rotation detection for page {page}")
        corrected_image, rotation_applied = rotation_detector.detect_and_correct(image)

        # Step 2: Detect document layout regions
        logger.info(f"Step 2: Layout detection for page {page}")
        regions = layout_detector.detect_regions(corrected_image)

        # Step 3: Process regions with OCR
        logger.info(f"Step 3: OCR processing {len(regions)} regions for page {page}")
        text, confidence = ocr_processor.process_regions(corrected_image, regions)

        # Update the page with results
        page.ocr_text = text
        page.ocr_confidence = confidence
        page.rotation_applied = rotation_applied
        page.ocr_completed = True
        page.save()

        # Update parent document OCR status
        page.document.update_ocr_status()

        logger.info(f"OCR completed for page {page}. Confidence: {confidence:.1f}%")

        return {
            "success": True,
            "message": "OCR completed successfully",
            "text": text,
            "confidence": confidence,
            "rotation_applied": rotation_applied,
            "page_id": str(page_id),
        }

    except ValidationError:
        error_msg = f"Invalid UUID format: {page_id}"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except DocumentPage.DoesNotExist:
        error_msg = f"DocumentPage with id {page_id} not found"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"OCR processing failed for page {page_id}: {e!s}"
        logger.error(error_msg, exc_info=True)

        # Update page to indicate failure (but don't mark as completed)
        try:
            page = DocumentPage.objects.get(id=page_id)
            # Could add an error field to track failures
        except DocumentPage.DoesNotExist:
            pass

        return {
            "success": False,
            "error": error_msg,
            "page_id": str(page_id),
        }


@shared_task(bind=True)
def process_document_ocr(self, document_id: str):  # noqa: ARG001
    """
    Process OCR for all pages in a document

    Args:
        document_id: UUID string of the Document to process

    Returns:
        dict: Processing result summary
    """
    try:
        # Get the document
        document = Document.objects.get(id=document_id)
        logger.info(f"Starting OCR processing for document {document}")

        # Get all unprocessed pages
        unprocessed_pages = document.pages.filter(ocr_completed=False)

        if not unprocessed_pages.exists():
            logger.info(f"No unprocessed pages found for document {document_id}")
            return {
                "success": True,
                "message": "No pages to process",
                "pages_processed": 0,
            }

        # Start OCR tasks for each page
        task_ids = []
        for page in unprocessed_pages:
            task = process_page_ocr.delay(str(page.id))
            task_ids.append(task.id)

        logger.info(f"Started OCR processing for {len(task_ids)} pages in document {document}")

        return {
            "success": True,
            "message": f"OCR processing started for {len(task_ids)} pages",
            "pages_processed": len(task_ids),
            "task_ids": task_ids,
            "document_id": str(document_id),
        }

    except ValidationError:
        error_msg = f"Invalid UUID format: {document_id}"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Document.DoesNotExist:
        error_msg = f"Document with id {document_id} not found"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"Document OCR processing failed for {document_id}: {e!s}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "document_id": str(document_id),
        }


class GenealogyChunker:
    """Helper class for chunking documents based on genealogical structure"""

    # Dutch generation names to Roman numerals
    GENERATION_MAPPING = {
        "eerste generatie": 1,
        "tweede generatie": 2,
        "derde generatie": 3,
        "vierde generatie": 4,
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
        # Try to get the neural network extractor
        # Switched back to NER-first extraction after Phase 3.2 training completion
        self.ner_extractor = get_default_ner_extractor()
        if self.ner_extractor:
            logger.info("Using neural network-based genealogy entity extraction")
        else:
            logger.info("Using regex-based genealogy entity extraction (fallback)")

    def chunk_document_text(self, document: Document) -> list[dict]:
        """
        Chunk document text based on genealogical structure

        Returns:
            List of dicts with keys: text_content, start_page, end_page, chunk_type,
            generation_number, genealogy_ids, dates, places, person_names,
            family_groups, extraction_method
        """
        full_text = document.get_combined_ocr_text()
        if not full_text.strip():
            return []

        chunks = []
        lines = full_text.split("\n")
        current_chunk_lines = []
        current_start_page = 1
        current_end_page = 1
        current_generation = 1
        current_family_context = None  # Track current family group (e.g., "X.9")
        current_genealogy_entry = None  # Track current genealogy entry for linking narrative chunks

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

                # Create family group header chunk
                header_chunk = self._create_chunk_dict(
                    text_content=line.strip(),
                    chunk_type="HEADER",
                    start_page=current_end_page,
                    end_page=current_end_page,
                    generation_number=current_generation,
                )
                chunks.append(header_chunk)

                # Update family context for subsequent individual entries
                current_family_context = self._extract_family_context(line, current_generation)

                # Reset for new chunk
                current_chunk_lines = []
                current_start_page = current_end_page
                i += 1
                continue

            # Check for genealogical entries (dense biographical entries)
            if self._is_genealogy_entry(line):
                # Finish current chunk if it has content
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        genealogical_data = self._extract_genealogical_data(
                            chunk_text, current_generation, current_family_context
                        )

                        # Determine chunk type for previous content
                        chunk_type = "NARRATIVE_CONTEXT" if current_genealogy_entry is not None else "CONTENT"

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

                # Start new genealogy entry - collect all lines until we hit another entry or generation
                genealogy_lines = [line]

                # Look ahead to collect multi-line genealogy entry
                j = i + 1
                consecutive_narrative_lines = 0
                while j < len(lines):
                    next_line = lines[j]

                    # Stop if we hit another genealogy entry, generation header, or page marker
                    if (
                        self._is_genealogy_entry(next_line)
                        or self._extract_generation_number(next_line)
                        or re.match(r"=== Page (\d+) ===", next_line)
                    ):
                        break

                    # Check for narrative content but be more robust
                    if self._is_narrative_content(next_line):
                        consecutive_narrative_lines += 1
                        # Only stop after 2 consecutive narrative lines to avoid false positives
                        if consecutive_narrative_lines >= 2:
                            break
                    else:
                        consecutive_narrative_lines = 0

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

                # If chunk gets too large (>2000 chars), split it
                chunk_text = "\n".join(current_chunk_lines)
                if len(chunk_text) > 2000:
                    genealogical_data = self._extract_genealogical_data(
                        chunk_text, current_generation, current_family_context
                    )

                    # Determine chunk type based on context
                    chunk_type = "NARRATIVE_CONTEXT" if current_genealogy_entry is not None else "CONTENT"

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

                # Determine chunk type based on context
                chunk_type = "NARRATIVE_CONTEXT" if current_genealogy_entry is not None else "CONTENT"

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
                "genealogy_ids": [],
                "dates": [],
                "places": [],
                "person_names": [],
                "family_groups": [],
                "occupations": [],
                "source_citations": [],
                "extraction_method": "regex",
            }

        return {
            "text_content": text_content,
            "start_page": start_page,
            "end_page": end_page,
            "chunk_type": chunk_type,
            "generation_number": generation_number,
            "genealogy_ids": genealogical_data["genealogy_ids"],
            "dates": genealogical_data["dates"],
            "places": genealogical_data["places"],
            "person_names": genealogical_data["person_names"],
            "family_groups": genealogical_data["family_groups"],
            "occupations": genealogical_data["occupations"],
            "source_citations": genealogical_data["source_citations"],
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

    def _is_family_group_header(self, line: str) -> bool:
        """Detect family group headers like 'II.1. Kinderen van...' or '1. Kinderen van...' (OCR corrupted)"""
        line = line.strip()

        # Standard pattern: Roman numeral + number + "Children of"/"Kinderen van"
        # II.1. Kinderen van Gerrit van Santen en Lijsbet
        standard_pattern = r"\b[IVXLCDMilvxlcdm]+\.\d+\.\s+(?:Kinderen\s+van|Children\s+of)"

        # OCR-corrupted pattern: Missing Roman numeral, just number + "Children of"/"Kinderen van"
        # 1. Kinderen van Gerrit van Santen en Lijsbet (missing "II.")
        corrupted_pattern = r"^\d+\.\s+(?:Kinderen\s+van|Children\s+of)"

        return bool(
            re.search(standard_pattern, line, re.IGNORECASE) or re.match(corrupted_pattern, line, re.IGNORECASE)
        )

    def _extract_family_context(self, line: str, current_generation: int) -> str | None:
        """Extract family context from family group headers"""
        # Pattern for family group headers: "X.9. Children of..." or
        # "X.9. Kinderen van..."
        family_pattern = r"\b([IVXLCDMilvxlcdm]+|[A-Z]+)\.(\d+)\.\s+(?:Children\s+of|Kinderen\s+van)"
        match = re.search(family_pattern, line, re.IGNORECASE)

        if match:
            roman_part, number_part = match.groups()
            # Apply OCR corrections
            corrected_roman = self.OCR_CORRECTIONS.get(roman_part, roman_part)
            if not self._is_valid_roman(corrected_roman):
                corrected_roman = self._number_to_roman(current_generation)
            return f"{corrected_roman}.{number_part}"

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
                'genealogy_ids': list[str],
                'dates': list[str],  # Standardized ISO dates
                'places': list[str],
                'person_names': list[str],
                'family_groups': list[str],
                'occupations': list[str],
                'extraction_method': str
            }
        """
        result = {
            "genealogy_ids": [],
            "dates": [],
            "places": [],
            "person_names": [],
            "family_groups": [],
            "occupations": [],
            "source_citations": [],
            "extraction_method": "regex",  # Default
        }

        # Extract genealogy IDs using existing method
        genealogy_ids, extraction_method = self._extract_genealogy_ids(text, current_generation, current_family_context)
        result["genealogy_ids"] = genealogy_ids
        result["extraction_method"] = extraction_method

        # Extract other genealogical data using regex patterns
        # Extract dates - use comprehensive pattern first, then fill gaps with year pattern
        raw_dates = []

        # Find comprehensive dates and their positions
        comprehensive_matches = list(GenealogyPatterns.DATE_COMPREHENSIVE.finditer(text))
        comprehensive_dates = [match.group(0) for match in comprehensive_matches]
        raw_dates.extend(comprehensive_dates)

        # Find year-only dates that don't overlap with comprehensive matches
        year_matches = list(GenealogyPatterns.YEAR.finditer(text))
        for year_match in year_matches:
            year_start, year_end = year_match.span()
            # Check if this year overlaps with any comprehensive match
            overlaps = any(
                comp_start <= year_start < comp_end or comp_start < year_end <= comp_end
                for comp_start, comp_end in [comp_match.span() for comp_match in comprehensive_matches]
            )
            if not overlaps:
                raw_dates.append(year_match.group(0))

        if raw_dates:
            result["dates"] = standardize_dates(raw_dates)

        # Extract places, person names, and occupations using NER model if available
        if self.ner_extractor:
            try:
                ner_entities = self.ner_extractor.extract_entities(text)
                result["places"] = [entity["text"] for entity in ner_entities.get("PLACE", [])]
                result["person_names"] = [entity["text"] for entity in ner_entities.get("PERSON_NAME", [])]
                result["occupations"] = [entity["text"] for entity in ner_entities.get("OCCUPATION", [])]
                result["source_citations"] = [entity["text"] for entity in ner_entities.get("SOURCE", [])]
                result["extraction_method"] = "ner+regex"
            except Exception as e:
                logger.warning(f"NER extraction failed, using regex fallback: {e}")
                result["places"] = []
                result["person_names"] = []
                result["occupations"] = []
        else:
            # Fallback to regex patterns if NER not available
            result["places"] = []
            result["person_names"] = []
            result["occupations"] = []

        # Extract family groups (full text matches)
        family_group_matches = GenealogyPatterns.FAMILY_GROUP.finditer(text)
        family_groups = [match.group(0).strip() for match in family_group_matches]
        result["family_groups"] = family_groups

        logger.debug(
            f"Extracted data - Dates: {len(result['dates'])}, "
            f"Places: {len(result['places'])}, "
            f"Names: {len(result['person_names'])}, "
            f"Occupations: {len(result['occupations'])}"
        )

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


@shared_task(bind=True)
def create_document_chunks(self, document_id: str):  # noqa: ARG001
    """
    Phase 1: Create text chunks with genealogical anchors for a document

    Args:
        document_id: UUID string of the Document to chunk

    Returns:
        dict: Chunking result summary
    """
    try:
        # Get the document
        document = Document.objects.get(id=document_id)
        logger.info(f"Starting chunking for document {document}")

        if not document.ocr_completed:
            return {
                "success": False,
                "error": "Document OCR must be completed before chunking",
                "document_id": str(document_id),
            }

        # Clear existing chunks for this document
        document.text_chunks.all().delete()

        # Create chunks
        chunker = GenealogyChunker()
        chunk_data = chunker.chunk_document_text(document)

        chunks_created = 0
        for sequence_num, chunk_dict in enumerate(chunk_data, 1):
            # For header chunks, store the clean header text
            generation_header = ""
            if chunk_dict["chunk_type"] == "HEADER":
                generation_header = chunk_dict["text_content"]  # Clean header text

            TextChunk.objects.create(
                document=document,
                text_content=chunk_dict["text_content"],
                chunk_type=chunk_dict["chunk_type"],
                start_page=chunk_dict["start_page"],
                end_page=chunk_dict["end_page"],
                sequence_number=sequence_num,
                generation_number=chunk_dict["generation_number"],
                generation_header=generation_header,
                genealogy_ids=chunk_dict["genealogy_ids"],
                person_names=chunk_dict["person_names"],
                dates=chunk_dict["dates"],
                places=chunk_dict["places"],
                family_groups=chunk_dict["family_groups"],
                occupations=chunk_dict["occupations"],
                source_citations=chunk_dict["source_citations"],
                extraction_method=chunk_dict["extraction_method"],
            )
            chunks_created += 1

        logger.info(f"Created {chunks_created} chunks for document {document}")

        return {
            "success": True,
            "message": f"Created {chunks_created} text chunks",
            "document_id": str(document_id),
            "chunks_created": chunks_created,
        }

    except ValidationError:
        error_msg = f"Invalid UUID format: {document_id}"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Document.DoesNotExist:
        error_msg = f"Document with id {document_id} not found"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"Document chunking failed for {document_id}: {e!s}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "document_id": str(document_id),
        }


@shared_task(bind=True)
def extract_entities_from_chunks(self, document_id: str):  # noqa: ARG001
    """
    Phase 2: Extract genealogy entities from document chunks using LLM

    Args:
        document_id: UUID string of the Document to extract from

    Returns:
        dict: Extraction result summary
    """
    try:
        # Get the document
        document = Document.objects.get(id=document_id)
        logger.info(f"Starting entity extraction for document {document}")

        # Get unprocessed content chunks
        unprocessed_chunks = document.text_chunks.filter(entities_extracted=False, chunk_type="CONTENT").order_by(
            "sequence_number"
        )

        if not unprocessed_chunks.exists():
            # Mark document as extraction completed
            document.extraction_completed = True
            document.save(update_fields=["extraction_completed"])

            return {
                "success": True,
                "message": "No chunks to process - extraction completed",
                "document_id": str(document_id),
                "chunks_processed": 0,
            }

        # For now, just mark chunks as processed (we'll implement LLM extraction later)
        chunks_processed = 0
        for chunk in unprocessed_chunks:
            # TODO: Implement LLM-based entity extraction here
            # For now, just mark as processed
            chunk.entities_extracted = True
            chunk.save(update_fields=["entities_extracted"])
            chunks_processed += 1

        # Mark document as extraction completed
        document.extraction_completed = True
        document.save(update_fields=["extraction_completed"])

        logger.info(f"Processed {chunks_processed} chunks for document {document}")

        return {
            "success": True,
            "message": f"Processed {chunks_processed} chunks",
            "document_id": str(document_id),
            "chunks_processed": chunks_processed,
        }

    except ValidationError:
        error_msg = f"Invalid UUID format: {document_id}"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Document.DoesNotExist:
        error_msg = f"Document with id {document_id} not found"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"Entity extraction failed for {document_id}: {e!s}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "document_id": str(document_id),
        }


@shared_task(bind=True)
def process_genealogy_extraction(self, document_id: str):  # noqa: ARG001
    """
    Multi-phase genealogy extraction: chunking + entity extraction

    Args:
        document_id: UUID string of the Document to process

    Returns:
        dict: Processing result summary
    """
    try:
        # Use Celery chain to orchestrate the multi-phase extraction
        logger.info(f"Starting multi-phase extraction chain for document {document_id}")

        # Create a chain: chunking -> entity extraction
        # Use .si() (immutable) for the second task so it doesn't receive the first task's result as an argument
        extraction_chain = chain(
            create_document_chunks.s(str(document_id)), extract_entities_from_chunks.si(str(document_id))
        )

        # Start the chain
        result = extraction_chain.apply_async()

        return {
            "success": True,
            "message": "Multi-phase extraction chain started successfully",
            "document_id": str(document_id),
            "chain_id": result.id,
        }

    except Exception as e:
        error_msg = f"Multi-phase extraction failed for {document_id}: {e!s}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "document_id": str(document_id),
        }
