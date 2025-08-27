import logging
import os
import re

from django.core.exceptions import ValidationError

from celery import shared_task

from .date_standardizer import standardize_dates
from .models import Document, DocumentPage, TextChunk
from .ner_extractor import get_default_ner_extractor
from .ocr_processor import OCRProcessor

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

        # Initialize OCR processor
        processor = OCRProcessor(language=language)

        # Process the image file
        file_path = page.image_file.path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Perform OCR
        text, confidence, rotation_applied = processor.process_file(file_path)

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
    }

    def __init__(self):
        self.current_generation = 1
        # Try to get the neural network extractor
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

        for line in lines:
            # Track page numbers
            page_match = re.match(r"=== Page (\d+) ===", line)
            if page_match:
                current_end_page = int(page_match.group(1))
                if not current_chunk_lines:
                    current_start_page = current_end_page
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
                        chunks.append(
                            {
                                "text_content": chunk_text,
                                "start_page": current_start_page,
                                "end_page": current_end_page,
                                "chunk_type": "CONTENT",
                                "generation_number": current_generation,
                                "genealogy_ids": genealogical_data["genealogy_ids"],
                                "dates": genealogical_data["dates"],
                                "places": genealogical_data["places"],
                                "person_names": genealogical_data["person_names"],
                                "family_groups": genealogical_data["family_groups"],
                                "extraction_method": genealogical_data["extraction_method"],
                            }
                        )

                # Create header chunk
                chunks.append(
                    {
                        "text_content": line.strip(),
                        "start_page": current_end_page,
                        "end_page": current_end_page,
                        "chunk_type": "HEADER",
                        "generation_number": generation_num,
                        "genealogy_ids": [],
                        "dates": [],
                        "places": [],
                        "person_names": [],
                        "family_groups": [],
                        "extraction_method": "regex",  # Headers don't use neural network extraction
                    }
                )

                # Reset for new chunk
                current_chunk_lines = []
                current_start_page = current_end_page
                current_generation = generation_num
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
                    chunks.append(
                        {
                            "text_content": chunk_text,
                            "start_page": current_start_page,
                            "end_page": current_end_page,
                            "chunk_type": "CONTENT",
                            "generation_number": current_generation,
                            "genealogy_ids": genealogical_data["genealogy_ids"],
                            "dates": genealogical_data["dates"],
                            "places": genealogical_data["places"],
                            "person_names": genealogical_data["person_names"],
                            "family_groups": genealogical_data["family_groups"],
                            "extraction_method": genealogical_data["extraction_method"],
                        }
                    )
                    current_chunk_lines = []
                    current_start_page = current_end_page

        # Handle final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                genealogical_data = self._extract_genealogical_data(
                    chunk_text, current_generation, current_family_context
                )
                chunks.append(
                    {
                        "text_content": chunk_text,
                        "start_page": current_start_page,
                        "end_page": current_end_page,
                        "chunk_type": "CONTENT",
                        "generation_number": current_generation,
                        "genealogy_ids": genealogical_data["genealogy_ids"],
                        "dates": genealogical_data["dates"],
                        "places": genealogical_data["places"],
                        "person_names": genealogical_data["person_names"],
                        "family_groups": genealogical_data["family_groups"],
                        "extraction_method": genealogical_data["extraction_method"],
                    }
                )

        return chunks

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

    def _extract_family_context(self, line: str, current_generation: int) -> str | None:
        """Extract family context from family group headers"""
        # Pattern for family group headers: "X.9. Children of..." or "X.9. Kinderen van..."
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
        """Extract and correct genealogical IDs from text, including inferred IDs

        Returns:
            tuple: (corrected_ids, extraction_method)
        """
        corrected_ids = []
        extraction_method = "regex"  # Default fallback

        # Try neural network extraction first
        if self.ner_extractor:
            try:
                entities = self.ner_extractor.extract_entities(text)

                # Extract genealogical IDs from NER results
                for genealogy_id in entities.get("GENEALOGY_ID", []):
                    if genealogy_id["confidence"] > 0.3:  # Lowered confidence threshold
                        # Parse and normalize neural network output using correction logic
                        raw_text = genealogy_id["text"].strip("():.,!? \t\n")

                        # Try to extract components using regex pattern
                        pattern_match = re.search(
                            r"([IVXLCDMilvxlcdm]+|[A-Z]+)\.(\d+)(?:\.([a-zA-Z]))?",
                            raw_text,
                        )
                        if pattern_match:
                            roman_part, number_part = (
                                pattern_match.groups()[0],
                                pattern_match.groups()[1],
                            )
                            letter_part = (
                                pattern_match.groups()[2]
                                if len(pattern_match.groups()) > 2 and pattern_match.groups()[2]
                                else None
                            )

                            if letter_part:
                                # Use existing correction method for full IDs
                                corrected_id = self._correct_genealogy_id(
                                    roman_part,
                                    number_part,
                                    letter_part,
                                    current_generation,
                                )
                                if corrected_id:
                                    corrected_ids.append(corrected_id)
                            else:
                                # Handle family group IDs (no letter part)
                                corrected_roman = self.OCR_CORRECTIONS.get(roman_part, roman_part)
                                if not self._is_valid_roman(corrected_roman):
                                    corrected_roman = self._number_to_roman(current_generation)
                                corrected_ids.append(f"{corrected_roman}.{number_part}")
                        else:
                            # If no pattern match, try to use raw text (might be clean)
                            corrected_ids.append(raw_text)

                # Extract family groups
                for family_group in entities.get("FAMILY_GROUP", []):
                    if family_group["confidence"] > 0.3:
                        # Parse family group header for ID
                        family_text = family_group["text"]
                        family_match = re.search(r"([IVX]+)\.(\d+)\.", family_text)
                        if family_match:
                            roman_part, number_part = family_match.groups()
                            corrected_roman = self.OCR_CORRECTIONS.get(roman_part, roman_part)
                            if not self._is_valid_roman(corrected_roman):
                                corrected_roman = self._number_to_roman(current_generation)
                            family_id = f"{corrected_roman}.{number_part}"
                            corrected_ids.append(family_id)

                # If we got results from NER, use them preferentially
                if corrected_ids:
                    extraction_method = "neural_network"
                    logger.debug(f"NER extracted {len(corrected_ids)} genealogy IDs from text chunk")
                    # Still add inferred IDs from family context
                    if current_family_context:
                        individual_pattern = r"\b([a-z])\.\s+([A-Z][a-zA-Z\s]+)"
                        individual_matches = re.findall(individual_pattern, text)
                        for letter, _name in individual_matches:
                            inferred_id = f"{current_family_context}.{letter}"
                            corrected_ids.append(inferred_id)

                    return list(set(corrected_ids)), extraction_method  # Remove duplicates

            except Exception as e:
                logger.warning(f"NER extraction failed, falling back to regex: {e}")
                extraction_method = "hybrid"  # Attempted NER but fell back

        # Fallback to regex-based extraction
        # 1. Extract explicit genealogical IDs (Roman.Number.letter format)
        explicit_pattern = r"\b([IVXLCDMilvxlcdm]+|[A-Z]+)\.(\d+)\.([a-zA-Z])\b"
        explicit_matches = re.findall(explicit_pattern, text)

        for roman_part, number_part, letter_part in explicit_matches:
            corrected_id = self._correct_genealogy_id(roman_part, number_part, letter_part, current_generation)
            if corrected_id:
                corrected_ids.append(corrected_id)

        # 2. Extract family group headers (both English and Dutch)
        family_group_pattern = r"\b([IVXLCDMilvxlcdm]+|[A-Z]+)\.(\d+)\.\s+(?:Children\s+of|Kinderen\s+van)"
        family_matches = re.findall(family_group_pattern, text, re.IGNORECASE)

        for roman_part, number_part in family_matches:
            corrected_roman = self.OCR_CORRECTIONS.get(roman_part, roman_part)
            if not self._is_valid_roman(corrected_roman):
                corrected_roman = self._number_to_roman(current_generation)
            family_id = f"{corrected_roman}.{number_part}"
            corrected_ids.append(family_id)  # Store family group ID

        # 3. Infer IDs for individuals with letter prefixes (if we have family context)
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
                'extraction_method': str
            }
        """
        result = {
            "genealogy_ids": [],
            "dates": [],
            "places": [],
            "person_names": [],
            "family_groups": [],
            "extraction_method": "regex",  # Default
        }

        # Extract genealogy IDs using existing method
        genealogy_ids, extraction_method = self._extract_genealogy_ids(text, current_generation, current_family_context)
        result["genealogy_ids"] = genealogy_ids
        result["extraction_method"] = extraction_method

        # Extract other genealogical data using neural network if available
        if self.ner_extractor:
            try:
                ner_results = self.ner_extractor.extract_entities(text)

                # Extract dates and standardize them
                raw_dates = [
                    date_item["text"] for date_item in ner_results.get("DATE", []) if date_item["confidence"] > 0.3
                ]
                if raw_dates:
                    result["dates"] = standardize_dates(raw_dates)

                # Extract places
                result["places"] = [
                    place_item["text"].strip()
                    for place_item in ner_results.get("PLACE", [])
                    if place_item["confidence"] > 0.3
                ]

                # Extract person names
                result["person_names"] = [
                    name_item["text"].strip()
                    for name_item in ner_results.get("PERSON_NAME", [])
                    if name_item["confidence"] > 0.3
                ]

                # Extract family groups (raw text, distinct from genealogy IDs)
                family_groups = [
                    fg_item["text"].strip()
                    for fg_item in ner_results.get("FAMILY_GROUP", [])
                    if fg_item["confidence"] > 0.3
                ]
                result["family_groups"] = family_groups

                logger.debug(
                    f"Extracted data - Dates: {len(result['dates'])}, "
                    f"Places: {len(result['places'])}, Names: {len(result['person_names'])}"
                )

            except Exception as e:
                logger.warning(f"Genealogical data extraction failed, using IDs only: {e}")

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
        # Phase 1: Create chunks
        logger.info(f"Starting Phase 1: Chunking for document {document_id}")
        chunk_result = create_document_chunks.delay(str(document_id))
        chunk_result.get(timeout=300)  # Wait for chunking to complete

        # Phase 2: Extract entities
        logger.info(f"Starting Phase 2: Entity extraction for document {document_id}")
        extract_result = extract_entities_from_chunks.delay(str(document_id))
        extract_result.get(timeout=600)  # Wait for extraction to complete

        return {
            "success": True,
            "message": "Multi-phase genealogy extraction completed",
            "document_id": str(document_id),
        }

    except Exception as e:
        error_msg = f"Multi-phase extraction failed for {document_id}: {e!s}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "document_id": str(document_id),
        }
