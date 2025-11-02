"""Text chunking tasks for genealogy documents

This module handles chunking of OCR text using section-specific strategies.
Different book sections (DESCENDANT_GENEALOGY, KWARTIERSTATEN, etc.) use
different chunking approaches.

The business logic has been extracted to ChunkingService for better testability.
"""
import logging

from django.core.exceptions import ValidationError

from celery import shared_task

from ..models import Document
from ..services import ChunkingService

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def create_document_chunks(self, document_id: str):  # noqa: ARG001
    """
    Create text chunks for a document using section-specific strategies.

    Processes each BookSection separately using its appropriate chunking strategy:
    - DESCENDANT_GENEALOGY: Complex genealogical chunking with context tracking
    - Other sections: Skipped for now (can add strategies later)

    Args:
        document_id: UUID string of the Document to chunk

    Returns:
        dict: Chunking result summary with success status and chunk counts
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

        # Get all book sections
        sections = document.book_sections.all().order_by('start_page')

        if not sections.exists():
            logger.warning(f"No BookSections defined for document {document}")
            return {
                "success": False,
                "error": "No BookSections defined - please configure book sections first",
                "document_id": str(document_id),
            }

        # Initialize chunking service
        chunking_service = ChunkingService()

        total_chunks_created = 0
        total_pages_processed = 0
        sections_processed = {}

        # Process each section with its appropriate strategy
        for section in sections:
            logger.info(f"Processing section: {section.title} ({section.section_type}, pages {section.start_page}-{section.end_page})")

            # Check if strategy wants to process this section
            if not chunking_service.should_process_section(section.section_type, section):
                logger.info(f"Strategy declined to process section")
                sections_processed[section.title] = {
                    'strategy': section.section_type,
                    'chunks': 0,
                    'status': 'skipped'
                }
                continue

            # Get pages for this section
            section_pages = document.pages.filter(
                page_number__gte=section.start_page,
                page_number__lte=section.end_page,
                ocr_completed=True
            ).order_by('page_number')

            if not section_pages.exists():
                logger.warning(f"No OCR-completed pages found for section {section.title}")
                sections_processed[section.title] = {
                    'strategy': section.section_type,
                    'chunks': 0,
                    'status': 'no_pages'
                }
                continue

            # Concatenate pages for this section
            section_text_parts = []
            section_page_map = []

            for page in section_pages:
                if not page.ocr_text or not page.ocr_text.strip():
                    logger.warning(f"Skipping page {page.page_number} - no OCR text")
                    continue

                start_char = sum(len(part) + 2 for part in section_text_parts)
                page_text = page.ocr_text
                end_char = start_char + len(page_text)

                section_page_map.append({
                    'page_number': page.page_number,
                    'page_obj': page,
                    'start_char': start_char,
                    'end_char': end_char,
                })

                section_text_parts.append(page_text)

            section_text = "\n\n".join(section_text_parts)
            logger.info(f"Section text: {len(section_text)} characters across {len(section_page_map)} pages")

            # Use service to chunk the section
            result = chunking_service.chunk_section(
                section_type=section.section_type,
                section_text=section_text,
                document=document,
                page_map=section_page_map,
                start_sequence=total_chunks_created + 1
            )

            if result['success']:
                chunks_created = result['chunks_created']
                total_chunks_created += chunks_created
                total_pages_processed += len(section_page_map)

                sections_processed[section.title] = {
                    'strategy': section.section_type,
                    'chunks': chunks_created,
                    'pages': len(section_page_map),
                    'status': 'success'
                }

                logger.info(f"Section '{section.title}' complete: {chunks_created} chunks created")
            else:
                logger.error(f"Failed to chunk section {section.title}: {result.get('error')}")
                sections_processed[section.title] = {
                    'strategy': section.section_type,
                    'chunks': 0,
                    'status': 'error',
                    'error': result.get('error')
                }

        logger.info(f"Chunking complete: {total_chunks_created} chunks across {len(sections_processed)} sections")

        return {
            "success": True,
            "message": f"Created {total_chunks_created} text chunks from {len(sections_processed)} sections",
            "document_id": str(document_id),
            "chunks_created": total_chunks_created,
            "pages_processed": total_pages_processed,
            "sections_processed": sections_processed,
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
