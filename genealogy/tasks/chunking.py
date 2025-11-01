"""Text chunking tasks for genealogy documents

This module handles chunking of OCR text into hierarchical genealogical chunks.
It processes the entire book at once to maintain context across pages.
"""
import logging

from django.core.exceptions import ValidationError

from celery import shared_task

from ..models import Document
from ..chunking import GenealogicalTextChunker, save_chunks_to_db

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def create_document_chunks(self, document_id: str):  # noqa: ARG001
    """
    Create text chunks with genealogical anchors for an entire document.

    Processes the ENTIRE BOOK at once by concatenating all pages, which allows:
    - Maintaining genealogical context (generation, family group) across pages
    - Detecting info boxes that span multiple pages
    - Proper text flow restoration (main narrative vs info boxes)

    Uses GenealogicalTextChunker with DeepSeek-OCR grounding tokens.

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

        # Get all OCR-completed pages in order
        pages = document.pages.filter(ocr_completed=True).order_by('page_number')

        if not pages.exists():
            return {
                "success": False,
                "error": "No OCR-completed pages found",
                "document_id": str(document_id),
            }

        # Concatenate all pages into one book-level OCR text
        logger.info(f"Concatenating {pages.count()} pages into single book text")

        book_text_parts = []
        page_map = []  # Track which character ranges belong to which pages

        for page in pages:
            if not page.ocr_text or not page.ocr_text.strip():
                logger.warning(f"Skipping page {page.page_number} - no OCR text")
                continue

            # Track the character position where this page starts
            start_char = sum(len(part) + 2 for part in book_text_parts)  # +2 for "\n\n"
            page_text = page.ocr_text
            end_char = start_char + len(page_text)

            page_map.append({
                'page_number': page.page_number,
                'page_obj': page,
                'start_char': start_char,
                'end_char': end_char,
            })

            book_text_parts.append(page_text)

        # Join all pages with double newline separator
        book_text = "\n\n".join(book_text_parts)
        logger.info(f"Created book text: {len(book_text)} characters across {len(page_map)} pages")

        # Create chunker for whole book
        # Pass document so chunker can look up BookSections
        chunker = GenealogicalTextChunker(document=document)

        # Parse entire book into chunks
        # This maintains genealogical context across pages and handles:
        # - Generation headers that affect multiple pages
        # - Family groups that span pages
        # - Info boxes that start on one page and continue on the next
        logger.info("Chunking entire book text...")
        book_chunks = chunker.chunk(book_text)
        logger.info(f"Created {len(book_chunks)} chunks from book text")

        # Save chunks to database with page mapping
        # page_map and book_text are used to determine page numbers from grounding tokens
        saved_chunks = save_chunks_to_db(
            book_chunks,
            document,
            page_map,
            book_text
        )
        chunks_created = len(saved_chunks)

        logger.info(f"Saved {chunks_created} chunks for document {document}")

        return {
            "success": True,
            "message": f"Created {chunks_created} text chunks from entire book",
            "document_id": str(document_id),
            "chunks_created": chunks_created,
            "pages_processed": len(page_map),
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
