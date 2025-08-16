import logging
import os

from django.core.exceptions import ValidationError

from celery import shared_task

from .models import Document, DocumentPage
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

        logger.info(
            f"Started OCR processing for {len(task_ids)} pages in document {document}"
        )

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
