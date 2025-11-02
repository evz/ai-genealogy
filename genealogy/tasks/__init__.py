"""
Genealogy tasks package

This package contains all Celery tasks for processing genealogy documents.
Tasks are organized by functionality:
- ocr: OCR processing tasks
- chunking: Text chunking tasks (book-level hierarchical chunking)
- extraction: Entity extraction tasks (LLM-based entity extraction)
"""

# Import all tasks for backward compatibility
from .chunking import create_document_chunks
from .extraction import extract_entities_from_chunks
from .ocr import process_document_ocr, process_page_ocr

__all__ = [
    # OCR tasks
    "process_page_ocr",
    "process_document_ocr",
    # Chunking tasks
    "create_document_chunks",
    # Extraction tasks
    "extract_entities_from_chunks",
]
