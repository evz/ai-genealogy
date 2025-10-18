"""
Genealogy tasks package

This package contains all Celery tasks for processing genealogy documents.
Tasks are organized by functionality:
- ocr: OCR processing tasks
- extraction: Text chunking and entity extraction tasks
- chunking: Text chunking logic (GenealogyChunker class)
"""

# Import all tasks for backward compatibility
from .extraction import create_document_chunks, extract_entities_from_chunks, extract_entities_from_chunk
from .ocr import process_document_ocr, process_page_ocr

__all__ = [
    # OCR tasks
    "process_page_ocr",
    "process_document_ocr",
    # Extraction tasks
    "create_document_chunks",
    "extract_entities_from_chunks",
    # Utility functions
    "extract_entities_from_chunk",
]
