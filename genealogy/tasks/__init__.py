"""
Genealogy tasks package

This package contains all Celery tasks for processing genealogy documents.
Tasks are organized by functionality:
- ocr: OCR processing tasks
- chunking: Text chunking tasks (book-level hierarchical chunking)
- extraction: Entity extraction tasks (LLM-based entity extraction)
- build_genealogy_graph: Build Person/Relationship/Partnership from genealogical IDs
- conversation: Conversation-related tasks (title generation, etc.)
- chat_agent: Chat agent processing with SSE streaming
- source_image: Transcription + translation of archival SourceImage uploads
"""

# Import all tasks for backward compatibility
from .build_genealogy_graph import build_genealogy_graph
from .chat_agent import process_chat_message
from .chunking import create_document_chunks
from .conversation import generate_conversation_title
from .extraction import extract_entities_from_chunks
from .ocr import process_document_ocr, process_page_ocr
from .persist_entities import persist_extracted_entities
from .source_image import transcribe_source_image, translate_source_image
from .summarization import summarize_chunk, summarize_all_chunks

__all__ = [
    # OCR tasks
    "process_page_ocr",
    "process_document_ocr",
    # Source image tasks
    "transcribe_source_image",
    "translate_source_image",
    # Chunking tasks
    "create_document_chunks",
    # Extraction tasks
    "extract_entities_from_chunks",
    # Entity persistence tasks
    "persist_extracted_entities",
    # Graph building tasks
    "build_genealogy_graph",
    # Conversation tasks
    "generate_conversation_title",
    # Chat agent tasks
    "process_chat_message",
    # Summarization tasks
    "summarize_chunk",
    "summarize_all_chunks",
]
