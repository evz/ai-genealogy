"""Genealogical text chunking package

This package provides tools for chunking OCR text from genealogical documents
into semantic units with hierarchical genealogical context.

Main exports:
- save_chunks_to_db: Database persistence function
- ChunkType, TextChunk, etc.: Data models
- Parser functions for detecting chunk types and extracting entities

Note: Chunking logic is implemented via the strategy pattern in
genealogy.chunking_strategies (see DescendantGenealogyChunkingStrategy).
"""

from .models import BoundingBox, ChunkType, GroundingToken, TextChunk
from .parser import (
    detect_chunk_type,
    detect_info_box_boundary,
    extract_person_from_individual_entry,
    is_biographical_text,
    is_source_citation,
    parse_grounding_tokens,
)
from .persistence import save_chunks_to_db

__all__ = [
    # Main functions
    'save_chunks_to_db',
    # Data models
    'ChunkType',
    'BoundingBox',
    'GroundingToken',
    'TextChunk',
    # Parser functions
    'parse_grounding_tokens',
    'detect_chunk_type',
    'is_source_citation',
    'is_biographical_text',
    'extract_person_from_individual_entry',
    'detect_info_box_boundary',
]
