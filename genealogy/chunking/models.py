"""Data models for genealogical text chunking"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class ChunkType(Enum):
    """Types of content chunks"""
    GENERATION_HEADER = "generation_header"
    FAMILY_GROUP_HEADER = "family_group_header"
    INDIVIDUAL_ENTRY = "individual_entry"
    BIOGRAPHICAL_TEXT = "biographical_text"
    SOURCE_CITATION = "source_citation"
    NARRATIVE_CONTEXT = "narrative_context"
    INFO_BOX = "info_box"
    IMAGE = "image"
    IMAGE_CAPTION = "image_caption"
    TABLE = "table"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box coordinates from grounding tokens"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class GroundingToken:
    """Parsed grounding token from DeepSeek-OCR"""
    element_type: str  # 'text', 'sub_title', 'image', 'image_caption', 'table'
    bbox: BoundingBox
    content: str
    raw_match: str  # Original grounding token string
    is_inverted: bool = False  # True if region has inverted colors (white text on black background)


@dataclass
class TextChunk:
    """Hierarchical text chunk with genealogical context"""
    chunk_type: ChunkType
    content: str
    grounding_tokens: List[GroundingToken] = field(default_factory=list)

    # Genealogical context (inherited from parent chunks)
    generation: Optional[str] = None  # e.g., "Tweede generatie"
    family_group: Optional[str] = None  # e.g., "II.2 Kinderen van X en Y"
    family_group_id: Optional[str] = None  # e.g., "II.2"
    parents: Optional[Tuple[str, str]] = None  # (father, mother) names

    # Metadata
    individual_marker: Optional[str] = None  # e.g., "a.", "b.", "n."
    is_info_box: bool = False

    # For source citations - link to the fact they support
    supports_chunk_index: Optional[int] = None

    # Extracted entities (populated during chunking for DESCENDANT_GENEALOGY)
    extracted_people: List[str] = field(default_factory=list)  # Person names mentioned
    extracted_relationships: List[dict] = field(default_factory=list)  # Relationship triples
    extracted_events: List[dict] = field(default_factory=list)  # Events (Phase 2, populated by LLM)

    def __repr__(self):
        return f"TextChunk(type={self.chunk_type.value}, generation={self.generation}, family={self.family_group_id}, marker={self.individual_marker})"
