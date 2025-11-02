"""OCR text parsing and token classification for genealogical documents"""

import re
from typing import List, Optional

from .models import BoundingBox, ChunkType, GroundingToken


# Regex patterns for token parsing and classification

# Grounding token pattern - captures DeepSeek-OCR grounding tokens
# Format: <|ref|>type<|/ref|><|det|>[[coords]]<|/det|><|inverted|>true<|/inverted|>\ncontent
GROUNDING_TOKEN_PATTERN = re.compile(
    r'<\|ref\|>([^<]+)<\|/ref\|><\|det\|>\[\[([0-9,\s]+)\]\]<\|/det\|>(<\|inverted\|>true<\|/inverted\|>)?\s*\n(.*?)(?=\n*<\|ref\||$)',
    re.DOTALL
)

# Generation pattern - matches generation headers
# Don't rely on markdown hashes since those vary, just match the generation words
GENERATION_PATTERN = re.compile(
    r'(Eerste|Tweede|Derde|Vierde|ierde|Vijfde|Zesde|Zevende|Achtste|Negende|Tiende|Elfde|Twaalfde)\s+[Gg]eneratie',
    re.IGNORECASE
)

# Family group pattern - matches various formats including OCR corruptions
# Standard: "II.1. Kinderen van X en Y" or "II.1. Children of X and Y"
# With reference: "II.1. Kinderen van X en Y (VI.1.c):"
# Also matches without period: "X.14 Children of..." (OCR variation)
FAMILY_GROUP_PATTERN = re.compile(
    r'([IVX]+\.\d+[a-z]?)\.?\s+(?:Kinderen?|Kind|Children?|Child)\s+(?:van|of)\s+(.+?)(?:\s+(?:en|and)\s+(.+?))?(?:\s*\([^)]+\))?:?\s*$',
    re.IGNORECASE
)

# Remarriage pattern - same parent with different spouse, NO roman numeral prefix
# Example: "Kinderen van Hendrik Willem van Zanten en Hendrika Vuijst (V.1.c):"
REMARRIAGE_FAMILY_GROUP_PATTERN = re.compile(
    r'(?:Kinderen?|Kind|Children?|Child)\s+(?:van|of)\s+(.+?)(?:\s+(?:en|and)\s+(.+?))?(?:\s*\([^)]+\))?:?\s*$',
    re.IGNORECASE
)

# Match individual entries like "a. Name" or "l. Name"
# Also match "I." (uppercase I) which is a common OCR error for "l." (lowercase L)
# Allow optional parenthetical notes like "a. (hypothetisch) Name" or "a. (Misschien) Name"
INDIVIDUAL_ENTRY_PATTERN = re.compile(r'^([a-zI])\.\s+(?:\([^)]+\)\s+)?([A-Z])', re.MULTILINE)

# Source citation keywords
SOURCE_KEYWORDS = ['RGV', 'SSANO', 'DTB', 'BS', 'Bevolkingsregister', 'Weesregister',
                  'Archiefnummer', 'Bronverwijzing', 'Source indication']


def parse_grounding_tokens(ocr_text: str) -> List[GroundingToken]:
    """
    Extract grounding tokens from OCR text.

    Args:
        ocr_text: DeepSeek-OCR formatted text with grounding tokens

    Returns:
        List of parsed grounding tokens in document order
    """
    tokens = []

    for match in GROUNDING_TOKEN_PATTERN.finditer(ocr_text):
        element_type = match.group(1)
        bbox_str = match.group(2)
        inverted_tag = match.group(3)  # <|inverted|>true<|/inverted|> or None
        content = match.group(4).strip()

        # Parse bounding box
        try:
            coords = [int(x.strip()) for x in bbox_str.split(',')]
        except ValueError:
            import pdb
            pdb.set_trace()
        bbox = BoundingBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

        # Check if inverted tag was captured
        is_inverted = inverted_tag is not None

        tokens.append(GroundingToken(
            element_type=element_type,
            bbox=bbox,
            content=content,
            raw_match=match.group(0),
            is_inverted=is_inverted
        ))

    # Don't sort - preserve document order!
    # When multiple pages are concatenated, bbox coordinates are page-relative
    # and sorting would mix up content from different pages
    return tokens


def detect_chunk_type(token: GroundingToken) -> ChunkType:
    """
    Determine the type of content chunk from a token.

    Args:
        token: Grounding token to classify

    Returns:
        ChunkType enum value
    """
    content = token.content
    element_type = token.element_type

    # Image types
    if element_type == 'image':
        return ChunkType.IMAGE
    if element_type == 'image_caption':
        return ChunkType.IMAGE_CAPTION
    if element_type == 'table':
        return ChunkType.TABLE

    # Headers
    if element_type == 'sub_title':
        # Check if it's inverted - inverted regions are ALWAYS info boxes
        if token.is_inverted:
            return ChunkType.INFO_BOX

        # Strip markdown hashes for pattern matching (DeepSeek-OCR includes them)
        content_stripped = content.lstrip('#').strip()

        # Check if it's an individual entry (DeepSeek-OCR sometimes labels these as sub_title)
        # Individual entries like "a. Name" or "b. (hypothetisch) Name"
        if INDIVIDUAL_ENTRY_PATTERN.match(content_stripped):
            return ChunkType.INDIVIDUAL_ENTRY

        # Check if it's a generation header
        if GENERATION_PATTERN.search(content_stripped):
            return ChunkType.GENERATION_HEADER

        # Check if it's a family group header (with roman numeral)
        if FAMILY_GROUP_PATTERN.match(content_stripped):
            return ChunkType.FAMILY_GROUP_HEADER

        # Check if it's a remarriage family group header (without roman numeral)
        if REMARRIAGE_FAMILY_GROUP_PATTERN.match(content_stripped):
            return ChunkType.FAMILY_GROUP_HEADER  # Same chunk type, different pattern

        # Check if it's a source citation header
        content_lower = content_stripped.lower()
        if 'bronverwijzing' in content_lower or 'source indication' in content_lower or 'literatuur' in content_lower:
            return ChunkType.SOURCE_CITATION

        # Otherwise it's an info box header
        return ChunkType.INFO_BOX

    # Text content
    if element_type == 'text':
        # Check if it's a source citation
        if is_source_citation(content):
            return ChunkType.SOURCE_CITATION

        # Check if it's a family group header (can be labeled as 'text' instead of 'sub_title')
        if FAMILY_GROUP_PATTERN.match(content):
            return ChunkType.FAMILY_GROUP_HEADER

        # Check if it's a remarriage family group header
        if REMARRIAGE_FAMILY_GROUP_PATTERN.match(content):
            return ChunkType.FAMILY_GROUP_HEADER

        # Check if it's an individual entry (starts with letter + period)
        if INDIVIDUAL_ENTRY_PATTERN.match(content):
            return ChunkType.INDIVIDUAL_ENTRY

        # Check if it's narrative vs biographical
        if is_biographical_text(content):
            return ChunkType.BIOGRAPHICAL_TEXT
        else:
            return ChunkType.NARRATIVE_CONTEXT

    return ChunkType.UNKNOWN


def is_source_citation(content: str) -> bool:
    """
    Detect if content is a source citation.

    Args:
        content: Text content to check

    Returns:
        True if content appears to be a source citation
    """
    # Archive codes at start
    if re.match(r'^[A-Z]{2,}\s+[A-Z]{2,}', content):
        return True

    # Contains source keywords
    if any(keyword in content for keyword in SOURCE_KEYWORDS):
        return True

    return False


def is_biographical_text(content: str) -> bool:
    """
    Detect if text is dense biographical facts vs narrative.

    Args:
        content: Text content to check

    Returns:
        True if content appears to be biographical facts
    """
    # Contains genealogical markers
    # Use word boundaries to avoid false positives like "advies" containing "dv"

    # Birth/death symbols - but only if followed by typical genealogical context
    # Avoid matching (* YYYY) in narrative like "Pieter van Zanten (* 1944) helped..."
    # Look for *, ~ or † followed by space and location/date patterns
    if re.search(r'[*~†]\s+\d{1,2}[\./-]', content):  # * 10.1.1850 or † 12-1-1920
        return True
    if re.search(r'[*~†]\s+[A-Z][a-z]+', content):  # * Amsterdam or † Naarden
        return True

    # Marriage markers: x 1., x 2.
    if re.search(r'\bx\s+\d+\.', content):
        return True

    # Genealogical abbreviations (son/daughter of, widow)
    if re.search(r'\b(zv|dv|wed|weduwe)\b', content, re.IGNORECASE):
        return True

    # Residence pattern: year followed by colon and place name
    # Must be at start of content or after whitespace, and followed by capital letter (place name)
    # Example: "1850: Naarden" not "In 2007: 5461"
    if re.search(r'(^|\s)\d{4}:\s+[A-Z][a-z]+', content):
        return True

    # Address pattern with street number: "Amsterdam 123,"
    if re.search(r'[A-Z][a-z]+\s+\d{1,3},', content):
        return True

    return False


def extract_person_from_individual_entry(content: str) -> Optional[str]:
    """
    Extract person name from individual entry content.

    Example input: "a. Thomas [Tom] van Zanten, */~ Naarden 10/22.6.1862, † Holland Township..."
    Returns: "Thomas van Zanten"

    Args:
        content: Individual entry text

    Returns:
        Person name or None if not found
    """
    # Match individual entry pattern: "a. FirstName [Nickname] LastName, ..."
    # Also match "I." (uppercase I) which is a common OCR error for "l." (lowercase L)
    match = re.match(r'^[a-zI]\.\s+([A-Z][^,*†x~]+?)(?:\s*\[[^\]]+\])?\s*[,*†x~]', content)
    if match:
        name = match.group(1).strip()
        # Remove bracketed nicknames if they slipped through
        name = re.sub(r'\s*\[[^\]]+\]\s*', ' ', name).strip()
        return name
    return None


def detect_info_box_boundary(tokens: List[GroundingToken], start_index: int) -> Optional[int]:
    """
    Detect if we're entering an info box section and find where it ends.

    Info boxes are detected by:
    - Inverted text (white on black background)
    - Horizontal shift (x-coordinate significantly different from main flow)

    Args:
        tokens: List of all tokens
        start_index: Index where potential info box starts

    Returns:
        End index (exclusive) of info box, or None if not an info box
    """
    if start_index >= len(tokens):
        return None

    start_token = tokens[start_index]

    # Inverted text is always an info box
    if start_token.is_inverted:
        # Find the end of the inverted region
        end_idx = start_index + 1
        while end_idx < len(tokens) and tokens[end_idx].is_inverted:
            end_idx += 1
        return end_idx

    # Not an info box
    return None
