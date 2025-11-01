"""Genealogical text chunker with hierarchical context

This module chunks OCR text from genealogical documents into semantic units
with inherited genealogical context (generation, family group, parents).
"""

import logging
from typing import List, Tuple

from .handlers import CHUNK_HANDLERS
from .models import ChunkType, GroundingToken, TextChunk
from .parser import detect_chunk_type, detect_info_box_boundary, parse_grounding_tokens

logger = logging.getLogger(__name__)


class GenealogicalTextChunker:
    """
    Chunks genealogical text with hierarchical context tracking.

    Uses DeepSeek-OCR grounding tokens to preserve spatial structure and
    maintain genealogical context (generation → family group → individual).
    """

    def __init__(self, document=None):
        """
        Initialize the chunker.

        Args:
            document: Optional Document model instance for looking up BookSections
        """
        self.document = document

        # Genealogical context tracking (updated as we parse)
        self.current_generation = None  # e.g., "Tweede generatie"
        self.current_family_group = None  # e.g., "II.2 Kinderen van X en Y"
        self.current_family_group_id = None  # e.g., "II.2"
        self.current_parents = None  # Tuple of (father, mother) names

    def chunk(self, ocr_text: str) -> List[TextChunk]:
        """
        Parse OCR text into hierarchical chunks with genealogical context.

        Uses a two-pass approach:
        1. Extract images and info boxes (simple, out-of-flow content)
        2. Chunk main flow (complex genealogical processing)

        Returns chunks in reading order with inherited context.
        """
        # Parse all grounding tokens
        all_tokens = list(parse_grounding_tokens(ocr_text))
        logger.info(f"Parsed {len(all_tokens)} grounding tokens")

        # PASS 1: Extract images and info boxes from the flow
        chunks = []
        images, info_boxes, main_flow_tokens = self._extract_out_of_flow_content(all_tokens)

        logger.info(f"Pass 1: Extracted {len(images)} images, {len(info_boxes)} info boxes")
        logger.info(f"Pass 1: {len(main_flow_tokens)} tokens remaining in main flow")

        # Create chunks for images (one chunk per image)
        for img_token in images:
            chunks.append(TextChunk(
                chunk_type=ChunkType.IMAGE if img_token.element_type == 'image' else ChunkType.IMAGE_CAPTION,
                content=img_token.content,
                grounding_tokens=[img_token],
            ))

        # Create chunks for info boxes (group consecutive inverted tokens)
        for info_box_group in info_boxes:
            info_box_content = '\n\n'.join(t.content for t in info_box_group)
            chunks.append(TextChunk(
                chunk_type=ChunkType.INFO_BOX,
                content=info_box_content,
                grounding_tokens=info_box_group,
                is_info_box=True,
            ))

        # PASS 2: Chunk the main flow with genealogical context
        main_flow_chunks = self._chunk_main_flow(main_flow_tokens)

        # Adjust supports_chunk_index for main flow chunks
        # They were indexed relative to main_flow_chunks, but need to be relative to final chunks list
        offset = len(chunks)  # Number of chunks already added (images + info boxes)
        for chunk in main_flow_chunks:
            if chunk.supports_chunk_index is not None:
                chunk.supports_chunk_index += offset

        chunks.extend(main_flow_chunks)

        logger.info(f"Pass 2: Created {len(main_flow_chunks)} main flow chunks")
        logger.info(f"Total chunks: {len(chunks)}")

        return chunks

    def _extract_out_of_flow_content(
        self,
        tokens: List[GroundingToken]
    ) -> Tuple[List[GroundingToken], List[List[GroundingToken]], List[GroundingToken]]:
        """
        Extract images and info boxes from token stream.

        Also extracts text tokens that are part of image captions (continuation text
        that follows image_caption tokens but isn't marked as such by OCR).

        Returns:
            (images, info_box_groups, main_flow_tokens)
        """
        images = []
        inverted_text_tokens = []
        main_flow_tokens = []

        # Estimate main flow x1 baseline from text tokens
        # (used to detect caption text that's indented differently)
        main_flow_x1_values = []
        for token in tokens:
            if token.element_type == 'text' and not token.is_inverted:
                main_flow_x1_values.append(token.bbox.x1)

        # Find the most common x1 value (mode) as the baseline
        if main_flow_x1_values:
            from collections import Counter
            x1_counter = Counter(main_flow_x1_values)
            baseline_x1, _ = x1_counter.most_common(1)[0]
        else:
            baseline_x1 = None

        # Track indices to skip (text tokens that are part of image captions)
        skip_indices = set()

        # First pass: identify text tokens that are part of image captions
        for i, token in enumerate(tokens):
            if token.element_type == 'image_caption':
                # Look ahead for text tokens that are spatially close and not main flow
                j = i + 1
                last_y2 = token.bbox.y2
                while j < len(tokens):
                    next_token = tokens[j]
                    if next_token.element_type != 'text':
                        break

                    # Check if vertically contiguous (y1 is close to previous y2)
                    y_gap = abs(next_token.bbox.y1 - last_y2)
                    if y_gap > 50:  # Not vertically contiguous
                        break

                    # Check if x1 differs from main flow baseline
                    # (caption text is usually indented/centered differently)
                    if baseline_x1 is not None:
                        x1_diff = abs(next_token.bbox.x1 - baseline_x1)
                        if x1_diff < 20:  # Too close to baseline - probably main flow
                            break

                    # This token is part of the caption
                    skip_indices.add(j)
                    last_y2 = next_token.bbox.y2
                    j += 1

        # Second pass: extract tokens based on type and skip list
        for i, token in enumerate(tokens):
            if i in skip_indices:
                # This text token is part of an image caption
                images.append(token)
            elif token.element_type in ['image', 'image_caption']:
                images.append(token)
            # Extract inverted text (info boxes)
            elif token.is_inverted and token.element_type in ['text', 'sub_title']:
                inverted_text_tokens.append(token)
            # Everything else goes to main flow
            else:
                main_flow_tokens.append(token)

        # Group consecutive inverted tokens into info box groups
        info_box_groups = []
        if inverted_text_tokens:
            current_group = [inverted_text_tokens[0]]

            for i in range(1, len(inverted_text_tokens)):
                # Check if this token is consecutive with the previous one
                prev_idx = tokens.index(inverted_text_tokens[i-1])
                curr_idx = tokens.index(inverted_text_tokens[i])

                if curr_idx == prev_idx + 1:
                    # Consecutive - add to current group
                    current_group.append(inverted_text_tokens[i])
                else:
                    # Gap - start new group
                    info_box_groups.append(current_group)
                    current_group = [inverted_text_tokens[i]]

            # Don't forget the last group
            info_box_groups.append(current_group)

        return images, info_box_groups, main_flow_tokens

    def _chunk_main_flow(self, tokens: List[GroundingToken]) -> List[TextChunk]:
        """
        Chunk the main flow tokens with genealogical context.

        Uses the handler pattern to process different chunk types.
        This is much cleaner than the original 229-line method!
        """
        chunks = []
        i = 0

        # Context dictionary that handlers will update
        context = {
            'generation': self.current_generation,
            'family_group': self.current_family_group,
            'family_group_id': self.current_family_group_id,
            'parents': self.current_parents,
        }

        while i < len(tokens):
            token = tokens[i]
            chunk_type = detect_chunk_type(token)

            # Find the appropriate handler for this chunk type
            for handler in CHUNK_HANDLERS:
                if handler.can_handle(chunk_type, token):
                    chunk, new_index = handler.create_chunk(
                        chunk_type=chunk_type,
                        token=token,
                        tokens=tokens,
                        index=i,
                        context=context,
                        chunks=chunks,
                    )
                    chunks.append(chunk)
                    i = new_index
                    break
            else:
                # Should never happen (DefaultChunkHandler always matches)
                logger.warning(f"No handler found for chunk type {chunk_type} at index {i}")
                i += 1

        # Update instance variables from context
        self.current_generation = context['generation']
        self.current_family_group = context['family_group']
        self.current_family_group_id = context['family_group_id']
        self.current_parents = context['parents']

        return chunks
