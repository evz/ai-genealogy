"""Chunking strategy for descendant genealogy sections"""
import logging
from collections import Counter
from typing import List

from ..chunking.handlers import CHUNK_HANDLERS
from ..chunking.models import ChunkType, TextChunk
from ..chunking.parser import detect_chunk_type, parse_grounding_tokens
from .base import ChunkingStrategy

logger = logging.getLogger(__name__)


class DescendantGenealogyChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy for descendant genealogy sections.

    Handles:
    - Generation headers
    - Family group headers
    - Individual entries with biographical context
    - Source citations
    - Info boxes
    """

    def __init__(self):
        # Genealogical context tracking
        self.current_generation = None
        self.current_family_group = None
        self.current_family_group_id = None
        self.current_parents = None

    @property
    def strategy_name(self) -> str:
        return "Descendant Genealogy Chunking"

    def chunk_section(self, section_text: str, document, page_map: List[dict]) -> List:
        """
        Chunk descendant genealogy section.

        Uses a two-pass approach:
        1. Extract images and info boxes (simple, out-of-flow content)
        2. Chunk main flow (complex genealogical processing)

        Args:
            section_text: OCR text from genealogy pages, concatenated
            document: Document model instance
            page_map: List of dicts mapping character positions to page numbers

        Returns:
            List of TextChunk objects
        """
        logger.info(f"Chunking descendant genealogy section with {len(section_text)} characters")

        # Parse all grounding tokens
        all_tokens = list(parse_grounding_tokens(section_text))
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
        main_flow_chunks = self._chunk_main_flow(main_flow_tokens, document)

        # Adjust supports_chunk_index for main flow chunks
        offset = len(chunks)  # Number of chunks already added (images + info boxes)
        for chunk in main_flow_chunks:
            if chunk.supports_chunk_index is not None:
                chunk.supports_chunk_index += offset

        chunks.extend(main_flow_chunks)

        logger.info(f"Pass 2: Created {len(main_flow_chunks)} main flow chunks")
        logger.info(f"Total chunks: {len(chunks)}")

        return chunks

    def _extract_out_of_flow_content(self, tokens):
        """Extract images and info boxes from token stream"""
        images = []
        inverted_text_tokens = []
        main_flow_tokens = []

        # Estimate main flow x1 baseline
        main_flow_x1_values = []
        for token in tokens:
            if token.element_type == 'text' and not token.is_inverted:
                main_flow_x1_values.append(token.bbox.x1)

        if main_flow_x1_values:
            x1_counter = Counter(main_flow_x1_values)
            baseline_x1, _ = x1_counter.most_common(1)[0]
        else:
            baseline_x1 = None

        # Track indices to skip (text tokens that are part of image captions)
        skip_indices = set()

        # First pass: identify text tokens that are part of image captions
        for i, token in enumerate(tokens):
            if token.element_type == 'image_caption':
                j = i + 1
                last_y2 = token.bbox.y2
                while j < len(tokens):
                    next_token = tokens[j]
                    if next_token.element_type != 'text':
                        break

                    y_gap = abs(next_token.bbox.y1 - last_y2)
                    if y_gap > 50:
                        break

                    if baseline_x1 is not None:
                        x1_diff = abs(next_token.bbox.x1 - baseline_x1)
                        if x1_diff <= 25:  # Fixed: use <= and slightly larger threshold
                            break

                    skip_indices.add(j)
                    last_y2 = next_token.bbox.y2
                    j += 1

        # Second pass: extract tokens
        for i, token in enumerate(tokens):
            if i in skip_indices:
                images.append(token)
            elif token.element_type in ['image', 'image_caption']:
                images.append(token)
            elif token.is_inverted and token.element_type in ['text', 'sub_title']:
                inverted_text_tokens.append(token)
            else:
                main_flow_tokens.append(token)

        # Group consecutive inverted tokens into info box groups
        info_box_groups = []
        if inverted_text_tokens:
            current_group = [inverted_text_tokens[0]]

            for i in range(1, len(inverted_text_tokens)):
                prev_idx = tokens.index(inverted_text_tokens[i-1])
                curr_idx = tokens.index(inverted_text_tokens[i])

                if curr_idx == prev_idx + 1:
                    current_group.append(inverted_text_tokens[i])
                else:
                    info_box_groups.append(current_group)
                    current_group = [inverted_text_tokens[i]]

            info_box_groups.append(current_group)

        return images, info_box_groups, main_flow_tokens

    def _split_multi_sibling_tokens(self, tokens):
        """
        Pre-process tokens to split any that contain multiple sibling entries.

        Some OCR tokens contain multiple individual entries (a., b., c.) in a single token.
        This splits them into separate tokens to ensure each sibling gets their own chunk.
        """
        import re
        from ..chunking.parser import INDIVIDUAL_ENTRY_PATTERN
        from ..chunking.models import GroundingToken

        split_tokens = []

        for token in tokens:
            # Check if token contains multiple lines starting with individual entry markers
            lines = token.content.split('\n')
            entry_line_indices = []

            for i, line in enumerate(lines):
                if INDIVIDUAL_ENTRY_PATTERN.match(line.strip()):
                    entry_line_indices.append(i)

            # If we found multiple entry markers, split the token
            if len(entry_line_indices) > 1:
                # Split into separate tokens
                for idx_pos, line_idx in enumerate(entry_line_indices):
                    # Determine the end line for this entry
                    if idx_pos < len(entry_line_indices) - 1:
                        end_line_idx = entry_line_indices[idx_pos + 1]
                    else:
                        end_line_idx = len(lines)

                    # Extract lines for this entry
                    entry_lines = lines[line_idx:end_line_idx]
                    entry_content = '\n'.join(entry_lines).strip()

                    # Create a new token with the same metadata but split content
                    split_token = GroundingToken(
                        element_type=token.element_type,
                        bbox=token.bbox,  # Keep same bbox (approximate)
                        content=entry_content,
                        raw_match=token.raw_match,  # Keep original raw_match
                        is_inverted=token.is_inverted,
                    )
                    split_tokens.append(split_token)
            else:
                # No splitting needed
                split_tokens.append(token)

        return split_tokens

    def _chunk_main_flow(self, tokens, document):
        """Chunk main flow tokens using handler pattern"""
        # Pre-process: split tokens that contain multiple sibling entries
        tokens = self._split_multi_sibling_tokens(tokens)

        chunks = []
        i = 0

        context = {
            'generation': self.current_generation,
            'family_group': self.current_family_group,
            'family_group_id': self.current_family_group_id,
            'parents': self.current_parents,
            'document': document,
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
                logger.warning(f"No handler found for chunk type {chunk_type} at index {i}")
                i += 1

        # Update instance variables from context
        self.current_generation = context['generation']
        self.current_family_group = context['family_group']
        self.current_family_group_id = context['family_group_id']
        self.current_parents = context['parents']

        return chunks
