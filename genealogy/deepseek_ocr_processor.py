#!/usr/bin/env python3
"""
DeepSeek-OCR Processor

End-to-end OCR processor using DeepSeek-OCR in gundam mode.
Processes full pages without region detection for maximum accuracy.
"""

import logging
import re

import cv2
import numpy as np
from PIL import Image

from genealogy.deepseek_ocr_client import DeepSeekOCRClient

logger = logging.getLogger(__name__)


class DeepSeekOCRProcessor:
    """
    OCR processor using DeepSeek-OCR in gundam mode for maximum accuracy.

    Processes full pages end-to-end without region detection.
    """

    def __init__(
        self,
        deepseek_host: str = "192.168.1.234",
        deepseek_port: int = 5555,
        mode: str = "gundam",
    ) -> None:
        """
        Initialize the DeepSeek OCR processor.

        Args:
            deepseek_host: DeepSeek-OCR server hostname or IP
            deepseek_port: DeepSeek-OCR server port
            mode: Resolution mode ('tiny', 'small', 'base', 'large', 'gundam')
        """
        self.deepseek_host = deepseek_host
        self.deepseek_port = deepseek_port
        self.mode = mode

        # Create client connection
        self.client = DeepSeekOCRClient(host=deepseek_host, port=deepseek_port)

        logger.info(
            f"DeepSeekOCRProcessor initialized - host: {deepseek_host}:{deepseek_port}, "
            f"mode: {mode}"
        )

    def is_inverted_region(self, image: Image.Image, bbox: tuple) -> bool:
        """
        Detect if a region contains inverted text (white text on black/dark background).

        Uses histogram-based contrast polarity detection to identify bimodal distribution
        where the background peak is dark (inverted) vs light (normal).

        Args:
            image: Full page image
            bbox: Bounding box as (x1, y1, x2, y2) in normalized 1000-bin coordinates

        Returns:
            True if the region appears to be inverted (dark background, light text)
        """
        x1, y1, x2, y2 = bbox

        # Convert from normalized 1000-bin coordinates to actual pixel coordinates
        # According to DeepSeek-OCR paper: "All coordinates are normalized into 1000 bins"
        width, height = image.size
        x1 = int((x1 / 1000.0) * width)
        y1 = int((y1 / 1000.0) * height)
        x2 = int((x2 / 1000.0) * width)
        y2 = int((y2 / 1000.0) * height)

        # Ensure valid coordinates
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        # Crop the region
        region = image.crop((x1, y1, x2, y2))

        # Convert to grayscale numpy array for OpenCV
        if region.mode != 'L':
            region = region.convert('L')
        gray = np.array(region)

        # Apply slight blur to smooth noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        # Find the two largest peaks
        # We'll look for peaks in the histogram by smoothing and finding local maxima
        # Split into dark half (0-127) and bright half (128-255)
        dark_half_mass = hist[:128].sum()
        bright_half_mass = hist[128:].sum()

        # Find which half has more pixels (should be background)
        # Also compute weighted mean to see where the bulk of pixels lie
        total_pixels = dark_half_mass + bright_half_mass
        if total_pixels == 0:
            return False

        dark_ratio = dark_half_mass / total_pixels

        # If most pixels are in the dark half, likely inverted (dark background)
        # Use a threshold of 60% to be robust
        is_inverted = dark_ratio > 0.6

        # Debug output
        print(f"Region at {bbox}: dark_ratio={dark_ratio:.2f}, inverted={is_inverted}")

        return is_inverted

    def annotate_inverted_regions(self, ocr_text: str, image: Image.Image) -> str:
        """
        Parse OCR text with grounding tokens and add inversion annotations.

        Args:
            ocr_text: OCR text with grounding tokens
            image: Original page image

        Returns:
            OCR text with <|inverted|>true<|/inverted|> annotations added
        """
        # Pattern to match grounding token blocks
        # Format: <|ref|>type<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>\nContent
        # Capture element type to skip images
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|/det\|>)(\n)'

        # Count matches for debugging
        matches = list(re.finditer(pattern, ocr_text))
        print(f"DEBUG: Found {len(matches)} grounding token matches to check for inversion")

        def add_inversion_tag(match):
            full_match = match.group(1)
            element_type = match.group(2)
            bbox_str = match.group(3)
            newline = match.group(4)

            # Skip images - inverted flag only applies to text content
            if element_type in ['image', 'image_caption']:
                return full_match + newline

            # Parse bounding box coordinates
            try:
                coords = [int(x.strip()) for x in bbox_str.split(',')]
                if len(coords) == 4:
                    bbox = tuple(coords)

                    # Check if region is inverted (text/sub_title only)
                    if self.is_inverted_region(image, bbox):
                        # Add inverted annotation after the grounding tokens
                        return f"{full_match}<|inverted|>true<|/inverted|>{newline}"
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse bounding box '{bbox_str}': {e}")

            # Return original match if not inverted or failed to parse
            return full_match + newline

        # Add inversion annotations
        annotated_text = re.sub(pattern, add_inversion_tag, ocr_text)

        return annotated_text

    def process_page(self, image: Image.Image) -> str:
        """
        Process full page with DeepSeek-OCR end-to-end.

        Args:
            image: Full page image

        Returns:
            Extracted text with grounding tokens, markdown formatting, and inversion annotations
        """
        logger.info(f"Processing full page with DeepSeek-OCR (mode: {self.mode})...")

        # Process full page with DeepSeek-OCR
        result = self.client.process_image(
            image,
            mode=self.mode,
            preserve_layout=True
        )

        # Get OCR text with grounding tokens
        text = result['text'].strip()

        # Annotate inverted regions
        text = self.annotate_inverted_regions(text, image)

        logger.info(
            f"DeepSeek-OCR processing complete - {len(text)} characters extracted"
        )
        return text

    def close(self):
        """Close connection to DeepSeek-OCR server"""
        if self.client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
