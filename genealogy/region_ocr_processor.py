#!/usr/bin/env python3
"""
Region-based OCR processing for genealogy documents.

Handles OCR on detected document regions with smart text ordering,
deduplication, and genealogy-specific optimizations.
"""

import logging
from typing import Any

import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class RegionOCRProcessor:
    """
    OCR processor for document regions with genealogy-specific optimizations.

    Handles text extraction from detected regions, smart ordering (main content
    vs inset content), and deduplication to produce clean final text.
    """

    def __init__(
        self,
        tesseract_language: str = "eng+nld",
        tesseract_config: str = "--oem 2 --psm 6",
        page_width_threshold: float = 0.2,
    ) -> None:
        """
        Initialize the region OCR processor.

        Args:
            tesseract_language: Language codes for Tesseract (e.g., "eng+nld")
            tesseract_config: Tesseract configuration string
            page_width_threshold: Threshold for separating main vs inset content (0.0-1.0)
        """
        self.tesseract_language = tesseract_language
        self.tesseract_config = tesseract_config
        self.page_width_threshold = page_width_threshold

        logger.info(
            f"RegionOCRProcessor initialized - lang: {tesseract_language}, "
            f"config: {tesseract_config}, threshold: {page_width_threshold}"
        )

    def process_regions(self, image: Image.Image, regions: list[dict[str, Any]]) -> tuple[str, float]:
        """
        Process all regions with OCR and return final stitched text with confidence.

        Args:
            image: Original document image
            regions: List of detected regions with bounding boxes

        Returns:
            Tuple of (final_stitched_text, overall_confidence_score)
        """
        logger.info(f"Processing {len(regions)} regions with OCR...")

        if not regions:
            logger.warning("No regions to process")
            return "", 0.0

        # Step 1: Run OCR on individual regions
        ocr_results = self._extract_text_from_regions(image, regions)

        # Step 2: Apply smart text ordering and deduplication
        final_text = self._stitch_text_results(ocr_results)

        # Step 3: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(ocr_results)

        logger.info(
            f"OCR processing complete - {len(final_text)} characters extracted, "
            f"confidence: {overall_confidence:.1f}%"
        )
        return final_text, overall_confidence

    def _extract_text_from_regions(self, image: Image.Image, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Extract text from individual regions using OCR.

        Args:
            image: Original document image
            regions: List of detected regions

        Returns:
            List of OCR results with text and metadata
        """
        logger.debug(f"Running OCR on {len(regions)} regions...")

        ocr_results = []

        for idx, region in enumerate(regions):
            bbox = region["bbox"]
            element_type = region["element"]
            confidence = region["confidence"]

            try:
                # Extract region from image
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                region_image = image.crop((x1, y1, x2, y2))

                # Run OCR
                text = pytesseract.image_to_string(
                    region_image, lang=self.tesseract_language, config=self.tesseract_config
                ).strip()

                # Get confidence scores
                try:
                    tsv_data = pytesseract.image_to_data(
                        region_image,
                        lang=self.tesseract_language,
                        config=self.tesseract_config,
                        output_type=pytesseract.Output.DICT,
                    )
                    ocr_confidences = [int(conf) for conf in tsv_data["conf"] if int(conf) > 0]
                    avg_confidence = np.mean(ocr_confidences) if ocr_confidences else 0
                except Exception:
                    avg_confidence = 0

                ocr_results.append(
                    {
                        "region_idx": idx,
                        "element_type": element_type,
                        "detection_confidence": confidence,
                        "bbox": bbox,
                        "text": text,
                        "ocr_confidence": avg_confidence,
                        "character_count": len(text),
                    }
                )

                logger.debug(f"Region {idx} ({element_type}): {len(text)} characters extracted")

            except Exception as e:
                logger.warning(f"OCR failed for region {idx}: {e}")
                ocr_results.append(
                    {
                        "region_idx": idx,
                        "element_type": element_type,
                        "detection_confidence": confidence,
                        "bbox": bbox,
                        "error": str(e),
                        "character_count": 0,
                    }
                )

        return ocr_results

    def _deduplicate_text_lines(self, text_lines: list[str]) -> list[str]:
        """
        Remove duplicate and near-duplicate text lines from OCR results.

        Args:
            text_lines: List of text lines to deduplicate

        Returns:
            Deduplicated list of text lines
        """
        if not text_lines:
            return text_lines

        # Remove exact duplicates while preserving order
        seen = set()
        unique_lines = []

        for line in text_lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                unique_lines.append(line)

        # Remove near-duplicates (lines that are substrings of others)
        filtered_lines = []

        for i, line1 in enumerate(unique_lines):
            is_substring = False
            line1_clean = line1.strip()

            for j, line2 in enumerate(unique_lines):
                if i != j:
                    line2_clean = line2.strip()
                    # If line1 is a substring of line2 and significantly shorter
                    if line1_clean in line2_clean and len(line1_clean) < len(line2_clean) * 0.8:
                        is_substring = True
                        break

            if not is_substring:
                filtered_lines.append(line1)

        logger.debug(f"Text deduplication: {len(text_lines)} -> {len(filtered_lines)} lines")
        return filtered_lines

    def _stitch_text_results(self, ocr_results: list[dict[str, Any]]) -> str:
        """
        Stitch OCR results together with smart ordering and deduplication.

        Args:
            ocr_results: List of OCR result dictionaries

        Returns:
            Final stitched text
        """
        if not ocr_results:
            return ""

        # Filter to only successful OCR results
        valid_results = [r for r in ocr_results if r.get("text")]

        if not valid_results:
            logger.warning("No valid OCR results to stitch")
            return ""

        # Calculate page width threshold for main vs inset content
        max_x = max(result["bbox"][2] for result in valid_results)
        page_width_threshold = max_x * self.page_width_threshold

        # Separate main content from inset content
        main_content = []
        inset_content = []

        for result in valid_results:
            bbox = result["bbox"]
            x_start = bbox[0]

            if x_start <= page_width_threshold:
                main_content.append(result)
            else:
                inset_content.append(result)

        logger.debug(f"Split content: {len(main_content)} main regions, {len(inset_content)} inset regions")

        # Sort by reading order (top-to-bottom, left-to-right)
        def reading_order_key(result: dict[str, Any]) -> tuple[float, float]:
            bbox = result["bbox"]
            return (bbox[1], bbox[0])  # (y1, x1)

        main_sorted = sorted(main_content, key=reading_order_key)
        inset_sorted = sorted(inset_content, key=reading_order_key)

        # Combine: main content first, then inset content
        sorted_results = main_sorted + inset_sorted

        # Extract all text lines
        all_text_lines = []
        for result in sorted_results:
            text = result["text"].strip()
            if text:
                # Split into lines and add each line
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                all_text_lines.extend(lines)

        # Deduplicate text lines
        deduplicated_lines = self._deduplicate_text_lines(all_text_lines)

        # Join into final text
        final_text = "\n".join(deduplicated_lines)

        logger.debug(f"Text stitching complete: {len(deduplicated_lines)} lines, {len(final_text)} characters")
        return final_text

    def _calculate_overall_confidence(self, ocr_results: list[dict[str, Any]]) -> float:
        """
        Calculate overall confidence score from individual region OCR results.

        Args:
            ocr_results: List of OCR result dictionaries

        Returns:
            Overall confidence score (0-100)
        """
        if not ocr_results:
            return 0.0

        # Get valid results with OCR confidence scores
        valid_results = [
            r for r in ocr_results if "ocr_confidence" in r and "character_count" in r and r["character_count"] > 0
        ]

        if not valid_results:
            # If no confidence data, use detection confidence as fallback
            detection_confidences = [
                r["detection_confidence"] * 100 for r in ocr_results if "detection_confidence" in r
            ]
            if detection_confidences:
                return sum(detection_confidences) / len(detection_confidences)
            return 50.0  # Default fallback

        # Weight by character count - regions with more text have more influence
        total_weighted_confidence = 0.0
        total_characters = 0

        for result in valid_results:
            ocr_conf = result["ocr_confidence"]
            char_count = result["character_count"]

            total_weighted_confidence += ocr_conf * char_count
            total_characters += char_count

        if total_characters == 0:
            return 50.0

        overall_confidence = total_weighted_confidence / total_characters

        # Clamp to reasonable range
        overall_confidence = max(0.0, min(100.0, overall_confidence))

        logger.debug(
            f"Calculated overall confidence: {overall_confidence:.1f}% "
            f"from {len(valid_results)} regions ({total_characters} chars)"
        )

        return overall_confidence
