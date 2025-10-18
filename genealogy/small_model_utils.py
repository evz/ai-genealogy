"""Small model utilities for focused OCR post-processing and line classification"""

import logging
from typing import List

from .ollama_utils import OllamaClient
from .patterns import GenealogyPatterns

logger = logging.getLogger(__name__)


class SmallModelProcessor:
    """Processor for focused small model tasks: OCR cleanup and line classification"""

    def __init__(self, ocr_model: str = "llama3.2:3b", classification_model: str = "qwen2.5:7b"):
        self.client = OllamaClient()
        self.ocr_model = ocr_model
        self.classification_model = classification_model

    def clean_ocr_genealogy_ids(self, text_lines: List[str]) -> List[str]:
        """
        Clean genealogical ID OCR errors one line at a time.
        Only processes lines that might contain genealogical IDs.
        """
        if not self.client.is_available():
            logger.warning("Ollama not available, skipping OCR cleanup")
            return text_lines

        cleaned_lines = []
        for line in text_lines:
            if self._might_contain_genealogy_id(line):
                cleaned_line = self._clean_single_id_line(line)
                cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append(line)

        return cleaned_lines

    def classify_single_line(self, line: str) -> str:
        """
        Classify a single line quickly. Returns: GEN_HDR | FAMILY_GROUP | INDIVIDUAL | OTHER
        """
        if not self.client.is_available():
            return self._regex_classify_line(line)

        # Use small model for classification
        prompt = f"""Classify this line as one of: GEN_HDR | FAMILY_GROUP | INDIVIDUAL | OTHER

GEN_HDR: Generation headers like "EERSTE GENERATIE", "SECOND GENERATION"
FAMILY_GROUP: Family headers like "II.1. Children of John Smith"
INDIVIDUAL: Individual entries like "a. Maria van Bergen"
OTHER: Everything else

Return only the classification, nothing else.

Line: "{line}"
"""

        try:
            response = self.client.generate(
                model=self.classification_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_ctx": 256}
            )

            if response and response.strip().upper() in ["GEN_HDR", "FAMILY_GROUP", "INDIVIDUAL", "OTHER"]:
                return response.strip().upper()
            else:
                # Fallback to regex
                return self._regex_classify_line(line)

        except Exception as e:
            logger.warning(f"Small model classification failed for '{line}': {e}")
            return self._regex_classify_line(line)

    def _might_contain_genealogy_id(self, line: str) -> bool:
        """Check if line might contain a genealogical ID that needs cleaning"""
        # Use existing pattern to detect potential IDs (including OCR corrupted ones)
        return bool(GenealogyPatterns.GENEALOGY_ID.search(line))

    def _clean_single_id_line(self, line: str) -> str:
        """Clean genealogical ID OCR errors in a single line"""
        prompt = f"""Fix only OCR errors in genealogical IDs. Only fix these 4 patterns:
- IL.1.d → II.1.d
- XIL.4.b → XII.4.b
- VIL.2.a → VII.2.a
- V.2.F → V.2.f

Return the corrected line exactly, no commentary.

Line: "{line}"
"""

        try:
            response = self.client.generate(
                model=self.ocr_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_ctx": 256}
            )

            if response and response.strip():
                cleaned = response.strip()
                # Basic sanity check - don't accept drastically different results
                if 0.5 <= len(cleaned) / len(line) <= 2.0:
                    return cleaned

            return line

        except Exception as e:
            logger.warning(f"OCR cleanup failed for '{line}': {e}")
            return line

    def _regex_classify_line(self, line: str) -> str:
        """Fast regex classification using existing patterns"""
        # Generation header
        if GenealogyPatterns.GENERATION_HEADER.search(line):
            return "GEN_HDR"

        # Family group
        if GenealogyPatterns.FAMILY_GROUP.search(line):
            return "FAMILY_GROUP"

        # Individual entry (simple pattern)
        line_stripped = line.strip()
        if line_stripped and len(line_stripped) > 2:
            if line_stripped[0].islower() and line_stripped[1] in '.):' and line_stripped[2:3] == ' ':
                return "INDIVIDUAL"

        return "OTHER"
