import logging

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processing with automatic orientation detection and multi-language support"""

    def __init__(self, language: str = "eng+nld"):
        """
        Initialize OCR processor

        Args:
            language: Tesseract language string ('eng', 'nld', 'eng+nld')
        """
        self.language = language

        # Tesseract configuration with PSM 1 (automatic page segmentation with OSD)
        # PSM 1 automatically handles rotation detection and provides better results
        self.tesseract_config = "--oem 3 --psm 1"

    def process_file(self, file_path: str) -> tuple[str, float, float]:
        """
        Process a file (PDF or image) and extract text with OCR

        Args:
            file_path: Path to the file to process

        Returns:
            Tuple of (extracted_text, confidence_score, rotation_applied)
            Note: rotation_applied is always 0 since PSM 1 handles rotation internally
        """
        try:
            # Determine file type and load image
            image = self._pdf_to_image(file_path) if file_path.lower().endswith(".pdf") else Image.open(file_path)

            # Perform OCR with PSM 1 (automatic page segmentation +
            # orientation detection)
            # PSM 1 automatically handles rotation, no manual rotation detection needed
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.tesseract_config,
            )

            # Get confidence score
            confidence = self._get_confidence_score(image)

            # Return 0 for rotation_applied since PSM 1 handles it internally
            return text.strip(), confidence, 0.0

        except Exception as e:
            logger.exception(f"OCR processing failed for {file_path}: {e}")
            raise

    def _pdf_to_image(self, file_path: str) -> Image.Image:
        """Convert first page of PDF to image"""
        try:
            with open(file_path, "rb") as pdf_file:
                images = convert_from_bytes(pdf_file.read(), first_page=1, last_page=1)
                if not images:
                    raise ValueError("Could not extract image from PDF")
                return images[0]
        except Exception as e:
            logger.exception(f"PDF conversion failed for {file_path}: {e}")
            raise

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance grayscale image quality for better OCR results"""
        try:
            # Increase contrast slightly for grayscale images
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(1.2)

            # Increase sharpness slightly
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            return sharpness_enhancer.enhance(1.1)

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image

    def _get_confidence_score(self, image: Image.Image) -> float:
        """Get OCR confidence score from Tesseract"""
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )

            # Calculate average confidence of words with confidence > 0
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]

            if confidences:
                return sum(confidences) / len(confidences)
            return 0.0

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.0
