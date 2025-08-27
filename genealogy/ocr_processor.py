import logging

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance

from .rotation_detector import RotationDetector

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processing with rotation correction and multi-language support"""

    def __init__(self, language: str = "eng+nld"):
        """
        Initialize OCR processor

        Args:
            language: Tesseract language string ('eng', 'nld', 'eng+nld')
        """
        self.language = language
        self.rotation_detector = RotationDetector()

        # Tesseract configuration for better accuracy
        # Use simpler config without character whitelist to avoid shell quoting issues
        self.tesseract_config = "--oem 3 --psm 3"

    def process_file(self, file_path: str) -> tuple[str, float, float]:
        """
        Process a file (PDF or image) and extract text with OCR

        Args:
            file_path: Path to the file to process

        Returns:
            Tuple of (extracted_text, confidence_score, rotation_applied)
        """
        try:
            # Determine file type and load image
            image = self._pdf_to_image(file_path) if file_path.lower().endswith(".pdf") else Image.open(file_path)

            # Convert to grayscale for better OCR
            if image.mode != "L":
                image = image.convert("L")

            # Detect and correct rotation using new method
            rotation_applied, confidence = self.rotation_detector.detect_rotation(image)
            if abs(rotation_applied) > 0.5:  # Only rotate if angle is significant
                image = image.rotate(-rotation_applied, expand=True)
                logger.info(f"Applied rotation correction: {rotation_applied:.2f}° (confidence: {confidence:.2f})")

            # Enhance image for better OCR
            image = self._enhance_image(image)

            # Perform OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.tesseract_config,
            )

            # Get confidence score
            confidence = self._get_confidence_score(image)

            return text.strip(), confidence, rotation_applied

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
