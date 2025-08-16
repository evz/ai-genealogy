import logging

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance

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
            if file_path.lower().endswith(".pdf"):
                image = self._pdf_to_image(file_path)
            else:
                image = Image.open(file_path)

            # Convert to grayscale for better OCR
            if image.mode != "L":
                image = image.convert("L")

            # Detect and correct rotation
            rotation_applied = self._detect_and_correct_rotation(image)
            if rotation_applied != 0:
                image = image.rotate(-rotation_applied, expand=True)
                logger.info(f"Applied rotation correction: {rotation_applied} degrees")

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

    def _detect_and_correct_rotation(self, image: Image.Image) -> float:
        """
        Detect rotation angle using OSD (Orientation and Script Detection)

        Returns:
            Rotation angle in degrees (0, 90, 180, 270, or fine-tuned angle)
        """
        try:
            # Use Tesseract's OSD to detect orientation
            osd_data = pytesseract.image_to_osd(image, config="--psm 0")

            # Parse orientation from OSD output
            rotation_angle = 0
            for line in osd_data.split("\n"):
                if "Rotate:" in line:
                    rotation_angle = int(line.split(":")[1].strip())
                    break

            return rotation_angle

        except Exception as e:
            logger.warning(f"Rotation detection failed, assuming no rotation: {e}")
            return 0

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
