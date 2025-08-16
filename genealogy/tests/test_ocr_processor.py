import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from django.test import TestCase

from genealogy.ocr_processor import OCRProcessor


class OCRProcessorTests(TestCase):
    """Test OCR processor business logic - mock external dependencies"""

    def setUp(self):
        """Create OCR processor"""
        self.processor = OCRProcessor(language="eng+nld")

    @patch("genealogy.ocr_processor.ImageEnhance.Sharpness")
    @patch("genealogy.ocr_processor.ImageEnhance.Contrast")
    @patch("genealogy.ocr_processor.pytesseract.image_to_string")
    @patch("genealogy.ocr_processor.pytesseract.image_to_osd")
    @patch("genealogy.ocr_processor.pytesseract.image_to_data")
    @patch("genealogy.ocr_processor.Image.open")
    def test_process_image_file_success(
        self,
        mock_image_open,
        mock_image_to_data,
        mock_image_to_osd,
        mock_image_to_string,
        mock_contrast,
        mock_sharpness,
    ):
        """Processing image file should return text, confidence, and rotation"""
        # Mock image
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image.convert.return_value = mock_image
        mock_image.rotate.return_value = mock_image
        mock_image_open.return_value = mock_image

        # Mock image enhancers
        mock_contrast_enhancer = Mock()
        mock_contrast_enhancer.enhance.return_value = mock_image
        mock_contrast.return_value = mock_contrast_enhancer

        mock_sharpness_enhancer = Mock()
        mock_sharpness_enhancer.enhance.return_value = mock_image
        mock_sharpness.return_value = mock_sharpness_enhancer

        # Mock OCR results
        mock_image_to_osd.return_value = "Rotate: 0\nOrientation confidence: 1.23"
        mock_image_to_string.return_value = "Sample OCR text from image"
        mock_image_to_data.return_value = {"conf": ["95", "87", "92", "88"]}

        # Create temp image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            text, confidence, rotation = self.processor.process_file(tmp_path)

            self.assertEqual(text, "Sample OCR text from image")
            self.assertEqual(confidence, 90.5)  # Average of [95, 87, 92, 88]
            self.assertEqual(rotation, 0)

            # Verify image was converted to grayscale
            mock_image.convert.assert_called_with("L")

        finally:
            Path(tmp_path).unlink()

    @patch("genealogy.ocr_processor.ImageEnhance.Sharpness")
    @patch("genealogy.ocr_processor.ImageEnhance.Contrast")
    @patch("genealogy.ocr_processor.convert_from_bytes")
    @patch("genealogy.ocr_processor.pytesseract.image_to_string")
    @patch("genealogy.ocr_processor.pytesseract.image_to_osd")
    @patch("genealogy.ocr_processor.pytesseract.image_to_data")
    def test_process_pdf_file_success(
        self,
        mock_image_to_data,
        mock_image_to_osd,
        mock_image_to_string,
        mock_convert_from_bytes,
        mock_contrast,
        mock_sharpness,
    ):
        """Processing PDF file should convert to image and return OCR results"""
        # Mock PDF conversion
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image.convert.return_value = mock_image
        mock_image.rotate.return_value = mock_image
        mock_convert_from_bytes.return_value = [mock_image]

        # Mock image enhancers
        mock_contrast_enhancer = Mock()
        mock_contrast_enhancer.enhance.return_value = mock_image
        mock_contrast.return_value = mock_contrast_enhancer

        mock_sharpness_enhancer = Mock()
        mock_sharpness_enhancer.enhance.return_value = mock_image
        mock_sharpness.return_value = mock_sharpness_enhancer

        # Mock OCR results
        mock_image_to_osd.return_value = "Rotate: 90\nOrientation confidence: 1.23"
        mock_image_to_string.return_value = "PDF OCR text content"
        mock_image_to_data.return_value = {"conf": ["80", "75", "85"]}

        # Create temp PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            tmp_path = tmp_file.name

        try:
            text, confidence, rotation = self.processor.process_file(tmp_path)

            self.assertEqual(text, "PDF OCR text content")
            self.assertEqual(confidence, 80.0)  # Average of [80, 75, 85]
            self.assertEqual(rotation, 90)

            # Verify PDF was converted
            mock_convert_from_bytes.assert_called_once()

        finally:
            Path(tmp_path).unlink()

    @patch("genealogy.ocr_processor.pytesseract.image_to_osd")
    def test_detect_rotation_failure_returns_zero(self, mock_image_to_osd):
        """Rotation detection failure should return 0 degrees"""
        mock_image_to_osd.side_effect = Exception("OSD failed")

        mock_image = Mock()
        rotation = self.processor._detect_and_correct_rotation(mock_image)

        self.assertEqual(rotation, 0)

    @patch("genealogy.ocr_processor.pytesseract.image_to_data")
    def test_confidence_calculation_filters_invalid_scores(self, mock_image_to_data):
        """Confidence calculation should average only valid scores"""
        mock_image_to_data.return_value = {"conf": ["95", "0", "87", "-1", "92"]}

        mock_image = Mock()
        confidence = self.processor._get_confidence_score(mock_image)

        # Should average only positive scores: (95 + 87 + 92) / 3 = 91.33
        self.assertAlmostEqual(confidence, 91.33, places=1)

    @patch("genealogy.ocr_processor.convert_from_bytes")
    def test_pdf_conversion_no_pages_raises_error(self, mock_convert_from_bytes):
        """PDF with no pages should raise ValueError"""
        mock_convert_from_bytes.return_value = []

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            tmp_path = tmp_file.name

        try:
            with self.assertRaises(ValueError) as cm:
                self.processor._pdf_to_image(tmp_path)
            self.assertIn("Could not extract image from PDF", str(cm.exception))
        finally:
            Path(tmp_path).unlink()

    def test_process_nonexistent_file_raises_error(self):
        """Processing nonexistent file should raise exception"""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_file("/nonexistent/file.pdf")
