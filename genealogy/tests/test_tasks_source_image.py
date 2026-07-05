import tempfile
import uuid
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from genealogy.models import Archive, SourceImage
from genealogy.tasks import transcribe_source_image, translate_source_image


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class SourceImageTaskTests(TestCase):
    """Test transcription/translation Celery tasks for SourceImage - mock external services"""

    def setUp(self):
        self.archive = Archive.objects.create(abbreviation="RAR", name="Regionaal Archief")

    def _create_image(self, is_handwritten: bool, **kwargs) -> SourceImage:
        defaults = {
            "archive": self.archive,
            "toegangsnummer": "123",
            "inventarisnummer": "45",
            "page_number": 1,
            "image_file": SimpleUploadedFile("page.jpg", b"fake image bytes", content_type="image/jpeg"),
            "is_handwritten": is_handwritten,
        }
        defaults.update(kwargs)
        return SourceImage.objects.create(**defaults)

    # -- transcribe_source_image --

    def test_transcribe_invalid_uuid(self):
        result = transcribe_source_image("invalid-uuid-format")

        self.assertFalse(result["success"])
        self.assertIn("Invalid UUID format", result["error"])

    def test_transcribe_nonexistent(self):
        result = transcribe_source_image(str(uuid.uuid4()))

        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    @patch("genealogy.tasks.source_image.translate_source_image.delay")
    @patch("genealogy.tasks.source_image.LoghiClient")
    def test_transcribe_handwritten_success(self, mock_loghi_cls, mock_translate_delay):
        mock_loghi_cls.return_value.transcribe.return_value = "raw dutch text"
        image = self._create_image(is_handwritten=True)

        result = transcribe_source_image(str(image.id))

        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "loghi")

        image.refresh_from_db()
        self.assertEqual(image.transcription_status, "completed")
        self.assertEqual(image.transcription_method, "loghi")
        self.assertEqual(image.raw_transcription, "raw dutch text")
        self.assertIsNotNone(image.transcribed_at)
        mock_translate_delay.assert_called_once_with(str(image.id))

    @patch("genealogy.tasks.source_image.translate_source_image.delay")
    @patch("genealogy.tasks.source_image.OllamaClient")
    def test_transcribe_printed_success(self, mock_ollama_cls, mock_translate_delay):
        mock_ollama_cls.return_value.generate_with_image.return_value = "raw printed text"
        image = self._create_image(is_handwritten=False)

        result = transcribe_source_image(str(image.id))

        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "ollama_deepseek_ocr")

        image.refresh_from_db()
        self.assertEqual(image.transcription_status, "completed")
        self.assertEqual(image.transcription_method, "ollama_deepseek_ocr")
        self.assertEqual(image.raw_transcription, "raw printed text")
        mock_translate_delay.assert_called_once_with(str(image.id))

    @patch("genealogy.tasks.source_image.LoghiClient")
    def test_transcribe_failure_records_error(self, mock_loghi_cls):
        mock_loghi_cls.return_value.transcribe.side_effect = RuntimeError("orchestrator unreachable")
        image = self._create_image(is_handwritten=True)

        result = transcribe_source_image(str(image.id))

        self.assertFalse(result["success"])

        image.refresh_from_db()
        self.assertEqual(image.transcription_status, "failed")
        self.assertIn("orchestrator unreachable", image.transcription_error)

    # -- translate_source_image --

    def test_translate_requires_completed_transcription(self):
        image = self._create_image(is_handwritten=True)  # transcription_status defaults to "pending"

        result = translate_source_image(str(image.id))

        self.assertFalse(result["success"])
        self.assertIn("not 'completed'", result["error"])

    @patch("genealogy.tasks.source_image.OllamaClient")
    def test_translate_success(self, mock_ollama_cls):
        mock_ollama_cls.return_value.generate.return_value = "translated english text"
        image = self._create_image(
            is_handwritten=True,
            transcription_status="completed",
            transcription_method="loghi",
            raw_transcription="raw dutch text",
        )

        result = translate_source_image(str(image.id))

        self.assertTrue(result["success"])

        image.refresh_from_db()
        self.assertEqual(image.translation_status, "completed")
        self.assertEqual(image.translation, "translated english text")
        self.assertIsNotNone(image.translated_at)

    @patch("genealogy.tasks.source_image.OllamaClient")
    def test_translate_failure_records_error(self, mock_ollama_cls):
        mock_ollama_cls.return_value.generate.side_effect = RuntimeError("ollama unreachable")
        image = self._create_image(
            is_handwritten=True,
            transcription_status="completed",
            transcription_method="loghi",
            raw_transcription="raw dutch text",
        )

        result = translate_source_image(str(image.id))

        self.assertFalse(result["success"])

        image.refresh_from_db()
        self.assertEqual(image.translation_status, "failed")
        self.assertIn("ollama unreachable", image.translation_error)
