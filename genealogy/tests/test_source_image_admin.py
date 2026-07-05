import tempfile
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from genealogy.admin import SourceImageAdmin
from genealogy.models import Archive, SourceImage


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class SourceImageAdminActionTests(TestCase):
    """Test SourceImage admin actions enqueue the expected tasks"""

    def setUp(self):
        self.archive = Archive.objects.create(abbreviation="RAR", name="Regionaal Archief")
        self.admin = SourceImageAdmin(model=SourceImage, admin_site=None)

    def _create_image(self, **kwargs) -> SourceImage:
        defaults = {
            "archive": self.archive,
            "toegangsnummer": "123",
            "inventarisnummer": "45",
            "page_number": 1,
            "image_file": SimpleUploadedFile("page.jpg", b"fake image bytes", content_type="image/jpeg"),
            "is_handwritten": True,
        }
        defaults.update(kwargs)
        return SourceImage.objects.create(**defaults)

    @patch.object(SourceImageAdmin, "message_user")
    @patch("genealogy.admin.source_image.transcribe_source_image.delay")
    def test_process_selected_queues_transcription(self, mock_delay, mock_message_user):  # noqa: ARG002
        image = self._create_image()
        queryset = SourceImage.objects.filter(id=image.id)

        self.admin.process_selected(request=None, queryset=queryset)

        mock_delay.assert_called_once_with(str(image.id))

    @patch.object(SourceImageAdmin, "message_user")
    @patch("genealogy.admin.source_image.translate_source_image.delay")
    def test_retranslate_selected_only_queues_completed_transcriptions(self, mock_delay, mock_message_user):  # noqa: ARG002
        completed_image = self._create_image(transcription_status="completed")
        pending_image = self._create_image(transcription_status="pending")
        queryset = SourceImage.objects.filter(id__in=[completed_image.id, pending_image.id])

        self.admin.retranslate_selected(request=None, queryset=queryset)

        mock_delay.assert_called_once_with(str(completed_image.id))
