"""Admin interface for SourceImage model"""
from django.contrib import admin
from django.utils.html import format_html

from ..models import SourceImage
from ..tasks import transcribe_source_image, translate_source_image


@admin.register(SourceImage)
class SourceImageAdmin(admin.ModelAdmin):
    list_display = [
        "__str__",
        "is_handwritten",
        "transcription_status",
        "translation_status",
        "persons_display",
    ]
    list_filter = ["is_handwritten", "transcription_status", "translation_status", "archive"]
    search_fields = ["toegangsnummer", "inventarisnummer"]
    autocomplete_fields = ["archive"]
    filter_horizontal = ["persons"]
    readonly_fields = [
        "id",
        "transcription_method",
        "transcription_status",
        "raw_transcription",
        "transcription_error",
        "transcribed_at",
        "translation_status",
        "translation",
        "translation_model",
        "translation_error",
        "translated_at",
        "created_at",
        "updated_at",
    ]
    actions = ["process_selected", "retranslate_selected"]

    def persons_display(self, obj: SourceImage) -> str:
        """Display the people this image is attached to"""
        persons = obj.persons.all()
        if not persons:
            return format_html('<em style="color: #999;">No people attached</em>')
        return format_html(", ".join(p.full_name for p in persons))

    persons_display.short_description = "People"  # type: ignore

    def process_selected(self, request, queryset):
        """Admin action: queue selected images for transcription + translation"""
        for image in queryset:
            transcribe_source_image.delay(str(image.id))
        self.message_user(request, f"Queued {queryset.count()} image(s) for transcription + translation.")

    process_selected.short_description = "Process selected images (transcribe + translate)"  # type: ignore

    def retranslate_selected(self, request, queryset):
        """Admin action: re-run translation only, for images that already have a transcription"""
        completed = queryset.filter(transcription_status="completed")
        for image in completed:
            translate_source_image.delay(str(image.id))
        self.message_user(request, f"Queued {completed.count()} image(s) for re-translation.")

    retranslate_selected.short_description = "Re-translate selected images"  # type: ignore
