import logging

from django.contrib import admin, messages
from django.utils.html import format_html

from ..models import DocumentPage
from ..tasks import process_page_ocr

logger = logging.getLogger(__name__)


@admin.register(DocumentPage)
class DocumentPageAdmin(admin.ModelAdmin):
    list_display = [
        "__str__",
        "document",
        "page_number",
        "filename",
        "ocr_status",
        "ocr_confidence",
    ]
    list_filter = ["ocr_completed", "document", "created_at"]
    search_fields = ["document__title", "original_filename"]
    readonly_fields = ["id", "filename", "created_at", "updated_at"]
    actions = ["process_ocr", "reprocess_ocr"]

    def ocr_status(self, obj: DocumentPage) -> str:
        if obj.ocr_completed:
            return format_html('<span style="color: green;">✓ Completed</span>')
        return format_html('<span style="color: orange;">⏳ Pending</span>')

    ocr_status.short_description = "OCR Status"  # type: ignore

    def process_ocr(self, request, queryset):
        """Admin action: Start OCR processing for selected pages (unprocessed only)"""
        processed_count = 0
        skipped_count = 0
        error_count = 0

        for page in queryset:
            try:
                if page.ocr_completed:
                    skipped_count += 1
                    continue

                page.validate_for_ocr()
                task = process_page_ocr.delay(str(page.id))
                processed_count += 1
                logger.info("Started OCR task %s for page %s", task.id, page)

            except ValueError as e:
                error_count += 1
                self.message_user(request, f"Error processing {page}: {e}", level=messages.ERROR)

        if processed_count:
            self.message_user(request, f"OCR processing started for {processed_count} pages.")
        if skipped_count:
            self.message_user(
                request,
                f"{skipped_count} pages skipped (already processed).",
                level=messages.WARNING,
            )
        if error_count:
            self.message_user(
                request,
                f"{error_count} pages could not be processed.",
                level=messages.ERROR,
            )

    process_ocr.short_description = "Process OCR for selected pages (unprocessed only)"  # type: ignore

    def reprocess_ocr(self, request, queryset):
        """Admin action: Reprocess OCR for selected pages (including already processed)"""
        processed_count = 0
        error_count = 0

        for page in queryset:
            try:
                # Reset OCR status to allow reprocessing
                page.ocr_completed = False
                page.ocr_text = ""
                page.ocr_confidence = None
                page.rotation_applied = 0.0
                page.save(
                    update_fields=[
                        "ocr_completed",
                        "ocr_text",
                        "ocr_confidence",
                        "rotation_applied",
                    ]
                )

                page.validate_for_ocr()
                task = process_page_ocr.delay(str(page.id))
                processed_count += 1
                logger.info("Started OCR reprocessing task %s for page %s", task.id, page)

            except ValueError as e:
                error_count += 1
                self.message_user(request, f"Error reprocessing {page}: {e}", level=messages.ERROR)

        if processed_count:
            self.message_user(request, f"OCR reprocessing started for {processed_count} pages.")
        if error_count:
            self.message_user(
                request,
                f"{error_count} pages could not be reprocessed.",
                level=messages.ERROR,
            )

    reprocess_ocr.short_description = (  # type: ignore
        "Reprocess OCR for selected pages (force reprocess)"
    )
