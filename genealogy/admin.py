import logging
import os
import re
from typing import TYPE_CHECKING

from django import forms
from django.contrib import admin, messages
from django.shortcuts import redirect, render
from django.urls import path
from django.utils.html import format_html

if TYPE_CHECKING:
    from django.core.files.uploadedfile import UploadedFile

from .models import (
    Document,
    DocumentPage,
    Event,
    ParentChildRelationship,
    Partnership,
    Person,
    Place,
    TextChunk,
)
from .ollama_utils import get_default_models
from .tasks import (
    create_document_chunks,
    process_genealogy_extraction,
    process_page_ocr,
)

logger = logging.getLogger(__name__)


class TextChunkAdminForm(forms.ModelForm):
    """Custom form for TextChunk admin with user-friendly curation fields"""

    class Meta:
        model = TextChunk
        fields = [
            "text_content",
            "chunk_type",
            "start_page",
            "end_page",
            "sequence_number",
            "generation_number",
            "generation_header",
            "genealogy_ids",
            "person_names",
            "dates",
            "places",
            "family_groups",
            "entities_extracted",
            "manually_reviewed",
            "extraction_method",
        ]
        widgets = {
            "genealogy_ids": forms.Textarea(
                attrs={
                    "rows": 2,
                    "cols": 80,
                    "placeholder": "II.1.a, II.1.b, III.5.c (leave empty if no IDs in this chunk)",
                }
            ),
            "person_names": forms.Textarea(
                attrs={
                    "rows": 2,
                    "cols": 80,
                    "placeholder": "Johan van der Berg, Maria Janssen, Pieter de Wit",
                }
            ),
            "dates": forms.Textarea(
                attrs={
                    "rows": 2,
                    "cols": 80,
                    "placeholder": "1654-03-15, 1658-12-22, 1675-01-01",
                }
            ),
            "places": forms.Textarea(
                attrs={
                    "rows": 2,
                    "cols": 80,
                    "placeholder": "Amsterdam, Utrecht, Haarlem",
                }
            ),
            "family_groups": forms.Textarea(
                attrs={
                    "rows": 2,
                    "cols": 80,
                    "placeholder": "II.9. Children of..., III.1. Kinderen van...",
                }
            ),
            "text_content": forms.Textarea(attrs={"rows": 10, "cols": 80}),
        }


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "page_count",
        "languages",
        "upload_date",
        "ocr_status",
        "extraction_status",
        "llm_model_used",
        "embedding_model_used",
    ]
    list_filter = ["ocr_completed", "extraction_completed", "upload_date"]
    search_fields = ["title"]
    readonly_fields = ["id", "upload_date"]

    def get_urls(self):
        """Add custom URLs for batch upload"""
        urls = super().get_urls()
        custom_urls = [
            path(
                "batch-upload/",
                self.admin_site.admin_view(self.batch_upload_view),
                name="genealogy_document_batch_upload",
            ),
        ]
        return custom_urls + urls

    def ocr_status(self, obj: Document) -> str:
        if obj.ocr_completed:
            return format_html('<span style="color: green;">✓ Completed</span>')
        return format_html('<span style="color: orange;">⏳ Pending</span>')

    ocr_status.short_description = "OCR Status"  # type: ignore

    def extraction_status(self, obj: Document) -> str:
        if obj.extraction_completed:
            return format_html('<span style="color: green;">✓ Completed</span>')
        return format_html('<span style="color: orange;">⏳ Pending</span>')

    extraction_status.short_description = "Extraction Status"  # type: ignore

    actions = ["extract_genealogy_data", "rerun_text_chunking"]

    def extract_genealogy_data(self, request, queryset):
        """Admin action: Start multi-phase genealogy extraction for selected documents"""
        success_count = 0
        error_count = 0

        for doc in queryset:
            if not doc.can_extract_genealogy():
                error_count += 1
                self.message_user(
                    request,
                    f"Document {doc.title} is not ready for extraction (OCR must be completed first).",
                    level=messages.WARNING,
                )
                continue

            try:
                # Get default models from environment
                defaults = get_default_models()

                # Store model configuration in document
                doc.llm_model_used = defaults["llm_model"]
                doc.embedding_model_used = defaults["embedding_model"]
                doc.save(update_fields=["llm_model_used", "embedding_model_used"])

                # Start multi-phase extraction process
                task = process_genealogy_extraction.delay(str(doc.id))
                success_count += 1
                logger.info(f"Started genealogy extraction task {task.id} for document {doc}")

            except Exception as e:
                error_count += 1
                self.message_user(
                    request,
                    f"Error starting extraction for {doc.title}: {e}",
                    level=messages.ERROR,
                )

        if success_count:
            defaults = get_default_models()
            self.message_user(
                request,
                f"Genealogy extraction started for {success_count} documents using {defaults['llm_model']} (LLM) and {defaults['embedding_model']} (embeddings).",
            )
        if error_count:
            self.message_user(
                request,
                f"{error_count} documents could not be processed.",
                level=messages.WARNING,
            )

    extract_genealogy_data.short_description = (  # type: ignore
        "Extract genealogy data from processed documents"
    )

    def rerun_text_chunking(self, request, queryset):
        """Admin action: Re-run text chunking process for selected documents"""
        success_count = 0
        error_count = 0

        for doc in queryset:
            if not doc.ocr_completed:
                error_count += 1
                self.message_user(
                    request,
                    f"Document {doc.title} is not ready for chunking (OCR must be completed first).",
                    level=messages.WARNING,
                )
                continue

            try:
                # Start text chunking task
                task = create_document_chunks.delay(str(doc.id))
                success_count += 1
                logger.info(f"Started text chunking task {task.id} for document {doc}")

            except Exception as e:
                error_count += 1
                self.message_user(
                    request,
                    f"Error starting text chunking for {doc.title}: {e}",
                    level=messages.ERROR,
                )

        if success_count:
            self.message_user(
                request,
                f"Text chunking started for {success_count} documents. "
                f"This will clear and recreate all text chunks for these documents.",
            )
        if error_count:
            self.message_user(
                request,
                f"{error_count} documents could not be processed.",
                level=messages.WARNING,
            )

    rerun_text_chunking.short_description = (  # type: ignore
        "Re-run text chunking for selected documents (clears existing chunks)"
    )

    def batch_upload_view(self, request):
        """Custom view for batch file upload"""
        if request.method == "POST":
            return self._handle_batch_upload(request)

        # Render the upload form
        context = {
            "title": "Batch Upload Documents",
            "opts": self.model._meta,
            "has_permission": True,
        }
        return render(request, "admin/genealogy/document/batch_upload.html", context)

    def _handle_batch_upload(self, request):
        """Process the batch upload form submission"""
        try:
            files = request.FILES.getlist("files")
            language = request.POST.get("language", "en")
            upload_mode = request.POST.get("upload_mode", "single_document")
            document_title = request.POST.get("document_title", "").strip()

            # Debug: Log the number of files received
            logger.info("Batch upload: Received %d files from request", len(files))
            for i, f in enumerate(files):
                logger.info("Batch upload: File %d: %s (%d bytes)", i + 1, f.name, f.size)

            if not files:
                messages.error(request, "No files were selected for upload.")
                return redirect("admin:genealogy_document_batch_upload")

            # Filter valid files
            valid_files = []
            for uploaded_file in files:
                if self._is_valid_file_type(uploaded_file):
                    valid_files.append(uploaded_file)
                else:
                    messages.warning(request, f"Skipped {uploaded_file.name}: unsupported file type")

            if not valid_files:
                messages.error(request, "No valid files to upload.")
                return redirect("admin:genealogy_document_batch_upload")

            documents_created = 0
            pages_created = 0
            created_documents = []

            if upload_mode == "single_document":
                # Create one document with multiple pages
                if not document_title:
                    document_title = self._get_document_title_from_filename(valid_files[0].name)

                document = Document.objects.create(
                    title=document_title,
                    languages=language,
                )
                documents_created = 1
                created_documents.append(document)

                # Sort files by extracted page number for proper ordering
                files_with_page_numbers: list[tuple[int, UploadedFile]] = []
                for uploaded_file in valid_files:
                    page_num = self._extract_page_number_from_filename(uploaded_file.name)
                    if page_num is None:
                        logger.warning(
                            "Could not extract page number from %s, using filename order",
                            uploaded_file.name,
                        )
                        # Use file index as fallback
                        page_num = len(files_with_page_numbers) + 1
                    files_with_page_numbers.append((page_num, uploaded_file))

                # Sort by page number
                files_with_page_numbers.sort(key=lambda x: x[0])

                # Create pages with extracted page numbers
                for page_num, uploaded_file in files_with_page_numbers:
                    page = DocumentPage.objects.create(
                        document=document,
                        page_number=page_num,
                        image_file=uploaded_file,
                        original_filename=uploaded_file.name,
                    )
                    pages_created += 1

            else:
                # Create separate documents (original behavior)
                for uploaded_file in valid_files:
                    document_title = self._get_document_title_from_filename(uploaded_file.name)
                    document = Document.objects.create(
                        title=document_title,
                        languages=language,
                    )
                    documents_created += 1
                    created_documents.append(document)

                    # Create document page
                    page = DocumentPage.objects.create(
                        document=document,
                        page_number=1,
                        image_file=uploaded_file,
                        original_filename=uploaded_file.name,
                    )
                    pages_created += 1

            # Automatically start OCR processing for uploaded files
            ocr_started = 0
            for document in created_documents:
                if document.can_process_ocr():
                    # Start OCR for all unprocessed pages in this document
                    unprocessed_pages = document.pages.filter(ocr_completed=False)
                    for page in unprocessed_pages:
                        try:
                            page.validate_for_ocr()
                            process_page_ocr.delay(str(page.id))
                            ocr_started += 1
                        except ValueError as e:
                            messages.warning(request, f"Could not start OCR for {page}: {e}")

            if ocr_started > 0:
                messages.success(
                    request,
                    f"Successfully uploaded {documents_created} documents with {pages_created} pages. "
                    f"OCR processing started for {ocr_started} pages.",
                )
            else:
                messages.success(
                    request,
                    f"Successfully uploaded {documents_created} documents with {pages_created} pages. "
                    f"No pages were ready for OCR processing.",
                )

            return redirect("admin:genealogy_document_changelist")

        except Exception as e:
            messages.error(request, f"Error during batch upload: {e!s}")
            return redirect("admin:genealogy_document_batch_upload")

    def _is_valid_file_type(self, uploaded_file):
        """Check if uploaded file is a supported image or PDF"""
        allowed_extensions = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"]
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        return file_ext in allowed_extensions

    def _get_document_title_from_filename(self, filename):
        """Extract a clean title from the filename"""
        # Remove extension and clean up
        title = os.path.splitext(filename)[0]
        # Replace underscores and hyphens with spaces
        title = title.replace("_", " ").replace("-", " ")
        # Capitalize words
        return title.title()

    def _extract_page_number_from_filename(self, filename):
        """
        Extract page number from filename ending.
        Expected format: filename ending with 3-digit page number like '026.pdf'

        Args:
            filename: The uploaded filename

        Returns:
            int: Page number if found, None otherwise
        """
        # Remove extension and get the base name
        base_name = os.path.splitext(filename)[0]

        # Look for 3-digit number at the end of filename
        match = re.search(r"(\d{3})$", base_name)
        if match:
            return int(match.group(1))

        # Fallback: look for any number at the end
        match = re.search(r"(\d+)$", base_name)
        if match:
            return int(match.group(1))

        return None


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


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = [
        "full_name",
        "gender",
        "birth_date",
        "death_date",
        "genealogical_id",
    ]
    list_filter = ["gender", "birth_date", "death_date"]
    search_fields = ["given_names", "surname", "maiden_name", "genealogical_id"]
    readonly_fields = ["id", "created_at", "updated_at"]
    filter_horizontal = ["source_documents"]

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "given_names",
                    "surname",
                    "maiden_name",
                    "gender",
                    "genealogical_id",
                ),
            },
        ),
        (
            "Birth Details",
            {
                "fields": ("birth_date", "birth_date_estimated", "birth_place"),
            },
        ),
        (
            "Death Details",
            {
                "fields": ("death_date", "death_date_estimated", "death_place"),
            },
        ),
        (
            "Sources",
            {
                "fields": ("source_documents",),
            },
        ),
        (
            "Metadata",
            {
                "fields": ("id", "created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )


@admin.register(Place)
class PlaceAdmin(admin.ModelAdmin):
    list_display = ["name", "locality", "region", "country"]
    list_filter = ["region", "country"]
    search_fields = ["name", "locality", "region", "country"]
    readonly_fields = ["id"]


@admin.register(Partnership)
class PartnershipAdmin(admin.ModelAdmin):
    list_display = ["__str__", "partnership_type", "start_date", "end_date"]
    list_filter = ["partnership_type", "start_date", "end_date"]
    readonly_fields = ["id", "created_at", "updated_at"]
    filter_horizontal = ["partners", "source_documents"]

    fieldsets = (
        (
            "Partnership Details",
            {
                "fields": ("partners", "partnership_type"),
            },
        ),
        (
            "Start Details",
            {
                "fields": ("start_date", "start_date_estimated", "start_place"),
            },
        ),
        (
            "End Details",
            {
                "fields": ("end_date", "end_date_estimated", "end_reason"),
            },
        ),
        (
            "Sources",
            {
                "fields": ("source_documents",),
            },
        ),
        (
            "Metadata",
            {
                "fields": ("id", "created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ["__str__", "event_type", "date", "place"]
    list_filter = ["event_type", "date", "date_estimated"]
    search_fields = ["description"]
    readonly_fields = ["id", "created_at", "updated_at"]
    filter_horizontal = ["source_documents"]


@admin.register(ParentChildRelationship)
class ParentChildRelationshipAdmin(admin.ModelAdmin):
    list_display = ["child", "parent", "relationship_type", "partnership"]
    list_filter = ["relationship_type"]
    search_fields = [
        "child__given_names",
        "child__surname",
        "parent__given_names",
        "parent__surname",
    ]
    readonly_fields = ["id", "created_at"]
    filter_horizontal = ["source_documents"]


@admin.register(TextChunk)
class TextChunkAdmin(admin.ModelAdmin):
    form = TextChunkAdminForm
    list_display = [
        "__str__",
        "document",
        "chunk_type",
        "sequence_number",
        "generation_number",
        "extraction_method",
        "manually_reviewed",
        "entities_extracted",
        "genealogy_ids_count",
    ]
    list_filter = [
        "chunk_type",
        "extraction_method",
        "manually_reviewed",
        "entities_extracted",
        "generation_number",
        "document",
    ]
    search_fields = ["text_content", "generation_header", "genealogy_ids"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "document",
                    "chunk_type",
                    "sequence_number",
                ),
            },
        ),
        (
            "Position",
            {
                "fields": ("start_page", "end_page"),
            },
        ),
        (
            "Genealogical Anchors",
            {
                "fields": (
                    "generation_number",
                    "generation_header",
                    "genealogy_ids",
                    "person_names",
                    "dates",
                    "places",
                    "family_groups",
                    "manually_reviewed",
                ),
                "description": "All genealogical data extracted from this chunk. You can manually review and correct these values.",
            },
        ),
        (
            "Content",
            {
                "fields": ("text_content",),
            },
        ),
        (
            "Processing Status",
            {
                "fields": ("entities_extracted", "extraction_method"),
            },
        ),
        (
            "Metadata",
            {
                "fields": ("id", "created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    def genealogy_ids_count(self, obj: TextChunk) -> int:
        return len(obj.genealogy_ids)

    genealogy_ids_count.short_description = "ID Count"  # type: ignore

    def extraction_method(self, obj: TextChunk) -> str:
        """Display extraction method with color coding"""
        method = obj.extraction_method
        if method == "neural_network":
            return format_html('<span style="color: blue; font-weight: bold;">🧠 Neural Network</span>')
        if method == "hybrid":
            return format_html('<span style="color: orange; font-weight: bold;">⚡ Hybrid</span>')
        # regex
        return format_html('<span style="color: gray;">🔍 Regex</span>')

    extraction_method.short_description = "Extraction Method"  # type: ignore

    def manually_reviewed(self, obj: TextChunk) -> str:
        """Display manual review status with color coding"""
        if obj.manually_reviewed:
            return format_html('<span style="color: green; font-weight: bold;">✅ Reviewed</span>')
        return format_html('<span style="color: orange;">⏳ Pending</span>')

    manually_reviewed.short_description = "Manual Review"  # type: ignore
