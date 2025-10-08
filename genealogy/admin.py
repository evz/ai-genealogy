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
from .ollama_utils import OllamaClient, get_default_models
from .tasks import (
    create_document_chunks,
    extract_entities_from_chunks,
    extract_entities_from_chunk,
    process_page_ocr,
)

logger = logging.getLogger(__name__)


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

    actions = ["extract_genealogy_data", "rerun_text_chunking", "reset_extraction_status"]

    def extract_genealogy_data(self, request, queryset):
        """Admin action: Start LLM-based entity extraction from chunks for selected documents"""
        success_count = 0
        error_count = 0

        for doc in queryset:
            # Check if document has chunks ready for extraction
            if not doc.text_chunks.filter(chunk_type="GENEALOGY_ENTRY").exists():
                error_count += 1
                self.message_user(
                    request,
                    f"Document {doc.title} has no GENEALOGY_ENTRY chunks. Run text chunking first.",
                    level=messages.WARNING,
                )
                continue

            try:
                # Get default models from environment and update document
                defaults = get_default_models()

                # Always update to current environment defaults
                doc.llm_model_used = defaults["llm_model"]
                doc.embedding_model_used = defaults["embedding_model"]
                doc.save(update_fields=["llm_model_used", "embedding_model_used"])

                # Start entity extraction process
                task = extract_entities_from_chunks.delay(str(doc.id))
                success_count += 1
                logger.info(f"Started entity extraction task {task.id} for document {doc}")

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
                f"Entity extraction started for {success_count} documents using {defaults['llm_model']}.",
            )
        if error_count:
            self.message_user(
                request,
                f"{error_count} documents could not be processed.",
                level=messages.WARNING,
            )

    extract_genealogy_data.short_description = (  # type: ignore
        "Extract entities from chunks (LLM extraction)"
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

    def reset_extraction_status(self, request, queryset):
        """Admin action: Reset extraction status to allow re-extraction"""
        # Bulk reset all chunks for selected documents
        chunks_reset = TextChunk.objects.filter(
            document__in=queryset,
            entities_extracted=True
        ).update(
            entities_extracted=False,
            extracted_people=[],
            extracted_relationships=[],
            extracted_events=[],
            dm_codes=[]
        )

        # Reset document extraction status
        queryset.update(extraction_completed=False)

        self.message_user(
            request,
            f"Reset extraction status for {queryset.count()} documents ({chunks_reset} chunks). "
            f"You can now re-run entity extraction.",
        )

    reset_extraction_status.short_description = (  # type: ignore
        "Reset extraction status (clear extracted data)"
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
    list_display = [
        "__str__",
        "document",
        "chunk_type",
        "sequence_number",
        "generation_number",
        "char_count",
        "token_count",
    ]
    list_filter = [
        "chunk_type",
        "extraction_method",
        "manually_reviewed",
        "entities_extracted",
        "generation_number",
        "document",
    ]
    search_fields = ["text_content", "generation_header"]
    readonly_fields = [
        "id", "created_at", "updated_at", "document", "chunk_type", "sequence_number",
        "start_page", "end_page", "entities_extracted", "generation_number",
        "generation_header", "family_groups", "text_content", "extraction_method",
        "formatted_people", "formatted_relationships", "formatted_events"
    ]
    actions = ["reextract_entities"]

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
                    "entities_extracted",
                    "generation_number",
                    "generation_header",
                    "family_groups",
                ),
                "description": "Genealogical context for this chunk.",
            },
        ),
        (
            "Structured Extraction (Graph Data)",
            {
                "fields": ("formatted_people", "formatted_relationships", "formatted_events"),
                "description": "Structured person and relationship data extracted by LLM for graph building.",
            },
        ),
        (
            "Content",
            {
                "fields": ("text_content",),
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

    def char_count(self, obj: TextChunk) -> str:
        """Display character count"""
        count = len(obj.text_content)
        return f"{count:,}"

    char_count.short_description = "Chars"  # type: ignore

    def token_count(self, obj: TextChunk) -> str:
        """Display approximate token count (chars / 4)"""
        count = len(obj.text_content) // 4
        return f"~{count:,}"

    token_count.short_description = "Tokens"  # type: ignore

    def formatted_people(self, obj: TextChunk) -> str:
        """Display extracted people in a readable format"""
        if not obj.extracted_people:
            return format_html('<em>No people extracted yet</em>')

        html_parts = []
        for i, person_name in enumerate(obj.extracted_people, 1):
            html_parts.append(f"{i}. <strong>{person_name}</strong>")

        return format_html("<br>".join(html_parts))

    formatted_people.short_description = "Extracted People"  # type: ignore

    def formatted_relationships(self, obj: TextChunk) -> str:
        """Display extracted relationships in a readable format"""
        if not obj.extracted_relationships:
            return format_html('<em>No relationships extracted yet</em>')

        html_parts = []
        for i, rel in enumerate(obj.extracted_relationships, 1):
            person1 = rel.get('person1', 'Unknown')
            person2 = rel.get('person2', 'Unknown')
            rel_type = rel.get('relationship_type', 'unknown')

            html_parts.append(
                f"<strong>{i}.</strong> {person1} → <em>{rel_type}</em> → {person2}"
            )

        return format_html("<br>".join(html_parts))

    formatted_relationships.short_description = "Extracted Relationships"  # type: ignore

    def formatted_events(self, obj: TextChunk) -> str:
        """Display extracted events in a readable format"""
        if not obj.extracted_events:
            return format_html('<em>No events extracted yet</em>')

        html_parts = []
        for i, event in enumerate(obj.extracted_events, 1):
            person = event.get('person', 'Unknown')
            event_type = event.get('event_type', 'UNKNOWN')
            date = event.get('date', '')
            place = event.get('place', '')

            parts = [f"<strong>{i}.</strong> {person}"]
            parts.append(f"<span style='color: #0066cc;'>{event_type}</span>")

            if date:
                parts.append(f"on {date}")
            if place:
                parts.append(f"in {place}")

            html_parts.append(" ".join(parts))

        return format_html("<br>".join(html_parts))

    formatted_events.short_description = "Extracted Events"  # type: ignore

    def has_add_permission(self, request):
        """Disable adding text chunks through admin"""
        return False

    def has_change_permission(self, request, obj=None):
        """Disable editing text chunks through admin"""
        return False

    def has_delete_permission(self, request, obj=None):
        """Disable deleting text chunks through admin"""
        return False

    def reextract_entities(self, request, queryset):
        """Admin action: Re-run entity extraction for selected chunks"""
        # Only process GENEALOGY_ENTRY chunks
        chunks = queryset.filter(chunk_type="GENEALOGY_ENTRY")

        if not chunks.exists():
            self.message_user(
                request,
                "No GENEALOGY_ENTRY chunks selected. Only genealogy entries can be extracted.",
                level=messages.WARNING
            )
            return

        # Initialize Ollama client
        ollama = OllamaClient(timeout=600)
        if not ollama.is_available():
            self.message_user(
                request,
                "Ollama server is not available. Cannot perform extraction.",
                level=messages.ERROR
            )
            return

        # Get model from first chunk's document or use default
        first_chunk = chunks.first()
        model = first_chunk.document.llm_model_used or get_default_models()["llm_model"]

        success_count = 0
        failed_count = 0

        for chunk in chunks:
            result = extract_entities_from_chunk(chunk, ollama, model)
            if result['success']:
                success_count += 1
            else:
                failed_count += 1

        # Show summary message
        if success_count > 0:
            self.message_user(
                request,
                f"Successfully re-extracted {success_count} chunk(s) using {model}.",
                level=messages.SUCCESS
            )

        if failed_count > 0:
            self.message_user(
                request,
                f"Failed to extract {failed_count} chunk(s). Check logs for details.",
                level=messages.ERROR
            )

    reextract_entities.short_description = "Re-extract entities from selected chunks"
