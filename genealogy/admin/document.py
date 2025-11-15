import logging
import os
import re
from typing import TYPE_CHECKING

from django import forms
from django.contrib import admin, messages
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html

if TYPE_CHECKING:
    from django.core.files.uploadedfile import UploadedFile

from ..models import (Document, DocumentPage, Event, Place, TextChunk)
from ..ollama_utils import OllamaClient, get_default_models
from ..tasks import (build_genealogy_graph, create_document_chunks,
                     extract_entities_from_chunks, process_page_ocr)

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

    actions = ["build_genealogy_graph", "persist_extracted_entities", "extract_genealogy_data", "rerun_text_chunking", "reset_extraction_status"]

    def build_genealogy_graph(self, request, queryset):
        """Admin action: Build genealogy graph (Person/Relationship/Partnership) from chunks"""
        success_count = 0
        error_count = 0

        for doc in queryset:
            # Check if document has chunks with genealogical identifiers
            chunks_count = doc.text_chunks.filter(
                genealogical_identifier__isnull=False,
                subject__isnull=False,
            ).count()

            if chunks_count == 0:
                error_count += 1
                self.message_user(
                    request,
                    f"Document {doc.title} has no chunks with genealogical identifiers. Run text chunking first.",
                    level=messages.WARNING,
                )
                continue

            try:
                # Start graph building task
                task = build_genealogy_graph.delay(str(doc.id))
                success_count += 1
                logger.info(f"Started build_genealogy_graph task {task.id} for document {doc}")

            except Exception as e:
                error_count += 1
                self.message_user(
                    request,
                    f"Error starting graph building for {doc.title}: {e}",
                    level=messages.ERROR,
                )

        if success_count:
            self.message_user(
                request,
                f"Genealogy graph building started for {success_count} documents. "
                f"This will create Person, Relationship, and Partnership records.",
            )
        if error_count:
            self.message_user(
                request,
                f"{error_count} documents could not be processed.",
                level=messages.WARNING,
            )

    build_genealogy_graph.short_description = (  # type: ignore
        "Build genealogy graph (Person/Relationship/Partnership)"
    )

    def persist_extracted_entities(self, request, queryset):
        """Admin action: Persist extracted entities (events) to database records"""
        from genealogy.tasks import persist_extracted_entities as persist_task

        success_count = 0
        error_count = 0

        for doc in queryset:
            # Check if document has extracted entities
            chunks_with_events = doc.text_chunks.filter(
                entities_extracted=True,
                extracted_events__isnull=False,
            ).exclude(extracted_events=[]).count()

            if chunks_with_events == 0:
                error_count += 1
                self.message_user(
                    request,
                    f"Document {doc.title} has no chunks with extracted events. Run LLM extraction first.",
                    level=messages.WARNING,
                )
                continue

            try:
                # Start persistence task
                task = persist_task.delay(str(doc.id))
                success_count += 1
                logger.info(f"Started persist_extracted_entities task {task.id} for document {doc}")

            except Exception as e:
                error_count += 1
                self.message_user(
                    request,
                    f"Error starting entity persistence for {doc.title}: {e}",
                    level=messages.ERROR,
                )

        if success_count:
            self.message_user(
                request,
                f"Entity persistence started for {success_count} documents. "
                f"This will create Event records linked to Person records.",
            )
        if error_count:
            self.message_user(
                request,
                f"{error_count} documents could not be processed.",
                level=messages.WARNING,
            )

    persist_extracted_entities.short_description = (  # type: ignore
        "Persist extracted entities (create Event records)"
    )

    def extract_genealogy_data(self, request, queryset):
        """Admin action: Start LLM-based entity extraction from chunks for selected documents"""
        success_count = 0
        error_count = 0

        for doc in queryset:
            # Check if document has chunks in genealogy sections ready for extraction
            genealogy_section_types = ['DESCENDANT_GENEALOGY', 'KWARTIERSTATEN']
            genealogy_sections = doc.book_sections.filter(section_type__in=genealogy_section_types)

            if not genealogy_sections.exists():
                error_count += 1
                self.message_user(
                    request,
                    f"Document {doc.title} has no genealogy sections (DESCENDANT_GENEALOGY or KWARTIERSTATEN). Define book sections first.",
                    level=messages.WARNING,
                )
                continue

            # Check if there are chunks in those sections
            has_chunks = False
            for section in genealogy_sections:
                if doc.text_chunks.filter(
                    start_page__gte=section.start_page,
                    start_page__lte=section.end_page
                ).exists():
                    has_chunks = True
                    break

            if not has_chunks:
                error_count += 1
                self.message_user(
                    request,
                    f"Document {doc.title} has no chunks in genealogy sections. Run text chunking first.",
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
