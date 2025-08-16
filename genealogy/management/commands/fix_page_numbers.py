import re
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand
from django.db import transaction

if TYPE_CHECKING:
    from uuid import UUID

from genealogy.models import DocumentPage


class Command(BaseCommand):
    help = "Fix page numbers for existing DocumentPages based on filenames"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be changed without making actual changes",
        )
        parser.add_argument(
            "--document-id",
            type=str,
            help="Fix page numbers for a specific document ID only",
        )

    def handle(self, **options):
        dry_run = options["dry_run"]
        document_id = options["document_id"]

        if dry_run:
            self.stdout.write(
                self.style.WARNING("DRY RUN MODE - No changes will be made")
            )

        # Get pages to process
        queryset = DocumentPage.objects.all()
        if document_id:
            queryset = queryset.filter(document_id=document_id)

        pages = queryset.select_related("document").order_by("document", "page_number")

        if not pages.exists():
            self.stdout.write(self.style.WARNING("No pages found to process"))
            return

        updated_count = 0
        error_count = 0

        # Group pages by document to handle each document separately
        documents_to_update: dict[UUID, list[tuple[DocumentPage, int]]] = {}

        for page in pages:
            extracted_page_num = self.extract_page_number_from_filename(
                page.original_filename
            )

            if extracted_page_num is None:
                self.stdout.write(
                    self.style.WARNING(
                        f"Could not extract page number from: {page.original_filename}"
                    )
                )
                error_count += 1
                continue

            if page.page_number != extracted_page_num:
                if page.document_id not in documents_to_update:
                    documents_to_update[page.document_id] = []
                documents_to_update[page.document_id].append((page, extracted_page_num))

        # Process each document separately to avoid unique constraint conflicts
        for document_id, page_updates in documents_to_update.items():
            with transaction.atomic():
                try:
                    # First, set all pages to high numbers (10000+) to avoid conflicts
                    for i, (page, new_page_num) in enumerate(page_updates):
                        if dry_run:
                            self.stdout.write(
                                f"Would update: {page.original_filename} "
                                f"page {page.page_number} → {new_page_num} "
                                f"(Document: {page.document.title})"
                            )
                        else:
                            # Temporarily set to high number to avoid constraint
                            # conflicts
                            temp_page_num = 10000 + i
                            page.page_number = temp_page_num
                            page.save(update_fields=["page_number"])

                    # Then, set all pages to their correct numbers
                    if not dry_run:
                        for page, new_page_num in page_updates:
                            page.page_number = new_page_num
                            page.save(update_fields=["page_number"])
                            self.stdout.write(
                                f"Updated: {page.original_filename} to page "
                                f"{new_page_num} (Document: {page.document.title})"
                            )

                    updated_count += len(page_updates)

                    if dry_run:
                        # Rollback the transaction in dry run mode
                        transaction.set_rollback(True)

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f"Error processing document {document_id}: {e}"
                        )
                    )
                    error_count += len(page_updates)

        # Summary
        self.stdout.write("\n" + "=" * 50)
        if dry_run:
            self.stdout.write(self.style.SUCCESS("DRY RUN COMPLETE"))
            self.stdout.write(f"Would update: {updated_count} pages")
        else:
            self.stdout.write(self.style.SUCCESS("OPERATION COMPLETE"))
            self.stdout.write(f"Updated: {updated_count} pages")

        if error_count > 0:
            self.stdout.write(self.style.WARNING(f"Errors: {error_count} pages"))

    def extract_page_number_from_filename(self, filename):
        """
        Extract page number from filename.
        Handles patterns like: 014_fwlK4fY.pdf → 14
        """
        # Remove extension and get the base name
        base_name = filename.rsplit(".", 1)[0] if "." in filename else filename

        # Look for 3-digit number at the beginning (Django upload pattern)
        match = re.search(r"^(\d{3})_", base_name)
        if match:
            return int(match.group(1))

        # Look for 3-digit number at the end (original naming pattern)
        match = re.search(r"(\d{3})$", base_name)
        if match:
            return int(match.group(1))

        # Fallback: look for any number at the beginning or end
        match = re.search(r"^(\d+)_", base_name)
        if match:
            return int(match.group(1))

        match = re.search(r"(\d+)$", base_name)
        if match:
            return int(match.group(1))

        return None
