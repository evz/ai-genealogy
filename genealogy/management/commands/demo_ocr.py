"""
Django management command to demonstrate OCR pipeline with sample documents.

Usage:
    python manage.py demo_ocr
    python manage.py demo_ocr --clear  # Clear previous demo data first
"""

import os
from pathlib import Path

from django.core.files import File
from django.core.management.base import BaseCommand
from django.conf import settings

from genealogy.models import Document, DocumentPage
from genealogy.tasks import process_page_ocr


class Command(BaseCommand):
    help = "Demonstrate OCR pipeline with sample genealogy documents"

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear previous demo data before running",
        )
        parser.add_argument(
            "--sync",
            action="store_true", 
            help="Run OCR synchronously instead of using Celery tasks",
        )

    def handle(self, **options):
        if options["clear"]:
            self._clear_demo_data()

        self.stdout.write("🔍 OCR Pipeline Demo")
        self.stdout.write("=" * 50)

        # Get sample files
        samples_dir = Path(settings.BASE_DIR) / "samples"
        sample_files = [
            ("025.pdf", "Sample genealogy document - single page"),
            ("032.pdf", "Sample genealogy document - multi-page"),
        ]

        if not samples_dir.exists():
            self.stdout.write(
                self.style.ERROR("❌ Samples directory not found at: %s") % samples_dir
            )
            return

        # Process each sample file
        for filename, description in sample_files:
            file_path = samples_dir / filename
            if not file_path.exists():
                self.stdout.write(
                    self.style.WARNING("⚠️ Sample file not found: %s") % filename
                )
                continue

            self.stdout.write(f"\n📄 Processing: {filename}")
            self.stdout.write(f"   Description: {description}")

            # Create document
            document = self._create_demo_document(file_path, description)
            self.stdout.write(f"   ✅ Created document: {document.title}")

            # Create and process pages
            pages_created = self._create_pages_for_document(document, file_path)
            self.stdout.write(f"   📄 Created {pages_created} page(s)")

            # Process OCR
            if options["sync"]:
                self._process_ocr_sync(document)
            else:
                self._process_ocr_async(document)

        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("✅ Demo complete!")
        self.stdout.write(
            "📱 View results at: http://localhost:8000/admin/genealogy/document/"
        )

    def _clear_demo_data(self):
        """Remove any existing demo data"""
        demo_docs = Document.objects.filter(title__startswith="Demo:")
        count = demo_docs.count()
        if count > 0:
            demo_docs.delete()
            self.stdout.write(f"🧹 Cleared {count} previous demo documents")

    def _create_demo_document(self, file_path: Path, description: str) -> Document:
        """Create a document for demo purposes"""
        title = f"Demo: {file_path.stem} - {description}"
        
        document = Document.objects.create(
            title=title,
            languages="eng",  # Default to English for demo
        )
        return document

    def _create_pages_for_document(self, document: Document, file_path: Path) -> int:
        """Create document pages from the PDF file"""
        with open(file_path, "rb") as f:
            django_file = File(f, name=file_path.name)
            
            # For demo, treat each PDF as a single page
            # In reality, the admin interface would handle multi-page PDFs
            page = DocumentPage.objects.create(
                document=document,
                page_number=1,
                image_file=django_file,
                original_filename=file_path.name,
            )
            
        return 1

    def _process_ocr_sync(self, document: Document):
        """Process OCR synchronously for immediate results"""
        self.stdout.write("   🔄 Processing OCR (synchronous)...")
        
        for page in document.pages.all():
            try:
                page.validate_for_ocr()
                
                # Import here to avoid import issues
                from genealogy.ocr_processor import OCRProcessor
                
                processor = OCRProcessor()
                file_path = page.image_file.path
                
                text, confidence, rotation = processor.process_file(file_path)
                
                page.ocr_text = text
                page.ocr_confidence = confidence
                page.rotation_applied = rotation
                page.ocr_completed = True
                page.save()
                
                self.stdout.write(
                    f"   ✅ OCR complete - {confidence:.1f}% confidence, "
                    f"{len(text)} characters extracted"
                )
                
                # Show first 100 characters of extracted text
                preview = text[:100].replace("\n", " ").strip()
                if len(text) > 100:
                    preview += "..."
                self.stdout.write(f"   📝 Preview: {preview}")
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"   ❌ OCR failed for page {page.page_number}: {e}")
                )

    def _process_ocr_async(self, document: Document):
        """Process OCR using Celery tasks"""
        self.stdout.write("   🔄 Queuing OCR tasks (asynchronous)...")
        
        task_count = 0
        for page in document.pages.all():
            try:
                page.validate_for_ocr()
                task = process_page_ocr.delay(str(page.id))
                task_count += 1
                self.stdout.write(f"   📋 Queued OCR task {task.id} for page {page.page_number}")
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"   ❌ Failed to queue OCR for page {page.page_number}: {e}")
                )
        
        if task_count > 0:
            self.stdout.write(
                f"   ⏱️ {task_count} OCR task(s) queued. "
                "Check the admin interface to see results as they complete."
            )