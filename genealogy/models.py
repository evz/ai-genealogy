import uuid

from django.db import models
from django.utils import timezone


class Document(models.Model):
    """Source document containing genealogical information"""

    LANGUAGE_CHOICES = [
        ("eng", "English"),
        ("nld", "Dutch"),
        ("eng+nld", "English + Dutch"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    languages = models.CharField(
        max_length=10,
        choices=LANGUAGE_CHOICES,
        default="eng+nld",
        help_text="Languages for OCR processing",
    )
    upload_date = models.DateTimeField(default=timezone.now)
    ocr_completed = models.BooleanField(default=False)
    extraction_completed = models.BooleanField(default=False)

    def __str__(self):
        return self.title

    @property
    def page_count(self):
        return self.pages.count()

    def update_ocr_status(self):
        """Update document OCR status based on all pages"""
        if not self.pages.exists():
            return

        # Check if all pages are OCR completed
        total_pages = self.pages.count()
        completed_pages = self.pages.filter(ocr_completed=True).count()

        if total_pages > 0 and completed_pages == total_pages:
            self.ocr_completed = True
            self.save(update_fields=["ocr_completed"])

    def start_genealogy_extraction(self):
        """Start genealogy data extraction for this document"""
        if not self.ocr_completed:
            raise ValueError("OCR must be completed before extraction")

        if self.extraction_completed:
            return None

        # TODO: Trigger Celery extraction task

        # For now, just mark as started
        return f"extraction-task-{self.id}"

    def can_process_ocr(self):
        """Check if document has pages ready for OCR processing"""
        return bool(
            self.pages.exists()
            and not self.ocr_completed
            and self.pages.filter(ocr_completed=False).exists()
        )

    def can_extract_genealogy(self):
        """Check if document is ready for genealogy extraction"""
        return bool(self.ocr_completed and not self.extraction_completed)

    @property
    def ocr_progress(self):
        """Get OCR progress for multi-page documents"""
        total_pages = self.pages.count()
        if total_pages == 0:
            return None

        completed_pages = self.pages.filter(ocr_completed=True).count()
        return {
            "completed": completed_pages,
            "total": total_pages,
            "percentage": (completed_pages / total_pages) * 100,
        }

    def get_combined_ocr_text(self):
        """Get combined OCR text from all pages"""
        if not self.pages.exists():
            return ""

        return "\n\n".join(
            [
                f"=== Page {page.page_number} ===\n{page.ocr_text}"
                for page in self.pages.filter(ocr_completed=True).order_by(
                    "page_number"
                )
                if page.ocr_text.strip()
            ]
        )


class DocumentPage(models.Model):
    """Individual page/image within a document"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="pages"
    )
    page_number = models.PositiveIntegerField()
    image_file = models.FileField(upload_to="document_pages/")

    # OCR processing status
    ocr_completed = models.BooleanField(default=False)
    ocr_text = models.TextField(blank=True, help_text="Extracted text from OCR")
    ocr_confidence = models.FloatField(
        null=True, blank=True, help_text="OCR confidence score 0-100"
    )

    # Image processing metadata
    rotation_applied = models.FloatField(
        default=0.0, help_text="Rotation correction applied in degrees"
    )
    original_filename = models.CharField(max_length=255, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ["document", "page_number"]
        ordering = ["document", "page_number"]

    def __str__(self):
        return f"{self.document.title} - Page {self.page_number}"

    @property
    def filename(self):
        return self.image_file.name.split("/")[-1] if self.image_file else ""

    def validate_for_ocr(self):
        """Validate that page is ready for OCR processing"""
        if self.ocr_completed:
            raise ValueError("OCR already completed for this page")

        if not self.image_file:
            raise ValueError("No image file attached to process")

    def can_process_ocr(self):
        """Check if page is ready for OCR processing"""
        return bool(self.image_file and not self.ocr_completed)


class Place(models.Model):
    """Geographic location"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    locality = models.CharField(max_length=255, blank=True)  # City/town
    region = models.CharField(max_length=255, blank=True)  # State/province
    country = models.CharField(max_length=255, blank=True)
    latitude = models.DecimalField(
        max_digits=9, decimal_places=6, null=True, blank=True
    )
    longitude = models.DecimalField(
        max_digits=9, decimal_places=6, null=True, blank=True
    )

    class Meta:
        unique_together = ["name", "locality", "region", "country"]

    def __str__(self):
        parts = [self.name]
        if self.locality:
            parts.append(self.locality)
        if self.region:
            parts.append(self.region)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)


class Person(models.Model):
    """Individual person in genealogical records"""

    GENDER_CHOICES = [
        ("M", "Male"),
        ("F", "Female"),
        ("N", "Non-binary"),
        ("U", "Unknown"),
        ("O", "Other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    given_names = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    maiden_name = models.CharField(
        max_length=255, blank=True, help_text="Previous surname if changed"
    )
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default="U")

    # Life events
    birth_date = models.DateField(null=True, blank=True)
    birth_date_estimated = models.BooleanField(default=False)
    birth_place = models.ForeignKey(
        Place, on_delete=models.SET_NULL, null=True, blank=True, related_name="births"
    )

    death_date = models.DateField(null=True, blank=True)
    death_date_estimated = models.BooleanField(default=False)
    death_place = models.ForeignKey(
        Place, on_delete=models.SET_NULL, null=True, blank=True, related_name="deaths"
    )

    # Genealogical identifiers (from Dutch family books)
    genealogical_id = models.CharField(
        max_length=50, blank=True, help_text="e.g., II.1.a"
    )

    # Source tracking
    source_documents = models.ManyToManyField(
        Document, blank=True, related_name="persons"
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["surname", "given_names"]

    def __str__(self):
        name_parts = [self.given_names, self.surname]
        if self.maiden_name:
            name_parts.insert(-1, f"({self.maiden_name})")
        return " ".join(name_parts)

    @property
    def full_name(self):
        return str(self)


class Partnership(models.Model):
    """Partnership/relationship between people (marriage, civil union, etc.)"""

    PARTNERSHIP_TYPES = [
        ("MARRIAGE", "Marriage"),
        ("CIVIL_UNION", "Civil Union"),
        ("DOMESTIC_PARTNERSHIP", "Domestic Partnership"),
        ("RELATIONSHIP", "Relationship"),
        ("OTHER", "Other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    partners = models.ManyToManyField(Person, related_name="partnerships")
    partnership_type = models.CharField(
        max_length=20, choices=PARTNERSHIP_TYPES, default="MARRIAGE"
    )

    # Partnership start details
    start_date = models.DateField(null=True, blank=True)
    start_date_estimated = models.BooleanField(default=False)
    start_place = models.ForeignKey(
        Place,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="partnership_starts",
    )

    # Partnership end details
    end_date = models.DateField(null=True, blank=True)
    end_date_estimated = models.BooleanField(default=False)
    end_reason = models.CharField(
        max_length=50, blank=True, help_text="divorce, death, separation, etc."
    )

    # Source tracking
    source_documents = models.ManyToManyField(
        Document, blank=True, related_name="partnerships"
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        partner_names = [partner.full_name for partner in self.partners.all()]
        return f"{' & '.join(partner_names)} ({self.get_partnership_type_display()})"


class Event(models.Model):
    """Genealogical events (baptism, burial, etc.)"""

    EVENT_TYPES = [
        ("BIRT", "Birth"),
        ("DEAT", "Death"),
        ("MARR", "Marriage"),
        ("DIVR", "Divorce"),
        ("BAPT", "Baptism"),
        ("BURI", "Burial"),
        ("RESI", "Residence"),
        ("OCCU", "Occupation"),
        ("EDUC", "Education"),
        ("IMMI", "Immigration"),
        ("EMIG", "Emigration"),
        ("OTHER", "Other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    event_type = models.CharField(max_length=5, choices=EVENT_TYPES)
    person = models.ForeignKey(
        Person, on_delete=models.CASCADE, related_name="events", null=True, blank=True
    )
    partnership = models.ForeignKey(
        Partnership,
        on_delete=models.CASCADE,
        related_name="events",
        null=True,
        blank=True,
    )

    date = models.DateField(null=True, blank=True)
    date_estimated = models.BooleanField(default=False)
    place = models.ForeignKey(
        Place, on_delete=models.SET_NULL, null=True, blank=True, related_name="events"
    )

    description = models.TextField(blank=True)

    # Source tracking
    source_documents = models.ManyToManyField(
        Document, blank=True, related_name="events"
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["date", "event_type"]

    def __str__(self):
        subject = self.person.full_name if self.person else str(self.partnership)
        return f"{self.get_event_type_display()}: {subject}"


class ParentChildRelationship(models.Model):
    """Relationship between child and parent(s)"""

    RELATIONSHIP_TYPES = [
        ("BIOLOGICAL", "Biological"),
        ("ADOPTED", "Adopted"),
        ("STEP", "Step"),
        ("FOSTER", "Foster"),
        ("GUARDIAN", "Guardian"),
        ("UNKNOWN", "Unknown"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    child = models.ForeignKey(
        Person, on_delete=models.CASCADE, related_name="parent_relationships"
    )
    parent = models.ForeignKey(
        Person, on_delete=models.CASCADE, related_name="child_relationships"
    )
    relationship_type = models.CharField(
        max_length=15, choices=RELATIONSHIP_TYPES, default="BIOLOGICAL"
    )

    # Optional: link to partnership if child is from a specific partnership
    partnership = models.ForeignKey(
        Partnership,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )

    # Source tracking
    source_documents = models.ManyToManyField(
        Document, blank=True, related_name="parent_child_relationships"
    )

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ["child", "parent"]

    def __str__(self):
        return (
            f"{self.child.full_name} - "
            f"{self.get_relationship_type_display().lower()} child of "
            f"{self.parent.full_name}"
        )
