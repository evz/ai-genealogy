import uuid

from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone
from pgvector.django import IvfflatIndex, VectorField

from .fields import CommaSeparatedArrayField


class Document(models.Model):
    """Source document containing genealogical information"""

    LANGUAGE_CHOICES = [
        ("eng", "English"),
        ("nld", "Dutch"),
        ("eng+nld", "English + Dutch"),
    ]

    DATE_FORMAT_CHOICES = [
        ("DMY", "Day-Month-Year (European: 15.3.1850)"),
        ("MDY", "Month-Day-Year (US: 3/15/1850)"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    languages = models.CharField(
        max_length=10,
        choices=LANGUAGE_CHOICES,
        default="eng+nld",
        help_text="Languages for OCR processing",
    )
    date_format = models.CharField(
        max_length=3,
        choices=DATE_FORMAT_CHOICES,
        default="DMY",
        help_text="Expected date format in this document for parsing ambiguous dates",
    )
    upload_date = models.DateTimeField(default=timezone.now)
    ocr_completed = models.BooleanField(default=False)
    extraction_completed = models.BooleanField(default=False)

    # Model configuration for processing
    llm_model_used = models.CharField(max_length=100, blank=True, help_text="LLM model used for entity extraction")
    embedding_model_used = models.CharField(
        max_length=100, blank=True, help_text="Embedding model used for RAG processing"
    )

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

    def can_process_ocr(self):
        """Check if document has pages ready for OCR processing"""
        return bool(self.pages.exists() and not self.ocr_completed and self.pages.filter(ocr_completed=False).exists())

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
                for page in self.pages.filter(ocr_completed=True).order_by("page_number")
                if page.ocr_text.strip()
            ]
        )


class BookSection(models.Model):
    """
    Defines a section of a book with specific processing requirements.

    Allows flexible configuration of how different page ranges should be processed
    during chunking and entity extraction.
    """

    SECTION_TYPES = [
        ("FRONT_MATTER", "Front Matter (no processing)"),
        ("DESCENDANT_GENEALOGY", "Descendant Genealogy (main processing)"),
        ("KWARTIERSTATEN", "Kwartierstaten/Ancestor Tables"),
        ("APPENDIX_NARRATIVE", "Appendix Narrative (no processing)"),
        ("GLOSSARY", "Glossary (no processing)"),
        ("INDEX", "Index (future: 6-column processing)"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="book_sections",
        help_text="The document this section belongs to"
    )

    # Section identification
    title = models.CharField(
        max_length=200,
        help_text="Descriptive title for this section (e.g., 'Main Genealogy', 'Index')"
    )
    section_type = models.CharField(
        max_length=30,
        choices=SECTION_TYPES,
        help_text="Type of content and processing to apply"
    )

    # Page range
    start_page = models.PositiveIntegerField(
        help_text="First page number of this section (inclusive)"
    )
    end_page = models.PositiveIntegerField(
        help_text="Last page number of this section (inclusive)"
    )

    # Optional notes
    notes = models.TextField(
        blank=True,
        help_text="Optional notes about this section (content description, OCR issues, etc.)"
    )

    # Ordering
    sequence = models.PositiveIntegerField(
        default=0,
        help_text="Display order (auto-set based on start_page if not specified)"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "genealogy_booksection"
        ordering = ["document", "sequence", "start_page"]
        unique_together = [["document", "start_page"]]  # No overlapping sections
        indexes = [
            models.Index(fields=["document", "start_page", "end_page"]),
        ]

    def __str__(self):
        return f"{self.document.title}: {self.title} (pp. {self.start_page}-{self.end_page})"

    def save(self, *args, **kwargs):
        # Auto-set sequence based on start_page if not explicitly set
        if self.sequence == 0:
            self.sequence = self.start_page
        super().save(*args, **kwargs)

    def clean(self):
        """Validate that start_page <= end_page and no overlap with other sections"""
        if self.start_page > self.end_page:
            raise ValidationError("start_page must be <= end_page")

        # Check for overlapping sections in the same document
        overlapping = BookSection.objects.filter(
            document=self.document
        ).exclude(id=self.id).filter(
            models.Q(start_page__lte=self.end_page, end_page__gte=self.start_page)
        )

        if overlapping.exists():
            overlapping_section = overlapping.first()
            raise ValidationError(
                f"This section overlaps with '{overlapping_section.title}' "
                f"(pp. {overlapping_section.start_page}-{overlapping_section.end_page})"
            )

    @classmethod
    def get_section_for_page(cls, document, page_number):
        """
        Get the BookSection that contains the given page number.

        Args:
            document: Document instance
            page_number: Page number to look up

        Returns:
            BookSection instance or None if no section defined for this page
        """
        return cls.objects.filter(
            document=document,
            start_page__lte=page_number,
            end_page__gte=page_number
        ).first()


class DocumentPage(models.Model):
    """Individual page/image within a document"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="pages")
    page_number = models.PositiveIntegerField()
    image_file = models.FileField(upload_to="document_pages/")

    # OCR processing status
    ocr_completed = models.BooleanField(default=False)
    ocr_text = models.TextField(blank=True, help_text="Extracted text from OCR")
    ocr_confidence = models.FloatField(null=True, blank=True, help_text="OCR confidence score 0-100")

    # Image processing metadata
    rotation_applied = models.FloatField(default=0.0, help_text="Rotation correction applied in degrees")
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
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)

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
    """
    A person identified by their genealogical ID.

    This is the canonical entity for a person - one genealogical ID = one Person.
    No clustering or merging needed since genealogical IDs are unique identifiers.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    genealogical_id = models.CharField(
        max_length=50,
        unique=True,
        db_index=True,
        help_text="Format: II.3.a (generation.family_group.individual)"
    )
    given_names = models.CharField(max_length=200)
    surname = models.CharField(max_length=200)
    generation = models.IntegerField(null=True, blank=True)

    # Links back to source material
    source_documents = models.ManyToManyField('Document', related_name='people')
    source_chunks = models.ManyToManyField('TextChunk', related_name='people')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['genealogical_id']
        indexes = [
            models.Index(fields=['genealogical_id']),
            models.Index(fields=['surname', 'given_names']),
        ]

    @property
    def full_name(self):
        return f"{self.given_names} {self.surname}".strip()

    def __str__(self):
        return f"{self.full_name} ({self.genealogical_id})"


class Partnership(models.Model):
    """Partnership (marriage, etc.) between two people"""

    PARTNERSHIP_TYPES = [
        ('MARRIAGE', 'Marriage'),
        ('DOMESTIC_PARTNERSHIP', 'Domestic Partnership'),
        ('UNKNOWN', 'Unknown'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    partner1 = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='partnerships_as_partner1'
    )
    partner2 = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='partnerships_as_partner2'
    )
    partnership_type = models.CharField(
        max_length=30,
        choices=PARTNERSHIP_TYPES,
        default='MARRIAGE'
    )

    source_documents = models.ManyToManyField('Document', related_name='partnerships')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                check=~models.Q(partner1=models.F('partner2')),
                name='partners_different'
            )
        ]
        indexes = [
            models.Index(fields=['partner1']),
            models.Index(fields=['partner2']),
        ]

    def __str__(self):
        return f"{self.partner1.full_name} & {self.partner2.full_name}"


class Relationship(models.Model):
    """Parent-child relationship between two people"""

    RELATIONSHIP_TYPES = [
        ('BIOLOGICAL', 'Biological'),
        ('ADOPTED', 'Adopted'),
        ('STEP', 'Step'),
        ('FOSTER', 'Foster'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    parent = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='children_relationships'
    )
    child = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='parent_relationships'
    )
    relationship_type = models.CharField(
        max_length=20,
        choices=RELATIONSHIP_TYPES,
        default='BIOLOGICAL'
    )

    source_documents = models.ManyToManyField('Document', related_name='relationships')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['parent', 'child', 'relationship_type']
        indexes = [
            models.Index(fields=['parent']),
            models.Index(fields=['child']),
        ]

    def __str__(self):
        return f"{self.parent.full_name} → {self.child.full_name}"


class Event(models.Model):
    """An event (birth, death, marriage, etc.) associated with a person"""

    EVENT_TYPES = [
        ('BIRT', 'Birth'),
        ('DEAT', 'Death'),
        ('BAPT', 'Baptism'),
        ('BURI', 'Burial'),
        ('MARR', 'Marriage'),
        ('DIVR', 'Divorce'),
        ('OCCU', 'Occupation'),
        ('RESI', 'Residence'),
        ('EDUC', 'Education'),
        ('IMMI', 'Immigration'),
        ('EMIG', 'Emigration'),
        ('OTHER', 'Other'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='events', null=True, blank=True)
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    date = models.DateField(null=True, blank=True, help_text="Parsed date")
    date_original = models.CharField(max_length=200, blank=True, help_text="Original date string from source")
    date_approximate = models.BooleanField(default=False, help_text="Whether the date is approximate/estimated")
    place = models.CharField(max_length=500, blank=True, null=True)
    description = models.TextField(blank=True)

    source_chunk = models.ForeignKey(
        'TextChunk',
        on_delete=models.SET_NULL,
        null=True,
        related_name='events'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['person', 'event_type', 'date']
        indexes = [
            models.Index(fields=['person', 'event_type']),
        ]

    def __str__(self):
        return f"{self.person.full_name} - {self.event_type}"


class TextChunk(models.Model):
    """Text chunk extracted from a document with genealogical anchors"""

    CHUNK_TYPES = [
        ("generation_header", "Generation Header"),
        ("family_group_header", "Family Group Header"),
        ("individual_entry", "Individual Entry"),
        ("biographical_text", "Biographical Text"),
        ("source_citation", "Source Citation"),
        ("narrative_context", "Narrative Context"),
        ("info_box", "Info Box"),
        ("image", "Image"),
        ("image_caption", "Image Caption"),
        ("table", "Table"),
        ("unknown", "Unknown"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="text_chunks")

    # Content
    text_content = models.TextField(help_text="The actual text content of this chunk")
    chunk_type = models.CharField(max_length=30, choices=CHUNK_TYPES, default="unknown")

    # Position information
    start_page = models.PositiveIntegerField(help_text="First page number this chunk appears on")
    end_page = models.PositiveIntegerField(help_text="Last page number this chunk appears on")
    sequence_number = models.PositiveIntegerField(help_text="Order within document")

    # Genealogical anchors (stored in corrected/canonical form)
    generation_number = models.PositiveIntegerField(
        null=True, blank=True, help_text="Generation number (I=1, II=2, etc.)"
    )
    generation_header = models.CharField(
        max_length=100,
        blank=True,
        help_text="Generation header text if this chunk contains one",
    )
    family_groups = CommaSeparatedArrayField(
        models.CharField(max_length=100),
        default=list,
        blank=True,
        help_text="Family group headers found in this chunk " "(Enter comma-separated values: II.9. Children of...)",
    )

    # Relationships between chunks
    related_genealogy_entry = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        help_text="For narrative chunks: the genealogy entry they provide context for",
    )

    # Processing status
    entities_extracted = models.BooleanField(
        default=False, help_text="Whether LLM has extracted entities into JSON fields"
    )
    entities_persisted = models.BooleanField(
        default=False, help_text="Whether Event/Occupation records have been created from extracted JSON"
    )
    manually_reviewed = models.BooleanField(
        default=False,
        help_text="Whether this chunk has been manually reviewed and curated",
    )
    extraction_method = models.CharField(
        max_length=20,
        choices=[
            ("regex", "Regex patterns"),
            ("neural_network", "Neural network (NER)"),
            ("hybrid", "Neural network with regex fallback"),
        ],
        default="regex",
        help_text="Method used to extract genealogical anchors",
    )

    # RAG + RRF fields for hybrid search
    embedding = VectorField(
        dimensions=1024,
        null=True, blank=True,
        help_text="Vector embedding for semantic search"
    )
    dm_codes = ArrayField(
        models.CharField(max_length=10),
        default=list, blank=True,
        help_text="Daitch-Mokotoff phonetic codes for surname matching"
    )

    # Structured extraction fields (temporary staging for entity resolution)
    extracted_people = ArrayField(
        models.CharField(max_length=150),
        default=list, blank=True,
        help_text="List of person names extracted from this chunk"
    )
    extracted_relationships = models.JSONField(
        default=list, blank=True,
        help_text="Relationship triples: [{\"person1\": \"...\", \"relationship_type\": \"parent|child|spouse\", \"person2\": \"...\"}]"
    )
    extracted_events = models.JSONField(
        default=list, blank=True,
        help_text="Events: [{\"person\": \"...\", \"event_type\": \"BIRT|DEAT|MARR|etc\", \"date\": \"...\", \"place\": \"...\"}]"
    )

    # Subject tracking for individual entries
    subject = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        help_text="The primary subject/person of this chunk (for INDIVIDUAL_ENTRY chunks)"
    )
    genealogical_identifier = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        db_index=True,
        help_text="Unique genealogical identifier (e.g., 'II.2.a' = generation.family_group.individual_marker)"
    )

    # Link to the primary Person for this chunk (for INDIVIDUAL_ENTRY chunks)
    primary_person = models.ForeignKey(
        'Person',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='primary_chunks',
        help_text="The Person for the subject of this chunk (only for INDIVIDUAL_ENTRY chunks)"
    )

    # Anchor positioning for chunk expansion
    doc_id = models.CharField(
        max_length=100, blank=True,
        help_text="Document identifier for chunk grouping"
    )
    chunk_no = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Sequential chunk number within doc_id"
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["document", "sequence_number"]
        unique_together = ["document", "sequence_number"]
        indexes = [
            # RAG + RRF indexes for hybrid search
            IvfflatIndex(
                name="textchunk_embedding_ivfflat",
                fields=["embedding"],
                lists=100,
                opclasses=["vector_cosine_ops"]
            ),
            GinIndex(
                fields=["text_content"],
                name="textchunk_content_gin_trgm",
                opclasses=["gin_trgm_ops"]
            ),
            GinIndex(
                fields=["dm_codes"],
                name="textchunk_dm_codes_gin"
            ),
            models.Index(
                fields=["doc_id", "chunk_no"],
                name="textchunk_doc_chunk_idx"
            ),
        ]

    def __str__(self):
        chunk_preview = self.text_content[:50] + "..." if len(self.text_content) > 50 else self.text_content
        return f"{self.document.title} - Chunk {self.sequence_number}: {chunk_preview}"


class Conversation(models.Model):
    """A chat conversation with the genealogy assistant"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Session-based for anonymous users (future: add user FK for auth)
    session_key = models.CharField(max_length=40, null=True, blank=True, db_index=True)

    # Optional: filter search to specific documents
    document_filter = models.ManyToManyField(
        Document,
        blank=True,
        related_name='conversations',
        help_text="Limit search to these documents (empty = search all)"
    )

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.title or f'Conversation {self.id}'


class Message(models.Model):
    """A single message in a conversation"""

    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    # Metadata for retrieval (assistant messages only)
    retrieved_chunks = models.JSONField(default=list, blank=True)
    retrieval_metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"
