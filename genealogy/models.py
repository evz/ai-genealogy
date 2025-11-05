import uuid

from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from django.utils import timezone
from pgvector.django import VectorField, IvfflatIndex

from .fields import CommaSeparatedArrayField


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
        from django.core.exceptions import ValidationError

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


class PersonMention(models.Model):
    """
    Immutable person mention extracted from source text.

    Represents a single extraction of a person from the source documents.
    Never modified after creation. Multiple mentions can be resolved to
    a single Identity through the MentionToIdentity mapping.
    """

    GENDER_CHOICES = [
        ("M", "Male"),
        ("F", "Female"),
        ("N", "Non-binary"),
        ("U", "Unknown"),
        ("O", "Other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Extracted attributes (immutable)
    given_names = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    maiden_name = models.CharField(max_length=255, blank=True, help_text="Previous surname if changed")
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default="U")

    # Genealogical identifiers (from Dutch family books)
    genealogical_id = models.CharField(max_length=50, blank=True, null=True, help_text="e.g., II.1.a")
    generation = models.PositiveIntegerField(null=True, blank=True, help_text="Generation number (I=1, II=2, etc.)")

    # Source tracking (immutable)
    source_documents = models.ManyToManyField(Document, blank=True, related_name="person_mentions")
    source_chunks = models.ManyToManyField('TextChunk', blank=True, related_name="person_mentions")

    # Duplicate detection - self-referential many-to-many for tracking potential duplicates
    potential_duplicates = models.ManyToManyField(
        'self',
        through='PotentialDuplicate',
        symmetrical=False,
        related_name='duplicate_of',
        blank=True
    )

    # Quality flags
    is_extraction_error = models.BooleanField(
        default=False,
        help_text="Mark true if this mention was incorrectly extracted (not a real person)"
    )

    # Never modified after creation
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["surname", "given_names"]
        db_table = "genealogy_personmention"  # Keep DB table name consistent

    def __str__(self):
        name_parts = [self.given_names, self.surname]
        if self.maiden_name:
            name_parts.insert(-1, f"({self.maiden_name})")
        return " ".join(name_parts)

    @property
    def full_name(self):
        return str(self)


class PotentialDuplicate(models.Model):
    """Through model for tracking potential duplicate person mentions"""

    REVIEW_STATUS_CHOICES = [
        ('PENDING', 'Pending Review'),
        ('CONFIRMED', 'Confirmed Duplicate - Needs Merge'),
        ('REJECTED', 'Not a Duplicate'),
        ('MERGED', 'Already Merged'),
    ]

    mention1 = models.ForeignKey(PersonMention, on_delete=models.CASCADE, related_name='duplicate_links_from')
    mention2 = models.ForeignKey(PersonMention, on_delete=models.CASCADE, related_name='duplicate_links_to')

    # Matching metadata
    confidence_score = models.FloatField(
        help_text="Matching confidence score (0-100). Higher = more likely duplicate"
    )
    match_reasons = models.JSONField(
        default=list,
        blank=True,
        help_text="List of matching signals: ['same_generation', 'same_parents', 'similar_name', etc.]"
    )

    # Review tracking
    review_status = models.CharField(
        max_length=20,
        choices=REVIEW_STATUS_CHOICES,
        default='PENDING'
    )
    reviewed_by = models.CharField(max_length=100, blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    review_notes = models.TextField(blank=True)

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ['mention1', 'mention2']
        ordering = ['-confidence_score', 'review_status']

    def __str__(self):
        return f"{self.mention1.full_name} ≈ {self.mention2.full_name} ({self.confidence_score:.0f}%)"


class Identity(models.Model):
    """
    Canonical resolved person entity.

    Represents the "real person" that one or more PersonMentions refer to.
    Soft-deleted when absorbed into another Identity during merges.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    display_name = models.CharField(max_length=500, help_text="Display name for this identity")
    notes = models.TextField(blank=True, help_text="Curator notes about this identity")

    # Genealogical identifier (from Dutch family books)
    genealogical_identifier = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        db_index=True,
        help_text="Unique genealogical identifier (e.g., 'II.2.a' = generation.family_group.individual_marker)"
    )

    # Soft-delete for reversibility
    is_deleted = models.BooleanField(default=False, help_text="True if this identity was absorbed into another")

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Identities"
        ordering = ["display_name"]

    def __str__(self):
        return self.display_name


class MentionToIdentity(models.Model):
    """
    Mapping layer - the ONLY mutable part of the system.

    Maps PersonMentions to Identities. This is where merges/unmerges happen.
    All other data (mentions, events, relationships) remains immutable.
    """

    mention = models.OneToOneField(
        PersonMention,
        primary_key=True,
        on_delete=models.CASCADE,
        help_text="The person mention being mapped"
    )
    identity = models.ForeignKey(
        Identity,
        on_delete=models.CASCADE,
        related_name="mention_mappings",
        help_text="The identity this mention maps to"
    )

    # Audit trail
    mapped_at = models.DateTimeField(auto_now=True, help_text="When this mapping was last updated")
    mapped_by = models.CharField(max_length=100, default="AUTO", help_text="Who/what updated this mapping")

    class Meta:
        indexes = [
            models.Index(fields=['identity']),
        ]

    def __str__(self):
        return f"{self.mention.full_name} → {self.identity.display_name}"


class MergeEvent(models.Model):
    """
    Audit log of all merge/unmerge operations.

    Records full transaction details for reversibility. Never delete from this table.
    """

    EVENT_TYPE_CHOICES = [
        ('merge', 'Merge'),
        ('unmerge', 'Unmerge'),
        ('split', 'Split'),
    ]

    id = models.BigAutoField(primary_key=True)
    event_type = models.CharField(max_length=20, choices=EVENT_TYPE_CHOICES)

    # Full transaction payload (JSON)
    # For merge: {survivor_identity, absorbed_identities, mention_moves: [{mention_id, from, to}]}
    # For unmerge: same structure but reversed
    payload = models.JSONField(help_text="Full transaction details for reversibility")

    # Attribution
    performed_by = models.CharField(max_length=100, default="AUTO")
    performed_at = models.DateTimeField(default=timezone.now)

    # If this event reverses another event
    reversed_event = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reversals",
        help_text="The event this undoes (for unmerge operations)"
    )

    class Meta:
        ordering = ['-performed_at']
        indexes = [
            models.Index(fields=['-performed_at']),
            models.Index(fields=['event_type']),
        ]

    def __str__(self):
        return f"{self.get_event_type_display()} at {self.performed_at.strftime('%Y-%m-%d %H:%M')} by {self.performed_by}"


class PartnershipMention(models.Model):
    """
    Immutable partnership mention extracted from source text.

    Represents partnerships between PersonMentions (marriage, civil union, etc.)
    Never modified after creation.
    """

    PARTNERSHIP_TYPES = [
        ("MARRIAGE", "Marriage"),
        ("CIVIL_UNION", "Civil Union"),
        ("DOMESTIC_PARTNERSHIP", "Domestic Partnership"),
        ("RELATIONSHIP", "Relationship"),
        ("OTHER", "Other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    partners = models.ManyToManyField(PersonMention, related_name="partnerships")
    partnership_type = models.CharField(max_length=20, choices=PARTNERSHIP_TYPES, default="MARRIAGE")

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
    end_reason = models.CharField(max_length=50, blank=True, help_text="divorce, death, separation, etc.")

    # Source tracking (immutable)
    source_documents = models.ManyToManyField(Document, blank=True, related_name="partnerships")

    # Never modified after creation
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "genealogy_partnershipmention"

    def __str__(self):
        partner_names = [partner.full_name for partner in self.partners.all()]
        return f"{' & '.join(partner_names)} ({self.get_partnership_type_display()})"


class Event(models.Model):
    """
    Immutable genealogical event extracted from source text.

    Events are attached to PersonMentions or PartnershipMentions and
    never modified after creation.
    """

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
    mention = models.ForeignKey(PersonMention, on_delete=models.CASCADE, related_name="events", null=True, blank=True)
    partnership = models.ForeignKey(
        PartnershipMention,
        on_delete=models.CASCADE,
        related_name="events",
        null=True,
        blank=True,
    )

    date = models.DateField(null=True, blank=True)
    date_estimated = models.BooleanField(default=False)
    place = models.ForeignKey(Place, on_delete=models.SET_NULL, null=True, blank=True, related_name="events")

    description = models.TextField(blank=True)

    # Source tracking (immutable)
    source_documents = models.ManyToManyField(Document, blank=True, related_name="events")

    # Never modified after creation
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["date", "event_type"]

    def __str__(self):
        subject = self.mention.full_name if self.mention else str(self.partnership)
        return f"{self.get_event_type_display()}: {subject}"


class RelationshipMention(models.Model):
    """
    Immutable parent-child relationship extracted from source text.

    Represents relationships between PersonMentions.
    Never modified after creation.
    """

    RELATIONSHIP_TYPES = [
        ("BIOLOGICAL", "Biological"),
        ("ADOPTED", "Adopted"),
        ("STEP", "Step"),
        ("FOSTER", "Foster"),
        ("GUARDIAN", "Guardian"),
        ("UNKNOWN", "Unknown"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    child_mention = models.ForeignKey(PersonMention, on_delete=models.CASCADE, related_name="parent_relationships")
    parent_mention = models.ForeignKey(PersonMention, on_delete=models.CASCADE, related_name="child_relationships")
    relationship_type = models.CharField(max_length=15, choices=RELATIONSHIP_TYPES, default="BIOLOGICAL")

    # Optional: link to partnership if child is from a specific partnership
    partnership = models.ForeignKey(
        PartnershipMention,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )

    # Source tracking (immutable)
    source_documents = models.ManyToManyField(Document, blank=True, related_name="parent_child_relationships")

    # Never modified after creation
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ["child_mention", "parent_mention"]
        db_table = "genealogy_relationshipmention"

    def __str__(self):
        return (
            f"{self.child_mention.full_name} - "
            f"{self.get_relationship_type_display().lower()} child of "
            f"{self.parent_mention.full_name}"
        )


class TextChunk(models.Model):
    """Text chunk extracted from a document with genealogical anchors"""

    CHUNK_TYPES = [
        ("HEADER", "Generation Header"),
        ("GENEALOGY_ENTRY", "Dense Biographical Entry"),
        ("CITATION", "Source Citation"),
        ("NARRATIVE", "Narrative/Context"),
        ("NARRATIVE_CONTEXT", "Related Context/Story"),
        ("CONTENT", "General Genealogy Content"),
        ("INDEX", "Index/Reference"),
        ("KWARTIERSTATEN_HEADER", "Kwartierstaten Section Header"),
        ("KWARTIERSTATEN", "Kwartierstaten Content"),
        ("APPENDIX_HEADER", "Appendix/Register Header"),
        ("APPENDIX", "Appendix/Register Content"),
        ("OTHER", "Other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="text_chunks")

    # Content
    text_content = models.TextField(help_text="The actual text content of this chunk")
    chunk_type = models.CharField(max_length=30, choices=CHUNK_TYPES, default="CONTENT")

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
        default=False, help_text="Whether entities have been extracted from this chunk"
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

    # Link to the primary PersonMention for this chunk (for INDIVIDUAL_ENTRY chunks)
    primary_person_mention = models.ForeignKey(
        'PersonMention',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='primary_chunks',
        help_text="The PersonMention for the subject of this chunk (only for INDIVIDUAL_ENTRY chunks)"
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
