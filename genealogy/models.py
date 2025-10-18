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
    """Individual person in genealogical records"""

    GENDER_CHOICES = [
        ("M", "Male"),
        ("F", "Female"),
        ("N", "Non-binary"),
        ("U", "Unknown"),
        ("O", "Other"),
    ]

    ENTITY_TYPE_CHOICES = [
        ('EXTRACTED', 'Original Extracted Entity'),
        ('CANONICAL', 'Merged Canonical Entity'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    given_names = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    maiden_name = models.CharField(max_length=255, blank=True, help_text="Previous surname if changed")
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default="U")

    # Genealogical identifiers (from Dutch family books)
    genealogical_id = models.CharField(max_length=50, blank=True, help_text="e.g., II.1.a")
    generation = models.PositiveIntegerField(null=True, blank=True, help_text="Generation number (I=1, II=2, etc.)")

    # Entity resolution tracking
    entity_type = models.CharField(
        max_length=20,
        choices=ENTITY_TYPE_CHOICES,
        default='EXTRACTED',
        help_text="Whether this is an original extraction or a merged canonical entity"
    )

    # For EXTRACTED entities that have been merged: points to their canonical version
    # For CANONICAL entities: should be NULL
    canonical_entity = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='source_entities',
        help_text="The canonical entity this record was merged into (NULL if not merged)"
    )

    # Many-to-many relationship for tracking merge composition via EntityMerge through model
    merged_from = models.ManyToManyField(
        'self',
        through='EntityMerge',
        symmetrical=False,
        related_name='merged_into',
        blank=True,
        help_text="For CANONICAL entities: the source entities that were merged to create this"
    )

    # Source tracking
    source_documents = models.ManyToManyField(Document, blank=True, related_name="persons")
    source_chunks = models.ManyToManyField('TextChunk', blank=True, related_name="persons")

    # Duplicate detection - self-referential many-to-many for tracking potential duplicates
    potential_duplicates = models.ManyToManyField(
        'self',
        through='PotentialDuplicate',
        symmetrical=False,
        related_name='duplicate_of',
        blank=True
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


class PotentialDuplicate(models.Model):
    """Through model for tracking potential duplicate persons"""

    REVIEW_STATUS_CHOICES = [
        ('PENDING', 'Pending Review'),
        ('CONFIRMED', 'Confirmed Duplicate - Needs Merge'),
        ('REJECTED', 'Not a Duplicate'),
        ('MERGED', 'Already Merged'),
    ]

    person1 = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='duplicate_links_from')
    person2 = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='duplicate_links_to')

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
        unique_together = ['person1', 'person2']
        ordering = ['-confidence_score', 'review_status']

    def __str__(self):
        return f"{self.person1.full_name} ≈ {self.person2.full_name} ({self.confidence_score:.0f}%)"


class EntityMerge(models.Model):
    """
    Through model tracking the composition of merged canonical entities.

    Records the provenance of entity resolution: which source entities were
    merged together to create a canonical entity, along with merge metadata.

    Each record represents one source entity being merged into the canonical entity,
    with pairwise confidence scores and detailed provenance tracking.
    """

    # The canonical (merged) entity that was created
    canonical_entity = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='merge_sources',
        help_text="The canonical entity created from merging"
    )

    # The source entity that was merged into the canonical entity
    source_entity = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='merge_targets',
        help_text="The source entity that was merged"
    )

    # Merge metadata - pairwise between canonical and this source
    confidence_score = models.FloatField(
        help_text="Pairwise confidence score for merging this source into canonical (0-100)"
    )

    # Store all pairwise similarities within the cluster for full auditability
    pairwise_similarities = models.JSONField(
        default=dict,
        blank=True,
        help_text="Dict of {other_source_entity_id: confidence_score} for all pairs in cluster"
    )

    merge_algorithm = models.CharField(
        max_length=50,
        default='manual',
        help_text="Algorithm or method used for merge (e.g., 'graph_clustering_v1', 'manual')"
    )

    merge_reason = models.JSONField(
        default=dict,
        blank=True,
        help_text="Detailed merge provenance: matching signals, cluster info, atomic similarities, etc."
    )

    # Attribution
    merged_by = models.CharField(
        max_length=100,
        default='AUTO',
        help_text="Who/what performed the merge: username or 'AUTO'"
    )

    merged_at = models.DateTimeField(
        default=timezone.now,
        help_text="When this merge occurred"
    )

    class Meta:
        unique_together = ['canonical_entity', 'source_entity']
        ordering = ['-merged_at']
        indexes = [
            models.Index(fields=['canonical_entity', '-merged_at']),
            models.Index(fields=['source_entity']),
        ]

    def __str__(self):
        return f"{self.source_entity.full_name} → {self.canonical_entity.full_name} ({self.confidence_score:.0f}%, {self.merged_at.strftime('%Y-%m-%d')})"


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

    # Source tracking
    source_documents = models.ManyToManyField(Document, blank=True, related_name="partnerships")

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
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name="events", null=True, blank=True)
    partnership = models.ForeignKey(
        Partnership,
        on_delete=models.CASCADE,
        related_name="events",
        null=True,
        blank=True,
    )

    date = models.DateField(null=True, blank=True)
    date_estimated = models.BooleanField(default=False)
    place = models.ForeignKey(Place, on_delete=models.SET_NULL, null=True, blank=True, related_name="events")

    description = models.TextField(blank=True)

    # Source tracking
    source_documents = models.ManyToManyField(Document, blank=True, related_name="events")

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
    child = models.ForeignKey(Person, on_delete=models.CASCADE, related_name="parent_relationships")
    parent = models.ForeignKey(Person, on_delete=models.CASCADE, related_name="child_relationships")
    relationship_type = models.CharField(max_length=15, choices=RELATIONSHIP_TYPES, default="BIOLOGICAL")

    # Optional: link to partnership if child is from a specific partnership
    partnership = models.ForeignKey(
        Partnership,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )

    # Source tracking
    source_documents = models.ManyToManyField(Document, blank=True, related_name="parent_child_relationships")

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ["child", "parent"]

    def __str__(self):
        return (
            f"{self.child.full_name} - "
            f"{self.get_relationship_type_display().lower()} child of "
            f"{self.parent.full_name}"
        )


class TextChunk(models.Model):
    """Text chunk extracted from a document with genealogical anchors"""

    CHUNK_TYPES = [
        ("HEADER", "Generation Header"),
        ("GENEALOGY_ENTRY", "Dense Biographical Entry"),
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
