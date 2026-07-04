# Genealogy Data Model Simplification Plan

## Overview

This plan removes the complex "reversible provenance" architecture (PersonMention, Identity, MentionToIdentity, etc.) and reverts to a simple Person/Event/Relationship model. With reliable OCR and genealogical IDs, we can use the genealogical ID as the canonical unique identifier, eliminating the need for clustering and mention-to-identity mapping.

## Problem Statement

The current architecture was designed to handle:
- Unreliable OCR leading to duplicate person extractions
- Uncertainty about which mentions refer to the same person
- Need to merge/cluster mentions into canonical identities

However, with improved OCR quality and reliable genealogical identifiers, this complexity is unnecessary. The genealogical ID directly identifies the person - no clustering needed.

## Goals

1. **Simplify data model**: Remove PersonMention/Identity/MentionToIdentity complexity
2. **Use genealogical IDs as source of truth**: One genealogical ID = one Person
3. **Remove clustering**: No need to find duplicates or merge mentions
4. **Deterministic extraction**: Same document always produces same graph
5. **Focus on valuable data**: Only track people with genealogical IDs
6. **Celery-based workflow**: Trigger extraction via admin actions

## Architecture Changes

### Before (Complex)
```
TextChunk → PersonMention → MentionToIdentity → Identity
           → RelationshipMention
           → PartnershipMention

Clustering → PotentialDuplicate → Merge → Update MentionToIdentity
```

### After (Simple)
```
TextChunk → Person (via genealogical_id)
         → Relationship
         → Partnership
         → Event
```

## Implementation Phases

### Phase 1: Database Model Changes

#### Files to Modify
- `genealogy/models.py`

#### Changes

**REMOVE these models entirely**:
- `PersonMention`
- `Identity`
- `MentionToIdentity`
- `PartnershipMention`
- `RelationshipMention`
- `PotentialDuplicate`

**ADD/RESTORE simplified models**:

```python
class Person(models.Model):
    """A person identified by their genealogical ID"""
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
        # Ensure we don't create duplicate partnerships (order-independent)
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


class Event(models.Model):
    """An event (birth, death, marriage, etc.) associated with a person"""
    EVENT_TYPES = [
        ('BIRTH', 'Birth'),
        ('DEATH', 'Death'),
        ('BAPTISM', 'Baptism'),
        ('MARRIAGE', 'Marriage'),
        ('OCCUPATION', 'Occupation'),
        ('RESIDENCE', 'Residence'),
        ('OTHER', 'Other'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='events')
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    date = models.CharField(max_length=200, blank=True)  # Free text initially
    place = models.CharField(max_length=500, blank=True)
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
```

**UPDATE TextChunk model**:
```python
class TextChunk(models.Model):
    # ... existing fields ...

    # Add link to primary person (the subject of this chunk)
    primary_person = models.ForeignKey(
        Person,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='primary_chunks',
        help_text="The main person this chunk is about"
    )
```

#### Migration
Create migration to:
1. Drop old tables: `PersonMention`, `Identity`, `MentionToIdentity`, `PartnershipMention`, `RelationshipMention`, `PotentialDuplicate`
2. Create new tables: `Person`, `Relationship`, `Partnership`, `Event`
3. Add `primary_person` FK to `TextChunk`

**Note**: This is a destructive migration. We'll need to re-run extraction after applying it.

### Phase 2: Simplify Chunking

#### Files to Modify
- `genealogy/chunking/persistence.py`

#### Changes

**REMOVE**:
- All imports of PersonMention, Identity, MentionToIdentity, PartnershipMention, RelationshipMention
- `_create_person_mention_with_identity()` helper function
- All person/relationship creation logic in `save_chunks_to_db()`

**KEEP**:
- TextChunk creation with metadata fields:
  - `subject` (the person's name from the chunk)
  - `genealogical_identifier` (e.g., "II.3.a")
  - `family_groups` (list of family group headers)
  - `extracted_people`, `extracted_relationships`, `extracted_events` (for later LLM enrichment)

**Result**: `save_chunks_to_db()` should ONLY create TextChunk records, no Person records.

### Phase 3: Create `build_genealogy_graph` Task

#### Files to Create
- `genealogy/tasks/build_genealogy_graph.py`
- `genealogy/tests/test_build_genealogy_graph.py`

#### Task Implementation

```python
"""Build genealogy graph from genealogical identifiers"""
import logging
from typing import Dict, List, Optional, Set, Tuple

from celery import shared_task
from django.db import transaction

from genealogy.models import Document, Person, Partnership, Relationship, TextChunk
from genealogy.utils import parse_family_group_header, parse_name

logger = logging.getLogger(__name__)


@shared_task
def build_genealogy_graph(document_id: str) -> Dict[str, int]:
    """
    Build Person, Relationship, and Partnership records from genealogical IDs.

    This task:
    1. Creates Person records from chunks with genealogical_identifier
    2. Creates Relationship records based on family structure
    3. Creates Partnership records for parents

    Args:
        document_id: UUID of the document to process

    Returns:
        dict with counts of created/updated records
    """
    document = Document.objects.get(id=document_id)

    logger.info(f"Building genealogy graph for document: {document.title}")

    # Get all chunks with genealogical identifiers
    chunks = TextChunk.objects.filter(
        document=document,
        genealogical_identifier__isnull=False,
        subject__isnull=False,
    ).order_by('genealogical_identifier')

    if not chunks.exists():
        logger.warning(f"No chunks with genealogical identifiers found for document {document_id}")
        return {
            'people_created': 0,
            'relationships_created': 0,
            'partnerships_created': 0,
        }

    stats = {
        'people_created': 0,
        'relationships_created': 0,
        'partnerships_created': 0,
    }

    with transaction.atomic():
        # Phase 1: Create/update Person records
        logger.info(f"Phase 1: Creating Person records from {chunks.count()} chunks")
        person_index = _create_people_from_chunks(chunks, document, stats)

        # Phase 2: Create Relationship records (parent-child)
        logger.info(f"Phase 2: Creating Relationship records")
        _create_relationships(chunks, person_index, document, stats)

        # Phase 3: Create Partnership records (marriages)
        logger.info(f"Phase 3: Creating Partnership records")
        _create_partnerships(chunks, person_index, document, stats)

    logger.info(
        f"Genealogy graph built: "
        f"{stats['people_created']} people, "
        f"{stats['relationships_created']} relationships, "
        f"{stats['partnerships_created']} partnerships"
    )

    return stats


def _create_people_from_chunks(
    chunks,
    document: Document,
    stats: Dict[str, int]
) -> Dict[str, Person]:
    """
    Create Person records from chunks with genealogical identifiers.

    Returns:
        Dict mapping genealogical_id to Person instance
    """
    person_index = {}

    for chunk in chunks:
        gen_id = chunk.genealogical_identifier

        # Parse name
        given_names, surname = parse_name(chunk.subject)

        # Get or create Person
        person, created = Person.objects.get_or_create(
            genealogical_id=gen_id,
            defaults={
                'given_names': given_names,
                'surname': surname,
                'generation': chunk.generation_number,
            }
        )

        if created:
            stats['people_created'] += 1
            logger.debug(f"Created Person: {person.full_name} ({gen_id})")

        # Link to source material
        person.source_documents.add(document)
        person.source_chunks.add(chunk)

        # Link chunk back to person
        chunk.primary_person = person
        chunk.save(update_fields=['primary_person'])

        person_index[gen_id] = person

    return person_index


def _create_relationships(
    chunks,
    person_index: Dict[str, Person],
    document: Document,
    stats: Dict[str, int]
):
    """Create parent-child Relationship records."""

    for chunk in chunks:
        child_id = chunk.genealogical_identifier
        child = person_index.get(child_id)

        if not child or not chunk.family_groups:
            continue

        # Parse family group header to get parent genealogical_id
        parent_names, parent_gen_id = parse_family_group_header(chunk.family_groups)

        if not parent_gen_id:
            continue

        # Look up parent by genealogical_id
        parent = person_index.get(parent_gen_id)

        if not parent:
            logger.warning(
                f"Parent {parent_gen_id} not found for child {child_id} "
                f"(mentioned in family group header)"
            )
            continue

        # Create relationship
        relationship, created = Relationship.objects.get_or_create(
            parent=parent,
            child=child,
            relationship_type='BIOLOGICAL',
        )

        if created:
            relationship.source_documents.add(document)
            stats['relationships_created'] += 1
            logger.debug(f"Created Relationship: {parent.full_name} → {child.full_name}")


def _create_partnerships(
    chunks,
    person_index: Dict[str, Person],
    document: Document,
    stats: Dict[str, int]
):
    """Create Partnership records for parents mentioned in family groups."""

    # Group chunks by family_group to avoid creating duplicate partnerships
    family_groups_seen = set()

    for chunk in chunks:
        if not chunk.family_groups:
            continue

        family_group = chunk.family_groups[0]

        # Skip if we've already processed this family group
        if family_group in family_groups_seen:
            continue
        family_groups_seen.add(family_group)

        # Parse family group header
        parent_names, parent1_gen_id = parse_family_group_header([family_group])

        if len(parent_names) < 2 or not parent1_gen_id:
            continue

        # Look up parent1 by genealogical_id
        parent1 = person_index.get(parent1_gen_id)

        if not parent1:
            logger.warning(f"Parent1 {parent1_gen_id} not found for partnership")
            continue

        # Find parent2 by looking at children of parent1 and checking their family groups
        # Parent2 should be mentioned in the same family group as parent1
        parent2_name = parent_names[1]
        parent2_given, parent2_surname = parse_name(parent2_name)

        # Try to find parent2 in person_index by matching name and checking if they share children
        parent2 = None
        for gen_id, person in person_index.items():
            if (person.given_names == parent2_given and
                person.surname == parent2_surname and
                person.generation == parent1.generation):
                parent2 = person
                break

        # If parent2 not found by exact match, create them
        if not parent2:
            # Parent2 doesn't have a genealogical_id, so we skip creating them
            # (only create people with genealogical IDs)
            logger.debug(
                f"Parent2 '{parent2_name}' not found with genealogical_id "
                f"for partnership with {parent1.full_name}"
            )
            continue

        # Create partnership (handle order-independence)
        if parent1.id > parent2.id:
            parent1, parent2 = parent2, parent1

        partnership, created = Partnership.objects.get_or_create(
            partner1=parent1,
            partner2=parent2,
            partnership_type='MARRIAGE',
        )

        if created:
            partnership.source_documents.add(document)
            stats['partnerships_created'] += 1
            logger.debug(f"Created Partnership: {parent1.full_name} & {parent2.full_name}")
```

#### Admin Action

Add to `genealogy/admin/document.py`:

```python
from genealogy.tasks.build_genealogy_graph import build_genealogy_graph

@admin.action(description="Build genealogy graph from genealogical IDs")
def action_build_genealogy_graph(modeladmin, request, queryset):
    """Trigger genealogy graph building for selected documents"""
    for document in queryset:
        build_genealogy_graph.delay(str(document.id))
        messages.success(
            request,
            f"Started building genealogy graph for '{document.title}'"
        )

class DocumentAdmin(admin.ModelAdmin):
    actions = [action_build_genealogy_graph]
```

### Phase 4: Simplify Entity Extraction

#### Files to Modify
- `genealogy/tasks/extraction.py`

#### Files to DELETE
- `genealogy/management/commands/create_entities.py` (convert to task, then delete)

#### Changes

**REMOVE**:
- ALL person creation logic
- ALL family group parsing
- ALL clustering/merging logic

**KEEP/UPDATE**:
- Event extraction (birth, death, etc.)
- Occupation extraction
- Place extraction
- Link extracted entities to existing Person records by genealogical_id lookup

**Example**:
```python
@shared_task
def extract_entities(document_id: str) -> Dict[str, int]:
    """
    Extract events, occupations, and places from chunks.
    Link to existing Person records.
    """
    document = Document.objects.get(id=document_id)
    chunks = TextChunk.objects.filter(
        document=document,
        entities_extracted=False,
    )

    stats = {'events_created': 0}

    for chunk in chunks:
        # Only process chunks that have a primary_person
        if not chunk.primary_person:
            continue

        # Extract events using LLM
        events = llm_extract_events(chunk.text_content)

        for event_data in events:
            event, created = Event.objects.get_or_create(
                person=chunk.primary_person,
                event_type=event_data['type'],
                date=event_data.get('date', ''),
                place=event_data.get('place', ''),
                defaults={
                    'description': event_data.get('description', ''),
                    'source_chunk': chunk,
                }
            )

            if created:
                stats['events_created'] += 1

        chunk.entities_extracted = True
        chunk.save(update_fields=['entities_extracted'])

    return stats
```

### Phase 5: Cleanup - Remove Clustering Code

#### Files to DELETE

1. **Admin templates**:
   - `genealogy/templates/admin/genealogy/personmention/change_list.html`
   - Any other PersonMention/Identity admin templates

2. **Management commands** (DELETE ALL):
   - `genealogy/management/commands/cluster_mentions.py` (if exists)
   - `genealogy/management/commands/create_entities.py` (replaced by task)
   - `genealogy/management/commands/chunk_genealogy.py` (if not already a task)
   - Any other clustering-related commands

3. **Tasks**:
   - Any clustering tasks in `genealogy/tasks/`

4. **Tests**:
   - `genealogy/tests/test_clustering.py`
   - `genealogy/tests/test_clustering_graph.py`
   - Any other clustering-related tests

5. **Documentation** (mark as deprecated/obsolete):
   - References to PersonMention/Identity architecture
   - Clustering documentation

#### Files to MODIFY

1. **`genealogy/admin/`**:
   - Remove PersonMention, Identity, MentionToIdentity, PartnershipMention, RelationshipMention admin classes
   - Create simple admin classes for Person, Relationship, Partnership, Event
   - Remove identity grouping logic
   - Remove merge actions

2. **`genealogy/tests/test_chunking_persistence.py`**:
   - Remove: `test_creates_identity_for_person_mention`
   - Remove: `test_creates_parent_mentions_and_identity`
   - Remove: `test_creates_partnership_for_parents`
   - Remove: `test_creates_relationship_mentions`
   - Remove: `test_no_duplicates_when_siblings_processed`
   - Keep: Basic chunking tests that verify TextChunk creation

### Phase 6: Update Tests

#### Files to Create
- `genealogy/tests/test_build_genealogy_graph.py`

#### Test Cases

```python
@pytest.mark.django_db
class TestBuildGenealogyGraph:
    """Test build_genealogy_graph task"""

    def test_creates_person_from_genealogical_id(self):
        """Test Person is created from chunk with genealogical_identifier"""
        # Create chunk with genealogical_identifier
        # Run task
        # Assert Person created with correct genealogical_id

    def test_creates_parent_child_relationship(self):
        """Test Relationship created from family group header"""
        # Create parent chunk with gen_id "II.1.a"
        # Create child chunk with gen_id "III.5.a" and family_group referring to "II.1.a"
        # Run task
        # Assert Relationship exists: parent → child

    def test_creates_partnership_for_parents(self):
        """Test Partnership created from family group header"""
        # Create chunks for both parents with gen_ids
        # Create child chunk with family_group mentioning both parents
        # Run task
        # Assert Partnership exists between parents

    def test_idempotency(self):
        """Test running task twice doesn't create duplicates"""
        # Create chunks
        # Run task twice
        # Assert Person count unchanged after second run

    def test_links_chunks_to_people(self):
        """Test chunks are linked to their primary person"""
        # Create chunk with genealogical_identifier
        # Run task
        # Assert chunk.primary_person is set

    def test_multi_generation_family(self):
        """Test creating relationships across multiple generations"""
        # Create chunks for 3 generations
        # Run task
        # Assert all Relationships created correctly

    def test_siblings_share_parents(self):
        """Test siblings (same family group) share same parent relationships"""
        # Create parent chunk
        # Create 3 sibling chunks with same family_group
        # Run task
        # Assert all 3 siblings have Relationship to same parent
```

### Phase 7: Update Workflow Documentation

#### New Workflow

**Via Admin UI** (Primary interface):
1. Upload/create Document
2. Click "Build genealogy graph" admin action
3. Click "Extract entities" admin action

**Via Celery** (Programmatic/testing):
```python
from genealogy.tasks import chunk_document, build_genealogy_graph, extract_entities

# 1. Chunk document (creates TextChunk records with metadata)
chunk_document.delay(document_id)

# 2. Build genealogy graph (creates Person/Relationship/Partnership)
build_genealogy_graph.delay(document_id)

# 3. Extract events/occupations (enriches existing Person records)
extract_entities.delay(document_id)
```

## Implementation Order

### Step 1: Create Migration & Update Models
1. Create migration to drop old tables
2. Update `genealogy/models.py` with new simplified models
3. Run migration
4. Verify schema in database

### Step 2: Simplify Chunking
1. Update `genealogy/chunking/persistence.py`
2. Remove all entity creation logic
3. Update tests in `test_chunking_persistence.py`
4. Run tests to verify chunking still works

### Step 3: Build Graph Task
1. Create `genealogy/tasks/build_genealogy_graph.py`
2. Create `genealogy/tests/test_build_genealogy_graph.py`
3. Implement task logic
4. Run tests

### Step 4: Simplify Extraction
1. Update `genealogy/tasks/extraction.py`
2. Remove person creation logic
3. Focus on Event extraction only
4. Update tests

### Step 5: Admin Integration
1. Update `genealogy/admin/document.py` to add admin action
2. Create simple admin for Person/Relationship/Partnership/Event
3. Remove old PersonMention/Identity admin

### Step 6: Cleanup
1. Delete clustering code
2. Delete old tests
3. Delete old admin templates
4. Update documentation

### Step 7: Integration Testing
1. Upload test document
2. Run chunking
3. Run build_genealogy_graph
4. Run extract_entities
5. Verify Person/Relationship/Partnership/Event created correctly
6. Test RAG queries still work

## Benefits of This Approach

1. **Dramatically simpler**: Remove ~50% of codebase complexity
2. **Deterministic**: Same input always produces same output
3. **No duplicates**: Genealogical ID is unique key
4. **No merging needed**: No clustering, no manual review
5. **Easy to debug**: Clear separation of phases
6. **Admin-friendly**: Trigger via buttons
7. **Faster**: No expensive clustering algorithms
8. **Maintainable**: Easier for future developers to understand

## Risks & Mitigation

### Risk: People without genealogical IDs are lost
**Mitigation**: Accept this - they're rarely the focus of queries and still appear in chunk text for context.

### Risk: OCR errors in genealogical IDs create wrong people
**Mitigation**: With improved OCR, this should be rare. Can add manual correction UI if needed.

### Risk: Second parent without genealogical ID not tracked
**Mitigation**: Accept for now. Most important parent (with genealogical ID) is tracked. Can enhance later if needed.

## Success Criteria

- [ ] All old models removed (PersonMention, Identity, etc.)
- [ ] New simple models working (Person, Relationship, Partnership, Event)
- [ ] Chunking creates only TextChunk records
- [ ] build_genealogy_graph creates complete family tree
- [ ] No duplicate Person records for same genealogical_id
- [ ] Parent-child relationships correct
- [ ] Partnership records created
- [ ] All tests passing
- [ ] Admin actions work
- [ ] RAG queries still return correct results
- [ ] Codebase is simpler and easier to understand

## Implementation Status

### ✅ COMPLETED - Phase 1: Database Model Changes

**Date Completed**: 2025-11-11

**What was done**:
1. ✅ Created new simplified models in `genealogy/models.py`:
   - `Person` - One per genealogical_id (unique)
   - `Relationship` - Parent-child relationships
   - `Partnership` - Spousal relationships
   - `Event` - Updated to reference Person instead of PersonMention
2. ✅ Removed old complex models:
   - `PersonMention`, `Identity`, `MentionToIdentity` (deleted)
   - `PartnershipMention`, `RelationshipMention` (deleted)
   - `PotentialDuplicate`, `MergeEvent` (deleted)
3. ✅ Updated TextChunk model:
   - Renamed `primary_person_mention` → `primary_person` (FK to Person)
   - Added `entities_persisted` field (tracks if Events created from JSON)
   - Updated `entities_extracted` help text (now means "LLM extracted to JSON")
4. ✅ Created migration `0037_simplify_to_person_model.py`:
   - Fixed operation order (AlterUniqueTogether before RemoveField)
   - Migration applied successfully
5. ✅ Simplified chunking persistence (`genealogy/chunking/persistence.py`):
   - Removed all PersonMention/Identity/Relationship creation logic
   - `save_chunks_to_db()` now ONLY creates TextChunk records with metadata
   - No entity creation during chunking

**Files Modified**:
- `genealogy/models.py` - New models, removed old ones
- `genealogy/chunking/persistence.py` - Simplified to only save TextChunks
- `genealogy/tasks/chunking.py` - Removed PersonMention import
- `genealogy/admin/__init__.py` - Temporarily disabled old admin imports
- `genealogy/admin/document.py` - Removed PotentialDuplicate import
- `genealogy/migrations/0037_simplify_to_person_model.py` - Migration file
- Removed: `genealogy/models/book_section.py` (redundant)

**Database State**:
- Old tables dropped (genealogy_personmention, genealogy_identity, etc.)
- New tables created (genealogy_person, genealogy_relationship, genealogy_partnership)
- TextChunk.primary_person field exists (FK to Person, null=True)
- TextChunk.entities_persisted field added (default=False)
- **TextChunks retain all metadata**: genealogical_identifier, subject, family_groups, extracted_events, etc.

### ✅ COMPLETED - Phase 2: Create `build_genealogy_graph` Task

**Date Completed**: 2025-11-11

**What was done**:
1. ✅ Created `genealogy/tasks/build_genealogy_graph.py`:
   - Implemented `GenealogyGraphBuilder` class for clean, testable code
   - Creates Person records from genealogical_identifier
   - Creates Relationship records from family_groups
   - Creates Partnership records for married couples
   - Handles spouse ID minting (e.g., `II.3.a.spouse1`) for partners without genealogical IDs
   - Supports multiple marriages (spouse1, spouse2, etc.)
   - Idempotent - can be run multiple times safely
2. ✅ Added admin action to `genealogy/admin/document.py`:
   - "Build genealogy graph (Person/Relationship/Partnership)" action
   - Validates chunks exist before running
3. ✅ Created comprehensive tests in `genealogy/tests/test_build_genealogy_graph.py`:
   - 12 tests covering all scenarios
   - All tests passing
4. ✅ Updated `genealogy/tasks/__init__.py` to export new task
5. ✅ Fixed `genealogy/utils/family_parsing.py`:
   - Updated regex to handle colons after genealogical IDs: `(II.1.a):`
   - Added support for simple Roman numeral format: `(I):`
6. ✅ Fixed `genealogy/chunking/parser.py`:
   - Updated `extract_person_from_individual_entry()` regex pattern
   - Now handles: periods, commas, parenthetical prefixes like "(Misschien)", trailing refs like (III.1)
   - Fixed 8 chunks that had genealogical_identifier but no subject

**Real Data Validation** (Jan van Bulhuis Book):
- ✅ 284 chunks with genealogical_identifier processed
- ✅ 372 people created (284 + 88 minted spouses)
- ✅ 531 parent-child relationships created
- ✅ 89 partnerships created
- ✅ All partnerships verified to have shared children
- ✅ 70% of generation 12 can trace ancestry back to generation 2 (9/30 blocked by missing parent data in source)

**Files Modified/Created**:
- `genealogy/tasks/build_genealogy_graph.py` - New task implementation
- `genealogy/admin/document.py` - Added admin action
- `genealogy/tasks/__init__.py` - Export new task
- `genealogy/tests/test_build_genealogy_graph.py` - Comprehensive tests
- `genealogy/utils/family_parsing.py` - Fixed regex patterns
- `genealogy/chunking/parser.py` - Fixed subject extraction

### ✅ COMPLETED - Phase 2.1: Fix Chunking Bug - IX.5.a Misclassification

**Date Completed**: 2025-11-11

**Issue Found**:
IX.5.a (Thomas Frans van Zanten) was misclassified as `image_caption` instead of `individual_entry`, preventing it from getting a genealogical ID and being included in the family tree. This blocked 9 out of 30 generation 12 people from tracing ancestry back to generation 2.

**Root Cause Identified**:
In `genealogy/chunking_strategies/descendant_genealogy.py` lines 121-142, there's logic that tries to detect "text tokens that are part of image captions" by checking if tokens after an `image_caption` token are:
1. Vertically close (y-gap < 50 pixels), AND
2. Horizontally offset from baseline (x1 far from baseline)

**The Bug**:
Line 137 had an off-by-one error in the threshold check:
```python
if x1_diff < 20:  # BUG: should be <=
    break
```

For the IX.5.a chunk:
- Token 9 (individual entry) had x1=135
- Baseline x1 was 155 (from earlier text tokens)
- x1_diff = |135 - 155| = **exactly 20 pixels**
- Check: `if 20 < 20` → False, so didn't break
- Result: Token 9 incorrectly added to skip_indices and treated as image caption

**The Fix**:
Changed line 137 from `if x1_diff < 20:` to `if x1_diff <= 25:`
- Now tokens within 25 pixels of baseline are correctly recognized as main text
- Provides small buffer for slight OCR coordinate variations

**Next Steps**:
- ⏸️ **Re-run chunking** on Jan van Bulhuis Book to fix the IX.5.a chunk
- ⏸️ **Re-run build_genealogy_graph** to include IX.5.a in the family tree
- ⏸️ **Verify** that more generation 12 people can now trace back to generation 2

**Files Modified**:
- `genealogy/chunking_strategies/descendant_genealogy.py` (line 137)

**Validation Data**:
- OCR correctly labeled token 9 as `element_type: text`
- Our chunking code incorrectly merged it with the image caption
- Fix prevents this merging and allows proper individual entry classification

### ✅ COMPLETED - Phase 3: Simplify Entity Extraction

**Date Completed**: 2025-11-13

**What was done**:
1. ✅ Created `genealogy/tasks/persist_entities.py`:
   - Reads `extracted_events` JSON from TextChunk records
   - Creates Event records linked to Person (via genealogical_id lookup)
   - Handles field ordering issues from LLM with `correct_event_fields()` heuristic function
   - Fuzzy name matching to link events to correct Person records
   - Idempotent - can be run multiple times safely
2. ✅ Updated `genealogy/prompts/extraction.py`:
   - Removed examples to prevent LLM hallucination (copying events from examples)
   - Updated to 5-field format: person|event_type|date|place|description
   - Clarified field usage (OCCU events put occupation in description, not place)
3. ✅ Fixed `genealogy/models.py`:
   - Updated `Event.EVENT_TYPES` to include GEDCOM codes (BIRT, DEAT, MARR, OCCU, RESI, etc.)
   - Event now links to Person (not PersonMention)
4. ✅ Created `genealogy/admin/event.py`:
   - Simple admin with list_display showing event_type, person_link, date, place, description
   - Links to Person and source TextChunk for easy navigation
5. ✅ Updated `genealogy/admin/textchunk.py`:
   - Shows created events with description field
   - Links to Event admin for details
6. ✅ Created comprehensive tests in `genealogy/tests/test_persist_entities.py`:
   - 9 tests covering all scenarios (basic persistence, occupations, fuzzy matching, error handling)
   - All tests passing

**Files Modified/Created**:
- `genealogy/tasks/persist_entities.py` - New entity persistence task
- `genealogy/prompts/extraction.py` - Fixed hallucination, updated format
- `genealogy/models.py` - Updated Event.EVENT_TYPES
- `genealogy/admin/event.py` - New Event admin
- `genealogy/admin/textchunk.py` - Shows created events
- `genealogy/admin/__init__.py` - Enabled EventAdmin
- `genealogy/tests/test_persist_entities.py` - Comprehensive tests

**Key Features**:
- **Field correction**: `correct_event_fields()` function fixes common LLM mistakes:
  - Dates in place field → swapped to date field
  - Places in description field → swapped to place field (except OCCU events)
  - Special handling for RESI and OCCU events
- **No re-extraction needed**: Field corrections run during persistence, allowing iteration without re-running expensive LLM extraction
- **Event types**: Full support for GEDCOM codes (BIRT, DEAT, MARR, OCCU, RESI, EDUC, IMMI, EMIG, etc.)

### ✅ COMPLETED - Phase 4: Cleanup

**Date Completed**: 2025-11-13

**What was done**:
1. ✅ **Deleted old admin files** (6 files):
   - `genealogy/admin/person_mention.py` - Old PersonMention admin
   - `genealogy/admin/identity.py` - Old Identity admin
   - `genealogy/admin/duplicate_clusters.py` - Clustering admin
   - `genealogy/admin/merge_logic.py` - Merge logic admin
   - `genealogy/admin/partnership.py` - Old PartnershipMention admin
   - `genealogy/admin/relationship.py` - Old RelationshipMention admin

2. ✅ **Deleted clustering code** (5 items):
   - `genealogy/clustering/` - Entire clustering directory
   - `genealogy/tests/test_clustering.py` - Clustering tests
   - `genealogy/tests/test_clustering_graph.py` - Clustering graph tests
   - `genealogy/management/commands/cluster_entities.py` - Clustering command
   - `genealogy/templates/admin/genealogy/potentialduplicate/` - Clustering templates

3. ✅ **Deleted old management commands** (2 files):
   - `genealogy/management/commands/create_entities.py` - Replaced by tasks
   - `genealogy/management/commands/backfill_person_mentions.py` - Old architecture

4. ✅ **Deleted old tests** (1 file):
   - `genealogy/tests/test_person_mention_lifecycle.py` - PersonMention lifecycle tests

5. ✅ **Cleaned up chunking/persistence.py**:
   - Removed all commented-out PersonMention/Identity code
   - Removed `_create_person_mention_with_identity_DISABLED()` function (50 lines)
   - Updated comments to reference new tasks (build_genealogy_graph, persist_entities)
   - Cleaned up imports

6. ✅ **Updated test file**:
   - `genealogy/tests/test_chunking_persistence.py` - Removed PersonMention import

7. ✅ **Created admin for all new models**:
   - ✅ `genealogy/admin/event.py` - Event admin with person links
   - ✅ `genealogy/admin/person.py` - Person admin (already existed)
   - ✅ `genealogy/admin/relationship_admin.py` - Relationship admin (already existed)
   - ✅ `genealogy/admin/partnership_admin.py` - Partnership admin (created new)

8. ✅ **Cleaned up admin/__init__.py**:
   - Removed all commented-out old admin imports
   - Added PartnershipAdmin to imports and __all__
   - Clean, organized import list

**Files Modified/Created**:
- `genealogy/admin/partnership_admin.py` - New Partnership admin
- `genealogy/admin/__init__.py` - Cleaned up comments, added PartnershipAdmin
- `genealogy/chunking/persistence.py` - Removed 50+ lines of commented code
- `genealogy/tests/test_chunking_persistence.py` - Removed old import

**Files Deleted**: 14 files total
- 6 old admin files
- 2 old management commands
- 1 old test file
- 1 clustering directory (multiple files)
- 4 individual test/command/template files

**Remaining Files Using Old Architecture** (Intentional - Future Phase):
These files support the **agentic workflow feature** which still uses the old Identity/PersonMention architecture. They will be migrated when we update the agent feature to use the new Person model:
- `genealogy/tests/test_query_genealogy_command.py` - Agent CLI tests
- `genealogy/services/genealogy_tools.py` - Agent tools (search, relationships)
- `genealogy/tests/test_agent_integration.py` - Agent integration tests
- `genealogy/tests/test_genealogy_tools.py` - Agent tools tests
- `genealogy/services/agent_executor.py` - Agent execution logic

**Result**: The codebase is now **significantly cleaner** with the complex PersonMention/Identity/clustering architecture completely removed. All new models (Person, Relationship, Partnership, Event) have working admin interfaces.

### ⏸️ PENDING - Phase 5: Create Tests

**Files to create**:
- `genealogy/tests/test_build_genealogy_graph.py`

**Test cases needed**:
1. `test_creates_person_from_genealogical_id` - Verify Person creation
2. `test_creates_parent_child_relationship` - Verify Relationship from family_groups
3. `test_creates_partnership_for_parents` - Verify Partnership creation
4. `test_idempotency` - Running twice doesn't duplicate
5. `test_links_chunks_to_people` - Verify chunk.primary_person set
6. `test_multi_generation_family` - Test across generations
7. `test_siblings_share_parents` - Siblings point to same parents

### ⏸️ PENDING - Phase 6: Integration Testing

**What to test**:
1. Upload test document
2. Run chunking task
3. Run build_genealogy_graph task
4. Verify Person/Relationship/Partnership created
5. Run extract_entities task (when implemented)
6. Verify Event records created
7. Test RAG queries still work

## Timeline Estimate

- ✅ Phase 1 (Models): 2-3 hours **[COMPLETED]**
- ✅ Phase 2 (Build graph task): 4-5 hours **[COMPLETED]**
- ✅ Phase 2.1 (Fix chunking bug): 2 hours **[COMPLETED]**
- ⏸️ Phase 3 (Entity extraction): 2 hours **[PENDING]**
- ⏸️ Phase 4 (Cleanup): 2 hours **[PENDING]**
- ⏸️ Phase 5 (Admin UI): 1 hour **[PENDING]**
- ⏸️ Phase 6 (Integration testing): 2-3 hours **[PENDING]**
- ⏸️ Phase 7 (Documentation): 1 hour **[PENDING]**

**Total: ~16-19 hours** | **Completed: ~14-16 hours** | **Remaining: ~2-3 hours**

## Current Status & How to Resume

### Where We Are Now (2025-11-11)

**Phase 2 is COMPLETE** - The `build_genealogy_graph` task is fully implemented, tested, and working on real data.

**Critical bug FIXED but NOT YET DEPLOYED**:
- Fixed chunking bug in `genealogy/chunking_strategies/descendant_genealogy.py` (line 137)
- Bug was causing individual entries to be misclassified as image captions
- Specifically blocked IX.5.a (Thomas Frans van Zanten) from being extracted
- **Fix NOT yet applied to database** - need to re-run chunking

### Immediate Next Steps

**Step 1: Re-run chunking on Jan van Bulhuis Book**
```bash
# In Django admin:
# 1. Go to Documents
# 2. Select "Jan van Bulhuis Book"
# 3. Click "Create text chunks" action
# This will re-chunk with the fixed logic
```

**Step 2: Re-run build_genealogy_graph**
```bash
# In Django admin:
# 1. Select "Jan van Bulhuis Book"
# 2. Click "Build genealogy graph" action
# This will create Person/Relationship/Partnership records with IX.5.a included
```

**Step 3: Verify the fix worked**
```python
# Check if IX.5.a now exists
from genealogy.models import Person
Person.objects.filter(genealogical_id="IX.5.a").exists()  # Should be True

# Check if generation 12 people can now trace back further
# Run the tree traversal script from previous session
```

### Known Source Data Issues (NOT code bugs)

These are typos/inconsistencies in the original book that we discovered:

1. **Systematic IX.x vs X.x confusion** (multiple typos in family group headers)

   a. **IX.6.c should be X.6.c**
   - **Impact**: Blocks XII.2.a (Greyson Woods) and XI.10.a/b/c (Kamp siblings) from tracing back
   - **Location**: Family group headers for XI.10 and XII.2
   - **Says**: "(IX.6.c)"
   - **Should say**: "(X.6.c)"
   - **Verification**: X.6.c exists (Thomas Gregory Kamp MSc, page 67)

   b. **IX.6.d should be X.6.d**
   - **Impact**: Blocks XI.11.a/b/c (Ringling siblings) from tracing back
   - **Location**: Family group header for XI.11
   - **Says**: "(IX.6.d)"
   - **Should say**: "(X.6.d)"
   - **Verification**: X.6.d exists (Nancy Suzanne Kamp BSc, page 67)

   c. **X.3.c should be X.4.c**
   - **Impact**: Blocks XI.6.a (Michelle van der Reijden) from tracing back
   - **Location**: Family group header for XI.6
   - **Says**: "Kinderen van Ewout van der Reijden (X.3.c):"
   - **Should say**: "(X.4.c):"
   - **Verification**: X.4.c exists (Ewout van der Reijden), X.3.c does not exist

   **Pattern**: The book has systematic numbering errors in generation IX/X family references

2. **XI.16.b (Jamie Nicole Hall) has no individual entry**
   - **Impact**: Blocks XII.10.a and XII.10.b (Abercrombie children) from tracing back
   - **Location**: Only mentioned in family group header on page 77
   - **Says**: "Children of Jamie Nicole Hall and Joshua Abercrombie (XI.16.b):"
   - **Issue**: No individual entry for XI.16.b exists anywhere in the book (checked pages 75-76, entries go from XI.30.b to XII.1.a)
   - **Workaround**: Would need to mint a Person record from family group header mention only

3. **Other potential issues** (mentioned in previous docs but now resolved)
   - IX.6.d and X.6.d: VERIFIED - not duplicates, X.6.d is correct, IX.6.d references are typos
   - X.3.c inconsistency: VERIFIED - X.3.c reference should be X.4.c

These should be handled as data corrections, not code fixes.

### Graph Quality Analysis (2025-11-12)

**Generation 12 Ancestry Tracing Results:**
- **Success rate**: 90% (27/30 people can trace back to generation 2)
- **Blocked people**: 3 (all due to source data issues above)
  - XII.10.a (Finley Jo Abercrombie) - parent XI.16.b missing from book
  - XII.10.b (Howard James Abercrombie) - parent XI.16.b missing from book
  - XII.2.a (Greyson Woods) - parent genealogical ID typo (IX.6.c → X.6.c)

**Generation 11 Ancestry Tracing Results:**
- **Success rate**: 79% (49/62 people can trace back to generation 2)
- **Blocked people**: 13 total
  - **7 minted spouses** (*.spouse1) - Expected behavior: partners without genealogical IDs don't have ancestry
    - XI.10.c.spouse1 (Travis Chase Woods)
    - XI.20.a.spouse1 (Tom Krans)
    - XI.20.b.spouse1 (Maaike van Oppenraay)
    - XI.21.a.spouse1 (Ane Veldman)
    - XI.21.b.spouse1 (Alieke Zwerver)
    - XI.8.a.spouse1 (Jesse Newton)
  - **6 people blocked by source data typos**:
    - XI.10.a, XI.10.b, XI.10.c (Kamp siblings) - parent ID typo (IX.6.c → X.6.c)
    - XI.11.a, XI.11.b, XI.11.c (Ringling siblings) - parent ID typo (IX.6.d → X.6.d)
    - XI.6.a (Michelle van der Reijden) - parent ID typo (X.3.c → X.4.c)

**Generation 10 Ancestry Tracing Results:**
- **Success rate**: 63.4% (59/93 people can trace back to generation 2)
- **People with genealogical IDs**: **59/59 = 100% success rate!** 🎉
- **Blocked people**: 34 minted spouses (all have .spouse1, .spouse2, or .spouse3 suffixes)
- **Zero failures** for people with genealogical IDs

**Generation 9 Ancestry Tracing Results:**
- **Success rate**: 66.7% (42/63 people can trace back to generation 2)
- **People with genealogical IDs**: **42/42 = 100% success rate!** 🎉
- **Blocked people**: 21 minted spouses
- **Zero failures** for people with genealogical IDs

**Generation 8 Ancestry Tracing Results:**
- **Success rate**: 62.1% (18/29 people can trace back to generation 2)
- **People with genealogical IDs**: **18/18 = 100% success rate!** 🎉
- **Blocked people**: 11 minted spouses
- **Zero failures** for people with genealogical IDs

**Analysis Summary (Before Sibling Fix):**
- **Generation 12**: 27/30 people with IDs = 90% success (3 blocked by source data issues)
- **Generation 11**: 49/55 people with IDs = 89% success (6 blocked by source data typos)
- **Generation 10**: 59/59 people with IDs = **100% success!** ✨
- **Generation 9**: 42/42 people with IDs = **100% success!** ✨
- **Generation 8**: 18/18 people with IDs = **100% success!** ✨
- **Overall**: **195/202 people with genealogical IDs (96.5%)** can trace back to generation 2
- **All 7 failures are due to documented source data issues** (not code bugs)

### ✅ COMPLETED - Phase 2.2: Fix Sibling Entry Chunking Bug

**Date Completed**: 2025-11-12

**Issue Found**:
The chunking strategy was missing people when siblings were listed in compact format:
```
a. Nathaniel James Hall, * 18.1.1984
b. Jamie Nicole Hall, * 14.7.1986
c. Joseph Steven Hall, * 16.5.1990
```

The OCR sometimes returns multiple sibling entries in a **single grounding token**, and the chunking strategy was treating the entire token as one chunk for only the first sibling (a.), completely missing siblings b. and c.

**Root Cause**:
In `IndividualEntryHandler`, when collecting tokens for an individual entry, it would consume the entire multi-sibling token, preventing subsequent siblings from being processed.

**The Fix**:
Added `_split_multi_sibling_tokens()` method in `DescendantGenealogyChunkingStrategy` that:
1. Pre-processes all tokens before main flow chunking
2. Detects tokens containing multiple lines starting with individual entry markers (a., b., c.)
3. Splits these into separate `GroundingToken` objects
4. Each sibling now gets processed as a separate chunk

**Files Modified**:
- `genealogy/chunking_strategies/descendant_genealogy.py` - Added token splitting pre-processor

**Results After Fix:**

**Generation 12** (2025-11-12 after fix):
- **Success rate**: 96.8% (30/31 people can trace back to generation 2)
- **Improvement**: From 27/30 to 30/31 🎉
- **Remaining blocked**: 1 person
  - XII.2.a (Greyson Woods) - blocked by parent XI.10.c who has the IX.6.c → X.6.c typo

**Generation 11** (2025-11-12 after fix):
- **Success rate**: 80% (60/75 people can trace back)
- **New people discovered**: XI.16.b, XI.16.c, XI.6.b (were hidden in multi-sibling tokens)
- **Remaining blocked (non-spouse)**: 7 people
  - XI.10.a, XI.10.b, XI.10.c - parent ID typo (IX.6.c → X.6.c)
  - XI.11.a, XI.11.b, XI.11.c - parent ID typo (IX.6.d → X.6.d)
  - XI.6.a, XI.6.b - parent ID typo (X.3.c → X.4.c)
- **Minted spouses blocked**: 8 (expected behavior)

**Key Improvements:**
- ✅ XI.16.b (Jamie Nicole Hall) now exists - was the missing parent for XII.10.a and XII.10.b
- ✅ XI.16.c (Joseph Steven Hall) now exists
- ✅ XI.6.b (Floor van der Reijden) now exists
- ✅ XII.10.a and XII.10.b (Abercrombie children) can now trace back!

**Validation**:
Found 9 chunks with multi-sibling tokens before the fix. All are now properly split into individual chunks.

### ✅ COMPLETED - Phase 2.3: Implement Genealogical ID Correction System

**Date Completed**: 2025-11-12

**Issue**:
Even after fixing the sibling chunking bug, several people couldn't trace back due to genealogical ID typos in the source document's family group headers. For example, "Children of X (IX.6.c):" when it should be "(X.6.c):".

**The Solution**:
Created a centralized correction system in `genealogy/utils/id_corrections.py` that:
1. Defines known genealogical ID typos in `GENEALOGICAL_ID_CORRECTIONS` dict
2. Provides `correct_genealogical_id()` function to apply corrections
3. Automatically corrects IDs when parsing family group headers

**Corrections Implemented**:
- `IX.6.c` → `X.6.c` (Thomas Gregory Kamp MSc)
- `IX.6.d` → `X.6.d` (Nancy Suzanne Kamp BSc)
- `X.3.c` → `X.4.c` (Ewout van der Reijden)

**Integration**:
Modified `genealogy/utils/family_parsing.py`:
- `parse_family_group_header()` now calls `correct_genealogical_id()` before returning
- All parent genealogical IDs are automatically corrected during graph building

**Files Modified**:
- `genealogy/utils/id_corrections.py` - New correction utility module
- `genealogy/utils/family_parsing.py` - Integrated corrections into parsing

**Final Results** (2025-11-12 after ID corrections):

**Generation 12**:
- ✅ **100% success (31/31 people with IDs can trace back to generation 2)** 🎉
- Zero failures!

**Generation 11**:
- ✅ **100% success for people with genealogical IDs (68/68)** 🎉
- 7 minted spouses blocked (expected - they don't have genealogical IDs by design)

**Overall Achievement**:
- **Generations 8-12**: All people with genealogical IDs can successfully trace back to generation 2
- **Only remaining "blocked" people**: Minted spouses (partners without genealogical IDs in source)
- **All source data typos are now automatically corrected** during graph building

**Conclusion**:
The genealogy graph building system is now **production-ready**:
1. ✅ Chunking correctly splits multi-sibling entries
2. ✅ Genealogical ID typos are automatically corrected
3. ✅ 100% of people with genealogical IDs can trace their ancestry
4. ✅ Minted spouses work as designed (no ancestry, used for partnership tracking)

### ✅ COMPLETED - Phase 4.1: Update Agent Tools for New Person Model

**Date Completed**: 2025-11-13

**What was done**:
1. ✅ **Updated genealogy/services/genealogy_tools.py**:
   - Completely rewrote from 510 lines to 395 lines (115 lines removed)
   - Migrated from Identity/PersonMention/MentionToIdentity architecture to Person/Relationship/Partnership
   - Removed all complex mention-to-identity mapping logic
   - Updated all methods:
     - `_get_person()` - Look up by genealogical_id or UUID
     - `search_person_by_name()` - Search Person table directly
     - `get_person_details()` - Return Person details with events/parents/children/partners
     - `search_by_birth_year()` - Use proper date queries (`date__year__gte`, `date__year__lte`)
     - `get_children()` - Return children via Relationship model
     - `get_parents()` - Return parents via Relationship model

2. ✅ **Updated Event model with proper date storage**:
   - Changed `Event.date` from `CharField` to `DateField`
   - Added `Event.date_original` (CharField) to preserve original string
   - Added `Event.date_approximate` (BooleanField) for approximate dates
   - Updated event types to use only GEDCOM codes (removed "BIRTH"/"DEATH" duplicates, kept "BIRT"/"DEAT")

3. ✅ **Created date parsing utility**:
   - New file: `genealogy/utils/date_parsing.py`
   - Function: `parse_genealogical_date(date_str)` returns `(date, is_approximate)`
   - Uses `python-dateutil` for flexible parsing
   - Handles formats: ISO (1845-03-12), European (dd.mm.yyyy), year-only (1845), approximate indicators (ca., circa, ~, about)
   - Returns `date` object or None if unparseable

4. ✅ **Created migration for date field change**:
   - New file: `genealogy/migrations/0039_change_event_date_to_datefield.py`
   - Multi-step migration:
     1. Add date_approximate and date_original fields
     2. Copy existing date (CharField) to date_original via SQL
     3. Drop old date field
     4. Create new date as DateField
     5. Use RunPython to parse all date_original values into date using dateutil
   - Includes forward and backward migration functions
   - Migration ran successfully

5. ✅ **Re-parsed all existing dates**:
   - Ran date parser on 507 unparsed events
   - Successfully parsed 470 dates (93% success rate)
   - 37 failed dates are date ranges like "1746/1747" or "1806-1829" (expected failures)

6. ✅ **Migrated old event types**:
   - Converted 237 "BIRTH" → "BIRT" events
   - Converted 104 "DEATH" → "DEAT" events
   - Now have 239 birth events with dates available for searching

7. ✅ **Updated genealogy/tests/test_genealogy_tools.py**:
   - Completely rewrote from 301 lines to 221 lines
   - Migrated from Identity/PersonMention setup to Person/Relationship/Partnership
   - Updated to use date objects instead of strings in test data
   - Updated assertions to expect ISO format date strings from API
   - **All 13 tests passing** ✅

8. ✅ **Updated persist_entities.py to use date parser**:
   - Now calls `parse_genealogical_date()` when creating Event records
   - Stores parsed date object in Event.date
   - Stores original string in Event.date_original
   - Stores approximate flag in Event.date_approximate

9. ✅ **End-to-end verification**:
   - Tested all genealogy tools methods with real database data:
     - `search_person_by_name()` - Found people and returned correct data
     - `get_person_details()` - Showed events, parents, children, partners
     - `get_children()` - Returned children with birth years
     - `get_parents()` - Returned parents correctly
     - `search_by_birth_year()` - Found people in date ranges using proper SQL queries
   - All tools working correctly with 390 people, 239 birth events with dates

**Files Modified/Created**:
- `genealogy/services/genealogy_tools.py` - Migrated to new Person model (510→395 lines)
- `genealogy/tests/test_genealogy_tools.py` - Migrated tests (301→221 lines), 13/13 passing
- `genealogy/models.py` - Updated Event model with DateField and new event types
- `genealogy/utils/date_parsing.py` - NEW: Date parsing utility
- `genealogy/migrations/0039_change_event_date_to_datefield.py` - NEW: Migration
- `genealogy/tasks/persist_entities.py` - Updated to use date parser

**Key Achievements**:
- ✅ Agent tools now fully compatible with simplified Person model
- ✅ Proper date storage enables SQL date range queries (birth year search works!)
- ✅ Flexible date parsing handles multiple formats and approximations
- ✅ All existing data migrated successfully
- ✅ All tests passing
- ✅ End-to-end verification complete

**Remaining Agent Files** (All Updated):
- ~~`genealogy/services/genealogy_tools.py`~~ ✅ UPDATED
- ~~`genealogy/tests/test_genealogy_tools.py`~~ ✅ UPDATED
- `genealogy/tests/test_agent_integration.py` - May need updates (not checked yet)
- `genealogy/tests/test_query_genealogy_command.py` - May need updates (not checked yet)
- `genealogy/services/agent_executor.py` - Depends on genealogy_tools.py (should work)

### What's Left to Do

**Phase 5 - Integration Testing** (1-2 hours):
- Test full workflow: chunk → build_graph → extract_entities → persist_entities
- Verify RAG queries still work
- Test agent queries on real data
- Test on multiple documents

**Phase 6 - Documentation** (1 hour):
- Update workflow docs
- Update README
- Document new admin actions
- Document date parsing system
