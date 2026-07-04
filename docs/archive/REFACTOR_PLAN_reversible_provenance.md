# Refactor Plan: Reversible Provenance Architecture

**Goal**: Refactor the entity resolution system to implement a fully reversible, auditable provenance model based on the design in `research/reversible_provenance.md`.

**Status**: Planning phase
**Started**: 2025-10-18
**Owner**: eric

---

## Executive Summary

The current merge system has architectural issues:
- Events are copied to canonical entities (creates duplicates)
- Canonical entities can be merged into other canonicals (creates nested mess)
- Relationships are moved, breaking reversibility
- No transactional event log for merge operations

The new architecture will:
- Keep all extracted data immutable (person mentions, events, relationships)
- Use a mapping layer (`mention_to_identity`) as the ONLY mutable component
- Log all merge/unmerge operations as reversible transactions
- Enable full audit trail and temporal queries

---

## Current State Analysis

### Current Models (as of 2025-10-18)

```python
class Person(models.Model):
    # Has both EXTRACTED and CANONICAL entity_type
    entity_type = models.CharField(max_length=20, choices=[('EXTRACTED', 'Extracted'), ('CANONICAL', 'Canonical')])
    canonical_entity = models.ForeignKey('self', null=True, blank=True)  # FK to canonical
    # ... name, gender, generation fields ...

class Event(models.Model):
    person = models.ForeignKey(Person)  # Points to either EXTRACTED or CANONICAL
    event_type = models.CharField(...)  # BIRTH, DEATH, MARRIAGE, etc.
    date = models.DateField(...)
    place = models.ForeignKey(Place, ...)
    # Currently COPIED during merge

class ParentChildRelationship(models.Model):
    parent = models.ForeignKey(Person)
    child = models.ForeignKey(Person)
    # Currently moved to point to canonical entities

class Partnership(models.Model):
    partners = models.ManyToManyField(Person)
    # Currently moved to point to canonical entities

class EntityMerge(models.Model):
    canonical_entity = models.ForeignKey(Person)
    source_entity = models.ForeignKey(Person)
    confidence_score = models.FloatField()
    pairwise_similarities = models.JSONField()
    # Tracks pairwise merges but not full transaction
```

### Current Problems

1. **Circular merges**: Canonical entity 924bbb7b-dc6e-44a7-b7d1-6871cbd4bc62 has been merged into itself
2. **Nested canonicals**: Multiple CANONICAL entities are sources in EntityMerge (should be impossible)
3. **Event duplication**: Canonical entities accumulate copies of events from all sources
4. **Non-reversible relationships**: When relationships are moved, can't restore them on unmerge
5. **No transaction log**: EntityMerge is pairwise, not transactional

---

## Target Architecture

### New Models

```python
class PersonMention(models.Model):
    """Immutable extraction from source text"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    # Source tracking
    source_chunk = models.ForeignKey(TextChunk)
    source_document = models.ForeignKey(Document)

    # Extracted attributes (immutable)
    given_names = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    maiden_name = models.CharField(max_length=255, blank=True)
    gender = models.CharField(max_length=10, blank=True)
    generation = models.IntegerField(null=True, blank=True)

    # Never modified after creation
    created_at = models.DateTimeField(auto_now_add=True)

class Identity(models.Model):
    """Canonical person - the resolved entity"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    display_name = models.CharField(max_length=500)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)  # Soft delete for reversibility

class MentionToIdentity(models.Model):
    """Mapping layer - THE ONLY MUTABLE PART"""
    mention = models.OneToOneField(PersonMention, primary_key=True, on_delete=models.CASCADE)
    identity = models.ForeignKey(Identity, on_delete=models.CASCADE)
    mapped_at = models.DateTimeField(auto_now=True)
    mapped_by = models.CharField(max_length=100)

class MergeEvent(models.Model):
    """Audit log - never delete from this table"""
    id = models.BigAutoField(primary_key=True)
    event_type = models.CharField(max_length=20, choices=[
        ('merge', 'Merge'),
        ('unmerge', 'Unmerge'),
        ('split', 'Split')
    ])
    payload = models.JSONField()  # Contains full transaction details
    performed_by = models.CharField(max_length=100)
    performed_at = models.DateTimeField(auto_now_add=True)
    reversed_event = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL)

class Event(models.Model):
    """Events stay attached to mentions - never moved"""
    mention = models.ForeignKey(PersonMention, related_name='events')  # Changed from Person
    event_type = models.CharField(...)
    date = models.DateField(...)
    place = models.ForeignKey(Place, ...)
    # Never modified after creation

class RelationshipMention(models.Model):
    """Relationships between mentions - immutable"""
    parent_mention = models.ForeignKey(PersonMention, related_name='child_relationships')
    child_mention = models.ForeignKey(PersonMention, related_name='parent_relationships')
    relationship_type = models.CharField(max_length=50)
    # Never modified after creation

class PartnershipMention(models.Model):
    """Partnerships stay at mention level"""
    mentions = models.ManyToManyField(PersonMention, related_name='partnerships')
    partnership_type = models.CharField(max_length=50)
    # Never modified after creation
```

---

## Migration Strategy

**Simplified Approach**: No data migration needed! Just nuke the tables and re-run extraction.

Since `create_entities` and `cluster_entities` management commands can be re-run relatively quickly, we'll:
1. Delete all existing Person/Event/Relationship data
2. Update the models to new architecture
3. Update `create_entities` and `cluster_entities` commands
4. Re-run extraction from scratch

This is cleaner than trying to migrate the broken data.

### Phase 1: Model Changes

#### Step 1.1: Rename and repurpose existing models
- [ ] Rename `Person` model to `PersonMention`
- [ ] Remove `entity_type` field (no longer needed)
- [ ] Remove `canonical_entity` FK (replaced by MentionToIdentity)
- [ ] Make all fields immutable (remove any update logic)

#### Step 1.2: Create new models
- [ ] Add `Identity` model
- [ ] Add `MentionToIdentity` model
- [ ] Add `MergeEvent` model
- [ ] Run `makemigrations` and `migrate`

#### Step 1.3: Update relationship models
- [ ] Rename `ParentChildRelationship` to `RelationshipMention`
- [ ] Update FKs to point to `PersonMention` (not Person)
- [ ] Rename `Partnership` to `PartnershipMention`
- [ ] Update FKs to point to `PersonMention`

#### Step 1.4: Update Event model
- [ ] Rename FK from `person` to `mention`
- [ ] Ensure it points to `PersonMention`

#### Step 1.5: Update PotentialDuplicate model
- [ ] Rename FKs from `person1`/`person2` to `mention1`/`mention2`
- [ ] Ensure it points to `PersonMention`

#### Step 1.6: Remove obsolete models
- [ ] Delete `EntityMerge` model (replaced by MergeEvent)

### Phase 2: Update Management Commands

#### Step 2.1: Update `create_entities` command
- [ ] Modify to create `PersonMention` objects (not Person with entity_type='EXTRACTED')
- [ ] Create singleton `Identity` + `MentionToIdentity` for each mention
- [ ] Extract events and link to `PersonMention` (rename FK to `mention`)
- [ ] Extract relationships as `RelationshipMention` (linking PersonMentions)
- [ ] Extract partnerships as `PartnershipMention` (linking PersonMentions)

#### Step 2.2: Update `cluster_entities` command
- [ ] Modify `genealogy/clustering/graph.py` to work with PersonMention
- [ ] Create `PotentialDuplicate` linking PersonMentions (rename person1/person2 to mention1/mention2)
- [ ] No merge logic here - just detection

#### Step 2.3: Rewrite merge logic in admin
- [ ] Replace `_process_cluster_merge()` in `duplicate_clusters.py`
- [ ] Implement transactional merge:
  ```python
  def merge_identities(survivor_identity_id, absorbed_mention_ids, user):
      with transaction.atomic():
          # Collect current mappings
          moved = []
          for mention_id in absorbed_mention_ids:
              mapping = MentionToIdentity.objects.get(mention_id=mention_id)
              moved.append({
                  'mention_id': str(mention_id),
                  'from': str(mapping.identity_id),
                  'to': str(survivor_identity_id)
              })

          # Update mappings
          MentionToIdentity.objects.filter(
              mention_id__in=absorbed_mention_ids
          ).update(
              identity_id=survivor_identity_id,
              mapped_by=user
          )

          # Soft-delete absorbed identities
          absorbed_ids = list(set(m['from'] for m in moved))
          Identity.objects.filter(id__in=absorbed_ids).update(is_deleted=True)

          # Log event
          MergeEvent.objects.create(
              event_type='merge',
              payload={
                  'survivor_identity': str(survivor_identity_id),
                  'absorbed_identities': absorbed_ids,
                  'mention_moves': moved
              },
              performed_by=user
          )
  ```

#### Step 2.4: Implement unmerge logic
- [ ] Read MergeEvent and replay in reverse
- [ ] Restore absorbed identities (is_deleted=False)
- [ ] Log unmerge event:
  ```python
  def unmerge_identity(merge_event_id, user):
      with transaction.atomic():
          event = MergeEvent.objects.get(id=merge_event_id)
          payload = event.payload

          # Restore absorbed identities
          absorbed_ids = payload['absorbed_identities']
          Identity.objects.filter(id__in=absorbed_ids).update(is_deleted=False)

          # Reverse mention moves
          for move in payload['mention_moves']:
              MentionToIdentity.objects.filter(
                  mention_id=move['mention_id']
              ).update(
                  identity_id=move['from'],
                  mapped_by=user
              )

          # Log unmerge
          MergeEvent.objects.create(
              event_type='unmerge',
              payload=payload,
              performed_by=user,
              reversed_event_id=merge_event_id
          )
  ```

#### Step 2.5: Update query logic for admin views
- [ ] Rewrite views to query through MentionToIdentity
- [ ] Example: Get all events for an identity:
  ```python
  def get_identity_events(identity_id):
      return Event.objects.filter(
          mention__mentiontoidentity__identity_id=identity_id
      ).distinct()
  ```
- [ ] Example: Get all relationships for an identity:
  ```python
  def get_identity_parents(identity_id):
      # Get mentions mapped to this identity
      mention_ids = MentionToIdentity.objects.filter(
          identity_id=identity_id
      ).values_list('mention_id', flat=True)

      # Get parent relationships through those mentions
      parent_mentions = RelationshipMention.objects.filter(
          child_mention_id__in=mention_ids
      ).values_list('parent_mention_id', flat=True)

      # Resolve to parent identities
      return Identity.objects.filter(
          mentiontoidentity__mention_id__in=parent_mentions
      ).distinct()
  ```

### Phase 3: Update Admin UI

#### Step 3.1: Split Person admin into two
- [ ] Create `PersonMentionAdmin` - read-only view of extractions
  - List display: name, generation, events, source chunks
  - Show which Identity this mention maps to
  - Link to source chunks and documents
- [ ] Create `IdentityAdmin` - main interface for resolved entities
  - List display: display_name, num_mentions, generation
  - Detail view: shows all mentions, events, relationships rolled up
  - Actions: split identity, manual merge

#### Step 3.2: Update PotentialDuplicateAdmin
- [ ] Update cluster computation to work with PersonMentions
- [ ] Update merge form to work with new transactional merge logic
- [ ] Update cluster detail view to show mentions (not persons)

#### Step 3.3: Rewrite provenance display
- [ ] Show MergeEvent log for an identity (audit trail)
- [ ] Show which mentions map to an identity (with confidence scores)
- [ ] Show events grouped by mention (to see which mention contributed which event)
- [ ] Add "Undo Merge" button that reverses a MergeEvent

#### Step 3.4: Update relationship displays
- [ ] Query through RelationshipMention, resolve via MentionToIdentity
- [ ] Show parents/children/partners as Identities
- [ ] Add badges showing how many mentions contribute to each relationship

### Phase 4: Testing and Cleanup

#### Step 4.1: Nuke existing data
- [ ] Delete all Person records
- [ ] Delete all Event records
- [ ] Delete all ParentChildRelationship records
- [ ] Delete all Partnership records
- [ ] Delete all EntityMerge records
- [ ] Delete all PotentialDuplicate records

#### Step 4.2: Re-run extraction
- [ ] Run `create_entities` management command
- [ ] Verify PersonMention objects created with singleton Identities
- [ ] Verify Events linked to mentions
- [ ] Verify RelationshipMentions created

#### Step 4.3: Re-run clustering
- [ ] Run `cluster_entities` management command
- [ ] Verify PotentialDuplicate records created
- [ ] Test merge workflow in admin
- [ ] Test unmerge workflow in admin

#### Step 4.4: Update documentation
- [ ] Update README with new architecture
- [ ] Document merge/unmerge workflows
- [ ] Add notes about reversibility and audit trail

---

## Testing Strategy

### Unit Tests
- [ ] Test merge transaction (multiple mentions → single identity)
- [ ] Test unmerge transaction (restore absorbed identities)
- [ ] Test split transaction (move subset of mentions to new identity)
- [ ] Test event rollup (query all events for an identity)
- [ ] Test relationship rollup (query all relationships for an identity)

### Integration Tests
- [ ] End-to-end: Extract → Cluster → Merge → Unmerge → Verify
- [ ] Test circular merge prevention
- [ ] Test nested merge handling

### Manual Testing
- [ ] Use Bessel van Zanten case as test case
- [ ] Verify can untangle the mess by:
  1. Unmerging all mentions
  2. Re-clustering properly
  3. Merging correctly this time

---

## Design Decisions (RESOLVED)

1. **PotentialDuplicate model**: ✅ Link PersonMentions (duplicates exist at mention level, pre-merge)

2. **Display name logic**: ✅ Option A - Copy from base mention (simple, matches current approach)

3. **Singleton identities**: ✅ Yes - Auto-create Identity for every unmapped PersonMention (makes querying simpler)

4. **Event deduplication**: ✅ Keep all events, deduplicate in display layer (current approach is fine)

---

## Success Criteria

- [ ] All existing Person records migrated to PersonMention + Identity
- [ ] All EntityMerge history reconstructed as MergeEvent transactions
- [ ] Can untangle Bessel van Zanten mess (924bbb7b-dc6e-44a7-b7d1-6871cbd4bc62)
- [ ] Merge/unmerge operations work correctly
- [ ] No data loss during migration
- [ ] All admin UI functions work with new models
- [ ] Clustering algorithm works with new models

---

## Timeline Estimate

- **Phase 1** (Model Changes): 3-4 hours
- **Phase 2** (Management Commands + Merge Logic): 6-8 hours
- **Phase 3** (Admin UI): 6-8 hours
- **Phase 4** (Testing and Cleanup): 2-3 hours

**Total**: ~17-23 hours of focused development (reduced from original estimate since no data migration)

---

## Current Status (2025-10-18)

### Completed
- ✅ Phase 1.1-1.6: All model changes complete
  - Renamed `Person` → `PersonMention`
  - Removed `entity_type` and `canonical_entity` fields
  - Renamed `ParentChildRelationship` → `RelationshipMention`
  - Renamed `Partnership` → `PartnershipMention`
  - Updated `Event.person` → `Event.mention`
  - Updated `PotentialDuplicate.person1/person2` → `mention1/mention2`
  - Deleted `EntityMerge` model
  - Created `Identity`, `MentionToIdentity`, `MergeEvent` models

### Next Steps
1. **Update all admin imports** - Admin files still import old model names:
   - Change `Person` → `PersonMention`
   - Change `ParentChildRelationship` → `RelationshipMention`
   - Change `Partnership` → `PartnershipMention`
   - Remove `EntityMerge` imports
   - Add `Identity`, `MentionToIdentity`, `MergeEvent` imports

2. **Run migrations** after fixing imports

3. **Nuke existing data** before testing

4. **Update management commands** to work with new models

### Files Needing Import Updates
- `genealogy/admin/person.py`
- `genealogy/admin/duplicate_clusters.py`
- `genealogy/admin/__init__.py`
- `genealogy/admin/textchunk.py`
- `genealogy/admin/relationship.py`
- `genealogy/admin/event.py`
- `genealogy/admin/partnership.py`
- `genealogy/admin/place.py`
- `genealogy/admin/document_page.py`
- `genealogy/admin/document.py`

---

## Notes for Future Sessions

### Execution Order

1. **Start with Phase 1** (Model Changes)
   - Work through Step 1.1 → 1.6 sequentially
   - Run migrations after each step
   - Don't nuke data until models are updated

2. **Then Phase 2** (Commands)
   - Update `create_entities` first
   - Update `cluster_entities` second
   - Implement merge/unmerge logic in admin last

3. **Then Phase 3** (Admin UI)
   - Split person.py into two admin classes
   - Update PotentialDuplicateAdmin
   - Add provenance/audit displays

4. **Finally Phase 4** (Test)
   - Nuke all data
   - Re-run extraction
   - Test merge/unmerge workflows

### Key Files to Modify

**Models** (`genealogy/models.py`):
- Rename `Person` → `PersonMention`
- Add `Identity`, `MentionToIdentity`, `MergeEvent`
- Rename `ParentChildRelationship` → `RelationshipMention`
- Rename `Partnership` → `PartnershipMention`
- Update `Event` FK from `person` to `mention`
- Update `PotentialDuplicate` FKs from `person1/person2` to `mention1/mention2`

**Extraction** (`genealogy/management/commands/create_entities.py`):
- Create `PersonMention` instead of `Person(entity_type='EXTRACTED')`
- Create singleton `Identity` + `MentionToIdentity` for each mention
- Link events to `mention` FK

**Clustering** (`genealogy/clustering/graph.py`):
- Work with `PersonMention` instead of `Person`
- Create `PotentialDuplicate` with mention1/mention2

**Admin** (`genealogy/admin/`):
- Split `person.py` into `person_mention.py` and `identity.py`
- Update `duplicate_clusters.py` merge logic to use transactional approach
- Add MergeEvent display and undo functionality

### Reference Implementation (from research doc)

See `research/reversible_provenance.md` for:
- SQL pseudocode for merge transaction
- SQL pseudocode for undo transaction
- Query patterns for rolling up mention data to identities

### Current Broken State (Don't Try to Fix)

The Bessel van Zanten entity (924bbb7b-dc6e-44a7-b7d1-6871cbd4bc62) is hopelessly broken:
- Has 11 source entities, several are CANONICAL (wrong!)
- Circular reference (entity merged into itself)
- Multiple conflicting birth/death events

Don't try to fix this - just nuke and re-extract after refactor is complete.
