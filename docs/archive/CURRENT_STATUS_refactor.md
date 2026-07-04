# Refactor Status - Reversible Provenance Architecture

**Last Updated**: 2025-10-20
**Current Session**: Continuing refactor work

---

## ✅ Completed

### Phase 1: Model Changes
All core model changes are complete in `genealogy/models.py`:

- ✅ Renamed `Person` → `PersonMention`
- ✅ Renamed `ParentChildRelationship` → `RelationshipMention`
- ✅ Renamed `Partnership` → `PartnershipMention`
- ✅ Updated `Event.person` → `Event.mention`
- ✅ Updated `PotentialDuplicate.person1/person2` → `mention1/mention2`
- ✅ Deleted `EntityMerge` model
- ✅ Created `Identity` model
- ✅ Created `MentionToIdentity` model
- ✅ Created `MergeEvent` model

### Phase 2: Management Commands
- ✅ Updated `create_entities.py` to create `PersonMention` + singleton `Identity`
- ✅ Updated `create_entities.py` to use `RelationshipMention` and `PartnershipMention`
- ✅ Updated `cluster_entities.py` to use `PersonMention`
- ✅ Fixed parent/child relationship direction bug (relationship_type 'parent' was backwards)
- ✅ Added on-the-fly PersonMention creation for missing relationship references
- ✅ Enhanced date parsing to handle uncertainty (circa, ranges, Dutch terms)

### Phase 3: Admin UI - Merge/Unmerge System
- ✅ `genealogy/admin/merge_logic.py` - NEW, transactional merge/unmerge functions
- ✅ `genealogy/admin/duplicate_clusters.py` - Updated to use new merge logic
- ✅ `genealogy/admin/person_mention.py` - NEW, read-only admin with ad hoc merge action
- ✅ `genealogy/admin/identity.py` - NEW, full Identity admin with:
  - Events rollup display with mention links
  - Relationships rollup display (deduplicated by identity)
  - Partnerships rollup display (deduplicated by identity)
  - Source chunks rollup display
  - Merge history display with unmerge buttons
  - Unmerge view with custom URL route
- ✅ `genealogy/admin/relationship.py` - Updated to use RelationshipMention
- ✅ `genealogy/admin/partnership.py` - Updated to use PartnershipMention
- ✅ `genealogy/admin/event.py` - No changes needed (already correct)
- ✅ `genealogy/admin/__init__.py` - Updated imports

---

## 🚧 In Progress - Admin Import Updates

These files still reference old model names and need updates:

### 1. `genealogy/admin/textchunk.py`
**Changes needed**:
```python
# Line 7: Change imports
from ..models import TextChunk, PersonMention, Event, RelationshipMention  # was: Person, ParentChildRelationship

# Line 178: Update query
persons = PersonMention.objects.filter(source_chunks=obj)...  # was: Person.objects

# Line 188: Update URL
person_url = reverse('admin:genealogy_personmention_change', args=[person.id])  # was: genealogy_person_change

# Line 207: Update query
persons = PersonMention.objects.filter(source_chunks=obj)  # was: Person.objects

# Line 213: Update FK reference
events = Event.objects.filter(mention__in=persons)...  # was: person__in

# Line 228: Update URL
person_url = reverse('admin:genealogy_personmention_change', args=[person.id])

# Line 254: Update query
persons = PersonMention.objects.filter(source_chunks=obj)

# Line 262: Update model and FK
relationships = RelationshipMention.objects.filter(child_mention_id__in=person_ids)  # was: ParentChildRelationship, child_id

# Line 266: Update model and FK
parent_relationships = RelationshipMention.objects.filter(parent_mention_id__in=person_ids)  # was: parent_id

# Line 269: Update FK
.exclude(child_mention_id__in=person_ids)  # was: child_id

# Line 285: Update URL
rel_url = reverse('admin:genealogy_relationshipmention_change', args=[rel.id])

# Line 286-287: Update URLs
child_url = reverse('admin:genealogy_personmention_change', args=[rel.child_mention.id])
parent_url = reverse('admin:genealogy_personmention_change', args=[rel.parent_mention.id])

# Lines 291-292: Update attribute references
{rel.child_mention.full_name} ... {rel.parent_mention.full_name}

# Lines 303-305: Same updates for parent_relationships section
```

### 2. `genealogy/admin/document.py`
**Check if it imports old models** - likely minimal changes

### 3. `genealogy/admin/duplicate_clusters.py`
**This is the BIG one** - contains all the merge logic

This file needs extensive refactoring because:
- It references `Person`, `EntityMerge`, `ParentChildRelationship`, `Partnership`
- All merge logic needs to be rewritten to use new transactional approach
- Should create `MergeEvent` records instead of `EntityMerge`
- Should update `MentionToIdentity` instead of moving relationships

**Recommendation**: Save this for a future session with fresh context. The merge logic is complex and needs careful refactoring per the design in `research/reversible_provenance.md`.

### 4. `genealogy/admin/place.py` and `genealogy/admin/document_page.py`
**Check if they need updates** - probably minimal/none

---

## 📋 Next Steps (Priority Order)

### Immediate (Can complete with remaining context):

1. **Fix textchunk.py imports** (straightforward find-replace)
2. **Check document.py, place.py, document_page.py** for any old model references
3. **Run makemigrations** to generate migration files
4. **Try to run migrate** (may fail if there are issues)

### Future Session (Requires fresh context):

5. **Refactor duplicate_clusters.py merge logic**
   - This is complex and needs the full design from `research/reversible_provenance.md`
   - Implement transactional merge using `MentionToIdentity` + `MergeEvent`
   - See Phase 2 in `REFACTOR_PLAN_reversible_provenance.md`

6. **Update management commands**:
   - `create_entities` - Change `Person` → `PersonMention`, create singleton `Identity`
   - `cluster_entities` - Update to work with `PersonMention`

7. **Nuke existing data and test**:
   - Delete all Person/Event/Relationship records
   - Re-run extraction
   - Test clustering
   - Test merge/unmerge workflows

---

## 🔑 Key Architecture Principles

Remember these when continuing:

1. **PersonMention is immutable** - never edit after creation
2. **MentionToIdentity is the ONLY mutable part** - all merges happen here
3. **MergeEvent records full transactions** - for complete reversibility
4. **Events and relationships stay on mentions** - don't copy/move them
5. **Query pattern**: Mention → MentionToIdentity → Identity (then aggregate)

---

## 📝 Command to Resume

When you resume in a new session:

```bash
# 1. Check which files still need updates
grep -r "from.*Person\|EntityMerge\|Partnership\|ParentChild" genealogy/admin/*.py

# 2. After fixing imports, try migrations
docker compose exec web python manage.py makemigrations
docker compose exec web python manage.py migrate

# 3. If migrations succeed, nuke old data
docker compose exec web python manage.py shell -c "
from genealogy.models import *
PersonMention.objects.all().delete()
Event.objects.all().delete()
RelationshipMention.objects.all().delete()
PartnershipMention.objects.all().delete()
Identity.objects.all().delete()
PotentialDuplicate.objects.all().delete()
"
```

---

## 🐛 Known Issues

None yet - migrations haven't been attempted

---

## 📚 Reference Documents

- **Design**: `research/reversible_provenance.md` - Core architecture
- **Plan**: `docs/REFACTOR_PLAN_reversible_provenance.md` - Implementation roadmap
- **This Doc**: `docs/CURRENT_STATUS_refactor.md` - Current session state
