# Hierarchical Chunking Implementation Plan

**STATUS: ✅ COMPLETED**

All 8 phases implemented successfully. See summary at end of document.

## Executive Summary

Implement a two-tier search architecture to separate structured metadata (names, dates, places) from narrative biographical content. This will dramatically improve semantic search precision by only embedding chunks with meaningful biographical content.

## Problem Statement

Current issues:
- 46% of individual_entry chunks are < 100 characters (just "a. Name, * date")
- These stub entries have no semantic content but get embedded anyway
- Vector search returns irrelevant stubs with similar scores to relevant content
- Example: "militaire dienst" query returns 4/5 irrelevant stubs

## Solution Architecture

### Two-Tier System

**Tier 1: Metadata (No Embeddings)**
- Short entries with just vital statistics (< 200 chars)
- Headers and structural chunks
- Searchable via: trigram fuzzy matching, phonetic (DM codes), subject field
- Use case: "Find Pieter van Zanten" (name-based queries)

**Tier 2: Narrative (Full Embeddings)**
- Longer entries with biographical content (>= 200 chars)
- Rich text with occupations, life events, relationships
- Searchable via: vector embeddings + trigram + phonetic + subject
- Use case: "Who served in the military?" (semantic queries)

## Implementation Phases

---

## Phase 1: Database Schema Changes

### 1.1 Add search_tier field to TextChunk model

**File**: `genealogy/models.py`

**Changes**:
```python
class TextChunk(models.Model):
    # ... existing fields ...

    search_tier = models.CharField(
        max_length=20,
        choices=[
            ('metadata', 'Metadata Only - No Embedding'),
            ('narrative', 'Narrative Content - Full Embedding'),
        ],
        default='metadata',
        db_index=True,  # Index for fast filtering in queries
        help_text="Determines search strategy: metadata uses trigram/phonetic only, narrative uses full embeddings"
    )
```

### 1.2 Create Django migration

**Command**:
```bash
docker compose exec web python manage.py makemigrations genealogy --name add_search_tier
```

**Expected migration**:
- Add `search_tier` field with default='metadata'
- Add index on `search_tier` column

### 1.3 Apply migration

**Command**:
```bash
docker compose exec web python manage.py migrate
```

**Verification**:
```bash
docker compose exec web python manage.py shell
>>> from genealogy.models import TextChunk
>>> TextChunk.objects.first().search_tier
'metadata'
```

---

## Phase 2: Classification Logic

### 2.1 Create classification utility

**File**: `genealogy/utils/chunk_classification.py` (new file)

**Content**:
```python
"""
Utilities for classifying text chunks into search tiers.
"""

def classify_chunk_tier(
    text_content: str,
    chunk_type: str,
    extracted_events: list = None,
    extracted_relationships: list = None
) -> str:
    """
    Determine if a chunk should be metadata-only or narrative-tier.

    Args:
        text_content: The chunk's text content
        chunk_type: The chunk type (individual_entry, generation_header, etc.)
        extracted_events: List of extracted events (optional, for heuristics)
        extracted_relationships: List of extracted relationships (optional)

    Returns:
        'metadata' or 'narrative'

    Logic:
    - Headers always -> metadata
    - individual_entry < 200 chars -> metadata
    - individual_entry >= 200 chars -> narrative
    - biographical_text, narrative_context -> narrative

    Future enhancements could check for:
    - Presence of occupation/military/education keywords
    - Number of events extracted
    - Presence of narrative sentences (not just vital stats)
    """

    # Headers and structural chunks -> metadata tier
    if chunk_type in ['generation_header', 'family_group_header']:
        return 'metadata'

    # Length-based classification for individual entries
    if chunk_type == 'individual_entry':
        text_length = len(text_content.strip())

        # Short entries are just vital statistics -> metadata tier
        if text_length < 200:
            return 'metadata'

        # Long entries likely have biographical narrative -> narrative tier
        return 'narrative'

    # Explicit biographical/narrative chunks -> narrative tier
    if chunk_type in ['biographical_text', 'narrative_context']:
        return 'narrative'

    # Default to metadata tier for safety
    return 'metadata'


def get_tier_statistics(chunk_queryset=None):
    """
    Get statistics about chunk tier distribution.

    Args:
        chunk_queryset: Optional queryset to analyze. If None, uses all TextChunks.

    Returns:
        dict with tier distribution statistics
    """
    from genealogy.models import TextChunk
    from django.db.models import Count

    if chunk_queryset is None:
        chunk_queryset = TextChunk.objects.all()

    stats = chunk_queryset.values('search_tier').annotate(count=Count('id'))

    total = chunk_queryset.count()

    result = {
        'total': total,
        'by_tier': {},
        'percentages': {}
    }

    for stat in stats:
        tier = stat['search_tier']
        count = stat['count']
        result['by_tier'][tier] = count
        result['percentages'][tier] = (count / total * 100) if total > 0 else 0

    return result
```

### 2.2 Create management command to classify existing chunks

**File**: `genealogy/management/commands/classify_chunk_tiers.py` (new file)

**Content**:
```python
"""
Management command to classify existing TextChunks into search tiers.
"""

from django.core.management.base import BaseCommand
from django.db.models import Count
from genealogy.models import TextChunk
from genealogy.utils.chunk_classification import classify_chunk_tier, get_tier_statistics


class Command(BaseCommand):
    help = 'Classify existing TextChunks into metadata or narrative tiers'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be changed without actually updating',
        )
        parser.add_argument(
            '--chunk-type',
            type=str,
            help='Only classify chunks of this type (e.g., individual_entry)',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        chunk_type_filter = options.get('chunk_type')

        self.stdout.write(self.style.WARNING('Starting chunk tier classification...\n'))

        # Get chunks to classify
        queryset = TextChunk.objects.all()
        if chunk_type_filter:
            queryset = queryset.filter(chunk_type=chunk_type_filter)
            self.stdout.write(f'Filtering to chunk_type={chunk_type_filter}\n')

        total = queryset.count()
        self.stdout.write(f'Total chunks to classify: {total}\n')

        # Track changes
        changes = {
            'metadata': 0,
            'narrative': 0,
            'unchanged': 0
        }

        # Classify each chunk
        batch_size = 500
        for i in range(0, total, batch_size):
            batch = queryset[i:i+batch_size]

            for chunk in batch:
                new_tier = classify_chunk_tier(
                    text_content=chunk.text_content,
                    chunk_type=chunk.chunk_type,
                    extracted_events=chunk.extracted_events,
                    extracted_relationships=chunk.extracted_relationships
                )

                if chunk.search_tier != new_tier:
                    changes[new_tier] += 1

                    if not dry_run:
                        chunk.search_tier = new_tier
                        chunk.save(update_fields=['search_tier'])
                else:
                    changes['unchanged'] += 1

            # Progress indicator
            self.stdout.write(f'Processed {min(i+batch_size, total)}/{total} chunks...', ending='\r')

        self.stdout.write('\n')

        # Report results
        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No changes made\n'))

        self.stdout.write(self.style.SUCCESS('\nClassification Summary:'))
        self.stdout.write(f'  Set to metadata tier: {changes["metadata"]}')
        self.stdout.write(f'  Set to narrative tier: {changes["narrative"]}')
        self.stdout.write(f'  Unchanged: {changes["unchanged"]}\n')

        # Get final statistics
        stats = get_tier_statistics(queryset)
        self.stdout.write(self.style.SUCCESS('Final Tier Distribution:'))
        for tier, count in stats['by_tier'].items():
            pct = stats['percentages'][tier]
            self.stdout.write(f'  {tier}: {count} chunks ({pct:.1f}%)')

        self.stdout.write(self.style.SUCCESS('\n✓ Classification complete'))
```

### 2.3 Run classification on existing data

**Commands**:
```bash
# Dry run first to see what would change
docker compose exec web python manage.py classify_chunk_tiers --dry-run

# Review output, then run for real
docker compose exec web python manage.py classify_chunk_tiers
```

**Expected output**:
```
Starting chunk tier classification...
Total chunks to classify: 450

Classification Summary:
  Set to metadata tier: 250
  Set to narrative tier: 200
  Unchanged: 0

Final Tier Distribution:
  metadata: 250 chunks (55.6%)
  narrative: 200 chunks (44.4%)

✓ Classification complete
```

---

## Phase 3: Update Embedding Generation

### 3.1 Modify embedding generation to skip metadata tier

**File**: `genealogy/tasks/embeddings.py` (or wherever embeddings are generated)

**Find the current embedding generation logic and update it**:

```python
# BEFORE (current):
chunks_to_embed = TextChunk.objects.filter(embedding=None)

# AFTER (updated):
chunks_to_embed = TextChunk.objects.filter(
    embedding=None,
    search_tier='narrative'  # Only embed narrative tier
)
```

### 3.2 Create management command to remove metadata embeddings

**File**: `genealogy/management/commands/clean_metadata_embeddings.py` (new file)

**Content**:
```python
"""
Remove embeddings from metadata-tier chunks to save space.
"""

from django.core.management.base import BaseCommand
from genealogy.models import TextChunk


class Command(BaseCommand):
    help = 'Remove embeddings from metadata-tier chunks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        # Find metadata chunks with embeddings
        metadata_with_embeddings = TextChunk.objects.filter(
            search_tier='metadata',
            embedding__isnull=False
        )

        count = metadata_with_embeddings.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS('No metadata chunks have embeddings. Nothing to clean.'))
            return

        self.stdout.write(f'Found {count} metadata chunks with embeddings')

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - No changes will be made'))
        else:
            # Set embeddings to None
            metadata_with_embeddings.update(embedding=None)
            self.stdout.write(self.style.SUCCESS(f'✓ Removed embeddings from {count} metadata chunks'))
```

**Run**:
```bash
docker compose exec web python manage.py clean_metadata_embeddings --dry-run
docker compose exec web python manage.py clean_metadata_embeddings
```

---

## Phase 4: Update HybridRetriever

### 4.1 Add tier-aware filtering to hybrid search

**File**: `genealogy/retrieval.py`

**Location**: In `_hybrid_search()` method, after line 225 where we define `length_filter`

**Changes**:

```python
# Add tier-aware filtering logic
def _detect_query_type(self, query_text: str, query_dm_codes: list) -> str:
    """
    Detect if query is name-based or semantic.

    Returns:
        'name' for name-based queries (has capitalized names, DM codes)
        'semantic' for semantic queries (cross-cutting questions)
    """
    # If we have DM codes (capitalized names), it's likely a name query
    if query_dm_codes:
        return 'name'

    # Check for semantic query patterns
    semantic_patterns = [
        'who served',
        'who lived',
        'who worked',
        'any musician',
        'any military',
        'occupation',
        'profession',
        'emigrated',
        'moved to',
    ]

    query_lower = query_text.lower()
    if any(pattern in query_lower for pattern in semantic_patterns):
        return 'semantic'

    # Default to semantic if uncertain
    return 'semantic'
```

Then update the SQL generation:

```python
# After line 225 (where length_filter is defined), add:

# Detect query type
query_type = self._detect_query_type(query_text, query_dm_codes)

# Build tier filter based on query type
if query_type == 'semantic':
    # Semantic queries: only search narrative tier
    tier_filter = " AND search_tier = 'narrative'"
    logger.info(f"Semantic query detected - searching narrative tier only")
else:
    # Name queries: search both tiers
    tier_filter = " AND search_tier IN ('metadata', 'narrative')"
    logger.info(f"Name query detected - searching both tiers")

# Vector search ALWAYS only applies to narrative tier (only tier with embeddings)
vec_tier_filter = " AND search_tier = 'narrative'"
```

Then update the CTE WHERE clauses:

```python
# Line 241 - vec CTE
WHERE  embedding IS NOT NULL{vec_tier_filter}{person_filter_clause}

# Line 250 - trgm CTE
WHERE  text_content % q_text{length_filter}{tier_filter}{person_filter_clause}

# Line 258 - phon CTE
WHERE  dm_codes::text[] && q_dm{length_filter}{tier_filter}{person_filter_clause}

# Line 268 - subj CTE
WHERE  subject IS NOT NULL
       AND subject != ''
       AND subject % q_text{length_filter}{tier_filter}{person_filter_clause}
```

### 4.2 Update person filter extraction to avoid location false positives

**File**: `genealogy/retrieval.py`

**Location**: `_extract_person_names_from_query()` method (line 155)

**Current problem**: "Minneapolis" and "Amsterdam" are detected as person names

**Fix**:
```python
def _extract_person_names_from_query(self, query: str) -> Optional[List[str]]:
    """
    Extract potential person names from the query for pre-filtering.

    Looks for capitalized names and common Dutch/genealogical surname patterns.
    Filters out common place names and question words.
    """
    names = []

    # Extract capitalized words (potential names)
    words = re.findall(r'\b[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+\b', query)

    # Filter out common English/Dutch question words
    stop_words = {
        'Tell', 'Who', 'What', 'Where', 'When', 'How', 'Was', 'Were',
        'About', 'The', 'Are', 'There', 'Any', 'Did', 'Does', 'Is'
    }

    # Filter out common place names (US states, countries, major cities)
    place_names = {
        'Minneapolis', 'Minnesota', 'America', 'Amsterdam', 'Rotterdam',
        'Utrecht', 'Netherlands', 'Holland', 'United', 'States',
        'Iowa', 'Texas', 'California', 'Chicago', 'Boston', 'Detroit',
        'Watergraafsmeer', 'Haaften', 'Culemborg', 'Breda', 'Naarden'
    }

    names = [w for w in words if w not in stop_words and w not in place_names]

    # If we found names, return them
    if names:
        logger.info(f"Extracted potential person names from query: {names}")
        return names

    return None
```

---

## Phase 5: Update search_source_text Tool

### 5.1 Remove redundant length filtering

**File**: `genealogy/services/genealogy_tools.py`

**Location**: `search_source_text()` method

**Changes**:
Already done in previous fix - the HybridRetriever now handles tier filtering, so we just need to ensure the comment is accurate:

```python
# Use the hybrid retriever
# Request more chunks than needed since we'll filter by chunk_type
# Note: HybridRetriever filters by search_tier (semantic queries only search narrative tier)
retriever = HybridRetriever()
chunks = retriever.retrieve(query=query, top_k=max_results * 3, expand_window=0)
```

---

## Phase 6: Testing & Validation

### 6.1 Create test fixtures

**File**: `genealogy/tests/fixtures/hierarchical_chunks.py` (new file)

**Content**: Sample chunks for both tiers to use in tests

### 6.2 Unit tests for classification logic

**File**: `genealogy/tests/test_chunk_classification.py` (new file)

**Tests**:
- Short individual_entry (< 200 chars) → metadata
- Long individual_entry (>= 200 chars) → narrative
- Generation headers → metadata
- Biographical text → narrative

### 6.3 Integration tests for tiered search

**File**: `genealogy/tests/test_tiered_search.py` (new file)

**Tests**:
- Semantic query only searches narrative tier
- Name query searches both tiers
- Metadata chunks don't have embeddings
- Place names don't trigger person filter

### 6.4 Manual validation queries

**Test queries**:
```python
# Should search narrative tier only, return only relevant biographical chunks
search_source_text("militaire dienst soldaat leger")
search_source_text("musician flute violin")
search_source_text("emigrated to America")

# Should search both tiers, find metadata stubs too
# (if we make a name-based search tool or update search_person_by_name)
```

### 6.5 Verify embedding reduction

**Command**:
```bash
docker compose exec web python manage.py shell
```

```python
from genealogy.models import TextChunk

# Check tier distribution
metadata_count = TextChunk.objects.filter(search_tier='metadata').count()
narrative_count = TextChunk.objects.filter(search_tier='narrative').count()

print(f"Metadata tier: {metadata_count}")
print(f"Narrative tier: {narrative_count}")

# Check embeddings only on narrative tier
metadata_with_emb = TextChunk.objects.filter(
    search_tier='metadata',
    embedding__isnull=False
).count()
narrative_with_emb = TextChunk.objects.filter(
    search_tier='narrative',
    embedding__isnull=False
).count()

print(f"\nMetadata chunks with embeddings: {metadata_with_emb} (should be 0)")
print(f"Narrative chunks with embeddings: {narrative_with_emb}")
```

---

## Phase 7: Agent Prompt Updates

### 7.1 Update agent system prompts

**Files**:
- `modelfiles/gene-chat-main-agent.Modelfile`
- `modelfiles/gene-chat-fast-agent.Modelfile`
- `modelfiles/gene-reasoner-agent.Modelfile`

**Add explanation**:
```
SEARCH SYSTEM ARCHITECTURE:
The search system uses a two-tier architecture:
- Metadata tier: Short entries with just names/dates (no biographical content)
- Narrative tier: Longer entries with occupations, life events, stories

The search_source_text tool automatically searches the narrative tier for
semantic queries like "Who served in the military?" This means you'll only
get results with actual biographical content, not just stub entries.

For name-based queries, use search_person_by_name which searches both tiers.
```

---

## Phase 8: Documentation

### 8.1 Update README

**File**: `README.md`

**Add section** about the two-tier search architecture

### 8.2 Update API documentation

**File**: `docs/API.md` (if exists)

**Document** the tier system and how it affects search results

### 8.3 Architecture diagram

**File**: `docs/SEARCH_ARCHITECTURE.md` (new)

**Content**: Visual diagram showing metadata vs narrative tiers, which search methods apply to which tier

---

## Rollback Plan

If issues arise, rollback steps:

1. **Revert HybridRetriever changes**: Remove tier filtering from SQL queries
2. **Set all chunks to narrative tier**: `TextChunk.objects.all().update(search_tier='narrative')`
3. **Re-generate embeddings**: Run embedding generation for all chunks
4. **Revert migration**: `python manage.py migrate genealogy <previous_migration_name>`

---

## Success Metrics

### Before Implementation
- Semantic query "militaire dienst soldaat leger": 4/5 results irrelevant stubs
- 300 individual_entry chunks, all embedded (100%)
- Embedding count: ~450 chunks

### After Implementation
- Semantic query returns only narrative chunks with actual military content
- ~250 metadata chunks (55%), 0 embeddings
- ~200 narrative chunks (45%), all embedded
- Embedding count: ~200 chunks (56% reduction)
- Faster vector search (smaller index)
- Higher precision for semantic queries

---

## Timeline Estimate

- Phase 1 (Schema): 30 minutes
- Phase 2 (Classification): 2 hours
- Phase 3 (Embedding cleanup): 30 minutes
- Phase 4 (HybridRetriever): 2 hours
- Phase 5 (Tool update): 15 minutes
- Phase 6 (Testing): 2-3 hours
- Phase 7 (Agent prompts): 30 minutes
- Phase 8 (Documentation): 1 hour

**Total: 8-9 hours of development time**

---

## Dependencies

- Django migrations working
- PostgreSQL database access
- Ollama embeddings service running
- Test framework set up

---

## Questions to Resolve

1. Should we make the 200-char threshold configurable?
2. Should we add a management command to recalculate tiers periodically?
3. Do we want to expose the tier in the search results JSON?
4. Should `get_person_details` also indicate which tier the source_texts come from?

---

## Next Steps

Ready to proceed with Phase 1?
