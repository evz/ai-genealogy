# LLM Extraction Improvement Plan

## Current State Analysis

### Phase 1: Deterministic Extraction (Already Complete)
From the text chunker, we already extract:
- **extracted_people**: Names from individual entries + their parents from family group context
- **extracted_relationships**: Parent-child relationships (parent → child)
- **extracted_events**: Empty (reserved for Phase 2)

Example from Chunk 249:
- People: `['Marretje (Marritje', 'Aart van Zanten', 'Anna Antonia Kappers']`
- Relationships: Parent-child links for both parents
- Family Group Context: `'VI.1. Kinderen van Aart van Zanten en Anna Antonia Kappers (V.1.a):'`

### Phase 2: LLM Extraction (Current Implementation)
Location: `genealogy/tasks/extraction.py` + `genealogy/prompts/extraction.py`

**Current approach:**
1. Only processes `GENEALOGY_ENTRY` chunks
2. Extracts from scratch (ignores Phase 1 data)
3. Provides family group context but doesn't use Phase 1 people/relationships
4. Overwrites Phase 1 data completely

**Current prompt provides:**
- `chunk.text_content` - the raw text
- `family_group` - first item from family_groups array

## Problems with Current Approach

1. **Duplicate Work**: LLM re-extracts parent names and parent-child relationships already extracted in Phase 1
2. **Lost Context**: Phase 1 already did the hard work of parsing family group headers - LLM doesn't benefit from this structured data
3. **Overwriting Good Data**: Phase 1 deterministic extraction is highly accurate for parent-child relationships, but Phase 2 LLM might make mistakes and overwrite it
4. **Missing Rich Context**: We have `generation_number`, `generation_header`, and `family_groups` that could help the LLM

## Proposed Improvements

### Option A: Complementary Extraction (Recommended)
**Phase 1 extracts:**
- Individual names from "a. Name" patterns
- Parent names from family group headers
- Parent-child relationships (deterministic, high accuracy)

**Phase 2 LLM extracts:**
- Events (births, deaths, marriages, occupations, residence)
- Spouse relationships
- Additional people mentioned in narrative (siblings, witnesses, etc.)
- Better name parsing (handle nicknames, alternate spellings)

**Changes needed:**
1. Update prompt to show Phase 1 extracted data as "ALREADY EXTRACTED"
2. Tell LLM to focus on events and partnerships
3. Merge Phase 2 results WITH Phase 1 data instead of overwriting
4. Provide full genealogical context: generation_number, family_groups, etc.

### Option B: LLM-Only Extraction
**Phase 1:** Just chunk text (no extraction)
**Phase 2:** LLM does everything

**Pros:** Simpler architecture
**Cons:**
- Slower (LLM processing everything)
- More expensive (more tokens)
- Less accurate (LLM might misparse family headers)
- Harder to debug extraction issues

### Option C: Hybrid Verification
Keep both extractions separate and use them to verify each other.
**Too complex for now** - recommend starting with Option A.

## Recommended Changes

### 1. Update Prompt to Provide Rich Context

```python
def build_extraction_prompt(chunk, examples=None):
    generation_info = f"Generation {chunk.generation_number}" if chunk.generation_number else "Unknown generation"
    family_group = chunk.family_groups[0] if chunk.family_groups else "None"

    # Show Phase 1 extracted data
    phase1_people = ", ".join(chunk.extracted_people) if chunk.extracted_people else "None"
    phase1_rels = len(chunk.extracted_relationships) if chunk.extracted_relationships else 0

    prompt = f"""You are extracting genealogical information from Dutch genealogy text.

GENEALOGICAL CONTEXT:
- Generation: {generation_info}
- Family Group: {family_group}

ALREADY EXTRACTED (Phase 1 - DO NOT RE-EXTRACT):
- People: {phase1_people}
- Parent-child relationships: {phase1_rels} relationships already captured

YOUR TASK (Phase 2):
Focus on extracting:
1. EVENTS: births, deaths, marriages, baptisms, burials, occupations, residences
2. PARTNERSHIPS: spouse relationships (marriages)
3. Any additional people mentioned but not in Phase 1 list

CONTENT TO EXTRACT FROM:
{chunk.text_content}
...
```

### 2. Update Extraction Logic to Merge Results

```python
def extract_entities_from_chunk(chunk, ollama, model):
    # ... existing code ...

    # Parse LLM output
    extracted_data = parse_extraction_output(response)

    # MERGE with Phase 1 data instead of overwriting
    # Keep Phase 1 people and relationships, add LLM's discoveries
    all_people = list(set(chunk.extracted_people + extracted_data['people']))
    all_relationships = chunk.extracted_relationships + extracted_data['parent_child'] + extracted_data['partnerships']

    # Events are new (Phase 1 doesn't extract events)
    chunk.extracted_people = all_people
    chunk.extracted_relationships = all_relationships
    chunk.extracted_events = extracted_data['events']
    chunk.entities_extracted = True
    chunk.save(...)
```

### 3. Improve Prompt Instructions

Add clearer instructions about:
- Don't re-extract people already listed
- Don't re-extract parent-child relationships (already done)
- Focus on dates, places, occupations, marriage info
- Extract spouse names from marriage indicators (x, tr., getrouwd)

### 4. Consider Adding Generation Header Context

The generation header (e.g., "ZESDE GENERATIE") provides useful context. Include it in the prompt.

## Migration Path

1. **First:** Update prompt to provide Phase 1 context (non-breaking change)
2. **Then:** Update merge logic to preserve Phase 1 data
3. **Test:** Run on sample chunks and compare quality
4. **Deploy:** Roll out to full document

## Expected Benefits

- **Faster:** LLM focuses on events/partnerships, not redundant name extraction
- **More accurate:** Deterministic extraction for parent-child, LLM for complex events
- **Better debugging:** Can see what Phase 1 vs Phase 2 extracted
- **Cheaper:** Fewer tokens needed when LLM has context
