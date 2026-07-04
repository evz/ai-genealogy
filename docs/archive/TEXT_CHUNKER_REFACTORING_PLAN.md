# Text Chunker Refactoring Plan

## Current State
- **951 lines** in single file
- **18 methods/functions**
- Largest methods:
  - `_chunk_main_flow`: 229 lines (24% of file!)
  - `save_chunks_to_db`: 136 lines
  - `_extract_out_of_flow_content`: 96 lines
  - `detect_info_box`: 67 lines

## Problems
1. **Single file is too large** - hard to navigate
2. **_chunk_main_flow is massive** - 229 lines handling all chunk types
3. **Mixed responsibilities** - parsing, chunking, entity extraction, DB persistence all in one file
4. **Pattern duplication** - Similar chunk creation code repeated many times

## Proposed Refactoring

### Option A: Split by Responsibility (Recommended)

**Create 4 files:**

1. **`genealogy/chunking/models.py`** (70 lines)
   - `ChunkType` enum
   - `BoundingBox` dataclass
   - `GroundingToken` dataclass
   - `TextChunk` dataclass

2. **`genealogy/chunking/parser.py`** (150 lines)
   - `parse_grounding_tokens()` - extract tokens from OCR
   - `detect_chunk_type()` - classify token types
   - `is_source_citation()`, `is_biographical_text()` - helpers
   - `detect_info_box()` - info box detection

3. **`genealogy/chunking/chunker.py`** (450 lines)
   - `GenealogicalTextChunker` class
   - `chunk()` - main entry point
   - `_extract_out_of_flow_content()` - Pass 1
   - `_chunk_main_flow()` - Pass 2 (still large, see Option B)
   - `extract_person_from_individual_entry()` - name extraction
   - `extract_entities_from_chunk()` - Phase 1 extraction

4. **`genealogy/chunking/persistence.py`** (280 lines)
   - `_get_page_numbers_from_tokens()` - page mapping
   - `save_chunks_to_db()` - DB persistence

**Package structure:**
```
genealogy/chunking/
  __init__.py  # Export main classes
  models.py
  parser.py
  chunker.py
  persistence.py
```

**Benefits:**
- Clear separation of concerns
- Each file < 500 lines
- Easy to test individual components
- Future: could add `kwartierstaten_chunker.py` for different section types

### Option B: Extract Chunk Handlers from _chunk_main_flow

The `_chunk_main_flow` method has repeated patterns for each chunk type:

```python
# Pattern repeated 7+ times:
if chunk_type == ChunkType.X:
    # Collect tokens
    # Create chunk with context
    # Append to chunks
    # Update index
    continue
```

**Refactor into:**

```python
class ChunkHandler:
    """Base class for handling specific chunk types"""
    def can_handle(self, chunk_type: ChunkType) -> bool:
        ...

    def create_chunk(self, token: GroundingToken, tokens: List, index: int, context) -> Tuple[TextChunk, int]:
        ...

class IndividualEntryHandler(ChunkHandler):
    ...

class SourceCitationHandler(ChunkHandler):
    ...

# In _chunk_main_flow:
handlers = [
    IndividualEntryHandler(),
    SourceCitationHandler(),
    FamilyGroupHeaderHandler(),
    ...
]

for handler in handlers:
    if handler.can_handle(chunk_type):
        chunk, new_index = handler.create_chunk(token, tokens, i, context)
        chunks.append(chunk)
        i = new_index
        break
```

**Benefits:**
- `_chunk_main_flow` reduces from 229 → ~50 lines
- Each handler is testable
- Easy to add new chunk types
- **Downside:** More files, more abstraction

### Option C: Extract Just save_chunks_to_db

Simplest option - just move the DB persistence code out.

**Benefits:**
- Quick win
- `text_chunker.py` focuses on chunking logic only
- DB code separate

**Downside:**
- Still leaves a 815-line file

## Recommendation

**Start with Option A** - split by responsibility into 4 files. This gives:
- Immediate improvement (4 files < 500 lines each vs 1 file > 900 lines)
- Clear boundaries
- Can add Option B (handlers) later if `_chunk_main_flow` is still too complex

**Then consider Option B** if we need to add more chunk types or if maintenance is difficult.

## Migration Plan

1. Create `genealogy/chunking/` package
2. Move dataclasses to `models.py`
3. Move parsing functions to `parser.py`
4. Move DB code to `persistence.py`
5. Keep main chunker in `chunker.py`
6. Update imports in:
   - `tasks/chunking.py`
   - Anywhere else that imports from `text_chunker`
7. Delete old `text_chunker.py`
8. Run tests

## Estimated Effort
- 30-45 minutes for Option A
- +60 minutes for Option B (handler pattern)
