# Extraction Strategy Design

## Overview

Different book sections require different extraction approaches. This document outlines the strategy pattern for section-specific extraction.

## Architecture

### Strategy Pattern

Each `BookSection.section_type` maps to an extraction strategy:

```
FRONT_MATTER → SkipExtractionStrategy (no processing)
DESCENDANT_GENEALOGY → DescendantGenealogyStrategy (current approach)
KWARTIERSTATEN → AncestorTableStrategy (future)
APPENDIX_NARRATIVE → NarrativeExtractionStrategy (future)
GLOSSARY → SkipExtractionStrategy (no processing)
INDEX → IndexExtractionStrategy (future: 6-column format)
```

### Base Strategy Class

```python
class ExtractionStrategy(ABC):
    """Base class for section-specific extraction strategies"""

    @abstractmethod
    def should_process(self, chunk: TextChunk) -> bool:
        """Determine if this chunk should be processed"""
        pass

    @abstractmethod
    def extract(self, chunk: TextChunk, ollama: OllamaClient, model: str) -> dict:
        """Extract entities from the chunk"""
        pass

    @abstractmethod
    def get_chunk_filter(self) -> dict:
        """Get Django ORM filter for chunks to process"""
        pass
```

## Strategy Implementations

### 1. SkipExtractionStrategy

**Used for**: FRONT_MATTER, GLOSSARY

**Behavior**:
- `should_process()` → Always returns False
- `extract()` → Not called
- `get_chunk_filter()` → Returns empty queryset

**Rationale**: Front matter and glossaries contain no genealogical data to extract.

### 2. DescendantGenealogyStrategy (Current Implementation)

**Used for**: DESCENDANT_GENEALOGY

**Behavior**:
- **Phase 1** (during chunking): Extract people and parent-child relationships deterministically
- **Phase 2** (LLM extraction): Extract events, partnerships, additional people
- Processes only `GENEALOGY_ENTRY` chunks
- Uses complementary Phase 1 + Phase 2 approach

**Chunk Filter**:
```python
chunk_type="GENEALOGY_ENTRY"
```

**Prompt**: Current simplified prompt with Phase 1 context

### 3. IndexExtractionStrategy (Future)

**Used for**: INDEX

**Behavior**:
- Extract person names and page numbers from 6-column index format
- No need for Phase 1/Phase 2 split
- Simple structured extraction: `Person Name → Page Numbers`

**Chunk Filter**:
```python
chunk_type="OTHER"  # Index is formatted differently
```

**Future Prompt**:
```
Extract person names and page numbers from genealogical index.

FORMAT:
- 6 columns per page
- Format: "LastName, FirstName page1, page2, page3"

OUTPUT:
PEOPLE:
- PersonName|PageNumbers

Example:
- Zanten, Jan van|15, 23, 45, 67
- Voorhaar, Geertruij|23, 24, 25
```

### 4. AncestorTableStrategy (Future)

**Used for**: KWARTIERSTATEN (Ahnentafel / Ancestor Tables)

**Behavior**:
- Different genealogical structure (ancestor-focused, not descendant-focused)
- Numbered system (1 = subject, 2 = father, 3 = mother, etc.)
- Extract ancestors and their relationships to subject

**Chunk Filter**:
```python
chunk_type="GENEALOGY_ENTRY" or chunk_type="TABLE"
```

**Future Prompt**: TBD based on Kwartierstaten format

### 5. NarrativeExtractionStrategy (Future)

**Used for**: APPENDIX_NARRATIVE

**Behavior**:
- Extract people, places, and events from narrative text
- No structured genealogical format
- Focus on story extraction, not genealogical facts

**Chunk Filter**:
```python
chunk_type="NARRATIVE"
```

**Future Prompt**: TBD based on narrative format

## Implementation Plan

### Phase 1: Create Strategy Infrastructure (Now)

1. Create `genealogy/extraction_strategies/` package
2. Create base `ExtractionStrategy` class
3. Create `SkipExtractionStrategy`
4. Create `DescendantGenealogyStrategy` (refactor current code)
5. Create `StrategyRegistry` to map section types to strategies

### Phase 2: Update Extraction Task

1. Modify `extract_entities_from_chunks()` to:
   - Group chunks by BookSection
   - Select appropriate strategy for each section
   - Apply strategy-specific filtering and extraction

### Phase 3: Add Index Strategy (Future)

1. Implement `IndexExtractionStrategy`
2. Create index-specific prompt
3. Test on index pages (94-101)

### Phase 4: Add Other Strategies (Future)

As needed for different book types

## Benefits

1. **Extensibility**: Easy to add new section types and extraction approaches
2. **Maintainability**: Each strategy is self-contained
3. **Flexibility**: Different sections can use different models, prompts, chunk filters
4. **Testing**: Each strategy can be tested independently
5. **Configuration**: Book sections define which strategy to use

## Migration Path

1. ✅ Current state: Single extraction approach, manual BookSection filtering
2. → Create strategy infrastructure (next step)
3. → Migrate current code to DescendantGenealogyStrategy
4. → Add SkipExtractionStrategy
5. → Future: Add IndexExtractionStrategy when needed
6. → Future: Add other strategies as needed

## Code Location

```
genealogy/extraction_strategies/
  __init__.py              # Registry and exports
  base.py                  # ExtractionStrategy base class
  skip.py                  # SkipExtractionStrategy
  descendant_genealogy.py  # DescendantGenealogyStrategy (current approach)
  index.py                 # IndexExtractionStrategy (future)
  ancestor_table.py        # AncestorTableStrategy (future)
  narrative.py             # NarrativeExtractionStrategy (future)
```

## Example Usage

```python
from genealogy.extraction_strategies import StrategyRegistry

# Get document and sections
document = Document.objects.get(id=document_id)
sections = document.book_sections.all()

for section in sections:
    # Get strategy for this section type
    strategy = StrategyRegistry.get_strategy(section.section_type)

    # Get chunks to process for this section
    chunks = document.text_chunks.filter(
        start_page__gte=section.start_page,
        start_page__lte=section.end_page,
        entities_extracted=False,
        **strategy.get_chunk_filter()
    )

    # Process each chunk with the strategy
    for chunk in chunks:
        if strategy.should_process(chunk):
            result = strategy.extract(chunk, ollama, model)
```
