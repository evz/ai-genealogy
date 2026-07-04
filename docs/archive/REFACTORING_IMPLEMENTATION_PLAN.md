# Refactoring Implementation Plan

## Executive Summary

This document provides a detailed implementation plan for refactoring the OCR, chunking, and extraction code in the genealogy_extractor project. The audit identified critical code duplication, dead code, and complexity hotspots that impact maintainability.

**Key Issues:**
- 100 lines of duplicated chunking logic
- Duplicated extraction prompt code
- 318 lines of dead code (legacy Tesseract)
- Complex methods mixing business logic with infrastructure
- Name collisions and unclear interfaces

**Approach:**
- Phased implementation (7 priorities)
- Incremental changes with testing at each step
- Maintain backward compatibility during migration
- Extract reusable utilities before deletion

**Timeline:** 3-4 weeks for all priorities (can be done incrementally)

---

## Priority 1: Extract Shared Chunking Utilities

### Problem
100 lines of code duplicated between:
- `genealogy/chunking/chunker.py` (GenealogicalTextChunker)
- `genealogy/chunking_strategies/descendant_genealogy.py` (DescendantGenealogyChunkingStrategy)

Specifically:
- `_extract_out_of_flow_content()` (~75 lines)
- `_chunk_main_flow()` (~45 lines)

### Root Cause
DescendantGenealogyChunkingStrategy was created by copying code from GenealogicalTextChunker instead of reusing it.

### Solution
Create a shared `ChunkingEngine` class that both can use.

### Implementation Steps

#### Step 1: Create ChunkingEngine Class
**File:** `genealogy/chunking/engine.py` (new file)

```python
"""Shared chunking engine for processing grounded OCR tokens"""

from typing import List, Dict, Any, Optional
from genealogy.models import OCRPage


class ChunkingEngine:
    """Reusable chunking logic for grounded token processing"""

    def __init__(self, page: OCRPage):
        """Initialize with OCR page data

        Args:
            page: OCRPage with grounded_tokens
        """
        self.page = page
        self.grounded_tokens = page.grounded_tokens or []

    def extract_out_of_flow_content(
        self,
        inverted_regions: List[Dict[str, Any]],
        target_types: Optional[List[str]] = None
    ) -> tuple[List[str], List[int]]:
        """Extract content that should be chunked separately

        Args:
            inverted_regions: List of inverted content regions
            target_types: Element types to extract (default: ['image', 'figure', 'table'])

        Returns:
            (extracted_content, used_token_indices)
        """
        if target_types is None:
            target_types = ['image', 'figure', 'table']

        extracted = []
        used_indices = []

        # Implementation extracted from GenealogicalTextChunker
        # ... (exact code from chunker.py:96-194)

        return extracted, used_indices

    def chunk_main_flow(
        self,
        handlers: List['ChunkHandler'],
        used_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """Process main flow tokens using provided handlers

        Args:
            handlers: List of ChunkHandler instances (ordered by priority)
            used_indices: Token indices already used (will be skipped)

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_handler = None
        current_chunk_tokens = []

        # Implementation extracted from GenealogicalTextChunker
        # ... (exact code from chunker.py:196-244)

        return chunks
```

#### Step 2: Create ChunkHandler Base Class
**File:** `genealogy/chunking/handlers.py` (update existing)

Add base class that standardizes handler interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ChunkHandler(ABC):
    """Base class for chunk handlers"""

    @abstractmethod
    def can_handle(self, token: Dict[str, Any]) -> bool:
        """Check if this handler can process the token"""
        pass

    @abstractmethod
    def process_token(self, token: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process token and return extracted metadata (or None)"""
        pass

    @abstractmethod
    def should_end_chunk(self, token: Dict[str, Any]) -> bool:
        """Check if this token should end the current chunk"""
        pass

    @property
    @abstractmethod
    def chunk_type(self) -> str:
        """The chunk type this handler produces"""
        pass
```

#### Step 3: Refactor GenealogicalTextChunker
**File:** `genealogy/chunking/chunker.py`

```python
"""Genealogical text chunker using shared engine"""

from .engine import ChunkingEngine
from .handlers import (
    GenerationHeaderHandler,
    FamilyGroupHeaderHandler,
    IndividualEntryHandler,
    SourceCitationHandler
)


class GenealogicalTextChunker:
    """Chunks genealogical text using grounding tokens"""

    def __init__(self, page):
        self.page = page
        self.engine = ChunkingEngine(page)

        # Initialize handlers (order matters!)
        self.handlers = [
            GenerationHeaderHandler(),
            FamilyGroupHeaderHandler(),
            IndividualEntryHandler(),
            SourceCitationHandler(),
        ]

    def chunk(self):
        """Chunk the page into structured sections"""
        inverted_regions = self._detect_inverted_regions()

        # Extract out-of-flow content (images, info boxes)
        out_of_flow, used_indices = self.engine.extract_out_of_flow_content(
            inverted_regions
        )

        # Chunk main flow
        main_flow_chunks = self.engine.chunk_main_flow(
            self.handlers,
            used_indices
        )

        return {
            'out_of_flow': out_of_flow,
            'main_flow': main_flow_chunks
        }

    def _detect_inverted_regions(self):
        """Detect inverted content regions (existing logic)"""
        # Keep existing implementation
        pass
```

#### Step 4: Refactor DescendantGenealogyChunkingStrategy
**File:** `genealogy/chunking_strategies/descendant_genealogy.py`

```python
"""Descendant genealogy chunking strategy"""

from genealogy.chunking.engine import ChunkingEngine
from genealogy.chunking.handlers import (
    GenerationHeaderHandler,
    FamilyGroupHeaderHandler,
    IndividualEntryHandler,
    SourceCitationHandler
)
from .base import ChunkingStrategy


class DescendantGenealogyChunkingStrategy(ChunkingStrategy):
    """Strategy for chunking descendant genealogy sections"""

    @property
    def strategy_name(self) -> str:
        return "Descendant Genealogy Chunking"

    def chunk_section(self, pages, start_page, end_page):
        """Chunk pages in descendant genealogy section"""
        all_chunks = []

        for page in pages:
            engine = ChunkingEngine(page)

            # Initialize handlers
            handlers = [
                GenerationHeaderHandler(),
                FamilyGroupHeaderHandler(),
                IndividualEntryHandler(),
                SourceCitationHandler(),
            ]

            # Extract out-of-flow
            inverted_regions = self._detect_inverted_regions(page)
            out_of_flow, used_indices = engine.extract_out_of_flow_content(
                inverted_regions
            )

            # Save out-of-flow chunks
            for content in out_of_flow:
                chunk = self._create_chunk(page, content, 'OUT_OF_FLOW')
                all_chunks.append(chunk)

            # Chunk main flow
            main_flow_chunks = engine.chunk_main_flow(handlers, used_indices)

            # Save main flow chunks
            for chunk_data in main_flow_chunks:
                chunk = self._create_chunk(
                    page,
                    chunk_data['text'],
                    chunk_data['chunk_type'],
                    metadata=chunk_data
                )
                all_chunks.append(chunk)

        return all_chunks
```

#### Step 5: Update Tests
**File:** `genealogy/tests/test_chunking_engine.py` (new file)

```python
"""Tests for shared chunking engine"""

import pytest
from genealogy.chunking.engine import ChunkingEngine
from genealogy.models import OCRPage


class TestChunkingEngine:
    """Test ChunkingEngine reusable logic"""

    def test_extract_out_of_flow_content(self, sample_page_with_images):
        """Test out-of-flow content extraction"""
        engine = ChunkingEngine(sample_page_with_images)

        inverted_regions = [
            {'bbox': [100, 100, 200, 200], 'type': 'image'}
        ]

        content, indices = engine.extract_out_of_flow_content(inverted_regions)

        assert len(content) > 0
        assert len(indices) > 0

    def test_chunk_main_flow(self, sample_page, sample_handlers):
        """Test main flow chunking with handlers"""
        engine = ChunkingEngine(sample_page)

        chunks = engine.chunk_main_flow(sample_handlers, used_indices=[])

        assert len(chunks) > 0
        assert all('chunk_type' in c for c in chunks)
```

Update existing tests to verify both chunker and strategy produce same results.

### Testing Strategy
1. Create unit tests for ChunkingEngine in isolation
2. Run existing chunker tests to ensure no regression
3. Run existing strategy tests to ensure no regression
4. Add integration test comparing chunker vs strategy output (should be identical)

### Rollback Plan
If issues arise:
1. Keep old code in place during migration
2. Use feature flag to toggle between old/new implementation
3. Can revert by removing engine.py and restoring original methods

### Estimated Effort
- **Development:** 4-6 hours
- **Testing:** 2-3 hours
- **Review/Deploy:** 1 hour
- **Total:** ~1 day

### Success Metrics
- Zero code duplication between chunker and strategy
- All existing tests pass
- Code coverage maintained or improved
- LOC reduced by ~100 lines

---

## Priority 2: Consolidate Extraction Prompts

### Problem
Extraction prompt code duplicated between:
- `genealogy/prompts/extraction.py` (212 lines)
- `genealogy/extraction_strategies/descendant_genealogy.py` (344 lines)

Duplicated elements:
- `DUTCH_ABBREVIATIONS` constant
- `EVENT_TYPE_CODES` constant
- `build_extraction_prompt()` function
- `parse_extraction_output()` function

### Root Cause
Unclear whether `prompts/extraction.py` or the strategy should be the source of truth.

### Solution
Make `prompts/extraction.py` the single source of truth. Strategy imports and uses these utilities.

### Implementation Steps

#### Step 1: Verify prompts/extraction.py is Complete
**File:** `genealogy/prompts/extraction.py`

Ensure it has everything needed:
- ✓ DUTCH_ABBREVIATIONS
- ✓ EVENT_TYPE_CODES
- ✓ build_extraction_prompt(chunk, examples=None)
- ✓ parse_extraction_output(output_text)
- ✓ load_examples()

#### Step 2: Refactor DescendantGenealogyStrategy
**File:** `genealogy/extraction_strategies/descendant_genealogy.py`

```python
"""Descendant genealogy extraction strategy"""

import logging
from typing import Dict, Any

from genealogy.prompts.extraction import (
    DUTCH_ABBREVIATIONS,
    EVENT_TYPE_CODES,
    build_extraction_prompt,
    parse_extraction_output,
    load_examples
)
from .base import ExtractionStrategy

logger = logging.getLogger(__name__)


class DescendantGenealogyStrategy(ExtractionStrategy):
    """Strategy for descendant genealogy sections"""

    def __init__(self):
        """Initialize strategy and load examples"""
        self._examples = None

    @property
    def strategy_name(self) -> str:
        return "Descendant Genealogy Extraction"

    def should_process(self, chunk) -> bool:
        """Process GENEALOGY_ENTRY chunks that have genealogical context"""
        return chunk.chunk_type == "GENEALOGY_ENTRY"

    def get_chunk_filter(self) -> Dict[str, Any]:
        """Filter for GENEALOGY_ENTRY chunks only"""
        return {"chunk_type": "GENEALOGY_ENTRY"}

    def extract(self, chunk, ollama, model: str) -> Dict[str, Any]:
        """Extract entities from a descendant genealogy chunk using LLM"""
        try:
            logger.info(f"Extracting from chunk {chunk.sequence_number}")

            # Build extraction prompt using shared utility
            if self._examples is None:
                self._examples = load_examples()

            prompt = build_extraction_prompt(chunk, self._examples)

            # Calculate required context window
            estimated_tokens = len(chunk.text_content) // 4 + 4000
            num_ctx = max(4096, min(131072, 2 ** (estimated_tokens - 1).bit_length()))

            if num_ctx > 8192:
                logger.info(f"Large chunk: using context window {num_ctx} tokens")

            # Query LLM
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'num_ctx': num_ctx,
                    'temperature': 0.0,
                }
            )

            if not response:
                logger.error(f"No response from LLM for chunk {chunk.sequence_number}")
                return {'success': False, 'error': 'No response from LLM'}

            # Parse output using shared utility
            extracted_data = parse_extraction_output(response)

            # Merge Phase 2 (LLM) data with Phase 1 (deterministic) data
            all_people = self._merge_people(chunk, extracted_data)
            all_relationships = self._merge_relationships(chunk, extracted_data)
            all_events = extracted_data['events']

            # Save merged data
            chunk.extracted_people = all_people
            chunk.extracted_relationships = all_relationships
            chunk.extracted_events = all_events
            chunk.entities_extracted = True
            chunk.save(update_fields=[
                'extracted_people',
                'extracted_relationships',
                'extracted_events',
                'entities_extracted'
            ])

            result = {
                'success': True,
                'people_count': len(all_people),
                'relationships_count': len(all_relationships),
                'events_count': len(all_events),
                'phase1_people': len(chunk.extracted_people or []),
                'phase2_people_added': len(all_people) - len(chunk.extracted_people or []),
            }

            logger.info(
                f"Chunk {chunk.sequence_number}: "
                f"{result['people_count']} people, "
                f"{result['relationships_count']} relationships, "
                f"{result['events_count']} events"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk.sequence_number}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _merge_people(self, chunk, extracted_data):
        """Merge Phase 1 and Phase 2 people lists"""
        phase1_people = chunk.extracted_people or []
        phase2_people = extracted_data['people']

        all_people = phase1_people.copy()
        for person in phase2_people:
            if person not in all_people:
                all_people.append(person)

        return all_people

    def _merge_relationships(self, chunk, extracted_data):
        """Merge Phase 1 and Phase 2 relationships"""
        phase1_relationships = chunk.extracted_relationships or []
        phase2_parent_child = extracted_data['parent_child']
        phase2_partnerships = extracted_data['partnerships']

        all_relationships = phase1_relationships.copy()
        for rel in phase2_parent_child + phase2_partnerships:
            # Check for duplicates
            is_duplicate = any(
                existing['person1'] == rel['person1'] and
                existing['relationship_type'] == rel['relationship_type'] and
                existing['person2'] == rel['person2']
                for existing in all_relationships
            )
            if not is_duplicate:
                all_relationships.append(rel)

        return all_relationships
```

#### Step 3: Remove Duplicated Code
Delete from `descendant_genealogy.py`:
- Lines 17-28 (DUTCH_ABBREVIATIONS)
- Lines 30-43 (EVENT_TYPE_CODES)
- Lines 58-68 (_load_examples method)
- Lines 82-160 (build_prompt method - replaced by imported function)
- Lines 162-237 (parse_output method - replaced by imported function)

#### Step 4: Update Tests
**File:** `genealogy/tests/test_extraction_prompts.py` (update)

```python
"""Tests for extraction prompt utilities"""

import pytest
from genealogy.prompts.extraction import (
    build_extraction_prompt,
    parse_extraction_output,
    DUTCH_ABBREVIATIONS,
    EVENT_TYPE_CODES
)


class TestExtractionPrompts:
    """Test prompt building and parsing utilities"""

    def test_build_extraction_prompt_structure(self, sample_chunk):
        """Test prompt contains all required sections"""
        prompt = build_extraction_prompt(sample_chunk)

        assert 'DUTCH GENEALOGICAL ABBREVIATIONS' in prompt
        assert 'EVENT TYPE CODES' in prompt
        assert 'EXAMPLES:' in prompt
        assert 'YOUR TASK' in prompt
        assert 'ALREADY EXTRACTED (Phase 1)' in prompt
        assert 'CONTENT TO EXTRACT FROM' in prompt

    def test_parse_extraction_output_people(self):
        """Test parsing people section"""
        output = """
PEOPLE:
- Jan van Zanten
- Maria Pieters

PARENT_CHILD:
None

PARTNERSHIPS:
None

EVENTS:
None
"""
        result = parse_extraction_output(output)

        assert result['people'] == ['Jan van Zanten', 'Maria Pieters']
        assert result['parent_child'] == []
        assert result['partnerships'] == []
        assert result['events'] == []

    def test_parse_extraction_output_events(self):
        """Test parsing events section"""
        output = """
PEOPLE:
None

PARENT_CHILD:
None

PARTNERSHIPS:
None

EVENTS:
- Jan van Zanten|BIRT|12 maart 1850|Amsterdam
- Jan van Zanten|OCCU|1875|metselaar
"""
        result = parse_extraction_output(output)

        assert len(result['events']) == 2
        assert result['events'][0]['person'] == 'Jan van Zanten'
        assert result['events'][0]['event_type'] == 'BIRT'
        assert result['events'][1]['event_type'] == 'OCCU'
```

### Testing Strategy
1. Run extraction tests with mocked LLM responses
2. Verify parsing logic handles all edge cases
3. Integration test with real chunk data
4. Verify Phase 1 + Phase 2 merging still works correctly

### Rollback Plan
- Keep old methods commented out temporarily
- Can restore by uncommenting and removing imports

### Estimated Effort
- **Development:** 2-3 hours
- **Testing:** 1-2 hours
- **Review/Deploy:** 30 min
- **Total:** ~4-5 hours

### Success Metrics
- Zero duplication of prompt code
- All extraction tests pass
- LOC reduced by ~100 lines

---

## Priority 3: Delete Dead Code

### Problem
`genealogy/region_ocr_processor.py` (318 lines) is legacy Tesseract code that's no longer used.

### Root Cause
Migrated to DeepSeek-OCR but didn't remove old code.

### Solution
Delete the file after verifying it's truly unused.

### Implementation Steps

#### Step 1: Verify No Imports
**Command:**
```bash
cd /home/eric/code/genealogy_extractor
grep -r "region_ocr_processor" --include="*.py" .
grep -r "RegionOCRProcessor" --include="*.py" .
```

Expected: No imports found (already verified in audit).

#### Step 2: Check Git History
**Command:**
```bash
git log --all --oneline -- genealogy/region_ocr_processor.py
```

Document when it was last modified and by which commit.

#### Step 3: Archive Before Delete
**Command:**
```bash
mkdir -p docs/archive
git show HEAD:genealogy/region_ocr_processor.py > docs/archive/region_ocr_processor.py.backup
echo "Archived from commit $(git rev-parse HEAD) on $(date)" >> docs/archive/region_ocr_processor.py.backup
```

#### Step 4: Delete File
**Command:**
```bash
git rm genealogy/region_ocr_processor.py
git commit -m "Remove legacy Tesseract OCR code (replaced by DeepSeek-OCR)

This file contained the old RegionOCRProcessor that used Tesseract for
OCR. We've migrated to DeepSeek-OCR with grounding tokens, making this
code obsolete.

Archived to docs/archive/region_ocr_processor.py.backup for reference.
"
```

#### Step 5: Update Documentation
**File:** `docs/DESIGN_LESSONS_LEARNED.md`

Add entry:
```markdown
## Legacy OCR Approaches

### Tesseract with Region Detection
**When:** Initial implementation (pre-DeepSeek)
**What:** RegionOCRProcessor that detected regions, then ran Tesseract
**Why it didn't work:**
- Tesseract doesn't understand document structure
- Required complex heuristics for layout
- No semantic understanding of text regions

**What replaced it:**
- DeepSeek-OCR with grounding tokens
- Layout detection via DocLayout-YOLO
- Semantic types embedded in OCR output

**Code removed:** genealogy/region_ocr_processor.py (commit XXXXX)
```

### Testing Strategy
1. Run full test suite to ensure no hidden dependencies
2. Run full pipeline to ensure OCR still works
3. Check for any dynamic imports (`importlib`, `__import__`)

### Rollback Plan
- File archived in `docs/archive/`
- Can restore with: `git show <commit>:genealogy/region_ocr_processor.py > genealogy/region_ocr_processor.py`

### Estimated Effort
- **Development:** 30 min
- **Testing:** 1 hour
- **Review/Deploy:** 15 min
- **Total:** ~2 hours

### Success Metrics
- File deleted from codebase
- All tests still pass
- LOC reduced by 318 lines

---

## Priority 4: Create Service Layer

### Problem
Business logic mixed with Django ORM and Celery dependencies, making testing difficult.

Examples:
- `genealogy/tasks/extraction.py` (Celery task)
- `genealogy/tasks/chunking.py` (Celery task)
- Entity creation logic embedded in management commands

### Root Cause
Initial implementation focused on "make it work" without separating concerns.

### Solution
Extract pure business logic into service classes that are testable in isolation.

### Implementation Steps

#### Step 1: Create Service Layer Structure
**Files:**
```
genealogy/services/
├── __init__.py
├── chunking_service.py
├── extraction_service.py
└── entity_service.py
```

#### Step 2: Create ChunkingService
**File:** `genealogy/services/chunking_service.py`

```python
"""Chunking service - pure business logic (no Django/Celery)"""

from typing import List, Dict, Any, Optional
from genealogy.chunking_strategies import get_strategy_for_section


class ChunkingService:
    """Service for chunking OCR pages into text chunks"""

    def chunk_section(
        self,
        section_type: str,
        pages: List[Any],  # List of OCRPage objects
        start_page: int,
        end_page: int
    ) -> List[Dict[str, Any]]:
        """Chunk a book section using appropriate strategy

        Args:
            section_type: Type of section (DESCENDANT_GENEALOGY, etc.)
            pages: OCRPage objects to chunk
            start_page: Starting page number
            end_page: Ending page number

        Returns:
            List of chunk dictionaries ready for DB save
        """
        strategy = get_strategy_for_section(section_type)

        if not strategy:
            raise ValueError(f"No chunking strategy for section type: {section_type}")

        chunks = strategy.chunk_section(pages, start_page, end_page)

        return chunks

    def chunk_pages(
        self,
        pages: List[Any],
        default_strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Chunk arbitrary pages (no section context)

        Args:
            pages: OCRPage objects to chunk
            default_strategy: Strategy to use (default: DESCENDANT_GENEALOGY)

        Returns:
            List of chunk dictionaries
        """
        if default_strategy is None:
            default_strategy = "DESCENDANT_GENEALOGY"

        return self.chunk_section(
            section_type=default_strategy,
            pages=pages,
            start_page=pages[0].page_number if pages else 1,
            end_page=pages[-1].page_number if pages else 1
        )
```

#### Step 3: Create ExtractionService
**File:** `genealogy/services/extraction_service.py`

```python
"""Extraction service - pure business logic (no Django/Celery)"""

from typing import Dict, Any, List, Optional
from genealogy.extraction_strategies import get_strategy_for_section


class ExtractionService:
    """Service for extracting entities from text chunks"""

    def __init__(self, ollama_client):
        """Initialize with Ollama client

        Args:
            ollama_client: Ollama client for LLM calls
        """
        self.ollama = ollama_client

    def extract_from_chunk(
        self,
        chunk: Any,  # TextChunk model instance
        model: str = "llama3.1:70b"
    ) -> Dict[str, Any]:
        """Extract entities from a single chunk

        Args:
            chunk: TextChunk to extract from
            model: LLM model to use

        Returns:
            Extraction result dictionary
        """
        # Get appropriate strategy based on section type
        section_type = chunk.book_section.section_type if chunk.book_section else None
        strategy = get_strategy_for_section(section_type)

        if not strategy:
            return {
                'success': False,
                'error': f'No extraction strategy for section: {section_type}'
            }

        if not strategy.should_process(chunk):
            return {
                'success': False,
                'error': f'Strategy {strategy.strategy_name} cannot process chunk type {chunk.chunk_type}'
            }

        # Extract using strategy
        result = strategy.extract(chunk, self.ollama, model)

        return result

    def extract_from_chunks(
        self,
        chunks: List[Any],
        model: str = "llama3.1:70b"
    ) -> Dict[str, Any]:
        """Extract entities from multiple chunks

        Args:
            chunks: List of TextChunk instances
            model: LLM model to use

        Returns:
            Summary of extraction results
        """
        results = {
            'total': len(chunks),
            'success': 0,
            'failed': 0,
            'errors': []
        }

        for chunk in chunks:
            result = self.extract_from_chunk(chunk, model)

            if result.get('success'):
                results['success'] += 1
            else:
                results['failed'] += 1
                results['errors'].append({
                    'chunk_id': chunk.id,
                    'error': result.get('error')
                })

        return results
```

#### Step 4: Create EntityService
**File:** `genealogy/services/entity_service.py`

```python
"""Entity service - pure business logic for entity creation"""

from typing import List, Dict, Any


class EntityService:
    """Service for creating entities from extracted data"""

    def create_person_mention(
        self,
        chunk: Any,  # TextChunk
        person_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a PersonMention from extracted data

        Args:
            chunk: Source chunk
            person_name: Name of person
            metadata: Additional metadata

        Returns:
            PersonMention data dict ready for DB save
        """
        return {
            'text_chunk': chunk,
            'name': person_name,
            'metadata': metadata or {}
        }

    def create_event(
        self,
        person_mention: Any,  # PersonMention
        event_type: str,
        date: Optional[str] = None,
        place: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an Event from extracted data

        Args:
            person_mention: PersonMention this event belongs to
            event_type: Event type code (BIRT, DEAT, etc.)
            date: Event date string
            place: Event place string
            metadata: Additional metadata

        Returns:
            Event data dict ready for DB save
        """
        return {
            'person_mention': person_mention,
            'event_type': event_type,
            'date_string': date or '',
            'place_string': place or '',
            'metadata': metadata or {}
        }

    def create_relationship(
        self,
        person1_mention: Any,
        person2_mention: Any,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a RelationshipMention from extracted data

        Args:
            person1_mention: First PersonMention
            person2_mention: Second PersonMention
            relationship_type: Type (child, spouse, etc.)
            metadata: Additional metadata

        Returns:
            RelationshipMention data dict ready for DB save
        """
        return {
            'person1': person1_mention,
            'person2': person2_mention,
            'relationship_type': relationship_type,
            'metadata': metadata or {}
        }
```

#### Step 5: Refactor Tasks to Use Services
**File:** `genealogy/tasks/chunking.py`

Before:
```python
@shared_task
def chunk_pages_task(document_id):
    # Business logic mixed with task code
    document = Document.objects.get(id=document_id)
    pages = OCRPage.objects.filter(document=document)

    strategy = DescendantGenealogyChunkingStrategy()
    chunks = strategy.chunk_section(pages, 1, 100)

    for chunk_data in chunks:
        TextChunk.objects.create(**chunk_data)
```

After:
```python
@shared_task
def chunk_pages_task(document_id):
    """Celery task wrapper for chunking service"""
    from genealogy.services.chunking_service import ChunkingService
    from genealogy.models import Document, OCRPage, TextChunk

    # Fetch data (Django ORM)
    document = Document.objects.get(id=document_id)
    pages = list(OCRPage.objects.filter(document=document).order_by('page_number'))

    # Business logic (service, testable)
    service = ChunkingService()
    chunks = service.chunk_pages(pages)

    # Save results (Django ORM)
    for chunk_data in chunks:
        TextChunk.objects.create(**chunk_data)

    return {'status': 'success', 'chunks_created': len(chunks)}
```

#### Step 6: Write Service Tests
**File:** `genealogy/tests/test_chunking_service.py`

```python
"""Tests for chunking service (no Django required)"""

import pytest
from genealogy.services.chunking_service import ChunkingService


class MockOCRPage:
    """Mock OCRPage for testing"""
    def __init__(self, page_number, grounded_tokens):
        self.page_number = page_number
        self.grounded_tokens = grounded_tokens


class TestChunkingService:
    """Test ChunkingService without Django dependencies"""

    def test_chunk_section_descendant_genealogy(self):
        """Test chunking with descendant genealogy strategy"""
        service = ChunkingService()

        pages = [
            MockOCRPage(1, [
                {'text': 'I. EERSTE GENERATIE', 'element_type': 'title'},
                {'text': 'a. Jan van Zanten', 'element_type': 'text'}
            ])
        ]

        chunks = service.chunk_section(
            section_type='DESCENDANT_GENEALOGY',
            pages=pages,
            start_page=1,
            end_page=1
        )

        assert len(chunks) > 0
        assert chunks[0]['chunk_type'] in ['GENERATION_HEADER', 'GENEALOGY_ENTRY']

    def test_chunk_section_invalid_type(self):
        """Test error handling for unknown section type"""
        service = ChunkingService()

        with pytest.raises(ValueError, match="No chunking strategy"):
            service.chunk_section('INVALID_TYPE', [], 1, 1)
```

### Testing Strategy
1. Write service unit tests (no Django/Celery dependencies)
2. Write integration tests (services + Django models)
3. Test Celery tasks still work with service layer
4. Verify same output before/after refactoring

### Rollback Plan
- Services are additive (don't break existing code)
- Tasks can be reverted to inline business logic
- Can phase migration one task at a time

### Estimated Effort
- **Development:** 6-8 hours
- **Testing:** 4-5 hours
- **Review/Deploy:** 1-2 hours
- **Total:** ~2 days

### Success Metrics
- Service classes testable without Django
- Test coverage for services ≥80%
- Celery tasks simplified (orchestration only)
- Clear separation of concerns

---

## Priority 5: Simplify Complex Methods

### Problem
Several methods exceed 70 lines with deep nesting:
- `GenealogicalTextChunker._chunk_main_flow()` (48 lines, 4 levels deep)
- `DescendantGenealogyStrategy.extract()` (105 lines)
- Various handler methods

### Root Cause
Incremental feature additions without refactoring.

### Solution
Extract sub-methods with single responsibilities.

### Implementation Steps

#### Step 1: Refactor _chunk_main_flow()
**File:** `genealogy/chunking/chunker.py`

Before (48 lines, complex):
```python
def _chunk_main_flow(self, handlers, used_indices):
    chunks = []
    current_handler = None
    current_chunk_tokens = []

    for idx, token in enumerate(self.grounded_tokens):
        if idx in used_indices:
            continue

        # Find handler
        for handler in handlers:
            if handler.can_handle(token):
                # If switching handlers, save current chunk
                if current_handler and current_handler != handler:
                    if current_chunk_tokens:
                        chunks.append({
                            'text': self._tokens_to_text(current_chunk_tokens),
                            'chunk_type': current_handler.chunk_type,
                            'metadata': current_handler.get_metadata()
                        })
                        current_chunk_tokens = []

                current_handler = handler
                break

        # Process token
        if current_handler:
            current_chunk_tokens.append(token)

        # Check if chunk should end
        if current_handler and current_handler.should_end_chunk(token):
            chunks.append({
                'text': self._tokens_to_text(current_chunk_tokens),
                'chunk_type': current_handler.chunk_type,
                'metadata': current_handler.get_metadata()
            })
            current_chunk_tokens = []
            current_handler = None

    # Save final chunk
    if current_chunk_tokens and current_handler:
        chunks.append({
            'text': self._tokens_to_text(current_chunk_tokens),
            'chunk_type': current_handler.chunk_type,
            'metadata': current_handler.get_metadata()
        })

    return chunks
```

After (split into focused methods):
```python
def _chunk_main_flow(self, handlers, used_indices):
    """Main flow chunking (orchestration only)"""
    chunks = []
    state = ChunkingState()

    for idx, token in enumerate(self.grounded_tokens):
        if idx in used_indices:
            continue

        # Process token with current state
        chunk_completed = self._process_token(token, handlers, state)

        if chunk_completed:
            chunks.append(self._create_chunk_from_state(state))
            state.reset()

    # Save final chunk
    if state.has_tokens():
        chunks.append(self._create_chunk_from_state(state))

    return chunks

def _process_token(self, token, handlers, state):
    """Process a single token, updating state

    Returns:
        bool: True if chunk should be saved
    """
    # Find handler for this token
    handler = self._find_handler(token, handlers)

    # Check if handler changed
    if state.handler and state.handler != handler:
        return True  # Save current chunk

    # Update state
    state.handler = handler
    if handler:
        state.add_token(token)

    # Check if chunk should end
    if handler and handler.should_end_chunk(token):
        return True

    return False

def _find_handler(self, token, handlers):
    """Find first handler that can process this token"""
    for handler in handlers:
        if handler.can_handle(token):
            return handler
    return None

def _create_chunk_from_state(self, state):
    """Create chunk dictionary from current state"""
    return {
        'text': self._tokens_to_text(state.tokens),
        'chunk_type': state.handler.chunk_type,
        'metadata': state.handler.get_metadata()
    }


class ChunkingState:
    """State for chunking process"""
    def __init__(self):
        self.handler = None
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def has_tokens(self):
        return len(self.tokens) > 0

    def reset(self):
        self.handler = None
        self.tokens = []
```

#### Step 2: Refactor DescendantGenealogyStrategy.extract()
**File:** `genealogy/extraction_strategies/descendant_genealogy.py`

Before (105 lines):
```python
def extract(self, chunk, ollama, model):
    try:
        logger.info(f"Extracting from chunk {chunk.sequence_number}")

        # Build prompt
        prompt = self.build_prompt(chunk)

        # Calculate context window
        estimated_tokens = len(chunk.text_content) // 4 + 4000
        num_ctx = max(4096, min(131072, 2 ** (estimated_tokens - 1).bit_length()))

        if num_ctx > 8192:
            logger.info(f"Large chunk: using context window {num_ctx} tokens")

        # Query LLM
        response = ollama.generate(model=model, prompt=prompt, options={'num_ctx': num_ctx})

        if not response:
            return {'success': False, 'error': 'No response from LLM'}

        # Parse
        extracted_data = self.parse_output(response)

        # Merge Phase 1 and Phase 2 data
        phase1_people = chunk.extracted_people or []
        phase2_people = extracted_data['people']
        all_people = phase1_people.copy()
        for person in phase2_people:
            if person not in all_people:
                all_people.append(person)

        # [... 60 more lines of merging logic ...]

        return result
    except Exception as e:
        logger.error(f"Failed to extract: {e}")
        return {'success': False, 'error': str(e)}
```

After (split into focused methods):
```python
def extract(self, chunk, ollama, model):
    """Extract entities from chunk (orchestration only)"""
    try:
        logger.info(f"Extracting from chunk {chunk.sequence_number}")

        # Query LLM
        response = self._query_llm(chunk, ollama, model)
        if not response:
            return {'success': False, 'error': 'No response from LLM'}

        # Parse and merge with Phase 1 data
        extracted_data = parse_extraction_output(response)
        merged_data = self._merge_phases(chunk, extracted_data)

        # Save to database
        self._save_extraction(chunk, merged_data)

        # Build result summary
        return self._build_result_summary(chunk, merged_data)

    except Exception as e:
        logger.error(f"Failed to extract: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}

def _query_llm(self, chunk, ollama, model):
    """Query LLM with appropriate context window"""
    prompt = build_extraction_prompt(chunk, self._examples)
    num_ctx = self._calculate_context_window(chunk)

    if num_ctx > 8192:
        logger.info(f"Large chunk: using {num_ctx} token context")

    return ollama.generate(
        model=model,
        prompt=prompt,
        options={'num_ctx': num_ctx, 'temperature': 0.0}
    )

def _calculate_context_window(self, chunk):
    """Calculate required context window size"""
    estimated_tokens = len(chunk.text_content) // 4 + 4000
    return max(4096, min(131072, 2 ** (estimated_tokens - 1).bit_length()))

def _merge_phases(self, chunk, extracted_data):
    """Merge Phase 1 (deterministic) and Phase 2 (LLM) data"""
    return {
        'people': self._merge_people(chunk, extracted_data),
        'relationships': self._merge_relationships(chunk, extracted_data),
        'events': extracted_data['events']
    }

def _merge_people(self, chunk, extracted_data):
    """Merge Phase 1 and Phase 2 people lists (deduplicated)"""
    phase1 = chunk.extracted_people or []
    phase2 = extracted_data['people']

    all_people = phase1.copy()
    for person in phase2:
        if person not in all_people:
            all_people.append(person)

    return all_people

def _merge_relationships(self, chunk, extracted_data):
    """Merge Phase 1 and Phase 2 relationships (deduplicated)"""
    phase1 = chunk.extracted_relationships or []
    phase2 = extracted_data['parent_child'] + extracted_data['partnerships']

    all_rels = phase1.copy()
    for rel in phase2:
        if not self._is_duplicate_relationship(rel, all_rels):
            all_rels.append(rel)

    return all_rels

def _is_duplicate_relationship(self, rel, existing):
    """Check if relationship already exists"""
    return any(
        e['person1'] == rel['person1'] and
        e['relationship_type'] == rel['relationship_type'] and
        e['person2'] == rel['person2']
        for e in existing
    )

def _save_extraction(self, chunk, merged_data):
    """Save extracted data to chunk"""
    chunk.extracted_people = merged_data['people']
    chunk.extracted_relationships = merged_data['relationships']
    chunk.extracted_events = merged_data['events']
    chunk.entities_extracted = True
    chunk.save(update_fields=[
        'extracted_people',
        'extracted_relationships',
        'extracted_events',
        'entities_extracted'
    ])

def _build_result_summary(self, chunk, merged_data):
    """Build extraction result summary"""
    phase1_people = len(chunk.extracted_people or [])
    phase1_rels = len(chunk.extracted_relationships or [])

    return {
        'success': True,
        'people_count': len(merged_data['people']),
        'relationships_count': len(merged_data['relationships']),
        'events_count': len(merged_data['events']),
        'phase1_people': phase1_people,
        'phase2_people_added': len(merged_data['people']) - phase1_people,
        'phase1_relationships': phase1_rels,
        'phase2_relationships_added': len(merged_data['relationships']) - phase1_rels,
    }
```

### Testing Strategy
1. Ensure all existing tests still pass (behavior unchanged)
2. Add unit tests for new sub-methods
3. Verify complexity metrics improved (cyclomatic complexity, nesting depth)

### Rollback Plan
- Changes are internal to methods (no API changes)
- Can revert by restoring original method bodies

### Estimated Effort
- **Development:** 4-5 hours
- **Testing:** 2-3 hours
- **Review/Deploy:** 1 hour
- **Total:** ~1 day

### Success Metrics
- No method >50 lines
- Nesting depth ≤3 levels
- Cyclomatic complexity ≤10 per method
- All tests pass

---

## Priority 6: Improve Interfaces

### Problem
**Name Collision:**
- `TextChunk` dataclass in `genealogy/chunking/types.py`
- `TextChunk` model in `genealogy/models.py`

**Unclear Abstractions:**
- `page_mapping` dictionary exposed (should be encapsulated)
- Handler interface not formalized

### Root Cause
Incremental development without interface design.

### Solution
1. Rename dataclass to `ChunkCandidate`
2. Encapsulate page_mapping in PageMapper class
3. Formalize handler interface with ABC

### Implementation Steps

#### Step 1: Rename TextChunk Dataclass
**File:** `genealogy/chunking/types.py`

```python
"""Chunking types and data structures"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ChunkCandidate:
    """A candidate chunk during chunking process (before DB save)

    This represents a chunk being built up during the chunking process.
    Once finalized, it's saved as a TextChunk model instance.
    """
    text: str
    chunk_type: str
    tokens: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    start_token_idx: Optional[int] = None
    end_token_idx: Optional[int] = None


@dataclass
class GroundingToken:
    """A grounded OCR token with position and semantic type"""
    text: str
    bbox: List[float]  # [x1, y1, x2, y2]
    element_type: str  # 'title', 'text', 'list', etc.
    page_number: int
    confidence: Optional[float] = None
```

Update all imports:
```bash
find genealogy/chunking -name "*.py" -exec sed -i 's/from genealogy.chunking.types import TextChunk/from genealogy.chunking.types import ChunkCandidate/g' {} +
find genealogy/chunking -name "*.py" -exec sed -i 's/TextChunk(/ChunkCandidate(/g' {} +
```

#### Step 2: Create PageMapper Class
**File:** `genealogy/chunking/page_mapper.py` (new file)

```python
"""Page mapping utilities for chunking"""

from typing import Dict, List


class PageMapper:
    """Maps token indices to page numbers for provenance tracking"""

    def __init__(self):
        self._mapping: Dict[int, int] = {}

    def add_token(self, token_idx: int, page_number: int):
        """Record page number for a token index"""
        self._mapping[token_idx] = page_number

    def get_page(self, token_idx: int) -> int:
        """Get page number for a token index"""
        return self._mapping.get(token_idx, -1)

    def get_page_range(self, start_idx: int, end_idx: int) -> List[int]:
        """Get all pages spanned by token range"""
        pages = set()
        for idx in range(start_idx, end_idx + 1):
            page = self.get_page(idx)
            if page != -1:
                pages.add(page)
        return sorted(pages)

    def get_primary_page(self, start_idx: int, end_idx: int) -> int:
        """Get primary (most common) page for token range"""
        pages = self.get_page_range(start_idx, end_idx)
        if not pages:
            return -1

        # Count tokens per page
        page_counts = {}
        for idx in range(start_idx, end_idx + 1):
            page = self.get_page(idx)
            if page != -1:
                page_counts[page] = page_counts.get(page, 0) + 1

        # Return page with most tokens
        return max(page_counts.items(), key=lambda x: x[1])[0]
```

Refactor usage:
```python
# Before
page_mapping = {}
for idx, token in enumerate(grounded_tokens):
    page_mapping[idx] = token['page_number']

primary_page = max(set(page_mapping.values()), key=list(page_mapping.values()).count)

# After
page_mapper = PageMapper()
for idx, token in enumerate(grounded_tokens):
    page_mapper.add_token(idx, token['page_number'])

primary_page = page_mapper.get_primary_page(start_idx, end_idx)
```

#### Step 3: Formalize Handler Interface
**File:** `genealogy/chunking/handlers.py` (update)

```python
"""Chunk handler interface and implementations"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from genealogy.chunking.types import ChunkCandidate


class ChunkHandler(ABC):
    """Abstract base class for chunk handlers

    Handlers identify and process specific chunk types during
    the chunking process. Each handler is responsible for:
    - Recognizing when it should handle a token
    - Processing the token and extracting metadata
    - Determining when a chunk should end
    """

    @property
    @abstractmethod
    def chunk_type(self) -> str:
        """The chunk type this handler produces

        Examples: 'GENERATION_HEADER', 'GENEALOGY_ENTRY', 'SOURCE_CITATION'
        """
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Handler priority (lower = higher priority)

        When multiple handlers can handle a token, the one with
        lower priority value is chosen.
        """
        pass

    @abstractmethod
    def can_handle(self, token: Dict[str, Any]) -> bool:
        """Check if this handler can process the token

        Args:
            token: Grounding token dictionary

        Returns:
            True if handler can process this token
        """
        pass

    @abstractmethod
    def process_token(self, token: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process token and extract metadata

        Args:
            token: Grounding token dictionary

        Returns:
            Metadata dictionary or None if no metadata extracted
        """
        pass

    @abstractmethod
    def should_end_chunk(self, token: Dict[str, Any]) -> bool:
        """Check if this token should end the current chunk

        Args:
            token: Grounding token dictionary

        Returns:
            True if chunk should end after this token
        """
        pass

    def reset(self):
        """Reset handler state (called when starting new chunk)

        Override if handler maintains state between tokens.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get accumulated metadata for current chunk

        Override if handler accumulates metadata across tokens.

        Returns:
            Metadata dictionary
        """
        return {}
```

Update existing handlers to inherit from ABC:
```python
class GenerationHeaderHandler(ChunkHandler):
    """Handles generation headers (I, II, III, ...)"""

    @property
    def chunk_type(self) -> str:
        return 'GENERATION_HEADER'

    @property
    def priority(self) -> int:
        return 1  # Highest priority

    def can_handle(self, token: Dict[str, Any]) -> bool:
        # Implementation
        pass

    # ... etc
```

### Testing Strategy
1. Run all tests after rename to ensure nothing broke
2. Test PageMapper in isolation
3. Verify handler interface is properly implemented by all handlers
4. Integration test with real chunking pipeline

### Rollback Plan
- Rename can be reversed with search-replace
- PageMapper is additive (can revert to dict if needed)
- Handler ABC is non-breaking (just adds type checking)

### Estimated Effort
- **Development:** 3-4 hours
- **Testing:** 2 hours
- **Review/Deploy:** 1 hour
- **Total:** ~1 day

### Success Metrics
- No name collisions
- PageMapper encapsulates all page mapping logic
- All handlers implement ChunkHandler ABC
- Type checking passes

---

## Priority 7: Extract Image Processing Utilities

### Problem
Image processing code (rotation correction, projection profiles) scattered across multiple files.

### Root Cause
Utility functions added where needed without centralization.

### Solution
Create dedicated `genealogy/image_processing/` module with clean interfaces.

### Implementation Steps

#### Step 1: Create Image Processing Module
**Files:**
```
genealogy/image_processing/
├── __init__.py
├── rotation.py
├── projection.py
└── layout.py
```

#### Step 2: Extract Rotation Correction
**File:** `genealogy/image_processing/rotation.py`

```python
"""Image rotation correction utilities"""

import logging
from typing import Tuple, Optional
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class RotationCorrector:
    """Detects and corrects document rotation"""

    def __init__(self, device: str = 'cuda'):
        """Initialize corrector

        Args:
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.device = device

    def detect_rotation(self, image: Image.Image) -> float:
        """Detect rotation angle of document

        Args:
            image: PIL Image

        Returns:
            Rotation angle in degrees (0, 90, 180, 270)
        """
        try:
            # Try Tesseract OSD first
            angle = self._tesseract_osd(image)
            if angle is not None:
                return angle

            # Fall back to projection profile analysis
            return self._projection_profile_rotation(image)

        except Exception as e:
            logger.warning(f"Rotation detection failed: {e}")
            return 0.0

    def correct_rotation(
        self,
        image: Image.Image,
        angle: Optional[float] = None
    ) -> Tuple[Image.Image, float]:
        """Detect and correct image rotation

        Args:
            image: PIL Image to correct
            angle: Rotation angle (if None, will be detected)

        Returns:
            (corrected_image, angle_used)
        """
        if angle is None:
            angle = self.detect_rotation(image)

        if angle == 0:
            return image, angle

        # Rotate image
        corrected = image.rotate(-angle, expand=True)

        logger.info(f"Corrected rotation by {angle} degrees")
        return corrected, angle

    def _tesseract_osd(self, image: Image.Image) -> Optional[float]:
        """Use Tesseract OSD for rotation detection"""
        import pytesseract

        try:
            osd = pytesseract.image_to_osd(image)
            for line in osd.split('\n'):
                if line.startswith('Rotate: '):
                    return float(line.split(': ')[1])
        except Exception:
            return None

        return None

    def _projection_profile_rotation(self, image: Image.Image) -> float:
        """Use projection profile analysis for rotation detection"""
        from genealogy.image_processing.projection import ProjectionAnalyzer

        analyzer = ProjectionAnalyzer(device=self.device)
        return analyzer.detect_rotation(image)
```

#### Step 3: Extract Projection Profile Analysis
**File:** `genealogy/image_processing/projection.py`

```python
"""Projection profile analysis for layout detection"""

import torch
import kornia
from PIL import Image
import numpy as np


class ProjectionAnalyzer:
    """Analyzes horizontal/vertical projection profiles"""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def compute_horizontal_profile(self, image: Image.Image) -> np.ndarray:
        """Compute horizontal projection profile

        Args:
            image: PIL Image (grayscale or RGB)

        Returns:
            1D array of horizontal projections
        """
        # Convert to tensor
        img_array = np.array(image.convert('L'))
        img_tensor = torch.from_numpy(img_array).float().to(self.device)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Compute horizontal projection
        profile = img_tensor.sum(dim=3).squeeze()  # Sum along width

        return profile.cpu().numpy()

    def compute_vertical_profile(self, image: Image.Image) -> np.ndarray:
        """Compute vertical projection profile

        Args:
            image: PIL Image (grayscale or RGB)

        Returns:
            1D array of vertical projections
        """
        # Convert to tensor
        img_array = np.array(image.convert('L'))
        img_tensor = torch.from_numpy(img_array).float().to(self.device)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        # Compute vertical projection
        profile = img_tensor.sum(dim=2).squeeze()  # Sum along height

        return profile.cpu().numpy()

    def detect_rotation(self, image: Image.Image) -> float:
        """Detect rotation using projection profile variance

        The correct orientation will have the highest variance
        in horizontal projection (text lines create peaks/valleys).

        Args:
            image: PIL Image

        Returns:
            Rotation angle (0, 90, 180, or 270)
        """
        angles = [0, 90, 180, 270]
        variances = []

        for angle in angles:
            rotated = image.rotate(-angle, expand=True)
            profile = self.compute_horizontal_profile(rotated)
            variance = np.var(profile)
            variances.append(variance)

        # Return angle with highest variance
        best_idx = np.argmax(variances)
        return angles[best_idx]
```

#### Step 4: Extract Layout Detection
**File:** `genealogy/image_processing/layout.py`

```python
"""Document layout detection utilities"""

from typing import List, Dict, Any
from PIL import Image


class LayoutDetector:
    """Detects document layout regions using DocLayout-YOLO"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize layout detector

        Args:
            model_path: Path to DocLayout-YOLO model
            device: Device to run on ('cuda' or 'cpu')
        """
        from doclayout_yolo import YOLOv10

        self.model = YOLOv10(model_path)
        self.device = device

    def detect_regions(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """Detect layout regions in image

        Args:
            image: PIL Image
            conf_threshold: Confidence threshold (0-1)

        Returns:
            List of region dictionaries with keys:
                - bbox: [x1, y1, x2, y2]
                - type: Region type ('title', 'text', 'image', etc.)
                - confidence: Detection confidence
        """
        results = self.model.predict(
            image,
            conf=conf_threshold,
            device=self.device
        )

        regions = []
        for result in results:
            for box in result.boxes:
                regions.append({
                    'bbox': box.xyxy[0].tolist(),
                    'type': self._class_id_to_type(box.cls),
                    'confidence': float(box.conf)
                })

        return regions

    def _class_id_to_type(self, class_id: int) -> str:
        """Map DocLayout-YOLO class ID to region type"""
        # Mapping based on DocLayout-YOLO classes
        mapping = {
            0: 'title',
            1: 'text',
            2: 'list',
            3: 'table',
            4: 'figure',
            5: 'caption',
        }
        return mapping.get(int(class_id), 'unknown')
```

#### Step 5: Update Imports Across Codebase
Replace scattered imports with centralized utilities:

```python
# Before
from genealogy.deepseek_ocr_processor import detect_rotation
from genealogy.chunking.utils import compute_projection_profile
from genealogy.tasks.ocr import layout_detection

# After
from genealogy.image_processing.rotation import RotationCorrector
from genealogy.image_processing.projection import ProjectionAnalyzer
from genealogy.image_processing.layout import LayoutDetector
```

### Testing Strategy
1. Unit tests for each image processing class
2. Integration tests with real document images
3. Verify OCR pipeline still works end-to-end
4. Performance benchmarks (should be same or better)

### Rollback Plan
- Keep old utility functions temporarily
- Migrate one file at a time
- Can revert by removing new module

### Estimated Effort
- **Development:** 4-5 hours
- **Testing:** 3 hours
- **Review/Deploy:** 1 hour
- **Total:** ~1 day

### Success Metrics
- All image processing in dedicated module
- Clear, tested interfaces
- No scattered utility functions
- Performance maintained

---

## Overall Timeline

### Week 1
- **Priority 1:** Extract shared chunking utilities (1 day)
- **Priority 2:** Consolidate extraction prompts (0.5 days)
- **Priority 3:** Delete dead code (0.25 days)
- **Testing/Buffer:** 0.25 days

### Week 2
- **Priority 4:** Create service layer (2 days)
- **Testing/Buffer:** 0.5 days

### Week 3
- **Priority 5:** Simplify complex methods (1 day)
- **Priority 6:** Improve interfaces (1 day)
- **Testing/Buffer:** 0.5 days

### Week 4
- **Priority 7:** Extract image processing utilities (1 day)
- **Integration testing:** 1 day
- **Documentation updates:** 0.5 days
- **Final review:** 0.5 days

**Total Estimated Time:** ~3-4 weeks (can be done part-time or in parallel)

---

## Risk Mitigation

### Risk 1: Breaking Existing Functionality
**Mitigation:**
- Comprehensive test coverage before refactoring
- Run full test suite after each priority
- Use feature flags for incremental rollout
- Keep old code temporarily during migration

### Risk 2: Merge Conflicts
**Mitigation:**
- Work on one priority at a time
- Frequent small commits
- Communicate with team about refactoring work
- Use git branches for each priority

### Risk 3: Performance Regression
**Mitigation:**
- Benchmark before/after each change
- Profile critical paths (OCR, chunking, extraction)
- Keep monitoring in place

### Risk 4: Scope Creep
**Mitigation:**
- Stick to defined priorities
- Resist temptation to "fix everything"
- Track additional issues for future work
- Time-box each priority

---

## Success Criteria

### Code Quality
- [ ] Zero code duplication (DRY violations)
- [ ] All methods <50 lines
- [ ] Cyclomatic complexity <10
- [ ] Clear separation of concerns

### Testing
- [ ] All existing tests pass
- [ ] Test coverage ≥80%
- [ ] Integration tests pass
- [ ] No silent failures

### Performance
- [ ] OCR pipeline speed maintained
- [ ] Memory usage not increased
- [ ] Database query count not increased

### Maintainability
- [ ] Clear module boundaries
- [ ] Consistent naming conventions
- [ ] Well-documented interfaces
- [ ] Easy to add new strategies

---

## Post-Refactoring Tasks

After completing all priorities:

1. **Update Documentation**
   - Architecture diagram
   - Code organization guide
   - Contributing guidelines

2. **Code Review**
   - Full codebase review
   - Identify any remaining tech debt
   - Document design decisions

3. **Performance Audit**
   - Profile end-to-end pipeline
   - Identify bottlenecks
   - Optimize if needed

4. **Future Improvements**
   - Strategy for other section types
   - Improved error handling
   - Better logging/observability
