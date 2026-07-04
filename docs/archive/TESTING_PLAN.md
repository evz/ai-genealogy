# Testing Plan: 85-90% Coverage for Processing Pipeline

## Executive Summary

This document outlines a comprehensive testing strategy to achieve 85-90% test coverage for the genealogy extraction processing pipeline (steps 3-6: Chunking → LLM Extraction → Entity Creation → Clustering).

**Current State:** Minimal test coverage (~10-15%)
**Target:** 85-90% coverage
**Timeline:** Implement over 2-3 weeks in phases

## Testing Philosophy

### What We're Testing For
1. **API Stability:** Ensure public interfaces don't break
2. **Correctness:** Business logic produces expected outputs
3. **Edge Cases:** Handle malformed input gracefully
4. **Regression Prevention:** Catch bugs before they reach production

### What We're NOT Testing
- OCR quality (external dependency)
- LLM output quality (non-deterministic, requires human eval)
- UI/Admin interfaces (out of scope for pipeline)

## Pipeline Modules Breakdown

Based on the README diagram, our testing coverage targets:

### Phase 3: Text Chunking (40% of coverage goal)
- `genealogy/chunking/parser.py` (detect_chunk_type, pattern matching)
- `genealogy/chunking/handlers.py` (handler classes)
- `genealogy/chunking/persistence.py` (save_chunks_to_db)
- `genealogy/chunking_strategies/descendant_genealogy.py`
- `genealogy/services/chunking_service.py`
- `genealogy/tasks/chunking.py`

### Phase 4: LLM Extraction (25% of coverage goal)
- `genealogy/prompts/extraction.py` (prompt building, output parsing)
- `genealogy/extraction_strategies/descendant_genealogy.py`
- `genealogy/services/extraction_service.py`
- `genealogy/tasks/extraction.py`

### Phase 5: Entity Creation (20% of coverage goal)
- Management command logic (create_entities.py)
- Entity mapping from chunks to PersonMention/Event/Relationship

### Phase 6: Clustering (15% of coverage goal)
- `genealogy/clustering/person_record.py`
- `genealogy/clustering/graph.py`
- `genealogy/clustering/nodes.py`

---

## Phase 1: Core Chunking Logic (Week 1)

### 1.1 Parser Tests (`test_chunking_parser.py`)

**Target: 90% coverage of parser.py**

#### Test Cases:

```python
class TestChunkTypeDetection:
    """Test detect_chunk_type() function"""

    def test_generation_header_detection():
        # Test: "Tweede generatie" → GENERATION_HEADER
        # Test: Handles OCR corruptions like "DERDE generatie"

    def test_family_group_header_detection():
        # Test: "II.3. Kinderen van X en Y" → FAMILY_GROUP_HEADER
        # Test: "XII.1. Children of X and Y" (English, generation 12)
        # Test: Text element (not sub_title) still matches
        # Test: Remarriage pattern (no roman numeral)

    def test_individual_entry_detection():
        # Test: "a. Pieter van Zanten" → INDIVIDUAL_ENTRY
        # Test: Handles hypothetical entries "a. (hypothetisch) Name"
        # Test: OCR error "I." → "l." normalization

    def test_source_citation_detection():
        # Test: Keywords like "RGV", "DTB", "Bronverwijzing"
        # Test: Both as text and sub_title elements

    def test_biographical_vs_narrative():
        # Test: Date-heavy text → BIOGRAPHICAL_TEXT
        # Test: Story text → NARRATIVE_CONTEXT
```

**Mock Strategy:** No mocks needed - pure functions with string input

**Fixtures:**
- Sample OCR tokens (GroundingToken objects)
- Real-world examples from Van Zanten book

---

### 1.2 Handler Tests (`test_chunking_handlers.py`)

**Target: 85% coverage of handlers.py**

#### Test Cases:

```python
class TestGenerationHeaderHandler:
    def test_extracts_generation_number():
        # Input: "Derde generatie" → generation=3
        # Input: "Twaalfde generatie" → generation=12

    def test_handles_ocr_corruption():
        # Input: "ierde generatie" (missing 'V') → generation=4

class TestFamilyGroupHeaderHandler:
    def test_extracts_family_group_id():
        # Input: "II.3. Kinderen van..." → family_group_id="II.3"

    def test_extracts_parents():
        # Input: "Kinderen van Pieter en Maria" → parents=("Pieter", "Maria")
        # Input: "Kinderen van Pieter" → parents=("Pieter",)

    def test_remarriage_inherits_id():
        # Context has family_group_id="VII.1"
        # Input: "Kinderen van Jan en Truus" (no roman numeral)
        # → Inherits family_group_id="VII.1", updates parents

class TestIndividualEntryHandler:
    def test_consolidates_biographical_text():
        # Token sequence: "a. Pieter" → bio text → bio text → STOP at next "b."
        # Returns single chunk with all bio text consolidated

    def test_stops_at_structural_boundaries():
        # Stops at: next individual entry, family group header, generation header

    def test_phase1_extraction():
        # Extracts person name from entry
        # Creates parent-child relationships from context
```

**Mock Strategy:**
- Mock `detect_chunk_type()` to control boundaries
- Use real GroundingToken fixtures

**Fixtures:**
- Token sequences representing typical genealogy structures
- Context dict with generation/family_group data

---

### 1.3 Integration Tests (`test_chunking_integration.py`)

**Target: Test full chunking pipeline end-to-end**

#### Test Cases:

```python
class TestDescendantGenealogyChunking:
    def test_chunks_generation_12_correctly(self):
        """
        Regression test for generation 12 bug.
        Family group headers labeled as 'text' should still be detected.
        """
        # Input: Page 77 OCR text with "XII.1. Children of..."
        # Expected: Separate chunk for family group header
        # Expected: Individual entries not merged with header

    def test_out_of_flow_extraction():
        # Input: Tokens with images and inverted content
        # Expected: Separate chunks for images/info boxes
        # Expected: Main flow tokens don't include out-of-flow content

    def test_phase1_entity_extraction():
        # Input: Full generation with headers and entries
        # Expected: extracted_people populated
        # Expected: extracted_relationships has parent-child links
```

**Mock Strategy:**
- Use real OCR text samples
- Mock database saves (use in-memory objects)

**Fixtures:**
- `fixtures/page_77_ocr.txt` (generation 12 example)
- `fixtures/generation_with_entries.txt` (complete generation)

---

## Phase 2: Services & Tasks (Week 1-2)

### 2.1 Service Layer Tests

**Target: 95% coverage (services are thin, should be easy)**

```python
class TestChunkingService:
    def test_chunk_section_success(self):
        # Mock strategy.chunk_section()
        # Mock save_chunks_to_db()
        # Verify result structure

    def test_chunk_section_unknown_type(self):
        # Input: Invalid section_type
        # Expected: Returns {'success': False, 'error': ...}

    def test_should_process_section(self):
        # Verify delegates to strategy.should_process()

class TestExtractionService:
    def test_extract_from_chunk_success(self):
        # Mock strategy.extract()
        # Verify result structure

    def test_extract_from_chunks_in_section(self):
        # Input: Multiple chunks
        # Expected: Aggregates results (processed, failed counts)
```

**Mock Strategy:**
- Heavy mocking - services are thin wrappers
- Mock strategies and database operations

---

### 2.2 Task Tests (`test_tasks_comprehensive.py`)

**Target: 80% coverage (tasks have lots of error handling)**

```python
class TestChunkingTask:
    def test_create_document_chunks_success(self):
        # Create document with pages and book sections
        # Mock service layer
        # Verify chunks created

    def test_handles_no_book_sections(self):
        # Input: Document without BookSections
        # Expected: Returns error

    def test_handles_ocr_not_completed(self):
        # Input: Document with ocr_completed=False
        # Expected: Returns error

    def test_clears_existing_chunks(self):
        # Pre-populate with old chunks
        # Run task
        # Verify old chunks deleted

class TestExtractionTask:
    def test_extract_entities_from_chunks_success(self):
        # Create document with chunks
        # Mock service layer and Ollama
        # Verify chunks updated with entities

    def test_handles_ollama_unavailable(self):
        # Mock ollama.is_available() → False
        # Expected: Raises RuntimeError

    def test_marks_document_complete(self):
        # Run extraction
        # Verify document.extraction_completed = True
```

**Mock Strategy:**
- Use Django's TestCase for DB operations
- Mock Ollama client
- Mock service layer for happy path tests

---

## Phase 3: Extraction & Prompts (Week 2)

### 3.1 Prompt Tests (`test_extraction_prompts.py`)

**Target: 95% coverage of prompts/extraction.py**

```python
class TestPromptBuilding:
    def test_build_extraction_prompt():
        # Input: Chunk with generation/family context
        # Expected: Prompt includes context, examples, instructions
        # Expected: Dutch abbreviations and event codes included

    def test_handles_missing_context():
        # Input: Chunk without generation
        # Expected: Prompt still builds successfully

class TestOutputParsing:
    def test_parse_extraction_output_valid():
        # Input: Well-formed LLM output with PEOPLE/PARENT_CHILD/PARTNERSHIPS/EVENTS
        # Expected: Returns structured dict with all entities

    def test_parse_extraction_output_malformed():
        # Input: Missing sections or invalid format
        # Expected: Returns empty lists, doesn't crash

    def test_parse_handles_occu_events():
        # Input: Output with occupation events
        # Expected: OCCU events parsed with person/place
```

**Mock Strategy:** None - pure functions

**Fixtures:**
- Sample LLM outputs (valid and edge cases)
- Chunk objects with various contexts

---

### 3.2 Extraction Strategy Tests

```python
class TestDescendantGenealogyExtraction:
    def test_merges_phase1_and_phase2_data():
        # Chunk with Phase 1 data (people, relationships)
        # Mock LLM response with Phase 2 data
        # Expected: Merged without duplicates

    def test_deduplicates_people():
        # Phase 1: ["Pieter van Zanten"]
        # Phase 2: ["Pieter van Zanten", "Maria Jansen"]
        # Expected: ["Pieter van Zanten", "Maria Jansen"] (no duplicate)

    def test_deduplicates_relationships():
        # Similar logic for relationships
```

**Mock Strategy:**
- Mock Ollama client
- Use real prompt building/parsing

---

## Phase 4: Entity Creation & Clustering (Week 3)

### 4.1 Entity Creation Tests

```python
class TestEntityCreation:
    def test_creates_person_mentions():
        # Input: Chunk with extracted_people
        # Expected: PersonMention objects created

    def test_creates_relationship_mentions():
        # Input: Chunk with extracted_relationships
        # Expected: RelationshipMention objects created

    def test_creates_events():
        # Input: Chunk with extracted_events
        # Expected: Event objects created and linked to mentions
```

---

### 4.2 Clustering Tests

```python
class TestPersonRecord:
    def test_name_similarity():
        # "Pieter van Zanten" vs "P. van Zanten" → high similarity
        # "Pieter van Zanten" vs "Jan Jansen" → low similarity

    def test_date_proximity():
        # Birth years within 2 years → high score
        # Birth years >10 years apart → low score

class TestGraphClustering:
    def test_bootstrap_high_confidence_matches():
        # Input: Mentions with similarity >= 0.75
        # Expected: Merged into same identity

    def test_iterative_refinement():
        # Input: Mentions with 0.60 <= similarity < 0.75
        # After bootstrap merges, recalculate
        # Expected: Additional merges based on graph context
```

---

## Test Infrastructure Setup

### Fixtures & Test Data

Create `genealogy/tests/fixtures/`:
```
fixtures/
├── ocr_samples/
│   ├── page_77_generation_12.txt
│   ├── generation_with_entries.txt
│   └── info_box_example.txt
├── llm_outputs/
│   ├── valid_extraction.txt
│   ├── malformed_extraction.txt
│   └── occupation_events.txt
└── grounding_tokens/
    ├── family_group_header.json
    ├── individual_entry_sequence.json
    └── out_of_flow_content.json
```

### Helper Functions

Create `genealogy/tests/helpers.py`:
```python
def create_grounding_token(content, element_type='text', bbox=None):
    """Factory for test GroundingToken objects"""

def create_test_chunk(text, generation=None, family_group=None):
    """Factory for TextChunk objects"""

def create_mock_ollama_response(people, relationships, events):
    """Generate mock LLM output"""
```

### Coverage Configuration

Update `.coveragerc` or `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["genealogy"]
omit = [
    "*/migrations/*",
    "*/tests/*",
    "*/admin/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

---

## Running Tests

### Commands

```bash
# Run all tests with coverage
pytest --cov=genealogy --cov-report=html --cov-report=term

# Run specific module
pytest genealogy/tests/test_chunking_parser.py -v

# Run with markers
pytest -m "not slow"  # Exclude slow integration tests
```

### CI Integration

Update `.github/workflows/quality-gate.yml`:
```yaml
- name: Run tests with coverage
  run: |
    pytest --cov=genealogy --cov-report=xml --cov-fail-under=85

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
```

---

## Success Metrics

### Coverage Targets by Module

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| `chunking/parser.py` | 90% | HIGH |
| `chunking/handlers.py` | 85% | HIGH |
| `chunking_strategies/` | 80% | HIGH |
| `services/` | 95% | HIGH |
| `prompts/extraction.py` | 95% | MEDIUM |
| `extraction_strategies/` | 80% | MEDIUM |
| `clustering/` | 75% | MEDIUM |
| `tasks/` | 80% | MEDIUM |

**Overall Target:** 85-90% coverage

### Definition of Done

- [ ] All target modules have >= target coverage
- [ ] CI fails if coverage drops below 85%
- [ ] No flaky tests (all tests pass consistently)
- [ ] Test runtime < 2 minutes (excluding slow integration tests)
- [ ] Documentation updated with testing patterns

---

## Implementation Order

### Week 1: Foundation
1. Set up test infrastructure (fixtures, helpers)
2. Parser tests (test_chunking_parser.py)
3. Handler tests (test_chunking_handlers.py)
4. Service tests (test_chunking_service.py, test_extraction_service.py)

### Week 2: Integration & Extraction
5. Chunking integration tests (test_chunking_integration.py)
6. Prompt tests (test_extraction_prompts.py)
7. Extraction strategy tests
8. Task tests (test_tasks_comprehensive.py)

### Week 3: Entities & Clustering
9. Entity creation tests
10. Clustering tests
11. Fill coverage gaps
12. Documentation

---

## Notes & Assumptions

1. **LLM Mocking:** We mock Ollama responses rather than testing actual LLM quality
2. **Database:** Use Django's TestCase with transaction rollback
3. **Fixtures:** Use real OCR samples from Van Zanten book
4. **Regression Tests:** Add test for any bug we find (like generation 12 issue)
5. **Performance:** Mark slow integration tests with `@pytest.mark.slow`

---

## Questions for Discussion

1. Should we use `pytest` or Django's `unittest`? (Recommendation: pytest for better fixtures)
2. Do we want mutation testing (with `mutmut`) to validate test quality?
3. Should we separate unit tests from integration tests in directory structure?
4. Do we need property-based testing (with `hypothesis`) for parser edge cases?
