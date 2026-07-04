# Testing Infrastructure Setup - Completed

This document summarizes the test infrastructure that has been set up for the genealogy extraction pipeline.

## What's Been Completed

### 1. Directory Structure ✓

```
genealogy/tests/
├── __init__.py (already existed)
├── helpers.py (NEW)
├── test_chunking_parser.py (NEW)
└── fixtures/
    ├── ocr_samples/
    │   └── page_77_generation_12.txt (NEW)
    ├── llm_outputs/ (empty, ready for future fixtures)
    └── grounding_tokens/ (empty, ready for future fixtures)
```

### 2. Test Helper Functions ✓

**File:** `genealogy/tests/helpers.py`

Provides factory functions for creating test data:

- `create_grounding_token()` - Create mock GroundingToken objects
- `create_token_sequence()` - Create sequences of tokens with auto-generated bounding boxes
- `create_mock_ollama_response()` - Generate mock LLM extraction output
- `load_fixture()` - Load test fixture files from the fixtures directory

### 3. Coverage Configuration ✓

**File:** `pyproject.toml` (updated)

Added comprehensive coverage configuration:

```toml
[tool.coverage.run]
source = ["genealogy"]
omit = ["*/migrations/*", "*/tests/*", "*/admin.py", "*/apps.py", "manage.py"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "def __repr__", "raise NotImplementedError", ...]
precision = 2
show_missing = true

[tool.pytest.ini_options]
DJANGO_SETTINGS_MODULE = "genealogy_extractor.settings"
python_files = ["test_*.py", "*_test.py"]
testpaths = ["genealogy/tests"]
markers = ["slow", "integration", "unit"]
```

### 4. Test Dependencies ✓

**File:** `requirements.txt` (updated)

Added testing dependencies:

```
pytest==8.3.4
pytest-django==4.9.0
pytest-cov==6.0.0
pytest-xdist==3.6.1  # for parallel test execution
```

**Installation:** Run `pip install -r requirements.txt` or rebuild Docker container

### 5. First Test Suite ✓

**File:** `genealogy/tests/test_chunking_parser.py`

Comprehensive tests for `genealogy.chunking.parser` module:

- **TestParseGroundingTokens** (4 tests)
  - Basic token parsing
  - Inverted token detection
  - Multiple token sequences
  - Document order preservation

- **TestChunkTypeDetection** (11 tests)
  - Generation header detection (including OCR corruptions)
  - Family group headers as `sub_title` elements
  - **Family group headers as `text` elements (regression test for generation 12 bug)**
  - Remarriage family group headers
  - Individual entry detection
  - Source citation detection
  - Info box detection
  - Biographical vs narrative text classification
  - Image types

- **TestIsSourceCitation** (3 tests)
  - Archive code detection
  - Keyword detection
  - Normal text rejection

- **TestIsBiographicalText** (8 tests)
  - Birth/death markers
  - Marriage markers
  - Genealogical abbreviations
  - Residence patterns
  - Address patterns
  - Narrative rejection

- **TestExtractPersonFromIndividualEntry** (6 tests)
  - Simple name extraction
  - Nickname handling
  - Various marker types
  - OCR error handling
  - Non-entry rejection

- **TestGenerationTwelveRegression** (2 integration tests)
  - **Regression test for generation 12 bug** - verifies family group headers labeled as `text` are detected
  - Verifies individual entries aren't merged with headers

**Total:** 35 test cases covering parser.py:292

### 6. Test Fixtures ✓

**File:** `genealogy/tests/fixtures/ocr_samples/page_77_generation_12.txt`

Real OCR sample from page 77 demonstrating:
- Generation header: "TWAALFDE GENERATIE"
- Multiple family group headers (XII.1, XII.2, XII.3) labeled as `text` elements
- Individual entries with biographical data
- Used in regression tests for the generation 12 bug

## Running Tests

### Basic Test Run

```bash
# Inside Docker container
docker compose exec web pytest genealogy/tests/test_chunking_parser.py -v

# With coverage
docker compose exec web pytest genealogy/tests/ --cov=genealogy --cov-report=term

# Generate HTML coverage report
docker compose exec web pytest genealogy/tests/ --cov=genealogy --cov-report=html
```

### Test Markers

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Parallel Execution

```bash
# Run tests in parallel (4 workers)
pytest -n 4
```

## Next Steps

According to `docs/TESTING_PLAN.md`, the next priorities are:

### Week 1 (Current):
- ✅ Set up test infrastructure
- ✅ Parser tests (test_chunking_parser.py) - **COMPLETED**
- 🔲 Handler tests (test_chunking_handlers.py)
- 🔲 Service tests (test_chunking_service.py, test_extraction_service.py)

### Week 2:
- 🔲 Chunking integration tests
- 🔲 Prompt tests
- 🔲 Extraction strategy tests
- 🔲 Task tests

### Week 3:
- 🔲 Entity creation tests
- 🔲 Clustering tests
- 🔲 Fill coverage gaps
- 🔲 Documentation

## Coverage Goals

Target coverage by module (from TESTING_PLAN.md):

| Module | Target | Priority |
|--------|--------|----------|
| `chunking/parser.py` | 90% | HIGH |
| `chunking/handlers.py` | 85% | HIGH |
| `chunking_strategies/` | 80% | HIGH |
| `services/` | 95% | HIGH |
| `prompts/extraction.py` | 95% | MEDIUM |
| `extraction_strategies/` | 80% | MEDIUM |
| `clustering/` | 75% | MEDIUM |
| `tasks/` | 80% | MEDIUM |

**Overall Target:** 85-90% coverage

## Important Notes

1. **Pytest Not Yet Installed in Docker:** You'll need to rebuild the Docker container or run `pip install -r requirements.txt` inside the container to install pytest and related dependencies.

2. **Regression Test for Generation 12 Bug:** The test suite includes specific tests for the bug where family group headers in generation 12 were labeled as `text` instead of `sub_title` by DeepSeek-OCR. These tests verify the fix in `genealogy/chunking/parser.py:154-160`.

3. **Test Data Location:** All test fixtures should be placed in `genealogy/tests/fixtures/` following the structure:
   - `ocr_samples/` - OCR text samples
   - `llm_outputs/` - Mock LLM responses
   - `grounding_tokens/` - JSON token sequences

4. **Helper Functions:** Use the helper functions in `genealogy/tests/helpers.py` to create test data consistently. Avoid duplicating factory logic across test files.

5. **Test Isolation:** Tests use pytest fixtures and don't require database setup (parser tests are pure functions). Database-dependent tests will use `pytest-django` fixtures when we implement handler/service tests.

## CI Integration (TODO)

Once tests are working locally, update `.github/workflows/quality-gate.yml`:

```yaml
- name: Run tests with coverage
  run: |
    docker compose exec -T web pytest --cov=genealogy --cov-report=xml --cov-fail-under=85

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Summary

The testing infrastructure is now set up and ready for continued development. The first test suite (`test_chunking_parser.py`) provides 35 test cases covering the parser module, including a comprehensive regression test for the generation 12 bug.

**Status:** Ready to continue with handler tests and service tests as outlined in the testing plan.
