# Testing Opportunities Analysis

## Current State

### Test Files by Type

**Integration Tests (No Mocks):**
- `test_chunking_handlers.py` (818 lines) - Chunking handlers with real text
- `test_chunking_parser.py` (494 lines) - Parser logic
- `test_chunking_persistence.py` (355 lines) - Database persistence
- `test_chunking_strategies.py` (179 lines) - Chunking strategies
- `test_clustering_graph.py` (394 lines) - Clustering graph algorithms
- `test_clustering.py` (294 lines) - Clustering nodes and logic
- `test_genealogy_tools.py` (301 lines) - Genealogy tools for agent
- `test_models.py` (146 lines) - Django model tests
- `test_person_mention_lifecycle.py` (166 lines) - Entity lifecycle
- `test_agent_integration.py` (369 lines) - NEW: Agent workflow integration

**Unit Tests (Heavy Mocks):**
- `test_admin.py` (275 lines) - Admin actions
- `test_agent_executor.py` (324 lines) - Agent executor logic
- `test_extraction_strategies.py` (254 lines) - Extraction strategies
- `test_prompts_extraction.py` (316 lines) - Prompt building
- `test_services.py` (361 lines) - Service orchestration
- `test_tasks_chunking.py` (337 lines) - Chunking Celery tasks
- `test_tasks_extraction.py` (295 lines) - Extraction Celery tasks
- `test_tasks.py` (113 lines) - Task utilities

**Total: ~5,791 lines of tests**

## Key Findings

### ✅ Strong Integration Testing
1. **Chunking Pipeline**: Well-tested end-to-end without mocks
   - Parser → Handler → Persistence all have integration tests
   - Tests use real OCR text and verify database state

2. **Clustering**: Well-tested with real graph algorithms
   - Node creation, similarity calculations
   - Constraint validation

3. **Agent Workflow**: NEW comprehensive integration tests
   - Tools work with real database
   - Multi-turn reasoning flows
   - Family relationship tracing

### 🔍 Integration Test Opportunities

#### 1. **Extraction Pipeline Integration Tests**
**Current:** Heavily mocked (`test_extraction_strategies.py`, `test_tasks_extraction.py`)
**Opportunity:** Create integration tests that:
- Use real TextChunks with actual genealogy text
- Mock only the LLM call (like we did for agent)
- Verify PersonMention, Event, RelationshipMention creation in database
- Test full flow: chunk → extract → persist

**Example Test:**
```python
@patch('genealogy.extraction_strategies.OllamaClient')
def test_extraction_creates_real_entities(mock_ollama):
    """Test that extraction creates actual database entities"""
    # Setup: Create real chunk with Dutch genealogy text
    chunk = TextChunk.objects.create(
        text_content="a. Pieter van Zanten, * Amsterdam 1850, † Rotterdam 1920"
    )

    # Mock only LLM response
    mock_ollama.generate.return_value = '''
    {
        "people": ["Pieter van Zanten"],
        "events": [
            {"person": "Pieter van Zanten", "event_type": "BIRT", "date": "1850", "place": "Amsterdam"},
            {"person": "Pieter van Zanten", "event_type": "DEAT", "date": "1920", "place": "Rotterdam"}
        ]
    }
    '''

    # Execute
    strategy = DescendantGenealogyStrategy()
    result = strategy.extract(chunk, mock_ollama, "test-model")

    # Verify: Check real database entities created
    assert PersonMention.objects.filter(given_names="Pieter", surname="van Zanten").exists()
    assert Event.objects.filter(event_type="BIRT", date__year=1850).exists()
```

**Benefits:**
- Catches database constraint violations
- Verifies data transformations
- Tests prompt parsing → model creation flow

#### 2. **RAG+RRF Retrieval Integration Tests**
**Current:** No dedicated tests for `retrieval.py`
**Opportunity:** Create integration tests for hybrid retrieval:
- Setup database with real TextChunks, embeddings, DM codes
- Test vector search, trigram search, phonetic search, subject search
- Verify RRF fusion logic
- Test window expansion

**Example Test:**
```python
@pytest.mark.django_db
def test_hybrid_retrieval_finds_relevant_chunks():
    """Test that RRF correctly ranks chunks"""
    # Setup: Create chunks about multiple Pieters
    pieter1 = TextChunk.objects.create(
        text_content="a. Pieter van Zanten, mason, * 1850",
        subject="Pieter van Zanten",
        embedding=embed("Pieter van Zanten mason 1850"),
        dm_codes=["P361", "F523", "Z353"]
    )

    pieter2 = TextChunk.objects.create(
        text_content="b. Pieter van Zanten, farmer, * 1880",
        subject="Pieter van Zanten",
        embedding=embed("Pieter van Zanten farmer 1880"),
        dm_codes=["P361", "F523", "Z353"]
    )

    # Execute
    retriever = HybridRetriever()
    results = retriever.retrieve("Pieter van Zanten the mason", top_k=2)

    # Verify: Mason chunk should rank higher
    assert results[0]["id"] == pieter1.id
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
```

**Benefits:**
- Validates search quality
- Tests subject boosting (k=10)
- Catches person filter bugs (AND vs OR)

#### 3. **Entity Clustering Integration Tests**
**Current:** Graph algorithm tests exist, but not end-to-end clustering
**Opportunity:** Test the full clustering workflow:
- Create multiple PersonMentions for same person
- Run clustering algorithm
- Verify Identity creation and MentionToIdentity mappings

**Example Test:**
```python
@pytest.mark.django_db
def test_clustering_merges_duplicate_mentions():
    """Test that clustering correctly identifies same person"""
    # Create mentions for same person
    mention1 = PersonMention.objects.create(
        given_names="Pieter", surname="van Zanten",
        birth_date="1850-01-01"
    )

    mention2 = PersonMention.objects.create(
        given_names="P.", surname="van Zanten",
        birth_date="1850-01-01"  # Same birth date
    )

    # Execute clustering
    from genealogy.clustering import cluster_person_mentions
    cluster_person_mentions()

    # Verify: Both mentions map to same identity
    identity1 = mention1.identity_mappings.first().identity
    identity2 = mention2.identity_mappings.first().identity
    assert identity1 == identity2
    assert identity1.display_name == "Pieter van Zanten"
```

#### 4. **Admin Action Integration Tests**
**Current:** `test_admin.py` has mocks
**Opportunity:** Test admin actions with real database operations:
- Test bulk re-extraction
- Test section-based filtering
- Verify Celery task queueing

**Example Test:**
```python
@pytest.mark.django_db
@patch('genealogy.admin.textchunk.ExtractionService')
def test_reextract_entities_filters_by_section(mock_service):
    """Test that reextract only processes genealogy sections"""
    # Setup: Create chunks in different sections
    doc = Document.objects.create(title="Test")

    gen_section = BookSection.objects.create(
        document=doc, section_type="DESCENDANT_GENEALOGY",
        start_page=1, end_page=10
    )

    other_section = BookSection.objects.create(
        document=doc, section_type="INTRODUCTION",
        start_page=11, end_page=20
    )

    gen_chunk = TextChunk.objects.create(
        document=doc, start_page=5, end_page=5
    )

    other_chunk = TextChunk.objects.create(
        document=doc, start_page=15, end_page=15
    )

    # Execute
    admin = TextChunkAdmin(TextChunk, admin.site)
    queryset = TextChunk.objects.all()
    admin.reextract_entities_action(request, queryset)

    # Verify: Only genealogy chunk processed
    call_args = mock_service.return_value.extract_from_chunks_in_section.call_args
    processed_chunks = call_args[0][0]
    assert gen_chunk in processed_chunks
    assert other_chunk not in processed_chunks
```

#### 5. **CLI Command Integration Tests**
**Current:** No tests for `query_genealogy.py`
**Opportunity:** Test the full CLI workflow:
- Test regular RAG mode
- Test agent mode
- Test argument parsing and output formatting

**Example Test:**
```python
@pytest.mark.django_db
@patch('genealogy.services.agent_executor.OllamaClient')
def test_query_genealogy_agent_mode(mock_ollama):
    """Test CLI with --agent flag"""
    # Setup test data
    setup_test_identities()

    mock_ollama.generate.side_effect = [
        'TOOL_CALL: search_person_by_name\nARGUMENTS: {"name": "Pieter"}',
        'ANSWER: Pieter van Zanten was born in 1850.'
    ]

    # Execute
    from io import StringIO
    out = StringIO()
    call_command('query_genealogy', 'Who is Pieter van Zanten?',
                 '--agent', stdout=out)

    # Verify
    output = out.getvalue()
    assert '🤖 Using agentic workflow' in output
    assert '🔧 TOOL CALLS:' in output
    assert 'search_person_by_name' in output
    assert 'Pieter van Zanten was born in 1850' in output
```

## Recommendations

### Priority 1: High Value, Low Effort
1. **RAG+RRF Integration Tests** - Critical for search quality
   - Setup test fixtures with varied chunks
   - Test edge cases (AND vs OR filtering, subject boosting)
   - ~200 lines of tests

2. **CLI Command Tests** - User-facing, easy to break
   - Test both RAG and agent modes
   - Test error handling
   - ~150 lines of tests

### Priority 2: Medium Value, Medium Effort
3. **Extraction Integration Tests** - Complex flow, many edge cases
   - Test entity creation from real text
   - Test prompt building and parsing
   - ~300 lines of tests

4. **Admin Action Integration Tests** - Prevent production bugs
   - Test section filtering
   - Test bulk operations
   - ~200 lines of tests

### Priority 3: Lower Priority
5. **Clustering Integration Tests** - Already has good graph tests
   - Test end-to-end clustering workflow
   - Test Identity creation
   - ~200 lines of tests

## Testing Principles

### What to Mock
- ✅ External LLM calls (OllamaClient)
- ✅ Celery task execution (in unit tests)
- ✅ File I/O for large operations

### What NOT to Mock
- ❌ Database operations (use test database)
- ❌ Django ORM queries
- ❌ Business logic (services, strategies)
- ❌ Data transformations

### Integration Test Pattern
```python
@pytest.mark.django_db
@patch('module.path.to.OllamaClient')  # Mock only external calls
def test_feature_end_to_end(mock_ollama):
    # Setup: Create real database objects
    obj = Model.objects.create(...)

    # Mock: Only external/slow dependencies
    mock_ollama.generate.return_value = "LLM response"

    # Execute: Run actual code
    result = service.method(obj)

    # Verify: Check real database state
    assert Model.objects.filter(...).exists()
    assert result['success'] is True
```

## Coverage Gaps

Based on file analysis, these modules lack integration tests:
1. `genealogy/retrieval.py` - No tests at all
2. `genealogy/management/commands/query_genealogy.py` - No tests
3. `genealogy/services/extraction_service.py` - Only mocked tests
4. `genealogy/admin/*.py` - Only mocked tests

## Next Steps

1. Start with Priority 1 (RAG+RRF and CLI tests)
2. Add integration tests incrementally, don't rewrite existing tests
3. Use `@pytest.mark.integration` to distinguish from unit tests
4. Run integration tests in CI but allow skipping for faster local dev
5. Measure coverage improvement (aim for 80%+ on critical paths)
