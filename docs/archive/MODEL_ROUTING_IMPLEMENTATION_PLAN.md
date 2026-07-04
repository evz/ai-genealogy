# Model Routing Implementation Plan

## Overview

Replace manual model selection with intelligent automatic routing based on query characteristics. This will ensure the right model is used for each use case:
- **gene-chat-fast** (llama3.1:8b) - Fast interactive queries
- **gene-chat-main** (qwen2.5:14b) - Complex multi-document reasoning
- **gene-reasoner** (deepseek-r1) - Identity resolution and merge conflicts

## Phase 0: DeepSeek-R1 Model Size Evaluation ✅ COMPLETED

**Goal**: Determine whether deepseek-r1:32b provides meaningful accuracy improvements over deepseek-r1:14b for genealogy merge/conflict resolution, and whether the speed tradeoff is acceptable.

### Results

**Decision**: Use **deepseek-r1:14b** as the base for gene-reasoner

After manual evaluation of both models on 11 genealogy reasoning test cases:
- deepseek-r1:14b provided sufficient accuracy for identity resolution and conflict reasoning
- Speed advantage of 14b model outweighed marginal accuracy gains from 32b
- Custom models created in Ollama:
  - ✅ gene-chat-fast (llama3.1:8b)
  - ✅ gene-chat-main (qwen2.5:14b)
  - ✅ gene-reasoner (deepseek-r1:14b)

### Test Scenarios

Create a test script that evaluates both models on identical genealogy reasoning tasks:

1. **Identity Resolution Tests** (5-7 test cases)
   - Same person across different records with spelling variations
   - Two different people with similar names and dates
   - Ambiguous case requiring careful reasoning
   - Example: "Are these baptism records for the same person?"

2. **Conflict Resolution Tests** (3-5 test cases)
   - Contradictory dates in different sources
   - Marriage records with conflicting spouse names
   - Death records with different ages/places
   - Example: "These three records show different birth years for Jan Pieters. Which is most likely correct?"

3. **Multi-Step Reasoning Tests** (3-5 test cases)
   - Reconstruct timeline from fragmentary evidence
   - Identify most likely explanation for apparent contradictions
   - Example: "Given these 5 records, construct the most plausible life history"

### Test Implementation

**File**: `genealogy/tests/test_model_evaluation.py`

```python
"""
Manual test runner for evaluating deepseek-r1:14b vs deepseek-r1:32b.

Run with: python manage.py test genealogy.tests.test_model_evaluation --keepdb
"""

import time
from dataclasses import dataclass
from typing import List

@dataclass
class TestCase:
    """A genealogy reasoning test case"""
    name: str
    query: str
    documents: List[str]  # Simulated retrieved records
    expected_conclusion: str  # What we expect the model to conclude
    reasoning_points: List[str]  # Key points a good answer should mention

# Define test cases
IDENTITY_TESTS = [
    TestCase(
        name="Clear Same Person - Spelling Variation",
        query="Are these baptism records for the same person?",
        documents=[
            "[Doc 1] Baptism: Pieter Jansen, born 15 March 1845, Amsterdam, parents: Jan Pieters & Maria de Vries",
            "[Doc 2] Baptism: Pieter Janszoon, born 15 Maart 1845, Amsterdam, parents: Jan Pieters & Maria de Vries"
        ],
        expected_conclusion="same person",
        reasoning_points=[
            "identical date",
            "identical location",
            "identical parents",
            "Jansen/Janszoon are patronymic variations"
        ]
    ),
    TestCase(
        name="Ambiguous - Could Be Same or Different",
        query="Do these records refer to the same Jan van Zanten?",
        documents=[
            "[Doc 1] Marriage: Jan van Zanten, age 28, married Anna Bakker, 5 June 1872, Haarlem",
            "[Doc 2] Death: Jan van Zanten, age 65, died 12 March 1910, Haarlem, occupation: baker",
            "[Doc 3] Census: Jan van Zanten, age 45, living in Haarlem, 1889, wife: Anna"
        ],
        expected_conclusion="probably same person",
        reasoning_points=[
            "ages are roughly consistent (28 in 1872 → 65 in 1910 → 38 years)",
            "same location (Haarlem)",
            "wife's name matches (Anna)",
            "timeline works if married at 28",
            "lack of father's name makes it uncertain"
        ]
    ),
    # Add 3-5 more test cases...
]

CONFLICT_TESTS = [
    TestCase(
        name="Conflicting Birth Years",
        query="These records show different birth years. Which is most reliable?",
        documents=[
            "[Doc 1] Marriage record 1885: Bessel van Zanten, age 23 (implies birth ~1862)",
            "[Doc 2] Death record 1911: Bessel van Zanten, age 70 (implies birth ~1841)",
            "[Doc 3] Baptism record: Bessel van Zanten, baptized 17 August 1841, Naarden"
        ],
        expected_conclusion="birth year 1841 is most reliable",
        reasoning_points=[
            "baptism record is primary source",
            "age at death often inaccurate",
            "1841 makes death age correct (70 years)",
            "marriage age would be 23 if born 1862 (too young given typical patterns)",
            "suggests marriage record age may be error or different person"
        ]
    ),
    # Add 2-4 more conflict resolution tests...
]

MULTI_STEP_TESTS = [
    TestCase(
        name="Reconstruct Life Timeline",
        query="Construct the most plausible timeline for this person's life",
        documents=[
            "[Doc 1] Baptism: Anna Pieters, 12 May 1835, Rotterdam, father: Pieter Jansen",
            "[Doc 2] Marriage: Anna Jansen married to Willem Smit, 3 June 1858, Rotterdam",
            "[Doc 3] Birth record: Pieter Smit, born 15 March 1859, Rotterdam, mother: Anna Jansen",
            "[Doc 4] Census 1869: Willem Smit (35), Anna (34), children: Pieter (10), Maria (5)",
            "[Doc 5] Death: Anna Smit, died 4 Jan 1892, Amsterdam, age 57"
        ],
        expected_conclusion="single timeline with Anna Pieters → Anna Jansen → Anna Smit",
        reasoning_points=[
            "patronymic usage (Anna Pieters = daughter of Pieter)",
            "married in 1858 at age ~23",
            "had children 1859 (Pieter), ~1864 (Maria)",
            "moved to Amsterdam sometime between 1869-1892",
            "ages roughly consistent across records"
        ]
    ),
    # Add 2-4 more timeline reconstruction tests...
]

def evaluate_model(model_name: str, test_cases: List[TestCase]) -> dict:
    """
    Evaluate a model on test cases and return results.

    Returns:
        {
            "model": model_name,
            "accuracy": percentage,
            "avg_time": seconds,
            "results": [individual test results]
        }
    """
    from genealogy.ollama_utils import OllamaClient

    client = OllamaClient()
    results = []

    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"Test: {test.name}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")

        # Build prompt
        docs_text = "\n\n".join(test.documents)
        prompt = f"""You are a genealogy specialist evaluating historical records.

Documents:
{docs_text}

Question: {test.query}

Instructions:
1. Analyze the evidence step by step
2. State your conclusion clearly: same person / probably same person / uncertain / different people
3. List the key evidence that supports your conclusion
4. Identify any contradictions or uncertainties

Your response:"""

        # Time the generation
        start = time.time()
        response = client.generate(model_name, prompt, num_ctx=32768, temperature=0.3)
        elapsed = time.time() - start

        print(f"\nResponse ({elapsed:.1f}s):")
        print(response)

        # Manual evaluation (for now)
        print(f"\nExpected conclusion: {test.expected_conclusion}")
        print(f"Expected reasoning points: {test.reasoning_points}")

        correct = input("\nWas this correct? (y/n/partial): ").lower()
        score = 1.0 if correct == 'y' else 0.5 if correct == 'partial' else 0.0

        results.append({
            "test": test.name,
            "time": elapsed,
            "score": score,
            "response": response
        })

    accuracy = sum(r["score"] for r in results) / len(results) * 100
    avg_time = sum(r["time"] for r in results) / len(results)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "results": results
    }

# Run comparison
if __name__ == "__main__":
    print("DeepSeek-R1 Model Size Evaluation")
    print("=" * 80)

    all_tests = IDENTITY_TESTS + CONFLICT_TESTS + MULTI_STEP_TESTS

    print(f"\nEvaluating deepseek-r1:14b...")
    results_14b = evaluate_model("deepseek-r1:14b", all_tests)

    print(f"\n\nEvaluating deepseek-r1:32b...")
    results_32b = evaluate_model("deepseek-r1:32b", all_tests)

    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\ndeepseek-r1:14b:")
    print(f"  Accuracy: {results_14b['accuracy']:.1f}%")
    print(f"  Avg Time: {results_14b['avg_time']:.1f}s")

    print(f"\ndeepseek-r1:32b:")
    print(f"  Accuracy: {results_32b['accuracy']:.1f}%")
    print(f"  Avg Time: {results_32b['avg_time']:.1f}s")

    print(f"\nAccuracy Improvement: {results_32b['accuracy'] - results_14b['accuracy']:.1f} percentage points")
    print(f"Speed Tradeoff: {results_32b['avg_time'] / results_14b['avg_time']:.2f}x slower")

    print("\nRecommendation:")
    if results_32b['accuracy'] - results_14b['accuracy'] > 10:
        print("  → Use deepseek-r1:32b (significant accuracy gain)")
    elif results_32b['accuracy'] - results_14b['accuracy'] > 5:
        print("  → Use deepseek-r1:32b if speed is acceptable")
    else:
        print("  → Use deepseek-r1:14b (minimal accuracy difference)")
```

### Decision Criteria

After running the evaluation:

- **If accuracy improvement > 10%**: Use 32b despite slower speed
- **If accuracy improvement 5-10%**: User's call based on speed tolerance
- **If accuracy improvement < 5%**: Use 14b for better speed

### Deliverable

Document in this file:
```markdown
## Phase 0 Results

**Model Evaluated**: deepseek-r1:14b vs deepseek-r1:32b
**Date**: [date of test]
**Test Cases**: [number] identity + [number] conflict + [number] multi-step

### Results

| Model | Accuracy | Avg Response Time | Notes |
|-------|----------|-------------------|-------|
| deepseek-r1:14b | X% | Xs | ... |
| deepseek-r1:32b | Y% | Ys | ... |

### Decision

✅ Selected: **deepseek-r1:[14b/32b]** for gene-reasoner

Rationale: [brief explanation]
```

---

## Phase 1: Create Model Router Service ✅ COMPLETED

**Goal**: Implement flexible routing logic that can be easily swapped/tuned.

### Implemented Files

- ✅ `genealogy/services/model_router.py` - Main ModelRouter class
- ✅ `genealogy/services/routing_strategies.py` - Pluggable strategies (KeywordBasedStrategy, AgentOnlyStrategy, AlwaysFastStrategy, AlwaysMainStrategy)
- ✅ `genealogy/tests/test_model_routing.py` - 31 unit tests (all passing)
- ✅ Updated `genealogy/services/__init__.py` to export router classes

### Testing Results

✅ **31/31 unit tests passing**

Test coverage:
- Merge/conflict detection (English & Dutch keywords including "dezelfde")
- Agent mode routing (gene-chat-main for non-merge agent queries)
- Merge queries take precedence even in agent mode → gene-reasoner
- Complexity-based routing (query length + chunk count/size)
- Edge cases (empty chunks, None chunks, case insensitivity)
- Strategy pattern (pluggable strategies work correctly)
- ModelRouter class (delegation, strategy swapping)

---

## ~~Phase 1: Create Model Router Service~~

**Goal**: ~~Implement flexible routing logic that can be easily swapped/tuned.~~

### File Structure

```
genealogy/
└── services/
    ├── model_router.py          # Main routing logic
    └── routing_strategies.py    # Pluggable strategy implementations
```

### 1.1 Core Router (`model_router.py`)

```python
"""
Model routing for genealogy queries.

Usage:
    from genealogy.services.model_router import route_model

    model = route_model(
        query="Are these the same person?",
        chunks=[...],
        use_agent=True
    )
    # Returns: "gene-reasoner"
"""

import logging
from typing import List, Protocol

logger = logging.getLogger(__name__)

# Model name constants
MODEL_FAST = "gene-chat-fast"
MODEL_MAIN = "gene-chat-main"
MODEL_REASONER = "gene-reasoner"


class RoutingStrategy(Protocol):
    """Interface for routing strategies - allows easy swapping"""

    def choose_model(
        self,
        query: str,
        chunks: List[dict] | None,
        use_agent: bool
    ) -> str:
        """Choose model based on query characteristics"""
        ...


def route_model(
    query: str,
    chunks: List[dict] | None = None,
    use_agent: bool = False,
    strategy: RoutingStrategy | None = None
) -> str:
    """
    Route query to appropriate model.

    Args:
        query: User's question
        chunks: Retrieved RAG chunks (optional)
        use_agent: Whether using agentic tool execution
        strategy: Custom routing strategy (defaults to KeywordComplexityStrategy)

    Returns:
        Model name to use
    """
    if strategy is None:
        from genealogy.services.routing_strategies import KeywordComplexityStrategy
        strategy = KeywordComplexityStrategy()

    model = strategy.choose_model(query, chunks, use_agent)
    logger.info(f"Routed query to {model}: query_len={len(query)} chunks={len(chunks or [])}")

    return model
```

### 1.2 Routing Strategies (`routing_strategies.py`)

This file contains pluggable strategies that can be easily swapped:

```python
"""
Routing strategy implementations.

Create new strategies by implementing the RoutingStrategy protocol.
"""

import re
from typing import List


class KeywordComplexityStrategy:
    """
    Route based on keywords (merge/conflict detection) and complexity scoring.

    This is the initial simple strategy from the MODEL_ROUTING.md doc.
    """

    MERGE_KEYWORDS = [
        r"\bsame person\b",
        r"\bsame man\b",
        r"\bsame woman\b",
        r"\bduplicate(s)?\b",
        r"\breconcile\b",
        r"\bmerge\b",
        # ... rest of keywords
    ]

    def is_merge_query(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(pat, q) for pat in self.MERGE_KEYWORDS)

    def estimate_complexity(self, query: str, chunks: List[dict]) -> int:
        """Score: 0-3+ based on query length and chunk count/size"""
        # Implementation from MODEL_ROUTING.md
        ...

    def choose_model(self, query: str, chunks: List[dict] | None, use_agent: bool) -> str:
        from genealogy.services.model_router import MODEL_FAST, MODEL_MAIN, MODEL_REASONER

        chunks = chunks or []

        # 1) Merge/conflict → reasoner
        if self.is_merge_query(query):
            return MODEL_REASONER

        # 2) Agent mode → main model
        if use_agent:
            return MODEL_MAIN

        # 3) Complexity-based
        complexity = self.estimate_complexity(query, chunks)
        if complexity >= 2:
            return MODEL_MAIN

        return MODEL_FAST


class MLBasedStrategy:
    """
    Future: ML classifier to predict optimal model.

    Could train on:
    - Query embeddings
    - Chunk count/types
    - Historical performance data
    - User feedback signals
    """

    def choose_model(self, query: str, chunks: List[dict] | None, use_agent: bool) -> str:
        # Placeholder for future ML-based routing
        raise NotImplementedError("ML routing not yet implemented")


class HeuristicStrategy:
    """
    Future: More sophisticated heuristics.

    Could consider:
    - Dutch text detection in chunks
    - Follow-up question detection
    - Named entity density
    - Date/place density
    """

    def choose_model(self, query: str, chunks: List[dict] | None, use_agent: bool) -> str:
        raise NotImplementedError("Advanced heuristics not yet implemented")
```

### 1.3 Configuration

Add to `genealogy_extractor/settings.py`:

```python
# Model routing configuration
MODEL_ROUTING_STRATEGY = "KeywordComplexityStrategy"  # Can swap to other strategies
```

### Testing

**File**: `genealogy/tests/test_model_router.py`

```python
"""Tests for model routing logic"""

from genealogy.services.model_router import route_model, MODEL_FAST, MODEL_MAIN, MODEL_REASONER


def test_merge_query_routes_to_reasoner():
    result = route_model("Are these the same person?", chunks=[])
    assert result == MODEL_REASONER


def test_simple_query_routes_to_fast():
    result = route_model("Who was Bessel?", chunks=[])
    assert result == MODEL_FAST


def test_complex_query_routes_to_main():
    chunks = [{"text_content": "x" * 5000} for _ in range(6)]
    result = route_model("Complex multi-doc question", chunks=chunks)
    assert result == MODEL_MAIN


def test_agent_mode_routes_to_main():
    result = route_model("Simple question", chunks=[], use_agent=True)
    assert result == MODEL_MAIN
```

---

## Phase 2: Update Chat View Integration

**Goal**: Replace manual model selection with automatic routing.

### 2.1 Remove Model Selection UI

**Files to modify**:
- `genealogy/templates/genealogy/chat/conversation.html` - Remove model dropdown
- `genealogy/views/chat.py` - Remove `selected_model` from POST parameters

### 2.2 Integrate Routing in View

**File**: `genealogy/views/chat.py`

```python
# BEFORE (line ~73):
selected_model = request.POST.get('model', get_default_models()['llm_model'])

# AFTER:
from genealogy.services.model_router import route_model

# Don't route yet - need to know if we have chunks
# For agent mode, route immediately
if use_agent:
    selected_model = route_model(
        query=user_message,
        chunks=None,  # Agent doesn't use RAG upfront
        use_agent=True
    )
else:
    # For RAG mode, route after retrieval
    selected_model = None  # Will set after retrieval
```

Then later in RAG mode (around line 209):

```python
# BEFORE:
chunks = retriever.retrieve(
    query=user_message,
    top_k=5,
    expand_window=1
)

# AFTER:
chunks = retriever.retrieve(
    query=user_message,
    top_k=5,
    expand_window=1
)

# NOW route based on retrieved chunks
selected_model = route_model(
    query=user_message,
    chunks=chunks,
    use_agent=False
)

# Emit model selection to frontend for transparency
yield f"data: {json.dumps({
    'status': 'model_selected',
    'model': selected_model
})}\n\n"
```

### 2.3 Update Frontend to Show Model Selection

**File**: `genealogy/templates/genealogy/chat/conversation.html` (or wherever SSE is handled)

Add handler for `model_selected` event to show user which model is being used:

```javascript
if (data.status === 'model_selected') {
    // Show subtle indicator: "Using fast model" / "Using reasoning model"
    const modelName = data.model.replace('gene-chat-', '').replace('gene-', '');
    console.log(`Model selected: ${data.model}`);
    // Could show badge in UI if desired
}
```

---

## Phase 3: Update Agent Executor

**Goal**: Ensure AgentExecutor uses routed model correctly.

### 3.1 Current State

Agent executor receives model in constructor (line 110 of chat.py):
```python
agent = AgentExecutor(model=selected_model, max_iterations=10, timeout=300)
```

### 3.2 No Changes Needed

Since we're routing in the view before constructing AgentExecutor, no changes needed in `agent_executor.py`. The routed model is passed in and used throughout.

**Future Enhancement** (Phase 4+): Could route *per iteration* if needed:
- Use fast model for tool parsing
- Use main/reasoner model for final synthesis
- Requires more complex logic in `execute_streaming()`

---

## Phase 4: Update Default Models & Environment

**Goal**: Update defaults to use new model lineup.

### 4.1 Update `ollama_utils.py`

```python
# BEFORE:
def get_default_models() -> dict[str, str]:
    return {
        "llm_model": os.getenv("OLLAMA_LLM_MODEL", "llama3.1:70b"),
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "zylonai/multilingual-e5-large:latest"),
    }

# AFTER:
def get_default_models() -> dict[str, str]:
    return {
        "llm_model": os.getenv("OLLAMA_LLM_MODEL", "gene-chat-fast"),  # Fast by default
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "zylonai/multilingual-e5-large:latest"),
    }
```

### 4.2 Update `.env` / `docker-compose.yml`

```bash
# .env
OLLAMA_LLM_MODEL=gene-chat-fast
```

---

## Phase 5: Testing & Validation

### 5.1 Manual Testing Checklist

Test each routing path:

- [ ] **Merge query** → Should route to gene-reasoner
  - Query: "Are these baptism records for the same person?"

- [ ] **Simple question, few chunks** → Should route to gene-chat-fast
  - Query: "Who was Bessel van Zanten?"

- [ ] **Complex question, many chunks** → Should route to gene-chat-main
  - Query: "Compare these 5 marriage records and identify patterns"

- [ ] **Agent mode** → Should route to gene-chat-main
  - Query: "Tell me about Bessel van Zanten" (with agent enabled)

- [ ] **Follow-up question** → Should route appropriately
  - Query: "What about his children?" (after previous question)

### 5.2 Integration Tests

**File**: `genealogy/tests/test_chat_routing_integration.py`

```python
"""Integration tests for model routing in chat"""

@pytest.mark.django_db
def test_chat_routes_merge_query_to_reasoner(client):
    """Merge queries should use reasoning model"""
    # Create conversation
    # POST message "Are these the same person?"
    # Capture SSE stream
    # Assert model_selected event shows gene-reasoner
    ...

@pytest.mark.django_db
def test_chat_routes_simple_query_to_fast(client):
    """Simple queries should use fast model"""
    ...

@pytest.mark.django_db
def test_chat_routes_agent_to_main(client):
    """Agent mode should use main model"""
    ...
```

### 5.3 Performance Monitoring

Add logging to track:
- Model selection distribution (how often each model is used)
- Response times by model
- User feedback signals (if available)

```python
# In model_router.py
logger.info(
    f"Model routed: {model}",
    extra={
        "query_length": len(query),
        "chunk_count": len(chunks or []),
        "use_agent": use_agent,
        "model": model
    }
)
```

---

## Phase 6: Documentation & Rollout

### 6.1 Update User Documentation

Create `docs/MODEL_SELECTION.md`:

```markdown
# Automatic Model Selection

The system automatically selects the optimal model for each query:

## Models

1. **Fast Model** (llama3.1:8b)
   - Used for: Simple questions, quick lookups
   - Speed: ~2-3s response time

2. **Main Model** (qwen2.5:14b)
   - Used for: Complex queries, multiple documents, agentic reasoning
   - Speed: ~5-10s response time

3. **Reasoning Model** (deepseek-r1:14b or 32b)
   - Used for: Identity resolution, conflicting records, merge decisions
   - Speed: ~10-15s response time

## How Routing Works

The system analyzes your question and automatically picks the best model based on:
- Keywords indicating merge/conflict resolution
- Number and size of relevant documents
- Query complexity
- Whether agent tools are needed

You'll see a subtle indicator showing which model is being used.
```

### 6.2 Update Admin Documentation

Document for yourself how to:
- Swap routing strategies
- Monitor model usage
- Tune routing thresholds
- Add new models to the lineup

---

## Success Criteria

### Phase 0 (Model Evaluation)
- [ ] Completed evaluation of 14b vs 32b on 10+ test cases
- [ ] Documented results and made decision
- [ ] Created custom Modelfile for gene-reasoner

### Phase 1 (Router Implementation)
- [ ] `model_router.py` created with pluggable strategy design
- [ ] `routing_strategies.py` with KeywordComplexityStrategy
- [ ] Unit tests passing

### Phase 2 (View Integration)
- [ ] Removed model selection UI
- [ ] Integrated routing in chat view (both RAG and agent modes)
- [ ] Frontend shows model selection transparently

### Phase 3 (Agent Updates)
- [ ] Verified agent uses routed model correctly
- [ ] No breaking changes

### Phase 4 (Defaults)
- [ ] Updated default model to gene-chat-fast
- [ ] Environment configured correctly

### Phase 5 (Testing)
- [ ] Manual testing checklist completed
- [ ] Integration tests written and passing
- [ ] Performance monitoring in place

### Phase 6 (Documentation)
- [ ] User documentation created
- [ ] Admin documentation updated
- [ ] Deployment notes written

---

## Rollout Plan

1. **Week 1**: Phase 0 - Model evaluation
   - Run tests
   - Make 14b vs 32b decision
   - Create gene-reasoner Modelfile

2. **Week 1-2**: Phases 1-3 - Implementation
   - Build router service
   - Integrate into views
   - Update agent if needed

3. **Week 2**: Phases 4-5 - Testing & Defaults
   - Update defaults
   - Manual testing
   - Write integration tests

4. **Week 2-3**: Phase 6 - Documentation & Deploy
   - Write docs
   - Deploy to production
   - Monitor usage patterns

---

## Future Enhancements

### Post-Launch Improvements

1. **Adaptive Routing**: Learn from user feedback
   - Track which queries work well with which models
   - Adjust routing thresholds over time

2. **Per-Iteration Agent Routing**: Route differently within agent execution
   - Fast model for tool parsing
   - Main/reasoner model for synthesis

3. **Hybrid Strategies**: Combine multiple signals
   - Dutch text detection
   - Named entity density
   - Query embeddings similarity to past queries

4. **User Override**: Allow power users to manually select models
   - Hidden setting or admin-only feature
   - For testing and comparison

5. **A/B Testing Framework**: Systematically compare routing strategies
   - Track metrics per strategy
   - Auto-select winning strategy
