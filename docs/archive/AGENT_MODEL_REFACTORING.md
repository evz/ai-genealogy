# Agent Model Refactoring Summary

## Problem Analysis

When investigating conversation `8589bd03-1611-4688-9929-68b0df630d68`, we found the agent failed with "Max iterations reached" when asked "how is Eugene van Zanten related to Pieter van Zanten?"

### Root Causes

1. **Massive, redundant prompts**: The `_build_agent_prompt()` method sent ~100 lines of instructions with every iteration, including tool protocol, error handling, workflow specs, and examples.

2. **Conflicting instructions**: The base models had existing system prompts focused on "short summaries with bullet lists" while the agentic prompt demanded "2-3 narrative paragraphs", creating confusion.

3. **No clear error recovery**: When the agent hit duplicate call errors, it didn't understand how to extract previous results from context.

4. **Workflow not enforced**: For relationship queries, the agent should search both people → extract IDs → call find_relationship. Instead it often tried calling find_relationship with names directly.

## Solution: System Prompt Architecture

We moved invariant instructions from the dynamic prompt into dedicated system prompts, creating three specialized agent models.

### What Goes in System Prompt vs Dynamic Prompt

**System Prompt (in Modelfile - set once):**
```
- Tool calling protocol (TOOL_CALL/ARGUMENTS/REASONING format)
- Error recovery patterns ("DUPLICATE CALL" → review previous iteration)
- ID extraction rules (use "id" or "genealogical_id", never "display_name")
- Required workflows (relationship → search both → extract IDs → find_relationship)
- Response style (narrative paragraphs, check source_texts, etc.)
- Domain expertise (Dutch patronymics, name variants, etc.)
```

**Dynamic Prompt (generated each iteration):**
```
- USER QUERY: <the question>
- AVAILABLE TOOLS: <list of tools>
- PREVIOUS TOOL CALLS: <what was already called>
- CURRENT CONTEXT: <tool results so far>
- IMPORTANT REMINDERS: <brief, specific to current state>
```

## Models Created

### 1. `gene-chat-fast-agent` (Llama 3.1 8B)
- **Temperature:** 0.3
- **Context:** 16K tokens
- **Best for:** Simple lookups (1-3 tool calls)
- **System prompt highlights:**
  - Concise responses (2-3 sentences)
  - Basic workflows only
  - Fast disambiguation

### 2. `gene-chat-main-agent` (Qwen2.5 14B) ⭐ Default
- **Temperature:** 0.1 (very focused)
- **Context:** 32K tokens
- **Best for:** Standard queries (2-6 tool calls)
- **System prompt highlights:**
  - Detailed error recovery with reflection
  - Complete workflow specifications
  - Narrative biographical responses
  - Full ID extraction protocol

### 3. `gene-reasoner-agent` (DeepSeek-R1 14B)
- **Temperature:** 0.2
- **Context:** 32K tokens
- **Best for:** Complex reasoning (4-10 tool calls)
- **System prompt highlights:**
  - Explicit `<think>` tags for reasoning
  - Identity merging protocols
  - Multi-hypothesis evaluation
  - Evidence comparison frameworks

## Benefits

### 1. Token Efficiency
- **Before:** ~100 lines sent every iteration × 10 iterations = ~1000 lines of redundant instructions
- **After:** ~18 lines per iteration, core instructions baked into system prompt
- **Savings:** ~80% reduction in prompt tokens per iteration

### 2. Consistency
- System prompt is always applied, can't be "forgotten" across iterations
- No conflicts between base model behavior and agentic instructions
- Clearer separation of concerns

### 3. Maintainability
- Change workflow rules? Update Modelfile and rebuild once
- No need to carefully edit mega-prompts buried in Python code
- Easier to test individual models: `ollama run gene-chat-main-agent`

### 4. Error Recovery
- Explicit patterns for common errors in system prompt
- Models "know" to review iteration N when duplicate call blocked
- Clear protocol for ID extraction from search results

## Code Changes

### `genealogy/services/agent_executor.py`

**Before (`_build_agent_prompt`):**
```python
return f"""You are a genealogy research assistant...

INSTRUCTIONS:
1. Review the CURRENT CONTEXT first...
2. Use conversation history...
3. CRITICAL: EXTRACTING IDs FROM TOOL RESULTS
   - When search_person_by_name returns...
   [... 90+ more lines ...]
"""
```

**After (`_build_agent_prompt`):**
```python
return f"""USER QUERY: {user_query}

AVAILABLE TOOLS:
{tools_description}

PREVIOUS TOOL CALLS ({len(tool_calls_made)}/{self.max_iterations} calls made):
{previous_tools}

CURRENT CONTEXT:
{context if context else "No context yet - search for information using tools."}

IMPORTANT REMINDERS:
- Use conversation context to resolve pronouns
- Extract "id" or "genealogical_id" from search results
- If DUPLICATE CALL error, review that iteration's result
- You have {self.max_iterations - len(tool_calls_made)} tool calls remaining

Respond with either TOOL_CALL or ANSWER:"""
```

**Line count reduction:** 115 lines → 18 lines (84% reduction)

### New Default Model

Update line 80 in `agent_executor.py`:
```python
def __init__(self, model: str = "gene-chat-main-agent", max_iterations: int = 10, ...):
```

## Testing the Fix

To verify this solves the original problem, test with the failing query:

```bash
# Rebuild models
cd modelfiles
ollama create gene-chat-main-agent -f gene-chat-main-agent.Modelfile

# Test in Django shell
docker compose exec web python manage.py shell

from genealogy.services.agent_executor import AgentExecutor

executor = AgentExecutor(model="gene-chat-main-agent")
result = executor.execute("How is Eugene van Zanten related to Pieter van Zanten?")
print(result)
```

### Expected Behavior

**Iteration 1:**
```
TOOL_CALL: search_person_by_name
ARGUMENTS: {"name": "Eugene van Zanten"}
REASONING: Need to find Eugene's ID first
```

**Iteration 2:**
```
TOOL_CALL: search_person_by_name
ARGUMENTS: {"name": "Pieter van Zanten"}
REASONING: Need to find Pieter's ID
```

**Iteration 3:**
```
TOOL_CALL: find_relationship
ARGUMENTS: {"person_id_1": "VIII.3.c", "person_id_2": "VII.3.a"}
REASONING: Now I have both IDs, can compute relationship
```

**Iteration 4:**
```
ANSWER: Eugene van Zanten (VIII.3.c) and Pieter van Zanten (VII.3.a) are second cousins.
They share a common ancestor, Bessel van Zanten (VI.1.n)...
```

## Migration Checklist

- [x] Create Modelfiles for three agent variants
- [x] Simplify `_build_agent_prompt()` in agent_executor.py
- [ ] Update default model in agent_executor.py to `gene-chat-main-agent`
- [ ] Build the new models: `ollama create gene-chat-main-agent -f ...`
- [ ] Test with the failing conversation query
- [ ] Add model routing logic (optional - route complex queries to gene-reasoner-agent)
- [ ] Update any documentation that references model names

## Future Enhancements

### 1. Automatic Model Routing
```python
def select_agent_model(query: str) -> str:
    if has_keywords(query, ["same person", "merge", "conflicting"]):
        return "gene-reasoner-agent"
    elif is_simple_lookup(query):
        return "gene-chat-fast-agent"
    else:
        return "gene-chat-main-agent"
```

### 2. Iteration Budget by Model
```python
ITERATION_BUDGETS = {
    "gene-chat-fast-agent": 5,
    "gene-chat-main-agent": 10,
    "gene-reasoner-agent": 15,
}
```

### 3. Tool Result Caching
Since we now block duplicate calls, consider caching tool results across conversations:
```python
# Cache key: (tool_name, frozen_args)
# Cache value: tool result
# TTL: 1 hour
```

### 4. Prompt Template Variants
For different query types, inject specific reminders:
```python
if query_type == "relationship":
    reminders += "\n- Remember: search both people first, then find_relationship"
```

## Monitoring

Track these metrics to validate the improvement:

1. **Success rate:** % of queries that complete without hitting max iterations
2. **Iterations per query:** Should decrease with clearer instructions
3. **Duplicate call rate:** Should be near zero with better workflow adherence
4. **Token usage:** Track prompt tokens per iteration (should be ~80% lower)

## References

- Original failing conversation: `8589bd03-1611-4688-9929-68b0df630d68`
- System prompts: `/modelfiles/*.Modelfile`
- Agent executor: `/genealogy/services/agent_executor.py`
- Tool definitions: `/genealogy/services/genealogy_tools.py`
