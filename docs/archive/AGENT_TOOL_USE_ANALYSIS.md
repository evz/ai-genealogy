# Agent Tool Use Analysis

Analysis of three problematic conversations to identify why the LLM agent is struggling to use tools effectively.

## Problem Summary

**Conversation 1 (2f9d4ae9)**: Agent failed to get children of Aart van Zanten
- Used `get_children({'person_id': 'Aart (Arie) van Zanten'})`
- Returned error: "Person not found: Aart (Arie) van Zanten"
- Agent repeated the same failing call 10 times

**Conversation 2 (39e6d0a4)**: Agent failed to compute relationship between Eugene and Julia
- Made 10 tool calls but couldn't find common ancestor
- Kept calling `get_parents` with wrong IDs
- Hit max iterations without completing task

**Conversation 3 (4c1be85a)**: Agent missed military service and orphan status for Bessel VI.1.n
- Called `get_person_details({'person_id': 'VI.1.n'})`
- Response mentioned "ontslag uit het leger" (discharge from army) under "Other" events
- Agent said "no explicit mention" of military service despite the discharge event
- Never mentioned orphan status

## Root Causes

### 1. **LLM Is Passing Wrong person_id Format**

The most critical issue in Conversation 1:

**What LLM did:**
```python
get_children({'person_id': 'Aart (Arie) van Zanten'})  # Name string
```

**What it should do:**
```python
get_children({'person_id': 'eabfafc3-4899-4291-804f-300cf12cefc3'})  # UUID from previous search
# OR
get_children({'person_id': 'VI.1.a'})  # Genealogical ID
```

The tool documentation says `person_id: "Identity UUID or genealogical ID (e.g., 'II.3.a')"` but the LLM is passing full name strings like "Aart (Arie) van Zanten" which aren't valid identifiers.

**Why this happens:**
- `search_person_by_name` returns descriptive text: "Aart (Arie) van Zanten - born in Naarden, died January 5, 1848..."
- The LLM copies this full description as the `person_id` argument
- The tool then fails because it expects a UUID or genealogical_id

**Fix needed:**
- Change `search_person_by_name` to return structured data with explicit `id` and `genealogical_id` fields
- Update agent prompt to be VERY explicit: "Use the 'id' field from search results, NOT the full description"

### 2. **Tool Output Isn't Structured Enough**

Looking at Conversation 1, when `get_person_details` was called with a UUID, it worked fine. But when `search_person_by_name` returns results, they look like:

```
"Aart (Arie) van Zanten - born in Naarden, died January 5, 1848 in Naarden; parents were Ariën van Zanten and Marritje Bakker."
```

This is human-readable text, not a structured object. The LLM struggles to parse out the UUID/ID to use in subsequent calls.

**Fix needed:**
```json
{
  "count": 3,
  "people": [
    {
      "id": "eabfafc3-4899-4291-804f-300cf12cefc3",
      "genealogical_id": "VI.1.a",
      "name": "Aart (Arie) van Zanten",
      "birth_year": null,
      "birth_place": "Naarden",
      "death_year": 1848,
      "death_place": "Naarden",
      "summary": "Born in Naarden, died January 5, 1848 in Naarden"
    }
  ]
}
```

### 3. **LLM Doesn't Understand Tool Return Values**

In Conversation 3, `get_person_details` returned events including:

```json
{
  "events": [
    ...
    {
      "event_type": "OTHR",
      "description": "ontslag uit het leger",  // discharge from army
      ...
    }
  ]
}
```

The agent said "no explicit mention of military service" despite this event clearly indicating army service.

**Why:**
- Event type is "OTHR" (Other), not "MILI" (Military)
- Description is in Dutch: "ontslag uit het leger"
- Agent may not connect "discharge" with "served in military"

**Fix needed:**
- Add explicit military event categorization during extraction
- Include translations in event descriptions: "ontslag uit het leger (discharge from army)"
- Improve agent prompt: "OTHR events may contain important military, orphan, or other status information - read descriptions carefully"

### 4. **Agent Repeats Failed Calls**

In Conversation 1, the agent called `get_children({'person_id': 'Aart (Arie) van Zanten'})` **10 times in a row**, getting the same error each time.

The prompt says "Avoid calling the same tool repeatedly with the same arguments" but clearly the agent isn't following this.

**Fix needed:**
- Stronger prompt language: "NEVER repeat a tool call that returned an error. If a tool fails, try a different approach or different arguments."
- Add error reflection step: "If you get an error, explain what went wrong and what you'll try differently next time"
- Consider programmatic deduplication: track recent calls and return synthetic error if duplicate detected

### 5. **Relationship Computation Too Complex**

Conversation 2 shows the agent struggled to compute relationships between cousins. It needed to:
1. Get Eugene's parents
2. Get Harold's parents
3. Find common ancestor (Pieter - Eugene's grandfather, Harold's father)
4. Compute relationship (Eugene is Harold's nephew, so Eugene's son is Harold's grand-nephew, making Julia a second cousin once removed)

The agent hit 10 iterations without completing this.

**Why:**
- No `find_common_ancestor` tool
- Agent must manually traverse tree with multiple `get_parents` calls
- Complex genealogical calculations not well-suited for LLM

**Fix needed:**
- Add `find_relationship(person_id_1, person_id_2)` tool that does the graph traversal automatically
- Returns: `{"relationship": "second cousin once removed", "common_ancestor": {...}, "path_description": "..."}`

### 6. **Search Results Don't Include IDs**

From Conversation 1, `search_person_by_name` returns:
```
"1. Aart (Arie) van Zanten - born in Naarden, died January 5, 1848..."
```

But to call `get_person_details` or `get_children`, you need the UUID or genealogical_id. The search results don't explicitly show these.

Later the agent tried `search_by_birth_year` which did seem to return UUIDs (it successfully called `get_person_details` with a UUID after).

**Fix needed:**
- Make all search tools return consistent structured format with explicit `id` and `genealogical_id` fields
- Update prompt examples to show: "Use the 'id' field from search results, e.g., get_person_details({'person_id': result['id']})"

## Implementation Status

### ✅ COMPLETED

1. **Added source text to get_person_details**
   - Modified `genealogy/services/genealogy_tools.py` to include `source_texts` field
   - Returns full narrative text from chunks where person is the subject
   - Updated tool description in agent_executor.py to inform LLM about narrative context
   - Result: Agent now has access to rich narrative details (military service, orphan status, etc.) that aren't in structured events

2. **Updated agent prompt to be explicit about IDs**
   - Added section 3 "CRITICAL: EXTRACTING IDs FROM TOOL RESULTS" in agent_executor.py (lines 468-476)
   - Includes explicit examples of correct vs incorrect ID usage
   - Warns against using display_name or full name as person_id

3. **Improved tool parameter descriptions**
   - Updated parameter descriptions for get_person_details, get_children, get_parents (lines 36, 52, 59)
   - All now explicitly state: "NEVER use person's name" or "NEVER use display_name"
   - More detailed examples of valid ID formats (UUID and genealogical_id)

4. **Add programmatic duplicate detection**
   - Created reusable `_check_duplicate_call()` method (lines 87-110)
   - Applied to both execute_streaming() and execute() methods
   - Returns explicit error message directing agent to try different approach

5. **Add error reflection**
   - Enhanced section 4 "ERROR HANDLING AND REFLECTION" (lines 478-489)
   - Requires agent to include ERROR_REFLECTION and NEXT_ATTEMPT in THOUGHT when errors occur
   - Provides common error patterns and specific guidance on how to fix them
   - Includes concrete example of proper error reflection workflow

6. **Verified search_person_by_name returns structured JSON**
   - Confirmed search_person_by_name already returns structured format with id, genealogical_id, display_name fields (lines 186-199 in genealogy_tools.py)
   - This was already implemented correctly - no changes needed

## Recommended Fixes (Priority Order)

### HIGH PRIORITY

All HIGH PRIORITY items have been completed! ✅

7. **Add find_relationship tool** ✅ COMPLETED
   - Implemented BFS-based graph traversal to find most recent common ancestor (MRCA)
   - Computes relationships: parent/child, grandparent/grandchild, siblings, cousins, aunt/uncle/niece/nephew
   - Handles "removed" relationships (first cousin once removed, etc.)
   - Added to agent_executor.py AVAILABLE_TOOLS list (line 70-77)
   - Tested successfully with parent-child and sibling relationships

### MEDIUM PRIORITY

1. **Improve event categorization**
   - "ontslag uit het leger" should create MILI event, not OTHR
   - Add bilingual descriptions: "ontslag uit het leger (discharge from army)"
   - Update extraction prompt to recognize military-related terms

### LOW PRIORITY

1. **Better orphan status detection**
   - Extract orphan status as explicit OTHR event with clear label
   - Add to person summary: "Orphan status: Yes/No/Unknown"

## Test Cases to Validate Fixes

After implementing fixes, re-test these scenarios:

1. **Get children test:**
   ```
   Q: "Tell me about Aart van Zanten"
   Q: "How many of his children survived until adulthood?"
   Expected: Agent successfully gets children list
   ```

2. **Relationship test:**
   ```
   Q: "I am Eugene van Zanten's son. How am I related to Julia van Zanten?"
   Expected: Agent finds common ancestor and computes relationship
   ```

3. **Military service test:**
   ```
   Q: "Tell me about Bessel van Zanten VI.1.n"
   Q: "Did he ever serve in the military?"
   Expected: Agent recognizes "ontslag uit het leger" means military service
   ```

## Additional Observations

- **Model:** All three conversations used `gene-chat-main` (qwen2.5:14b) in agent mode
- **Token length:** Qwen2.5 14b should be capable of following complex instructions
- **Iteration limit:** 10 iterations often not enough for multi-step queries
- **No RAG retrieval:** Agent mode starts with empty context, forcing tool use (this is intentional)

The core problem isn't the model capability - it's the tool interface design. The LLM is trying to follow instructions but the tool outputs aren't structured in a way that makes the next steps obvious.
