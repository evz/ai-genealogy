# Prompt Auditing System

## Overview

This document describes the new prompt auditing and management system implemented to improve LLM chat quality through systematic prompt engineering, versioning, and effectiveness tracking.

## Problem Statement

The original chat system had several issues that made it difficult to improve LLM responses:

1. **Fragmented prompt logic**: System prompts baked into Modelfiles, runtime prompts in Python code, example prompts in text files
2. **No auditability**: No way to see which exact prompt generated which response
3. **Difficult iteration**: Changing prompts required rebuilding Ollama Modelfiles
4. **No metrics**: No data on which prompts work better

## Solution Architecture

### 1. Base Models with Dynamic System Prompts

**Before**: Custom Ollama Modelfiles with baked-in system prompts
- `gene-chat-fast-agent` (Llama 3.1 8B + custom system prompt)
- `gene-chat-main-agent` (Qwen 2.5 14B + custom system prompt)
- `gene-reasoner-agent` (DeepSeek-R1 14B + custom system prompt)

**After**: Base models with runtime system prompts
- `llama3.1:8b` (fast)
- `qwen2.5:14b-instruct-q5_K_M` (main)
- `deepseek-r1:14b` (reasoning)

System prompts are now passed dynamically on each API call.

### 2. Centralized Template System

**Location**: `genealogy/prompts/templates/`

Templates are versioned text files:
- `agent_system_v1.txt` - System prompt for agent mode
- `agent_user_v1.txt` - User prompt template with variables

**Template Variables** (for `agent_user_v1.txt`):
- `{user_query}` - The user's question
- `{tools_description}` - Available tools and parameters
- `{num_calls}` - Number of tool calls made so far
- `{max_iterations}` - Maximum iterations allowed
- `{previous_calls}` - History of previous tool calls
- `{context}` - Accumulated context from tool results

### 3. PromptRegistry Class

**Location**: `genealogy/services/prompt_registry.py`

Manages prompt templates and logging:

```python
registry = PromptRegistry()

# Load a template
registry.load_template(
    template_key="agent_v1",
    system_filename="agent_system_v1.txt",
    user_filename="agent_user_v1.txt",
    version="1"
)

# Render with variables
prompt_data = registry.render_prompt(
    template_name="agent_v1",
    user_query="Who is Bessel van Zanten?",
    tools_description="...",
    ...
)

# Returns:
# {
#     "system": "...",
#     "user": "...",
#     "template_name": "agent",
#     "template_version": "1",
#     "variables": {...}
# }
```

### 4. PromptLog Model

**Location**: `genealogy/models.py`

Logs every prompt sent to an LLM:

```python
class PromptLog(models.Model):
    message = ForeignKey(Message)  # Link to conversation

    # Template info
    prompt_template_name = CharField()  # e.g., "agent"
    prompt_version = CharField()  # e.g., "1"

    # Prompts
    system_prompt = TextField()
    user_prompt = TextField()
    full_prompt = TextField()
    prompt_variables = JSONField()

    # Execution
    model_name = CharField()  # e.g., "qwen2.5:14b-instruct-q5_K_M"
    iteration = PositiveIntegerField()
    tool_calls = JSONField()

    # Response
    llm_response = TextField()
    parsed_successfully = BooleanField()
    parse_error = TextField()

    # Performance
    latency_ms = IntegerField()
    token_count_prompt = IntegerField()
    token_count_response = IntegerField()
```

### 5. Updated AgentExecutor

**Location**: `genealogy/services/agent_executor.py`

Now supports:
- Dynamic system prompts via PromptRegistry
- Automatic prompt logging to database
- Base model support

```python
agent = AgentExecutor(
    model="qwen2.5:14b-instruct-q5_K_M",  # Base model
    max_iterations=20,
    timeout=300,
    prompt_template="agent_v1",  # Optional, defaults to agent_v1
    message_instance=assistant_msg  # For logging
)
```

### 6. Updated Model Router

**Location**: `genealogy/services/routing_strategies.py`

Routes to base models instead of custom Modelfiles:
- Simple queries → `llama3.1:8b`
- Standard complexity → `qwen2.5:14b-instruct-q5_K_M`
- Complex reasoning/merging → `deepseek-r1:14b`

### 7. Django Admin Interface

**Location**: `genealogy/admin/prompt_log.py`, `genealogy/admin/conversation.py`

Browse and analyze prompts through Django admin:
- View all prompts for a message
- See exact system and user prompts sent
- Filter by template version, model, success rate
- Analyze performance metrics (latency, tokens)

## How to Use This System

### Viewing Prompts for Debugging

1. **Via Django Admin**:
   - Go to `/admin/genealogy/promptlog/`
   - Filter by date, model, or template version
   - Click on a log to see full prompts and response

2. **Via Message**:
   - Go to `/admin/genealogy/message/`
   - Click on a message
   - Click "View X prompt log(s)" link

### Testing New Prompts (A/B Testing)

1. **Create new template version**:
   ```bash
   # Create new prompt files
   genealogy/prompts/templates/agent_system_v2.txt
   genealogy/prompts/templates/agent_user_v2.txt
   ```

2. **Update AgentExecutor initialization**:
   ```python
   agent = AgentExecutor(
       model=selected_model,
       prompt_template="agent_v2",  # Changed from v1
       message_instance=assistant_msg
   )
   ```

3. **Compare effectiveness**:
   ```python
   from genealogy.services.prompt_registry import PromptRegistry

   registry = PromptRegistry()

   # Get metrics for v1
   v1_metrics = registry.get_effectiveness_metrics(
       template_name="agent",
       template_version="1"
   )

   # Get metrics for v2
   v2_metrics = registry.get_effectiveness_metrics(
       template_name="agent",
       template_version="2"
   )

   # Compare:
   # - success_rate (% of prompts that parsed correctly)
   # - avg_latency_ms
   # - avg_iterations (how many tool calls needed)
   # - avg_prompt_tokens
   ```

### Simplifying Prompts

Current system prompt (`agent_system_v1.txt`) is intentionally minimal. To test if instructions can be simplified:

1. Create `agent_system_v2.txt` with even fewer instructions
2. Deploy and test with real queries
3. Use Django admin to compare:
   - Parse success rate (v1 vs v2)
   - Average iterations (does simpler prompt need more tool calls?)
   - Response quality (manual review)

### Analyzing Failures

When a query returns a bad answer:

1. Find the conversation in Django admin
2. Click through to the assistant message
3. View all prompt logs for that message
4. Examine:
   - Which iteration failed? (parse_error field)
   - What was the exact prompt at that iteration?
   - What did the LLM respond?
   - Which tool calls were made?

## Files Changed

### New Files
- `genealogy/models.py` - Added `PromptLog` model
- `genealogy/services/prompt_registry.py` - Template management and logging
- `genealogy/prompts/templates/agent_system_v1.txt` - System prompt
- `genealogy/prompts/templates/agent_user_v1.txt` - User prompt template
- `genealogy/admin/prompt_log.py` - Admin interface for PromptLog
- `genealogy/admin/conversation.py` - Admin for Conversation/Message
- `genealogy/migrations/0043_promptlog.py` - Database migration

### Modified Files
- `genealogy/services/agent_executor.py` - Use PromptRegistry, log prompts, support base models
- `genealogy/services/routing_strategies.py` - Route to base models
- `genealogy/views/chat.py` - Create message early, pass to AgentExecutor
- `genealogy/ollama_utils.py` - Add `system` parameter support
- `genealogy/admin/__init__.py` - Register new admin classes

## Next Steps

### Immediate
1. ✅ Test chat interface to ensure prompts are being logged
2. Run test queries and verify PromptLog entries appear in admin
3. Review logged prompts to identify issues

### Short-term
1. Create `agent_system_v2.txt` with simplified instructions
2. A/B test v1 vs v2 on known "problem queries"
3. Analyze which version has better parse success rate
4. Build a script to bulk-test queries against different prompt versions

### Long-term
1. Create prompt templates for other LLM tasks (extraction, OCR correction, etc.)
2. Build web UI for prompt effectiveness dashboard
3. Implement automatic prompt optimization based on success metrics
4. Add support for conversation-level prompt switching

## Key Design Decisions

1. **No global singleton for PromptRegistry**: Each AgentExecutor creates its own instance. Simpler, more testable.

2. **Version in filename**: Templates use `agent_system_v1.txt` not `agent_system.txt` + metadata. Makes versions explicit and prevents accidental overwrites.

3. **Separate system/user templates**: Allows testing different combinations. System prompt controls behavior, user prompt structures the request.

4. **Logging at iteration level**: Each iteration (tool call) gets its own PromptLog entry. Critical for debugging agent loops.

5. **Message created before execution**: Allows linking PromptLogs to Message during execution, not after.

## Troubleshooting

### Prompts not appearing in admin
- Check that `message_instance` is passed to AgentExecutor
- Verify migration ran: `docker compose exec web python manage.py migrate genealogy`
- Check Django logs for errors during logging

### Template not found error
- Ensure template files exist in `genealogy/prompts/templates/`
- Check filename matches pattern: `agent_system_v{N}.txt`, `agent_user_v{N}.txt`
- Verify template_key matches (e.g., "agent_v1")

### System prompt not being used
- Check OllamaClient.generate_stream() includes `system` parameter
- Verify Ollama version supports system prompts (should be v0.1.0+)
- Try logging the payload sent to Ollama

## Metrics to Track

Key metrics for evaluating prompt effectiveness:

1. **Parse Success Rate**: % of responses that matched expected format (TOOL_CALL vs ANSWER)
2. **Average Iterations**: How many tool calls before reaching answer
3. **Latency**: Response time per iteration
4. **Token Efficiency**: Prompt tokens vs response quality
5. **Duplicate Call Rate**: How often LLM repeats same tool call

Access via:
```python
registry.get_effectiveness_metrics(template_name="agent", template_version="1")
```
