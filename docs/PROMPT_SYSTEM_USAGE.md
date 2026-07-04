# Prompt Template System - Usage Guide

## What Changed

Your genealogy chat now uses **database-backed prompt templates** instead of hard-coded prompts or Modelfiles. This makes it easy to test and improve prompts without touching code.

## Quick Start

### View Your Current Template

1. Go to Django admin: `/admin/genealogy/prompttemplate/`
2. You'll see "agent v1 (ACTIVE)" - this is your current prompt
3. Click it to view/edit

### Test a New Prompt Version

1. In Django admin, find "agent v1" and click it
2. Click the "Clone" button (or use the "Clone as new version" action)
3. You'll get "agent v2" with the same content
4. Edit the system_template or user_template
5. Add description: "Testing simpler instructions"
6. Save (leave is_active = False for now)

### Test v2 in a Specific Conversation

**Option A: Per-conversation (safer)**
1. Go to `/admin/genealogy/conversation/`
2. Find your test conversation
3. Set "Prompt template override" to "agent v2"
4. Save
5. Only THIS conversation uses v2, everyone else still uses v1

**Option B: Global switch (affects everyone)**
1. Go to `/admin/genealogy/prompttemplate/`
2. Click "agent v2"
3. Check "is_active" checkbox
4. Save
5. This automatically deactivates v1
6. All NEW conversations now use v2

### Compare Results

1. Go to `/admin/genealogy/promptlog/`
2. Filter by "Prompt version":
   - Select "1" to see v1 results
   - Select "2" to see v2 results
3. Compare:
   - **Success rate**: % of responses that parsed correctly
   - **Avg latency**: How fast the model responds
   - **Avg iterations**: How many tool calls before answering

### Roll Back if Needed

If v2 is worse:
1. Go back to `/admin/genealogy/prompttemplate/`
2. Click "agent v1"
3. Check "is_active"
4. Save
5. Instant rollback!

## Example Workflow: Simplifying the Prompt

Current v1 has 5 "CRITICAL RULES". Let's test if we can reduce to 3:

**Step 1: Clone v1 as v2**
- Django admin → PromptTemplate → agent v1 → Clone as new version
- Auto-creates "agent v2"

**Step 2: Edit v2**
```
CRITICAL RULES:
1. Start responses with "TOOL_CALL:" or "ANSWER:"
2. Extract IDs from search results, don't use names
3. If multiple matches, list ALL and ask user to clarify
```

Description: "Simplified to 3 rules, removed duplicate call and context tracking (let model figure it out)"

**Step 3: Test on one conversation**
- Go to Conversations → pick a test conversation
- Set "prompt_template_override" to "agent v2"
- Ask it queries you know the answers to

**Step 4: Compare**
- Go to PromptLog
- Filter by "prompt_version = 2"
- Check success rate

**Step 5: Decide**
- If v2 success rate >= v1: Make v2 active (global)
- If v2 worse: Archive it, keep using v1

## Template Variables

Your agent prompt uses these variables (auto-filled by AgentExecutor):

| Variable | Example | Purpose |
|----------|---------|---------|
| `{user_query}` | "Who is Bessel van Zanten?" | The user's question |
| `{tools_description}` | "- search_person_by_name: ..." | List of available tools |
| `{num_calls}` | 2 | How many tool calls so far |
| `{max_iterations}` | 20 | Maximum allowed |
| `{previous_calls}` | "Iteration 1: search_person_by_name(...)" | Tool call history |
| `{context}` | Tool results accumulated | What the agent has learned |

You can test template rendering with example_variables in admin.

## Admin Features

### Actions

- **Set as active**: Make this version the default
- **Archive**: Hide old experiments
- **Clone as new version**: Copy to create v3, v4, etc.

### Stats for Each Version

Click a template to see:
- **Total uses**: How many times it's been used
- **Success rate**: % of successful parses
- **Avg latency**: Response time
- **Avg iterations**: Tool calls per query
- **Avg tokens**: Input → output size

### Preview

Set `example_variables` to see how template renders with real data.

## Advanced: A/B Testing

Want to compare v1 vs v2 scientifically?

1. Keep v1 active (most users get this)
2. Set 5 test conversations to use v2 (prompt_template_override)
3. After 20 queries each, compare stats
4. Winner becomes active

## Troubleshooting

### "No active template found for 'agent'"

- Go to `/admin/genealogy/prompttemplate/`
- Find "agent v1" (or any version)
- Check "is_active"
- Save

### Template not rendering

- Click the template in admin
- Scroll to "Preview (with example variables)"
- Error shown here if variables are wrong
- Check `required_variables` matches what's in template

### Changes not taking effect

- Remember: is_active templates are cached per request
- Try in new browser session / incognito
- Or: restart Django (`docker compose restart web`)

## Best Practices

1. **Always test in one conversation first** before making a template globally active
2. **Write good descriptions** so you remember what you changed 6 months later
3. **Archive old experiments** to keep the list clean
4. **Never delete v1** - it's your baseline for comparison
5. **Check stats before rolling out** - at least 10 uses before trusting metrics

## What's Logged

Every time the agent calls the LLM:
- Full system + user prompt sent
- Which template version was used
- All variables that were filled in
- LLM's raw response
- Whether it parsed successfully
- How long it took
- Token counts

This lets you debug exactly what prompt caused a bad answer.

## Next Steps

Now that you have this system:
1. Test the chat with queries you know answers to
2. Find where it fails
3. Look at the PromptLog to see which prompt iteration caused the failure
4. Create v2 with improvements
5. Test and compare
6. Repeat until you have prompts that work reliably!
