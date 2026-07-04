# Database-Backed Prompt Template System

## Problem

Currently, changing prompt versions requires code changes (modifying AgentExecutor initialization). This makes A/B testing and iteration cumbersome.

## Solution

Store prompt templates in the database with an "active" flag, allowing no-code template switching.

## Model Design

```python
class PromptTemplate(models.Model):
    """Versioned prompt templates stored in database."""

    # Identity
    name = CharField(max_length=100)  # e.g., "agent"
    version = CharField(max_length=20)  # e.g., "1", "2", "3"

    # Template content
    system_template = TextField()
    user_template = TextField()

    # Metadata
    description = TextField(blank=True)  # What changed in this version
    created_by = CharField(max_length=100, blank=True)  # Who created it
    created_at = DateTimeField(auto_now_add=True)

    # Status
    is_active = BooleanField(default=False)  # Only one active per name
    is_archived = BooleanField(default=False)  # Hide old experiments

    # Template variable documentation
    required_variables = JSONField(default=list)  # ["user_query", "tools_description", ...]

    class Meta:
        unique_together = ['name', 'version']
        ordering = ['name', '-version']
```

## Usage Flow

### 1. Admin creates new prompt version

In Django admin:
- Click "Add Prompt Template"
- Name: "agent"
- Version: "2" (auto-increment would be nice)
- Paste system prompt
- Paste user prompt
- Description: "Simplified instructions, removed disambiguation rules"
- Save as draft (is_active=False)

### 2. Admin tests new version

Option A - Per-conversation override:
- In chat UI, dropdown: "Prompt Version: [Active (v1)] [v2 (testing)] [v3 (testing)]"
- Select v2 for just this conversation
- Test queries

Option B - Global switch:
- In Django admin, check "is_active" on v2
- System automatically uses v2 for all new queries
- Old conversations stay with their original version (stored in Message.retrieval_metadata)

### 3. Compare results

In Django admin:
- Filter PromptLog by template_version="1" vs "2"
- Compare metrics side-by-side
- See which has better success rate

### 4. Rollback if needed

If v2 is worse:
- Uncheck is_active on v2
- Check is_active on v1
- Instant rollback, no code deployment

## Implementation Changes

### PromptRegistry changes

```python
class PromptRegistry:
    def get_active_template(self, name: str) -> PromptTemplate:
        """Get the currently active template for a given name."""
        from genealogy.models import PromptTemplate as DBTemplate

        template = DBTemplate.objects.filter(
            name=name,
            is_active=True,
            is_archived=False
        ).first()

        if not template:
            raise ValueError(f"No active template found for '{name}'")

        return template

    def get_template_by_version(self, name: str, version: str) -> PromptTemplate:
        """Get a specific template version (for testing/comparison)."""
        from genealogy.models import PromptTemplate as DBTemplate

        template = DBTemplate.objects.get(name=name, version=version)
        return template
```

### AgentExecutor changes

```python
class AgentExecutor:
    def __init__(
        self,
        model: str = "qwen2.5:14b-instruct-q5_K_M",
        max_iterations: int = 20,
        timeout: int = 300,
        prompt_template_name: str = "agent",  # Changed from prompt_template
        prompt_template_version: str = None,  # Optional: specify version
        message_instance=None
    ):
        self.prompt_template_name = prompt_template_name

        # Get template from database
        if prompt_template_version:
            # Use specific version (for testing)
            self.template = self.prompt_registry.get_template_by_version(
                prompt_template_name,
                prompt_template_version
            )
        else:
            # Use active version (default)
            self.template = self.prompt_registry.get_active_template(
                prompt_template_name
            )
```

### Chat view changes

```python
# Option 1: Always use active version (simplest)
agent = AgentExecutor(
    model=selected_model,
    prompt_template_name="agent",  # Uses active version
    message_instance=assistant_msg
)

# Option 2: Allow per-conversation override
template_version = conversation.metadata.get('prompt_version')  # User-selected
agent = AgentExecutor(
    model=selected_model,
    prompt_template_name="agent",
    prompt_template_version=template_version,  # None = use active
    message_instance=assistant_msg
)
```

## Admin UI Features

### PromptTemplate Admin

- List view: Show all templates with active indicator
- Filter by: name, is_active, is_archived, created_at
- Actions:
  - "Clone as new version" (copy template → increment version → edit)
  - "Set as active" (unsets other active templates with same name)
  - "Compare with version X" (side-by-side diff)
- Inline preview of rendered template with sample variables

### Template Editor

- Code editor with syntax highlighting
- Variable validation (warns if required variable missing)
- Preview pane showing rendered output
- Test button (renders with sample data)

## Migration Strategy

1. Create PromptTemplate model
2. Migration script to import v1 templates from files
3. Set v1 as active
4. Keep file-based templates as fallback (git history)
5. Update PromptRegistry to check DB first, fall back to files

## Benefits

✅ **No-code template switching** - Toggle active version in admin
✅ **A/B testing** - Run v1 and v2 simultaneously on different conversations
✅ **Version history** - Never lose old prompts, can always roll back
✅ **Collaboration** - Multiple people can create/test templates
✅ **Audit trail** - See who created each version, when
✅ **Documentation** - Description field explains what changed
✅ **Safety** - Test in one conversation before making active globally

## Potential Additions

### 1. Template Variables as First-Class Objects

```python
class TemplateVariable(models.Model):
    template = ForeignKey(PromptTemplate)
    name = CharField()  # "user_query"
    description = TextField()  # "The user's question"
    example_value = TextField()  # "Who is John Smith?"
    is_required = BooleanField(default=True)
```

### 2. A/B Test Tracking

```python
class PromptABTest(models.Model):
    name = CharField()  # "Simplification experiment"
    template_a = ForeignKey(PromptTemplate, related_name='tests_as_a')
    template_b = ForeignKey(PromptTemplate, related_name='tests_as_b')
    traffic_split = FloatField(default=0.5)  # 50/50
    started_at = DateTimeField(auto_now_add=True)
    ended_at = DateTimeField(null=True)

    def get_metrics(self):
        # Compare PromptLogs for template_a vs template_b
        # Return success rates, latencies, etc.
```

### 3. Conversation-Level Template Override

Add to Conversation model:
```python
class Conversation(models.Model):
    # ... existing fields ...
    prompt_template_override = ForeignKey(
        'PromptTemplate',
        null=True,
        blank=True,
        help_text="Override active template for this conversation (for testing)"
    )
```

Then in chat UI, add dropdown to select template version for this conversation.

## Questions for You

1. **Template selection scope**: Should templates be:
   - A) Global (one active version for everyone)
   - B) Per-conversation (user can test different versions)
   - C) Both (global default + per-conversation override)

2. **Version numbering**: Should we:
   - A) Simple integers (1, 2, 3)
   - B) Semantic versioning (1.0.0, 1.1.0, 2.0.0)
   - C) Timestamps (2024-12-06-001)

3. **Template editor**: Do you want:
   - A) Basic textarea in Django admin
   - B) Monaco/CodeMirror editor with syntax highlighting
   - C) Separate template management page outside admin

4. **Fallback behavior**: If no active template found:
   - A) Raise error (strict)
   - B) Use most recent version
   - C) Fall back to file-based templates

Let me know your preferences and I'll implement this!
