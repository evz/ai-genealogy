"""Admin interface for PromptLog model."""

from django.contrib import admin
from django.utils.html import format_html
from genealogy.models import PromptLog


@admin.register(PromptLog)
class PromptLogAdmin(admin.ModelAdmin):
    """Admin interface for viewing and analyzing prompt logs."""

    list_display = [
        'created_at',
        'prompt_template_name',
        'prompt_version',
        'model_name',
        'iteration',
        'parsed_successfully',
        'latency_ms',
        'token_counts',
        'view_prompt_link'
    ]

    list_filter = [
        'prompt_template_name',
        'prompt_version',
        'model_name',
        'parsed_successfully',
        'created_at'
    ]

    search_fields = [
        'message__content',
        'llm_response',
        'prompt_template_name'
    ]

    readonly_fields = [
        'message',
        'prompt_template_name',
        'prompt_version',
        'system_prompt_display',
        'user_prompt_display',
        'full_prompt_display',
        'prompt_variables_display',
        'model_name',
        'iteration',
        'tool_calls_display',
        'llm_response_display',
        'parsed_successfully',
        'parse_error',
        'latency_ms',
        'token_count_prompt',
        'token_count_response',
        'created_at'
    ]

    fieldsets = [
        ('Message Context', {
            'fields': ['message', 'created_at']
        }),
        ('Template Info', {
            'fields': ['prompt_template_name', 'prompt_version', 'prompt_variables_display']
        }),
        ('Prompts', {
            'fields': ['system_prompt_display', 'user_prompt_display', 'full_prompt_display'],
            'classes': ['collapse']
        }),
        ('Model & Execution', {
            'fields': ['model_name', 'iteration', 'tool_calls_display']
        }),
        ('Response', {
            'fields': ['llm_response_display', 'parsed_successfully', 'parse_error']
        }),
        ('Performance', {
            'fields': ['latency_ms', 'token_count_prompt', 'token_count_response']
        })
    ]

    def token_counts(self, obj):
        """Display token counts in a compact format."""
        return f"{obj.token_count_prompt or 0}→{obj.token_count_response or 0}"
    token_counts.short_description = 'Tokens (in→out)'

    def view_prompt_link(self, obj):
        """Link to detailed prompt view."""
        return format_html(
            '<a href="/admin/genealogy/promptlog/{}/change/" target="_blank">View</a>',
            obj.id
        )
    view_prompt_link.short_description = 'Actions'

    def system_prompt_display(self, obj):
        """Display system prompt in a code block."""
        return format_html('<pre style="white-space: pre-wrap;">{}</pre>', obj.system_prompt)
    system_prompt_display.short_description = 'System Prompt'

    def user_prompt_display(self, obj):
        """Display user prompt in a code block."""
        return format_html('<pre style="white-space: pre-wrap;">{}</pre>', obj.user_prompt)
    user_prompt_display.short_description = 'User Prompt'

    def full_prompt_display(self, obj):
        """Display full combined prompt in a code block."""
        return format_html('<pre style="white-space: pre-wrap;">{}</pre>', obj.full_prompt)
    full_prompt_display.short_description = 'Full Prompt (Combined)'

    def llm_response_display(self, obj):
        """Display LLM response in a code block."""
        return format_html('<pre style="white-space: pre-wrap;">{}</pre>', obj.llm_response)
    llm_response_display.short_description = 'LLM Response'

    def prompt_variables_display(self, obj):
        """Display prompt variables as formatted JSON."""
        import json
        return format_html(
            '<pre style="white-space: pre-wrap;">{}</pre>',
            json.dumps(obj.prompt_variables, indent=2)
        )
    prompt_variables_display.short_description = 'Template Variables'

    def tool_calls_display(self, obj):
        """Display tool calls as formatted JSON."""
        import json
        return format_html(
            '<pre style="white-space: pre-wrap;">{}</pre>',
            json.dumps(obj.tool_calls, indent=2)
        )
    tool_calls_display.short_description = 'Tool Calls Made'

    def has_add_permission(self, request):
        """Disable adding prompt logs manually."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Allow deleting old logs for cleanup."""
        return True
