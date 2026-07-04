"""Admin interface for Conversation and Message models."""

from django.contrib import admin
from django.utils.html import format_html
from genealogy.models import Conversation, Message


class MessageInline(admin.TabularInline):
    """Inline display of messages in a conversation."""
    model = Message
    extra = 0
    fields = ['role', 'content_preview', 'created_at', 'view_prompts_link']
    readonly_fields = ['role', 'content_preview', 'created_at', 'view_prompts_link']
    can_delete = False

    def content_preview(self, obj):
        """Show first 100 chars of message content."""
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content'

    def view_prompts_link(self, obj):
        """Link to view prompt logs for this message."""
        count = obj.prompt_logs.count()
        if count > 0:
            return format_html(
                '<a href="/admin/genealogy/promptlog/?message__id__exact={}" target="_blank">{} prompts</a>',
                obj.id,
                count
            )
        return '-'
    view_prompts_link.short_description = 'Prompts'


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    """Admin interface for conversations."""

    list_display = ['title', 'created_at', 'updated_at', 'message_count', 'session_key']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['title', 'session_key']
    readonly_fields = ['created_at', 'updated_at', 'message_count']
    inlines = [MessageInline]

    fieldsets = [
        (None, {
            'fields': ['title', 'session_key']
        }),
        ('Timestamps', {
            'fields': ['created_at', 'updated_at', 'message_count']
        })
    ]

    def message_count(self, obj):
        """Display number of messages in conversation."""
        return obj.messages.count()
    message_count.short_description = 'Messages'


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """Admin interface for individual messages."""

    list_display = ['conversation', 'role', 'content_preview', 'created_at', 'prompt_log_count']
    list_filter = ['role', 'created_at']
    search_fields = ['content', 'conversation__title']
    readonly_fields = ['conversation', 'role', 'content', 'created_at', 'retrieval_metadata', 'prompt_log_link']

    fieldsets = [
        (None, {
            'fields': ['conversation', 'role', 'content', 'created_at']
        }),
        ('Retrieval Metadata', {
            'fields': ['retrieval_metadata'],
            'classes': ['collapse']
        }),
        ('Prompt Logs', {
            'fields': ['prompt_log_link']
        })
    ]

    def content_preview(self, obj):
        """Show first 100 chars of message content."""
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content'

    def prompt_log_count(self, obj):
        """Display number of prompt logs for this message."""
        count = obj.prompt_logs.count()
        return format_html(
            '<a href="/admin/genealogy/promptlog/?message__id__exact={}">{}</a>',
            obj.id,
            count
        ) if count > 0 else '0'
    prompt_log_count.short_description = 'Prompts'

    def prompt_log_link(self, obj):
        """Link to view all prompt logs for this message."""
        count = obj.prompt_logs.count()
        if count > 0:
            return format_html(
                '<a href="/admin/genealogy/promptlog/?message__id__exact={}" target="_blank">View {} prompt log(s)</a>',
                obj.id,
                count
            )
        return 'No prompt logs yet'
    prompt_log_link.short_description = 'Prompt Logs'

    def has_add_permission(self, request):
        """Disable manual message creation."""
        return False
