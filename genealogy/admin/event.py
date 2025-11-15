from django.contrib import admin
from django.utils.html import format_html

from ..models import Event


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ["event_type", "person_link", "date", "place", "description_short", "source_chunk_link"]
    list_filter = ["event_type"]
    search_fields = ["person__full_name", "person__genealogical_id", "description", "place", "date"]
    readonly_fields = ["id", "created_at", "person_display", "source_chunk_display"]

    def description_short(self, obj):
        """Show first 50 chars of description"""
        if obj.description:
            return obj.description[:50] + "..." if len(obj.description) > 50 else obj.description
        return "-"
    description_short.short_description = "Description"

    def person_link(self, obj):
        """Link to the Person record"""
        if obj.person:
            url = f"/admin/genealogy/person/{obj.person.id}/change/"
            return format_html(
                '<a href="{}">{} ({})</a>',
                url,
                obj.person.full_name,
                obj.person.genealogical_id
            )
        return "-"
    person_link.short_description = "Person"

    def source_chunk_link(self, obj):
        """Link to the source TextChunk"""
        if obj.source_chunk:
            url = f"/admin/genealogy/textchunk/{obj.source_chunk.id}/change/"
            return format_html(
                '<a href="{}">Chunk {}</a>',
                url,
                obj.source_chunk.sequence_number
            )
        return "-"
    source_chunk_link.short_description = "Source"

    def person_display(self, obj):
        """Display person in detail view"""
        if obj.person:
            url = f"/admin/genealogy/person/{obj.person.id}/change/"
            return format_html(
                '<a href="{}">{} ({})</a>',
                url,
                obj.person.full_name,
                obj.person.genealogical_id
            )
        return "-"
    person_display.short_description = "Person"

    def source_chunk_display(self, obj):
        """Display source chunk in detail view"""
        if obj.source_chunk:
            url = f"/admin/genealogy/textchunk/{obj.source_chunk.id}/change/"
            return format_html(
                '<a href="{}">Chunk {} - {}</a><br><pre>{}</pre>',
                url,
                obj.source_chunk.sequence_number,
                obj.source_chunk.subject,
                obj.source_chunk.text_content[:500]
            )
        return "-"
    source_chunk_display.short_description = "Source Chunk"
