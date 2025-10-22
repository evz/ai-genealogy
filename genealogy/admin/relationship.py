from django.contrib import admin

from ..models import RelationshipMention


@admin.register(RelationshipMention)
class RelationshipMentionAdmin(admin.ModelAdmin):
    list_display = ["child_mention", "parent_mention", "relationship_type", "partnership"]
    list_filter = ["relationship_type"]
    search_fields = [
        "child_mention__given_names",
        "child_mention__surname",
        "parent_mention__given_names",
        "parent_mention__surname",
    ]
    readonly_fields = ["id", "created_at"]
    filter_horizontal = ["source_documents"]

    def has_add_permission(self, request):
        """Relationships are created by extraction commands"""
        return False
