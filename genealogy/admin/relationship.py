from django.contrib import admin

from ..models import ParentChildRelationship


@admin.register(ParentChildRelationship)
class ParentChildRelationshipAdmin(admin.ModelAdmin):
    list_display = ["child", "parent", "relationship_type", "partnership"]
    list_filter = ["relationship_type"]
    search_fields = [
        "child__given_names",
        "child__surname",
        "parent__given_names",
        "parent__surname",
    ]
    readonly_fields = ["id", "created_at"]
    filter_horizontal = ["source_documents"]
