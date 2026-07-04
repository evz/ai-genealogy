from django.contrib import admin
from genealogy.models import Archive


@admin.register(Archive)
class ArchiveAdmin(admin.ModelAdmin):
    list_display = ["abbreviation", "name", "city", "country", "website"]
    search_fields = ["abbreviation", "name", "city"]
    ordering = ["abbreviation"]
    fieldsets = [
        (None, {"fields": ["abbreviation", "name"]}),
        ("Contact", {"fields": ["address", "city", "country", "phone", "website", "opening_hours"]}),
        ("Notes", {"fields": ["notes"]}),
    ]
