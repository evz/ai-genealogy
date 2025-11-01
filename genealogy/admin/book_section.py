"""Admin for BookSection model"""
from django.contrib import admin
from django.core.exceptions import ValidationError

from ..models import BookSection


@admin.register(BookSection)
class BookSectionAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "document",
        "section_type",
        "start_page",
        "end_page",
        "page_count",
        "sequence",
    ]
    list_filter = ["section_type", "document"]
    search_fields = ["title", "notes"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "document",
                    "title",
                    "section_type",
                ),
            },
        ),
        (
            "Page Range",
            {
                "fields": ("start_page", "end_page"),
            },
        ),
        (
            "Additional Information",
            {
                "fields": ("notes", "sequence"),
                "classes": ("collapse",),
            },
        ),
        (
            "Metadata",
            {
                "fields": ("id", "created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    def page_count(self, obj: BookSection) -> int:
        """Calculate number of pages in this section"""
        return obj.end_page - obj.start_page + 1

    page_count.short_description = "Pages"  # type: ignore

    def save_model(self, request, obj, form, change):
        """Validate before saving"""
        try:
            obj.clean()
            super().save_model(request, obj, form, change)
        except ValidationError as e:
            self.message_user(request, str(e), level='error')
            return
