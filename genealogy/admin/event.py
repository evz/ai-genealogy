from django.contrib import admin

from ..models import Event


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ["__str__", "event_type", "date", "place"]
    list_filter = ["event_type", "date", "date_estimated"]
    search_fields = ["description"]
    readonly_fields = ["id", "created_at", "updated_at"]
    filter_horizontal = ["source_documents"]
