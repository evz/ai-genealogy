from django.contrib import admin

from ..models import Place


@admin.register(Place)
class PlaceAdmin(admin.ModelAdmin):
    list_display = ["name", "locality", "region", "country"]
    list_filter = ["region", "country"]
    search_fields = ["name", "locality", "region", "country"]
    readonly_fields = ["id"]
