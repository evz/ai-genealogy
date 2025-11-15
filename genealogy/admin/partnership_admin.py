"""Admin interface for Partnership model"""
from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from ..models import Partnership


@admin.register(Partnership)
class PartnershipAdmin(admin.ModelAdmin):
    list_display = [
        'partner1_id_display',
        'partner2_id_display',
        'partnership_type',
        'created_at',
    ]
    list_filter = ['partnership_type', 'created_at']
    search_fields = [
        'partner1__genealogical_id',
        'partner1__given_names',
        'partner1__surname',
        'partner2__genealogical_id',
        'partner2__given_names',
        'partner2__surname',
    ]
    readonly_fields = [
        'id',
        'partner1_display',
        'partner2_display',
        'partnership_type',
        'source_documents_display',
        'created_at',
    ]

    fieldsets = (
        (
            'Partnership',
            {
                'fields': (
                    'partner1_display',
                    'partner2_display',
                    'partnership_type',
                ),
            },
        ),
        (
            'Sources',
            {
                'fields': ('source_documents_display',),
            },
        ),
        (
            'Metadata',
            {
                'fields': ('id', 'created_at'),
                'classes': ('collapse',),
            },
        ),
    )

    def partner1_id_display(self, obj: Partnership) -> str:
        """Display partner1 genealogical ID in list view"""
        return obj.partner1.genealogical_id if obj.partner1 else "-"

    partner1_id_display.short_description = "Partner 1 ID"  # type: ignore
    partner1_id_display.admin_order_field = "partner1__genealogical_id"  # type: ignore

    def partner2_id_display(self, obj: Partnership) -> str:
        """Display partner2 genealogical ID in list view"""
        return obj.partner2.genealogical_id if obj.partner2 else "-"

    partner2_id_display.short_description = "Partner 2 ID"  # type: ignore
    partner2_id_display.admin_order_field = "partner2__genealogical_id"  # type: ignore

    def partner1_display(self, obj: Partnership) -> str:
        """Display partner1 with link in detail view"""
        if obj.partner1:
            url = reverse('admin:genealogy_person_change', args=[obj.partner1.id])
            return format_html(
                '<a href="{}">{} ({})</a>',
                url,
                obj.partner1.full_name,
                obj.partner1.genealogical_id
            )
        return "-"

    partner1_display.short_description = "Partner 1"  # type: ignore

    def partner2_display(self, obj: Partnership) -> str:
        """Display partner2 with link in detail view"""
        if obj.partner2:
            url = reverse('admin:genealogy_person_change', args=[obj.partner2.id])
            return format_html(
                '<a href="{}">{} ({})</a>',
                url,
                obj.partner2.full_name,
                obj.partner2.genealogical_id
            )
        return "-"

    partner2_display.short_description = "Partner 2"  # type: ignore

    def source_documents_display(self, obj: Partnership) -> str:
        """Display source documents with links"""
        if not obj.source_documents.exists():
            return "-"

        links = []
        for doc in obj.source_documents.all():
            url = reverse('admin:genealogy_document_change', args=[doc.id])
            links.append(format_html('<a href="{}">{}</a>', url, doc.title))

        return format_html('<br>'.join(links))

    source_documents_display.short_description = "Source Documents"  # type: ignore
