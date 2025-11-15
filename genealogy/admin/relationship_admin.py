"""Admin interface for Relationship model"""
from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from ..models import Relationship


@admin.register(Relationship)
class RelationshipAdmin(admin.ModelAdmin):
    list_display = [
        'parent_id_display',
        'child_id_display',
        'relationship_type',
        'created_at',
    ]
    list_filter = ['relationship_type', 'created_at']
    search_fields = [
        'parent__genealogical_id',
        'parent__given_names',
        'parent__surname',
        'child__genealogical_id',
        'child__given_names',
        'child__surname',
    ]
    readonly_fields = [
        'id',
        'parent_display',
        'child_display',
        'relationship_type',
        'source_documents_display',
        'created_at',
    ]

    fieldsets = (
        (
            'Relationship',
            {
                'fields': (
                    'parent_display',
                    'child_display',
                    'relationship_type',
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

    def parent_id_display(self, obj: Relationship) -> str:
        """Display parent genealogical ID"""
        return obj.parent.genealogical_id

    parent_id_display.short_description = "Parent ID"  # type: ignore
    parent_id_display.admin_order_field = "parent__genealogical_id"  # type: ignore

    def child_id_display(self, obj: Relationship) -> str:
        """Display child genealogical ID"""
        return obj.child.genealogical_id

    child_id_display.short_description = "Child ID"  # type: ignore
    child_id_display.admin_order_field = "child__genealogical_id"  # type: ignore

    def parent_display(self, obj: Relationship) -> str:
        """Display parent with link"""
        parent_url = reverse('admin:genealogy_person_change', args=[obj.parent.id])
        return format_html(
            '<div style="padding: 10px; border: 1px solid #0066cc; border-radius: 4px; background: #e3f2fd;">'
            '<div><strong>{}</strong></div>'
            '<div style="margin-top: 5px;">{}</div>'
            '<div style="margin-top: 5px;"><a href="{}" target="_blank">View person →</a></div>'
            '</div>',
            obj.parent.genealogical_id,
            obj.parent.full_name,
            parent_url
        )

    parent_display.short_description = "Parent"  # type: ignore

    def child_display(self, obj: Relationship) -> str:
        """Display child with link"""
        child_url = reverse('admin:genealogy_person_change', args=[obj.child.id])
        return format_html(
            '<div style="padding: 10px; border: 1px solid #ff9800; border-radius: 4px; background: #fff8f0;">'
            '<div><strong>{}</strong></div>'
            '<div style="margin-top: 5px;">{}</div>'
            '<div style="margin-top: 5px;"><a href="{}" target="_blank">View person →</a></div>'
            '</div>',
            obj.child.genealogical_id,
            obj.child.full_name,
            child_url
        )

    child_display.short_description = "Child"  # type: ignore

    def source_documents_display(self, obj: Relationship) -> str:
        """Display source documents"""
        documents = obj.source_documents.all()

        if not documents:
            return format_html('<em style="color: #999;">No source documents</em>')

        html = []
        for doc in documents:
            doc_url = reverse('admin:genealogy_document_change', args=[doc.id])
            html.append(f'<a href="{doc_url}" target="_blank">{doc.title}</a>')

        return format_html(', '.join(html))

    source_documents_display.short_description = "Source Documents"  # type: ignore

    def has_add_permission(self, request):
        """Disable manual creation (should be created by build_genealogy_graph task)"""
        return False

    def has_delete_permission(self, request, obj=None):
        """Allow deletion"""
        return True
