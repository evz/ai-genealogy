"""Admin interface for Person model"""
from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from ..models import Person, Relationship, Partnership


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = [
        'genealogical_id',
        'full_name',
        'generation',
        'parent_count',
        'child_count',
        'partnership_count',
    ]
    list_filter = ['generation']
    search_fields = ['genealogical_id', 'given_names', 'surname']
    readonly_fields = [
        'id',
        'genealogical_id',
        'given_names',
        'surname',
        'generation',
        'created_at',
        'updated_at',
        'source_documents_display',
        'source_chunks_display',
        'parents_display',
        'children_display',
        'partnerships_display',
    ]

    fieldsets = (
        (
            'Identity',
            {
                'fields': (
                    'genealogical_id',
                    'given_names',
                    'surname',
                    'generation',
                ),
            },
        ),
        (
            'Relationships',
            {
                'fields': (
                    'parents_display',
                    'children_display',
                    'partnerships_display',
                ),
            },
        ),
        (
            'Sources',
            {
                'fields': (
                    'source_documents_display',
                    'source_chunks_display',
                ),
            },
        ),
        (
            'Metadata',
            {
                'fields': ('id', 'created_at', 'updated_at'),
                'classes': ('collapse',),
            },
        ),
    )

    def parent_count(self, obj: Person) -> int:
        """Count of parent relationships"""
        return Relationship.objects.filter(child=obj).count()

    parent_count.short_description = "Parents"  # type: ignore

    def child_count(self, obj: Person) -> int:
        """Count of child relationships"""
        return Relationship.objects.filter(parent=obj).count()

    child_count.short_description = "Children"  # type: ignore

    def partnership_count(self, obj: Person) -> int:
        """Count of partnerships"""
        return Partnership.objects.filter(
            partner1=obj
        ).count() + Partnership.objects.filter(
            partner2=obj
        ).count()

    partnership_count.short_description = "Partnerships"  # type: ignore

    def parents_display(self, obj: Person) -> str:
        """Display parent relationships"""
        relationships = Relationship.objects.filter(child=obj).select_related('parent')

        if not relationships:
            return format_html('<em style="color: #999;">No parents</em>')

        html = []
        for rel in relationships:
            parent_url = reverse('admin:genealogy_person_change', args=[rel.parent.id])
            html.append(
                f'<div style="padding: 8px; margin-bottom: 5px; border-left: 3px solid #4caf50; background: #f1f8f4;">'
                f'<a href="{parent_url}" target="_blank" style="font-weight: bold; color: #0066cc;">{rel.parent.genealogical_id}</a> '
                f'{rel.parent.full_name} '
                f'<span style="color: #666;">({rel.relationship_type})</span>'
                f'</div>'
            )

        return format_html(''.join(html))

    parents_display.short_description = "Parents"  # type: ignore

    def children_display(self, obj: Person) -> str:
        """Display child relationships"""
        relationships = Relationship.objects.filter(parent=obj).select_related('child')

        if not relationships:
            return format_html('<em style="color: #999;">No children</em>')

        html = []
        html.append(f'<div style="margin-bottom: 10px;"><strong>Total: {relationships.count()} children</strong></div>')

        for rel in relationships:
            child_url = reverse('admin:genealogy_person_change', args=[rel.child.id])
            html.append(
                f'<div style="padding: 8px; margin-bottom: 5px; border-left: 3px solid #ff9800; background: #fff8f0;">'
                f'<a href="{child_url}" target="_blank" style="font-weight: bold; color: #ff9800;">{rel.child.genealogical_id}</a> '
                f'{rel.child.full_name} '
                f'<span style="color: #666;">({rel.relationship_type})</span>'
                f'</div>'
            )

        return format_html(''.join(html))

    children_display.short_description = "Children"  # type: ignore

    def partnerships_display(self, obj: Person) -> str:
        """Display partnerships"""
        partnerships_as_1 = Partnership.objects.filter(partner1=obj).select_related('partner2')
        partnerships_as_2 = Partnership.objects.filter(partner2=obj).select_related('partner1')

        total = partnerships_as_1.count() + partnerships_as_2.count()

        if total == 0:
            return format_html('<em style="color: #999;">No partnerships</em>')

        html = []
        html.append(f'<div style="margin-bottom: 10px;"><strong>Total: {total} partnerships</strong></div>')

        for partnership in partnerships_as_1:
            partner_url = reverse('admin:genealogy_person_change', args=[partnership.partner2.id])
            html.append(
                f'<div style="padding: 8px; margin-bottom: 5px; border: 1px solid #9c27b0; background: #f3e5f5;">'
                f'<a href="{partner_url}" target="_blank" style="font-weight: bold; color: #9c27b0;">{partnership.partner2.genealogical_id}</a> '
                f'{partnership.partner2.full_name} '
                f'<span style="color: #666;">({partnership.partnership_type})</span>'
                f'</div>'
            )

        for partnership in partnerships_as_2:
            partner_url = reverse('admin:genealogy_person_change', args=[partnership.partner1.id])
            html.append(
                f'<div style="padding: 8px; margin-bottom: 5px; border: 1px solid #9c27b0; background: #f3e5f5;">'
                f'<a href="{partner_url}" target="_blank" style="font-weight: bold; color: #9c27b0;">{partnership.partner1.genealogical_id}</a> '
                f'{partnership.partner1.full_name} '
                f'<span style="color: #666;">({partnership.partnership_type})</span>'
                f'</div>'
            )

        return format_html(''.join(html))

    partnerships_display.short_description = "Partnerships"  # type: ignore

    def source_documents_display(self, obj: Person) -> str:
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

    def source_chunks_display(self, obj: Person) -> str:
        """Display source chunks"""
        chunks = obj.source_chunks.all()

        if not chunks:
            return format_html('<em style="color: #999;">No source chunks</em>')

        html = []
        html.append(f'<div style="margin-bottom: 10px;"><strong>Total: {chunks.count()} chunks</strong></div>')

        for chunk in chunks[:10]:  # Limit to first 10
            chunk_url = reverse('admin:genealogy_textchunk_change', args=[chunk.id])
            html.append(
                f'<div style="padding: 8px; margin-bottom: 5px; border: 1px solid #ddd; background: #f9f9f9;">'
                f'<a href="{chunk_url}" target="_blank">Chunk #{chunk.sequence_number}</a> '
                f'<span style="color: #666;">Page {chunk.start_page}, {chunk.chunk_type}</span>'
                f'</div>'
            )

        if chunks.count() > 10:
            html.append(f'<div style="color: #666; font-style: italic;">... and {chunks.count() - 10} more</div>')

        return format_html(''.join(html))

    source_chunks_display.short_description = "Source Chunks"  # type: ignore

    def has_add_permission(self, request):
        """Disable manual creation (should be created by build_genealogy_graph task)"""
        return False

    def has_delete_permission(self, request, obj=None):
        """Allow deletion"""
        return True
