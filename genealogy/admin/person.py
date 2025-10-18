import logging

from django.contrib import admin, messages
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.html import format_html

from ..models import Person, ParentChildRelationship, PotentialDuplicate, EntityMerge

logger = logging.getLogger(__name__)


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = [
        "full_name",
        "generation",
        "spouses_display",
        "parents_display",
        "children_display",
    ]
    list_filter = ["gender", "generation"]
    search_fields = ["given_names", "surname", "maiden_name", "genealogical_id"]
    readonly_fields = ["id", "created_at", "updated_at", "source_chunk_links", "source_document_links", "events_display", "relationships_display", "entity_provenance_display"]
    actions = ["merge_selected_persons"]

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "given_names",
                    "surname",
                    "maiden_name",
                    "gender",
                    "generation",
                    "genealogical_id",
                ),
            },
        ),
        (
            "Entity Provenance",
            {
                "fields": ("entity_provenance_display",),
                "description": "Shows the merge history for canonical entities",
            },
        ),
        (
            "Relationships",
            {
                "fields": ("relationships_display",),
            },
        ),
        (
            "Events",
            {
                "fields": ("events_display",),
            },
        ),
        (
            "Sources",
            {
                "fields": ("source_document_links", "source_chunk_links"),
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

    def relationships_display(self, obj):
        """Display family relationships in a visual layout"""
        if not obj.pk:
            return "—"

        html = []

        # Parents section
        parents = [rel.parent for rel in ParentChildRelationship.objects.filter(child=obj).select_related('parent')]
        if parents:
            html.append('<div style="margin-bottom: 20px;">')
            html.append('<h3 style="margin: 0 0 10px 0; color: #0066cc;">Parents</h3>')
            html.append('<div style="display: flex; gap: 15px; flex-wrap: wrap;">')
            for parent in parents:
                parent_url = reverse('admin:genealogy_person_change', args=[parent.id])
                html.append(
                    f'<div style="padding: 10px; border: 2px solid #0066cc; border-radius: 5px; background: #e3f2fd;">'
                    f'<a href="{parent_url}" target="_blank" style="font-weight: bold; font-size: 1.1em;">{parent.full_name}</a><br>'
                    f'<small>Gen {parent.generation or "?"}</small>'
                    f'</div>'
                )
            html.append('</div></div>')

        # Current person (centered)
        html.append('<div style="margin: 20px 0; text-align: center; padding: 15px; background: #fff3cd; border: 3px solid #ff9800; border-radius: 5px;">')
        html.append(f'<strong style="font-size: 1.3em;">{obj.full_name}</strong><br>')
        html.append(f'<small>Generation {obj.generation or "?"}</small>')
        html.append('</div>')

        # Siblings section (people who share at least one parent)
        if parents:
            # Get all children of the parents
            parent_ids = [p.id for p in parents]
            siblings = set()
            for parent_id in parent_ids:
                sibling_rels = ParentChildRelationship.objects.filter(parent_id=parent_id).exclude(child=obj).select_related('child')
                for rel in sibling_rels:
                    siblings.add(rel.child)

            if siblings:
                html.append('<div style="margin: 20px 0;">')
                html.append('<h3 style="margin: 0 0 10px 0; color: #ff9800;">Siblings</h3>')
                html.append('<div style="display: flex; gap: 15px; flex-wrap: wrap;">')
                for sibling in sorted(siblings, key=lambda s: s.full_name):
                    sibling_url = reverse('admin:genealogy_person_change', args=[sibling.id])
                    html.append(
                        f'<div style="padding: 10px; border: 2px solid #ff9800; border-radius: 5px; background: #fff3e0;">'
                        f'<a href="{sibling_url}" target="_blank" style="font-weight: bold; font-size: 1.1em;">{sibling.full_name}</a><br>'
                        f'<small>Gen {sibling.generation or "?"}</small>'
                        f'</div>'
                    )
                html.append('</div></div>')

        # Children section
        children = [rel.child for rel in ParentChildRelationship.objects.filter(parent=obj).select_related('child')]
        if children:
            html.append('<div style="margin-top: 20px;">')
            html.append('<h3 style="margin: 0 0 10px 0; color: #4caf50;">Children</h3>')
            html.append('<div style="display: flex; gap: 15px; flex-wrap: wrap;">')
            for child in children:
                child_url = reverse('admin:genealogy_person_change', args=[child.id])
                html.append(
                    f'<div style="padding: 10px; border: 2px solid #4caf50; border-radius: 5px; background: #e8f5e9;">'
                    f'<a href="{child_url}" target="_blank" style="font-weight: bold; font-size: 1.1em;">{child.full_name}</a><br>'
                    f'<small>Gen {child.generation or "?"}</small>'
                    f'</div>'
                )
            html.append('</div></div>')

        # Partnerships
        partnerships = obj.partnerships.all().prefetch_related('partners')
        if partnerships:
            html.append('<div style="margin-top: 20px;">')
            html.append('<h3 style="margin: 0 0 10px 0; color: #9c27b0;">Partnerships</h3>')
            for partnership in partnerships:
                partners = [p for p in partnership.partners.all() if p.id != obj.id]
                if partners:
                    html.append('<div style="display: flex; gap: 15px; flex-wrap: wrap;">')
                    for partner in partners:
                        partner_url = reverse('admin:genealogy_person_change', args=[partner.id])
                        html.append(
                            f'<div style="padding: 10px; border: 2px solid #9c27b0; border-radius: 5px; background: #f3e5f5;">'
                            f'<a href="{partner_url}" target="_blank" style="font-weight: bold; font-size: 1.1em;">{partner.full_name}</a><br>'
                            f'<small>{partnership.partnership_type} • Gen {partner.generation or "?"}</small>'
                            f'</div>'
                        )
                    html.append('</div>')
            html.append('</div>')

        if not parents and not children and not partnerships:
            return "No relationships recorded"

        return format_html(''.join(html))

    relationships_display.short_description = 'Family Relationships'

    def events_display(self, obj):
        """Display all events for this person with links"""
        if not obj.pk:
            return "—"

        events = obj.events.all().order_by('date')
        if not events:
            return "No events recorded"

        html = ['<table style="width: 100%; border-collapse: collapse;">']
        html.append('<tr style="background: #e0e0e0;"><th style="padding: 8px; border: 1px solid #ccc;">Event Type</th><th style="padding: 8px; border: 1px solid #ccc;">Date</th><th style="padding: 8px; border: 1px solid #ccc;">Place</th><th style="padding: 8px; border: 1px solid #ccc;">Description</th></tr>')

        for event in events:
            event_url = reverse('admin:genealogy_event_change', args=[event.id])

            date_str = str(event.date) if event.date else "—"
            if event.date and event.date_estimated:
                date_str += " (est.)"
            place_str = event.place.name if event.place else "—"
            desc_str = event.description if event.description else "—"

            html.append(f'<tr>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;"><a href="{event_url}" target="_blank"><strong>{event.event_type}</strong></a></td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{date_str}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{place_str}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{desc_str}</td>')
            html.append(f'</tr>')

        html.append('</table>')
        return format_html(''.join(html))

    events_display.short_description = 'Events'

    def source_chunk_links(self, obj):
        """Display source chunks with links and previews"""
        if not obj.pk:
            return "—"

        chunks = obj.source_chunks.all()
        if not chunks:
            return "No source chunks"

        html = []
        for chunk in chunks:
            url = reverse('admin:genealogy_textchunk_change', args=[chunk.id])
            preview = chunk.text_content[:150] + '...' if len(chunk.text_content) > 150 else chunk.text_content
            html.append(
                f'<div style="margin-bottom: 10px; padding: 8px; border: 1px solid #e0e0e0; border-radius: 3px;">'
                f'<a href="{url}" target="_blank"><strong>Chunk {chunk.sequence_number}</strong></a><br>'
                f'<small style="color: #666;">{preview}</small>'
                f'</div>'
            )

        return format_html(''.join(html))

    source_chunk_links.short_description = 'Source Chunks'

    def source_document_links(self, obj):
        """Display source documents as links"""
        if not obj.pk:
            return "—"

        docs = obj.source_documents.all()
        if not docs:
            return "No source documents"

        html = []
        for doc in docs:
            url = reverse('admin:genealogy_document_change', args=[doc.id])
            html.append(f'<a href="{url}" target="_blank">{doc.title}</a>')

        return format_html('<br>'.join(html))

    source_document_links.short_description = 'Source Documents'

    def spouses_display(self, obj):
        """Show spouses in list view"""
        spouses = []
        for partnership in obj.partnerships.all():
            for partner in partnership.partners.exclude(id=obj.id):
                spouses.append(partner.full_name)
        if not spouses:
            return "—"
        return ", ".join(spouses)

    spouses_display.short_description = 'Spouses'

    def parents_display(self, obj):
        """Show parents in list view"""
        parents = [rel.parent for rel in ParentChildRelationship.objects.filter(child=obj).select_related('parent')]
        if not parents:
            return "—"
        return ", ".join([p.full_name for p in parents])

    parents_display.short_description = 'Parents'

    def children_display(self, obj):
        """Show children in list view"""
        children = [rel.child for rel in ParentChildRelationship.objects.filter(parent=obj).select_related('child')]
        if not children:
            return "—"
        return ", ".join([c.full_name for c in children])

    children_display.short_description = 'Children'

    def entity_provenance_display(self, obj):
        """Display merge provenance for canonical entities"""
        if not obj.pk:
            return "—"

        # Check if this is a canonical entity
        if obj.entity_type != 'CANONICAL':
            # Check if this entity has been merged into a canonical entity
            if obj.canonical_entity:
                canonical_url = reverse('admin:genealogy_person_change', args=[obj.canonical_entity.id])
                return format_html(
                    '<div style="padding: 10px; background: #fff3cd; border: 2px solid #ff9800; border-radius: 5px;">'
                    '<strong>⚠ This is an EXTRACTED entity that has been merged into:</strong><br>'
                    '<a href="{}" target="_blank" style="font-size: 1.1em;">{}</a>'
                    '</div>',
                    canonical_url,
                    obj.canonical_entity.full_name
                )
            else:
                return format_html(
                    '<div style="padding: 10px; background: #e3f2fd; border: 1px solid #2196f3; border-radius: 5px;">'
                    'This is an EXTRACTED entity (not yet merged)'
                    '</div>'
                )

        # This is a canonical entity - show what was merged into it
        source_merges = EntityMerge.objects.filter(
            canonical_entity=obj
        ).select_related('source_entity').order_by('merged_at')

        if not source_merges.exists():
            return format_html(
                '<div style="padding: 10px; background: #f3e5f5; border: 1px solid #9c27b0; border-radius: 5px;">'
                'This is a CANONICAL entity (no merge history found)'
                '</div>'
            )

        html = []
        html.append(
            '<div style="padding: 15px; background: #e8f5e9; border: 2px solid #4caf50; border-radius: 5px; margin-bottom: 15px;">'
            '<strong style="font-size: 1.2em;">✓ This is a CANONICAL entity</strong><br>'
            f'<small>Merged from {source_merges.count()} source entities</small>'
            '</div>'
        )

        html.append('<div style="margin-top: 10px;">')
        html.append('<h4 style="margin: 10px 0; color: #666;">Source Entities Merged:</h4>')
        html.append('<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">')
        html.append(
            '<tr style="background: #f5f5f5;">'
            '<th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Source Entity</th>'
            '<th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Confidence</th>'
            '<th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Merged By</th>'
            '<th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Merged At</th>'
            '<th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Pairwise Similarities</th>'
            '</tr>'
        )

        for merge in source_merges:
            source_url = reverse('admin:genealogy_person_change', args=[merge.source_entity.id])

            # Format pairwise similarities
            pairwise_html = []
            if merge.pairwise_similarities:
                for other_id, confidence in merge.pairwise_similarities.items():
                    try:
                        other_person = Person.objects.get(id=other_id)
                        other_url = reverse('admin:genealogy_person_change', args=[other_person.id])
                        pairwise_html.append(f'<a href="{other_url}" target="_blank">{other_person.full_name}</a>: {confidence:.1f}%')
                    except Person.DoesNotExist:
                        pairwise_html.append(f'Person {other_id[:8]}...: {confidence:.1f}%')
            pairwise_display = '<br>'.join(pairwise_html) if pairwise_html else '—'

            html.append(
                f'<tr>'
                f'<td style="padding: 8px; border: 1px solid #ddd;">'
                f'<a href="{source_url}" target="_blank">{merge.source_entity.full_name}</a><br>'
                f'<small style="color: #666;">ID: {str(merge.source_entity.id)[:8]}...</small>'
                f'</td>'
                f'<td style="padding: 8px; border: 1px solid #ddd;">{merge.confidence_score:.1f}%</td>'
                f'<td style="padding: 8px; border: 1px solid #ddd;">{merge.merged_by}</td>'
                f'<td style="padding: 8px; border: 1px solid #ddd;">{merge.merged_at.strftime("%Y-%m-%d %H:%M")}</td>'
                f'<td style="padding: 8px; border: 1px solid #ddd; font-size: 0.9em;">{pairwise_display}</td>'
                f'</tr>'
            )

        html.append('</table>')
        html.append('</div>')

        return format_html(''.join(html))

    entity_provenance_display.short_description = 'Entity Merge Provenance'

    def merge_selected_persons(self, request, queryset):
        """Manually merge two selected persons"""
        if queryset.count() != 2:
            self.message_user(
                request,
                "Please select exactly 2 persons to merge.",
                level=messages.ERROR
            )
            return

        person1, person2 = queryset.order_by('id')

        # Create or get a temporary PotentialDuplicate record
        duplicate, created = PotentialDuplicate.objects.get_or_create(
            person1=person1,
            person2=person2,
            defaults={
                'confidence_score': 100.0,
                'match_reasons': ['manual_merge'],
                'review_status': 'PENDING'
            }
        )

        # Import here to avoid circular import
        from .duplicate_clusters import PotentialDuplicateAdmin

        # Find the cluster containing these persons
        admin_instance = PotentialDuplicateAdmin(PotentialDuplicate, admin.site)
        clusters = admin_instance._compute_clusters(review_status='PENDING')

        # Find which cluster contains person1 or person2
        target_cluster_id = None
        for cluster in clusters:
            if person1.id in cluster['person_ids'] or person2.id in cluster['person_ids']:
                target_cluster_id = cluster['id']
                break

        if target_cluster_id is not None:
            # Redirect to cluster merge view
            merge_url = reverse('admin:genealogy_potentialduplicate_cluster_merge', args=[target_cluster_id])
        else:
            # No cluster found - should not happen, but fall back to cluster list
            messages.warning(request, "Created duplicate link. Check the cluster list to merge.")
            merge_url = reverse('admin:genealogy_potentialduplicate_cluster_list')

        return redirect(merge_url)

    merge_selected_persons.short_description = "Merge selected 2 persons (manual)"
