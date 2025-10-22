import logging

from django.contrib import admin, messages
from django.shortcuts import redirect
from django.urls import path, reverse
from django.utils.html import format_html

from ..models import Identity, MentionToIdentity, MergeEvent
from .merge_logic import unmerge_mentions

logger = logging.getLogger(__name__)


@admin.register(Identity)
class IdentityAdmin(admin.ModelAdmin):
    """
    Admin for Identity - the canonical resolved person.

    This is where you work with "real people" after resolving duplicates.
    """

    list_display = [
        "display_name",
        "num_mentions",
        "is_deleted",
        "created_at",
    ]
    list_filter = ["is_deleted"]
    search_fields = ["display_name", "notes"]

    # Hide deleted identities by default
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.GET.get('is_deleted__exact'):
            # Default to showing only non-deleted
            qs = qs.filter(is_deleted=False)
        return qs
    readonly_fields = [
        "id",
        "mentions_display",
        "merge_history_display",
        "events_rollup_display",
        "relationships_rollup_display",
        "partnerships_rollup_display",
        "source_chunks_rollup_display",
        "created_at",
        "updated_at",
    ]

    fieldsets = (
        (
            "Identity Information",
            {
                "fields": (
                    "display_name",
                    "notes",
                    "is_deleted",
                ),
            },
        ),
        (
            "Mentions",
            {
                "fields": ("mentions_display",),
                "description": "PersonMentions mapped to this Identity",
            },
        ),
        (
            "Events (from all mentions)",
            {
                "fields": ("events_rollup_display",),
            },
        ),
        (
            "Relationships (from all mentions)",
            {
                "fields": ("relationships_rollup_display",),
            },
        ),
        (
            "Partnerships (from all mentions)",
            {
                "fields": ("partnerships_rollup_display",),
            },
        ),
        (
            "Source Chunks (from all mentions)",
            {
                "fields": ("source_chunks_rollup_display",),
            },
        ),
        (
            "Merge History",
            {
                "fields": ("merge_history_display",),
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

    def get_urls(self):
        """Add custom URL for unmerge action"""
        urls = super().get_urls()
        custom_urls = [
            path(
                'unmerge/<int:merge_event_id>/',
                self.admin_site.admin_view(self.unmerge_view),
                name='genealogy_identity_unmerge',
            ),
        ]
        return custom_urls + urls

    def num_mentions(self, obj):
        """Count of mentions mapped to this identity"""
        return obj.mention_mappings.count()

    num_mentions.short_description = '# Mentions'

    def mentions_display(self, obj):
        """Display all mentions mapped to this identity"""
        if not obj.pk:
            return "—"

        mappings = obj.mention_mappings.select_related('mention').all()
        if not mappings:
            return "No mentions mapped"

        html = ['<table style="width: 100%; border-collapse: collapse;">']
        html.append('<tr style="background: #e0e0e0;"><th style="padding: 8px; border: 1px solid #ccc;">Mention</th><th style="padding: 8px; border: 1px solid #ccc;">Mapped By</th><th style="padding: 8px; border: 1px solid #ccc;">Mapped At</th></tr>')

        for mapping in mappings:
            mention_url = reverse('admin:genealogy_personmention_change', args=[mapping.mention.id])
            html.append(f'<tr>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;"><a href="{mention_url}" target="_blank">{mapping.mention.full_name}</a></td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{mapping.mapped_by}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{mapping.mapped_at.strftime("%Y-%m-%d %H:%M")}</td>')
            html.append(f'</tr>')

        html.append('</table>')
        return format_html(''.join(html))

    mentions_display.short_description = 'Mapped Mentions'

    def events_rollup_display(self, obj):
        """Display all events from all mentions"""
        if not obj.pk:
            return "—"

        # Get all mentions
        mention_ids = obj.mention_mappings.values_list('mention_id', flat=True)

        # Get all events from those mentions
        from ..models import Event
        events = Event.objects.filter(mention_id__in=mention_ids).select_related('mention', 'place').order_by('date')

        if not events:
            return "No events"

        html = ['<table style="width: 100%; border-collapse: collapse;">']
        html.append('<tr style="background: #e0e0e0;"><th>Type</th><th>Date</th><th>Place</th><th>From Mention</th></tr>')

        for event in events:
            date_str = str(event.date) if event.date else "—"
            if event.date_estimated:
                date_str += " (est.)"
            place_str = event.place.name if event.place else "—"

            # Link to the mention
            if event.mention:
                mention_url = reverse('admin:genealogy_personmention_change', args=[event.mention.id])
                mention_link = f'<a href="{mention_url}" target="_blank">{event.mention.full_name}</a>'
            else:
                mention_link = "—"

            html.append(f'<tr>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;"><strong>{event.get_event_type_display()}</strong></td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{date_str}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{place_str}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;"><small>{mention_link}</small></td>')
            html.append(f'</tr>')

        html.append('</table>')
        return format_html(''.join(html))

    events_rollup_display.short_description = 'Events'

    def relationships_rollup_display(self, obj):
        """Display relationships rolled up through mentions, deduplicated by identity"""
        if not obj.pk:
            return "—"

        # Get all mentions for this identity
        mention_ids = list(obj.mention_mappings.values_list('mention_id', flat=True))

        # Get relationships
        from ..models import RelationshipMention
        parent_rels = RelationshipMention.objects.filter(
            child_mention_id__in=mention_ids
        ).select_related('parent_mention')

        child_rels = RelationshipMention.objects.filter(
            parent_mention_id__in=mention_ids
        ).select_related('child_mention')

        html = []

        # Parents - deduplicate by identity
        if parent_rels.exists():
            html.append('<div style="margin-bottom: 15px;"><strong>Parents:</strong><br>')
            # Resolve parent mentions to identities
            parent_mention_ids = parent_rels.values_list('parent_mention_id', flat=True)
            parent_mappings = MentionToIdentity.objects.filter(
                mention_id__in=parent_mention_ids
            ).select_related('identity')

            # Use dict to deduplicate by identity ID
            parent_identities = {}
            for mapping in parent_mappings:
                parent_identities[mapping.identity.id] = mapping.identity

            for identity in parent_identities.values():
                identity_url = reverse('admin:genealogy_identity_change', args=[identity.id])
                html.append(f'<a href="{identity_url}" target="_blank">{identity.display_name}</a><br>')
            html.append('</div>')

        # Children - deduplicate by identity
        if child_rels.exists():
            html.append('<div style="margin-bottom: 15px;"><strong>Children:</strong><br>')
            # Resolve child mentions to identities
            child_mention_ids = child_rels.values_list('child_mention_id', flat=True)
            child_mappings = MentionToIdentity.objects.filter(
                mention_id__in=child_mention_ids
            ).select_related('identity')

            # Use dict to deduplicate by identity ID
            child_identities = {}
            for mapping in child_mappings:
                child_identities[mapping.identity.id] = mapping.identity

            for identity in child_identities.values():
                identity_url = reverse('admin:genealogy_identity_change', args=[identity.id])
                html.append(f'<a href="{identity_url}" target="_blank">{identity.display_name}</a><br>')
            html.append('</div>')

        if not html:
            return "No relationships"

        return format_html(''.join(html))

    relationships_rollup_display.short_description = 'Relationships'

    def partnerships_rollup_display(self, obj):
        """Display partnerships rolled up through mentions, deduplicated by partner identity"""
        if not obj.pk:
            return "—"

        # Get all mentions for this identity
        mention_ids = list(obj.mention_mappings.values_list('mention_id', flat=True))

        # Get partnerships involving any of these mentions
        from ..models import PartnershipMention
        partnerships = PartnershipMention.objects.filter(
            partners__id__in=mention_ids
        ).distinct().prefetch_related('partners')

        if not partnerships:
            return "No partnerships"

        # Deduplicate by partner identity
        partner_identities = {}  # identity_id -> (identity, partnership_types, dates)

        for partnership in partnerships:
            # Get partner mentions (excluding this identity's mentions)
            partner_mentions = partnership.partners.exclude(id__in=mention_ids)

            # Resolve partners to identities
            for partner_mention in partner_mentions:
                try:
                    partner_mapping = MentionToIdentity.objects.get(mention=partner_mention)
                    partner_identity = partner_mapping.identity

                    # Deduplicate by identity ID
                    if partner_identity.id not in partner_identities:
                        partner_identities[partner_identity.id] = {
                            'identity': partner_identity,
                            'types': set(),
                            'dates': []
                        }

                    partner_identities[partner_identity.id]['types'].add(partnership.get_partnership_type_display())
                    if partnership.start_date:
                        date_str = str(partnership.start_date)
                        if partnership.start_date_estimated:
                            date_str += " (est.)"
                        partner_identities[partner_identity.id]['dates'].append(date_str)

                except MentionToIdentity.DoesNotExist:
                    # Partner mention not mapped to identity yet - track by mention name
                    mention_key = f"unmerged_{partner_mention.id}"
                    if mention_key not in partner_identities:
                        partner_identities[mention_key] = {
                            'mention': partner_mention,
                            'types': set(),
                            'dates': []
                        }
                    partner_identities[mention_key]['types'].add(partnership.get_partnership_type_display())
                    if partnership.start_date:
                        date_str = str(partnership.start_date)
                        if partnership.start_date_estimated:
                            date_str += " (est.)"
                        partner_identities[mention_key]['dates'].append(date_str)

        if not partner_identities:
            return "No partnerships"

        html = ['<div style="margin-bottom: 15px;">']

        for key, data in partner_identities.items():
            if 'identity' in data:
                # Merged partner
                partner_identity = data['identity']
                partner_url = reverse('admin:genealogy_identity_change', args=[partner_identity.id])
                types_str = ", ".join(data['types'])
                dates_str = ", ".join(set(data['dates'])) if data['dates'] else ""

                html.append(
                    f'<div style="padding: 8px; margin-bottom: 5px; background: #f5f5f5; border-left: 3px solid #4caf50;">'
                    f'<strong>{types_str}</strong>: '
                    f'<a href="{partner_url}" target="_blank">{partner_identity.display_name}</a>'
                )
                if dates_str:
                    html.append(f' - {dates_str}')
                html.append('</div>')
            else:
                # Unmerged partner
                partner_mention = data['mention']
                types_str = ", ".join(data['types'])
                html.append(
                    f'<div style="padding: 8px; margin-bottom: 5px; background: #fff3cd; border-left: 3px solid #ffc107;">'
                    f'<strong>{types_str}</strong>: {partner_mention.full_name} (not merged yet)'
                    f'</div>'
                )

        html.append('</div>')
        return format_html(''.join(html))

    partnerships_rollup_display.short_description = 'Partnerships'

    def source_chunks_rollup_display(self, obj):
        """Display all source chunks from all mentions"""
        if not obj.pk:
            return "—"

        # Get all mentions for this identity
        mention_ids = list(obj.mention_mappings.values_list('mention_id', flat=True))

        # Get all source chunks from those mentions
        from ..models import TextChunk
        chunks = TextChunk.objects.filter(
            person_mentions__id__in=mention_ids
        ).select_related('document').order_by(
            'document__title', 'start_page', 'sequence_number'
        ).distinct()

        if not chunks:
            return "No source chunks"

        html = ['<div style="max-height: 600px; overflow-y: auto;">']

        current_doc = None
        for chunk in chunks:
            # Show document header when it changes
            if current_doc != chunk.document:
                current_doc = chunk.document
                doc_url = reverse('admin:genealogy_document_change', args=[chunk.document.id])
                html.append(
                    f'<div style="background: #417690; color: white; padding: 8px; margin-top: 10px; font-weight: bold;">'
                    f'<a href="{doc_url}" target="_blank" style="color: white;">{chunk.document.title}</a>'
                    f'</div>'
                )

            # Show chunk
            chunk_url = reverse('admin:genealogy_textchunk_change', args=[chunk.id])
            if chunk.start_page and chunk.end_page and chunk.start_page != chunk.end_page:
                page_info = f"Pages {chunk.start_page}-{chunk.end_page}"
            elif chunk.start_page:
                page_info = f"Page {chunk.start_page}"
            else:
                page_info = "No page"

            html.append(
                f'<div style="margin: 10px 0; padding: 12px; background: #f9f9f9; border-left: 3px solid #417690;">'
                f'<div style="margin-bottom: 5px;">'
                f'<small style="color: #666;">'
                f'<a href="{chunk_url}" target="_blank">Chunk #{chunk.sequence_number}</a> | {page_info}'
                f'</small>'
                f'</div>'
                f'<div style="font-family: monospace; white-space: pre-wrap;">{chunk.text_content}</div>'
                f'</div>'
            )

        html.append('</div>')
        return format_html(''.join(html))

    source_chunks_rollup_display.short_description = 'Source Chunks'

    def merge_history_display(self, obj):
        """Display merge event history for this identity"""
        if not obj.pk:
            return "—"

        # Get merge events involving this identity
        events = MergeEvent.objects.filter(
            payload__target_identity_id=str(obj.id)
        ).order_by('-performed_at')

        if not events:
            return "No merge history"

        html = ['<table style="width: 100%; border-collapse: collapse;">']
        html.append('<tr style="background: #e0e0e0;"><th>Event</th><th>Performed By</th><th>At</th><th>Details</th><th>Action</th></tr>')

        for event in events:
            payload = event.payload
            num_mentions = len(payload.get('old_mappings', []))
            unmerge_url = reverse('admin:genealogy_identity_unmerge', args=[event.id])

            html.append(f'<tr>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{event.get_event_type_display()}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{event.performed_by}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{event.performed_at.strftime("%Y-%m-%d %H:%M")}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;"><small>{num_mentions} mentions merged</small></td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">'
                       f'<a href="{unmerge_url}" '
                       f'style="background: #f44336; color: white; padding: 4px 8px; text-decoration: none; border-radius: 3px; font-size: 0.85em;" '
                       f'onclick="return confirm(\'Are you sure you want to reverse this merge? This will restore the {num_mentions} mentions to their previous identities.\');">'
                       f'↶ Unmerge</a>'
                       f'</td>')
            html.append(f'</tr>')

        html.append('</table>')
        return format_html(''.join(html))

    merge_history_display.short_description = 'Merge History'

    def unmerge_view(self, request, merge_event_id):
        """View to handle unmerging a previous merge event"""
        try:
            merge_event = MergeEvent.objects.get(id=merge_event_id)
        except MergeEvent.DoesNotExist:
            messages.error(request, f"Merge event {merge_event_id} not found")
            return redirect('admin:genealogy_identity_changelist')

        # Get the target identity ID from the payload
        target_identity_id = merge_event.payload.get('target_identity_id')
        if not target_identity_id:
            messages.error(request, "Cannot unmerge: no target identity in merge event")
            return redirect('admin:genealogy_identity_changelist')

        try:
            # Perform the unmerge
            restored_identities = unmerge_mentions(
                merge_event_id=merge_event_id,
                performed_by=request.user.username
            )

            messages.success(
                request,
                f"Successfully unmerged {len(restored_identities)} identities from merge event #{merge_event_id}"
            )

            # Redirect back to the original target identity (which should still exist)
            return redirect('admin:genealogy_identity_change', target_identity_id)

        except Exception as e:
            logger.exception("Unmerge failed")
            messages.error(request, f"Error during unmerge: {e}")
            return redirect('admin:genealogy_identity_changelist')
