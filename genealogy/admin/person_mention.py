import logging

from django.contrib import admin, messages
from django.urls import reverse
from django.utils.html import format_html

from ..models import Identity, PersonMention, RelationshipMention, MentionToIdentity
from .merge_logic import merge_mentions

logger = logging.getLogger(__name__)


@admin.register(PersonMention)
class PersonMentionAdmin(admin.ModelAdmin):
    """
    Read-only admin for PersonMention - immutable extractions.

    PersonMentions are never edited after creation. To resolve duplicates,
    use the Identity admin and merge interface.
    """

    list_display = [
        "full_name",
        "generation",
        "gender",
        "identity_link",
        "source_docs_count",
    ]
    list_filter = ["gender", "generation"]
    search_fields = ["given_names", "surname", "maiden_name", "genealogical_id"]
    actions = ["merge_selected_mentions"]
    readonly_fields = [
        "id",
        "given_names",
        "surname",
        "maiden_name",
        "gender",
        "genealogical_id",
        "generation",
        "identity_link",
        "events_display",
        "relationships_display",
        "source_chunk_links",
        "source_document_links",
        "created_at",
        "updated_at",
    ]

    fieldsets = (
        (
            "Extracted Attributes (Read-Only)",
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
            "Identity Mapping",
            {
                "fields": ("identity_link",),
                "description": "Which Identity this mention maps to",
            },
        ),
        (
            "Events",
            {
                "fields": ("events_display",),
            },
        ),
        (
            "Relationships",
            {
                "fields": ("relationships_display",),
            },
        ),
        (
            "Sources",
            {
                "fields": ("source_document_links", "source_chunk_links"),
            },
        ),
        (
            "Quality Flags",
            {
                "fields": ("is_extraction_error",),
                "description": "Mark if this mention was incorrectly extracted",
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

    def has_add_permission(self, request):
        """Mentions are created by extraction commands, not manually"""
        return False

    def has_delete_permission(self, request, obj=None):
        """Mentions are immutable - don't allow deletion"""
        return False

    def save_model(self, request, obj, form, change):
        """Override to handle extraction error flag changes"""
        # Save first
        super().save_model(request, obj, form, change)

        # Then handle side effects if extraction error was set
        if obj.is_extraction_error:
            from ..models import PotentialDuplicate
            from django.utils import timezone

            # Mark all PotentialDuplicate pairs involving this mention as REJECTED
            count1 = PotentialDuplicate.objects.filter(
                mention1=obj,
                review_status='PENDING'
            ).update(
                review_status='REJECTED',
                reviewed_by=request.user.username,
                reviewed_at=timezone.now()
            )

            count2 = PotentialDuplicate.objects.filter(
                mention2=obj,
                review_status='PENDING'
            ).update(
                review_status='REJECTED',
                reviewed_by=request.user.username,
                reviewed_at=timezone.now()
            )

            from django.contrib import messages
            if count1 + count2 > 0:
                messages.info(request, f"Marked {count1 + count2} potential duplicate pairs as REJECTED for this extraction error.")

    def identity_link(self, obj):
        """Show which Identity this mention maps to"""
        try:
            mapping = obj.mentiontoidentity
            identity = mapping.identity
            identity_url = reverse('admin:genealogy_identity_change', args=[identity.id])
            # Show first 8 chars of UUID + display name for easy identification
            uuid_short = str(identity.id)[:8]
            return format_html(
                '<a href="{}" target="_blank">[{}] {}</a>',
                identity_url,
                uuid_short,
                identity.display_name
            )
        except:
            return "—"

    identity_link.short_description = 'Mapped to Identity'

    def source_docs_count(self, obj):
        """Show count of source documents"""
        return obj.source_documents.count()

    source_docs_count.short_description = 'Docs'

    def events_display(self, obj):
        """Display all events for this mention"""
        if not obj.pk:
            return "—"

        events = obj.events.all().order_by('date')
        if not events:
            return "No events recorded"

        html = ['<table style="width: 100%; border-collapse: collapse;">']
        html.append('<tr style="background: #e0e0e0;"><th style="padding: 8px; border: 1px solid #ccc;">Event Type</th><th style="padding: 8px; border: 1px solid #ccc;">Date</th><th style="padding: 8px; border: 1px solid #ccc;">Place</th></tr>')

        for event in events:
            date_str = str(event.date) if event.date else "—"
            if event.date and event.date_estimated:
                date_str += " (est.)"
            place_str = event.place.name if event.place else "—"

            html.append(f'<tr>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;"><strong>{event.get_event_type_display()}</strong></td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{date_str}</td>')
            html.append(f'<td style="padding: 8px; border: 1px solid #ccc;">{place_str}</td>')
            html.append(f'</tr>')

        html.append('</table>')
        return format_html(''.join(html))

    events_display.short_description = 'Events'

    def relationships_display(self, obj):
        """Display relationships for this mention"""
        if not obj.pk:
            return "—"

        html = []

        # Parents
        parents = [rel.parent_mention for rel in obj.parent_relationships.select_related('parent_mention')]
        if parents:
            html.append('<div style="margin-bottom: 10px;"><strong>Parents:</strong><br>')
            for parent in parents:
                html.append(f'{parent.full_name}<br>')
            html.append('</div>')

        # Children
        children = [rel.child_mention for rel in obj.child_relationships.select_related('child_mention')]
        if children:
            html.append('<div style="margin-bottom: 10px;"><strong>Children:</strong><br>')
            for child in children:
                html.append(f'{child.full_name}<br>')
            html.append('</div>')

        # Partners
        partnerships = obj.partnerships.all().prefetch_related('partners')
        if partnerships:
            html.append('<div style="margin-bottom: 10px;"><strong>Partners:</strong><br>')
            for partnership in partnerships:
                partners = [p for p in partnership.partners.all() if p.id != obj.id]
                for partner in partners:
                    html.append(f'{partner.full_name}<br>')
            html.append('</div>')

        if not html:
            return "No relationships recorded"

        return format_html(''.join(html))

    relationships_display.short_description = 'Relationships'

    def source_chunk_links(self, obj):
        """Display source chunks with links"""
        if not obj.pk:
            return "—"

        chunks = obj.source_chunks.all()
        if not chunks:
            return "No source chunks"

        html = []
        for chunk in chunks:
            url = reverse('admin:genealogy_textchunk_change', args=[chunk.id])
            preview = chunk.text_content[:100] + '...' if len(chunk.text_content) > 100 else chunk.text_content
            html.append(
                f'<div style="margin-bottom: 8px;">'
                f'<a href="{url}" target="_blank">Chunk {chunk.sequence_number}</a>: '
                f'<small>{preview}</small>'
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

    @admin.action(description='Merge selected mentions into same Identity')
    def merge_selected_mentions(self, request, queryset):
        """
        Admin action to merge selected PersonMentions into a single Identity.

        This is useful for ad hoc merging when the clustering algorithm missed
        some duplicates, or when you manually identify mentions that should be merged.
        """
        mention_ids = list(queryset.values_list('id', flat=True))

        if len(mention_ids) < 2:
            self.message_user(
                request,
                "Please select at least 2 mentions to merge.",
                level=messages.ERROR
            )
            return

        # Get current Identity mappings
        mappings = MentionToIdentity.objects.filter(
            mention_id__in=mention_ids
        ).select_related('identity')

        # Check if all mentions already map to the same Identity
        identity_ids = set(m.identity_id for m in mappings)
        if len(identity_ids) == 1:
            identity = list(mappings)[0].identity
            self.message_user(
                request,
                f"All {len(mention_ids)} selected mentions already belong to the same Identity: {identity.display_name}",
                level=messages.WARNING
            )
            return

        # Get all unique identities involved
        identities = {m.identity for m in mappings}
        identity_names = ", ".join(f"[{str(i.id)[:8]}] {i.display_name}" for i in identities)

        # Choose target identity - use the one with the most mentions (most "established")
        identity_mention_counts = {}
        for mapping in mappings:
            identity_id = mapping.identity_id
            if identity_id not in identity_mention_counts:
                identity_mention_counts[identity_id] = MentionToIdentity.objects.filter(
                    identity_id=identity_id
                ).count()

        # Get the identity with the highest mention count
        target_identity_id = max(identity_mention_counts.items(), key=lambda x: x[1])[0]
        target_identity = Identity.objects.get(id=target_identity_id)

        merge_reason = {
            'merged_via': 'admin_action',
            'source': 'ad_hoc_merge',
            'note': f'Manual merge of {len(mention_ids)} mentions from {len(identity_ids)} identities'
        }

        try:
            result_identity = merge_mentions(
                mention_ids=mention_ids,
                target_identity_id=target_identity.id,
                merged_by=request.user.username,
                merge_reason=merge_reason
            )

            self.message_user(
                request,
                f"Successfully merged {len(mention_ids)} mentions from {len(identity_ids)} identities "
                f"({identity_names}) into Identity: [{str(result_identity.id)[:8]}] {result_identity.display_name}",
                level=messages.SUCCESS
            )
        except Exception as e:
            logger.exception("Ad hoc merge failed")
            self.message_user(
                request,
                f"Error during merge: {e}",
                level=messages.ERROR
            )
