from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from ..models import PartnershipMention, PersonMention, RelationshipMention


@admin.register(PartnershipMention)
class PartnershipMentionAdmin(admin.ModelAdmin):
    list_display = ["__str__", "partnership_type", "start_date", "end_date"]
    list_filter = ["partnership_type", "start_date", "end_date"]
    search_fields = ["partners__given_names", "partners__surname"]
    readonly_fields = ["id", "created_at", "updated_at", "family_overview_display", "partners_display"]
    filter_horizontal = ["source_documents"]

    fieldsets = (
        (
            "Family Group",
            {
                "fields": ("family_overview_display",),
            },
        ),
        (
            "Partnership Details",
            {
                "fields": ("partners_display", "partnership_type"),
            },
        ),
        (
            "Start Details",
            {
                "fields": ("start_date", "start_date_estimated", "start_place"),
                "classes": ("collapse",),
            },
        ),
        (
            "End Details",
            {
                "fields": ("end_date", "end_date_estimated", "end_reason"),
                "classes": ("collapse",),
            },
        ),
        (
            "Sources",
            {
                "fields": ("source_documents",),
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

    def family_overview_display(self, obj):
        """Display shared children in a visual family group layout"""
        if not obj.pk:
            return "—"

        html = []

        # Get partners
        partners = list(obj.partners.all())

        if not partners:
            return "No partners"

        # Partners section
        html.append('<div style="margin-bottom: 30px;">')
        html.append('<h2 style="margin: 0 0 15px 0; color: #4a148c;">Partners</h2>')
        html.append('<div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">')

        for partner in partners:
            partner_url = reverse('admin:genealogy_personmention_change', args=[partner.id])
            html.append(
                f'<div style="padding: 15px; border: 3px solid #4a148c; border-radius: 8px; background: #ede7f6; min-width: 200px;">'
                f'<a href="{partner_url}" target="_blank" style="font-weight: bold; font-size: 1.3em; color: #4a148c;">{partner.full_name}</a><br>'
                f'<small style="color: #666;">Generation {partner.generation or "?"}</small>'
                f'</div>'
            )

        html.append('</div></div>')

        # Get shared children (children who have both partners as parents)
        if len(partners) == 2:
            # Get children of each partner
            children_of_partner1 = set(rel.child_mention_id for rel in RelationshipMention.objects.filter(parent_mention=partners[0]))
            children_of_partner2 = set(rel.child_mention_id for rel in RelationshipMention.objects.filter(parent_mention=partners[1]))

            # Find shared children
            shared_child_ids = children_of_partner1.intersection(children_of_partner2)

            if shared_child_ids:
                shared_children = PersonMention.objects.filter(id__in=shared_child_ids).order_by('generation', 'given_names', 'surname')

                html.append('<div style="margin-top: 20px;">')
                html.append(f'<h2 style="margin: 0 0 15px 0; color: #2e7d32;">Children ({len(shared_children)})</h2>')
                html.append('<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 10px;">')

                for child in shared_children:
                    child_url = reverse('admin:genealogy_personmention_change', args=[child.id])
                    html.append(
                        f'<div style="padding: 10px; border: 2px solid #2e7d32; border-radius: 5px; background: #e8f5e9;">'
                        f'<a href="{child_url}" target="_blank" style="font-weight: bold; font-size: 1.1em; color: #2e7d32;">{child.full_name}</a><br>'
                        f'<small style="color: #666;">Gen {child.generation or "?"}</small>'
                        f'</div>'
                    )

                html.append('</div></div>')
            else:
                html.append('<div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-left: 4px solid #ff9800;">')
                html.append('<em>No shared children found</em>')
                html.append('</div>')

        return format_html(''.join(html))

    family_overview_display.short_description = "Family Overview"

    def partners_display(self, obj):
        """Display partners as readonly links"""
        if not obj.pk:
            return "—"

        partners = obj.partners.all()
        if not partners:
            return "—"

        links = []
        for partner in partners:
            partner_url = reverse('admin:genealogy_personmention_change', args=[partner.id])
            links.append(f'<a href="{partner_url}" target="_blank">{partner.full_name}</a>')

        return format_html(' and '.join(links))

    partners_display.short_description = "Partners"

    def has_add_permission(self, request):
        """Partnerships are created by extraction commands"""
        return False
