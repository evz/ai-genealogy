"""Admin interface for PromptTemplate model."""

from django.contrib import admin
from django.utils.html import format_html
from django import forms
from genealogy.models import PromptTemplate


class PromptTemplateForm(forms.ModelForm):
    """Custom form with better widgets for template editing."""

    class Meta:
        model = PromptTemplate
        fields = '__all__'
        widgets = {
            'system_template': forms.Textarea(attrs={'rows': 20, 'cols': 100, 'style': 'font-family: monospace;'}),
            'user_template': forms.Textarea(attrs={'rows': 20, 'cols': 100, 'style': 'font-family: monospace;'}),
            'description': forms.Textarea(attrs={'rows': 3, 'cols': 100}),
        }


@admin.register(PromptTemplate)
class PromptTemplateAdmin(admin.ModelAdmin):
    """Admin interface for managing prompt templates."""

    form = PromptTemplateForm

    list_display = [
        'name',
        'version',
        'is_active_indicator',
        'created_at',
        'created_by',
        'usage_count',
        'quick_actions'
    ]

    list_filter = [
        'name',
        'is_active',
        'is_archived',
        'created_at'
    ]

    search_fields = [
        'name',
        'version',
        'description',
        'created_by'
    ]

    readonly_fields = [
        'created_at',
        'updated_at',
        'usage_stats',
        'preview_rendered'
    ]

    fieldsets = [
        ('Identity', {
            'fields': ['name', 'version', 'description', 'created_by']
        }),
        ('Template Content', {
            'fields': ['system_template', 'user_template'],
        }),
        ('Status', {
            'fields': ['is_active', 'is_archived']
        }),
        ('Documentation', {
            'fields': ['required_variables', 'example_variables'],
            'classes': ['collapse']
        }),
        ('Metadata', {
            'fields': ['created_at', 'updated_at', 'usage_stats', 'preview_rendered'],
            'classes': ['collapse']
        })
    ]

    actions = ['set_as_active', 'archive_templates', 'clone_as_new_version']

    def is_active_indicator(self, obj):
        """Display active status with color."""
        if obj.is_active:
            return format_html('<span style="color: green; font-weight: bold;">✓ ACTIVE</span>')
        elif obj.is_archived:
            return format_html('<span style="color: gray;">ARCHIVED</span>')
        else:
            return format_html('<span style="color: orange;">Inactive</span>')
    is_active_indicator.short_description = 'Status'

    def usage_count(self, obj):
        """Display number of times this template has been used."""
        count = obj.prompt_logs.count() if hasattr(obj, 'prompt_logs') else 0
        return format_html(
            '<a href="/admin/genealogy/promptlog/?prompt_template_name={}&prompt_version={}">{} uses</a>',
            obj.name,
            obj.version,
            count
        )
    usage_count.short_description = 'Usage'

    def quick_actions(self, obj):
        """Quick action buttons."""
        buttons = []

        if not obj.is_active:
            buttons.append(
                f'<a href="javascript:void(0)" onclick="return false;" '
                f'style="padding: 2px 5px; background: green; color: white; text-decoration: none; border-radius: 3px;">'
                f'Activate</a>'
            )

        buttons.append(
            f'<a href="/admin/genealogy/prompttemplate/add/?clone_from={obj.id}" '
            f'style="padding: 2px 5px; background: blue; color: white; text-decoration: none; border-radius: 3px;">'
            f'Clone</a>'
        )

        return format_html(' '.join(buttons))
    quick_actions.short_description = 'Actions'

    def usage_stats(self, obj):
        """Display usage statistics for this template."""
        from genealogy.services.prompt_registry import PromptRegistry

        registry = PromptRegistry()
        metrics = registry.get_effectiveness_metrics(obj.name, obj.version)

        if metrics.get('total_uses', 0) == 0:
            return "No usage data yet"

        return format_html('''
            <table style="border-collapse: collapse;">
                <tr><th style="text-align: left; padding-right: 20px;">Total Uses:</th><td>{total_uses}</td></tr>
                <tr><th style="text-align: left; padding-right: 20px;">Success Rate:</th><td>{success_rate:.1f}%</td></tr>
                <tr><th style="text-align: left; padding-right: 20px;">Avg Latency:</th><td>{avg_latency_ms:.0f}ms</td></tr>
                <tr><th style="text-align: left; padding-right: 20px;">Avg Iterations:</th><td>{avg_iterations:.1f}</td></tr>
                <tr><th style="text-align: left; padding-right: 20px;">Avg Tokens:</th><td>{avg_prompt_tokens:.0f} → {avg_response_tokens:.0f}</td></tr>
            </table>
        ''', **metrics)
    usage_stats.short_description = 'Usage Statistics'

    def preview_rendered(self, obj):
        """Show preview of rendered template with example variables."""
        if not obj.example_variables:
            return "No example variables set"

        try:
            rendered = obj.render(**obj.example_variables)
            return format_html('''
                <h4>System Prompt:</h4>
                <pre style="background: #f5f5f5; padding: 10px; white-space: pre-wrap;">{system}</pre>
                <h4>User Prompt:</h4>
                <pre style="background: #f5f5f5; padding: 10px; white-space: pre-wrap;">{user}</pre>
            ''', system=rendered['system'][:500], user=rendered['user'][:500])
        except Exception as e:
            return format_html('<span style="color: red;">Error rendering: {}</span>', str(e))
    preview_rendered.short_description = 'Preview (with example variables)'

    def set_as_active(self, request, queryset):
        """Set selected template as active."""
        if queryset.count() > 1:
            self.message_user(request, "Can only activate one template at a time", level='error')
            return

        template = queryset.first()
        template.is_active = True
        template.save()

        self.message_user(request, f"Set {template.name} v{template.version} as active")
    set_as_active.short_description = "Set as active"

    def archive_templates(self, request, queryset):
        """Archive selected templates."""
        count = queryset.update(is_archived=True, is_active=False)
        self.message_user(request, f"Archived {count} template(s)")
    archive_templates.short_description = "Archive selected templates"

    def clone_as_new_version(self, request, queryset):
        """Clone selected template as new version."""
        if queryset.count() > 1:
            self.message_user(request, "Can only clone one template at a time", level='error')
            return

        template = queryset.first()

        # Find next available version number
        existing_versions = PromptTemplate.objects.filter(
            name=template.name
        ).values_list('version', flat=True)

        # Convert to integers and find max
        version_nums = []
        for v in existing_versions:
            try:
                version_nums.append(int(v))
            except ValueError:
                pass

        next_version = str(max(version_nums) + 1) if version_nums else "2"

        new_template = template.clone_as_new_version(
            new_version=next_version,
            created_by=request.user.username if request.user.is_authenticated else ""
        )

        self.message_user(request, f"Cloned as {new_template.name} v{new_template.version}")
    clone_as_new_version.short_description = "Clone as new version"

    def get_changeform_initial_data(self, request):
        """Pre-fill form when cloning."""
        initial = super().get_changeform_initial_data(request)

        # Check if we're cloning
        clone_from_id = request.GET.get('clone_from')
        if clone_from_id:
            try:
                source = PromptTemplate.objects.get(id=clone_from_id)

                # Find next version
                existing_versions = PromptTemplate.objects.filter(
                    name=source.name
                ).values_list('version', flat=True)

                version_nums = []
                for v in existing_versions:
                    try:
                        version_nums.append(int(v))
                    except ValueError:
                        pass

                next_version = str(max(version_nums) + 1) if version_nums else "2"

                initial.update({
                    'name': source.name,
                    'version': next_version,
                    'system_template': source.system_template,
                    'user_template': source.user_template,
                    'description': f"Cloned from v{source.version}",
                    'required_variables': source.required_variables,
                    'example_variables': source.example_variables,
                    'is_active': False,
                })
            except PromptTemplate.DoesNotExist:
                pass

        return initial
