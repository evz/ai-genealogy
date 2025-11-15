import logging
from collections import defaultdict

from django.contrib import admin, messages
from django.urls import reverse
from django.utils.html import format_html

from ..extraction_strategies import get_strategy
from ..models import BookSection, Event, Person, Relationship, TextChunk
from ..ollama_utils import OllamaClient, get_default_models
from ..services import ChunkEnrichmentService

logger = logging.getLogger(__name__)


@admin.register(TextChunk)
class TextChunkAdmin(admin.ModelAdmin):
    list_display = [
        "__str__",
        "chunk_type",
        "sequence_number",
        "generation_number",
        "subject",
        "genealogical_identifier",
    ]
    list_filter = [
        "chunk_type",
        "extraction_method",
        "manually_reviewed",
        "entities_extracted",
        "generation_number",
        "document",
    ]
    search_fields = ["text_content", "generation_header", "subject", "genealogical_identifier"]
    readonly_fields = [
        "id", "created_at", "updated_at", "document", "chunk_type", "sequence_number",
        "start_page", "end_page", "entities_extracted", "generation_number",
        "generation_header", "family_groups", "subject", "genealogical_identifier",
        "related_entry_display", "text_content", "extraction_method",
        "formatted_people", "formatted_relationships", "formatted_events",
        "created_persons_display", "created_events_display", "created_relationships_display"
    ]
    actions = ["reextract_entities", "enrich_chunks"]

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "document",
                    "chunk_type",
                    "sequence_number",
                ),
            },
        ),
        (
            "Position",
            {
                "fields": ("start_page", "end_page"),
            },
        ),
        (
            "Genealogical Anchors",
            {
                "fields": (
                    "entities_extracted",
                    "generation_number",
                    "generation_header",
                    "family_groups",
                    "subject",
                    "genealogical_identifier",
                    "related_entry_display",
                ),
                "description": "Genealogical context for this chunk.",
            },
        ),
        (
            "Structured Extraction (Graph Data)",
            {
                "fields": ("formatted_people", "formatted_relationships", "formatted_events"),
                "description": "Structured person and relationship data extracted by LLM for graph building.",
            },
        ),
        (
            "Created Entities (Database Records)",
            {
                "fields": ("created_persons_display", "created_events_display", "created_relationships_display"),
                "description": "Links to actual Person, Event, and Relationship records created from this chunk.",
            },
        ),
        (
            "Content",
            {
                "fields": ("text_content",),
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

    def char_count(self, obj: TextChunk) -> str:
        """Display character count"""
        count = len(obj.text_content)
        return f"{count:,}"

    char_count.short_description = "Chars"  # type: ignore

    def token_count(self, obj: TextChunk) -> str:
        """Display approximate token count (chars / 4)"""
        count = len(obj.text_content) // 4
        return f"~{count:,}"

    token_count.short_description = "Tokens"  # type: ignore

    def formatted_people(self, obj: TextChunk) -> str:
        """Display extracted people in a readable format"""
        if not obj.extracted_people:
            return format_html('<em>No people extracted yet</em>')

        html_parts = []
        for i, person_name in enumerate(obj.extracted_people, 1):
            html_parts.append(f"{i}. <strong>{person_name}</strong>")

        return format_html("<br>".join(html_parts))

    formatted_people.short_description = "Extracted People"  # type: ignore

    def formatted_relationships(self, obj: TextChunk) -> str:
        """Display extracted relationships in a readable format"""
        if not obj.extracted_relationships:
            return format_html('<em>No relationships extracted yet</em>')

        html_parts = []
        for i, rel in enumerate(obj.extracted_relationships, 1):
            person1 = rel.get('person1', 'Unknown')
            person2 = rel.get('person2', 'Unknown')
            rel_type = rel.get('relationship_type', 'unknown')

            html_parts.append(
                f"<strong>{i}.</strong> {person1} → <em>{rel_type}</em> → {person2}"
            )

        return format_html("<br>".join(html_parts))

    formatted_relationships.short_description = "Extracted Relationships"  # type: ignore

    def formatted_events(self, obj: TextChunk) -> str:
        """Display extracted events in a readable format"""
        if not obj.extracted_events:
            return format_html('<em>No events extracted yet</em>')

        html_parts = []
        for i, event in enumerate(obj.extracted_events, 1):
            person = event.get('person', 'Unknown')
            event_type = event.get('event_type', 'UNKNOWN')
            date = event.get('date', '')
            place = event.get('place', '')

            parts = [f"<strong>{i}.</strong> {person}"]
            parts.append(f"<span style='color: #0066cc;'>{event_type}</span>")

            if date:
                parts.append(f"on {date}")
            if place:
                parts.append(f"in {place}")

            html_parts.append(" ".join(parts))

        return format_html("<br>".join(html_parts))

    formatted_events.short_description = "Extracted Events"  # type: ignore

    def related_entry_display(self, obj: TextChunk) -> str:
        """Display link to related genealogy entry (for source citations)"""
        if not obj.related_genealogy_entry:
            if obj.chunk_type == "CITATION":
                return format_html('<em style="color: #999;">No related entry linked</em>')
            return "—"

        entry = obj.related_genealogy_entry
        entry_url = reverse('admin:genealogy_textchunk_change', args=[entry.id])

        # Show preview of the related entry
        content_preview = entry.text_content[:100].replace('\n', ' ')
        if len(entry.text_content) > 100:
            content_preview += "..."

        return format_html(
            '<div style="padding: 10px; border: 1px solid #0066cc; border-radius: 4px; background: #e3f2fd;">'
            '<div><strong>Chunk #{}</strong> ({})</div>'
            '<div style="margin-top: 5px; color: #666; font-size: 0.9em;">{}</div>'
            '<div style="margin-top: 5px;"><a href="{}" target="_blank">View entry →</a></div>'
            '</div>',
            entry.sequence_number,
            entry.chunk_type,
            content_preview,
            entry_url
        )

    related_entry_display.short_description = "Related Entry (for citations)"  # type: ignore

    def created_persons_display(self, obj: TextChunk) -> str:
        """Display links to Person records created from this chunk"""
        if not obj.pk:
            return "—"

        persons = Person.objects.filter(source_chunks=obj).order_by('generation', 'given_names', 'surname')

        if not persons:
            return format_html('<em style="color: #999;">No persons created yet</em>')

        html = []
        html.append(f'<div style="margin-bottom: 10px;"><strong>Total: {persons.count()} persons</strong></div>')
        html.append('<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px;">')

        for person in persons:
            person_url = reverse('admin:genealogy_person_change', args=[person.id])
            html.append(
                f'<div style="padding: 8px; border: 1px solid #0066cc; border-radius: 4px; background: #e3f2fd;">'
                f'<a href="{person_url}" target="_blank" style="font-weight: bold; color: #0066cc;">{person.full_name}</a><br>'
                f'<small style="color: #666;">Gen {person.generation or "?"} | {person.genealogical_id}</small>'
                f'</div>'
            )

        html.append('</div>')
        return format_html(''.join(html))

    created_persons_display.short_description = "Created Persons"  # type: ignore

    def created_events_display(self, obj: TextChunk) -> str:
        """Display links to Event records created from persons in this chunk"""
        if not obj.pk:
            return "—"

        # Get all persons from this chunk
        persons = Person.objects.filter(source_chunks=obj)

        if not persons:
            return format_html('<em style="color: #999;">No persons from this chunk</em>')

        # Get events for these persons
        events = Event.objects.filter(person__in=persons).select_related('person').order_by('person__given_names', 'event_type')

        if not events:
            return format_html('<em style="color: #999;">No events created yet</em>')

        html = []
        html.append(f'<div style="margin-bottom: 10px;"><strong>Total: {events.count()} events</strong></div>')

        # Group by person
        events_by_person = defaultdict(list)
        for event in events:
            events_by_person[event.person].append(event)

        for person, person_events in events_by_person.items():
            person_url = reverse('admin:genealogy_person_change', args=[person.id])
            html.append(f'<div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9;">')
            html.append(f'<div style="font-weight: bold; margin-bottom: 5px;"><a href="{person_url}" target="_blank" style="color: #0066cc;">{person.full_name}</a></div>')
            html.append('<ul style="margin: 0; padding-left: 20px;">')

            for event in person_events:
                event_url = reverse('admin:genealogy_event_change', args=[event.id])
                parts = [f'<a href="{event_url}" target="_blank" style="color: #2e7d32;">{event.event_type}</a>']
                if event.date:
                    parts.append(f'on {event.date}')
                if event.place:
                    parts.append(f'in {event.place}')
                if event.description:
                    parts.append(f'- {event.description}')
                html.append(f'<li>{" ".join(parts)}</li>')

            html.append('</ul></div>')

        return format_html(''.join(html))

    created_events_display.short_description = "Created Events"  # type: ignore

    def created_relationships_display(self, obj: TextChunk) -> str:
        """Display links to Relationship records involving persons from this chunk"""
        if not obj.pk:
            return "—"

        # Get all persons from this chunk
        persons = Person.objects.filter(source_chunks=obj)

        if not persons:
            return format_html('<em style="color: #999;">No persons from this chunk</em>')

        person_ids = [p.id for p in persons]

        # Get relationships where either parent or child is from this chunk
        relationships = Relationship.objects.filter(
            child_id__in=person_ids
        ).select_related('child', 'parent').order_by('child__given_names')

        parent_relationships = Relationship.objects.filter(
            parent_id__in=person_ids
        ).select_related('child', 'parent').exclude(
            child_id__in=person_ids  # Don't duplicate relationships already shown
        ).order_by('child__given_names')

        total_count = relationships.count() + parent_relationships.count()

        if total_count == 0:
            return format_html('<em style="color: #999;">No relationships created yet</em>')

        html = []
        html.append(f'<div style="margin-bottom: 10px;"><strong>Total: {total_count} relationships</strong></div>')

        # Show child relationships (person from chunk is the child)
        if relationships:
            html.append('<div style="margin-bottom: 15px;">')
            html.append('<h4 style="margin: 0 0 10px 0; color: #0066cc;">As Child</h4>')
            for rel in relationships:
                rel_url = reverse('admin:genealogy_relationship_change', args=[rel.id])
                child_url = reverse('admin:genealogy_person_change', args=[rel.child.id])
                parent_url = reverse('admin:genealogy_person_change', args=[rel.parent.id])

                html.append(
                    f'<div style="padding: 8px; margin-bottom: 5px; border-left: 3px solid #4caf50; background: #f1f8f4;">'
                    f'<a href="{child_url}" target="_blank" style="font-weight: bold; color: #0066cc;">{rel.child.full_name}</a> '
                    f'← <a href="{parent_url}" target="_blank" style="color: #666;">{rel.parent.full_name}</a> '
                    f'<a href="{rel_url}" target="_blank" style="color: #999; font-size: 0.9em;">[edit]</a>'
                    f'</div>'
                )
            html.append('</div>')

        # Show parent relationships (person from chunk is the parent)
        if parent_relationships:
            html.append('<div style="margin-bottom: 15px;">')
            html.append('<h4 style="margin: 0 0 10px 0; color: #ff9800;">As Parent</h4>')
            for rel in parent_relationships:
                rel_url = reverse('admin:genealogy_relationship_change', args=[rel.id])
                child_url = reverse('admin:genealogy_person_change', args=[rel.child.id])
                parent_url = reverse('admin:genealogy_person_change', args=[rel.parent.id])

                html.append(
                    f'<div style="padding: 8px; margin-bottom: 5px; border-left: 3px solid #ff9800; background: #fff8f0;">'
                    f'<a href="{parent_url}" target="_blank" style="font-weight: bold; color: #ff9800;">{rel.parent.full_name}</a> '
                    f'→ <a href="{child_url}" target="_blank" style="color: #666;">{rel.child.full_name}</a> '
                    f'<a href="{rel_url}" target="_blank" style="color: #999; font-size: 0.9em;">[edit]</a>'
                    f'</div>'
                )
            html.append('</div>')

        return format_html(''.join(html))

    created_relationships_display.short_description = "Created Relationships"  # type: ignore

    def has_add_permission(self, request):
        """Disable adding text chunks through admin"""
        return False

    def has_change_permission(self, request, obj=None):
        """Disable editing text chunks through admin"""
        return False

    def has_delete_permission(self, request, obj=None):
        """Disable deleting text chunks through admin"""
        return False

    def reextract_entities(self, request, queryset):
        """Admin action: Re-run entity extraction for selected chunks"""
        # Get chunks in genealogy sections (DESCENDANT_GENEALOGY, KWARTIERSTATEN)
        genealogy_section_types = ['DESCENDANT_GENEALOGY', 'KWARTIERSTATEN']

        # Get all genealogy sections
        genealogy_sections = BookSection.objects.filter(
            section_type__in=genealogy_section_types
        )

        # Filter chunks to those in genealogy sections
        chunks_in_genealogy = []
        for chunk in queryset:
            # Find which section this chunk belongs to
            section = chunk.document.book_sections.filter(
                start_page__lte=chunk.start_page,
                end_page__gte=chunk.start_page
            ).first()

            if section and section.section_type in genealogy_section_types:
                chunks_in_genealogy.append(chunk.id)

        chunks = queryset.filter(id__in=chunks_in_genealogy)

        if not chunks.exists():
            self.message_user(
                request,
                "No chunks in genealogy sections selected. Only chunks from DESCENDANT_GENEALOGY or KWARTIERSTATEN sections can be extracted.",
                level=messages.WARNING
            )
            return

        # Initialize Ollama client
        ollama = OllamaClient(timeout=600)
        if not ollama.is_available():
            self.message_user(
                request,
                "Ollama server is not available. Cannot perform extraction.",
                level=messages.ERROR
            )
            return

        # Get model from first chunk's document or use default
        first_chunk = chunks.first()
        model = first_chunk.document.llm_model_used or get_default_models()["llm_model"]

        success_count = 0
        failed_count = 0

        for chunk in chunks:
            # Get the appropriate strategy based on the chunk's section
            # Find which BookSection this chunk belongs to
            section = chunk.document.book_sections.filter(
                start_page__lte=chunk.start_page,
                end_page__gte=chunk.start_page
            ).first()

            if not section:
                self.message_user(
                    request,
                    f"Chunk {chunk.sequence_number} has no BookSection defined for page {chunk.start_page}",
                    level=messages.WARNING
                )
                failed_count += 1
                continue

            try:
                strategy = get_strategy(section.section_type)
            except KeyError:
                self.message_user(
                    request,
                    f"Unknown section type: {section.section_type}",
                    level=messages.ERROR
                )
                failed_count += 1
                continue

            # Extract using the strategy
            result = strategy.extract(chunk, ollama, model)
            if result['success']:
                success_count += 1
            else:
                failed_count += 1

        # Show summary message
        if success_count > 0:
            self.message_user(
                request,
                f"Successfully re-extracted {success_count} chunk(s) using {model}.",
                level=messages.SUCCESS
            )

        if failed_count > 0:
            self.message_user(
                request,
                f"Failed to extract {failed_count} chunk(s). Check logs for details.",
                level=messages.ERROR
            )

    reextract_entities.short_description = "Re-extract entities from selected chunks"

    @admin.action(description="Generate embeddings and DM codes for selected chunks")
    def enrich_chunks(self, request, queryset):
        """Admin action: Generate embeddings and DM codes for selected chunks"""
        # Initialize Ollama client
        ollama = OllamaClient(timeout=600)
        if not ollama.is_available():
            self.message_user(
                request,
                "Ollama server is not available. Cannot generate enrichments.",
                level=messages.ERROR
            )
            return

        # Initialize enrichment service
        enrichment_service = ChunkEnrichmentService(ollama)

        # Get embedding model
        embedding_model = get_default_models()["embedding_model"]

        # Enrich all selected chunks
        result = enrichment_service.enrich_chunks_batch(
            chunks=queryset,
            embedding_model=embedding_model,
            generate_embedding=True,
            generate_dm_codes=True,
            force=True  # Force regeneration even if already exists
        )

        # Show summary message
        if result['processed'] > 0:
            msg_parts = [f"Successfully enriched {result['processed']} chunk(s):"]
            if result['embeddings_generated'] > 0:
                msg_parts.append(f"{result['embeddings_generated']} embeddings")
            if result['dm_codes_generated'] > 0:
                msg_parts.append(f"{result['dm_codes_generated']} DM code sets ({result['total_dm_codes']} total codes)")

            self.message_user(
                request,
                " ".join(msg_parts),
                level=messages.SUCCESS
            )

        if result['failed'] > 0:
            self.message_user(
                request,
                f"Failed to enrich {result['failed']} chunk(s). Check logs for details.",
                level=messages.ERROR
            )
