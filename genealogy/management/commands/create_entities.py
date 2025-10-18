"""
Management command to create Person, Partnership, and Event entities from extracted chunk data.

Uses conservative matching - only auto-merges on very high confidence (exact name + generation + parents).
Otherwise creates separate records for later duplicate detection/review.
"""
import logging
import re
from datetime import datetime
from typing import Optional

from django.core.management.base import BaseCommand
from django.db import transaction

from genealogy.models import Document, Person, Partnership, Event, ParentChildRelationship, TextChunk, Place

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Create Person/Partnership/Event entities from extracted chunk data"

    def add_arguments(self, parser):
        parser.add_argument(
            '--document-id',
            type=str,
            help='Process only chunks from specific document ID'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without saving to database'
        )
        parser.add_argument(
            '--clean',
            action='store_true',
            help='Delete all existing Person/Partnership/Event records before creating new ones'
        )

    def handle(self, *args, **options):
        self.dry_run = options['dry_run']
        self.stats = {
            'persons_created': 0,
            'partnerships_created': 0,
            'events_created': 0,
            'parent_child_created': 0,
        }

        if self.dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN MODE - No changes will be saved"))

        # Clean existing entities if requested
        if options['clean']:
            if self.dry_run:
                self.stdout.write(self.style.WARNING("Would delete existing Person/Partnership/Event records"))
            else:
                self.stdout.write("Deleting existing entities...")
                Event.objects.all().delete()
                ParentChildRelationship.objects.all().delete()
                Partnership.objects.all().delete()
                Person.objects.all().delete()
                self.stdout.write(self.style.SUCCESS("Existing entities deleted"))

        # Get chunks to process
        chunks = TextChunk.objects.filter(
            chunk_type='GENEALOGY_ENTRY',
            entities_extracted=True
        )

        if options['document_id']:
            chunks = chunks.filter(document_id=options['document_id'])

        chunks = chunks.order_by('document', 'sequence_number')

        total_chunks = chunks.count()
        self.stdout.write(f"Processing {total_chunks} chunks...")

        for i, chunk in enumerate(chunks, 1):
            if i % 50 == 0:
                self.stdout.write(f"  Progress: {i}/{total_chunks} chunks")

            try:
                with transaction.atomic():
                    self._process_chunk(chunk)
                    if self.dry_run:
                        raise Exception("Dry run - rollback")
            except Exception as e:
                if not self.dry_run:
                    logger.error(f"Error processing chunk {chunk.id}: {e}", exc_info=True)
                    self.stdout.write(self.style.ERROR(f"  Error in chunk {chunk.sequence_number}: {e}"))

        # Print summary
        self.stdout.write(self.style.SUCCESS("\nEntity Creation Complete!"))
        self.stdout.write(f"  Persons created: {self.stats['persons_created']}")
        self.stdout.write(f"  Partnerships created: {self.stats['partnerships_created']}")
        self.stdout.write(f"  Parent-child relationships created: {self.stats['parent_child_created']}")
        self.stdout.write(f"  Events created: {self.stats['events_created']}")
        self.stdout.write(f"\nNote: All persons created separately - use detect_duplicates to find matches")

    def _process_chunk(self, chunk: TextChunk):
        """Process a single chunk to create entities"""

        # Parse family group to identify parent names and generation context
        parent_names_in_header = self._parse_family_group(chunk.family_groups)

        # Track persons created in this chunk for relationship linking
        chunk_persons = {}  # name -> Person mapping

        # Step 1: Create/match Person records
        for person_name in chunk.extracted_people:
            # Determine this person's generation
            generation = self._determine_generation(person_name, chunk, parent_names_in_header)

            # Create person (always create new, never merge during initial extraction)
            person = self._create_person(person_name, generation, chunk)
            chunk_persons[person_name] = person

        # Step 1.5: Create Partnership from family group header
        # This ensures parent partnerships are created even if LLM didn't extract them
        if len(parent_names_in_header) == 2:
            parent1_name = parent_names_in_header[0]
            parent2_name = parent_names_in_header[1]

            parent1 = chunk_persons.get(parent1_name)
            parent2 = chunk_persons.get(parent2_name)

            if parent1 and parent2:
                if not self.dry_run:
                    # Check if partnership already exists between these two people
                    existing = Partnership.objects.filter(
                        partners=parent1
                    ).filter(
                        partners=parent2
                    ).first()

                    if existing:
                        # Add source document if not already there
                        existing.source_documents.add(chunk.document)
                        logger.debug(f"Partnership already exists for {parent1_name} and {parent2_name}")
                    else:
                        # Create new partnership from family group header
                        partnership = Partnership.objects.create(
                            partnership_type='MARRIAGE'
                        )
                        partnership.partners.add(parent1, parent2)
                        partnership.source_documents.add(chunk.document)
                        self.stats['partnerships_created'] += 1
                        logger.debug(f"Created partnership from family group header: {parent1_name} and {parent2_name}")
                else:
                    self.stats['partnerships_created'] += 1

        # Step 2: Create ParentChildRelationship records
        for rel in chunk.extracted_relationships:
            person1_name = rel.get('person1')
            person2_name = rel.get('person2')
            rel_type = rel.get('relationship_type')

            if rel_type in ['parent', 'child']:
                # Determine who is parent and who is child
                # relationship_type 'parent': person1 is child, person2 is parent
                # relationship_type 'child': person1 is child, person2 is parent
                if rel_type == 'parent':
                    child = chunk_persons.get(person1_name)
                    parent = chunk_persons.get(person2_name)
                else:  # child
                    child = chunk_persons.get(person1_name)  # person1 is the child
                    parent = chunk_persons.get(person2_name)  # person2 is the parent

                if child and parent:
                    if not self.dry_run:
                        ParentChildRelationship.objects.get_or_create(
                            child=child,
                            parent=parent,
                            defaults={'relationship_type': 'BIOLOGICAL'}
                        )
                    self.stats['parent_child_created'] += 1

            elif rel_type == 'spouse':
                # Create partnership
                person1 = chunk_persons.get(person1_name)
                person2 = chunk_persons.get(person2_name)

                if person1 and person2:
                    if not self.dry_run:
                        # Check if partnership already exists between these two people
                        existing = Partnership.objects.filter(
                            partners=person1
                        ).filter(
                            partners=person2
                        ).first()

                        if existing:
                            # Add source document if not already there
                            existing.source_documents.add(chunk.document)
                        else:
                            # Create new partnership
                            partnership = Partnership.objects.create(
                                partnership_type='MARRIAGE'
                            )
                            partnership.partners.add(person1, person2)
                            partnership.source_documents.add(chunk.document)
                            self.stats['partnerships_created'] += 1
                    else:
                        self.stats['partnerships_created'] += 1

        # Step 3: Create Event records
        for event_data in chunk.extracted_events:
            person_name = event_data.get('person')
            person = chunk_persons.get(person_name)

            if person:
                event_type = event_data.get('event_type', 'OTHER')

                # Map to valid event types (max 5 chars in model)
                valid_types = ['BIRT', 'DEAT', 'MARR', 'DIVR', 'BAPT', 'BURI', 'RESI', 'OCCU', 'EDUC', 'IMMI', 'EMIG', 'OTHER']
                if event_type not in valid_types:
                    # Map common variations to OTHER
                    logger.debug(f"Mapping unknown event type '{event_type}' to OTHER")
                    event_type = 'OTHER'
                date_str = event_data.get('date', '')
                place_name = event_data.get('place', '')

                # Parse date
                date_obj = self._parse_date(date_str) if date_str else None

                # Get or create place
                place_obj = None
                if place_name:
                    place_obj, _ = Place.objects.get_or_create(name=place_name)

                if not self.dry_run:
                    # Use get_or_create to avoid duplicates
                    event, created = Event.objects.get_or_create(
                        event_type=event_type,
                        person=person,
                        date=date_obj,
                        place=place_obj,
                        defaults={}
                    )
                    if created:
                        self.stats['events_created'] += 1
                else:
                    self.stats['events_created'] += 1

    def _parse_family_group(self, family_groups: list) -> list:
        """Extract parent names from family group header"""
        if not family_groups:
            return []

        # Example: "XII.14. Kinderen van Marinus Wilhelmus Borsten en Alieke Zwerver"
        # Example: "XlI.10. Children of Jamie Nicole Hall and Joshua Abercrombie (X1.16.b)"
        family_group = family_groups[0]

        # Pattern to extract everything after "Kinderen van" or "Children of"
        # Stop at opening parenthesis if present
        pattern = r'(?:Kinderen\s+van|Children\s+of)\s+(.+?)(?:\s*\(|$)'
        match = re.search(pattern, family_group, re.IGNORECASE)

        if not match:
            return []

        names_part = match.group(1).strip()

        # Split on "en" or "and" to get both parent names
        parent_names = re.split(r'\s+(?:en|and)\s+', names_part, flags=re.IGNORECASE)
        parent_names = [p.strip() for p in parent_names if p.strip()]

        return parent_names

    def _determine_generation(self, person_name: str, chunk: TextChunk, parent_names: list) -> int:
        """
        Determine person's generation based on extracted relationships.

        Uses two-pass logic to handle cases where a chunk mentions multiple generations:
        - Pass 1: Identify "primary generation" (people who are children of parents NOT also described as children)
        - Pass 2: Assign gen+1 to children of primary generation, gen-1 to people only mentioned as parents
        """

        # Build relationship maps
        children_map = {}  # person -> set of their parents
        parents_map = {}   # person -> set of their children

        for rel in chunk.extracted_relationships:
            if rel.get('relationship_type') == 'child':
                child = rel.get('person1')
                parent = rel.get('person2')

                if child:
                    if child not in children_map:
                        children_map[child] = set()
                    if parent:
                        children_map[child].add(parent)

                if parent:
                    if parent not in parents_map:
                        parents_map[parent] = set()
                    if child:
                        parents_map[parent].add(child)

        # Pass 1: Identify "primary generation"
        # These are people described as children whose parents are NOT also described as children
        primary_generation = set()
        for person in chunk.extracted_people:
            if person in children_map:
                # This person is described as a child
                parents = children_map[person]
                # Check if ANY of their parents are also described as children
                parents_are_also_children = any(p in children_map for p in parents)
                if not parents_are_also_children:
                    # This person's parents are NOT described as children in this chunk
                    # So this person is in the primary generation
                    primary_generation.add(person)

        # Pass 2: Assign generation based on role

        # Priority 1: If in primary generation → chunk generation
        if person_name in primary_generation:
            return chunk.generation_number

        # Priority 2: If child of someone in primary generation → chunk generation + 1
        if person_name in children_map:
            parents = children_map[person_name]
            if any(p in primary_generation for p in parents):
                return chunk.generation_number + 1

        # Priority 3: If ONLY a parent (never mentioned as a child) → chunk generation - 1
        if person_name in parents_map and person_name not in children_map:
            return chunk.generation_number - 1

        # Default: chunk generation
        return chunk.generation_number

    def _create_person(self, name: str, generation: int, chunk: TextChunk) -> Person:
        """
        Create a new Person record - never merges.

        All deduplication happens later via the detect_duplicates command.
        """
        # Parse name into parts
        given_names, surname = self._parse_name(name)

        if not self.dry_run:
            person = Person.objects.create(
                given_names=given_names,
                surname=surname,
                generation=generation,
            )
            person.source_documents.add(chunk.document)
            person.source_chunks.add(chunk)
        else:
            # In dry run, create temporary person object
            person = Person(
                given_names=given_names,
                surname=surname,
                generation=generation,
            )

        self.stats['persons_created'] += 1
        return person

    def _parse_name(self, full_name: str) -> tuple[str, str]:
        """Parse full name into given_names and surname"""
        # Remove brackets: "Pieter [Peter]" -> "Pieter"
        full_name = re.sub(r'\[.*?\]', '', full_name).strip()

        # Clean up parenthetical notes that are not part of the name
        # e.g. "Bessel van Zanten (son of Pieter)" -> "Bessel van Zanten"
        full_name = re.sub(r'\s*\([^)]*\)\s*$', '', full_name).strip()

        if not full_name:
            return '', ''

        parts = full_name.split()

        # Handle Dutch name prefixes (van, de, van de, van der, van den, etc.)
        if len(parts) >= 2:
            # Check for two-word prefixes: "van de", "van der", "van den"
            if len(parts) >= 3 and parts[-3].lower() == 'van' and parts[-2].lower() in ['de', 'der', 'den']:
                surname = ' '.join(parts[-3:])
                given_names = ' '.join(parts[:-3])
            # Check for single-word prefixes: "van", "de", "der", "den"
            elif parts[-2].lower() in ['van', 'de', 'der', 'den']:
                surname = ' '.join(parts[-2:])
                given_names = ' '.join(parts[:-2])
            else:
                # No prefix - last word is surname
                surname = parts[-1]
                given_names = ' '.join(parts[:-1])
        elif len(parts) == 1:
            # Single word - treat as surname
            surname = parts[0]
            given_names = ''
        else:
            return '', ''

        return given_names, surname

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to date object"""
        if not date_str:
            return None

        # Try various formats
        formats = [
            '%Y-%m-%d',  # 2024-01-15
            '%Y-%m',     # 2024-01
            '%Y',        # 2024
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        return None
