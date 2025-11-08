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

from genealogy.models import (Document, Event, Identity, MentionToIdentity,
                              PartnershipMention, PersonMention, Place,
                              RelationshipMention, TextChunk)
from genealogy.utils import parse_name

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Create PersonMention/Identity/PartnershipMention/Event entities from extracted chunk data"

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
            help='Delete all existing PersonMention/Identity/PartnershipMention/Event records before creating new ones'
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
                self.stdout.write(self.style.WARNING("Would delete existing PersonMention/Identity/PartnershipMention/Event records"))
                self.stdout.write(self.style.WARNING("(PersonMentions with genealogical_id will be preserved)"))
            else:
                self.stdout.write("Deleting existing entities...")

                # Delete events for PersonMentions WITHOUT genealogical_id
                # (Events linked to chunking-created PersonMentions are preserved)
                Event.objects.filter(mention__genealogical_id__isnull=True).delete()

                # Delete all relationship mentions (these are recreated each time)
                RelationshipMention.objects.all().delete()

                # Delete all partnership mentions (these are recreated each time)
                PartnershipMention.objects.all().delete()

                # Delete MentionToIdentity mappings ONLY for PersonMentions without genealogical_id
                MentionToIdentity.objects.filter(mention__genealogical_id__isnull=True).delete()

                # Delete Identity records that have no remaining mappings
                # (mention_mappings is the related_name from MentionToIdentity)
                Identity.objects.filter(mention_mappings__isnull=True).delete()

                # Delete PersonMentions that were created by create_entities (no genealogical_id)
                # Preserve PersonMentions created during chunking (have genealogical_id)
                deleted_count = PersonMention.objects.filter(genealogical_id__isnull=True).delete()[0]
                preserved_count = PersonMention.objects.filter(genealogical_id__isnull=False).count()

                self.stdout.write(self.style.SUCCESS(
                    f"Existing entities deleted ({deleted_count} PersonMentions without genealogical_id)"
                ))
                self.stdout.write(self.style.SUCCESS(
                    f"Preserved {preserved_count} PersonMentions with genealogical_id (from chunking)"
                ))

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

    def _has_relationships(self, person_name: str, chunk: TextChunk) -> bool:
        """
        Check if a person name appears in any relationships in the chunk.

        Returns True if the person is mentioned in parent-child or spouse relationships.
        """
        for rel in chunk.extracted_relationships:
            if rel.get('person1') == person_name or rel.get('person2') == person_name:
                return True
        return False

    def _process_chunk(self, chunk: TextChunk):
        """Process a single chunk to create entities"""

        # Parse family group to identify parent names and generation context
        parent_names_in_header = self._parse_family_group(chunk.family_groups)

        # Track persons created in this chunk for relationship linking
        chunk_persons = {}  # name -> PersonMention mapping

        # Step 0: Check if this chunk already has a primary PersonMention (created during chunking)
        # If so, add it to chunk_persons so relationships can reference it
        if chunk.primary_person_mention and chunk.subject:
            chunk_persons[chunk.subject] = chunk.primary_person_mention
            logger.debug(f"Using existing PersonMention for subject '{chunk.subject}' (id={chunk.primary_person_mention.id})")

        # Step 1: Create PersonMention records for other people mentioned in the chunk
        # (Skip the subject if we already have a PersonMention for them)
        for person_name in chunk.extracted_people:
            # Skip if we already have a PersonMention for this person
            if person_name in chunk_persons:
                continue

            # Determine this person's generation
            generation = self._determine_generation(person_name, chunk, parent_names_in_header)

            # Check if this person has relational context
            has_rels = self._has_relationships(person_name, chunk)

            # Create person mention and singleton identity
            person = self._create_person_mention(person_name, generation, chunk, has_relationships=has_rels)
            if person:
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
                    # Check if partnership already exists between these two mentions
                    existing = PartnershipMention.objects.filter(
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
                        partnership = PartnershipMention.objects.create(
                            partnership_type='MARRIAGE'
                        )
                        partnership.partners.add(parent1, parent2)
                        partnership.source_documents.add(chunk.document)
                        self.stats['partnerships_created'] += 1
                        logger.debug(f"Created partnership from family group header: {parent1_name} and {parent2_name}")
                else:
                    self.stats['partnerships_created'] += 1

        # Step 2: Create RelationshipMention records
        for rel in chunk.extracted_relationships:
            person1_name = rel.get('person1')
            person2_name = rel.get('person2')
            rel_type = rel.get('relationship_type')

            if rel_type in ['parent', 'child']:
                # Determine who is parent and who is child
                # relationship_type 'parent': person1 IS A parent (of person2)
                # relationship_type 'child': person1 IS A child (of person2)
                if rel_type == 'parent':
                    parent_name = person1_name
                    child_name = person2_name
                else:  # child
                    child_name = person1_name
                    parent_name = person2_name

                # Get or create mentions for both people
                parent = chunk_persons.get(parent_name)
                if not parent and parent_name:
                    # Create PersonMention on-the-fly for missing parent
                    # By definition, this person has relationships (they're being created here)
                    logger.warning(f"Creating on-the-fly PersonMention for '{parent_name}' (referenced in relationship but not in extracted_people)")
                    parent = self._create_person_mention(parent_name, None, chunk, has_relationships=True)
                    if parent:
                        chunk_persons[parent_name] = parent

                child = chunk_persons.get(child_name)
                if not child and child_name:
                    # Create PersonMention on-the-fly for missing child
                    # By definition, this person has relationships (they're being created here)
                    logger.warning(f"Creating on-the-fly PersonMention for '{child_name}' (referenced in relationship but not in extracted_people)")
                    child = self._create_person_mention(child_name, None, chunk, has_relationships=True)
                    if child:
                        chunk_persons[child_name] = child

                if child and parent:
                    if not self.dry_run:
                        RelationshipMention.objects.get_or_create(
                            child_mention=child,
                            parent_mention=parent,
                            defaults={'relationship_type': 'BIOLOGICAL'}
                        )
                    self.stats['parent_child_created'] += 1

            elif rel_type == 'spouse':
                # Create partnership mention
                # Get or create mentions for both people
                person1 = chunk_persons.get(person1_name)
                if not person1 and person1_name:
                    # By definition, this person has relationships (they're being created here)
                    logger.warning(f"Creating on-the-fly PersonMention for '{person1_name}' (referenced in spouse relationship but not in extracted_people)")
                    person1 = self._create_person_mention(person1_name, None, chunk, has_relationships=True)
                    if person1:
                        chunk_persons[person1_name] = person1

                person2 = chunk_persons.get(person2_name)
                if not person2 and person2_name:
                    # By definition, this person has relationships (they're being created here)
                    logger.warning(f"Creating on-the-fly PersonMention for '{person2_name}' (referenced in spouse relationship but not in extracted_people)")
                    person2 = self._create_person_mention(person2_name, None, chunk, has_relationships=True)
                    if person2:
                        chunk_persons[person2_name] = person2

                if person1 and person2:
                    if not self.dry_run:
                        # Check if partnership already exists between these two mentions
                        existing = PartnershipMention.objects.filter(
                            partners=person1
                        ).filter(
                            partners=person2
                        ).first()

                        if existing:
                            # Add source document if not already there
                            existing.source_documents.add(chunk.document)
                        else:
                            # Create new partnership mention
                            partnership = PartnershipMention.objects.create(
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
                date_obj, date_estimated = self._parse_date(date_str) if date_str else (None, False)

                # Get or create place
                place_obj = None
                if place_name:
                    place_obj, _ = Place.objects.get_or_create(name=place_name)

                if not self.dry_run:
                    # Use get_or_create to avoid duplicates
                    event, created = Event.objects.get_or_create(
                        event_type=event_type,
                        mention=person,
                        date=date_obj,
                        place=place_obj,
                        defaults={'date_estimated': date_estimated}
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

    def _create_person_mention(self, name: str, generation: int, chunk: TextChunk,
                               has_relationships: bool = False) -> PersonMention:
        """
        Create a new PersonMention record and singleton Identity.

        Each mention gets its own Identity initially.
        All deduplication happens later via the cluster_entities command.

        Args:
            name: Full name string to parse
            generation: Generation number
            chunk: Source TextChunk
            has_relationships: Whether this person has parent/child/spouse relationships
                             (allows creation with just given_names if true)

        Returns:
            PersonMention instance or None if validation fails
        """
        # Parse name into parts
        given_names, surname = parse_name(name)

        # Validate name quality:
        # 1. MUST have both given_names and surname, OR
        # 2. If only given_names, MUST have relational context (parents/children/spouse)
        has_both_names = given_names.strip() and surname.strip()
        has_contextual_given_name = given_names.strip() and has_relationships

        if not (has_both_names or has_contextual_given_name):
            logger.warning(
                f"Skipping PersonMention creation for '{name}': insufficient name information "
                f"(given_names='{given_names}', surname='{surname}', has_relationships={has_relationships})"
            )
            return None

        if not self.dry_run:
            # Create the immutable person mention
            person = PersonMention.objects.create(
                given_names=given_names,
                surname=surname,
                generation=generation,
            )
            person.source_documents.add(chunk.document)
            person.source_chunks.add(chunk)

            # Create singleton Identity for this mention
            identity = Identity.objects.create(
                display_name=person.full_name,
                notes=f"Auto-created for {person.full_name}"
            )

            # Create the mapping
            MentionToIdentity.objects.create(
                mention=person,
                identity=identity,
                mapped_by="AUTO"
            )
        else:
            # In dry run, create temporary person object
            person = PersonMention(
                given_names=given_names,
                surname=surname,
                generation=generation,
            )

        self.stats['persons_created'] += 1
        return person

    def _parse_date(self, date_str: str) -> tuple[Optional[datetime], bool]:
        """
        Parse date string to date object.

        Returns:
            tuple: (date_object, is_estimated)
                - date_object: parsed date or None
                - is_estimated: True if date was uncertain/estimated
        """
        if not date_str:
            return None, False

        # Check for uncertainty indicators
        is_estimated = False
        date_str = date_str.strip()

        # English uncertainty words
        english_uncertain = ['circa', 'c.', 'ca.', 'about', 'around', 'approx', 'before', 'after', 'early', 'late', 'mid']

        # Dutch uncertainty words
        dutch_uncertain = [
            'omstreeks', 'ca', 'voor', 'vóór', 'na', 'rond', 'ongeveer',
            'begin', 'eind', 'midden', 'vroeg', 'laat'
        ]

        # Check for < and > symbols
        if '<' in date_str or '>' in date_str:
            is_estimated = True
            date_str = date_str.replace('<', '').replace('>', '').strip()

        # Check for uncertainty indicators
        lower = date_str.lower()
        all_uncertain = english_uncertain + dutch_uncertain
        if any(word in lower for word in all_uncertain):
            is_estimated = True

        # Handle ranges: "1746 or 1747", "1746/1747", "1746-1747"
        if ' or ' in lower or ' of ' in lower:
            is_estimated = True

        # Extract first reasonable year/date from uncertain strings
        cleaned_date = date_str

        # Handle "YYYY or YYYY" or "YYYY/YYYY" or "YYYY-YYYY" patterns (ranges)
        # Extract first year
        year_match = re.search(r'\b(\d{4})\b', date_str)
        if year_match and (' or ' in lower or ' of ' in lower or re.search(r'\d{4}[/-]\d{4}', date_str)):
            cleaned_date = year_match.group(1)
            is_estimated = True

        # Remove uncertainty words to clean the date
        for word in all_uncertain:
            cleaned_date = re.sub(r'\b' + re.escape(word) + r'\b', '', cleaned_date, flags=re.IGNORECASE).strip()

        # Clean up extra whitespace
        cleaned_date = re.sub(r'\s+', ' ', cleaned_date).strip()

        # Try various formats
        formats = [
            '%Y-%m-%d',  # 2024-01-15
            '%Y-%m',     # 2024-01
            '%Y',        # 2024
        ]

        for fmt in formats:
            try:
                return datetime.strptime(cleaned_date, fmt).date(), is_estimated
            except ValueError:
                continue

        # If we couldn't parse, return None but preserve the estimated flag if we detected uncertainty
        return None, is_estimated
