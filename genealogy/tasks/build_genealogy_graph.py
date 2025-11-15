"""Build genealogy graph from genealogical identifiers"""
import logging
from typing import Dict

from celery import shared_task
from django.db import transaction

from genealogy.models import Document, Person, Partnership, Relationship, TextChunk
from genealogy.utils.family_parsing import parse_family_group_header
from genealogy.utils.name_parsing import parse_name

logger = logging.getLogger(__name__)


class GenealogyGraphBuilder:
    """Builds Person, Relationship, and Partnership records from TextChunks."""

    def __init__(self, document: Document):
        self.document = document
        self.person_index: Dict[str, Person] = {}
        self.stats = {
            'people_created': 0,
            'people_updated': 0,
            'relationships_created': 0,
            'partnerships_created': 0,
        }

    def build(self) -> Dict[str, int]:
        """
        Build the genealogy graph from chunks with genealogical identifiers.

        Returns:
            dict with counts of created/updated records
        """
        logger.info(f"Building genealogy graph for document: {self.document.title}")

        # Get all chunks with genealogical identifiers
        chunks = TextChunk.objects.filter(
            document=self.document,
            genealogical_identifier__isnull=False,
            subject__isnull=False,
        ).order_by('genealogical_identifier')

        if not chunks.exists():
            logger.warning(f"No chunks with genealogical identifiers found for document {self.document.id}")
            return self.stats

        with transaction.atomic():
            # Phase 1: Create/update Person records from chunks with genealogical IDs
            logger.info(f"Phase 1: Creating Person records from {chunks.count()} chunks")
            self._create_people_from_chunks(chunks)

            # Phase 2: Create Partnership records and mint spouse IDs
            # Must run before Phase 3 so spouse Person records exist for relationships
            logger.info(f"Phase 2: Creating Partnership records (and minting spouse IDs)")
            self._create_partnerships(chunks)

            # Phase 3: Create Relationship records (parent-child)
            # Runs after partnerships so spouse Person records are available
            logger.info(f"Phase 3: Creating Relationship records")
            self._create_relationships(chunks)

        logger.info(
            f"Genealogy graph built: "
            f"{self.stats['people_created']} people created, "
            f"{self.stats['people_updated']} people updated, "
            f"{self.stats['relationships_created']} relationships, "
            f"{self.stats['partnerships_created']} partnerships"
        )

        return self.stats

    def _create_people_from_chunks(self, chunks):
        """Create Person records from chunks with genealogical identifiers."""
        for chunk in chunks:
            gen_id = chunk.genealogical_identifier

            # Parse name
            given_names, surname = parse_name(chunk.subject)

            # Get or create Person
            person, created = Person.objects.get_or_create(
                genealogical_id=gen_id,
                defaults={
                    'given_names': given_names,
                    'surname': surname,
                    'generation': chunk.generation_number,
                }
            )

            if created:
                self.stats['people_created'] += 1
                logger.debug(f"Created Person: {person.full_name} ({gen_id})")
            else:
                # Update if names have changed
                updated = False
                if person.given_names != given_names:
                    person.given_names = given_names
                    updated = True
                if person.surname != surname:
                    person.surname = surname
                    updated = True
                if person.generation != chunk.generation_number:
                    person.generation = chunk.generation_number
                    updated = True

                if updated:
                    person.save()
                    self.stats['people_updated'] += 1
                    logger.debug(f"Updated Person: {person.full_name} ({gen_id})")

            # Link to source material
            person.source_documents.add(self.document)
            person.source_chunks.add(chunk)

            # Link chunk back to person
            chunk.primary_person = person
            chunk.save(update_fields=['primary_person'])

            self.person_index[gen_id] = person

    def _create_relationships(self, chunks):
        """Create parent-child Relationship records.

        Note: This method should be called AFTER _create_partnerships() so that
        spouse Person records have been created and are available in person_index.
        """
        for chunk in chunks:
            child_id = chunk.genealogical_identifier
            child = self.person_index.get(child_id)

            if not child or not chunk.family_groups:
                continue

            # Parse family group header to get parent genealogical_id and names
            parent_names, parent1_gen_id = parse_family_group_header(chunk.family_groups)

            if not parent1_gen_id:
                logger.debug(
                    f"No parent genealogical_id found in family group for child {child_id}"
                )
                continue

            # Look up parent1 by genealogical_id
            parent1 = self.person_index.get(parent1_gen_id)

            if not parent1:
                logger.warning(
                    f"Parent {parent1_gen_id} not found for child {child_id} "
                    f"(mentioned in family group header)"
                )
                continue

            # Create relationship for parent1
            relationship, created = Relationship.objects.get_or_create(
                parent=parent1,
                child=child,
                relationship_type='BIOLOGICAL',
            )

            if created:
                relationship.source_documents.add(self.document)
                self.stats['relationships_created'] += 1
                logger.debug(f"Created Relationship: {parent1.full_name} → {child.full_name}")

            # If there's a second parent, create relationship for them too
            if len(parent_names) >= 2:
                parent2_name = parent_names[1]
                parent2_given, parent2_surname = parse_name(parent2_name)

                # Try to find parent2 in person_index (may be minted spouse or actual person)
                parent2 = None
                for gen_id, person in self.person_index.items():
                    if (person.given_names.lower() == parent2_given.lower() and
                        person.surname.lower() == parent2_surname.lower() and
                        person.generation == parent1.generation):
                        parent2 = person
                        break

                if parent2:
                    # Create relationship for parent2
                    relationship2, created2 = Relationship.objects.get_or_create(
                        parent=parent2,
                        child=child,
                        relationship_type='BIOLOGICAL',
                    )

                    if created2:
                        relationship2.source_documents.add(self.document)
                        self.stats['relationships_created'] += 1
                        logger.debug(f"Created Relationship: {parent2.full_name} → {child.full_name}")

    def _create_partnerships(self, chunks):
        """Create Partnership records for parents mentioned in family groups."""
        # Group chunks by family_group to avoid creating duplicate partnerships
        family_groups_seen = set()

        # Track minted spouse IDs to handle multiple marriages
        # Key: (person_gen_id, spouse_surname_lower) -> minted_gen_id
        spouse_id_map = {}

        for chunk in chunks:
            if not chunk.family_groups:
                continue

            family_group = chunk.family_groups[0]

            # Skip if we've already processed this family group
            if family_group in family_groups_seen:
                continue
            family_groups_seen.add(family_group)

            # Parse family group header
            parent_names, parent1_gen_id = parse_family_group_header([family_group])

            # Need 2 parents and first parent must have genealogical_id
            if len(parent_names) < 2 or not parent1_gen_id:
                continue

            # Look up parent1 by genealogical_id
            parent1 = self.person_index.get(parent1_gen_id)

            if not parent1:
                logger.warning(f"Parent1 {parent1_gen_id} not found for partnership")
                continue

            # Try to find parent2 by name and generation match
            parent2_name = parent_names[1]
            parent2_given, parent2_surname = parse_name(parent2_name)

            # Try to find parent2 in person_index by matching name and generation
            parent2 = None
            for gen_id, person in self.person_index.items():
                if (person.given_names.lower() == parent2_given.lower() and
                    person.surname.lower() == parent2_surname.lower() and
                    person.generation == parent1.generation):
                    parent2 = person
                    break

            # If parent2 not found, create them with a minted genealogical ID
            if not parent2:
                # Generate a unique minted ID for this spouse
                # Use surname to make it readable, with index if there are multiple with same surname
                spouse_key = (parent1_gen_id, parent2_surname.lower())

                if spouse_key in spouse_id_map:
                    # Already created this spouse
                    minted_gen_id = spouse_id_map[spouse_key]
                    parent2 = self.person_index.get(minted_gen_id)
                else:
                    # Check if there are existing spouses for this person to determine index
                    existing_spouse_count = sum(1 for key in spouse_id_map.keys() if key[0] == parent1_gen_id)
                    spouse_index = existing_spouse_count + 1

                    # Format: II.3.a.spouse1, II.3.a.spouse2, etc.
                    minted_gen_id = f"{parent1_gen_id}.spouse{spouse_index}"

                    # Create Person for spouse with minted ID
                    parent2, created = Person.objects.get_or_create(
                        genealogical_id=minted_gen_id,
                        defaults={
                            'given_names': parent2_given,
                            'surname': parent2_surname,
                            'generation': parent1.generation,
                        }
                    )

                    if created:
                        self.stats['people_created'] += 1
                        logger.debug(f"Created spouse Person: {parent2.full_name} ({minted_gen_id})")

                    # Link to source
                    parent2.source_documents.add(self.document)

                    # Add to tracking maps
                    self.person_index[minted_gen_id] = parent2
                    spouse_id_map[spouse_key] = minted_gen_id

            # Create partnership (handle order-independence)
            # Ensure consistent ordering by using ID comparison
            partner1, partner2 = (parent1, parent2) if parent1.id < parent2.id else (parent2, parent1)

            partnership, created = Partnership.objects.get_or_create(
                partner1=partner1,
                partner2=partner2,
                partnership_type='MARRIAGE',
            )

            if created:
                partnership.source_documents.add(self.document)
                self.stats['partnerships_created'] += 1
                logger.debug(f"Created Partnership: {partner1.full_name} & {partner2.full_name}")


@shared_task
def build_genealogy_graph(document_id: str) -> Dict[str, int]:
    """
    Build Person, Relationship, and Partnership records from genealogical IDs.

    This task:
    1. Creates Person records from chunks with genealogical_identifier
    2. Creates Relationship records based on family structure
    3. Creates Partnership records for parents

    Args:
        document_id: UUID of the document to process

    Returns:
        dict with counts of created/updated records
    """
    document = Document.objects.get(id=document_id)
    builder = GenealogyGraphBuilder(document)
    return builder.build()
