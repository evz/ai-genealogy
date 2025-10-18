"""
PersonRecord class for entity resolution.

Enriches a Person model instance with extracted attributes and relationships
for use in similarity calculation and clustering.
"""
from typing import Dict

from genealogy.models import Person


class PersonRecord:
    """Enriched person record with extracted attributes and relationships"""

    def __init__(self, person: Person):
        self.id = person.id
        self.person = person

        # Basic attributes
        self.given_names = person.given_names or ""
        self.surname = person.surname or ""
        self.generation = person.generation

        # Extract events (birth, death, etc.)
        self.birth_date = None
        self.birth_place = None
        self.death_date = None
        self.death_place = None
        self.birth_year = None
        self.death_year = None

        for event in person.events.all():
            if event.event_type == 'BIRT':
                self.birth_date = event.date
                self.birth_place = event.place.name if event.place else None
                if event.date:
                    self.birth_year = event.date.year
            elif event.event_type == 'DEAT':
                self.death_date = event.date
                self.death_place = event.place.name if event.place else None
                if event.date:
                    self.death_year = event.date.year

        # Relationships - store both IDs and names for overlap calculation
        self.parent_ids = set()
        self.child_ids = set()
        self.spouse_ids = set()

        # Normalized names for relationship overlap scoring
        self.parent_names = set()
        self.child_names = set()
        self.spouse_names = set()

        # Extract parent relationships
        for rel in person.parent_relationships.all():
            self.parent_ids.add(rel.parent_id)
            # Normalize parent name for comparison
            parent_name = self._normalize_name(rel.parent.given_names, rel.parent.surname)
            self.parent_names.add(parent_name)

        # Extract child relationships
        for rel in person.child_relationships.all():
            self.child_ids.add(rel.child_id)
            # Normalize child name for comparison
            child_name = self._normalize_name(rel.child.given_names, rel.child.surname)
            self.child_names.add(child_name)

        # Extract spouse relationships
        for partnership in person.partnerships.all():
            for partner in partnership.partners.exclude(id=person.id):
                self.spouse_ids.add(partner.id)
                # Normalize spouse name for comparison
                spouse_name = self._normalize_name(partner.given_names, partner.surname)
                self.spouse_names.add(spouse_name)

    def _normalize_name(self, given_names: str, surname: str) -> str:
        """Normalize a name for comparison (lowercase, no spaces)"""
        full_name = f"{given_names or ''} {surname or ''}".strip()
        return full_name.lower().replace(' ', '')

    def get_attributes(self) -> Dict[str, any]:
        """Get all non-null attributes as a dictionary"""
        attrs = {}

        if self.given_names:
            attrs['given_names'] = self.given_names
        if self.surname:
            attrs['surname'] = self.surname
        if self.generation is not None:
            attrs['generation'] = self.generation
        if self.birth_year:
            attrs['birth_year'] = self.birth_year
        if self.death_year:
            attrs['death_year'] = self.death_year
        if self.birth_place:
            attrs['birth_place'] = self.birth_place
        if self.death_place:
            attrs['death_place'] = self.death_place

        return attrs
