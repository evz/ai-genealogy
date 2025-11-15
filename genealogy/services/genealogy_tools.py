"""
Genealogy tools for LLM agentic workflows.

These tools allow the LLM to iteratively request information
to answer complex queries like relationship tracing and disambiguation.

Works with the simplified Person model - one genealogical_id = one Person.
"""

import logging
from typing import Dict, Optional
from uuid import UUID

from django.db.models import Q

from genealogy.models import Event, Partnership, Person, Relationship

logger = logging.getLogger(__name__)


class GenealogyTools:
    """Tools for LLM to interact with genealogy database"""

    def __init__(self):
        self.max_results = 20  # Safety limit for queries

    def _get_person(self, person_id: str) -> Optional[Person]:
        """
        Get person by genealogical_id or UUID.

        Args:
            person_id: Person UUID or genealogical_id

        Returns:
            Person object or None if not found
        """
        # Try genealogical_id first
        person = Person.objects.filter(genealogical_id=person_id).first()

        if person:
            return person

        # Try UUID - validate format first
        try:
            UUID(person_id)  # Validate UUID format
            return Person.objects.filter(id=person_id).first()
        except (ValueError, TypeError):
            # Not a valid UUID
            return None

    def search_person_by_name(self, name: str, max_results: int = 10) -> Dict:
        """
        Search for people by name with disambiguating details.

        Args:
            name: Full or partial name to search for (e.g., 'Pieter van Zanten', 'Aart')
            max_results: Maximum number of results to return (default: 10)

        Returns:
            {
                "count": int,
                "people": [
                    {
                        "id": "uuid",
                        "display_name": "Pieter van Zanten",
                        "genealogical_id": "II.3.a",
                        "birth": {"date": "1845-03-12", "place": "Amsterdam"},
                        "death": {"date": "1920-11-03", "place": "Den Haag"},
                        "parents": ["Johannes van Zanten", "Maria de Vries"]
                    }
                ],
                "truncated": bool
            }
        """
        # Limit max_results for safety
        max_results = min(max_results, self.max_results)

        # Search by given names or surname
        people = Person.objects.filter(
            Q(given_names__icontains=name) | Q(surname__icontains=name)
        ).order_by('genealogical_id')[:max_results]

        results = []
        for person in people:
            # Get birth/death events
            birth = Event.objects.filter(
                person=person,
                event_type='BIRT'
            ).first()

            death = Event.objects.filter(
                person=person,
                event_type='DEAT'
            ).first()

            # Get parents
            parent_rels = Relationship.objects.filter(
                child=person
            ).select_related('parent')

            parents = [rel.parent.full_name for rel in parent_rels]

            results.append({
                "id": str(person.id),
                "display_name": person.full_name,
                "genealogical_id": person.genealogical_id,
                "birth": {
                    "date": birth.date.isoformat() if birth and birth.date else None,
                    "place": birth.place if birth else None
                } if birth else None,
                "death": {
                    "date": death.date.isoformat() if death and death.date else None,
                    "place": death.place if death else None
                } if death else None,
                "parents": parents
            })

        return {
            "count": len(results),
            "people": results,
            "truncated": len(people) >= max_results
        }

    def get_person_details(self, person_id: str) -> Dict:
        """
        Get detailed information about a specific person.

        Args:
            person_id: Person UUID or genealogical_id (e.g., "II.3.a")

        Returns:
            {
                "id": "uuid",
                "display_name": "Pieter van Zanten",
                "genealogical_id": "II.3.a",
                "events": [
                    {"type": "Birth", "date": "1845-03-12", "place": "Amsterdam"},
                    {"type": "Marriage", "date": "1870-06-15", "place": "Rotterdam"}
                ],
                "parents": [{"id": "uuid", "name": "Johannes van Zanten"}],
                "children": [{"id": "uuid", "name": "Anna van Zanten"}],
                "partners": [{"id": "uuid", "name": "Maria de Vries", "type": "Marriage"}]
            }
        """
        # Get person by genealogical_id or UUID
        person = self._get_person(person_id)

        if not person:
            return {"error": f"Person not found: {person_id}"}

        # Get all events
        events = Event.objects.filter(person=person).order_by('date')

        events_data = []
        for event in events:
            events_data.append({
                "type": event.get_event_type_display(),
                "date": event.date.isoformat() if event.date else None,
                "place": event.place,
                "description": event.description
            })

        # Get parents
        parent_rels = Relationship.objects.filter(
            child=person
        ).select_related('parent')

        parents = [
            {"id": str(rel.parent.id), "name": rel.parent.full_name, "genealogical_id": rel.parent.genealogical_id}
            for rel in parent_rels
        ]

        # Get children
        child_rels = Relationship.objects.filter(
            parent=person
        ).select_related('child')

        children = [
            {"id": str(rel.child.id), "name": rel.child.full_name, "genealogical_id": rel.child.genealogical_id}
            for rel in child_rels
        ]

        # Get partnerships
        partnerships = Partnership.objects.filter(
            Q(partner1=person) | Q(partner2=person)
        ).select_related('partner1', 'partner2')

        partners = []
        for partnership in partnerships:
            partner = partnership.partner2 if partnership.partner1 == person else partnership.partner1
            partners.append({
                "id": str(partner.id),
                "name": partner.full_name,
                "genealogical_id": partner.genealogical_id,
                "type": partnership.get_partnership_type_display()
            })

        return {
            "id": str(person.id),
            "display_name": person.full_name,
            "genealogical_id": person.genealogical_id,
            "events": events_data,
            "parents": parents,
            "children": children,
            "partners": partners
        }

    def search_by_birth_year(
        self,
        name: str,
        birth_year_min: Optional[int] = None,
        birth_year_max: Optional[int] = None
    ) -> Dict:
        """
        Search for people by name and birth year range.

        Args:
            name: Person's name
            birth_year_min: Minimum birth year (inclusive)
            birth_year_max: Maximum birth year (inclusive)

        Returns:
            Same format as search_person_by_name
        """
        # First, get people matching the name
        people = Person.objects.filter(
            Q(given_names__icontains=name) | Q(surname__icontains=name)
        )

        # Filter by birth year if specified
        if birth_year_min or birth_year_max:
            # Get person IDs with birth events in the year range
            birth_events = Event.objects.filter(
                event_type='BIRT',
                date__isnull=False
            )

            if birth_year_min:
                birth_events = birth_events.filter(date__year__gte=birth_year_min)
            if birth_year_max:
                birth_events = birth_events.filter(date__year__lte=birth_year_max)

            person_ids_with_birth = birth_events.values_list('person_id', flat=True)

            # Filter people to only those with matching birth years
            people = people.filter(id__in=person_ids_with_birth)

        # Limit results
        people = people.order_by('genealogical_id')[:self.max_results]

        # Build results using same logic as search_person_by_name
        results = []
        for person in people:
            birth = Event.objects.filter(
                person=person,
                event_type='BIRT'
            ).first()

            death = Event.objects.filter(
                person=person,
                event_type='DEAT'
            ).first()

            parent_rels = Relationship.objects.filter(
                child=person
            ).select_related('parent')

            parents = [rel.parent.full_name for rel in parent_rels]

            results.append({
                "id": str(person.id),
                "display_name": person.full_name,
                "genealogical_id": person.genealogical_id,
                "birth": {
                    "date": birth.date.isoformat() if birth and birth.date else None,
                    "place": birth.place if birth else None
                } if birth else None,
                "death": {
                    "date": death.date.isoformat() if death and death.date else None,
                    "place": death.place if death else None
                } if death else None,
                "parents": parents
            })

        return {
            "count": len(results),
            "people": results,
            "truncated": len(people) >= self.max_results
        }

    def get_children(self, person_id: str) -> Dict:
        """
        Get all children of a person.

        Args:
            person_id: Person UUID or genealogical_id

        Returns:
            {
                "person": {"id": "uuid", "name": "Pieter van Zanten"},
                "children": [
                    {"id": "uuid", "name": "Anna van Zanten", "birth_year": "1870"}
                ],
                "count": int
            }
        """
        # Get person by genealogical_id or UUID
        person = self._get_person(person_id)

        if not person:
            return {"error": f"Person not found: {person_id}"}

        # Get child relationships
        child_rels = Relationship.objects.filter(
            parent=person
        ).select_related('child')

        # Get birth years for children
        children = []
        for rel in child_rels:
            child = rel.child
            birth = Event.objects.filter(
                person=child,
                event_type='BIRT'
            ).first()

            # Get birth year from date object
            birth_year = birth.date.year if birth and birth.date else None

            children.append({
                "id": str(child.id),
                "name": child.full_name,
                "genealogical_id": child.genealogical_id,
                "birth_year": birth_year
            })

        return {
            "person": {
                "id": str(person.id),
                "name": person.full_name
            },
            "children": children,
            "count": len(children)
        }

    def get_parents(self, person_id: str) -> Dict:
        """
        Get parents of a person.

        Args:
            person_id: Person UUID or genealogical_id

        Returns:
            {
                "person": {"id": "uuid", "name": "Pieter van Zanten"},
                "parents": [
                    {"id": "uuid", "name": "Johannes van Zanten"}
                ],
                "count": int
            }
        """
        # Get person by genealogical_id or UUID
        person = self._get_person(person_id)

        if not person:
            return {"error": f"Person not found: {person_id}"}

        # Get parent relationships
        parent_rels = Relationship.objects.filter(
            child=person
        ).select_related('parent')

        parents = [
            {
                "id": str(rel.parent.id),
                "name": rel.parent.full_name,
                "genealogical_id": rel.parent.genealogical_id
            }
            for rel in parent_rels
        ]

        return {
            "person": {
                "id": str(person.id),
                "name": person.full_name
            },
            "parents": parents,
            "count": len(parents)
        }
