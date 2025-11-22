"""
Genealogy tools for LLM agentic workflows.

These tools allow the LLM to iteratively request information
to answer complex queries like relationship tracing and disambiguation.

Works with the simplified Person model - one genealogical_id = one Person.
"""

import logging
import re
from typing import Dict, Optional
from uuid import UUID

from django.db.models import Q

from genealogy.models import Event, Partnership, Person, Relationship, TextChunk
from genealogy.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class GenealogyTools:
    """Tools for LLM to interact with genealogy database"""

    def __init__(self):
        self.max_results = 20  # Safety limit for queries

    def _generate_name_variations(self, name: str) -> list:
        """
        Generate spelling variations for Dutch surnames.

        Handles common variations like:
        - "van Zanten" → ["van zanten", "vanzanten", "Van Zanten", "VanZanten"]
        - "de Vries" → ["de vries", "devries", "De Vries", "DeVries"]
        - "van der Berg" → ["van der berg", "vanderberg", "Van der Berg", "VanderBerg"]

        Args:
            name: Name to generate variations for

        Returns:
            List of name variations to search for
        """
        variations = [name.lower()]  # Always include lowercase original

        # Common Dutch prefixes
        dutch_prefixes = ['van', 'de', 'der', 'den', 'het', 'ten', 'ter', 'te']

        # Check if name contains Dutch prefixes with spaces
        lower_name = name.lower()
        for prefix in dutch_prefixes:
            # Look for "prefix " (with space after)
            if f'{prefix} ' in lower_name:
                # Remove all spaces to create concatenated version
                # "van Zanten" → "vanzanten"
                variations.append(lower_name.replace(' ', ''))

                # Also add version with only prefix spaces removed
                # "Pieter van Zanten" → "Pieter vanzanten"
                for p in dutch_prefixes:
                    if f'{p} ' in lower_name:
                        variations.append(lower_name.replace(f'{p} ', p))
                break

        # If name has NO spaces but could be a compound (like "vanzanten")
        if ' ' not in name:
            for prefix in dutch_prefixes:
                if lower_name.startswith(prefix) and len(lower_name) > len(prefix):
                    # "vanzanten" → "van zanten"
                    # Insert space after the prefix
                    spaced_version = lower_name[:len(prefix)] + ' ' + lower_name[len(prefix):]
                    variations.append(spaced_version)

        return list(set(variations))  # Remove duplicates

    def _build_name_query(self, name: str):
        """
        Build a Django Q object for flexible name matching.

        Handles both single names and full names by searching across
        given_names and surname fields intelligently. Also handles
        spelling variations like "van Zanten" vs "vanzanten".

        Args:
            name: Name to search for (can be given name, surname, or full name)

        Returns:
            Django Q object for filtering Person queryset
        """
        # Generate spelling variations
        name_variations = self._generate_name_variations(name)

        # Build query for all variations
        query = Q()

        for variant in name_variations:
            if ' ' in variant:
                # Full name with space - try multiple matching strategies
                parts = variant.split()
                query |= (
                    Q(given_names__icontains=variant) |  # Entire string in given_names
                    Q(surname__icontains=variant) |  # Entire string in surname
                    # First part in given_names, rest in surname (e.g., "Bessel van Zanten")
                    (Q(given_names__icontains=parts[0], surname__icontains=' '.join(parts[1:])) if len(parts) > 1 else Q())
                )
            else:
                # Single word - search in either field
                query |= Q(given_names__icontains=variant) | Q(surname__icontains=variant)

        return query

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

        # Search using flexible name matching
        people = Person.objects.filter(
            self._build_name_query(name)
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
                "partners": [{"id": "uuid", "name": "Maria de Vries", "type": "Marriage"}],
                "source_texts": [
                    {
                        "chunk_id": 123,
                        "sequence_number": 137,
                        "chunk_type": "individual_entry",
                        "text": "Full narrative text from the source document...",
                        "page_range": "85-87",
                        "subject": "Pieter van Zanten"
                    }
                ]
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

        # Get source text from chunks where this person is the subject
        from genealogy.models import TextChunk
        chunks = TextChunk.objects.filter(
            genealogical_identifier=person.genealogical_id
        ).order_by('sequence_number')

        source_texts = []
        for chunk in chunks:
            source_texts.append({
                "chunk_id": str(chunk.id),
                "sequence_number": chunk.sequence_number,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text_content,
                "page_range": f"{chunk.start_page}-{chunk.end_page}",
                "subject": chunk.subject
            })

        return {
            "id": str(person.id),
            "display_name": person.full_name,
            "genealogical_id": person.genealogical_id,
            "events": events_data,
            "parents": parents,
            "children": children,
            "partners": partners,
            "source_texts": source_texts
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
            name: Person's name (can be given name, surname, or full name)
            birth_year_min: Minimum birth year (inclusive)
            birth_year_max: Maximum birth year (inclusive)

        Returns:
            Same format as search_person_by_name
        """
        # Search using flexible name matching
        people = Person.objects.filter(self._build_name_query(name))

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

    def search_by_occupation(self, occupation: str, max_results: int = 10) -> Dict:
        """
        Search for people by occupation.

        Args:
            occupation: Occupation term(s) to search for. Can be a single term or
                       multiple terms separated by commas for multilingual search
                       (e.g., "teacher, onderwijzer, meester")
            max_results: Maximum number of results to return (default: 10)

        Returns:
            {
                "count": int,
                "people": [
                    {
                        "id": "uuid",
                        "display_name": "Bessel van Zanten",
                        "genealogical_id": "VI.1.n",
                        "occupation": "spoorwegarbeider",
                        "birth": {"date": "1841-08-17", "place": "Naarden"}
                    }
                ],
                "truncated": bool
            }
        """
        # Limit max_results for safety
        max_results = min(max_results, self.max_results)

        # Parse multiple occupation terms (comma-separated)
        occupation_terms = [term.strip() for term in occupation.split(',')]

        # Build OR query for all occupation terms
        query = Q()
        for term in occupation_terms:
            if term:  # Skip empty strings
                query |= Q(description__icontains=term)

        # Find occupation events matching any of the search terms
        occupation_events = Event.objects.filter(
            event_type='OCCU'
        ).filter(query).select_related('person').order_by('person__genealogical_id')[:max_results * 2]  # Get more to handle dedup

        results = []
        seen_person_ids = set()

        for event in occupation_events:
            person = event.person

            # Skip duplicates (person might have multiple occupation events)
            if person.id in seen_person_ids:
                continue
            seen_person_ids.add(person.id)

            # Stop if we've reached max_results
            if len(results) >= max_results:
                break

            # Get birth event
            birth = Event.objects.filter(
                person=person,
                event_type='BIRT'
            ).first()

            results.append({
                "id": str(person.id),
                "display_name": person.full_name,
                "genealogical_id": person.genealogical_id,
                "occupation": event.description,
                "birth": {
                    "date": birth.date.isoformat() if birth and birth.date else None,
                    "place": birth.place if birth else None
                } if birth else None
            })

        return {
            "count": len(results),
            "people": results,
            "truncated": len(seen_person_ids) >= max_results,
            "search_terms": occupation_terms  # Show what terms were searched
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

    def find_relationship(self, person_id_1: str, person_id_2: str) -> Dict:
        """
        Compute genealogical relationship between two people by finding their common ancestor.

        Uses BFS to find the most recent common ancestor (MRCA) and then
        computes the relationship based on generational distances.

        Args:
            person_id_1: Person UUID or genealogical_id for first person
            person_id_2: Person UUID or genealogical_id for second person

        Returns:
            {
                "relationship": "second cousin once removed",
                "relationship_type": "cousin",
                "common_ancestor": {
                    "id": "uuid",
                    "name": "Pieter van Zanten",
                    "genealogical_id": "VII.3.a"
                },
                "generations_from_person_1": 2,
                "generations_from_person_2": 3,
                "explanation": "Person 1 is the grandchild of the common ancestor..."
            }

        Returns {"error": "..."} if people not found or not related.
        """
        # Resolve both person IDs
        person1 = self._get_person(person_id_1)
        person2 = self._get_person(person_id_2)

        if not person1:
            return {"error": f"Person not found: {person_id_1}"}
        if not person2:
            return {"error": f"Person not found: {person_id_2}"}

        # Check if same person
        if person1.id == person2.id:
            return {
                "relationship": "self",
                "relationship_type": "self",
                "generations_from_person_1": 0,
                "generations_from_person_2": 0,
                "explanation": "Same person"
            }

        # Find all ancestors for both people with generation distances
        ancestors1 = self._get_all_ancestors(person1)
        ancestors2 = self._get_all_ancestors(person2)

        # Check if one is direct ancestor of the other
        if person1.id in ancestors2:
            gen_distance = ancestors2[person1.id]
            rel_name = self._get_ancestor_relationship(gen_distance)
            return {
                "relationship": rel_name,
                "relationship_type": "ancestor",
                "generations_from_person_1": 0,
                "generations_from_person_2": gen_distance,
                "explanation": f"{person2.full_name} is {gen_distance} generation(s) below {person1.full_name}, making them {person1.full_name}'s {rel_name}"
            }

        if person2.id in ancestors1:
            gen_distance = ancestors1[person2.id]
            rel_name = self._get_descendant_relationship(gen_distance)
            return {
                "relationship": rel_name,
                "relationship_type": "descendant",
                "generations_from_person_1": gen_distance,
                "generations_from_person_2": 0,
                "explanation": f"{person1.full_name} is {gen_distance} generation(s) below {person2.full_name}, making them {person2.full_name}'s {rel_name}"
            }

        # Find most recent common ancestor (MRCA)
        common_ancestors = set(ancestors1.keys()) & set(ancestors2.keys())

        if not common_ancestors:
            return {
                "relationship": "none",
                "relationship_type": "none",
                "error": "No common ancestor found - people are not related in this dataset"
            }

        # Get MRCA (minimum combined generational distance)
        mrca_id = min(common_ancestors, key=lambda aid: ancestors1[aid] + ancestors2[aid])
        mrca = Person.objects.get(id=mrca_id)

        gen_from_1 = ancestors1[mrca_id]
        gen_from_2 = ancestors2[mrca_id]

        # Compute relationship
        relationship, rel_type = self._compute_cousin_relationship(gen_from_1, gen_from_2)

        return {
            "relationship": relationship,
            "relationship_type": rel_type,
            "common_ancestor": {
                "id": str(mrca.id),
                "name": mrca.full_name,
                "genealogical_id": mrca.genealogical_id
            },
            "generations_from_person_1": gen_from_1,
            "generations_from_person_2": gen_from_2,
            "explanation": f"{person1.full_name} is {gen_from_1} generation(s) from common ancestor {mrca.full_name}, {person2.full_name} is {gen_from_2} generation(s) from the same ancestor. This makes them {relationship}."
        }

    def _get_all_ancestors(self, person: Person) -> Dict:
        """
        Get all ancestors of a person with their generational distances.

        Uses BFS to traverse up the family tree.

        Returns:
            Dictionary mapping ancestor UUID to generation distance
        """
        from uuid import UUID

        ancestors = {}
        queue = [(person.id, 0)]
        visited = {person.id}

        while queue:
            current_id, gen_dist = queue.pop(0)

            # Get parents
            parent_rels = Relationship.objects.filter(child_id=current_id).select_related('parent')

            for rel in parent_rels:
                parent = rel.parent
                if parent.id not in visited:
                    visited.add(parent.id)
                    ancestors[parent.id] = gen_dist + 1
                    queue.append((parent.id, gen_dist + 1))

        return ancestors

    def _get_ancestor_relationship(self, generations: int) -> str:
        """Get relationship name when person2 is ancestor of person1"""
        if generations == 1:
            return "parent"
        elif generations == 2:
            return "grandparent"
        elif generations == 3:
            return "great-grandparent"
        else:
            greats = "great-" * (generations - 2)
            return f"{greats}grandparent"

    def _get_descendant_relationship(self, generations: int) -> str:
        """Get relationship name when person2 is descendant of person1"""
        if generations == 1:
            return "child"
        elif generations == 2:
            return "grandchild"
        elif generations == 3:
            return "great-grandchild"
        else:
            greats = "great-" * (generations - 2)
            return f"{greats}grandchild"

    def _compute_cousin_relationship(self, gen_from_1: int, gen_from_2: int) -> tuple:
        """
        Compute cousin relationship based on generational distances.

        Returns:
            Tuple of (relationship_string, relationship_type)
        """
        # Siblings share parents (both are 1 generation from common ancestor)
        if gen_from_1 == 1 and gen_from_2 == 1:
            return ("sibling", "sibling")

        min_gen = min(gen_from_1, gen_from_2)
        max_gen = max(gen_from_1, gen_from_2)
        removed = max_gen - min_gen

        if min_gen == 1:
            # Aunt/uncle/niece/nephew territory
            if removed == 1:
                return ("aunt/uncle or niece/nephew", "aunt_uncle_niece_nephew")
            else:
                greats = "great-" * (removed - 1)
                return (f"{greats}aunt/uncle or {greats}niece/nephew", "aunt_uncle_niece_nephew")

        # Standard cousin calculation
        cousin_degree = min_gen - 1

        if removed == 0:
            # Same generation
            if cousin_degree == 1:
                return ("first cousin", "cousin")
            elif cousin_degree == 2:
                return ("second cousin", "cousin")
            elif cousin_degree == 3:
                return ("third cousin", "cousin")
            else:
                return (f"{cousin_degree}th cousin", "cousin")
        else:
            # Different generations (removed)
            if removed == 1:
                removed_str = "once removed"
            elif removed == 2:
                removed_str = "twice removed"
            else:
                removed_str = f"{removed} times removed"

            if cousin_degree == 1:
                return (f"first cousin {removed_str}", "cousin")
            elif cousin_degree == 2:
                return (f"second cousin {removed_str}", "cousin")
            elif cousin_degree == 3:
                return (f"third cousin {removed_str}", "cousin")
            else:
                return (f"{cousin_degree}th cousin {removed_str}", "cousin")

    def search_source_text(self, query: str, max_results: int = 10) -> Dict:
        """
        Search genealogical source texts for information using semantic search.

        This tool is useful for cross-cutting queries that aren't about specific people,
        such as: "Who lived in Minneapolis?", "Are there any musicians?", "Who served in the military?"

        Args:
            query: Search query describing what you're looking for
            max_results: Maximum number of text chunks to return (default: 10)

        Returns:
            {
                "count": int,
                "results": [
                    {
                        "chunk_id": "uuid",
                        "text": "Full narrative text from the source...",
                        "subject": "Pieter van Zanten",
                        "genealogical_id": "VII.3.a",
                        "page_range": "45-47",
                        "score": 0.85,
                        "mentioned_people": [
                            {"name": "Pieter van Zanten", "genealogical_id": "VII.3.a"},
                            {"name": "Dina Schouten", "genealogical_id": "VII.3.a.spouse1"}
                        ]
                    }
                ]
            }
        """
        # Limit max_results for safety
        max_results = min(max_results, self.max_results)

        # Use the hybrid retriever
        retriever = HybridRetriever()
        chunks = retriever.retrieve(query=query, top_k=max_results, expand_window=0)

        results = []
        for chunk in chunks:
            # Extract genealogical IDs mentioned in the text
            # Pattern matches genealogical IDs like "VII.3.a" or "IX.10.b.spouse1"
            text = chunk.get('text_content', '')
            id_pattern = r'\b([IVX]+\.\d+\.[a-z]+(?:\.\w+)?)\b'
            mentioned_ids = re.findall(id_pattern, text)

            # Get unique mentioned people with their names
            mentioned_people = []
            seen_ids = set()

            for gen_id in mentioned_ids:
                if gen_id in seen_ids:
                    continue
                seen_ids.add(gen_id)

                # Try to find person with this genealogical_id
                person = Person.objects.filter(genealogical_id=gen_id).first()
                if person:
                    mentioned_people.append({
                        "name": person.full_name,
                        "genealogical_id": gen_id
                    })

            results.append({
                "chunk_id": str(chunk['id']),
                "text": text,
                "subject": chunk.get('subject', ''),
                "genealogical_id": chunk.get('genealogical_identifier', ''),
                "page_range": f"{chunk.get('start_page', '?')}-{chunk.get('end_page', '?')}",
                "score": float(chunk.get('rrf_score', 0.0)),
                "mentioned_people": mentioned_people
            })

        return {
            "count": len(results),
            "results": results,
            "query": query
        }
