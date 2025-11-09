"""
Genealogy tools for LLM agentic workflows.

These tools allow the LLM to iteratively request information
to answer complex queries like relationship tracing and disambiguation.

All tools work with Identity objects (canonical people) rather than
raw PersonMentions, providing complete aggregated information.
"""

import logging
from typing import List, Dict, Optional
from uuid import UUID
from django.db.models import Q, Count
from genealogy.models import (
    Identity,
    MentionToIdentity,
    PersonMention,
    RelationshipMention,
    Event,
    PartnershipMention,
)

logger = logging.getLogger(__name__)


class GenealogyTools:
    """Tools for LLM to interact with genealogy database"""

    def __init__(self):
        self.max_results = 20  # Safety limit for queries

    def _get_identity(self, person_id: str, annotate_mentions: bool = False):
        """
        Get identity by genealogical_id or UUID.

        Args:
            person_id: Identity UUID or genealogical_id
            annotate_mentions: If True, annotate with mention_count

        Returns:
            Identity object or None if not found
        """
        # Build base queryset
        base_qs = Identity.objects.filter(is_deleted=False)
        if annotate_mentions:
            base_qs = base_qs.annotate(mention_count=Count('mention_mappings'))

        # Try genealogical_id first
        identity = base_qs.filter(genealogical_identifier=person_id).first()

        if identity:
            return identity

        # Try UUID - validate format first
        try:
            UUID(person_id)  # Validate UUID format
            return base_qs.filter(id=person_id).first()
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
                        "parents": ["Johannes van Zanten", "Maria de Vries"],
                        "num_mentions": 3
                    }
                ],
                "truncated": bool
            }
        """
        # Limit max_results for safety
        max_results = min(max_results, self.max_results)

        # Search identities by display name
        identities = Identity.objects.filter(
            display_name__icontains=name,
            is_deleted=False
        ).annotate(
            mention_count=Count('mention_mappings')
        ).order_by('-mention_count', 'display_name')[:max_results]

        results = []
        for identity in identities:
            # Get birth/death events from all mentions
            mention_ids = identity.mention_mappings.values_list('mention_id', flat=True)

            birth = Event.objects.filter(
                mention_id__in=mention_ids,
                event_type='BIRT'
            ).select_related('place').first()

            death = Event.objects.filter(
                mention_id__in=mention_ids,
                event_type='DEAT'
            ).select_related('place').first()

            # Get parents (deduplicated by identity)
            parent_rels = RelationshipMention.objects.filter(
                child_mention_id__in=mention_ids
            ).select_related('parent_mention')

            parent_mention_ids = parent_rels.values_list('parent_mention_id', flat=True)
            parent_mappings = MentionToIdentity.objects.filter(
                mention_id__in=parent_mention_ids
            ).select_related('identity')

            parent_identities = {}
            for mapping in parent_mappings:
                parent_identities[mapping.identity.id] = mapping.identity.display_name

            parents = list(parent_identities.values())

            results.append({
                "id": str(identity.id),
                "display_name": identity.display_name,
                "genealogical_id": identity.genealogical_identifier,
                "birth": {
                    "date": birth.date.isoformat() if birth and birth.date else None,
                    "place": birth.place.name if birth and birth.place else None
                } if birth else None,
                "death": {
                    "date": death.date.isoformat() if death and death.date else None,
                    "place": death.place.name if death and death.place else None
                } if death else None,
                "parents": parents,
                "num_mentions": identity.mention_count
            })

        return {
            "count": len(results),
            "people": results,
            "truncated": identities.count() >= max_results
        }

    def get_person_details(self, person_id: str) -> Dict:
        """
        Get detailed information about a specific person.

        Args:
            person_id: Identity UUID or genealogical_id (e.g., "II.3.a")

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
                "num_mentions": 3
            }
        """
        # Get identity by genealogical_id or UUID (with mention count)
        identity = self._get_identity(person_id, annotate_mentions=True)

        if not identity:
            return {"error": f"Person not found: {person_id}"}

        # Get all mention IDs for this identity
        mention_ids = list(identity.mention_mappings.values_list('mention_id', flat=True))

        # Get all events from all mentions
        events = Event.objects.filter(
            mention_id__in=mention_ids
        ).select_related('place').order_by('date')

        events_data = []
        for event in events:
            date_str = event.date.isoformat() if event.date else None
            if event.date_estimated and date_str:
                date_str += " (estimated)"

            events_data.append({
                "type": event.get_event_type_display(),
                "date": date_str,
                "place": event.place.name if event.place else None,
                "description": event.description
            })

        # Get parents (deduplicated by identity)
        parent_rels = RelationshipMention.objects.filter(
            child_mention_id__in=mention_ids
        ).select_related('parent_mention')

        parent_mention_ids = parent_rels.values_list('parent_mention_id', flat=True)
        parent_mappings = MentionToIdentity.objects.filter(
            mention_id__in=parent_mention_ids
        ).select_related('identity')

        parent_identities = {}
        for mapping in parent_mappings:
            parent_identities[mapping.identity.id] = mapping.identity

        parents = [
            {"id": str(identity.id), "name": identity.display_name}
            for identity in parent_identities.values()
        ]

        # Get children (deduplicated by identity)
        child_rels = RelationshipMention.objects.filter(
            parent_mention_id__in=mention_ids
        ).select_related('child_mention')

        child_mention_ids = child_rels.values_list('child_mention_id', flat=True)
        child_mappings = MentionToIdentity.objects.filter(
            mention_id__in=child_mention_ids
        ).select_related('identity')

        child_identities = {}
        for mapping in child_mappings:
            child_identities[mapping.identity.id] = mapping.identity

        children = [
            {"id": str(identity.id), "name": identity.display_name}
            for identity in child_identities.values()
        ]

        # Get partnerships (deduplicated by partner identity)
        partnerships = PartnershipMention.objects.filter(
            partners__id__in=mention_ids
        ).distinct().prefetch_related('partners')

        partner_identities = {}
        for partnership in partnerships:
            # Get partner mentions (excluding this identity's mentions)
            partner_mentions = partnership.partners.exclude(id__in=mention_ids)

            for partner_mention in partner_mentions:
                try:
                    partner_mapping = MentionToIdentity.objects.get(mention=partner_mention)
                    partner_identity = partner_mapping.identity

                    if partner_identity.id not in partner_identities:
                        partner_identities[partner_identity.id] = {
                            'identity': partner_identity,
                            'types': set()
                        }

                    partner_identities[partner_identity.id]['types'].add(
                        partnership.get_partnership_type_display()
                    )
                except MentionToIdentity.DoesNotExist:
                    # Partner not merged yet - skip
                    continue

        partners = [
            {
                "id": str(data['identity'].id),
                "name": data['identity'].display_name,
                "type": ", ".join(data['types'])
            }
            for data in partner_identities.values()
        ]

        return {
            "id": str(identity.id),
            "display_name": identity.display_name,
            "genealogical_id": identity.genealogical_identifier,
            "events": events_data,
            "parents": parents,
            "children": children,
            "partners": partners,
            "num_mentions": identity.mention_count,
            "notes": identity.notes
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
        # First, get identities matching the name
        identities = Identity.objects.filter(
            display_name__icontains=name,
            is_deleted=False
        )

        # Filter by birth year if specified
        if birth_year_min or birth_year_max:
            # Get mention IDs with birth events in the year range
            birth_events = Event.objects.filter(event_type='BIRT')

            if birth_year_min:
                birth_events = birth_events.filter(date__year__gte=birth_year_min)
            if birth_year_max:
                birth_events = birth_events.filter(date__year__lte=birth_year_max)

            # Get mention IDs with matching birth events
            mention_ids_with_birth = birth_events.values_list('mention_id', flat=True)

            # Get identity IDs that have any of these mentions
            identity_ids_with_birth = MentionToIdentity.objects.filter(
                mention_id__in=mention_ids_with_birth
            ).values_list('identity_id', flat=True)

            # Filter identities to only those with matching birth years
            identities = identities.filter(id__in=identity_ids_with_birth)

        # Annotate and limit
        identities = identities.annotate(
            mention_count=Count('mention_mappings')
        ).order_by('-mention_count', 'display_name')[:self.max_results]

        # Build results using same logic as search_person_by_name
        results = []
        for identity in identities:
            mention_ids = identity.mention_mappings.values_list('mention_id', flat=True)

            birth = Event.objects.filter(
                mention_id__in=mention_ids,
                event_type='BIRT'
            ).select_related('place').first()

            death = Event.objects.filter(
                mention_id__in=mention_ids,
                event_type='DEAT'
            ).select_related('place').first()

            parent_rels = RelationshipMention.objects.filter(
                child_mention_id__in=mention_ids
            ).select_related('parent_mention')

            parent_mention_ids = parent_rels.values_list('parent_mention_id', flat=True)
            parent_mappings = MentionToIdentity.objects.filter(
                mention_id__in=parent_mention_ids
            ).select_related('identity')

            parent_identities = {}
            for mapping in parent_mappings:
                parent_identities[mapping.identity.id] = mapping.identity.display_name

            parents = list(parent_identities.values())

            results.append({
                "id": str(identity.id),
                "display_name": identity.display_name,
                "genealogical_id": identity.genealogical_identifier,
                "birth": {
                    "date": birth.date.isoformat() if birth and birth.date else None,
                    "place": birth.place.name if birth and birth.place else None
                } if birth else None,
                "death": {
                    "date": death.date.isoformat() if death and death.date else None,
                    "place": death.place.name if death and death.place else None
                } if death else None,
                "parents": parents,
                "num_mentions": identity.mention_count
            })

        return {
            "count": len(results),
            "people": results,
            "truncated": identities.count() >= self.max_results
        }

    def get_children(self, person_id: str) -> Dict:
        """
        Get all children of a person.

        Args:
            person_id: Identity UUID or genealogical_id

        Returns:
            {
                "person": {"id": "uuid", "name": "Pieter van Zanten"},
                "children": [
                    {"id": "uuid", "name": "Anna van Zanten", "birth_year": 1870}
                ],
                "count": int
            }
        """
        # Get identity by genealogical_id or UUID
        identity = self._get_identity(person_id)

        if not identity:
            return {"error": f"Person not found: {person_id}"}

        # Get all mention IDs for this identity
        mention_ids = list(identity.mention_mappings.values_list('mention_id', flat=True))

        # Get child relationships
        child_rels = RelationshipMention.objects.filter(
            parent_mention_id__in=mention_ids
        ).select_related('child_mention')

        # Get child mention IDs and resolve to identities
        child_mention_ids = child_rels.values_list('child_mention_id', flat=True)
        child_mappings = MentionToIdentity.objects.filter(
            mention_id__in=child_mention_ids
        ).select_related('identity')

        child_identities = {}
        for mapping in child_mappings:
            child_identities[mapping.identity.id] = mapping.identity

        # Get birth years for children
        children = []
        for child_identity in child_identities.values():
            child_mention_ids = child_identity.mention_mappings.values_list('mention_id', flat=True)
            birth = Event.objects.filter(
                mention_id__in=child_mention_ids,
                event_type='BIRT'
            ).first()

            children.append({
                "id": str(child_identity.id),
                "name": child_identity.display_name,
                "genealogical_id": child_identity.genealogical_identifier,
                "birth_year": birth.date.year if birth and birth.date else None
            })

        return {
            "person": {
                "id": str(identity.id),
                "name": identity.display_name
            },
            "children": children,
            "count": len(children)
        }

    def get_parents(self, person_id: str) -> Dict:
        """
        Get parents of a person.

        Args:
            person_id: Identity UUID or genealogical_id

        Returns:
            {
                "person": {"id": "uuid", "name": "Pieter van Zanten"},
                "parents": [
                    {"id": "uuid", "name": "Johannes van Zanten"}
                ],
                "count": int
            }
        """
        # Get identity by genealogical_id or UUID
        identity = self._get_identity(person_id)

        if not identity:
            return {"error": f"Person not found: {person_id}"}

        # Get all mention IDs for this identity
        mention_ids = list(identity.mention_mappings.values_list('mention_id', flat=True))

        # Get parent relationships
        parent_rels = RelationshipMention.objects.filter(
            child_mention_id__in=mention_ids
        ).select_related('parent_mention')

        parent_mention_ids = parent_rels.values_list('parent_mention_id', flat=True)
        parent_mappings = MentionToIdentity.objects.filter(
            mention_id__in=parent_mention_ids
        ).select_related('identity')

        parent_identities = {}
        for mapping in parent_mappings:
            parent_identities[mapping.identity.id] = mapping.identity

        parents = [
            {
                "id": str(identity.id),
                "name": identity.display_name,
                "genealogical_id": identity.genealogical_identifier
            }
            for identity in parent_identities.values()
        ]

        return {
            "person": {
                "id": str(identity.id),
                "name": identity.display_name
            },
            "parents": parents,
            "count": len(parents)
        }
