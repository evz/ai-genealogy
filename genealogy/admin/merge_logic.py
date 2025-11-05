"""
New merge/unmerge logic using reversible provenance architecture.

Key principles:
- PersonMention, RelationshipMention, PartnershipMention are IMMUTABLE
- Only MentionToIdentity mappings change during merge/unmerge
- MergeEvent provides complete audit trail for reversibility
"""
import logging
from typing import List
from uuid import UUID

from django.db import transaction
from django.utils import timezone

from ..models import Identity, MentionToIdentity, MergeEvent, PersonMention

logger = logging.getLogger(__name__)


def merge_mentions(
    mention_ids: List[UUID],
    target_identity_id: UUID = None,
    merged_by: str = "unknown",
    merge_reason: dict = None,
    preferred_genealogical_identifier: str = None
) -> Identity:
    """
    Merge multiple PersonMentions into a single Identity.

    Args:
        mention_ids: List of PersonMention IDs to merge
        target_identity_id: Optional - reuse existing Identity, or create new one
        merged_by: Username performing the merge
        merge_reason: Dict with merge metadata (confidence, reasons, etc.)
        preferred_genealogical_identifier: Optional - explicitly choose which genealogical_identifier to use

    Returns:
        The target Identity that all mentions now map to
    """
    if len(mention_ids) < 2:
        raise ValueError("Must provide at least 2 mentions to merge")

    with transaction.atomic():
        # Get all the mentions
        mentions = PersonMention.objects.filter(id__in=mention_ids)
        if mentions.count() != len(mention_ids):
            raise ValueError("Some mention IDs not found")

        # Get current identity mappings
        mappings = MentionToIdentity.objects.filter(mention_id__in=mention_ids).select_related('identity')

        # Collect all involved identities (before merge)
        old_identities = {m.identity for m in mappings}

        # Determine best genealogical_identifier
        # Priority: 1) explicit preference, 2) from mentions, 3) from old identities
        genealogical_identifier = preferred_genealogical_identifier
        if not genealogical_identifier:
            # Try to get from mentions
            for mention in mentions:
                if mention.genealogical_id:
                    genealogical_identifier = mention.genealogical_id
                    break

            # If still not found, try from old identities
            if not genealogical_identifier:
                for identity in old_identities:
                    if identity.genealogical_identifier:
                        genealogical_identifier = identity.genealogical_identifier
                        break

        # Determine target identity
        if target_identity_id:
            target_identity = Identity.objects.get(id=target_identity_id)
            # Update genealogical_identifier if we found a better one
            if genealogical_identifier and not target_identity.genealogical_identifier:
                target_identity.genealogical_identifier = genealogical_identifier
                target_identity.save()
        else:
            # Create a new identity
            # Use the first mention's name as display name
            first_mention = mentions.first()
            target_identity = Identity.objects.create(
                display_name=first_mention.full_name,
                notes=f"Merged from {len(mention_ids)} mentions",
                genealogical_identifier=genealogical_identifier
            )

        # Build merge event payload for reversibility
        payload = {
            'operation': 'merge',
            'mention_ids': [str(mid) for mid in mention_ids],
            'target_identity_id': str(target_identity.id),
            'old_mappings': [
                {
                    'mention_id': str(m.mention_id),
                    'old_identity_id': str(m.identity_id),
                }
                for m in mappings
            ],
            'merge_reason': merge_reason or {},
            'timestamp': timezone.now().isoformat()
        }

        # Create merge event BEFORE making changes
        merge_event = MergeEvent.objects.create(
            event_type='merge',
            payload=payload,
            performed_by=merged_by,
        )

        # Update all mappings to point to target identity
        mappings.update(
            identity=target_identity,
            mapped_at=timezone.now(),
            mapped_by=merged_by
        )

        # Soft-delete old identities that are now empty
        for old_identity in old_identities:
            if old_identity.id != target_identity.id:
                # Check if this identity has any remaining mentions
                remaining_count = MentionToIdentity.objects.filter(identity=old_identity).count()
                if remaining_count == 0:
                    old_identity.is_deleted = True
                    old_identity.notes = f"{old_identity.notes}\n[Absorbed into {target_identity.id} on {timezone.now()}]"
                    old_identity.save()

        logger.info(f"Merged {len(mention_ids)} mentions into identity {target_identity.id}")

        return target_identity


def unmerge_mentions(
    merge_event_id: int,
    performed_by: str = "unknown"
) -> List[Identity]:
    """
    Reverse a previous merge operation.

    Args:
        merge_event_id: ID of the MergeEvent to reverse
        performed_by: Username performing the unmerge

    Returns:
        List of Identities after unmerge (restored singleton identities)
    """
    with transaction.atomic():
        # Get the merge event
        merge_event = MergeEvent.objects.get(id=merge_event_id)

        if merge_event.event_type != 'merge':
            raise ValueError(f"Event {merge_event_id} is not a merge event")

        # Extract old mappings from payload
        old_mappings = merge_event.payload['old_mappings']

        # Restore each mention to its original identity
        restored_identities = []

        for mapping_data in old_mappings:
            mention_id = UUID(mapping_data['mention_id'])
            old_identity_id = UUID(mapping_data['old_identity_id'])

            # Get or restore the old identity
            old_identity = Identity.objects.get(id=old_identity_id)
            if old_identity.is_deleted:
                old_identity.is_deleted = False
                old_identity.notes = f"{old_identity.notes}\n[Restored on {timezone.now()}]"
                old_identity.save()

            # Update the mapping
            MentionToIdentity.objects.filter(mention_id=mention_id).update(
                identity=old_identity,
                mapped_at=timezone.now(),
                mapped_by=performed_by
            )

            restored_identities.append(old_identity)

        # Create unmerge event
        unmerge_payload = {
            'operation': 'unmerge',
            'reversed_event_id': merge_event_id,
            'mention_ids': [m['mention_id'] for m in old_mappings],
            'restored_identities': [str(i.id) for i in restored_identities],
            'timestamp': timezone.now().isoformat()
        }

        MergeEvent.objects.create(
            event_type='unmerge',
            payload=unmerge_payload,
            performed_by=performed_by,
            reversed_event=merge_event
        )

        logger.info(f"Unmerged event {merge_event_id}, restored {len(restored_identities)} identities")

        return restored_identities
