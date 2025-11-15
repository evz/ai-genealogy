"""Task to persist extracted entities (events) to database records"""
import logging
import re

from celery import shared_task
from django.db import transaction

from ..models import Document, Event, Person, TextChunk
from ..utils.date_parsing import parse_genealogical_date

logger = logging.getLogger(__name__)


def correct_event_fields(event_data):
    """
    Correct common field ordering mistakes from LLM extraction.

    Common issues:
    - Date in place field (looks like year: 1850, 1875, etc.)
    - Place in description field (looks like address/city)
    - Missing fields or wrong positions

    Args:
        event_data: Dict with keys: person, event_type, date, place, description

    Returns:
        Corrected event_data dict
    """
    corrected = event_data.copy()

    # Helper: Check if string looks like a year (4 digits)
    def looks_like_year(s):
        if not s or not isinstance(s, str):
            return False
        return bool(re.match(r'^\d{4}$', s.strip()))

    # Helper: Check if string looks like a date (has dots, slashes, or dashes)
    def looks_like_date(s):
        if not s or not isinstance(s, str):
            return False
        s = s.strip()
        # Matches patterns like "15.3.1850", "1850-03-15", "3/15/1850"
        return bool(re.search(r'\d+[./-]\d+[./-]?\d*', s)) or looks_like_year(s)

    # Helper: Check if string looks like a place (has address keywords or is alphabetic)
    def looks_like_place(s):
        if not s or not isinstance(s, str):
            return False
        s = s.strip().lower()
        # Common address/place indicators
        place_keywords = ['straat', 'laan', 'weg', 'plein', 'kade', 'gracht', 'huis']
        has_keyword = any(kw in s for kw in place_keywords)
        # Or mostly alphabetic (city names)
        mostly_alpha = len([c for c in s if c.isalpha() or c.isspace()]) > len(s) * 0.6
        return has_keyword or mostly_alpha

    date = corrected.get('date', '').strip()
    place = corrected.get('place', '').strip()
    description = corrected.get('description', '').strip()

    # Fix 1: If date is empty but place looks like a date, swap them
    if not date and looks_like_date(place):
        logger.debug(f"Swapping date/place: date='{date}' place='{place}'")
        corrected['date'] = place
        corrected['place'] = ''
        place = ''
        date = corrected['date']

    # Fix 2: If place is empty but description looks like a place, swap them
    # (BUT NOT for OCCU events where description should be the occupation)
    if not place and looks_like_place(description) and corrected.get('event_type') != 'OCCU':
        logger.debug(f"Swapping place/description: place='{place}' description='{description}'")
        corrected['place'] = description
        corrected['description'] = ''

    # Fix 3: For RESI events, if description has the place and place has the date
    if corrected.get('event_type') == 'RESI':
        if not date and looks_like_date(place):
            # Date is in place field
            if looks_like_place(description):
                # Place is in description field
                logger.debug(f"RESI fix: moving date from place, place from description")
                corrected['date'] = place
                corrected['place'] = description
                corrected['description'] = ''

    # Fix 4: For OCCU events, ensure occupation is in description and place is empty
    if corrected.get('event_type') == 'OCCU':
        if place and not description:
            # Occupation might be in place field
            if not looks_like_place(place) and not looks_like_date(place):
                logger.debug(f"OCCU fix: moving occupation from place to description")
                corrected['description'] = place
                corrected['place'] = ''

    return corrected


@shared_task(bind=True)
def persist_extracted_entities(self, document_id: str):
    """
    Persist extracted entities from TextChunk JSON fields to database records.

    This task reads extracted_events from TextChunks and creates Event records
    linked to Person records via genealogical_id.

    Args:
        document_id: UUID string of the Document to process

    Returns:
        dict: Result summary with counts
    """
    try:
        document = Document.objects.get(id=document_id)
        logger.info(f"Starting entity persistence for document: {document.title}")

        # Get all chunks with extracted entities that haven't been persisted yet
        chunks_to_persist = TextChunk.objects.filter(
            document=document,
            entities_extracted=True,
            extracted_events__isnull=False,
        ).exclude(extracted_events=[])

        logger.info(f"Found {chunks_to_persist.count()} chunks with extracted events to persist")

        if not chunks_to_persist.exists():
            return {
                "success": True,
                "message": "No chunks with extracted events found",
                "document_id": str(document_id),
                "events_created": 0,
            }

        # Create a lookup of genealogical_id -> Person for fast lookups
        people_by_gen_id = {}
        for person in Person.objects.filter(source_documents=document):
            people_by_gen_id[person.genealogical_id] = person

        logger.info(f"Loaded {len(people_by_gen_id)} people from document")

        stats = {
            'events_created': 0,
            'events_skipped_no_person': 0,
            'events_skipped_no_gen_id': 0,
            'chunks_processed': 0,
        }

        with transaction.atomic():
            for chunk in chunks_to_persist:
                chunk_events_created = 0

                # Process extracted_events
                for event_data in chunk.extracted_events:
                    # Correct common field ordering issues
                    event_data = correct_event_fields(event_data)

                    # Extract the person name from the event
                    person_name = event_data.get('person')
                    if not person_name:
                        logger.debug(f"Event in chunk {chunk.sequence_number} has no person field")
                        stats['events_skipped_no_person'] += 1
                        continue

                    # Try to find the Person record
                    # First, try using the chunk's genealogical_identifier if it matches the person
                    person = None
                    if chunk.genealogical_identifier:
                        person = people_by_gen_id.get(chunk.genealogical_identifier)
                        # Verify the person's name roughly matches
                        if person and person_name.lower() not in person.full_name.lower():
                            # Might be a different person mentioned in the chunk
                            person = None

                    if not person:
                        # Try to find by name match (fuzzy)
                        # This is a fallback for events of people mentioned but not the primary subject
                        for gen_id, p in people_by_gen_id.items():
                            if person_name.lower() in p.full_name.lower():
                                person = p
                                break

                    if not person:
                        logger.debug(
                            f"Could not find Person for event: {event_data.get('event_type')} "
                            f"for {person_name} in chunk {chunk.sequence_number}"
                        )
                        stats['events_skipped_no_person'] += 1
                        continue

                    # Parse the date string
                    date_str = event_data.get('date', '')
                    parsed_date, is_approximate = parse_genealogical_date(date_str)

                    # Create Event record
                    event = Event.objects.create(
                        person=person,
                        event_type=event_data.get('event_type', 'OTHER'),
                        date=parsed_date,
                        date_original=date_str,
                        date_approximate=is_approximate,
                        place=event_data.get('place', ''),
                        description=event_data.get('description', ''),
                        source_chunk=chunk,
                    )

                    chunk_events_created += 1
                    stats['events_created'] += 1

                    logger.debug(
                        f"Created {event.event_type} event for {person.genealogical_id} "
                        f"({person.full_name}) from chunk {chunk.sequence_number}"
                    )

                stats['chunks_processed'] += 1

                if chunk_events_created > 0:
                    logger.info(
                        f"Chunk {chunk.sequence_number}: Created {chunk_events_created} events"
                    )

        logger.info(
            f"Entity persistence complete for {document.title}: "
            f"{stats['events_created']} events created from {stats['chunks_processed']} chunks, "
            f"{stats['events_skipped_no_person']} skipped (no person match), "
            f"{stats['events_skipped_no_gen_id']} skipped (no gen ID)"
        )

        return {
            "success": True,
            "message": f"Created {stats['events_created']} events from {stats['chunks_processed']} chunks",
            "document_id": str(document_id),
            **stats,
        }

    except Document.DoesNotExist:
        error_msg = f"Document with id {document_id} not found"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"Entity persistence failed for {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "document_id": str(document_id),
        }
