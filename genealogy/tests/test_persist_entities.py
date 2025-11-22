"""Tests for persist_extracted_entities task"""
import uuid
from unittest.mock import Mock

from django.test import TestCase

from genealogy.models import Document, Event, Person, TextChunk
from genealogy.tasks.persist_entities import persist_extracted_entities


class PersistExtractedEntitiesTestCase(TestCase):
    """Test suite for persist_extracted_entities task"""

    def setUp(self):
        """Set up test data"""
        # Create a document
        self.document = Document.objects.create(
            title="Test Document",
            languages="nl"
        )

        # Create people with genealogical IDs
        self.person_ix_5_a = Person.objects.create(
            genealogical_id="IX.5.a",
            given_names="Jan",
            surname="van Zanten",
            generation=9
        )
        self.person_ix_5_a.source_documents.add(self.document)

        self.person_ix_5_b = Person.objects.create(
            genealogical_id="IX.5.b",
            given_names="Marie",
            surname="van Zanten",
            generation=9
        )
        self.person_ix_5_b.source_documents.add(self.document)

    def test_persist_events_basic(self):
        """Test basic event persistence from chunk JSON"""
        # Create chunk with extracted events
        chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=1,
            start_page=1,
            end_page=1,
            text_content="Jan van Zanten, * Amsterdam 15.3.1850, † Utrecht 3.11.1920",
            chunk_type="individual_entry",
            genealogical_identifier="IX.5.a",
            subject="Jan van Zanten",
            entities_extracted=True,
            extracted_events=[
                {
                    "person": "Jan van Zanten",
                    "event_type": "BIRTH",
                    "date": "15.3.1850",
                    "place": "Amsterdam"
                },
                {
                    "person": "Jan van Zanten",
                    "event_type": "DEATH",
                    "date": "3.11.1920",
                    "place": "Utrecht"
                }
            ]
        )

        # Run persistence task
        result = persist_extracted_entities(str(self.document.id))

        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(result['events_created'], 2)
        self.assertEqual(result['chunks_processed'], 1)

        # Verify events were created
        events = Event.objects.filter(person=self.person_ix_5_a).order_by('event_type')
        self.assertEqual(events.count(), 2)

        birth_event = events.filter(event_type='BIRTH').first()
        self.assertEqual(str(birth_event.date), "1850-03-15")
        self.assertEqual(birth_event.place, "Amsterdam")
        self.assertEqual(birth_event.source_chunk, chunk)

        death_event = events.filter(event_type='DEATH').first()
        self.assertEqual(str(death_event.date), "1920-11-03")
        self.assertEqual(death_event.place, "Utrecht")
        self.assertEqual(death_event.source_chunk, chunk)

    def test_persist_events_with_occupation(self):
        """Test persisting occupation events"""
        chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=2,
            start_page=1,
            end_page=1,
            text_content="Jan van Zanten, onderwijzer",
            chunk_type="individual_entry",
            genealogical_identifier="IX.5.a",
            subject="Jan van Zanten",
            entities_extracted=True,
            extracted_events=[
                {
                    "person": "Jan van Zanten",
                    "event_type": "OCCU",
                    "date": "",
                    "place": "",
                    "description": "onderwijzer"
                }
            ]
        )

        result = persist_extracted_entities(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['events_created'], 1)

        event = Event.objects.get(person=self.person_ix_5_a, event_type='OCCU')
        self.assertEqual(event.description, "onderwijzer")
        self.assertEqual(event.place, "")
        self.assertIsNone(event.date)

    def test_persist_events_no_person_match(self):
        """Test that events without matching person are skipped"""
        chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=3,
            start_page=1,
            end_page=1,
            text_content="Unknown person mentioned",
            chunk_type="individual_entry",
            genealogical_identifier="IX.5.a",
            subject="Jan van Zanten",
            entities_extracted=True,
            extracted_events=[
                {
                    "person": "Unknown Person",
                    "event_type": "BIRTH",
                    "date": "1800",
                    "place": "Nowhere"
                }
            ]
        )

        result = persist_extracted_entities(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['events_created'], 0)
        self.assertEqual(result['events_skipped_no_person'], 1)

    def test_persist_events_empty_extracted_events(self):
        """Test handling of chunks with empty extracted_events"""
        chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=4,
            start_page=1,
            end_page=1,
            text_content="No events here",
            chunk_type="individual_entry",
            genealogical_identifier="IX.5.a",
            subject="Jan van Zanten",
            entities_extracted=True,
            extracted_events=[]
        )

        result = persist_extracted_entities(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['events_created'], 0)
        self.assertIn("No chunks with extracted events found", result['message'])

    def test_persist_events_multiple_chunks(self):
        """Test persisting events from multiple chunks"""
        # First chunk with birth
        chunk1 = TextChunk.objects.create(
            document=self.document,
            sequence_number=5,
            start_page=1,
            end_page=1,
            text_content="Jan van Zanten, * Amsterdam 1850",
            chunk_type="individual_entry",
            genealogical_identifier="IX.5.a",
            subject="Jan van Zanten",
            entities_extracted=True,
            extracted_events=[
                {
                    "person": "Jan van Zanten",
                    "event_type": "BIRTH",
                    "date": "1850",
                    "place": "Amsterdam"
                }
            ]
        )

        # Second chunk with marriage
        chunk2 = TextChunk.objects.create(
            document=self.document,
            sequence_number=6,
            start_page=1,
            end_page=1,
            text_content="Jan van Zanten x Marie Jansen",
            chunk_type="individual_entry",
            genealogical_identifier="IX.5.a",
            subject="Jan van Zanten",
            entities_extracted=True,
            extracted_events=[
                {
                    "person": "Jan van Zanten",
                    "event_type": "MARR",
                    "date": "1875",
                    "place": "Rotterdam"
                }
            ]
        )

        result = persist_extracted_entities(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['events_created'], 2)
        self.assertEqual(result['chunks_processed'], 2)

        # Verify events from different chunks
        events = Event.objects.filter(person=self.person_ix_5_a)
        self.assertEqual(events.count(), 2)

    def test_persist_events_name_fuzzy_match(self):
        """Test that events can match person by partial name"""
        chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=7,
            start_page=1,
            end_page=1,
            text_content="Jan van Zanten geboren",
            chunk_type="individual_entry",
            genealogical_identifier="IX.5.b",  # Marie's chunk
            subject="Marie van Zanten",
            entities_extracted=True,
            extracted_events=[
                {
                    "person": "Jan",  # Partial name
                    "event_type": "BIRTH",
                    "date": "1850",
                    "place": "Amsterdam"
                }
            ]
        )

        result = persist_extracted_entities(str(self.document.id))

        # Should match Jan van Zanten by fuzzy name matching
        self.assertTrue(result['success'])
        self.assertEqual(result['events_created'], 1)

        event = Event.objects.get(event_type='BIRTH')
        self.assertEqual(event.person, self.person_ix_5_a)

    def test_persist_events_no_chunks_with_events(self):
        """Test handling when no chunks have extracted events"""
        result = persist_extracted_entities(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['events_created'], 0)
        self.assertIn("No chunks with extracted events found", result['message'])

    def test_persist_events_document_not_found(self):
        """Test error handling when document doesn't exist"""
        fake_id = str(uuid.uuid4())
        result = persist_extracted_entities(fake_id)

        self.assertFalse(result['success'])
        self.assertIn('not found', result['error'])

    def test_persist_events_with_description(self):
        """Test persisting events with description field"""
        chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=8,
            start_page=1,
            end_page=1,
            text_content="Jan moved to Utrecht for work",
            chunk_type="individual_entry",
            genealogical_identifier="IX.5.a",
            subject="Jan van Zanten",
            entities_extracted=True,
            extracted_events=[
                {
                    "person": "Jan van Zanten",
                    "event_type": "RESI",
                    "date": "1880",
                    "place": "Utrecht",
                    "description": "Moved for work"
                }
            ]
        )

        result = persist_extracted_entities(str(self.document.id))

        self.assertTrue(result['success'])
        self.assertEqual(result['events_created'], 1)

        event = Event.objects.get(event_type='RESI')
        self.assertEqual(event.description, "Moved for work")
