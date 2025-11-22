"""
Integration tests for the full genealogy extraction workflow.

Tests the complete pipeline:
1. Chunking (creates TextChunk records)
2. Build genealogy graph (creates Person/Relationship/Partnership from genealogical IDs)
3. Extract entities (LLM extracts events to JSON)
4. Persist entities (creates Event records from JSON)
"""

import pytest
from django.test import TestCase
from genealogy.models import Document, Person, Relationship, Partnership, Event, TextChunk
from genealogy.tasks.build_genealogy_graph import build_genealogy_graph
from genealogy.tasks.persist_entities import persist_extracted_entities


@pytest.mark.django_db
class TestWorkflowIntegration(TestCase):
    """Integration tests for full extraction workflow"""

    def setUp(self):
        """Set up test document with sample chunks"""
        self.document = Document.objects.create(
            title="Test Family Tree",
            languages="nld"
        )

    def test_full_workflow_creates_complete_graph(self):
        """Test that full workflow creates Person, Relationship, Partnership, and Event records"""

        # Step 1: Create chunks (simulates chunking task output)
        # First create Jan (Generation I parent)
        TextChunk.objects.create(
            document=self.document,
            sequence_number=0,
            chunk_type="individual_entry",
            text_content="I.1.a Jan van Zanten",
            subject="Jan van Zanten",
            genealogical_identifier="I.1.a",
            start_page=9,
            end_page=9,
        )

        # Then Pieter (Generation II)
        pieter_chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=1,
            chunk_type="individual_entry",
            text_content="Pieter van Zanten, * Amsterdam 15.3.1850, metselaar, † Rotterdam 3.11.1920",
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            family_groups=["Kinderen van Jan van Zanten (I.1.a):"],
            start_page=10,
            end_page=10,
        )

        family_header = TextChunk.objects.create(
            document=self.document,
            sequence_number=2,
            chunk_type="family_group_header",
            text_content="Kinderen van Pieter van Zanten en Maria de Vries (II.3.a):",
            start_page=11,
            end_page=11,
        )

        # Then Anna (Generation III)
        anna_chunk = TextChunk.objects.create(
            document=self.document,
            sequence_number=3,
            chunk_type="individual_entry",
            text_content="a. Anna van Zanten, * Rotterdam 15.6.1870",
            subject="Anna van Zanten",
            genealogical_identifier="III.5.a",
            family_groups=["Kinderen van Pieter van Zanten en Maria de Vries (II.3.a):"],
            start_page=11,
            end_page=11,
        )

        # Verify chunks created
        self.assertEqual(TextChunk.objects.filter(document=self.document).count(), 4)

        # Step 2: Build genealogy graph
        result = build_genealogy_graph(str(self.document.id))

        # Verify Person records created (Jan, Pieter, Anna, Maria minted spouse)
        self.assertEqual(result['people_created'], 4)

        jan = Person.objects.get(genealogical_id="I.1.a")
        pieter = Person.objects.get(genealogical_id="II.3.a")
        self.assertEqual(pieter.given_names, "Pieter")
        self.assertEqual(pieter.surname, "van Zanten")

        anna = Person.objects.get(genealogical_id="III.5.a")
        self.assertEqual(anna.given_names, "Anna")

        maria = Person.objects.get(genealogical_id="II.3.a.spouse1")
        self.assertEqual(maria.given_names, "Maria")
        self.assertEqual(maria.surname, "de Vries")

        # Verify Relationships created (Jan -> Pieter, Pieter -> Anna, Maria -> Anna)
        self.assertEqual(result['relationships_created'], 3)

        # Check Pieter -> Anna relationship
        parent_relationship = Relationship.objects.filter(parent=pieter, child=anna).first()
        self.assertIsNotNone(parent_relationship)
        self.assertEqual(parent_relationship.relationship_type, "BIOLOGICAL")

        # Check Maria -> Anna relationship
        maria_relationship = Relationship.objects.filter(parent=maria, child=anna).first()
        self.assertIsNotNone(maria_relationship)

        # Check Jan -> Pieter relationship
        grandparent_relationship = Relationship.objects.filter(parent=jan, child=pieter).first()
        self.assertIsNotNone(grandparent_relationship)

        # Verify Partnership created (Pieter & Maria)
        self.assertEqual(result['partnerships_created'], 1)

        partnership = Partnership.objects.filter(
            partner1=pieter,
            partner2=maria
        ).first()
        if not partnership:
            # Try reverse order
            partnership = Partnership.objects.filter(
                partner1=maria,
                partner2=pieter
            ).first()
        self.assertIsNotNone(partnership)
        self.assertEqual(partnership.partnership_type, "MARRIAGE")

        # Step 3: Add extracted events (simulates LLM extraction)
        pieter_chunk.extracted_events = [
            {
                "person": "Pieter van Zanten",
                "event_type": "BIRT",
                "date": "15.3.1850",
                "place": "Amsterdam",
                "description": ""
            },
            {
                "person": "Pieter van Zanten",
                "event_type": "OCCU",
                "date": "",
                "place": "",
                "description": "metselaar"
            },
            {
                "person": "Pieter van Zanten",
                "event_type": "DEAT",
                "date": "3.11.1920",
                "place": "Rotterdam",
                "description": ""
            }
        ]
        pieter_chunk.entities_extracted = True
        pieter_chunk.save()

        anna_chunk.extracted_events = [
            {
                "person": "Anna van Zanten",
                "event_type": "BIRT",
                "date": "15.6.1870",
                "place": "Rotterdam",
                "description": ""
            }
        ]
        anna_chunk.entities_extracted = True
        anna_chunk.save()

        # Step 4: Persist extracted entities
        result = persist_extracted_entities(str(self.document.id))

        # Verify Event records created
        self.assertEqual(result['events_created'], 4)

        # Check Pieter's events
        pieter_events = Event.objects.filter(person=pieter).order_by('event_type')
        self.assertEqual(pieter_events.count(), 3)

        birth = pieter_events.filter(event_type='BIRT').first()
        self.assertIsNotNone(birth)
        self.assertEqual(birth.place, "Amsterdam")
        self.assertEqual(str(birth.date), "1850-03-15")
        self.assertFalse(birth.date_approximate)

        occupation = pieter_events.filter(event_type='OCCU').first()
        self.assertIsNotNone(occupation)
        self.assertEqual(occupation.description, "metselaar")

        death = pieter_events.filter(event_type='DEAT').first()
        self.assertIsNotNone(death)
        self.assertEqual(death.place, "Rotterdam")
        self.assertEqual(str(death.date), "1920-11-03")

        # Check Anna's event
        anna_birth = Event.objects.filter(person=anna, event_type='BIRT').first()
        self.assertIsNotNone(anna_birth)
        self.assertEqual(anna_birth.place, "Rotterdam")
        self.assertEqual(str(anna_birth.date), "1870-06-15")

    def test_workflow_is_idempotent(self):
        """Test that running workflow twice doesn't create duplicates"""

        # Create chunk
        TextChunk.objects.create(
            document=self.document,
            sequence_number=1,
            chunk_type="individual_entry",
            text_content="Jan van Santen, * 1800",
            subject="Jan van Santen",
            genealogical_identifier="I.1.a",
            start_page=5,
            end_page=5,
        )

        # Run build_genealogy_graph twice
        result1 = build_genealogy_graph(str(self.document.id))
        result2 = build_genealogy_graph(str(self.document.id))

        # Second run should create nothing (idempotent)
        self.assertEqual(result1['people_created'], 1)
        self.assertEqual(result2['people_created'], 0)

        # Only one Person record should exist
        self.assertEqual(Person.objects.filter(genealogical_id="I.1.a").count(), 1)

    def test_workflow_handles_multiple_generations(self):
        """Test workflow with 3 generations of family"""

        # Generation I
        TextChunk.objects.create(
            document=self.document,
            sequence_number=1,
            chunk_type="individual_entry",
            text_content="I. Jan van Zanten",
            subject="Jan van Zanten",
            genealogical_identifier="I.1.a",
            start_page=5,
            end_page=5,
        )

        # Generation II
        TextChunk.objects.create(
            document=self.document,
            sequence_number=2,
            chunk_type="family_group_header",
            text_content="Kinderen van Jan van Zanten (I.1.a):",
            start_page=6,
            end_page=6,
        )

        TextChunk.objects.create(
            document=self.document,
            sequence_number=3,
            chunk_type="individual_entry",
            text_content="a. Pieter van Zanten",
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            family_groups=["Kinderen van Jan van Zanten (I.1.a):"],
            start_page=6,
            end_page=6,
        )

        # Generation III
        TextChunk.objects.create(
            document=self.document,
            sequence_number=4,
            chunk_type="family_group_header",
            text_content="Kinderen van Pieter van Zanten (II.3.a):",
            start_page=7,
            end_page=7,
        )

        TextChunk.objects.create(
            document=self.document,
            sequence_number=5,
            chunk_type="individual_entry",
            text_content="a. Anna van Zanten",
            subject="Anna van Zanten",
            genealogical_identifier="III.5.a",
            family_groups=["Kinderen van Pieter van Zanten (II.3.a):"],
            start_page=7,
            end_page=7,
        )

        result = build_genealogy_graph(str(self.document.id))

        # Verify all 3 people created
        self.assertEqual(result['people_created'], 3)

        jan = Person.objects.get(genealogical_id="I.1.a")
        pieter = Person.objects.get(genealogical_id="II.3.a")
        anna = Person.objects.get(genealogical_id="III.5.a")

        # Verify relationships across 3 generations
        self.assertEqual(result['relationships_created'], 2)

        # Jan -> Pieter
        self.assertTrue(Relationship.objects.filter(parent=jan, child=pieter).exists())

        # Pieter -> Anna
        self.assertTrue(Relationship.objects.filter(parent=pieter, child=anna).exists())

        # Test ancestry chain
        anna_parents = Relationship.objects.filter(child=anna).select_related('parent')
        self.assertEqual(anna_parents.count(), 1)
        self.assertEqual(anna_parents.first().parent, pieter)

        pieter_parents = Relationship.objects.filter(child=pieter).select_related('parent')
        self.assertEqual(pieter_parents.count(), 1)
        self.assertEqual(pieter_parents.first().parent, jan)

    def test_workflow_handles_siblings(self):
        """Test workflow correctly handles siblings with same parents"""

        # Parent
        TextChunk.objects.create(
            document=self.document,
            sequence_number=1,
            chunk_type="individual_entry",
            text_content="I. Jan van Zanten",
            subject="Jan van Zanten",
            genealogical_identifier="I.1.a",
            start_page=5,
            end_page=5,
        )

        # Siblings
        TextChunk.objects.create(
            document=self.document,
            sequence_number=2,
            chunk_type="family_group_header",
            text_content="Kinderen van Jan van Zanten (I.1.a):",
            start_page=6,
            end_page=6,
        )

        TextChunk.objects.create(
            document=self.document,
            sequence_number=3,
            chunk_type="individual_entry",
            text_content="a. Pieter van Zanten",
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            family_groups=["Kinderen van Jan van Zanten (I.1.a):"],
            start_page=6,
            end_page=6,
        )

        TextChunk.objects.create(
            document=self.document,
            sequence_number=4,
            chunk_type="individual_entry",
            text_content="b. Maria van Zanten",
            subject="Maria van Zanten",
            genealogical_identifier="II.3.b",
            family_groups=["Kinderen van Jan van Zanten (I.1.a):"],
            start_page=6,
            end_page=6,
        )

        TextChunk.objects.create(
            document=self.document,
            sequence_number=5,
            chunk_type="individual_entry",
            text_content="c. Anna van Zanten",
            subject="Anna van Zanten",
            genealogical_identifier="II.3.c",
            family_groups=["Kinderen van Jan van Zanten (I.1.a):"],
            start_page=6,
            end_page=6,
        )

        result = build_genealogy_graph(str(self.document.id))

        # Verify 4 people created (Jan + 3 siblings)
        self.assertEqual(result['people_created'], 4)

        jan = Person.objects.get(genealogical_id="I.1.a")
        pieter = Person.objects.get(genealogical_id="II.3.a")
        maria = Person.objects.get(genealogical_id="II.3.b")
        anna = Person.objects.get(genealogical_id="II.3.c")

        # Verify all 3 siblings have Jan as parent
        self.assertEqual(result['relationships_created'], 3)

        self.assertTrue(Relationship.objects.filter(parent=jan, child=pieter).exists())
        self.assertTrue(Relationship.objects.filter(parent=jan, child=maria).exists())
        self.assertTrue(Relationship.objects.filter(parent=jan, child=anna).exists())

        # Verify siblings share parent
        jan_children = Relationship.objects.filter(parent=jan).select_related('child')
        child_ids = set(rel.child.genealogical_id for rel in jan_children)
        self.assertEqual(child_ids, {"II.3.a", "II.3.b", "II.3.c"})

    def test_event_field_correction_during_persist(self):
        """Test that persist_entities corrects common field ordering mistakes"""

        # Create Person first
        TextChunk.objects.create(
            document=self.document,
            sequence_number=1,
            chunk_type="individual_entry",
            text_content="Jan van Zanten",
            subject="Jan van Zanten",
            genealogical_identifier="I.1.a",
            start_page=5,
            end_page=5,
        )

        build_genealogy_graph(str(self.document.id))

        # Add extracted events with field ordering errors
        chunk = TextChunk.objects.get(genealogical_identifier="I.1.a")
        chunk.extracted_events = [
            # Error: Date in place field
            {
                "person": "Jan van Zanten",
                "event_type": "BIRT",
                "date": "",
                "place": "1850",  # Wrong! Date in place field
                "description": "Amsterdam"  # Wrong! Place in description field
            },
        ]
        chunk.entities_extracted = True
        chunk.save()

        # Persist should auto-correct the fields
        persist_extracted_entities(str(self.document.id))

        jan = Person.objects.get(genealogical_id="I.1.a")
        birth = Event.objects.get(person=jan, event_type='BIRT')

        # Field correction should have moved date from place to date field
        self.assertEqual(birth.date.year, 1850)
        # And place from description to place field
        self.assertEqual(birth.place, "Amsterdam")
