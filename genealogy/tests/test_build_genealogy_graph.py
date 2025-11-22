"""Tests for build_genealogy_graph task"""
import pytest
from django.test import TestCase

from genealogy.models import Document, Person, Partnership, Relationship, TextChunk
from genealogy.tasks.build_genealogy_graph import GenealogyGraphBuilder, build_genealogy_graph


@pytest.mark.django_db
class TestBuildGenealogyGraph(TestCase):
    """Test build_genealogy_graph task and GenealogyGraphBuilder class"""

    def setUp(self):
        """Create test document"""
        self.document = Document.objects.create(
            title="Test Genealogy Document",
            languages="nld",
        )

    def test_creates_person_from_genealogical_id(self):
        """Test Person is created from chunk with genealogical_identifier"""
        # Create chunk with genealogical_identifier
        TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            generation_number=2,
        )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        stats = builder.build()

        # Assert Person created
        self.assertEqual(stats['people_created'], 1)
        self.assertEqual(stats['people_updated'], 0)

        person = Person.objects.get(genealogical_id="II.3.a")
        self.assertEqual(person.given_names, "Pieter")
        self.assertEqual(person.surname, "van Zanten")
        self.assertEqual(person.generation, 2)

    def test_creates_parent_child_relationship(self):
        """Test Relationship created from family group header"""
        # Create parent chunk
        TextChunk.objects.create(
            document=self.document,
            text_content="Jan van Zanten was the founder.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Jan van Zanten",
            genealogical_identifier="II.1.a",
            generation_number=2,
        )

        # Create child chunk with family_group referring to parent
        TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=2,
            end_page=2,
            sequence_number=1,
            subject="Pieter van Zanten",
            genealogical_identifier="III.5.a",
            generation_number=3,
            family_groups=["III.5. Kinderen van Jan van Zanten (II.1.a)"],
        )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        stats = builder.build()

        # Assert both people created
        self.assertEqual(stats['people_created'], 2)
        self.assertEqual(stats['relationships_created'], 1)

        # Assert relationship exists
        parent = Person.objects.get(genealogical_id="II.1.a")
        child = Person.objects.get(genealogical_id="III.5.a")

        relationship = Relationship.objects.get(parent=parent, child=child)
        self.assertEqual(relationship.relationship_type, 'BIOLOGICAL')

    def test_creates_partnership_for_parents(self):
        """Test Partnership created from family group header with both parents"""
        # Create parent1 chunk
        TextChunk.objects.create(
            document=self.document,
            text_content="Jan van Zanten was the founder.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Jan van Zanten",
            genealogical_identifier="II.1.a",
            generation_number=2,
        )

        # Create parent2 chunk
        TextChunk.objects.create(
            document=self.document,
            text_content="Maria Pieterse married Jan.",
            start_page=1,
            end_page=1,
            sequence_number=1,
            subject="Maria Pieterse",
            genealogical_identifier="II.1.b",
            generation_number=2,
        )

        # Create child chunk mentioning both parents
        TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=2,
            end_page=2,
            sequence_number=2,
            subject="Pieter van Zanten",
            genealogical_identifier="III.5.a",
            generation_number=3,
            family_groups=["III.5. Kinderen van Jan van Zanten en Maria Pieterse (II.1.a)"],
        )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        stats = builder.build()

        # Assert all people created
        self.assertEqual(stats['people_created'], 3)
        self.assertEqual(stats['partnerships_created'], 1)

        # Assert partnership exists
        parent1 = Person.objects.get(genealogical_id="II.1.a")
        parent2 = Person.objects.get(genealogical_id="II.1.b")

        # Partnership should exist (order-independent)
        partnership = Partnership.objects.filter(
            partner1__in=[parent1, parent2],
            partner2__in=[parent1, parent2],
        ).first()

        self.assertIsNotNone(partnership)
        self.assertEqual(partnership.partnership_type, 'MARRIAGE')

    def test_idempotency(self):
        """Test running task twice doesn't create duplicates"""
        # Create chunk
        TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            generation_number=2,
        )

        # Run task first time
        builder1 = GenealogyGraphBuilder(self.document)
        stats1 = builder1.build()
        self.assertEqual(stats1['people_created'], 1)

        # Run task second time
        builder2 = GenealogyGraphBuilder(self.document)
        stats2 = builder2.build()
        self.assertEqual(stats2['people_created'], 0)  # No new people
        self.assertEqual(stats2['people_updated'], 0)  # No updates (same data)

        # Assert only one person exists
        self.assertEqual(Person.objects.count(), 1)

    def test_links_chunks_to_people(self):
        """Test chunks are linked to their primary person"""
        # Create chunk
        chunk = TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            generation_number=2,
        )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        builder.build()

        # Refresh chunk from database
        chunk.refresh_from_db()

        # Assert chunk.primary_person is set
        self.assertIsNotNone(chunk.primary_person)
        self.assertEqual(chunk.primary_person.genealogical_id, "II.3.a")

    def test_multi_generation_family(self):
        """Test creating relationships across multiple generations"""
        # Generation 1 (root)
        TextChunk.objects.create(
            document=self.document,
            text_content="Gerrit van Zanten, the patriarch.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Gerrit van Zanten",
            genealogical_identifier="I.1.a",
            generation_number=1,
        )

        # Generation 2 (child of I.1.a)
        TextChunk.objects.create(
            document=self.document,
            text_content="Jan van Zanten, son of Gerrit.",
            start_page=2,
            end_page=2,
            sequence_number=1,
            subject="Jan van Zanten",
            genealogical_identifier="II.1.a",
            generation_number=2,
            family_groups=["II.1. Kinderen van Gerrit van Zanten (I.1.a)"],
        )

        # Generation 3 (child of II.1.a)
        TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten, son of Jan.",
            start_page=3,
            end_page=3,
            sequence_number=2,
            subject="Pieter van Zanten",
            genealogical_identifier="III.5.a",
            generation_number=3,
            family_groups=["III.5. Kinderen van Jan van Zanten (II.1.a)"],
        )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        stats = builder.build()

        # Assert all created correctly
        self.assertEqual(stats['people_created'], 3)
        self.assertEqual(stats['relationships_created'], 2)

        # Check relationships
        gen1 = Person.objects.get(genealogical_id="I.1.a")
        gen2 = Person.objects.get(genealogical_id="II.1.a")
        gen3 = Person.objects.get(genealogical_id="III.5.a")

        # Check I.1.a → II.1.a relationship
        self.assertTrue(
            Relationship.objects.filter(parent=gen1, child=gen2).exists()
        )

        # Check II.1.a → III.5.a relationship
        self.assertTrue(
            Relationship.objects.filter(parent=gen2, child=gen3).exists()
        )

    def test_siblings_share_parents(self):
        """Test siblings (same family group) share same parent relationships"""
        # Create parent
        TextChunk.objects.create(
            document=self.document,
            text_content="Jan van Zanten was the founder.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Jan van Zanten",
            genealogical_identifier="II.1.a",
            generation_number=2,
        )

        # Create 3 siblings with same family_group
        for i, name in enumerate(["Pieter", "Maria", "Hendrik"]):
            TextChunk.objects.create(
                document=self.document,
                text_content=f"{name} van Zanten was born.",
                start_page=2,
                end_page=2,
                sequence_number=i + 1,
                subject=f"{name} van Zanten",
                genealogical_identifier=f"III.5.{chr(97 + i)}",  # a, b, c
                generation_number=3,
                family_groups=["III.5. Kinderen van Jan van Zanten (II.1.a)"],
            )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        stats = builder.build()

        # Assert all created
        self.assertEqual(stats['people_created'], 4)  # 1 parent + 3 children
        self.assertEqual(stats['relationships_created'], 3)  # 3 parent-child relationships

        # Check all siblings have relationship to parent
        parent = Person.objects.get(genealogical_id="II.1.a")

        for letter in ['a', 'b', 'c']:
            child = Person.objects.get(genealogical_id=f"III.5.{letter}")
            self.assertTrue(
                Relationship.objects.filter(parent=parent, child=child).exists()
            )

    def test_no_chunks_with_genealogical_ids(self):
        """Test task returns empty stats when no chunks with genealogical IDs"""
        # Create chunk without genealogical_identifier
        TextChunk.objects.create(
            document=self.document,
            text_content="Some random text.",
            start_page=1,
            end_page=1,
            sequence_number=0,
        )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        stats = builder.build()

        # Assert nothing created
        self.assertEqual(stats['people_created'], 0)
        self.assertEqual(stats['relationships_created'], 0)
        self.assertEqual(stats['partnerships_created'], 0)

    def test_updates_person_when_name_changes(self):
        """Test Person is updated when chunk has different name for same genealogical_id"""
        # Create initial chunk
        chunk1 = TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            generation_number=2,
        )

        # Run task first time
        builder1 = GenealogyGraphBuilder(self.document)
        stats1 = builder1.build()
        self.assertEqual(stats1['people_created'], 1)

        # Update chunk with corrected name
        chunk1.subject = "Pieter Cornelis van Zanten"
        chunk1.save()

        # Run task second time
        builder2 = GenealogyGraphBuilder(self.document)
        stats2 = builder2.build()
        self.assertEqual(stats2['people_created'], 0)
        self.assertEqual(stats2['people_updated'], 1)

        # Check person was updated
        person = Person.objects.get(genealogical_id="II.3.a")
        self.assertEqual(person.given_names, "Pieter Cornelis")
        self.assertEqual(person.surname, "van Zanten")

    def test_celery_task_wrapper(self):
        """Test the Celery task wrapper works correctly"""
        # Create chunk
        TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Pieter van Zanten",
            genealogical_identifier="II.3.a",
            generation_number=2,
        )

        # Call the Celery task directly (not via .delay())
        stats = build_genealogy_graph(str(self.document.id))

        # Assert it worked
        self.assertEqual(stats['people_created'], 1)
        self.assertEqual(Person.objects.count(), 1)

    def test_partnership_created_with_minted_spouse_id(self):
        """Test Partnership is created with minted ID when second parent doesn't have explicit genealogical ID"""
        # Create only first parent
        TextChunk.objects.create(
            document=self.document,
            text_content="Jan van Zanten was the founder.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Jan van Zanten",
            genealogical_identifier="II.1.a",
            generation_number=2,
        )

        # Create child mentioning both parents, but second parent has no genealogical ID
        TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=2,
            end_page=2,
            sequence_number=1,
            subject="Pieter van Zanten",
            genealogical_identifier="III.5.a",
            generation_number=3,
            family_groups=["III.5. Kinderen van Jan van Zanten en Unknown Person (II.1.a)"],
        )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        stats = builder.build()

        # Assert partnership IS created with minted spouse ID
        # We should acknowledge that someone existed, even if name is "Unknown Person"
        self.assertEqual(stats['partnerships_created'], 1)
        self.assertEqual(Partnership.objects.count(), 1)

        # Verify the spouse was created with a minted ID
        self.assertEqual(stats['people_created'], 3)  # Jan, Pieter, Unknown Person (minted)

        # Check that Unknown Person has a minted ID
        unknown_spouse = Person.objects.get(given_names="Unknown", surname="Person")
        self.assertEqual(unknown_spouse.genealogical_id, "II.1.a.spouse1")
        self.assertEqual(unknown_spouse.generation, 2)

    def test_relationship_not_created_when_parent_missing(self):
        """Test Relationship is not created when parent genealogical ID doesn't exist"""
        # Create child chunk referencing non-existent parent
        TextChunk.objects.create(
            document=self.document,
            text_content="Pieter van Zanten was born in 1850.",
            start_page=1,
            end_page=1,
            sequence_number=0,
            subject="Pieter van Zanten",
            genealogical_identifier="III.5.a",
            generation_number=3,
            family_groups=["III.5. Kinderen van Unknown Parent (II.99.z)"],
        )

        # Run task
        builder = GenealogyGraphBuilder(self.document)
        stats = builder.build()

        # Assert relationship NOT created
        self.assertEqual(stats['relationships_created'], 0)
        self.assertEqual(Relationship.objects.count(), 0)
