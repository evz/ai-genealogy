"""Tests for clustering modules"""
from django.test import TestCase

from genealogy.clustering.nodes import AtomicNode, RelationalNode
from genealogy.clustering.person_record import PersonRecord
from genealogy.models import (Event, PartnershipMention, PersonMention, Place,
                              RelationshipMention)


class TestAtomicNode(TestCase):
    """Test AtomicNode dataclass"""

    def test_creates_atomic_node(self):
        """Should create atomic node with attributes"""
        node = AtomicNode(
            person_id=1,
            attribute="given_names",
            value="Pieter"
        )

        self.assertEqual(node.person_id, 1)
        self.assertEqual(node.attribute, "given_names")
        self.assertEqual(node.value, "Pieter")

    def test_hash_is_consistent(self):
        """Should have consistent hash for same values"""
        node1 = AtomicNode(person_id=1, attribute="given_names", value="Pieter")
        node2 = AtomicNode(person_id=1, attribute="given_names", value="Pieter")

        self.assertEqual(hash(node1), hash(node2))

    def test_equality(self):
        """Should be equal for same person_id, attribute, value"""
        node1 = AtomicNode(person_id=1, attribute="given_names", value="Pieter")
        node2 = AtomicNode(person_id=1, attribute="given_names", value="Pieter")

        self.assertEqual(node1, node2)

    def test_not_equal_different_values(self):
        """Should not be equal for different values"""
        node1 = AtomicNode(person_id=1, attribute="given_names", value="Pieter")
        node2 = AtomicNode(person_id=1, attribute="given_names", value="Jan")

        self.assertNotEqual(node1, node2)

    def test_not_equal_different_attributes(self):
        """Should not be equal for different attributes"""
        node1 = AtomicNode(person_id=1, attribute="given_names", value="Pieter")
        node2 = AtomicNode(person_id=1, attribute="surname", value="Pieter")

        self.assertNotEqual(node1, node2)

    def test_not_equal_different_person_ids(self):
        """Should not be equal for different person IDs"""
        node1 = AtomicNode(person_id=1, attribute="given_names", value="Pieter")
        node2 = AtomicNode(person_id=2, attribute="given_names", value="Pieter")

        self.assertNotEqual(node1, node2)


class TestRelationalNode(TestCase):
    """Test RelationalNode dataclass"""

    def test_creates_relational_node(self):
        """Should create relational node with default values"""
        node = RelationalNode(person1_id=1, person2_id=2)

        self.assertEqual(node.person1_id, 1)
        self.assertEqual(node.person2_id, 2)
        self.assertEqual(node.similarity, 0.0)
        self.assertEqual(node.atomic_sims, {})
        self.assertTrue(node.constraints_valid)
        self.assertEqual(node.constraint_violations, [])
        self.assertEqual(node.supporting_matches, set())

    def test_creates_with_similarity(self):
        """Should create with custom similarity score"""
        node = RelationalNode(person1_id=1, person2_id=2, similarity=0.85)

        self.assertEqual(node.similarity, 0.85)

    def test_hash_is_order_independent(self):
        """Should have same hash regardless of person order"""
        node1 = RelationalNode(person1_id=1, person2_id=2)
        node2 = RelationalNode(person1_id=2, person2_id=1)

        self.assertEqual(hash(node1), hash(node2))

    def test_equality_is_order_independent(self):
        """Should be equal regardless of person order"""
        node1 = RelationalNode(person1_id=1, person2_id=2)
        node2 = RelationalNode(person1_id=2, person2_id=1)

        self.assertEqual(node1, node2)

    def test_not_equal_different_persons(self):
        """Should not be equal for different person pairs"""
        node1 = RelationalNode(person1_id=1, person2_id=2)
        node2 = RelationalNode(person1_id=1, person2_id=3)

        self.assertNotEqual(node1, node2)

    def test_stores_atomic_similarities(self):
        """Should store atomic attribute similarities"""
        node = RelationalNode(person1_id=1, person2_id=2)
        node.atomic_sims = {
            'given_names': 0.9,
            'surname': 1.0,
            'birth_year': 0.8
        }

        self.assertEqual(node.atomic_sims['given_names'], 0.9)
        self.assertEqual(len(node.atomic_sims), 3)

    def test_tracks_constraint_violations(self):
        """Should track constraint violations"""
        node = RelationalNode(person1_id=1, person2_id=2)
        node.constraints_valid = False
        node.constraint_violations = ['birth_year_diff_too_large', 'same_parent']

        self.assertFalse(node.constraints_valid)
        self.assertEqual(len(node.constraint_violations), 2)

    def test_tracks_supporting_matches(self):
        """Should track supporting relational matches"""
        node = RelationalNode(person1_id=1, person2_id=2)
        node.supporting_matches = {(3, 4), (5, 6)}

        self.assertEqual(len(node.supporting_matches), 2)
        self.assertIn((3, 4), node.supporting_matches)


class TestPersonRecord(TestCase):
    """Test PersonRecord wrapper"""

    def setUp(self):
        """Create test person mention"""
        self.person = PersonMention.objects.create(
            given_names="Pieter",
            surname="Jansen",
            generation=2,
        )

    def test_creates_person_record(self):
        """Should create person record from mention"""
        record = PersonRecord(self.person)

        self.assertEqual(record.id, self.person.id)
        self.assertEqual(record.given_names, "Pieter")
        self.assertEqual(record.surname, "Jansen")
        self.assertEqual(record.generation, 2)

    def test_extracts_birth_event(self):
        """Should extract birth date and place"""
        place = Place.objects.create(name="Amsterdam")
        Event.objects.create(
            mention=self.person,
            event_type="BIRT",
            date="1850-01-15",
            place=place,
        )

        record = PersonRecord(self.person)

        self.assertEqual(str(record.birth_date), "1850-01-15")
        self.assertEqual(record.birth_place, "Amsterdam")
        self.assertEqual(record.birth_year, 1850)

    def test_extracts_death_event(self):
        """Should extract death date and place"""
        place = Place.objects.create(name="Rotterdam")
        Event.objects.create(
            mention=self.person,
            event_type="DEAT",
            date="1920-05-20",
            place=place,
        )

        record = PersonRecord(self.person)

        self.assertEqual(str(record.death_date), "1920-05-20")
        self.assertEqual(record.death_place, "Rotterdam")
        self.assertEqual(record.death_year, 1920)

    def test_handles_events_without_place(self):
        """Should handle events without place"""
        Event.objects.create(
            mention=self.person,
            event_type="BIRT",
            date="1850-01-15",
            place=None,
        )

        record = PersonRecord(self.person)

        self.assertIsNone(record.birth_place)

    def test_handles_events_without_date(self):
        """Should handle events without date"""
        Event.objects.create(
            mention=self.person,
            event_type="BIRT",
            date=None,
        )

        record = PersonRecord(self.person)

        self.assertIsNone(record.birth_date)
        self.assertIsNone(record.birth_year)

    def test_extracts_parent_relationships(self):
        """Should extract parent IDs and names"""
        parent = PersonMention.objects.create(
            given_names="Jan",
            surname="Jansen",
        )

        RelationshipMention.objects.create(
            parent_mention=parent,
            child_mention=self.person,
            relationship_type="biological",
        )

        record = PersonRecord(self.person)

        self.assertIn(parent.id, record.parent_ids)
        self.assertEqual(len(record.parent_names), 1)

    def test_extracts_child_relationships(self):
        """Should extract child IDs and names"""
        child = PersonMention.objects.create(
            given_names="Klaas",
            surname="Jansen",
        )

        RelationshipMention.objects.create(
            parent_mention=self.person,
            child_mention=child,
            relationship_type="biological",
        )

        record = PersonRecord(self.person)

        self.assertIn(child.id, record.child_ids)
        self.assertEqual(len(record.child_names), 1)

    def test_extracts_spouse_relationships(self):
        """Should extract spouse IDs and names"""
        spouse = PersonMention.objects.create(
            given_names="Maria",
            surname="Pietersen",
        )

        partnership = PartnershipMention.objects.create(
            partnership_type="marriage",
        )
        partnership.partners.add(self.person, spouse)

        record = PersonRecord(self.person)

        self.assertIn(spouse.id, record.spouse_ids)
        self.assertEqual(len(record.spouse_names), 1)

    def test_initializes_empty_relationship_sets(self):
        """Should initialize empty relationship sets"""
        record = PersonRecord(self.person)

        self.assertEqual(len(record.parent_ids), 0)
        self.assertEqual(len(record.child_ids), 0)
        self.assertEqual(len(record.spouse_ids), 0)
        self.assertEqual(len(record.parent_names), 0)
        self.assertEqual(len(record.child_names), 0)
        self.assertEqual(len(record.spouse_names), 0)

    def test_handles_multiple_parents(self):
        """Should handle multiple parent relationships"""
        father = PersonMention.objects.create(given_names="Jan", surname="Jansen")
        mother = PersonMention.objects.create(given_names="Maria", surname="Pietersen")

        RelationshipMention.objects.create(
            parent_mention=father,
            child_mention=self.person,
            relationship_type="biological",
        )
        RelationshipMention.objects.create(
            parent_mention=mother,
            child_mention=self.person,
            relationship_type="biological",
        )

        record = PersonRecord(self.person)

        self.assertEqual(len(record.parent_ids), 2)
        self.assertEqual(len(record.parent_names), 2)
