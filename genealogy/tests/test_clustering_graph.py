"""Tests for DependencyGraph clustering algorithm"""
from django.test import TestCase

from genealogy.clustering.graph import DependencyGraph
from genealogy.clustering.person_record import PersonRecord
from genealogy.models import (Event, PartnershipMention, PersonMention, Place,
                              RelationshipMention)


class TestDependencyGraph(TestCase):
    """Test the DependencyGraph class"""

    def setUp(self):
        """Create test persons"""
        self.person1 = PersonMention.objects.create(
            given_names="Pieter",
            surname="Jansen",
            generation=2,
        )

        self.person2 = PersonMention.objects.create(
            given_names="Pieter",
            surname="Jansen",
            generation=2,
        )

        self.graph = DependencyGraph()

    def test_add_person(self):
        """Should add person to graph and create atomic nodes"""
        record = PersonRecord(self.person1)
        self.graph.add_person(record)

        self.assertIn(self.person1.id, self.graph.persons)
        self.assertIn(self.person1.id, self.graph.atomic_nodes)
        # Should have atomic nodes for given_names and surname
        self.assertGreater(len(self.graph.atomic_nodes[self.person1.id]), 0)

    def test_calculate_atomic_similarity_exact_match(self):
        """Should return 1.0 for exact matches"""
        sim = self.graph.calculate_atomic_similarity('given_names', 'Pieter', 'Pieter')
        self.assertEqual(sim, 1.0)

    def test_calculate_atomic_similarity_none_values(self):
        """Should return 0.0 when either value is None"""
        sim = self.graph.calculate_atomic_similarity('given_names', None, 'Pieter')
        self.assertEqual(sim, 0.0)

        sim = self.graph.calculate_atomic_similarity('given_names', 'Pieter', None)
        self.assertEqual(sim, 0.0)

    def test_calculate_atomic_similarity_similar_strings(self):
        """Should calculate string similarity for similar names"""
        # Very similar names should have high similarity
        sim = self.graph.calculate_atomic_similarity('given_names', 'Pieter', 'Peter')
        self.assertGreater(sim, 0.7)

    def test_calculate_atomic_similarity_substring(self):
        """Should recognize substring matches"""
        # "John" in "John William" should get 0.85
        sim = self.graph.calculate_atomic_similarity('given_names', 'John', 'John William')
        self.assertEqual(sim, 0.85)

    def test_calculate_atomic_similarity_numeric(self):
        """Should calculate numeric similarity for years"""
        # Exact match
        sim = self.graph.calculate_atomic_similarity('birth_year', 1850, 1850)
        self.assertEqual(sim, 1.0)

        # Within tolerance (5 years) should be high
        sim = self.graph.calculate_atomic_similarity('birth_year', 1850, 1852)
        self.assertGreater(sim, 0.8)

        # Beyond tolerance should decay
        sim = self.graph.calculate_atomic_similarity('birth_year', 1850, 1870)
        self.assertLess(sim, 0.5)

    def test_calculate_disambiguation_weight(self):
        """Should calculate disambiguation weights based on frequency"""
        # Add multiple persons with different attributes
        for i in range(5):
            person = PersonMention.objects.create(
                given_names="Common",
                surname=f"Surname{i}",
                generation=i,
            )
            record = PersonRecord(person)
            self.graph.add_person(record)

        # Common given_names should have lower weight
        common_weight = self.graph.calculate_disambiguation_weight('given_names', 'Common')
        self.assertLess(common_weight, 1.0)

        # Rare surname should have higher weight
        rare_weight = self.graph.calculate_disambiguation_weight('surname', 'Surname0')
        self.assertGreater(rare_weight, common_weight)

    def test_calculate_relationship_overlap_empty_sets(self):
        """Should handle empty relationship sets"""
        overlap = self.graph.calculate_relationship_overlap(set(), set())
        self.assertEqual(overlap, 0.0)

        overlap = self.graph.calculate_relationship_overlap({'name1'}, set())
        self.assertEqual(overlap, 0.0)

    def test_calculate_relationship_overlap_jaccard(self):
        """Should calculate Jaccard similarity correctly"""
        set1 = {'jan', 'maria', 'hendrik'}
        set2 = {'jan', 'maria', 'klaas'}

        overlap = self.graph.calculate_relationship_overlap(set1, set2)
        # Intersection: 2 (jan, maria), Union: 4 (jan, maria, hendrik, klaas)
        self.assertEqual(overlap, 0.5)

    def test_calculate_overall_similarity_identical_names(self):
        """Should calculate high similarity for identical names"""
        record1 = PersonRecord(self.person1)
        record2 = PersonRecord(self.person2)

        self.graph.add_person(record1)
        self.graph.add_person(record2)

        rel_node = self.graph.calculate_overall_similarity(self.person1.id, self.person2.id)

        self.assertTrue(rel_node.constraints_valid)
        self.assertGreater(rel_node.similarity, 0.25)  # Should have decent similarity from matching names

    def test_calculate_overall_similarity_generation_mismatch(self):
        """Should fail generation constraint check"""
        person3 = PersonMention.objects.create(
            given_names="Pieter",
            surname="Jansen",
            generation=5,  # Different generation
        )

        record1 = PersonRecord(self.person1)
        record3 = PersonRecord(person3)

        self.graph.add_person(record1)
        self.graph.add_person(record3)

        rel_node = self.graph.calculate_overall_similarity(self.person1.id, person3.id)

        self.assertFalse(rel_node.constraints_valid)
        self.assertEqual(rel_node.similarity, 0.0)
        self.assertIn("Generation mismatch", rel_node.constraint_violations[0])

    def test_calculate_overall_similarity_with_birth_events(self):
        """Should incorporate birth year similarity"""
        place = Place.objects.create(name="Amsterdam")

        Event.objects.create(
            mention=self.person1,
            event_type="BIRT",
            date="1850-01-15",
            place=place,
        )

        Event.objects.create(
            mention=self.person2,
            event_type="BIRT",
            date="1851-03-20",
            place=place,
        )

        record1 = PersonRecord(self.person1)
        record2 = PersonRecord(self.person2)

        self.graph.add_person(record1)
        self.graph.add_person(record2)

        rel_node = self.graph.calculate_overall_similarity(self.person1.id, self.person2.id)

        # Should have birth_year and birth_place similarities
        self.assertIn('birth_year', rel_node.atomic_sims)
        self.assertIn('birth_place', rel_node.atomic_sims)
        self.assertGreater(rel_node.atomic_sims['birth_year'], 0.9)  # 1 year difference
        self.assertEqual(rel_node.atomic_sims['birth_place'], 1.0)  # Exact match

    def test_calculate_overall_similarity_with_spouse_overlap(self):
        """Should incorporate spouse relationship overlap"""
        # Create spouse for both persons
        spouse = PersonMention.objects.create(
            given_names="Maria",
            surname="Pietersen",
            generation=2,
        )

        # Create partnerships
        partnership1 = PartnershipMention.objects.create(partnership_type='MARRIAGE')
        partnership1.partners.add(self.person1, spouse)

        partnership2 = PartnershipMention.objects.create(partnership_type='MARRIAGE')
        partnership2.partners.add(self.person2, spouse)

        record1 = PersonRecord(self.person1)
        record2 = PersonRecord(self.person2)

        self.graph.add_person(record1)
        self.graph.add_person(record2)

        rel_node = self.graph.calculate_overall_similarity(self.person1.id, self.person2.id)

        # Should have spouse overlap
        self.assertIn('spouse_overlap', rel_node.atomic_sims)
        self.assertGreater(rel_node.atomic_sims['spouse_overlap'], 0.0)
        # Spouse overlap is heavily weighted, so similarity should be high
        self.assertGreater(rel_node.similarity, 0.5)

    def test_calculate_overall_similarity_sibling_detection(self):
        """Should detect siblings (shared parents, different names)"""
        # Create parents
        parent = PersonMention.objects.create(
            given_names="Jan",
            surname="Jansen",
            generation=1,
        )

        # Create two children with same parents but different given names
        child1 = PersonMention.objects.create(
            given_names="Pieter",
            surname="Jansen",
            generation=2,
        )

        child2 = PersonMention.objects.create(
            given_names="Klaas",  # Different given name
            surname="Jansen",
            generation=2,
        )

        # Link parent relationships
        RelationshipMention.objects.create(
            parent_mention=parent,
            child_mention=child1,
            relationship_type='BIOLOGICAL',
        )

        RelationshipMention.objects.create(
            parent_mention=parent,
            child_mention=child2,
            relationship_type='BIOLOGICAL',
        )

        record1 = PersonRecord(child1)
        record2 = PersonRecord(child2)

        self.graph.add_person(record1)
        self.graph.add_person(record2)

        rel_node = self.graph.calculate_overall_similarity(child1.id, child2.id)

        # Should detect as siblings and reject the match
        self.assertFalse(rel_node.constraints_valid)
        self.assertEqual(rel_node.similarity, 0.0)
        self.assertTrue(any('Siblings' in v for v in rel_node.constraint_violations))

    def test_calculate_overall_similarity_normalizes_by_max_possible(self):
        """Should normalize by maximum possible weight, not actual weight"""
        # Person with sparse data
        sparse_person = PersonMention.objects.create(
            given_names="Jan",
            surname="Doe",
            generation=1,
        )

        # Person with rich data
        rich_person = PersonMention.objects.create(
            given_names="Jan",
            surname="Doe",
            generation=1,
        )

        Event.objects.create(
            mention=rich_person,
            event_type="BIRT",
            date="1850-01-15",
        )

        Event.objects.create(
            mention=rich_person,
            event_type="DEAT",
            date="1920-01-15",
        )

        record_sparse = PersonRecord(sparse_person)
        record_rich = PersonRecord(rich_person)

        self.graph.add_person(record_sparse)
        self.graph.add_person(record_rich)

        rel_node = self.graph.calculate_overall_similarity(sparse_person.id, rich_person.id)

        # Similarity should be calculated based on available data
        # Names match perfectly, so even though we're missing birth/death for sparse person,
        # similarity should still be reasonable
        self.assertGreater(rel_node.similarity, 0.2)
        self.assertLess(rel_node.similarity, 1.0)

    def test_detect_partial_match_group(self):
        """Should detect partial match groups (siblings)"""
        # Create parents
        parent = PersonMention.objects.create(
            given_names="Jan",
            surname="Jansen",
            generation=1,
        )

        # Create siblings
        child1 = PersonMention.objects.create(
            given_names="Pieter",
            surname="Jansen",
            generation=2,
        )

        child2 = PersonMention.objects.create(
            given_names="Klaas",
            surname="Jansen",
            generation=2,
        )

        # Link relationships
        RelationshipMention.objects.create(
            parent_mention=parent,
            child_mention=child1,
            relationship_type='BIOLOGICAL',
        )

        RelationshipMention.objects.create(
            parent_mention=parent,
            child_mention=child2,
            relationship_type='BIOLOGICAL',
        )

        record1 = PersonRecord(child1)
        record2 = PersonRecord(child2)

        self.graph.add_person(record1)
        self.graph.add_person(record2)

        # Test detection
        cluster = {child1.id, child2.id}
        is_partial = self.graph.detect_partial_match_group(cluster)

        # Should detect as partial match (siblings with different names)
        self.assertTrue(is_partial)

    def test_detect_partial_match_group_same_names(self):
        """Should not detect partial match when names are identical"""
        # Create parents
        parent = PersonMention.objects.create(
            given_names="Jan",
            surname="Jansen",
            generation=1,
        )

        # Create two persons with same name (potential duplicates)
        child1 = PersonMention.objects.create(
            given_names="Pieter",
            surname="Jansen",
            generation=2,
        )

        child2 = PersonMention.objects.create(
            given_names="Pieter",
            surname="Jansen",
            generation=2,
        )

        # Link relationships
        RelationshipMention.objects.create(
            parent_mention=parent,
            child_mention=child1,
            relationship_type='BIOLOGICAL',
        )

        RelationshipMention.objects.create(
            parent_mention=parent,
            child_mention=child2,
            relationship_type='BIOLOGICAL',
        )

        record1 = PersonRecord(child1)
        record2 = PersonRecord(child2)

        self.graph.add_person(record1)
        self.graph.add_person(record2)

        # Test detection
        cluster = {child1.id, child2.id}
        is_partial = self.graph.detect_partial_match_group(cluster)

        # Should NOT detect as partial match (same names = likely duplicates)
        self.assertFalse(is_partial)
