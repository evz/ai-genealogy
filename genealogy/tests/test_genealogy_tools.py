"""
Tests for GenealogyTools service - agentic workflow tools.
"""

import pytest
from django.test import TestCase

from genealogy.models import Event, Partnership, Person, Relationship
from genealogy.services.genealogy_tools import GenealogyTools


@pytest.mark.django_db
class TestGenealogyTools(TestCase):
    """Test the genealogy tools for agentic workflows"""

    def setUp(self):
        """Set up test data"""
        self.tools = GenealogyTools()

        # Create people
        self.pieter = Person.objects.create(
            genealogical_id="II.3.a",
            given_names="Pieter",
            surname="van Zanten",
            generation=2
        )

        self.maria = Person.objects.create(
            genealogical_id="II.3.b",
            given_names="Maria",
            surname="de Vries",
            generation=2
        )

        self.anna = Person.objects.create(
            genealogical_id="III.5.a",
            given_names="Anna",
            surname="van Zanten",
            generation=3
        )

        # Create events (using date objects now)
        from datetime import date

        Event.objects.create(
            person=self.pieter,
            event_type="BIRT",
            date=date(1845, 3, 12),
            date_original="1845-03-12",
            place="Amsterdam"
        )

        Event.objects.create(
            person=self.pieter,
            event_type="DEAT",
            date=date(1920, 11, 3),
            date_original="1920-11-03",
            place="Rotterdam"
        )

        Event.objects.create(
            person=self.anna,
            event_type="BIRT",
            date=date(1870, 6, 15),
            date_original="1870-06-15",
            place="Rotterdam"
        )

        # Create relationships (Anna is child of Pieter and Maria)
        Relationship.objects.create(
            parent=self.pieter,
            child=self.anna,
            relationship_type="BIOLOGICAL"
        )

        Relationship.objects.create(
            parent=self.maria,
            child=self.anna,
            relationship_type="BIOLOGICAL"
        )

        # Create partnership (Pieter married to Maria)
        Partnership.objects.create(
            partner1=self.pieter,
            partner2=self.maria,
            partnership_type="MARRIAGE"
        )

    def test_search_person_by_name(self):
        """Test searching for people by name"""
        result = self.tools.search_person_by_name("Pieter", max_results=10)

        self.assertEqual(result["count"], 1)
        self.assertFalse(result["truncated"])

        person = result["people"][0]
        self.assertEqual(person["display_name"], "Pieter van Zanten")
        self.assertEqual(person["genealogical_id"], "II.3.a")
        self.assertEqual(person["birth"]["date"], "1845-03-12")
        self.assertEqual(person["birth"]["place"], "Amsterdam")
        self.assertEqual(person["death"]["date"], "1920-11-03")
        self.assertEqual(person["death"]["place"], "Rotterdam")

    def test_search_person_by_partial_name(self):
        """Test searching with partial name match"""
        result = self.tools.search_person_by_name("van Zanten", max_results=10)

        # Should find both Pieter and Anna van Zanten
        self.assertEqual(result["count"], 2)
        names = [p["display_name"] for p in result["people"]]
        self.assertIn("Pieter van Zanten", names)
        self.assertIn("Anna van Zanten", names)

    def test_search_with_max_results_limit(self):
        """Test that max_results parameter limits output"""
        result = self.tools.search_person_by_name("van Zanten", max_results=1)

        self.assertEqual(result["count"], 1)
        self.assertTrue(result["truncated"])

    def test_get_person_details_by_genealogical_id(self):
        """Test getting person details by genealogical ID"""
        result = self.tools.get_person_details("II.3.a")

        self.assertEqual(result["display_name"], "Pieter van Zanten")
        self.assertEqual(result["genealogical_id"], "II.3.a")

        # Check events
        self.assertEqual(len(result["events"]), 2)
        event_types = [e["type"] for e in result["events"]]
        # Event types use GEDCOM codes (BIRT, DEAT) which display as "Birth", "Death"
        self.assertIn("Birth", event_types)
        self.assertIn("Death", event_types)

        # Check children
        self.assertEqual(len(result["children"]), 1)
        self.assertEqual(result["children"][0]["name"], "Anna van Zanten")
        self.assertEqual(result["children"][0]["genealogical_id"], "III.5.a")

        # Check partners
        self.assertEqual(len(result["partners"]), 1)
        self.assertEqual(result["partners"][0]["name"], "Maria de Vries")

    def test_get_person_details_by_uuid(self):
        """Test getting person details by UUID"""
        result = self.tools.get_person_details(str(self.pieter.id))

        self.assertEqual(result["display_name"], "Pieter van Zanten")
        self.assertEqual(result["genealogical_id"], "II.3.a")

    def test_get_person_details_not_found(self):
        """Test error handling when person not found"""
        result = self.tools.get_person_details("NONEXISTENT")

        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    def test_get_children(self):
        """Test getting children of a person"""
        result = self.tools.get_children("II.3.a")

        self.assertEqual(result["person"]["name"], "Pieter van Zanten")
        self.assertEqual(result["count"], 1)
        self.assertEqual(len(result["children"]), 1)
        self.assertEqual(result["children"][0]["name"], "Anna van Zanten")
        self.assertEqual(result["children"][0]["genealogical_id"], "III.5.a")
        self.assertEqual(result["children"][0]["birth_year"], 1870)

    def test_get_children_none(self):
        """Test getting children when person has no children"""
        result = self.tools.get_children("III.5.a")

        self.assertEqual(result["count"], 0)
        self.assertEqual(len(result["children"]), 0)

    def test_get_parents(self):
        """Test getting parents of a person"""
        result = self.tools.get_parents("III.5.a")

        self.assertEqual(result["person"]["name"], "Anna van Zanten")
        self.assertEqual(result["count"], 2)

        parent_names = [p["name"] for p in result["parents"]]
        self.assertIn("Pieter van Zanten", parent_names)
        self.assertIn("Maria de Vries", parent_names)

    def test_get_parents_none(self):
        """Test getting parents when person has no recorded parents"""
        result = self.tools.get_parents("II.3.a")

        self.assertEqual(result["count"], 0)
        self.assertEqual(len(result["parents"]), 0)

    def test_search_by_birth_year(self):
        """Test searching by birth year range"""
        result = self.tools.search_by_birth_year(
            name="van Zanten",
            birth_year_min=1840,
            birth_year_max=1850
        )

        # Should find only Pieter (born 1845)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["people"][0]["display_name"], "Pieter van Zanten")
        # Birth year should be 1845
        self.assertEqual(result["people"][0]["birth"]["date"], "1845-03-12")

    def test_search_by_birth_year_no_match(self):
        """Test searching with birth year that doesn't match"""
        result = self.tools.search_by_birth_year(
            name="van Zanten",
            birth_year_min=1900,
            birth_year_max=1950
        )

        # Should find nobody
        self.assertEqual(result["count"], 0)

    def test_search_includes_parents(self):
        """Test that search results include parent information"""
        result = self.tools.search_person_by_name("Anna", max_results=10)

        self.assertEqual(result["count"], 1)
        person = result["people"][0]
        self.assertEqual(len(person["parents"]), 2)
        parent_names = person["parents"]
        self.assertIn("Pieter van Zanten", parent_names)
        self.assertIn("Maria de Vries", parent_names)


@pytest.mark.django_db
class TestFuzzyNameMatching(TestCase):
    """Test fuzzy name matching variations for Dutch surnames"""

    def setUp(self):
        """Set up test data with various name spellings"""
        self.tools = GenealogyTools()

        # Create people with various Dutch surname spellings
        self.pieter_spaced = Person.objects.create(
            genealogical_id="II.1.a",
            given_names="Pieter",
            surname="van Zanten",
            generation=2
        )

        self.jan_nospace = Person.objects.create(
            genealogical_id="II.1.b",
            given_names="Jan",
            surname="VanZanten",  # No space, capitalized
            generation=2
        )

        self.willem_lower = Person.objects.create(
            genealogical_id="II.1.c",
            given_names="Willem",
            surname="vanzanten",  # No space, lowercase
            generation=2
        )

        self.maria_de = Person.objects.create(
            genealogical_id="II.2.a",
            given_names="Maria",
            surname="de Vries",
            generation=2
        )

        self.anna_der = Person.objects.create(
            genealogical_id="II.3.a",
            given_names="Anna",
            surname="van der Berg",
            generation=2
        )

    def test_generate_name_variations_van_zanten(self):
        """Test name variation generation for 'van Zanten'"""
        variations = self.tools._generate_name_variations("van Zanten")

        # Should generate both spaced and non-spaced versions
        self.assertIn("van zanten", variations)
        self.assertIn("vanzanten", variations)
        self.assertEqual(len(variations), 2)

    def test_generate_name_variations_vanzanten(self):
        """Test name variation generation for 'vanzanten' (no space)"""
        variations = self.tools._generate_name_variations("vanzanten")

        # Should generate both versions
        self.assertIn("vanzanten", variations)
        self.assertIn("van zanten", variations)
        self.assertEqual(len(variations), 2)

    def test_generate_name_variations_de_vries(self):
        """Test name variation generation for 'de Vries'"""
        variations = self.tools._generate_name_variations("de Vries")

        self.assertIn("de vries", variations)
        self.assertIn("devries", variations)

    def test_generate_name_variations_van_der_berg(self):
        """Test name variation generation for 'van der Berg'"""
        variations = self.tools._generate_name_variations("van der Berg")

        # Should generate multiple variations
        self.assertIn("van der berg", variations)
        self.assertIn("vanderberg", variations)
        # Should also handle partial space removal
        self.assertTrue(any("vander" in v for v in variations))

    def test_generate_name_variations_full_name(self):
        """Test variation generation for full name with prefix"""
        variations = self.tools._generate_name_variations("Bessel van Zanten")

        # Should include variations
        self.assertIn("bessel van zanten", variations)
        self.assertIn("besselvanzanten", variations)
        self.assertIn("bessel vanzanten", variations)

    def test_search_finds_spaced_spelling(self):
        """Test that searching 'van Zanten' finds person with spaced surname"""
        result = self.tools.search_person_by_name("van Zanten", max_results=10)

        # Should find Pieter with "van Zanten"
        names = [p["display_name"] for p in result["people"]]
        self.assertIn("Pieter van Zanten", names)

    def test_search_finds_nospace_spelling(self):
        """Test that searching 'vanzanten' finds person with no-space surname"""
        result = self.tools.search_person_by_name("vanzanten", max_results=10)

        # Should find Jan with "VanZanten"
        names = [p["display_name"] for p in result["people"]]
        self.assertIn("Jan VanZanten", names)

    def test_search_spaced_finds_nospace(self):
        """Test that searching 'van Zanten' (spaced) also finds 'VanZanten' (no space)"""
        result = self.tools.search_person_by_name("van Zanten", max_results=10)

        # Should find ALL variations
        names = [p["display_name"] for p in result["people"]]
        self.assertIn("Pieter van Zanten", names)
        self.assertIn("Jan VanZanten", names)
        self.assertIn("Willem vanzanten", names)

        # Should find at least 3 people
        self.assertGreaterEqual(result["count"], 3)

    def test_search_nospace_finds_spaced(self):
        """Test that searching 'vanzanten' (no space) also finds 'van Zanten' (spaced)"""
        result = self.tools.search_person_by_name("vanzanten", max_results=10)

        # Should find ALL variations
        names = [p["display_name"] for p in result["people"]]
        self.assertIn("Pieter van Zanten", names)
        self.assertIn("Jan VanZanten", names)
        self.assertIn("Willem vanzanten", names)

        # Should find at least 3 people
        self.assertGreaterEqual(result["count"], 3)

    def test_search_equivalence(self):
        """Test that 'van Zanten' and 'vanzanten' return same results"""
        result1 = self.tools.search_person_by_name("van Zanten", max_results=10)
        result2 = self.tools.search_person_by_name("vanzanten", max_results=10)

        # Should find same number of people
        self.assertEqual(result1["count"], result2["count"])

        # Should find same people
        ids1 = set([p["id"] for p in result1["people"]])
        ids2 = set([p["id"] for p in result2["people"]])
        self.assertEqual(ids1, ids2)

    def test_search_de_vries_variations(self):
        """Test fuzzy matching for 'de Vries'"""
        result1 = self.tools.search_person_by_name("de Vries", max_results=10)
        result2 = self.tools.search_person_by_name("devries", max_results=10)

        # Both should find Maria
        names1 = [p["display_name"] for p in result1["people"]]
        names2 = [p["display_name"] for p in result2["people"]]

        self.assertIn("Maria de Vries", names1)
        self.assertIn("Maria de Vries", names2)

    def test_case_insensitive_search(self):
        """Test that search is case insensitive"""
        result1 = self.tools.search_person_by_name("VAN ZANTEN", max_results=10)
        result2 = self.tools.search_person_by_name("van zanten", max_results=10)
        result3 = self.tools.search_person_by_name("Van Zanten", max_results=10)

        # All should find same people
        self.assertEqual(result1["count"], result2["count"])
        self.assertEqual(result2["count"], result3["count"])

    def test_search_by_birth_year_with_fuzzy_name(self):
        """Test that birth year search also uses fuzzy matching"""
        from datetime import date

        # Add birth event to Jan VanZanten
        Event.objects.create(
            person=self.jan_nospace,
            event_type="BIRT",
            date=date(1850, 1, 1),
            date_original="1850-01-01"
        )

        # Search with spaced version for person with non-spaced surname
        result = self.tools.search_by_birth_year(
            name="van Zanten",
            birth_year_min=1849,
            birth_year_max=1851
        )

        # Should find Jan even though his surname is "VanZanten"
        names = [p["display_name"] for p in result["people"]]
        self.assertIn("Jan VanZanten", names)
