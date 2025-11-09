"""
Tests for GenealogyTools service - agentic workflow tools.
"""

import pytest
from django.test import TestCase
from genealogy.models import (
    Identity,
    MentionToIdentity,
    PersonMention,
    Event,
    Place,
    RelationshipMention,
    PartnershipMention,
)
from genealogy.services.genealogy_tools import GenealogyTools


@pytest.mark.django_db
class TestGenealogyTools(TestCase):
    """Test the genealogy tools for agentic workflows"""

    def setUp(self):
        """Set up test data"""
        self.tools = GenealogyTools()

        # Create places
        self.amsterdam = Place.objects.create(name="Amsterdam")
        self.rotterdam = Place.objects.create(name="Rotterdam")

        # Create identities
        self.pieter_identity = Identity.objects.create(
            display_name="Pieter van Zanten",
            genealogical_identifier="II.3.a"
        )

        self.maria_identity = Identity.objects.create(
            display_name="Maria de Vries",
            genealogical_identifier="II.3.b"
        )

        self.anna_identity = Identity.objects.create(
            display_name="Anna van Zanten",
            genealogical_identifier="III.5.a"
        )

        # Create person mentions
        self.pieter_mention = PersonMention.objects.create(
            given_names="Pieter",
            surname="van Zanten",
            genealogical_id="II.3.a"
        )

        self.maria_mention = PersonMention.objects.create(
            given_names="Maria",
            surname="de Vries",
            genealogical_id="II.3.b"
        )

        self.anna_mention = PersonMention.objects.create(
            given_names="Anna",
            surname="van Zanten",
            genealogical_id="III.5.a"
        )

        # Link mentions to identities
        MentionToIdentity.objects.create(
            mention=self.pieter_mention,
            identity=self.pieter_identity,
            mapped_by="test"
        )

        MentionToIdentity.objects.create(
            mention=self.maria_mention,
            identity=self.maria_identity,
            mapped_by="test"
        )

        MentionToIdentity.objects.create(
            mention=self.anna_mention,
            identity=self.anna_identity,
            mapped_by="test"
        )

        # Create events
        Event.objects.create(
            mention=self.pieter_mention,
            event_type="BIRT",
            date="1845-03-12",
            place=self.amsterdam
        )

        Event.objects.create(
            mention=self.pieter_mention,
            event_type="DEAT",
            date="1920-11-03",
            place=self.rotterdam
        )

        Event.objects.create(
            mention=self.anna_mention,
            event_type="BIRT",
            date="1870-06-15",
            place=self.rotterdam
        )

        # Create relationships (Anna is child of Pieter and Maria)
        RelationshipMention.objects.create(
            child_mention=self.anna_mention,
            parent_mention=self.pieter_mention,
            relationship_type="BIOLOGICAL"
        )

        RelationshipMention.objects.create(
            child_mention=self.anna_mention,
            parent_mention=self.maria_mention,
            relationship_type="BIOLOGICAL"
        )

        # Create partnership (Pieter married to Maria)
        partnership = PartnershipMention.objects.create(
            partnership_type="MARR",
            start_date="1869-01-15"
        )
        partnership.partners.add(self.pieter_mention, self.maria_mention)

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

    def test_get_person_details_by_id(self):
        """Test getting person details by identity UUID"""
        result = self.tools.get_person_details(str(self.pieter_identity.id))

        self.assertEqual(result["display_name"], "Pieter van Zanten")
        self.assertEqual(result["genealogical_id"], "II.3.a")
        self.assertEqual(len(result["events"]), 2)  # birth and death

        # Check events
        event_types = [e["type"] for e in result["events"]]
        self.assertIn("Birth", event_types)
        self.assertIn("Death", event_types)

        # Check children
        self.assertEqual(len(result["children"]), 1)
        self.assertEqual(result["children"][0]["name"], "Anna van Zanten")

        # Check partners
        self.assertEqual(len(result["partners"]), 1)
        self.assertEqual(result["partners"][0]["name"], "Maria de Vries")

    def test_get_person_details_by_genealogical_id(self):
        """Test getting person details by genealogical identifier"""
        result = self.tools.get_person_details("II.3.a")

        self.assertEqual(result["display_name"], "Pieter van Zanten")
        self.assertEqual(result["genealogical_id"], "II.3.a")

    def test_get_person_details_not_found(self):
        """Test getting details for non-existent person"""
        result = self.tools.get_person_details("nonexistent-id")

        self.assertIn("error", result)
        self.assertIn("not found", result["error"].lower())

    def test_search_by_birth_year(self):
        """Test searching by birth year range"""
        # Search for people born in 1845
        result = self.tools.search_by_birth_year(
            "van Zanten",
            birth_year_min=1845,
            birth_year_max=1845
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["people"][0]["display_name"], "Pieter van Zanten")

        # Search for people born in 1870
        result = self.tools.search_by_birth_year(
            "van Zanten",
            birth_year_min=1870,
            birth_year_max=1870
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["people"][0]["display_name"], "Anna van Zanten")

        # Search for people born between 1845-1880 (should find both)
        result = self.tools.search_by_birth_year(
            "van Zanten",
            birth_year_min=1845,
            birth_year_max=1880
        )

        self.assertEqual(result["count"], 2)

    def test_get_children(self):
        """Test getting children of a person"""
        result = self.tools.get_children(str(self.pieter_identity.id))

        self.assertEqual(result["person"]["name"], "Pieter van Zanten")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["children"][0]["name"], "Anna van Zanten")
        self.assertEqual(result["children"][0]["birth_year"], 1870)

    def test_get_children_by_genealogical_id(self):
        """Test getting children using genealogical ID"""
        result = self.tools.get_children("II.3.a")

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["children"][0]["name"], "Anna van Zanten")

    def test_get_parents(self):
        """Test getting parents of a person"""
        result = self.tools.get_parents(str(self.anna_identity.id))

        self.assertEqual(result["person"]["name"], "Anna van Zanten")
        self.assertEqual(result["count"], 2)

        parent_names = [p["name"] for p in result["parents"]]
        self.assertIn("Pieter van Zanten", parent_names)
        self.assertIn("Maria de Vries", parent_names)

    def test_max_results_limit(self):
        """Test that max_results is enforced"""
        # Create 25 identities
        for i in range(25):
            identity = Identity.objects.create(
                display_name=f"Test Person {i}"
            )
            mention = PersonMention.objects.create(
                given_names="Test",
                surname=f"Person{i}"
            )
            MentionToIdentity.objects.create(
                mention=mention,
                identity=identity,
                mapped_by="test"
            )

        # Request 30, should get max 20
        result = self.tools.search_person_by_name("Test", max_results=30)

        self.assertEqual(result["count"], 20)
        self.assertTrue(result["truncated"])

    def test_person_with_no_events(self):
        """Test getting details for person with no events"""
        # Create identity with no events
        identity = Identity.objects.create(display_name="No Events Person")
        mention = PersonMention.objects.create(
            given_names="No Events",
            surname="Person"
        )
        MentionToIdentity.objects.create(
            mention=mention,
            identity=identity,
            mapped_by="test"
        )

        result = self.tools.get_person_details(str(identity.id))

        self.assertEqual(result["display_name"], "No Events Person")
        self.assertEqual(len(result["events"]), 0)
        self.assertEqual(len(result["children"]), 0)
        self.assertEqual(len(result["parents"]), 0)

    def test_deleted_identities_excluded(self):
        """Test that deleted identities are excluded from search"""
        # Mark Pieter as deleted
        self.pieter_identity.is_deleted = True
        self.pieter_identity.save()

        result = self.tools.search_person_by_name("Pieter", max_results=10)

        self.assertEqual(result["count"], 0)

        # Should also not be found by ID
        result = self.tools.get_person_details(str(self.pieter_identity.id))
        self.assertIn("error", result)
