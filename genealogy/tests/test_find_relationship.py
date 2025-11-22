"""Tests for find_relationship genealogy tool"""

import pytest
from genealogy.services.genealogy_tools import GenealogyTools
from genealogy.models import Person, Relationship


@pytest.mark.django_db
class TestFindRelationship:
    """Test the find_relationship tool for computing genealogical relationships"""

    @pytest.fixture
    def tools(self):
        """Create GenealogyTools instance"""
        return GenealogyTools()

    @pytest.fixture
    def test_family(self):
        """Create a test family tree for relationship testing

        Family structure:
            Aart (VI.1) [ancestor]
                |
            Bessel (VI.1.n) [parent]
                |
            +---+---+---+
            |   |   |   |
          Pieter VII.3.a   Bessel VII.3.c   Anna VII.3.d   Jan VII.3.e
            |   (siblings)
         Eugene VIII.3.a
        """
        # Create ancestor (Aart)
        aart = Person.objects.create(
            genealogical_id="VI.1",
            given_names="Aart",
            surname="van Zanten"
        )

        # Create parent (Bessel VI.1.n)
        bessel = Person.objects.create(
            genealogical_id="VI.1.n",
            given_names="Bessel",
            surname="van Zanten"
        )

        # Link Bessel to Aart
        Relationship.objects.create(parent=aart, child=bessel)

        # Create Bessel's children (siblings)
        pieter = Person.objects.create(
            genealogical_id="VII.3.a",
            given_names="Pieter",
            surname="van Zanten"
        )

        bessel_jr = Person.objects.create(
            genealogical_id="VII.3.c",
            given_names="Bessel",
            surname="van Zanten"
        )

        anna = Person.objects.create(
            genealogical_id="VII.3.d",
            given_names="Anna",
            surname="van Zanten"
        )

        jan = Person.objects.create(
            genealogical_id="VII.3.e",
            given_names="Jan",
            surname="van Zanten"
        )

        # Link children to Bessel
        Relationship.objects.create(parent=bessel, child=pieter)
        Relationship.objects.create(parent=bessel, child=bessel_jr)
        Relationship.objects.create(parent=bessel, child=anna)
        Relationship.objects.create(parent=bessel, child=jan)

        # Create grandchild (Eugene)
        eugene = Person.objects.create(
            genealogical_id="VIII.3.a",
            given_names="Eugene",
            surname="van Zanten"
        )

        # Link Eugene to Pieter
        Relationship.objects.create(parent=pieter, child=eugene)

        return {
            "aart": aart,
            "bessel": bessel,
            "pieter": pieter,
            "bessel_jr": bessel_jr,
            "anna": anna,
            "jan": jan,
            "eugene": eugene
        }

    def test_find_relationship_parent_child(self, tools, test_family):
        """
        Test parent-child relationship detection.

        VI.1.n (Bessel) is the parent of VII.3.a (Pieter).
        """
        result = tools.find_relationship("VI.1.n", "VII.3.a")

        assert result["relationship_type"] == "ancestor"
        assert result["relationship"] == "parent"
        assert result["generations_from_person_1"] == 0
        assert result["generations_from_person_2"] == 1

    def test_find_relationship_child_parent(self, tools, test_family):
        """
        Test child-parent relationship (reverse of parent-child).

        VII.3.a (Pieter) is the child of VI.1.n (Bessel).
        """
        result = tools.find_relationship("VII.3.a", "VI.1.n")

        assert result["relationship_type"] == "descendant"
        assert result["relationship"] == "child"
        assert result["generations_from_person_1"] == 1
        assert result["generations_from_person_2"] == 0

    def test_find_relationship_siblings(self, tools, test_family):
        """
        Test sibling relationship detection.

        VII.3.a (Pieter) and VII.3.c (Bessel Jr) are siblings
        (both children of VI.1.n).
        """
        result = tools.find_relationship("VII.3.a", "VII.3.c")

        assert result["relationship_type"] == "sibling"
        assert result["relationship"] == "sibling"
        assert result["generations_from_person_1"] == 1
        assert result["generations_from_person_2"] == 1
        assert "common_ancestor" in result
        assert result["common_ancestor"]["genealogical_id"] == "VI.1.n"

    def test_find_relationship_grandparent_grandchild(self, tools, test_family):
        """
        Test grandparent-grandchild relationship.

        VI.1.n (Bessel) is grandparent of VIII.3.a (Eugene).
        """
        result = tools.find_relationship("VI.1.n", "VIII.3.a")

        assert result["relationship_type"] == "ancestor"
        assert result["relationship"] == "grandparent"
        assert result["generations_from_person_1"] == 0
        assert result["generations_from_person_2"] == 2

    def test_find_relationship_self(self, tools, test_family):
        """
        Test that same person returns 'self' relationship.
        """
        result = tools.find_relationship("VI.1.n", "VI.1.n")

        assert result["relationship_type"] == "self"
        assert result["relationship"] == "self"
        assert result["generations_from_person_1"] == 0
        assert result["generations_from_person_2"] == 0

    def test_find_relationship_person_not_found(self, tools):
        """
        Test error handling when person doesn't exist.
        """
        result = tools.find_relationship("INVALID.ID", "VI.1.n")

        assert "error" in result
        assert "Person not found" in result["error"]

    def test_find_relationship_no_common_ancestor(self, tools, test_family):
        """
        Test when two people are not related (no common ancestor).
        """
        # Create an unrelated person
        unrelated = Person.objects.create(
            genealogical_id="X.1.a",
            given_names="Unrelated",
            surname="Person"
        )

        result = tools.find_relationship("VI.1.n", unrelated.genealogical_id)

        # Should return "none" with error message
        assert result["relationship_type"] == "none"
        assert "error" in result
        assert "No common ancestor" in result["error"]

    def test_find_relationship_with_uuid(self, tools, test_family):
        """
        Test that find_relationship works with UUID as well as genealogical_id.
        """
        # Get person VI.1.n
        person = Person.objects.get(genealogical_id="VI.1.n")
        uuid_str = str(person.id)

        # Test with UUID instead of genealogical_id
        result = tools.find_relationship(uuid_str, "VII.3.a")

        assert result["relationship_type"] == "ancestor"
        assert result["relationship"] == "parent"

    def test_find_relationship_aunt_uncle(self, tools, test_family):
        """
        Test aunt/uncle or niece/nephew relationship.

        VII.3.c (Bessel Jr) is the aunt/uncle of VIII.3.a (Eugene).
        Eugene is VII.3.a's child, making Bessel Jr the uncle.
        """
        result = tools.find_relationship("VII.3.c", "VIII.3.a")

        assert result["relationship_type"] == "aunt_uncle_niece_nephew"
        assert "aunt/uncle" in result["relationship"] or "niece/nephew" in result["relationship"]
        # Bessel Jr is 1 generation from common ancestor (Bessel VI.1.n)
        # Eugene is 2 generations from common ancestor
        assert result["generations_from_person_1"] == 1
        assert result["generations_from_person_2"] == 2
