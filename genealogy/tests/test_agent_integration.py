"""
Integration tests for the agentic workflow.

These tests use real database models and GenealogyTools but mock the LLM calls.
This validates that the entire flow works end-to-end with actual data.
"""

import pytest
from unittest.mock import patch, Mock
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
from genealogy.services.agent_executor import AgentExecutor
from genealogy.services.genealogy_tools import GenealogyTools


@pytest.mark.django_db
class TestAgentIntegration(TestCase):
    """Integration tests for agent executor with real database"""

    def setUp(self):
        """Set up test data"""
        # Create places
        self.amsterdam = Place.objects.create(name="Amsterdam")
        self.rotterdam = Place.objects.create(name="Rotterdam")
        self.den_haag = Place.objects.create(name="Den Haag")

        # Create a family: Pieter (father) + Maria (mother) -> Anna (child)
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

        # Create mentions
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
            mention=self.maria_mention,
            event_type="BIRT",
            date="1848-06-20",
            place=self.den_haag
        )

        Event.objects.create(
            mention=self.anna_mention,
            event_type="BIRT",
            date="1870-06-15",
            place=self.rotterdam
        )

        # Create relationships
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

        # Create partnership
        partnership = PartnershipMention.objects.create(
            partnership_type="MARR",
            start_date="1869-01-15"
        )
        partnership.partners.add(self.pieter_mention, self.maria_mention)

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_agent_searches_and_gets_details(self, mock_ollama_class):
        """
        Test that agent can search for a person and get their details.

        Flow: User asks "Who is Pieter van Zanten?"
        1. Agent searches by name
        2. Agent gets person details
        3. Agent provides final answer
        """
        mock_ollama = Mock()

        # Mock LLM responses
        mock_ollama.generate.side_effect = [
            # First iteration: search by name
            'TOOL_CALL: search_person_by_name\nARGUMENTS: {"name": "Pieter van Zanten"}\nREASONING: Search for Pieter van Zanten',
            # Second iteration: get details
            'TOOL_CALL: get_person_details\nARGUMENTS: {"person_id": "II.3.a"}\nREASONING: Get full details',
            # Third iteration: provide answer
            'ANSWER: Pieter van Zanten was born on March 12, 1845 in Amsterdam and died on November 3, 1920 in Rotterdam. He was married to Maria de Vries and had a daughter named Anna van Zanten.'
        ]
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        result = agent.execute("Who is Pieter van Zanten?")

        # Verify success
        self.assertTrue(result["success"])
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(len(result["tool_calls"]), 2)

        # Verify first tool call was search
        first_call = result["tool_calls"][0]
        self.assertEqual(first_call["tool"], "search_person_by_name")
        self.assertEqual(first_call["arguments"]["name"], "Pieter van Zanten")
        self.assertEqual(first_call["result"]["count"], 1)
        self.assertEqual(first_call["result"]["people"][0]["display_name"], "Pieter van Zanten")

        # Verify second tool call was get_person_details
        second_call = result["tool_calls"][1]
        self.assertEqual(second_call["tool"], "get_person_details")
        self.assertIn(second_call["arguments"]["person_id"], ["II.3.a", str(self.pieter_identity.id)])
        self.assertEqual(second_call["result"]["display_name"], "Pieter van Zanten")
        self.assertEqual(len(second_call["result"]["events"]), 2)  # birth and death
        self.assertEqual(len(second_call["result"]["children"]), 1)
        self.assertEqual(second_call["result"]["children"][0]["name"], "Anna van Zanten")

        # Verify answer
        self.assertIn("Pieter van Zanten", result["answer"])
        self.assertIn("1845", result["answer"])

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_agent_disambiguates_by_birth_year(self, mock_ollama_class):
        """
        Test that agent can disambiguate people with same name using birth year.

        Flow: User asks "Tell me about Pieter van Zanten born in 1845"
        1. Agent uses search_by_birth_year to narrow down
        2. Agent provides answer
        """
        # Create another Pieter with different birth year
        pieter2_identity = Identity.objects.create(
            display_name="Pieter van Zanten",
            genealogical_identifier="IV.2.b"
        )
        pieter2_mention = PersonMention.objects.create(
            given_names="Pieter",
            surname="van Zanten",
            genealogical_id="IV.2.b"
        )
        MentionToIdentity.objects.create(
            mention=pieter2_mention,
            identity=pieter2_identity,
            mapped_by="test"
        )
        Event.objects.create(
            mention=pieter2_mention,
            event_type="BIRT",
            date="1880-05-10",
            place=self.amsterdam
        )

        mock_ollama = Mock()
        mock_ollama.generate.side_effect = [
            # First iteration: search by birth year
            'TOOL_CALL: search_by_birth_year\nARGUMENTS: {"name": "Pieter van Zanten", "birth_year_min": 1845, "birth_year_max": 1845}\nREASONING: Find Pieter born in 1845',
            # Second iteration: provide answer
            'ANSWER: Found Pieter van Zanten (II.3.a) born March 12, 1845 in Amsterdam.'
        ]
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        result = agent.execute("Tell me about Pieter van Zanten born in 1845")

        # Verify success
        self.assertTrue(result["success"])
        self.assertEqual(len(result["tool_calls"]), 1)

        # Verify search found only the right Pieter
        search_result = result["tool_calls"][0]["result"]
        self.assertEqual(search_result["count"], 1)
        self.assertEqual(search_result["people"][0]["genealogical_id"], "II.3.a")
        self.assertEqual(search_result["people"][0]["birth"]["date"], "1845-03-12")

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_agent_traces_family_relationships(self, mock_ollama_class):
        """
        Test that agent can trace family relationships.

        Flow: User asks "Who are the children of Pieter van Zanten?"
        1. Agent searches for Pieter
        2. Agent gets children
        3. Agent provides answer
        """
        mock_ollama = Mock()
        mock_ollama.generate.side_effect = [
            # Search for Pieter
            'TOOL_CALL: search_person_by_name\nARGUMENTS: {"name": "Pieter van Zanten"}\nREASONING: Find Pieter',
            # Get children
            'TOOL_CALL: get_children\nARGUMENTS: {"person_id": "II.3.a"}\nREASONING: Get his children',
            # Answer
            'ANSWER: Pieter van Zanten had one child: Anna van Zanten, born in 1870.'
        ]
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        result = agent.execute("Who are the children of Pieter van Zanten?")

        # Verify success
        self.assertTrue(result["success"])
        self.assertEqual(len(result["tool_calls"]), 2)

        # Verify get_children result
        children_call = result["tool_calls"][1]
        self.assertEqual(children_call["tool"], "get_children")
        self.assertEqual(children_call["result"]["count"], 1)
        self.assertEqual(children_call["result"]["children"][0]["name"], "Anna van Zanten")
        self.assertEqual(children_call["result"]["children"][0]["birth_year"], 1870)

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_agent_with_initial_rag_context(self, mock_ollama_class):
        """
        Test that agent can use initial RAG context to answer without tools.

        Flow: Agent receives RAG context with answer, doesn't need to call tools
        """
        mock_ollama = Mock()
        mock_ollama.generate.return_value = 'ANSWER: Based on the context, Pieter van Zanten was born in Amsterdam in 1845.'
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        initial_context = """
        [II.3.a] Pieter van Zanten
        Born: March 12, 1845 in Amsterdam
        Died: November 3, 1920 in Rotterdam
        """

        result = agent.execute("When was Pieter van Zanten born?", initial_context=initial_context)

        # Verify success without any tool calls
        self.assertTrue(result["success"])
        self.assertEqual(len(result["tool_calls"]), 0)
        self.assertEqual(result["iterations"], 1)
        self.assertIn("1845", result["answer"])

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_agent_handles_not_found(self, mock_ollama_class):
        """
        Test that agent gracefully handles person not found.
        """
        mock_ollama = Mock()
        mock_ollama.generate.side_effect = [
            # Search returns no results
            'TOOL_CALL: search_person_by_name\nARGUMENTS: {"name": "Nonexistent Person"}\nREASONING: Search for person',
            # Agent acknowledges not found
            'ANSWER: I could not find any person named "Nonexistent Person" in the database.'
        ]
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        result = agent.execute("Who is Nonexistent Person?")

        # Verify success with informative answer
        self.assertTrue(result["success"])
        self.assertEqual(result["tool_calls"][0]["result"]["count"], 0)
        self.assertIn("could not find", result["answer"].lower())

    def test_genealogy_tools_without_mocks(self):
        """
        Test that GenealogyTools work correctly with real database data.
        This is a pure integration test without any mocks.
        """
        tools = GenealogyTools()

        # Test search
        result = tools.search_person_by_name("Pieter van Zanten")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["people"][0]["display_name"], "Pieter van Zanten")

        # Test get_person_details
        details = tools.get_person_details("II.3.a")
        self.assertEqual(details["display_name"], "Pieter van Zanten")
        self.assertEqual(len(details["events"]), 2)
        self.assertEqual(len(details["children"]), 1)
        self.assertEqual(len(details["partners"]), 1)

        # Test get_children
        children = tools.get_children("II.3.a")
        self.assertEqual(children["count"], 1)
        self.assertEqual(children["children"][0]["name"], "Anna van Zanten")

        # Test get_parents
        parents = tools.get_parents(str(self.anna_identity.id))
        self.assertEqual(parents["count"], 2)
        parent_names = {p["name"] for p in parents["parents"]}
        self.assertEqual(parent_names, {"Pieter van Zanten", "Maria de Vries"})

        # Test search_by_birth_year
        result = tools.search_by_birth_year("van Zanten", birth_year_min=1845, birth_year_max=1845)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["people"][0]["display_name"], "Pieter van Zanten")
