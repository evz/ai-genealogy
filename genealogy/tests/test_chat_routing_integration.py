"""Integration tests for model routing in chat views"""

import json
import pytest
from unittest.mock import patch, MagicMock
from django.test import Client
from django.contrib.sessions.middleware import SessionMiddleware
from django.http import HttpRequest
from genealogy.models import Conversation

@pytest.mark.django_db
class TestChatRouting:
    """Test model routing integration in chat views"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        client = Client()
        # Force session creation by accessing it
        session = client.session
        session.save()
        return client

    @pytest.fixture
    def conversation(self, client):
        """Create a test conversation"""
        # Create conversation with the client's session key
        conv = Conversation.objects.create(
            title="Test Conversation",
            session_key=client.session.session_key
        )
        return conv

    @pytest.fixture
    def mock_retriever(self):
        """Mock the HybridRetriever for RAG tests"""
        with patch('genealogy.views.chat.HybridRetriever') as mock:
            retriever = MagicMock()
            # Mock retrieve() to return empty chunks (routing doesn't need real chunks)
            retriever.retrieve.return_value = []
            # Mock build_context() to return empty context
            retriever.build_context.return_value = ""
            mock.return_value = retriever
            yield mock

    @pytest.fixture
    def mock_ollama(self):
        """Mock the OllamaClient for generation"""
        with patch('genealogy.views.chat.OllamaClient') as mock:
            ollama = MagicMock()
            # Mock generate_stream() to return a simple response
            ollama.generate_stream.return_value = iter(["Test", " response"])
            mock.return_value = ollama
            yield mock

    def test_merge_query_routes_to_reasoner(self, client, conversation, mock_retriever, mock_ollama):
        """
        Merge/identity queries should route to gene-reasoner.

        Tests that queries with keywords like "same person" trigger the
        merge detection and route to the reasoning model.
        """
        response = client.post(
            f'/chat/{conversation.id}/stream/',
            data={
                'message': 'Are these the same person?',
                'use_agent': 'false'
            }
        )

        assert response.status_code == 200
        assert response['Content-Type'] == 'text/event-stream'

        # Parse SSE stream
        content = b''.join(response.streaming_content).decode('utf-8')
        events = [line for line in content.split('\n') if line.startswith('data: ')]

        # Find model_selected event
        model_selected = None
        for event in events:
            data = json.loads(event[6:])  # Skip 'data: ' prefix
            if data.get('status') == 'model_selected':
                model_selected = data.get('model')
                break

        assert model_selected == 'gene-reasoner', \
            f"Expected gene-reasoner for merge query, got {model_selected}"

    def test_dutch_merge_query_routes_to_reasoner(self, client, conversation, mock_retriever, mock_ollama):
        """
        Dutch merge queries should also route to gene-reasoner.

        Tests multilingual keyword detection.
        """
        response = client.post(
            f'/chat/{conversation.id}/stream/',
            data={
                'message': 'Is dit dezelfde persoon?',
                'use_agent': 'false'
            }
        )

        assert response.status_code == 200

        # Parse SSE stream
        content = b''.join(response.streaming_content).decode('utf-8')
        events = [line for line in content.split('\n') if line.startswith('data: ')]

        # Find model_selected event
        model_selected = None
        for event in events:
            data = json.loads(event[6:])
            if data.get('status') == 'model_selected':
                model_selected = data.get('model')
                break

        assert model_selected == 'gene-reasoner', \
            f"Expected gene-reasoner for Dutch merge query, got {model_selected}"

    def test_simple_query_routes_to_fast(self, client, conversation, mock_retriever, mock_ollama):
        """
        Simple queries with no special keywords should route to gene-chat-fast.

        Tests default routing behavior for basic queries.
        """
        response = client.post(
            f'/chat/{conversation.id}/stream/',
            data={
                'message': 'Tell me about Jan Pieters',
                'use_agent': 'false'
            }
        )

        assert response.status_code == 200

        # Parse SSE stream
        content = b''.join(response.streaming_content).decode('utf-8')
        events = [line for line in content.split('\n') if line.startswith('data: ')]

        # Find model_selected event
        model_selected = None
        for event in events:
            data = json.loads(event[6:])
            if data.get('status') == 'model_selected':
                model_selected = data.get('model')
                break

        assert model_selected == 'gene-chat-fast', \
            f"Expected gene-chat-fast for simple query, got {model_selected}"

    def test_agent_mode_routes_to_main(self, client, conversation):
        """
        Agent mode should route to gene-chat-main.

        Tests that agentic queries use the main model for better reasoning.
        """
        response = client.post(
            f'/chat/{conversation.id}/stream/',
            data={
                'message': 'Tell me about Jan Pieters',
                'use_agent': 'true'
            }
        )

        assert response.status_code == 200

        # Parse SSE stream
        content = b''.join(response.streaming_content).decode('utf-8')
        events = [line for line in content.split('\n') if line.startswith('data: ')]

        # Find model_selected event
        model_selected = None
        for event in events:
            data = json.loads(event[6:])
            if data.get('status') == 'model_selected':
                model_selected = data.get('model')
                break

        assert model_selected == 'gene-chat-main', \
            f"Expected gene-chat-main for agent mode, got {model_selected}"

    def test_merge_query_in_agent_mode_uses_reasoner(self, client, conversation):
        """
        Merge queries should use gene-reasoner even in agent mode.

        Tests that merge detection takes precedence over agent mode.
        """
        response = client.post(
            f'/chat/{conversation.id}/stream/',
            data={
                'message': 'Are these the same person?',
                'use_agent': 'true'
            }
        )

        assert response.status_code == 200

        # Parse SSE stream
        content = b''.join(response.streaming_content).decode('utf-8')
        events = [line for line in content.split('\n') if line.startswith('data: ')]

        # Find model_selected event
        model_selected = None
        for event in events:
            data = json.loads(event[6:])
            if data.get('status') == 'model_selected':
                model_selected = data.get('model')
                break

        assert model_selected == 'gene-reasoner', \
            f"Expected gene-reasoner for merge query in agent mode, got {model_selected}"

    def test_model_selected_event_is_emitted(self, client, conversation, mock_retriever, mock_ollama):
        """
        All queries should emit a model_selected SSE event.

        Tests that the routing decision is communicated to the frontend.
        """
        response = client.post(
            f'/chat/{conversation.id}/stream/',
            data={
                'message': 'Test query',
                'use_agent': 'false'
            }
        )

        assert response.status_code == 200

        # Parse SSE stream
        content = b''.join(response.streaming_content).decode('utf-8')
        events = [line for line in content.split('\n') if line.startswith('data: ')]

        # Check that model_selected event exists
        model_selected_events = [
            json.loads(event[6:]) for event in events
            if json.loads(event[6:]).get('status') == 'model_selected'
        ]

        assert len(model_selected_events) == 1, \
            "Expected exactly one model_selected event"

        assert 'model' in model_selected_events[0], \
            "model_selected event should contain 'model' field"
