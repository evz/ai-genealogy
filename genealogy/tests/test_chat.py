"""Tests for chat interface"""

from django.test import Client, TestCase
from django.urls import reverse

from genealogy.models import Conversation, Message


class ChatViewsTest(TestCase):
    """Test chat views"""

    def setUp(self):
        """Set up test client and session"""
        self.client = Client()
        # Create session
        session = self.client.session
        session.save()
        self.session_key = session.session_key

    def test_chat_index_view(self):
        """Test chat index page loads"""
        response = self.client.get(reverse('chat_index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Genealogy Chat')

    def test_new_conversation_creates_conversation(self):
        """Test creating a new conversation"""
        response = self.client.get(reverse('chat_new'))
        self.assertEqual(response.status_code, 302)  # Redirect

        # Check conversation was created
        self.assertEqual(Conversation.objects.count(), 1)
        conversation = Conversation.objects.first()
        self.assertEqual(conversation.session_key, self.session_key)

    def test_conversation_detail_view(self):
        """Test conversation detail page"""
        conversation = Conversation.objects.create(
            title="Test Conversation",
            session_key=self.session_key
        )

        response = self.client.get(reverse('chat_conversation', args=[conversation.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Conversation')

    def test_conversation_with_messages(self):
        """Test conversation with messages displays them"""
        conversation = Conversation.objects.create(
            title="Test",
            session_key=self.session_key
        )

        Message.objects.create(
            conversation=conversation,
            role='user',
            content='Hello?'
        )

        Message.objects.create(
            conversation=conversation,
            role='assistant',
            content='Hi there!'
        )

        response = self.client.get(reverse('chat_conversation', args=[conversation.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Hello?')
        self.assertContains(response, 'Hi there!')

    def test_session_isolation(self):
        """Test that conversations are isolated by session"""
        # Create conversation with this session
        conv1 = Conversation.objects.create(
            title="My Conversation",
            session_key=self.session_key
        )

        # Create conversation with different session
        conv2 = Conversation.objects.create(
            title="Other Conversation",
            session_key="different_session"
        )

        # Index should only show my conversation
        response = self.client.get(reverse('chat_index'))
        self.assertContains(response, 'My Conversation')
        self.assertNotContains(response, 'Other Conversation')

        # Should not be able to access other session's conversation
        response = self.client.get(reverse('chat_conversation', args=[conv2.id]))
        self.assertEqual(response.status_code, 302)  # Redirect away
