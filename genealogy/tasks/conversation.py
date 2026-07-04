"""
Conversation-related Celery tasks
"""
import logging
from celery import shared_task
from django.db import transaction

logger = logging.getLogger(__name__)


@shared_task
def generate_conversation_title(conversation_id: str, user_message: str):
    """
    Generate a title for a conversation based on the first user message.

    This task runs asynchronously to avoid blocking the chat response.

    Args:
        conversation_id: UUID of the conversation
        user_message: First user message to base title on
    """
    try:
        from genealogy.models import Conversation
        from genealogy.ollama_utils import OllamaClient

        conversation = Conversation.objects.get(id=conversation_id)

        # Only generate if still "New Conversation"
        if conversation.title != "New Conversation":
            logger.info(f"Conversation {conversation_id} already has title: {conversation.title}")
            return

        ollama = OllamaClient()
        new_title = ollama.generate_conversation_title(user_message)

        with transaction.atomic():
            conversation.title = new_title
            conversation.save(update_fields=['title'])

        logger.info(f"Generated title for conversation {conversation_id}: {new_title}")

    except Exception as e:
        logger.exception(f"Failed to generate title for conversation {conversation_id}: {e}")
