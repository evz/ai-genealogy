"""Celery task for processing chat messages with agent"""

import json
import logging
from typing import Dict, Any

from celery import shared_task
from django.core.cache import cache

from ..models import Conversation, Message
from ..services.agent_executor import AgentExecutor
from ..services import ModelRouter

logger = logging.getLogger(__name__)


def publish_event(message_id: str, event: Dict[str, Any]):
    """Publish an event to the SSE stream for this message"""
    # Use Redis list to store events for this message
    # SSE endpoint will read from this list
    cache_key = f"chat_events:{message_id}"

    # Append event to list
    events = cache.get(cache_key, [])
    events.append(event)
    cache.set(cache_key, events, timeout=3600)  # Keep for 1 hour

    # Also publish to a channel for real-time streaming
    cache.set(f"chat_event_latest:{message_id}", event, timeout=60)


@shared_task(bind=True, max_retries=0)
def process_chat_message(
    self,
    conversation_id: str,
    user_message_id: str,
    assistant_message_id: str,
    user_message_text: str
):
    """
    Process a chat message using the agent.

    Publishes events to Redis that the SSE endpoint can stream to the browser.

    Args:
        conversation_id: UUID of the conversation
        user_message_id: UUID of the user's message
        assistant_message_id: UUID of the assistant's message (placeholder)
        user_message_text: The user's query
    """
    try:
        conversation = Conversation.objects.get(id=conversation_id)
        assistant_msg = Message.objects.get(id=assistant_message_id)

        # Publish starting event
        publish_event(assistant_message_id, {
            'type': 'status',
            'status': 'started',
            'message': 'Starting agent...'
        })

        # Get conversation history
        history = conversation.messages.filter(
            created_at__lt=assistant_msg.created_at
        ).order_by('-created_at')[:50]

        # Build conversation history text
        history_text = ""
        for msg in reversed(list(history)):
            role_label = "USER" if msg.role == "user" else "ASSISTANT"
            history_text += f"{role_label}: {msg.content}\n\n"

        # Initialize model router
        router = ModelRouter()
        selected_model = router.route(
            query=user_message_text,
            chunks=None,
            use_agent=True
        )

        # Publish model selection
        publish_event(assistant_message_id, {
            'type': 'model_selected',
            'status': 'model_selected',
            'model': selected_model
        })

        # Prepare initial context
        initial_context = f"""CONVERSATION HISTORY:
{history_text if history_text else "No previous messages"}"""

        # Get prompt template version (from conversation override or use active)
        prompt_template_version = None
        if conversation.prompt_template_override:
            prompt_template_version = conversation.prompt_template_override.version

        # Initialize agent
        agent = AgentExecutor(
            model=selected_model,
            max_iterations=20,
            timeout=600,  # 10 minutes - much longer since we're async
            prompt_template_name="agent",
            prompt_template_version=prompt_template_version,
            message_instance=assistant_msg
        )

        # Stream agent execution
        full_response = ""
        tool_calls_made = []
        reasoning_steps = []

        for event in agent.execute_streaming(
            user_query=user_message_text,
            initial_context=initial_context
        ):
            # Forward all events to SSE stream
            publish_event(assistant_message_id, event)

            # Track state for final save
            if event.get("type") == "answer":
                full_response = event['answer']
                tool_calls_made = event.get('tool_calls', [])
            elif event.get("type") == "thinking_end":
                if event.get('content'):
                    reasoning_steps.append({
                        'iteration': event.get('iteration'),
                        'content': event.get('content'),
                        'type': 'thinking'
                    })
            elif event.get("type") == "error":
                full_response = event.get('answer', f"Error: {event.get('error', 'Unknown error')}")
                tool_calls_made = event.get('tool_calls', [])

        # Update assistant message with final content
        assistant_msg.content = full_response
        assistant_msg.retrieval_metadata = {
            'chunks_count': 0,
            'model_used': selected_model,
            'agent_mode': True,
            'tool_calls': tool_calls_made,
            'reasoning_steps': reasoning_steps
        }
        assistant_msg.save()

        # Publish completion
        publish_event(assistant_message_id, {
            'type': 'complete',
            'status': 'complete',
            'message_id': str(assistant_message_id),
            'content': full_response
        })

        logger.info(f"Chat message {assistant_message_id} processed successfully")

    except Exception as e:
        logger.exception(f"Error processing chat message: {e}")

        # Publish error
        publish_event(assistant_message_id, {
            'type': 'error',
            'status': 'error',
            'error': str(e)
        })

        # Update message with error
        try:
            assistant_msg = Message.objects.get(id=assistant_message_id)
            assistant_msg.content = f"Error: {str(e)}"
            assistant_msg.save()
        except:
            pass
