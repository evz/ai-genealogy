"""Chat interface views for genealogy assistant"""

import json
import logging
import time

from django.core.cache import cache
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from ..models import Conversation, Message
from ..ollama_utils import get_default_models
from ..services.agent_executor import AgentExecutor
from ..services import ModelRouter
from ..tasks import generate_conversation_title
from ..tasks.chat_agent import process_chat_message

logger = logging.getLogger(__name__)


def chat_index(request):
    """Main chat interface - list of conversations"""
    # Get or create session
    if not request.session.session_key:
        request.session.create()

    conversations = Conversation.objects.filter(
        session_key=request.session.session_key
    ).order_by('-updated_at')[:20]

    return render(request, 'genealogy/chat/index.html', {
        'conversations': conversations
    })


def conversation_detail(request, conversation_id):
    """View a specific conversation"""
    from ..models import PromptTemplate

    conversation = get_object_or_404(Conversation, id=conversation_id)

    # Verify session ownership
    if conversation.session_key != request.session.session_key:
        return redirect('chat_index')

    messages = conversation.messages.all()

    # Get available prompt templates for the selector
    available_templates = PromptTemplate.objects.filter(
        name='agent',
        is_archived=False
    ).order_by('version')

    # Determine which template is currently in use
    if conversation.prompt_template_override:
        current_template = conversation.prompt_template_override
    else:
        current_template = PromptTemplate.objects.filter(
            name='agent',
            is_active=True
        ).first()

    return render(request, 'genealogy/chat/conversation.html', {
        'conversation': conversation,
        'messages': messages,
        'available_templates': available_templates,
        'current_template': current_template,
    })


def new_conversation(request):
    """Create a new conversation"""
    # Get or create session
    if not request.session.session_key:
        request.session.create()

    conversation = Conversation.objects.create(
        title="New Conversation",
        session_key=request.session.session_key
    )

    return redirect('chat_conversation', conversation_id=conversation.id)


@require_http_methods(["POST"])
def set_prompt_template(request, conversation_id):
    """Set the prompt template override for this conversation"""
    from ..models import PromptTemplate

    conversation = get_object_or_404(Conversation, id=conversation_id)

    # Verify session ownership
    if conversation.session_key != request.session.session_key:
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    template_id = request.POST.get('template_id')

    if template_id == 'default':
        # Clear override, use global active template
        conversation.prompt_template_override = None
        conversation.save()
        return JsonResponse({
            'status': 'success',
            'message': 'Using global active template'
        })

    try:
        template = PromptTemplate.objects.get(id=template_id)
        conversation.prompt_template_override = template
        conversation.save()
        return JsonResponse({
            'status': 'success',
            'message': f'Now using template v{template.version}'
        })
    except PromptTemplate.DoesNotExist:
        return JsonResponse({'error': 'Template not found'}, status=404)


@require_http_methods(["GET"])
def stream_events(request, message_id):
    """
    SSE endpoint that streams events for a specific message.

    The browser connects to this endpoint and receives real-time updates
    as the Celery task processes the message.
    """
    def event_stream():
        """Generate SSE stream from Redis events"""
        cache_key = f"chat_events:{message_id}"
        last_index = 0
        timeout_counter = 0
        max_timeout = 120  # 2 minutes of no events before giving up

        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected'})}\n\n"

        while timeout_counter < max_timeout:
            # Get events from Redis
            events = cache.get(cache_key, [])

            # Send any new events
            if len(events) > last_index:
                for event in events[last_index:]:
                    yield f"data: {json.dumps(event)}\n\n"

                    # Check if this is a terminal event
                    if event.get('type') in ['complete', 'error']:
                        # Clean up events after sending completion
                        cache.delete(cache_key)
                        return

                last_index = len(events)
                timeout_counter = 0  # Reset timeout on activity
            else:
                # No new events, send keepalive
                yield f": keepalive\n\n"
                timeout_counter += 1
                time.sleep(1)

        # Timeout - send error and close
        yield f"data: {json.dumps({'type': 'error', 'error': 'Stream timeout'})}\n\n"

    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response


@require_http_methods(["POST"])
def send_message(request, conversation_id):
    """
    Receive a chat message and start async processing.

    Creates message records and starts a Celery task to process the message.
    Browser should connect to /stream/<message_id>/ to receive updates.
    """
    conversation = get_object_or_404(Conversation, id=conversation_id)

    # Verify session ownership
    if conversation.session_key != request.session.session_key:
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    user_message = request.POST.get('message', '').strip()

    if not user_message:
        return JsonResponse({'error': 'Empty message'}, status=400)

    # Save user message
    user_msg = Message.objects.create(
        conversation=conversation,
        role='user',
        content=user_message
    )

    # Create assistant message placeholder
    assistant_msg = Message.objects.create(
        conversation=conversation,
        role='assistant',
        content='',  # Will be updated by Celery task
        retrieval_metadata={'agent_mode': True}
    )

    # Start Celery task to process the message
    process_chat_message.delay(
        conversation_id=str(conversation.id),
        user_message_id=str(user_msg.id),
        assistant_message_id=str(assistant_msg.id),
        user_message_text=user_message
    )

    # Enqueue background task to generate conversation title if this is the first message
    if conversation.title == "New Conversation":
        generate_conversation_title.delay(str(conversation.id), user_message)

    # Return message IDs so browser can connect to SSE stream
    return JsonResponse({
        'status': 'started',
        'user_message_id': str(user_msg.id),
        'assistant_message_id': str(assistant_msg.id),
        'stream_url': f'/chat/stream/{assistant_msg.id}/'
    })
