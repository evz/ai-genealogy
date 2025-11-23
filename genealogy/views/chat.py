"""Chat interface views for genealogy assistant"""

import json
import logging

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from ..models import Conversation, Message
from ..ollama_utils import get_default_models
from ..services.agent_executor import AgentExecutor
from ..services import ModelRouter
from ..tasks import generate_conversation_title

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
    conversation = get_object_or_404(Conversation, id=conversation_id)

    # Verify session ownership
    if conversation.session_key != request.session.session_key:
        return redirect('chat_index')

    messages = conversation.messages.all()

    return render(request, 'genealogy/chat/conversation.html', {
        'conversation': conversation,
        'messages': messages
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
def stream_message(request, conversation_id):
    """Stream LLM response using Server-Sent Events (SSE)"""
    conversation = get_object_or_404(Conversation, id=conversation_id)

    # Verify session ownership
    if conversation.session_key != request.session.session_key:
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    user_message = request.POST.get('message', '').strip()

    if not user_message:
        return JsonResponse({'error': 'Empty message'}, status=400)

    # Initialize model router
    router = ModelRouter()

    # Save user message
    user_msg = Message.objects.create(
        conversation=conversation,
        role='user',
        content=user_message
    )

    def event_stream():
        """Generate SSE stream"""
        try:
            # Get conversation history (shared by both modes)
            history = conversation.messages.filter(
                created_at__lt=user_msg.created_at
            ).order_by('-created_at')[:50]  # Last 50 messages (25 exchanges)

            # Build conversation history text
            history_text = ""
            for msg in reversed(list(history)):
                role_label = "USER" if msg.role == "user" else "ASSISTANT"
                history_text += f"{role_label}: {msg.content}\n\n"

            # Agentic mode - use AgentExecutor with tools
            yield f"data: {json.dumps({'status': 'agent_starting'})}\n\n"

            # Route model for agent mode (no chunks yet)
            selected_model = router.route(
                query=user_message,
                chunks=None,
                use_agent=True
            )

            # Emit model selection event
            yield f"data: {json.dumps({
                'status': 'model_selected',
                'model': selected_model
            })}\n\n"

            # Agent mode: Only provide conversation history, no RAG retrieval
            # This forces the agent to use tools, giving us control over disambiguation
            initial_context = f"""CONVERSATION HISTORY:
{history_text if history_text else "No previous messages"}"""

            # Initialize agent
            agent = AgentExecutor(model=selected_model, max_iterations=20, timeout=300)

            # We'll still need chunks for saving metadata, so retrieve them after the agent finishes
            chunks = []

            # Stream agent execution
            full_response = ""
            tool_calls_made = []

            for event in agent.execute_streaming(
                user_query=user_message,
                initial_context=initial_context
            ):
                event_type = event.get("type")

                if event_type == "thinking_start":
                    yield f"data: {json.dumps({
                        'status': 'thinking_start',
                        'iteration': event['iteration'],
                        'max_iterations': event['max_iterations']
                    })}\n\n"

                elif event_type == "thinking_token":
                    yield f"data: {json.dumps({
                        'status': 'thinking_token',
                        'token': event['token'],
                        'iteration': event['iteration']
                    })}\n\n"

                elif event_type == "thinking_end":
                    yield f"data: {json.dumps({
                        'status': 'thinking_end',
                        'iteration': event['iteration']
                    })}\n\n"

                elif event_type == "tool_call":
                    yield f"data: {json.dumps({
                        'status': 'tool_call',
                        'tool': event['tool'],
                        'arguments': event['arguments'],
                        'reasoning': event.get('reasoning', '')
                    })}\n\n"

                elif event_type == "tool_result":
                    yield f"data: {json.dumps({
                        'status': 'tool_result',
                        'tool': event['tool'],
                        'result': event['result']
                    })}\n\n"

                elif event_type == "answer":
                    full_response = event['answer']
                    tool_calls_made = event.get('tool_calls', [])

                    # Stream the final answer
                    yield f"data: {json.dumps({
                        'status': 'generating'
                    })}\n\n"

                    # Stream answer word by word for better UX
                    words = full_response.split()
                    for word in words:
                        yield f"data: {json.dumps({
                            'token': word + ' ',
                            'done': False
                        })}\n\n"

                elif event_type == "error":
                    # Handle error
                    error_msg = event.get('error', 'Unknown error')
                    full_response = event.get('answer', f"Error: {error_msg}")
                    tool_calls_made = event.get('tool_calls', [])

                    yield f"data: {json.dumps({
                        'error': error_msg,
                        'partial_answer': full_response
                    })}\n\n"

            # Save assistant message with tool call metadata
            Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=full_response,
                retrieved_chunks=[{
                    'id': str(c['id']),
                    'text': c['text_content'][:200],
                    'page': c['start_page'],
                    'score': float(c.get('rrf_score', 0.0))
                } for c in chunks],
                retrieval_metadata={
                    'chunks_count': len(chunks),
                    'model_used': selected_model,
                    'agent_mode': True,
                    'tool_calls': tool_calls_made
                }
            )

            # Send completion
            yield f"data: {json.dumps({
                'done': True,
                'full_text': full_response
            })}\n\n"

            # Enqueue background task to generate conversation title if this is the first message
            if conversation.title == "New Conversation":
                generate_conversation_title.delay(str(conversation.id), user_message)

        except Exception as e:
            logger.exception(f"Error streaming message: {e}")
            yield f"data: {json.dumps({
                'error': str(e),
                'done': True
            })}\n\n"

    return StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
