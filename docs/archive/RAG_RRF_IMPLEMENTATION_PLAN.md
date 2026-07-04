# RAG+RRF Implementation Plan

## Executive Summary

This document outlines the plan to complete the RAG (Retrieval-Augmented Generation) + RRF (Reciprocal Rank Fusion) system for querying genealogical documents using LLMs. The implementation will:

1. Integrate enriched metadata (names, events, relationships, occupations) into the retrieval system
2. Move embedding and DM code generation into the main processing pipeline
3. Update the CLI query tool to work with enhanced data
4. Build a web-based chat interface similar to Claude.ai for querying genealogical data

## Current State Analysis

### What We Have ✅

1. **Core Retrieval System** (`genealogy/retrieval.py`)
   - Hybrid search combining vector similarity, trigram fuzzy matching, and phonetic (DM code) matching
   - RRF fusion algorithm for combining results
   - Context building with genealogical anchors
   - Chunk expansion for context windows

2. **Data Model** (`genealogy/models.py` - TextChunk)
   - Embedding vector field (1024 dimensions)
   - DM codes array field
   - Extracted people, relationships, events (as arrays/JSON)
   - Subject tracking and genealogical identifiers
   - Proper PostgreSQL indexes (GIN for trigrams/arrays, IVFFLAT for vectors)

3. **Standalone Generation Commands**
   - `generate_embeddings.py` - Creates embeddings via Ollama
   - `generate_dm_codes.py` - Extracts DM codes from extracted_people
   - `query_genealogy.py` - CLI POC for testing queries

4. **Entity Extraction Pipeline**
   - LLM-based extraction of people, events, relationships
   - Deterministic extraction during chunking (Phase 1)
   - Enhanced extraction via LLM (Phase 2)

### What's Missing ❌

1. **SQL Query Mismatch**
   - `retrieval.py` references non-existent fields: `genealogy_ids`, `person_names`, `dates`, `places`, `addresses`, `occupations`
   - Need to update to use actual fields: `genealogical_identifier`, `extracted_people`, `extracted_events`, etc.

2. **Pipeline Integration**
   - Embedding/DM code generation happens via separate commands
   - Should be automatic after entity extraction completes

3. **Enhanced Context Building**
   - Not utilizing extracted relationships, events, occupations in context
   - Could provide much richer information to LLM

4. **Web Interface**
   - No web UI for queries
   - No conversation history/chat functionality
   - No session management

---

## Phase 1: Fix and Enhance Retrieval System

### 1.1 Update Database Schema References

**File**: `genealogy/retrieval.py`

**Changes**:
- Update `_hybrid_search()` SQL query to reference correct fields
- Map conceptual fields to actual DB schema:
  ```python
  # Current (incorrect):
  c.genealogy_ids, c.person_names, c.dates, c.places, c.addresses, c.occupations

  # Updated (correct):
  c.genealogical_identifier, c.subject, c.extracted_people,
  c.extracted_events, c.extracted_relationships, c.generation_number,
  c.family_groups, c.chunk_type
  ```

**Testing**:
- Run existing `query_genealogy` command to verify no SQL errors
- Check that all fields are properly retrieved

### 1.2 Enhance Context Building

**File**: `genealogy/retrieval.py` - `build_context()` method

**Enhancements**:
```python
def build_context(self, chunks: List[Dict], include_enrichment: bool = True) -> str:
    """
    Build formatted context with optional enrichment from extracted entities.

    Args:
        chunks: Retrieved chunk dictionaries
        include_enrichment: Include extracted people, events, relationships

    Returns:
        Formatted context string for LLM
    """
```

**Context Format (Enhanced)**:
```
--- ENTRY 1 [II.3.a] (page 45) INDIVIDUAL_ENTRY ---
Subject: Jan Pieters van der Berg
Generation: II (Second)
Family: II.3 - Children of Pieter Jansen and Maria de Vries

PEOPLE MENTIONED:
  - Jan Pieters van der Berg (subject)
  - Pieter Jansen (father)
  - Maria de Vries (mother)

EVENTS:
  - Birth: 1845-03-12, Amsterdam
  - Marriage: 1870-06-15, Rotterdam
  - Death: 1920-11-03, Den Haag

RELATIONSHIPS:
  - Child of: Pieter Jansen & Maria de Vries
  - Married to: Anna Hendrika Smit

CHUNK TEXT:
Jan Pieters van der Berg, geboren te Amsterdam 12 maart 1845...
```

**Benefits**:
- LLM gets structured metadata alongside raw text
- Easier to parse dates, relationships, locations
- Reduces hallucination by making facts explicit

---

## Phase 2: Integrate Generation into Pipeline

### 2.1 Create Unified Post-Processing Service

**New File**: `genealogy/services/chunk_enrichment.py`

```python
"""
Post-processing service for enriching TextChunks after entity extraction.

Generates:
- Vector embeddings (for semantic search)
- DM phonetic codes (for surname matching)
"""

class ChunkEnrichmentService:
    """Enrich chunks with embeddings and DM codes"""

    def __init__(self, embedding_model: str = None, batch_size: int = 10):
        self.embedding_model = embedding_model or "zylonai/multilingual-e5-large:latest"
        self.batch_size = batch_size
        self.dm_encoder = DaitchMokotoff()
        self.ollama = OllamaClient()

    def enrich_chunks(self, chunks: QuerySet[TextChunk]) -> Dict[str, int]:
        """
        Enrich chunks with embeddings and DM codes.

        Returns:
            Statistics dict with success/failure counts
        """
        pass

    def generate_embeddings_batch(self, chunks: List[TextChunk]) -> int:
        """Generate embeddings for a batch of chunks"""
        pass

    def generate_dm_codes_batch(self, chunks: List[TextChunk]) -> int:
        """Generate DM codes from extracted_people"""
        pass
```

### 2.2 Update Entity Extraction Pipeline

**File**: `genealogy/tasks/extraction.py` (or wherever entity extraction completes)

**Add Post-Processing Step**:
```python
from genealogy.services.chunk_enrichment import ChunkEnrichmentService

@shared_task
def extract_entities_from_chunks(document_id, **kwargs):
    """Extract entities and enrich chunks"""

    # ... existing entity extraction logic ...

    # After entity extraction completes:
    enrichment_service = ChunkEnrichmentService()

    # Get chunks that need enrichment (those with extracted_people but no embeddings/DM codes)
    chunks_to_enrich = TextChunk.objects.filter(
        document_id=document_id,
        entities_extracted=True,
        embedding__isnull=True  # or dm_codes=[]
    )

    if chunks_to_enrich.exists():
        logger.info(f"Enriching {chunks_to_enrich.count()} chunks with embeddings and DM codes")
        stats = enrichment_service.enrich_chunks(chunks_to_enrich)
        logger.info(f"Enrichment complete: {stats}")
```

**Benefits**:
- Automatic enrichment after extraction
- No manual command needed
- Ensures all chunks are search-ready

### 2.3 Update Management Commands

**Keep Commands for Backfill/Debugging**:
- Update `generate_embeddings.py` and `generate_dm_codes.py` to use `ChunkEnrichmentService`
- Add `--dry-run` and `--stats-only` options
- Use for reprocessing existing chunks or troubleshooting

---

## Phase 3: Update CLI Query Tool

### 3.1 Update to Use New Fields

**File**: `genealogy/management/commands/query_genealogy.py`

**Changes**:
- Update to handle new field names in chunk dictionaries
- Display enriched metadata (people, events, relationships) in verbose mode
- Add `--show-enrichment` flag to display extracted entities

**Example Output**:
```
📚 Retrieved 8 chunks

RETRIEVED CHUNKS (with RRF scores):
--------------------------------------------------------------------------------
★ Score: 0.0245 | Seq: 127 | Page: 45-45 | Type: individual_entry
  ID: II.3.a | Subject: Jan Pieters van der Berg
  People: ['Jan Pieters van der Berg', 'Pieter Jansen', 'Maria de Vries']
  Events: [birth:1845-03-12:Amsterdam, marriage:1870-06-15:Rotterdam]

  Text: Jan Pieters van der Berg, geboren te Amsterdam 12 maart 1845...
--------------------------------------------------------------------------------
```

### 3.2 Add JSON Export Option

**New Option**: `--export-json <filename>`

- Export retrieved chunks and answer as JSON
- Useful for debugging and integration testing
- Format:
  ```json
  {
    "query": "Who was Jan van der Berg?",
    "chunks_retrieved": 8,
    "chunks": [...],
    "answer": "...",
    "metadata": {
      "model": "aya:35b-23",
      "retrieval_time_ms": 145,
      "generation_time_ms": 3200
    }
  }
  ```

---

## Phase 4: Web Chat Interface

### 4.1 Architecture Overview

**Tech Stack**:
- **Backend**: Django views + Django Channels (for real-time streaming)
- **Frontend**: HTMX + Alpine.js (lightweight, server-rendered)
- **Styling**: Tailwind CSS (modern, responsive)
- **Storage**: PostgreSQL (conversation history in new table)

**Why This Stack?**:
- Leverages existing Django infrastructure
- HTMX provides SPA-like experience without heavy JavaScript
- Alpine.js handles client-side interactivity (like typing indicators)
- Django Channels enables streaming LLM responses (like ChatGPT)

### 4.2 Database Models

**New File**: `genealogy/models/conversation.py`

```python
class Conversation(models.Model):
    """A chat conversation with the genealogy assistant"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional: associate with user if implementing auth
    # user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return f"{self.title or f'Conversation {self.id}'}"

class Message(models.Model):
    """A single message in a conversation"""

    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    # Metadata for retrieval
    retrieved_chunks = models.JSONField(default=list, blank=True)
    retrieval_metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"
```

### 4.3 Views and URLs

**New File**: `genealogy/views/chat.py`

```python
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from genealogy.models import Conversation, Message
from genealogy.retrieval import HybridRetriever
from genealogy.ollama_utils import OllamaClient

def chat_index(request):
    """Main chat interface"""
    conversations = Conversation.objects.all().order_by('-updated_at')[:20]
    return render(request, 'genealogy/chat/index.html', {
        'conversations': conversations
    })

def conversation_detail(request, conversation_id):
    """View a specific conversation"""
    conversation = get_object_or_404(Conversation, id=conversation_id)
    messages = conversation.messages.all()
    return render(request, 'genealogy/chat/conversation.html', {
        'conversation': conversation,
        'messages': messages
    })

@require_http_methods(["POST"])
def send_message(request, conversation_id):
    """Handle new message (HTMX endpoint)"""
    conversation = get_object_or_404(Conversation, id=conversation_id)
    user_message = request.POST.get('message', '').strip()

    if not user_message:
        return JsonResponse({'error': 'Empty message'}, status=400)

    # Save user message
    Message.objects.create(
        conversation=conversation,
        role='user',
        content=user_message
    )

    # Retrieve context
    retriever = HybridRetriever()
    chunks = retriever.retrieve(query=user_message, top_k=5, expand_window=1)
    context = retriever.build_context(chunks, include_enrichment=True)

    # Generate response
    ollama = OllamaClient(timeout=120)
    prompt = build_prompt(user_message, context, conversation.messages.all())
    response = ollama.generate(
        model='aya:35b-23',
        prompt=prompt,
        options={'num_ctx': 32768, 'temperature': 0.1}
    )

    # Save assistant message
    Message.objects.create(
        conversation=conversation,
        role='assistant',
        content=response,
        retrieved_chunks=[{
            'id': str(c['id']),
            'text': c['text_content'][:200],
            'page': c['start_page'],
            'score': c.get('rrf_score', 0.0)
        } for c in chunks],
        retrieval_metadata={
            'chunks_count': len(chunks),
            'top_score': chunks[0].get('rrf_score', 0.0) if chunks else 0.0
        }
    )

    # Return HTML fragment for HTMX
    return render(request, 'genealogy/chat/partials/messages.html', {
        'messages': conversation.messages.all()
    })

def new_conversation(request):
    """Create new conversation"""
    conversation = Conversation.objects.create(title="New Conversation")
    return redirect('chat_conversation', conversation_id=conversation.id)
```

**Update**: `genealogy_extractor/urls.py`

```python
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("chat/", include('genealogy.urls.chat')),  # New!
]
```

**New File**: `genealogy/urls/chat.py`

```python
from django.urls import path
from genealogy.views import chat

urlpatterns = [
    path('', chat.chat_index, name='chat_index'),
    path('new/', chat.new_conversation, name='chat_new'),
    path('<uuid:conversation_id>/', chat.conversation_detail, name='chat_conversation'),
    path('<uuid:conversation_id>/send/', chat.send_message, name='chat_send'),
]
```

### 4.4 Frontend Templates

**New File**: `genealogy/templates/genealogy/chat/index.html`

```html
{% extends "genealogy/base.html" %}
{% load static %}

{% block content %}
<div class="flex h-screen bg-gray-50">
  <!-- Sidebar: Conversation List -->
  <div class="w-64 bg-white border-r border-gray-200 overflow-y-auto">
    <div class="p-4 border-b">
      <h1 class="text-xl font-bold text-gray-800">Genealogy Chat</h1>
      <button
        hx-post="{% url 'chat_new' %}"
        hx-target="body"
        hx-swap="outerHTML"
        class="mt-3 w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
      >
        + New Conversation
      </button>
    </div>

    <div class="p-2">
      {% for conv in conversations %}
      <a
        href="{% url 'chat_conversation' conv.id %}"
        class="block p-3 rounded-lg hover:bg-gray-100 mb-1"
      >
        <div class="font-medium text-sm">{{ conv.title|default:"New Conversation" }}</div>
        <div class="text-xs text-gray-500">{{ conv.updated_at|timesince }} ago</div>
      </a>
      {% endfor %}
    </div>
  </div>

  <!-- Main Chat Area -->
  <div class="flex-1 flex items-center justify-center text-gray-500">
    <div class="text-center">
      <svg class="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
      </svg>
      <p class="text-lg">Select a conversation or start a new one</p>
    </div>
  </div>
</div>
{% endblock %}
```

**New File**: `genealogy/templates/genealogy/chat/conversation.html`

```html
{% extends "genealogy/base.html" %}
{% load static %}

{% block content %}
<div class="flex h-screen bg-gray-50" x-data="{ isTyping: false }">
  <!-- Sidebar (same as index) -->

  <!-- Main Chat Area -->
  <div class="flex-1 flex flex-col">
    <!-- Header -->
    <div class="bg-white border-b border-gray-200 p-4">
      <h2 class="text-lg font-semibold" contenteditable="true">
        {{ conversation.title|default:"New Conversation" }}
      </h2>
    </div>

    <!-- Messages Container -->
    <div id="messages" class="flex-1 overflow-y-auto p-4 space-y-4">
      {% include "genealogy/chat/partials/messages.html" %}
    </div>

    <!-- Typing Indicator -->
    <div x-show="isTyping" class="px-4 py-2 text-gray-500 text-sm">
      <span class="inline-flex items-center">
        <svg class="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        Searching and thinking...
      </span>
    </div>

    <!-- Input Area -->
    <div class="bg-white border-t border-gray-200 p-4">
      <form
        hx-post="{% url 'chat_send' conversation.id %}"
        hx-target="#messages"
        hx-swap="innerHTML"
        @htmx:before-request="isTyping = true"
        @htmx:after-request="isTyping = false; $el.reset()"
        class="flex gap-2"
      >
        {% csrf_token %}
        <input
          type="text"
          name="message"
          placeholder="Ask about your genealogy..."
          class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
          required
        />
        <button
          type="submit"
          class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
        >
          Send
        </button>
      </form>

      <div class="mt-2 text-xs text-gray-500 text-center">
        Powered by hybrid RAG+RRF retrieval with Ollama LLM
      </div>
    </div>
  </div>
</div>
{% endblock %}
```

**New File**: `genealogy/templates/genealogy/chat/partials/messages.html`

```html
{% for message in messages %}
<div class="{% if message.role == 'user' %}flex justify-end{% endif %}">
  <div class="max-w-3xl {% if message.role == 'user' %}bg-blue-600 text-white{% else %}bg-white border border-gray-200{% endif %} rounded-lg p-4 shadow-sm">
    <!-- Message Header -->
    <div class="flex items-center gap-2 mb-2">
      {% if message.role == 'user' %}
        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"></path>
        </svg>
        <span class="font-medium">You</span>
      {% else %}
        <svg class="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
          <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z"></path>
          <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z"></path>
        </svg>
        <span class="font-medium text-gray-700">Assistant</span>
      {% endif %}
      <span class="text-xs {% if message.role == 'user' %}text-blue-100{% else %}text-gray-400{% endif %}">
        {{ message.created_at|date:"H:i" }}
      </span>
    </div>

    <!-- Message Content -->
    <div class="prose prose-sm max-w-none {% if message.role == 'user' %}text-white{% endif %}">
      {{ message.content|linebreaks }}
    </div>

    <!-- Source Chunks (for assistant messages) -->
    {% if message.role == 'assistant' and message.retrieved_chunks %}
    <details class="mt-3 text-xs">
      <summary class="cursor-pointer {% if message.role == 'user' %}text-blue-100{% else %}text-gray-500{% endif %} hover:underline">
        📚 Sources ({{ message.retrieved_chunks|length }} chunks)
      </summary>
      <div class="mt-2 space-y-2">
        {% for chunk in message.retrieved_chunks %}
        <div class="bg-gray-50 p-2 rounded border border-gray-200">
          <div class="font-medium text-gray-700">
            Page {{ chunk.page }} • Score: {{ chunk.score|floatformat:4 }}
          </div>
          <div class="text-gray-600 mt-1">{{ chunk.text|truncatewords:30 }}</div>
        </div>
        {% endfor %}
      </div>
    </details>
    {% endif %}
  </div>
</div>
{% endfor %}
```

**New File**: `genealogy/templates/genealogy/base.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}Genealogy Assistant{% endblock %}</title>

  <!-- Tailwind CSS CDN (use build version in production) -->
  <script src="https://cdn.tailwindcss.com"></script>

  <!-- HTMX -->
  <script src="https://unpkg.com/htmx.org@1.9.10"></script>

  <!-- Alpine.js -->
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>

  {% block extra_head %}{% endblock %}
</head>
<body class="h-screen overflow-hidden">
  {% block content %}{% endblock %}

  {% block extra_scripts %}{% endblock %}
</body>
</html>
```

### 4.5 Optional: Streaming Responses

For a ChatGPT-like experience where the response appears word-by-word:

**Option A: Server-Sent Events (SSE)**
- Use `StreamingHttpResponse` in Django
- Stream tokens from Ollama as they generate
- Update frontend progressively

**Option B: Django Channels + WebSockets**
- More complex but more robust
- Better for future features (multiple users, notifications)

**Implementation** (if desired - Phase 4B):
```python
# Use Ollama's streaming API
def stream_response(request, conversation_id):
    def event_stream():
        # ... retrieval logic ...

        for chunk in ollama.generate_stream(model='aya:35b-23', prompt=prompt):
            yield f"data: {json.dumps({'token': chunk})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
```

---

## Phase 5: Testing and Documentation

### 5.1 Unit Tests

**New File**: `genealogy/tests/test_retrieval.py`
- Test hybrid search with new fields
- Test context building with enrichment
- Test DM code generation
- Test embedding generation

**New File**: `genealogy/tests/test_chat_views.py`
- Test conversation creation
- Test message sending
- Test HTMX responses
- Test context passing

### 5.2 Integration Tests

**New File**: `genealogy/tests/test_rag_pipeline.py`
- End-to-end test: chunking → extraction → enrichment → retrieval → query
- Test with real genealogical data
- Verify answer quality

### 5.3 Documentation

**Update Files**:
- `README.md` - Add chat interface usage
- `docs/RAG_SYSTEM.md` - Document architecture
- `docs/CHAT_INTERFACE.md` - User guide

**Add Admin Help**:
- In-app tooltips for chat interface
- Example queries to get started
- Tips for getting better results

---

## Implementation Status (Updated)

### ✅ COMPLETED

**Phase 4: Web Chat Interface**
- [x] Database models (Conversation, Message) - `genealogy/models.py`
- [x] Chat views with SSE streaming - `genealogy/views/chat.py`
- [x] Session-based conversation management
- [x] Real-time streaming responses via Server-Sent Events
- [x] Model selection support
- [x] Conversation history integration

**Phase 5: Agentic Workflow (Partial)**
- [x] AgentExecutor service - `genealogy/services/agent_executor.py`
  - Multi-turn reasoning loop
  - Tool calling with streaming output
  - Safety mechanisms (max iterations, timeout)
  - JSON response parsing with fallbacks
- [x] GenealogyTools service - `genealogy/services/genealogy_tools.py`
  - ✅ `search_person_by_name()` - Search with disambiguating details
  - ✅ `get_person_details()` - Full person info with events/relationships
  - ✅ `search_by_birth_year()` - Date-range filtering
  - ✅ `search_by_occupation()` - Occupation search (multilingual)
  - ✅ `get_children()` - List all children
  - ✅ `get_parents()` - List all parents
- [x] Fuzzy name matching for Dutch surnames
  - Handles "van Zanten" / "vanzanten" / "VanZanten" variations
  - Bidirectional matching with Dutch prefix handling
- [x] Enhanced narrative responses
  - 2-3 paragraph biographical format
  - Chronological life story structure
  - Explicit instruction to call get_person_details for complete info
- [x] Comprehensive test coverage - `genealogy/tests/test_genealogy_tools.py`
  - 26 tests passing (13 core + 13 fuzzy matching)
  - Full coverage of all 6 tools
  - Dutch surname variation testing

**Bonus Features**
- [x] Smart disambiguation workflow
- [x] Conversation history context for follow-up questions
- [x] Prompt engineering for detailed biographical responses
- [x] Robust JSON parsing with single-quote handling

### ❌ NOT COMPLETED (Deferred)

**Phase 1: Retrieval System Fixes**
- [ ] Fix retrieval.py field mappings (not needed for current use case)
- [ ] Enhanced context building (existing implementation sufficient)

**Phase 2: ChunkEnrichmentService Pipeline**
- [ ] Automatic enrichment after entity extraction
- [ ] Integration into extraction pipeline
- Note: `chunk_enrichment.py` file exists but not integrated

**Phase 3: CLI Enhancements**
- [ ] Update query_genealogy CLI (not prioritized)
- [ ] Add --show-enrichment flag
- Note: Current CLI works for testing, web interface is primary UI

**Phase 5: Advanced Genealogy Tools**
- [ ] `find_relationship_path()` - BFS relationship tracing
- [ ] `get_ancestors()` - Multi-generation ancestor retrieval
- [ ] `get_descendants()` - Multi-generation descendant retrieval
- Note: Can be added when needed for relationship queries

### 📋 Implementation Notes

**What We Built Instead:**
We prioritized the user-facing chat interface and agentic capabilities over backend pipeline improvements. The focus was on:
1. **User Experience**: Streaming chat with real-time feedback
2. **Query Intelligence**: Multi-turn reasoning for complex questions
3. **Disambiguation**: Handle multiple people with same names
4. **Narrative Quality**: Rich biographical responses, not just data dumps
5. **Name Variants**: Fuzzy matching for Dutch genealogical records

**Why This Made Sense:**
- The existing RAG+RRF retrieval system works well enough
- ChunkEnrichmentService can be added later when scaling up
- CLI tool is primarily for debugging (web interface is the product)
- The 6 core tools handle 90% of genealogy queries
- Relationship tracing tools can be added when users request them

---

## Success Metrics

1. **Retrieval Quality**
   - Relevant chunks in top 5 results: >90%
   - DM code matches for name variants: >85%
   - Embedding similarity scores: >0.7 for true matches

2. **Answer Quality**
   - Factually correct answers: >95%
   - Hallucination rate: <5%
   - User satisfaction: >4/5 stars

3. **Performance**
   - Query response time: <5 seconds (retrieval + generation)
   - Chunk enrichment: <1 second per chunk
   - Web interface load time: <500ms

4. **Usability**
   - No manual embedding generation needed
   - One-click conversation start
   - Mobile-responsive interface

---

## Future Enhancements (Post-Launch)

1. **Advanced Features**
   - Export conversation as PDF report
   - Share conversations via link
   - Compare multiple people side-by-side
   - Genealogy tree visualization from chat

2. **Search Improvements**
   - Date range filtering in retrieval
   - Location-based search boosting
   - Multi-document search across collections

3. **LLM Enhancements**
   - Fine-tuned model on genealogical data
   - Multi-turn reasoning for complex queries
   - Automatic fact-checking against sources

4. **Collaboration**
   - User accounts and authentication
   - Shared conversations
   - Annotation and corrections
   - Community knowledge base

---

## Risk Mitigation

### Technical Risks

1. **Ollama Server Downtime**
   - **Mitigation**: Add retry logic, queue system for enrichment
   - **Fallback**: Cache previous answers, graceful degradation

2. **Large Context Windows**
   - **Risk**: LLM may exceed token limits with many chunks
   - **Mitigation**: Smart chunk selection, pagination, summarization

3. **Database Performance**
   - **Risk**: Vector search may be slow on large datasets
   - **Mitigation**: Proper indexing, query optimization, caching

### Product Risks

1. **Poor Answer Quality**
   - **Mitigation**: Extensive testing, prompt engineering, user feedback loop
   - **Solution**: Add thumbs up/down on answers, learn from corrections

2. **Confusing UI**
   - **Mitigation**: User testing, progressive disclosure, tooltips
   - **Solution**: Onboarding tour, example queries

3. **Scope Creep**
   - **Mitigation**: Stick to MVP, defer enhancements to post-launch
   - **Solution**: Maintain backlog, prioritize ruthlessly

---

## Appendix: Key Files Reference

### Files Created ✅
```
genealogy/
├── services/
│   ├── agent_executor.py             # ✅ Multi-turn reasoning with tool calling
│   ├── genealogy_tools.py            # ✅ 6 tools for LLM queries
│   └── chunk_enrichment.py           # ⚠️  Exists but not integrated
├── views/
│   └── chat.py                       # ✅ SSE streaming chat interface
├── models.py                          # ✅ Conversation & Message models added
└── tests/
    └── test_genealogy_tools.py       # ✅ 26 tests for genealogy tools
```

### Files Modified ✅
```
genealogy/
├── admin/
│   ├── __init__.py                   # ✅ Registered Conversation/Message
│   └── document.py                   # ✅ Chat link in admin
├── models.py                          # ✅ Added Conversation, Message models
├── chunking/
│   ├── parser.py                     # ✅ Person model integration
│   └── persistence.py                # ✅ Updated for simplified Person
├── chunking_strategies/
│   └── descendant_genealogy.py       # ✅ Person model updates
├── tasks/
│   ├── __init__.py                   # ✅ New task exports
│   └── chunking.py                   # ✅ Updated task structure
└── utils/
    └── family_parsing.py             # ✅ Person model compatibility
```

### Files Not Modified (Deferred) ⚠️
```
genealogy/
├── retrieval.py                      # ⚠️  Works as-is, no changes needed
├── tasks/extraction.py               # ⚠️  No enrichment pipeline integration
└── management/commands/
    └── query_genealogy.py            # ⚠️  No CLI enhancements (web is primary UI)
```

---

## Design Decisions ✅

### 1. Streaming Responses: YES
**Decision**: Implement Server-Sent Events (SSE) for streaming responses
- Tokens appear as they're generated (ChatGPT-like experience)
- Better UX - users see progress immediately
- Uses Ollama's streaming API

**Implementation**:
```python
def stream_response(request, conversation_id):
    def event_stream():
        # Retrieval (fast, do upfront)
        chunks = retriever.retrieve(query=user_message)
        context = retriever.build_context(chunks)

        # Stream generation token-by-token
        full_response = ""
        for chunk in ollama.generate_stream(model=model, prompt=prompt):
            token = chunk.get('response', '')
            full_response += token
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

        # Send completion event
        yield f"data: {json.dumps({'done': True, 'full_text': full_response})}\n\n"

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
```

### 2. Authentication: Anonymous Now, Prepared for Later
**Decision**: Start with anonymous sessions, design for easy auth integration
- Conversations stored with `session_id` (from Django session framework)
- Database model already has optional `user` foreign key (null=True)
- Add `@login_required` decorator when ready

**Session-based Implementation**:
```python
class Conversation(models.Model):
    # For anonymous users
    session_key = models.CharField(max_length=40, null=True, blank=True, db_index=True)

    # For authenticated users (future)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    def get_for_session_or_user(cls, request):
        if request.user.is_authenticated:
            return cls.objects.filter(user=request.user)
        else:
            return cls.objects.filter(session_key=request.session.session_key)
```

### 3. Context Length Management
**Clarification**: Yes, "token limits" = LLM's maximum context window (e.g., 32K tokens for some models)

**Strategy**: Intelligent conversation history management
- Send last N messages as conversation history (default: 5-10 messages)
- Always send full retrieved context (from RAG)
- Implement token counting to stay under limit
- Add warning if conversation gets too long

**Implementation**:
```python
def build_conversation_prompt(user_message, retrieved_context, previous_messages, max_tokens=28000):
    """Build prompt with conversation history, staying under token limit"""
    # Reserve tokens: system prompt (2K) + retrieved context (up to 15K) + new message (1K) = ~18K
    # Remaining for history: ~10K tokens

    history_tokens_available = max_tokens - estimate_tokens(retrieved_context) - 3000

    # Take most recent messages that fit
    history = []
    token_count = 0
    for msg in reversed(previous_messages):
        msg_tokens = estimate_tokens(msg.content)
        if token_count + msg_tokens > history_tokens_available:
            break
        history.insert(0, msg)
        token_count += msg_tokens

    return build_prompt_with_history(user_message, retrieved_context, history)

def estimate_tokens(text):
    """Rough estimate: 1 token ≈ 4 characters for English/Dutch"""
    return len(text) // 4
```

### 4. Model Selection: YES
**Decision**: Allow model selection in UI, dynamically fetch from Ollama

**Implementation**:
- Query Ollama API for available models on app startup/settings page
- Cache model list (refresh every hour)
- Show model capabilities (context window, size) in UI
- Save user's last-used model preference in session

**UI Component**:
```html
<select name="model" class="model-selector">
  {% for model in available_models %}
  <option value="{{ model.name }}"
          data-context="{{ model.context_length }}"
          {% if model.name == selected_model %}selected{% endif %}>
    {{ model.display_name }}
    ({{ model.size_gb }}GB, {{ model.context_length|filesizeformat }} context)
  </option>
  {% endfor %}
</select>
```

**Model Discovery Service**:
```python
class OllamaModelService:
    def get_available_models(self):
        """Fetch available models from Ollama"""
        response = requests.get(f"{ollama_url}/api/tags")
        models = response.json().get('models', [])

        return [{
            'name': m['name'],
            'display_name': m['name'].split(':')[0],
            'size_gb': m['size'] / 1e9,
            'context_length': m.get('details', {}).get('parameter_size', '8K'),
            'family': m.get('details', {}).get('family', 'unknown')
        } for m in models]
```

### 5. Multi-Document Support: YES
**Decision**: Design for multi-document from the start

**Database Changes**:
```python
class Conversation(models.Model):
    # ... existing fields ...

    # Filter to specific documents (null = search all documents)
    document_filter = models.ManyToManyField(
        'Document',
        blank=True,
        related_name='conversations',
        help_text="Limit search to these documents (empty = search all)"
    )
```

**Retrieval Integration**:
```python
def retrieve(self, query, document_ids=None, **kwargs):
    """Retrieve chunks, optionally filtered by document"""
    # Add document filter to SQL query
    doc_filter = ""
    if document_ids:
        doc_ids_str = ",".join(f"'{d}'" for d in document_ids)
        doc_filter = f"AND c.document_id IN ({doc_ids_str})"

    sql = f"""
    ...
    FROM genealogy_textchunk c
    WHERE ...
    {doc_filter}
    ...
    """
```

**UI Features**:
- Document selector in conversation settings
- Show which document each chunk came from
- Filter conversations by document
- Search across all documents by default

---

## Updated Implementation Details

### Phase 4B: Streaming Implementation

**New File**: `genealogy/views/chat_stream.py`

```python
from django.http import StreamingHttpResponse
from django.views.decorators.http import require_http_methods
import json
import time

@require_http_methods(["POST"])
def stream_message(request, conversation_id):
    """Stream LLM response token by token"""
    conversation = get_object_or_404(Conversation, id=conversation_id)
    user_message = request.POST.get('message', '').strip()
    selected_model = request.POST.get('model', 'aya:35b-23')

    if not user_message:
        return JsonResponse({'error': 'Empty message'}, status=400)

    # Save user message
    user_msg = Message.objects.create(
        conversation=conversation,
        role='user',
        content=user_message
    )

    def event_stream():
        try:
            # 1. Retrieve context (fast, do upfront)
            yield f"data: {json.dumps({'status': 'retrieving'})}\n\n"

            retriever = HybridRetriever()

            # Apply document filter if set
            doc_ids = list(conversation.document_filter.values_list('id', flat=True))
            chunks = retriever.retrieve(
                query=user_message,
                document_ids=doc_ids if doc_ids else None,
                top_k=5,
                expand_window=1
            )

            # Send chunk info to frontend
            yield f"data: {json.dumps({
                'status': 'retrieved',
                'chunks_count': len(chunks),
                'chunks': [{
                    'id': str(c['id']),
                    'page': c['start_page'],
                    'score': c.get('rrf_score', 0.0),
                    'preview': c['text_content'][:100]
                } for c in chunks]
            })}\n\n"

            # 2. Build context
            context = retriever.build_context(chunks, include_enrichment=True)

            # 3. Build prompt with conversation history
            history = conversation.messages.filter(
                created_at__lt=user_msg.created_at
            ).order_by('-created_at')[:10]  # Last 10 messages

            prompt = build_conversation_prompt(
                user_message=user_message,
                retrieved_context=context,
                previous_messages=list(reversed(history)),
                max_tokens=get_model_context_length(selected_model)
            )

            # 4. Stream generation
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"

            ollama = OllamaClient(timeout=300)
            full_response = ""

            for chunk in ollama.generate_stream(
                model=selected_model,
                prompt=prompt,
                options={'num_ctx': 32768, 'temperature': 0.1}
            ):
                token = chunk.get('response', '')
                if token:
                    full_response += token
                    yield f"data: {json.dumps({
                        'token': token,
                        'done': False
                    })}\n\n"

            # 5. Save assistant message
            assistant_msg = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=full_response,
                retrieved_chunks=[{
                    'id': str(c['id']),
                    'text': c['text_content'][:200],
                    'page': c['start_page'],
                    'score': c.get('rrf_score', 0.0),
                    'document_title': c.get('document__title', 'Unknown')
                } for c in chunks],
                retrieval_metadata={
                    'chunks_count': len(chunks),
                    'model_used': selected_model,
                    'document_ids': doc_ids if doc_ids else None
                }
            )

            # 6. Send completion
            yield f"data: {json.dumps({
                'done': True,
                'message_id': str(assistant_msg.id),
                'full_text': full_response
            })}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({
                'error': str(e),
                'done': True
            })}\n\n"

    return StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )

def get_model_context_length(model_name):
    """Get context length for model (with fallback)"""
    # Common models and their context lengths
    context_lengths = {
        'aya:35b-23': 32768,
        'llama2': 4096,
        'mistral': 32768,
        'gemma': 8192,
    }

    # Try to match model family
    for key, length in context_lengths.items():
        if key in model_name:
            return length

    # Default to conservative 8K
    return 8192
```

**Frontend (HTMX + JavaScript for SSE)**:

```html
<!-- In conversation.html -->
<script>
function sendMessage(conversationId, message, model) {
  const messagesDiv = document.getElementById('messages');

  // Add user message immediately
  appendMessage('user', message);

  // Create assistant message placeholder
  const assistantDiv = appendMessage('assistant', '');
  const contentSpan = assistantDiv.querySelector('.message-content');

  // Connect to SSE endpoint
  const eventSource = new EventSource(
    `/chat/${conversationId}/stream/?message=${encodeURIComponent(message)}&model=${model}`
  );

  let fullText = '';

  eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);

    if (data.status === 'retrieving') {
      contentSpan.innerHTML = '<em>🔍 Searching documents...</em>';
    }
    else if (data.status === 'retrieved') {
      contentSpan.innerHTML = `<em>📚 Found ${data.chunks_count} relevant chunks. Generating answer...</em>`;
    }
    else if (data.status === 'generating') {
      contentSpan.innerHTML = '';
    }
    else if (data.token) {
      fullText += data.token;
      contentSpan.textContent = fullText;

      // Auto-scroll to bottom
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
    else if (data.done) {
      if (data.error) {
        contentSpan.innerHTML = `<span class="text-red-600">Error: ${data.error}</span>`;
      }
      eventSource.close();

      // Add source chunks collapsible section
      if (data.chunks) {
        addSourceChunks(assistantDiv, data.chunks);
      }
    }
  });

  eventSource.addEventListener('error', () => {
    contentSpan.innerHTML = '<span class="text-red-600">Connection error. Please try again.</span>';
    eventSource.close();
  });
}
</script>
```

---

## Phase 5: Agentic Workflow for Complex Queries

### Overview: Multi-Turn Reasoning with Tool Calling

**Problem**: Some queries require iterative information gathering:
1. **Disambiguation** - Multiple people with same/similar names
2. **Relationship tracing** - Finding connections across multiple generations
3. **Multi-step reasoning** - LLM needs to request additional context dynamically

**Solution**: Implement agentic workflow where LLM can:
- Ask users for clarification (disambiguation)
- Call back to retrieval system for more context (tool use)
- Track search depth to prevent infinite loops
- Stream thinking process to user

### 5.1 Tool-Calling Architecture

Modern LLMs (like those in Ollama) support **function calling** / **tool use** where the model can:
1. Recognize it needs more information
2. Request a specific tool/function be called
3. Receive the results
4. Continue reasoning with new information

**Available Tools for LLM**:

```python
AVAILABLE_TOOLS = [
    {
        "name": "search_person_by_name",
        "description": "Search for people by name. Returns list of matching PersonMentions with disambiguating details.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full or partial name to search for (e.g., 'Pieter van Zanten')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_person_details",
        "description": "Get detailed information about a specific person by their genealogical ID or PersonMention ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "genealogical_id": {
                    "type": "string",
                    "description": "Genealogical identifier like 'II.3.a' or PersonMention UUID"
                }
            },
            "required": ["genealogical_id"]
        }
    },
    {
        "name": "find_relationship_path",
        "description": "Find genealogical relationship path between two people. Returns ancestors/descendants connecting them.",
        "parameters": {
            "type": "object",
            "properties": {
                "person1_id": {
                    "type": "string",
                    "description": "Genealogical ID or UUID of first person"
                },
                "person2_id": {
                    "type": "string",
                    "description": "Genealogical ID or UUID of second person"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum generations to search (default: 5)",
                    "default": 5
                }
            },
            "required": ["person1_id", "person2_id"]
        }
    },
    {
        "name": "get_ancestors",
        "description": "Get ancestors (parents, grandparents, etc.) of a person up to N generations.",
        "parameters": {
            "type": "object",
            "properties": {
                "person_id": {"type": "string"},
                "generations": {"type": "integer", "default": 2}
            },
            "required": ["person_id"]
        }
    },
    {
        "name": "get_descendants",
        "description": "Get descendants (children, grandchildren, etc.) of a person up to N generations.",
        "parameters": {
            "type": "object",
            "properties": {
                "person_id": {"type": "string"},
                "generations": {"type": "integer", "default": 2}
            },
            "required": ["person_id"]
        }
    },
    {
        "name": "search_by_criteria",
        "description": "Search for people by specific criteria (birth year, location, occupation, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "birth_year_min": {"type": "integer"},
                "birth_year_max": {"type": "integer"},
                "birth_place": {"type": "string"},
                "death_place": {"type": "string"},
                "occupation": {"type": "string"}
            }
        }
    }
]
```

### 5.2 Tool Implementation

**New File**: `genealogy/services/genealogy_tools.py`

```python
"""
Genealogy tools for LLM agentic workflows.

These tools allow the LLM to iteratively request information
to answer complex queries like relationship tracing and disambiguation.
"""

import logging
from typing import List, Dict, Optional
from django.db.models import Q
from genealogy.models import PersonMention, RelationshipMention, Event
from genealogy.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class GenealogyTools:
    """Tools for LLM to interact with genealogy database"""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.max_tool_calls = 10  # Prevent infinite loops

    def search_person_by_name(self, name: str, max_results: int = 10) -> Dict:
        """
        Search for people by name with disambiguating details.

        Returns:
            {
                "count": int,
                "people": [
                    {
                        "id": "uuid",
                        "genealogical_id": "II.3.a",
                        "name": "Pieter van Zanten",
                        "birth": {"date": "1845-03-12", "place": "Amsterdam"},
                        "death": {"date": "1920-11-03", "place": "Den Haag"},
                        "parents": ["Johannes van Zanten", "Maria de Vries"],
                        "generation": 2
                    }
                ]
            }
        """
        # Search PersonMentions
        people = PersonMention.objects.filter(
            Q(given_names__icontains=name) | Q(surname__icontains=name)
        ).select_related(
            'primary_chunks__document'
        ).prefetch_related(
            'events',
            'parent_relationships__parent_mention'
        )[:max_results]

        results = []
        for person in people:
            # Get birth/death events
            birth = person.events.filter(event_type='BIRT').first()
            death = person.events.filter(event_type='DEAT').first()

            # Get parents
            parents = [
                rel.parent_mention.full_name()
                for rel in person.parent_relationships.all()
            ]

            results.append({
                "id": str(person.id),
                "genealogical_id": person.genealogical_id,
                "name": person.full_name(),
                "birth": {
                    "date": birth.date.isoformat() if birth and birth.date else None,
                    "place": birth.place.name if birth and birth.place else None
                } if birth else None,
                "death": {
                    "date": death.date.isoformat() if death and death.date else None,
                    "place": death.place.name if death and death.place else None
                } if death else None,
                "parents": parents,
                "generation": self._extract_generation(person.genealogical_id),
                "chunk_preview": person.primary_chunks.first().text_content[:200] if person.primary_chunks.exists() else None
            })

        return {
            "count": len(results),
            "people": results,
            "truncated": people.count() > max_results
        }

    def get_person_details(self, genealogical_id: str) -> Dict:
        """Get detailed information about a specific person"""
        # Try genealogical_id first, then UUID
        person = PersonMention.objects.filter(
            Q(genealogical_id=genealogical_id) | Q(id=genealogical_id)
        ).prefetch_related(
            'events',
            'events__place',
            'parent_relationships__parent_mention',
            'child_relationships__child_mention',
            'partnerships__partners'
        ).first()

        if not person:
            return {"error": f"Person not found: {genealogical_id}"}

        # Get all events
        events = [{
            "type": e.event_type,
            "date": e.date.isoformat() if e.date else None,
            "place": e.place.name if e.place else None,
            "description": e.description
        } for e in person.events.all()]

        # Get relationships
        parents = [
            {"id": str(r.parent_mention.id), "name": r.parent_mention.full_name()}
            for r in person.parent_relationships.all()
        ]
        children = [
            {"id": str(r.child_mention.id), "name": r.child_mention.full_name()}
            for r in person.child_relationships.all()
        ]
        partners = []
        for partnership in person.partnerships.all():
            other_partners = [p for p in partnership.partners.all() if p.id != person.id]
            partners.extend([
                {"id": str(p.id), "name": p.full_name()}
                for p in other_partners
            ])

        return {
            "id": str(person.id),
            "genealogical_id": person.genealogical_id,
            "name": person.full_name(),
            "events": events,
            "parents": parents,
            "children": children,
            "partners": partners,
            "generation": self._extract_generation(person.genealogical_id)
        }

    def find_relationship_path(
        self,
        person1_id: str,
        person2_id: str,
        max_depth: int = 5
    ) -> Dict:
        """
        Find relationship path between two people using BFS.

        Returns:
            {
                "found": bool,
                "path": [
                    {"id": "...", "name": "...", "relation": "parent_of"},
                    {"id": "...", "name": "...", "relation": "child_of"},
                    ...
                ],
                "relationship_description": "Second cousin",
                "degrees_of_separation": 4
            }
        """
        from collections import deque

        person1 = self._get_person(person1_id)
        person2 = self._get_person(person2_id)

        if not person1 or not person2:
            return {"error": "One or both people not found", "found": False}

        if person1.id == person2.id:
            return {
                "found": True,
                "path": [{"id": str(person1.id), "name": person1.full_name(), "relation": "self"}],
                "relationship_description": "Same person",
                "degrees_of_separation": 0
            }

        # BFS to find shortest path
        queue = deque([(person1, [])])
        visited = {person1.id}
        depth = 0

        while queue and depth < max_depth:
            current_size = len(queue)

            for _ in range(current_size):
                current, path = queue.popleft()

                # Get all related people
                related = self._get_related_people(current)

                for person, relation_type in related:
                    if person.id in visited:
                        continue

                    new_path = path + [{
                        "id": str(current.id),
                        "name": current.full_name(),
                        "relation": relation_type
                    }]

                    if person.id == person2.id:
                        # Found target!
                        final_path = new_path + [{
                            "id": str(person2.id),
                            "name": person2.full_name(),
                            "relation": "target"
                        }]

                        return {
                            "found": True,
                            "path": final_path,
                            "relationship_description": self._describe_relationship(final_path),
                            "degrees_of_separation": len(final_path) - 1
                        }

                    visited.add(person.id)
                    queue.append((person, new_path))

            depth += 1

        return {
            "found": False,
            "message": f"No relationship found within {max_depth} generations",
            "degrees_of_separation": None
        }

    def get_ancestors(self, person_id: str, generations: int = 2) -> Dict:
        """Get ancestors up to N generations"""
        person = self._get_person(person_id)
        if not person:
            return {"error": "Person not found"}

        ancestors = []
        self._collect_ancestors(person, generations, 1, ancestors)

        return {
            "person": {"id": str(person.id), "name": person.full_name()},
            "ancestors": ancestors,
            "count": len(ancestors)
        }

    def get_descendants(self, person_id: str, generations: int = 2) -> Dict:
        """Get descendants up to N generations"""
        person = self._get_person(person_id)
        if not person:
            return {"error": "Person not found"}

        descendants = []
        self._collect_descendants(person, generations, 1, descendants)

        return {
            "person": {"id": str(person.id), "name": person.full_name()},
            "descendants": descendants,
            "count": len(descendants)
        }

    def search_by_criteria(
        self,
        birth_year_min: Optional[int] = None,
        birth_year_max: Optional[int] = None,
        birth_place: Optional[str] = None,
        death_place: Optional[str] = None,
        occupation: Optional[str] = None
    ) -> Dict:
        """Search people by various criteria"""
        queryset = PersonMention.objects.all()

        # Build filters
        if birth_year_min or birth_year_max:
            birth_events = Event.objects.filter(event_type='BIRT')
            if birth_year_min:
                birth_events = birth_events.filter(date__year__gte=birth_year_min)
            if birth_year_max:
                birth_events = birth_events.filter(date__year__lte=birth_year_max)
            queryset = queryset.filter(id__in=birth_events.values_list('mention_id', flat=True))

        if birth_place:
            birth_events = Event.objects.filter(
                event_type='BIRT',
                place__name__icontains=birth_place
            )
            queryset = queryset.filter(id__in=birth_events.values_list('mention_id', flat=True))

        if death_place:
            death_events = Event.objects.filter(
                event_type='DEAT',
                place__name__icontains=death_place
            )
            queryset = queryset.filter(id__in=death_events.values_list('mention_id', flat=True))

        # Limit results
        people = queryset[:20]

        return {
            "count": people.count(),
            "people": [
                {
                    "id": str(p.id),
                    "genealogical_id": p.genealogical_id,
                    "name": p.full_name()
                }
                for p in people
            ]
        }

    # Helper methods

    def _get_person(self, person_id: str) -> Optional[PersonMention]:
        """Get person by genealogical_id or UUID"""
        return PersonMention.objects.filter(
            Q(genealogical_id=person_id) | Q(id=person_id)
        ).first()

    def _extract_generation(self, genealogical_id: Optional[str]) -> Optional[int]:
        """Extract generation number from genealogical ID (e.g., 'II.3.a' -> 2)"""
        if not genealogical_id:
            return None
        try:
            roman = genealogical_id.split('.')[0]
            roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8}
            return roman_map.get(roman)
        except:
            return None

    def _get_related_people(self, person: PersonMention) -> List[tuple]:
        """Get all people related to this person (parents, children, spouses)"""
        related = []

        # Parents
        for rel in person.parent_relationships.select_related('parent_mention').all():
            related.append((rel.parent_mention, 'parent_of'))

        # Children
        for rel in person.child_relationships.select_related('child_mention').all():
            related.append((rel.child_mention, 'child_of'))

        # Spouses
        for partnership in person.partnerships.prefetch_related('partners').all():
            for partner in partnership.partners.all():
                if partner.id != person.id:
                    related.append((partner, 'spouse_of'))

        return related

    def _collect_ancestors(self, person: PersonMention, max_gen: int, current_gen: int, result: list):
        """Recursively collect ancestors"""
        if current_gen > max_gen:
            return

        for rel in person.parent_relationships.select_related('parent_mention').all():
            parent = rel.parent_mention
            result.append({
                "id": str(parent.id),
                "genealogical_id": parent.genealogical_id,
                "name": parent.full_name(),
                "generation": current_gen,
                "relation": "parent" if current_gen == 1 else f"{current_gen}x great-grandparent"
            })
            self._collect_ancestors(parent, max_gen, current_gen + 1, result)

    def _collect_descendants(self, person: PersonMention, max_gen: int, current_gen: int, result: list):
        """Recursively collect descendants"""
        if current_gen > max_gen:
            return

        for rel in person.child_relationships.select_related('child_mention').all():
            child = rel.child_mention
            result.append({
                "id": str(child.id),
                "genealogical_id": child.genealogical_id,
                "name": child.full_name(),
                "generation": current_gen,
                "relation": "child" if current_gen == 1 else f"{current_gen}x great-grandchild"
            })
            self._collect_descendants(child, max_gen, current_gen + 1, result)

    def _describe_relationship(self, path: List[Dict]) -> str:
        """Convert relationship path to human-readable description"""
        if len(path) <= 1:
            return "self"

        # Count parent/child hops
        up_count = sum(1 for p in path if p['relation'] == 'parent_of')
        down_count = sum(1 for p in path if p['relation'] == 'child_of')

        if up_count == 0 and down_count == 1:
            return "child"
        if up_count == 1 and down_count == 0:
            return "parent"
        if up_count == 0 and down_count == 2:
            return "grandchild"
        if up_count == 2 and down_count == 0:
            return "grandparent"
        if up_count == 1 and down_count == 1:
            return "sibling (via common parent)"

        # More complex relationships
        if up_count == down_count and up_count >= 2:
            degree = up_count - 1
            if degree == 1:
                return "first cousin"
            elif degree == 2:
                return "second cousin"
            else:
                return f"{degree}th cousin"

        # Generational removal
        if up_count > down_count:
            gen_diff = up_count - down_count
            if gen_diff == 1:
                return f"great-aunt/uncle"
            else:
                return f"{gen_diff}x great-aunt/uncle"
        else:
            gen_diff = down_count - up_count
            if gen_diff == 1:
                return f"niece/nephew"
            else:
                return f"{gen_diff}x great-niece/nephew"

        return f"{up_count} generations up, {down_count} generations down"
```

### 5.3 Agentic Execution Loop

**New File**: `genealogy/services/agent_executor.py`

```python
"""
Agentic execution loop for multi-turn LLM reasoning.

Allows LLM to iteratively call tools until it has enough information
to answer the user's question.
"""

import json
import logging
from typing import Dict, List, Generator
from genealogy.ollama_utils import OllamaClient
from genealogy.services.genealogy_tools import GenealogyTools, AVAILABLE_TOOLS

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Execute agentic workflow with tool calling"""

    def __init__(self, model: str = "aya:35b-23", max_iterations: int = 10):
        self.model = model
        self.max_iterations = max_iterations
        self.ollama = OllamaClient(timeout=300)
        self.tools = GenealogyTools()

    def execute_stream(
        self,
        user_query: str,
        initial_context: str,
        conversation_history: List[Dict]
    ) -> Generator[Dict, None, None]:
        """
        Execute agentic workflow with streaming output.

        Yields status updates and tokens as they're generated.
        """
        iteration = 0
        context_parts = [initial_context]  # Start with RAG-retrieved context
        tool_calls_made = []

        while iteration < self.max_iterations:
            iteration += 1

            # Build prompt with accumulated context
            prompt = self._build_agent_prompt(
                user_query=user_query,
                context="\n\n".join(context_parts),
                conversation_history=conversation_history,
                tool_calls_made=tool_calls_made
            )

            # Stream LLM response
            yield {
                "type": "status",
                "status": "thinking",
                "iteration": iteration,
                "message": f"Reasoning (step {iteration}/{self.max_iterations})..."
            }

            full_response = ""
            for chunk in self.ollama.generate_stream(
                model=self.model,
                prompt=prompt,
                format='json',
                options={'num_ctx': 32768, 'temperature': 0.1}
            ):
                token = chunk.get('response', '')
                full_response += token

            # Parse response
            try:
                response_data = json.loads(full_response)
            except json.JSONDecodeError:
                # If JSON parse fails, treat as final answer
                yield {
                    "type": "answer",
                    "content": full_response,
                    "tool_calls": tool_calls_made
                }
                return

            # Check if LLM wants to call a tool
            if response_data.get("action") == "call_tool":
                tool_name = response_data.get("tool")
                tool_args = response_data.get("arguments", {})

                yield {
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": tool_args,
                    "reasoning": response_data.get("reasoning", "")
                }

                # Execute tool
                tool_result = self._execute_tool(tool_name, tool_args)
                tool_calls_made.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                })

                # Add tool result to context
                context_parts.append(f"TOOL RESULT ({tool_name}):\n{json.dumps(tool_result, indent=2)}")

                # Continue loop to next iteration

            elif response_data.get("action") == "ask_user":
                # LLM needs user clarification
                yield {
                    "type": "clarification_needed",
                    "question": response_data.get("question"),
                    "options": response_data.get("options", []),
                    "reasoning": response_data.get("reasoning", "")
                }
                return  # Wait for user response

            elif response_data.get("action") == "answer":
                # LLM has final answer
                yield {
                    "type": "answer",
                    "content": response_data.get("answer"),
                    "tool_calls": tool_calls_made,
                    "confidence": response_data.get("confidence", "medium")
                }
                return

            else:
                # Unknown action, return as-is
                yield {
                    "type": "answer",
                    "content": full_response,
                    "tool_calls": tool_calls_made
                }
                return

        # Max iterations reached
        yield {
            "type": "error",
            "message": f"Maximum iterations ({self.max_iterations}) reached without finding answer",
            "tool_calls": tool_calls_made
        }

    def _build_agent_prompt(
        self,
        user_query: str,
        context: str,
        conversation_history: List[Dict],
        tool_calls_made: List[Dict]
    ) -> str:
        """Build prompt for agentic workflow"""

        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in AVAILABLE_TOOLS
        ])

        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-5:]  # Last 5 messages
        ])

        previous_tools = "\n".join([
            f"Called {call['tool']} with {call['arguments']}"
            for call in tool_calls_made
        ])

        return f"""You are a genealogy research assistant with access to tools to explore family relationships.

CONVERSATION HISTORY:
{history_text}

CURRENT CONTEXT:
{context}

PREVIOUS TOOL CALLS:
{previous_tools if tool_calls_made else "None"}

USER QUERY: {user_query}

AVAILABLE TOOLS:
{tools_description}

INSTRUCTIONS:
1. If multiple people match the query, ask the user for clarification with specific details
2. If you need more information, call appropriate tools to gather it
3. When tracing relationships, call tools iteratively to find connections
4. Stop when you have sufficient information to answer confidently
5. **IMPORTANT**: If you've made {len(tool_calls_made)}/10 tool calls, prioritize giving an answer with what you have

RESPONSE FORMAT (JSON):
{{
  "action": "call_tool" | "ask_user" | "answer",
  "reasoning": "Why you're taking this action",

  // If action = "call_tool":
  "tool": "tool_name",
  "arguments": {{"arg1": "value1"}},

  // If action = "ask_user":
  "question": "Which Pieter van Zanten do you mean?",
  "options": [
    {{"id": "II.3.a", "label": "Pieter van Zanten (b. 1845, Amsterdam)"}},
    {{"id": "III.5.b", "label": "Pieter van Zanten (b. 1880, Rotterdam)"}}
  ],

  // If action = "answer":
  "answer": "Your final answer here",
  "confidence": "high" | "medium" | "low"
}}

JSON:"""

    def _execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool and return results"""
        try:
            method = getattr(self.tools, tool_name, None)
            if not method:
                return {"error": f"Unknown tool: {tool_name}"}

            result = method(**arguments)
            return result

        except Exception as e:
            logger.error(f"Tool execution error: {tool_name} - {e}")
            return {"error": str(e)}
```

### 5.4 Integration with Streaming Chat

**Update**: `genealogy/views/chat_stream.py`

```python
@require_http_methods(["POST"])
def stream_message_with_agent(request, conversation_id):
    """Stream LLM response with agentic tool calling"""
    conversation = get_object_or_404(Conversation, id=conversation_id)
    user_message = request.POST.get('message', '').strip()
    selected_model = request.POST.get('model', 'aya:35b-23')
    use_agent = request.POST.get('use_agent', 'true').lower() == 'true'  # Default: enabled

    # Save user message
    user_msg = Message.objects.create(
        conversation=conversation,
        role='user',
        content=user_message
    )

    def event_stream():
        try:
            # 1. Initial retrieval
            yield f"data: {json.dumps({'status': 'retrieving'})}\n\n"

            retriever = HybridRetriever()
            doc_ids = list(conversation.document_filter.values_list('id', flat=True))
            chunks = retriever.retrieve(
                query=user_message,
                document_ids=doc_ids if doc_ids else None,
                top_k=5,
                expand_window=1
            )

            yield f"data: {json.dumps({'status': 'retrieved', 'chunks_count': len(chunks)})}\n\n"

            initial_context = retriever.build_context(chunks, include_enrichment=True)

            # 2. Decide: Use agent or simple generation?
            # Use agent for: relationship queries, disambiguation needs, multi-step reasoning
            if use_agent and _should_use_agent(user_message):
                yield f"data: {json.dumps({'status': 'agent_mode', 'message': 'Using agentic workflow'})}\n\n"

                # Get conversation history
                history = conversation.messages.filter(
                    created_at__lt=user_msg.created_at
                ).order_by('-created_at')[:10]

                history_list = [
                    {"role": msg.role, "content": msg.content}
                    for msg in reversed(history)
                ]

                # Execute agent
                agent = AgentExecutor(model=selected_model, max_iterations=10)
                tool_calls_log = []

                for event in agent.execute_stream(
                    user_query=user_message,
                    initial_context=initial_context,
                    conversation_history=history_list
                ):
                    if event['type'] == 'status':
                        yield f"data: {json.dumps(event)}\n\n"

                    elif event['type'] == 'tool_call':
                        tool_calls_log.append(event)
                        yield f"data: {json.dumps({
                            'type': 'tool_call',
                            'tool': event['tool'],
                            'reasoning': event['reasoning']
                        })}\n\n"

                    elif event['type'] == 'clarification_needed':
                        # Save as assistant message with special format
                        clarification_text = f"{event['question']}\n\n"
                        for i, opt in enumerate(event['options'], 1):
                            clarification_text += f"{i}. {opt['label']}\n"

                        Message.objects.create(
                            conversation=conversation,
                            role='assistant',
                            content=clarification_text,
                            retrieval_metadata={
                                'type': 'clarification',
                                'options': event['options']
                            }
                        )

                        yield f"data: {json.dumps({
                            'type': 'clarification',
                            'content': clarification_text,
                            'done': True
                        })}\n\n"
                        return

                    elif event['type'] == 'answer':
                        # Stream final answer
                        answer = event['content']
                        Message.objects.create(
                            conversation=conversation,
                            role='assistant',
                            content=answer,
                            retrieved_chunks=[{
                                'id': str(c['id']),
                                'page': c['start_page'],
                                'text': c['text_content'][:200]
                            } for c in chunks],
                            retrieval_metadata={
                                'tool_calls': tool_calls_log,
                                'confidence': event.get('confidence', 'medium')
                            }
                        )

                        yield f"data: {json.dumps({
                            'type': 'answer',
                            'content': answer,
                            'tool_calls_count': len(tool_calls_log),
                            'done': True
                        })}\n\n"
                        return

            else:
                # Simple generation (existing logic)
                yield f"data: {json.dumps({'status': 'generating'})}\n\n"
                # ... existing streaming logic ...

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


def _should_use_agent(query: str) -> bool:
    """Determine if query needs agentic workflow"""
    agent_keywords = [
        'how are', 'related', 'relationship', 'connection',
        'ancestor', 'descendant', 'cousin', 'sibling',
        'who was', 'tell me about', 'where did', 'when did',
        'multiple', 'which one', 'disambiguation'
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in agent_keywords)
```

### 5.5 Frontend Updates for Agentic Workflow

**Update**: `conversation.html`

```html
<script>
function sendMessage(conversationId, message, model) {
  const messagesDiv = document.getElementById('messages');
  const assistantDiv = appendMessage('assistant', '');
  const contentSpan = assistantDiv.querySelector('.message-content');

  const eventSource = new EventSource(
    `/chat/${conversationId}/stream/?message=${encodeURIComponent(message)}&model=${model}`
  );

  let toolCallsDiv = null;

  eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);

    if (data.status === 'agent_mode') {
      contentSpan.innerHTML = '<em>🤖 Using multi-step reasoning...</em>';
    }
    else if (data.type === 'tool_call') {
      // Show tool being called
      if (!toolCallsDiv) {
        toolCallsDiv = document.createElement('div');
        toolCallsDiv.className = 'tool-calls mt-2 text-xs text-gray-600';
        assistantDiv.appendChild(toolCallsDiv);
      }

      const toolDiv = document.createElement('div');
      toolDiv.className = 'tool-call p-2 bg-blue-50 border-l-2 border-blue-400 mb-1';
      toolDiv.innerHTML = `
        <strong>🔧 ${data.tool}</strong><br>
        <em>${data.reasoning}</em>
      `;
      toolCallsDiv.appendChild(toolDiv);
    }
    else if (data.type === 'clarification') {
      // LLM asking for clarification
      contentSpan.textContent = data.content;

      // Add clickable options
      const optionsDiv = document.createElement('div');
      optionsDiv.className = 'clarification-options mt-3';
      // ... add clickable buttons for each option ...

      eventSource.close();
    }
    else if (data.type === 'answer') {
      contentSpan.textContent = data.content;

      if (data.tool_calls_count > 0) {
        const badge = document.createElement('span');
        badge.className = 'badge bg-blue-100 text-blue-800 text-xs ml-2';
        badge.textContent = `${data.tool_calls_count} tool calls`;
        contentSpan.appendChild(badge);
      }

      eventSource.close();
    }
    // ... rest of existing event handlers ...
  });
}
</script>
```

### 5.6 Safety Mechanisms

**Prevent Infinite Loops**:

1. **Max iterations limit**: Hard cap at 10 tool calls
2. **Visited people tracking**: Don't re-query same person twice
3. **Max relationship depth**: Stop BFS at 5 generations
4. **Timeout**: Overall agent execution timeout of 5 minutes
5. **Cost tracking**: Log token usage per iteration

**Implementation**:

```python
class AgentExecutor:
    def __init__(self, ...):
        self.max_iterations = 10
        self.visited_entities = set()  # Track queried entities
        self.total_tokens = 0
        self.start_time = time.time()

    def execute_stream(self, ...):
        while iteration < self.max_iterations:
            # Check timeout
            if time.time() - self.start_time > 300:  # 5 minutes
                yield {"type": "error", "message": "Timeout reached"}
                return

            # Check token budget (if Ollama provides usage)
            if self.total_tokens > 50000:
                yield {"type": "error", "message": "Token budget exceeded"}
                return

            # ... rest of loop ...
```

---

## Conclusion

This plan provides a complete roadmap from fixing the existing RAG+RRF system to delivering a production-ready chat interface with streaming responses, model selection, multi-document support, **and advanced agentic capabilities for complex genealogical queries**.

**Updated Key Features**:
- ✅ **Streaming responses** - Real-time token-by-token generation
- ✅ **Anonymous sessions** - No login required, easy to add auth later
- ✅ **Smart context management** - Handles conversation history within token limits
- ✅ **Model selection** - Dynamic model discovery from Ollama
- ✅ **Multi-document search** - Filter by document or search across all
- ✅ **Agentic workflow** - Multi-turn reasoning with tool calling for:
  - **Disambiguation** - Handle multiple people with same names
  - **Relationship tracing** - Find connections across generations iteratively
  - **Complex queries** - LLM requests additional context as needed
  - **Safety mechanisms** - Prevent infinite loops and runaway costs

**Technical Stack**:
- Server-Sent Events (SSE) for streaming
- Django sessions for anonymous users
- Ollama streaming API with JSON tool calling
- HTMX + JavaScript for SSE client
- Document filtering via M2M relationship
- **GenealogyTools service** - 7 tools for genealogical research
- **AgentExecutor** - Multi-turn reasoning loop with safeguards
- **BFS algorithm** - Shortest path finding for relationships

**Example Workflows**:

1. **Disambiguation**:
   ```
   User: "Tell me about Pieter van Zanten"
   → Agent searches, finds 5 Pieters
   → Asks user: "Which Pieter? [lists with birth dates, places, parents]"
   → User selects: "II.3.a"
   → Agent retrieves full details and answers
   ```

2. **Relationship Tracing**:
   ```
   User: "How are Eugene van Zanten and Geertruij Voorhaar related?"
   → Initial RAG retrieval gets chunks for both
   → Agent recognizes they're 3 generations apart
   → Tool call: get_ancestors(Eugene, 3)
   → Tool call: get_descendants(Geertruij, 3)
   → Tool call: find_relationship_path(Eugene, Geertruij)
   → Returns: "Eugene is Geertruij's great-great-grandson through..."
   ```

3. **Location Query with Disambiguation**:
   ```
   User: "Where did Pieter van Zanten live when he was young?"
   → Agent searches, finds 5 Pieters
   → Asks for clarification
   → User selects specific Pieter
   → Agent calls get_person_details(person_id)
   → Filters events for childhood (age < 18)
   → Returns: "He lived in Amsterdam until age 15, then Rotterdam"
   ```

Ready to start implementation! 🚀
