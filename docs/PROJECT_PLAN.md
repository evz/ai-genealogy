# Genealogy Extractor - Project Plan

## Project Overview
AI-powered genealogy digitization application that processes Dutch family history books using OCR and LLM technology to extract structured family data.

**Core Features:**
- PDF OCR processing → structured genealogy data
- GEDCOM format export
- Wiki page generation for entities
- Free-form document queries
- Genealogical research question generation

**Tech Stack:**
- Django (web interface, data models)
- Celery (background processing)
- PostgreSQL + pgvector (data storage, vector search)
- Redis (Celery backend)
- Docker (containerization)
- Ollama (local LLM at the-area.local)

## Development Phases

### Phase 1: Foundation (Start Simple)
**Goal**: Basic Django app that can accept PDFs and store genealogy data

1. **Django Models** - Simple data models (Person, Family, Place, Event, Document)
2. **Upload Interface** - Basic form to upload PDF files
3. **Document Storage** - Simple file handling and database storage

### Phase 2: Core Processing Pipeline
**Goal**: Transform PDFs into searchable text

3. **OCR Processing** - Celery task for PDF → text extraction
4. **Database Setup** - PostgreSQL + pgvector for text storage and vector search
5. **Text Chunking** - Split OCR text with genealogical anchors (IDs, dates, names)

### Phase 3: AI-Powered Extraction
**Goal**: Extract structured data from text

5. **Ollama Integration** - Connect to your local LLM server
6. **Entity Extraction** - Extract Person/Family/Place data from text chunks
7. **Relationship Mapping** - Build connections between entities

### Phase 4: Output Generation
**Goal**: Provide useful formats for genealogy work

6. **GEDCOM Export** - Standard genealogy format output
7. **Q&A Interface** - Simple form to query document content
8. **Wiki Pages** - Basic templates for Person/Family/Place pages

### Phase 5: Development Infrastructure
**Goal**: Reliable development and deployment

9. **Docker Setup** - Containerize for easy local development
10. **Testing Framework** - Following your two-layer integration testing pattern

## Design Principles

Based on lessons learned from previous over-engineered attempt:

### Simplicity First
- Start with Django models containing business logic, not separate service layers
- Use simple views that call models/tasks directly
- Add abstraction only when you have 3+ similar cases
- Avoid repository pattern unless multiple data sources exist
- Functions over classes when possible

### Testing Strategy
- Mock only external systems (Ollama API, file system) in tests
- Build real user workflows from upload → processing → output
- Use two-layer integration testing:
  1. Service layer tests with real database
  2. Blueprint tests with mocked services
- Fix bugs when found, don't document them as "expected behavior"

### Architecture Patterns
- **Routes** → **Models/Tasks** (skip service layers unless truly needed)
- Business logic in models or simple functions
- No premature abstractions or "enterprise patterns"
- Optimize for understandability over architectural purity

## RAG Pipeline Design

"Anchor-aware, local RAG pipeline":
1. Clean OCR text
2. Split into small chunks
3. Attach deterministic anchors (genealogical IDs like II.1.a, page/time, names/dates/places)
4. Store text + embeddings + trigram + phonetic keys in PostgreSQL
5. At query time: hybrid retriever (vectors + trigram/BM25 + Daitch–Mokotoff)
6. Fuse with RRF, pull candidates, expand to adjacent chunks within same anchor
7. Optionally traverse lightweight relationship graph
8. LLM answers over curated context, keeping same-name individuals separate

## Current Status
- Fresh Django project created
- Ready to begin Phase 1: Foundation

## Next Steps
1. Set up basic Django models for genealogy data
2. Create simple upload interface for PDF documents
3. Implement basic document storage
