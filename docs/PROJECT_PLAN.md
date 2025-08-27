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

## Current Status (August 2025)

### ✅ Phase 1: Foundation - COMPLETED
- Django models for genealogy data (Person, Place, Partnership, Event, ParentChildRelationship)
- Document and DocumentPage models for file handling
- Admin interface for data management
- PostgreSQL with UUID primary keys and proper relationships

### 🔄 Phase 2: Core Processing Pipeline - TESTING & REFINEMENT
- Multi-format OCR processing (PDF, JPG, PNG, TIFF) with Tesseract
- Multi-language support (English/Dutch)
- Celery task processing with Redis backend
- Batch upload functionality via Django admin
- Page-by-page processing with confidence scoring and rotation correction
- **Current focus**: Resolving OCR quality issues and rotation detection edge cases

### 🔄 Phase 3: AI-Powered Extraction - IN PROGRESS

#### ✅ Completed Components:
1. **Smart Text Chunking System**
   - Generation-aware chunking (splits on "EERSTE GENERATIE", "TWEEDE GENERATIE", etc.)
   - Family context tracking for ID inference
   - OCR error correction for genealogical IDs
   - TextChunk model with PostgreSQL arrays for anchor storage

2. **Genealogical ID Processing**
   - Automatic correction of OCR misreadings (IL→II, XIL→XII, VIL→VII, etc.)
   - Family group detection ("X.9. Children of..." / "X.9. Kinderen van...")
   - Individual ID inference (a. Name → X.9.a when in family X.9 context)
   - Support for both English and Dutch patterns

3. **Ollama Integration & Model Configuration**
   - Docker mDNS resolution for `the-area.local` → IP conversion
   - Configurable LLM and embedding models via environment variables
   - Model discovery and validation through Ollama API
   - Document-level tracking of which models were used

4. **Multi-Phase Extraction Architecture**
   - Phase 1: Create text chunks with genealogical anchors
   - Phase 2: Extract entities from chunks (placeholder ready)
   - Admin action triggers both phases sequentially
   - Robust error handling and progress tracking

#### 🔄 Advanced Extraction Methods Implemented:

5. **Neural Network Named Entity Recognition (NER)**
   - Named Entity Recognition (NER): machine learning approach to identify and classify genealogical entities in text
   - Custom BIO (Beginning-Inside-Outside) tagging scheme for entities: PERSON_NAME, DATE, PLACE, GENEALOGY_ID, FAMILY_GROUP
   - BERT-based (Bidirectional Encoder Representations from Transformers) architecture fine-tuned for Dutch/English genealogy text
   - Achieved 96.84% overall F1 score (harmonic mean of precision and recall) across all entity types
   - Training data generation from existing regex extractions with quality validation and manual review
   - Confidence-based filtering (0.9+ threshold) for production deployment

6. **Dual Extraction Pipeline with Method Tracking**
   - Hybrid approach: traditional regex patterns + neural network NER processing
   - Extraction method tracking in database for performance comparison and validation
   - Visual admin interface to compare regex vs. neural network results side-by-side
   - Fallback logic: neural network primary, regex backup for low-confidence predictions

7. **Enhanced OCR with Advanced Rotation Detection**
   - Two-stage rotation correction: major angle detection (0°/90°/180°/270°) + fine-angle adjustments (±10°)
   - Computer vision techniques: Hough line detection, projection profile analysis, Canny edge detection
   - Addresses complex rotation issues where pages need 182.3° correction instead of simple 180°
   - OpenCV integration with morphological operations for robust text line detection
   - Recovers previously missing OCR content from improperly rotated genealogy book pages

8. **Comprehensive Date Standardization System**
   - Multi-format Dutch/English date parsing: "15 maart 1654" → "1654-03-15" (ISO format)
   - Pattern recognition for genealogical date conventions: dot notation (22.12.1658), word dates, partial dates ("maart 1654"), circa dates ("circa 1680")
   - Integration into extraction pipeline for consistent temporal data representation
   - Handles uncertainty markers and genealogical date range expressions

9. **Gold Standard Curation Workflow**
   - Manual correction interface in Django admin for refining genealogical anchor extraction
   - Custom PostgreSQL ArrayField implementation with comma-separated input/display for user-friendly editing
   - Structured correction fields: person_names, dates, places, genealogy_ids, family_groups
   - Manual review tracking system for training data quality assurance and neural network improvement
   - Organized fieldset layout optimized for efficient genealogical data curation

#### 🔄 Currently Active:
- **OCR Quality Optimization**: Testing enhanced rotation detection on problematic pages
- **Training Data Refinement**: Manual curation of genealogical anchors for improved neural network performance
- **Method Performance Analysis**: Comparing extraction accuracy between regex and neural network approaches

#### ⏳ Next Phase Priorities:
- **Relationship Inference Engine**: Parse genealogical IDs to automatically create parent-child relationship records
- **Entity Deduplication**: Merge similar persons/places across text chunks using fuzzy matching
- **LLM Integration**: Convert refined text chunks into structured Person/Place/Partnership/Event database records

### ⏳ Phase 4: RAG & Query System - PENDING
- Text embeddings generation and storage
- Vector search with pgvector
- Hybrid retrieval (semantic + keyword + phonetic)
- Natural language Q&A interface

### ⏳ Phase 5: Output Generation - PENDING
- GEDCOM export functionality
- Wiki page generation for entities
- Research question generation

## Technical Architecture

### Database Schema
```
Document (title, languages, models_used)
├── DocumentPage (image_file, ocr_text, confidence)
├── TextChunk (text_content, genealogy_ids[], generation_number)
├── Person (names, dates, places, genealogical_id)
├── Place (name, locality, region, country)
├── Partnership (partners, dates, type)
├── Event (type, person, date, place, description)
└── ParentChildRelationship (child, parent, type)
```

### Processing Pipeline
1. **Upload** → Document + DocumentPages created
2. **OCR** → Tesseract extracts text with confidence scores
3. **Chunking** → Smart segmentation with genealogical anchors
4. **Extraction** → LLM processes chunks → entities
5. **Relationships** → Genealogical IDs → parent/child links

### Key Technical Innovations

#### OCR Processing & Quality Enhancement
- **Advanced Rotation Detection**: Two-stage computer vision approach using Hough line detection, projection profiles, and OpenCV morphological operations to handle complex page orientations beyond simple 90° increments
- **OCR Correction Mapping**: Systematic correction of Roman numeral OCR errors (IL→II, XIL→XII, VIL→VII) based on empirical analysis of genealogical text patterns
- **Multi-Format Processing Pipeline**: Unified handling of PDF, JPG, PNG, TIFF with Tesseract OCR and confidence scoring

#### Intelligent Text Processing
- **Generation-Aware Chunking**: Preserves genealogical document structure by segmenting on generation headers ("EERSTE GENERATIE", "TWEEDE GENERATIE")
- **Family Context Tracking**: Infers individual genealogical IDs (e.g., "a. John" → "X.9.a") from family group context ("X.9. Children of...")
- **Multi-Language Pattern Recognition**: Handles both Dutch ("Kinderen van") and English ("Children of") genealogical conventions

#### Machine Learning & Entity Recognition
- **Domain-Specific NER Architecture**: Custom BERT fine-tuning for genealogical Named Entity Recognition with BIO tagging scheme
- **Hybrid Extraction Strategy**: Combines rule-based regex patterns with neural network predictions, including confidence-based method selection
- **Training Data Quality Pipeline**: Automated generation and manual curation of training examples with built-in validation and review tracking

#### Data Standardization & Quality Control
- **Comprehensive Date Parsing**: Multi-format Dutch/English date standardization handling genealogical conventions ("circa 1680", "15 maart 1654")
- **PostgreSQL Array Field Management**: Custom form fields for user-friendly comma-separated input/display of genealogical anchor data
- **Method Performance Tracking**: Database-level tracking of extraction methods (regex vs. neural network) for performance analysis and validation

## Configuration

### Environment Variables (.env)
```bash
# Database
DB_HOST=localhost  # (resolved from the-area.local in Docker)
DB_NAME=genealogy_extractor
DB_USER=postgres
DB_PASSWORD=postgres

# Ollama Models
OLLAMA_HOST=the-area.local
OLLAMA_PORT=11434
OLLAMA_LLM_MODEL=aya:35b-23
OLLAMA_EMBEDDING_MODEL=zylonai/multilingual-e5-large:latest
```

### Docker Development
```bash
make up-build    # Resolves mDNS and starts services
make demo        # Complete OCR demo with sample files
make shell       # Django shell access
```

## Current Development Challenges & Focus Areas

### OCR Quality & Data Recovery
**Challenge**: Complex document rotations causing missing genealogical content
- **Specific Issue**: Page 16 missing "e. Hendrikje [Heintje] van Zanten" due to 182.3° rotation (not simple 180°)
- **Solution Implemented**: Two-stage rotation detection with computer vision techniques
- **Current Status**: Enhanced rotation detection complete, testing on problematic pages
- **Trade-off Consideration**: Reprocessing pages invalidates existing manual curation work on text chunks

### Neural Network vs. Traditional Extraction
**Challenge**: Balancing accuracy improvements with system complexity
- **Current Performance**: Neural network NER achieving 96.84% F1 score vs. regex baseline
- **Implementation**: Dual extraction pipeline with method tracking for validation
- **Ongoing Work**: Manual curation of training data to improve neural network precision
- **Decision Point**: Whether to fully transition from regex to neural network or maintain hybrid approach

### Training Data Quality Assurance
**Challenge**: Maintaining high-quality training examples for machine learning models
- **Manual Process**: Django admin interface for genealogical anchor curation
- **Scale Issue**: Time-intensive manual review of extracted entities for gold standard creation
- **Current Focus**: Optimizing curation workflow while preserving data quality

## Next Development Priorities

### 1. **OCR Quality Validation** 🔥
- Test enhanced rotation detection on full document set
- Quantify content recovery improvements from advanced rotation correction
- **Decision Required**: Whether to reprocess existing documents with improved OCR
- **Estimated Impact**: Recovery of missing genealogical content, potential invalidation of existing curation work

### 2. **Neural Network Performance Optimization**
- Complete manual curation of training examples for improved model accuracy
- Analyze false positive vs. false negative patterns in current NER predictions
- Fine-tune confidence thresholds for production deployment
- **Estimated effort**: 2-3 focused curation sessions

### 3. **Database Relationship Architecture**
- Implement genealogical ID parsing for automatic parent-child relationship creation
- Design entity deduplication strategy for persons/places across text chunks
- **Estimated effort**: 1-2 development sessions

### 4. **RAG Foundation & Query System**
- Embedding generation pipeline for semantic search capabilities
- Vector search implementation with pgvector for genealogical queries
- **Estimated effort**: 2-3 sessions once extraction pipeline is stabilized

## Data Examples

### Genealogical ID Patterns
- `II.1.a` → 2nd generation, 1st family, 1st child
- `X.9. Children of John & Mary` → Family group header
- `a. Peter Smith` → Individual (becomes `X.9.a`)

### OCR Corrections Applied
- `IL.1.d` → `II.1.d`
- `XIL.4.b` → `XII.4.b`
- `VIL.2.a` → `VII.2.a`
- `V.2.F` → `V.2.f`

### Chunk Anchoring
Each TextChunk includes:
- Generation number (1-12)
- Page range (start_page, end_page)
- Genealogical IDs found/inferred
- Family context for ID resolution
