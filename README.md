# Genealogy Extractor

[![Quality Gate](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml)

Extracts structured genealogical data from Dutch family history books using OCR, layout analysis, and local LLMs. Processes scanned documents into a database of people, relationships, partnerships, and events with hybrid RAG-based querying.

## What It Does

Takes printed family history books (Dutch "familiegeschiedenis" with complex layouts, marginal annotations, and genealogical notation) and turns them into structured, queryable data. The goal is eventually building a family wiki where facts and stories live together, but right now it's focused on accurate extraction and intelligent querying using hybrid RAG retrieval.

The interesting problems are mostly around OCR (handling complex layouts with semantic understanding), entity extraction (building a structured genealogy graph from narrative text), and intelligent retrieval (hybrid RAG combining vector similarity, phonetic matching, and trigram search).

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. OCR WITH GROUNDING TOKENS (DeepSeek-OCR)                             │
├─────────────────────────────────────────────────────────────────────────┤
│ • Rotation correction (Tesseract OSD + Kornia projection profiles)      │
│ • Layout detection (DocLayout-YOLO finds regions, types)                │
│ • DeepSeek-OCR with grounding tokens (bounding boxes + element types)   │
│                                                                         │
│ Output: OCRPage with grounded tokens (position + semantic type)         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. BOOK SECTION IDENTIFICATION                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ • Detect section types (FRONT_MATTER, DESCENDANT_GENEALOGY,             │
│   KWARTIERSTATEN, APPENDIX, GLOSSARY, INDEX)                            │
│ • Store page ranges for each section                                    │
│                                                                         │
│ Output: BookSection records with start/end pages                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. TEXT CHUNKING (Section-Specific Strategies)                          │
├─────────────────────────────────────────────────────────────────────────┤
│ For DESCENDANT_GENEALOGY sections:                                      │
│   Pass 1: Extract out-of-flow content (images, info boxes)              │
│   Pass 2: Handler-based chunking of main flow                           │
│     - GenerationHeaderHandler    (I, II, III, ...)                      │
│     - FamilyGroupHeaderHandler   (II.3. Kinderen van...)                │
│     - IndividualEntryHandler     (a. Pieter van Zanten, ...)            │
│     - SourceCitationHandler      (bibliographic references)             │
│                                                                         │
│ For other sections: SkipChunkingStrategy (not implemented yet)          │
│                                                                         │
│ Output: TextChunk records with chunk_type, generation_number,           │
│         family_groups, genealogical_id                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. BUILD GENEALOGY GRAPH (from genealogical IDs)                        │
├─────────────────────────────────────────────────────────────────────────┤
│ • Create Person records from genealogical IDs (II.3.a, III.5.b, etc.)   │
│ • Parse family headers to identify parents/children                     │
│ • Mint spouse IDs for partners without explicit IDs (II.3.a.spouse1)    │
│ • Create Relationship records (parent-child links)                      │
│ • Create Partnership records (spouse relationships)                     │
│                                                                         │
│ Output: Person, Relationship, Partnership records                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. LLM EXTRACTION (Section-Specific Strategies)                         │
├─────────────────────────────────────────────────────────────────────────┤
│ For DESCENDANT_GENEALOGY chunks (chunk_type='individual_entry'):        │
│                                                                         │
│   LLM Extraction:                                                       │
│     • Extract events: BIRT, DEAT, MARR, BAPT, BURI, OCCU, RESI, etc.    │
│     • Extract event details (date, place, description)                  │
│     • Parse dates in document-specific format (DMY or MDY)              │
│     • Dynamic context window (4K-128K) based on chunk size              │
│                                                                         │
│ For other sections: SkipExtractionStrategy (not implemented yet)        │
│                                                                         │
│ Output: TextChunk.extracted_events (JSON array)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. PERSIST ENTITIES (create Event records from LLM output)              │
├─────────────────────────────────────────────────────────────────────────┤
│ • Parse extracted_events JSON from each chunk                           │
│ • Create Event records linked to Person by genealogical_id              │
│ • Parse dates using document's date_format setting                      │
│ • Create Place records for event locations                              │
│                                                                         │
│ Output: Event records linked to Person records                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. CHUNK ENRICHMENT (embeddings + phonetic codes)                       │
├─────────────────────────────────────────────────────────────────────────┤
│ • Generate embeddings for ALL chunks (not just individual_entry)        │
│ • Extract surnames from Person records linked to chunks                 │
│ • Generate Daitch-Mokotoff codes for phonetic surname matching          │
│                                                                         │
│ Output: TextChunk.embedding (pgvector), TextChunk.dm_codes (array)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. HYBRID RAG RETRIEVAL (Reciprocal Rank Fusion)                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Semantic Query Expansion (for queries without capitalized names):       │
│   • LLM generates Dutch/English synonyms and archaic terminology        │
│   • Example: "military" → "ruiter, soldaat, cavalerie, militie"         │
│   • Increases retrieval limits (2x top_k) for better coverage           │
│                                                                         │
│ Combine three retrieval methods:                                        │
│   1. Vector similarity (cosine distance on embeddings)                  │
│   2. Phonetic matching (Daitch-Mokotoff codes for surnames)             │
│   3. Trigram similarity (pg_trgm for fuzzy text matching)               │
│                                                                         │
│ Reciprocal Rank Fusion merges results:                                  │
│   • Each method ranks chunks independently                              │
│   • RRF score = Σ(1 / (k + rank)) across all methods                    │
│   • Chunks scoring well in multiple methods rank highest                │
│                                                                         │
│ Output: Top-k ranked chunks for LLM context                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 9. INTELLIGENT MODEL ROUTING                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Three-tier model architecture with automatic routing:                   │
│                                                                         │
│   gene-chat-fast (llama3.1:8b)                                           │
│     • Fast interactive queries                                          │
│     • Simple factual lookups                                            │
│                                                                         │
│   gene-chat-main (qwen2.5:14b)                                           │
│     • Complex reasoning tasks                                           │
│     • Agent mode with tool calling                                      │
│                                                                         │
│   gene-reasoner (deepseek-r1:14b)                                        │
│     • Identity resolution queries                                       │
│     • Merge conflict detection                                          │
│     • Explicit reasoning via <think> tags                               │
│                                                                         │
│ Routing precedence:                                                     │
│   1. Merge detection (keywords: "same person", "dezelfde", etc.)        │
│   2. Agent mode flag (complex multi-step queries)                       │
│   3. Query complexity analysis (word count, question marks)             │
│   4. Default: gene-chat-fast                                            │
│                                                                         │
│ Frontend: Real-time model selection display + streaming reasoning       │
│                                                                         │
│ Output: Streaming SSE response with model_selected events               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Graph-First Architecture:** Build genealogy graph directly from genealogical IDs (II.3.a, etc.) before LLM extraction. This creates a reliable structural skeleton that LLM enriches with events and details. Spouse IDs are minted for partners without explicit IDs (e.g., II.3.a.spouse1).

**Section-Based Processing:** Different book sections (descendant genealogy, ancestor charts, indexes) need different chunking and extraction logic. Strategy pattern allows clean separation.

**Grounding Tokens:** DeepSeek-OCR provides bounding boxes and semantic types (title, text, list, etc.) that enable intelligent chunking based on document structure, not just text patterns.

**Hybrid RAG Retrieval:** Combines vector similarity, phonetic surname matching (Daitch-Mokotoff), and trigram text matching using Reciprocal Rank Fusion. This captures semantic similarity, name variants, and fuzzy text matches in a single ranking.

**Hierarchical Search Tiers:** Chunks are classified into two tiers based on content richness:
- **Metadata tier** (66%): Short entries (< 100 chars) with just vital statistics. No embeddings generated. Searchable via trigram/phonetic matching for name-based queries.
- **Narrative tier** (34%): Longer entries (>= 100 chars) with biographical content. Full embeddings generated. Used for semantic queries like "Who served in the military?"

This reduces embedding storage by 66% and dramatically improves semantic search precision by excluding stub entries from vector search.

**Intelligent Model Routing:** Three-tier model architecture automatically routes queries to the optimal LLM based on complexity and query type. Merge/identity queries use gene-reasoner (DeepSeek-R1) for explicit reasoning, complex multi-step queries use gene-chat-main (Qwen2.5), and simple lookups use gene-chat-fast (Llama3.1:8b). Frontend displays model selection in real-time and streams DeepSeek's reasoning tokens for transparency.

## Tech Stack

- **Backend**: Django + PostgreSQL (pgvector + pg_trgm) + Celery + Redis
- **OCR**: DeepSeek-OCR, DocLayout-YOLO, Tesseract OSD
- **LLM**: Ollama with three-tier routing
  - gene-chat-fast: llama3.1:8b (interactive queries)
  - gene-chat-main: qwen2.5:14b (complex reasoning, agent mode)
  - gene-reasoner: deepseek-r1:14b (identity resolution, explicit reasoning)
  - Embeddings: multilingual-e5-large
- **Image Processing**: Kornia/PyTorch (GPU)
- **RAG Retrieval**: Hybrid search with Reciprocal Rank Fusion
- **Phonetic Matching**: Daitch-Mokotoff soundex for surname variants

## Current Status

End-to-end pipeline working with hybrid RAG retrieval. Currently processing the Van Zanten family book.

**Data:** 390 people, 513 text chunks (all with embeddings), 268 tests passing

Recent work:
- Semantic query expansion (LLM-based Dutch/English synonym generation for archaic terminology)
- Intelligent model routing (3-tier LLM architecture with automatic selection)
- Streaming reasoning tokens from DeepSeek-R1 to frontend
- Simplified architecture (removed PersonMention/Identity clustering)
- Graph-first extraction from genealogical IDs
- Spouse ID minting for partners without explicit IDs
- Hybrid RAG retrieval with RRF (vector + phonetic + trigram)
- Full-chunk embeddings for better RAG context
- Document-specific date format handling (DMY/MDY)

Next:
- RAG quality evaluation and improvements
- Strategies for other section types (ancestor charts, indexes)
- Agent-based extraction workflows

## References

**Layout Analysis:**
- Zhao et al. (2024) - DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception. arXiv:2410.12628
- Ptak et al. (2017) - Projection-Based Text Line Segmentation with a Variable Threshold. *Int. J. Applied Math and CS*, 27:195-206

## License

MIT
