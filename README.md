# Genealogy Extractor

[![Quality Gate](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml)

Extracts structured genealogical data from Dutch family history books using OCR, layout analysis, and local LLMs. Currently processes scanned documents into a database of person mentions, events, and relationships with graph-based duplicate detection.

## What It Does

Takes printed family history books (Dutch "familiegeschiedenis" with complex layouts, marginal annotations, and genealogical notation) and turns them into structured, queryable data. The goal is eventually building a family wiki where facts and stories live together, but right now it's focused on accurate extraction and entity resolution.

The interesting problems are mostly around OCR (handling complex layouts with semantic understanding) and entity resolution (merging duplicate person mentions using both attribute similarity and family relationship graphs).

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. OCR WITH GROUNDING TOKENS (DeepSeek-OCR)                            │
├─────────────────────────────────────────────────────────────────────────┤
│ • Rotation correction (Tesseract OSD + Kornia projection profiles)     │
│ • Layout detection (DocLayout-YOLO finds regions, types)               │
│ • DeepSeek-OCR with grounding tokens (bounding boxes + element types)  │
│                                                                         │
│ Output: OCRPage with grounded tokens (position + semantic type)        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. BOOK SECTION IDENTIFICATION                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ • Detect section types (FRONT_MATTER, DESCENDANT_GENEALOGY,            │
│   KWARTIERSTATEN, APPENDIX, GLOSSARY, INDEX)                           │
│ • Store page ranges for each section                                   │
│                                                                         │
│ Output: BookSection records with start/end pages                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. TEXT CHUNKING (Section-Specific Strategies)                         │
├─────────────────────────────────────────────────────────────────────────┤
│ For DESCENDANT_GENEALOGY sections:                                     │
│   Pass 1: Extract out-of-flow content (images, info boxes)             │
│   Pass 2: Handler-based chunking of main flow                          │
│     - GenerationHeaderHandler    (I, II, III, ...)                     │
│     - FamilyGroupHeaderHandler   (II.3. Kinderen van...)               │
│     - IndividualEntryHandler     (a. Pieter van Zanten, ...)           │
│     - SourceCitationHandler      (bibliographic references)            │
│                                                                         │
│   Phase 1 Extraction (Deterministic):                                  │
│     • Parse generation numbers, family groups                          │
│     • Extract people from headers (parents, children)                  │
│     • Extract parent-child relationships from structure                │
│                                                                         │
│ For other sections: SkipChunkingStrategy (not implemented yet)         │
│                                                                         │
│ Output: TextChunk records with chunk_type, generation_number,          │
│         family_groups, extracted_people, extracted_relationships       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. LLM EXTRACTION (Section-Specific Strategies)                        │
├─────────────────────────────────────────────────────────────────────────┤
│ For DESCENDANT_GENEALOGY chunks (chunk_type='GENEALOGY_ENTRY'):        │
│                                                                         │
│   Phase 2 Extraction (LLM):                                            │
│     • Merge with Phase 1 data (people, relationships)                  │
│     • Extract events: BIRT, DEAT, MARR, BAPT, BURI, OCCU, RESI, etc.   │
│     • Extract partnerships (spouse relationships)                      │
│     • Enrich with additional people from narrative text                │
│     • Dynamic context window (4K-128K) based on chunk size             │
│                                                                         │
│ For other sections: SkipExtractionStrategy (not implemented yet)       │
│                                                                         │
│ Output: Enhanced TextChunk with extracted_events, enriched people      │
│         and relationships                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. ENTITY CREATION (create_entities.py)                                │
├─────────────────────────────────────────────────────────────────────────┤
│ • Create PersonMention for each person (immutable record)              │
│ • Create singleton Identity for each mention                           │
│ • Create RelationshipMention (parent-child links)                      │
│ • Create PartnershipMention (spouse relationships)                     │
│ • Create Event records (BIRT, DEAT, MARR, OCCU, etc.)                  │
│ • Link events to person mentions                                       │
│                                                                         │
│ Output: PersonMention, Identity, RelationshipMention,                  │
│         PartnershipMention, Event, Place records                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. ENTITY CLUSTERING (cluster_entities.py)                             │
├─────────────────────────────────────────────────────────────────────────┤
│ Graph-based entity resolution (Kirielle et al. 2022):                  │
│                                                                         │
│ Phase 1: Bootstrap high-confidence matches                             │
│   • Calculate name similarity (Levenshtein)                            │
│   • Calculate date proximity (birth year)                              │
│   • Calculate relationship overlap (spouse/child Jaccard)              │
│   • Validate constraints (age, gender, temporal)                       │
│   • Merge clusters with similarity >= 0.75                             │
│                                                                         │
│ Phase 2: Iterative refinement                                          │
│   • Re-calculate similarities with merge-aware matching                │
│   • Merge clusters with similarity >= 0.60                             │
│   • Infer partnerships from shared children                            │
│   • Refine with transitive relationship matching                       │
│                                                                         │
│ Output: PotentialDuplicate records for review                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. MANUAL REVIEW & MERGE                                               │
├─────────────────────────────────────────────────────────────────────────┤
│ • Review PotentialDuplicate suggestions in admin UI                    │
│ • Merge PersonMentions into consolidated Identities                    │
│ • Audit log tracks all merge operations (reversible)                   │
│                                                                         │
│ Output: Merged Identity records representing unique individuals        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Two-Phase Extraction:** Phase 1 (deterministic, during chunking) provides reliable anchor data that Phase 2 (LLM) enhances. This reduces hallucination and improves quality.

**Section-Based Processing:** Different book sections (descendant genealogy, ancestor charts, indexes) need different chunking and extraction logic. Strategy pattern allows clean separation.

**Grounding Tokens:** DeepSeek-OCR provides bounding boxes and semantic types (title, text, list, etc.) that enable intelligent chunking based on document structure, not just text patterns.

**Immutable Mentions:** All `PersonMention` records are immutable with full provenance. Deduplication happens through mutable `Identity` mappings that can be reviewed and reversed.

## Tech Stack

- **Backend**: Django + PostgreSQL (pgvector) + Celery + Redis
- **OCR**: DeepSeek-OCR, DocLayout-YOLO, Tesseract OSD
- **LLM**: Ollama (llama3.1:70b)
- **Image Processing**: Kornia/PyTorch (GPU)
- **Clustering**: Custom dependency graph + Union-Find
- **String Matching**: textdistance (Levenshtein)

## Current Status

End-to-end pipeline working. Currently processing the Van Zanten family book.

Recent work:
- DeepSeek-OCR integration with grounding tokens
- Section-based processing (strategy pattern for chunking + extraction)
- Two-phase extraction (deterministic + LLM)
- Occupation extraction (inline + narrative)

Next:
- Strategies for other section types (ancestor charts, indexes)
- Clustering quality metrics
- Merge review UI polish

## Docs

- [docs/doclayout_yolo_pipeline.md](docs/doclayout_yolo_pipeline.md) - OCR pipeline design
- [docs/pedigree_construction_notes.md](docs/pedigree_construction_notes.md) - Clustering algorithm notes (Kirielle et al. 2022)
- [docs/family_clustering.md](docs/family_clustering.md) - Entity resolution approach
- [docs/clustering_issues_analysis.md](docs/clustering_issues_analysis.md) - Edge cases
- [docs/REFACTOR_PLAN_reversible_provenance.md](docs/REFACTOR_PLAN_reversible_provenance.md) - Architecture refactor plan
- [docs/DESIGN_LESSONS_LEARNED.md](docs/DESIGN_LESSONS_LEARNED.md) - Things that didn't work

## References

**Layout Analysis:**
- Zhao et al. (2024) - DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception. arXiv:2410.12628
- Ptak et al. (2017) - Projection-Based Text Line Segmentation with a Variable Threshold. *Int. J. Applied Math and CS*, 27:195-206

**Entity Resolution:**
- Kirielle et al. (2022) - Unsupervised Graph-based Entity Resolution for Accurate and Efficient Family Pedigree Search
- Fu et al. (2025) - In-context Clustering-based Entity Resolution with Large Language Models. arXiv:2506.02509v1

## License

MIT
