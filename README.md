# Genealogy Extractor

[![Quality Gate](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml)

Extracts structured genealogical data from Dutch family history books using OCR, layout analysis, and local LLMs. Currently processes scanned documents into a database of person mentions, events, and relationships with graph-based duplicate detection.

## What It Does

Takes printed family history books (Dutch "familiegeschiedenis" with complex layouts, marginal annotations, and genealogical notation) and turns them into structured, queryable data. The goal is eventually building a family wiki where facts and stories live together, but right now it's focused on accurate extraction and entity resolution.

The interesting problems are mostly around OCR (handling complex layouts with semantic understanding) and entity resolution (merging duplicate person mentions using both attribute similarity and family relationship graphs).

## Pipeline

### 1. OCR with Layout Understanding

The OCR pipeline does semantic document layout analysis before text extraction:

- **Rotation correction** - Coarse detection with Tesseract OSD, then GPU-accelerated projection profiles (Kornia) for fine adjustment (0.1° precision)
- **Layout detection** - DocLayout-YOLO finds text blocks, titles, tables, figures, captions
- **Region-based OCR** - Tesseract processes each region independently, separates main text from marginal annotations based on position

This approach came after trying morphological segmentation and variance-based methods. Layout-aware processing handles Dutch family book formats much better than naive line-by-line OCR. Details in [docs/doclayout_yolo_pipeline.md](docs/doclayout_yolo_pipeline.md).

### 2. Entity Extraction

Local LLM (via Ollama) extracts structured data from OCR text:

- Generation-aware chunking preserves genealogical structure (family groups, generation headers)
- Extracts person mentions, relationships (parent-child, partnerships), events (birth, death, marriage)
- Dynamic context window (4K-128K tokens) based on chunk complexity

All extractions stored as immutable `PersonMention` records with source provenance. Each mention gets mapped to an `Identity` (the "real person") through a mutable mapping layer.

### 3. Graph-Based Duplicate Detection

Clustering algorithm (based on Kirielle et al. 2022) finds duplicate person mentions using attribute + relational similarity:

- **Attribute matching** - Levenshtein distance for names, decay functions for dates, inverse frequency weighting for rare values
- **Relationship matching** - Jaccard similarity on spouse/parent/child overlap, with merge-aware transitive matching (if two mentions share a spouse that's already been identified as the same person, that's strong evidence they're also the same)
- **Constraint validation** - Temporal (birth/death compatibility), biological (lifespan limits), sibling detection (shared parents but different names)
- **Iterative refinement** - After merging some clusters, re-run clustering to find new matches that depend on previous merges

Creates `PotentialDuplicate` records for manual review. Merge operations are fully reversible through an audit log.

Details in [docs/pedigree_construction_notes.md](docs/pedigree_construction_notes.md) and [docs/family_clustering.md](docs/family_clustering.md).

## Tech Stack

- Django + PostgreSQL (pgvector for future embedding work) + Celery + Redis
- Tesseract 5.x + DocLayout-YOLO
- Ollama for local LLM inference
- Kornia/PyTorch for GPU-accelerated image processing

## Current Status

The pipeline works end-to-end (OCR → extraction → clustering → manual merge review). Currently refining the entity resolution logic and building out the admin UI for merge operations.

Recent work:
- Implemented reversible provenance architecture (all extractions immutable, merges tracked in audit log)
- Added merge-aware clustering (re-running clustering after merges picks up new matches based on previously merged identities)
- Improved sibling detection and constraint validation

Still todo:
- Better LLM prompts for relationship extraction (currently misses some edge cases)
- Evaluation metrics for clustering quality
- UI polish for the merge review workflow

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
