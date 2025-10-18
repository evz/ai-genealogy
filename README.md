# Genealogy Extractor

[![Quality Gate](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml)

AI-powered genealogy digitization that processes Dutch family history books using OCR and LLM technology.

## About

This project addresses a gap in commercial genealogy tools, which focus heavily on names and dates while overlooking the fact that these refer to real people who lived full lives with stories worth preserving. The goal is to extract structured data from family documents and transform it into a collaborative family wiki where relatives can contribute not only genealogical facts but also stories about family gatherings, migrations, and daily life.

The extracted documents will serve as a searchable corpus, allowing family members to ask natural language questions about family history and receive answers that include suggestions for further research. This approach preserves both the factual genealogical data and the human stories that make family history meaningful.

## Technical Implementation

### Document Layout Analysis & OCR

**Three-stage pipeline:**

1. **Rotation Detection** (`genealogy/rotation_detector.py`)
   - Coarse detection: Tesseract OSD for 0°/180° rotations
   - Fine correction: GPU-accelerated projection profiles for -3° to +3° adjustments (0.1° precision using Kornia)

2. **Layout Detection** (`genealogy/document_layout_detector.py`)
   - DocLayout-YOLO model for semantic document understanding
   - Detects text blocks, titles, tables, figures, and captions with bounding boxes

3. **Region-Based OCR** (`genealogy/region_ocr_processor.py`)
   - Processes each detected region independently with Tesseract
   - Separates main content from inset annotations based on horizontal position (20% threshold)
   - Deduplicates overlapping text from region boundaries

**Research background:** Explored morphological segmentation (Ptak et al., 2017), OCR quality assessment (Schneider & Maurer, 2020), and variance-based image detection before adopting semantic layout analysis. Full research journey documented in [docs/doclayout_yolo_pipeline.md](docs/doclayout_yolo_pipeline.md).

### Entity Extraction

**Implementation:** `genealogy/tasks/extraction.py`, `genealogy/tasks/chunking.py`

- Generation-aware text chunking preserves genealogical document structure (family groups, generation headers)
- Local LLM inference via Ollama for structured extraction
- Extracts: person names, parent-child relationships, partnerships, events (birth, death, marriage)
- Dynamic context window sizing based on chunk length (4K-128K tokens)

### Graph-Based Entity Resolution

**Implementation:** `genealogy/clustering/`

Resolves duplicate person records across multiple text mentions using combined attribute and relational similarity.

**Core techniques:**

- **Attribute Similarity**: Levenshtein distance for strings, tolerance-based decay for numeric values
- **Disambiguation Weighting (AMB)**: Rare attribute values weighted higher than common ones using inverse frequency
- **Relational Signals**: Jaccard similarity on spouse/parent/child overlap with cluster-aware transitive matching
- **Constraint Propagation**:
  - Temporal constraints (birth/death date compatibility within 5 years)
  - Biological constraints (lifespan limits, death-before-birth detection)
  - Sibling detection (shared parents + different names OR incompatible birth dates)
- **Provenance Tracking**: Preserves all source entities, tracks pairwise confidence scores, creates canonical entities without destructive merging

**Research background:** Relational entity resolution techniques documented in [docs/family_clustering.md](docs/family_clustering.md), [docs/clustering_issues_analysis.md](docs/clustering_issues_analysis.md), and [research/pedigree_construction_notes.md](research/pedigree_construction_notes.md).

## Architecture

- **Framework**: Django + PostgreSQL (with pgvector) + Celery + Redis
- **OCR**: Tesseract 5.x with DocLayout-YOLO for layout analysis
- **LLM**: Local inference via Ollama
- **GPU Acceleration**: Kornia for rotation detection, PyTorch for models

## Documentation

- [docs/doclayout_yolo_pipeline.md](docs/doclayout_yolo_pipeline.md) - OCR pipeline research and implementation
- [docs/family_clustering.md](docs/family_clustering.md) - Entity resolution approach
- [docs/clustering_issues_analysis.md](docs/clustering_issues_analysis.md) - Clustering edge cases and solutions
- [research/pedigree_construction_notes.md](research/pedigree_construction_notes.md) - Pedigree construction notes
- [docs/DESIGN_LESSONS_LEARNED.md](docs/DESIGN_LESSONS_LEARNED.md) - Architecture lessons
- [docs/TESTING_LESSONS_LEARNED.md](docs/TESTING_LESSONS_LEARNED.md) - Testing failures and solutions

## References

### Document Layout Analysis & OCR

1. Zhao, Z., Kang, H., Wang, B., & He, C. (2024). "DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception." arXiv:2410.12628.

2. Ptak, R., Zygadlo, B., Unold, O. (2017). "Projection–Based Text Line Segmentation with a Variable Threshold." *International Journal of Applied Mathematics and Computer Science*, 27:195-206.

3. Dos Santos, R., Clemente, G., Ren, T., Calvalcanti, G. (2009). "Text Line Segmentation Based on Morphology and Histogram Projection."

4. Schneider, P., Maurer, Y. (2020). "Rerunning OCR: A Machine Learning Approach to Quality Assessment and Enhancement Prediction." National Library of Luxembourg.

### Entity Resolution

5. Kirielle, N., Nanayakkara, C., Christen, P., Dibben, C., Williamson, L., Garrett, E., & Manson, C. (2022). "Unsupervised Graph-based Entity Resolution for Accurate and Efficient Family Pedigree Search."

6. Fu, J., Tang, H., Khan, A., Mehrotra, S., Ke, X., & Gao, Y. (2025). "In-context Clustering-based Entity Resolution with Large Language Models: A Design Space Exploration." arXiv:2506.02509v1.

## License

MIT License - see [LICENSE](LICENSE) file for details.
