# Genealogy Extractor

[![Quality Gate](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/evz/ai-genealogy/actions/workflows/quality-gate.yml)

AI-powered genealogy digitization that processes Dutch family history books using OCR and LLM technology.

## About

This project addresses a gap in commercial genealogy tools, which focus heavily on names and dates while overlooking the fact that these refer to real people who lived full lives with stories worth preserving. The goal is to extract structured data from family documents and transform it into a collaborative family wiki where relatives can contribute not only genealogical facts but also stories about family gatherings, migrations, and daily life.

The extracted documents will serve as a searchable corpus, allowing family members to ask natural language questions about family history and receive answers that include suggestions for further research. This approach preserves both the factual genealogical data and the human stories that make family history meaningful.

This project builds on lessons learned from an earlier [family-wiki](https://github.com/evz/family-wiki/) project that highlighted the importance of clear requirements, comprehensive testing, and iterative development when working with AI assistants on complex software projects.

## Quick Demo

Try the OCR processing with sample genealogy documents:

**Requirements:** Git, Docker and make

```bash
git clone <repository>
cd ai-genealogy
cp .env.example .env
make demo
```

**Access:** http://localhost:8000/admin/ (admin/admin)

The demo processes a couple sample pages from a book about my family and extracts multilingual text.

## Current Status

### OCR Processing Pipeline - Optimized
- Multi-format document processing (PDF, JPG, PNG, TIFF) with Tesseract PSM 1
- Multi-language support (English/Dutch) for genealogical texts
- Automatic orientation detection using Tesseract's built-in OSD (Orientation and Script Detection)
- 92-94% OCR confidence on genealogy documents (significantly improved from 45-55%)
- Batch upload functionality and background processing with Celery/Redis

### AI-Powered Entity Extraction - Implemented
- **Neural Network NER (Named Entity Recognition)**: Custom BERT-based model fine-tuned for genealogical entities
- **Performance**: 96.84% F1 score (harmonic mean of precision and recall) across PERSON_NAME, DATE, PLACE, GENEALOGY_ID, FAMILY_GROUP entities
- **Dual Extraction Pipeline**: Hybrid approach combining traditional regex patterns with neural network predictions
- **Training Data Curation**: Django admin interface for manual refinement of genealogical anchor extractions

### Text Processing & Data Standardization
- **Generation-Aware Chunking**: Intelligent segmentation preserving genealogical document structure
- **Date Standardization**: Multi-format Dutch/English date parsing ("15 maart 1654" → "1654-03-15")
- **Genealogical ID Correction**: Systematic fixes for OCR errors in Roman numerals (IL→II, XIL→XII)
- **Family Context Tracking**: Infers individual IDs from family group headers ("a. John" → "X.9.a")

### Development Approach
Uses Django admin interface to prototype and test business logic before building custom UI. This approach enables rapid iteration on data models and processing workflows while maintaining data quality through manual review capabilities.

**Current Focus**: Refining neural network training data and relationship inference
**Next Phase**: LLM integration for natural language queries and relationship inference

## Sample Data

The `samples/` directory contains a couple sample pages from a book about my family with mixed English/Dutch text.

## Tested With

- Python 3.12
- Tesseract OCR 5.x with English and Dutch language packs
- Docker 28.x
- PostgreSQL 16 with pgvector extension

## Setup

**Docker (Recommended):**
```bash
cp .env.example .env
make up-build
```

**Local Development:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate && python manage.py createsuperuser
python manage.py runserver
```


## Usage

**Document Processing Workflow:**
1. Upload documents via Django admin interface
2. Automatic OCR processing with PSM 1 orientation detection
3. Intelligent text chunking with genealogical structure preservation
4. Dual entity extraction (regex + neural network NER)
5. Review extracted text, confidence scores, and genealogical anchors
6. Manual curation of training data for neural network refinement

**Current Capabilities:**
- Multi-page document OCR with confidence scoring
- Genealogical entity recognition and extraction
- Date standardization and genealogical ID correction
- Visual comparison of extraction methods (regex vs. neural network)
- Manual anchor curation for gold standard training data

## Development

**Quality checks:** `make quality-gate` (linting, formatting, type checking, security, tests)

**Tests:** `make test`

**Architecture:** Django + PostgreSQL + Celery + Redis + Tesseract OCR + PyTorch

**Key Technologies:**
- **Machine Learning**: PyTorch + Transformers (BERT) for genealogical Named Entity Recognition
- **Data Storage**: PostgreSQL with custom ArrayField handling for genealogical anchors
- **Background Processing**: Celery with Redis for scalable document processing
- **OCR**: Tesseract PSM 1 with automatic orientation detection and multi-language support

Run `make help` to see all available development commands.

## Documentation

- [docs/INSTRUCTIONS.md](docs/INSTRUCTIONS.md) - Original project requirements and specifications
- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) - Development phases and implementation plan
- [docs/DESIGN_LESSONS_LEARNED.md](docs/DESIGN_LESSONS_LEARNED.md) - Critical architecture lessons to avoid over-engineering
- [docs/TESTING_LESSONS_LEARNED.md](docs/TESTING_LESSONS_LEARNED.md) - Real-world testing failures and solutions

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

For issues and feature requests, please check the existing documentation and create detailed bug reports with reproduction steps.
