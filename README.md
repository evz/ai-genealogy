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

OCR processing pipeline implemented: multi-format documents (PDF, JPG, PNG, TIFF), multi-language OCR (English/Dutch), batch upload, background processing with Celery.

Uses Django admin interface to prototype and test business logic before building custom UI. This approach allows rapid iteration on data models and processing workflows.

**Next**: AI-powered extraction to structured genealogy data.

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

Upload documents via Django admin → automatic OCR processing → review extracted text and confidence scores.

## Development

**Quality checks:** `make quality-gate` (linting, formatting, type checking, security, tests)

**Tests:** `make test`

**Architecture:** Django + PostgreSQL + Celery + Redis + Tesseract OCR

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
