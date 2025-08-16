# Genealogy Extractor

[![Quality Gate](https://github.com/evanzanten/ai-genealogy/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/evanzanten/ai-genealogy/actions/workflows/quality-gate.yml)

AI-powered genealogy digitization that processes Dutch family history books using OCR and LLM technology.

## Current Status

OCR processing pipeline implemented: multi-format documents (PDF, JPG, PNG, TIFF), multi-language OCR (English/Dutch), batch upload, background processing with Celery.

**Next**: AI-powered extraction to structured genealogy data.

## Tested With

- Python 3.12
- Tesseract OCR 5.x with English and Dutch language packs
- Docker 28.x
- PostgreSQL 16 with pgvector extension

## Sample Data

The `samples/` directory contains Dutch genealogy documents (025.pdf, 032.pdf) used for testing multilingual OCR extraction. These demonstrate the system's ability to process historical family records with mixed English/Dutch text.

## Quick Demo

**Requirements:** Git, Docker and make

```bash
git clone <repository>
cd ai-genealogy
cp .env.example .env
make demo
```

**Access:** http://localhost:8000/admin/ (admin/admin)

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
