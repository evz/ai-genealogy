# Genealogy Extractor

An AI-powered genealogy digitization application that processes Dutch family history books using OCR and LLM technology to extract structured family data.

## Current Status

**Current Status**: OCR processing pipeline implemented and tested
- Multi-format document processing (PDF, JPG, PNG, TIFF)  
- Multi-language OCR (English, Dutch, bilingual) with Tesseract
- Batch upload via Django admin interface
- Background OCR processing with Celery
- Quality control with confidence scoring and rotation correction
- Test suite: 36 tests passing
- Code quality tools: linting, formatting, type checking, security scanning

**Next Phase**: AI-powered extraction and structured data generation

## Features

### Implemented Features
- Multi-format document processing: PDF, JPG, PNG, TIFF
- Multi-language OCR: English, Dutch, and bilingual text extraction with Tesseract
- Batch upload interface: Upload multiple documents via Django admin with two modes:
  - Single document with multiple pages (for scanned books)
  - Multiple separate documents (for individual files)
- Background processing: Celery-based async OCR processing
- Quality control: OCR confidence scoring, automatic rotation correction
- Progress monitoring: Real-time OCR progress tracking in admin interface
- Error handling: Comprehensive error handling and logging

### Planned Features
- **LLM Extraction**: Convert OCR text to structured genealogy data using local Ollama
- **GEDCOM Export**: Generate family tree files in standard format
- **Wiki Generation**: Automatic wiki page creation for families
- **RAG Pipeline**: Query documents with natural language using pgvector
- **Research Questions**: AI-generated genealogical research suggestions

## Architecture

### Technology Stack
- **Backend**: Django 5.2.5 with PostgreSQL + pgvector (planned)
- **Task Queue**: Celery with Redis backend
- **OCR**: Tesseract with English and Dutch language packs
- **Image Processing**: Pillow, pdf2image for document conversion
- **AI/LLM**: Ollama integration (planned for the-area.local)
- **Containerization**: Docker Compose for development and deployment
- **Code Quality**: Ruff (linting), MyPy (type checking), Bandit (security)

### Current Data Models
- **Document**: Main genealogy document with language settings and metadata
- **DocumentPage**: Individual pages/images with OCR results and confidence scores

### Planned Data Models
- **Person**: Individual family members with genealogical data
- **Partnership**: Gender-neutral relationships (marriage, partnership)
- **Place**: Geographic locations with hierarchical structure
- **Event**: Life events (birth, death, marriage, etc.)

## Quick Start

### Development Setup

1. **Clone and setup virtual environment:**
   ```bash
   git clone <repository>
   cd genealogy_extractor
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup database:**
   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

4. **Run development server:**
   ```bash
   python manage.py runserver
   ```

### Docker Setup (Recommended)

1. **Start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Setup database:**
   ```bash
   docker-compose exec web python manage.py migrate
   docker-compose exec web python manage.py createsuperuser
   ```

3. **Access application:**
   - Admin interface: http://localhost:8000/admin/
   - Upload documents: http://localhost:8000/admin/genealogy/document/

See [DOCKER_README.md](DOCKER_README.md) for detailed Docker instructions.

## Usage

### Document Processing Workflow

1. **Upload Documents**:
   - Access Django admin at `/admin/genealogy/document/`
   - Use "Batch Upload Documents" for multiple files
   - Choose upload mode:
     - **Single Document (Multiple Pages)**: Creates one document with numbered pages (ideal for scanned multi-page books)
     - **Multiple Separate Documents**: Creates separate documents for each file
   - Select appropriate language (en, nl, or en+nl)
   - OCR processing starts automatically after upload

2. **Monitor OCR Progress**:
   - View progress in document list view
   - Check individual page results in DocumentPage admin
   - Use "Process OCR" admin action for manual processing if needed

3. **Review Results**:
   - View extracted text in DocumentPage admin
   - Check OCR confidence scores
   - Review any rotation corrections applied

### Testing OCR

Test the multilingual OCR workflow:
```bash
# With virtual environment
python test-ocr-workflow.py

# With Docker
docker-compose exec web python test-ocr-workflow.py
```

## Development

### Code Quality

The project uses comprehensive linting and code quality tools:

```bash
# Run all quality checks
make quality-gate

# Individual checks
make lint           # Run ruff linting
make lint-fix       # Auto-fix linting issues
make format         # Format code with ruff
make format-check   # Check formatting without changes
make type-check     # Run mypy type checking
make security       # Run bandit security checks
make test           # Run test suite
```

**Quality Gate Requirements:**
- ✅ Ruff linting (code style, imports, complexity)
- ✅ Code formatting (consistent style)
- ✅ Type checking with MyPy
- ✅ Security scanning with Bandit
- ✅ Django system checks
- ✅ Full test suite passing

**Pre-commit Hooks:**
```bash
# Install pre-commit hooks (requires git repository)
make pre-commit
```

### Project Structure
```
genealogy_extractor/
├── genealogy/              # Main Django app
│   ├── models.py          # Data models
│   ├── admin.py           # Admin interface customizations
│   ├── tasks.py           # Celery background tasks
│   ├── ocr_processor.py   # OCR processing logic
│   ├── tests/             # Test suite
│   ├── templates/         # Admin templates
│   └── static/            # CSS and assets
├── genealogy_extractor/   # Django project settings
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Container orchestration
└── Dockerfile            # Container definition
```

### Running Tests

```bash
# All tests
python manage.py test genealogy.tests

# Specific test modules
python manage.py test genealogy.tests.test_models
python manage.py test genealogy.tests.test_ocr_processor
python manage.py test genealogy.tests.test_tasks
python manage.py test genealogy.tests.test_admin

# With Docker
docker-compose exec web python manage.py test genealogy.tests
```

### Design Philosophy

This project follows hard-learned lessons from previous iterations to avoid over-engineering:

1. **Simplicity First**: Django admin over custom interfaces, functions over classes when possible
2. **Boring Technology**: Proven, simple solutions over shiny/complex frameworks
3. **Explicit Over Clever**: Clear, direct code over abstractions and patterns
4. **Real Integration Testing**: Test actual user workflows, mock only external dependencies
5. **Quality Gate Enforcement**: Comprehensive linting, type checking, and security scanning
6. **Inclusive Design**: Gender-neutral relationship modeling for modern genealogy

## Configuration

### Environment Variables

- `DEBUG`: Enable Django debug mode (default: False)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection for Celery
- `MEDIA_ROOT`: File storage location

### OCR Settings

Tesseract configuration in `ocr_processor.py`:
- Character whitelist optimized for genealogy documents
- Rotation detection and correction
- Multi-language support (English + Dutch)
- Confidence score calculation

## Contributing

### Development Guidelines

1. **Follow existing patterns**: Check similar implementations before adding new features
2. **Write tests first**: All new functionality should have corresponding tests
3. **Use type hints**: Maintain code clarity with proper typing
4. **Document decisions**: Update relevant `.md` files for significant changes

### Testing Strategy

- **Models**: Test business logic and validation
- **OCR Processor**: Mock external dependencies (Tesseract)
- **Tasks**: Test Celery task success/failure scenarios
- **Admin**: Test batch upload and UI functionality

## Documentation

- [PROJECT_PLAN.md](PROJECT_PLAN.md) - Original project specification and development phases
- [DESIGN_LESSONS_LEARNED.md](DESIGN_LESSONS_LEARNED.md) - Critical architecture lessons to avoid over-engineering
- [TESTING_LESSONS_LEARNED.md](TESTING_LESSONS_LEARNED.md) - Real-world testing failures and solutions
- [QUALITY_GATE.md](QUALITY_GATE.md) - Code quality standards and tools
- [INSTRUCTIONS.md](INSTRUCTIONS.md) - Development guidelines and requirements
- [DOCKER_README.md](DOCKER_README.md) - Container setup and usage

## License

[License information]

## Support

For issues and feature requests, please check the existing documentation and create detailed bug reports with reproduction steps.
