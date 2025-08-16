# Docker Setup for Genealogy Extractor

This setup provides a complete containerized environment for the genealogy extractor application with OCR processing capabilities.

## Services

- **web**: Django application server (port 8000)
- **celery**: Background task worker for OCR processing
- **celery-beat**: Task scheduler (for future periodic tasks)
- **db**: PostgreSQL database with pgvector extension (port 5432)
- **redis**: Redis cache and Celery message broker (port 6379)

## Quick Start

1. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Run migrations:**
   ```bash
   docker-compose exec web python manage.py migrate
   ```

3. **Create admin user:**
   ```bash
   docker-compose exec web python manage.py createsuperuser
   ```

4. **Access the application:**
   - Web interface: http://localhost:8000/admin/
   - Database: localhost:5432 (postgres/postgres)
   - Redis: localhost:6379

## Testing OCR Workflow

1. **Test the complete setup:**
   ```bash
   ./docker-test.sh
   ```

2. **Test multilingual OCR processing:**
   ```bash
   docker-compose exec web python test-ocr-workflow.py
   ```

## Development Workflow

1. **Upload documents via admin:**
   - Go to http://localhost:8000/admin/genealogy/document/
   - Use "Batch Upload Documents" to upload PDF/image files
   - Select language (English, Dutch, or English+Dutch)

2. **Process OCR:**
   - Select uploaded documents in admin
   - Use "Process OCR for selected documents" action
   - Monitor Celery worker logs: `docker-compose logs -f celery`

3. **View results:**
   - Check document pages for extracted text
   - View OCR confidence scores and status

## Container Commands

### View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f celery
docker-compose logs -f web
```

### Execute commands in containers:
```bash
# Django management commands
docker-compose exec web python manage.py [command]

# Celery worker status
docker-compose exec celery celery -A genealogy_extractor inspect active

# Database access
docker-compose exec db psql -U postgres genealogy_extractor
```

### Stop services:
```bash
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v
```

## File Storage

- **Media files**: Shared volume `media_files` between web and celery containers
- **Database**: Persistent volume `postgres_data`
- **Development**: Current directory mounted to `/app` for live code changes

## OCR Features

- **Languages**: English, Dutch, English+Dutch
- **Formats**: PDF, JPG, PNG, TIFF
- **Features**: Automatic rotation correction, confidence scoring
- **Processing**: Asynchronous via Celery tasks

## Troubleshooting

### Celery worker not processing tasks:
```bash
docker-compose logs celery
docker-compose restart celery
```

### Database connection issues:
```bash
docker-compose exec web python manage.py check --database default
```

### OCR processing errors:
- Check tesseract installation: `docker-compose exec celery tesseract --version`
- Verify language packs: `docker-compose exec celery tesseract --list-langs`

### Permission issues with media files:
```bash
docker-compose exec web ls -la /app/media/
```

## Performance Tuning

### Celery Configuration
```bash
# Increase worker concurrency
docker-compose exec celery celery -A genealogy_extractor worker --concurrency=4

# Monitor worker performance
docker-compose exec celery celery -A genealogy_extractor inspect stats
```

### Resource Limits
Edit `docker-compose.yml` to add resource limits:
```yaml
celery:
  deploy:
    resources:
      limits:
        memory: 2G
        cpus: '1.0'
```