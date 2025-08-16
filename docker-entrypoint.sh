#!/bin/bash
set -e

# Run database migrations
python manage.py migrate --noinput

# Create superuser if it doesn't exist (will skip if user already exists)
python manage.py createsuperuser --noinput || true

# Execute the main command
exec "$@"