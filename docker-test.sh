#!/bin/bash

# Docker setup test script
echo "Testing Docker setup for genealogy extractor..."

# Build containers
echo "Building containers..."
docker-compose build

# Start services in background
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check if all services are running
echo "Checking service status..."
docker-compose ps

# Test web service health
echo "Testing web service..."
if curl -f http://localhost:8000/admin/ -o /dev/null -s; then
    echo "✓ Web service is responding"
else
    echo "✗ Web service not responding"
fi

# Test database connection
echo "Testing database connection..."
docker-compose exec -T web python manage.py check --database default

# Test Celery worker
echo "Testing Celery worker..."
docker-compose logs celery | tail -5

# Test Redis connection
echo "Testing Redis connection..."
docker-compose exec -T redis redis-cli ping

echo "Docker setup test completed!"
echo "To stop services: docker-compose down"
echo "To view logs: docker-compose logs [service_name]"
