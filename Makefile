.PHONY: help install install-dev lint format type-check security test test-fast clean build up down logs shell migrate createsuperuser collectstatic reset-db docker-test quality-gate pre-commit setup

# Colors for output
YELLOW := \033[1;33m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m

# Default target
help: ## Show this help message
	@echo "$(YELLOW)Genealogy Extractor - Development Commands$(NC)"
	@echo "============================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Setup and Installation  
setup: ## Initial project setup (Docker-first development)
	@echo "$(YELLOW)🚀 Setting up genealogy-extractor project (Docker)...$(NC)"
	$(MAKE) install-dev
	$(MAKE) up-build
	sleep 15
	$(MAKE) migrate
	@echo "$(GREEN)✅ Setup complete! Run 'make help' to see available commands$(NC)"
	@echo "$(GREEN)🌐 Access the application at: http://localhost:8000/admin/$(NC)"

install: ## Install production dependencies
	@echo "$(YELLOW)📦 Installing production dependencies...$(NC)"
	pip install -r requirements.txt

install-dev: ## Install development dependencies (includes linting tools)
	@echo "$(YELLOW)📦 Installing development dependencies...$(NC)"
	pip install -r requirements.txt
	@if command -v pre-commit >/dev/null 2>&1; then \
		echo "$(YELLOW)🪝 Installing pre-commit hooks...$(NC)"; \
		pre-commit install; \
	fi

##@ Code Quality
lint: ## Run all linting checks (ruff)
	@echo "$(YELLOW)🔍 Running ruff linting...$(NC)"
	ruff check .

lint-fix: ## Run linting with auto-fix
	@echo "$(YELLOW)🔧 Running ruff linting with auto-fix...$(NC)"
	ruff check . --fix

format: ## Format code with ruff
	@echo "$(YELLOW)✨ Formatting code with ruff...$(NC)"
	ruff format .

format-check: ## Check code formatting without making changes
	@echo "$(YELLOW)📏 Checking code formatting...$(NC)"
	ruff format --check .

type-check: ## Run type checking with mypy
	@echo "$(YELLOW)🔍 Running type checking...$(NC)"
	mypy genealogy genealogy_extractor --config-file pyproject.toml

security: ## Run security checks with bandit (development config)
	@echo "$(YELLOW)🛡️ Running security checks (development)...$(NC)"
	bandit -r genealogy genealogy_extractor -f txt -c .bandit

security-prod: ## Run strict security checks for production
	@echo "$(YELLOW)🛡️ Running strict security checks (production)...$(NC)"
	bandit -r genealogy genealogy_extractor -f txt -c .bandit.prod

quality-gate: ensure-containers ## Run complete quality gate (linting, formatting, types, security, tests)
	@echo "$(YELLOW)🚪 Running complete quality gate...$(NC)"
	@echo "=================================="
	$(MAKE) lint
	$(MAKE) format-check  
	$(MAKE) type-check
	$(MAKE) security
	$(MAKE) django-check
	$(MAKE) test
	@echo "$(GREEN)🎉 All quality checks passed!$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(YELLOW)🪝 Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

##@ Django Development
ensure-containers: ## Ensure Docker containers are running
	@if ! docker compose ps web | grep -q "Up"; then \
		echo "$(YELLOW)🚀 Starting Docker containers...$(NC)"; \
		docker compose up -d; \
		echo "$(YELLOW)⏳ Waiting for containers to be ready...$(NC)"; \
		sleep 10; \
	fi

django-check: ensure-containers ## Run Django system checks in Docker
	@echo "$(YELLOW)🔍 Running Django system checks in Docker...$(NC)"
	docker compose exec web python manage.py check

migrate: ensure-containers ## Run Django migrations in Docker
	@echo "$(YELLOW)📊 Running Django migrations in Docker...$(NC)"
	docker compose exec web python manage.py migrate

makemigrations: ensure-containers ## Create new Django migrations in Docker
	@echo "$(YELLOW)📝 Creating Django migrations in Docker...$(NC)"
	docker compose exec web python manage.py makemigrations

createsuperuser: ensure-containers ## Create Django superuser in Docker
	@echo "$(YELLOW)👤 Creating Django superuser in Docker...$(NC)"
	docker compose exec web python manage.py createsuperuser

collectstatic: ensure-containers ## Collect static files in Docker
	@echo "$(YELLOW)📁 Collecting static files in Docker...$(NC)"
	docker compose exec web python manage.py collectstatic --noinput

runserver: ensure-containers ## Run Django development server (use 'make up' instead)
	@echo "$(YELLOW)🌐 Django server is running in Docker at http://localhost:8000$(NC)"
	@echo "$(YELLOW)💡 Use 'make logs-web' to view logs$(NC)"

shell: ensure-containers ## Open Django shell in Docker
	@echo "$(YELLOW)🐚 Opening Django shell in Docker...$(NC)"
	docker compose exec web python manage.py shell

##@ Testing
test: ensure-containers ## Run all tests in Docker
	@echo "$(YELLOW)🧪 Running all tests in Docker...$(NC)"
	docker compose exec web python manage.py test genealogy.tests

test-fast: ensure-containers ## Run tests with faster settings in Docker
	@echo "$(YELLOW)⚡ Running tests (fast mode) in Docker...$(NC)"
	docker compose exec web python manage.py test genealogy.tests --parallel --keepdb

test-coverage: ensure-containers ## Run tests with coverage report in Docker
	@echo "$(YELLOW)📊 Running tests with coverage in Docker...$(NC)"
	docker compose exec web coverage run --source='.' manage.py test genealogy.tests
	docker compose exec web coverage report
	docker compose exec web coverage html

test-models: ensure-containers ## Run only model tests in Docker
	@echo "$(YELLOW)🗃️ Running model tests in Docker...$(NC)"
	docker compose exec web python manage.py test genealogy.tests.test_models

test-admin: ensure-containers ## Run only admin tests in Docker
	@echo "$(YELLOW)⚙️ Running admin tests in Docker...$(NC)"
	docker compose exec web python manage.py test genealogy.tests.test_admin

test-tasks: ensure-containers ## Run only task tests in Docker
	@echo "$(YELLOW)⚡ Running task tests in Docker...$(NC)"
	docker compose exec web python manage.py test genealogy.tests.test_tasks

test-ocr: ensure-containers ## Run OCR workflow test in Docker
	@echo "$(YELLOW)👁️ Running OCR workflow test in Docker...$(NC)"
	docker compose exec web python test-ocr-workflow.py

##@ Docker Commands
build: ## Build Docker containers
	@echo "$(YELLOW)🐳 Building Docker containers...$(NC)"
	docker compose build

up: ## Start all Docker services
	@echo "$(YELLOW)🚀 Starting Docker services...$(NC)"
	docker compose up -d

up-build: ## Build and start all Docker services
	@echo "$(YELLOW)🚀 Building and starting Docker services...$(NC)"
	docker compose up --build -d

down: ## Stop all Docker services
	@echo "$(YELLOW)⏹️ Stopping Docker services...$(NC)"
	docker compose down

down-volumes: ## Stop Docker services and remove volumes (WARNING: deletes data)
	@echo "$(RED)⚠️ Stopping Docker services and removing volumes...$(NC)"
	docker compose down -v

logs: ## View Docker logs for all services
	@echo "$(YELLOW)📋 Viewing Docker logs...$(NC)"
	docker compose logs -f

logs-web: ## View Docker logs for web service
	@echo "$(YELLOW)📋 Viewing web service logs...$(NC)"
	docker compose logs -f web

logs-celery: ## View Docker logs for celery service
	@echo "$(YELLOW)📋 Viewing celery service logs...$(NC)"
	docker compose logs -f celery

shell-web: ## Open shell in web container
	@echo "$(YELLOW)🐚 Opening shell in web container...$(NC)"
	docker compose exec web bash

shell-db: ## Open database shell
	@echo "$(YELLOW)🗃️ Opening database shell...$(NC)"
	docker compose exec db psql -U postgres genealogy_extractor


reset-db: ## Reset database (WARNING: deletes all data)
	@echo "$(RED)⚠️ Resetting database...$(NC)"
	docker compose down
	docker volume rm genealogy_extractor_postgres_data || true
	docker compose up -d db
	sleep 5
	$(MAKE) migrate

##@ Cleanup
clean: ## Clean up temporary files and caches
	@echo "$(YELLOW)🧹 Cleaning up temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

clean-docker: ## Clean up Docker containers and images
	@echo "$(YELLOW)🐳 Cleaning up Docker...$(NC)"
	docker compose down --rmi all --volumes --remove-orphans

##@ Development Workflow
dev-setup: ## Complete development setup (Docker + dependencies)
	@echo "$(YELLOW)🏗️ Setting up development environment...$(NC)"
	$(MAKE) install-dev
	$(MAKE) up-build
	sleep 10
	$(MAKE) migrate
	@echo "$(GREEN)✅ Development environment ready!$(NC)"
	@echo "$(GREEN)🌐 Access the application at: http://localhost:8000/admin/$(NC)"

dev-reset: ## Reset development environment
	@echo "$(YELLOW)🔄 Resetting development environment...$(NC)"
	$(MAKE) down-volumes
	$(MAKE) clean
	$(MAKE) dev-setup

ci: ## Run CI pipeline (quality gate + tests)
	@echo "$(YELLOW)🚀 Running CI pipeline...$(NC)"
	$(MAKE) quality-gate
	@echo "$(GREEN)✅ CI pipeline completed successfully!$(NC)"

# Quick aliases
start: up ## Alias for 'up'
stop: down ## Alias for 'down'
restart: down up ## Restart Docker services