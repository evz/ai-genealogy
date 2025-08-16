#!/bin/bash

# Quality Gate: Linting and Code Quality Checks
# This script runs all code quality checks that must pass before committing

set -e  # Exit on any error

echo "🔍 Running Code Quality Checks..."
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2 passed${NC}"
    else
        echo -e "${RED}❌ $2 failed${NC}"
        return 1
    fi
}

# Function to run a check
run_check() {
    echo -e "\n${YELLOW}Running $1...${NC}"
    if $2; then
        print_status 0 "$1"
    else
        print_status 1 "$1"
        return 1
    fi
}

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  No virtual environment detected. Activating .venv...${NC}"
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo -e "${RED}❌ No virtual environment found. Please create one with: python -m venv .venv${NC}"
        exit 1
    fi
fi

# Install dependencies if needed
if ! command -v ruff &> /dev/null; then
    echo -e "${YELLOW}📦 Installing linting dependencies...${NC}"
    pip install -r requirements.txt
fi

echo "🏠 Project: genealogy-extractor"
echo "📍 Directory: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "📦 Virtual Env: $VIRTUAL_ENV"

# 1. Ruff Linting
run_check "Ruff Linting" "ruff check ."

# 2. Ruff Formatting
run_check "Ruff Formatting" "ruff format --check ."

# 3. Type Checking with MyPy
run_check "MyPy Type Checking" "mypy genealogy genealogy_extractor --config-file pyproject.toml"

# 4. Django System Check
run_check "Django System Check" "python manage.py check"

# 5. Security Check with Bandit
run_check "Security Check (Bandit)" "bandit -r genealogy genealogy_extractor -f json -o /tmp/bandit-report.json || bandit -r genealogy genealogy_extractor"

# 6. Test Suite
run_check "Test Suite" "python manage.py test genealogy.tests --verbosity=1"

echo -e "\n${GREEN}🎉 All quality checks passed!${NC}"
echo -e "${GREEN}✨ Code is ready for commit${NC}"

# Optional: Show summary statistics
echo -e "\n📊 Summary:"
echo "- Linting: ✅ Clean"
echo "- Formatting: ✅ Consistent" 
echo "- Type Hints: ✅ Valid"
echo "- Security: ✅ Safe"
echo "- Tests: ✅ Passing"
echo "- Django: ✅ Valid"