# Code Quality Gate

This document outlines the code quality standards and tools used in the genealogy extractor project.

## Quality Gate Process

All code must pass the quality gate before being committed. Run the complete quality gate with:

```bash
make quality-gate
```

This runs the following checks in sequence:

### 1. Linting with Ruff ✅
- **Tool**: [Ruff](https://docs.astral.sh/ruff/) - Fast Python linter
- **Command**: `make lint`
- **Auto-fix**: `make lint-fix`
- **Config**: `pyproject.toml`

**Rules enabled:**
- Code style (pycodestyle E/W)
- Import sorting (isort)
- Code complexity (pylint)
- Security checks (bandit)
- Django best practices
- Type checking compatibility

**Rules ignored for Django:**
- `RUF012`: Mutable class attributes (Django admin config)
- `SLF001`: Private member access (`_meta` is common in Django)
- `DJ001`: null=True on string fields (sometimes needed)

### 2. Code Formatting ✅
- **Tool**: Ruff formatter (Black-compatible)
- **Command**: `make format-check`
- **Auto-format**: `make format`
- **Style**: 88 character line length, double quotes

### 3. Type Checking ✅
- **Tool**: [MyPy](https://mypy.readthedocs.io/) with django-stubs
- **Command**: `make type-check`
- **Config**: `pyproject.toml`
- **Coverage**: Gradual typing (lenient to start, will tighten over time)

### 4. Security Scanning ✅
- **Tool**: [Bandit](https://bandit.readthedocs.io/)
- **Command**: `make security`
- **Scope**: Identifies common security issues
- **Exclusions**: Test files (expected to have assertions)

### 5. Django System Checks ✅
- **Tool**: Django's built-in check framework
- **Command**: `make django-check`
- **Scope**: Django configuration, model validation, etc.

### 6. Test Suite ✅
- **Tool**: Django's test runner
- **Command**: `make test`
- **Coverage**: All genealogy app functionality
- **Types**: Unit tests, integration tests, admin tests

## Development Workflow

### Local Development
```bash
# Install development dependencies
make install-dev

# Before committing - run quality gate
make quality-gate

# Quick fixes
make lint-fix && make format
```

### Docker Development
```bash
# Run quality gate in Docker
docker-compose exec web make quality-gate

# Run individual checks
docker-compose exec web make lint
docker-compose exec web make test
```

### CI/CD Integration
```bash
# Complete CI pipeline
make ci
```

## Configuration Files

- **`pyproject.toml`** - Ruff, MyPy, and tool configuration
- **`.pre-commit-config.yaml`** - Pre-commit hooks (when using git)
- **`Makefile`** - Development commands and quality gate

## Quality Metrics

| Tool | Current Status | Target |
|------|----------------|--------|
| Ruff Linting | 🟡 123 issues | 🟢 < 50 issues |
| Code Formatting | 🟢 Consistent | 🟢 Consistent |
| Type Coverage | 🟡 Basic | 🟢 Comprehensive |
| Security | 🟢 Clean | 🟢 Clean |
| Test Coverage | 🟢 34/34 passing | 🟢 > 90% coverage |

## Ignored Rules Rationale

The following rules are intentionally ignored for Django development:

- **Long lines in admin.py**: Django admin configuration often requires longer lines
- **Print statements in scripts**: Test and utility scripts need console output  
- **Mutable class attributes**: Django admin/model configuration uses lists/dicts
- **Private member access**: Django's `_meta` attribute is commonly used
- **Unittest style assertions**: Prefer Django's TestCase assertions over plain assert

## Future Improvements

- [ ] Add test coverage reporting
- [ ] Implement git pre-commit hooks
- [ ] Tighten MyPy strictness gradually
- [ ] Add import sorting validation
- [ ] Configure IDE integration docs