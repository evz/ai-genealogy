"""Pytest configuration for genealogy tests

This module sets up test fixtures and database configuration for the test suite.

Note: PostgreSQL extensions (vector, pg_trgm) are created automatically by
migration 0016_add_rag_rrf_fields.py, so no special setup is needed here.
"""

import pytest
