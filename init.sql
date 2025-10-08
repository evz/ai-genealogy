-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm extension for trigram similarity matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;
