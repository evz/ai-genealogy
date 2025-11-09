"""Chunk enrichment service for generating embeddings and DM phonetic codes

This service provides a unified interface for enriching TextChunk instances with:
- Vector embeddings (for semantic search)
- Daitch-Mokotoff phonetic codes (for fuzzy surname matching)

It can be called from:
- Entity extraction pipeline (automatic after extraction)
- Management commands (batch processing)
- Admin actions (manual enrichment)

This service handles the business logic (generating embeddings/codes).
The persistence layer (genealogy.chunking.persistence) handles saving to DB.
"""

import logging
import re
from typing import Dict, List, Optional, Set

from abydos.phonetic import DaitchMokotoff

from genealogy.chunking.persistence import save_chunk_enrichment

logger = logging.getLogger(__name__)


class ChunkEnrichmentService:
    """Service for enriching text chunks with embeddings and DM codes"""

    def __init__(self, ollama_client):
        """Initialize with Ollama client

        Args:
            ollama_client: OllamaClient instance for generating embeddings
        """
        self.ollama = ollama_client
        self.dm_encoder = DaitchMokotoff()

    def enrich_chunk(
        self,
        chunk,
        embedding_model: str = "zylonai/multilingual-e5-large:latest",
        generate_embedding: bool = True,
        generate_dm_codes: bool = True,
        force: bool = False,
    ) -> Dict[str, any]:
        """Enrich a single chunk with embeddings and/or DM codes

        Args:
            chunk: TextChunk model instance to enrich
            embedding_model: Ollama model to use for embeddings
            generate_embedding: Whether to generate embedding
            generate_dm_codes: Whether to generate DM codes
            force: Regenerate even if already exists

        Returns:
            dict with:
                - success: bool
                - embedding_generated: bool
                - dm_codes_generated: bool
                - dm_code_count: int
                - error: str (if failed)
        """
        result = {
            "success": True,
            "embedding_generated": False,
            "dm_codes_generated": False,
            "dm_code_count": 0,
        }

        try:
            embedding_to_save = None
            dm_codes_to_save = None

            # Generate embedding if requested
            if generate_embedding and (force or not chunk.embedding):
                embedding = self._generate_embedding(chunk.text_content, embedding_model)
                if embedding:
                    embedding_to_save = embedding
                    result["embedding_generated"] = True
                else:
                    result["success"] = False
                    result["error"] = "Failed to generate embedding"
                    return result

            # Generate DM codes if requested
            if generate_dm_codes and (force or not chunk.dm_codes):
                dm_codes = self._extract_dm_codes_from_names(chunk.extracted_people or [])
                dm_codes_to_save = dm_codes
                result["dm_codes_generated"] = True
                result["dm_code_count"] = len(dm_codes)

            # Save changes using persistence layer
            if embedding_to_save is not None or dm_codes_to_save is not None:
                save_result = save_chunk_enrichment(
                    chunk=chunk,
                    embedding=embedding_to_save,
                    dm_codes=dm_codes_to_save
                )
                # Verify save succeeded
                if not save_result.get("embedding_saved") and embedding_to_save is not None:
                    logger.warning(f"Failed to save embedding for chunk {chunk.id}")
                if not save_result.get("dm_codes_saved") and dm_codes_to_save is not None:
                    logger.warning(f"Failed to save DM codes for chunk {chunk.id}")

            return result

        except Exception as e:
            logger.error(f"Error enriching chunk {chunk.id}: {e}", exc_info=True)
            return {
                "success": False,
                "embedding_generated": False,
                "dm_codes_generated": False,
                "dm_code_count": 0,
                "error": str(e),
            }

    def enrich_chunks_batch(
        self,
        chunks,
        embedding_model: str = "zylonai/multilingual-e5-large:latest",
        generate_embedding: bool = True,
        generate_dm_codes: bool = True,
        force: bool = False,
    ) -> Dict[str, any]:
        """Enrich multiple chunks in batch

        Args:
            chunks: QuerySet or list of TextChunk instances
            embedding_model: Ollama model to use for embeddings
            generate_embedding: Whether to generate embeddings
            generate_dm_codes: Whether to generate DM codes
            force: Regenerate even if already exists

        Returns:
            dict with:
                - success: bool
                - processed: int (chunks successfully enriched)
                - failed: int (chunks that failed)
                - embeddings_generated: int
                - dm_codes_generated: int
                - total_dm_codes: int
                - errors: list of error messages
        """
        processed = 0
        failed = 0
        embeddings_generated = 0
        dm_codes_generated = 0
        total_dm_codes = 0
        errors = []

        for chunk in chunks:
            result = self.enrich_chunk(
                chunk=chunk,
                embedding_model=embedding_model,
                generate_embedding=generate_embedding,
                generate_dm_codes=generate_dm_codes,
                force=force,
            )

            if result["success"]:
                processed += 1
                if result["embedding_generated"]:
                    embeddings_generated += 1
                if result["dm_codes_generated"]:
                    dm_codes_generated += 1
                    total_dm_codes += result["dm_code_count"]
            else:
                failed += 1
                errors.append(f"Chunk {chunk.id}: {result.get('error', 'Unknown error')}")

        return {
            "success": failed == 0,
            "processed": processed,
            "failed": failed,
            "embeddings_generated": embeddings_generated,
            "dm_codes_generated": dm_codes_generated,
            "total_dm_codes": total_dm_codes,
            "errors": errors,
        }

    def _generate_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Generate embedding for text using Ollama

        Args:
            text: Text to embed
            model: Ollama embedding model

        Returns:
            List of floats (embedding vector) or None if failed
        """
        # Clean and prepare text
        clean_text = text.strip()
        if not clean_text:
            logger.warning("Cannot generate embedding for empty text")
            return None

        # Call Ollama API
        embedding = self.ollama.embed(model, clean_text)

        if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
            logger.warning(f"Invalid embedding response from Ollama for text: {clean_text[:50]}...")
            return None

        return embedding

    def _extract_dm_codes_from_names(self, extracted_people: List[str]) -> List[str]:
        """Extract DM codes from LLM-extracted person names

        Args:
            extracted_people: List of person names from chunk

        Returns:
            Sorted list of unique DM codes
        """
        if not extracted_people:
            return []

        dm_codes: Set[str] = set()

        for name in extracted_people:
            codes = self._get_dm_codes_for_name(name)
            dm_codes.update(codes)

        # Return as sorted list for consistency
        return sorted(list(dm_codes))

    def _get_dm_codes_for_name(self, name: str) -> List[str]:
        """Generate DM codes for a single name

        Args:
            name: Person name to encode

        Returns:
            List of DM codes for name parts
        """
        if not name or not isinstance(name, str):
            return []

        # Clean the name
        name = name.strip()
        if not name:
            return []

        codes = []

        # Split name into parts (first, middle, last names)
        parts = re.split(r'[,\s]+', name)

        for part in parts:
            part = part.strip()
            if len(part) >= 2:  # Only process names with at least 2 characters
                try:
                    # Clean the name part (remove prefixes, non-letters, etc.)
                    cleaned_part = self._clean_name_part(part)
                    if cleaned_part and len(cleaned_part) >= 2:
                        dm_code_set = self.dm_encoder.encode(cleaned_part)
                        # DaitchMokotoff.encode() returns a set of codes
                        if isinstance(dm_code_set, set):
                            for code in dm_code_set:
                                if code and code.strip():
                                    codes.append(code.strip())
                        elif dm_code_set and dm_code_set.strip():
                            codes.append(dm_code_set.strip())
                except Exception as e:
                    logger.debug(f"Failed to encode name part '{part}': {e}")

        return codes

    def _clean_name_part(self, name_part: str) -> str:
        """Clean a name part before DM encoding

        Removes common prefixes, suffixes, and non-alphabetic characters.

        Args:
            name_part: Raw name part

        Returns:
            Cleaned name part ready for encoding
        """
        # Remove common Dutch/German prefixes
        prefixes = ['van', 'der', 'de', 'du', 'von', 'da', 'di', 'del', 'ter', 'ten', 'op']
        suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'zoon', 'dochter']

        # Convert to lowercase for comparison
        lower_part = name_part.lower()

        # Remove prefixes
        for prefix in prefixes:
            if lower_part.startswith(prefix + ' '):
                name_part = name_part[len(prefix)+1:].strip()
                lower_part = name_part.lower()
                break

        # Remove suffixes
        for suffix in suffixes:
            if lower_part.endswith(' ' + suffix):
                name_part = name_part[:-len(suffix)-1].strip()
                break

        # Remove non-alphabetic characters except hyphens and apostrophes
        cleaned = re.sub(r'[^a-zA-Z\-\']', '', name_part)

        return cleaned
