"""
Hybrid RAG+RRF retrieval system combining:
- Vector similarity search (semantic)
- Trigram fuzzy matching (spelling variants)
- Daitch-Mokotoff phonetic matching (surname variants)

Uses Reciprocal Rank Fusion to combine results from all three approaches.
"""

import logging
import re
from typing import Dict, List, Optional

from abydos.phonetic import DaitchMokotoff
from django.db import connection

from .ollama_utils import OllamaClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval using PostgreSQL CTEs for vector + trigram + phonetic search.
    """

    def __init__(self, embedding_model: str = "zylonai/multilingual-e5-large:latest"):
        self.ollama = OllamaClient()
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 12,
        vec_limit: int = 25,
        trigram_limit: int = 20,
        phonetic_limit: int = 40,
        expand_window: int = 2,
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid RAG+RRF.

        Args:
            query: User's question
            top_k: Final number of results to return
            vec_limit: Max results from vector search
            trigram_limit: Max results from trigram search
            phonetic_limit: Max results from phonetic search
            expand_window: Number of chunks to expand before/after each hit

        Returns:
            List of dicts with chunk data, anchor info, and RRF scores
        """
        # 1. Prepare query features
        query_features = self._extract_query_features(query)

        # 2. Run hybrid search
        results = self._hybrid_search(
            query_text=query_features["text"],
            query_embedding=query_features["embedding"],
            query_dm_codes=query_features["dm_codes"],
            top_k=top_k,
            vec_limit=vec_limit,
            trigram_limit=trigram_limit,
            phonetic_limit=phonetic_limit,
        )

        # 3. Expand results to include neighboring chunks
        if expand_window > 0:
            results = self._expand_results(results, window=expand_window)

        return results

    def _extract_query_features(self, query: str) -> Dict:
        """
        Extract features from query:
        - Cleaned text
        - Embedding vector
        - Capitalized names (for phonetic matching)
        - DM codes
        """
        # Clean query text
        text = query.strip()

        # Generate embedding
        embedding = self.ollama.embed(self.embedding_model, text)
        if not embedding:
            logger.warning("Failed to generate embedding for query")
            embedding = [0.0] * 1024  # fallback to zero vector

        # Extract capitalized names (potential surnames)
        # Pattern: word starting with capital, at least 2 chars
        names = re.findall(r'\b[A-ZÀ-Ý][A-Za-zÀ-ÿ]{1,}\b', text)

        # Generate DM codes for names
        dm_codes = []
        if names:
            dm = DaitchMokotoff()
            for name in names:
                try:
                    code_set = dm.encode(name)
                    # DaitchMokotoff.encode() returns a set of codes
                    if isinstance(code_set, set):
                        for code in code_set:
                            if code and code.strip():
                                dm_codes.append(code.strip())
                    elif code_set and code_set.strip():
                        dm_codes.append(code_set.strip())
                except Exception as e:
                    logger.warning(f"Failed to encode name '{name}': {e}")

        # Remove duplicates
        dm_codes = list(set(dm_codes))

        return {
            "text": text,
            "embedding": embedding,
            "names": names,
            "dm_codes": dm_codes,
        }

    def _hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        query_dm_codes: List[str],
        top_k: int,
        vec_limit: int,
        trigram_limit: int,
        phonetic_limit: int,
    ) -> List[Dict]:
        """
        Execute hybrid search using PostgreSQL CTE with RRF fusion.

        Based on the pattern from RAG_RRF_notes.md - all three retrieval
        legs (vector, trigram, phonetic) in one SQL query.
        """
        # Convert embedding to PostgreSQL vector literal
        emb_literal = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"

        # Convert DM codes to PostgreSQL array literal
        if query_dm_codes:
            dm_array = "ARRAY[" + ",".join(f"'{c}'" for c in query_dm_codes) + "]::text[]"
        else:
            dm_array = "ARRAY[]::text[]"  # empty array if no names found

        # Escape single quotes in query text
        safe_query = query_text.replace("'", "''")

        # SQL query with three-leg hybrid search + RRF fusion
        sql = f"""
        WITH
        params AS (
            SELECT
                '{emb_literal}'::vector AS q_vec,
                '{safe_query}' AS q_text,
                {dm_array} AS q_dm
        ),

        vec AS (
            SELECT id,
                   row_number() OVER (ORDER BY embedding <=> q_vec) AS vec_rank
            FROM   genealogy_textchunk, params
            WHERE  embedding IS NOT NULL
            ORDER  BY embedding <=> q_vec
            LIMIT  {vec_limit}
        ),

        trgm AS (
            SELECT id,
                   row_number() OVER (ORDER BY similarity(text_content, q_text) DESC) AS tg_rank
            FROM   genealogy_textchunk, params
            WHERE  text_content % q_text
            LIMIT  {trigram_limit}
        ),

        phon AS (
            SELECT id,
                   row_number() OVER () AS ph_rank
            FROM   genealogy_textchunk, params
            WHERE  dm_codes::text[] && q_dm
            LIMIT  {phonetic_limit}
        ),

        rrf AS (
            SELECT id, 1.0/(60+vec_rank) AS score FROM vec
            UNION ALL
            SELECT id, 1.0/(60+tg_rank)  AS score FROM trgm
            UNION ALL
            SELECT id, 1.0/(80+ph_rank)  AS score FROM phon
        )

        SELECT
            c.id,
            c.text_content,
            c.document_id,
            c.sequence_number,
            c.start_page,
            c.end_page,
            c.chunk_type,
            c.genealogy_ids,
            c.person_names,
            c.dates,
            c.places,
            c.addresses,
            c.occupations,
            SUM(rrf.score) AS rrf_score
        FROM   rrf
        JOIN   genealogy_textchunk c ON c.id = rrf.id
        GROUP  BY c.id, c.text_content, c.document_id, c.sequence_number,
                  c.start_page, c.end_page, c.chunk_type, c.genealogy_ids,
                  c.person_names, c.dates, c.places, c.addresses, c.occupations
        ORDER  BY rrf_score DESC
        LIMIT  {top_k};
        """

        with connection.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                results.append(result)

        logger.info(f"Hybrid search returned {len(results)} results for query: {query_text[:50]}...")
        return results

    def _expand_results(self, results: List[Dict], window: int = 2) -> List[Dict]:
        """
        Expand each result to include neighboring chunks from same document.

        This ensures we retrieve complete context around the matched chunk.
        For example, if chunk 7 matches, also retrieve chunks 5-9.
        """
        if not results:
            return results

        expanded = []
        seen_chunks = set()

        for result in results:
            doc_id = result["document_id"]
            center_seq = result["sequence_number"]

            # Get neighboring chunks from same document
            min_seq = max(0, center_seq - window)
            max_seq = center_seq + window

            sql = """
            SELECT
                id,
                text_content,
                document_id,
                sequence_number,
                start_page,
                end_page,
                chunk_type,
                genealogy_ids,
                person_names,
                dates,
                places,
                addresses,
                occupations
            FROM genealogy_textchunk
            WHERE document_id = %s
              AND sequence_number BETWEEN %s AND %s
            ORDER BY sequence_number
            """

            with connection.cursor() as cursor:
                cursor.execute(sql, [doc_id, min_seq, max_seq])
                columns = [col[0] for col in cursor.description]
                for row in cursor.fetchall():
                    chunk_dict = dict(zip(columns, row))
                    chunk_id = chunk_dict["id"]

                    # Only add if we haven't seen this chunk yet
                    if chunk_id not in seen_chunks:
                        # Preserve RRF score for center chunk, set to 0 for neighbors
                        chunk_dict["rrf_score"] = result.get("rrf_score", 0.0) if chunk_id == result["id"] else 0.0
                        chunk_dict["is_center"] = (chunk_id == result["id"])
                        expanded.append(chunk_dict)
                        seen_chunks.add(chunk_id)

        # Sort by document and sequence to maintain reading order
        expanded.sort(key=lambda x: (x["document_id"], x["sequence_number"]))

        logger.info(f"Expanded {len(results)} results to {len(expanded)} chunks (window={window})")
        return expanded

    def build_context(self, chunks: List[Dict], include_anchors: bool = True) -> str:
        """
        Build formatted context string from retrieved chunks for LLM prompt.

        Args:
            chunks: Retrieved chunk dictionaries
            include_anchors: Whether to include anchor metadata in context

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            if include_anchors:
                # Build more descriptive anchor label
                anchor_parts = []

                # Entry number for clarity
                anchor_parts.append(f"ENTRY {i}")

                # Genealogical ID if available
                if chunk.get("genealogy_ids"):
                    anchor_parts.append(f"[{chunk['genealogy_ids'][0]}]")

                # Page number
                anchor_parts.append(f"(page {chunk['start_page']})")

                # Chunk type for context
                chunk_type = chunk.get("chunk_type", "")
                if chunk_type == "GENEALOGY_ENTRY":
                    anchor_parts.append("GENEALOGY")
                elif chunk_type == "HEADER":
                    anchor_parts.append("HEADER")

                # Person names if available
                if chunk.get("person_names"):
                    names = chunk['person_names'][:2]  # Just first 2 names
                    anchor_parts.append(f"Names: {', '.join(names)}")

                anchor = " ".join(anchor_parts)
                context_parts.append(f"--- {anchor} ---\n{chunk['text_content']}\n")
            else:
                context_parts.append(chunk["text_content"])

        return "\n".join(context_parts)
