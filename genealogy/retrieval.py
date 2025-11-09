"""
Hybrid RAG+RRF retrieval system combining:
- Vector similarity search (semantic)
- Trigram fuzzy matching (spelling variants)
- Daitch-Mokotoff phonetic matching (surname variants)

Uses Reciprocal Rank Fusion to combine results from all three approaches.
"""

import json
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
        subject_limit: int = 15,
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
        # Handle empty query
        if not query or not query.strip():
            return []

        # 1. Prepare query features
        query_features = self._extract_query_features(query)

        # 1.5. Extract potential person names from query for pre-filtering
        person_filter = self._extract_person_names_from_query(query)

        # 2. Run hybrid search
        results = self._hybrid_search(
            query_text=query_features["text"],
            query_embedding=query_features["embedding"],
            query_dm_codes=query_features["dm_codes"],
            top_k=top_k,
            vec_limit=vec_limit,
            trigram_limit=trigram_limit,
            phonetic_limit=phonetic_limit,
            subject_limit=subject_limit,
            person_filter=person_filter,
        )

        # 2.5. Post-boost chunks where subject matches query names
        if person_filter:
            results = self._boost_subject_matches(results, person_filter)

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

    def _extract_person_names_from_query(self, query: str) -> Optional[List[str]]:
        """
        Extract potential person names from the query for pre-filtering.

        Looks for capitalized names and common Dutch/genealogical surname patterns.
        """
        # Pattern: Capitalized words that might be names
        # Common pattern: "FirstName van/de/der Surname"
        names = []

        # Extract capitalized words (potential names)
        words = re.findall(r'\b[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+\b', query)

        # Filter out common English/Dutch question words
        stop_words = {'Tell', 'Who', 'What', 'Where', 'When', 'How', 'Was', 'Were', 'About', 'The'}
        names = [w for w in words if w not in stop_words]

        # If we found names, return them
        if names:
            logger.info(f"Extracted potential person names from query: {names}")
            return names

        return None

    def _hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        query_dm_codes: List[str],
        top_k: int,
        vec_limit: int,
        trigram_limit: int,
        phonetic_limit: int,
        subject_limit: int,
        person_filter: Optional[List[str]] = None,
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

        # Build person filter WHERE clause if names detected
        person_filter_clause = ""
        if person_filter:
            # Create ILIKE conditions for subject or extracted_people
            # Use AND to require ALL names be present (not just any one)
            name_conditions = []
            for name in person_filter:
                safe_name = name.replace("'", "''")
                # Match in subject field OR check if any array element contains the name
                name_conditions.append(f"(subject ILIKE '%{safe_name}%' OR EXISTS (SELECT 1 FROM unnest(extracted_people) AS person WHERE person ILIKE '%{safe_name}%'))")
            person_filter_clause = " AND (" + " AND ".join(name_conditions) + ")"
            logger.info(f"Applying person filter: {person_filter}")

        # SQL query with four-leg hybrid search + RRF fusion
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
            WHERE  embedding IS NOT NULL{person_filter_clause}
            ORDER  BY embedding <=> q_vec
            LIMIT  {vec_limit}
        ),

        trgm AS (
            SELECT id,
                   row_number() OVER (ORDER BY similarity(text_content, q_text) DESC) AS tg_rank
            FROM   genealogy_textchunk, params
            WHERE  text_content % q_text{person_filter_clause}
            LIMIT  {trigram_limit}
        ),

        phon AS (
            SELECT id,
                   row_number() OVER () AS ph_rank
            FROM   genealogy_textchunk, params
            WHERE  dm_codes::text[] && q_dm{person_filter_clause}
            LIMIT  {phonetic_limit}
        ),

        subj AS (
            SELECT id,
                   row_number() OVER (ORDER BY similarity(subject, q_text) DESC) AS subj_rank
            FROM   genealogy_textchunk, params
            WHERE  subject IS NOT NULL
                   AND subject != ''
                   AND subject % q_text{person_filter_clause}
            LIMIT  {subject_limit}
        ),

        rrf AS (
            SELECT id, 1.0/(60+vec_rank) AS score FROM vec
            UNION ALL
            SELECT id, 1.0/(60+tg_rank)  AS score FROM trgm
            UNION ALL
            SELECT id, 1.0/(80+ph_rank)  AS score FROM phon
            UNION ALL
            SELECT id, 1.0/(10+subj_rank) AS score FROM subj
        )

        SELECT
            c.id,
            c.text_content,
            c.document_id,
            c.sequence_number,
            c.start_page,
            c.end_page,
            c.chunk_type,
            c.genealogical_identifier,
            c.subject,
            c.generation_number,
            c.generation_header,
            c.family_groups,
            c.extracted_people,
            c.extracted_events,
            c.extracted_relationships,
            SUM(rrf.score) AS rrf_score
        FROM   rrf
        JOIN   genealogy_textchunk c ON c.id = rrf.id
        GROUP  BY c.id, c.text_content, c.document_id, c.sequence_number,
                  c.start_page, c.end_page, c.chunk_type, c.genealogical_identifier,
                  c.subject, c.generation_number, c.generation_header, c.family_groups,
                  c.extracted_people, c.extracted_events, c.extracted_relationships
        ORDER  BY rrf_score DESC
        LIMIT  {top_k};
        """

        with connection.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse JSONB fields that come back as strings from raw SQL
                result = self._parse_jsonb_fields(result)
                results.append(result)

        logger.info(f"Hybrid search returned {len(results)} results for query: {query_text[:50]}...")
        return results

    def _boost_subject_matches(self, results: List[Dict], person_filter: List[str]) -> List[Dict]:
        """
        Post-process results to boost chunks where subject exactly matches query names.

        This helps surface the actual person's entry when their name appears in many chunks.
        """
        for result in results:
            subject = result.get("subject", "")
            if subject:
                # Check if any filter name appears in the subject
                for name in person_filter:
                    if name.lower() in subject.lower():
                        # Boost the RRF score significantly
                        original_score = result.get("rrf_score", 0.0)
                        result["rrf_score"] = float(original_score) * 2.0  # Double the score
                        logger.debug(f"Boosted {result.get('genealogical_identifier')} (subject: {subject}) from {original_score:.4f} to {result['rrf_score']:.4f}")
                        break

        # Re-sort by the new scores
        results.sort(key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)

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
                genealogical_identifier,
                subject,
                generation_number,
                generation_header,
                family_groups,
                extracted_people,
                extracted_events,
                extracted_relationships
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
                    # Parse JSONB fields that come back as strings from raw SQL
                    chunk_dict = self._parse_jsonb_fields(chunk_dict)
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

    def _parse_jsonb_fields(self, chunk_dict: Dict) -> Dict:
        """
        Parse JSONB fields that come back as strings from raw SQL queries.

        The Django ORM handles this automatically, but raw SQL returns JSONB as strings.
        """
        jsonb_fields = ['family_groups', 'extracted_people', 'extracted_events', 'extracted_relationships']

        for field in jsonb_fields:
            if field in chunk_dict and isinstance(chunk_dict[field], str):
                try:
                    chunk_dict[field] = json.loads(chunk_dict[field])
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, keep as-is or set to empty list
                    logger.warning(f"Failed to parse JSONB field '{field}' for chunk")
                    chunk_dict[field] = []

        return chunk_dict

    def build_context(self, chunks: List[Dict], include_anchors: bool = True, include_enrichment: bool = False) -> str:
        """
        Build formatted context string from retrieved chunks for LLM prompt.

        Args:
            chunks: Retrieved chunk dictionaries
            include_anchors: Whether to include anchor metadata in context
            include_enrichment: Whether to include extracted people, events, relationships

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
                genealogical_id = chunk.get("genealogical_identifier")
                if genealogical_id:
                    anchor_parts.append(f"[{genealogical_id}]")

                # Page number
                anchor_parts.append(f"(page {chunk['start_page']})")

                # Chunk type for context
                chunk_type = chunk.get("chunk_type", "")
                if chunk_type == "individual_entry":
                    anchor_parts.append("INDIVIDUAL_ENTRY")
                elif chunk_type == "generation_header":
                    anchor_parts.append("GENERATION_HEADER")
                elif chunk_type == "family_group_header":
                    anchor_parts.append("FAMILY_GROUP_HEADER")

                anchor = " ".join(anchor_parts)
                chunk_parts = [f"--- {anchor} ---"]

                # Add subject if available
                subject = chunk.get("subject")
                if subject:
                    chunk_parts.append(f"Subject: {subject}")

                # Add generation info if available
                generation_number = chunk.get("generation_number")
                generation_header = chunk.get("generation_header")
                if generation_number:
                    gen_text = f"Generation: {generation_number}"
                    if generation_header:
                        gen_text += f" ({generation_header})"
                    chunk_parts.append(gen_text)

                # Add family group info if available
                family_groups = chunk.get("family_groups")
                if family_groups and len(family_groups) > 0:
                    chunk_parts.append(f"Family: {', '.join(family_groups)}")

                # Add enrichment if requested
                if include_enrichment:
                    # People mentioned
                    extracted_people = chunk.get("extracted_people")
                    if extracted_people and len(extracted_people) > 0:
                        people_list = ', '.join(extracted_people[:5])  # First 5 names
                        if len(extracted_people) > 5:
                            people_list += f" (and {len(extracted_people) - 5} more)"
                        chunk_parts.append(f"\nPEOPLE MENTIONED: {people_list}")

                    # Events
                    extracted_events = chunk.get("extracted_events")
                    if extracted_events and len(extracted_events) > 0:
                        chunk_parts.append("\nEVENTS:")
                        for event in extracted_events[:10]:  # First 10 events
                            # Skip if event is not a dict (defensive coding)
                            if not isinstance(event, dict):
                                continue
                            event_parts = []
                            if event.get("event_type"):
                                event_parts.append(event["event_type"])
                            if event.get("person"):
                                event_parts.append(f"({event['person']})")
                            if event.get("date"):
                                event_parts.append(event["date"])
                            if event.get("place"):
                                event_parts.append(event["place"])
                            if event_parts:
                                chunk_parts.append(f"  - {': '.join(event_parts)}")

                    # Relationships
                    extracted_relationships = chunk.get("extracted_relationships")
                    if extracted_relationships and len(extracted_relationships) > 0:
                        chunk_parts.append("\nRELATIONSHIPS:")
                        for rel in extracted_relationships[:10]:  # First 10 relationships
                            person1 = rel.get("person1", "")
                            person2 = rel.get("person2", "")
                            rel_type = rel.get("relationship_type", "related to")
                            if person1 and person2:
                                chunk_parts.append(f"  - {person1} ({rel_type}) {person2}")

                chunk_parts.append(f"\nCHUNK TEXT:\n{chunk['text_content']}\n")
                context_parts.append("\n".join(chunk_parts))
            else:
                context_parts.append(chunk["text_content"])

        return "\n".join(context_parts)
