"""
Routing strategies for selecting optimal LLM models based on query characteristics.

Each strategy implements the RoutingStrategy protocol and can be swapped transparently.
"""

import os
import re
from typing import Protocol, List


# Model configuration from environment variables
MODEL_FAST = os.getenv("LLM_MODEL_FAST", "llama3.1:8b")
MODEL_MAIN = os.getenv("LLM_MODEL_MAIN", "qwen2.5:14b-instruct-q5_K_M")
MODEL_REASONER = os.getenv("LLM_MODEL_REASONER", "deepseek-r1:14b")


class RoutingStrategy(Protocol):
    """Protocol for model routing strategies"""

    def choose_model(self, query: str, chunks: List[dict] | None = None, use_agent: bool = False) -> str:
        """
        Choose the optimal model for a given query.

        Args:
            query: The user's query text
            chunks: Retrieved chunks (if in RAG mode), None if in agent mode
            use_agent: Whether agent mode is being used

        Returns:
            Model name (e.g., "gene-chat-fast", "gene-chat-main", "gene-reasoner")
        """
        ...


class KeywordBasedStrategy:
    """
    Routes based on keyword patterns in the query.

    - Detects merge/conflict/identity queries → deepseek-r1:14b (reasoning)
    - Estimates complexity from query length and chunk count → qwen2.5:14b (main)
    - Default to fast model for simple queries → llama3.1:8b

    Supports both English and Dutch queries.

    Model mappings (base models, no custom Modelfiles):
    - Fast: llama3.1:8b
    - Main: qwen2.5:14b-instruct-q5_K_M
    - Reasoner: deepseek-r1:14b
    """

    # Keywords that indicate merge/conflict/identity resolution tasks
    # Includes both English and Dutch terms
    MERGE_KEYWORDS = [
        # English
        r"\bsame person\b",
        r"\bsame man\b",
        r"\bsame woman\b",
        r"\bduplicate(s)?\b",
        r"\breconcile\b",
        r"\bmerge\b",
        r"\bwhich of these\b",
        r"\bmore likely\b",
        r"\bconflicting\b",
        r"\bcontradict(?:ion|ory)?\b",
        r"\bidentif(?:y|ying)\b.*same\b",
        # Dutch
        r"\bdezelfde\b",
        r"\bdubbel(e)?\b",
        r"\bverzoenen\b",
        r"\bsamenvoegen\b",
        r"\bwelk(?:e)? van deze\b",
        r"\bwaarschijnlijk(?:er)?\b",
        r"\btegenstrijd(?:ig|ige|igheden)?\b",
        r"\bconflict(?:en)?\b",
        r"\bidentificeren\b.*dezelfde\b",
    ]

    def choose_model(self, query: str, chunks: List[dict] | None = None, use_agent: bool = False) -> str:
        """Choose model based on keyword patterns and complexity heuristics"""

        # 0. Check for manual reasoning model override
        # User can type "use reasoning model" or "@reasoning" to force deepseek-r1
        if self._is_reasoning_override(query):
            return MODEL_REASONER

        # 1. Check for merge/conflict/identity resolution queries FIRST
        # (these should use deepseek-r1 even in agent mode)
        if self._is_merge_or_conflict_query(query):
            return MODEL_REASONER

        # 2. If using agent mode, use main model (agents need good reasoning)
        if use_agent:
            return MODEL_MAIN

        # 3. Estimate complexity based on query and retrieved chunks
        complexity = self._estimate_complexity(query, chunks)

        # 4. Route based on complexity
        if complexity >= 2:
            return MODEL_MAIN

        return MODEL_FAST

    def _is_reasoning_override(self, query: str) -> bool:
        """Check if user explicitly requested reasoning model"""
        q = query.lower()
        patterns = [
            r"@reasoning\b",
            r"\buse reasoning model\b",
            r"\buse reasoning\b",
            r"\breasoning model\b",
            r"@reasoner\b",
        ]
        return any(re.search(pat, q) for pat in patterns)

    def _is_merge_or_conflict_query(self, query: str) -> bool:
        """Check if query matches merge/conflict keywords"""
        q = query.lower()
        return any(re.search(pat, q) for pat in self.MERGE_KEYWORDS)

    def _estimate_complexity(self, query: str, chunks: List[dict] | None) -> int:
        """
        Estimate query complexity (0-3+).

        Heuristics:
        - Long query (40+ words) → +1
        - Many chunks (5+) → +1
        - Large total chunk content (8000+ chars) → +1
        """
        score = 0

        # Query length
        q_len = len(query.split())
        if q_len > 40:
            score += 1

        if chunks:
            n_chunks = len(chunks)
            total_chars = sum(len(c.get('text', '')) for c in chunks)

            # Number of chunks
            if n_chunks >= 5:
                score += 1

            # Total content size
            if total_chars > 8000:
                score += 1

        return score


class AgentOnlyStrategy:
    """
    Simple strategy for testing: always routes agent queries to reasoner model,
    everything else to fast model.
    """

    def choose_model(self, query: str, chunks: List[dict] | None = None, use_agent: bool = False) -> str:
        return MODEL_REASONER if use_agent else MODEL_FAST


class AlwaysFastStrategy:
    """Testing strategy: always use fast model"""

    def choose_model(self, query: str, chunks: List[dict] | None = None, use_agent: bool = False) -> str:
        return MODEL_FAST


class AlwaysMainStrategy:
    """Testing strategy: always use main model"""

    def choose_model(self, query: str, chunks: List[dict] | None = None, use_agent: bool = False) -> str:
        return MODEL_MAIN
