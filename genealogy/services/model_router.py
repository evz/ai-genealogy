"""
Model router service for intelligent LLM model selection.

Provides a clean interface for routing queries to the optimal model based on
query characteristics, with pluggable strategies for flexibility.
"""

import logging
from typing import List

from .routing_strategies import KeywordBasedStrategy, RoutingStrategy

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Routes queries to optimal LLM models based on configurable strategies.

    Usage:
        router = ModelRouter()
        model = router.route(query="Tell me about Jan Pieters", chunks=chunks)
        # Returns: "gene-chat-fast"

        model = router.route(
            query="Are these the same person?",
            chunks=chunks
        )
        # Returns: "gene-reasoner"
    """

    def __init__(self, strategy: RoutingStrategy | None = None):
        """
        Initialize router with a strategy.

        Args:
            strategy: Routing strategy to use. Defaults to KeywordBasedStrategy.
        """
        self.strategy = strategy or KeywordBasedStrategy()

    def route(
        self,
        query: str,
        chunks: List[dict] | None = None,
        use_agent: bool = False
    ) -> str:
        """
        Route a query to the optimal model.

        Args:
            query: The user's query text
            chunks: Retrieved chunks (if in RAG mode)
            use_agent: Whether agent mode is being used

        Returns:
            Model name (e.g., "gene-chat-fast", "gene-chat-main", "gene-reasoner")
        """
        model = self.strategy.choose_model(query, chunks, use_agent)

        logger.info(
            f"Routed query to model: {model}",
            extra={
                "query_preview": query[:100],
                "model": model,
                "use_agent": use_agent,
                "num_chunks": len(chunks) if chunks else 0,
                "strategy": self.strategy.__class__.__name__
            }
        )

        return model

    def set_strategy(self, strategy: RoutingStrategy):
        """
        Change the routing strategy.

        Args:
            strategy: New strategy to use
        """
        logger.info(
            f"Changing routing strategy from {self.strategy.__class__.__name__} "
            f"to {strategy.__class__.__name__}"
        )
        self.strategy = strategy
