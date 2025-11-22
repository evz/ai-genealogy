"""
Unit tests for model routing strategies and router.
"""

import pytest

from genealogy.services import (
    ModelRouter,
    KeywordBasedStrategy,
    AgentOnlyStrategy,
    AlwaysFastStrategy,
    AlwaysMainStrategy,
)


class TestKeywordBasedStrategy:
    """Tests for KeywordBasedStrategy"""

    def setup_method(self):
        self.strategy = KeywordBasedStrategy()

    # Identity/merge query detection tests

    def test_merge_query_english_same_person(self):
        """Should route 'same person' queries to gene-reasoner"""
        model = self.strategy.choose_model("Are these the same person?")
        assert model == "gene-reasoner"

    def test_merge_query_english_same_man(self):
        """Should route 'same man' queries to gene-reasoner"""
        model = self.strategy.choose_model("Is this the same man in both records?")
        assert model == "gene-reasoner"

    def test_merge_query_english_duplicate(self):
        """Should route duplicate detection queries to gene-reasoner"""
        model = self.strategy.choose_model("Are these duplicates?")
        assert model == "gene-reasoner"

    def test_merge_query_english_reconcile(self):
        """Should route reconciliation queries to gene-reasoner"""
        model = self.strategy.choose_model("Can you reconcile these records?")
        assert model == "gene-reasoner"

    def test_merge_query_english_conflicting(self):
        """Should route conflicting evidence queries to gene-reasoner"""
        model = self.strategy.choose_model("These dates are conflicting")
        assert model == "gene-reasoner"

    def test_merge_query_dutch_dezelfde_persoon(self):
        """Should route Dutch 'dezelfde persoon' queries to gene-reasoner"""
        model = self.strategy.choose_model("Is dit dezelfde persoon?")
        assert model == "gene-reasoner"

    def test_merge_query_dutch_zelfde(self):
        """Should route Dutch 'zelfde' queries to gene-reasoner"""
        model = self.strategy.choose_model("Zijn dit dezelfde mensen?")
        assert model == "gene-reasoner"

    def test_merge_query_dutch_tegenstridig(self):
        """Should route Dutch conflict queries to gene-reasoner"""
        model = self.strategy.choose_model("Deze gegevens zijn tegenstrijdig")
        assert model == "gene-reasoner"

    def test_merge_query_dutch_samenvoegen(self):
        """Should route Dutch merge queries to gene-reasoner"""
        model = self.strategy.choose_model("Kunnen we deze records samenvoegen?")
        assert model == "gene-reasoner"

    # Agent mode tests

    def test_agent_mode_uses_main_model(self):
        """Agent mode should use gene-chat-main"""
        model = self.strategy.choose_model(
            "Tell me about Jan Pieters",
            use_agent=True
        )
        assert model == "gene-chat-main"

    def test_merge_detection_takes_precedence_over_agent_mode(self):
        """Merge queries should use gene-reasoner even in agent mode"""
        # Merge detection happens BEFORE agent mode check
        model = self.strategy.choose_model(
            "Are these the same person?",
            use_agent=True
        )
        assert model == "gene-reasoner"

    # Complexity-based routing tests

    def test_simple_query_routes_to_fast(self):
        """Simple queries should route to gene-chat-fast"""
        model = self.strategy.choose_model("Tell me about Jan Pieters")
        assert model == "gene-chat-fast"

    def test_long_query_alone_routes_to_fast(self):
        """Long query alone (score=1) should still route to gene-chat-fast"""
        # Complexity score 1 is not enough (need >= 2)
        long_query = " ".join(["word"] * 50)
        model = self.strategy.choose_model(long_query)
        assert model == "gene-chat-fast"

    def test_many_chunks_alone_routes_to_fast(self):
        """Many chunks alone (score=1) should still route to gene-chat-fast"""
        # Complexity score 1 is not enough (need >= 2)
        chunks = [{"text": "chunk content"} for _ in range(5)]
        model = self.strategy.choose_model("Tell me about this", chunks=chunks)
        assert model == "gene-chat-fast"

    def test_large_content_alone_routes_to_fast(self):
        """Large content alone (score=1) should still route to gene-chat-fast"""
        # Complexity score 1 is not enough (need >= 2)
        chunks = [{"text": "x" * 3000} for _ in range(3)]
        model = self.strategy.choose_model("Tell me about this", chunks=chunks)
        assert model == "gene-chat-fast"

    def test_complexity_score_2_routes_to_main(self):
        """Complexity score >= 2 should route to gene-chat-main"""
        # Long query (40+ words) + many chunks (5+) = complexity 2
        long_query = " ".join(["word"] * 50)
        chunks = [{"text": "chunk"} for _ in range(5)]
        model = self.strategy.choose_model(long_query, chunks=chunks)
        assert model == "gene-chat-main"

    def test_empty_chunks_list(self):
        """Empty chunks list should not cause errors"""
        model = self.strategy.choose_model("Tell me about Jan", chunks=[])
        assert model == "gene-chat-fast"

    def test_none_chunks(self):
        """None chunks should not cause errors"""
        model = self.strategy.choose_model("Tell me about Jan", chunks=None)
        assert model == "gene-chat-fast"

    # Edge cases

    def test_case_insensitive_keyword_matching(self):
        """Keyword matching should be case insensitive"""
        model = self.strategy.choose_model("Are These The SAME Person?")
        assert model == "gene-reasoner"

    def test_keyword_in_middle_of_query(self):
        """Keywords should be detected anywhere in query"""
        model = self.strategy.choose_model(
            "I found two records and wonder if they are the same person or not"
        )
        assert model == "gene-reasoner"


class TestAgentOnlyStrategy:
    """Tests for AgentOnlyStrategy"""

    def setup_method(self):
        self.strategy = AgentOnlyStrategy()

    def test_agent_mode_routes_to_reasoner(self):
        """Agent mode should route to gene-reasoner"""
        model = self.strategy.choose_model("Any query", use_agent=True)
        assert model == "gene-reasoner"

    def test_non_agent_mode_routes_to_fast(self):
        """Non-agent mode should route to gene-chat-fast"""
        model = self.strategy.choose_model("Any query", use_agent=False)
        assert model == "gene-chat-fast"

    def test_default_non_agent_routes_to_fast(self):
        """Default (no use_agent param) should route to gene-chat-fast"""
        model = self.strategy.choose_model("Any query")
        assert model == "gene-chat-fast"


class TestAlwaysFastStrategy:
    """Tests for AlwaysFastStrategy"""

    def setup_method(self):
        self.strategy = AlwaysFastStrategy()

    def test_always_returns_fast(self):
        """Should always return gene-chat-fast"""
        assert self.strategy.choose_model("query") == "gene-chat-fast"
        assert self.strategy.choose_model("query", use_agent=True) == "gene-chat-fast"
        assert self.strategy.choose_model(
            "Are these the same person?"
        ) == "gene-chat-fast"


class TestAlwaysMainStrategy:
    """Tests for AlwaysMainStrategy"""

    def setup_method(self):
        self.strategy = AlwaysMainStrategy()

    def test_always_returns_main(self):
        """Should always return gene-chat-main"""
        assert self.strategy.choose_model("query") == "gene-chat-main"
        assert self.strategy.choose_model("query", use_agent=True) == "gene-chat-main"
        assert self.strategy.choose_model(
            "Are these the same person?"
        ) == "gene-chat-main"


class TestModelRouter:
    """Tests for ModelRouter class"""

    def test_default_strategy_is_keyword_based(self):
        """ModelRouter should use KeywordBasedStrategy by default"""
        router = ModelRouter()
        assert isinstance(router.strategy, KeywordBasedStrategy)

    def test_custom_strategy(self):
        """ModelRouter should accept custom strategy"""
        custom_strategy = AlwaysFastStrategy()
        router = ModelRouter(strategy=custom_strategy)
        assert router.strategy == custom_strategy

    def test_route_delegates_to_strategy(self):
        """ModelRouter.route() should delegate to strategy"""
        router = ModelRouter(strategy=AlwaysFastStrategy())
        model = router.route("Any query")
        assert model == "gene-chat-fast"

    def test_set_strategy(self):
        """Should be able to change strategy"""
        router = ModelRouter(strategy=AlwaysFastStrategy())
        router.set_strategy(AlwaysMainStrategy())

        model = router.route("Any query")
        assert model == "gene-chat-main"

    def test_route_with_all_parameters(self):
        """route() should pass all parameters to strategy"""
        router = ModelRouter()
        chunks = [{"text": "test"}]

        # This should route to gene-chat-main (agent mode)
        model = router.route(
            query="Tell me about Jan",
            chunks=chunks,
            use_agent=True
        )
        assert model == "gene-chat-main"

    def test_integration_keyword_strategy(self):
        """Integration test with KeywordBasedStrategy"""
        router = ModelRouter()

        # Simple query
        assert router.route("Tell me about Jan") == "gene-chat-fast"

        # Merge query
        assert router.route("Are these the same person?") == "gene-reasoner"

        # Agent mode
        assert router.route("Tell me about Jan", use_agent=True) == "gene-chat-main"

        # Complex query
        chunks = [{"text": "x" * 2000} for _ in range(6)]
        assert router.route("Tell me about this", chunks=chunks) == "gene-chat-main"
