"""
Tests for AgentExecutor - agentic workflow execution.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from genealogy.services.agent_executor import AgentExecutor


class TestAgentExecutor:
    """Test the agent executor for multi-turn reasoning"""

    def test_parse_tool_call_response(self):
        """Test parsing a tool call response from LLM"""
        agent = AgentExecutor()

        response = """TOOL_CALL: search_person_by_name
ARGUMENTS: {"name": "Pieter van Zanten", "max_results": 5}
REASONING: Need to find people named Pieter van Zanten"""

        result = agent._parse_response(response)

        assert result["type"] == "tool_call"
        assert result["tool"] == "search_person_by_name"
        assert result["arguments"] == {"name": "Pieter van Zanten", "max_results": 5}
        assert "Pieter van Zanten" in result["reasoning"]

    def test_parse_answer_response(self):
        """Test parsing a final answer response from LLM"""
        agent = AgentExecutor()

        response = """ANSWER: Pieter van Zanten was born in Amsterdam in 1845."""

        result = agent._parse_response(response)

        assert result["type"] == "answer"
        assert "Pieter van Zanten" in result["answer"]
        assert "1845" in result["answer"]

    def test_parse_simple_args(self):
        """Test parsing simple key=value argument format"""
        agent = AgentExecutor()

        args_str = 'name="Pieter van Zanten", max_results=10'
        result = agent._parse_simple_args(args_str)

        assert result["name"] == "Pieter van Zanten"
        assert result["max_results"] == 10

    def test_parse_simple_args_with_spaces(self):
        """Test parsing arguments with various formats"""
        agent = AgentExecutor()

        args_str = "name='Pieter', birth_year_min=1840, birth_year_max=1850"
        result = agent._parse_simple_args(args_str)

        assert result["name"] == "Pieter"
        assert result["birth_year_min"] == 1840
        assert result["birth_year_max"] == 1850

    def test_execute_tool_success(self):
        """Test executing a tool successfully"""
        agent = AgentExecutor()

        # Mock the tools.search_person_by_name method
        agent.tools.search_person_by_name = Mock(return_value={
            "count": 1,
            "people": [{"name": "Pieter van Zanten"}]
        })

        result = agent._execute_tool(
            "search_person_by_name",
            {"name": "Pieter", "max_results": 10}
        )

        assert "count" in result
        assert result["count"] == 1
        agent.tools.search_person_by_name.assert_called_once_with(
            name="Pieter",
            max_results=10
        )

    def test_execute_tool_unknown(self):
        """Test executing an unknown tool returns error"""
        agent = AgentExecutor()

        result = agent._execute_tool("unknown_tool", {})

        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_tool_invalid_args(self):
        """Test executing a tool with invalid arguments"""
        agent = AgentExecutor()

        # Mock the tool to raise TypeError
        agent.tools.search_person_by_name = Mock(side_effect=TypeError("missing required argument"))

        result = agent._execute_tool("search_person_by_name", {})

        assert "error" in result
        assert "Invalid arguments" in result["error"]

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_execute_immediate_answer(self, mock_ollama_class):
        """Test execution when LLM provides immediate answer without tool calls"""
        # Mock OllamaClient
        mock_ollama = Mock()
        mock_ollama.generate.return_value = "ANSWER: Pieter van Zanten was born in 1845."
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        result = agent.execute("Who is Pieter van Zanten?")

        assert result["success"] is True
        assert result["iterations"] == 1
        assert len(result["tool_calls"]) == 0
        assert "Pieter van Zanten" in result["answer"]
        assert "1845" in result["answer"]

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_execute_with_tool_call(self, mock_ollama_class):
        """Test execution with one tool call then answer"""
        # Mock OllamaClient
        mock_ollama = Mock()

        # First call: tool call, Second call: answer
        mock_ollama.generate.side_effect = [
            'TOOL_CALL: search_person_by_name\nARGUMENTS: {"name": "Pieter"}\nREASONING: Need to search',
            'ANSWER: Found Pieter van Zanten, born 1845.'
        ]
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        # Mock the tool
        agent.tools.search_person_by_name = Mock(return_value={
            "count": 1,
            "people": [{"name": "Pieter van Zanten", "birth": {"date": "1845-01-01"}}]
        })

        result = agent.execute("Who is Pieter van Zanten?")

        assert result["success"] is True
        assert result["iterations"] == 2
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "search_person_by_name"
        assert "Pieter van Zanten" in result["answer"]

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_execute_max_iterations(self, mock_ollama_class):
        """Test that execution stops at max iterations"""
        # Mock OllamaClient
        mock_ollama = Mock()

        # Always return tool calls, never answer
        mock_ollama.generate.return_value = 'TOOL_CALL: search_person_by_name\nARGUMENTS: {"name": "Test"}\nREASONING: Searching'
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor(max_iterations=3)
        agent.ollama = mock_ollama

        # Mock the tool
        agent.tools.search_person_by_name = Mock(return_value={"count": 0, "people": []})

        result = agent.execute("Test query")

        assert result["success"] is False
        assert result["iterations"] == 3
        assert len(result["tool_calls"]) == 3
        assert "Max iterations" in result["error"]

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_execute_llm_error(self, mock_ollama_class):
        """Test handling of LLM errors"""
        # Mock OllamaClient
        mock_ollama = Mock()
        mock_ollama.generate.side_effect = Exception("Connection error")
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        result = agent.execute("Test query")

        assert result["success"] is False
        assert result["answer"] is None
        assert "LLM error" in result["error"]

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_execute_empty_response(self, mock_ollama_class):
        """Test handling of empty LLM response"""
        # Mock OllamaClient
        mock_ollama = Mock()
        mock_ollama.generate.return_value = None
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        result = agent.execute("Test query")

        assert result["success"] is False
        assert "empty response" in result["error"]

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_execute_with_initial_context(self, mock_ollama_class):
        """Test execution with initial RAG context"""
        # Mock OllamaClient
        mock_ollama = Mock()
        mock_ollama.generate.return_value = "ANSWER: Based on the context, Pieter was born in 1845."
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        initial_context = "Pieter van Zanten was born in Amsterdam in 1845."
        result = agent.execute("When was Pieter born?", initial_context=initial_context)

        assert result["success"] is True
        assert "1845" in result["answer"]

        # Check that initial context was passed to LLM
        call_args = mock_ollama.generate.call_args
        prompt = call_args[1]["prompt"]
        assert "Amsterdam" in prompt
        assert "1845" in prompt

    @patch('genealogy.services.agent_executor.OllamaClient')
    @patch('genealogy.services.agent_executor.time')
    def test_execute_timeout(self, mock_time, mock_ollama_class):
        """Test that execution respects timeout"""
        # Mock time to simulate timeout
        mock_time.time.side_effect = [0, 400]  # Start time, then after timeout

        # Mock OllamaClient
        mock_ollama = Mock()
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor(timeout=300)
        agent.ollama = mock_ollama

        result = agent.execute("Test query")

        assert result["success"] is False
        assert "Timeout" in result["error"]

    def test_build_prompt_structure(self):
        """Test that prompt is properly structured"""
        agent = AgentExecutor()

        prompt = agent._build_agent_prompt(
            user_query="Who is Pieter?",
            context="Some context about Pieter",
            tool_calls_made=[]
        )

        # Check key sections are present
        assert "CURRENT CONTEXT:" in prompt
        assert "USER QUERY:" in prompt
        assert "AVAILABLE TOOLS:" in prompt
        assert "RESPONSE FORMAT:" in prompt
        assert "TOOL_CALL:" in prompt
        assert "ANSWER:" in prompt

    def test_build_prompt_with_tool_calls(self):
        """Test prompt includes previous tool calls"""
        agent = AgentExecutor()

        tool_calls = [
            {
                "iteration": 1,
                "tool": "search_person_by_name",
                "arguments": {"name": "Pieter"}
            }
        ]

        prompt = agent._build_agent_prompt(
            user_query="Who is Pieter?",
            context="",
            tool_calls_made=tool_calls
        )

        assert "PREVIOUS TOOL CALLS:" in prompt
        assert "search_person_by_name" in prompt
        assert "1/" in prompt  # Shows 1 of max iterations

    @patch('genealogy.services.agent_executor.OllamaClient')
    def test_execute_multiple_tool_calls(self, mock_ollama_class):
        """Test execution with multiple sequential tool calls"""
        # Mock OllamaClient
        mock_ollama = Mock()

        # Sequence: search -> get_details -> answer
        mock_ollama.generate.side_effect = [
            'TOOL_CALL: search_person_by_name\nARGUMENTS: {"name": "Pieter"}\nREASONING: Search',
            'TOOL_CALL: get_person_details\nARGUMENTS: {"person_id": "123"}\nREASONING: Get details',
            'ANSWER: Pieter van Zanten was born in 1845.'
        ]
        mock_ollama_class.return_value = mock_ollama

        agent = AgentExecutor()
        agent.ollama = mock_ollama

        # Mock the tools
        agent.tools.search_person_by_name = Mock(return_value={
            "count": 1,
            "people": [{"id": "123", "name": "Pieter van Zanten"}]
        })
        agent.tools.get_person_details = Mock(return_value={
            "display_name": "Pieter van Zanten",
            "events": [{"type": "Birth", "date": "1845-01-01"}]
        })

        result = agent.execute("Who is Pieter van Zanten?")

        assert result["success"] is True
        assert result["iterations"] == 3
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["tool"] == "search_person_by_name"
        assert result["tool_calls"][1]["tool"] == "get_person_details"
