"""
Agentic execution loop for multi-turn LLM reasoning.

Allows LLM to iteratively call tools until it has enough information
to answer the user's question.
"""

import json
import logging
import time
from typing import Dict, List, Optional

from genealogy.ollama_utils import OllamaClient
from genealogy.services.genealogy_tools import GenealogyTools

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Execute agentic workflow with tool calling"""

    # Tool definitions for the LLM
    AVAILABLE_TOOLS = [
        {
            "name": "search_person_by_name",
            "description": "Search for people by name. Returns list with disambiguating details (birth, death, parents).",
            "parameters": {
                "name": "Person's name (full or partial)",
                "max_results": "Maximum results (default 10)"
            }
        },
        {
            "name": "get_person_details",
            "description": "Get detailed information about a specific person by their ID or genealogical identifier.",
            "parameters": {
                "person_id": "Identity UUID or genealogical ID (e.g., 'II.3.a')"
            }
        },
        {
            "name": "search_by_birth_year",
            "description": "Search for people by name and birth year range to disambiguate.",
            "parameters": {
                "name": "Person's name",
                "birth_year_min": "Minimum birth year (optional)",
                "birth_year_max": "Maximum birth year (optional)"
            }
        },
        {
            "name": "get_children",
            "description": "Get all children of a specific person.",
            "parameters": {
                "person_id": "Identity UUID or genealogical ID"
            }
        },
        {
            "name": "get_parents",
            "description": "Get parents of a specific person.",
            "parameters": {
                "person_id": "Identity UUID or genealogical ID"
            }
        }
    ]

    def __init__(self, model: str = "llama3.1:70b", max_iterations: int = 10, timeout: int = 300):
        """
        Initialize agent executor.

        Args:
            model: LLM model to use
            max_iterations: Maximum tool calls before stopping
            timeout: Total timeout in seconds
        """
        self.model = model
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.ollama = OllamaClient(timeout=timeout)
        self.tools = GenealogyTools()

    def execute(
        self,
        user_query: str,
        initial_context: Optional[str] = None
    ) -> Dict:
        """
        Execute agentic workflow synchronously.

        Args:
            user_query: The user's question
            initial_context: Optional initial context from RAG retrieval

        Returns:
            {
                "answer": str,
                "tool_calls": List[Dict],
                "iterations": int,
                "success": bool,
                "error": Optional[str]
            }
        """
        iteration = 0
        context_parts = [initial_context] if initial_context else []
        tool_calls_made = []
        start_time = time.time()

        while iteration < self.max_iterations:
            # Check timeout
            if time.time() - start_time > self.timeout:
                return {
                    "answer": None,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration,
                    "success": False,
                    "error": f"Timeout after {self.timeout}s"
                }

            iteration += 1
            logger.info(f"Agent iteration {iteration}/{self.max_iterations}")

            # Build prompt
            prompt = self._build_agent_prompt(
                user_query=user_query,
                context="\n\n".join(context_parts),
                tool_calls_made=tool_calls_made
            )

            # Get LLM response
            try:
                response = self.ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={'num_ctx': 32768, 'temperature': 0.1}
                )

                if not response:
                    return {
                        "answer": None,
                        "tool_calls": tool_calls_made,
                        "iterations": iteration,
                        "success": False,
                        "error": "LLM returned empty response"
                    }

            except Exception as e:
                logger.exception("LLM generation failed")
                return {
                    "answer": None,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration,
                    "success": False,
                    "error": f"LLM error: {str(e)}"
                }

            # Parse response to check if it's a tool call or final answer
            action = self._parse_response(response)

            if action["type"] == "tool_call":
                # Execute tool
                tool_name = action["tool"]
                tool_args = action["arguments"]

                logger.info(f"Calling tool: {tool_name} with args: {tool_args}")

                tool_result = self._execute_tool(tool_name, tool_args)
                tool_calls_made.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                })

                # Add result to context
                context_parts.append(
                    f"TOOL RESULT ({tool_name}):\n{json.dumps(tool_result, indent=2)}"
                )

                # Continue to next iteration

            elif action["type"] == "answer":
                # LLM has final answer
                return {
                    "answer": action["answer"],
                    "tool_calls": tool_calls_made,
                    "iterations": iteration,
                    "success": True,
                    "error": None
                }

            else:
                # Unknown action type, treat as final answer
                return {
                    "answer": response,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration,
                    "success": True,
                    "error": None
                }

        # Max iterations reached
        return {
            "answer": f"Could not find complete answer after {self.max_iterations} attempts. Here's what I found:\n\n{context_parts[-1] if context_parts else 'No information gathered.'}",
            "tool_calls": tool_calls_made,
            "iterations": iteration,
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached"
        }

    def _build_agent_prompt(
        self,
        user_query: str,
        context: str,
        tool_calls_made: List[Dict]
    ) -> str:
        """Build prompt for agentic workflow"""

        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}\n  Parameters: {tool['parameters']}"
            for tool in self.AVAILABLE_TOOLS
        ])

        previous_tools = "\n".join([
            f"Iteration {call['iteration']}: Called {call['tool']} with {call['arguments']}"
            for call in tool_calls_made
        ]) if tool_calls_made else "None"

        return f"""You are a genealogy research assistant with access to tools to find information about people and relationships.

CURRENT CONTEXT:
{context if context else "No context yet - you may need to search for information."}

PREVIOUS TOOL CALLS:
{previous_tools}

USER QUERY: {user_query}

AVAILABLE TOOLS:
{tools_description}

INSTRUCTIONS:
1. If you need more information to answer the question, use a tool
2. If multiple people match the query, use search_by_birth_year or get_person_details to disambiguate
3. When you have enough information to answer confidently, provide the final answer
4. You've made {len(tool_calls_made)}/{self.max_iterations} tool calls - prioritize answering if approaching limit

RESPONSE FORMAT:
You must respond in one of two ways:

OPTION 1 - Call a tool (if you need more information):
TOOL_CALL: <tool_name>
ARGUMENTS: <json_dict_of_arguments>
REASONING: <why you're calling this tool>

OPTION 2 - Provide final answer (if you have enough information):
ANSWER: <your complete answer to the user's question>

Choose one format and respond now:"""

    def _parse_response(self, response: str) -> Dict:
        """
        Parse LLM response to determine if it's a tool call or final answer.

        Returns:
            {"type": "tool_call", "tool": str, "arguments": dict, "reasoning": str}
            or
            {"type": "answer", "answer": str}
        """
        response = response.strip()

        # Check if it's a tool call
        if response.startswith("TOOL_CALL:"):
            lines = response.split("\n")
            tool_name = None
            arguments = {}
            reasoning = ""

            for line in lines:
                if line.startswith("TOOL_CALL:"):
                    tool_name = line.replace("TOOL_CALL:", "").strip()
                elif line.startswith("ARGUMENTS:"):
                    args_str = line.replace("ARGUMENTS:", "").strip()
                    try:
                        arguments = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Try to parse as simple key=value
                        logger.warning(f"Could not parse arguments as JSON: {args_str}")
                        # Extract simple key=value pairs
                        arguments = self._parse_simple_args(args_str)
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            if tool_name:
                return {
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": arguments,
                    "reasoning": reasoning
                }

        # Check if it's an answer
        if response.startswith("ANSWER:"):
            answer = response.replace("ANSWER:", "").strip()
            return {
                "type": "answer",
                "answer": answer
            }

        # Default: treat as final answer
        return {
            "type": "answer",
            "answer": response
        }

    def _parse_simple_args(self, args_str: str) -> Dict:
        """Parse simple key=value argument format"""
        args = {}
        # Try to extract key=value pairs
        parts = args_str.split(",")
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip().strip('"\'')
                value = value.strip().strip('"\'')
                # Try to convert to int if it looks like a number
                try:
                    if value.isdigit():
                        value = int(value)
                except (ValueError, AttributeError):
                    # Keep as string if conversion fails
                    pass
                args[key] = value
        return args

    def _execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool and return results"""
        try:
            method = getattr(self.tools, tool_name, None)
            if not method:
                return {"error": f"Unknown tool: {tool_name}"}

            result = method(**arguments)
            return result

        except TypeError as e:
            logger.error(f"Tool execution type error: {tool_name} - {e}")
            return {"error": f"Invalid arguments for {tool_name}: {str(e)}"}
        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return {"error": f"Tool error: {str(e)}"}
