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
            "description": "Get detailed information about a specific person by their ID or genealogical identifier. Returns structured events (birth, death, marriage, etc.) AND the full narrative source text from the original document, which often contains additional context like military service, orphan status, occupations, and life details not captured in structured events.",
            "parameters": {
                "person_id": "UUID from search results 'id' field (e.g., 'abc-123-def') OR genealogical_id field (e.g., 'VI.1.n'). NEVER use display_name or person's full name."
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
                "person_id": "UUID from search results 'id' field OR 'genealogical_id' field. NEVER use person's name."
            }
        },
        {
            "name": "get_parents",
            "description": "Get parents of a specific person.",
            "parameters": {
                "person_id": "UUID from search results 'id' field OR 'genealogical_id' field. NEVER use person's name."
            }
        },
        {
            "name": "search_by_occupation",
            "description": "Search for people by occupation. Supports multilingual search - provide multiple occupation terms separated by commas (e.g., 'teacher, onderwijzer' or 'railway worker, spoorwegarbeider'). Documents are in Dutch, so include Dutch translations when searching.",
            "parameters": {
                "occupation": "Occupation term(s), comma-separated for multilingual (e.g., 'teacher, onderwijzer, meester')",
                "max_results": "Maximum results (default 10)"
            }
        },
        {
            "name": "find_relationship",
            "description": "Compute the genealogical relationship between two people by finding their most recent common ancestor. Returns relationship type (e.g., 'second cousin once removed', 'grandparent', 'sibling'), common ancestor details, and generational distances. Use this when asked 'How are X and Y related?' or 'What is the relationship between X and Y?'",
            "parameters": {
                "person_id_1": "UUID or genealogical_id of first person. NEVER use person's name.",
                "person_id_2": "UUID or genealogical_id of second person. NEVER use person's name."
            }
        },
        {
            "name": "search_source_text",
            "description": "Search genealogical source texts using semantic search. Use this for cross-cutting queries that aren't about specific people, such as: 'Who lived in Minneapolis?', 'Are there any musicians?', 'Who served in the military?', 'Tell me about people who emigrated to America'. Returns narrative text chunks with relevance scores AND automatically extracts mentioned people with their genealogical IDs, making it easy to follow up with get_person_details.",
            "parameters": {
                "query": "Search query describing what you're looking for (e.g., 'musicians', 'lived in Minneapolis', 'military service')",
                "max_results": "Maximum number of text chunks to return (default 10)"
            }
        }
    ]

    def __init__(self, model: str = "gene-chat-main", max_iterations: int = 10, timeout: int = 300):
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

    def _check_duplicate_call(self, tool_name: str, tool_args: Dict, tool_calls_made: List[Dict]) -> Optional[Dict]:
        """
        Check if this tool call is a duplicate of a previous call.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments for the tool
            tool_calls_made: List of previous tool calls

        Returns:
            Error dict if duplicate detected, None otherwise
        """
        call_signature = (tool_name, json.dumps(tool_args, sort_keys=True))

        # Find if this exact call was made before and get its result
        for call in tool_calls_made:
            call_sig = (call["tool"], json.dumps(call["arguments"], sort_keys=True))
            if call_signature == call_sig:
                logger.warning(f"Duplicate tool call blocked: {tool_name}({tool_args})")

                # Get the result from the previous call
                previous_result = call.get("result", {})
                prev_iter = call.get("iteration", "unknown")

                return {
                    "error": f"STOP! DUPLICATE CALL BLOCKED. You already called {tool_name} with these EXACT same arguments in iteration {prev_iter}. You MUST use the result from that previous call instead of calling again. Review the results from iteration {prev_iter} in this conversation. If you need different information, call a DIFFERENT tool or use DIFFERENT arguments. Repeating this exact call wastes iterations.",
                    "previous_iteration": prev_iter,
                    "hint": "Look back at the tool results from earlier in this conversation to find what you need."
                }

        return None

    def execute_streaming(
        self,
        user_query: str,
        initial_context: Optional[str] = None
    ):
        """
        Execute agentic workflow with streaming updates.

        Args:
            user_query: The user's question
            initial_context: Optional initial context from RAG retrieval

        Yields:
            {"type": "status", "message": str}
            {"type": "tool_call", "tool": str, "arguments": dict, "reasoning": str}
            {"type": "tool_result", "tool": str, "result": dict}
            {"type": "thinking", "iteration": int, "max_iterations": int}
            {"type": "answer", "answer": str, "success": bool, "tool_calls": List[Dict], "iterations": int}
            {"type": "error", "error": str, "tool_calls": List[Dict], "iterations": int}
        """
        iteration = 0
        context_parts = [initial_context] if initial_context else []
        tool_calls_made = []
        start_time = time.time()

        while iteration < self.max_iterations:
            # Check timeout
            if time.time() - start_time > self.timeout:
                yield {
                    "type": "error",
                    "error": f"Timeout after {self.timeout}s",
                    "tool_calls": tool_calls_made,
                    "iterations": iteration
                }
                return

            iteration += 1

            yield {
                "type": "thinking",
                "iteration": iteration,
                "max_iterations": self.max_iterations
            }

            # Build prompt
            prompt = self._build_agent_prompt(
                user_query=user_query,
                context="\n\n".join(context_parts),
                tool_calls_made=tool_calls_made
            )

            # Get LLM response (streaming)
            try:
                response = ""
                in_think_tag = False
                think_content = ""

                for chunk in self.ollama.generate_stream(
                    model=self.model,
                    prompt=prompt,
                    num_ctx=32768,
                    temperature=0.1
                ):
                    response += chunk

                    # Track if we're inside <think> tags for streaming
                    if '<think>' in chunk:
                        in_think_tag = True
                        yield {
                            "type": "thinking_start",
                            "iteration": iteration,
                            "max_iterations": self.max_iterations
                        }

                    if in_think_tag:
                        # Extract just the thinking content (without tags)
                        think_chunk = chunk.replace('<think>', '').replace('</think>', '')
                        if think_chunk:
                            think_content += think_chunk
                            yield {
                                "type": "thinking_token",
                                "token": think_chunk,
                                "iteration": iteration
                            }

                    if '</think>' in chunk:
                        in_think_tag = False
                        yield {
                            "type": "thinking_end",
                            "iteration": iteration
                        }

                if not response:
                    yield {
                        "type": "error",
                        "error": "LLM returned empty response",
                        "tool_calls": tool_calls_made,
                        "iterations": iteration
                    }
                    return

            except Exception as e:
                logger.exception("LLM generation failed")
                yield {
                    "type": "error",
                    "error": f"LLM error: {str(e)}",
                    "tool_calls": tool_calls_made,
                    "iterations": iteration
                }
                return

            # Parse response to check if it's a tool call or final answer
            action = self._parse_response(response)

            if action["type"] == "tool_call":
                # Notify about tool call
                yield {
                    "type": "tool_call",
                    "tool": action["tool"],
                    "arguments": action["arguments"],
                    "reasoning": action.get("reasoning", "")
                }

                # Execute tool
                tool_name = action["tool"]
                tool_args = action["arguments"]

                logger.info(f"Calling tool: {tool_name} with args: {tool_args}")

                # Check for duplicate tool calls
                duplicate_error = self._check_duplicate_call(tool_name, tool_args, tool_calls_made)
                if duplicate_error:
                    tool_result = duplicate_error
                else:
                    tool_result = self._execute_tool(tool_name, tool_args)
                tool_calls_made.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                })

                # Notify about tool result
                yield {
                    "type": "tool_result",
                    "tool": tool_name,
                    "result": tool_result
                }

                # Add result to context
                context_parts.append(
                    f"TOOL RESULT ({tool_name}):\n{json.dumps(tool_result, indent=2)}"
                )

                # Continue to next iteration

            elif action["type"] == "answer":
                # LLM has final answer
                yield {
                    "type": "answer",
                    "answer": action["answer"],
                    "tool_calls": tool_calls_made,
                    "iterations": iteration,
                    "success": True
                }
                return

            else:
                # Unknown action type, treat as final answer
                yield {
                    "type": "answer",
                    "answer": response,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration,
                    "success": True
                }
                return

        # Max iterations reached
        yield {
            "type": "error",
            "error": f"Max iterations ({self.max_iterations}) reached",
            "answer": f"Could not find complete answer after {self.max_iterations} attempts. Here's what I found:\n\n{context_parts[-1] if context_parts else 'No information gathered.'}",
            "tool_calls": tool_calls_made,
            "iterations": iteration
        }

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

                # Check for duplicate tool calls
                duplicate_error = self._check_duplicate_call(tool_name, tool_args, tool_calls_made)
                if duplicate_error:
                    tool_result = duplicate_error
                else:
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
        """
        Build prompt for agentic workflow.

        NOTE: Core instructions (tool protocol, error recovery, workflows) are in the
        model's SYSTEM prompt. This prompt provides only the current state.
        """

        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}\n  Parameters: {tool['parameters']}"
            for tool in self.AVAILABLE_TOOLS
        ])

        previous_tools = "\n".join([
            f"Iteration {call['iteration']}: {call['tool']}({call['arguments']})"
            for call in tool_calls_made
        ]) if tool_calls_made else "None"

        return f"""USER QUERY: {user_query}

AVAILABLE TOOLS:
{tools_description}

PREVIOUS TOOL CALLS ({len(tool_calls_made)}/{self.max_iterations} calls made):
{previous_tools}

CURRENT CONTEXT:
{context if context else "No context yet - search for information using tools."}

IMPORTANT REMINDERS:
- Use conversation context to resolve pronouns (e.g., "his children" = the person just discussed)
- Extract "id" or "genealogical_id" from search results before calling other tools
- If you receive a DUPLICATE CALL error referencing iteration N, review that iteration's result
- You have {self.max_iterations - len(tool_calls_made)} tool calls remaining

Respond with either TOOL_CALL or ANSWER:"""

    def _parse_response(self, response: str) -> Dict:
        """
        Parse LLM response to determine if it's a tool call or final answer.

        Returns:
            {"type": "tool_call", "tool": str, "arguments": dict, "reasoning": str}
            or
            {"type": "answer", "answer": str}
        """
        response = response.strip()

        # Check if TOOL_CALL appears anywhere in the response (not just at start)
        # LLMs sometimes add thinking/reasoning text before the tool call
        if "TOOL_CALL:" in response:
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
                        # Try fixing single quotes to double quotes for Python dict format
                        try:
                            fixed_args_str = args_str.replace("'", '"')
                            arguments = json.loads(fixed_args_str)
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

        # Check if ANSWER appears anywhere in the response
        if "ANSWER:" in response:
            # Find the line that starts with ANSWER:
            for line in response.split("\n"):
                if line.startswith("ANSWER:"):
                    answer = response[response.index("ANSWER:") + len("ANSWER:"):].strip()
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
