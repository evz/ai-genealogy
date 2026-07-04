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
from genealogy.services.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Execute agentic workflow with tool calling"""

    # Tool definitions for the LLM
    AVAILABLE_TOOLS = [
        {
            "name": "search_person_by_name",
            "description": "Search for people by name. Returns all matches with birth, death, parents.",
            "parameters": {
                "name": "Person's name",
                "max_results": "Optional - max results (default: all)"
            }
        },
        {
            "name": "get_person_details",
            "description": "Get detailed info about a person by ID. Returns structured events AND narrative source text.",
            "parameters": {
                "person_id": "UUID or genealogical_id (e.g., 'VI.1.n'). Use ID from search results, not person's name."
            }
        },
        {
            "name": "search_by_birth_year",
            "description": "Search by name and birth year range to disambiguate.",
            "parameters": {
                "name": "Person's name",
                "birth_year_min": "Min year (optional)",
                "birth_year_max": "Max year (optional)"
            }
        },
        {
            "name": "get_children",
            "description": "Get all children of a person.",
            "parameters": {
                "person_id": "UUID or genealogical_id. Use ID, not name."
            }
        },
        {
            "name": "get_parents",
            "description": "Get parents of a person.",
            "parameters": {
                "person_id": "UUID or genealogical_id. Use ID, not name."
            }
        },
        {
            "name": "find_relationship",
            "description": "Find genealogical relationship between two people. Returns relationship type, common ancestor, generational distances.",
            "parameters": {
                "person_id_1": "UUID or genealogical_id of first person",
                "person_id_2": "UUID or genealogical_id of second person"
            }
        },
        {
            "name": "search_source_text",
            "description": "Semantic search of genealogical texts. Use for queries like 'Who lived in X?', 'Who worked as Y?'. Returns text chunks with mentioned people and their IDs.",
            "parameters": {
                "query": "Search query",
                "max_results": "Optional - max chunks (default 50)"
            }
        }
    ]

    def __init__(
        self,
        model: str = "qwen2.5:14b-instruct-q5_K_M",
        max_iterations: int = 20,
        timeout: int = 300,
        prompt_template_name: str = "agent",
        prompt_template_version: str = None,
        message_instance=None  # Message model instance for logging
    ):
        """
        Initialize agent executor.

        Args:
            model: Base LLM model to use (e.g., "qwen2.5:14b-instruct-q5_K_M", "llama3.1:8b")
            max_iterations: Maximum tool calls before stopping
            timeout: Total timeout in seconds
            prompt_template_name: Name of prompt template (e.g., "agent")
            prompt_template_version: Specific version to use (e.g., "2"), None = use active
            message_instance: Message model instance to link prompt logs to
        """
        self.model = model
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.prompt_template_name = prompt_template_name
        self.prompt_template_version = prompt_template_version
        self.message_instance = message_instance
        self.ollama = OllamaClient(timeout=timeout)
        self.tools = GenealogyTools()
        self.prompt_registry = PromptRegistry()

    def _check_duplicate_call(self, tool_name: str, tool_args: Dict, tool_calls_made: List[Dict]) -> Optional[Dict]:
        """
        Check if this tool call is a duplicate of a previous call.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments for the tool
            tool_calls_made: List of previous tool calls

        Returns:
            The previous result with a duplicate notice if detected, None otherwise
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

                # Return the previous result with a notice that this was a duplicate
                result_copy = previous_result.copy() if isinstance(previous_result, dict) else previous_result

                # Add a note at the top of the result
                duplicate_notice = f"[DUPLICATE CALL - Returning cached result from iteration {prev_iter}. You already called {tool_name} with these same arguments. Use this result instead of calling again.]"

                # If result is a dict, add the notice as a field
                if isinstance(result_copy, dict):
                    result_copy["_duplicate_notice"] = duplicate_notice
                    result_copy["_previous_iteration"] = prev_iter
                    return result_copy
                else:
                    # For non-dict results, wrap it
                    return {
                        "_duplicate_notice": duplicate_notice,
                        "_previous_iteration": prev_iter,
                        "result": result_copy
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

            # Build prompt using template system
            prompt_start_time = time.time()
            prompt_data = self._build_agent_prompt(
                user_query=user_query,
                context="\n\n".join(context_parts),
                tool_calls_made=tool_calls_made
            )

            # Get LLM response (streaming)
            try:
                response = ""
                in_think_tag = False
                think_content = ""
                llm_start_time = time.time()

                for chunk in self.ollama.generate_stream(
                    model=self.model,
                    prompt=prompt_data["user"],
                    system=prompt_data["system"],
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
                            "iteration": iteration,
                            "content": think_content  # Include full thinking content
                        }
                        think_content = ""  # Reset for next iteration

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

            # Calculate latency
            latency_ms = int((time.time() - llm_start_time) * 1000)

            # Parse response to check if it's a tool call or final answer
            action = self._parse_response(response)

            # Log this prompt execution if we have a message instance
            if self.message_instance:
                parsed_successfully = action["type"] != "error"
                parse_error = action.get("error", "") if not parsed_successfully else ""

                self.prompt_registry.log_prompt(
                    message=self.message_instance,
                    prompt_data=prompt_data,
                    model_name=self.model,
                    iteration=iteration,
                    tool_calls=tool_calls_made,
                    llm_response=response,
                    parsed_successfully=parsed_successfully,
                    parse_error=parse_error,
                    latency_ms=latency_ms
                )

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

            # Build prompt using template system
            prompt_data = self._build_agent_prompt(
                user_query=user_query,
                context="\n\n".join(context_parts),
                tool_calls_made=tool_calls_made
            )

            # Get LLM response
            try:
                llm_start_time = time.time()
                response = self.ollama.generate(
                    model=self.model,
                    prompt=prompt_data["user"],
                    system=prompt_data["system"],
                    num_ctx=32768,
                    temperature=0.1
                )
                latency_ms = int((time.time() - llm_start_time) * 1000)

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

            # Log this prompt execution if we have a message instance
            if self.message_instance:
                parsed_successfully = action["type"] != "error"
                parse_error = action.get("error", "") if not parsed_successfully else ""

                self.prompt_registry.log_prompt(
                    message=self.message_instance,
                    prompt_data=prompt_data,
                    model_name=self.model,
                    iteration=iteration,
                    tool_calls=tool_calls_made,
                    llm_response=response,
                    parsed_successfully=parsed_successfully,
                    parse_error=parse_error,
                    latency_ms=latency_ms
                )

            if action["type"] == "error":
                # Parsing error - add error message to context so LLM can correct itself
                error_msg = action["error"]
                logger.warning(f"Parse error at iteration {iteration}: {error_msg}")

                context_parts.append(
                    f"ERROR: {error_msg}\n\nPlease use the correct format:\nTOOL_CALL: <tool_name>\nARGUMENTS: {{...}}\nREASONING: <explanation>"
                )

                # Continue to next iteration to let LLM retry

            elif action["type"] == "tool_call":
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
    ) -> Dict[str, str]:
        """
        Build prompt for agentic workflow using database templates.

        Returns:
            Dict with keys: "system", "user", "template_name", "template_version", "variables"
        """

        # Get template from database
        if self.prompt_template_version:
            # Use specific version
            template = self.prompt_registry.get_template_by_version(
                self.prompt_template_name,
                self.prompt_template_version
            )
        else:
            # Use active version
            template = self.prompt_registry.get_active_template(
                self.prompt_template_name
            )

        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}\n  Parameters: {tool['parameters']}"
            for tool in self.AVAILABLE_TOOLS
        ])

        previous_calls = "\n".join([
            f"Iteration {call['iteration']}: {call['tool']}({call['arguments']})"
            for call in tool_calls_made
        ]) if tool_calls_made else "None"

        # Render prompt using template
        return self.prompt_registry.render_prompt(
            template=template,
            user_query=user_query,
            tools_description=tools_description,
            num_calls=len(tool_calls_made),
            max_iterations=self.max_iterations,
            previous_calls=previous_calls,
            context=context if context else "No context yet."
        )

    def _parse_response(self, response: str) -> Dict:
        """
        Parse LLM response to determine if it's a tool call or final answer.

        Returns:
            {"type": "tool_call", "tool": str, "arguments": dict, "reasoning": str}
            or
            {"type": "answer", "answer": str}
        """
        response = response.strip()

        # Strip <think>...</think> tags from reasoning models (e.g., deepseek-r1)
        # These contain chain-of-thought reasoning that shouldn't be in the final answer
        # Note: The original response with think tags is still saved in PromptLog for debugging
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Check if TOOL_CALL appears anywhere in the response (not just at start)
        # LLMs sometimes add thinking/reasoning text before the tool call
        if "TOOL_CALL" in response:
            # Check for correct format with colon
            if "TOOL_CALL:" not in response:
                return {
                    "type": "error",
                    "error": "Invalid tool call format. Use: 'TOOL_CALL: <tool_name>' with a colon after TOOL_CALL"
                }

            lines = response.split("\n")
            tool_name = None
            arguments = {}
            reasoning = ""

            for line in lines:
                if line.startswith("TOOL_CALL:"):
                    tool_name = line.replace("TOOL_CALL:", "").strip()
                    # Handle models that use function call syntax in tool name
                    # e.g., "get_person_details(person_id='VIII.3.d', toolbench_rapidapi_key='...')"
                    if '(' in tool_name and tool_name.endswith(')'):
                        # Parse arguments from function call syntax
                        func_call = tool_name
                        tool_name = func_call.split('(')[0].strip()

                        # Extract arguments from the function call
                        args_start = func_call.index('(') + 1
                        args_end = func_call.rindex(')')
                        args_str = func_call[args_start:args_end]

                        # Parse key=value pairs from function syntax
                        # This is a simple parser - handles key="value" or key='value'
                        import re
                        arg_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
                        parsed_args = dict(re.findall(arg_pattern, args_str))

                        # Only use these if ARGUMENTS line doesn't override later
                        # Store as fallback
                        arguments = parsed_args
                        logger.warning(f"Tool call used function syntax, parsed: {tool_name}({parsed_args})")
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
            else:
                return {
                    "type": "error",
                    "error": "TOOL_CALL found but no tool name specified. Format: 'TOOL_CALL: <tool_name>'"
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

            # Filter arguments to only include expected parameters
            # This prevents issues with models that add extra args like "toolbench_rapidapi_key"
            import inspect
            sig = inspect.signature(method)
            valid_params = set(sig.parameters.keys())

            # Filter out unexpected arguments
            filtered_args = {k: v for k, v in arguments.items() if k in valid_params}

            # Log if we filtered anything out
            if len(filtered_args) != len(arguments):
                filtered_out = set(arguments.keys()) - valid_params
                logger.warning(f"Filtered unexpected arguments from {tool_name}: {filtered_out}")

            result = method(**filtered_args)
            return result

        except TypeError as e:
            logger.error(f"Tool execution type error: {tool_name} - {e}")
            return {"error": f"Invalid arguments for {tool_name}: {str(e)}"}
        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return {"error": f"Tool error: {str(e)}"}
