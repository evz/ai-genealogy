"""
Centralized prompt template management and logging system.

This module provides:
1. Version-controlled prompt templates (database-backed)
2. Template rendering with variables
3. Prompt logging for auditability
4. Effectiveness metrics and A/B testing support
"""

import logging
from typing import Dict, Optional, Any

from django.db.models import Avg

from genealogy.models import PromptTemplate as DBTemplate, PromptLog

logger = logging.getLogger(__name__)


class PromptRegistry:
    """
    Central registry for all prompt templates.

    Loads templates from database and manages prompt logging.
    """

    def get_active_template(self, name: str) -> DBTemplate:
        """
        Get the currently active template for a given name.

        Args:
            name: Template name (e.g., "agent")

        Returns:
            PromptTemplate model instance from database

        Raises:
            ValueError: If no active template found
        """
        template = DBTemplate.objects.filter(
            name=name,
            is_active=True,
            is_archived=False
        ).first()

        if not template:
            raise ValueError(
                f"No active template found for '{name}'. "
                f"Create one in Django admin at /admin/genealogy/prompttemplate/"
            )

        logger.info(f"Using active template: {template.name} v{template.version}")
        return template

    def get_template_by_version(self, name: str, version: str) -> DBTemplate:
        """
        Get a specific template version (for testing/comparison).

        Args:
            name: Template name (e.g., "agent")
            version: Version number (e.g., "1", "2")

        Returns:
            PromptTemplate model instance from database

        Raises:
            PromptTemplate.DoesNotExist: If template not found
        """
        template = DBTemplate.objects.get(name=name, version=version)
        logger.info(f"Using specific template: {template.name} v{template.version}")
        return template

    def render_prompt(
        self,
        template: DBTemplate,
        **variables
    ) -> Dict[str, Any]:
        """
        Render a prompt template with the given variables.

        Args:
            template: PromptTemplate model instance
            **variables: Template variables

        Returns:
            {
                "system": str,
                "user": str,
                "template_name": str,
                "template_version": str,
                "variables": dict
            }
        """
        return template.render(**variables)

    def log_prompt(
        self,
        message,  # Message model instance
        prompt_data: Dict[str, Any],
        model_name: str,
        iteration: int,
        tool_calls: list,
        llm_response: str,
        parsed_successfully: bool,
        parse_error: str = "",
        latency_ms: Optional[int] = None
    ):
        """
        Log a prompt execution to the database.

        Args:
            message: Message model instance
            prompt_data: Output from render_prompt()
            model_name: LLM model used
            iteration: Iteration number in agent loop
            tool_calls: List of tool calls made
            llm_response: Raw LLM response
            parsed_successfully: Whether response was parsed successfully
            parse_error: Error message if parsing failed
            latency_ms: Response time in milliseconds
        """
        # Combine system and user for full prompt
        full_prompt = f"SYSTEM:\n{prompt_data['system']}\n\nUSER:\n{prompt_data['user']}"

        # Estimate token counts (rough approximation: 1 token ≈ 4 characters)
        token_count_prompt = len(full_prompt) // 4
        token_count_response = len(llm_response) // 4

        PromptLog.objects.create(
            message=message,
            prompt_template_name=prompt_data["template_name"],
            prompt_version=prompt_data["template_version"],
            system_prompt=prompt_data["system"],
            user_prompt=prompt_data["user"],
            full_prompt=full_prompt,
            prompt_variables=prompt_data["variables"],
            model_name=model_name,
            iteration=iteration,
            tool_calls=tool_calls,
            llm_response=llm_response,
            parsed_successfully=parsed_successfully,
            parse_error=parse_error,
            latency_ms=latency_ms,
            token_count_prompt=token_count_prompt,
            token_count_response=token_count_response
        )

        logger.info(
            f"Logged prompt: {prompt_data['template_name']} v{prompt_data['template_version']} "
            f"(model={model_name}, iter={iteration}, tokens={token_count_prompt}→{token_count_response})"
        )

    def get_effectiveness_metrics(self, template_name: str, template_version: str) -> Dict[str, Any]:
        """
        Get effectiveness metrics for a specific prompt template version.

        Returns:
            {
                "total_uses": int,
                "success_rate": float,  # % parsed successfully
                "avg_latency_ms": float,
                "avg_prompt_tokens": float,
                "avg_response_tokens": float,
                "avg_iterations": float,  # For agent prompts
            }
        """
        logs = PromptLog.objects.filter(
            prompt_template_name=template_name,
            prompt_version=template_version
        )

        total_uses = logs.count()
        if total_uses == 0:
            return {"total_uses": 0}

        success_count = logs.filter(parsed_successfully=True).count()

        aggregates = logs.aggregate(
            avg_latency=Avg('latency_ms'),
            avg_prompt_tokens=Avg('token_count_prompt'),
            avg_response_tokens=Avg('token_count_response'),
            avg_iterations=Avg('iteration')
        )

        return {
            "total_uses": total_uses,
            "success_rate": (success_count / total_uses) * 100,
            "avg_latency_ms": round(aggregates['avg_latency'] or 0, 2),
            "avg_prompt_tokens": round(aggregates['avg_prompt_tokens'] or 0, 2),
            "avg_response_tokens": round(aggregates['avg_response_tokens'] or 0, 2),
            "avg_iterations": round(aggregates['avg_iterations'] or 0, 2)
        }
