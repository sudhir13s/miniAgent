"""Per-agent LLM configuration.

Model IDs are provider-qualified ``<vendor>/<model>`` strings. The configured
``BaseLLMProvider`` interprets them; the default ``LiteLLMProvider`` uses the
vendor prefix to select which company's model is called:
    ``"openai/gpt-4o"``
    ``"gemini/gemini-2.0-flash"``
    ``"anthropic/claude-sonnet-4-6"``
    ``"groq/llama-3.1-70b-versatile"``
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for the LLM backing a specific agent.

    Attributes:
        id: Provider-qualified model ID, e.g. ``"openai/gpt-4o"``.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter. Only supported by some providers.
        extra_params: Additional provider-specific parameters passed
            through to the LLM provider unchanged.
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Provider-qualified model ID '<vendor>/<model>', e.g. 'openai/gpt-4o'.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=1_000_000,
        description="Maximum number of tokens to generate.",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k sampling parameter.",
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters passed to the LLM provider.",
    )

    model_config = {"frozen": True, "extra": "forbid"}
