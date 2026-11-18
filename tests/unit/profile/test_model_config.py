"""Tests for ModelConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miniagent.profile.model_config import ModelConfig


class TestModelConfig:
    """Tests for the ModelConfig model."""

    def test_create_model_config(self) -> None:
        """A valid model config should be created."""
        config = ModelConfig(id="gemini/gemini-1.5-pro")
        assert config.id == "gemini/gemini-1.5-pro"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_temperature_bounds(self) -> None:
        """Temperature must be between 0.0 and 2.0."""
        with pytest.raises(ValidationError):
            ModelConfig(
                id="openai/gpt-4",
                temperature=2.5,
            )
        with pytest.raises(ValidationError):
            ModelConfig(
                id="openai/gpt-4",
                temperature=-0.1,
            )

    def test_top_p_bounds(self) -> None:
        """top_p must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            ModelConfig(
                id="openai/gpt-4",
                top_p=1.5,
            )

    def test_max_tokens_bounds(self) -> None:
        """max_tokens must be at least 1."""
        with pytest.raises(ValidationError):
            ModelConfig(
                id="openai/gpt-4",
                max_tokens=0,
            )

    def test_empty_model_rejected(self) -> None:
        """An empty model string should be rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(id="")

    def test_missing_model_rejected(self) -> None:
        """The model ID is required."""
        with pytest.raises(ValidationError, match=r"(?m)^id\n\s+Field required"):
            ModelConfig()  # type: ignore[call-arg]

    def test_litellm_model_key_rejected(self) -> None:
        """The pre-rename `litellm_model` key fails loudly instead of being ignored."""
        with pytest.raises(ValidationError, match="litellm_model"):
            ModelConfig(litellm_model="openai/gpt-4")  # type: ignore[call-arg]

    def test_legacy_provider_fields_rejected(self) -> None:
        """The removed provider + model_name form fails loudly, not silently."""
        with pytest.raises(ValidationError, match="provider"):
            ModelConfig(id="openai/gpt-4", provider="openai")  # type: ignore[call-arg]

    def test_extra_params(self) -> None:
        """Extra params should be accepted."""
        config = ModelConfig(
            id="anthropic/claude-3",
            extra_params={"stop_sequences": ["END"]},
        )
        assert config.extra_params == {"stop_sequences": ["END"]}

    def test_frozen(self) -> None:
        """ModelConfig should be immutable."""
        config = ModelConfig(id="gemini/gemini-1.5-pro")
        with pytest.raises(ValidationError):
            config.temperature = 0.5  # type: ignore[misc]
