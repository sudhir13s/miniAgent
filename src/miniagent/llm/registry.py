"""The process-wide default LLM provider.

``build_agent`` uses this provider unless it is given one, so every agent
shares a single ``LiteLLMProvider`` by default::

    from miniagent.llm import get_default_provider, set_default_provider

    provider = get_default_provider()            # LiteLLMProvider, created once
    set_default_provider(MyGatewayProvider())    # use another gateway everywhere
"""

from __future__ import annotations

from miniagent.llm.base import BaseLLMProvider
from miniagent.llm.litellm_provider import LiteLLMProvider

_default_provider: BaseLLMProvider | None = None


def get_default_provider() -> BaseLLMProvider:
    """Return the default provider, creating a ``LiteLLMProvider`` on first use.

    Returns:
        The shared provider instance.
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = LiteLLMProvider()
    return _default_provider


def set_default_provider(provider: BaseLLMProvider) -> None:
    """Replace the default provider with a custom gateway.

    Args:
        provider: The provider every later ``build_agent`` call uses.

    Raises:
        TypeError: If ``provider`` is not a ``BaseLLMProvider`` instance.
    """
    if not isinstance(provider, BaseLLMProvider):
        raise TypeError(f"provider must be a BaseLLMProvider, got {type(provider).__name__}")
    global _default_provider
    _default_provider = provider


def reset_default_provider() -> None:
    """Forget the default provider; the next call creates a new ``LiteLLMProvider``."""
    global _default_provider
    _default_provider = None
