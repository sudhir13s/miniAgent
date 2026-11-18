"""Model access: the ``BaseLLMProvider`` contract and the default LiteLLM provider.

Quick import::

    from miniagent.llm import BaseLLMProvider, LiteLLMProvider, get_default_provider
"""

from miniagent.llm.base import BaseLLMProvider
from miniagent.llm.litellm_provider import LiteLLMProvider
from miniagent.llm.registry import (
    get_default_provider,
    reset_default_provider,
    set_default_provider,
)
from miniagent.llm.streaming import CompletionDone, StreamEvent, TextDelta

__all__ = [
    "BaseLLMProvider",
    "CompletionDone",
    "LiteLLMProvider",
    "StreamEvent",
    "TextDelta",
    "get_default_provider",
    "reset_default_provider",
    "set_default_provider",
]
