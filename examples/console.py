"""Console output shared by the example scripts."""

from __future__ import annotations

from miniagent import MessageRole, State

DIVIDER = "=" * 60


def print_conversation(state: State, *, max_tool_output_chars: int | None = None) -> None:
    """Print every user, AI, and tool message in ``state``, labelled by role.

    Args:
        state: The final workflow state to print.
        max_tool_output_chars: Truncate tool output beyond this many characters;
            ``None`` prints it in full.
    """
    for message in state.messages:
        if message.role == MessageRole.USER:
            print(f"[USER] {message.content}\n")
        elif message.role == MessageRole.AI:
            print(f"[AI] {message.content}\n")
        elif message.role == MessageRole.TOOL:
            output = _truncate(message.content, max_tool_output_chars)
            print(f"[TOOL:{message.tool_name}] {output}\n")


def _truncate(text: str, max_chars: int | None) -> str:
    """Shorten ``text`` to ``max_chars`` characters, marking the cut with an ellipsis."""
    if max_chars is None or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."
