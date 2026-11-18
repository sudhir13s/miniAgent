"""CalculatorTool — safely evaluates arithmetic expressions."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool.base import BaseTool, ToolSchema

_BINARY_OPERATIONS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPERATIONS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.expr) -> float:
    """Recursively evaluate an AST node using only safe arithmetic ops."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATIONS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        # float() rejects the complex result of e.g. (-8) ** 0.5.
        return float(_BINARY_OPERATIONS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATIONS:
        return float(_UNARY_OPERATIONS[type(node.op)](_safe_eval(node.operand)))
    raise ToolInvocationError(
        f"Unsafe expression: '{ast.dump(node)}'. "
        "Only basic arithmetic (+, -, *, /, **, %, //) is allowed."
    )


class CalculatorTool(BaseTool):
    """Evaluates arithmetic expressions safely without using eval().

    Parses the expression into an AST and walks only safe operators,
    preventing code injection while supporting +, -, *, /, **, %, //.

    Example::

        tool = CalculatorTool()
        result = await tool.invoke(expression="(2 + 3) * 4 / 2")
        # "10.0"
    """

    def __init__(self) -> None:
        """Initialise the calculator tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="calculator",
                description=(
                    "Evaluate an arithmetic expression and return the result. "
                    "Supports +, -, *, /, ** (power), % (modulo), // (floor division). "
                    "Example: '(2 + 3) * 4' returns '20.0'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": (
                                "The arithmetic expression to evaluate, e.g. '(2 + 3) * 4'."
                            ),
                        },
                    },
                    "required": ["expression"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Evaluate the ``expression`` argument and return the result as a string."""
        expression: str = kwargs.get("expression", "")
        if not expression.strip():
            raise ToolInvocationError("Expression cannot be empty.")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = _safe_eval(tree.body)
            return str(result)
        except ToolInvocationError:
            raise
        except ZeroDivisionError as exc:
            raise ToolInvocationError("Division by zero.") from exc
        except Exception as exc:
            raise ToolInvocationError(
                f"Failed to evaluate expression '{expression}': {exc}", cause=exc
            ) from exc
