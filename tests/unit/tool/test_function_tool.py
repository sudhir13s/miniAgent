"""Tests for the ``@tool`` decorator and ``FunctionTool``."""

from __future__ import annotations

import threading
from typing import Literal

import pytest
from pydantic import BaseModel

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool import FunctionTool, tool
from miniagent.tool.base import DEFAULT_TOOL_TIMEOUT_SECONDS


class Address(BaseModel):
    street: str
    city: str


@tool
async def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Paris".
        unit: Temperature unit for the answer.
    """
    return f"{city}: 21 {unit}"


class TestSchemaFromSignature:
    def test_decorator_returns_function_tool_named_after_function(self) -> None:
        assert isinstance(get_weather, FunctionTool)
        assert get_weather.name == "get_weather"

    def test_description_is_docstring_summary(self) -> None:
        assert get_weather.schema.description == "Get the current weather for a city."

    def test_parameters_follow_type_hints_and_defaults(self) -> None:
        parameters = get_weather.schema.parameters
        assert parameters["type"] == "object"
        assert parameters["required"] == ["city"]
        assert parameters["properties"]["city"]["type"] == "string"
        assert parameters["properties"]["unit"]["enum"] == ["celsius", "fahrenheit"]
        assert parameters["properties"]["unit"]["default"] == "celsius"

    def test_parameter_descriptions_come_from_args_section(self) -> None:
        properties = get_weather.schema.parameters["properties"]
        assert properties["city"]["description"] == 'City name, e.g. "Paris".'
        assert properties["unit"]["description"] == "Temperature unit for the answer."

    def test_unknown_arguments_are_forbidden_in_schema(self) -> None:
        assert get_weather.schema.parameters["additionalProperties"] is False

    def test_multiline_parameter_description_is_joined(self) -> None:
        @tool
        def search(query: str) -> str:
            """Search the docs.

            Args:
                query: Words to look for. Quote a phrase
                    to match it exactly.
            """
            return query

        description = search.schema.parameters["properties"]["query"]["description"]
        assert description == "Words to look for. Quote a phrase to match it exactly."

    def test_nested_model_parameter(self) -> None:
        @tool
        def ship(address: Address) -> str:
            """Ship a parcel."""
            return address.city

        assert ship.schema.parameters["required"] == ["address"]


class TestOptions:
    def test_overrides_name_description_and_timeout(self) -> None:
        @tool(name="lookup", description="Look something up.", timeout_seconds=5)
        def find(key: str) -> str:
            """Ignored docstring."""
            return key

        assert find.name == "lookup"
        assert find.schema.description == "Look something up."
        assert find.timeout_seconds == 5

    def test_default_timeout(self) -> None:
        assert get_weather.timeout_seconds == DEFAULT_TOOL_TIMEOUT_SECONDS


class TestInvoke:
    async def test_async_function_is_awaited(self) -> None:
        assert await get_weather.invoke(city="Paris") == "Paris: 21 celsius"

    async def test_sync_function_runs_off_the_event_loop_thread(self) -> None:
        @tool
        def current_thread() -> str:
            """Name the thread running this tool."""
            return threading.current_thread().name

        assert await current_thread.invoke() != threading.current_thread().name

    async def test_non_string_result_is_returned_as_json(self) -> None:
        @tool
        def stats(values: list[int]) -> dict[str, int]:
            """Summarise numbers."""
            return {"count": len(values), "total": sum(values)}

        assert await stats.invoke(values=[1, 2, 3]) == '{"count": 3, "total": 6}'

    async def test_arguments_are_coerced_to_declared_types(self) -> None:
        @tool
        def double(number: int) -> str:
            """Double a number."""
            return str(number * 2)

        assert await double.invoke(number="21") == "42"

    async def test_nested_model_argument_is_built_from_dict(self) -> None:
        @tool
        def ship(address: Address) -> str:
            """Ship a parcel."""
            return f"{type(address).__name__} in {address.city}"

        assert await ship.invoke(address={"street": "1 Main", "city": "Oslo"}) == "Address in Oslo"

    async def test_missing_argument_raises_tool_error_naming_it(self) -> None:
        with pytest.raises(ToolInvocationError, match="city"):
            await get_weather.invoke()

    async def test_invalid_argument_raises_tool_error_naming_it(self) -> None:
        with pytest.raises(ToolInvocationError, match="unit"):
            await get_weather.invoke(city="Paris", unit="kelvin")

    async def test_unknown_argument_raises_tool_error_naming_it(self) -> None:
        with pytest.raises(ToolInvocationError, match="bogus"):
            await get_weather.invoke(city="Paris", bogus=1)

    async def test_decorated_function_is_still_callable_directly(self) -> None:
        assert await get_weather("Rome") == "Rome: 21 celsius"


class TestDefinitionErrors:
    def test_missing_type_hint_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="type hint"):

            @tool
            def untyped(value) -> str:  # type: ignore[no-untyped-def]
                """Has no hint."""
                return str(value)

    def test_variadic_parameters_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="args"):

            @tool
            def variadic(*args: str) -> str:
                """Takes anything."""
                return ""

    def test_missing_description_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="description"):

            @tool
            def undocumented(value: str) -> str:
                return value

    def test_invalid_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):

            @tool(name="has space")
            def spaced(value: str) -> str:
                """Bad name."""
                return value

    def test_overlong_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="64"):

            @tool(name="x" * 65)
            def long_name(value: str) -> str:
                """Bad name."""
                return value
