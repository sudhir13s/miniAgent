"""Unit tests for describing validation errors to a model."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from miniagent.core.validation import describe_validation_error


class _Point(BaseModel):
    x: int
    y: int


def _error(raw: str) -> ValidationError:
    with pytest.raises(ValidationError) as excinfo:
        _Point.model_validate_json(raw)
    return excinfo.value


class TestDescribeValidationError:
    def test_should_name_each_failing_field(self) -> None:
        description = describe_validation_error(_error('{"x": "a"}'), root_label="reply")
        assert description == (
            "x: Input should be a valid integer, unable to parse string as an integer; "
            "y: Field required"
        )

    def test_should_use_the_root_label_when_the_whole_value_is_wrong(self) -> None:
        description = describe_validation_error(_error("not json"), root_label="reply")
        assert description.startswith("reply: Invalid JSON")
