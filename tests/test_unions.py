from typing import Any, Literal, Optional, Union

import pytest

from valgebra import ValidationError, validator


def test_pep604_union_accepts_either_branch() -> None:
    schema = validator(int | str)
    assert schema.is_valid(1)
    assert schema.is_valid("x")
    assert not schema.is_valid(1.0)


def test_optional_admits_none() -> None:
    schema = validator(int | None)
    assert schema.is_valid(3)
    assert schema.is_valid(None)
    assert not schema.is_valid("x")


def test_typing_optional_and_union_aliases() -> None:
    # The legacy aliases are exercised on purpose; the modern X | Y form is
    # covered above.
    assert validator(Optional[int]).is_valid(None)  # noqa: UP045
    assert validator(Union[int, str]).is_valid("x")  # noqa: UP007
    assert not validator(Union[int, str]).is_valid(1.0)  # noqa: UP007


def test_literal_with_several_values_is_a_union() -> None:
    schema = validator(Literal["red", "green"])
    assert schema.is_valid("red")
    assert schema.is_valid("green")
    assert not schema.is_valid("blue")


def test_literal_with_one_value_is_a_single_literal() -> None:
    schema = validator(Literal[5])
    assert schema.is_valid(5)
    assert not schema.is_valid(True)


def test_any_admits_every_value() -> None:
    schema = validator(Any)
    assert schema.is_valid(object())
    assert schema.is_valid(None)
    assert schema.is_valid([1, "x", {}])


def test_union_failure_reports_a_union_error() -> None:
    with pytest.raises(ValidationError) as info:
        validator(int | str).validate(1.0)
    assert info.value.code == "union_error"
