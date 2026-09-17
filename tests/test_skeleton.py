import pytest

from valgebra import ValidationError, Validator


def test_validator_returns_a_compiled_validator() -> None:
    assert isinstance(Validator(int), Validator)


def test_int_schema_accepts_an_int() -> None:
    assert Validator(int).is_valid(3)


def test_int_schema_rejects_a_str() -> None:
    assert not Validator(int).is_valid("x")


def test_validate_raises_validation_error_on_mismatch() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int).validate("x")
    assert info.value.code == "int_type"
    assert info.value.path == ()


def test_validate_answers_with_nothing_when_the_value_is_a_member() -> None:
    """The success side of the raising entry point, which its docstring states.

    `validate` is driven everywhere for what it raises and nowhere for what it
    answers, so a method documented to return `None` could start returning a
    verdict -- a value a caller would then be able to read as one, and read
    wrongly the day it went back to `None`.
    """
    assert Validator(int).validate(3) is None
    assert Validator(int).validate(3, fail_fast=True) is None


def test_cast_returns_the_validated_object() -> None:
    obj = [1, 2, 3]
    assert Validator([int]).ensure(obj) is obj
