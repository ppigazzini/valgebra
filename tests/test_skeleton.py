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


def test_cast_returns_the_validated_object() -> None:
    obj = [1, 2, 3]
    assert Validator([int]).ensure(obj) is obj
