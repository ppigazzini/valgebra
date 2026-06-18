import pytest

from valgebra import ValidationError, Validator


def test_literal_accepts_the_exact_value() -> None:
    assert Validator(5).is_valid(5)
    assert Validator("red").is_valid("red")
    assert Validator(b"x").is_valid(b"x")


def test_literal_rejects_a_different_value() -> None:
    assert not Validator(5).is_valid(6)
    assert not Validator("red").is_valid("green")


def test_literal_is_a_typed_singleton() -> None:
    # Python's == conflates 1, True, and 1.0; a literal keeps them distinct by
    # also requiring the same type.
    assert Validator(1).is_valid(1)
    assert not Validator(1).is_valid(True)
    assert not Validator(1).is_valid(1.0)
    assert Validator(True).is_valid(True)
    assert not Validator(True).is_valid(1)
    assert Validator(1.0).is_valid(1.0)
    assert not Validator(1.0).is_valid(1)


def test_literal_failure_reports_its_code() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(5).validate(6)
    assert info.value.code == "literal_error"
