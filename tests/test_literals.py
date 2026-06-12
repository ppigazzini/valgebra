import pytest

from valgebra import ValidationError, validator


def test_literal_accepts_the_exact_value() -> None:
    assert validator(5).is_valid(5)
    assert validator("red").is_valid("red")
    assert validator(b"x").is_valid(b"x")


def test_literal_rejects_a_different_value() -> None:
    assert not validator(5).is_valid(6)
    assert not validator("red").is_valid("green")


def test_literal_is_a_typed_singleton() -> None:
    # Python's == conflates 1, True, and 1.0; a literal keeps them distinct by
    # also requiring the same type.
    assert validator(1).is_valid(1)
    assert not validator(1).is_valid(True)
    assert not validator(1).is_valid(1.0)
    assert validator(True).is_valid(True)
    assert not validator(True).is_valid(1)
    assert validator(1.0).is_valid(1.0)
    assert not validator(1.0).is_valid(1)


def test_literal_failure_reports_its_code() -> None:
    with pytest.raises(ValidationError) as info:
        validator(5).validate(6)
    assert info.value.code == "literal_value"
