import copy
from typing import Any, Literal, Optional, Union

import pytest

from valgebra import ValidationError, Validator, union


def test_pep604_union_accepts_either_branch() -> None:
    schema = Validator(int | str)
    assert schema.is_valid(1)
    assert schema.is_valid("x")
    assert not schema.is_valid(1.0)


def test_optional_admits_none() -> None:
    schema = Validator(int | None)
    assert schema.is_valid(3)
    assert schema.is_valid(None)
    assert not schema.is_valid("x")


def test_typing_optional_and_union_aliases() -> None:
    # The legacy aliases are exercised on purpose; the modern X | Y form is
    # covered above.
    assert Validator(Optional[int]).is_valid(None)  # noqa: UP045
    assert Validator(Union[int, str]).is_valid("x")  # noqa: UP007
    assert not Validator(Union[int, str]).is_valid(1.0)  # noqa: UP007


def test_literal_with_several_values_is_a_union() -> None:
    schema = Validator(Literal["red", "green"])
    assert schema.is_valid("red")
    assert schema.is_valid("green")
    assert not schema.is_valid("blue")


def test_literal_with_one_value_is_a_single_literal() -> None:
    schema = Validator(Literal[5])
    assert schema.is_valid(5)
    assert not schema.is_valid(True)


def test_any_admits_every_value() -> None:
    schema = Validator(Any)
    assert schema.is_valid(object())
    assert schema.is_valid(None)
    assert schema.is_valid([1, "x", {}])


def test_union_failure_reports_a_union_error() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int | str).validate(1.0)
    assert info.value.code == "union_error"


def test_int_literal_union_keeps_same_type_semantics() -> None:
    # A multi-value literal compiles to a union of literals, which the fast path
    # decides by value; it must still honor the same-type rule, so a bool or
    # float never matches an int literal.
    schema = Validator(Literal[1, 2, 3])
    assert schema.is_valid(1)
    assert schema.is_valid(3)
    assert not schema.is_valid(4)
    assert not schema.is_valid(True)  # bool, not int
    assert not schema.is_valid(1.0)  # float, not int


def test_str_literal_union_decides_by_value() -> None:
    schema = Validator(Literal["red", "green", "blue"])
    assert schema.is_valid("green")
    assert not schema.is_valid("teal")
    assert not schema.is_valid(0)


def test_mixed_type_literal_union_matches_each_type() -> None:
    # int, str, and bool members together: the fast path decides int and str by
    # value while bool falls back to the linear scan, and all must agree.
    schema = Validator(Literal[1, "a", True])
    for member in (1, "a", True):
        assert schema.is_valid(member)
    assert not schema.is_valid(2)
    assert not schema.is_valid("b")
    assert not schema.is_valid(1.0)  # a float matches no member


def test_big_integer_literal_union_matches_outside_machine_range() -> None:
    # union(...) builds a union of literals from runtime values, so a literal
    # larger than a machine integer can be exercised; it is decided by the linear
    # fallback while the small member uses the fast path.
    big = 10**30
    schema = union(big, 5)
    assert schema.is_valid(big)
    assert schema.is_valid(5)
    assert not schema.is_valid(big + 1)


def test_literal_union_decision_survives_copy() -> None:
    # A copied validator rebuilds its own precompute, so the value-keyed fast
    # path stays correct rather than reusing the original's node addresses.
    schema = copy.deepcopy(Validator(Literal[1, 2, 3]))
    assert schema.is_valid(2)
    assert not schema.is_valid(9)
    assert not schema.is_valid(True)


def test_literal_union_validate_agrees_with_is_valid() -> None:
    # The explain path bypasses the fast set lookup; it must reach the same
    # verdict the membership fast path does.
    schema = Validator(Literal[1, 2, 3])
    schema.validate(2)
    with pytest.raises(ValidationError):
        schema.validate(4)
    with pytest.raises(ValidationError):
        schema.validate(True)
