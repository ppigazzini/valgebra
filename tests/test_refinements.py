from typing import Annotated

import annotated_types as at
import pytest

from valgebra import ValidationError, validator


def test_comparison_bounds() -> None:
    adult = validator(Annotated[int, at.Ge(18), at.Le(150)])
    assert adult.is_valid(18)
    assert adult.is_valid(150)
    assert not adult.is_valid(17)
    assert not adult.is_valid(151)


def test_strict_comparison_bounds() -> None:
    schema = validator(Annotated[int, at.Gt(0), at.Lt(10)])
    assert schema.is_valid(5)
    assert not schema.is_valid(0)
    assert not schema.is_valid(10)


def test_length_bounds() -> None:
    name = validator(Annotated[str, at.MinLen(1), at.MaxLen(3)])
    assert name.is_valid("ab")
    assert not name.is_valid("")
    assert not name.is_valid("abcd")


def test_length_bounds_on_a_list() -> None:
    schema = validator(Annotated[list[int], at.MinLen(2)])
    assert schema.is_valid([1, 2])
    assert not schema.is_valid([1])


def test_predicate_marker() -> None:
    even = validator(Annotated[int, at.Predicate(lambda x: x % 2 == 0)])
    assert even.is_valid(4)
    assert not even.is_valid(3)


def test_bare_callable_metadata_is_a_predicate() -> None:
    positive = validator(Annotated[int, lambda x: x > 0])
    assert positive.is_valid(1)
    assert not positive.is_valid(-1)


def test_base_failure_takes_precedence_over_constraints() -> None:
    schema = validator(Annotated[int, at.Ge(0)])
    with pytest.raises(ValidationError) as info:
        schema.validate("x")
    assert info.value.code == "int_type"


def test_constraint_failure_reports_its_code() -> None:
    schema = validator(Annotated[int, at.Ge(18)])
    with pytest.raises(ValidationError) as info:
        schema.validate(5)
    assert info.value.code == "greater_than_equal"


def test_raising_predicate_is_surfaced_as_predicate_error() -> None:
    def boom(_: object) -> bool:
        raise RuntimeError

    schema = validator(Annotated[int, at.Predicate(boom)])
    with pytest.raises(ValidationError) as info:
        schema.validate(1)
    assert info.value.code == "predicate_error"


def test_unrecognized_metadata_is_ignored() -> None:
    schema = validator(Annotated[int, "documentation"])
    assert schema.is_valid(3)
    assert not schema.is_valid("x")
