import sys
from typing import Literal, NoReturn

import pytest

from valgebra import (
    ValidationError,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    union,
)


def test_union_admits_any_branch() -> None:
    schema = union(int, str)
    assert schema.is_valid(1)
    assert schema.is_valid("x")
    assert not schema.is_valid(1.0)


def test_intersect_requires_every_member() -> None:
    schema = intersection(int, complement(bool))
    assert schema.is_valid(5)
    assert not schema.is_valid(True)  # an int, but also a bool
    assert not schema.is_valid("x")


def test_complement_inverts_membership() -> None:
    schema = complement(int)
    assert schema.is_valid("x")
    assert not schema.is_valid(5)


def test_lattice_bounds() -> None:
    assert anything.is_valid(object())
    assert anything.is_valid(None)
    assert not nothing.is_valid(5)
    # the complement of bottom is the top
    assert complement(nothing).is_valid(5)
    assert not complement(anything).is_valid(5)


def test_typing_native_bound_spellings() -> None:
    # `object` is the top spelling; `NoReturn` (3.8+) is a bottom spelling.
    assert Validator(object).is_equivalent(anything)
    assert Validator(NoReturn).is_equivalent(nothing)
    assert Validator(NoReturn).is_empty()
    assert not Validator(NoReturn).is_valid(5)
    assert not Validator(NoReturn).is_valid(None)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="typing.Never was added in 3.11")
def test_never_is_the_bottom() -> None:
    from typing import Never  # noqa: PLC0415 -- 3.11+ only, so import is gated

    assert Validator(Never).is_equivalent(nothing)
    assert Validator(Never).is_empty()
    assert not Validator(Never).is_valid(5)


def test_combinators_compose_over_compiled_validators() -> None:
    inner = Validator(list[int])
    schema = union(inner, str)
    assert schema.is_valid([1, 2, 3])
    assert schema.is_valid("x")
    assert not schema.is_valid(1.0)


def test_composition_preserves_pooled_literals() -> None:
    schema = union(Literal["a"], int)
    assert schema.is_valid("a")
    assert schema.is_valid(7)
    assert not schema.is_valid("b")


def test_complement_failure_reports_unexpected_match() -> None:
    with pytest.raises(ValidationError) as info:
        complement(int).validate(5)
    assert info.value.code == "unexpected_match"


def test_intersect_with_an_annotation() -> None:
    schema = intersection(int, complement(Literal[0]))
    assert schema.is_valid(1)
    assert not schema.is_valid(0)
