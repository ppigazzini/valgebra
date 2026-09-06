import itertools
import sys
from typing import Annotated, Literal, NoReturn

import annotated_types as at
import pytest

from valgebra import (
    Regex,
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


def test_a_predicate_does_not_fold_against_its_own_complement() -> None:
    """The complement laws are laws about sets, and a predicate is not one.

    `A & ~A = nothing` holds because a value is in `A` or it is not, once. A
    predicate is user code and the two occurrences of `A` are two calls, so one
    that does not answer from the value alone answers them differently. The
    witness below is admitted by the meet, and the dual rejects a value the top
    would have to admit -- so folding either would claim emptiness of a schema
    that admits values.

    Pinned rather than left to the reader of the law, because "`A & ~A` is
    empty" is exactly what somebody reading `docs/04-algebra.md` would add.
    `docs/dev/01-schema-ir.md` carries the refusal.
    """
    flip = itertools.count()

    def alternating(_: object) -> bool:
        return next(flip) % 2 == 0

    predicate = Validator(Annotated[int, at.Predicate(alternating)])
    assert predicate == predicate  # noqa: PLR0124 - the identity is the subject

    meet = intersection(predicate, complement(predicate))
    assert not meet.is_empty(), "a predicate is not a set, so the law does not apply"
    # The witness: some call lands inside a meet the fold would call empty.
    assert any(meet.is_valid(1) for _ in range(8))

    join = union(predicate, complement(predicate))
    assert not join.is_equivalent(anything)
    # And the dual witness: some call falls outside a join the fold would call
    # the top.
    assert not all(join.is_valid(1) for _ in range(8))


def test_a_pattern_does_fold_against_its_own_complement() -> None:
    """The contrast that makes the refusal about predicates, not about markers.

    A pattern is a function of the string, so the two occurrences agree and the
    law holds.
    """
    pattern = Validator(Annotated[str, Regex("a+")])
    assert intersection(pattern, complement(pattern)).is_empty()
    assert union(pattern, complement(pattern)).is_equivalent(anything)
