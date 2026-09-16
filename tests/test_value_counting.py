"""How many values an element denotes, and what a set of them can hold.

A set holds each member once, so `Annotated[set[T], MinLen(n)]` asks `T` for `n`
values that differ. Answering it means counting the values a schema denotes, and
the count is read in one direction each: an upper bound proves the refinement
empty, a lower bound proves it inhabited, and a schema this cannot count proves
neither.

A `Literal` looks like the easy row and is the one to get wrong. `Literal[c]`
denotes `{x | type(x) is type(c) and x == c}`, which is one value only where
`c`'s type's equality is the one the oracle trusts -- a builtin scalar, whose
equality is Python's, or a type comparing by identity. A constant whose `__eq__`
answers `True` for everything denotes *more* than one value, and a constant that
does not equal itself denotes none:

* counting the first as one proves `Annotated[set[Literal[c]], MinLen(2)]`
  empty, and a two-member set of that type validates against it;
* counting the second as one proves such a set *inhabited*, and reporting
  `"not_subset"` against an unrelated kind asserts a value that does not exist.

Both are answers no value supports, in the two directions the contract separates.
The rows below carry the value that settles each, and the rows beside them are
the constants the oracle does trust, where the count is exact and the relation is
decided.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal

import annotated_types as at

from valgebra import Validator


class Colour(Enum):
    """An ordinary enumeration: its members compare by identity."""

    RED = 1
    GREEN = 2


class AlwaysEqual(Enum):
    """An enumeration whose members equal everything.

    `Literal[AlwaysEqual.A]` denotes both members, so a set of two of them is a
    set of two values of that schema.
    """

    A = 1
    B = 2

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return id(self)


class Unhashable(Enum):  # noqa: PLW1641
    """An enumeration a set cannot hold: defining `__eq__` clears `__hash__`.

    The missing `__hash__` is the point: Python clears it for a class defining
    `__eq__`, so no set carries a member of this type and the schema below
    denotes none.
    """

    A = 1

    def __eq__(self, other: object) -> bool:
        return other is self


def test_a_constant_whose_equality_is_not_pythons_is_not_counted_as_one() -> None:
    """A literal denoting two values does not empty a set that asks for two."""
    schema = Annotated[set[Literal[AlwaysEqual.A]], at.MinLen(2)]
    both = {AlwaysEqual.A, AlwaysEqual.B}
    # The walk is the authority, and it admits the value.
    assert Validator(schema).is_valid(both) is True
    # So the schema is not empty, and nothing may prove it so.
    assert Validator(schema).is_empty() is False


def test_a_constant_that_does_not_equal_itself_is_not_counted_as_one() -> None:
    """`Literal[nan]` denotes no value, so a set of one of them holds none."""
    schema = Annotated[set[Literal[math.nan]], at.MinLen(1)]  # ty: ignore[invalid-type-form]
    assert Validator(schema).is_valid({math.nan}) is False
    assert Validator(schema).is_valid(set()) is False
    # A refutation claims a value of the subject outside the other schema, and
    # this subject has none, so the relation is not refuted.
    assert Validator(schema).relation_to(int) != "not_subset"


def test_a_constant_a_set_cannot_hold_is_not_counted_as_one() -> None:
    """A member with no hash is a member no set carries."""
    schema = Annotated[set[Literal[Unhashable.A]], at.MinLen(1)]
    assert Unhashable.__hash__ is None
    assert Validator(schema).relation_to(int) != "not_subset"


def test_a_constant_the_oracle_trusts_is_counted_exactly() -> None:
    """The rows the count is for, so a decline does not swallow them.

    `None` and an ordinary enumeration member are each one value, and a set
    asking for two of them holds none.
    """
    assert Validator(Annotated[set[None], at.MinLen(2)]).is_empty() is True
    assert Validator(Annotated[set[None], at.MinLen(1)]).is_empty() is False
    one_colour = Annotated[set[Literal[Colour.RED]], at.MinLen(2)]
    assert Validator(one_colour).is_empty() is True
    two_colours = Annotated[set[Literal[Colour.RED, Colour.GREEN]], at.MinLen(2)]
    assert Validator(two_colours).is_empty() is False
    assert Validator(two_colours).is_valid({Colour.RED, Colour.GREEN}) is True


def test_the_two_booleans_bound_a_set_of_them() -> None:
    """`bool` denotes two values, and a set asking for three holds none."""
    assert Validator(Annotated[set[bool], at.MinLen(3)]).is_empty() is True
    assert Validator(Annotated[set[bool], at.MinLen(2)]).is_empty() is False
    assert Validator(Annotated[set[bool], at.MinLen(2)]).is_valid({True, False}) is True


def test_a_bound_of_zero_is_met_by_the_empty_set() -> None:
    """Whatever the element denotes, the empty set has that many members."""
    assert Validator(Annotated[set[None], at.MaxLen(3)]).is_empty() is False
    empty_element = Annotated[set[Literal[math.nan]], at.MaxLen(3)]  # ty: ignore[invalid-type-form]
    assert Validator(empty_element).is_valid(set()) is True
    assert Validator(empty_element).is_empty() is False


def test_a_sequence_repeats_one_element_where_a_set_does_not() -> None:
    """The distinction the count exists for, held from the other side."""
    # A list takes any length by repeating one element.
    assert Validator(Annotated[list[None], at.MinLen(5)]).is_empty() is False
    assert Validator(Annotated[list[None], at.MinLen(5)]).is_valid([None] * 5) is True
    # A set of the same element cannot.
    assert Validator(Annotated[set[None], at.MinLen(5)]).is_empty() is True
