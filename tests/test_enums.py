"""An enumeration is the union of its members only when it *is* that union.

Reading a class as `Literal[*cls]` is sound only if every instance of the class
is one of the values `list(cls)` yields. Three enumeration kinds break that, and
two of them were read as the union anyway:

* a `Flag` builds instances with `|` that the class never listed, so
  `Validator(P).is_subtype_of(Literal[P.A, P.B])` was `True` with `P.A | P.B`
  standing against it;
* an `Enum` with no members can still be subclassed -- that is how an enum base
  class is written -- so reading it as the empty union made it a subtype of
  `nothing`, with a subclass's member standing against that.

Each row below pairs the relation with the value that refutes it, so a kind that
is read as its members again fails here rather than in a caller's data.
"""

from __future__ import annotations

import enum
import sys
from typing import Literal

import pytest

from valgebra import Validator, complement, intersection, nothing, union


class Colour(enum.Enum):
    """The shape the union reading is for: plain members, identity equality."""

    RED = 1
    GREEN = 2


class Permission(enum.Flag):
    """`|` makes instances of this class that `list(Permission)` never yields."""

    READ = 1
    WRITE = 2


class Access(enum.IntFlag):
    READ = 4
    WRITE = 2


class Base(enum.Enum):
    """No members, so it is a base a subclass may still add members to."""


class Derived(Base):
    X = 1


class Code(enum.IntEnum):
    OK = 1


# `StrEnum` reaches `enum` in 3.11, and nothing below it spells the class this
# row is about; the tests that read it skip beneath that floor.
if sys.version_info >= (3, 11):

    class Name(enum.StrEnum):
        RED = "red"


def test_a_plain_enumeration_is_the_union_of_its_members() -> None:
    members = Literal[Colour.RED, Colour.GREEN]
    assert Validator(Colour).is_subtype_of(members)
    assert Validator(members).is_subtype_of(Colour)
    assert Validator(Colour).is_equivalent(members)
    # And two members are two values, which is what the identity check buys.
    assert intersection(Literal[Colour.RED], Literal[Colour.GREEN]).is_empty()


def test_a_flag_is_not_the_union_of_the_members_it_lists() -> None:
    """`P.A | P.B` is the value that refutes the union reading."""
    both = Permission.READ | Permission.WRITE
    listed = Literal[Permission.READ, Permission.WRITE]

    # The witness: in the class, and not in the union of what it lists.
    assert Validator(Permission).is_valid(both)
    assert not Validator(listed).is_valid(both)

    # So neither direction may be decided, and the meet that admits the witness
    # may not be decided empty.
    assert not Validator(Permission).is_subtype_of(listed)
    assert not Validator(Permission).is_equivalent(listed)
    assert intersection(Permission, complement(listed)).is_valid(both)
    assert not intersection(Permission, complement(listed)).is_empty()


def test_an_int_flag_is_not_the_union_of_the_members_it_lists() -> None:
    both = Access.READ | Access.WRITE
    listed = Literal[Access.READ, Access.WRITE]
    assert Validator(Access).is_valid(both)
    assert not Validator(Access).is_subtype_of(listed)


def test_an_enumeration_with_no_members_is_not_the_empty_set() -> None:
    """A subclass's member is the value that refutes it."""
    assert Validator(Base).is_valid(Derived.X)
    assert not Validator(Base).is_subtype_of(nothing)
    assert not Validator(Base).is_empty()
    assert not Validator(Base).is_equivalent(nothing)


def test_a_subclass_that_has_members_is_the_union_of_them() -> None:
    # The exclusion is "no members", not "derives from an enum": once a class
    # has members it can no longer be subclassed, so the reading holds again.
    assert Validator(Derived).is_subtype_of(Literal[Derived.X])
    assert Validator(Derived).is_equivalent(Literal[Derived.X])


def test_an_enumeration_whose_members_are_not_two_values_stays_an_atom() -> None:
    # An `IntEnum` member equals the integer behind it, so two of them are not
    # two values and the union reading has nothing to stand on.
    assert not Validator(Code).is_subtype_of(Literal[Code.OK])
    # Conservative, not wrong: membership is unaffected.
    assert Validator(Code).is_valid(Code.OK)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="StrEnum")
def test_a_string_enumeration_stays_an_atom_for_the_same_reason() -> None:
    # A `StrEnum` member equals the string behind it, which is the row above
    # over the other value kind a member can carry.
    assert not Validator(Name).is_subtype_of(Literal[Name.RED])
    assert Validator(Name).is_valid(Name.RED)


def test_every_refused_kind_still_answers_membership() -> None:
    """The refusals cost completeness, never an accept or a reject."""
    cases = [
        (Permission, Permission.READ | Permission.WRITE),
        (Access, Access.READ | Access.WRITE),
        (Base, Derived.X),
        (Code, Code.OK),
    ]
    if sys.version_info >= (3, 11):
        cases.append((Name, Name.RED))
    for cls, value in cases:
        assert Validator(cls).is_valid(value), cls
        assert not Validator(cls).is_valid(object()), cls
        # And the class is still what a union with it admits.
        assert union(cls, int).is_valid(value), cls
