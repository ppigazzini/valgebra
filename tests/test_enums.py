"""An enumeration is the union of its members only when it *is* that union.

Reading a class as `Literal[*cls]` is sound only if every instance of the class
is one of the values `list(cls)` yields. Three enumeration kinds break that,
and two of them are easy to read as the union anyway:

* a `Flag` builds instances with `|` that the class never listed, so read as its
  members, `Validator(P).is_subtype_of(Literal[P.A, P.B])` is `True` with
  `P.A | P.B` standing against it;
* an `Enum` with no members can still be subclassed -- that is how an enum base
  class is written -- so reading it as the empty union makes it a subtype of
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


class Color(enum.Enum):
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
    members = Literal[Color.RED, Color.GREEN]
    assert Validator(Color).is_subtype_of(members)
    assert Validator(members).is_subtype_of(Color)
    assert Validator(Color).is_equivalent(members)
    # And two members are two values, which is what the identity check buys.
    assert intersection(Literal[Color.RED], Literal[Color.GREEN]).is_empty()


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


class _Missing(enum.Enum):
    """An enumeration whose ``_missing_`` makes an instance for any value."""

    A = 1
    B = 2

    @classmethod
    def _missing_(cls, value: object) -> _Missing:
        pseudo = object.__new__(cls)
        pseudo._name_, pseudo._value_ = f"V{value}", value
        return pseudo


class _HideMeta(enum.EnumMeta):
    """A metaclass leaving one member out of the members it lists."""

    def __iter__(cls):  # noqa: ANN204 - the metaclass's own signature
        members = super().__iter__()
        return (member for member in members if member.name != "HIDDEN")  # ty: ignore[unresolved-attribute]


class _Hidden(enum.Enum, metaclass=_HideMeta):
    A = 1
    B = 2
    HIDDEN = 3


def test_an_enumeration_making_an_instance_past_its_members_stays_an_atom() -> None:
    """The union reading stands on the class making no instance but its members.

    A ``_missing_`` of its own is the class making one -- ``_Missing(5)`` -- and
    a metaclass's own ``__iter__`` leaves ``HIDDEN`` out of ``list(_Hidden)``:
    each is an instance the class admits and no listed member's literal does.
    """
    pseudo = _Missing(5)
    listed = Literal[_Missing.A, _Missing.B]
    assert Validator(_Missing).is_valid(pseudo)
    assert not Validator(listed).is_valid(pseudo)
    assert Validator(_Missing).relation_to(listed) != "subset"
    assert not Validator(_Missing).is_equivalent(listed)

    shown = Literal[_Hidden.A, _Hidden.B]
    assert Validator(_Hidden).is_valid(_Hidden.HIDDEN)
    assert not Validator(shown).is_valid(_Hidden.HIDDEN)
    assert Validator(_Hidden).relation_to(shown) != "subset"


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


class _Color(enum.Enum):
    RED = 1
    GREEN = 2


@pytest.mark.xfail(
    strict=True,
    reason=(
        "an enumeration is proved the union of its members because the class "
        "makes no other instance, and `object.__new__` makes one past it"
    ),
)
# TRUST: An enumeration's instances are made through it.
def test_an_instance_made_past_the_class_is_one_of_its_members() -> None:
    """The known-unsound case of reading an enumeration as its members.

    `_Color` and the union of its two members are decided one set. An
    instance `object.__new__` builds is admitted by the class atom and by no
    member's literal, so the inclusion has a value against it.
    """
    ghost = object.__new__(_Color)
    members = Validator(Literal[_Color.RED, _Color.GREEN])

    assert Validator(_Color).is_valid(ghost), "the walk asks `isinstance`"
    assert not members.is_valid(ghost), "no member is the ghost"
    assert Validator(_Color).relation_to(members) != "subset"
