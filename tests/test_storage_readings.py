"""What a container holds is read from its storage, not from what it says.

A schema over a container denotes the values it *holds*: `set[int]` is the sets
whose every member is an integer, and `Annotated[str, MinLen(3)]` is the strings
of three characters or more. A subclass may override `__len__` or `__iter__` and
answer anything, and a walk that believes the override decides membership of a
set that is not the value -- admitting a container whose storage the schema
excludes, which is an accept no value supports.

The rule is one rule for every container the walk reads, and it is the rule
`docs/dev/04-walk.md` already states for a list and a tuple: a subclass that
*inherits* the base's slot is read where it lies, and one that **overrides** it
is read through the base type's own slot. Both halves are held here, because
reading every subclass through the base would cost the common subclass -- a
`NamedTuple`, an `IntEnum`'s container -- the copy it does not need.

Each row runs in both modes. A lie the fast walk believes and the explaining
walk catches is two answers for one value, which the walk page puts first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator

if TYPE_CHECKING:
    from collections.abc import Iterator


class LyingLen(set):
    """A set whose `__len__` is nine whatever it holds."""

    __slots__ = ()

    def __len__(self) -> int:
        return 9


class LyingIter(set):
    """A set whose `__iter__` yields integers whatever it holds."""

    __slots__ = ()

    def __iter__(self) -> Iterator[int]:
        return iter([1, 2, 3])


class LyingFrozenIter(frozenset):
    """The same lie, one kind over."""

    __slots__ = ()

    def __iter__(self) -> Iterator[int]:
        return iter([1, 2, 3])


class LyingFrozenLen(frozenset):
    __slots__ = ()

    def __len__(self) -> int:
        return 9


class LyingDictLen(dict):
    __slots__ = ()

    def __len__(self) -> int:
        return 9


class LyingStrLen(str):
    __slots__ = ()

    def __len__(self) -> int:
        return 9


class LyingBytesLen(bytes):
    __slots__ = ()

    def __len__(self) -> int:
        return 9


class LyingListLen(list):
    __slots__ = ()

    def __len__(self) -> int:
        return 9


class LyingTupleLen(tuple):
    __slots__ = ()

    def __len__(self) -> int:
        return 9


class QuietSet(set):
    """A subclass that overrides neither slot: read where it lies."""

    __slots__ = ()


def _both_modes(schema: object, value: object) -> tuple[bool, bool]:
    """Membership as `is_valid` and as `validate` report it.

    The two are one answer by the walk page's first sentence, so a row that
    returns `(True, False)` is a defect whichever of the two is right.
    """
    compiled = Validator(schema)
    fast = compiled.is_valid(value)
    try:
        compiled.validate(value)
        explained = True
    except ValidationError:
        explained = False
    return fast, explained


# Each row: the schema, a value whose storage the schema excludes, and the name
# of the slot the subclass lies through.
_OUTSIDE: list[tuple[str, object, object]] = [
    ("set __iter__", set[int], LyingIter({"a"})),
    ("set __iter__ under a bound", Annotated[set[int], at.MinLen(1)], LyingIter({"a"})),
    ("frozenset __iter__", frozenset[int], LyingFrozenIter({"a"})),
    ("set __len__", Annotated[set[int], at.MinLen(3)], LyingLen({1})),
    ("frozenset __len__", Annotated[frozenset[int], at.MinLen(3)], LyingFrozenLen({1})),
    ("dict __len__", Annotated[dict[str, int], at.MinLen(3)], LyingDictLen({"a": 1})),
    ("str __len__", Annotated[str, at.MinLen(3)], LyingStrLen("a")),
    ("bytes __len__", Annotated[bytes, at.MinLen(3)], LyingBytesLen(b"a")),
    ("list __len__", Annotated[list[int], at.MinLen(3)], LyingListLen([1])),
    ("tuple __len__", Annotated[tuple[int, ...], at.MinLen(3)], LyingTupleLen((1,))),
]


@pytest.mark.parametrize(
    ("name", "schema", "value"), _OUTSIDE, ids=[row[0] for row in _OUTSIDE]
)
def test_a_subclass_does_not_talk_its_way_into_a_schema(
    name: str, schema: object, value: object
) -> None:
    """The storage decides, so the value is outside the schema in both modes."""
    assert _both_modes(schema, value) == (False, False), name


# Each row: the schema and a value whose storage the schema admits. The lie is
# in the other direction here -- the override would refuse a value that belongs.
_INSIDE: list[tuple[str, object, object]] = [
    ("set __iter__", set[str], LyingIter({"a"})),
    ("set __len__", Annotated[set[int], at.MaxLen(1)], LyingLen({1})),
    ("frozenset __len__", Annotated[frozenset[int], at.MaxLen(1)], LyingFrozenLen({1})),
    ("dict __len__", Annotated[dict[str, int], at.MaxLen(1)], LyingDictLen({"a": 1})),
    ("str __len__", Annotated[str, at.MaxLen(1)], LyingStrLen("a")),
    ("bytes __len__", Annotated[bytes, at.MaxLen(1)], LyingBytesLen(b"a")),
]


@pytest.mark.parametrize(
    ("name", "schema", "value"), _INSIDE, ids=[row[0] for row in _INSIDE]
)
def test_a_subclass_does_not_talk_its_way_out_of_a_schema(
    name: str, schema: object, value: object
) -> None:
    """The storage decides here too, so the value belongs in both modes."""
    assert _both_modes(schema, value) == (True, True), name


def test_a_subclass_that_overrides_neither_slot_is_read_where_it_lies() -> None:
    """The common subclass keeps the reading its storage already gives.

    The base slot is asked only of a type that overrides, so a subclass adding
    behaviour beside the container's own pays nothing for this rule.
    """
    assert _both_modes(set[int], QuietSet({1, 2})) == (True, True)
    assert _both_modes(set[int], QuietSet({"a"})) == (False, False)
    bounded = Annotated[set[int], at.MinLen(2)]
    assert _both_modes(bounded, QuietSet({1, 2})) == (True, True)


def test_the_two_length_readings_agree_on_one_value() -> None:
    """A length bound and the shape beside it count the same items.

    A schema carrying both is where believing the override showed first: the
    bound read the lie and the element walk read the storage, so one value
    satisfied each constraint in a different sense.
    """
    schema = Annotated[set[int], at.MinLen(3)]
    value = LyingLen({1})
    assert Validator(schema).is_valid(value) is False
    # And the element walk beside it reads the same one member.
    assert Validator(set[int]).is_valid(value) is True


def test_an_element_a_set_does_not_hold_is_not_reported() -> None:
    """The explaining walk names the storage's member, not the iterator's."""
    with pytest.raises(ValidationError) as caught:
        Validator(set[int]).validate(LyingIter({"a"}))
    codes = [item["code"] for item in caught.value.errors]
    assert codes == ["int_type"]
    assert caught.value.errors[0]["value"] == "'a'"
