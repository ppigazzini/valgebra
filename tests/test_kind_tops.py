"""Each kind's top denotes what the table says, at the value that decides it.

A representation closed under complement is only as good as the set its top
names: a top too wide makes a difference inhabited that is empty, and one too
narrow makes an inclusion refutable that holds. The theory page carries a
table -- one row per kind, the set its top denotes, and the boundary value a
test carries for it. This is that table as a test: for every row the kind's
schema admits the deciding value, its complement refuses it, and the meet of
the two is decided empty, which is the complement law read at the one value
each row was written for.

Two rows are about what the kinds do *not* hold. A hashable subclass of an
unhashable kind is a set member, so `set[object]` holds it and the set top is
not "a set of the hashable kinds". And a class whose metaclass answers
`isinstance` by running code is not a set, so every relation over it declines
where a plain class is decided.
"""

from __future__ import annotations

import pytest

from valgebra import Validator, anything, complement, intersection


class _HashableList(list):  # type: ignore[type-arg]
    """A list that is hashable, so a set may hold it."""

    def __hash__(self) -> int:  # type: ignore[override]
        return 0


class _Flip(type):
    """A metaclass whose `isinstance` runs code, so the class is not a set."""

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, int)


class _Coin(metaclass=_Flip):
    """Its instances are whatever the metaclass says they are."""


class _Plain:
    """A class the bindings can read: no layout, no hook."""


#: The table: the kind's schema, and the values that decide its top.
_TOPS: dict[str, tuple[object, list[object]]] = {
    "NoneType": (type(None), [None]),
    "bool": (bool, [True, False]),
    "int": (int, [-(2**63) + 1, 2**63 - 1, 2**70, True]),
    "float": (float, [2.0**53 + 1, float("nan"), float("inf"), -0.0]),
    "str": (str, ["\n", "a\nb", "", "\ud800"]),
    "bytes": (bytes, [b"\n", b""]),
    "list": (list, [[], [2**70], [float("nan")]]),
    "tuple": (tuple, [(), (2**70,)]),
    "set": (set, [set(), {_HashableList([1])}]),
    "frozenset": (frozenset, [frozenset(), frozenset({_HashableList([1])})]),
    "dict": (dict, [{}, {1: "a"}, {True: "a"}]),
    "instance": (_Plain, [_Plain()]),
}


# THEORY: each-kind-is-closed
@pytest.mark.parametrize("kind", list(_TOPS))
def test_each_kinds_top_is_decided_at_the_value_that_decides_it(kind: str) -> None:
    schema, deciding = _TOPS[kind]
    top = Validator(schema)
    outside = complement(schema)
    for value in deciding:
        assert top.is_valid(value), f"{kind}'s top refuses {value!r}"
        assert not outside.is_valid(value), f"{kind}'s complement admits {value!r}"
    assert intersection(schema, complement(schema)).is_empty(), kind
    assert top.relation_to(anything) == "subset", kind


# THEORY: each-kind-is-closed
def test_a_hashable_subclass_of_an_unhashable_kind_is_a_set_member() -> None:
    """The set top is every hashable value, not every value of a hashable kind."""
    member = _HashableList([1])
    assert Validator(set).is_valid({member})
    assert Validator(set[object]).is_valid({member})
    assert Validator(set[list]).is_valid({member})
    assert not Validator(set[int]).is_valid({member})


# THEORY: each-kind-is-closed
def test_a_class_whose_metaclass_runs_isinstance_declines_every_relation() -> None:
    """The instance top is the class lattice, and a hook is not in it."""
    assert Validator(_Coin).is_valid(1)  # the walk asks, and reads the answer
    assert not Validator(_Coin).is_valid("a")
    assert Validator(_Coin).relation_to(int) == "undecided"
    assert Validator(int).relation_to(_Coin) == "undecided"
    assert Validator(_Coin).relation_to(_Plain) == "undecided"
    # A plain class beside it is decided, so the decline is the hook's: a
    # plain class holds an object, and an object is not an `int`, so the
    # inclusion is refuted; against the complement of a kind it is undecided,
    # since a subclass may derive from the kind too.
    assert Validator(_Plain).relation_to(int) == "not_subset"
    assert Validator(_Plain).relation_to(complement(int)) == "undecided"
    assert Validator(_Plain).relation_to(_Plain) == "subset"
