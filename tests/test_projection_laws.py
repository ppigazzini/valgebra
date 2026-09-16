"""What `open` and `close` are laws of, and where they are not.

Every other operation on this surface is a function of the *set* a schema
denotes: two schemas admitting the same values are one schema to `is_subtype_of`,
to `is_empty`, and to every fold a constructor applies. `open` and `close` are
not. They rewrite the records a schema is written out of, so two terms denoting
one set can open into two different sets -- and a caller who reads them as set
operations gets an answer that depends on how the schema was spelled.

That is a deliberate narrowing rather than a defect: the operators read a
spelling because the node they rewrite carries the field list as written. What is
owed is a test saying exactly where the law holds and exactly where it does not,
with the value that separates the two sides -- so that a widening changes this
file, and a reader who wants the law knows what to write instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import Validator, intersection, union

if TYPE_CHECKING:
    from collections.abc import Mapping

# --- Where the law holds --------------------------------------------------

_FIELDS = ["a", "b"]


def _record(spelling: Mapping[str, object]) -> Validator:
    return Validator(spelling)


@st.composite
def _equivalent_pairs(draw: st.DrawFn) -> tuple[Validator, Validator]:
    """Two spellings of one record, drawn so they denote the same set.

    The rewritings are ones that preserve the set *and* the field list: a member
    repeated in a union, a meet with the schema itself, and a record built with
    its keys in the other order. Over this fragment `open` and `close` are
    congruences, and the rows below say so.
    """
    names = draw(
        st.lists(st.sampled_from(_FIELDS), min_size=1, max_size=2, unique=True)
    )
    optional = draw(st.lists(st.booleans(), min_size=len(names), max_size=len(names)))
    types = draw(
        st.lists(
            st.sampled_from([int, str, bool]), min_size=len(names), max_size=len(names)
        )
    )
    spelling = {
        f"{name}?" if is_optional else name: kind
        for name, is_optional, kind in zip(names, optional, types, strict=True)
    }
    one = _record(spelling)
    rewriting = draw(st.sampled_from(["itself", "union", "meet", "reordered"]))
    if rewriting == "itself":
        return one, _record(dict(spelling))
    if rewriting == "union":
        return one, union(spelling, spelling)
    if rewriting == "meet":
        return one, intersection(spelling, spelling)
    return one, _record(dict(reversed(list(spelling.items()))))


@given(pair=_equivalent_pairs())
def test_a_rewriting_that_keeps_the_fields_opens_and_closes_alike(
    pair: tuple[Validator, Validator],
) -> None:
    """The fragment the law holds over, and it is the one a caller writes."""
    one, other = pair
    assert one.is_equivalent(other)
    assert one.open().is_equivalent(other.open())
    assert one.close().is_equivalent(other.close())


def test_the_projections_are_idempotent() -> None:
    """Opening an open record and closing a closed one change nothing."""
    record = Validator({"a": int, "b?": str})
    assert record.open().open().is_equivalent(record.open())
    assert record.close().close().is_equivalent(record.close())
    assert repr(record.open().open()) == repr(record.open())
    assert repr(record.close().close()) == repr(record.close())
    # And closing an opened record gives the record back, which is what makes
    # them projections rather than inverses.
    assert record.open().close().is_equivalent(record)


def test_closing_a_mapping_leaves_it_alone() -> None:
    """A clause over every key is not a record's catch-all.

    `close` drops a *record's* catch-all. A pure mapping declares no field, so
    there is no record to close and the clause is the schema rather than an
    addition to it. The theory's own operator would give the empty record here;
    this one reads the term, and the difference is the deviation below.
    """
    mapping = Validator(dict[str, int])
    assert mapping.close().is_equivalent(mapping)
    assert mapping.open().is_equivalent(mapping)
    assert mapping.close().is_valid({"a": 1}) is True
    # A record with a typed catch-all beside a field does close.
    mixed = Validator({"a": int, str: int})
    assert mixed.close().is_equivalent({"a": int})
    assert mixed.close().is_valid({"a": 1, "b": 2}) is False


# --- Where the law does not hold ------------------------------------------


def test_opening_is_not_a_function_of_the_set() -> None:
    """Two terms denoting one set open into two, and a value says which.

    `{"a?": int}` and `{} | {"a": int}` admit the same dicts: the empty one, and
    the one mapping `a` to an integer. Opening the first admits any *other* key
    while keeping `a` an integer; opening the second admits every dict, because
    the branch with no field declared opens to every dict on its own.
    """
    one = Validator({"a?": int})
    other = union({}, {"a": int})
    assert one.is_equivalent(other), "the same set, two spellings"

    witness = {"a": "x"}
    assert one.is_valid(witness) is False
    assert other.is_valid(witness) is False
    # And after opening they part, on that value.
    assert one.open().is_valid(witness) is False
    assert other.open().is_valid(witness) is True
    assert one.open().is_equivalent(other.open()) is False
    # The direction is not arbitrary: opening a term with fewer branches gives
    # the smaller set, so the relation is an inclusion rather than a disjointness.
    assert one.open().relation_to(other.open()) == "subset"
    assert other.open().relation_to(one.open()) == "not_subset"


def test_closing_is_not_a_function_of_the_set_either() -> None:
    """The same, one projection over, with the pair the core's own rows name."""
    free = Validator({"a?": Any})
    every_dict = Validator(dict[Any, Any])
    # Both admit every dict, by different routes.
    for value in ({}, {"a": 1}, {"b": "x"}, {"a": None, "b": 2}):
        assert free.open().is_valid(value) is True
        assert every_dict.is_valid(value) is True
    # Closing keeps the declared field on one and the clause on the other.
    assert free.close().is_valid({"b": "x"}) is False
    assert every_dict.close().is_valid({"b": "x"}) is True
    assert free.close().is_equivalent(every_dict.close()) is False


def test_the_relations_are_functions_of_the_set() -> None:
    """The control: every other answer on this surface reads the set.

    A respelling changes `repr` and `==`, and changes nothing a relation
    reports -- which is what makes the two rows above the exception rather than
    the rule.
    """
    record = Validator({"a": int})
    respelled = union(record, intersection(record, Validator(str)))
    assert respelled != record
    assert record.is_subtype_of(respelled)
    assert respelled.is_subtype_of(record)
    assert record.is_equivalent(respelled)
    assert respelled.is_empty() == record.is_empty()


@pytest.mark.parametrize(
    "spelling",
    [{"a": int}, {"a?": int}, {"a": int, str: int}, {str: int}, {}],
    ids=["closed", "optional", "catch-all", "mapping", "empty"],
)
def test_a_projection_returns_a_new_validator(spelling: object) -> None:
    """Neither operator changes the validator it is asked of."""
    original = Validator(spelling)
    before = repr(original)
    opened, closed = original.open(), original.close()
    assert repr(original) == before
    assert opened is not original
    assert closed is not original
