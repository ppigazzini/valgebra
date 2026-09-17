"""What `open` and `close` are laws of, and where they are not.

Every other operation on this surface is a function of the *set* a schema
denotes: two schemas admitting the same values are one schema to `is_subtype_of`,
to `is_empty`, and to every fold a constructor applies. `close` is one too.
`open` is not, and the reason is one place: it descends into a union, and a
branch declaring no field frees every key on its own, so `{"a?": int}` and
`{} | {"a": int}` -- one set, two spellings -- open into two.

That is a deliberate narrowing rather than a defect, and it is the only one:
what the operators read of a *keyed map* is the regions its clauses claim, which
is a set of keys rather than a way of writing one. What they must not do is read
an unrelated part of the term -- treat one clause differently according to
whether a field sits beside it -- and the rows below hold that apart from the
union case with the value that separates the two sides.
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


# THEORY: open-and-close-read-the-region
def test_a_clause_is_read_the_same_with_or_without_a_field_beside_it() -> None:
    """One clause, one answer, whatever else the term declares.

    Openness is the default of the key-type region no clause claims, so a clause
    names a region the operators do not touch. A `str => int` clause claims the
    `str` keys whether or not a field is declared beside it, and reading it one
    way in a record and another in a mapping would be the term's *unrelated*
    parts deciding one clause -- which is incoherence rather than the spelling
    sensitivity the union case below declares.
    """
    bare = Validator({str: int})
    beside_a_field = Validator({"a": int, str: int})

    for schema in (bare, beside_a_field):
        # The claimed region is not the operators' to touch, either way.
        assert schema.close().is_equivalent(schema), schema
        assert schema.close().is_valid({"a": 1}) is True, schema

    # And opening frees what the clause leaves over, in both -- a key no clause
    # claims, which is any key that is not a `str`.
    assert bare.open().is_valid({1: "anything at all"}) is True
    assert beside_a_field.open().is_valid({"a": 1, 2: "free"}) is True
    # while the region the clause claims still says what a `str` key maps to.
    assert beside_a_field.open().is_valid({"a": 1, "b": "not an int"}) is False


# --- Where the law does not hold ------------------------------------------


# THEORY: open-and-close-read-the-region
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


# THEORY: open-and-close-read-the-region
def test_closing_is_a_function_of_the_set() -> None:
    """The pair the core's own rows name, and the one `close` used to part.

    `{"a?": Any, ...}` and `dict[Any, Any]` admit every dict, by different
    routes, and they are one term: a keyed map with no field and a catch-all
    clause. Closing sends the region no clause claims to nothing, and `[top]`
    claims every key whichever way the term was written -- so the two close to
    one set. An operator picking a reading for that term would map one set to
    two, which is what puts it outside the algebra.
    """
    free = Validator({"a?": Any}).open()
    every_dict = Validator(dict[Any, Any])
    for value in ({}, {"a": 1}, {"b": "x"}, {"a": None, "b": 2}):
        assert free.is_valid(value) is True, value
        assert every_dict.is_valid(value) is True, value
    assert free.is_equivalent(every_dict), "the same set, two routes"
    assert free.close().is_equivalent(every_dict.close())
    # And the set they close to is the one that admits the empty dict alone.
    assert every_dict.close().is_valid({}) is True
    assert every_dict.close().is_valid({"b": "x"}) is False


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


# THEORY: open-and-close-read-the-region
def test_closing_an_opened_schema_is_not_closing_it() -> None:
    """The two values where the round trip is strict, and what each is about.

    `close` after `open` reads as a round trip and is not one, because `open`
    is not injective: it normalises, and both normalisations are forced.

    A declared name admitting everything says exactly what the catch-all an
    opening writes already says, so it goes -- and it has to, or `{"a?":
    anything}` and `{}`, which are one set once opened, would close to two.
    Two clauses carrying one value are one clause over the union of their keys,
    so opening `dict[str, anything]` gives a single clause over every key --
    and that has to be so too, since the long spelling is keyed by a complement
    and the set representation declines that shape.

    Neither is recoverable, which is why `tests/test_laws.py` holds the
    direction and not the equality.
    """
    # A declared name a full catch-all already says.
    record = Validator({"a?": object})
    assert record.close().is_valid({"a": 1})
    assert record.open().is_valid({"a": 1})
    assert not record.open().close().is_valid({"a": 1})
    assert record.open().close() == Validator({})

    # Two clauses carrying one value, folded into one over every key.
    mapping = Validator(dict[str, object])
    assert mapping.close().is_valid({"a": 1})
    assert repr(mapping.open()) == "dict[anything, anything]"
    assert not mapping.open().close().is_valid({"a": 1})

    # And the shape the round trip does hold of, so the strictness above reads
    # as the exception it is rather than as the rule.
    typed = Validator(dict[str, int])
    assert typed.open().close() == typed
