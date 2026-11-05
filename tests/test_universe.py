"""A universe is a set: the algebra below the JSON values' top is Boolean again.

`docs/dev/10-theory.md` claims that a value model embedding in Python's takes a
set for its top and no parameter on the algebra. This holds that against real
documents: the complement inside the universe is a meet with its top, a value
outside the universe is outside every set below it, and the relations asked
inside it answer as the walk does.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import annotated_types as at
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

NUMBER = intersection(int | float, complement(bool))
JSON_VALUE: Validator[Any] = recursive(
    lambda value: union(
        None,
        bool,
        NUMBER,
        str,
        list[value],  # ty: ignore[invalid-type-form]
        dict[str, value],  # ty: ignore[invalid-type-form]
    )
)

#: What a document parses to.
documents = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(alphabet=st.characters(exclude_categories=("Cs",))),
    lambda inner: (
        st.lists(inner, max_size=3)
        | st.dictionaries(
            st.text(alphabet=st.characters(exclude_categories=("Cs",))),
            inner,
            max_size=3,
        )
    ),
    max_leaves=6,
)
#: Values no document parses to, of every other kind.
strangers = st.sampled_from([b"x", (1, 2), {1, 2}, frozenset(), {1: 2}, object(), 1j])

#: Sets to read inside the universe, each a whole kind or part of one.
sets = st.sampled_from(
    [
        Validator(None),
        Validator(bool),
        Validator(int),
        Validator(float),
        NUMBER,
        Validator(str),
        Validator(list[Any]),
        Validator(list[int]),
        Validator(dict[str, Any]),
        Validator(dict[str, int]),
        Validator(bytes),
        Validator(tuple[int, ...]),
        anything,
        nothing,
    ]
)


def within(s: object) -> Validator[Any]:
    return intersection(JSON_VALUE, s)


def outside(s: object) -> Validator[Any]:
    return intersection(JSON_VALUE, complement(s))


# THEORY: a-universe-is-a-set
@given(documents, sets, sets)
def test_the_sets_below_a_universe_are_a_boolean_algebra(
    value: object, a: Validator[Any], b: Validator[Any]
) -> None:
    assert JSON_VALUE.is_valid(value), "the generator draws what a document parses to"
    # One of `a` and its complement holds a value of the universe, and never both.
    assert within(a).is_valid(value) != outside(a).is_valid(value)
    assert union(within(a), outside(a)).is_valid(value)
    assert not intersection(within(a), outside(a)).is_valid(value)
    # De Morgan, with the universe's complement.
    assert outside(union(a, b)).is_valid(value) == (
        outside(a).is_valid(value) and outside(b).is_valid(value)
    )
    assert outside(intersection(a, b)).is_valid(value) == (
        outside(a).is_valid(value) or outside(b).is_valid(value)
    )
    # A proof read inside the universe holds of the values it asks about.
    if within(a).is_subtype_of(b) and a.is_valid(value):
        assert b.is_valid(value)
    if intersection(within(a), complement(b)).is_empty() and a.is_valid(value):
        assert b.is_valid(value)


# THEORY: a-universe-is-a-set
@given(strangers, sets)
def test_a_value_outside_the_universe_is_outside_every_set_below_it(
    value: object, a: Validator[Any]
) -> None:
    assert not JSON_VALUE.is_valid(value)
    assert not within(a).is_valid(value)
    assert not outside(a).is_valid(value)


@given(documents)
def test_a_document_reads_as_the_value_it_parses_to(value: object) -> None:
    # The embedding the universe rests on: the two readers give one answer, so
    # the JSON path holds no value model of its own.
    text = json.dumps(value)
    assert JSON_VALUE.is_valid_json(text) == JSON_VALUE.is_valid(json.loads(text))


def test_kinds_a_document_cannot_hold_are_empty_inside_the_universe() -> None:
    assert within(bytes).is_empty()
    assert within(tuple[int, ...]).is_empty()
    assert within(set[int]).is_empty()
    assert not within(list[int]).is_empty()
    # `{}` is a dict with no key at all, so a mapping keyed by what no document
    # holds as a key keeps that one value.
    assert not within(dict[int, int]).is_empty()
    assert within(dict[int, int]).is_valid({})
    assert not within(dict[int, int]).is_valid({1: 1})


def test_one_element_is_a_complement_and_a_count_is_a_predicate() -> None:
    some_int = intersection(list, complement(list[complement(int)]))  # ty: ignore[invalid-type-form]
    assert some_int.is_valid([1, "a"])
    assert not some_int.is_valid(["a"])
    assert not some_int.is_valid([])
    assert (
        Validator(Annotated[list[int], at.MinLen(1)]).relation_to(some_int) == "subset"
    )
    assert Validator(list[int]).relation_to(some_int) == "not_subset"

    twice = Validator(
        Annotated[
            list[Any],
            at.Predicate(lambda items: sum(isinstance(i, int) for i in items) >= 2),
        ]
    )
    assert twice.is_valid([1, 2])
    assert not twice.is_valid([1, "a"])
    # The count is opaque: exact on membership, and every relation declines.
    assert twice.relation_to(some_int) == "undecided"
