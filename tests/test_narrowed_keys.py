"""A map key narrowed to the words of a regular language.

`docs/15-decidability.md` says a `str` narrowed by a pattern or a length is held by
a split of the map's `str` keys, so a relation about such a map is decided and a
key takes one answer however the clauses overlap. These hold that against the walk:
every answer is checked over dicts drawn from the keys the languages tell apart.
"""

from __future__ import annotations

import itertools
import json
from typing import Annotated, Any

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Regex,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    union,
)

A = Annotated[str, Regex("a.*")]
Z = Annotated[str, Regex(".*z")]
XA = Annotated[str, Regex("x-.*")]
TWO = Annotated[str, at.MinLen(2)]


@pytest.mark.parametrize(
    ("narrow", "wide", "expected"),
    [
        pytest.param(dict[A, int], dict[str, int], "subset", id="a-language-is-part"),
        pytest.param(
            dict[str, int], dict[A, int], "not_subset", id="and-not-the-whole"
        ),
        pytest.param(dict[A, int], dict[A, str], "not_subset", id="on-a-value"),
        pytest.param(dict[A, bool], dict[A, int], "subset", id="value-widens"),
        pytest.param(
            dict[Annotated[str, Regex("ab.*")], int],
            dict[A, int],
            "subset",
            id="a-smaller-language",
        ),
        pytest.param(
            dict[A, int],
            dict[Annotated[str, Regex("ab.*")], int],
            "not_subset",
            id="a-larger-language",
        ),
        pytest.param(
            dict[TWO, int],
            dict[Annotated[str, at.MinLen(1)], int],
            "subset",
            id="a-length-is-a-language",
        ),
        pytest.param(
            dict[Annotated[str, at.MinLen(1)], int],
            dict[TWO, int],
            "not_subset",
            id="a-shorter-length-is-outside",
        ),
        pytest.param(
            {A: int, Z: str},
            dict[str, int | str],
            "subset",
            id="overlap-reads-the-join",
        ),
        pytest.param(
            {A: int, Z: str},
            dict[A, int],
            "not_subset",
            id="overlap-holds-more-than-either",
        ),
        pytest.param(dict[A, int], {A: int, Z: str}, "subset", id="each-clause-holds"),
        pytest.param(
            {"ab": int},
            dict[A, int],
            "subset",
            id="a-field-a-language-holds",
        ),
        pytest.param(
            {"b": int},
            dict[A, int],
            "not_subset",
            id="a-field-no-language-holds",
        ),
        pytest.param(
            {"ab?": int, A: str},
            {A: str},
            "not_subset",
            id="a-field-is-read-before-its-clause",
        ),
        pytest.param(
            dict[Annotated[str, Regex("a|b")], int],
            {"a?": int, "b?": int},
            "subset",
            id="a-finite-language-is-its-keys",
        ),
        pytest.param(
            {"a?": int, "b?": int},
            dict[Annotated[str, Regex("a|b")], int],
            "subset",
            id="and-back",
        ),
        pytest.param(
            intersection(dict[A, int], dict[Z, int]),
            dict[Annotated[str, Regex("a(.*z)?|.*z")], int],
            "subset",
            id="a-meet-reads-the-refinement",
        ),
        pytest.param(
            dict[Annotated[str, Regex("(?s).*")], int],
            dict[str, int],
            "subset",
            id="a-pattern-holds-no-lone-surrogate",
        ),
        pytest.param(
            dict[str, int],
            dict[Annotated[str, Regex("(?s).*")], int],
            "not_subset",
            id="and-a-str-does",
        ),
        pytest.param(
            intersection(dict[str, Any], complement(dict[A, int])),
            nothing,
            "not_subset",
            id="a-complement-holds-its-failures",
        ),
    ],
)
def test_a_relation_over_narrowed_keys_is_decided(
    narrow: object, wide: object, expected: str
) -> None:
    assert Validator(narrow).relation_to(wide) == expected


def test_a_length_holds_the_lone_surrogate_a_pattern_does_not() -> None:
    # `(?s)` leaves the surrogate as the one word between the two: a length counts a
    # code point, and a pattern matches text, which a lone surrogate has none of. A
    # length read over text alone would prove this subset, which the walk refutes.
    counted = dict[Annotated[str, at.MinLen(1)], int]
    matched = dict[Annotated[str, Regex("(?s).+")], int]
    apart = {"\ud800": 1}
    assert Validator(counted).is_valid(apart)
    assert not Validator(matched).is_valid(apart)
    assert Validator(counted).relation_to(matched) == "not_subset"
    assert Validator(matched).relation_to(counted) == "subset"
    difference = Validator(intersection(counted, complement(matched)))
    assert difference.is_valid(apart)
    assert difference.relation_to(nothing) == "not_subset"


def test_clauses_that_overlap_give_a_key_one_answer() -> None:
    both = Validator({A: int, Z: str})
    assert both.is_valid({"az": 1})
    assert both.is_valid({"az": "s"})
    assert not both.is_valid({"az": 1.5})
    assert both.is_valid({"ab": 1})
    assert not both.is_valid({"ab": "s"})
    assert both.is_valid({"bz": "s"})
    assert not both.is_valid({"bz": 1})
    assert not both.is_valid({"b": 1})


def test_a_field_is_read_before_the_clause_holding_it() -> None:
    record = Validator({"ab": int, A: str})
    assert record.is_valid({"ab": 1})
    assert not record.is_valid({"ab": "s"})
    assert record.is_valid({"ab": 1, "ac": "s"})
    assert not record.is_valid({"ab": 1, "ac": 1})
    assert not record.is_valid({"ac": "s"}), "the field is required"


def test_a_narrowed_key_reads_a_document_as_it_reads_the_object() -> None:
    headers = Validator({XA: str, "n": int})
    for document in (
        '{"n": 1, "x-a": "s"}',
        '{"n": 1, "y": "s"}',
        '{"n": 1, "x-a": 1}',
    ):
        assert headers.is_valid_json(document) == headers.is_valid(
            json.loads(document)
        ), document
    assert headers.is_valid_json('{"n": 1, "x-a": "s"}')
    assert not headers.is_valid_json('{"n": 1, "y": "s"}')


def test_opening_and_closing_read_the_words_the_clause_leaves() -> None:
    closed = Validator(dict[A, int])
    opened = closed.open()
    assert opened.is_valid({"b": object()})
    assert opened.is_valid({"ab": 1})
    assert not opened.is_valid({"ab": object()})
    assert closed.relation_to(opened) == "subset"
    assert opened.relation_to(closed) == "not_subset"
    assert opened.close() == closed


def test_a_narrowed_key_renders_as_it_reads_back() -> None:
    schema = Validator(dict[A, int])
    assert repr(schema) == "dict[Annotated[str, Regex('a.*')], int]"
    assert Validator(dict[A, int]) == schema
    assert Validator(dict[Annotated[str, Regex("b.*")], int]) != schema
    assert hash(Validator(dict[A, int])) == hash(schema)


#: The keys the drawn languages tell apart, and the values their clauses do.
_KEYS = ("", "a", "b", "z", "az", "ab", "bz", "x-a", "x-z", "aa")
_VALUES = (0, "s")


def _dicts() -> list[dict[object, object]]:
    found: list[dict[object, object]] = [
        dict(zip(keys, values, strict=True))
        for size in (0, 1, 2)
        for keys in itertools.combinations(_KEYS, size)
        for values in itertools.product(_VALUES, repeat=size)
    ]
    found.extend(
        [
            dict.fromkeys(_KEYS, 0),
            dict.fromkeys(_KEYS, "s"),
            {"a": 0, "ab": "s", "z": 0},
            {7: 0},
            {b"a": 0},
            {"\ud800": 0},
        ]
    )
    return found


_DICTS = _dicts()

_language = st.sampled_from(
    [
        A,
        Z,
        XA,
        TWO,
        Annotated[str, Regex("a|b")],
        Annotated[str, Regex("a")],
        Annotated[str, at.MaxLen(1)],
        str,
    ]
)
_value = st.sampled_from([int, str, int | str, anything])


@st.composite
def _maps(draw: st.DrawFn) -> Validator[Any]:
    clauses = draw(st.lists(st.tuples(_language, _value), min_size=1, max_size=2))
    written: dict[object, object] = {}
    for key, value in clauses:
        written.setdefault(key, value)
    if draw(st.booleans()):
        written["ab?"] = draw(_value)
    return Validator(written)


_terms = st.recursive(
    _maps(),
    lambda inner: st.one_of(
        st.tuples(inner, inner).map(lambda pair: union(*pair)),
        st.tuples(inner, inner).map(lambda pair: intersection(*pair)),
        inner.map(lambda each: intersection(dict, complement(each))),
    ),
    max_leaves=3,
)


# THEORY: the-str-keys-split-by-languages
@given(_terms, _terms)
def test_a_relation_over_narrowed_keys_answers_as_the_walk_does(
    a: Validator[Any], b: Validator[Any]
) -> None:
    held = [value for value in _DICTS if a.is_valid(value)]
    answer = a.relation_to(b)
    if answer == "subset":
        assert all(b.is_valid(value) for value in held), (a, b)
    if any(not b.is_valid(value) for value in held):
        assert answer != "subset", (a, b)
    if a.is_empty():
        assert not held, a
