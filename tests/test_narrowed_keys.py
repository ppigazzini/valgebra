"""A map key narrowed to the words of a regular language.

`docs/15-decidability.md` says a `str` narrowed by a pattern or a length is held by
a split of the map's `str` keys, so a relation about such a map is decided and a
key takes one answer however the clauses overlap. These hold that against the walk:
every answer is checked over dicts drawn from the keys the languages tell apart.
"""

from __future__ import annotations

import itertools
import json
from typing import Annotated, Any, Literal

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
    recursive,
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


class _Apart(str):
    """A key of the text `a` that no lookup of the name finds."""

    __slots__ = ()

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other


class _Folded(str):
    """A key equal to the name `a` whose text is `A`, as a header name is."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.lower())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and self.lower() == other.lower()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a language's words are read as the keys it holds, and a key whose hash "
        "is not its text's is one no lookup of the word finds"
    ),
)
# TRUST: A `str` key's hash and `==` are those of its text.
def test_a_language_of_one_word_holds_a_key_the_word_does_not_reach() -> None:
    """The known-unsound case of reading a word as one key."""
    narrowed, record = (
        Validator(dict[Annotated[str, Regex("a")], int]),
        Validator({"a?": int}),
    )
    value = {_Apart("a"): 1}

    assert narrowed.is_valid(value), "the pattern reads the text"
    assert not record.is_valid(value), "and the name reads the hash and the `==`"
    assert narrowed.relation_to(record) != "subset"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a field's name is read as a word of a language, and a key equal to the "
        "name can hold another text"
    ),
)
def test_a_field_holds_a_key_whose_text_is_outside_the_language() -> None:
    """The known-unsound case of reading a name as its word."""
    record, narrowed = (
        Validator({"a": int}),
        Validator(dict[Annotated[str, Regex("a")], int]),
    )
    value = {_Folded("A"): 1}

    assert record.is_valid(value), "the name reads the hash and the `==`"
    assert not narrowed.is_valid(value), "and the pattern reads the text"
    assert record.relation_to(narrowed) != "subset"


def test_a_key_keeping_the_hash_of_its_text_stands_against_no_proof() -> None:
    """What the assumption holds of: a subclass with nothing of its own."""

    class Plain(str):
        __slots__ = ()

    narrowed = Validator(dict[Annotated[str, Regex("a")], int])
    record = Validator({"a?": int})
    assert narrowed.relation_to(record) == "subset"
    assert narrowed.is_valid({Plain("a"): 1})
    assert record.is_valid({Plain("a"): 1})


def test_a_clause_keyed_by_a_literal_is_refuted_beside_its_complement() -> None:
    """`open` writes the complement beside a literal key, which reads no constant."""
    one: Any = Literal["a"]
    assert (
        Validator({one: str}).open().relation_to(Validator({one: bool}).open())
        == "not_subset"
    )
    assert (
        Validator({one: bool}).open().relation_to(Validator({one: int}).open())
        == "subset"
    )
    assert (
        Validator({}).open().relation_to(Validator({one: bool}).open()) == "not_subset"
    )
    assert not Validator({one: bool}).open().is_valid({"a": "s"})
    assert Validator({}).open().is_valid({"a": "s"})


def test_a_name_a_table_leaves_out_is_held_by_its_complement() -> None:
    """A literal pins its text, so no key equal to another name is in the table."""
    one: Any = Literal["a"]
    outside = {complement(one): anything}
    assert Validator({"b": str}).relation_to(outside) == "subset"
    assert Validator({"a": str}).relation_to(outside) != "subset"
    assert Validator({"b": str}).is_valid({"b": "s"})
    assert Validator(outside).is_valid({"b": "s"})
    assert not Validator(outside).is_valid({"a": "s"})


@pytest.mark.parametrize(
    ("wide", "narrow", "witness"),
    [
        (Annotated[str, Regex("a|b")], Annotated[str, Regex("a")], {"b": 1}),
        (Annotated[str, at.MinLen(1)], Annotated[str, at.MinLen(2)], {"a": 1}),
    ],
)
def test_a_word_no_clause_reads_refutes_a_recursion_over_its_keys(
    wide: Any,
    narrow: Any,
    witness: dict[str, int],
) -> None:
    """The sets stop at the fixpoint, and read the two key languages alone."""

    def keyed(key: Any) -> Validator[Any]:
        return recursive(lambda s: union(int, {key: s}))

    assert keyed(wide).is_valid(witness)
    assert not keyed(narrow).is_valid(witness)
    assert keyed(wide).relation_to(keyed(narrow)) == "not_subset"
    assert keyed(narrow).relation_to(keyed(wide)) != "not_subset"
