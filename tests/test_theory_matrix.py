"""Relations checked against values rather than against a recorded answer.

Every other relation suite in this tree pins the answer the procedure gave when
the row was written. That holds a wrong answer exactly as firmly as a right one,
which is how a `True` that no value supports survives a green suite: the sweep
reports the arm as covered, and what covers it is an assertion copied from the
arm.

The rows here carry a **corpus** instead. Membership is decided by the walk,
which is a different procedure from the relations -- a different crate, no shared
code -- so it is an oracle the relations can be wrong against:

* a value in the subject and outside the supertype **refutes** the inclusion, so
  `is_subtype_of` returning `True` is a defect however the rule reached it;
* a relation reported `"not_subset"` claims such a value exists, so a row whose
  corpus carries the boundary values and finds none is a claim with nothing
  under it.

The first is checked for every row. The second is checked where the row says the
inclusion holds, which is what makes the corpus the load-bearing half: the values
are the ones at the edge of each kind's universe -- the newline a length bound
must count, the integer no float equals, the 64-bit end of a residue class, the
hashable subclass of an unhashable kind, the constant that does not equal itself
-- because those are the values a representation built by hand omits.

`expected` is the answer this build gives, which is a ratchet: a row that
regresses to `"undecided"` fails and is still sound, and a row that starts
deciding fails and is a widening someone has to write down. What the *model*
says is carried by the corpus instead, because a value is checkable and a
recorded string is not. A row whose `expected` its own values contradict fails
here, which is the check on whoever writes one.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import annotated_types as at
import pytest

from valgebra import (
    Regex,
    ValidationError,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)


class HashableList(list):
    """A list that a set can hold.

    Hashability is a property of the value, not of its kind, which is why the
    members of `set[list[int]]` are not the members of `set[nothing]`.
    """

    __slots__ = ()
    __hash__ = object.__hash__


class LyingKey(str):
    """A key carrying a field's text that the dict does not find under it."""

    __slots__ = ()

    def __hash__(self) -> int:
        return 12345

    def __eq__(self, other: object) -> bool:
        return False


class PlainKey(str):
    """A key whose text is the field's and whose equality is the text's."""

    __slots__ = ()


@dataclass
class Point:
    x: int


@dataclass
class Other:
    x: int


@runtime_checkable
class HasX(Protocol):
    x: int


class RaisingCheck(type):
    def __subclasscheck__(cls, other: type) -> bool:
        raise RuntimeError(cls)


class Raising(metaclass=RaisingCheck):
    pass


class SaysYes(type):
    def __subclasscheck__(cls, other: type) -> bool:
        return True

    def __instancecheck__(cls, other: object) -> bool:
        return True


class Everything(metaclass=SaysYes):
    pass


class Plain:
    pass


NAN = float("nan")
TWO_53 = 2**53
I64_MIN = -(2**63)
#: A `str` no codec encodes. It is one character long, it is a member of `str`,
#: and a pattern matches the text of a string, which this has none of -- so it is
#: the value that tells a word kind's universe from the language of a pattern
#: over it.
LONE_SURROGATE = "\ud800"

#: Every row: a name, the subject, the supertype, the relation the set model
#: gives, and the values that decide it.
#:
#: A corpus is not a sample. Each value is at a boundary of the kind the row is
#: about, and a row whose corpus reaches neither schema is a row that would pass
#: against any implementation -- which
#: `test_the_corpus_reaches_the_schemas_it_is_about` holds.
ROWS: list[tuple[str, Any, Any, str, list[Any]]] = [
    # -- a length bound counts every symbol, the newline included ------------
    (
        "a length bound counts the newline",
        Annotated[str, at.MinLen(1)],
        Annotated[str, Regex("[^\n]+")],
        "not_subset",
        ["", "\n", "a", "a\nb", "\n\n"],
    ),
    (
        "an upper length bound counts the newline",
        Annotated[str, at.MaxLen(1)],
        Annotated[str, Regex("[^\n]?")],
        "not_subset",
        ["", "\n", "a", "ab"],
    ),
    (
        "a length bound is below a looser one",
        Annotated[str, at.MinLen(2)],
        Annotated[str, at.MinLen(1)],
        "subset",
        ["", "\n", "a", "ab", "a\n"],
    ),
    # -- a set holds what the walk admits, not what a kind suggests ----------
    (
        "a set of an unhashable kind holds a hashable subclass",
        set[int | list[int]],
        union(set[int], int),
        "not_subset",
        [set(), {1}, {HashableList([1])}, 1],
    ),
    (
        "a set orders by its elements",
        set[bool],
        set[int],
        "subset",
        [set(), {True}, {1}, {HashableList([1])}],
    ),
    # -- an integer residue class at the end of the range --------------------
    (
        "the least integer is even",
        Annotated[int, at.Ge(I64_MIN), at.Le(I64_MIN)],
        Annotated[int, at.MultipleOf(2)],
        "subset",
        [I64_MIN, I64_MIN + 1, I64_MIN + 2, 0, 1, 2],
    ),
    (
        "a residue class at the end of the range keeps its members",
        Literal[I64_MIN + 1],  # ty: ignore[invalid-type-form]
        Annotated[int, at.MultipleOf(3)],
        "not_subset",
        [I64_MIN, I64_MIN + 1, I64_MIN + 3, 0, 3],
    ),
    # -- a float bound against an integer no float equals --------------------
    (
        "a float bound reads the integer exactly",
        Annotated[float, at.Lt(TWO_53 + 1)],
        Annotated[float, at.Lt(TWO_53)],
        "not_subset",
        [0.0, -0.0, float(TWO_53), float(TWO_53 + 2), math.inf, NAN],
    ),
    (
        "no float equals the bound, so strictness costs nothing",
        Annotated[float, at.Le(TWO_53 + 1)],
        Annotated[float, at.Lt(TWO_53 + 1)],
        "subset",
        [0.0, float(TWO_53), float(TWO_53 + 2), -math.inf, NAN],
    ),
    (
        "a bound between two neighbours admits no float",
        Annotated[float, at.Ge(TWO_53 + 1), at.Le(TWO_53 + 1)],
        nothing,
        "subset",
        [float(TWO_53), float(TWO_53 + 2), 0.0, NAN],
    ),
    # -- a constant that denotes nothing is in every set ---------------------
    (
        "a constant that does not equal itself is no witness",
        Literal[NAN, 1],  # ty: ignore[invalid-type-form]
        Literal[1],
        "subset",
        [1, NAN, 0, True, 1.0],
    ),
    # -- a key part the walk reads wider than the partition names ------------
    (
        "a bool key is an int key",
        dict[bool, int],
        dict[int, int],
        "subset",
        [{}, {True: 1}, {1: 1}, {"a": 1}],
    ),
    (
        "an int-keyed map covers the bool keys",
        intersection(dict[bool, int], complement(dict[int, int])),
        nothing,
        "subset",
        [{}, {True: 1}, {1: 1}],
    ),
    (
        "a finite key part is exhausted by its labels",
        dict[bool, int],
        dict[Literal[True, False], int],
        "subset",
        [{}, {True: 1}, {False: 1}, {True: 1, False: 2}],
    ),
    (
        "a dict has one entry for an int key and its boolean",
        intersection(
            complement(dict[Literal[1], str]),
            complement(dict[Literal[True], str]),
            dict[Literal[1, True], str],
        ),
        nothing,
        "subset",
        # `{1: "x", True: "y"}` is `{1: "y"}`, and ruff says so -- which is the
        # fact the row is about, so the literal stays and the rule is answered.
        [{}, {1: "x"}, {True: "x"}, {1: "x", True: "y"}, {0: "x"}],  # noqa: F601
    ),
    (
        "two integers are two keys, so the same shape is inhabited",
        intersection(
            complement(dict[Literal[1], str]),
            complement(dict[Literal[2], str]),
            dict[Literal[1, 2], str],
        ),
        nothing,
        "not_subset",
        [{}, {1: "x"}, {2: "x"}, {1: "x", 2: "y"}],
    ),
    # -- a clause governs only the keys it can spell -------------------------
    (
        "a clause keyed by another kind spells no field name",
        dict[int, str],
        {"a?": int, int: str},
        "subset",
        [{}, {1: "x"}, {"a": 1}, {1: "x", 2: "y"}],
    ),
    # -- a word kind's universe is the words the kind can hold ---------------
    (
        "a pattern that matches every text does not reach every string",
        str,
        union(Annotated[str, Regex(r"[\s\S]*")], int),
        "not_subset",
        ["", "a", "\n", "\u00e9", "\U0001f600", 1, LONE_SURROGATE],
    ),
    (
        "a string kind holds the words no pattern matches",
        intersection(str, complement(Annotated[str, Regex("(?s:.)*")])),
        nothing,
        "not_subset",
        ["", "a", "\n", "\u00e9", "\U0001f600", LONE_SURROGATE],
    ),
    (
        "every text a pattern matches is a string",
        Annotated[str, Regex("(?s:.)*")],
        str,
        "subset",
        ["", "a", "\n", "\u00e9", "\U0001f600", LONE_SURROGATE, 1],
    ),
    (
        "bytes keeps the wider universe a str does not have",
        bytes,
        complement(str),
        "subset",
        [b"", b"a", b"\xff", "a", ""],
    ),
    # -- the open world, inside a container ----------------------------------
    (
        "a meet of two unrelated classes has no value to refute with",
        list[intersection(Point, Other)],  # ty: ignore[invalid-type-form]
        list[str],
        "undecided",
        [[], ["a"], [Point(1)], [Other(1)]],
    ),
    # -- a cut reference widens the difference, so it cannot refute ----------
    (
        "a fixpoint reached through a cut is not refuted by the widening",
        list[Annotated[str, Regex("a+")]],
        recursive(
            lambda node: union(list[node], Annotated[str, Regex("a*")])  # ty: ignore[invalid-type-form]
        ),
        "undecided",
        [[], ["a"], ["aa", "a"], [[]], "a", "", "b", ["b"]],
    ),
    # -- a class whose metaclass answers the check is not a set --------------
    (
        "a protocol whose issubclass raises decides nothing",
        Plain,
        HasX,
        "undecided",
        [Plain(), Point(1), 1, "a"],
    ),
    (
        "a class whose subclass check raises decides nothing",
        Plain,
        Raising,
        "undecided",
        [Plain(), 1, "a"],
    ),
    (
        "a class whose instance check answers yes decides nothing",
        Plain,
        Everything,
        "undecided",
        [Plain(), 1, "a"],
    ),
    # -- rows that must keep deciding ----------------------------------------
    ("bool is below int", bool, int, "subset", [True, False, 0, 1, 2, "a"]),
    ("int is not below bool", int, bool, "not_subset", [True, 1, 0, 2]),
    (
        "a literal is below its kind's complement",
        Literal["a"],
        complement(int),
        "subset",
        ["a", "b", 1, True],
    ),
    (
        "a list of bools is a list of ints",
        list[bool],
        list[int],
        "subset",
        [[], [True], [1], [True, False]],
    ),
    (
        "a fixed sequence splits across the branches that cover it",
        tuple[int | str, int],
        union(tuple[int, int], tuple[str, int]),
        "subset",
        [(1, 1), ("a", 1), (1, "a"), ()],
    ),
    (
        "a closed record is below a catch-all mapping",
        {"a": int},
        dict[str, int],
        "subset",
        [{}, {"a": 1}, {"a": "x"}, {"a": 1, "b": 2}],
    ),
    (
        "a required key the subject does not declare refutes",
        dict[str, int],
        {"a": int},
        "not_subset",
        [{}, {"a": 1}, {"b": 1}],
    ),
    (
        "the top is spelled and is still the top",
        int,
        anything,
        "subset",
        [1, "a", None, [], object()],
    ),
]


def _admits(schema: Validator, value: Any) -> bool:
    """Membership, asked of the walk, which is the oracle the relations face."""
    return schema.is_valid(value)


def _admits(schema: Validator, value: Any) -> bool:
    """Membership, asked of the walk, which is the oracle the relations face."""
    return schema.is_valid(value)


def _witnesses(
    subject: Validator, supertype: Validator, corpus: list[Any]
) -> list[Any]:
    """Every value of the corpus in the subject and outside the supertype."""
    return [
        value
        for value in corpus
        if _admits(subject, value) and not _admits(supertype, value)
    ]


def _pair(row: tuple[str, Any, Any, str, list[Any]]) -> tuple[Validator, Validator]:
    return Validator(row[1]), Validator(row[2])


IDS = [row[0] for row in ROWS]


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_no_proof_is_refuted_by_a_value(
    row: tuple[str, Any, Any, str, list[Any]],
) -> None:
    """A `True` asserts every value of the subject is one of the supertype.

    One value the subject admits and the supertype refuses settles it, whatever
    rule reached the answer. This is the property the whole contract rests on and
    the one no other suite checks against values.
    """
    name, _subject, _supertype, _expected, corpus = row
    subject, supertype = _pair(row)
    found = _witnesses(subject, supertype, corpus)
    if found:
        assert not subject.is_subtype_of(supertype), (
            f"{name}: reported a subtype, and {found[0]!r} is in the subject "
            f"and outside the supertype"
        )


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_a_claimed_refutation_stands_on_a_value(
    row: tuple[str, Any, Any, str, list[Any]],
) -> None:
    """A `"not_subset"` claims a value of the subject lies outside the supertype.

    Where the row records an inclusion, no such value exists, so a refutation is
    false however conservative the procedure is entitled to be. `"undecided"` is
    always allowed; `"not_subset"` is a claim.
    """
    name, _subject, _supertype, expected, corpus = row
    if expected != "subset":
        return
    subject, supertype = _pair(row)
    assert not _witnesses(subject, supertype, corpus), f"{name}: the row is wrong"
    assert subject.relation_to(supertype) != "not_subset", (
        f"{name}: reported a refutation, and no value of the corpus is in the "
        f"subject and outside the supertype"
    )


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_a_recorded_refutation_has_its_witness(
    row: tuple[str, Any, Any, str, list[Any]],
) -> None:
    """A row recording `"not_subset"` carries the value that makes it one.

    The corpus is what a reader checks the row against, so a refutation nobody
    can witness is a row to rewrite rather than a fact to trust.
    """
    name, _subject, _supertype, expected, corpus = row
    if expected != "not_subset":
        return
    subject, supertype = _pair(row)
    assert _witnesses(subject, supertype, corpus), (
        f"{name}: records a refutation its own corpus cannot witness"
    )


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_the_relation_is_the_one_recorded(
    row: tuple[str, Any, Any, str, list[Any]],
) -> None:
    """The ratchet: a change in either direction is seen rather than absorbed."""
    name, _subject, _supertype, expected, _corpus = row
    subject, supertype = _pair(row)
    assert subject.relation_to(supertype) == expected, name


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_the_corpus_reaches_the_schemas_it_is_about(
    row: tuple[str, Any, Any, str, list[Any]],
) -> None:
    """A corpus neither schema admits anything from proves nothing about either.

    Two equal sets are never separated by a value, so separation is not the bar.
    Reaching them is: a row whose values are all refused by both sides would pass
    against any implementation at all.

    A row whose supertype is the bottom is the exception and is checked the other
    way round: what it claims is that the subject holds no value, so a corpus
    where none of the boundary values is in the subject is the evidence, and
    `test_a_claimed_refutation_stands_on_a_value` is where that is asserted.
    """
    name, _subject, _supertype, _expected, corpus = row
    subject, supertype = _pair(row)
    if supertype == Validator(nothing):
        return
    assert any(
        _admits(subject, value) or _admits(supertype, value) for value in corpus
    ), f"{name}: no value of the corpus is in either schema"


# The membership walk answers the same question the relations are checked
# against above, so a row here is a claim about the *oracle* rather than about a
# rule -- and the two readings of one schema must agree with each other before
# either can be an oracle for anything else.


def test_the_fast_and_explaining_walks_agree_at_the_depth_bound() -> None:
    """`is_valid` and `validate` are one answer, at every depth.

    The fast walk shortcuts a homogeneous list of a scalar kind and the
    explaining walk does not, so the level each element sits at is taken in one
    and skipped in the other unless the shortcut opens it. A fixpoint with four
    nodes per unfolding puts the innermost list at the walk's ceiling, which is
    where the two used to part.
    """
    deep = recursive(
        lambda node: union(
            Annotated[list[node], at.MinLen(1)],  # ty: ignore[invalid-type-form]
            Annotated[list[int], at.MinLen(1)],
        )
    )

    def nest(levels: int) -> Any:
        value: Any = 1
        for _ in range(levels):
            value = [value]
        return value

    for levels in (1, 8, 127, 128, 129):
        value = nest(levels)
        text = "[" * levels + "1" + "]" * levels
        fast = deep.is_valid(value)
        assert fast == (value in deep)
        assert fast == _raises_nothing(deep.validate, value)
        assert deep.is_valid_json(text) == _raises_nothing(deep.validate_json, text)


def _raises_nothing(check: Any, value: Any) -> bool:
    try:
        check(value)
    except ValidationError:
        return False
    return True


def test_a_record_resolves_a_key_the_way_the_dict_does() -> None:
    """A required field is present under the key the dict finds it under.

    A record is read two ways -- the declared keys probed, or the value's entries
    scanned -- and which one runs depends on whether a catch-all clause sits
    beside the field. They must answer alike, so neither may resolve a key by
    decoding its bytes: a `str` subclass carries a field's text without being
    that field.
    """
    closed = Validator({"a": int})
    open_record = Validator({"a": int, str: int})
    for record in (closed, open_record):
        assert record.is_valid({"a": 1})
        assert record.is_valid({PlainKey("a"): 1}), "a subclass the dict finds"
        assert not record.is_valid({LyingKey("a"): 1}), "a subclass it does not"
        assert not record.is_valid({"b": 1})


def test_a_multiple_is_a_remainder_of_zero() -> None:
    """`MultipleOf` is `value % operand == 0`, which is what the node denotes.

    Reading the remainder's truthiness instead asks a different question of any
    type whose `__bool__` and `__eq__` disagree.
    """
    multiples = Validator(Annotated[int, at.MultipleOf(3)])
    assert multiples.is_valid(9)
    assert multiples.is_valid(0)
    assert not multiples.is_valid(10)
    assert Validator(Annotated[float, at.MultipleOf(0.5)]).is_valid(1.5)
    assert not Validator(Annotated[float, at.MultipleOf(0.1)]).is_valid(0.3)

    span = datetime.timedelta
    remainder = span(seconds=6) % span(seconds=2)
    admitted = Validator(Annotated[span, at.MultipleOf(span(seconds=2))]).is_valid(
        span(seconds=6)
    )
    assert admitted == (remainder == 0)
