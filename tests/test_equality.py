"""Two spellings of one schema are one schema.

`==` is the question "are these the same schema", and until the constructors
settled it the answer was "were these built the same way": a record's fields kept
the order they were written in, a map's clauses theirs, and a literal or a
refinement marker kept the *pool slot* its constant landed in, which is the order
the constants were first seen. So `{"a": int, "b": int}` and `{"b": int, "a":
int}` -- one set, by every other measure the library offers -- compared unequal,
hashed apart, and stayed two members of a union that folds repeats.

Two mechanisms fix it and this holds both. Construction puts a record's fields, a
map's clauses and a refinement's constraints in a canonical order, so the term
itself stops carrying the spelling. And `==` reads *through* a pool slot to the
constant it names, matching a union's members and a refinement's constraints as
the sets they are -- because a slot is construction order and cannot be sorted
away.

What this is not is a decision. `int | ~int` and `anything` denote one set and
are not equal here; `is_equivalent` is the question for that, and the pair below
that asserts it stays separate is the one that says so.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import annotated_types as at
import pytest

from valgebra import (
    Regex,
    Validator,
    anything,
    intersection,
    nothing,
    recursive,
    union,
)

# Pairs that are one schema written two ways. Each row is a set, and the two
# spellings differ only in an order that is not part of it.
SAME = [
    ("record fields", {"a": int, "b": str}, {"b": str, "a": int}),
    ("optional fields", {"a?": int, "b": str}, {"b": str, "a?": int}),
    ("nested records", {"x": {"a": int, "b": int}}, {"x": {"b": int, "a": int}}),
    ("map clauses", {str: int, int: str}, {int: str, str: int}),
    ("record and clause", {"a": int, str: int}, {str: int, "a": int}),
    ("literal members", Literal[1, 2], Literal[2, 1]),
    ("literal members, strings", Literal["a", "b", "c"], Literal["c", "b", "a"]),
    ("union members", int | str, str | int),
    (
        "refinement markers",
        Annotated[int, at.Ge(0), at.Le(9)],
        Annotated[int, at.Le(9), at.Ge(0)],
    ),
    (
        "three markers",
        Annotated[str, at.MinLen(1), at.MaxLen(9), Regex("a+")],
        Annotated[str, Regex("a+"), at.MaxLen(9), at.MinLen(1)],
    ),
    (
        "a duplicated marker",
        Annotated[int, at.Ge(0), at.Ge(0)],
        Annotated[int, at.Ge(0)],
    ),
    (
        "records inside a union",
        union({"a": int, "b": int}, str),
        union(str, {"b": int, "a": int}),
    ),
    (
        "records inside a list",
        [{"a": int, "b": int}, ...],
        [{"b": int, "a": int}, ...],
    ),
]

# Pairs that are *not* one schema, so the reading above cannot be doing it by
# saying yes to everything. Each differs in something a set depends on.
DIFFERENT = [
    ("different constants", Literal[1], Literal[2]),
    ("a different member", Literal[1, 2], Literal[1, 3]),
    ("a literal is typed", Literal[1], Literal[True]),
    # A float is not a `Literal` argument the typing spec allows, and is a
    # constant this library pools like any other: the pair is the point.
    ("a literal is typed, floats", Literal[1], Literal[1.0]),  # ty: ignore[invalid-type-form]
    ("a field's type", {"a": int}, {"a": str}),
    ("a missing field", {"a": int, "b": int}, {"a": int}),
    ("required against optional", {"a": int}, {"a?": int}),
    ("open against closed", {"a": int}, {"a": int, anything: anything}),
    ("the bound's direction", Annotated[int, at.Ge(0)], Annotated[int, at.Le(0)]),
    ("the bound's value", Annotated[int, at.Ge(0)], Annotated[int, at.Ge(1)]),
    ("a sequence is ordered", tuple[int, str], tuple[str, int]),
    ("the container", list[int], set[int]),
    ("a clause's value", {str: int}, {str: str}),
]


@pytest.mark.parametrize(("name", "left", "right"), SAME, ids=[row[0] for row in SAME])
def test_one_set_written_two_ways_is_one_schema(
    name: str, left: object, right: object
) -> None:
    first, second = Validator(left), Validator(right)
    assert first == second, name
    assert hash(first) == hash(second), f"{name}: equal validators must hash alike"
    # And the union of the pair is one member, because the fold that drops a
    # repeat compares terms: this is the consequence that made the equality
    # worth fixing rather than the equality itself.
    assert union(first, second) == first, name


@pytest.mark.parametrize(
    ("name", "left", "right"), DIFFERENT, ids=[row[0] for row in DIFFERENT]
)
def test_two_sets_stay_two_schemas(name: str, left: object, right: object) -> None:
    assert Validator(left) != Validator(right), name


def test_a_validator_equals_itself_over_a_value_that_equals_nothing() -> None:
    """Identity before value, or a `nan` literal would not equal itself.

    The literals are built through a variable rather than written out: a float
    is not an argument the typing spec allows `Literal` to take, and a checker
    reading the annotation says so. This library pools the constant either way,
    which is what the case is about.
    """
    quiet = math.nan
    nan = Validator(Literal[quiet])  # ty: ignore[invalid-type-form]
    assert nan == nan  # noqa: PLR0124 - the identity is the subject
    # `math.nan` is one object, so a second validator over it pools the same
    # constant and the two are one schema.
    assert Validator(Literal[quiet]) == nan  # ty: ignore[invalid-type-form]
    # Two *different* nan objects are two constants: a nan is equal to no value,
    # itself included, so nothing but identity can join them.
    first, second = float("nan"), float("nan")
    assert Validator(Literal[first]) != Validator(  # ty: ignore[invalid-type-form]
        Literal[second]  # ty: ignore[invalid-type-form]
    )


def test_equality_is_not_a_decision() -> None:
    """Sets a theorem identifies are not equal terms, and must not become so.

    `==` is what the constructors settle; `is_equivalent` is what the decision
    procedures prove. Blurring them would make a cheap syntactic check look like
    an expensive semantic one, and the pairs below are the ones a reader most
    expects to be blurred.
    """
    pairs = [
        # Not `int | ~int`, which the constructor folds to the top: the
        # complement laws are settled where the schema is built, and these are
        # the ones a *theorem* identifies rather than a fold.
        (intersection(int, str), Validator(nothing)),
        (Validator(bool), Validator(Literal[True, False])),
        (union(int, Literal[1]), Validator(int)),
    ]
    for left, right in pairs:
        assert left.is_equivalent(right), "the pair is one set"
        assert left != right, "and two terms"


def test_a_recursive_schema_equals_the_same_one_built_again() -> None:
    # The fixpoint is a definition and a reference, and both sides build their
    # own; equality compares the definitions pairwise rather than by identity.
    json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    again = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    assert json == again
    assert hash(json) == hash(again)
    # A different body is a different schema, however alike the reference looks.
    assert json != recursive(lambda j: union(None, bool, int, str, [j], {str: j}))


def test_a_validator_is_usable_as_a_dictionary_key() -> None:
    # The point of the hash: a registry of contracts holds one entry per schema,
    # and the spelling a caller happened to use does not make a second.
    registry = {Validator({"a": int, "b": str}): "row"}
    registry[Validator({"b": str, "a": int})] = "row again"
    assert len(registry) == 1
    assert registry[Validator({"a": int, "b": str})] == "row again"
