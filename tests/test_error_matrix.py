"""Every code the walk can report, in every mode that can report it.

`tests/test_error_codes.py` pins each code once, through `fail_fast=True`. That
is one cell of four. A report has two modes -- stop at the first violation, or
aggregate them -- and two entry paths -- a Python value, or a JSON document
parsed on the way in -- and a caller reading `errors` in aggregate mode over a
document is running a combination the suite had not.

So each row below names a code and is driven through all of them:

* **fail fast**, which promises exactly one violation;
* **aggregate**, which promises at least this one;
* **nested**, where the same failure sits inside a container and `loc` must name
  the way down to it rather than the root;
* **the JSON path**, for every code a parsed document can carry -- and a written
  reason for every code it cannot, beside the code a document gets instead, so
  the reason is falsifiable. It had to be: `validate_json` parses and then runs
  the *object* walk, so most codes a first reading calls unreachable are
  reached. Four of this table's reasons were wrong when it was written, and the
  column that names the other code is what said so.

The codes are derived from the tree by `tests/test_use_case_ledger.py`, which
holds this file's tables to that list in both directions: a code added to the
walk arrives here without a row and fails there. The derivation lives there
rather than here because it reads the Rust, and this file runs against an
installed wheel.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import annotated_types as at
import pytest

from valgebra import (
    Regex,
    ValidationError,
    Validator,
    complement,
    nothing,
    recursive,
    union,
)


class Case(NamedTuple):
    """One code, and the values that reach it on each path."""

    spec: object
    """A schema that reports this code for `value`."""
    value: object
    """A value outside `spec`, failing with this code at the root."""
    path: tuple[str | int, ...] = ()
    """The `loc` the root case carries. Empty unless the code is *about* a key
    or an attribute, which `missing_key`, `extra_forbidden` and
    `missing_attribute` are: each names the one it is about, wherever it is."""
    nested: tuple[object, object, tuple[str | int, ...]] | None = None
    """A schema, a value, and the `loc` the same failure carries inside it."""
    document: str | None = None
    """A JSON document that reaches the code through `spec`, where one can."""
    no_json: str = ""
    """Why no document reaches it, where none can. One of the two is required."""
    instead: tuple[str, str] | None = None
    """A document, and the code it gets instead, where the reason above says a
    document cannot reach this one. Written so the reason is falsifiable: the
    first draft claimed a document reaches `tuple_length` and not `tuple_type`,
    which is the reverse of what the parser does, and nothing would have said
    so."""


class _Color(enum.Enum):
    RED = 1


@dataclass
class _Point:
    x: int
    y: int


class _Unset(_Point):
    """An instance of the class with the field never assigned."""

    def __init__(self) -> None:
        pass


def _raises(value: object) -> bool:
    message = "a predicate that will not answer"
    raise RuntimeError(message)


#: Every code the walk reports, with what reaches it.
#:
#: A row is the code's own shape rather than a convenient one: the JSON column
#: reaches the code *through the parser*, so a document is written as text and
#: not as a Python value handed to the object walk.
CASES: dict[str, Case] = {
    "none_type": Case(None, 1, nested=({"a": None}, {"a": 1}, ("a",)), document="1"),
    "bool_type": Case(bool, 1, nested=({"a": bool}, {"a": 1}, ("a",)), document="1"),
    "int_type": Case(int, "x", nested=({"a": int}, {"a": "x"}, ("a",)), document='"x"'),
    "float_type": Case(float, 1, nested=({"a": float}, {"a": 1}, ("a",)), document="1"),
    "string_type": Case(str, 1, nested=({"a": str}, {"a": 1}, ("a",)), document="1"),
    "bytes_type": Case(
        bytes,
        "x",
        nested=({"a": bytes}, {"a": "x"}, ("a",)),
        document='"x"',
    ),
    "list_type": Case(
        list[int], "x", nested=({"a": list[int]}, {"a": 1}, ("a",)), document='"x"'
    ),
    "dict_type": Case(
        dict[str, int],
        [],
        nested=({"a": dict[str, int]}, {"a": []}, ("a",)),
        document="[]",
    ),
    "tuple_type": Case(
        tuple[int, str],
        {1: "a"},
        nested=({"a": tuple[int, str]}, {"a": 1}, ("a",)),
        document="[1]",
    ),
    "set_type": Case(
        set[int],
        [1],
        nested=({"a": set[int]}, {"a": [1]}, ("a",)),
        document="[1]",
    ),
    "frozen_set_type": Case(
        frozenset[int],
        {1},
        nested=({"a": frozenset[int]}, {"a": [1]}, ("a",)),
        document="[1]",
    ),
    "literal_error": Case(
        "active",
        "paused",
        nested=({"a": "active"}, {"a": "paused"}, ("a",)),
        document='"paused"',
    ),
    "no_match": Case(
        nothing, 1, nested=({"a": nothing}, {"a": 1}, ("a",)), document="1"
    ),
    "list_length": Case(
        Validator([int, int]),
        [1],
        nested=({"a": Validator([int, int])}, {"a": [1]}, ("a",)),
        document="[1]",
    ),
    "tuple_length": Case(
        tuple[int, int],
        (1,),
        nested=({"a": tuple[int, int]}, {"a": (1,)}, ("a",)),
        no_json="a document's array is a list and never a tuple, so a tuple "
        "schema refuses one by kind before it counts the elements: what a "
        "document reaches is `tuple_type`",
        instead=("[1]", "tuple_type"),
    ),
    "missing_key": Case(
        {"a": int},
        {},
        path=("a",),
        nested=({"outer": {"a": int}}, {"outer": {}}, ("outer", "a")),
        document="{}",
    ),
    "extra_forbidden": Case(
        {"a": int},
        {"a": 1, "b": 2},
        path=("b",),
        nested=({"outer": {"a": int}}, {"outer": {"a": 1, "b": 2}}, ("outer", "b")),
        document='{"a": 1, "b": 2}',
    ),
    "instance_type": Case(
        _Color,
        1,
        nested=({"a": _Color}, {"a": 1}, ("a",)),
        document="1",
    ),
    "missing_attribute": Case(
        _Point,
        _Unset(),
        path=("x",),
        nested=({"a": _Point}, {"a": _Unset()}, ("a", "x")),
        no_json="the attribute record is met with the class, and the parser "
        "builds no instance of one: every document is refused by the class "
        "beside it before any attribute is asked for",
        instead=('{"x": 1, "y": 2}', "instance_type"),
    ),
    "unexpected_match": Case(
        complement(int),
        5,
        nested=({"a": complement(int)}, {"a": 5}, ("a",)),
        document="5",
    ),
    "union_error": Case(
        union(int, str),
        1.5,
        nested=({"a": union(int, str)}, {"a": 1.5}, ("a",)),
        document="1.5",
    ),
    "greater_than_equal": Case(
        Annotated[int, at.Ge(0)],
        -1,
        nested=({"a": Annotated[int, at.Ge(0)]}, {"a": -1}, ("a",)),
        document="-1",
    ),
    "greater_than": Case(
        Annotated[int, at.Gt(0)],
        0,
        nested=({"a": Annotated[int, at.Gt(0)]}, {"a": 0}, ("a",)),
        document="0",
    ),
    "less_than_equal": Case(
        Annotated[int, at.Le(0)],
        1,
        nested=({"a": Annotated[int, at.Le(0)]}, {"a": 1}, ("a",)),
        document="1",
    ),
    "less_than": Case(
        Annotated[int, at.Lt(0)],
        0,
        nested=({"a": Annotated[int, at.Lt(0)]}, {"a": 0}, ("a",)),
        document="0",
    ),
    "too_short": Case(
        Annotated[str, at.MinLen(2)],
        "a",
        nested=({"a": Annotated[str, at.MinLen(2)]}, {"a": "a"}, ("a",)),
        document='"a"',
    ),
    "too_long": Case(
        Annotated[str, at.MaxLen(1)],
        "ab",
        nested=({"a": Annotated[str, at.MaxLen(1)]}, {"a": "ab"}, ("a",)),
        document='"ab"',
    ),
    "multiple_of": Case(
        Annotated[int, at.MultipleOf(3)],
        5,
        nested=({"a": Annotated[int, at.MultipleOf(3)]}, {"a": 5}, ("a",)),
        document="5",
    ),
    "string_pattern_mismatch": Case(
        Annotated[str, Regex(r"\\d+")],
        "ab",
        nested=({"a": Annotated[str, Regex(r"\\d+")]}, {"a": "ab"}, ("a",)),
        document='"ab"',
    ),
    "predicate_failed": Case(
        Annotated[int, at.Predicate(lambda v: v > 0)],
        -1,
        nested=(
            {"a": Annotated[int, at.Predicate(lambda v: v > 0)]},
            {"a": -1},
            ("a",),
        ),
        document="-1",
    ),
    "predicate_error": Case(
        Annotated[int, at.Predicate(_raises)],
        1,
        nested=({"a": Annotated[int, at.Predicate(_raises)]}, {"a": 1}, ("a",)),
        document="1",
    ),
}

#: Codes whose shape is not a value beside a schema, with the test that holds
#: each. A deep value is built by a loop, a cyclic one refers to itself, a
#: mutating one runs Python while it is read, and a malformed document never
#: becomes a value at all -- none of them is a row in a table, and each is
#: named here so the ledger below can tell "handled elsewhere" from "missing".
ELSEWHERE: dict[str, str] = {
    "recursion_limit": "test_a_value_past_the_depth_bound_reports_the_bound",
    "recursion_loop": "test_a_value_containing_itself_reports_the_cycle",
    "mutated_during_validation": "test_a_value_that_moves_under_the_walk_is_reported",
    "json_invalid": "test_a_document_the_parser_refuses_reports_the_parse",
}


def _fail_fast(spec: object, value: object) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        Validator(spec).validate(value, fail_fast=True)
    return caught.value


def _aggregate(spec: object, value: object) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        Validator(spec).validate(value)
    return caught.value


def _from_json(spec: object, document: str) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        Validator(spec).validate_json(document)
    return caught.value


def _codes_of(error: ValidationError) -> list[str]:
    return [str(entry["code"]) for entry in error.errors]


@pytest.mark.parametrize("code", sorted(CASES))
def test_a_code_is_reported_in_both_modes(code: str) -> None:
    """Stopping at the first violation and aggregating report the same code.

    The modes differ in how many violations come back, never in what the
    failure *is*: a caller who switches to `fail_fast` to get one error must
    get the one the aggregate would have led with.
    """
    case = CASES[code]
    first = _fail_fast(case.spec, case.value)
    assert first.code == code
    assert first.path == case.path
    assert len(first.errors) == 1

    every = _aggregate(case.spec, case.value)
    assert code in _codes_of(every), _codes_of(every)
    assert every.code == code


@pytest.mark.parametrize("code", sorted(CASES))
def test_a_nested_failure_names_the_way_down_to_it(code: str) -> None:
    """The same failure inside a container carries the path to it.

    A `loc` of `()` for a failure one level in sends a reader to the whole
    value, which is the report saying the container is wrong when a field is.
    """
    case = CASES[code]
    if case.nested is None:
        pytest.skip(f"{code} is driven by a test of its own")
    spec, value, where = case.nested
    error = _fail_fast(spec, value)
    assert error.code == code
    assert error.path == where
    # The aggregating mode reports the same location for the same failure.
    aggregated = _aggregate(spec, value)
    assert where in [tuple(entry["path"]) for entry in aggregated.errors]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("code", sorted(CASES))
def test_a_code_a_document_can_carry_is_reported_from_one(code: str) -> None:
    """Every code a parsed document can reach is reached through the parser.

    A row with no document carries the reason the grammar cannot produce one,
    because "JSON cannot reach it" is a claim about the grammar rather than a
    gap, and an unwritten one is indistinguishable from a gap.
    """
    case = CASES[code]
    if case.document is None:
        assert len(case.no_json) > 40, f"{code}: {case.no_json!r}"
        if case.instead is not None:
            document, other = case.instead
            assert other != code
            assert _from_json(case.spec, document).code == other
        return
    error = _from_json(case.spec, case.document)
    assert error.code == code
    assert error.path == case.path


def test_a_value_past_the_depth_bound_reports_the_bound() -> None:
    """A value nested deeper than the walk descends reports the bound.

    Both modes and both paths: a document nests as a Python value does, and
    the bound is the walk's rather than the parser's.
    """
    deep = recursive(lambda s: union(int, [s]))
    value: object = 0
    for _ in range(200):
        value = [value]
    assert _fail_fast(deep, value).code == "recursion_limit"
    assert "recursion_limit" in _codes_of(_aggregate(deep, value))
    assert _from_json(deep, json.dumps(value)).code == "recursion_limit"


def test_a_value_containing_itself_reports_the_cycle() -> None:
    """A self-containing value is reported rather than walked forever."""
    deep = recursive(lambda s: union(int, [s]))
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert _fail_fast(deep, cyclic).code == "recursion_loop"
    assert "recursion_loop" in _codes_of(_aggregate(deep, cyclic))


def test_a_value_that_moves_under_the_walk_is_reported() -> None:
    """A container that resizes while it is read is reported, not answered for.

    The walk reads a dict by the count it began with, so a value that grows
    under it has not been read. Reporting a verdict would be answering about a
    value that no longer exists. The key the predicate adds is one the record
    *declares*, so the report is the move rather than an undeclared key --
    which is the case that separates the two.
    """
    moved: dict[str, int] = {"a": 1, "b": 2}

    def grow(_: object) -> bool:
        moved.setdefault("c", 3)
        return True

    schema = Validator({"a": Annotated[int, at.Predicate(grow)], "b": int, "c?": int})
    assert schema.is_valid(moved) is False

    moved = {"a": 1, "b": 2}
    error = _fail_fast(schema, moved)
    assert error.code == "mutated_during_validation"
    # And the aggregating mode reports it too: a value that moved is not a
    # failure the walk can look past to find another.
    moved = {"a": 1, "b": 2}
    assert "mutated_during_validation" in _codes_of(_aggregate(schema, moved))


def test_a_document_the_parser_refuses_reports_the_parse() -> None:
    """A document that is not JSON fails before any schema is consulted."""
    error = _from_json(int, "{ not json")
    assert error.code == "json_invalid"
    assert error.path == ()
    # And the same in aggregate mode: a parse failure is one failure, whatever
    # the caller asked for, because there is no value to find a second in.
    with pytest.raises(ValidationError) as caught:
        Validator(int).validate_json("{ not json", fail_fast=True)
    assert caught.value.code == "json_invalid"
