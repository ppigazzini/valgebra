"""The in-place JSON walk against every node kind, versus the object path.

The JSON fast path validates a parsed JSON value without materializing Python
objects for the structure it walks, but nodes that compare against a Python
object (literals, refinements, instance and object checks) materialize the value
at that node. Both must reach the same decision as validating ``json.loads`` of
the same document; every case asserts that equivalence and the expected verdict.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from functools import reduce
from types import GenericAlias
from typing import Annotated, Literal

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Validator,
    complement,
    intersection,
    recursive,
    union,
)

_BIG = 123456789012345678901234567890
# Valid at runtime but not as a static type expression; build them off the
# type-checked path.
_FLOAT_LIT = Literal[1.0]  # ty: ignore[invalid-type-form]
_BIGNUM_LIT = Literal[_BIG]  # ty: ignore[invalid-type-form]


class _Color(enum.Enum):
    RED = 1
    GREEN = 2


@dataclass
class _Point:
    x: int
    y: int


# (label, schema spec, JSON document, expected membership).
CASES: list[tuple[str, object, str, bool]] = [
    # literals materialize the scalar at the node
    ("lit-str", Literal["a"], '"a"', True),
    ("lit-str-no", Literal["a"], '"b"', False),
    ("lit-int", Literal[1], "1", True),
    ("lit-int-vs-bool", Literal[1], "true", False),
    ("lit-bool", Literal[True], "true", True),
    ("lit-float", _FLOAT_LIT, "1.0", True),
    ("lit-float-vs-int", _FLOAT_LIT, "1", False),
    ("lit-bignum", _BIGNUM_LIT, str(_BIG), True),
    # refinements materialize the value, including container length
    ("ge", Annotated[int, at.Ge(0)], "5", True),
    ("ge-no", Annotated[int, at.Ge(0)], "-1", False),
    ("minlen-str", Annotated[str, at.MinLen(2)], '"ab"', True),
    ("minlen-str-no", Annotated[str, at.MinLen(2)], '"a"', False),
    ("minlen-list", Annotated[list[int], at.MinLen(1)], "[1]", True),
    ("maxlen-list", Annotated[list[int], at.MaxLen(1)], "[1, 2]", False),
    ("multiple", Annotated[int, at.MultipleOf(3)], "9", True),
    ("multiple-no", Annotated[int, at.MultipleOf(3)], "5", False),
    # the big-integer path
    ("int-bignum", int, str(_BIG), True),
    ("float-vs-bignum", float, str(_BIG), False),
    # instance/object never match a JSON value (it materializes to a builtin)
    ("enum-no", _Color, "1", False),
    ("dataclass-no", _Point, '{"x": 1, "y": 2}', False),
    # bytes, tuples, sets, and frozensets have no JSON form: never members
    ("bytes-no", bytes, '"x"', False),
    ("tuple-no", tuple[int, int], "[1, 2]", False),
    ("set-no", set[int], "[1, 2]", False),
    ("frozenset-no", frozenset[int], "[1]", False),
    # combinators over JSON
    ("union", union(int, str), "1", True),
    ("union-no", union(int, str), "1.5", False),
    ("complement", complement(int), '"x"', True),
    ("intersection", intersection(int, complement(bool)), "5", True),
    ("intersection-bool", intersection(int, complement(bool)), "true", False),
    # nested structure walked in place
    ("nested", list[dict[str, int]], '[{"a": 1}, {"b": 2}]', True),
    ("nested-no", list[dict[str, int]], '[{"a": "x"}]', False),
    ("record", {"name": str, "age?": int}, '{"name": "Ada", "age": 36}', True),
    ("record-extra", {"name": str}, '{"name": "Ada", "x": 1}', False),
    ("mapping", dict[str, int], '{"a": 1, "b": 2}', True),
]


@pytest.mark.parametrize(
    ("label", "spec", "doc", "expected"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_json_matches_object_path_and_expected(
    label: str,
    spec: object,
    doc: str,
    expected: bool,  # noqa: FBT001
) -> None:
    v = Validator(spec)
    assert v.is_valid_json(doc) is expected
    assert v.is_valid_json(doc) == v.is_valid(json.loads(doc))


def test_recursive_schema_over_json_in_place() -> None:
    json_value = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    doc = '{"a": [1, "x", {"b": null}], "c": [true, 3.5]}'
    assert json_value.is_valid_json(doc)
    assert json_value.is_valid_json(doc) == json_value.is_valid(json.loads(doc))


def test_recursive_tree_over_json() -> None:
    tree = recursive(lambda t: {"value": int, "left?": t, "right?": t})
    assert tree.is_valid_json('{"value": 1, "left": {"value": 2}}')
    assert not tree.is_valid_json('{"value": 1, "left": {"value": "x"}}')


def test_deeply_nested_json_recursion_is_bounded() -> None:
    # A recursive schema over a very deep document fails cleanly (depth guard),
    # never overflowing the native stack.
    deep = recursive(lambda s: union(int, [s]))
    doc = "[" * 200 + "1" + "]" * 200
    assert deep.is_valid_json(doc) is False
    assert deep.is_valid_json(doc) == deep.is_valid(json.loads(doc))


def test_duplicate_object_keys_keep_the_last_value() -> None:
    # json.loads keeps the last value for a duplicate key; the in-place walk must
    # agree, for both records and mappings.
    for spec, doc in [
        (Validator({"a": int}), '{"a": "x", "a": 1}'),
        (Validator({"a": int}), '{"a": 1, "a": "x"}'),
        (Validator(dict[str, int]), '{"k": "x", "k": 1}'),
        (Validator(dict[str, int]), '{"k": 1, "k": "x"}'),
    ]:
        assert spec.is_valid_json(doc) == spec.is_valid(json.loads(doc))


def test_malformed_and_wrong_type_json_are_not_members() -> None:
    assert Validator(int).is_valid_json("not json") is False
    assert Validator(int).is_valid_json("") is False
    assert Validator(int).is_valid_json(b"5") is True


def test_is_valid_json_rejects_non_string_input() -> None:
    # Neither str nor bytes: not a member, and never raises.
    assert Validator(int).is_valid_json(123) is False  # ty: ignore[invalid-argument-type]


def test_fixed_length_list_over_json() -> None:
    v = Validator([int, str])
    assert v.is_valid_json('[1, "a"]') is True
    assert v.is_valid_json("[1, 2]") is False
    assert v.is_valid_json("[1]") is False  # wrong length


def test_open_record_over_json_admits_extra_keys() -> None:
    v = Validator({"a": int}).open()
    assert v.is_valid_json('{"a": 1, "b": 2}') is True
    assert v.is_valid_json('{"a": "x"}') is False  # declared value still checked


def test_json_null_materializes_for_a_predicate() -> None:
    # A predicate over a JSON null forces materialization to None.
    v = Validator(Annotated[object, at.Predicate(lambda x: x is None)])
    assert v.is_valid_json("null") is True
    assert v.is_valid_json("1") is False


def test_mapping_and_record_reject_a_non_object_json() -> None:
    # A dict-shaped schema against a JSON array (or scalar) is not a member.
    assert Validator(dict[str, int]).is_valid_json("[1, 2]") is False
    assert Validator({"a": int}).is_valid_json("[1]") is False
    assert Validator({"a": int}).is_valid_json("5") is False


# --- A fuzzer pinning the in-place JSON walk to the object walk -------------
# Both must reach the same verdict for every JSON-meaningful schema and every
# JSON document: is_valid_json walks the parsed value in place, is_valid walks
# the object json.loads produces. Independent schema and value drive both the
# type-mismatch and the structural-recursion paths.


def _json_schemas() -> st.SearchStrategy[object]:
    leaf = st.sampled_from([int, float, str, bool, type(None)])
    refined = st.one_of(
        st.integers(min_value=-5, max_value=5).map(lambda k: Annotated[int, at.Ge(k)]),
        st.integers(min_value=0, max_value=5).map(
            lambda k: Annotated[str, at.MinLen(k)]
        ),
    )
    return st.recursive(
        st.one_of(leaf, refined),
        lambda child: st.one_of(
            child.map(lambda t: GenericAlias(list, (t,))),  # list[T]
            child.map(lambda t: GenericAlias(dict, (str, t))),  # dict[str, T]
            st.tuples(child, child).map(lambda ab: [ab[0], ab[1], ...]),  # [A, B, ...]
            st.lists(leaf, min_size=2, max_size=3, unique=True).map(
                lambda ts: reduce(lambda a, b: a | b, ts)
            ),  # a union of scalars
            st.lists(child, min_size=1, max_size=2).map(
                lambda cs: {f"f{i}": c for i, c in enumerate(cs)}
            ),  # a closed record
            st.tuples(child, child).map(
                lambda ab: {str: ab[0], int: ab[1]}
            ),  # a heterogeneous map (over JSON only the str clause can match)
        ),
        max_leaves=4,
    )


def _json_values() -> st.SearchStrategy[object]:
    return st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-8, max_value=8),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=3),
        ),
        lambda child: st.one_of(
            st.lists(child, max_size=4),
            st.dictionaries(st.text(max_size=3), child, max_size=3),
        ),
        max_leaves=6,
    )


@given(spec=_json_schemas(), value=_json_values())
def test_json_walk_matches_the_object_walk(spec: object, value: object) -> None:
    schema = Validator(spec)
    doc = json.dumps(value)
    assert schema.is_valid_json(doc) == schema.is_valid(json.loads(doc))


def test_fixed_and_variadic_sequences_reject_wrong_json_shapes() -> None:
    # A fixed-length list schema rejects a non-array JSON; a tuple (fixed or
    # variadic) never matches JSON, which has no tuples.
    assert Validator([int, str]).is_valid_json("{}") is False
    assert Validator([int, str]).is_valid_json("5") is False
    assert Validator(tuple[int, ...]).is_valid_json("[1, 2]") is False
