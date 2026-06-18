"""Every error code the explain walk can emit, with its code and path.

The explain walk (`validate`) reports a machine-readable code and a path per
failure. This pins the code and the location for each node kind, so the error
contract is locked and the reporting branches are exercised.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Annotated

import annotated_types as at
import pytest

from valgebra import (
    ValidationError,
    Validator,
    complement,
    fixed_sequence,
    intersection,
    nothing,
    recursive,
    union,
)


def _first(spec: object, value: object) -> tuple[str, tuple[str | int, ...]]:
    with pytest.raises(ValidationError) as info:
        Validator(spec).validate(value, fail_fast=True)
    return info.value.code, info.value.path


class _Color(enum.Enum):
    RED = 1


@dataclass
class _Point:
    x: int
    y: int


def test_scalar_codes() -> None:
    assert _first(None, 1) == ("none_type", ())
    assert _first(bool, 1) == ("bool_type", ())
    assert _first(int, "x") == ("int_type", ())
    assert _first(float, 1) == ("float_type", ())
    assert _first(str, 1) == ("string_type", ())
    assert _first(bytes, "x") == ("bytes_type", ())


def test_bottom_and_literal_codes() -> None:
    assert _first(nothing, 1) == ("no_match", ())
    assert _first("active", "paused") == ("literal_value", ())


def test_collection_type_codes() -> None:
    assert _first(list[int], "x") == ("list_type", ())
    assert _first(set[int], [1]) == ("set_type", ())
    assert _first(frozenset[int], {1}) == ("frozenset_type", ())
    assert _first(tuple[int, str], [1, "a"]) == ("tuple_type", ())
    assert _first(dict[str, int], []) == ("dict_type", ())


def test_length_codes() -> None:
    # A fixed-length list and a fixed tuple report a length mismatch.
    assert _first(fixed_sequence(int, int), [1]) == ("list_length", ())
    assert _first(tuple[int, int], (1,)) == ("tuple_length", ())


def test_record_codes_and_paths() -> None:
    assert _first({"a": int}, {}) == ("missing_key", ("a",))
    assert _first({"a": int}, {"a": 1, "b": 2}) == ("extra_key", ("b",))
    assert _first({"a": int}, {"a": "x"}) == ("int_type", ("a",))


def test_class_codes_and_paths() -> None:
    assert _first(_Color, 1) == ("instance_type", ())
    assert _first(_Point, {"x": 1, "y": 2}) == ("instance_type", ())
    assert _first(_Point, _Point(1, "y")) == ("int_type", ("y",))  # ty: ignore[invalid-argument-type]


def test_missing_attribute_code() -> None:
    @dataclass
    class Has:
        a: int

    class Lacks:
        pass

    assert Validator(Has).is_valid(Lacks()) is False
    code, _ = _first(Has, Lacks())
    assert code in {"instance_type", "missing_attribute"}


def test_combinator_codes() -> None:
    assert _first(complement(int), 5) == ("unexpected_match", ())
    assert _first(union(int, str), 1.5) == ("union_error", ())
    # An intersection reports the member that failed.
    assert _first(intersection(int, complement(bool)), True) == ("unexpected_match", ())


def test_nested_path_reporting() -> None:
    code, path = _first({"items": [dict[str, int]]}, {"items": [{"a": "x"}]})
    assert code == "int_type"
    assert path == ("items", 0, "a")


def test_constraint_codes() -> None:
    assert _first(Annotated[int, at.Ge(0)], -1) == ("greater_than_equal", ())
    assert _first(Annotated[int, at.Gt(0)], 0) == ("greater_than", ())
    assert _first(Annotated[int, at.Le(0)], 1) == ("less_than_equal", ())
    assert _first(Annotated[int, at.Lt(0)], 0) == ("less_than", ())
    assert _first(Annotated[str, at.MinLen(2)], "a") == ("too_short", ())
    assert _first(Annotated[str, at.MaxLen(1)], "ab") == ("too_long", ())
    assert _first(Annotated[int, at.MultipleOf(3)], 5) == ("not_multiple_of", ())


def test_recursion_codes() -> None:
    deep = recursive(lambda s: union(int, [s]))
    # Depth bound: a value nested past the limit fails with recursion_limit.
    value: object = 0
    for _ in range(200):
        value = [value]
    assert _first(deep, value)[0] == "recursion_limit"

    # A self-containing value fails with recursion_loop rather than looping.
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert _first(deep, cyclic)[0] == "recursion_loop"


def test_json_invalid_code() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int).validate_json("{ not json")
    assert info.value.code == "json_invalid"
    assert info.value.path == ()
