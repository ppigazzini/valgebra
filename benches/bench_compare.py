"""Comparison benchmarks: valgebra against pydantic-core and jsonschema.

The same synthetic shapes run through three checkers so the recorded baseline in
``docs/performance.md`` is reproducible. The three do not do identical work, and
the docs state the caveats:

- valgebra checks membership of the object already in hand: no copy, no coercion.
- jsonschema (``Draft202012Validator.is_valid``) is also a pure check with no
  coercion, the closest semantic analogue, but it is pure Python.
- pydantic (``TypeAdapter.validate_python`` in strict mode) validates and
  *constructs*; strict mode disables coercion, but it still builds and returns a
  value, so it does strictly more work than a membership check.

Run with ``pytest benches/bench_compare.py --benchmark-group-by=group`` after
installing the ``bench`` dependency group.
"""

from __future__ import annotations

from typing import TypedDict

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

from valgebra import validator

ARRAY_LEN = 10_000
RECORD_WIDTH = 50
NESTING_DEPTH = 25

LIBRARIES = ["valgebra", "pydantic", "jsonschema"]


def _nested_list_type(depth: int) -> object:
    schema: object = int
    for _ in range(depth):
        schema = list[schema]  # type: ignore[valid-type]
    return schema


def _nested_list_value(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = [value]
    return value


def _nested_json_schema(depth: int) -> dict:
    schema: dict = {"type": "integer"}
    for _ in range(depth):
        schema = {"type": "array", "items": schema}
    return schema


def make_large_array(lib: str) -> tuple[object, object]:
    data = list(range(ARRAY_LEN))
    if lib == "valgebra":
        return validator(list[int]).is_valid, data
    if lib == "pydantic":
        adapter = TypeAdapter(list[int])
        return lambda d: adapter.validate_python(d, strict=True), data
    schema = {"type": "array", "items": {"type": "integer"}}
    return Draft202012Validator(schema).is_valid, data


def make_wide_record(lib: str) -> tuple[object, object]:
    data = {f"f{i}": i for i in range(RECORD_WIDTH)}
    fields = {f"f{i}": int for i in range(RECORD_WIDTH)}
    if lib == "valgebra":
        return validator(fields).is_valid, data
    if lib == "pydantic":
        # A named, closed record's analogue is a TypedDict, not dict[str, int]
        # (which is valgebra's Mapping). Build one with the same fields.
        record = TypedDict("Wide", fields)
        adapter = TypeAdapter(record)
        return lambda d: adapter.validate_python(d, strict=True), data
    schema = {
        "type": "object",
        "properties": {f"f{i}": {"type": "integer"} for i in range(RECORD_WIDTH)},
        "required": [f"f{i}" for i in range(RECORD_WIDTH)],
        "additionalProperties": False,
    }
    return Draft202012Validator(schema).is_valid, data


def make_deep_nesting(lib: str) -> tuple[object, object]:
    data = _nested_list_value(NESTING_DEPTH)
    if lib == "valgebra":
        return validator(_nested_list_type(NESTING_DEPTH)).is_valid, data
    if lib == "pydantic":
        adapter = TypeAdapter(_nested_list_type(NESTING_DEPTH))
        return lambda d: adapter.validate_python(d, strict=True), data
    return Draft202012Validator(_nested_json_schema(NESTING_DEPTH)).is_valid, data


SHAPES = {
    "large_array": make_large_array,
    "wide_record": make_wide_record,
    "deep_nesting": make_deep_nesting,
}


@pytest.mark.parametrize("lib", LIBRARIES)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_compare(benchmark: object, shape: str, lib: str) -> None:
    benchmark.group = shape  # type: ignore[attr-defined]
    check, data = SHAPES[shape](lib)
    benchmark(check, data)
