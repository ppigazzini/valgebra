"""JSON-path benchmarks: validate_json versus parse-then-validate.

The exit criterion for the JSON path is that it is measurably faster than
parsing with the standard library and then validating. Each shape is timed three
ways on the same JSON document:

- **valgebra-json**: ``is_valid_json`` — jiter parses and the value validates on
  the Rust path, one boundary crossing.
- **valgebra-loads**: ``json.loads`` then ``is_valid`` — the standard-library
  parse-then-validate baseline.
- **pydantic-json**: a strict ``TypeAdapter.validate_json`` — an external
  Rust-cored reference point.

Not collected by the default test run (``testpaths = tests``); run with
``uv run --group bench pytest benches/bench_json.py``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter

from valgebra import Validator

RECORD_WIDTH = 50
ARRAY_LEN = 10_000
NESTED_ROWS = 200

LIBRARIES = ["valgebra-json", "valgebra-loads", "pydantic-json"]


def shape_record() -> tuple[object, object, str]:
    spec = {f"f{i}": int for i in range(RECORD_WIDTH)}
    data = {f"f{i}": i for i in range(RECORD_WIDTH)}
    return spec, TypeAdapter(dict[str, int]), json.dumps(data)


def shape_array() -> tuple[object, object, str]:
    data = list(range(ARRAY_LEN))
    return list[int], TypeAdapter(list[int]), json.dumps(data)


def shape_nested() -> tuple[object, object, str]:
    data = [{"a": i, "b": i + 1} for i in range(NESTED_ROWS)]
    return list[dict[str, int]], TypeAdapter(list[dict[str, int]]), json.dumps(data)


SHAPES = {
    "record": shape_record,
    "array": shape_array,
    "nested": shape_nested,
}


@pytest.mark.parametrize("lib", LIBRARIES)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_json(benchmark: object, shape: str, lib: str) -> None:
    benchmark.group = shape  # type: ignore[attr-defined]
    spec, adapter, text = SHAPES[shape]()
    if lib == "valgebra-json":
        check = Validator(spec).is_valid_json
        assert benchmark(check, text) is True
    elif lib == "valgebra-loads":
        v = Validator(spec)
        benchmark(lambda doc: v.is_valid(json.loads(doc)), text)
    else:
        benchmark(adapter.validate_json, text)
