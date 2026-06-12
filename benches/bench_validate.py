"""End-to-end validation benchmarks for valgebra's public API.

Each benchmark compiles a schema once and times a single boundary-crossing call
(``is_valid``/``validate``) over a synthetic shape that stresses one cost
dimension: large flat arrays, wide records, deep nesting, and union dispatch.
Compilation cost is measured on its own.

These are not collected by the default ``pytest`` run (``testpaths = tests``);
run them with ``pytest benches/bench_validate.py`` after installing the
``bench`` dependency group. The harness is pytest-benchmark.
"""

from __future__ import annotations

from typing import Literal

from valgebra import union, validator

# Sizes chosen so each shape runs in microseconds-to-milliseconds and the
# relative costs stay visible; they are recorded alongside results in the docs.
ARRAY_LEN = 10_000
RECORD_WIDTH = 50
NESTING_DEPTH = 25
UNION_BRANCHES = 32


def nested_list_schema(depth: int) -> object:
    """Build a ``list[list[... int]]`` schema nested ``depth`` levels deep."""
    schema: object = int
    for _ in range(depth):
        schema = list[schema]  # type: ignore[valid-type]
    return schema


def nested_list_value(depth: int) -> object:
    """Return a value matching :func:`nested_list_schema` at the same depth."""
    value: object = 0
    for _ in range(depth):
        value = [value]
    return value


def wide_record_schema(width: int) -> dict[str, type]:
    """Build a closed record with ``width`` required integer fields."""
    return {f"f{i}": int for i in range(width)}


def wide_record_value(width: int) -> dict[str, int]:
    return {f"f{i}": i for i in range(width)}


def test_large_array(benchmark: object) -> None:
    check = validator(list[int]).is_valid
    data = list(range(ARRAY_LEN))
    assert benchmark(check, data) is True


def test_wide_record_is_valid(benchmark: object) -> None:
    check = validator(wide_record_schema(RECORD_WIDTH)).is_valid
    data = wide_record_value(RECORD_WIDTH)
    assert benchmark(check, data) is True


def test_wide_record_validate(benchmark: object) -> None:
    # The aggregating explain walk on a passing value, versus the bool fast path.
    validate = validator(wide_record_schema(RECORD_WIDTH)).validate
    data = wide_record_value(RECORD_WIDTH)
    benchmark(validate, data)


def test_deep_nesting(benchmark: object) -> None:
    check = validator(nested_list_schema(NESTING_DEPTH)).is_valid
    data = nested_list_value(NESTING_DEPTH)
    assert benchmark(check, data) is True


def test_union_dispatch(benchmark: object) -> None:
    # Worst case: the matching branch is last, so the walk scans every member.
    schema = union(*(Literal[i] for i in range(UNION_BRANCHES)))
    check = schema.is_valid
    target = UNION_BRANCHES - 1
    assert benchmark(check, target) is True


def test_compile_wide_record(benchmark: object) -> None:
    schema = wide_record_schema(RECORD_WIDTH)
    benchmark(validator, schema)
