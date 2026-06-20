"""Worst-case timing on pathological inputs.

Each shape drives a resource guard valgebra enforces against untrusted input: a
deeply nested value (object and JSON paths), a wide union, and a hostile mapping.
The point is to *measure* the bounded cost rather than reason about it; the guards
are correctness-tested in ``tests/test_adversarial_bounds.py``. Record the numbers
in ``docs/performance.md``.

Run with ``pytest benches/bench_adversarial.py`` after installing the ``bench``
dependency group.
"""

from __future__ import annotations

from typing import Literal

import pytest

from valgebra import Validator, recursive, union

DEPTH = 5_000
UNION_WIDTH = 5_000
DICT_KEYS = 100_000

_DEEP = Validator(recursive(lambda j: union(int, [j])))


def _nested_value(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = [value]
    return value


def test_deep_object_rejection(benchmark: object) -> None:
    value = _nested_value(DEPTH)
    benchmark(_DEEP.is_valid, value)  # type: ignore[operator]


def test_deep_json_rejection(benchmark: object) -> None:
    document = "[" * DEPTH + "1" + "]" * DEPTH
    benchmark(_DEEP.is_valid_json, document)  # type: ignore[operator]


def test_wide_union_membership(benchmark: object) -> None:
    wide = union(*[Literal[i] for i in range(UNION_WIDTH)])
    benchmark(wide.is_valid, UNION_WIDTH - 1)  # type: ignore[operator]


def test_hostile_mapping(benchmark: object) -> None:
    mapping = Validator(dict[str, int])
    data = {str(i): i for i in range(DICT_KEYS)}
    benchmark(mapping.is_valid, data)  # type: ignore[operator]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
