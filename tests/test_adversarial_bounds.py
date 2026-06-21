"""Adversarial inputs meet their resource bounds instead of exhausting one.

A validator processes untrusted values, so every recursive walk and every
error-reporting probe is bounded by a gated limit rather than the input. These
tests drive each pathological shape and assert the bound engages: the operation
terminates with a graceful verdict or a specific error code, never a native
stack overflow, a Python ``RecursionError``, or an unbounded hang. The limits
themselves are:

- schema build recursion (deeply nested annotation),
- value-walk recursion (deeply nested value, on both the object and JSON paths),
- the object-identity loop guard (a value that contains itself),
- the closest-branch probe cap (error reporting over a wide union).

Worst-case *timing* on these shapes is recorded by ``benches/bench_adversarial``;
here the guarantee is that the guard fires, which is what keeps a hostile input
from turning into denial of service. Union width is part of the developer-written
schema, not untrusted input, so the bounds here are about the value, not the
schema's declared size.
"""

from __future__ import annotations

from typing import Literal

import pytest

from valgebra import ValidationError, Validator, recursive, union


def _nested_annotation(depth: int) -> object:
    schema: object = int
    for _ in range(depth):
        schema = list[schema]  # type: ignore[valid-type]
    return schema


def _nested_value(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = [value]
    return value


def test_build_depth_guard_rejects_an_overdeep_schema() -> None:
    # A reasonably nested annotation compiles; one past the build-depth guard is
    # rejected at compile time with a clean exception, not a stack overflow.
    assert Validator(_nested_annotation(50)).is_valid(_nested_value(50))
    with pytest.raises((NotImplementedError, ValidationError, ValueError)):
        Validator(_nested_annotation(1000))


def test_deeply_nested_object_hits_the_recursion_limit() -> None:
    schema = Validator(recursive(lambda j: union(int, [j])))
    deep = _nested_value(5000)
    # is_valid swallows the bound as a non-membership; validate names it.
    assert not schema.is_valid(deep)
    with pytest.raises(ValidationError) as info:
        schema.validate(deep)
    assert info.value.code == "recursion_limit"


def test_deeply_nested_json_is_rejected_cleanly() -> None:
    schema = Validator(recursive(lambda j: union(int, [j])))
    document = "[" * 5000 + "1" + "]" * 5000
    assert not schema.is_valid_json(document)
    with pytest.raises(ValidationError) as info:
        schema.validate_json(document)
    assert info.value.code == "json_invalid"


def test_self_referential_value_is_caught_as_a_loop() -> None:
    schema = Validator(recursive(lambda j: union(int, [j])))
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert not schema.is_valid(cyclic)
    with pytest.raises(ValidationError) as info:
        schema.validate(cyclic)
    # The value's self-reference is caught by the cycle guard, not a generic union
    # miss: pin the exact code so a regression to `union_error` is visible.
    assert info.value.code == "recursion_loop"


def test_wide_union_membership_is_decided_and_bounded() -> None:
    wide = union(*[Literal[i] for i in range(5000)])  # ty: ignore[invalid-type-form]
    # The value-driven work (the linear scan and the capped closest-branch probe)
    # terminates with the right verdict for both a member and a non-member.
    assert wide.is_valid(4999)
    assert not wide.is_valid(10_000)
    with pytest.raises(ValidationError) as info:
        wide.validate(10_000)
    # One aggregated union failure: the probe cap keeps the report from rewalking
    # every branch of the union.
    assert info.value.code == "union_error"
    assert len(info.value.errors) == 1


def test_hostile_dict_keys_are_handled() -> None:
    mapping = Validator(dict[str, int])
    assert mapping.is_valid({str(i): i for i in range(100_000)})
    assert mapping.is_valid({"k" * 1_000_000: 1})
    assert not mapping.is_valid({"bad": "not an int"})
