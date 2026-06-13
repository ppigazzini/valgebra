"""Edge branches: explain-path details, defensive paths, and error propagation.

These exercise the corners that the main suites do not reach: per-element
reporting in sets and tuples, fail-fast aborts mid-collection, a missing
attribute on an instance, an unrepresentable value, mapping key paths, and the
frontend rejecting an unbuildable inner schema.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from types import GenericAlias

import pytest

from valgebra import (
    CompiledValidator,
    ValidationError,
    fixed_sequence,
    lax,
    validator,
)


def _errors(spec: object, value: object) -> list[object]:
    with pytest.raises(ValidationError) as info:
        validator(spec).validate(value)
    return [item["code"] for item in info.value.errors]


def test_missing_attribute_on_an_instance() -> None:
    @dataclass
    class P:
        x: int
        y: int

    # An uninitialized instance is a P but has no attributes set.
    instance = P.__new__(P)
    assert validator(P).is_valid(instance) is False
    with pytest.raises(ValidationError) as info:
        validator(P).validate(instance, fail_fast=True)
    assert info.value.code == "missing_attribute"
    assert info.value.path == ("x",)


def test_unrepresentable_value_is_summarized_safely() -> None:
    class BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError

    with pytest.raises(ValidationError) as info:
        validator(int).validate(BadRepr(), fail_fast=True)
    assert info.value.value == "<unrepresentable>"


def test_set_and_frozenset_report_each_bad_element() -> None:
    assert _errors(set[int], {"a", "b"}) == ["int_type", "int_type"]
    assert _errors(frozenset[int], frozenset({"a"})) == ["int_type"]


def test_variadic_tuple_reports_each_bad_element() -> None:
    codes = _errors(tuple[int, ...], (1, "x", "y"))
    assert codes == ["int_type", "int_type"]


def test_mapping_reports_key_and_value_paths() -> None:
    with pytest.raises(ValidationError) as info:
        validator(dict[str, int]).validate({"k": "x"})
    assert info.value.code == "int_type"
    assert info.value.path == ("k",)


def test_aggregation_collects_every_failure_then_fail_fast_stops() -> None:
    spec = [int]  # homogeneous list
    bad = ["a", "b", "c"]
    assert _errors(spec, bad) == ["int_type", "int_type", "int_type"]
    with pytest.raises(ValidationError) as info:
        validator(spec).validate(bad, fail_fast=True)
    assert len(info.value.errors) == 1


def test_fixed_length_list_wrong_type_reports_list_type() -> None:
    with pytest.raises(ValidationError) as info:
        fixed_sequence(int, int).validate("not a list", fail_fast=True)
    assert info.value.code == "list_type"


def test_frontend_rejects_an_unbuildable_inner_schema() -> None:
    # A container whose element cannot compile propagates the build error.
    with pytest.raises(NotImplementedError):
        validator(set[list])  # bare `list` is not a usable element schema
    with pytest.raises(NotImplementedError):
        validator(list[dict])  # bare `dict` likewise


def test_too_many_set_arguments_is_rejected() -> None:
    # The native set form admits exactly one element type.
    with pytest.raises(NotImplementedError):
        validator({int, str})


def test_unsupported_typing_form_is_rejected() -> None:
    # A parametrized generic whose origin valgebra does not handle is rejected.
    with pytest.raises(NotImplementedError):
        validator(Iterator[int])


def test_malformed_schema_forms_are_rejected() -> None:
    with pytest.raises(NotImplementedError):
        validator(GenericAlias(dict, (int,)))  # dict needs key and value
    with pytest.raises(NotImplementedError):
        validator([int, str, float])  # a list schema is [T] or [T, ...]


def test_native_list_forms_compile_and_check() -> None:
    # The native [T] and [T, ...] list literals, distinct from typing list[T].
    assert validator([int]).is_valid([1, 2])
    assert not validator([int]).is_valid([1, "x"])
    assert validator([int, ...]).is_valid([1, 2, 3])
    assert not validator([int, ...]).is_valid([1, "x"])


def test_open_record_explain_path_admits_extra_keys() -> None:
    # The aggregating walk over an open record returns without flagging extras.
    lax(validator({"a": int})).validate({"a": 1, "b": 2, "c": 3})


def test_native_mapping_form_compiles_and_checks() -> None:
    # A single type-keyed dict literal is a mapping.
    v = validator({str: int})
    assert v.is_valid({"a": 1, "b": 2})
    assert not v.is_valid({"a": "x"})
    assert not v.is_valid({1: 2})


def test_fail_fast_stops_at_the_first_extra_key() -> None:
    extra = {"a": 1, "b": 2, "c": 3}
    assert len(_errors({"a": int}, extra)) == 2  # both extra keys aggregated
    with pytest.raises(ValidationError) as info:
        validator({"a": int}).validate(extra, fail_fast=True)
    assert len(info.value.errors) == 1
    assert info.value.code == "extra_key"


def test_native_set_form_compiles_and_checks() -> None:
    v = validator({int})
    assert v.is_valid({1, 2})
    assert not v.is_valid({1, "x"})
    assert not v.is_valid([1])


@pytest.mark.parametrize(
    "spec",
    [
        set[list],
        frozenset[list],
        dict[list, int],
        dict[int, list],
        list[dict],
        tuple[list, ...],
        {list},
        [list],  # native [T] with an unbuildable element
        [dict, ...],  # native [T, ...] with an unbuildable element
    ],
)
def test_unbuildable_inner_schema_propagates(spec: object) -> None:
    # A container whose element/key/value cannot compile fails to build, rather
    # than silently producing a wrong validator.
    with pytest.raises(NotImplementedError):
        validator(spec)


def test_single_argument_generics_reject_extra_args() -> None:
    # Built with GenericAlias because they are deliberately malformed type forms.
    with pytest.raises(NotImplementedError):
        validator(GenericAlias(set, (int, str)))
    with pytest.raises(NotImplementedError):
        validator(GenericAlias(tuple, (int, ..., str)))  # ellipsis not as the tail


# (validator, value with two wrong-typed elements). Each collection's explain
# walk aggregates by default and stops at the first failure under fail_fast.
_COLLECTIONS = [
    ("fixed_sequence", validator(fixed_sequence(int, int)), ["x", "y"]),
    ("tuple", validator(tuple[int, int]), ("x", "y")),
    ("variadic_tuple", validator(tuple[int, ...]), (1, "x", "y")),
    ("set", validator(set[int]), {"a", "b"}),
    ("frozenset", validator(frozenset[int]), frozenset({"a", "b"})),
]


@pytest.mark.parametrize(
    ("label", "v", "bad"),
    _COLLECTIONS,
    ids=[c[0] for c in _COLLECTIONS],
)
def test_fail_fast_stops_at_first_element_in_every_collection(
    label: str,
    v: CompiledValidator,
    bad: object,
) -> None:
    full = []
    try:
        v.validate(bad)
    except ValidationError as err:
        full = list(err.errors)
    assert len(full) >= 2
    with pytest.raises(ValidationError) as info:
        v.validate(bad, fail_fast=True)
    assert len(info.value.errors) == 1
