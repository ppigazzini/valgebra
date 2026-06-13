"""Heterogeneous maps and records with a typed catch-all.

A dict schema's string keys are named fields; any other key is a schema keying a
default clause for the rest. Several schema keys give a heterogeneous mapping
(the value type depends on which key schema matches); named fields plus a schema
key give a record with a typed catch-all. Named fields take precedence.
"""

from typing import Annotated

import annotated_types as at
import pytest

from valgebra import ValidationError, lazy, union, validator


def test_heterogeneous_mapping_by_key_schema() -> None:
    schema = validator({str: int, int: str})  # str keys -> int, int keys -> str
    assert schema.is_valid({"a": 1, "b": 2})
    assert schema.is_valid({1: "x", 2: "y"})
    assert schema.is_valid({"a": 1, 2: "y"})  # both clauses in one dict
    assert not schema.is_valid({"a": "x"})  # a str key needs an int value
    assert not schema.is_valid({1: 2})  # an int key needs a str value
    assert schema.is_valid({})  # no key violates any clause


def test_record_with_a_typed_catch_all() -> None:
    schema = validator({"name": str, str: int})  # name: str, other str keys: int
    assert schema.is_valid({"name": "Ada"})
    assert schema.is_valid({"name": "Ada", "age": 36})  # extra str key -> int
    assert not schema.is_valid({"name": "Ada", "age": "old"})  # catch-all wants int
    assert not schema.is_valid({"name": 5})  # the named field still wins
    assert not schema.is_valid({"name": "Ada", 1: 1})  # a non-str key is uncovered


def test_named_field_takes_precedence_over_the_catch_all() -> None:
    # "id" is a named int field even though a str catch-all would also match it.
    schema = validator({"id": int, str: str})
    assert schema.is_valid({"id": 1, "tag": "x"})
    assert not schema.is_valid({"id": "1"})  # the field wins: id must be int


def test_repr_round_trips_the_forms() -> None:
    assert repr(validator({str: int, int: str})) == "{str: int, int: str}"
    assert repr(validator({"name": str, str: int})) == "{'name': str, str: int}"


def test_uncovered_key_reports_a_violation() -> None:
    with pytest.raises(ValidationError):
        validator({str: int}).validate({"a": "not an int"})


def test_a_constant_key_is_a_literal_keyed_clause() -> None:
    # A non-string key is a key schema; a constant becomes a typed singleton, so
    # {1: 2} keys by the literal 1 with the literal-2 value.
    schema = validator({1: 2})
    assert schema.is_valid({1: 2})
    assert not schema.is_valid({1: 3})  # value must be the literal 2
    assert not schema.is_valid({2: 2})  # key must be the literal 1
    assert not schema.is_valid({True: 2})  # 1 and True are distinct singletons


def test_bool_key_is_covered_by_the_int_clause() -> None:
    # bool is a subclass of int, so a bool key matches an int key schema.
    schema = validator({str: int, int: str})
    assert schema.is_valid({True: "x"})  # True is an int key -> str value
    assert not schema.is_valid({True: 1})  # int key needs a str value


def test_optional_field_with_a_catch_all() -> None:
    schema = validator({"a?": int, str: str})
    assert schema.is_valid({})  # the optional field may be absent
    assert schema.is_valid({"a": 1})
    assert schema.is_valid({"b": "x"})  # other str keys take the catch-all
    assert not schema.is_valid({"a": "x"})  # the field still wins: a must be int


def test_a_refinement_key_schema() -> None:
    schema = validator({Annotated[str, at.MinLen(2)]: int})
    assert schema.is_valid({"ab": 1})
    assert not schema.is_valid({"a": 1})  # the key is too short
    assert not schema.is_valid({"ab": "x"})  # the value must be an int


def test_a_non_string_key_in_a_closed_record_is_rejected() -> None:
    schema = validator({"a": int})  # closed: no default clause
    assert schema.is_valid({"a": 1})
    assert not schema.is_valid({"a": 1, ("tuple",): 2})  # a non-string key


def test_recursion_through_a_field_and_a_catch_all() -> None:
    # A recursive reference sits inside both a named field's value and the
    # catch-all clause's value; the map guards it, so the fixpoint is contractive.
    tree = lazy(lambda s: {"v": int, str: union(s, None)})
    assert tree.is_valid({"v": 1})
    assert tree.is_valid({"v": 1, "child": {"v": 2}})
    assert not tree.is_valid({"v": 1, "child": {"v": "x"}})
    cyclic: dict[str, object] = {"v": 1}
    cyclic["self"] = cyclic
    assert not tree.is_valid(cyclic)  # a cyclic value is rejected, never looping
