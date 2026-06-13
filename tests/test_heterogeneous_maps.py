"""Heterogeneous maps and records with a typed catch-all.

A dict schema's string keys are named fields; any other key is a schema keying a
default clause for the rest. Several schema keys give a heterogeneous mapping
(the value type depends on which key schema matches); named fields plus a schema
key give a record with a typed catch-all. Named fields take precedence.
"""

import pytest

from valgebra import ValidationError, validator


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
