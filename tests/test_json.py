"""JSON path consistency: validate_json/is_valid_json against the object path.

The JSON path parses with jiter and runs the same validation walk as a native
object, so for every schema and every JSON document the decision and the errors
must match validating ``json.loads`` of the same document. These tests lock that
equivalence, plus the str/bytes input handling and the malformed-JSON contract.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from valgebra import ValidationError, validator

if TYPE_CHECKING:
    from collections.abc import Callable


def _outcome(call: Callable[[], object]) -> tuple[str | int, ...] | None:
    """Return the first error's (code, path) if validation raises, else None."""
    try:
        call()
    except ValidationError as err:
        return (err.code, *err.path)
    return None


# (label, schema spec, JSON documents to probe). The documents exercise the JSON
# value model: numbers (int vs float), strings, booleans, null, arrays, and
# objects, against scalar, collection, union, and record schemas.
CORPUS: list[tuple[str, object, list[str]]] = [
    ("int", int, ["1", "0", "-5", "true", "1.5", '"x"', "null"]),
    ("float", float, ["1.5", "1", "true", '"x"']),
    ("bool", bool, ["true", "false", "1", "0"]),
    ("str", str, ['"x"', '""', "1", "null"]),
    ("none", None, ["null", "0", '""', "false"]),
    ("list", list[int], ["[]", "[1,2,3]", '[1,"x"]', "{}", '"x"']),
    ("mapping", dict[str, int], ["{}", '{"a":1}', '{"a":"x"}', "[]"]),
    ("optional", str | None, ["null", '"x"', "5"]),
    (
        "record",
        {"name": str, "age?": int},
        [
            '{"name":"Ada"}',
            '{"name":"Ada","age":36}',
            '{"name":"Ada","age":"old"}',
            '{"name":"Ada","extra":1}',
            "{}",
            '{"name":5}',
        ],
    ),
    (
        "nested",
        list[dict[str, int]],
        ["[]", '[{"a":1}]', '[{"a":"x"}]', "[1]", '[{"a":1},{"b":2}]'],
    ),
]

# Flatten to one parameter per (schema, document) pair for readable test ids.
PAIRS = [
    (f"{label}-{i}", spec, doc)
    for label, spec, docs in CORPUS
    for i, doc in enumerate(docs)
]


@pytest.mark.parametrize(("label", "spec", "doc"), PAIRS, ids=[p[0] for p in PAIRS])
def test_json_path_agrees_with_object_path(label: str, spec: object, doc: str) -> None:
    v = validator(spec)
    obj = json.loads(doc)
    # The bool fast path agrees.
    assert v.is_valid_json(doc) == v.is_valid(obj)
    # The aggregating walk agrees on whether it raises and on the first error.
    assert _outcome(lambda: v.validate_json(doc)) == _outcome(lambda: v.validate(obj))


def test_validate_json_accepts_bytes() -> None:
    v = validator({"name": str})
    v.validate_json(b'{"name": "Ada"}')
    assert v.is_valid_json(b'{"name": "Ada"}')
    assert not v.is_valid_json(b'{"name": 5}')


def test_validate_json_returns_none_on_success() -> None:
    assert validator(list[int]).validate_json("[1, 2, 3]") is None


def test_malformed_json_raises_a_structured_error() -> None:
    v = validator(int)
    with pytest.raises(ValidationError) as info:
        v.validate_json("{not json")
    assert info.value.code == "json_invalid"
    assert info.value.path == ()
    # The error model is uniform: malformed JSON appears in `errors` too.
    assert info.value.errors[0]["code"] == "json_invalid"


def test_malformed_json_is_not_valid() -> None:
    # is_valid_json never raises; unparseable input is simply not a member.
    assert not validator(int).is_valid_json("{not json")
    assert not validator(int).is_valid_json("")


def test_validate_json_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        validator(int).validate_json(123)  # ty: ignore[invalid-argument-type]


def test_json_aggregates_every_failure_like_the_object_path() -> None:
    v = validator({"a": int, "b": int, "c": int})
    doc = '{"a": "x", "b": "y", "c": "z"}'
    with pytest.raises(ValidationError) as json_info:
        v.validate_json(doc)
    with pytest.raises(ValidationError) as obj_info:
        v.validate(json.loads(doc))
    assert json_info.value.errors == obj_info.value.errors


def test_fail_fast_stops_at_first_failure_on_the_json_path() -> None:
    v = validator({"a": int, "b": int})
    doc = '{"a": "x", "b": "y"}'
    with pytest.raises(ValidationError) as info:
        v.validate_json(doc, fail_fast=True)
    assert len(info.value.errors) == 1
