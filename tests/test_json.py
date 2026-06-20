"""JSON path consistency: validate_json/is_valid_json against the object path.

The JSON path parses with jiter and runs the same validation walk as a native
object, so for every schema and every JSON document the decision and the errors
must match validating ``json.loads`` of the same document. These tests lock that
equivalence, plus the str/bytes input handling and the malformed-JSON contract.
"""

from __future__ import annotations

import json
from types import GenericAlias
from typing import TYPE_CHECKING, Annotated

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import ValidationError, Validator, union

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
    v = Validator(spec)
    obj = json.loads(doc)
    # The bool fast path agrees.
    assert v.is_valid_json(doc) == v.is_valid(obj)
    # The aggregating walk agrees on whether it raises and on the first error.
    assert _outcome(lambda: v.validate_json(doc)) == _outcome(lambda: v.validate(obj))


def test_validate_json_accepts_bytes() -> None:
    v = Validator({"name": str})
    v.validate_json(b'{"name": "Ada"}')
    assert v.is_valid_json(b'{"name": "Ada"}')
    assert not v.is_valid_json(b'{"name": 5}')


def test_validate_json_returns_none_on_success() -> None:
    assert Validator(list[int]).validate_json("[1, 2, 3]") is None


def test_duplicate_json_keys_keep_the_last_value() -> None:
    # A JSON object may repeat a key; json.loads keeps the last, and the keyed-map
    # walk covers each non-field key by its last value in a single pass.
    v = Validator(dict[str, int])
    assert v.is_valid_json('{"a": 1, "a": 2}')
    assert not v.is_valid_json('{"a": 1, "a": "x"}')
    assert v.is_valid_json('{"a": "x", "a": 3}')


def test_many_duplicate_json_keys_validate_in_one_pass() -> None:
    # A document with thousands of repeated keys against an open mapping is covered
    # without a per-key tail rescan; this finishes promptly rather than quadratically.
    v = Validator(dict[str, int])
    doc = "{" + ", ".join(f'"k": {i}' for i in range(20000)) + "}"
    assert v.is_valid_json(doc)


def test_malformed_json_raises_a_structured_error() -> None:
    v = Validator(int)
    with pytest.raises(ValidationError) as info:
        v.validate_json("{not json")
    assert info.value.code == "json_invalid"
    assert info.value.path == ()
    # The error model is uniform: malformed JSON appears in `errors` too.
    assert info.value.errors[0]["code"] == "json_invalid"


def test_malformed_json_is_not_valid() -> None:
    # is_valid_json never raises; unparseable input is simply not a member.
    assert not Validator(int).is_valid_json("{not json")
    assert not Validator(int).is_valid_json("")


def test_validate_json_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        Validator(int).validate_json(123)  # ty: ignore[invalid-argument-type]


def test_json_aggregates_every_failure_like_the_object_path() -> None:
    v = Validator({"a": int, "b": int, "c": int})
    doc = '{"a": "x", "b": "y", "c": "z"}'
    with pytest.raises(ValidationError) as json_info:
        v.validate_json(doc)
    with pytest.raises(ValidationError) as obj_info:
        v.validate(json.loads(doc))
    assert json_info.value.errors == obj_info.value.errors


def test_fail_fast_stops_at_first_failure_on_the_json_path() -> None:
    v = Validator({"a": int, "b": int})
    doc = '{"a": "x", "b": "y"}'
    with pytest.raises(ValidationError) as info:
        v.validate_json(doc, fail_fast=True)
    assert len(info.value.errors) == 1


def _json_schemas() -> st.SearchStrategy[object]:
    leaf = st.one_of(
        st.sampled_from([int, float, bool, str, None, object]),
        st.sampled_from([0, 1, "a", "", True, 1.5]),
        st.integers(min_value=-3, max_value=3).map(lambda k: Annotated[int, at.Ge(k)]),
    )
    return st.recursive(
        leaf,
        lambda child: st.one_of(
            child.map(lambda x: GenericAlias(list, (x,))),
            child.map(lambda x: GenericAlias(dict, (str, x))),
            st.tuples(child, child).map(lambda ab: {"a": ab[0], "b?": ab[1]}),
            st.tuples(child, child).map(lambda ab: union(ab[0], ab[1])),
        ),
        max_leaves=8,
    )


def _json_values() -> st.SearchStrategy[object]:
    leaf = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=5),
    )
    return st.recursive(
        leaf,
        lambda child: st.one_of(
            st.lists(child, max_size=4),
            st.dictionaries(st.text(max_size=3), child, max_size=4),
        ),
        max_leaves=10,
    )


@given(spec=_json_schemas(), value=_json_values())
def test_json_path_fuzz_agrees_with_object_path(spec: object, value: object) -> None:
    # The in-place JSON walk must reach the same verdict as validating the
    # json.loads of the same document on the object path.
    v = Validator(spec)
    doc = json.dumps(value)
    assert v.is_valid_json(doc) == v.is_valid(json.loads(doc))


def test_load_returns_the_parsed_value() -> None:
    # load parses, validates, and hands back the parsed object (no second parse).
    v = Validator({"name": str, "age?": int})
    parsed = v.load('{"name": "Ada", "age": 36}')
    assert parsed == {"name": "Ada", "age": 36}
    # bytes input works too
    assert v.load(b'{"name": "Ada"}') == {"name": "Ada"}


def test_load_raises_on_a_non_member() -> None:
    v = Validator(list[int])
    with pytest.raises(ValidationError) as info:
        v.load('[1, "x"]')
    assert info.value.code == "int_type"


def test_load_raises_on_malformed_json() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int).load("{not json")
    assert info.value.code == "json_invalid"


@given(spec=_json_schemas(), value=_json_values())
def test_load_round_trips_with_json_loads(spec: object, value: object) -> None:
    # When the document is a member, load returns exactly what json.loads would.
    v = Validator(spec)
    doc = json.dumps(value)
    if v.is_valid(json.loads(doc)):
        assert v.load(doc) == json.loads(doc)
