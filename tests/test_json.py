"""JSON path consistency: validate_json/is_valid_json against the object path.

The JSON path parses with jiter and runs the same validation walk as a native
object, so for every schema and every JSON document the decision and the errors
must match validating ``json.loads`` of the same document. These tests lock that
equivalence, plus the str/bytes input handling and the malformed-JSON contract.
"""

from __future__ import annotations

import json
from types import GenericAlias
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

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
# PROMISE: JSON output
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


def test_a_declared_key_repeated_keeps_its_last_value() -> None:
    # The same rule for a *declared* field, which a closed record resolves
    # through its own plan: the document's later entry is the one it means, so an
    # earlier one that would fail is not the value checked. Read the other way
    # too -- a later entry that fails is a failure however good the earlier one
    # was.
    v = Validator({"a": int, "b?": str})
    assert v.is_valid_json('{"a": "x", "a": 1}')
    assert not v.is_valid_json('{"a": 1, "a": "x"}')
    # And the rest of a closed record's rules over the same path.
    assert not v.is_valid_json('{"a": 1, "z": 2}')
    assert v.is_valid_json('{"a": 1}')
    assert v.is_valid_json('{"a": 1, "b": "s"}')
    assert not v.is_valid_json('{"b": "s"}')


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
    # is_valid_json never raises; unparseable input is not a member.
    assert not Validator(int).is_valid_json("{not json")
    assert not Validator(int).is_valid_json("")


def test_validate_json_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        Validator(int).validate_json(123)  # ty: ignore[invalid-argument-type]


def test_undecodable_json_string_agrees_across_entry_points() -> None:
    # A lone surrogate cannot encode to UTF-8. Both JSON entry points must treat
    # it as the same malformed-input condition: the check returns not-a-member,
    # and the raising entries report a structured `json_invalid` error rather than
    # leaking a `UnicodeEncodeError` the contract forbids.
    v = Validator(int)
    lone_surrogate = "\udc80"
    assert not v.is_valid_json(lone_surrogate)
    for entry in (v.validate_json, v.load):
        with pytest.raises(ValidationError) as info:
            entry(lone_surrogate)
        assert info.value.code == "json_invalid"
        assert info.value.errors[0]["code"] == "json_invalid"


# TRUST: The JSON parser (jiter) agrees with `json.loads` where both accept.
def test_the_two_documents_the_grammar_refuses_and_the_module_accepts() -> None:
    """What the trust base buys, and the two documents it does not cover.

    The JSON path's denotation is the object path's because the parser builds
    the value `json.loads` builds. That holds on every document both accept,
    and the grammar is the stricter of the two: a non-standard float token and
    an escape naming a lone surrogate are documents Python's module parses and
    this parser refuses ([the JSON path](../docs/07-json.md)).

    Refusing is the sound direction -- the JSON path admits a subset of what
    the object path does, never a different value -- so each is reported as
    `json_invalid` before a schema sees the document. Driving them is what
    keeps the trust base honest about which documents it covers: an
    implementation that started *accepting* one of these would be taking a
    value the object path holds and the grammar does not name.
    """
    beyond_the_grammar = ["NaN", "Infinity", "-Infinity", r'"\ud800"']
    for text in beyond_the_grammar:
        # Python's own parser builds a value, and the object path holds it.
        built = json.loads(text)
        assert Validator(object).is_valid(built)
        # The JSON path reports the document, rather than the value.
        assert not Validator(object).is_valid_json(text)
        with pytest.raises(ValidationError) as info:
            Validator(object).validate_json(text)
        assert info.value.code == "json_invalid"

    # And within the grammar the two build the same value, which is the half
    # of the assumption the suites lean on everywhere else.
    for text in ('{"a": [1, 2.5, null, true]}', '"\u00e9"', "1e400", "-0.0"):
        assert Validator(object).is_valid_json(text) is Validator(object).is_valid(
            json.loads(text)
        )


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
        # Spelled `Literal[v]`: the leaf is also a generic's argument, where a
        # bare value names a type rather than being one.
        st.sampled_from([0, 1, "a", "", True, 1.5]).map(lambda value: Literal[value]),  # ty: ignore[invalid-type-form]
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


def test_a_json_entry_refuses_a_value_that_is_not_a_document() -> None:
    """`str` or `bytes`, and anything else is a `TypeError` rather than a failure.

    The three entries take a *document*, so a caller handing one an `int` has
    made a type error and not written an invalid document -- and reading it as
    the second would report a `ValidationError` about a document nobody wrote.
    The distinction is the docstrings' own, and it was held for `validate_json`
    and for neither of the others.
    """
    validator = Validator(int)
    with pytest.raises(TypeError, match="str or bytes"):
        validator.load(123)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="str or bytes"):
        validator.validate_json(123)  # ty: ignore[invalid-argument-type]
    # `is_valid_json` answers a question rather than raising one, so a value
    # that is not a document is not a member.
    assert validator.is_valid_json(123) is False  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("width", [1, 15, 16, 40])
def test_a_record_reads_a_document_at_every_width(width: int) -> None:
    """A record decides a parsed object alike on each side of the table width.

    The walk gathers a document's value for each declared field into a table
    held on the stack up to sixteen fields and on the heap past them. With the
    optional field the widths give records of 2, 16, 17 and 41 fields, so each
    assertion runs on both sides of that line, and against the object path.
    """
    v = Validator({f"k{i}": int for i in range(width)} | {"opt?": str})
    whole = {f"k{i}": i for i in range(width)}
    docs = [
        whole,
        whole | {"opt": "s"},
        whole | {f"k{width - 1}": "x"},
        {k: val for k, val in whole.items() if k != "k0"},
        whole | {"undeclared": 1},
    ]
    for doc in docs:
        text = json.dumps(doc)
        assert v.is_valid_json(text) == v.is_valid(doc), text
    assert v.is_valid_json(json.dumps(whole))
    assert not v.is_valid_json(json.dumps(whole | {f"k{width - 1}": "x"}))
    # The last of two entries for one key is the one read, whichever side of
    # the width the record sits.
    last = json.dumps(whole)[:-1] + ', "k0": "x", "k0": 0}'
    assert v.is_valid_json(last)
    assert not v.is_valid_json(json.dumps(whole)[:-1] + ', "k0": 0, "k0": "x"}')


class _OpenRecord(TypedDict):
    a: int


@pytest.mark.parametrize("spec", [_OpenRecord, object, dict[str, Any]])
def test_invalid_utf8_in_an_unread_value_is_not_a_document(spec: object) -> None:
    """A string the schema never looks at is still decoded, and a bad one refuses.

    The tree parser decodes every string it meets, so `is_valid_json` answers
    for a document with a byte that is not UTF-8 exactly as `validate_json`
    does, whether or not a field reads it. A reader that skips the unread value
    -- jiter's `next_skip` does not check UTF-8 -- would call the document a
    member, which is what `docs/dev/04-walk.md` refuses a streaming check for.
    """
    v = Validator(spec)
    doc = b'{"a": 1, "x": "\xff"}'
    assert not v.is_valid_json(doc)
    assert _outcome(lambda: v.validate_json(doc)) == ("json_invalid",)


def _nested(depth: int) -> str:
    return "[" * depth + "1" + "]" * depth


def test_two_hundred_containers_deep_is_a_document() -> None:
    """The parser's nesting limit admits two hundred containers.

    The row beside it refuses one more, and the pair is the limit's edge.
    """
    v = Validator(object)
    assert v.is_valid_json(_nested(200))
    assert _outcome(lambda: v.validate_json(_nested(200))) is None


@pytest.mark.parametrize("depth", [201, 300])
def test_the_nesting_limit_counts_from_the_root(depth: int) -> None:
    """One container past the limit refuses, whatever the schema reads.

    The limit is the parser's over the whole document, counted from the root,
    so a schema that reads none of the nesting still refuses a document past
    it. A reader that hands each unread subtree to a fresh parser call gets a
    fresh budget per call and would accept the deeper documents.
    """
    v = Validator(object)
    assert not v.is_valid_json(_nested(depth))
    assert _outcome(lambda: v.validate_json(_nested(depth))) == ("json_invalid",)


def test_a_predicate_sees_only_what_a_parsed_document_holds() -> None:
    """The document is parsed before any of the schema's Python runs.

    So a predicate is never called for a document that turns out malformed,
    and a repeated key calls it with the value the document means -- the last
    -- and never with an earlier one. Both are orders a caller can observe, and
    both hold because the check reads a finished parse.
    """
    calls: list[object] = []

    def seen(x: object) -> bool:
        calls.append(x)
        return True

    items = Validator(list[Annotated[int, at.Predicate(seen)]])
    assert not items.is_valid_json("[1, 2, 3, oops")
    assert calls == []

    record = Validator({"a": Annotated[int, at.Predicate(seen)]})
    assert record.is_valid_json('{"a": 1, "a": 2}')
    assert calls == [2]
    calls.clear()
    assert record.is_valid_json('{"a": "no", "a": 2}')
    assert calls == [2]
