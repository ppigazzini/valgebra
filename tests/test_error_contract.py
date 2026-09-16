"""The promises `docs/08-error-model.md` makes, held at the sites that make them.

Two of them are short enough to quote. `fail_fast=True` "stops at the first
failure", and the page's own example asserts `len(err.errors) == 1`. And a path
segment is the key it names: "a string key is itself, in full, and an **integer
key is itself as an integer** ... so walking the path back down reaches the
value".

Each is one rule with several sites, and a rule held at one site and not another
is the shape this file is for: a union and a mapping clause each aggregated a
whole branch under `fail_fast`, and an undeclared integer key came back as the
string of its digits from one reading of a record and as the integer from the
other. A caller indexing `d[path[-1]]` finds nothing in the first case, which is
the sentence the page wrote the rule for.
"""

from __future__ import annotations

from typing import Annotated

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator, union


def _errors(schema: object, value: object, *, fail_fast: bool = False) -> list[dict]:
    with pytest.raises(ValidationError) as caught:
        Validator(schema).validate(value, fail_fast=fail_fast)
    return list(caught.value.errors)


def _json_errors(schema: object, text: str, *, fail_fast: bool = False) -> list[dict]:
    with pytest.raises(ValidationError) as caught:
        Validator(schema).validate_json(text, fail_fast=fail_fast)
    return list(caught.value.errors)


# Each row: a name, a schema, and a value failing it in more than one place.
_MANY_FAILURES: list[tuple[str, object, object]] = [
    ("a record", {"a": int, "b": str}, {"a": "x", "b": 1}),
    ("a list", list[int], ["x", "y"]),
    ("a set", set[int], {"x", "y"}),
    ("a mapping clause", dict[str, int], {1: "x"}),
    ("a union", union(int, {"p": int, "q": int}), {"p": "s", "q": "s"}),
    ("a union of records", union({"a": int}, {"a": str, "b": int}), {"a": 1.5}),
    ("a nested record", {"a": {"x": int, "y": int}}, {"a": {"x": "s", "y": "s"}}),
]


@pytest.mark.parametrize(
    ("name", "schema", "value"), _MANY_FAILURES, ids=[row[0] for row in _MANY_FAILURES]
)
def test_fail_fast_reports_one_failure(
    name: str, schema: object, value: object
) -> None:
    """The page's own assertion, at every site that can report more than one."""
    assert len(_errors(schema, value, fail_fast=True)) == 1, name
    # And the same value without it reports what it finds, which is what makes
    # the row above a choice rather than the only answer available.
    assert len(_errors(schema, value)) >= 1, name


def test_fail_fast_reports_one_failure_on_the_json_path() -> None:
    """The JSON entry points carry the same promise."""
    schema = union(int, {"p": int, "q": int})
    assert len(_json_errors(schema, '{"p": "s", "q": "s"}', fail_fast=True)) == 1
    assert len(_json_errors(dict[int, int], '{"a": "x"}', fail_fast=True)) == 1


def test_fail_fast_keeps_the_failure_the_aggregate_would_lead_with() -> None:
    """Stopping early reports the first of what it would otherwise report."""
    schema = union(int, {"p": int, "q": int})
    value = {"p": "s", "q": "s"}
    assert _errors(schema, value, fail_fast=True) == _errors(schema, value)[:1]


def test_load_stops_at_the_first_failure_too() -> None:
    """`load` takes the same keyword and means the same thing by it."""
    with pytest.raises(ValidationError) as caught:
        Validator({"a": int, "b": int}).load('{"a": "x", "b": "y"}', fail_fast=True)
    assert len(caught.value.errors) == 1
    with pytest.raises(ValidationError) as caught:
        Validator({"a": int, "b": int}).load('{"a": "x", "b": "y"}')
    assert len(caught.value.errors) == 2


# Each row: a name, a record schema, a dict carrying a key it does not declare,
# and the segment the path must name it by.
_KEYS: list[tuple[str, object, object, object]] = [
    ("an undeclared int key, closed", {"a": int}, {"a": 1, 2: 3}, 2),
    (
        "an undeclared int key, with a catch-all",
        {"a": int, str: int},
        {"a": 1, 2: 3},
        2,
    ),
    ("a large int key", {"a": int}, {"a": 1, 2**70: 3}, 2**70),
    ("a bool key", {"a": int}, {"a": 1, True: 3}, 1),
    ("a str key", {"a": int}, {"a": 1, "b": 3}, "b"),
]


@pytest.mark.parametrize(
    ("name", "schema", "value", "segment"), _KEYS, ids=[row[0] for row in _KEYS]
)
def test_a_path_names_the_key_it_points_at(
    name: str, schema: object, value: object, segment: object
) -> None:
    """An integer key is itself, so the path indexes back down to the value."""
    errors = _errors(schema, value)
    paths = [item["path"] for item in errors]
    assert any(path and path[-1] == segment for path in paths), (name, paths)
    # The sentence the rule is written for: the segment reaches the entry.
    reached = [path[-1] for path in paths if path]
    assert all(key in value for key in reached), (name, reached)  # ty: ignore[unsupported-operator]


def test_the_two_readings_of_a_record_name_a_key_alike() -> None:
    """A catch-all beside a field changes which reading runs, not the answer."""
    closed = _errors({"a": int}, {"a": 1, 2: 3})[0]["path"]
    with_catch_all = _errors({"a": int, str: int}, {"a": 1, 2: 3})[0]["path"]
    assert closed == with_catch_all == (2,)


def test_a_key_that_is_neither_a_string_nor_an_integer_is_named_by_its_repr() -> None:
    """The page's third case, which has no spelling a caller indexes with."""
    errors = _errors({"a": int}, {"a": 1, (1, 2): 3})
    assert any(item["path"] == ("(1, 2)",) for item in errors), errors


def test_a_predicate_that_raises_names_its_error_without_flooding_the_message() -> None:
    """A large error is summarised, as every other value in a message is."""

    def boom(_value: object) -> bool:
        raise ValueError("Z" * 5000)

    errors = _errors(Annotated[int, at.Predicate(boom)], 1)
    assert errors[0]["code"] == "predicate_error"
    assert len(errors[0]["expected"]) < 200
    assert len(errors[0]["message"]) < 400
    # And the message still says what raised.
    assert "Z" in errors[0]["expected"]


def test_a_value_summary_is_bounded_and_says_it_was_cut() -> None:
    """The style guide's bound, at the edge rather than in the middle."""
    errors = _errors(int, "x" * 500)
    assert len(errors[0]["value"]) <= 84
    assert errors[0]["value"].endswith("...")
    # A short value is not cut at all.
    assert _errors(int, "xy")[0]["value"] == "'xy'"


def test_a_fatal_signal_raised_while_a_message_is_built_propagates() -> None:
    """A fatal signal is never folded, at every site -- the summary included."""

    class Unrepresentable:
        def __repr__(self) -> str:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Validator(int).validate(Unrepresentable())
    # `is_valid` builds no message, so nothing asks the value to render and the
    # signal has no site to arise at. That is the answer, not a fold: the
    # membership question was decided without the `__repr__` being called.
    assert Validator(int).is_valid(Unrepresentable()) is False


def test_an_ordinary_exception_from_a_repr_is_folded_into_the_summary() -> None:
    """The control: an ordinary failure to render is not the interpreter ending."""

    class Awkward:
        def __repr__(self) -> str:
            raise ValueError("no repr for you")

    errors = _errors(int, Awkward())
    assert errors[0]["code"] == "int_type"
    assert errors[0]["value"] == "<unrepresentable>"
