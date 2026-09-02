import os
import subprocess
import sys
import textwrap

import pytest

from valgebra import ValidationError, Validator


def test_record_aggregates_every_field_failure() -> None:
    schema = Validator({"a": int, "b": str, "c": int})
    with pytest.raises(ValidationError) as info:
        schema.validate({"a": "x", "b": 1, "c": "y"})
    codes = [item["code"] for item in info.value.errors]
    assert codes == ["int_type", "string_type", "int_type"]


def test_sequence_aggregates_every_element_failure() -> None:
    with pytest.raises(ValidationError) as info:
        Validator([int]).validate([1, "x", 2, "y"])
    paths = [item["path"] for item in info.value.errors]
    assert paths == [(1,), (3,)]


def test_aggregated_str_is_a_counted_summary() -> None:
    with pytest.raises(ValidationError) as info:
        Validator({"a": int, "b": int}).validate({"a": "x", "b": "y"})
    summary = str(info.value)
    assert summary.startswith("2 validation errors:")
    assert summary.count("\n") == 2


def test_fail_fast_stops_at_the_first_failure() -> None:
    schema = Validator({"a": int, "b": str, "c": int})
    with pytest.raises(ValidationError) as info:
        schema.validate({"a": "x", "b": 1, "c": "y"}, fail_fast=True)
    assert len(info.value.errors) == 1
    assert info.value.code == "int_type"


def test_aggregation_order_is_deterministic() -> None:
    schema = Validator({"items": [int], "name": str})
    with pytest.raises(ValidationError) as info:
        schema.validate({"items": [1, "x"], "name": 5})
    paths = [item["path"] for item in info.value.errors]
    assert paths == [("items", 1), ("name",)]


def test_a_single_failure_is_one_item() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int).validate("x")
    assert len(info.value.errors) == 1
    assert str(info.value) == info.value.message


def test_a_set_reports_its_failures_in_an_order_the_value_fixes() -> None:
    # A set has no positions, so a failing element carries no index and only what
    # it reports distinguishes it. Iteration order is the interpreter's and moves
    # with the hash seed, so the report is ordered by what it says instead.
    schema = Validator(set[int])
    with pytest.raises(ValidationError) as info:
        schema.validate({"d", "b", "a", "c"})
    reported = [str(item["value"]) for item in info.value.errors]
    assert reported == sorted(reported)
    assert len(reported) == 4


def test_a_set_reports_the_same_first_failure_under_fail_fast() -> None:
    # Fail-fast keeps the first of that order rather than the first the
    # interpreter happened to hand over.
    schema = Validator(set[int])
    with pytest.raises(ValidationError) as info:
        schema.validate({"d", "b", "a", "c"}, fail_fast=True)
    assert [item["value"] for item in info.value.errors] == ["'a'"]


def test_the_report_does_not_move_with_the_hash_seed() -> None:
    # The property above holds in one process; this drives two interpreters whose
    # string hashing differs, which is what the seed changes.
    program = textwrap.dedent(
        """
        from valgebra import ValidationError, Validator

        try:
            Validator(set[int]).validate({"d", "b", "a", "c"})
        except ValidationError as error:
            print([item["value"] for item in error.errors])
        """
    )
    runs = {
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3", "4")
    }
    assert len(runs) == 1


def test_a_key_that_is_not_a_string_names_itself_in_full() -> None:
    # A path is made of strings and integers, so a key of another type appears as
    # its repr: it names the key without pretending to be one a caller can index
    # back with. A string key is itself, whole, for the opposite reason.
    with pytest.raises(ValidationError) as info:
        Validator(dict[str, int]).validate({(1, 2): 5})
    assert info.value.path == ("(1, 2)",)

    long_key = "k" * 100
    with pytest.raises(ValidationError) as info:
        Validator(dict[str, int]).validate({long_key: "x"})
    assert info.value.path == (long_key,)
