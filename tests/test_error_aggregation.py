import pytest

from valgebra import ValidationError, validator


def test_record_aggregates_every_field_failure() -> None:
    schema = validator({"a": int, "b": str, "c": int})
    with pytest.raises(ValidationError) as info:
        schema.validate({"a": "x", "b": 1, "c": "y"})
    codes = [item["code"] for item in info.value.errors]
    assert codes == ["int_type", "string_type", "int_type"]


def test_sequence_aggregates_every_element_failure() -> None:
    with pytest.raises(ValidationError) as info:
        validator([int]).validate([1, "x", 2, "y"])
    paths = [item["path"] for item in info.value.errors]
    assert paths == [(1,), (3,)]


def test_aggregated_str_is_a_counted_summary() -> None:
    with pytest.raises(ValidationError) as info:
        validator({"a": int, "b": int}).validate({"a": "x", "b": "y"})
    summary = str(info.value)
    assert summary.startswith("2 validation errors:")
    assert summary.count("\n") == 2


def test_fail_fast_stops_at_the_first_failure() -> None:
    schema = validator({"a": int, "b": str, "c": int})
    with pytest.raises(ValidationError) as info:
        schema.validate({"a": "x", "b": 1, "c": "y"}, fail_fast=True)
    assert len(info.value.errors) == 1
    assert info.value.code == "int_type"


def test_aggregation_order_is_deterministic() -> None:
    schema = validator({"items": [int], "name": str})
    with pytest.raises(ValidationError) as info:
        schema.validate({"items": [1, "x"], "name": 5})
    paths = [item["path"] for item in info.value.errors]
    assert paths == [("items", 1), ("name",)]


def test_a_single_failure_is_one_item() -> None:
    with pytest.raises(ValidationError) as info:
        validator(int).validate("x")
    assert len(info.value.errors) == 1
    assert str(info.value) == info.value.message
