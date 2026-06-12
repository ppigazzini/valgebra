import pytest

from valgebra import ValidationError, validator


def test_record_accepts_a_matching_dict() -> None:
    user = validator({"name": str, "age": int})
    assert user.is_valid({"name": "Ada", "age": 36})


def test_record_optional_key_may_be_absent() -> None:
    user = validator({"name": str, "age?": int})
    assert user.is_valid({"name": "Ada"})
    assert user.is_valid({"name": "Ada", "age": 36})


def test_record_optional_key_is_checked_when_present() -> None:
    user = validator({"name": str, "age?": int})
    assert not user.is_valid({"name": "Ada", "age": "old"})


def test_record_required_key_must_be_present() -> None:
    user = validator({"name": str, "age": int})
    with pytest.raises(ValidationError) as info:
        user.validate({"name": "Ada"})
    assert info.value.code == "missing_key"
    assert info.value.path == ("age",)


def test_record_is_closed_by_default() -> None:
    user = validator({"name": str})
    with pytest.raises(ValidationError) as info:
        user.validate({"name": "Ada", "extra": 1})
    assert info.value.code == "extra_key"


def test_empty_record_matches_only_the_empty_dict() -> None:
    empty = validator({})
    assert empty.is_valid({})
    assert not empty.is_valid({"a": 1})


def test_record_rejects_a_non_dict() -> None:
    assert not validator({"name": str}).is_valid(["name"])


def test_nested_record_failure_reports_the_path() -> None:
    schema = validator({"user": {"name": str}})
    with pytest.raises(ValidationError) as info:
        schema.validate({"user": {"name": 5}})
    assert info.value.code == "string_type"
    assert info.value.path == ("user", "name")
