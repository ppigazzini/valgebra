import json

import pytest

from valgebra import ValidationError, Validator


def test_error_carries_scalar_attributes() -> None:
    with pytest.raises(ValidationError) as info:
        Validator({"user": {"name": str}}).validate({"user": {"name": 5}})
    err = info.value
    assert err.code == "string_type"
    assert err.path == ("user", "name")
    assert err.expected == "str"
    assert err.value == "5"
    assert "string_type" in err.message


def test_errors_tuple_is_structured_and_json_serializable() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int).validate("x")
    err = info.value
    assert isinstance(err.errors, tuple)
    assert len(err.errors) == 1
    item = err.errors[0]
    assert set(item) == {"code", "path", "message", "expected", "value"}
    # round-trips through json.dumps
    decoded = json.loads(json.dumps(err.errors))
    assert decoded[0]["code"] == "int_type"
    assert decoded[0]["path"] == []


def test_scalar_attributes_mirror_the_first_error_item() -> None:
    with pytest.raises(ValidationError) as info:
        Validator([int]).validate([1, "x"])
    err = info.value
    first = err.errors[0]
    assert err.code == first["code"]
    assert err.path == first["path"]
    assert err.message == first["message"]


def test_str_is_the_single_message_for_one_failure() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int).validate("x")
    assert str(info.value) == info.value.message
