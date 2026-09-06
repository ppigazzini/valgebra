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


def test_an_integer_key_stays_an_integer_in_the_path() -> None:
    """`d[2]` and `d["2"]` are different entries, and the path says which.

    The path is what a caller walks back down to the offending value, and every
    key used to arrive as text -- so a dict keyed by numbers reported a location
    that indexed nothing. A string key is still itself; anything that is neither
    a string nor an integer has no spelling in a path made of the two, and
    appears as its repr.
    """
    numbered = Validator(dict[int, int])
    with pytest.raises(ValidationError) as failure:
        numbered.validate({1: 1, 2: "x"})
    (error,) = failure.value.errors
    path = error["path"]
    assert path == (2,)
    assert "at [2]" in str(error["message"])

    # And the value the path names is reachable by walking it, which is the
    # whole claim: the items are typed `object` in the error model, so the walk
    # is written the way a caller writes it.
    walked: object = {1: 1, 2: "x"}
    assert isinstance(path, tuple)
    for step in path:
        assert isinstance(walked, dict)
        walked = walked[step]
    assert walked == "x"

    # A string key is unchanged, and the two do not collide.
    text = Validator(dict[str, int])
    with pytest.raises(ValidationError) as failure:
        text.validate({"2": "x"})
    assert failure.value.errors[0]["path"] == ("2",)

    # A key that is neither names itself without pretending to be walkable.
    with pytest.raises(ValidationError) as failure:
        text.validate({1.5: 1})
    assert failure.value.errors[0]["path"] == ("1.5",)
