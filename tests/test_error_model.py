import copy
import json
import pickle

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


def test_every_integer_key_reaches_its_value_from_the_path() -> None:
    """`docs/08` promises a path a caller can walk back down.

    A key that is an `int` must arrive as an `int`, whatever its size: it used
    to be `repr`-ed into the path once it left the range of a machine word, and
    a `bool` was excluded outright -- so `path` held `'1180591620717411303424'`
    and `'True'`, strings that index nothing. Each row below indexes the very
    dict the error came from, which is the property the page states.
    """
    for key in (2, -1, 0, 2**70, -(2**70), True, False):
        holder = {key: "not an int"}
        with pytest.raises(ValidationError) as info:
            Validator(dict[int, int]).validate(holder)
        path = info.value.errors[0]["path"]
        assert isinstance(path, tuple)
        (segment,) = path
        assert isinstance(segment, int), f"{key!r} left the path as {segment!r}"
        assert holder[segment] == "not an int", f"{key!r} does not index back"


def test_a_key_that_is_neither_a_string_nor_an_integer_is_named_not_spelled() -> None:
    """The other half: a path is strings and integers, so the rest is a name."""
    with pytest.raises(ValidationError) as info:
        Validator(dict[float, int]).validate({1.5: "x"})
    path = info.value.errors[0]["path"]
    assert isinstance(path, tuple)
    (segment,) = path
    assert segment == "1.5"
    # A string key that looks like a number stays a string, which is the
    # distinction the integer path exists to preserve.
    with pytest.raises(ValidationError) as info:
        Validator(dict[str, int]).validate({"2": "x"})
    assert info.value.errors[0]["path"] == ("2",)


def test_the_error_model_is_built_when_it_is_asked_for() -> None:
    """A caller that logs `str(error)` should not pay for the whole model.

    Populating the six documented attributes at raise time cost a dict per
    violation, a path per violation and six attribute writes -- and a report
    over 10,000 failing rows spent 26 ms of it whether or not anybody read a
    row. The failures are carried in Rust and the attributes are built by the
    first access that asks, so what a caller does not read is not built.

    Observable rather than timed: the value is on the instance once it has been
    read, and absent before.
    """
    with pytest.raises(ValidationError) as info:
        Validator({"a": int}).validate({"a": "x"})
    error = info.value

    # Nothing built yet, and the carried failures are the reason.
    assert "code" not in vars(error)
    assert "errors" not in vars(error)

    assert error.code == "int_type"
    assert "code" in vars(error), "the built value is cached on the instance"
    assert "errors" not in vars(error), "and only what was asked for is built"

    assert len(error.errors) == 1
    assert "errors" in vars(error)
    # A second read is the same object, not a second build.
    assert error.errors is error.errors


def test_every_documented_attribute_answers_after_the_change() -> None:
    with pytest.raises(ValidationError) as info:
        Validator({"a": int, "b": str}).validate({"a": "x", "b": 1})
    error = info.value
    assert error.code == "int_type"
    assert error.path == ("a",)
    assert error.message == "at a: expected int, got 'x' [int_type]"
    assert error.expected == "int"
    assert error.value == "'x'"
    assert len(error.errors) == 2
    assert str(error).startswith("2 validation errors:")


def test_an_error_built_by_hand_reports_an_empty_model() -> None:
    """The model describes *failures*, and one built by hand has none.

    Class defaults used to say this. They cannot now -- a default makes ordinary
    lookup succeed, so the hook that builds the attributes would never run -- so
    the empty answers come from the hook instead, and the type keeps one shape.
    """
    hand = ValidationError("built by hand")
    assert hand.code == ""
    assert hand.path == ()
    assert hand.message == ""
    assert hand.expected == ""
    assert hand.value == ""
    assert hand.errors == ()
    assert str(hand) == "built by hand"


def test_an_attribute_the_model_does_not_have_is_still_an_attribute_error() -> None:
    """The hook answers six names; everything else must fall through."""
    with pytest.raises(ValidationError) as info:
        Validator(int).validate("x")
    with pytest.raises(AttributeError, match="nonesuch"):
        _ = info.value.nonesuch  # ty: ignore[unresolved-attribute]


def test_the_model_survives_a_process_boundary() -> None:
    """Pickling carries the built data, not the Rust object behind it."""
    with pytest.raises(ValidationError) as info:
        Validator({"a": int, "b": str}).validate({"a": "x", "b": 1})
    error = info.value

    restored = pickle.loads(pickle.dumps(error))  # noqa: S301
    assert restored.errors == error.errors
    assert restored.code == error.code
    assert restored.path == error.path
    assert str(restored) == str(error)
    # And the carrier does not travel: what crossed is the plain model.
    assert "_failures" not in vars(restored)


def test_copying_an_error_keeps_the_model() -> None:
    with pytest.raises(ValidationError) as info:
        Validator({"a": int}).validate({"a": "x"})
    error = info.value
    assert copy.copy(error).errors == error.errors
    assert copy.deepcopy(error).code == error.code
