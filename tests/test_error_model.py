import copy
import dataclasses
import json
import pickle
from typing import Annotated, Literal, NamedTuple, TypedDict

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator, complement, intersection, union


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

    The path is what a caller walks back down to the offending value, so a key
    arriving as text makes a dict keyed by numbers report a location that
    indexes nothing. A string key is itself; anything that is neither a string
    nor an integer has no spelling in a path made of the two, and appears as its
    repr.
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


def test_a_big_integer_key_is_named_by_its_stored_digits() -> None:
    """The digits come from the storage, as a small integer's value does.

    A subclass's `__str__` answers what it likes: a spelling of another key, or
    a raise the path folded into the key's `repr` -- a string, which indexes
    nothing.
    """

    class Lying(int):
        __hash__ = int.__hash__

        def __str__(self) -> str:
            return "7"

    class Raising(int):
        __hash__ = int.__hash__

        def __str__(self) -> str:
            raise KeyboardInterrupt

    for key in (Lying(2**70), Raising(2**70)):
        with pytest.raises(ValidationError) as info:
            Validator(dict[int, int]).validate({key: "x"})
        assert info.value.errors[0]["path"] == (2**70,)


def test_a_string_key_with_no_text_is_named_not_read_as_empty() -> None:
    """A lone surrogate has no UTF-8 spelling, so the key appears as its repr.

    Read as the empty string, it named the entry `d[""]` is, and the two
    failures below carried one path.
    """
    with pytest.raises(ValidationError) as info:
        Validator(dict[str, int]).validate({"\ud800": "x", "": "y"})
    paths = {error["path"] for error in info.value.errors}
    assert paths == {("",), (repr("\ud800"),)}


def test_a_value_a_union_admits_is_never_summarized() -> None:
    """A member builds no report, so nothing reads its repr.

    `validate` explained a union by walking each branch in explaining mode, and
    a branch refusing the value summarized it -- running the value's
    `__repr__`, or a field's, once for each branch before the one that matched.
    A repr that raised `MemoryError` made `validate` raise for a member. The
    union is decided before any branch is explained, whatever the branch's
    kind: a named tuple, a meet, a complement, a list, a set or a tuple before
    the matching branch read the repr as a record or a class once did.
    """
    reprs = []

    class Seen:
        def __repr__(self) -> str:
            reprs.append(self)
            return "Seen()"

    class Unrepresentable:
        def __repr__(self) -> str:
            raise MemoryError

    class Other:
        pass

    @dataclasses.dataclass
    class Point:
        x: int

    class Counts(TypedDict):
        a: int

    class Holds(TypedDict):
        a: Seen

    class Pair(NamedTuple):
        a: int

    for schema, value in [
        (int | Seen, Seen()),
        (list[int | Seen], [Seen(), Seen()]),
        (Literal[1, "a"] | Seen, Seen()),
        (Annotated[list[int], at.MinLen(1)] | Seen, Seen()),
        (dict[str, int] | Seen, Seen()),
        (int | Unrepresentable, Unrepresentable()),
        (Other | Seen, Seen()),
        (Point | Seen, Seen()),
        (Point | Unrepresentable, Unrepresentable()),
        (union({"a": int}, {"a": Seen}), {"a": Seen()}),
        (Counts | Holds, {"a": Seen()}),
        (list[Counts | Holds], [{"a": Seen()}, {"a": 1}]),
        (dict[str, int] | dict[str, Seen], {"k": Seen()}),
        (union({"a": int}, {"a": Unrepresentable}), {"a": Unrepresentable()}),
        (Pair | Seen, Seen()),
        (Pair | Unrepresentable, Unrepresentable()),
        (union(intersection(int, complement(Literal[0])), Seen), Seen()),
        (union(complement(Seen), Seen), Seen()),
        (list[int] | list[Seen], [Seen()]),
        (set[int] | set[Seen], {Seen()}),
        (tuple[int] | tuple[Seen], (Seen(),)),
        (list[int] | list[Unrepresentable], [Unrepresentable()]),
    ]:
        compiled = Validator(schema)
        compiled.validate(value)
        compiled.validate(value, fail_fast=True)
        assert compiled.ensure(value) is value
    assert reprs == []


def test_a_message_names_each_key_on_one_line() -> None:
    """A key that is not a bare name is written as a subscript of its literal.

    Bare, the key `"a.b"` read as the path `a` then `b`, `"[0]"` as the index 0,
    and a key holding a newline broke the one-line message across two. The
    path itself was always exact; the message is what a log line keeps.
    """
    cases = [
        ({"a.b": "x"}, dict[str, int], "at ['a.b']: "),
        ({"a": {"b": "x"}}, {"a": {"b": int}}, "at a.b: "),
        ({"[0]": "x"}, dict[str, int], "at ['[0]']: "),
        ({"": "x"}, dict[str, int], "at ['']: "),
        ({"a\nb": "x"}, dict[str, int], "at ['a\\nb']: "),
        ({"user-id": "x"}, dict[str, int], "at user-id: "),
    ]
    for value, schema, location in cases:
        with pytest.raises(ValidationError) as info:
            Validator(schema).validate(value)
        assert info.value.message.startswith(location), info.value.message
        assert "\n" not in info.value.message


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

    Class defaults cannot say it: a default makes ordinary lookup succeed, so
    the hook that builds the attributes would never run. The empty answers come
    from the hook instead, and the type keeps one shape.
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


def test_a_failure_raised_while_handling_another_chains_it() -> None:
    """A failure is raised the way a Python `raise` inside a handler is.

    The exception being handled is the failure's `__context__`, and no
    `__cause__` is claimed; outside a handler there is no context at all.
    """
    validator = Validator({"a": int})
    handled = KeyError("outer")

    def validate_while_handling() -> None:
        try:
            raise handled
        except KeyError:
            validator.validate({"a": "x"})

    with pytest.raises(ValidationError) as info:
        validate_while_handling()
    assert info.value.__context__ is handled
    assert info.value.__cause__ is None
    assert not info.value.__suppress_context__
    assert info.value.code == "int_type"

    with pytest.raises(ValidationError) as info:
        validator.validate({"a": "x"})
    assert info.value.__context__ is None
