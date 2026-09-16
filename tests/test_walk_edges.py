"""Values at the edge of what the walk reads, and the answer each gets.

The node matrix asks every node about a representative value. These are the
values that are not representative: a container of the wrong flavour, a
subclass that answers a question by running code, a value that contains itself,
a refinement over a carrier arithmetic reaches and the kinds do not.

Each row is here because nothing else in the suite asks it, and because the
answer is one a reader would otherwise have to guess at. Where the answer is a
refusal, the code is asserted too: a value refused for the wrong reason is a
message that sends a reader somewhere else.
"""

from __future__ import annotations

import collections
import datetime
import decimal
import enum
import fractions
from typing import Annotated, Literal

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator, complement, recursive, union


def _codes(schema: object, value: object) -> list[object]:
    with pytest.raises(ValidationError) as caught:
        Validator(schema).validate(value)
    return [item["code"] for item in caught.value.errors]


class Level(enum.IntEnum):
    LOW = 1


class MyInt(int):
    __slots__ = ()


class RaisingKey(str):
    """A key that answers no comparison."""

    __slots__ = ()
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("no comparison")


def test_a_buffer_is_not_a_bytes() -> None:
    """`bytes` is the kind, not everything shaped like it."""
    assert Validator(bytes).is_valid(bytearray(b"a")) is False
    assert Validator(bytes).is_valid(memoryview(b"a")) is False
    assert _codes(bytes, bytearray(b"a")) == ["bytes_type"]
    # And neither is a document the JSON entry points read.
    with pytest.raises(TypeError):
        Validator(int).validate_json(bytearray(b"1"))  # ty: ignore[invalid-argument-type]
    assert Validator(int).is_valid_json(bytearray(b"1")) is False  # ty: ignore[invalid-argument-type]


def test_a_dict_subclass_is_read_through_its_storage() -> None:
    """The mapping kinds a program really holds."""
    assert Validator(dict[str, int]).is_valid(collections.OrderedDict(a=1)) is True
    assert Validator(dict[str, int]).is_valid(collections.OrderedDict(a="x")) is False
    counted = collections.defaultdict(int, a=1)
    assert Validator(dict[str, int]).is_valid(counted) is True
    assert Validator({"a": int}).is_valid(collections.OrderedDict(a=1)) is True


def test_a_key_that_answers_no_comparison_is_a_key_the_record_lacks() -> None:
    """A dict whose key cannot be compared is a dict, and not a member.

    The comparison-raises rule makes the value a non-member. What it must not
    do is report the *dict* as the wrong type: the value is a dict, and a
    reader told otherwise looks at the wrong thing.
    """
    value = {RaisingKey("name"): "Ada"}
    assert Validator({"name": str}).is_valid(value) is False
    codes = _codes({"name": str}, value)
    assert "dict_type" not in codes, codes
    assert "missing_key" in codes, codes


def test_a_recursive_schema_under_a_complement_walks() -> None:
    """A reference below a negation is a set, and membership reads it."""
    never_a_list = recursive(lambda t: complement(list[t]))  # ty: ignore[invalid-type-form]
    assert never_a_list.is_valid(1) is True
    assert never_a_list.is_valid([]) is False
    assert _codes(never_a_list, []) == ["unexpected_match"]


def test_a_value_that_contains_itself_is_read_where_no_reference_does() -> None:
    """The identity guard is the recursive walk's, and nothing else asks it."""
    cycle: list = []
    cycle.append(cycle)
    # No reference to unfold, so nothing re-enters the value and it is a member.
    assert Validator(list[object]).is_valid(cycle) is True
    assert complement(list[int]).is_valid(cycle) is True
    # Under a reference the guard refuses it rather than looping.
    loop = recursive(lambda t: list[t])  # ty: ignore[invalid-type-form]
    assert _codes(loop, cycle) == ["recursion_loop"]


def test_a_predicate_answers_with_what_python_calls_truth() -> None:
    """A predicate is asked whether it returned truthy, not whether it is a bool.

    The marker's own annotation says a predicate answers `bool`; the page says
    valgebra "checks that it returned truthy", and a caller's predicate is
    ordinary Python that may answer with anything.
    """

    def wordy(_value: object) -> object:
        return "no"

    def nothing_at_all(_value: object) -> object:
        return []

    truthy = Annotated[int, at.Predicate(wordy)]  # ty: ignore[invalid-argument-type]
    falsy = Annotated[int, at.Predicate(nothing_at_all)]  # ty: ignore[invalid-argument-type]
    assert Validator(truthy).is_valid(1) is True
    assert Validator(falsy).is_valid(1) is False
    assert _codes(falsy, 1) == ["predicate_failed"]


def test_a_predicate_may_ask_a_validator_of_its_own() -> None:
    """The slow path is Python, so a predicate re-entering the library works."""
    inner = Validator(int)
    outer = Validator(Annotated[object, at.Predicate(inner.is_valid)])
    assert outer.is_valid(1) is True
    assert outer.is_valid("a") is False


def test_a_generator_exit_is_fatal_like_the_rest() -> None:
    """`is_fatal` names it, the page names it, and nothing drove it."""

    def leaves(_value: object) -> bool:
        raise GeneratorExit

    schema = Annotated[int, at.Predicate(leaves)]
    with pytest.raises(GeneratorExit):
        Validator(schema).validate(1)
    with pytest.raises(GeneratorExit):
        Validator(schema).is_valid(1)


# Each row: a name, a refinement over a carrier that is not a builtin scalar,
# a member, a non-member, and the code the refusal carries.
_CARRIERS: list[tuple[str, object, object, object, str]] = [
    (
        "a decimal bound",
        Annotated[decimal.Decimal, at.Ge(decimal.Decimal(1))],
        decimal.Decimal(2),
        decimal.Decimal(0),
        "greater_than_equal",
    ),
    (
        "a fraction bound",
        Annotated[fractions.Fraction, at.Ge(fractions.Fraction(1))],
        fractions.Fraction(2),
        fractions.Fraction(0),
        "greater_than_equal",
    ),
    (
        "a length over a set",
        Annotated[set[int], at.MinLen(2)],
        {1, 2},
        {1},
        "too_short",
    ),
    (
        "a length over a dict",
        Annotated[dict[str, int], at.MaxLen(1)],
        {"a": 1},
        {"a": 1, "b": 2},
        "too_long",
    ),
    (
        "a length over bytes",
        Annotated[bytes, at.MinLen(2)],
        b"ab",
        b"a",
        "too_short",
    ),
]


@pytest.mark.parametrize(
    ("name", "schema", "member", "outsider", "code"),
    _CARRIERS,
    ids=[row[0] for row in _CARRIERS],
)
def test_a_refinement_reads_the_carrier_it_is_written_over(
    name: str, schema: object, member: object, outsider: object, code: str
) -> None:
    """A bound is Python's operator, so it reaches past the builtin scalars."""
    assert Validator(schema).is_valid(member) is True, name
    assert Validator(schema).is_valid(outsider) is False, name
    assert _codes(schema, outsider) == [code], name


def test_a_step_that_is_not_a_number_is_refused() -> None:
    """A `MultipleOf` denotes `value % n == 0`, which asks `n` to be a number.

    `timedelta(4) % timedelta(2)` is `timedelta(0)`, and that does not equal the
    integer zero -- so a step written as a duration names a schema no value
    belongs to, whatever the caller meant by it. That is refused at build, as a
    step of zero already is, rather than compiled into a validator that refuses
    everything without saying why.
    """
    with pytest.raises(ValueError, match=r"number"):
        Validator(
            Annotated[datetime.timedelta, at.MultipleOf(datetime.timedelta(seconds=2))]
        )
    # The numeric steps are unmoved, across the carriers arithmetic reaches.
    assert Validator(Annotated[int, at.MultipleOf(2)]).is_valid(4) is True
    assert Validator(Annotated[float, at.MultipleOf(0.5)]).is_valid(1.5) is True
    step = Annotated[decimal.Decimal, at.MultipleOf(decimal.Decimal("0.5"))]
    assert Validator(step).is_valid(decimal.Decimal("1.5")) is True
    assert Validator(step).is_valid(decimal.Decimal("1.6")) is False
    fraction = Annotated[fractions.Fraction, at.MultipleOf(fractions.Fraction(1, 2))]
    assert Validator(fraction).is_valid(fractions.Fraction(3, 2)) is True


def test_an_enumeration_member_is_its_kind_and_not_its_value() -> None:
    """`IntEnum` is an `int`, and a literal pins the type as well as the value."""
    assert Validator(int).is_valid(Level.LOW) is True
    assert Validator(Literal[1]).is_valid(Level.LOW) is False
    assert Validator(Literal[Level.LOW]).is_valid(Level.LOW) is True


def test_a_scalar_subclass_is_its_kind_and_not_its_literal() -> None:
    """The same rule for a subclass, through the plan a literal union builds."""
    assert Validator(int).is_valid(MyInt(1)) is True
    assert Validator(Literal[1]).is_valid(MyInt(1)) is False
    # The wide form takes a different path inside, and gives the same answer.
    assert Validator(Literal[1, 2, 3]).is_valid(MyInt(1)) is False
    wide = union(*[Literal[i] for i in range(80)])  # ty: ignore[invalid-type-form]
    assert Validator(wide).is_valid(MyInt(1)) is False
