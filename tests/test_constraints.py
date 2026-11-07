"""Every refinement constraint, on both the fast path and the explain path.

Covers each annotated-types marker valgebra reads, the boolean result and the
error code on failure, and the rendered form. A dropped constraint silently
admits invalid values, so every constraint is checked in both directions.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
import operator
import sys
import typing
from decimal import Decimal, FloatOperation, localcontext
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated, Any, Literal

import annotated_types as at
import pytest
import typing_extensions
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Regex,
    ValidationError,
    Validator,
    complement,
    intersection,
    nothing,
    union,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _code(spec: object, value: object) -> str | None:
    """Return the first error code if validation raises, else None."""
    try:
        Validator(spec).validate(value)
    except ValidationError as err:
        return err.code
    return None


# (label, spec, accepted value, rejected value, failure code, repr fragment).
CONSTRAINTS = [
    ("ge", Annotated[int, at.Ge(0)], 0, -1, "greater_than_equal", "Ge(0)"),
    ("gt", Annotated[int, at.Gt(0)], 1, 0, "greater_than", "Gt(0)"),
    ("le", Annotated[int, at.Le(10)], 10, 11, "less_than_equal", "Le(10)"),
    ("lt", Annotated[int, at.Lt(10)], 9, 10, "less_than", "Lt(10)"),
    ("min_len", Annotated[str, at.MinLen(2)], "ab", "a", "too_short", "MinLen(2)"),
    ("max_len", Annotated[str, at.MaxLen(2)], "ab", "abc", "too_long", "MaxLen(2)"),
    (
        "multiple_of",
        Annotated[int, at.MultipleOf(3)],
        9,
        5,
        "multiple_of",
        "MultipleOf(3)",
    ),
]


@pytest.mark.parametrize(
    ("label", "spec", "good", "bad", "code", "shown"),
    CONSTRAINTS,
    ids=[c[0] for c in CONSTRAINTS],
)
def test_constraint_accepts_rejects_codes_and_repr(  # noqa: PLR0913, PLR0917
    label: str,
    spec: object,
    good: object,
    bad: object,
    code: str,
    shown: str,
) -> None:
    v = Validator(spec)
    assert v.is_valid(good)
    assert not v.is_valid(bad)
    assert _code(spec, good) is None
    assert _code(spec, bad) == code
    assert shown in repr(v)


def test_multiple_of_handles_negatives_and_floats() -> None:
    assert Validator(Annotated[int, at.MultipleOf(3)]).is_valid(-6)
    assert Validator(Annotated[float, at.MultipleOf(0.5)]).is_valid(1.5)
    assert not Validator(Annotated[float, at.MultipleOf(0.5)]).is_valid(1.3)


@pytest.mark.parametrize(
    ("spec", "member", "non_member"),
    [
        pytest.param(
            Annotated[int | float, at.MultipleOf(0.5)],
            1,
            1.25,
            id="int-against-a-float-divisor",
        ),
        pytest.param(
            Annotated[int | float, at.MultipleOf(Fraction(1, 2))],
            1,
            Fraction(1, 4),
            id="int-against-a-fraction",
        ),
        pytest.param(
            Annotated[int | float, at.MultipleOf(Decimal(1))],
            3,
            Decimal("0.5"),
            id="int-against-a-decimal",
        ),
    ],
)
def test_a_divisor_of_another_type_divides(
    spec: object,
    member: object,
    non_member: object,
) -> None:
    # `%` is the operator, and half of what it means lives on the *operand*: a
    # value that does not know the divisor defers to its `__rmod__`. Reading only
    # the value's `__mod__` reports `NotImplemented`, which is truthy and so reads
    # as a non-zero remainder -- every such pair a non-multiple.
    schema = Validator(spec)
    assert schema.is_valid(member)
    assert not schema.is_valid(non_member)


def _fresh(build: Callable[[], object]) -> Validator:
    """Build a validator of `build()` with `typing`'s memo of `Annotated` cleared.

    `typing` memoizes `Annotated[...]` by equality, and `MultipleOf(6.0)` equals
    `MultipleOf(6)`, so a form built earlier would carry the other step.
    """
    for cleanup in getattr(typing, "_cleanups", ()):
        cleanup()
    return Validator(build())


@pytest.mark.parametrize(
    ("narrow", "wide", "witness"),
    [
        pytest.param(
            lambda: Annotated[int, at.MultipleOf(6.0)],
            lambda: Annotated[int, at.MultipleOf(3)],
            2**53 + 5,
            id="an-int-read-as-the-float-nearest-it",
        ),
        pytest.param(
            lambda: Annotated[int, at.MultipleOf(4)],
            lambda: Annotated[int, at.MultipleOf(0.5)],
            2**1030,
            id="an-int-no-float-holds",
        ),
        pytest.param(
            lambda: Annotated[float, at.MultipleOf(2**53 + 1)],
            lambda: Annotated[float, at.MultipleOf(3)],
            2.0**53,
            id="a-step-read-as-the-float-nearest-it",
        ),
        pytest.param(
            lambda: Annotated[Decimal, at.MultipleOf(4)],
            lambda: Annotated[Decimal, at.MultipleOf(2)],
            Decimal("2E+28"),
            id="a-quotient-past-the-decimal-precision",
        ),
        pytest.param(
            lambda: Annotated[object, at.MultipleOf(4)],
            lambda: Annotated[object, at.MultipleOf(2)],
            Decimal("2E+28"),
            id="a-base-holding-any-number",
        ),
    ],
)
def test_a_step_dividing_another_is_not_read_across_kinds_of_number(
    narrow: Callable[[], object],
    wide: Callable[[], object],
    witness: object,
) -> None:
    """Dividing is about the steps, and `%` reads each value in its own kind."""
    narrower, wider = _fresh(narrow), _fresh(wide)
    assert narrower.is_valid(witness)
    assert not wider.is_valid(witness)
    assert narrower.relation_to(wider) == "undecided"


@pytest.mark.parametrize(
    ("narrow", "wide"),
    [
        pytest.param(
            lambda: Annotated[int, at.MultipleOf(10**30)],
            lambda: Annotated[int, at.MultipleOf(10**10)],
            id="two-ints-past-the-machine-word",
        ),
        pytest.param(
            lambda: Annotated[float, at.MultipleOf(4)],
            lambda: Annotated[float, at.MultipleOf(0.5)],
            id="an-int-and-a-float-over-the-floats",
        ),
        pytest.param(
            lambda: Annotated[int | float, at.MultipleOf(4)],
            lambda: Annotated[int | float, at.MultipleOf(2)],
            id="two-ints-over-both",
        ),
    ],
)
def test_a_step_dividing_another_of_one_kind_of_number_is_below_it(
    narrow: Callable[[], object],
    wide: Callable[[], object],
) -> None:
    """Where `%` reads a value and both steps alike, dividing decides it."""
    assert _fresh(narrow).relation_to(_fresh(wide)) == "subset"


@pytest.mark.parametrize(
    ("spec", "missing"),
    [
        pytest.param(Annotated[int, at.MinLen(1)], "length", id="length-on-a-number"),
        pytest.param(Annotated[int, Regex("a")], "text", id="pattern-on-a-number"),
        pytest.param(Annotated[bytes, Regex("a")], "text", id="pattern-on-bytes"),
        pytest.param(Annotated[str, at.MultipleOf(3)], "number", id="divisor-on-text"),
        pytest.param(Annotated[int, at.Ge("a")], "order", id="bound-of-another-kind"),
        pytest.param(Annotated[str, at.Ge(1)], "order", id="bound-on-text"),
    ],
)
def test_a_constraint_the_base_cannot_answer_is_refused(
    spec: object,
    missing: str,
) -> None:
    # Asking a value something it cannot answer raises, and the walk reads a raise
    # as a non-member: the schema would admit nothing at all and say nothing about
    # why. That is not a schema anyone writes on purpose.
    with pytest.raises(NotImplementedError, match=missing):
        Validator(spec)


def test_a_constraint_one_branch_can_answer_narrows_rather_than_empties() -> None:
    # A union where some member carries the constraint is a narrowing of the
    # union, which is a set a reader can mean.
    schema = Validator(Annotated[int | str, at.MinLen(1)])
    assert schema.is_valid("a")
    assert not schema.is_valid(5)


def test_an_opaque_base_keeps_its_constraint() -> None:
    # A class may define whatever the constraint asks for, so the check stands
    # aside rather than guessing.
    assert Validator(Annotated[object, at.MinLen(1)]).is_valid("ab")


def test_multiple_of_zero_is_rejected_at_build() -> None:
    # No value is a multiple of zero, and checking one divides by zero, so the
    # unsatisfiable schema is rejected when compiled rather than swallowing a
    # ZeroDivisionError as a non-match at validation time.
    with pytest.raises(ValueError, match="MultipleOf"):
        Validator(Annotated[int, at.MultipleOf(0)])
    with pytest.raises(ValueError, match="MultipleOf"):
        Validator(Annotated[float, at.MultipleOf(0.0)])


def test_interval_marker_contributes_both_bounds() -> None:
    v = Validator(Annotated[int, at.Interval(ge=0, le=10)])
    assert v.is_valid(0)
    assert v.is_valid(10)
    assert not v.is_valid(-1)
    assert not v.is_valid(11)


def test_len_marker_contributes_min_and_max() -> None:
    v = Validator(Annotated[str, at.Len(2, 4)])
    assert v.is_valid("ab")
    assert v.is_valid("abcd")
    assert not v.is_valid("a")
    assert not v.is_valid("abcde")


def test_several_markers_combine() -> None:
    v = Validator(Annotated[int, at.Interval(ge=0, le=20), at.MultipleOf(5)])
    assert v.is_valid(10)
    assert not v.is_valid(7)
    assert not v.is_valid(25)


def test_unrecognized_marker_is_ignored_per_spec() -> None:
    # A non-constraint marker carries no membership meaning and is dropped, as
    # the typing spec directs for unrecognized Annotated metadata.
    v = Validator(Annotated[int, "just documentation"])
    assert v.is_valid(5)
    assert repr(v) == "int"


def test_predicate_passes_and_fails() -> None:
    v = Validator(Annotated[int, at.Predicate(lambda x: x % 2 == 0)])
    assert v.is_valid(4)
    assert not v.is_valid(3)
    assert (
        _code(Annotated[int, at.Predicate(lambda x: x % 2 == 0)], 3)
        == "predicate_failed"
    )
    assert "Predicate(...)" in repr(v)


def test_predicate_that_raises_is_reported_distinctly() -> None:
    def boom(_: object) -> bool:
        raise RuntimeError

    spec = Annotated[int, at.Predicate(boom)]
    assert not Validator(spec).is_valid(5)
    assert _code(spec, 5) == "predicate_error"


def test_refinement_reports_the_base_failure_before_constraints() -> None:
    # A value failing the base type never reaches the constraints.
    assert _code(Annotated[int, at.Ge(0)], "x") == "int_type"


def test_refinement_failure_reports_the_path() -> None:
    with pytest.raises(ValidationError) as info:
        Validator({"age": Annotated[int, at.Ge(0)]}).validate({"age": -1})
    assert info.value.code == "greater_than_equal"
    assert info.value.path == ("age",)


class _LyingList(list):  # type: ignore[type-arg]
    """A ``list`` whose ``__len__`` disagrees with what it holds."""

    def __len__(self) -> int:
        return 10


class _LyingTuple(tuple):  # type: ignore[type-arg]
    """The same, for a ``tuple``."""

    __slots__ = ()

    def __len__(self) -> int:
        return 10


def test_a_sequence_length_is_what_the_value_holds() -> None:
    """A value has one length, and for a sequence it is the storage.

    A ``list`` subclass may override ``__len__`` and say anything. The sequence
    walk counts what the value holds, so a length marker that believed
    ``__len__`` instead would put the value in ``MinLen(5)`` and outside the
    five-element shape though both were written to mean one narrowing. A
    membership that depends on which half of a schema asks is not a set.
    """
    one_item = _LyingList([1])
    assert list.__len__(one_item) == 1, "it holds one"
    assert len(one_item) == 10, "and reports ten"

    assert not Validator(Annotated[list[int], at.MinLen(5)]).is_valid(one_item)
    assert Validator(Annotated[list[int], at.MaxLen(2)]).is_valid(one_item)
    assert Validator(Annotated[list[int], at.MinLen(1)]).is_valid(one_item)
    # The shape counts the same items, which is the agreement the rule is for.
    assert Validator([int]).is_valid(one_item)

    assert not Validator(Annotated[tuple[int, ...], at.MinLen(5)]).is_valid(
        _LyingTuple((1,))
    )


def test_a_tuple_shape_walks_the_storage_and_not_the_dunder() -> None:
    """The shape reads the elements the value holds, on every interpreter.

    The marker above is one half; the walk over ``tuple[int, ...]`` is the
    other, and it read its element count from ``PyTuple_Size``. That is the
    storage on CPython and the object's own ``__len__`` on PyPy's ``cpyext``,
    so a subclass reporting ten over one element made the walk read nine slots
    past the end of the allocation -- which is a segfault rather than an answer.

    The rows below are membership, and they are also the crash: a process that
    dies here fails them by not reaching them.
    """
    one_int = _LyingTuple((1,))
    assert len(one_int) == 10, "it reports ten"
    assert tuple.__len__(one_int) == 1, "and holds one"

    assert Validator(tuple[int, ...]).is_valid(one_int)
    assert not Validator(tuple[str, ...]).is_valid(one_int)
    # A fixed shape counts the storage too, so one element is not two.
    assert Validator(tuple[int]).is_valid(one_int)
    assert not Validator(tuple[int, int]).is_valid(one_int)
    assert Validator(tuple[int, str]).is_valid(_LyingTuple((1, "a")))
    # The explaining walk answers over the same elements as the fast one.
    with pytest.raises(ValidationError) as raised:
        Validator(tuple[str, ...]).validate(one_int)
    assert raised.value.path == (0,)
    assert raised.value.code == "string_type"


def test_every_other_kind_still_answers_with_its_dunder() -> None:
    """The rule is one length per value, not distrust of ``__len__``.

    A ``str``, ``bytes``, ``set`` and ``dict`` are read through ``__len__`` by
    the rest of the walk too, so that is their length and nothing changes.
    """
    assert Validator(Annotated[str, at.MinLen(5)]).is_valid("abcdef")
    assert not Validator(Annotated[str, at.MinLen(5)]).is_valid("ab")
    assert Validator(Annotated[bytes, at.MaxLen(2)]).is_valid(b"ab")
    assert Validator(Annotated[set[int], at.MinLen(2)]).is_valid({1, 2})
    assert Validator(Annotated[dict[str, int], at.MinLen(2)]).is_valid({"a": 1, "b": 2})


NAN_BOUNDS = ["ge", "gt", "le", "lt"]


@pytest.mark.parametrize("bound", NAN_BOUNDS)
def test_an_order_bound_against_nan_is_refused(bound: str) -> None:
    """`nan` is not a bound; it is the absence of an order.

    Every comparison with `nan` is false, so `Annotated[float, Ge(nan)]` admits
    no value at all -- the empty set written as a bound, which no caller means
    and which neither decider proves empty. Dropping the marker instead would
    admit every float, so the refusal is what says which mistake was made.
    """
    marker = getattr(at, {"ge": "Ge", "gt": "Gt", "le": "Le", "lt": "Lt"}[bound])
    with pytest.raises(ValueError, match="cannot be nan"):
        Validator(Annotated[float, marker(float("nan"))])


def test_a_multiple_of_nan_is_refused() -> None:
    with pytest.raises(ValueError, match="MultipleOf\\(nan\\)"):
        Validator(Annotated[float, at.MultipleOf(float("nan"))])


@pytest.mark.parametrize("text", ["sNaN", "-sNaN", "NaN"])
def test_a_decimal_nan_is_refused_as_a_float_nan_is(text: str) -> None:
    """A `Decimal` not-a-number is the absence of an order too, signaling or not.

    `Decimal("sNaN")` refuses the conversion to a float, and every comparison
    or remainder with it raises `InvalidOperation`, which the walk reads as a
    refusal: as a bound or a step it admits no value, as `nan` does.
    """
    nan = Decimal(text)
    for marker in (at.Ge, at.Gt, at.Le, at.Lt):
        with pytest.raises(ValueError, match="cannot be nan"):
            Validator(Annotated[int, marker(nan)])
    with pytest.raises(ValueError, match="MultipleOf\\(nan\\)"):
        Validator(Annotated[Decimal, at.MultipleOf(nan)])


@pytest.mark.parametrize("bound", NAN_BOUNDS)
def test_an_order_bound_against_a_complex_number_is_refused(bound: str) -> None:
    """A complex number orders nothing either: `1 >= 1j` raises, as `1 >= 0j` does.

    So the bound is the empty set written as a bound, whatever the base: the
    `nan` case by another road, refused the same way.
    """
    marker = getattr(at, {"ge": "Ge", "gt": "Gt", "le": "Le", "lt": "Lt"}[bound])
    with pytest.raises(ValueError, match="orders no value against a complex one"):
        Validator(Annotated[float, marker(1j)])
    with pytest.raises(ValueError, match="orders no value against a complex one"):
        Validator(Annotated[object, marker(1j)])


def test_a_complex_step_is_refused() -> None:
    """Python takes no remainder against a complex number: `4 % 2j` raises."""
    with pytest.raises(ValueError, match="no remainder against a complex number"):
        Validator(Annotated[int, at.MultipleOf(2j)])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="no remainder against a complex number"):
        Validator(Annotated[object, at.MultipleOf(2j)])  # ty: ignore[invalid-argument-type]


_COMPARISONS = (
    (at.Ge, operator.ge),
    (at.Gt, operator.gt),
    (at.Le, operator.le),
    (at.Lt, operator.lt),
)


def _beside(value: Fraction | Decimal) -> list[float]:
    nearest = float(value)
    down = math.nextafter(nearest, -math.inf)
    up = math.nextafter(nearest, math.inf)
    return [
        math.nextafter(down, -math.inf),
        down,
        nearest,
        up,
        math.nextafter(up, math.inf),
    ]


@given(
    rational=st.one_of(
        st.fractions(min_value=-(10**6), max_value=10**6, max_denominator=10**4),
        st.decimals(
            min_value=-(10**6),
            max_value=10**6,
            places=12,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    which=st.sampled_from(_COMPARISONS),
)
def test_a_rational_bound_over_a_float_is_the_floats_it_compares_with(
    rational: Fraction | Decimal, which: tuple[Any, Any]
) -> None:
    # The walk and Python compare exactly, so the set the bound reads as is the one
    # the floats beside it spell, in both directions.
    marker, compare = which
    bound = Validator(Annotated[float, marker(rational)])
    near = _beside(rational)
    for x in [*near, math.inf, -math.inf, 0.0, -0.0]:
        assert bound.is_valid(x) == compare(x, rational), x
    assert not bound.is_valid(math.nan)
    kept = [x for x in near if compare(x, rational)]
    if marker in (at.Ge, at.Gt):
        same = Annotated[float, at.Ge(min(kept))]
    else:
        same = Annotated[float, at.Le(max(kept))]
    assert bound.relation_to(same) == "subset"
    assert Validator(same).relation_to(Annotated[float, marker(rational)]) == "subset"


@pytest.mark.parametrize(
    ("narrow", "wide", "expected"),
    [
        pytest.param(
            at.Ge(0.25), at.Ge(Decimal("0.5")), "not_subset", id="a-float-below"
        ),
        pytest.param(at.Ge(Decimal("0.5")), at.Ge(0.5), "subset", id="one-float"),
        pytest.param(at.Ge(0.5), at.Ge(Decimal("0.5")), "subset", id="and-back"),
        # The float 0.1 is a little above the decimal, so a bound on that holds it.
        pytest.param(
            at.Ge(Decimal("0.1")), at.Gt(0.1), "not_subset", id="the-float-above"
        ),
        pytest.param(at.Ge(0.1), at.Ge(Decimal("0.1")), "subset", id="the-float-holds"),
        pytest.param(
            at.Gt(Decimal("0.1")), at.Ge(0.1), "subset", id="strictness-costs-nothing"
        ),
        # The float 1/3 is a little below the fraction.
        pytest.param(
            at.Ge(Fraction(1, 3)), at.Gt(1 / 3), "subset", id="the-float-below"
        ),
        pytest.param(
            at.Le(Fraction(1, 3)), at.Le(1 / 3), "subset", id="and-its-other-side"
        ),
    ],
)
def test_a_rational_bound_over_a_float_is_decided(
    narrow: object, wide: object, expected: str
) -> None:
    subject = Validator(Annotated[float, narrow])
    assert subject.relation_to(Annotated[float, wide]) == expected


def test_no_float_lies_between_a_decimal_and_the_float_next_to_it() -> None:
    exact = Decimal("0.1000000000000000055511151231257827021181583404541015625")
    between = intersection(
        Annotated[float, at.Gt(Decimal("0.1"))], Annotated[float, at.Lt(exact)]
    )
    assert Validator(between).relation_to(nothing) == "subset"
    assert (
        Validator(Annotated[int | float, at.Ge(Decimal("0.5"))]).relation_to(
            union(Annotated[int, at.Ge(1)], Annotated[float, at.Ge(0.5)])
        )
        == "subset"
    )
    # Under the trap the comparison raises, so the operand is not read, and the
    # relation declines rather than raising. A value no bound above equals, since
    # `typing` caches an `Annotated` by its metadata's `==`, and `Ge(0.5)` is
    # `Ge(Decimal("0.5"))` to that cache.
    with localcontext() as context:
        context.traps[FloatOperation] = True
        trapped = Validator(Annotated[float, at.Ge(0.25)])
        assert (
            trapped.relation_to(Annotated[float, at.Ge(Decimal("0.625"))])
            == "undecided"
        )


def test_a_decimal_step_divides_integers_and_no_float() -> None:
    """`3 % Decimal("0.1")` is `Decimal("0.0")`, and `1.5 % Decimal("0.1")` raises.

    A float and a `Decimal` are the one pair of numbers `%` refuses to mix, so
    the step over a float base is the empty set; over an integer it divides,
    and over a union holding an integer it narrows, as every constraint one
    member answers does.
    """
    with pytest.raises(
        NotImplementedError, match="refuses to mix a float and a `Decimal`"
    ):
        Validator(Annotated[float, at.MultipleOf(Decimal("0.1"))])
    tenths = Validator(Annotated[int, at.MultipleOf(Decimal("0.1"))])
    assert tenths.is_valid(3)
    both = Validator(Annotated[int | float, at.MultipleOf(Decimal("0.1"))])
    assert both.is_valid(3)
    assert not both.is_valid(0.5)


def test_a_marker_equal_to_a_real_one_is_refused_wherever_it_arrives() -> None:
    """A refusal of a complex or `Decimal` marker also meets the real one it equals.

    `MultipleOf(2 + 0j)` equals `MultipleOf(2)`, and `MultipleOf(Decimal("0.5"))`
    equals `MultipleOf(0.5)`, so `typing` caches one `Annotated` for both and
    hands it to whichever spelling comes second. A truth value in a length marker
    is read rather than refused for that reason
    (`test_a_truth_value_is_read_as_the_length_it_equals`), because the value it
    holds reads as the length it equals. These do not: Python's `%` and `>=`
    raise on them, so the schema the second spelling meets admits nothing either
    way. Refused, the caller who wrote the real marker is told so, and the
    message says how the other one arrived.
    """
    collisions = [
        (
            lambda: Annotated[int, at.MultipleOf(2 + 0j)],  # ty: ignore[invalid-argument-type]
            lambda: Annotated[int, at.MultipleOf(2)],
            ValueError,
        ),
        (
            lambda: Annotated[float, at.MultipleOf(Decimal("0.5"))],
            lambda: Annotated[float, at.MultipleOf(0.5)],
            NotImplementedError,
        ),
        (
            lambda: Annotated[float, at.Ge(3 + 0j)],  # ty: ignore[invalid-argument-type]
            lambda: Annotated[float, at.Ge(3)],
            ValueError,
        ),
        (
            lambda: Annotated[str, at.MaxLen(2 + 0j)],  # ty: ignore[invalid-argument-type]
            lambda: Annotated[str, at.MaxLen(2)],
            ValueError,
        ),
    ]
    try:
        for written, real, raises in collisions:
            with pytest.raises(raises, match="caches `Annotated` by equality"):
                _fresh(written)
            assert real() is written()
            with pytest.raises(raises, match="caches `Annotated` by equality"):
                Validator(real())
    finally:
        for cleanup in getattr(typing, "_cleanups", ()):
            cleanup()


def test_a_bound_that_is_empty_because_the_order_says_so_is_kept() -> None:
    """The refusal is about `nan`, not about emptiness.

    `Gt(inf)` admits nothing either, and it is empty *because the order says
    so*, which is an answer a caller may well have meant.
    """
    beyond = Validator(Annotated[float, at.Gt(float("inf"))])
    assert not beyond.is_valid(1e308)
    assert not beyond.is_valid(float("inf"))


def test_a_whole_float_step_is_not_the_integer_step_it_equals() -> None:
    """`MultipleOf(2.0)` equals `MultipleOf(2)` and is another set over `int`.

    `%` reads an `int` against a `float` as the float nearest it, so
    `2**53 + 1` -- whose nearest float is `2**53` -- is a multiple of `2.0` and
    not of `2`. Each step is read as the number it is, and `typing`'s cache,
    which keys `Annotated` by equality, makes the two one marker in a process
    that wrote either first: `docs/05-refinements.md` says so, and this holds
    what it says.
    """
    past = 2**53 + 1
    try:
        assert not _fresh(lambda: Annotated[int, at.MultipleOf(2)]).is_valid(past)
        assert _fresh(lambda: Annotated[int, at.MultipleOf(2.0)]).is_valid(past)
        written = Annotated[int, at.MultipleOf(2.0)]
        assert Annotated[int, at.MultipleOf(2)] is written
        assert Validator(Annotated[int, at.MultipleOf(2)]).is_valid(past)
    finally:
        for cleanup in getattr(typing, "_cleanups", ()):
            cleanup()


def test_a_truth_value_is_read_as_the_length_it_equals() -> None:
    """`MinLen(True)` is `MinLen(1)`, and refusing it is not available.

    A `bool` is an `int` in Python, so a truth value in a length marker is a
    mistake worth catching -- but `at.MinLen(0)` and `at.MinLen(False)` compare
    equal and hash alike, so `typing` caches **one** `Annotated` object for both
    spellings and hands it to whichever comes second. Refusing the `bool` would
    therefore fail correct `MinLen(0)` code in any process where a
    `MinLen(False)` was built first, which is a worse failure than reading the
    value the marker holds.

    Pinned so the refusal is not added by the next reader who notices the same
    thing: the collision below is what makes it unavailable.
    """
    assert at.MinLen(0) == at.MinLen(False)
    assert hash(at.MinLen(0)) == hash(at.MinLen(False))
    assert Annotated[str, at.MinLen(0)] is Annotated[str, at.MinLen(False)]

    assert repr(Validator(Annotated[str, at.MinLen(True)])) == (
        "Annotated[str, MinLen(1)]"
    )


class _EqualToAll(float):
    """A `float` whose `==` holds every value, ordered as the float it stores."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = float.__hash__


class _ThreeAsInt(_EqualToAll):
    """The same, whose `int()` is 3."""

    __slots__ = ()

    def __int__(self) -> int:
        return 3


def test_a_number_equal_to_a_length_is_read_as_that_length() -> None:
    """`MaxLen(2.0)` is `MaxLen(2)`, for the reason `MinLen(True)` is `MinLen(1)`.

    A length is an `int`, and Python compares it with a `float`, a `Fraction`
    or a `Decimal` exactly, so a bound equal to a whole length admits what that
    length admits. `MaxLen(2.0)` equals `MaxLen(2)` and hashes alike, so
    `typing` hands one spelling the other's marker: refusing the float would
    refuse a correct `MaxLen(2)` wherever the float was written first. A bound
    equal to no whole length is refused as before, and so is one of any other
    type, whose `int()` and `==` are its own: `len("ab") >= _EqualToAll(2.5)`
    reads the float it stores, which is no length.
    """
    try:
        for whole in (2.0, Fraction(2), Decimal("2.0")):
            built = _fresh(lambda w=whole: Annotated[str, at.MaxLen(w)])  # ty: ignore[invalid-argument-type]
            assert repr(built) == "Annotated[str, MaxLen(2)]"
        _fresh(lambda: Annotated[str, at.MaxLen(2.0)])  # ty: ignore[invalid-argument-type]
        written = (at.MaxLen(2.0),)  # ty: ignore[invalid-argument-type]
        assert Annotated[str, at.MaxLen(2)].__metadata__ == written
        assert Validator(Annotated[str, at.MaxLen(2)]).is_valid("ab")
        for no_length in (
            2.5,
            Fraction(5, 2),
            Decimal("2.5"),
            float("nan"),
            Decimal("NaN"),
            -2.0,
            _EqualToAll(2.5),
            _ThreeAsInt(2.5),
        ):
            with pytest.raises(ValueError, match="must be a length a value can have"):
                _fresh(lambda n=no_length: Annotated[str, at.MaxLen(n)])  # ty: ignore[invalid-argument-type]
    finally:
        for cleanup in getattr(typing, "_cleanups", ()):
            cleanup()


def test_a_length_past_the_longest_a_value_has_is_refused() -> None:
    """No value is longer than `sys.maxsize`, the core's `LONGEST`.

    CPython counts a length in a `Py_ssize_t`, and `len()` raises past it, so a
    bound past it names no length a value can have and is refused, as a whole
    float equal to one is; the bound at it builds. Nothing is refuted below it
    for holding a longer value, and nothing below the bound one short of it
    either: a list or a set that long is one no interpreter builds.
    """
    longest = sys.maxsize
    for past in (
        at.MinLen(longest + 1),
        at.MaxLen(longest + 1),
        at.MinLen(float(longest + 1)),  # ty: ignore[invalid-argument-type]
    ):
        with pytest.raises(ValueError, match="must be a length a value can have"):
            Validator(Annotated[str, past])
    for subject, base in (
        (list[int], list[int]),
        (set[int], set[int]),
        (Annotated[str, at.MinLen(3)], str),
    ):
        assert (
            Validator(subject).relation_to(Annotated[base, at.MaxLen(longest)])
            != "not_subset"
        )
        assert (
            Validator(subject).relation_to(Annotated[base, at.MaxLen(longest - 1)])
            != "not_subset"
        )


def test_a_refutation_stands_on_a_length_every_interpreter_builds() -> None:
    """A value longer than some interpreter builds is no witness.

    A container holds its items in one allocation, which the allocator refuses
    past `sys.maxsize` bytes whatever the memory: CPython builds no list longer
    than `sys.maxsize // 8`, and a set stops at about a fifty-third of it. So
    `list[int]` is below `MaxLen(sys.maxsize // 8)` and the bare base is
    declined there rather than refuted by a list one longer, and the bound's
    complement is declined rather than held inhabited. A refutation is given a
    length every kind reaches on every interpreter, `sys.maxsize // 128`, and
    is refuted up to it, over each sized kind.
    """
    list_most = sys.maxsize // 8
    reached = sys.maxsize // 128
    assert (
        Validator(list[int]).relation_to(Annotated[list[int], at.MaxLen(list_most)])
        == "undecided"
    )
    assert (
        Validator(tuple[int, ...]).relation_to(
            Annotated[tuple[int, ...], at.MaxLen(list_most)]
        )
        == "undecided"
    )
    too_long = Validator(Annotated[list[int], at.MinLen(list_most + 1)])
    assert too_long.relation_to(nothing) == "undecided"
    one_or_most = union(
        Annotated[list[int], at.MaxLen(1)], Annotated[list[int], at.MaxLen(list_most)]
    )
    nonempty = Validator(Annotated[list[int], at.MinLen(1)])
    assert nonempty.relation_to(one_or_most) == "undecided"
    one_or_longest = union(
        Annotated[str, at.MaxLen(1)], Annotated[str, at.MaxLen(sys.maxsize - 64)]
    )
    assert (
        Validator(Annotated[str, at.MinLen(1)]).relation_to(one_or_longest)
        == "undecided"
    )
    for base in (str, bytes, list[int], tuple[int, ...], set[int], dict[str, int]):
        within = Annotated[base, at.MaxLen(reached - 1)]
        assert Validator(base).relation_to(within) == "not_subset", base
        past = Annotated[base, at.MaxLen(reached)]
        assert Validator(base).relation_to(past) == "undecided", base
        assert (
            Validator(Annotated[base, at.MinLen(reached)]).relation_to(nothing)
            == "not_subset"
        ), base
        assert (
            Validator(Annotated[base, at.MinLen(reached + 1)]).relation_to(nothing)
            == "undecided"
        ), base


def test_the_lengths_a_value_can_have_still_build() -> None:
    assert Validator(Annotated[str, at.MinLen(0)]).is_valid("")
    assert Validator(Annotated[str, at.MinLen(3)]).is_valid("abc")
    assert Validator(Annotated[str, at.MaxLen(0)]).is_valid("")


def test_a_length_every_branch_leaves_out_refutes_a_union() -> None:
    """A union of length bounds is refuted at a length no branch admits.

    `list[int]` of at least one element holds a list of every length, so a
    union of `MaxLen` variants is refuted by one longer than every branch
    admits, at any size, and so is a gap between two branches. Bounds covering
    every length refute nothing. The text form reads the same way.
    """
    nonempty = Validator(Annotated[list[int], at.MinLen(1)])
    for most in (1_000, 3_000, 100_000):
        variants = union(
            Annotated[list[int], at.MaxLen(1)], Annotated[list[int], at.MaxLen(most)]
        )
        assert nonempty.relation_to(variants) == "not_subset"
    gap = union(
        Annotated[list[int], at.MaxLen(2)], Annotated[list[int], at.MinLen(4_000)]
    )
    assert nonempty.relation_to(gap) == "not_subset"
    assert not gap.is_valid([0, 0, 0])
    covered = union(
        Annotated[list[int], at.MaxLen(3)], Annotated[list[int], at.MinLen(4)]
    )
    assert nonempty.relation_to(covered) != "not_subset"
    text = Validator(Annotated[str, at.MaxLen(3_000)])
    words = union(Annotated[str, at.MaxLen(64)], Annotated[str, at.MinLen(2_000)])
    assert text.relation_to(words) == "not_subset"


def test_a_container_holds_no_more_items_than_its_shape_can_name() -> None:
    """A length bound past the items a shape can hold empties it.

    A set of an element with no value holds `set()` alone, a mapping no more
    keys than its fields and what each clause with a value can take -- none
    where the value has no value, two for a `bool` key -- and a closed record
    no more than its fields. A bound those lengths reach stays inhabited.
    """

    class OneOptional(typing_extensions.TypedDict, total=False, closed=True):
        a: int

    class OneRequired(typing_extensions.TypedDict, closed=True):
        a: int
        b: typing_extensions.NotRequired[int]

    no_value = intersection(int, str)
    for empty in (
        Annotated[dict[str, nothing], at.MinLen(1)],  # ty: ignore[invalid-type-form]
        Annotated[dict[int, no_value], at.MinLen(1)],  # ty: ignore[invalid-type-form]
        Annotated[set[no_value], at.MinLen(1)],  # ty: ignore[invalid-type-form]
        Annotated[frozenset[no_value], at.MinLen(1)],  # ty: ignore[invalid-type-form]
        Annotated[dict[bool, int], at.MinLen(3)],
        Annotated[dict[None, int], at.MinLen(2)],
        Annotated[OneOptional, at.MinLen(2)],
    ):
        assert Validator(empty).relation_to(nothing) == "subset", empty
    for reached, value in (
        (Annotated[dict[bool, int], at.MinLen(2)], {True: 1, False: 2}),
        (Annotated[OneRequired, at.MinLen(2)], {"a": 1, "b": 2}),
    ):
        assert Validator(reached).is_valid(value)
        assert Validator(reached).relation_to(nothing) != "subset"


def test_a_meet_whose_members_each_hold_the_empty_container_holds_it() -> None:
    """The exact empty container in every member is a value of the meet.

    Each member is read for `set()` or `{}` off its shape: a length bound that
    admits zero keeps it, and a complement holds it where its inner set refuses
    it, as `{"a": int}` does for the key it requires. A member whose
    constraint the reading cannot read, a predicate, leaves the meet undecided.
    """
    at_most_none = Validator(Annotated[set[str], at.MaxLen(0)])
    at_most_one = Validator(Annotated[set, at.MaxLen(1)])
    no_items = Validator(Annotated[dict[str, int], at.MaxLen(0)])
    without_a = complement(Validator({"a": int}))
    for meet, empty in (
        (intersection(at_most_none, at_most_one), set()),
        (intersection(no_items, without_a), {}),
    ):
        assert meet.is_valid(empty)
        assert meet.relation_to(nothing) == "not_subset"
    called = Validator(Annotated[set[str], lambda value: len(value) < 2])
    assert intersection(at_most_none, called).relation_to(nothing) == "undecided"


def test_a_set_of_a_narrow_element_cannot_be_longer_than_the_element() -> None:
    """A set holds each member once, so its length bounds its element's values.

    A sequence takes any length by repeating one element; a set does not, and
    reading it as one claimed `set[None]` with `MinLen(2)` inhabited. An
    inhabited subject is what lets a kind mismatch refute an inclusion, so the
    relation against `None` asserted a value that does not exist.
    """
    two_nones = Validator(Annotated[set[None], at.MinLen(2)])
    assert two_nones.is_empty()
    assert two_nones.relation_to(Validator(None)) == "subset"
    assert not Validator(Annotated[set[None], at.MinLen(1)]).is_empty()

    # `bool` has two values, and the bound moves through them.
    assert not Validator(Annotated[set[bool], at.MinLen(2)]).is_empty()
    assert Validator(Annotated[frozenset[bool], at.MinLen(3)]).is_empty()

    # A kind with more values than a bound names supplies every bound, and a
    # list repeats, so the same element and bound decide the other way.
    assert not Validator(Annotated[set[int], at.MinLen(9)]).is_empty()
    assert not Validator(Annotated[list[None], at.MinLen(9)]).is_empty()
    assert Validator(Annotated[list[None], at.MinLen(2)]).is_valid([None, None])


class _Mod7(int):
    """An ``int`` equal to every integer it is congruent to modulo 7."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, int) and (int(self) - int(other)) % 7 == 0

    def __hash__(self) -> int:
        return int(self) % 7


class _CaseOrder(str):
    """A ``str`` equal and ordered by its lower case."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and self.lower() == other.lower()

    def __hash__(self) -> int:
        return hash(self.lower())

    def __lt__(self, other: str) -> bool:
        return self.lower() < other.lower()

    def __le__(self, other: str) -> bool:
        return self.lower() <= other.lower()

    def __gt__(self, other: str) -> bool:
        return self.lower() > other.lower()

    def __ge__(self, other: str) -> bool:
        return self.lower() >= other.lower()


class _Reversed(int):
    """An ``int`` ordered backwards, with ``int``'s own equality."""

    __slots__ = ()

    def __lt__(self, other: int) -> bool:
        return int.__gt__(self, other)

    def __le__(self, other: int) -> bool:
        return int.__ge__(self, other)

    def __gt__(self, other: int) -> bool:
        return int.__lt__(self, other)

    def __ge__(self, other: int) -> bool:
        return int.__le__(self, other)


class _Kept(int):
    """An ``int`` subclass keeping every method of ``int``."""

    __slots__ = ()


def test_a_bound_of_a_builtin_subclass_with_its_own_order_is_not_compared() -> None:
    """The walk reads such an operand otherwise than ``==`` and the order do.

    ``x >= b`` reads a builtin subclass's operand through the builtin's storage,
    or through the reflected method the subclass overrides: ``1 >= _Mod7(1)``
    reads ``1``, though ``_Mod7(1) == 8``, and ``0 >= _Reversed(5)`` holds. Two
    such bounds compared as their ``==`` and order say would be one bound to
    the relations and two to membership.
    """
    mod7 = _fresh(lambda: Annotated[int, at.Ge(_Mod7(1))])
    eight = _fresh(lambda: Annotated[int, at.Ge(8)])
    assert mod7.is_valid(1)
    assert not eight.is_valid(1)
    assert not mod7.is_equivalent(eight)
    box = _fresh(lambda: Annotated[int, at.Gt(_Mod7(1)), at.Lt(8)])
    assert box.is_valid(2)
    assert box.relation_to(nothing) != "subset"

    cased = _fresh(lambda: Annotated[str, at.Ge(_CaseOrder("a"))])
    plain = _fresh(lambda: Annotated[str, at.Ge("a")])
    assert cased.is_valid("B")
    assert not plain.is_valid("B")
    assert cased.relation_to(plain) != "subset"
    cased_box = _fresh(lambda: Annotated[str, at.Ge(_CaseOrder("b")), at.Lt("b")])
    assert cased_box.is_valid("C")
    assert cased_box.relation_to(nothing) != "subset"

    backwards = _fresh(lambda: Annotated[int, at.Ge(_Reversed(5))])
    assert backwards.is_valid(0)
    assert backwards.relation_to(_fresh(lambda: Annotated[int, at.Ge(5)])) != "subset"
    empty = _fresh(lambda: Annotated[int, at.Gt(5), at.Gt(_Reversed(5))])
    assert not any(empty.is_valid(n) for n in range(-20, 20))
    assert empty.relation_to(nothing) != "not_subset"
    holds = _fresh(lambda: Annotated[int, at.Gt(_Reversed(5)), at.Lt(_Reversed(3))])
    assert holds.is_valid(4)
    assert holds.relation_to(nothing) != "subset"


def test_a_bound_of_a_builtin_subclass_keeping_its_order_is_compared() -> None:
    """A subclass keeping every method of its builtin orders as the builtin."""
    kept = _fresh(lambda: Annotated[int, at.Ge(_Kept(1))])
    assert kept.relation_to(_fresh(lambda: Annotated[int, at.Ge(0)])) == "subset"
    assert _fresh(lambda: Annotated[int, at.Gt(_Kept(3)), at.Lt(4)]).is_empty()


class _OwnNumber:
    """A number of its own, registered as `numbers.Real`, equal to each other.

    Its `==` holds every instance and its order compares the number it holds,
    so two of its bounds are one to `==` and two to `>=`.
    """

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OwnNumber)

    def __hash__(self) -> int:
        return 0

    def __lt__(self, other: float) -> bool:
        return self.value < other

    def __le__(self, other: float) -> bool:
        return self.value <= other

    def __gt__(self, other: float) -> bool:
        return self.value > other

    def __ge__(self, other: float) -> bool:
        return self.value >= other


numbers.Real.register(_OwnNumber)


def test_a_bound_of_a_class_of_its_own_is_not_compared() -> None:
    """Python implies no relationship among a class's comparison operators.

    The walk reads `x >= b` alone, so `Ge(_OwnNumber(2.5))` over the integers
    holds 3 and `Ge(_OwnNumber(4))` does not, while `==` calls the two operands
    equal. Neither bound is compared with another. A date, the standard
    library's, orders as its `==` says, and its bounds are.
    """
    low = _fresh(lambda: Annotated[int, at.Ge(_OwnNumber(2.5))])  # ty: ignore[invalid-argument-type]
    high = _fresh(lambda: Annotated[int, at.Ge(_OwnNumber(4))])  # ty: ignore[invalid-argument-type]
    assert low.is_valid(3)
    assert not high.is_valid(3)
    assert not low.is_equivalent(high)
    assert low.relation_to(high) != "subset"
    later = Validator(Annotated[dt.date, at.Ge(dt.date(2020, 1, 2))])
    earlier = Annotated[dt.date, at.Ge(dt.date(2020, 1, 1))]
    assert later.relation_to(earlier) == "subset"


def test_two_list_or_tuple_bounds_are_not_compared() -> None:
    """A sequence orders item by item, and an item ordered against nothing breaks it.

    ``[2 + 0j] >= [2]`` holds, its one item equal to the bound's, and
    ``[2 + 0j] >= [1]`` orders a complex number against an integer, which
    raises, so the value is in the tighter bound and out of the looser one. Two
    sets order by inclusion, which raises for no value, and stay compared.
    """
    tighter = Validator(Annotated[list[object], at.Ge([2])])
    looser = Validator(Annotated[list[object], at.Ge([1])])
    assert tighter.is_valid([2 + 0j])
    assert not looser.is_valid([2 + 0j])
    assert tighter.relation_to(looser) != "subset"
    below = Validator(Annotated[list[object], at.Le([1])])
    assert below.is_valid([1 + 0j])
    assert below.relation_to(Annotated[list[object], at.Le([2])]) != "subset"
    pairs = Validator(Annotated[tuple[object, ...], at.Ge((2,))])
    assert pairs.relation_to(Annotated[tuple[object, ...], at.Ge((1,))]) != "subset"
    supersets = Validator(Annotated[set[int], at.Ge({1, 2})])
    assert supersets.relation_to(Annotated[set[int], at.Ge({1})]) == "subset"


# THEORY: a-refinement-narrows-its-base
def test_an_order_bound_over_words_is_decided_as_the_words_it_admits() -> None:
    """A `str` or `bytes` ordered against a word is a regular set of words.

    Python orders two strings code point by code point and two `bytes` byte by
    byte, a prefix before what extends it, and the words on one side of a word
    are what an automaton holds: so two such bounds relate as their sets do,
    and the set a bound leaves is read empty or not. A lone surrogate is a
    code point, ordered where its number puts it.
    """

    def text(bound: object) -> object:
        return Annotated[str, bound]

    assert Validator(text(at.Ge("a"))).relation_to(text(at.Ge("b"))) == "not_subset"
    assert Validator(text(at.Ge("b"))).relation_to(text(at.Ge("a"))) == "subset"
    assert Validator(text(at.Lt(""))).relation_to(nothing) == "subset"
    assert Validator(text(at.Ge(""))).relation_to(nothing) == "not_subset"
    assert (
        Validator(str).relation_to(union(text(at.Ge("a")), text(at.Lt("a"))))
        == "subset"
    )
    assert Validator(text(at.Gt("a"))).is_equivalent(text(at.Ge("a\x00")))

    def words(bound: object) -> object:
        return Annotated[bytes, bound]

    assert Validator(words(at.Gt(b"a"))).is_equivalent(words(at.Ge(b"a\x00")))
    assert Validator(words(at.Ge(b"a"))).relation_to(words(at.Ge(b"b"))) == "not_subset"
    surrogate = "\ud800"
    assert Validator(text(at.Gt("퟿"))).is_valid(surrogate)
    assert not Validator(text(at.Ge(""))).is_valid(surrogate)
    assert Validator(text(at.Gt("퟿"))).relation_to(text(at.Ge(""))) == "not_subset"


def test_a_sized_base_and_its_least_length_zero_are_one_set() -> None:
    """`X` and `Annotated[X, MinLen(0)]` are one set for a sized `X`, and read alike.

    A base written bare is its refinement under no bound, so a length a
    supertype's bounds leave out refutes it as it refutes the refinement, and
    a set or a dict holding its empty one is refuted below a supertype refusing
    it, through bounds that admit zero.
    """
    pairs = [
        (str, Annotated[str, at.MaxLen(1000)]),
        (dict[str, int], Annotated[dict, at.MaxLen(2)]),
        (set[int], Annotated[set, at.MaxLen(2)]),
        (set[int], Annotated[set, at.MinLen(1)]),
        (frozenset[int], Annotated[frozenset, at.Len(1, 3)]),
    ]
    for base, supertype in pairs:
        spelled = Annotated[base, at.MinLen(0)]
        assert Validator(base).relation_to(supertype) == "not_subset", base
        assert Validator(spelled).relation_to(supertype) == "not_subset", base
    empty_dict = Annotated[dict, at.MaxLen(0)]
    assert Validator(empty_dict).relation_to({"a": int}) == "not_subset"
    assert Validator(empty_dict).relation_to(Validator({"a": int}).open()) == (
        "not_subset"
    )


#: The floats a step over floats reads apart: whole on both sides of the edges
#: where the spacing changes, fractional, and the three no step holds.
_FLOATS = [
    *sorted(
        {
            *(float(n) for n in range(-13, 14)),
            0.5,
            -0.5,
            1.5,
            5e-324,
            *(
                sign * edge * factor
                for sign in (1.0, -1.0)
                for edge in (2.0**52, 2.0**53, 2.0**63, 2.0**1000)
                for factor in (1.0, 1 + 2.0**-52, 1 + 2.0**-51, 1 + 3 * 2.0**-52)
            ),
            1.7976931348623157e308,
            -1.7976931348623157e308,
        }
    ),
    math.inf,
    -math.inf,
    math.nan,
]


@pytest.mark.parametrize(
    ("narrow", "wide", "expected"),
    [
        pytest.param(
            Annotated[float, at.MultipleOf(1), at.Gt(30), at.Lt(31)],
            nothing,
            "subset",
            id="no-whole-float-between-two-neighbors",
        ),
        pytest.param(
            Annotated[float, at.MultipleOf(1), at.Ge(30), at.Lt(31)],
            nothing,
            "not_subset",
            id="a-whole-float-at-the-held-end",
        ),
        pytest.param(
            Annotated[float, at.MultipleOf(2)],
            Annotated[float, at.MultipleOf(1)],
            "subset",
            id="two-divides-one",
        ),
        pytest.param(
            Annotated[float, at.MultipleOf(1)],
            Annotated[float, at.MultipleOf(2)],
            "not_subset",
            id="one-is-no-multiple-of-two",
        ),
        pytest.param(
            Annotated[float, at.MultipleOf(3)],
            Annotated[float, at.MultipleOf(2)],
            "not_subset",
            id="neither-divides",
        ),
        pytest.param(
            Annotated[float, at.MultipleOf(6), at.Ge(0), at.Le(5)],
            Annotated[float, at.Le(0)],
            "subset",
            id="only-zero-is-a-multiple-under-five",
        ),
        pytest.param(
            intersection(float, complement(Annotated[float, at.MultipleOf(1)])),
            Annotated[float, at.Gt(0), at.Lt(1)],
            "not_subset",
            id="the-fractional-floats-reach-past-one",
        ),
        pytest.param(
            intersection(
                complement(Annotated[float, at.MultipleOf(1)]),
                Annotated[float, at.Ge(2.0**53)],
            ),
            Annotated[float, at.Ge(math.inf)],
            "subset",
            id="past-2-to-the-53-only-inf-is-not-whole",
        ),
        pytest.param(
            Annotated[float, at.MultipleOf(3), at.Ge(2.0**53), at.Le(2.0**53 + 2)],
            nothing,
            "subset",
            id="the-multiple-of-three-there-is-no-float",
        ),
        pytest.param(
            Annotated[int | float, at.MultipleOf(4)],
            Annotated[int | float, at.MultipleOf(2)],
            "subset",
            id="both-kinds-read-together",
        ),
        pytest.param(
            Annotated[int | float, at.MultipleOf(2)],
            Annotated[int | float, at.MultipleOf(4)],
            "not_subset",
            id="both-kinds-refuted-together",
        ),
    ],
)
def test_a_step_over_floats_is_decided_as_a_set(
    narrow: object, wide: object, expected: str
) -> None:
    assert Validator(narrow).relation_to(wide) == expected


def test_the_integers_of_json_are_decided_against_their_complement() -> None:
    # The set a JSON `integer` is: a number with no fractional part, however it is
    # written. Its complement is what a planner's other branch reads.
    integer = intersection(
        union(int, Annotated[float, at.MultipleOf(1)]), complement(bool)
    )
    number = intersection(int | float, complement(bool))

    def between(lo: int, hi: int) -> Validator[Any]:
        return intersection(
            integer,
            union(
                Annotated[int, at.Ge(lo), at.Le(hi)],
                Annotated[float, at.Ge(lo), at.Le(hi)],
            ),
        )

    rest = intersection(between(1, 1000), complement(between(1, 30)))
    assert rest.relation_to(between(31, 1000)) == "subset"
    assert between(31, 1000).relation_to(rest) == "subset"
    assert rest.is_valid(31)
    assert rest.is_valid(31.0)
    assert not rest.is_valid(30.0)
    assert integer.relation_to(number) == "subset"
    assert number.relation_to(integer) == "not_subset"
    assert intersection(integer, complement(int)).relation_to(float) == "subset"
    assert intersection(between(1, 30), between(31, 40)).is_empty()


_step = st.sampled_from([1, 2, 3, 4, 6, 7, 16, -3])
_bound = st.sampled_from([-3, 0, 0.5, 4, 2.0**52, 2.0**53, 2.0**63])


@st.composite
def _float_sets(draw: st.DrawFn) -> Validator[Any]:
    atoms = st.one_of(
        _step.map(lambda step: Validator(Annotated[float, at.MultipleOf(step)])),
        _bound.map(lambda bound: Validator(Annotated[float, at.Ge(bound)])),
        _bound.map(lambda bound: Validator(Annotated[float, at.Lt(bound)])),
        st.just(Validator(float)),
    )
    return draw(
        st.recursive(
            atoms,
            lambda inner: st.one_of(
                st.tuples(inner, inner).map(lambda pair: union(*pair)),
                st.tuples(inner, inner).map(lambda pair: intersection(*pair)),
                inner.map(lambda each: intersection(float, complement(each))),
            ),
            max_leaves=5,
        )
    )


@given(_float_sets(), _float_sets())
def test_a_relation_over_float_steps_answers_as_the_walk_does(
    a: Validator[Any], b: Validator[Any]
) -> None:
    held_by_a = [value for value in _FLOATS if a.is_valid(value)]
    answer = a.relation_to(b)
    if answer == "subset":
        assert all(b.is_valid(value) for value in held_by_a), (a, b)
    if a.is_empty():
        assert not held_by_a, a
    # A value the walk finds outside is a refutation no proof may contradict.
    if any(not b.is_valid(value) for value in held_by_a):
        assert answer != "subset", (a, b)


def test_a_float_refinement_holds_a_value_a_refutation_stands_on() -> None:
    """The sets read the interval, whatever kind of number bounds it."""
    half = Annotated[float, at.Ge(Fraction(5, 16))]
    assert Validator(half).relation_to(Literal[9223372036854775808]) == "not_subset"
    assert (
        Validator(Annotated[set[half], at.MinLen(1)]).relation_to(nothing)
        == "not_subset"
    )
    none = Annotated[float, at.Ge(1.3125), at.Le(0.3125)]
    assert Validator(none).relation_to(nothing) == "subset"


# Steps no other test writes: `typing` caches an `Annotated` by its metadata's `==`,
# and `MultipleOf(2.0)` is `MultipleOf(2)` to that cache.
@pytest.mark.parametrize("step", [0.0625, 1.125, 2.25, -0.375, math.inf])
def test_zero_is_a_multiple_of_every_float_step_over_the_integers(step: float) -> None:
    """A value needs no division, and the walk reads the same remainder."""
    multiple = Annotated[int, at.MultipleOf(step)]
    assert Validator(multiple).is_valid(0)
    assert Validator(multiple).relation_to(nothing) == "not_subset"
    assert Validator(multiple).relation_to(str) == "not_subset"


def test_an_operand_answering_the_remainder_of_zero_is_not_read_as_a_step() -> None:
    """A subclass's `__rmod__` is the operand's, which a relation does not read past."""

    class Refusing(float):
        def __rmod__(self, other: object) -> float:
            return 1.0

    multiple = Annotated[int, at.MultipleOf(Refusing(7.25))]
    assert not Validator(multiple).is_valid(0)
    assert Validator(multiple).relation_to(str) != "not_subset"
