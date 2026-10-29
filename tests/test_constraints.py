"""Every refinement constraint, on both the fast path and the explain path.

Covers each annotated-types marker valgebra reads, the boolean result and the
error code on failure, and the rendered form. A dropped constraint silently
admits invalid values, so every constraint is checked in both directions.
"""

from __future__ import annotations

import datetime as dt
import numbers
import typing
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated

import annotated_types as at
import pytest

from valgebra import Regex, ValidationError, Validator, nothing

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


def test_the_lengths_a_value_can_have_still_build() -> None:
    assert Validator(Annotated[str, at.MinLen(0)]).is_valid("")
    assert Validator(Annotated[str, at.MinLen(3)]).is_valid("abc")
    assert Validator(Annotated[str, at.MaxLen(0)]).is_valid("")


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
