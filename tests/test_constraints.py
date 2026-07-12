"""Every refinement constraint, on both the fast path and the explain path.

Covers each annotated-types marker valgebra reads, the boolean result and the
error code on failure, and the rendered form. A dropped constraint silently
admits invalid values, so every constraint is checked in both directions.
"""

from __future__ import annotations

from typing import Annotated

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator


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
def test_constraint_accepts_rejects_codes_and_repr(  # noqa: PLR0913
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


def test_multiple_of_on_a_non_number_is_not_a_multiple() -> None:
    # The base admits the value but the modulo is undefined: not a multiple.
    assert not Validator(Annotated[str, at.MultipleOf(3)]).is_valid("abc")


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
