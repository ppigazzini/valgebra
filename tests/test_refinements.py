import enum
from typing import Annotated

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator


def test_comparison_bounds() -> None:
    adult = Validator(Annotated[int, at.Ge(18), at.Le(150)])
    assert adult.is_valid(18)
    assert adult.is_valid(150)
    assert not adult.is_valid(17)
    assert not adult.is_valid(151)


def test_strict_comparison_bounds() -> None:
    schema = Validator(Annotated[int, at.Gt(0), at.Lt(10)])
    assert schema.is_valid(5)
    assert not schema.is_valid(0)
    assert not schema.is_valid(10)


def test_length_bounds() -> None:
    name = Validator(Annotated[str, at.MinLen(1), at.MaxLen(3)])
    assert name.is_valid("ab")
    assert not name.is_valid("")
    assert not name.is_valid("abcd")


def test_length_bounds_on_a_list() -> None:
    schema = Validator(Annotated[list[int], at.MinLen(2)])
    assert schema.is_valid([1, 2])
    assert not schema.is_valid([1])


def test_predicate_marker() -> None:
    even = Validator(Annotated[int, at.Predicate(lambda x: x % 2 == 0)])
    assert even.is_valid(4)
    assert not even.is_valid(3)


def test_bare_callable_metadata_is_a_predicate() -> None:
    positive = Validator(Annotated[int, lambda x: x > 0])
    assert positive.is_valid(1)
    assert not positive.is_valid(-1)


def test_a_callable_object_is_a_predicate_too() -> None:
    """A predicate carried by an instance is asked, like any other callable."""

    class Positive:
        def __call__(self, value: int) -> bool:
            return value > 0

    positive = Validator(Annotated[int, Positive()])
    assert positive.is_valid(1)
    assert not positive.is_valid(-1)


def test_a_class_is_metadata_rather_than_a_predicate() -> None:
    """A class is callable, and calling it constructs rather than asks.

    `Kilograms(1.5)` builds a unit marker; it does not answer whether 1.5 is in
    the set. Reading a class as a predicate turns a documentation marker into a
    schema that admits nothing, so a class is metadata valgebra does not
    recognise, and the typing spec says to ignore that.
    """

    class Kilograms:
        symbol = "kg"

    schema = Validator(Annotated[float, Kilograms])
    assert repr(schema) == "float"
    assert schema.is_valid(1.5)
    assert not schema.is_valid("x")


def test_an_enum_class_is_metadata_rather_than_a_predicate() -> None:
    """The same, for the marker shape a reader is most likely to reach for."""

    class Colour(enum.Enum):
        RED = "red"

    schema = Validator(Annotated[int, Colour])
    assert repr(schema) == "int"
    assert schema.is_valid(1)


def test_a_constraint_class_is_ignored_rather_than_read_for_its_slots() -> None:
    """A marker class exposes descriptors where an instance exposes values.

    `at.Ge(0)` carries `ge = 0`; `at.Ge` carries the slot descriptor that would
    read it. Taking the descriptor for a bound builds a comparison against an
    object no value is ordered against, so the schema admits nothing — the same
    trap as a class read as a predicate, one arm earlier.
    """
    schema = Validator(Annotated[int, at.Ge])
    assert repr(schema) == "int"
    assert schema.is_valid(1)
    assert schema.is_valid(-1)


def test_a_class_marker_beside_a_real_constraint_leaves_it_standing() -> None:
    """Ignoring one marker does not disturb the others in the same form."""

    class Kilograms:
        symbol = "kg"

    schema = Validator(Annotated[int, at.Ge(0), Kilograms])
    assert schema.is_valid(1)
    assert not schema.is_valid(-1)


def test_base_failure_takes_precedence_over_constraints() -> None:
    schema = Validator(Annotated[int, at.Ge(0)])
    with pytest.raises(ValidationError) as info:
        schema.validate("x")
    assert info.value.code == "int_type"


def test_constraint_failure_reports_its_code() -> None:
    schema = Validator(Annotated[int, at.Ge(18)])
    with pytest.raises(ValidationError) as info:
        schema.validate(5)
    assert info.value.code == "greater_than_equal"


def test_raising_predicate_is_surfaced_as_predicate_error() -> None:
    def boom(_: object) -> bool:
        raise RuntimeError

    schema = Validator(Annotated[int, at.Predicate(boom)])
    with pytest.raises(ValidationError) as info:
        schema.validate(1)
    assert info.value.code == "predicate_error"


def test_unrecognized_metadata_is_ignored() -> None:
    schema = Validator(Annotated[int, "documentation"])
    assert schema.is_valid(3)
    assert not schema.is_valid("x")


# --- What a decision query does to a predicate ---------------------------------
#
# The decision procedures do not reason about a predicate's satisfiability, and
# the pages say so. They do run one: deciding whether a literal is a subtype of a
# refinement is deciding whether that literal's value is a member of it, and
# membership runs the predicate. These pin the behaviour the pages describe, so a
# doc edit and a code change cannot disagree about which is true.


def test_a_subtyping_query_runs_the_predicate_of_a_literal_supertype() -> None:
    seen: list[object] = []

    def even(value: int) -> bool:
        seen.append(value)
        return value % 2 == 0

    assert Validator(4).is_subtype_of(Annotated[int, at.Predicate(even)])
    assert not Validator(5).is_subtype_of(Annotated[int, at.Predicate(even)])
    assert seen == [4, 5]


def test_a_subtyping_query_does_not_reason_about_a_predicate() -> None:
    # Two refinements carrying the same predicate are not related through it: the
    # procedure compares constraints, it does not analyse them.
    always = Annotated[int, at.Predicate(lambda _: True)]
    assert not Validator(always).is_subtype_of(Annotated[int, at.Ge(0)])


def test_an_emptiness_query_runs_a_bound_comparison() -> None:
    compared: list[str] = []

    class Counted(int):
        """An int that records the order comparisons a decision performs on it."""

        def __gt__(self, other: int, /) -> bool:
            compared.append("gt")
            return super().__gt__(other)

        def __lt__(self, other: int, /) -> bool:
            compared.append("lt")
            return super().__lt__(other)

    # `typing.Annotated` caches by the value of its metadata, and a `Counted` is
    # equal to the plain int it wraps, so a spec any other test already built
    # comes back holding *that* test's operands and this one instruments nothing.
    # The bounds are values nothing else uses, and the premise is asserted rather
    # than assumed: the failure to catch is the silent one.
    low, high = Counted(60_013), Counted(-60_013)
    spec = Annotated[int, at.Ge(low), at.Le(high)]
    assert spec.__metadata__[0].ge is low, "typing returned a cached spec"

    assert Validator(spec).is_empty()
    assert compared, "deciding the bound conjunction ordered the two operands"


class _CountedBound(str):
    """A bound that records every time its repr is taken."""

    __slots__ = ()
    reprs = 0

    def __repr__(self) -> str:
        type(self).reprs += 1
        return "<bound>"


def test_a_passing_check_builds_no_violation_message() -> None:
    """A membership answer costs nothing to explain until there is a failure.

    A violation names the bound it was measured against, and naming it takes the
    bound's `repr`. A value that *belongs* produces no violation, so that repr is
    never read — and building it anyway makes the cost of accepting a value scale
    with the size of the schema's operand rather than with the value.
    """
    bound = _CountedBound("m")
    schema = Validator(Annotated[str, at.Ge(bound)])

    _CountedBound.reprs = 0
    with pytest.raises(ValidationError, match="<bound>"):
        schema.validate("a")
    explained = _CountedBound.reprs
    assert explained, "the failure must name the bound, or the probe measures nothing"

    _CountedBound.reprs = 0
    for _ in range(100):
        assert schema.is_valid("z")
    assert _CountedBound.reprs == 0
