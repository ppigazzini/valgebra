"""What the numeric carriers can spell, and what they decline at the edge.

A relation between two schemas is decided on the *sets* they denote, and the
integer component spells a set as intervals over `i64` with a residue class per
step. Python's integers are unbounded and the carrier is not, so a bound the
carrier cannot name is a bound the relation declines -- which is deviation 11 of
the theory ledger, and it is sound: a decline is "not proven", never a wrong
proof.

The edge of the carrier is where declining is easy to get wrong. A span with no
lower end means *every* integer, and a span beginning at the smallest `i64` means
every integer from there up -- two different sets that the carrier writes the
same way, because the integer one place below the first is not a value it can
hold. Reading the second as the first proves `int` below
`Annotated[int, Ge(-2**63)]`, and `-2**63 - 1` is an `int` that bound refuses.

So each row here is a bound *at* the carrier's end, the relation it must not
prove, and the value that settles it. The rows beside them are one step in, where
the carrier does spell the set and the relation is decided exactly: a decline
that spreads inward is a completeness loss, and a test that only forbids the
wrong answer would not see it.
"""

from __future__ import annotations

from typing import Annotated

import annotated_types as at
import pytest

from valgebra import Validator, complement, intersection

# The ends of the integer carrier, and the values just outside them.
I64_MIN = -(2**63)
I64_MAX = 2**63 - 1
# The first integer no float equals, which is the float carrier's own edge.
FLOAT_EXACT_LIMIT = 2**53


# Each row: a name, a bound at the carrier's end, and a value the bound refuses
# that `int` admits. The witness is what makes the relation false rather than
# merely unproven, and the walk is the authority on it.
_AT_THE_EDGE: list[tuple[str, object, int]] = [
    ("Ge at the bottom", Annotated[int, at.Ge(I64_MIN)], I64_MIN - 1),
    ("Le at the top", Annotated[int, at.Le(I64_MAX)], I64_MAX + 1),
]


@pytest.mark.parametrize(
    ("name", "bounded", "outside"), _AT_THE_EDGE, ids=[row[0] for row in _AT_THE_EDGE]
)
def test_a_bound_at_the_carrier_s_end_proves_nothing_about_every_integer(
    name: str, bounded: object, outside: int
) -> None:
    """`int` is not below a bound that refuses an integer, and is not proved so."""
    # The witness first: the walk decides membership exactly, whatever the
    # carrier can spell.
    assert Validator(int).is_valid(outside) is True, name
    assert Validator(bounded).is_valid(outside) is False, name
    # So the inclusion is false, and a `subset` would be a proof against a value.
    assert Validator(int).relation_to(bounded) != "subset", name
    # The same relation asked as an emptiness: the difference holds the witness.
    difference = intersection(int, complement(Validator(bounded)))
    assert difference.is_valid(outside) is True, name
    assert difference.is_empty() is False, name


def test_a_bound_one_step_inside_the_carrier_is_decided_exactly() -> None:
    """The decline is at the end, and does not spread to the bounds beside it."""
    inside_low = Annotated[int, at.Ge(I64_MIN + 1)]
    inside_high = Annotated[int, at.Le(I64_MAX - 1)]
    # Refuted, and the witness is a value the carrier can hold.
    assert Validator(int).relation_to(inside_low) == "not_subset"
    assert Validator(int).relation_to(inside_high) == "not_subset"
    # And the order between two spellable bounds is still decided both ways.
    assert Validator(inside_low).relation_to(Annotated[int, at.Ge(I64_MIN + 2)]) == (
        "not_subset"
    )
    assert Validator(Annotated[int, at.Ge(I64_MIN + 2)]).relation_to(inside_low) == (
        "subset"
    )


def test_a_strict_bound_at_the_end_names_a_set_the_carrier_spells() -> None:
    """`Gt` at the bottom and `Lt` at the top move one step in by construction.

    The set they name begins one past the end, which the carrier holds, so these
    decide rather than decline -- and the row exists so that a repair to the two
    above does not take them with it.
    """
    assert Validator(int).relation_to(Annotated[int, at.Gt(I64_MIN)]) == "not_subset"
    assert Validator(int).relation_to(Annotated[int, at.Lt(I64_MAX)]) == "not_subset"


def test_a_bound_past_the_carrier_declines() -> None:
    """Deviation 11 as the page states it, with the queries it names."""
    wide = Annotated[int, at.Ge(2**70)]
    wider = Annotated[int, at.Ge(2**70 + 1)]
    assert Validator(wide).relation_to(wider) == "undecided"
    # The schema still validates exactly: membership reads the Python integer.
    assert Validator(wide).is_valid(2**70) is True
    assert Validator(wide).is_valid(2**70 - 1) is False


def test_a_step_past_the_period_bound_declines() -> None:
    """A modulus the residue representation does not materialise is declined."""
    assert (
        Validator(Annotated[int, at.MultipleOf(5000)]).relation_to(
            Annotated[int, at.MultipleOf(2500)]
        )
        == "undecided"
    )
    # Two steps whose least common multiple is past the bound meet undecided
    # rather than being rounded to a period that is not theirs.
    meet = intersection(
        Annotated[int, at.MultipleOf(64)], Annotated[int, at.MultipleOf(81)]
    )
    assert meet.relation_to(Annotated[int, at.MultipleOf(2)]) == "undecided"
    # And a step inside the bound is decided, in the direction that holds.
    assert (
        Validator(Annotated[int, at.MultipleOf(4)]).relation_to(
            Annotated[int, at.MultipleOf(2)]
        )
        == "subset"
    )


def test_a_float_bound_written_as_an_integer_no_float_equals() -> None:
    """The float carrier's own edge, read against the bound's two neighbours."""
    subject = Annotated[float, at.Lt(FLOAT_EXACT_LIMIT + 1)]
    supertype = Annotated[float, at.Lt(FLOAT_EXACT_LIMIT)]
    witness = float(FLOAT_EXACT_LIMIT)
    assert Validator(subject).is_valid(witness) is True
    assert Validator(supertype).is_valid(witness) is False
    assert Validator(subject).relation_to(supertype) == "not_subset"


def test_a_bool_base_is_decided_against_a_bound_it_meets() -> None:
    """A `bool` is an `int` of two values, and a spellable bound decides it."""
    assert Validator(bool).relation_to(Annotated[int, at.Ge(0)]) == "subset"
    assert Validator(bool).relation_to(Annotated[int, at.Ge(1)]) == "not_subset"
