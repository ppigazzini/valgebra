"""Membership at the float corner cases: NaN, signed zero, and infinities.

A validator's scalar semantics have to be exact, and floating point has three
traps that a value-membership test must get right:

- ``NaN`` compares false against everything, including itself, so a bound
  refinement (``Ge``/``Le``) must *reject* it and an equality ``Literal`` must
  not match it — silently accepting ``NaN`` because a comparison "didn't fail"
  is the classic soundness hole.
- ``-0.0`` equals ``0.0`` and hashes equal in Python, so a ``Literal[0.0]``
  admits ``-0.0`` (and vice versa) — the membership test follows Python
  equality, deliberately.
- ``inf``/``-inf`` are ordinary ``float`` members and order normally against
  finite bounds, so ``inf`` satisfies ``Ge(0)`` but not ``Le(1e308)``.

The JSON path is stricter than Python's ``json`` module: the non-standard
``Infinity``/``-Infinity``/``NaN`` tokens are rejected at parse time as
``json_invalid`` rather than parsed into float specials, while an overflowing
literal such as ``1e400`` parses to ``inf`` exactly as the object path holds it.
These cases are pinned here so a change to the scalar or JSON path has to face
them.
"""

from __future__ import annotations

import math
from typing import Annotated

import pytest
from annotated_types import Ge, Gt, Le, Lt

from valgebra import ValidationError, Validator

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")


def test_specials_are_float_members() -> None:
    # NaN, both infinities, and negative zero are all float instances.
    is_float = Validator(float)
    assert is_float.is_valid(NAN)
    assert is_float.is_valid(INF)
    assert is_float.is_valid(NINF)
    assert is_float.is_valid(-0.0)
    # ...but not int members: float and int stay disjoint even for specials.
    assert not Validator(int).is_valid(NAN)
    assert not Validator(int).is_valid(INF)


def test_nan_is_rejected_by_every_bound() -> None:
    # NaN compares false against every bound, so a refinement must reject it
    # rather than let a "comparison did not fail" slip it through.
    for constraint in (Ge(0), Le(0), Gt(-1), Lt(1)):
        schema = Validator(Annotated[float, constraint])
        assert not schema.is_valid(NAN), constraint


def test_infinities_order_against_finite_bounds() -> None:
    # Infinities are ordered normally against finite bounds.
    assert Validator(Annotated[float, Ge(0)]).is_valid(INF)
    assert not Validator(Annotated[float, Le(1e308)]).is_valid(INF)
    assert not Validator(Annotated[float, Ge(0)]).is_valid(NINF)
    assert Validator(Annotated[float, Le(0)]).is_valid(NINF)


def test_signed_zero_follows_python_equality_in_literals() -> None:
    # 0.0 == -0.0 in Python and they hash equal, so a float-constant literal
    # admits either spelling of zero. Membership follows Python equality by
    # design. A bare float constant is the literal spelling (typing.Literal
    # forbids float arguments).
    assert Validator(0.0).is_valid(-0.0)
    assert Validator(-0.0).is_valid(0.0)


def test_nan_never_matches_a_nan_literal() -> None:
    # A float-constant literal is equality membership, and NaN != NaN, so a NaN
    # literal admits no value — not even a NaN. This mirrors Python's equality.
    assert not Validator(NAN).is_valid(NAN)
    assert not Validator(NAN).is_valid(0.0)


def test_infinity_literal_matches_infinity() -> None:
    assert Validator(INF).is_valid(INF)
    assert not Validator(INF).is_valid(NINF)


@pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN"])
def test_json_rejects_the_nonstandard_float_tokens(token: str) -> None:
    # Python's json module accepts Infinity/-Infinity/NaN; the valgebra JSON path
    # is strict and rejects them at parse time, even though the parsed specials
    # would be valid float members on the object path.
    schema = Validator(float)
    assert not schema.is_valid_json(token)
    with pytest.raises(ValidationError) as info:
        schema.validate_json(token)
    assert info.value.code == "json_invalid"
    # The object path, in contrast, admits the corresponding float special.
    assert schema.is_valid(float(token))


def test_json_overflow_parses_to_infinity() -> None:
    # An overflowing literal is standard JSON and parses to the same infinity the
    # object path holds, so the two paths agree on it.
    schema = Validator(float)
    assert schema.is_valid_json("1e400")
    assert math.isinf(1e400)
    assert schema.is_valid(1e400)
    # Negative-zero and underflow-to-zero are ordinary JSON numbers.
    assert schema.is_valid_json("-0.0")
    assert schema.is_valid_json("1e-400")
