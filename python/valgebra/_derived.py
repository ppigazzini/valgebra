"""Derived combinators built from the core Boolean algebra.

These add no new IR: they compose `union`, `intersect`, and `complement`, so
they inherit the algebra's semantics and laws exactly.
"""

from ._valgebra import (
    CompiledValidator,
    anything,
    complement,
    intersect,
    union,
)


def ifthen(
    condition: object,
    then: object,
    otherwise: object = anything,
) -> CompiledValidator:
    """Require `then` when a value matches `condition`, else `otherwise`.

    `otherwise` admits anything by default. Denotation:
    ``(condition and then) or ((not condition) and otherwise)``; with the
    default `otherwise` this is exactly "condition implies then".
    """
    return union(
        intersect(condition, then),
        intersect(complement(condition), otherwise),
    )


def cond(
    *cases: tuple[object, object],
    default: object = anything,
) -> CompiledValidator:
    """Select the `then` of the first matching `(condition, then)` case.

    A value is checked against each condition in order; the first one it
    matches selects the `then` it must satisfy. If it matches no condition it
    must satisfy `default`. Equivalent to nesting `ifthen` from the last case
    inward, so the earliest matching case wins.
    """
    result: object = default
    for condition, then in reversed(cases):
        result = ifthen(condition, then, result)
    if isinstance(result, CompiledValidator):
        return result
    # No cases: coerce a bare default spec into a validator.
    return union(result)
