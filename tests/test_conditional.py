"""Conditional fields are a composition recipe, not a shipped combinator.

valgebra ships no `ifthen`/`cond`: material implication and first-match dispatch
are derived from the Boolean algebra (`union`/`intersection`/`complement`). These
helpers are the documented recipe; the tests assert the recipe reaches the
decisions the removed combinators did, so the derivation stays covered.
"""

from typing import Annotated

import annotated_types as at

from valgebra import (
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    union,
)


def implies(condition: object, then: object, otherwise: object = anything) -> Validator:
    """Require `then` of a value matching `condition`, else `otherwise`.

    The conditional-field recipe: a value either matches `condition` and must
    then satisfy `then`, or fails `condition` and must satisfy `otherwise`.
    """
    return union(
        intersection(condition, then),
        intersection(complement(condition), otherwise),
    )


def first_match(*cases: tuple[object, object], default: object = anything) -> Validator:
    """Select the consequent of the first matching `(condition, then)` case.

    Nests `implies` from the last case inward, so the earliest matching
    condition selects its consequent, falling back to `default`.
    """
    result: Validator = Validator(default)
    for condition, then in reversed(cases):
        result = implies(condition, then, result)
    return result


def test_implies_requires_consequent_when_condition_matches() -> None:
    schema = implies(int, Annotated[int, at.Ge(0)])
    assert schema.is_valid(5)  # int and non-negative
    assert not schema.is_valid(-1)  # int but negative
    assert schema.is_valid("x")  # not an int: admitted by default


def test_implies_with_explicit_otherwise() -> None:
    schema = implies(int, Annotated[int, at.Ge(0)], str)
    assert schema.is_valid(5)
    assert schema.is_valid("x")  # not an int: must be a str
    assert not schema.is_valid(1.0)  # neither an int nor a str


def test_first_match_selects_the_first_matching_case() -> None:
    schema = first_match(
        (str, Annotated[str, at.MinLen(1)]),
        (int, Annotated[int, at.Ge(0)]),
        default=nothing,
    )
    assert schema.is_valid("ok")
    assert schema.is_valid(3)
    assert not schema.is_valid("")  # str but empty
    assert not schema.is_valid(-1)  # int but negative
    assert not schema.is_valid(1.0)  # matches no case, default is nothing


def test_first_match_default_admits_unmatched_by_default() -> None:
    schema = first_match((int, Annotated[int, at.Ge(0)]))
    assert schema.is_valid(1.0)  # no case matches; default is anything
    assert not schema.is_valid(-1)


def test_first_match_with_no_cases_is_the_default() -> None:
    schema = first_match(default=int)
    assert schema.is_valid(5)
    assert not schema.is_valid("x")
