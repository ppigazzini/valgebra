from typing import Annotated

import annotated_types as at

from valgebra import cond, ifthen, nothing


def test_ifthen_requires_consequent_when_condition_matches() -> None:
    schema = ifthen(int, Annotated[int, at.Ge(0)])
    assert schema.is_valid(5)  # int and non-negative
    assert not schema.is_valid(-1)  # int but negative
    assert schema.is_valid("x")  # not an int: admitted by default


def test_ifthen_with_explicit_otherwise() -> None:
    schema = ifthen(int, Annotated[int, at.Ge(0)], str)
    assert schema.is_valid(5)
    assert schema.is_valid("x")  # not an int: must be a str
    assert not schema.is_valid(1.0)  # neither an int nor a str


def test_cond_selects_the_first_matching_case() -> None:
    schema = cond(
        (str, Annotated[str, at.MinLen(1)]),
        (int, Annotated[int, at.Ge(0)]),
        default=nothing,
    )
    assert schema.is_valid("ok")
    assert schema.is_valid(3)
    assert not schema.is_valid("")  # str but empty
    assert not schema.is_valid(-1)  # int but negative
    assert not schema.is_valid(1.0)  # matches no case, default is nothing


def test_cond_default_admits_unmatched_by_default() -> None:
    schema = cond((int, Annotated[int, at.Ge(0)]))
    assert schema.is_valid(1.0)  # no case matches; default is anything
    assert not schema.is_valid(-1)


def test_cond_with_no_cases_is_the_default() -> None:
    schema = cond(default=int)
    assert schema.is_valid(5)
    assert not schema.is_valid("x")
