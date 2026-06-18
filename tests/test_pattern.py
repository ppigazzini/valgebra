"""Native string-pattern refinements: `Annotated[str, Regex(...)]`.

The pattern is matched in Rust (anchored, `re.fullmatch` semantics), off the
Python-predicate slow path. These tests pin the semantics, the differential
parity with `re.fullmatch`, the build-time validation of a bad pattern, and that
it composes and reaches the JSON path like any other refinement.
"""

import json
import re
from typing import Annotated

import pytest

from valgebra import Regex, ValidationError, Validator, union


def test_pattern_fullmatch_anchored() -> None:
    schema = Validator(Annotated[str, Regex(r"a+")])
    assert schema.is_valid("aaa")
    assert not schema.is_valid("aXb")  # not a full match
    assert not schema.is_valid("")  # a+ needs at least one
    assert not schema.is_valid("baaa")  # anchored at the start


def test_pattern_rejects_non_string() -> None:
    schema = Validator(Annotated[str, Regex(r"\d+")])
    assert not schema.is_valid(123)  # the base is str; an int is not a member
    assert schema.is_valid("123")


def test_bare_re_pattern_is_accepted_as_metadata() -> None:
    schema = Validator(Annotated[str, re.compile(r"[0-9a-f]{24}")])
    assert schema.is_valid("0123456789abcdef01234567")
    assert not schema.is_valid("nope")


@pytest.mark.parametrize(
    "pattern",
    [r"[A-Za-z0-9_.-]+", r"\d{1,3}(\.\d{1,3}){3}", r"(foo|bar)baz", r"a*", r"[^x]+"],
)
def test_pattern_matches_re_fullmatch(pattern: str) -> None:
    schema = Validator(Annotated[str, Regex(pattern)])
    reference = re.compile(pattern)
    corpus = ["", "a", "abc", "a.b-c_1", "1.2.3.4", "foobaz", "barbaz", "xax", "x" * 50]
    for value in corpus:
        assert schema.is_valid(value) == (reference.fullmatch(value) is not None), value


def test_invalid_pattern_raises_at_compile_time() -> None:
    with pytest.raises(ValueError, match="invalid regular expression"):
        Validator(Annotated[str, Regex(r"(unclosed")])


def test_pattern_composes_and_reaches_json() -> None:
    oid = Annotated[str, Regex(r"[0-9a-f]{24}")]
    record = Validator({"id": oid, "name": str})
    good = {"id": "0123456789abcdef01234567", "name": "x"}
    bad = {"id": "short", "name": "x"}
    assert record.is_valid(good)
    assert not record.is_valid(bad)
    # The JSON path reaches the same decision as the object path.
    assert record.is_valid_json(json.dumps(good))
    assert not record.is_valid_json(json.dumps(bad))
    # And it composes in the algebra like any refinement.
    either = union(oid, Annotated[str, Regex(r"[A-Z]+")])
    assert either.is_valid("ABC")
    assert either.is_valid("0123456789abcdef01234567")
    assert not either.is_valid("abc")


def test_pattern_validate_reports_a_pattern_violation() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(Annotated[str, Regex(r"\d+")]).validate("abc")
    assert info.value.code == "string_pattern"
