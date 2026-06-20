r"""Surrogate-bearing strings: fast/slow agreement and field-name handling.

A Python ``str`` may carry lone surrogate code points (``"\ud800"``), which are
not valid UTF-8. valgebra stores field names as UTF-8 and compares string values
without assuming UTF-8 on the membership path. Two invariants matter:

- The fast membership check (``is_valid``) and the explaining walk (``validate``)
  must agree on every string, surrogate-bearing ones included.
- A field name cannot hold a lone surrogate (it could not round-trip), so such a
  key is refused at build time rather than silently replaced — which would make
  the field unmatchable.
"""

from typing import TypedDict

import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import ValidationError, Validator, union

# A lone high surrogate: a valid Python str, but not valid UTF-8.
SURROGATE = "\ud800"

# Strings mixing ordinary characters with surrogate code points.
_surrogate_text = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=0x20, max_codepoint=0x7E),
        st.characters(min_codepoint=0xD800, max_codepoint=0xDFFF),
    ),
    max_size=4,
)


def _agree(schema: object, value: object) -> None:
    """Assert the fast check and the explaining walk report equal membership."""
    validator = schema if isinstance(schema, Validator) else Validator(schema)
    fast = validator.is_valid(value)
    try:
        validator.validate(value)
        explain = True
    except ValidationError:
        explain = False
    assert fast == explain


@given(value=_surrogate_text)
def test_str_schema_agrees_on_surrogate_values(value: str) -> None:
    _agree(str, value)


@given(value=_surrogate_text)
def test_literal_agrees_on_surrogate_values(value: str) -> None:
    _agree(union("ok", SURROGATE), value)


@given(value=_surrogate_text)
def test_literal_union_plan_agrees_on_surrogate_values(value: str) -> None:
    # A wide all-string union builds the literal fast-path decision table; a
    # surrogate value must defer to the scan and agree with it.
    _agree(union("a", "b", "c", "d", SURROGATE), value)


@given(key=_surrogate_text, value=st.integers())
def test_mapping_agrees_on_surrogate_keys(key: str, value: int) -> None:
    _agree({str: int}, {key: value})


@given(key=_surrogate_text, value=st.integers())
def test_closed_record_agrees_on_surrogate_extra_keys(key: str, value: int) -> None:
    _agree({"x": int}, {"x": 1, key: value})


def test_surrogate_value_matches_where_expected() -> None:
    # A surrogate string is a member of str, of a literal union carrying it, and
    # is a valid mapping key.
    assert Validator(str).is_valid(SURROGATE)
    assert union("other", SURROGATE).is_valid(SURROGATE)
    assert Validator({str: int}).is_valid({SURROGATE: 1})


def test_dict_literal_record_rejects_a_surrogate_field_name() -> None:
    # A surrogate field name cannot round-trip through UTF-8, so it is refused at
    # build time rather than silently corrupted into a replacement character.
    with pytest.raises(ValueError, match="lone surrogate"):
        Validator({SURROGATE: int})


def test_typed_dict_rejects_a_surrogate_field_name() -> None:
    surrogate_typed_dict = TypedDict("surrogate_typed_dict", {SURROGATE: int})
    with pytest.raises((ValueError, UnicodeError)):
        Validator(surrogate_typed_dict)
