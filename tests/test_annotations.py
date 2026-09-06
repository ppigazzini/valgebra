import sys
from types import GenericAlias
from typing import ForwardRef, Literal, TypedDict

import pytest

import _deferred
from valgebra import ValidationError, Validator

# The floor each qualifier reaches `typing` at; see `_deferred`.
if sys.version_info >= (3, 11):
    from typing import NotRequired, Required


def test_list_annotation_is_a_sequence() -> None:
    assert Validator(list[int]).is_valid([1, 2, 3])
    assert not Validator(list[int]).is_valid([1, "x"])
    assert not Validator(list[int]).is_valid((1, 2))


def test_set_annotation() -> None:
    assert Validator(set[int]).is_valid({1, 2})
    assert not Validator(set[int]).is_valid({1, "x"})


def test_dict_annotation_is_a_mapping() -> None:
    schema = Validator(dict[str, int])
    assert schema.is_valid({"a": 1, "b": 2})
    assert not schema.is_valid({"a": "x"})
    assert not schema.is_valid({1: 1})


def test_fixed_tuple_annotation_matches_positionally() -> None:
    schema = Validator(tuple[int, str])
    assert schema.is_valid((1, "a"))
    assert not schema.is_valid((1, 2))
    assert not schema.is_valid((1,))


def test_nested_generic_annotations() -> None:
    schema = Validator(list[dict[str, int]])
    assert schema.is_valid([{"a": 1}, {"b": 2}])
    assert not schema.is_valid([{"a": "x"}])


def test_generic_annotation_reports_located_failure() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(list[int]).validate([1, "x"])
    assert info.value.code == "int_type"
    assert info.value.path == (1,)


# A name no module defines, so resolving it is impossible rather than merely
# unattempted. The forms are built at runtime: written as annotations they would
# be flagged by the linter for the very reason this test exists.
_UNRESOLVED = "Account"

_FORWARD_REFERENCES = [
    pytest.param(GenericAlias(list, (_UNRESOLVED,)), id="list"),
    pytest.param(GenericAlias(set, (_UNRESOLVED,)), id="set"),
    pytest.param(GenericAlias(dict, (_UNRESOLVED, int)), id="dict-key"),
    pytest.param(GenericAlias(dict, (str, _UNRESOLVED)), id="dict-value"),
    pytest.param(GenericAlias(tuple, (_UNRESOLVED, int)), id="tuple"),
    pytest.param(GenericAlias(list, (ForwardRef(_UNRESOLVED),)), id="ForwardRef"),
]


@pytest.mark.parametrize("spec", _FORWARD_REFERENCES)
def test_a_forward_reference_in_a_generic_argument_is_refused(spec: object) -> None:
    # The typing spec resolves a string in this position against the namespace the
    # annotation was written in, and a runtime object carries no namespace.
    # Reading it as a literal instead builds a container of the *word*, which
    # refuses what the annotation admits.
    with pytest.raises(NotImplementedError, match="forward reference"):
        Validator(spec)


def test_a_constant_is_still_a_literal_where_a_value_belongs() -> None:
    # The refusal is about the argument of a typing form, not about constants.
    assert Validator("active").is_valid("active")
    assert Validator(["active"]).is_valid(["active"])
    assert Validator(Literal["active"]).is_valid("active")
    assert Validator({"state": "active"}).is_valid({"state": "active"})


@pytest.mark.skipif(sys.version_info < (3, 11), reason="NotRequired marker")
def test_a_qualifier_survives_a_string_annotation() -> None:
    """`NotRequired` under `from __future__ import annotations` (PEP 563).

    CPython computes `__required_keys__` when the class is created, from the
    annotations *as written*. Under PEP 563 those are strings, so
    `"NotRequired[list[str]]"` is opaque to it and every key lands in
    `__required_keys__` -- which the builder read. Every optional key in every
    module using the future import was therefore compiled required, and correct
    data failed with `missing_key`.

    `get_type_hints` resolves the string and keeps the qualifier, so the
    resolved hint is what says whether the key must be present. The assertions
    below compare against CPython's own view precisely because the two disagree
    here; agreeing with CPython would be the bug.
    """
    assert sorted(_deferred.Deferred.__required_keys__) == ["a", "b", "c"]
    assert sorted(_deferred.Deferred.__optional_keys__) == []

    schema = Validator(_deferred.Deferred)
    assert repr(schema) == "{'a': int, 'b?': list[str], 'c': str, str: anything}"
    # The keys the author marked optional may be absent.
    assert schema.is_valid({"a": 1, "c": "x"})
    assert schema.is_valid({"a": 1, "c": "x", "b": ["s"]})
    # And the ones they did not, may not.
    assert not schema.is_valid({"a": 1})
    assert not schema.is_valid({"c": "x"})


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Required marker")
def test_a_required_qualifier_survives_it_too() -> None:
    """The other direction: `Required` inside a `total=False` class."""
    assert sorted(_deferred.DeferredTotalFalse.__required_keys__) == []

    schema = Validator(_deferred.DeferredTotalFalse)
    assert repr(schema) == "{'a?': int, 'b': str, str: anything}"
    assert schema.is_valid({"b": "x"})
    assert not schema.is_valid({"a": 1}), "`Required` must beat `total=False`"


@pytest.mark.skipif(sys.version_info < (3, 13), reason="ReadOnly marker")
def test_a_qualifier_wrapping_a_qualifier_survives_it_too() -> None:
    """`ReadOnly[NotRequired[T]]`: the search reads through the outer one.

    `ReadOnly` says nothing about presence, so a reading that stopped at the
    outermost qualifier would leave the key's required-ness to the class, and
    under PEP 563 the class calls every key required.
    """
    assert sorted(_deferred.DeferredReadOnly.__required_keys__) == ["a", "d"]

    schema = Validator(_deferred.DeferredReadOnly)
    assert repr(schema) == "{'a': int, 'd?': int, str: anything}"
    assert schema.is_valid({"a": 1})
    assert schema.is_valid({"a": 1, "d": 2})
    assert not schema.is_valid({"d": 2})


@pytest.mark.skipif(sys.version_info < (3, 11), reason="NotRequired marker")
def test_a_deferred_typed_dict_matches_the_same_class_written_plainly() -> None:
    """The whole claim in one line: the future import changes no set."""

    class Plain(TypedDict):
        a: int
        b: NotRequired[list[str]]
        c: Required[str]

    assert Validator(_deferred.Deferred) == Validator(Plain)
