from types import GenericAlias
from typing import ForwardRef, Literal, Union

import pytest

from valgebra import ValidationError, Validator


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
    pytest.param(Union[ForwardRef(_UNRESOLVED), None], id="optional"),  # noqa: UP007
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
