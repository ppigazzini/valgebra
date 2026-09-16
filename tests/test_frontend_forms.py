"""Forms the frontend reads as something other than what they say.

A schema is built from an annotation, and the failure that matters here is the
quiet one: a form read as a *different* schema rather than refused. A refusal a
caller sees is a message they act on; a schema that admits more than the
annotation names is a validator that passes what it was written to stop.

Each row below is a form whose reading was not its meaning, with the value that
shows it, and beside it the form it must keep reading as it did.
"""

from __future__ import annotations

import typing
from typing import Annotated, Literal

import annotated_types as at
import pytest

from valgebra import Regex, Validator


class Yields(at.GroupedMetadata):
    """A marker carrying its constraints through the grouping protocol.

    `annotated_types` documents this as the way to write one: a marker that is
    several constraints answers `__iter__` with them, which is how `Interval`
    and `Len` are written. Reading a marker by attribute alone finds nothing on
    this one, and metadata this frontend does not recognise is ignored -- which
    leaves a schema admitting exactly what the marker was written to exclude.
    """

    def __iter__(self) -> typing.Iterator[object]:
        yield at.Ge(0)
        yield at.Lt(10)


class YieldsItself(at.GroupedMetadata):
    """A marker whose grouping never bottoms out."""

    def __iter__(self) -> typing.Iterator[object]:
        yield self


def test_a_parametrised_legacy_tuple_is_the_tuple_it_names() -> None:
    """`typing.Tuple[()]` is the empty tuple, as `tuple[()]` is.

    The two spell one type, and the legacy one carries an empty argument list
    where the *bare* alias carries none at all -- a difference `get_args` does
    not report, since it answers with an empty tuple for both.
    """
    legacy = Validator(typing.Tuple[()])
    native = Validator(tuple[()])
    assert legacy == native
    assert repr(legacy) == "tuple[()]"
    assert legacy.is_valid(()) is True
    assert legacy.is_valid((1,)) is False


def test_a_bare_legacy_alias_is_the_class_it_aliases() -> None:
    """The control the row above must not take with it."""
    assert Validator(typing.Tuple) == Validator(tuple)
    assert Validator(typing.Tuple).is_valid((1, 2)) is True
    assert Validator(typing.List) == Validator(list)
    assert Validator(typing.Dict) == Validator(dict)
    # And a parametrised legacy alias is its parametrisation.
    assert Validator(typing.List[int]) == Validator(list[int])


def test_a_grouped_marker_carries_the_constraints_it_yields() -> None:
    """The documented way to write a marker of several constraints."""
    schema = Annotated[int, Yields()]
    assert Validator(schema).is_valid(5) is True
    assert Validator(schema).is_valid(-5) is False
    assert Validator(schema).is_valid(10) is False
    # And it is the same schema the constraints spell one at a time.
    assert Validator(schema) == Validator(Annotated[int, at.Ge(0), at.Lt(10)])


def test_the_grouped_markers_of_the_vocabulary_are_unmoved() -> None:
    """`Interval` and `Len` are grouped markers, and read as they did."""
    interval = Validator(Annotated[int, at.Interval(ge=0, lt=10)])
    assert interval == Validator(Annotated[int, at.Ge(0), at.Lt(10)])
    assert interval.is_valid(0) is True
    assert interval.is_valid(10) is False
    length = Validator(Annotated[str, at.Len(1, 3)])
    assert length.is_valid("ab") is True
    assert length.is_valid("") is False
    assert length.is_valid("abcd") is False


def test_a_grouped_marker_that_never_bottoms_out_is_refused() -> None:
    """A marker yielding itself is refused rather than followed forever."""
    with pytest.raises(ValueError, match=r"nested too deeply|does not bottom out"):
        Validator(Annotated[int, YieldsItself()])


def test_a_field_named_twice_is_refused() -> None:
    """One key cannot be required and optional at once.

    `{"a": int, "a?": str}` is two dict keys and one field name, so the record
    it names is a contradiction rather than a schema -- and building it gave a
    record admitting nothing, with no message saying why.
    """
    with pytest.raises(ValueError, match=r"declared twice|names the field"):
        Validator({"a": int, "a?": str})
    with pytest.raises(ValueError, match=r"declared twice|names the field"):
        Validator({"a?": str, "a": int})


def test_a_record_naming_each_field_once_builds() -> None:
    """The control: the optional marker is part of the name, not a second one."""
    record = Validator({"a": int, "b?": str})
    assert record.is_valid({"a": 1}) is True
    assert record.is_valid({"a": 1, "b": "x"}) is True
    assert record.is_valid({"a": 1, "b": 2}) is False


def test_a_literal_of_a_type_is_refused() -> None:
    """`Literal[int]` names no constant, and is not the `int` schema."""
    with pytest.raises(
        (TypeError, ValueError, NotImplementedError), match=r"Literal|constant"
    ):
        Validator(Literal[int])  # ty: ignore[invalid-type-form]


def test_a_literal_of_a_constant_builds() -> None:
    """The control: the spellings the typing spec allows."""
    assert Validator(Literal[1]).is_valid(1) is True
    assert Validator(Literal["a"]).is_valid("a") is True
    assert Validator(Literal[None]).is_valid(None) is True  # noqa: PYI061
    assert Validator(Literal[b"x"]).is_valid(b"x") is True
    assert Validator(Literal[True]).is_valid(True) is True


def test_a_pattern_that_is_not_text_says_what_it_is() -> None:
    """A refusal names the marker it read, not a kind it guessed."""
    with pytest.raises((TypeError, ValueError, NotImplementedError)) as caught:
        Validator(Annotated[str, Regex(123)])  # ty: ignore[invalid-argument-type]
    assert "bytes" not in str(caught.value)
