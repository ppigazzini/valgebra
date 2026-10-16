"""The prefix-plus-tail and non-empty list forms.

A list schema `[A, ..., Z, ...]` denotes a fixed positional prefix followed by
zero or more elements matching the element before the trailing `...`. `[T, ...]`
is the prefix-free (homogeneous) case, and `[T, T, ...]` is the non-empty list.
"""

import sys
import typing
from collections.abc import Callable
from typing import Any

import pytest

from valgebra import ValidationError, Validator


def test_prefix_then_repeated_tail() -> None:
    schema = Validator([str, int, ...])  # a str, then zero or more ints
    assert schema.is_valid(["a"])
    assert schema.is_valid(["a", 1, 2, 3])
    assert not schema.is_valid([1])  # prefix must be a str
    assert not schema.is_valid([])  # the prefix is required
    assert not schema.is_valid(["a", "b"])  # the tail must be ints


def test_homogeneous_form_is_unchanged() -> None:
    schema = Validator([int, ...])
    assert schema.is_valid([])
    assert schema.is_valid([1, 2, 3])
    assert not schema.is_valid(["a"])


def test_non_empty_list() -> None:
    schema = Validator([int, int, ...])  # at least one int
    assert not schema.is_valid([])
    assert schema.is_valid([1])
    assert schema.is_valid([1, 2, 3])
    assert not schema.is_valid(["a"])


def test_too_short_reports_a_length_code() -> None:
    with pytest.raises(ValidationError) as info:
        Validator([str, int, ...]).validate([])
    assert info.value.code == "list_length"
    assert "at least 1" in info.value.expected


def test_tail_element_failure_reports_its_index() -> None:
    with pytest.raises(ValidationError) as info:
        Validator([str, int, ...]).validate(["a", 1, "b"])
    assert info.value.path == (2,)


def test_repr_round_trips_the_forms() -> None:
    assert repr(Validator([str, int, ...])) == "[str, int, ...]"
    assert repr(Validator([int, ...])) == "list[int]"
    assert repr(Validator([int, int, ...])) == "[int, int, ...]"


def test_ellipsis_only_as_the_last_element() -> None:
    with pytest.raises(NotImplementedError):
        Validator([int, ..., ...])
    with pytest.raises(NotImplementedError):
        Validator([..., int])


# --- The tuple spelling of the same shape -------------------------------------
#
# PEP 646 lets a tuple say "a fixed prefix, then more of this" as an *unpacked*
# variadic tuple. It is the prefix-and-tail shape under another spelling, and a
# reader who writes it means what `tuple[A, B, ...]` means. The star syntax is
# 3.11+, so the forms are built rather than written: this file parses on the
# support floor.

requires_unpacking = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="unpacking a tuple type into another arrived in 3.11 (PEP 646)",
)


def _starred(alias: Any) -> object:
    """`*alias`, without the star syntax the support floor cannot parse."""
    return next(iter(alias))


def _unpacked(alias: Any) -> object:
    """`Unpack[alias]`, the other spelling of the same thing."""
    return typing.Unpack[alias]


@requires_unpacking
@pytest.mark.parametrize("spell", [_starred, _unpacked], ids=["star", "Unpack"])
def test_an_unpacked_variadic_tuple_is_the_prefix_and_tail_form(
    spell: Callable[[object], object],
) -> None:
    schema = Validator(tuple[int, spell(tuple[str, ...])])  # ty: ignore[invalid-type-form]
    assert schema.is_valid((1,))
    assert schema.is_valid((1, "a", "b"))
    assert not schema.is_valid((1, 2))
    assert not schema.is_valid(("a",))
    # Read as a nested tuple instead, this is the value it would have admitted.
    assert not schema.is_valid((1, ("a", "b")))
    assert repr(schema) == "tuple[int, str, ...]"


@requires_unpacking
def test_an_unpacked_fixed_tuple_splices_its_elements() -> None:
    schema = Validator(tuple[bool, _starred(tuple[int, str])])  # ty: ignore[invalid-type-form]
    assert schema.is_valid((True, 1, "a"))
    assert not schema.is_valid((True, 1))
    assert not schema.is_valid((1, 1, "a"))
    assert repr(schema) == "tuple[bool, int, str]"


@requires_unpacking
def test_an_unpacked_tuple_alone_is_its_own_shape() -> None:
    schema = Validator(tuple[_starred(tuple[str, ...])])  # ty: ignore[invalid-type-form]
    assert schema.is_valid(())
    assert schema.is_valid(("a", "b"))
    assert not schema.is_valid((1,))


@requires_unpacking
def test_an_element_after_the_tail_is_refused() -> None:
    # A sequence carries a fixed prefix and then a repeating tail, with nothing
    # after it. Reading this as any other shape would admit a different set.
    with pytest.raises(NotImplementedError, match="nothing may follow the tail"):
        Validator(tuple[_starred(tuple[int, ...]), str])  # ty: ignore[invalid-type-form]


@requires_unpacking
def test_an_unpacked_type_variable_tuple_is_refused() -> None:
    # `*Ts` binds no element types at runtime, so there is nothing to check.
    with pytest.raises(NotImplementedError, match="only a tuple can be unpacked"):
        Validator(
            # ty: ignore[invalid-type-form,invalid-legacy-type-variable]
            tuple[int, typing.Unpack[typing.TypeVarTuple("Ts")]]
        )


@requires_unpacking
@pytest.mark.parametrize("spell", [_starred, _unpacked], ids=["star", "Unpack"])
def test_a_spliced_tuple_unpacks_what_it_carries(
    spell: Callable[[object], object],
) -> None:
    # What a splice carries is read as the carrier's own arguments, as both
    # static checkers reveal it: a star inside it unpacks, and a `...` in it
    # repeats. Pushed in as elements instead, the inner star would be a nested
    # tuple and the inner `...` the literal `Ellipsis`.
    tail = tuple[str, spell(tuple[bool, ...])]  # ty: ignore[invalid-type-form]
    schema = Validator(tuple[int, spell(tail)])  # ty: ignore[invalid-type-form]
    assert repr(schema) == "tuple[int, str, bool, ...]"
    assert schema.is_valid((1, "a"))
    assert schema.is_valid((1, "a", True))
    assert not schema.is_valid((1, "a", (True,)))

    inner = spell(tuple[int, ...])
    assert repr(Validator(tuple[spell(tuple[inner])])) == "tuple[int, ...]"  # ty: ignore[invalid-type-form]
    assert repr(Validator(tuple[spell(tuple[int, str, ...])])) == "tuple[int, str, ...]"  # ty: ignore[invalid-type-form]
    spliced = Validator(tuple[spell(tuple[int, spell(tuple[()])]), str])  # ty: ignore[invalid-type-form]
    assert repr(spliced) == "tuple[int, str]"

    # A tail reached through a splice is still a tail, and nothing follows it.
    nested = tuple[int, spell(tuple[str, ...])]  # ty: ignore[invalid-type-form]
    with pytest.raises(NotImplementedError, match="nothing may follow the tail"):
        Validator(tuple[spell(nested), bool])  # ty: ignore[invalid-type-form]


@requires_unpacking
def test_an_ellipsis_repeats_only_an_element_written_before_it() -> None:
    # The elements a splice brings in are not one element, and reading the last
    # of them as the repeated one admits a set the spelling does not say.
    for spelling in (
        tuple[_starred(tuple[int, str]), ...],  # ty: ignore[invalid-type-form]
        tuple[int, _starred(tuple[()]), ...],  # ty: ignore[invalid-type-form]
    ):
        with pytest.raises(NotImplementedError, match="are not one element"):
            Validator(spelling)
    with pytest.raises(NotImplementedError, match="cannot begin with one"):
        Validator(tuple[int, _starred(tuple[..., str])])  # ty: ignore[invalid-type-form]


@requires_unpacking
@pytest.mark.parametrize("spell", [_starred, _unpacked], ids=["star", "Unpack"])
def test_an_unpacked_tuple_outside_a_tuple_is_refused(
    spell: Callable[[object], object],
) -> None:
    # Read as a type, `*tuple[int]` is `tuple[int]`, and `list[*tuple[int]]`
    # would be a list of one-element tuples, which one checker refuses and
    # another reads as `list[int]`.
    one = spell(tuple[int])
    for spelling in (
        one,
        list[one],  # ty: ignore[invalid-type-form]
        dict[one, int],  # ty: ignore[invalid-type-form]
        int | one,  # ty: ignore[unsupported-operator]
        [one],
    ):
        with pytest.raises(NotImplementedError, match="unpacks into the arguments"):
            Validator(spelling)


def test_an_unpacked_empty_tuple_splices_nothing() -> None:
    # 3.10 spells `tuple[()]`'s arguments as the empty tuple itself, which
    # spliced in as an element would be a tuple literal.
    extensions = pytest.importorskip("typing_extensions")
    schema = Validator(tuple[int, extensions.Unpack[tuple[()]]])
    assert schema == Validator(tuple[int])
