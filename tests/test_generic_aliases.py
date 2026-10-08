"""A generic type alias applied to its arguments is its body with them in place.

PEP 695 writes `type Pair[T] = tuple[T, T]`, and the typing spec reads
`Pair[int]` as `tuple[int, int]`. The runtime keeps everything that reading
needs -- the alias, its parameters, its body -- and checks none of it: it
builds `Pair[int, str]` without complaint. So the frontend substitutes the
arguments itself, counts them, fills a missing one from its default, and ties
a recursive alias once per argument list, refusing the one shape whose
argument lists never repeat.

The `type` statement is 3.12 syntax, so each alias is defined by running its
source, and `typing_extensions` spells the same aliases on every release.
"""

from __future__ import annotations

import sys

import pytest

from valgebra import Validator

needs_the_type_statement = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the type statement is 3.12 syntax"
)
needs_a_default_in_it = pytest.mark.skipif(
    sys.version_info < (3, 13), reason="a type parameter's default is 3.13 syntax"
)


def _aliases(source: str) -> dict[str, object]:
    """Define the aliases `source` writes, and give back their namespace."""
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - the syntax under test, not user input
    return namespace


@needs_the_type_statement
def test_an_applied_alias_is_its_body_with_the_arguments_substituted() -> None:
    aliases = _aliases(
        "type Pair[T] = tuple[T, T]\ntype Pairs[T] = list[Pair[T]]\n"
        "type Swap[T, U] = dict[U, T]\n"
    )
    pair = Validator(aliases["Pair"][int])  # ty: ignore[not-subscriptable]
    assert pair == Validator(tuple[int, int])
    assert pair.is_valid((1, 2))
    assert not pair.is_valid((1, "a"))
    assert not pair.is_valid((1,))
    # An alias applied inside another reads the same way, and the body's own
    # parameter order is the one the arguments land in.
    pairs = Validator(aliases["Pairs"][str])  # ty: ignore[not-subscriptable]
    assert pairs == Validator(list[tuple[str, str]])
    swap = Validator(aliases["Swap"][str, int])  # ty: ignore[not-subscriptable]
    assert swap == Validator(dict[int, str])


@needs_the_type_statement
def test_a_count_the_runtime_accepts_is_refused_by_the_aliases_name() -> None:
    aliases = _aliases("type Pair[T] = tuple[T, T]\ntype Two[T, U] = dict[T, U]\n")
    with pytest.raises(NotImplementedError, match="type argument, and") as surplus:
        Validator(aliases["Pair"][int, str])  # ty: ignore[not-subscriptable]
    assert str(surplus.value).endswith(
        "Pair takes 1 type argument, and Pair[int, str] gives it 2"
    )
    with pytest.raises(NotImplementedError, match="no default to stand in") as missing:
        Validator(aliases["Two"][int])  # ty: ignore[not-subscriptable]
    assert str(missing.value).endswith(
        "Two takes 2 type arguments, and Two[int] gives it 1: U has no default to "
        "stand in"
    )
    # Bare, a generic alias means `Any` for its parameter, which is rarely the
    # intent, so it is refused naming the spelling that says the argument.
    with pytest.raises(NotImplementedError, match=r"write Pair\[\.\.\.\]"):
        Validator(aliases["Pair"])


@needs_a_default_in_it
def test_a_default_stands_in_for_a_missing_argument() -> None:
    aliases = _aliases(
        "type PairD[T = int] = tuple[T, T]\ntype Same[T, U = T] = dict[T, U]\n"
    )
    assert Validator(aliases["PairD"]) == Validator(tuple[int, int])
    assert Validator(aliases["PairD"][str]) == Validator(tuple[str, str])  # ty: ignore[not-subscriptable]
    assert Validator(aliases["Same"][int]) == Validator(dict[int, int])  # ty: ignore[not-subscriptable]


def test_a_typing_extensions_alias_reads_as_the_statement_does() -> None:
    extensions = pytest.importorskip("typing_extensions")
    t = extensions.TypeVar("T")
    d = extensions.TypeVar("D", default=int)
    pair = extensions.TypeAliasType("Pair", tuple[t, t], type_params=(t,))
    defaulted = extensions.TypeAliasType("PairD", tuple[d, d], type_params=(d,))
    assert Validator(pair[int]) == Validator(tuple[int, int])
    assert Validator(defaulted) == Validator(tuple[int, int])
    with pytest.raises(NotImplementedError, match="has no default"):
        Validator(pair)


@needs_the_type_statement
def test_a_recursive_generic_alias_is_tied_once_per_argument_list() -> None:
    aliases = _aliases(
        "type Tree[T] = T | list[Tree[T]]\n"
        "type Swapping[T, U] = None | dict[T, Swapping[U, T]]\n"
    )
    tree = Validator(aliases["Tree"][int])  # ty: ignore[not-subscriptable]
    assert tree.is_valid([1, [2, [3]]])
    assert not tree.is_valid([1, ["a"]])
    # A fresh `Tree[int]` at every read, and one schema.
    assert tree == Validator(aliases["Tree"][int])  # ty: ignore[not-subscriptable]
    swapping = Validator(aliases["Swapping"][int, str])  # ty: ignore[not-subscriptable]
    assert swapping.is_valid({1: {"a": None}})
    assert not swapping.is_valid({1: {2: None}})


@needs_the_type_statement
def test_an_alias_recurring_on_a_growing_argument_is_refused() -> None:
    # `Nest[int]` unfolds to `Nest[list[int]]`, and that to a list of lists:
    # no argument list repeats, so no fixpoint ties it. mypy refuses the
    # shape at its definition, and so does the build.
    aliases = _aliases(
        "type Nest[T] = T | list[Nest[list[T]]]\n"
        "type Settles[T] = T | list[Settles[int]]\n"
    )
    with pytest.raises(NotImplementedError, match="nesting its own type parameter"):
        Validator(aliases["Nest"][int])  # ty: ignore[not-subscriptable]
    # An argument naming none of its parameters is a list met one step on:
    # `Settles[str]` is a `str` or a list of what `Settles[int]` holds.
    settles = Validator(aliases["Settles"][str])  # ty: ignore[not-subscriptable]
    assert settles.is_valid("a")
    assert settles.is_valid([1, [2]])
    assert not settles.is_valid(["a"])


@needs_the_type_statement
def test_a_parameter_that_is_no_type_variable_is_refused() -> None:
    aliases = _aliases(
        "from collections.abc import Callable\n"
        "type Spread[*Ts] = tuple[*Ts]\ntype Hook[**P] = Callable[P, int]\n"
    )
    with pytest.raises(NotImplementedError, match="Spread declares Ts"):
        Validator(aliases["Spread"][int])  # ty: ignore[not-subscriptable]
    with pytest.raises(NotImplementedError, match="Hook declares P"):
        Validator(aliases["Hook"][[int]])  # ty: ignore[not-subscriptable]


@needs_the_type_statement
def test_a_bound_is_a_checkers_to_hold_and_is_not_read() -> None:
    # The runtime builds `Bounded[str]`, and a checker refuses it; the schema is
    # the body with the argument in place, as the argument was written.
    aliases = _aliases("type Bounded[T: int] = list[T]\n")
    assert Validator(aliases["Bounded"][str]) == Validator(list[str])  # ty: ignore[not-subscriptable]
