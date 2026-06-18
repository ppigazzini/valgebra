from types import GenericAlias

import pytest

from valgebra import ValidationError, Validator, nothing


def _pt_tuple(*args: object) -> GenericAlias:
    """Build a prefix-plus-tail tuple schema `tuple[A, B, ...]` at runtime.

    Constructing it at runtime keeps the static type checker from reading it as
    an (invalid) `tuple` specialization.
    """
    return GenericAlias(tuple, args)


def test_frozenset_annotation() -> None:
    schema = Validator(frozenset[int])
    assert schema.is_valid(frozenset({1, 2, 3}))
    assert schema.is_valid(frozenset())
    assert not schema.is_valid(frozenset({1, "x"}))


def test_frozenset_rejects_a_plain_set() -> None:
    assert not Validator(frozenset[int]).is_valid({1, 2})


def test_variadic_tuple_accepts_any_length() -> None:
    schema = Validator(tuple[int, ...])
    assert schema.is_valid(())
    assert schema.is_valid((1,))
    assert schema.is_valid((1, 2, 3))


def test_variadic_tuple_rejects_a_bad_element() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(tuple[int, ...]).validate((1, "x", 3))
    assert info.value.code == "int_type"
    assert info.value.path == (1,)


def test_variadic_tuple_rejects_a_non_tuple() -> None:
    assert not Validator(tuple[int, ...]).is_valid([1, 2, 3])


def test_fixed_and_variadic_tuples_are_distinct() -> None:
    assert Validator(tuple[int, str]).is_valid((1, "a"))
    assert not Validator(tuple[int, str]).is_valid((1, 2, 3))
    assert Validator(tuple[int, ...]).is_valid((1, 2, 3))


def test_prefix_tail_tuple_fixes_a_prefix_then_repeats() -> None:
    # tuple[A, B, ...]: a fixed prefix then the last element repeated, mirroring
    # the [A, B, ...] list form.
    schema = Validator(_pt_tuple(str, int, ...))
    assert schema.is_valid(("head",))  # the repeated tail may be empty
    assert schema.is_valid(("head", 1))
    assert schema.is_valid(("head", 1, 2, 3))
    assert not schema.is_valid(())  # the prefix str is required
    assert not schema.is_valid((1, 2))  # first element must be a str
    assert not schema.is_valid(("head", "x"))  # tail must be ints


def test_prefix_tail_tuple_rejects_an_interior_ellipsis() -> None:
    with pytest.raises(NotImplementedError):
        Validator(_pt_tuple(int, ..., str))


def test_prefix_tail_tuple_decides_distinctly_from_lists() -> None:
    # The frontend builds the same prefix-plus-tail regex the list form does, but
    # under the tuple container, so the decision procedure honours the container
    # throughout subtyping, emptiness, and equivalence.

    # Subtyping is covariant in the prefix and the repeated tail.
    assert Validator(_pt_tuple(bool, bool, ...)).is_subtype_of(_pt_tuple(int, int, ...))
    assert not Validator(_pt_tuple(int, int, ...)).is_subtype_of(
        _pt_tuple(int, bool, ...)
    )
    # A fixed-length tuple is a subtype of a prefix-and-tail one it fits.
    assert Validator(tuple[bool, int]).is_subtype_of(_pt_tuple(int, int, ...))

    # The container is part of the type: the list form and the tuple form with an
    # identical element regex are unrelated.
    assert not Validator([int, int, ...]).is_subtype_of(_pt_tuple(int, int, ...))
    assert not Validator(_pt_tuple(int, int, ...)).is_subtype_of([int, int, ...])

    # An uninhabited prefix empties the tuple; an uninhabited tail only forbids
    # the repeats, so a single-element tuple matching the prefix still inhabits it.
    assert Validator(_pt_tuple(nothing, int, ...)).is_empty()
    assert not Validator(_pt_tuple(int, nothing, ...)).is_empty()

    # Equivalence collapses a redundant union in the tail (bool is a subtype of int).
    collapsed = Validator(_pt_tuple(int, bool | int, ...))
    assert collapsed.is_equivalent(_pt_tuple(int, int, ...))


def test_mapping_error_path_tolerates_a_key_whose_str_raises() -> None:
    # The aggregating walk labels each entry by its key; a key whose str()
    # raises falls back to a repr summary rather than crashing.
    class Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("no str")

    with pytest.raises(ValidationError):
        Validator(dict[int, int]).validate({Unprintable(): 1})
