"""A generic record class applied to its arguments is read with them in place.

A generic `TypedDict`, dataclass or `NamedTuple` names a field's type with its
parameter, so an argument reaches what a member holds: `Box[int]` is a `Box`
whose field of type `T` holds an `int`. The runtime keeps the class, its
parameters and every base's arguments, and binds none of them into the hints
`typing` hands back, so the frontend binds them itself.

No `from __future__ import annotations` here: the classes are declared inside
the tests, and their annotations are read as the objects they name.
"""

import dataclasses
import sys
import typing

import pytest

from valgebra import Validator

T = typing.TypeVar("T")
U = typing.TypeVar("U")


def test_a_generic_dataclass_applied_is_read_with_its_arguments_in_place() -> None:
    """`Box[int]` is a `Box` whose field of type `T` holds an `int`.

    The class's field types name its parameter, so the argument reaches what a
    member holds. A class deriving from an applied base reads the base's
    arguments off what it was written deriving from, however deep, and through a
    base that binds its own parameter to a form of another.
    """

    @dataclasses.dataclass
    class Box(typing.Generic[T]):
        item: T

    @dataclasses.dataclass
    class Listed(Box[list[U]], typing.Generic[U]):
        count: U

    @dataclasses.dataclass
    class Ints(Listed[int]):
        pass

    class Again(Ints):
        pass

    assert Validator(Box[int]) == Validator(Box[int])
    assert Validator(Box[int]) != Validator(Box[str])
    assert Validator(Box[int]).is_valid(Box(1))
    assert not Validator(Box[int]).is_valid(Box("one"))
    for derived in (Ints, Again):
        assert Validator(derived).is_valid(derived([1], 2))
        assert not Validator(derived).is_valid(derived(["one"], 2))  # ty: ignore[invalid-argument-type]
        assert not Validator(derived).is_valid(derived([1], "two"))  # ty: ignore[invalid-argument-type]
    assert Validator(list[Box[int]]).is_valid([Box(1)])


def test_a_generic_record_class_reads_its_parameters_or_refuses() -> None:
    """A parameter is bound by an argument or a base, or the class is refused.

    Written bare, a class whose field names a parameter with no default names
    no one set, and a type variable two bases bind two ways makes one field two
    types; each is refused naming the class and the variable.
    """

    @dataclasses.dataclass
    class Box(typing.Generic[T]):
        item: T

    @dataclasses.dataclass
    class Other(typing.Generic[T]):
        other: T

    @dataclasses.dataclass
    class Both(Box[int], Other[str]):
        pass

    with pytest.raises(
        NotImplementedError,
        match=r"is generic in .+, and its field .+ is declared with it",
    ):
        Validator(Box)
    with pytest.raises(
        NotImplementedError,
        match=r"binds .+ to .+ through one base and to .+ through another",
    ):
        Validator(Both)


@pytest.mark.skipif(
    sys.version_info < (3, 13), reason="a type parameter's default is 3.13"
)
def test_a_parameter_left_out_takes_its_default() -> None:
    """A bare generic class is its defaults, as PEP 696 reads it."""
    if sys.version_info >= (3, 13):
        defaulted = typing.TypeVar("defaulted", default=int)

        @dataclasses.dataclass
        class Holder(typing.Generic[defaulted]):
            item: defaulted

        assert Validator(Holder) == Validator(Holder[int])
        assert not Validator(Holder).is_valid(Holder("one"))
        assert Validator(Holder[str]).is_valid(Holder("one"))


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="a generic TypedDict or NamedTuple is 3.11"
)
def test_a_generic_typeddict_and_named_tuple_applied_read_their_arguments() -> None:
    """A generic `TypedDict` and `NamedTuple` read as the dataclass does.

    A `TypedDict`'s bases are not on its `__mro__`, so a child's arguments are
    read off its original bases; a named tuple's positions hold what its
    parameters are applied to.
    """

    class Pair(typing.TypedDict, typing.Generic[T]):
        first: T

    class Named(Pair[int]):
        second: str

    class Point(typing.NamedTuple, typing.Generic[T]):
        x: T

    class Plain(typing.TypedDict):
        first: int

    assert Validator(Pair[int]) == Validator(Plain)
    assert Validator(Named).is_valid({"first": 1, "second": "a"})
    assert not Validator(Named).is_valid({"first": "1", "second": "a"})
    assert Validator(Point[int]).is_valid(Point(1))
    assert not Validator(Point[int]).is_valid(Point("1"))
