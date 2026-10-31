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
    no one set, and a class two bases reach under two arguments makes a field
    it declares two types; each is refused naming the class and the variable.
    Two bases declared over one `TypeVar` object are two scopes, each field
    read in its own.
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

    @dataclasses.dataclass
    class OfInts(Box[int]):
        pass

    @dataclasses.dataclass
    class OfStrs(Box[str]):
        pass

    @dataclasses.dataclass
    class Twice(OfInts, OfStrs):  # ty: ignore[invalid-generic-class]
        pass

    with pytest.raises(
        NotImplementedError,
        match=r"is generic in .+, and its field .+ is declared with it",
    ):
        Validator(Box)
    assert Validator(Both).is_valid(Both("o", 1))
    assert not Validator(Both).is_valid(Both(1, "o"))  # ty: ignore[invalid-argument-type]
    with pytest.raises(
        NotImplementedError,
        match=r"through one base and to .* through another, so a field .* "
        r"declares with it names two types",
    ):
        Validator(Twice)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="a generic TypedDict is 3.11")
def test_a_type_variable_is_its_own_classs_parameter() -> None:
    """A class reusing the `TypeVar` its base is bound with has a parameter of its own.

    `Box[int]` closes `Box`'s `T`, and `IntBoxOf`'s `T` is another parameter,
    the same object in another scope: `IntBoxOf[str]` holds an `int` at `a`
    and a `str` at `b`. A dataclass redeclaring the base's field with its own
    parameter reads the field in its own scope -- the first class on the
    `__mro__` that annotates it, as `get_type_hints` reads it, so `Diamond`'s
    `a` is `Right`'s `U`, bound to `str` -- and a base applied to the
    parameters crossed binds them crossed.
    """

    class Box(typing.TypedDict, typing.Generic[T]):
        a: T

    class IntBoxOf(Box[int], typing.Generic[T]):
        b: T

    @dataclasses.dataclass
    class DBox(typing.Generic[T]):
        a: T

    @dataclasses.dataclass
    class DIntBoxOf(DBox[int], typing.Generic[T]):
        b: T

    @dataclasses.dataclass
    class Again(DBox[int], typing.Generic[T]):
        a: T  # ty: ignore[invalid-attribute-override]

    class Pair(typing.TypedDict, typing.Generic[T, U]):
        first: T
        second: U

    class Crossed(Pair[U, T], typing.Generic[T, U]):
        pass

    @dataclasses.dataclass
    class Left(DBox[int]):
        pass

    @dataclasses.dataclass
    class Right(DBox[int], typing.Generic[U]):
        a: U  # ty: ignore[invalid-attribute-override]

    @dataclasses.dataclass
    class Diamond(Left, Right[str]):
        pass

    boxed = Validator(IntBoxOf[str])
    assert boxed.is_valid({"a": 1, "b": "x"})
    assert not boxed.is_valid({"a": "x", "b": "x"})
    assert Validator(DIntBoxOf[str]).is_valid(DIntBoxOf(1, "x"))
    assert not Validator(DIntBoxOf[str]).is_valid(DIntBoxOf("x", "x"))  # ty: ignore[invalid-argument-type]
    assert Validator(Again[str]).is_valid(Again("x"))
    assert not Validator(Again[str]).is_valid(Again(1))
    assert Validator(Diamond).is_valid(Diamond("x"))
    assert not Validator(Diamond).is_valid(Diamond(1))  # ty: ignore[invalid-argument-type]
    assert Validator(Crossed[int, str]).is_valid({"first": "x", "second": 1})
    assert not Validator(Crossed[int, str]).is_valid({"first": 1, "second": "x"})


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


@pytest.mark.skipif(sys.version_info < (3, 11), reason="a TypeVarTuple is 3.11")
def test_a_class_binding_a_list_of_types_or_a_signature_is_refused() -> None:
    """A `TypeVarTuple` stands for a list of types, which no form here takes.

    `Row[int, str]` binds `shapes` to `(int, str)`, so its field is `tuple[int,
    str]`; read as one argument a parameter, it held `(1,)` and refused `(1,
    "a")`. The class applied, a class deriving from it applied, a dataclass
    alike, and a class over a `ParamSpec` applied are refused, as an alias
    declaring one is.
    """
    shapes = typing.TypeVarTuple("shapes")
    signature = typing.ParamSpec("signature")

    class Row(typing.TypedDict, typing.Generic[typing.Unpack[shapes]]):
        a: tuple[typing.Unpack[shapes]]

    class Derived(Row[int, str]):
        pass

    @dataclasses.dataclass
    class DRow(typing.Generic[typing.Unpack[shapes]]):
        a: tuple[typing.Unpack[shapes]]

    @dataclasses.dataclass
    class Hook(typing.Generic[signature]):
        call: typing.Callable[signature, int]

    for form in (Row[int, str], Row[int], Derived, DRow[int, str], Hook[[int]]):
        with pytest.raises(NotImplementedError, match="stands for a list of types"):
            Validator(form)


@pytest.mark.skipif(
    sys.version_info < (3, 13), reason="a type parameter's default is 3.13"
)
def test_a_default_naming_an_earlier_parameter_reads_its_binding() -> None:
    """A default may name a parameter before it, which its application binds.

    `Pairs[int]` binds `first` to `int`, and `second` to its default with
    `first` bound, `list[int]`, as PEP 696 reads it. `CPython` hands a base
    applied short back with the default as written, `Defaults[int, str,
    list[middle]]`, and the `middle` there is the one that base binds to `str`,
    not `middle`'s own default, `bytes`.
    """
    if sys.version_info >= (3, 13):
        first = typing.TypeVar("first")
        second = typing.TypeVar("second", default=list[first])

        class Pairs(typing.TypedDict, typing.Generic[first, second]):
            a: first
            b: second

        pairs = Validator(Pairs[int])
        assert pairs.is_valid({"a": 1, "b": [2]})
        assert not pairs.is_valid({"a": 1, "b": ["x"]})

        outer = typing.TypeVar("outer", default=int)
        middle = typing.TypeVar("middle", default=bytes)
        inner = typing.TypeVar("inner", default=list[middle])

        class Defaults(typing.TypedDict, typing.Generic[outer, middle, inner]):
            k: outer
            w: inner

        class Derived(Defaults[int, str]):
            pass

        derived = Validator(Derived)
        assert derived.is_valid({"k": 1, "w": ["s"]})
        assert not derived.is_valid({"k": 1, "w": [b"b"]})

        @dataclasses.dataclass
        class Same(typing.Generic[outer, middle, inner]):
            w: inner

        assert Validator(Same).is_valid(Same([b"b"]))
        assert not Validator(Same).is_valid(Same(["s"]))


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
