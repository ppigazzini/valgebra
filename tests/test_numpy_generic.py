"""numpy's own generic: the class met with a record of its two arguments.

`docs/dev/03-frontend.md` says what the form denotes. The stand-in below is as much
of numpy as the frontend reads, so these hold the reading on a lane with no numpy
installed, and the last test holds it against numpy where it is.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Annotated, Any, Literal

import annotated_types as at
import pytest

from valgebra import Validator, intersection

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Generic:
    def __class_getitem__(cls, args: object) -> types.GenericAlias:
        return types.GenericAlias(cls, args)


class _Float64:
    kind, itemsize = "f", 8


class _Float32:
    kind, itemsize = "f", 4


class _Long:
    kind, itemsize = "i", 8


class _LongLong:
    kind, itemsize = "i", 8


class _Str:
    kind, itemsize = "U", 0


class _Floating:
    """An abstract scalar type, of which `dtype` makes no dtype."""


class _Dtype(_Generic):
    def __init__(self, scalar: type) -> None:
        if scalar is _Floating:
            msg = "no dtype of an abstract scalar"
            raise TypeError(msg)
        self.type, self.kind, self.itemsize = scalar, scalar.kind, scalar.itemsize  # ty: ignore[unresolved-attribute]


class _Ndarray(_Generic):
    def __init__(self, shape: tuple[int, ...], scalar: type) -> None:
        self.shape = shape
        self.dtype = _Dtype(scalar)


@pytest.fixture
def numpy() -> Iterator[types.ModuleType]:
    stand_in = types.ModuleType("numpy")
    stand_in.ndarray = _Ndarray  # ty: ignore[unresolved-attribute]
    stand_in.dtype = _Dtype  # ty: ignore[unresolved-attribute]
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "numpy", stand_in)
        yield stand_in


def _array(shape: object, scalar: object = Any) -> object:
    dtype = Any if scalar is Any else _Dtype[scalar]
    return _Ndarray[shape, dtype]


@pytest.mark.usefixtures("numpy")
def test_the_form_reads_its_shape_and_its_elements() -> None:
    rows_of_three = Validator(_array(tuple[int, Literal[3]], _Float64))
    assert rows_of_three.is_valid(_Ndarray((4, 3), _Float64))
    assert not rows_of_three.is_valid(_Ndarray((4, 4), _Float64))
    assert not rows_of_three.is_valid(_Ndarray((3,), _Float64))
    assert not rows_of_three.is_valid(_Ndarray((4, 3), _Float32))
    assert not rows_of_three.is_valid((4, 3))


@pytest.mark.usefixtures("numpy")
def test_elements_are_read_by_kind_and_width() -> None:
    # Two classes of one kind and width are one element, as `int64` and `longlong` are
    # where the platform makes them one width.
    longs = Validator(_array(Any, _Long))
    assert longs.is_valid(_Ndarray((1,), _LongLong))
    # A scalar of no fixed width is its kind alone: a string dtype holds its length.
    strings = Validator(_array(Any, _Str))
    array = _Ndarray((1,), _Str)
    array.dtype.itemsize = 20
    assert strings.is_valid(array)


@pytest.mark.usefixtures("numpy")
def test_the_relations_read_what_the_form_says() -> None:
    wide = _array(tuple[Any, ...], _Float64)
    narrow = _array(tuple[int, int], _Float64)
    assert Validator(narrow).relation_to(wide) == "subset"
    assert Validator(narrow).relation_to(_Ndarray) == "subset"
    assert intersection(wide, _array(tuple[Any, ...], _Float32)).is_empty()
    assert intersection(narrow, _array(tuple[int], _Float64)).is_empty()
    assert not intersection(wide, narrow).is_empty()


@pytest.mark.usefixtures("numpy")
@pytest.mark.parametrize(
    ("form", "says"),
    [
        (lambda: _array(int), "is no shape"),
        (lambda: _Ndarray[Any, int], "is no dtype"),
        (lambda: _array(Any, _Floating), "names no one scalar type"),
        (lambda: _Ndarray[int], "needs a shape and a dtype"),
    ],
)
def test_what_names_no_array_is_refused(form: Any, says: str) -> None:
    with pytest.raises(NotImplementedError, match=says):
        Validator(form())


def test_a_class_of_that_name_with_no_numpy_is_refused() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.delitem(sys.modules, "numpy", raising=False)
        with pytest.raises(NotImplementedError, match="unsupported typing form"):
            Validator(_Ndarray[Any, Any])


def test_an_array_element_bound_is_a_predicate() -> None:
    np = pytest.importorskip("numpy")
    non_negative = Validator(
        Annotated[np.ndarray, at.Predicate(lambda x: bool((x >= 0).all()))]
    )
    assert non_negative.is_valid(np.zeros(3))
    assert not non_negative.is_valid(np.array([-1.0, 2.0]))
    assert non_negative.relation_to(np.ndarray) == "subset"
    assert Validator(np.ndarray).relation_to(non_negative) == "undecided"
    # And an order bound on the array reads the one-element arrays, which is the
    # stated denotation and not what the spelling suggests.
    bounded = Validator(Annotated[np.ndarray, at.Ge(0)])
    assert bounded.is_valid(np.zeros(1))
    assert not bounded.is_valid(np.zeros(3))


def test_the_form_reads_numpys_own_arrays() -> None:
    np = pytest.importorskip("numpy")
    typing = pytest.importorskip("numpy.typing")
    doubles = Validator(typing.NDArray[np.float64])
    assert doubles.is_valid(np.zeros(3))
    assert doubles.is_valid(np.zeros((2, 2), dtype=">f8"))
    assert not doubles.is_valid(np.zeros(3, dtype="float32"))
    assert Validator(typing.NDArray[np.longlong]).is_valid(np.zeros(1, dtype="int64"))
    assert intersection(doubles, typing.NDArray[np.float32]).is_empty()
    with pytest.raises(NotImplementedError, match="names no one scalar type"):
        Validator(typing.NDArray[np.floating])
