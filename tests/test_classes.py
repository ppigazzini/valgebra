import dataclasses
import enum
import sys
import typing
from typing import NamedTuple, NewType, Protocol, TypedDict, runtime_checkable

import pytest

from valgebra import ValidationError, validator


class User(TypedDict):
    name: str
    age: int


class PartialUser(TypedDict, total=False):
    nickname: str
    email: str


def test_typeddict_requires_all_keys_by_default() -> None:
    schema = validator(User)
    assert schema.is_valid({"name": "Ada", "age": 36})
    assert not schema.is_valid({"name": "Ada"})
    assert not schema.is_valid({"name": "Ada", "age": "old"})


def test_typeddict_total_false_makes_keys_optional() -> None:
    schema = validator(PartialUser)
    assert schema.is_valid({})
    assert schema.is_valid({"email": "a@b.c"})
    assert schema.is_valid({"email": "a@b.c", "nickname": "ada"})
    assert not schema.is_valid({"email": 1})  # checked when present


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Required marker")
def test_typeddict_required_marker_within_total_false() -> None:
    class Part(TypedDict, total=False):
        nickname: str
        email: typing.Required[str]

    schema = validator(Part)
    assert schema.is_valid({"email": "a@b.c"})
    assert not schema.is_valid({"nickname": "ada"})  # email is required


@dataclasses.dataclass
class Point:
    x: int
    y: int


def test_dataclass_checks_instance_and_attributes() -> None:
    assert validator(Point).is_valid(Point(1, 2))
    assert not validator(Point).is_valid(Point(1, "y"))  # ty: ignore[invalid-argument-type]
    assert not validator(Point).is_valid({"x": 1, "y": 2})


def test_dataclass_attribute_failure_reports_the_path() -> None:
    with pytest.raises(ValidationError) as info:
        validator(Point).validate(Point(1, "y"))  # ty: ignore[invalid-argument-type]
    assert info.value.code == "int_type"
    assert info.value.path == ("y",)


class Pair(NamedTuple):
    a: int
    b: str


def test_namedtuple_is_a_deep_instance_check() -> None:
    assert validator(Pair).is_valid(Pair(1, "x"))
    assert not validator(Pair).is_valid(Pair(1, 2))  # ty: ignore[invalid-argument-type]
    assert not validator(Pair).is_valid((1, "x"))


class Color(enum.Enum):
    RED = 1
    GREEN = 2


def test_enum_accepts_its_members() -> None:
    assert validator(Color).is_valid(Color.RED)
    assert validator(Color).is_valid(Color.GREEN)
    assert not validator(Color).is_valid(1)


@runtime_checkable
class Sized(Protocol):
    def __len__(self) -> int: ...


def test_runtime_checkable_protocol() -> None:
    assert validator(Sized).is_valid([1, 2])
    assert validator(Sized).is_valid("abc")
    assert not validator(Sized).is_valid(5)


class NotRuntime(Protocol):
    def ping(self) -> None: ...


def test_non_runtime_protocol_is_rejected() -> None:
    with pytest.raises(NotImplementedError):
        validator(NotRuntime)


UserId = NewType("UserId", int)


def test_newtype_validates_its_supertype() -> None:
    assert validator(UserId).is_valid(5)
    assert not validator(UserId).is_valid("x")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_pep695_type_alias_delegates_to_value() -> None:
    int_list = typing.TypeAliasType("int_list", list[int])
    assert validator(int_list).is_valid([1, 2, 3])
    assert not validator(int_list).is_valid([1, "x"])
