import dataclasses
import enum
import sys
import typing
from typing import (
    Annotated,
    NamedTuple,
    NewType,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator


class User(TypedDict):
    name: str
    age: int


class PartialUser(TypedDict, total=False):
    nickname: str
    email: str


def test_typeddict_requires_all_keys_by_default() -> None:
    schema = Validator(User)
    assert schema.is_valid({"name": "Ada", "age": 36})
    assert not schema.is_valid({"name": "Ada"})
    assert not schema.is_valid({"name": "Ada", "age": "old"})


def test_typeddict_total_false_makes_keys_optional() -> None:
    schema = Validator(PartialUser)
    assert schema.is_valid({})
    assert schema.is_valid({"email": "a@b.c"})
    assert schema.is_valid({"email": "a@b.c", "nickname": "ada"})
    assert not schema.is_valid({"email": 1})  # checked when present


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Required marker")
def test_typeddict_required_marker_within_total_false() -> None:
    class Part(TypedDict, total=False):
        nickname: str
        email: typing.Required[str]

    schema = Validator(Part)
    assert schema.is_valid({"email": "a@b.c"})
    assert not schema.is_valid({"nickname": "ada"})  # email is required


@dataclasses.dataclass
class Point:
    x: int
    y: int


def test_dataclass_checks_instance_and_attributes() -> None:
    assert Validator(Point).is_valid(Point(1, 2))
    assert not Validator(Point).is_valid(Point(1, "y"))  # ty: ignore[invalid-argument-type]
    assert not Validator(Point).is_valid({"x": 1, "y": 2})


def test_dataclass_attribute_failure_reports_the_path() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(Point).validate(Point(1, "y"))  # ty: ignore[invalid-argument-type]
    assert info.value.code == "int_type"
    assert info.value.path == ("y",)


class Pair(NamedTuple):
    a: int
    b: str


def test_namedtuple_is_a_deep_instance_check() -> None:
    assert Validator(Pair).is_valid(Pair(1, "x"))
    assert not Validator(Pair).is_valid(Pair(1, 2))  # ty: ignore[invalid-argument-type]
    assert not Validator(Pair).is_valid((1, "x"))


class Color(enum.Enum):
    RED = 1
    GREEN = 2


def test_enum_accepts_its_members() -> None:
    assert Validator(Color).is_valid(Color.RED)
    assert Validator(Color).is_valid(Color.GREEN)
    assert not Validator(Color).is_valid(1)


@runtime_checkable
class Sized(Protocol):
    def __len__(self) -> int: ...


def test_runtime_checkable_protocol() -> None:
    assert Validator(Sized).is_valid([1, 2])
    assert Validator(Sized).is_valid("abc")
    assert not Validator(Sized).is_valid(5)


class NotRuntime(Protocol):
    def ping(self) -> None: ...


def test_non_runtime_protocol_is_rejected() -> None:
    with pytest.raises(NotImplementedError):
        Validator(NotRuntime)


UserId = NewType("UserId", int)


def test_newtype_validates_its_supertype() -> None:
    assert Validator(UserId).is_valid(5)
    assert not Validator(UserId).is_valid("x")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_pep695_type_alias_delegates_to_value() -> None:
    int_list = typing.TypeAliasType("int_list", list[int])
    assert Validator(int_list).is_valid([1, 2, 3])
    assert not Validator(int_list).is_valid([1, "x"])


class BoundedUser(TypedDict):
    name: str
    age: Annotated[int, at.Ge(0)]


def test_typeddict_field_refinement_is_enforced() -> None:
    # A refinement on a field must constrain the field, not be dropped: the
    # Annotated metadata has to survive hint resolution.
    schema = Validator(BoundedUser)
    assert schema.is_valid({"name": "Ada", "age": 36})
    assert not schema.is_valid({"name": "Ada", "age": -5})
    assert "Ge(0)" in repr(schema)


@dataclasses.dataclass
class BoundedPoint:
    x: Annotated[int, at.Ge(0)]


def test_dataclass_field_refinement_is_enforced() -> None:
    schema = Validator(BoundedPoint)
    assert schema.is_valid(BoundedPoint(1))
    assert not schema.is_valid(BoundedPoint(-1))


class BoundedPair(NamedTuple):
    n: Annotated[int, at.Ge(0)]


def test_namedtuple_field_refinement_is_enforced() -> None:
    schema = Validator(BoundedPair)
    assert schema.is_valid(BoundedPair(1))
    assert not schema.is_valid(BoundedPair(-1))


@dataclasses.dataclass
class Node:
    value: int
    nxt: "Node | None" = None


def test_recursive_dataclass_is_rejected_not_crashed() -> None:
    # A class whose own type appears in a field is recursive; it must be written
    # with recursive. Compiling it directly is rejected cleanly, never crashing.
    with pytest.raises(NotImplementedError):
        Validator(Node)


def test_finite_deep_schema_still_compiles() -> None:
    # The recursion guard rejects only genuine recursion, not deep finite nesting.
    depth = 60
    schema: object = int
    value: object = 1
    for _ in range(depth):
        schema = list[schema]  # type: ignore[valid-type]
        value = [value]
    assert Validator(schema).is_valid(value)


def test_typeddict_inheritance_collects_all_fields() -> None:
    class Base(TypedDict):
        a: int

    class Derived(Base):
        b: str

    schema = Validator(Derived)
    assert schema.is_valid({"a": 1, "b": "x"})
    assert not schema.is_valid({"b": "x"})  # inherited required key missing
    assert not schema.is_valid({"a": 1})


def test_intenum_is_an_instance_check() -> None:
    class Level(enum.IntEnum):
        LOW = 1
        HIGH = 2

    schema = Validator(Level)
    assert schema.is_valid(Level.LOW)
    assert not schema.is_valid(1)  # a bare int is not an enum member
