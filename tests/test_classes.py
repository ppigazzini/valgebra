import collections
import dataclasses
import enum
import sys
import typing
from typing import (
    Annotated,
    Literal,
    NamedTuple,
    NewType,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator, complement, intersection


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


def test_typeddict_is_open_and_a_dict_literal_is_not() -> None:
    """A `TypedDict` denotes the set its own spec assigns it, which is open.

    `Validator(TD)` reads an annotation whose meaning is fixed elsewhere, and
    reading it as a narrower set is a deviation the class carries no mark of. The
    dict-literal form is this library's own spelling, and a schema written as a
    *shape* means that shape.
    """
    typed = Validator(User)
    assert typed.is_valid({"name": "Ada", "age": 36, "note": "extra"})
    # The keys it does name are still checked, and still required.
    assert not typed.is_valid({"name": "Ada", "note": "extra"})
    assert not typed.is_valid({"name": "Ada", "age": "old", "note": "extra"})

    shape = Validator({"name": str, "age": int})
    assert shape.is_valid({"name": "Ada", "age": 36})
    assert not shape.is_valid({"name": "Ada", "age": 36, "note": "extra"})


@pytest.mark.skipif(not hasattr(typing, "NoExtraItems"), reason="PEP 728 markers")
def test_typeddict_closed_and_extra_items_are_obeyed() -> None:
    """PEP 728's two markers say what a `TypedDict` allows besides the keys it names.

    A runtime that has them fills `__extra_items__` either way — with the type
    its author wrote, or with the `NoExtraItems` sentinel to say there was none.
    The sentinel is not a type, and reading it as one turns the open default into
    a record admitting exactly the sentinel: a closed record wearing an open
    one's spelling.
    """
    # The functional form, and the markers written past the floor this project
    # supports: both tools are right that they are 3.15's, and the skip above is
    # what keeps them from running anywhere they are not.
    shut = typing.TypedDict("shut", {"a": int}, closed=True)  # noqa: UP013  # ty: ignore[unknown-argument]
    extra = typing.TypedDict("extra", {"a": int}, extra_items=str)  # noqa: UP013  # ty: ignore[unknown-argument]
    plain = typing.TypedDict("plain", {"a": int})  # noqa: UP013

    assert not Validator(shut).is_valid({"a": 1, "x": 2})
    assert Validator(shut).is_valid({"a": 1})

    assert Validator(extra).is_valid({"a": 1, "x": "s"})
    assert not Validator(extra).is_valid({"a": 1, "x": 2})

    # No marker given is the spec's default, which is open.
    assert Validator(plain).is_valid({"a": 1, "x": 2})


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


# --- What a class declares is not every annotation on it ----------------------
#
# A dataclass carries annotations that name no attribute of its instances, and
# reading them all asks an instance for something it cannot have -- a schema no
# instance satisfies. Each kind of class keeps its own list of what it declares,
# and that list is what the frontend reads.


@dataclasses.dataclass
class WithInitVar:
    kept: int
    seed: dataclasses.InitVar[int]

    def __post_init__(self, seed: int) -> None:
        del seed


@dataclasses.dataclass
class WithClassVar:
    kept: int
    shared: typing.ClassVar[int] = 3


@dataclasses.dataclass
class WithLateField:
    kept: int
    derived: int = dataclasses.field(init=False, default=7)


def test_an_init_only_field_is_not_an_attribute() -> None:
    # `InitVar` names a constructor parameter; the instance does not keep it.
    schema = Validator(WithInitVar)
    assert schema.is_valid(WithInitVar(1, 2))
    assert not schema.is_valid(WithInitVar("no", 2))  # ty: ignore[invalid-argument-type]


def test_a_class_variable_is_not_an_attribute_of_the_instance() -> None:
    schema = Validator(WithClassVar)
    assert schema.is_valid(WithClassVar(1))
    assert not schema.is_valid(WithClassVar("no"))  # ty: ignore[invalid-argument-type]


def test_a_field_the_constructor_does_not_take_is_still_checked() -> None:
    # `init=False` keeps the field off the constructor, not off the instance.
    good = WithLateField(1)
    schema = Validator(WithLateField)
    assert schema.is_valid(good)
    good.derived = "no"  # ty: ignore[invalid-assignment]
    assert not schema.is_valid(good)


def test_an_unannotated_named_tuple_is_an_instance_check() -> None:
    plain = collections.namedtuple("plain", "a b")  # noqa: PYI024
    schema = Validator(plain)
    assert schema.is_valid(plain(1, "x"))
    assert not schema.is_valid((1, "x"))


@pytest.mark.skipif(sys.version_info < (3, 13), reason="ReadOnly marker")
def test_a_read_only_typed_dict_field_is_the_type_it_qualifies() -> None:
    # `ReadOnly` says whether a consumer may write the key back, which is a
    # statement about use rather than about which values belong.
    class Config(TypedDict):
        name: typing.ReadOnly[str]
        port: typing.NotRequired[typing.ReadOnly[int]]

    schema = Validator(Config)
    assert schema.is_valid({"name": "a"})
    assert schema.is_valid({"name": "a", "port": 1})
    assert not schema.is_valid({"name": "a", "port": "no"})
    assert not schema.is_valid({})


def test_a_bare_container_class_is_its_kind() -> None:
    """`list` and `list[object]` admit the same values, so they are one schema.

    An unparameterised generic names its kind's whole set — what the typing spec
    assigns it, and what the membership check always performed. Read as an
    `isinstance` atom it was a different sort of thing from the sequence node
    beside it, and neither spelling was decided below the other.
    """
    for bare, parameterised in (
        (list, list[object]),
        (tuple, tuple[object, ...]),
        (set, set[object]),
        (frozenset, frozenset[object]),
        (dict, dict[object, object]),
    ):
        assert Validator(bare) == Validator(parameterised), bare
        assert Validator(bare).is_equivalent(parameterised), bare


def test_a_bare_container_class_admits_what_it_did() -> None:
    class Sub(list):
        pass

    assert Validator(list).is_valid([])
    assert Validator(list).is_valid([1, "a"])
    assert Validator(list).is_valid(Sub([1]))
    assert not Validator(list).is_valid(())
    assert not Validator(list).is_valid({1: 2})
    assert Validator(dict).is_valid({})
    assert not Validator(dict).is_valid([])


def test_a_class_built_on_a_builtin_narrows_that_kind() -> None:
    """Every instance of a `str` subclass is a string, so the class is below it.

    The class constrains a value *within* the kind rather than standing beside
    it, which is what makes the relation decidable in one direction and refutes
    it in the other: `int` is not below `MyInt`, and `5` is not a `MyInt`.
    """

    class MyInt(int):
        pass

    class MyStr(str):
        __slots__ = ()

    assert Validator(MyInt).is_subtype_of(int)
    assert Validator(MyStr).is_subtype_of(str)
    assert Validator(MyStr).is_subtype_of(complement(int))
    assert intersection(MyInt, MyStr).is_empty()

    assert not Validator(int).is_subtype_of(MyInt)
    assert not Validator(MyInt).is_valid(5)
    assert Validator(MyInt).is_valid(MyInt(5))


def test_a_class_built_on_no_builtin_narrows_nothing() -> None:
    """A subclass of a plain class may lay down any layout, so it confines none.

    Placing such a class on one kind would be the one unsound direction — a
    claim that a value does not exist — and `Both` is that value.
    """

    class Plain:
        pass

    class MyStr(str):
        __slots__ = ()

    class Both(Plain, MyStr):
        pass

    assert Validator(Plain).is_valid(Both("x"))
    assert not intersection(Plain, str).is_empty()
    assert not Validator(Plain).is_subtype_of(str)
    assert not Validator(str).is_subtype_of(Plain)


def test_a_refutation_about_a_class_needs_a_value_of_the_kind_it_is_read_on() -> None:
    """A plain class meets a builtin kind in a set nothing here can name a value of.

    The descriptor carries a class that lays down no layout on every kind's
    line, which is right for inclusion -- a subclass may lay down any layout,
    so `Plain` is not below the complement of `int`. It is not a value. An
    integer that is an instance of `Plain` exists only if some class derives
    from both, and which classes exist is not something a snapshot of the order
    can say; the atom rule already declines exactly that for two unrelated
    classes, and a builtin kind is one more class.

    So the relation is undecided rather than refuted. A `"not_subset"` is a
    statement about a value, and this one had no value to stand on.
    """

    class Plain:
        pass

    class Laid(str):
        __slots__ = ()

    assert Validator(Plain).relation_to(complement(int)) == "undecided"
    assert Validator(int).relation_to(complement(Plain)) == "undecided"
    assert Validator(Plain).relation_to(complement(str)) == "undecided"

    # A class laid out as the kind is the case the reading is not conservative
    # about: every instance of it is a string, so it has a value on that line
    # and the refutation stands.
    assert Validator(Laid).relation_to(complement(str)) == "not_subset"
    assert Validator(str).relation_to(complement(Laid)) == "not_subset"
    # And it holds no value of another kind at all, which is a proof.
    assert Validator(Laid).relation_to(complement(int)) == "subset"

    # Two classes are unchanged: the question there is the order, not a kind.
    class Other:
        pass

    assert Validator(Plain).relation_to(Plain) == "subset"
    assert Validator(Plain).relation_to(Other) == "not_subset"


def test_a_kind_is_below_exactly_the_classes_its_builtin_derives_from() -> None:
    """A kind's own builtin builds its values, so one `issubclass` settles it.

    Every value of a kind that is not an instance of some subclass has
    `type(v)` equal to the kind's builtin, and a subclass of that builtin is
    below the kind already. So asking whether a kind is below a class is one
    question over the class order, and a `False` names a value -- an ordinary
    `5` -- rather than guessing which classes exist. That is what a refutation
    has to stand on.
    """

    class Plain:
        pass

    class MyInt(int):
        pass

    # Every kind names the builtin its values are built as, so every one of
    # them stands off a class that builtin does not derive from. `None` is the
    # exception the table itself states: its one value is a singleton rather
    # than a constructor's, so the reading declines and the refutation comes
    # from the value instead.
    for kind in (bool, int, str, bytes, float, tuple, frozenset, list, set, dict):
        assert Validator(kind).relation_to(Validator(Plain)) == "not_subset"
    assert Validator(None).relation_to(Validator(Plain)) == "not_subset"

    # The same reading proves, where the order runs the other way.
    assert Validator(bool).relation_to(Validator(int)) == "subset"
    assert Validator(int).relation_to(Validator(object)) == "subset"

    # And a subclass is below the kind rather than the kind below it: `5` is
    # not a `MyInt`.
    assert Validator(MyInt).relation_to(Validator(int)) == "subset"
    assert Validator(int).relation_to(Validator(MyInt)) == "not_subset"


def test_a_class_met_with_its_attributes_is_read_as_a_direct_instance() -> None:
    """A dataclass lowers to a class met with the attributes its instances carry.

    That meet holds a *direct* instance of the class, so `type(v) is C` settles
    `isinstance(v, D)` through the order alone and a class deriving from both
    changes nothing about that value. It is the one shape of meet that can
    refute: every other rule over a meet proves, since a member that is not
    below the supertype says nothing about the smaller set the meet denotes.

    A class laying down no builtin layout has a direct instance that is a plain
    object, carrying none of the kinds the partition names. A class laid out as
    a builtin keeps that kind, and the meet with its attributes keeps it too.
    """

    @dataclasses.dataclass
    class Point:
        x: int

    @dataclasses.dataclass
    class Counted(int):
        n: int = 0

    assert Validator(Point).relation_to(Validator(int)) == "not_subset"
    assert Validator(Point).relation_to(Validator(list)) == "not_subset"
    assert Validator(Point).relation_to(Validator(object)) == "subset"

    assert Validator(Counted).relation_to(Validator(int)) == "subset"
    assert Validator(Counted).relation_to(Validator(str)) == "not_subset"


def test_a_class_whose_metaclass_answers_isinstance_denotes_no_set() -> None:
    """`isinstance` a metaclass computes is not the class order, so nothing reads it.

    Both class readings ask the order -- one `issubclass` over a kind's own
    builtin, or the layout a class lays down -- and both are only about the
    order. A metaclass that overrides `__instancecheck__` or
    `__subclasscheck__` answers membership by running code instead, and no
    snapshot of the order predicts what that code says. So the reading declines
    rather than guessing, and the pair is undecided.

    Every abstract base class is such a class, `ABCMeta` defining both hooks,
    and so is a runtime-checkable `Protocol`.
    """
    for abstract in (collections.abc.Sequence, collections.abc.Iterable):
        assert Validator(int).relation_to(Validator(abstract)) == "undecided"
        assert Validator(list).relation_to(Validator(abstract)) == "undecided"

    @runtime_checkable
    class HasX(Protocol):
        x: int

    assert Validator(int).relation_to(Validator(HasX)) == "undecided"


def test_a_dataclass_outside_every_branch_of_a_union_is_outside_the_union() -> None:
    """The witness is one value, and it is the same value in every branch.

    A dataclass lowers to a class met with the attributes its instances carry,
    and that meet holds a *direct* instance of the class. A class laying down
    no builtin layout has a direct instance that is a plain object, carrying
    none of the kinds the partition names -- so it is in no branch that names a
    kind, and a value in no branch is outside the union.

    Reading the branches one at a time is what that single witness licenses. A
    branch naming no kind of its own ends it, since the witness may be in that
    branch for all the reading sees.
    """

    @dataclasses.dataclass
    class Point:
        x: int

    schema = Validator(Point)
    assert schema.relation_to(Validator(int | str)) == "not_subset"
    assert schema.relation_to(Validator(int | None)) == "not_subset"
    assert schema.relation_to(Validator(Literal[1, 2, 3])) == "not_subset"

    # A predicate is what the descriptor will not read through, so a union
    # carrying one is answered by the kinds or not at all.
    guarded = int | Annotated[list[int], at.Predicate(bool)]
    assert schema.relation_to(Validator(guarded)) == "not_subset"

    # Its own branch proves it instead, and the union is not refuted.
    assert schema.is_subtype_of(Point | int)
