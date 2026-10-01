"""Forms the frontend reads as something other than what they say.

A schema is built from an annotation, and the failure that matters here is the
quiet one: a form read as a *different* schema rather than refused. A refusal a
caller sees is a message they act on; a schema that admits more than the
annotation names is a validator that passes what it was written to stop.

Each row below is a form whose reading was not its meaning, with the value that
shows it, and beside it the form it must keep reading as it did.

Some rows carry a `ty: ignore`. A form read as something other than what it says
is often one a checker refuses outright, and the suppression marks that: the
annotation is deliberate, and what is asserted is the reading the frontend gives
it.
"""

from __future__ import annotations

import builtins
import collections.abc
import dataclasses
import subprocess
import sys
import typing
import warnings
from typing import Annotated, Literal

import annotated_types as at
import pytest

from valgebra import Regex, Validator, nothing

#: PEP 814's frozen dict, which arrives in 3.15; `None` below it.
FROZENDICT = getattr(builtins, "frozendict", None)


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


# BOUND: MAX_GROUPING_DEPTH
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


def test_an_object_carrying_metadata_and_no_origin_is_a_value() -> None:
    """`__metadata__` alone is not an `Annotated` form.

    The dispatch read any object carrying `__metadata__` as `Annotated` and then
    asked it for `__origin__`, so an object without one failed with a bare
    `AttributeError`, a type the build never raises. It is read as every other
    object is: a value, admitting itself.
    """

    class Carrier:
        __metadata__ = (1,)

    carrier = Carrier()
    validator = Validator(carrier)
    assert validator.is_valid(carrier) is True
    assert validator.is_valid(Carrier()) is False


def test_a_typing_extensions_form_reads_as_its_typing_spelling() -> None:
    """`typing_extensions` spells a form with an object of its own before a release.

    On 3.10 that is `Any`, `Never`, `Self`, `LiteralString`, the field
    qualifiers and `Unpack`, and before 3.15 `TypeAliasType`. Read against
    `typing` alone, `Never` was a literal of the form object, `Any` a class
    nothing is an instance of, `Required[int]` an unsupported form, and an alias
    a literal of itself. Where a release spells the form in `typing` the two are
    one object, and this reads the same.
    """
    extensions = pytest.importorskip("typing_extensions")
    assert Validator(extensions.Never) == Validator(nothing)
    assert Validator(extensions.Any) == Validator(typing.Any)
    assert Validator(list[extensions.Any]).is_valid([1, "a"])
    for qualifier in (extensions.Required, extensions.NotRequired, extensions.ReadOnly):
        assert Validator(qualifier[int]) == Validator(int)
    ints = extensions.TypeAliasType("Ints", list[int])
    assert Validator(ints) == Validator(list[int])
    tail = Validator(tuple[int, extensions.Unpack[tuple[str, ...]]])
    assert tail.is_valid((1, "a", "b"))
    assert not tail.is_valid((1, 2))
    for construct in (extensions.Self, extensions.LiteralString):
        with pytest.raises(NotImplementedError, match="typing construct"):
            Validator(construct)

    account = Validator(
        extensions.TypedDict(
            "Account",
            {"owner": extensions.Required[str], "balance": extensions.ReadOnly[int]},
            total=False,
        )
    )
    assert account.is_valid({"owner": "Ada"})
    assert not account.is_valid({"balance": 1})


#: A validator built on one thread while another imports `typing_extensions`,
#: with the interpreter switching threads as often as it can.
_BUILT_WHILE_IMPORTING = """
import sys
import threading
from typing import Literal

from valgebra import Validator

sys.setswitchinterval(1e-6)
importing = True


def build():
    while importing:
        Validator(Literal["a", 3])


thread = threading.Thread(target=build)
thread.start()
import typing_extensions as extensions

importing = False
thread.join()
assert Validator(extensions.ReadOnly[int]) == Validator(int)
assert Validator(extensions.TypeAliasType("Ints", list[int])) == Validator(list[int])
"""


def test_a_typing_extensions_form_read_while_the_module_imports_is_not_lost() -> None:
    """A module is in `sys.modules` before its body has run.

    A validator built on another thread while `typing_extensions` imports read
    the module's forms half-defined, and kept the ones the body had not reached
    as absent: `ReadOnly[int]` was an unsupported form for the rest of the
    process. Run in a fresh interpreter, where the module is not yet loaded.
    """
    pytest.importorskip("typing_extensions")
    subprocess.run(  # noqa: S603 - this interpreter, a fixed script, no shell
        [sys.executable, "-c", _BUILT_WHILE_IMPORTING], check=True
    )


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("typing", "TypedDict"),
        ("typing", "NamedTuple"),
        ("typing_extensions", "TypedDict"),
        ("typing_extensions", "NamedTuple"),
    ],
)
def test_a_bare_typed_dict_or_named_tuple_is_refused(module: str, name: str) -> None:
    """The base a class is declared from is not a type, and not a literal either.

    Each is a function or a form at runtime, and read as a literal it admitted
    that one object and nothing a caller has. A checker reads `NamedTuple` as
    every named tuple class and refuses `TypedDict` as a type.
    """
    form = getattr(pytest.importorskip(module), name)
    with pytest.raises(NotImplementedError, match="the base a class is declared from"):
        Validator(form)


def test_the_unsupported_form_refusal_names_the_forms_it_reads() -> None:
    """The fallback lists what the dispatch reads, and each of those builds.

    The list left `frozenset` out, which `frozenset[int]` builds, so a reader
    looking for the spelling that works was told there was none.
    """
    with pytest.raises(
        NotImplementedError, match="the subscripted forms read"
    ) as caught:
        Validator(collections.abc.Mapping[str, int])
    listed = str(caught.value).split("the subscripted forms read are ")[1]
    for container in (list, set, frozenset, dict, tuple):
        assert container.__name__ in listed
        Validator(container[int] if container is not dict else dict[int, int])


def test_an_annotation_python_cannot_evaluate_raises_what_evaluating_it_raises() -> (
    None
):
    """A name the class's module does not define raises `NameError` at build.

    The class's annotations are resolved when the validator is built, as
    `docs/16-api.md` says, so a name imported only under `TYPE_CHECKING` -- the
    name here, which is never defined -- is Python's error and not a refusal.
    """

    @dataclasses.dataclass
    class Price:
        amount: NeverDefined  # noqa: F821  # ty: ignore[unresolved-reference]

    with pytest.raises(NameError, match="NeverDefined"):
        Validator(Price)


def test_a_pattern_that_is_not_text_says_what_it_is() -> None:
    """A refusal names the marker it read, not a kind it guessed."""
    with pytest.raises((TypeError, ValueError, NotImplementedError)) as caught:
        Validator(Annotated[str, Regex(123)])  # ty: ignore[invalid-argument-type]
    assert "bytes" not in str(caught.value)


def test_a_frozenset_literal_is_refused_as_its_set_sibling_is() -> None:
    """`frozenset({int})` names a container of ints, and is not one.

    A set literal `{int}` is refused with a sentence naming `set[T]`. Its
    frozen sibling fell past every arm to the constant fallthrough and was
    interned: `Validator(frozenset({int}))` built `Literal[frozenset({<class
    'int'>}))`, a schema admitting one frozen set of one type object and no
    other value. The annotation names a frozen set of integers, so the reading
    is not a narrower version of what was written -- it is a different set,
    and the schema a caller gets refuses every value they have.
    """
    with pytest.raises(NotImplementedError, match="write a frozen set as"):
        Validator(frozenset({int}))
    # The empty one is the same mistake with nothing inside it.
    with pytest.raises(NotImplementedError, match="write a frozen set as"):
        Validator(frozenset())
    # And the form it names builds, which is what the message points at.
    assert Validator(frozenset[int]).is_valid(frozenset({1, 2}))


@pytest.mark.skipif(FROZENDICT is None, reason="frozendict arrives in 3.15")
def test_a_frozen_dict_is_a_mapping_and_not_a_dict() -> None:
    """A `frozendict` value is a `Mapping`, and no schema that denotes dicts admits it.

    The record and mapping node denotes dicts, and a `frozendict` is not a
    `dict` subclass, so `dict[K, V]` and a record refuse one exactly as they
    refuse any other mapping; the `Mapping` atom is the `isinstance` check that
    admits it. `frozendict[K, V]` names a set of frozen dicts, a carrier the
    node set does not have, and is refused rather than read as `dict[K, V]`.
    """
    assert FROZENDICT is not None
    with pytest.raises(NotImplementedError, match="a frozen dict literal is not a"):
        Validator(FROZENDICT(a=int))
    value = FROZENDICT(a=1)
    assert not Validator(dict[str, int]).is_valid(value)
    assert not Validator({"a": int}).is_valid(value)
    assert Validator(collections.abc.Mapping).is_valid(value)
    with pytest.raises(NotImplementedError, match="unsupported typing form"):
        Validator(FROZENDICT[str, int])


def test_a_frozen_set_of_constants_is_still_a_value() -> None:
    """A frozen set *of constants* is a value, and stays one.

    The refusal above is about a frozen set holding types. One holding values
    is an ordinary constant -- a `Literal` argument is exactly this -- so
    refusing every frozen set would take a spelling the typing spec allows.
    """
    inside = Validator({"tags": frozenset[str]})
    assert inside.is_valid({"tags": frozenset({"a"})})


#: Forms the typing spec gives a meaning to, which no runtime value belongs to.
#:
#: Read as the fallback literal, each would denote the form object itself and
#: the schema would refuse every value without saying why. `docs/03` lists them
#: with the reason each has no set; this is the table that holds the page.
NO_SET: list[tuple[str, object]] = [
    ("a type variable", typing.TypeVar("T")),  # ty: ignore[invalid-legacy-type-variable]
    ("a parameter specification", typing.ParamSpec("P")),  # ty: ignore[invalid-paramspec]
    ("Final", typing.Final),
    ("ClassVar", typing.ClassVar),
    ("a tuple literal", (int, str)),
    ("a set literal", {int}),
    ("a frozen set literal", frozenset({int})),
]

if sys.version_info >= (3, 11):
    # Two names the floor does not have. Read behind the guard rather than
    # dropped, because both are forms a typed-Python author writes and the
    # reading a schema would otherwise give them is the silent one.
    NO_SET += [("Self", typing.Self), ("LiteralString", typing.LiteralString)]

if FROZENDICT is not None:
    # The dict literal's frozen sibling. A `frozendict` is not a `dict`, so it
    # misses the dict literal's arm and, without one of its own, was read as a
    # constant: a schema admitting that one mapping of a type object.
    NO_SET += [("a frozen dict literal", FROZENDICT(a=int))]


@pytest.mark.parametrize(
    "form", [row[1] for row in NO_SET], ids=[row[0] for row in NO_SET]
)
def test_a_form_with_no_set_is_refused_rather_than_read_as_a_literal(
    form: object,
) -> None:
    """Refused at build, with a sentence, rather than silently denoting itself.

    The failure this pins is the quiet one: a schema that admits nothing and
    says nothing, reached by a caller who wrote a form the spec has a meaning
    for and this library does not.
    """
    with pytest.raises(NotImplementedError) as caught:
        Validator(form)
    assert len(str(caught.value)) > 40, caught.value


def test_a_generic_parametrisation_of_a_user_class_is_refused() -> None:
    """A user `Generic[T]` erases its parameter, so the argument narrows nothing.

    Built as the class alone it would admit every instance whatever its
    parameter, which is a wider set than the annotation names.
    """
    T = typing.TypeVar("T")

    class Box(typing.Generic[T]):  # type: ignore[misc]
        pass

    with pytest.raises(NotImplementedError, match="unsupported typing form"):
        Validator(Box[int])


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Unpack arrives in 3.11")
def test_an_unpack_outside_a_tuple_is_refused() -> None:
    """`Unpack[X]` binds element types into a tuple, so it has none alone."""

    class Fields(typing.TypedDict):
        a: int

    with pytest.raises(NotImplementedError, match="unsupported typing form"):
        Validator(typing.Unpack[Fields])


def test_a_bare_protocol_is_refused() -> None:
    """Membership of a protocol is `isinstance`, which the bare form refuses."""
    with pytest.raises(NotImplementedError, match="runtime_checkable"):
        Validator(typing.Protocol)


@typing.runtime_checkable
class Sized(typing.Protocol):
    """A protocol the decorator was applied to."""

    def __len__(self) -> int: ...


class InheritsTheMark(Sized, typing.Protocol):
    """A subclass protocol: it inherits the decorator's mark, not the decorator."""

    def __iter__(self) -> typing.Iterator[object]: ...


@typing.runtime_checkable
class CarriesTheMark(Sized, typing.Protocol):
    """The same subclass protocol, decorated itself."""

    def __iter__(self) -> typing.Iterator[object]: ...


def test_a_protocol_is_runtime_checkable_only_where_it_is_decorated() -> None:
    """A subclass protocol is refused unless the decorator was applied to it.

    It inherits the attribute `@runtime_checkable` sets, so `isinstance`
    answers for it, with a `DeprecationWarning` on 3.15 and a `TypeError` from
    3.20. The walk reads a warning raised as an error as a non-member, so under
    `-W error` such a schema would admit nothing and say nothing; it is refused
    by name on every release instead. Decorated itself, the same class is read,
    and a check against it raises no warning.
    """
    with pytest.raises(
        NotImplementedError, match=r"InheritsTheMark.* inherits @runtime_checkable"
    ):
        Validator(InheritsTheMark)
    schema = Validator(CarriesTheMark)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert schema.is_valid([1])
        assert not schema.is_valid(1)


@typing.runtime_checkable
class HoldsX(typing.Protocol):
    """A protocol with one data member."""

    x: int


@typing.runtime_checkable
class HoldsM(typing.Protocol):
    """A protocol with one method."""

    def m(self) -> int: ...


class XFromAHook:
    """An `x` a `__getattr__` hook serves, with nothing on the class to find."""

    def __getattr__(self, name: str) -> int:
        if name == "x":
            return 1
        raise AttributeError(name)


class MFromAHook:
    """An `m` a `__getattr__` hook serves, with nothing on the class to find."""

    def __getattr__(self, name: str) -> typing.Callable[[], int]:
        if name == "m":
            return lambda: 1
        raise AttributeError(name)


class XWhoseGetterRaises:
    """An `x` that is a property, whose getter raises an ordinary exception."""

    @property
    def x(self) -> int:
        raise RuntimeError("the getter fails")


class XInAnEmptySlot:
    """An `x` declared as a slot that no instance here assigns."""

    __slots__ = ("x",)


class NamesHoldsX(HoldsX):
    """A class naming the protocol as a base, and assigning no `x`."""


class XHoldsText:
    """An `x` that is present and holds a value outside `int`."""

    x = "text"


def test_a_protocol_admits_what_the_running_release_admits() -> None:
    """A protocol is Python's own check, and the check moved at 3.12.

    `isinstance` against a runtime-checkable protocol finds a member with
    `hasattr` on 3.10 and 3.11 and with `inspect.getattr_static` from 3.12.
    `hasattr` runs a `__getattr__` hook and a property's getter; the static
    lookup runs neither, and finds a slot's descriptor on the class whether or
    not the instance has filled it. So a member served by a hook, a getter that
    raises, and an empty slot change sides -- the four values below, each one
    of the rows the schema-language page tabulates. On 3.10 and 3.11 the raising
    getter makes `isinstance` raise, and the walk reads that as a non-member.

    Two things hold on every release: the check asks whether a member is
    present and not what it holds, and an instance of a class that names the
    protocol as a base passes without a member being asked for -- here one
    with no `x` at all.
    """
    reads_statically = sys.version_info >= (3, 12)
    holds_x = Validator(HoldsX)
    holds_m = Validator(HoldsM)
    assert holds_x.is_valid(XFromAHook()) is not reads_statically
    assert holds_m.is_valid(MFromAHook()) is not reads_statically
    assert holds_x.is_valid(XWhoseGetterRaises()) is reads_statically
    assert holds_x.is_valid(XInAnEmptySlot()) is reads_statically
    # mypy and pyright refuse to construct the class, since it declares `x` and
    # assigns none; Python constructs it, and that instance is the point.
    unassigned = NamesHoldsX()
    assert not hasattr(unassigned, "x")
    assert holds_x.is_valid(unassigned)
    assert holds_x.is_valid(XHoldsText())


def test_a_qualifier_is_unwrapped_wherever_it_is_written() -> None:
    """A field qualifier compiles the type it qualifies, inside a record or not.

    It survives hint resolution because field metadata is kept, so the frontend
    meets one at the top level too. Unwrapping is the reading in both places,
    which is what makes the two schemas equal rather than merely alike.
    """
    for name in ("Required", "NotRequired"):
        qualifier = getattr(typing, name, None)
        if qualifier is None:  # pragma: no cover - both arrive in 3.11
            continue
        assert repr(Validator(qualifier[int])) == repr(Validator(int))
        assert Validator(qualifier[int]).is_valid(1)
        assert not Validator(qualifier[int]).is_valid("1")


def test_a_union_of_one_member_is_that_member() -> None:
    """`Union[int]` is `int`: typing collapses it before the frontend sees it.

    Written through `getattr` because the linter rewrites the subscript into
    `int`, which is the very collapse this asserts and would leave the row
    asserting that `int` is `int`.
    """
    one = getattr(typing, "Union")[int]  # noqa: B009
    assert repr(Validator(one)) == "int"


def test_a_literal_of_an_int_subclass_denotes_the_value_it_was_given() -> None:
    """`Literal[Sub(1)]` is the subclass instance, not the integer beside it.

    The two are different sets -- neither admits the other's value -- and the
    repr is Python's repr of the constant. A subclass that does not override
    `__repr__` therefore renders `Literal[1]`, which reads back as the integer:
    the rendering is faithful to the object and does not round-trip. That is
    the general rule for a literal of an object whose repr is not its
    constructor, and it is pinned here rather than left for a reader to
    discover.
    """

    class Plain(int):
        pass

    class Shown(int):
        def __repr__(self) -> str:
            return f"Shown({int(self)})"

    plain, integer = Validator(Literal[Plain(1)]), Validator(Literal[1])  # ty: ignore[invalid-type-form]
    assert plain.is_valid(Plain(1))
    assert not plain.is_valid(1)
    assert not integer.is_valid(Plain(1))
    # The repr of the subclass with no `__repr__` of its own reads back as the
    # integer, which is a different set.
    assert repr(plain) == "Literal[1]"
    assert repr(integer) == "Literal[1]"
    # One that spells itself renders so, and reads back as what it is.
    assert repr(Validator(Literal[Shown(1)])) == "Literal[Shown(1)]"  # ty: ignore[invalid-type-form]


def test_a_named_tuple_prints_as_its_name_like_every_other_class() -> None:
    """`docs/03` says a class prints as its name, and a NamedTuple is one.

    A dataclass and a plain class print `DC` and `Plain`; a `NamedTuple`
    printed `intersection(tuple[int], NT)`. The schema is a meet either way --
    an `isinstance` beside a deep check of the fields -- and the renderer named
    the class for the dataclass shape only, because it looked for an attribute
    record and a NamedTuple's fields are positions. Two class forms, two
    renderings, and one rule on the page.
    """

    class Point(typing.NamedTuple):
        x: int
        y: str = "origin"

    assert repr(Validator(Point)) == "Point"
    # And the schema is the meet it was: the class *and* the field types.
    assert Validator(Point).is_valid(Point(1, "a"))
    assert not Validator(Point).is_valid((1, "a"))
    assert not Validator(Point).is_valid(Point(1, 2))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_a_named_tuple_with_defaults_admits_the_value_the_default_builds() -> None:
    """A default is supplied by the constructor, not a field that may be absent.

    The tuple has every position whether the caller wrote it or not.
    """

    class Point(typing.NamedTuple):
        x: int
        y: str = "origin"

    schema = Validator(Point)
    assert schema.is_valid(Point(1))
    assert schema.is_valid(Point(1, "given"))


def test_a_dataclass_is_read_through_its_fields_however_it_is_declared() -> None:
    """A declaration changes how a class is built, not what it holds.

    `slots`, `frozen` and `KW_ONLY` each name the same set of values.
    """

    @dataclasses.dataclass(slots=True)
    class Slotted:
        a: int

    @dataclasses.dataclass(frozen=True)
    class Frozen:
        a: int

    assert Validator(Slotted).is_valid(Slotted(1))
    assert not Validator(Slotted).is_valid(Frozen(1))
    assert Validator(Frozen).is_valid(Frozen(1))


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="on 3.10 a string annotation of KW_ONLY does not resolve to a type",
)
def test_a_keyword_only_marker_is_a_declaration_rather_than_a_field() -> None:
    """`KW_ONLY` says how the constructor takes its arguments, not what is held.

    It carries no value on an instance, so a schema naming it as a field would
    ask every value for an attribute none of them has.
    """

    @dataclasses.dataclass
    class KeywordOnly:
        a: int
        _: dataclasses.KW_ONLY
        b: str = "x"

    schema = Validator(KeywordOnly)
    assert schema.is_valid(KeywordOnly(1, b="y"))
    assert not schema.is_valid(KeywordOnly(1, b=2))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    assert "_" not in repr(schema)


#: A constraint over a literal whose constant cannot be asked it.
#:
#: The bare-kind rows beside them are already refused, and a literal is a value
#: *of* a kind -- so the two spellings are the same question and must get the
#: same answer.
LITERAL_MISFITS: list[tuple[str, object]] = [
    ("a length over an integer literal", Annotated[Literal[1], at.MinLen(1)]),
    ("a length over a boolean literal", Annotated[Literal[True], at.MaxLen(1)]),
    ("a length over a float literal", Annotated[Literal[1.5], at.MinLen(1)]),  # ty: ignore[invalid-type-form]
    ("an order over a string literal", Annotated[Literal["a"], at.Ge(0)]),
    ("an order over a bytes literal", Annotated[Literal[b"a"], at.Ge(0)]),
    ("a pattern over an integer literal", Annotated[Literal[1], Regex("a+")]),
    ("a divisor over a string literal", Annotated[Literal["a"], at.MultipleOf(2)]),
    # A union of them is the same question asked of each member: no member can
    # answer, so the union cannot either.
    (
        "a length over a union of integer literals",
        Annotated[Literal[1, 2], at.MinLen(1)],
    ),
]


@pytest.mark.parametrize(
    "form", [row[1] for row in LITERAL_MISFITS], ids=[row[0] for row in LITERAL_MISFITS]
)
def test_a_constraint_a_literal_cannot_answer_is_refused(form: object) -> None:
    """A literal is a value of a kind, and is asked what that kind can answer.

    `Annotated[int, MinLen(1)]` is refused because reading a length off an
    integer raises and the walk reads a raise as a non-member -- so the schema
    would admit nothing and say nothing about why. `Annotated[Literal[1],
    MinLen(1)]` is the same schema one value narrower, and it compiled: it
    admitted no value and reported itself *inhabited*, which is a set that
    exists according to the library and holds nothing according to the walk.

    The kind is the constant's, read from the pool where the constant lives.
    """
    with pytest.raises(NotImplementedError, match="values have no"):
        Validator(form)


def test_a_constraint_a_literal_can_answer_still_builds() -> None:
    """The refusal is about the kind, not about the form.

    A literal whose constant *can* be asked the constraint narrows exactly as
    its kind does: to the constant where it satisfies the bound, and to the
    empty set where it does not. Refusing every constrained literal would take
    a spelling that means something.
    """
    ok = Validator(Annotated[Literal["ab"], at.MinLen(1)])
    assert ok.is_valid("ab")
    assert not ok.is_valid("a")

    # The bound the constant misses is the empty set, and says so.
    missed = Validator(Annotated[Literal["ab"], at.MinLen(5)])
    assert missed.is_empty()
    assert not missed.is_valid("ab")

    # And the other families, each over a constant of the kind that answers it.
    assert Validator(Annotated[Literal[4], at.Ge(0)]).is_valid(4)
    assert Validator(Annotated[Literal[4], at.MultipleOf(2)]).is_valid(4)
    assert Validator(Annotated[Literal["ab"], Regex("a.")]).is_valid("ab")
