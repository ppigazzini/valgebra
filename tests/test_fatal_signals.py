"""Fatal interpreter signals propagate; ordinary exceptions fold to non-member.

A membership probe (``isinstance``, a rich comparison, ``__eq__``, ``getattr``,
``__mod__``, ``__len__``, a user predicate) that raises an *ordinary* exception
means the value cannot answer "are you in this set?", so it is a non-member (or a
``predicate_error`` for a predicate). A *fatal* signal -- a base exception that is
not an ordinary exception (KeyboardInterrupt, SystemExit, GeneratorExit), or a
MemoryError/RecursionError -- is not a membership answer at all: it propagates
instead of being silently read as a non-member, so an interrupted or
resource-exhausted check stops rather than continuing.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import TYPE_CHECKING, Annotated, Literal

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# The signals that must propagate. KeyboardInterrupt/SystemExit are base
# exceptions that are not ordinary exceptions; MemoryError/RecursionError *are*
# ordinary exceptions, so a plain "is it an Exception?" test would miss them.
FATAL = [KeyboardInterrupt, SystemExit, MemoryError, RecursionError]


def _class_raising(exc: BaseException) -> type:
    """Build a class whose ``isinstance`` check raises ``exc`` (via metaclass)."""

    class Meta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            raise exc

    class Probed(metaclass=Meta):
        pass

    return Probed


def test_ordinary_exception_in_an_isinstance_probe_is_a_non_member() -> None:
    validator = Validator(_class_raising(ValueError("boom")))
    assert validator.is_valid(object()) is False


@pytest.mark.parametrize("signal", FATAL)
def test_fatal_signal_in_isinstance_propagates(signal: type[BaseException]) -> None:
    validator = Validator(_class_raising(signal()))
    with pytest.raises(signal):
        validator.is_valid(object())


@pytest.mark.parametrize("signal", FATAL)
def test_fatal_signal_propagates_through_validate(signal: type[BaseException]) -> None:
    validator = Validator(_class_raising(signal()))
    with pytest.raises(signal):
        validator.validate(object())


def test_fatal_signal_propagates_through_membership_operator() -> None:
    validator = Validator(_class_raising(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _ = object() in validator


# -- Attribute access (getattr) ----------------------------------------------


@dataclasses.dataclass
class _Point:
    x: int


def _point_with_attr_raising(exc: BaseException) -> _Point:
    """Build a _Point instance whose ``x`` attribute access raises ``exc``."""

    class Evil(_Point):
        def __getattribute__(self, name: str) -> object:
            if name == "x":
                raise exc
            return super().__getattribute__(name)

    return Evil.__new__(Evil)  # bypass __init__, which would set x


def test_ordinary_exception_in_getattr_is_a_missing_attribute() -> None:
    validator = Validator(_Point)
    evil = _point_with_attr_raising(ValueError("boom"))
    assert validator.is_valid(evil) is False  # folded, not raised
    with pytest.raises(ValidationError) as info:
        validator.validate(evil)
    assert info.value.code == "missing_attribute"


@pytest.mark.parametrize("signal", FATAL)
def test_fatal_signal_in_getattr_propagates(signal: type[BaseException]) -> None:
    validator = Validator(_Point)
    evil = _point_with_attr_raising(signal())
    with pytest.raises(signal):
        validator.is_valid(evil)


# -- User predicates ----------------------------------------------------------


def _predicate_raising(exc: BaseException) -> Validator:
    def boom(_value: object) -> bool:
        raise exc

    return Validator(Annotated[int, at.Predicate(boom)])


def test_ordinary_exception_in_a_predicate_is_a_predicate_error() -> None:
    validator = _predicate_raising(ValueError("boom"))
    assert validator.is_valid(5) is False  # folded to a predicate_error, not raised
    with pytest.raises(ValidationError) as info:
        validator.validate(5)
    assert info.value.code == "predicate_error"


@pytest.mark.parametrize("signal", FATAL)
def test_fatal_signal_in_a_predicate_propagates(signal: type[BaseException]) -> None:
    validator = _predicate_raising(signal())
    with pytest.raises(signal):
        validator.is_valid(5)


def _key_raising_once(exc: BaseException) -> str:
    """Build a `str` key that raises `exc` from its first `__eq__`, as a signal does."""

    class Key(str):
        __slots__ = ()
        __hash__ = str.__hash__
        raised = False

        def __eq__(self, other: object) -> bool:
            if not Key.raised:
                Key.raised = True
                raise exc
            return str.__eq__(self, other)

    return Key("a")


@pytest.mark.parametrize("signal", FATAL)
@pytest.mark.parametrize(
    "schema", [{"a": int}, {"a": int, str: int}], ids=["closed", "clause"]
)
def test_fatal_signal_in_a_keys_eq_propagates(
    signal: type[BaseException], schema: dict[object, object]
) -> None:
    """A dict asks its key's `__eq__` when a record looks a field up by name."""
    with pytest.raises(signal):
        Validator(schema).is_valid({_key_raising_once(signal()): 1})


@pytest.mark.parametrize("signal", FATAL)
def test_fatal_signal_propagates_through_the_json_entries(
    signal: type[BaseException],
) -> None:
    """The JSON entries parse and then walk, so a fatal signal reaches them too.

    `validate_json` and `load` run the object walk over the parsed document, and
    `is_valid_json` runs the in-place one. All three promise a fatal interpreter
    signal propagates rather than being read as a non-member, and the promise
    was held on the object path alone.
    """
    validator = Validator(_class_raising(signal()))
    with pytest.raises(signal):
        validator.is_valid_json("1")
    with pytest.raises(signal):
        validator.validate_json("1")
    with pytest.raises(signal):
        validator.load("1")


def test_an_ordinary_exception_from_reprlib_leaves_the_plain_repr() -> None:
    """A container `reprlib` cannot render is summarized by its own `repr`.

    `reprlib` reads a class by its name and module, so a `list` subclass named
    `list` in `builtins` is read as the builtin, through its own `__len__`. One
    that raises reaches the summary as an ordinary exception, which is no
    signal: the value's `repr` still names it, where reading the raise as a
    signal would print it as `<unrepresentable>`.
    """

    def no_length(_: object) -> int:
        raise ValueError

    impostor = type("list", (list,), {"__module__": "builtins", "__len__": no_length})
    with pytest.raises(ValidationError) as caught:
        Validator(int).validate(impostor([1, 2]))
    assert caught.value.errors[0]["value"] == "[1, 2]"


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_a_signal_an_elements_repr_raises_once_propagates_from_the_summary(
    signal: type[BaseException],
) -> None:
    """A container's summary is `reprlib`'s, and it lets a signal through.

    Once through, the plain `repr` behind it ran the element a second time and
    read the answer as the value's summary, so a signal delivered once -- as a
    real interrupt is -- vanished into a `ValidationError`. `reprlib` catches
    `MemoryError` and `RecursionError` itself, which is the standard library's
    reading of them and not this one's.
    """
    shots = [signal]

    class Once:
        def __repr__(self) -> str:
            if shots:
                raise shots.pop()
            return "Once()"

    with pytest.raises(signal):
        Validator(int).validate([Once()])


def _forget_annotated() -> None:
    for cleanup in getattr(typing, "_cleanups", ()):
        cleanup()


@pytest.fixture
def forgetting_annotated() -> Iterator[None]:
    """Forget `typing`'s memo of `Annotated` before the test and after it.

    `typing` memoises `Annotated[...]` by equality, and a hostile operand here
    equals the ordinary one another test spells: `Ge(B(0))` is `Ge(0)`. Left in
    the memo, it is handed to that test, and before this one it hands back the
    previous signal's operand.
    """
    _forget_annotated()
    yield
    _forget_annotated()


class _LoudBound(int):
    """An integer bound whose order comparisons raise the signal it was given."""

    signal: type[BaseException] = KeyboardInterrupt

    def _compared(self, _other: object) -> bool:
        raise self.signal

    __lt__ = __gt__ = __le__ = __ge__ = _compared
    __hash__ = int.__hash__


@pytest.mark.usefixtures("forgetting_annotated")
@pytest.mark.parametrize("signal", FATAL)
def test_a_fatal_signal_in_a_relation_query_propagates(
    signal: type[BaseException],
) -> None:
    """A relation runs user code to decide, and an interrupted run is no verdict.

    Deciding `Literal[5]` against a refinement asks whether 5 belongs, which
    runs the predicate; deciding a bound conjunction orders the two bounds. An
    interrupted probe read as a decline -- `undecided`, or `False` from
    `is_empty` -- or, through the literal, as the refutation `not_subset`.
    """

    def interrupted(value: object) -> bool:
        raise signal

    probe = Validator(Literal[5])
    refined = Annotated[int, at.Predicate(interrupted)]
    for query in (probe.relation_to, probe.is_subtype_of, probe.is_equivalent):
        with pytest.raises(signal):
            query(refined)

    loud = type("Loud", (_LoudBound,), {"signal": signal})
    with pytest.raises(signal):
        Validator(Annotated[int, at.Gt(loud(5)), at.Lt(1)]).is_empty()


def _raising_zero(raised: Callable[..., object]) -> Callable[[int, object], bool]:
    """Build an `__eq__` raising for the build's own question, comparison with zero.

    Only that one: `typing` compares a new `Annotated` with a cached one of equal
    hash, and that comparison is not the build's.
    """

    def equal(self: int, other: object) -> bool:
        if other == 0:
            raised()
        return int.__eq__(self, other)

    return equal


# Each question building asks of user code, spelled with the hook that answers
# it by raising.
_BUILT: dict[str, Callable[[Callable[..., object]], object]] = {
    "a bound's float": lambda raised: Annotated[
        int, at.Ge(type("B", (int,), {"__float__": raised})(0))
    ],
    "a step's comparison with zero": lambda raised: Annotated[
        int,
        at.MultipleOf(
            type(
                "S", (int,), {"__eq__": _raising_zero(raised), "__hash__": int.__hash__}
            )(3)
        ),
    ],
    "the class a bound says it is": lambda raised: Annotated[
        int,
        at.Ge(type("Liar", (), {"__class__": property(raised)})()),  # ty: ignore[invalid-argument-type]
    ],
    "a marker class's module": lambda raised: Annotated[
        int,
        type("Meta", (type,), {"__module__": property(raised)})("Unknown", (), {})(),
    ],
    "the grouped-metadata flag": lambda raised: Annotated[
        int,
        type(
            "Grouped",
            (),
            {
                "__is_annotated_types_grouped_metadata__": type(
                    "Flag", (), {"__bool__": raised}
                )()
            },
        )(),
    ],
    "a class's protocol flag": lambda raised: type(
        "P", (), {"_is_protocol": type("Flag", (), {"__bool__": raised})()}
    ),
    "an element's unpacked flag": lambda raised: tuple[
        type("U", (), {"__unpacked__": type("Flag", (), {"__bool__": raised})()})()  # ty: ignore[invalid-type-form]
    ],
}


@pytest.mark.usefixtures("forgetting_annotated")
@pytest.mark.parametrize("question", list(_BUILT))
@pytest.mark.parametrize("signal", FATAL)
def test_a_fatal_signal_while_a_validator_is_built_propagates(
    signal: type[BaseException], question: str
) -> None:
    """Building asks user code about each marker, and an interrupted answer is none.

    Each question reads an ordinary exception as a documented fallback -- a
    bound that is not a float, a marker from somewhere else, a flag that is not
    set -- and read an interrupted one the same way, or, for the class a bound
    says it is, as a different refusal.
    """

    def raised(*_: object) -> object:
        raise signal

    annotation = _BUILT[question](raised)
    with pytest.raises(signal):
        Validator(annotation)


class _Loud(int):
    """An integer constant whose `==`, `hash` and `repr` raise once armed.

    Armed after the build, so the build and `typing`'s memo read it as the
    integer it is, and only the question under test meets the signal.
    """

    raised: BaseException | None = None

    def _raise(self) -> None:
        if self.raised is not None:
            raise self.raised

    def __eq__(self, other: object) -> bool:
        self._raise()
        return int.__eq__(self, other)

    def __hash__(self) -> int:
        self._raise()
        return int.__hash__(self)

    def __repr__(self) -> str:
        self._raise()
        return int.__repr__(self)


_ASKED: dict[str, Callable[[Validator, Validator], object]] = {
    "==": lambda left, right: left == right,
    "hash": lambda left, _: hash(left),
    "repr": lambda left, _: repr(left),
}


@pytest.mark.usefixtures("forgetting_annotated")
@pytest.mark.parametrize("question", list(_ASKED))
@pytest.mark.parametrize("signal", FATAL)
def test_a_fatal_signal_from_a_constant_propagates_from_eq_hash_and_repr(
    signal: type[BaseException], question: str
) -> None:
    """Comparing, hashing and printing a validator read each constant it pools."""
    left, right = _Loud(0), _Loud(0)
    built = Validator(Annotated[int, at.Ge(left)])
    _forget_annotated()
    other = Validator(Annotated[int, at.Ge(right)])
    left.raised = right.raised = signal()
    with pytest.raises(signal):
        _ASKED[question](built, other)


@pytest.mark.usefixtures("forgetting_annotated")
def test_an_ordinary_exception_from_a_constant_folds_in_eq_hash_and_repr() -> None:
    """A constant that cannot answer is unequal, hashless, and unrepresentable.

    Each check is one the answering constants would fail: two zeros are equal,
    and a zero and a one hash apart.
    """
    constants = _Loud(0), _Loud(0), _Loud(1)
    validators = []
    for constant in constants:
        _forget_annotated()
        validators.append(Validator(Annotated[int, at.Ge(constant)]))
    built, same, apart = validators
    for constant in constants:
        constant.raised = ValueError("boom")
    assert built != same
    assert hash(built) == hash(apart)
    assert repr(built) == "Annotated[int, Ge(<unrepresentable>)]"


class _Index:
    """A length bound whose `__index__` raises what it was given."""

    def __init__(self, raised: BaseException) -> None:
        self.raised = raised

    def __index__(self) -> int:
        raise self.raised

    def __repr__(self) -> str:
        return "_Index()"


class _Repr:
    """An object whose `__repr__` raises what it was given."""

    def __init__(self, raised: BaseException) -> None:
        self.raised = raised

    def __repr__(self) -> str:
        raise self.raised


@pytest.mark.usefixtures("forgetting_annotated")
@pytest.mark.parametrize("signal", FATAL)
@pytest.mark.parametrize("carrier", [_Index, _Repr])
def test_a_fatal_signal_while_a_refusal_is_written_propagates(
    signal: type[BaseException], carrier: type[_Index | _Repr]
) -> None:
    """A refused bound is read for its length and its repr, and either may raise.

    An ordinary exception is the refusal the bound earns, naming it as it can.
    """
    with pytest.raises(signal):
        Validator(Annotated[list[int], at.MinLen(carrier(signal()))])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="must be a length a value can have"):
        Validator(Annotated[list[int], at.MinLen(carrier(ValueError("boom")))])  # ty: ignore[invalid-argument-type]
