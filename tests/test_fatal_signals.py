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
from typing import Annotated

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator

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
