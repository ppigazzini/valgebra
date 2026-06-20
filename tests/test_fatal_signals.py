"""Fatal interpreter signals propagate; ordinary exceptions fold to non-member.

A membership probe (``isinstance``, a rich comparison, ``__eq__``, ``__mod__``,
``__len__``) that raises an ordinary exception means the value cannot answer "are
you in this set?", so it is a non-member. A *fatal* signal -- a base exception
that is not an ordinary exception, such as KeyboardInterrupt or SystemExit -- is
not a membership answer at all: it propagates instead of being silently read as a
non-member, so an interrupted check stops rather than continuing.
"""

from __future__ import annotations

import pytest

from valgebra import Validator


def _class_raising(exc: BaseException) -> type:
    """Build a class whose ``isinstance`` check raises ``exc`` (via metaclass)."""

    class Meta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            raise exc

    class Probed(metaclass=Meta):
        pass

    return Probed


def test_ordinary_exception_in_a_probe_is_a_non_member() -> None:
    validator = Validator(_class_raising(ValueError("boom")))
    assert validator.is_valid(object()) is False


def test_keyboard_interrupt_propagates_through_is_valid() -> None:
    validator = Validator(_class_raising(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        validator.is_valid(object())


def test_system_exit_propagates_through_is_valid() -> None:
    validator = Validator(_class_raising(SystemExit()))
    with pytest.raises(SystemExit):
        validator.is_valid(object())


def test_fatal_signal_propagates_through_validate() -> None:
    validator = Validator(_class_raising(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        validator.validate(object())


def test_fatal_signal_propagates_through_membership_operator() -> None:
    validator = Validator(_class_raising(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _ = object() in validator
