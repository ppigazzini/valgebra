"""A value that changes while it is checked is reported, not crashed on.

Membership runs Python at almost every entry of a container -- a predicate, an
``__eq__``, an ``isinstance`` hook -- and a free-threaded interpreter lets another
thread write to that container meanwhile. The walk therefore reads containers in
a way that survives the change: it reports ``mutated_during_validation`` and a
non-member, because nothing about the contents was decided, and never a
``BaseException`` the caller cannot catch as a validation failure.

The cross-thread case runs only where threads are genuinely parallel; under the
GIL a predicate is the shape that reaches the same read, which is what the rest
of the module drives.
"""

from __future__ import annotations

import sys
import threading
from typing import Annotated

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator

MUTATED = "mutated_during_validation"


def _gil_enabled() -> bool:
    getter = getattr(sys, "_is_gil_enabled", None)
    return True if getter is None else bool(getter())


def _record_growing(target: dict[str, int]) -> Validator:
    """Build a record whose first field grows `target` while its predicate runs."""

    def grow(_: object) -> bool:
        target.setdefault("c", 3)
        return True

    return Validator({"a": Annotated[int, at.Predicate(grow)], "b": int, "c?": int})


def _set_growing(target: set[int]) -> Validator:
    """Build a set schema whose element predicate grows `target` while it runs."""

    def grow(_: object) -> bool:
        target.add(99)
        return True

    return Validator(set[Annotated[int, at.Predicate(grow)]])


def test_a_dict_grown_by_a_predicate_is_reported_not_a_panic() -> None:
    checked = {"a": 1, "b": 2}
    assert _record_growing(checked).is_valid(checked) is False

    explained = {"a": 1, "b": 2}
    with pytest.raises(ValidationError) as info:
        _record_growing(explained).validate(explained)
    assert info.value.code == MUTATED


def test_a_set_grown_by_a_predicate_is_reported_not_a_panic() -> None:
    checked = {1, 2, 3}
    assert _set_growing(checked).is_valid(checked) is False

    explained = {1, 2, 3}
    with pytest.raises(ValidationError) as info:
        _set_growing(explained).validate(explained)
    assert info.value.code == MUTATED


def test_replacing_a_value_leaves_the_reading_intact() -> None:
    # Only a change in *size* costs the reading. A predicate that rewrites a value
    # in place leaves the entries where they are, so the walk still answers about
    # the dict rather than reporting that it moved.
    stable = {"a": 1, "b": 2}

    def rewrite(_: object) -> bool:
        stable["b"] = 5
        return True

    schema = Validator({"a": Annotated[int, at.Predicate(rewrite)], "b": int})
    assert schema.is_valid(stable) is True


def test_an_unmutated_container_is_unaffected() -> None:
    assert Validator({"a": int}).is_valid({"a": 1})
    assert not Validator({"a": int}).is_valid({"a": "x"})
    assert Validator(set[int]).is_valid({1, 2})
    assert not Validator(set[int]).is_valid({1, "x"})
    assert Validator(frozenset[int]).is_valid(frozenset({1}))
    assert not Validator(frozenset[int]).is_valid({1})


def _hammer(validator: Validator, shared: object, mutate: object) -> list[str]:
    """Read `shared` from four threads while two others resize it."""
    escaped: list[str] = []
    stop = threading.Event()

    def check() -> None:
        try:
            while not stop.is_set():
                validator.is_valid(shared)
        except BaseException as error:  # noqa: BLE001 - the point is that none escapes
            escaped.append(f"{type(error).__module__}.{type(error).__name__}")

    def churn() -> None:
        while not stop.is_set():
            mutate(shared)

    threads = [threading.Thread(target=check) for _ in range(4)]
    threads += [threading.Thread(target=churn) for _ in range(2)]
    for thread in threads:
        thread.start()
    stop.wait(1.0)
    stop.set()
    for thread in threads:
        thread.join()
    return escaped


def _resize_dict(container: dict[str, int]) -> None:
    container["spare"] = 1
    container.pop("spare", None)


def _resize_set(container: set[int]) -> None:
    container.add(10**6)
    container.discard(10**6)


@pytest.mark.skipif(
    _gil_enabled(),
    reason="under the GIL another thread cannot write to the container mid-walk; "
    "the predicate cases above are the reachable shape there",
)
@pytest.mark.parametrize(
    ("schema", "shared", "mutate"),
    [
        pytest.param(
            dict[str, int],
            {f"k{index}": index for index in range(200)},
            _resize_dict,
            id="dict",
        ),
        pytest.param(set[int], set(range(200)), _resize_set, id="set"),
    ],
)
def test_a_container_written_by_another_thread_never_escapes_as_an_exception(
    schema: object,
    shared: object,
    mutate: object,
) -> None:
    assert _hammer(Validator(schema), shared, mutate) == []
