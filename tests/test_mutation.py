"""A value that changes while it is checked is reported, not crashed on.

Membership runs Python at almost every entry of a container -- a predicate, an
``__eq__``, an ``isinstance`` hook -- and a free-threaded interpreter lets another
thread write to that container meanwhile. A dict, a set and a **list** are all
read against a count taken once, so all three are held here; a tuple cannot be
resized and needs no guard. The walk therefore reads containers in
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
from typing import TYPE_CHECKING, Annotated, Any

import annotated_types as at
import pytest

from valgebra import ValidationError, Validator

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _list_growing(target: list[int]) -> Validator:
    """Build a list schema whose element predicate grows `target` while it runs."""

    def grow(_: object) -> bool:
        target.append(99)
        return True

    return Validator(list[Annotated[int, at.Predicate(grow)]])


def _list_shrinking(target: list[int]) -> Validator:
    """Build a list schema whose element predicate shrinks `target` while it runs."""

    def shrink(_: object) -> bool:
        if len(target) > 1:
            target.pop()
        return True

    return Validator(list[Annotated[int, at.Predicate(shrink)]])


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


def test_a_key_added_while_a_record_is_explained_is_still_reported() -> None:
    """The report reads the value it has, not the one it counted.

    A closed record holding exactly the keys it declares has no undeclared key
    to look for, and the report reaches that by counting what the field walk
    found against the entries the value held. Checking a field runs Python, and
    Python can add a key -- so the count is a claim about the value as it was,
    and the length is read again before it is believed. Here the first field
    fails and the second grows the value, which leaves as many declared fields
    found as there were entries to begin with: the one arrangement where a stale
    count would say there is nothing to look for.
    """
    payload = {"a": 1, "b": 2}

    def grow(_: object) -> bool:
        payload[f"x{len(payload)}"] = 0
        return True

    record = Validator({"a": str, "b": Annotated[int, at.Predicate(grow)]})
    with pytest.raises(ValidationError) as info:
        record.validate(payload)
    codes = [item["code"] for item in info.value.errors]
    assert "string_type" in codes
    assert "extra_forbidden" in codes


def test_a_list_grown_by_a_predicate_is_reported_not_answered() -> None:
    """The positions past the length read at entry are never visited.

    A sequence is walked by position against a length read once, so a list that
    grows hides its new items -- and the walk that never saw them answered
    `True` for a value that is not a member. The pair below is the whole of it:
    the value ends as `[1, 99]`, `list[int]` admits that, and the schema whose
    element must also pass the predicate does not admit a reading at all.
    """
    checked = [1]
    assert _list_growing(checked).is_valid(checked) is False
    assert checked == [1, 99], "the predicate really did move it"

    explained = [1]
    with pytest.raises(ValidationError) as info:
        _list_growing(explained).validate(explained)
    assert info.value.code == MUTATED


def test_a_list_shrunk_by_a_predicate_is_reported_not_answered() -> None:
    """The same fact at the other end: the walk expected items that are gone."""
    checked = [1, 2, 3]
    assert _list_shrinking(checked).is_valid(checked) is False

    explained = [1, 2, 3]
    with pytest.raises(ValidationError) as info:
        _list_shrinking(explained).validate(explained)
    assert info.value.code == MUTATED


def test_a_list_that_was_never_a_member_is_not_made_one_by_shrinking() -> None:
    """The direction that would be an accept the value never supported.

    `[1, "x"]` is not a `list[int]` at entry. A predicate that drops the bad
    element while the walk is on the first one leaves a list that *is* one, and
    a walk that answered from the positions it managed to read would say `True`
    about neither state.
    """
    target: list[object] = [1, "x"]

    def drop(_: object) -> bool:
        if len(target) > 1:
            target.pop()
        return True

    schema = Validator(list[Annotated[int, at.Predicate(drop)]])
    assert schema.is_valid(target) is False


def test_a_tuple_needs_no_guard() -> None:
    """A tuple cannot be resized, so its walk keeps the plain iterator."""
    seen: list[object] = []

    def note(value: object) -> bool:
        seen.append(value)
        return True

    schema = Validator(tuple[Annotated[int, at.Predicate(note)], ...])
    assert schema.is_valid((1, 2, 3)) is True
    assert seen == [1, 2, 3], "every position, in order, exactly once"


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
    assert Validator(list[int]).is_valid([1, 2])
    assert not Validator(list[int]).is_valid([1, "x"])
    assert Validator([int, str]).is_valid([1, "x"])
    assert Validator({"a": int}).is_valid({"a": 1})
    assert not Validator({"a": int}).is_valid({"a": "x"})
    assert Validator(set[int]).is_valid({1, 2})
    assert not Validator(set[int]).is_valid({1, "x"})
    assert Validator(frozenset[int]).is_valid(frozenset({1}))
    assert not Validator(frozenset[int]).is_valid({1})


def _hammer(
    validator: Validator,
    shared: Any,
    mutate: Callable[[Any], None],
) -> list[str]:
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


def _resize_list(container: list[int]) -> None:
    container.append(1)
    container.pop()


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
        pytest.param(list[int], list(range(200)), _resize_list, id="list"),
    ],
)
def test_a_container_written_by_another_thread_never_escapes_as_an_exception(
    schema: object,
    shared: Any,
    mutate: Callable[[Any], None],
) -> None:
    assert _hammer(Validator(schema), shared, mutate) == []
