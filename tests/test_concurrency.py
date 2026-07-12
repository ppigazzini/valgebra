"""Validators are safe to share across threads.

A compiled validator is immutable (frozen, with an interned pool it never
mutates) and the validation walk keeps its recursion guard in a per-call local,
so the same validator can be validated from many threads at once. The lazy
per-validator precompute is the only shared mutable state, and it is a
thread-safe one-time init holding pure-Rust data, so first use races safely.

The extension module declares itself free-threading-ready, so a free-threaded
interpreter keeps the global interpreter lock disabled and this test exercises
true parallel access; under a regular interpreter it still checks correctness
under concurrency.
"""

from __future__ import annotations

import sys
import threading

import pytest

from valgebra import ValidationError, Validator, recursive


def _gil_enabled() -> bool:
    """Whether the interpreter holds a GIL, so threads do not run in parallel."""
    query = getattr(sys, "_is_gil_enabled", None)
    return query() if query is not None else True


_RECORD = Validator({"name": str, "age?": int, "tags": list[str]})
_JSON = Validator(list[dict[str, int]])
_TREE = recursive(lambda t: {"value": int, "left?": t, "right?": t})

_GOOD_RECORD = {"name": "Ada", "age": 36, "tags": ["a", "b"]}
_BAD_RECORD = {"name": 5, "tags": "x"}

THREADS = 8
ITERATIONS = 1000


def _fresh_validators() -> tuple[Validator, Validator, Validator]:
    """Build a fresh trio whose lazy precompute has not been built yet."""
    record = Validator({"name": str, "age?": int, "tags": list[str]})
    json_v = Validator(list[dict[str, int]])
    tree = recursive(lambda t: {"value": int, "left?": t, "right?": t})
    return record, json_v, tree


def _hammer(
    failures: list[str], record: Validator, json_v: Validator, tree: Validator
) -> None:
    try:
        for _ in range(ITERATIONS):
            assert record.is_valid(_GOOD_RECORD) is True
            assert record.is_valid(_BAD_RECORD) is False
            assert json_v.is_valid_json('[{"a": 1}, {"b": 2}]') is True
            assert tree.is_valid({"value": 1, "left": {"value": 2}}) is True
            try:
                record.validate(_BAD_RECORD)
            except ValidationError:
                pass
            else:
                failures.append("expected validate to raise")
    except Exception as exc:  # noqa: BLE001  (surface any thread failure)
        failures.append(repr(exc))


def _run_hammer_threads(
    record: Validator, json_v: Validator, tree: Validator
) -> list[str]:
    failures: list[str] = []
    threads = [
        threading.Thread(target=_hammer, args=(failures, record, json_v, tree))
        for _ in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def test_validators_are_thread_safe() -> None:
    # Always runs. Under the GIL this checks correctness under concurrency (threads
    # interleave but do not truly overlap); it is not, on its own, a data-race
    # detector — that is the free-threaded test below.
    assert not _run_hammer_threads(_RECORD, _JSON, _TREE)


@pytest.mark.skipif(
    _gil_enabled(),
    reason="under the GIL threads do not run in parallel, so first use cannot race; "
    "the free-threaded interpreter is where a real data race would surface",
)
def test_validators_run_truly_parallel_without_the_gil() -> None:
    # Honest about what is exercised: this body runs only when the GIL is disabled.
    # The validators are built fresh here, so their shared lazy precompute is hit
    # for the first time by genuinely parallel threads rather than already warmed
    # by an earlier test.
    assert not _gil_enabled()
    assert not _run_hammer_threads(*_fresh_validators())
