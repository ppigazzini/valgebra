"""Validators are safe to share across threads.

A compiled validator is immutable (frozen, with an interned pool it never
mutates) and the validation walk keeps its recursion guard in a per-call local,
so the same validator can be validated from many threads at once. Under a
free-threaded interpreter this runs with no GIL, so the test exercises true
parallel access; under a regular interpreter it still checks correctness under
concurrency.
"""

from __future__ import annotations

import threading

from valgebra import ValidationError, recursive, validator

_RECORD = validator({"name": str, "age?": int, "tags": list[str]})
_JSON = validator(list[dict[str, int]])
_TREE = recursive(lambda t: {"value": int, "left?": t, "right?": t})

_GOOD_RECORD = {"name": "Ada", "age": 36, "tags": ["a", "b"]}
_BAD_RECORD = {"name": 5, "tags": "x"}

THREADS = 8
ITERATIONS = 1000


def _hammer(failures: list[str]) -> None:
    try:
        for _ in range(ITERATIONS):
            assert _RECORD.is_valid(_GOOD_RECORD) is True
            assert _RECORD.is_valid(_BAD_RECORD) is False
            assert _JSON.is_valid_json('[{"a": 1}, {"b": 2}]') is True
            assert _TREE.is_valid({"value": 1, "left": {"value": 2}}) is True
            try:
                _RECORD.validate(_BAD_RECORD)
            except ValidationError:
                pass
            else:
                failures.append("expected validate to raise")
    except Exception as exc:  # noqa: BLE001  (surface any thread failure)
        failures.append(repr(exc))


def test_validators_are_thread_safe() -> None:
    failures: list[str] = []
    threads = [
        threading.Thread(target=_hammer, args=(failures,)) for _ in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures
