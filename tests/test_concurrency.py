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
import sysconfig
import threading
from typing import Annotated

import pytest

from valgebra import ValidationError, Validator, recursive


def _gil_enabled() -> bool:
    """Whether the interpreter holds a GIL, so threads do not run in parallel."""
    query = getattr(sys, "_is_gil_enabled", None)
    return query() if query is not None else True


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="a build with a GIL has none to keep off",
)
def test_a_free_threaded_build_keeps_the_gil_off_after_import() -> None:
    """Importing the extension leaves a free-threaded interpreter without a GIL.

    A module that does not declare itself free-threading-ready makes the
    interpreter turn the GIL back on at import, with a `RuntimeWarning` and no
    error, and the parallel tests below then skip rather than fail: the lane
    that exists to run them reads green having run none. This is the row that
    turns that red. A GIL the caller asked for (`PYTHON_GIL=1`, `-X gil=1`) is
    the caller's choice rather than the module's, and is left alone.
    """
    if getattr(sys.flags, "gil", None) == 1:
        pytest.skip("the GIL was asked for when the interpreter started")
    assert not _gil_enabled()


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


def _compile_markers(failures: list[str], seed: int) -> None:
    """Compile refinements whose marker *type* is new every time.

    A fresh type per compile is what makes this the concurrency test for the
    frontend's mask cache: every compile misses it and writes to it, so the
    threads contend on the one shared mutable Python object the compile path
    has. The answers are checked because a mask read from the wrong entry would
    drop the bound rather than raise, and a dropped bound is a validator that
    admits what it was written to refuse.
    """
    try:
        for n in range(MARKER_TYPES):
            marker = type(
                f"Ge{seed}_{n}",
                (),
                {"__slots__": ("ge",), "__init__": lambda s, v: setattr(s, "ge", v)},
            )(n)
            bounded = Validator(Annotated[int, marker])
            assert bounded.is_valid(n) is True
            assert bounded.is_valid(n - 1) is False
    except BaseException as err:  # noqa: BLE001 - any failure is the report
        failures.append(f"thread {seed}: {type(err).__name__}: {err}")


#: Fresh marker types per thread. Enough to run past the cache's own bound
#: (256 types) across eight threads, so the branch that stops remembering is
#: contended too and not only the branch that writes.
MARKER_TYPES = 100


def _run_marker_threads() -> list[str]:
    failures: list[str] = []
    threads = [
        threading.Thread(target=_compile_markers, args=(failures, seed))
        for seed in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def test_markers_compile_from_many_threads() -> None:
    """The frontend's mask cache is shared and mutable, and this contends it.

    Which attribute names a refinement marker can carry is read from its type
    and remembered in one `dict` for the process. That is the only shared
    *mutable* Python object on the compile path, and its safety argument is
    that a dict's reads and writes are atomic under the free-threaded
    interpreter's per-object lock, with two threads that miss on one type
    computing the same mask because the mask is the type's and not the thread's.

    The argument was written and not run. Every other row here validates with
    schemas built before the threads start; none compiles, so none reached the
    cache at all.
    """
    assert not _run_marker_threads()


@pytest.mark.skipif(_gil_enabled(), reason="threads do not run in parallel under a GIL")
def test_markers_compile_in_parallel_without_the_gil() -> None:
    """The same, where the contention is real rather than interleaved."""
    assert not _gil_enabled()
    assert not _run_marker_threads()
