"""PyO3's global reference pool is compiled out of every build.

PyO3 keeps a pool of reference-count decrements deferred while a thread is
detached, and every call into the extension asks it for them. Once the pool
exists -- and the first of PyO3's lazy initialisers to detach and attach again
creates it -- the question locks a mutex, which costs `Validator(int).is_valid`
forty instructions a call. The binding never detaches, so the pool only ever
holds nothing, and `.cargo/config.toml` compiles it out.

Nothing else holds the flags there. The instruction gate's boundary shape calls
the walk without crossing PyO3's call machinery, so it cannot see the pool, and
the comparison gate reads the cost only as nanoseconds a laptop's noise covers.
A cleanup that dropped the file would put the mutex back on every call with
every gate green.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from _toml import load

# A repository check: it reads the build configuration, which no wheel ships.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".cargo" / "config.toml"


def _cfgs() -> set[str]:
    """Read the `--cfg` names `[build] rustflags` passes to every crate."""
    flags = load(CONFIG)["build"]["rustflags"]
    return {value for flag, value in pairwise(flags) if flag == "--cfg"}


def test_the_reference_pool_is_compiled_out() -> None:
    assert "pyo3_disable_reference_pool" in _cfgs(), (
        "`.cargo/config.toml` no longer compiles out PyO3's reference pool, so "
        "every call into the extension locks its mutex again"
    )


def test_a_drop_while_detached_leaks_rather_than_aborts() -> None:
    # With the pool gone, PyO3 aborts the process on a `Py<T>` dropped while
    # detached unless this flag is set. The binding never detaches, so neither
    # happens; the flag decides what an unforeseen one costs a caller.
    assert "pyo3_leak_on_drop_without_reference_pool" in _cfgs(), (
        "without the leak flag, a `Py<T>` dropped while detached aborts the "
        "interpreter instead of leaking one object"
    )
