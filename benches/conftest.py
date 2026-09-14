"""Every figure this directory prints is of the shipped binary, or none is.

A wall-clock benchmark says what it timed or it says nothing: `maturin develop`
without `--release` installs a debug extension that looks exactly like the
release one from Python, and a run against it reads an order of magnitude slow
-- which is indistinguishable from a regression, and was read as a *win* of 56%
here on 2026-09-14 when the change was worth five.

`scripts/compare_gate.py` has refused a non-release extension since two of its
own readings were of a binary nobody meant to measure. This directory had no
such refusal, so the same rule is asked of it, from the same statement of the
rule rather than a second copy.

The check is by *size*: a release extension is a couple of megabytes and one
carrying its debug symbols is an order of magnitude larger, which separates
them without asking the compiler what it did.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def _compare_gate() -> ModuleType:
    """Load the gate's provenance check without importing it as a package."""
    path = ROOT / "scripts" / "compare_gate.py"
    spec = importlib.util.spec_from_file_location("compare_gate", path)
    if spec is None or spec.loader is None:  # pragma: no cover - a missing script
        pytest.exit(f"benches: {path} is not importable", returncode=2)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session", autouse=True)
def _the_extension_is_the_shipped_one() -> None:
    """Refuse the whole run when the loaded extension is not a release build."""
    wrong = _compare_gate().provenance()
    if wrong is not None:
        pytest.exit(f"benches: {wrong}", returncode=2)
