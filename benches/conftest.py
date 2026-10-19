"""Every figure this directory prints is of the shipped binary, or none is.

A wall-clock benchmark says what it timed or it says nothing: `maturin develop`
without `--release` installs a debug extension that looks exactly like the
release one from Python, and a run against it reads an order of magnitude slow
-- which is indistinguishable from a regression, and a comparison with one side
built that way reads a change worth five percent as a win of fifty.

`scripts/compare_gate.py` refuses a non-release extension, and the same rule is
asked of this directory, from the same statement of the rule rather than a
second copy.

The check is by *size*: a release extension is a couple of megabytes and one
carrying its debug symbols is an order of magnitude larger, which separates
them without asking the compiler what it did.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def _compare_gate() -> ModuleType:
    """Load the gate's provenance check without importing it as a package.

    A script finds `_lib` beside it on the import path its own run gives it,
    and pytest's run is not that one, so the directory goes there first.
    """
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
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
