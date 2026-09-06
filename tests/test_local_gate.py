"""The local gate runs the merge gate's steps, or names the ones it cannot.

`AGENTS.md` lists commands a developer runs in a full clone with a warm virtual
environment; CI is forty-odd jobs in shallow clones with pinned tools on three
operating systems. The first difference that mattered was found by a push: a
ledger reading `git describe` passed locally for a week and reddened eight jobs
at once, because a local clone has tags and a checkout does not.

`scripts/gate.py` closes that by running the workflow's own `run:` steps in a
clone shaped like the runner's. This holds the two halves of that claim:

* every `run:` step of every merge-gate job is either executed by the gate or
  named in its `NEEDS_A_RUNNER` list with the reason, so a step added to CI is
  one the local gate runs or refuses **by name**;
* nothing in that list has outlived its step, so the excuses cannot accumulate.

And the property the whole thing exists for: the tree it runs in is a shallow
clone with no tags.

LEDGER: every merge-gate step is run by the local gate or excused by name
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "gate.py"


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("valgebra_gate", GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _merge_gate_steps() -> list[tuple[str, str]]:
    spec = gate.workflow()
    return [
        (job, name)
        for job in gate.required_jobs(spec)
        for name, _ in gate.steps(spec, job)
    ]


def test_every_merge_gate_step_is_run_or_excused_by_name() -> None:
    steps = _merge_gate_steps()
    # The scan is the detector: no steps at all would pass having read nothing.
    assert len(steps) >= 20, f"the workflow scan found only {steps}"

    unaccounted = sorted(
        f"{job}: {name}"
        for job, name in steps
        if not gate.runnable(name) and name not in gate.NEEDS_A_RUNNER
    )
    assert not unaccounted, unaccounted
    # Every step is one or the other, by construction of `runnable`; what this
    # asserts is that both columns are non-empty, since a gate that ran nothing
    # and a list that excused nothing would each pass the line above.
    assert any(gate.runnable(name) for _, name in steps), "the gate runs nothing"
    assert any(not gate.runnable(name) for _, name in steps), (
        "no step needs a runner, which means the list stopped being read"
    )


def test_no_excuse_has_outlived_its_step() -> None:
    named = {name for _, name in _merge_gate_steps()}
    stale = sorted(set(gate.NEEDS_A_RUNNER) - named)
    assert not stale, (
        f"steps excused from the local gate that the workflow no longer has: "
        f"{stale}. Delete each with the reason beside it."
    )


def test_every_excuse_carries_a_reason() -> None:
    empty = sorted(name for name, why in gate.NEEDS_A_RUNNER.items() if not why.strip())
    assert not empty, f"steps excused with no reason: {empty}"


def test_the_python_suite_is_one_of_the_steps_it_runs() -> None:
    # The step that would have caught the failure this gate exists for: the
    # whole Python suite, run in a clone with no tags.
    runnable = {name for _, name in _merge_gate_steps() if gate.runnable(name)}
    assert "pytest" in runnable


def test_the_gate_runs_in_a_shallow_clone_with_no_tags(tmp_path: Path) -> None:
    """The property the gate exists for, checked on a real clone."""
    tree = gate.shallow_clone(tmp_path)
    depth = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(tree), "rev-list", "--count", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    assert depth.stdout.strip() == "1", "the clone is not shallow"
    tags = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(tree), "tag", "--list"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    assert tags.stdout.strip() == "", "the clone carries tags a checkout would not"


def test_the_three_exit_codes_are_distinct() -> None:
    assert (gate.EXIT_OK, gate.EXIT_FAIL, gate.EXIT_CANNOT_RUN) == (0, 1, 2)
