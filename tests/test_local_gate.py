"""The local gate runs the merge gate's steps, or names the ones it cannot.

`AGENTS.md` lists commands a developer runs in a full clone with a warm virtual
environment; CI is forty-odd jobs in shallow clones with pinned tools on three
operating systems. The first difference that mattered was found by a push: a
ledger reading `git describe` passed locally for a week and reddened eight jobs
at once, because a local clone has tags and a checkout does not.

`scripts/gate.py` closes that by running the workflow's own `run:` steps in a
clone shaped like the runner's. This holds the two halves of that claim:

* every `run:` step of every merge-gate job is either **in the plan the gate
  builds** or named in its `NEEDS_A_RUNNER` list with the reason, so a step
  added to CI is one the local gate runs or refuses by name;
* nothing in that list has outlived its step, so the excuses cannot accumulate.

The first of those is asked of the *plan*, not of `runnable`. Asked of
`runnable` it was `name in NEEDS_A_RUNNER and name not in NEEDS_A_RUNNER` -- a
contradiction, so the list it built was empty for every workflow and the
assertion could not fail. The defect it now catches is planted below.

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


def _planned(spec: dict) -> set[str]:
    """Collect the steps `gate.py` would actually run, as `job: name`."""
    plan, _ = gate.build_plan(spec, gate.required_jobs(spec))
    return {f"{job}: {name}" for job, name, _, _ in plan}


def test_every_merge_gate_step_is_planned_or_excused_by_name() -> None:
    spec = gate.workflow()
    steps = _merge_gate_steps()
    # The scan is the detector: no steps at all would pass having read nothing.
    assert len(steps) >= 20, f"the workflow scan found only {steps}"

    planned = _planned(spec)
    unaccounted = sorted(
        f"{job}: {name}"
        for job, name in steps
        if f"{job}: {name}" not in planned and name not in gate.NEEDS_A_RUNNER
    )
    assert not unaccounted, (
        f"merge-gate steps the local gate neither runs nor excuses: {unaccounted}. "
        "A step carrying an expression only a runner answers is skipped silently "
        "unless it is named in NEEDS_A_RUNNER with the reason."
    )
    # Both columns must be non-empty: a gate that planned nothing and a list
    # that excused nothing would each pass the assertion above.
    assert planned, "the gate runs nothing"
    assert any(name in gate.NEEDS_A_RUNNER for _, name in steps), (
        "no step needs a runner, which means the list stopped being read"
    )


def test_a_step_only_a_runner_can_fill_in_is_unaccounted() -> None:
    """The defect the assertion above exists to catch, planted on a spec.

    `${{ github.sha }}` is the runner's to answer and `gate.py` cannot. Before
    this, `resolved` searched for `env.` expressions alone, so such a step came
    back "resolved" with its braces intact, was planned, and would have been
    handed to bash verbatim -- while the ledger passed, because its filter was a
    contradiction. Planting it on a synthetic workflow keeps the case in the
    suite without a step in `ci.yml` that exists only to be caught.
    """
    planted = "A step only a runner can fill in"
    spec = {
        "env": {},
        "jobs": {
            "ci": {"needs": ["planted"]},
            "planted": {
                "steps": [{"name": planted, "run": 'echo "${{ github.sha }}"'}]
            },
        },
    }
    plan, unresolved = gate.build_plan(spec, gate.required_jobs(spec))
    assert plan == [], "a step the gate cannot fill in must not be planned"
    assert unresolved == [f"planted: {planted}"]
    assert planted not in gate.NEEDS_A_RUNNER
    assert f"planted: {planted}" not in _planned(spec), (
        "the filter must flag a step that is neither planned nor excused"
    )


def test_an_env_expression_the_workflow_defines_is_filled_in() -> None:
    """And the other direction, so the refusal is not a blanket one."""
    assert gate.resolved("echo ${{ env.X }}", {"X": "1"}) == "echo 1"
    assert gate.resolved("echo ${{ env.X }}", {}) is None
    assert gate.resolved("echo ${{ github.sha }}", {"X": "1"}) is None
    assert gate.resolved("echo ${{ env.X }} ${{ matrix.os }}", {"X": "1"}) is None
    assert gate.resolved("echo plain", {}) == "echo plain"


def test_no_excuse_has_outlived_its_step() -> None:
    named = {name for _, name in _merge_gate_steps()}
    stale = sorted(set(gate.NEEDS_A_RUNNER) - named)
    assert not stale, (
        f"steps excused from the local gate that the workflow no longer has: "
        f"{stale}. Delete each with the reason beside it."
    )


def test_every_stand_in_is_for_a_step_the_gate_excuses() -> None:
    """A stand-in for a step the gate already runs is a step run twice."""
    loose = sorted(set(gate.STANDINS) - set(gate.NEEDS_A_RUNNER))
    assert not loose, (
        f"stand-ins for steps the gate is not excused from: {loose}. A step the "
        "gate can run is run, not stood in for."
    )
    named = {name for _, name in _merge_gate_steps()}
    stale = sorted(set(gate.STANDINS) - named)
    assert not stale, f"stand-ins for steps the workflow no longer has: {stale}"


def test_the_interpreter_backed_binding_tests_are_in_the_plan() -> None:
    """The hole the stand-ins exist for, asked of the plan the gate builds.

    Seventy-odd Rust tests link an embedded interpreter, and the merge gate
    reaches them only inside a coverage rebuild the local gate cannot run. Both
    halves are needed: the command has to carry the feature, and it has to be in
    the plan rather than in a table nothing reads.
    """
    spec = gate.workflow()
    plan, _ = gate.build_plan(spec, gate.required_jobs(spec))
    running = [command for _, _, command, _ in plan if "interpreter-tests" in command]
    assert running, (
        "no step in the plan runs the binding's interpreter-backed tests; "
        "without one they are in no local step at all"
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
