"""Run the merge gate's own commands, in the clone the merge gate gets.

`AGENTS.md` lists a build-health gate: a handful of commands a developer runs in
a full clone with a warm virtual environment. CI is forty-odd jobs in *shallow*
clones with pinned tool versions on three operating systems, and the first
difference that mattered was found by a push -- a ledger that reads `git
describe` passed for a week locally and reddened eight jobs at once, because a
local clone has tags and a checkout does not.

So this runs the lane's steps rather than a list that resembles them. The
commands come out of ``.github/workflows/ci.yml``; the working tree is a fresh
shallow clone of `HEAD` with no tags, which is what `actions/checkout` produces;
and every step of every merge-gate job is either executed here or named in
`NEEDS_A_RUNNER` with the reason it cannot be. `tests/test_local_gate.py` holds
that list to the workflow in both directions, so a step added to CI is a step
this either runs or refuses by name.

What it is not is a CI replacement: the steps that need a PGO wheel, valgrind, a
mutation sweep or a second operating system are named and skipped. Those are the
lanes a push is for.

Usage:
    python scripts/gate.py                  # the runnable steps, in a shallow clone
    python scripts/gate.py --list           # what would run, and what would not
    python scripts/gate.py --job docs       # one job's steps
    python scripts/gate.py --here           # this working tree, no clone

Three outcomes, three exit codes: **0** every step passed, **1** a step failed,
**2** the gate could not run -- an unreadable workflow, a clone that failed. A
gate that could not run has proven nothing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: Steps a developer's machine cannot reproduce, and why. Each key is a step's
#: `name:` in the workflow; a step with no name is matched by its command's
#: first line. The reason is the point: a skip nobody can read is a hole.
NEEDS_A_RUNNER = {
    "Install valgrind": "installs a system package with sudo",
    "Build and install the optimized wheel": "a PGO build, minutes per run",
    "Name the merge base": "reads the event payload the runner provides",
    "Instruction-count regression gate": "cachegrind, and a second build of the base",
    "Decision-path instruction-count regression gate": "cachegrind, as above",
    "Binding-path instruction-count regression gate": "cachegrind, as above",
    "Recorded instruction-count budgets (nightly)": "cachegrind, and nightly only",
    "Smoke-run the benchmarks": "needs the optimized wheel above",
    "Competitive comparison gate": "needs the optimized wheel above",
    "List the Rust files the diff touches": "reads the event payload",
    "Sweep the core files the change touches": "a mutation sweep, tens of minutes",
    "Sweep the binding's swept files when the change touches one": (
        "a mutation sweep, tens of minutes"
    ),
    "Measure binding coverage via the Python suite and Rust unit tests": (
        "instrumented rebuild of the whole workspace"
    ),
    "Install the differential oracles": (
        "resolves the bench group; the suite below needs it"
    ),
    (
        "cargo llvm-cov nextest --locked --workspace --summary-only "
        "--ignore-filename-regex 'valgebra-py' --fail-under-lines 93"
    ): "an instrumented rebuild under nextest",
    "uv sync --locked --no-install-project --group bench": (
        "the bench group, for steps this list already skips"
    ),
    "Cross-check membership against pydantic-core and jsonschema": (
        "needs the bench group installed above"
    ),
    "Build the extension into the venv": (
        "maturin develop, which the caller has already run"
    ),
    "Install dev dependencies": "uv sync, which the caller has already run",
    "Stub matches the extension": "needs the built extension in the venv",
    "A caller's strict types, on the floor and on the current": (
        "needs both interpreters installed"
    ),
}

#: Jobs whose every step is a runner's, so naming each would say nothing more.
RUNNER_ONLY_JOBS = {
    "wheel",
    "wheel-macos",
    "wheel-windows",
    "sdist",
    "pages",
    "release",
}


def workflow() -> dict:
    try:
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        print(f"gate: cannot read {WORKFLOW}: {err}")
        raise SystemExit(EXIT_CANNOT_RUN) from err


def required_jobs(spec: dict) -> list[str]:
    """Read the jobs the `ci` aggregator waits for: the merge gate itself."""
    jobs = spec.get("jobs", {})
    needs = jobs.get("ci", {}).get("needs", [])
    if not needs:
        print("gate: the ci job lists no needs; there is no merge gate to run")
        raise SystemExit(EXIT_CANNOT_RUN)
    return [name for name in needs if name in jobs and name not in RUNNER_ONLY_JOBS]


def steps(spec: dict, job: str) -> Iterator[tuple[str, str]]:
    """Yield the `(name, command)` of every `run:` step of `job`, in order."""
    for step in spec["jobs"][job].get("steps", []):
        command = step.get("run")
        if command is None:
            continue  # a `uses:` step: an action, not a command
        yield step.get("name", command.strip().splitlines()[0]), command


def runnable(name: str) -> bool:
    return name not in NEEDS_A_RUNNER


def shallow_clone(into: Path) -> Path:
    """Make a clone of `HEAD` with one commit and no tags, as a checkout does.

    The difference this exists for. A local clone carries every tag and the
    whole history, so a check that reads either passes here and fails there --
    which is exactly what happened, on eight jobs at once.
    """
    tree = into / "tree"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--no-tags",
            "--quiet",
            f"file://{ROOT}",
            str(tree),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(tree), "checkout", "--quiet", "HEAD"], check=False)
    return tree


def run_step(name: str, command: str, cwd: Path) -> bool:
    print(f"\n=== {name}")
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", command],
        cwd=cwd,
        check=False,
        # One build directory across the jobs and across runs. The clone is what
        # this reproduces; a cold Rust build is not, and paying for one per job
        # would make the gate something nobody runs.
        env={
            **os.environ,
            "CI": "1",
            "CARGO_TARGET_DIR": str(ROOT / "target" / "gate"),
        },
    )
    if result.returncode != 0:
        print(f"FAILED: {name} (exit {result.returncode})")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show the plan and stop")
    parser.add_argument("--job", help="run one job's steps")
    parser.add_argument(
        "--here",
        action="store_true",
        help="use this working tree instead of a fresh shallow clone",
    )
    args = parser.parse_args()

    spec = workflow()
    jobs = [args.job] if args.job else required_jobs(spec)
    unknown = [job for job in jobs if job not in spec.get("jobs", {})]
    if unknown:
        print(f"gate: no such job: {', '.join(unknown)}")
        return EXIT_CANNOT_RUN

    plan = [
        (job, name, command)
        for job in jobs
        for name, command in steps(spec, job)
        if runnable(name)
    ]
    if args.list:
        for job, name, _ in plan:
            print(f"run   {job}: {name}")
        for job in jobs:
            for name, _ in steps(spec, job):
                if not runnable(name):
                    print(f"skip  {job}: {name} -- {NEEDS_A_RUNNER[name]}")
        return EXIT_OK

    if args.here:
        tree = ROOT
        holder = None
        print("gate: running in this working tree (--here)")
    else:
        holder = Path(tempfile.mkdtemp(prefix="valgebra-gate-"))
        tree = shallow_clone(holder)
        print(f"gate: shallow clone of HEAD, no tags, at {tree}")

    try:
        failures = [
            f"{job}: {name}"
            for job, name, command in plan
            if not run_step(f"{job}: {name}", command, tree)
        ]
    finally:
        if holder is not None:
            shutil.rmtree(holder, ignore_errors=True)

    print()
    if failures:
        print(f"gate: {len(failures)} step(s) failed: {', '.join(failures)}")
        return EXIT_FAIL
    print(f"gate: {len(plan)} step(s) passed in a clone shaped like the runner's.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
