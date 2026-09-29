"""A workflow installs the lock, once, and runs on what it installed.

`docs/dev/07-tooling-ci.md` says every lane installs with one `uv sync` and runs
every later command under `uv run --no-sync`. Both halves decide what a lane
tests. A `uv sync` without `--locked` resolves afresh where the lock is stale,
so the Pages deploy could publish a site built from a resolution no lane had
tested. A bare `uv run` resolves again before it runs, uninstalling the build
`maturin develop` made. Held over every workflow, so a job added tomorrow is
held the day it lands.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# The repository checks are not the product suite: this file reads the
# workflows, none of which ship in a wheel.
pytestmark = pytest.mark.repository

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _jobs() -> list[tuple[str, dict]]:
    return [
        (f"{path.name}:{name}", job)
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for name, job in (
            yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
        ).items()
    ]


def _commands(job: dict, tool: str) -> list[str]:
    """Every line of the job's `run:` steps invoking `uv <tool>`."""
    return [
        line.strip()
        for step in job.get("steps", [])
        for line in str(step.get("run", "")).splitlines()
        if re.search(rf"\buv {tool}\b", line)
    ]


def test_the_workflows_are_read() -> None:
    assert len(_jobs()) >= 10, "the workflow scan found almost no jobs"


def test_every_sync_installs_the_lock() -> None:
    unlocked = [
        f"{job}: {line}"
        for job, body in _jobs()
        for line in _commands(body, "sync")
        if "--locked" not in line
    ]
    assert not unlocked, f"a sync that may resolve afresh: {unlocked}"


def test_every_run_in_a_checkout_runs_on_what_the_job_installed() -> None:
    # A job with no checkout has no project for `uv run` to resolve, so the
    # rule is about the jobs that check the repository out.
    checked_out = [
        (job, body)
        for job, body in _jobs()
        if any(
            str(step.get("uses", "")).startswith("actions/checkout@")
            for step in body.get("steps", [])
        )
    ]
    assert checked_out, "no job checks the repository out"
    resolving = [
        f"{job}: {line}"
        for job, body in checked_out
        for line in _commands(body, "run")
        if "--no-sync" not in line
    ]
    assert not resolving, f"a run that resolves the environment again: {resolving}"
