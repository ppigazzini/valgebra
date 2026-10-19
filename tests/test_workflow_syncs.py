"""A workflow installs the lock, once, and runs on what it installed.

`docs/dev/07-tooling-ci.md` says every lane installs with one `uv sync` and runs
every later command under `uv run --no-sync`. Both halves decide what a lane
tests. A `uv sync` without `--locked` resolves afresh where the lock is stale,
so a Pages deploy can publish a site built from a resolution no lane tested. A
bare `uv run` resolves again before it runs, uninstalling the build `maturin
develop` made. Held over every workflow, so a job added tomorrow is held the day
it lands.

A build of the extension that does not run through the lock fetches its own
maturin: the build action resolves `[build-system] requires` against a manifest
it fetches on the day, and `pip install maturin` takes the newest. Each names
the version `uv.lock` resolves, so a wheel is built by the maturin the lanes
tested rather than by whichever is newest when it runs.

Cargo has a lock of its own, and a build without `--locked` rewrites one that no
longer matches the manifests where a build with it refuses: the lane then
tests, and `cargo deny` audits, a resolution nobody committed. So every cargo
command a workflow runs that resolves the graph to build passes it; `cargo
fuzz` and `cargo mutants` take no `--locked` and are not read.

LEDGER: every sync, cargo build and maturin reads the lock; every run runs on the sync
"""

from __future__ import annotations

import re

import pytest

from _toml import load
from _tree import ROOT, WORKFLOWS, read_workflow

# The repository checks are not the product suite: this file reads the
# workflows, none of which ship in a wheel.
pytestmark = pytest.mark.repository


def _jobs() -> list[tuple[str, dict]]:
    return [
        (f"{path.name}:{name}", job)
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for name, job in (read_workflow(path)["jobs"]).items()
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


#: The build action, which fetches a maturin of its own.
MATURIN_ACTION = "PyO3/maturin-action@"

#: A pip install of maturin, with the version it pins if it pins one.
PIP_MATURIN = re.compile(r"\bpip install\b.*?\bmaturin\b(?:==(\S+))?")


def _locked_maturin() -> str:
    """Give the maturin version `uv.lock` resolves."""
    packages = load(ROOT / "uv.lock")["package"]
    (version,) = {
        package["version"] for package in packages if package["name"] == "maturin"
    }
    return version


def test_every_maturin_fetched_past_the_lock_is_the_locks() -> None:
    locked = _locked_maturin()
    fetched = []
    for job, body in _jobs():
        for step in body.get("steps", []):
            if str(step.get("uses", "")).startswith(MATURIN_ACTION):
                version = (step.get("with") or {}).get("maturin-version")
                fetched.append((job, str(version or "the newest")))
            pinned = map(PIP_MATURIN.search, str(step.get("run", "")).splitlines())
            fetched.extend(
                (job, found.group(1) or "the newest") for found in pinned if found
            )
    # The scan is the detector: the release's two builds, the push lane's wheel
    # and the PyPy lane's venv.
    assert len(fetched) >= 4, f"the scan found {fetched}"
    wrong = [
        f"{job}: maturin {version}" for job, version in fetched if version != locked
    ]
    assert not wrong, f"a maturin past the lock, which resolves {locked}: {wrong}"


#: A cargo command that resolves the dependency graph, with the rest of its
#: line. `cargo fuzz` and `cargo mutants` take no `--locked` and are not read.
CARGO_BUILD = re.compile(
    r"\bcargo\s+(?:\+(?:\$\{\{[^}]*\}\}|\S+)\s+)?"
    r"(build|check|clippy|doc|test|run|deny|llvm-cov)\b([^;&|\n]*)"
)

#: The `cargo llvm-cov` subcommands that read a profile or print the
#: environment, and build nothing.
BUILDS_NOTHING = frozenset({"clean", "report", "show-env"})


def test_every_cargo_build_reads_the_lock() -> None:
    builds = []
    for job, body in _jobs():
        for step in body.get("steps", []):
            text = str(step.get("run", "")).replace("\\\n", " ")
            for line in text.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for found in CARGO_BUILD.finditer(line):
                    tool, rest = found.groups()
                    words = rest.split()
                    if tool == "llvm-cov" and words and words[0] in BUILDS_NOTHING:
                        continue
                    builds.append((job, " ".join(found.group(0).split()), rest))
    # The scan is the detector: the lint, test, MSRV, doc, coverage, deny and
    # fuzz harness lanes all build.
    assert len(builds) >= 10, f"the scan found {builds}"
    unlocked = [
        f"{job}: {command}" for job, command, rest in builds if "--locked" not in rest
    ]
    assert not unlocked, f"a cargo build that may rewrite the lock: {unlocked}"
