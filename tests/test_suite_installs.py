"""Every suite that installs its own packages installs the test group's lock.

Three suites run without the dev group: the CI leg on PyPy, the release's
product suite on each built wheel, and the same suite on each musllinux wheel in
its Alpine container, through the image's own pip. Each installs the `test`
dependency group, exported from `uv.lock`, so it reads the versions every other
lane reads: a list written into a step by hand resolves against the index on
every run, and a release of any package on it reddens a push with nothing in
the tree changed.

The group is held to the dev group: `dev` includes it, and names besides only
the tools excused below, each with the reason the suite does not read it. A
package the suite reads and the group leaves out would not redden a suite -- a
row reading an optional implementation asks `pytest.importorskip`, which skips
-- so the PEP 728 `TypedDict` rows would skip on PyPy while every CPython lane
ran them.

LEDGER: every suite apart from dev installs the test group at the lock's versions
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from _toml import load

# This reads the tree rather than the library: it is a claim about the
# workflows, and nothing it touches ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

#: The group a suite installed apart from the dev group installs.
GROUP = "test"

#: The dev group's packages a suite does not read, each with the reason.
EXCUSED: dict[str, str] = {
    "maturin": "builds the extension; these suites install a wheel built before",
    "mypy": "a checker the checker ledgers run, which skip where it is absent",
    "pyright": "a checker the checker ledgers run, which skip where it is absent",
    "pre-commit": "runs the hooks before a commit, not the suite",
    "pytest-cov": "measures coverage, which its own lane gates on CPython",
    "ruff": "lints and formats, once, on CPython",
    "ty": "type-checks, once, on CPython",
}

#: The export a suite installs from: the group alone, at the lock's versions,
#: written to a file the install then reads.
EXPORT = re.compile(
    rf"uv export (?:--project \S+ )?--locked --only-group {GROUP} --no-emit-project"
    r"\s*(?:\\\n\s*)?-o (\S+)"
)

#: A quoted requirement with a version bound, the shape a hand-written list takes.
REQUIREMENT = re.compile(r'"[A-Za-z0-9][A-Za-z0-9._-]*[<>=!~][^"]*"')


def _name(requirement: str) -> str:
    """Name a distribution as an index does: lower case, one separator."""
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def _groups() -> dict[str, list]:
    return load(ROOT / "pyproject.toml")["dependency-groups"]


def _jobs() -> list[tuple[str, dict]]:
    return [
        (f"{path.name}: {name}", job)
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for name, job in yaml.safe_load(path.read_text(encoding="utf-8"))
        .get("jobs", {})
        .items()
    ]


def _runs(job: dict) -> list[str]:
    return [str(step.get("run", "")) for step in job.get("steps", [])]


def _suites_apart() -> set[str]:
    """Every job that runs the suite and syncs no environment of its own."""
    return {
        name
        for name, job in _jobs()
        if any("-m pytest" in run for run in _runs(job))
        and not any("uv sync" in run for run in _runs(job))
    }


def test_the_dev_group_is_the_test_group_and_its_tools() -> None:
    """What `dev` holds besides the suite's packages is a tool, excused by name."""
    groups = _groups()
    dev = groups["dev"]
    assert {"include-group": GROUP} in dev, f"the dev group does not include {GROUP}"
    tools = {_name(entry) for entry in dev if isinstance(entry, str)}
    unexcused = sorted(tools - set(EXCUSED))
    assert not unexcused, (
        f"the dev group names {unexcused} beside the {GROUP} group. Move a "
        f"package the suite reads into `{GROUP}`, or excuse a tool here with the "
        "reason the suite does not read it."
    )
    stale = sorted(set(EXCUSED) - tools)
    assert not stale, f"excuses for packages the dev group does not name: {stale}"
    suite = {_name(entry) for entry in groups[GROUP] if isinstance(entry, str)}
    assert "pytest" in suite, f"the {GROUP} group does not hold pytest"
    assert not suite & set(EXCUSED), sorted(suite & set(EXCUSED))


def test_every_suite_apart_installs_the_test_group_from_the_lock() -> None:
    """The versions are the lock's, so the index moves no suite by itself.

    `--locked` refuses a lock the project has moved past rather than
    resolving anew, and the install reads the exported file and nothing
    written beside it.
    """
    apart = _suites_apart()
    assert len(apart) >= 4, f"only {sorted(apart)} run the suite apart from dev"
    drift = []
    for name, job in _jobs():
        if name not in apart:
            continue
        runs = "\n".join(_runs(job))
        exported = EXPORT.findall(runs)
        if not exported:
            drift.append(f"{name} exports no `{GROUP}` group from the lock")
            continue
        if not any(f"-r {path}" in runs or f"-r /{path}" in runs for path in exported):
            drift.append(f"{name} exports {exported} and installs none of them")
        if written := REQUIREMENT.findall(runs):
            drift.append(f"{name} installs {written} by hand beside the lock")
    assert not drift, "suites installed apart from the lock:\n  " + "\n  ".join(drift)
