"""Every suite that installs its own packages installs what the suite reads.

Two suites run without the dev group: the CI leg on PyPy, and the release's
product suite on each built wheel. Each installs a list written into its step
by hand, and a package the list leaves out does not redden the suite -- a row
reading an optional implementation asks `pytest.importorskip`, which skips. So
the PEP 728 `TypedDict` rows skipped on PyPy while every CPython lane ran them.

So each such list is read against the dev group in `pyproject.toml`, both ways:
every package the group names is on the list or is a tool excused below with
the reason the suite does not read it, and every package on the list is one the
group names.

LEDGER: every hand-written suite install names every package the suite reads
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

#: A quoted requirement in a step's `run`: the name, then a version bound.
REQUIREMENT = re.compile(r'"([A-Za-z0-9][A-Za-z0-9._-]*)[<>=!~][^"]*"')


def _name(requirement: str) -> str:
    """Name a distribution as an index does: lower case, one separator."""
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def _dev_group() -> set[str]:
    groups = load(ROOT / "pyproject.toml")["dependency-groups"]
    return {_name(entry) for entry in groups["dev"] if isinstance(entry, str)}


def _installs() -> dict[str, set[str]]:
    """Every step installing a suite's packages by hand, by where it is."""
    found: dict[str, set[str]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in workflow.get("jobs", {}).items():
            for step in job.get("steps", []):
                run = step.get("run", "")
                if "uv pip install" in run and '"pytest>=' in run:
                    where = f"{path.name}: {job_name}: {step.get('name', run[:40])}"
                    found[where] = {_name(name) for name in REQUIREMENT.findall(run)}
    return found


def test_every_hand_written_install_names_what_the_suite_reads() -> None:
    wanted = _dev_group() - set(EXCUSED)
    drift = []
    for where, installed in _installs().items():
        drift += [f"{where} leaves out {name}" for name in sorted(wanted - installed)]
        drift += [
            f"{where} installs {name}, which the dev group does not name"
            for name in sorted(installed - wanted)
        ]
    assert not drift, (
        "hand-written suite installs adrift from the dev group:\n  "
        + "\n  ".join(drift)
        + "\nAdd the package to each list, or excuse a tool here with the "
        "reason the suite does not read it."
    )


def test_every_excused_tool_is_in_the_dev_group() -> None:
    """An excuse for a package the group does not name excuses nothing."""
    assert set(EXCUSED) <= _dev_group(), sorted(set(EXCUSED) - _dev_group())


def test_the_scan_reads_the_installs_that_are_there() -> None:
    """The other direction: a scan finding nothing would pass having read nothing."""
    installs = _installs()
    assert len(installs) >= 2, sorted(installs)
    for where, installed in installs.items():
        assert "pytest" in installed, where
