"""A workflow installs its Rust toolchain with rustup, through a local action.

A third-party action is pinned by hash, with a comment naming the tag the hash
came from, and the workflow audit checks that the tag still names that hash. A
toolchain action that publishes one moving tag moves it with every upstream
change, so the pin and its comment part company and the audit fails every lane
behind it until somebody re-pins. Everything such an action ran for these lanes
is one rustup command, which every hosted runner carries, so
`.github/actions/setup-rust` runs it directly and there is no tag to drift.

Held over every workflow and every local action, so a toolchain action added
tomorrow fails here the day it lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# A repository check: it reads the workflows, which ship in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
GITHUB = ROOT / ".github"

#: The local action every lane installs its toolchain through, in the
#: self-repository form the workflow audit asks for.
WRAPPER = "$/.github/actions/setup-rust"


def _steps() -> list[tuple[str, dict]]:
    """Every step of every workflow job and every local action, with its place."""
    found: list[tuple[str, dict]] = []
    for path in sorted((GITHUB / "workflows").glob("*.yml")):
        jobs = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
        for name, job in jobs.items():
            found.extend((f"{path.name}:{name}", step) for step in job.get("steps", []))
    for path in sorted((GITHUB / "actions").glob("*/action.yml")):
        runs = yaml.safe_load(path.read_text(encoding="utf-8"))["runs"]
        where = str(path.relative_to(ROOT))
        found.extend((where, step) for step in runs.get("steps", []))
    return found


def _uses() -> list[tuple[str, str]]:
    return [(where, str(step["uses"])) for where, step in _steps() if "uses" in step]


def test_the_steps_are_read() -> None:
    # A scan that matches nothing passes every assertion below.
    assert len(_uses()) >= 20, "the workflow scan found almost no actions"
    assert any(uses == WRAPPER for _, uses in _uses()), "no lane uses the wrapper"


def test_no_toolchain_comes_from_a_third_party_action() -> None:
    third_party = [
        f"{where}: {uses}"
        for where, uses in _uses()
        if "toolchain" in uses.lower() and not uses.startswith(("$/", "./"))
    ]
    assert not third_party, (
        f"a toolchain installed by a third-party action: {third_party}; use {WRAPPER}"
    )


def test_the_wrapper_runs_rustup_and_no_action() -> None:
    wrapper = ROOT / WRAPPER.removeprefix("$/") / "action.yml"
    steps = yaml.safe_load(wrapper.read_text(encoding="utf-8"))["runs"]["steps"]
    assert not [step["uses"] for step in steps if "uses" in step]
    assert any("rustup" in str(step.get("run", "")) for step in steps)
