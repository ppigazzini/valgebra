"""A survivor's note that argues from the sweep's interpreter names that one.

An accepted survivor carries an argument for why no test kills it, and some of
those arguments are about the **interpreter the sweep runs**: a mutation can be
equivalent on one CPython and killable on another, because the objects the code
compares are not the same objects across releases.

Such an argument is a claim about the lane's own environment, and a lane can be
repinned. One was: the walk sweep moved from 3.14 to 3.12, where `typing.Union`
and `types.UnionType` are two objects rather than one, and a note reading
"equivalent on the interpreter this sweep runs" went on saying so for two days.
The sweep killed the mutant, the ratchet reported the entry stale, and the
nightly was red for a sentence.

Nothing held it. `tests/test_mutation_scope.py` holds every note to a mutant
that exists, and `tests/test_lane_interpreters.py` holds every lane to naming
its interpreter; neither reads what the notes say *about* that interpreter.

So this does. A note that argues from the sweep's own interpreter has to name
the version the sweep pins -- it may discuss any other release it likes, and
most of these notes do, but the one it claims to run on must be the one in the
workflow.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# This file reads the tree rather than the library: it holds two recorded files
# to each other, and exercises no schema.
pytestmark = pytest.mark.repository

#: Each baseline, and the workflow job whose interpreter its notes may argue
#: from. The core sweep installs no interpreter at all -- it is pure Rust -- so
#: a note there that argues from one is arguing from nothing, which is why it is
#: listed with `None` rather than left out.
BASELINES = {
    "scripts/mutation_baseline_walk.json": "nightly-mutants-walk",
    "scripts/mutation_baseline.json": "nightly-mutants",
}

#: A claim about the environment the sweep itself runs in, as these notes spell
#: it. Not every mention of a version is one: a note may say what another
#: release does, and several do, which is why the phrase is what is matched.
_ABOUT_THIS_SWEEP = re.compile(
    r"\b(?:the interpreter this sweep|this sweep (?:runs|links|is run)|"
    r"on the interpreter this sweep|the interpreter the sweep)\b",
    re.IGNORECASE,
)

#: A CPython release as a note writes one: `3.12`, `3.14t`.
_VERSION = re.compile(r"\b3\.\d{1,2}t?\b")


def _pinned(job: str) -> str | None:
    """Read the interpreter `job` installs, or `None` where it installs none."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][job].get("steps", [])
    versions = [
        (step.get("with") or {}).get("python-version")
        for step in steps
        if isinstance(step, dict)
    ]
    found = [str(version) for version in versions if version]
    return found[0] if found else None


def _notes(path: str) -> dict[str, str]:
    recorded = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return recorded.get("_accepted", {})


def test_the_baselines_and_the_workflow_are_read_at_all() -> None:
    # A scan that finds nothing makes every assertion below vacuous, which is
    # the failure mode a ledger has: it passes loudest when it is broken.
    assert _pinned("nightly-mutants-walk") == "3.12"
    assert _pinned("nightly-mutants") is None
    for path in BASELINES:
        assert len(_notes(path)) >= 5, f"{path} carries no accepted notes"


@pytest.mark.parametrize(("path", "job"), list(BASELINES.items()))
def test_a_note_arguing_from_the_sweeps_interpreter_names_it(
    path: str, job: str
) -> None:
    pinned = _pinned(job)
    wrong: list[str] = []
    for mutant, note in _notes(path).items():
        if not _ABOUT_THIS_SWEEP.search(note):
            continue
        if pinned is None:
            wrong.append(
                f"{mutant}: argues from the sweep's interpreter, and "
                f"`{job}` installs none"
            )
        elif pinned not in _VERSION.findall(note):
            wrong.append(
                f"{mutant}: argues from the sweep's interpreter and names "
                f"{sorted(set(_VERSION.findall(note)))}, not the pinned {pinned}"
            )
    assert not wrong, (
        "accepted notes argue from an interpreter the lane does not run:\n"
        + "\n".join(wrong)
        + f"\nThe lane pins {pinned}. Re-read the argument against that "
        "release: a mutant equivalent on one CPython is often killable on "
        "another, which is how this class of entry goes stale."
    )
