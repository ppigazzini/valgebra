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


# --- What the sweep's interpreter imports -------------------------------------

#: The line that starts a sweep's embedded interpreter from the venv the lock
#: fills, rather than from whichever `python3` the runner's `PATH` holds first.
_VENV_FIRST = re.compile(
    r'^\s*export PATH="\$\(dirname "\$PYO3_PYTHON"\):\$PATH"\s*$', re.MULTILINE
)


def _embedded_sweeps() -> dict[str, str]:
    """Give each step that sweeps against an embedded interpreter, by its job."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for name, job in workflow["jobs"].items():
        for step in job.get("steps") or []:
            run = step.get("run") or ""
            if "cargo mutants" in run and "interpreter-tests" in run:
                found[name] = run
    return found


def test_a_sweep_starts_its_embedded_interpreter_from_the_venv() -> None:
    """A note about what the lock installs holds only where the sweep imports it.

    The walk baseline accepts the mutants of the arms for `typing_extensions`'
    own objects, because on 3.12 that module's `Required`, `NotRequired`,
    `Unpack`, `Any` and `Never` are `typing`'s own: the notes argue from 3.12
    *with the module installed from the lock*. The embedded interpreter takes
    its prefix from the first `python3` on `PATH`, and `PYO3_PYTHON` only
    configures the build. With the venv named by `PYO3_PYTHON` alone, uv's build
    of 3.12 starts on its bare prefix, imports no `typing_extensions`, the
    corpus installs its stand-in, and all four mutants are caught: the nightly
    read four entries stale, and the notes stayed correct the whole time.

    So every sweep that embeds its interpreter puts the venv first on `PATH`
    before it runs, as the tooling page's recipe does.
    """
    sweeps = _embedded_sweeps()
    assert set(sweeps) >= {"nightly-mutants-walk", "mutants-diff-walk"}, sorted(sweeps)
    unstarted = sorted(
        job
        for job, run in sweeps.items()
        if not (
            (line := _VENV_FIRST.search(run)) is not None
            and line.start() < run.index("cargo mutants")
        )
    )
    assert not unstarted, (
        f"sweeps whose embedded interpreter starts outside the venv: {unstarted}. "
        "Export the venv's `bin` first on `PATH` before `cargo mutants`, or the "
        "interpreter imports what the runner's first `python3` has rather than "
        "what the baseline's notes argue from."
    )


#: An embedded interpreter started from the base installation, as the `python`
#: job's corpora start theirs: an interpreter with no environment to live in,
#: reading its standard library from the prefix it is handed.
_BASE_HOME = re.compile(r'\bPYTHONHOME="\$base"')


def _interpreter_steps() -> dict[str, str]:
    """Give each step that embeds an interpreter in a test binary, by job and step."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return {
        f"{name}: {step.get('name', '')}": str(step.get("run") or "")
        for name, job in workflow["jobs"].items()
        for step in job.get("steps") or []
        if "--features interpreter-tests" in str(step.get("run") or "")
    }


def test_every_embedded_interpreter_names_its_prefix() -> None:
    """A step linking the interpreter says which installation it starts from.

    The prefix decides what the interpreter imports: from the venv, the module
    the lock installs; from the base installation, the standard library alone,
    where the corpus installs its stand-in for `typing_extensions`. Both are
    readings a lane may want -- the sweeps' baseline notes argue from the first,
    the `python` job's corpora read the second on every release -- and neither
    is the one a step gets by naming nothing, which is whatever the runner's
    `PATH` holds first. So each step names exactly one, before it runs the
    tests: the venv's `bin` first on `PATH`, or `PYTHONHOME` at the base.
    """
    steps = _interpreter_steps()
    assert len(steps) >= 4, sorted(steps)
    unnamed = []
    for where, run in sorted(steps.items()):
        at = run.index("--features interpreter-tests")
        venv = (line := _VENV_FIRST.search(run)) is not None and line.start() < at
        base = (home := _BASE_HOME.search(run)) is not None and home.start() < at
        if venv == base:
            unnamed.append(f"{where} ({'both' if venv else 'neither'})")
    assert not unnamed, (
        f"steps embedding an interpreter without naming one prefix: {unnamed}. "
        "Export the venv's `bin` first on `PATH`, or hand the base installation "
        "as `PYTHONHOME`, before the tests run."
    )
