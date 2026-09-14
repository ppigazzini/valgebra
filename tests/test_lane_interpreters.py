"""Every lane that installs an interpreter names which one, and runs there.

A ceiling in `scripts/perf_compare.json`, a band in `scripts/perf_budget.json`
and a mutation sweep are each claims about an interpreter as much as about the
code: a ratio is a property of the pair running, a budget's band was widened to
cover the distance between two releases, and a mutant on a version-gated branch
is killable on one interpreter and unviable on the next.

The wrapper action's `python-version` defaults to the empty string, documented
as "left empty, uv picks one itself", and what uv picks is whatever the runner
image ships. So those claims rested on an image's default, and the day the
image moves they would all change meaning at once with every lane green.

Held over both workflows: a `setup-uv` use with no version fails, and the
version each lane names is the one its own comment argues for.

The second half is the implementations. The packaging metadata states CPython
and PyPy, which is a promise that a schema means the same thing on both -- and
for eight months only CPython ran the suite. PyPy's C API answers differently
as well as exporting differently: `PyTuple_Size` goes through the object's own
`__len__` there, and a `tuple` subclass that overrode it walked past the end of
its storage and killed the process. The import check the lane had could not see
that, because an import is not an answer. So a stated implementation has a lane
that runs the suite, and a lane that runs the suite on an implementation is one
the metadata states.

LEDGER: no lane installs an interpreter without naming it
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# A repository check: it reads the workflows, which ship in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

#: The wrapper every lane installs its interpreter through.
WRAPPER = "/.github/actions/setup-uv"

#: The implementation a bare version string means, which nobody spells out.
CPYTHON = "cpython"

#: A `run:` step that invokes pytest, rather than one that installs it: the
#: dependency is named in quotes beside its floor, and a match on the bare word
#: would read the install as a run.
_RUNS_PYTEST = re.compile(r"(?:^|\s)(?:uv run |[\w./-]*python -m )?pytest\b")


#: An interpreter uv is asked for by name: `pypy-3.11`, `graalpy-24`.
_NAMED = re.compile(r"\b([a-z][a-z_]*[a-z])-\d")

#: One asked for by version alone, which is how CPython is spelled.
_BARE = re.compile(r"(?<![\w.-])\d+\.\d+")


def _interpreter_text(job: dict, step: dict) -> str:
    """Read the interpreter a lane installs, with its matrix folded in.

    A matrix leg names its version through an expression, so the step alone
    says `${{ matrix.python-version }}` and nothing about an implementation.
    The matrix's own value -- a list, or the expression that chooses between
    two lists -- is the text that does, so it is read in the expression's
    place rather than the lane being read as CPython by default.
    """
    named = str((step.get("with") or {}).get("python-version") or "")
    if "matrix." not in named:
        return named
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    parts = [named]
    for key, value in matrix.items():
        if key == "include":
            parts.extend(str(row) for row in value)
        elif f"matrix.{key}" in named:
            parts.append(str(value))
    return " ".join(parts)


def _implementations_in(text: str) -> set[str]:
    """Which implementations a lane's interpreter text asks uv for.

    uv spells a non-CPython interpreter by naming it and leaves CPython unsaid,
    so a version standing on its own is CPython.
    """
    found = set(_NAMED.findall(text.lower()))
    if _BARE.search(text):
        found.add(CPYTHON)
    return found


def _stated_implementations() -> set[str]:
    """Read the implementations the packaging metadata promises to support.

    Every quoted string in the file, filtered by the classifier's own prefix,
    rather than parsed: `tomllib` is 3.11+ and this suite runs from 3.10, which
    `tests/test_mutation_scope.py` says beside the same decision. The prefix is
    distinctive enough that a string carrying it anywhere in this file is that
    classifier.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    prefix = "Programming Language :: Python :: Implementation :: "
    return {
        row.removeprefix(prefix).lower()
        for row in re.findall(r'"([^"]+)"', text)
        if row.startswith(prefix)
    }


def _implementations_running_the_suite() -> dict[str, list[str]]:
    """Group every lane that runs pytest by the implementation it installs."""
    running: dict[str, list[str]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            if not any(_RUNS_PYTEST.search(str(step.get("run", ""))) for step in steps):
                continue
            for step in steps:
                if not str(step.get("uses", "")).endswith(WRAPPER):
                    continue
                for name in _implementations_in(_interpreter_text(job, step)):
                    running.setdefault(name, []).append(f"{path.name}:{job_name}")
    return running


def _uses_without_a_version() -> list[str]:
    """Return `job/step` for every wrapper use that names no interpreter."""
    unnamed = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (workflow.get("jobs") or {}).items():
            for index, step in enumerate(job.get("steps") or []):
                uses = str(step.get("uses", ""))
                if not uses.endswith(WRAPPER):
                    continue
                version = (step.get("with") or {}).get("python-version")
                if not str(version or "").strip():
                    unnamed.append(f"{path.name}:{job_name}:step {index}")
    return unnamed


def test_no_lane_lets_the_image_choose_its_interpreter() -> None:
    unnamed = _uses_without_a_version()
    assert not unnamed, (
        f"lanes installing an interpreter without naming one: {unnamed}. "
        "Add `python-version` with the reason that lane needs that version, or "
        "the claims it makes belong to the runner image rather than to this tree."
    )


def test_the_wrapper_still_defaults_to_letting_uv_choose() -> None:
    """The rule above is about the lanes, and it is worth only what the default is.

    If the wrapper itself started naming a version, every lane would inherit one
    and this ledger would pass while saying nothing. The check is that the hole
    it closes is still open at the wrapper.
    """
    action = yaml.safe_load(
        (ROOT / ".github" / "actions" / "setup-uv" / "action.yml").read_text(
            encoding="utf-8"
        )
    )
    default = action["inputs"]["python-version"].get("default", "")
    assert not str(default).strip(), (
        "the wrapper names a default interpreter, so a lane naming none inherits "
        "it and this ledger checks nothing; either drop the default or hold the "
        "lanes to the wrapper's value instead"
    )


def test_every_stated_implementation_runs_the_suite_somewhere() -> None:
    stated = _stated_implementations()
    running = _implementations_running_the_suite()
    assert stated, "the packaging metadata states no implementation to hold"
    missing = sorted(stated - running.keys())
    assert not missing, (
        f"stated as targets with no lane running the suite: {missing}. "
        "A classifier is a promise that a schema means the same thing there, "
        "and an import check cannot see an answer that differs."
    )


def test_no_lane_runs_the_suite_on_an_implementation_nobody_states() -> None:
    """The other direction: a lane is the evidence for a claim somebody made."""
    unstated = sorted(
        _implementations_running_the_suite().keys() - _stated_implementations()
    )
    assert not unstated, (
        f"lanes run the suite on implementations the metadata does not state: "
        f"{unstated}. Add the classifier, or the lane is spending minutes on a "
        "target no consumer is told about."
    )
