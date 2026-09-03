"""Every job that runs on a pull request must be able to block the merge.

The workflow's `ci` job gates on each required job having *succeeded*, and it
names them twice: once in `needs`, so the gate waits for them, and once in an
`if` condition, so a skipped or cancelled job fails the gate rather than
slipping past a check that only catches failure. Two hand-written lists beside a
growing job set is the shape that drifts, and it did: a gate added to the
workflow ran on every pull request and could not block one, because neither list
named it.

Held in three directions:

* a job that runs on a pull request and is absent from `needs` fails, so a new
  gate blocks a merge the day it arrives;
* a job in `needs` that the `if` condition does not test fails, so the gate
  cannot wait for a job whose result it then ignores;
* a name in either list that is no longer a job fails, so a list cannot outlive
  what it describes.

The nightly jobs are exempt by their own condition: they run only on the
schedule, so requiring them would block every merge on work no pull request
does.

LEDGER: every pull-request job is required by the merge gate
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
# The gate itself, which cannot require its own success.
GATE = "ci"


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _jobs() -> dict[str, str]:
    """Every job in the workflow, mapped to its body.

    Read with a regex rather than a YAML parser: `tomllib` has no YAML
    counterpart in the standard library, and a third-party parser would be a
    dependency added for one file this repository owns. A job is a two-space
    key under `jobs:`, which is the only nesting level that shape occurs at.
    """
    text = _text()
    body = text.split("\njobs:\n", 1)[1]
    heads = re.finditer(r"^  ([a-z0-9-]+):$", body, re.MULTILINE)
    starts = [(m.group(1), m.start()) for m in heads]
    bounds = [*[s for _, s in starts[1:]], len(body)]
    return {
        name: body[start:end] for (name, start), end in zip(starts, bounds, strict=True)
    }


def _scheduled_only(job_body: str) -> bool:
    return "if: github.event_name == 'schedule'" in job_body


def _needs() -> set[str]:
    needs = r"^    needs:\n      \[(.*?)\]"
    match = re.search(needs, _text(), re.DOTALL | re.MULTILINE)
    assert match is not None, "the ci job lists no needs"
    return {name.strip() for name in match.group(1).split(",") if name.strip()}


def _tested() -> set[str]:
    return set(re.findall(r"needs\.([a-z0-9-]+)\.result != 'success'", _text()))


def test_every_pull_request_job_is_required() -> None:
    jobs = _jobs()
    # The glob is the detector: an empty job set would pass having read nothing.
    assert len(jobs) >= 15, f"the job scan found only {sorted(jobs)}"

    required = _needs()
    missing = sorted(
        name
        for name, body in jobs.items()
        if name != GATE and not _scheduled_only(body) and name not in required
    )
    assert not missing, (
        f"jobs that run on a pull request but cannot block it: {missing}. "
        "Add each to the ci job's `needs` and to its success condition."
    )


def test_every_required_job_has_its_result_tested() -> None:
    # Waiting for a job whose result is never read is a gate that reports
    # success for a job that failed to run at all.
    untested = sorted(_needs() - _tested())
    assert not untested, f"required jobs whose result the gate ignores: {untested}"


def test_no_required_job_has_gone() -> None:
    jobs = set(_jobs())
    stale = sorted((_needs() | _tested()) - jobs)
    assert not stale, f"the merge gate names jobs that do not exist: {stale}"


def test_a_nightly_job_is_not_required() -> None:
    # The exemption is real and must stay narrow: a job is excused only by its
    # own schedule condition, not by being forgotten.
    jobs = _jobs()
    nightly = {name for name, body in jobs.items() if _scheduled_only(body)}
    assert nightly, "no job is schedule-only; the exemption has no subject"
    assert not (nightly & _needs()), (
        f"schedule-only jobs required of every merge: {sorted(nightly & _needs())}"
    )
