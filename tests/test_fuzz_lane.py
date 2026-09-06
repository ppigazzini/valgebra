"""The fuzz soak bounds one allocation, and bounds memory per batch.

A libFuzzer run has two memory ceilings and they answer different questions.
`-rss_limit_mb` watches the *process*, which under the sanitizer grows with the
number of executions whatever the target does, because freed memory is
quarantined rather than returned. `-malloc_limit_mb` watches one *allocation*,
which is the thing a bound in this tree actually promises -- the automaton
builder's eight megabytes, the product's edge count -- and the thing worth
reddening a lane.

Leaving both unnamed gives the second the value of the first, so the only check
that ran was the one that cannot name a defect: the soak crossed the unnamed
2,048 MB default partway through its own budget and reported an out-of-memory
against whichever input was in flight, which replayed in 25 ms.

Held in both directions: the step must carry a fork mode, so the process ceiling
bounds a batch rather than the whole run, and it must name the allocation
ceiling rather than inherit one. A step that drops either fails here.

LEDGER: the fuzz soak names its allocation ceiling and forks its batches
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# A repository check: it reads the workflow, which ships in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: The step that runs the soak, by its `name:` in the workflow.
SOAK = "Fuzz the decision procedures"

#: The largest single allocation the soak may make, in megabytes. Above every
#: bound the code can ask for -- the regex builder's is eight -- and far below
#: the process ceiling it replaces, so it fires on a runaway allocation and on
#: nothing else.
ALLOCATION_CEILING_MB = 64


def _soak_step() -> str:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in spec["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == SOAK:
                return str(step["run"])
    message = f"the workflow has no {SOAK!r} step"
    raise AssertionError(message)


def test_the_soak_names_its_allocation_ceiling() -> None:
    """An unnamed `-malloc_limit_mb` is the process ceiling wearing its name."""
    step = _soak_step()
    found = re.search(r"-malloc_limit_mb=(\d+)", step)
    assert found, (
        "the soak does not name -malloc_limit_mb, so the only memory check it "
        "runs is the process one, which the sanitizer's own growth trips"
    )
    assert int(found.group(1)) == ALLOCATION_CEILING_MB, (
        f"the soak bounds one allocation at {found.group(1)} MB; this file "
        f"records {ALLOCATION_CEILING_MB}. Moving it is a decision with an "
        "argument, so move both."
    )


def test_the_soak_bounds_memory_per_batch() -> None:
    """Without a fork the process ceiling bounds the whole run, which grows."""
    step = _soak_step()
    assert re.search(r"-fork=\d+", step), (
        "the soak does not fork, so its memory ceiling applies to a process "
        "whose resident size grows with every execution it has done"
    )


def test_the_soak_still_has_a_time_budget() -> None:
    """The flags above are additions, not a replacement for the budget."""
    assert re.search(r"-max_total_time=\d+", _soak_step())
