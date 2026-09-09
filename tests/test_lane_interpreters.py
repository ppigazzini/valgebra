"""Every lane that installs an interpreter names which one.

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

LEDGER: no lane installs an interpreter without naming it
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# A repository check: it reads the workflows, which ship in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

#: The wrapper every lane installs its interpreter through.
WRAPPER = "/.github/actions/setup-uv"


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
