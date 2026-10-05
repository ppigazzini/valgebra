"""The property profiles draw what their names say, on a runner as on a laptop.

A profile is read in the environment it runs in, and a runner sets ``CI``, which
makes Hypothesis's own ``ci`` settings the default a registered profile inherits:
derandomised, no database. A deep profile that inherits them asks the same
examples every night. So the profiles are read here in a child interpreter with
``CI`` set, where the inheritance happens, rather than in this one, where it may
not.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from hypothesis import settings
from hypothesis.database import DirectoryBasedExampleDatabase

# A repository check: it reads the suite's own `conftest.py` and the workflow,
# neither of which ships in a wheel.
pytestmark = pytest.mark.repository

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

#: Load `conftest.py` under one profile and print the settings it selected.
READ = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); import conftest; "
    "from hypothesis import settings; d = settings.default; "
    "print(json.dumps({'derandomize': d.derandomize, "
    "'database': d.database is not None, 'max_examples': d.max_examples}))"
)


def _selected(profile: str) -> dict[str, object]:
    env = os.environ | {"CI": "true", "HYPOTHESIS_PROFILE": profile}
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, "-c", READ, str(TESTS)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_the_nightly_profile_draws_at_random_and_keeps_its_failures() -> None:
    """The deep lane asks new examples each night and stores what fails."""
    assert _selected("nightly") == {
        "derandomize": False,
        "database": True,
        "max_examples": 5000,
    }


@pytest.mark.parametrize("profile", ["ci", "dev"])
def test_a_merge_gate_profile_is_the_same_red_on_a_rerun(profile: str) -> None:
    """The ``ci`` profile is derandomised on purpose; ``dev`` keeps the default.

    ``dev`` is asked too, because it is what runs when a caller sets no profile,
    and on a runner it inherits Hypothesis's own derandomised ``ci`` settings.
    """
    selected = _selected(profile)
    assert selected["derandomize"] is True
    assert selected["database"] is False


def test_the_nightly_lane_hands_its_database_back_each_night() -> None:
    """What a red night keeps is replayed on the next, so a red stays red.

    A runner starts empty, and ``actions/cache`` saves after a green job only,
    while the database matters after a red one. So the lane restores the
    directory the ``nightly`` profile names before the deep suite and saves it
    after, whatever the suite answered.
    """
    database = settings.get_profile("nightly").database
    assert isinstance(database, DirectoryBasedExampleDatabase)
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["nightly-fuzz"]["steps"]
    (deep,) = (
        i
        for i, step in enumerate(steps)
        if step.get("env", {}).get("HYPOTHESIS_PROFILE") == "nightly"
    )

    def carries(step: dict[str, object], action: str) -> bool:
        with_ = step.get("with", {})
        return (
            str(step.get("uses", "")).startswith(f"actions/cache/{action}@")
            and isinstance(with_, dict)
            and Path(with_.get("path", "")) == Path(database.path)
        )

    assert any(carries(step, "restore") for step in steps[:deep])
    assert any(
        carries(step, "save") and step.get("if") == "always()"
        for step in steps[deep + 1 :]
    )


def test_every_property_file_runs_under_the_nightly_profile() -> None:
    """A property only the merge gate draws asks the same examples forever.

    The ``ci`` profile is derandomised: its draw is a function of the test's
    source, so every push asks the same examples until the test is edited. Only
    the nightly lane draws at random, and only in the files its deep suite
    names. Seven files with twelve properties were left off that list -- the
    four that hold the walk to pydantic-core and jsonschema among them -- so the
    tree's one independent oracle answered the same 2,000 questions on every
    push and none at night.
    """
    drawn = {
        path.relative_to(ROOT).as_posix()
        for path in TESTS.glob("test_*.py")
        if re.search(r"^\s*@given\(", path.read_text(encoding="utf-8"), re.MULTILINE)
    }
    assert drawn, "no test file draws a property"
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    (deep,) = (
        str(step["run"])
        for step in workflow["jobs"]["nightly-fuzz"]["steps"]
        if step.get("env", {}).get("HYPOTHESIS_PROFILE") == "nightly"
    )
    named = set(re.findall(r"tests/test_\w+\.py", deep))
    missing = sorted(drawn - named)
    assert not missing, (
        f"property files the nightly profile never draws: {missing}. Name each "
        "in the nightly lane's deep suite."
    )
