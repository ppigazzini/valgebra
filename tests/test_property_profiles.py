"""The property profiles draw what their names say, on a runner as on a laptop.

A profile is read in the environment it runs in, and a runner sets ``CI``, which
makes Hypothesis's own ``ci`` settings the default a registered profile inherits:
derandomized, no database. A deep profile that inherits them asks the same
examples every night. So the profiles are read here in a child interpreter with
``CI`` set, where the inheritance happens, rather than in this one, where it may
not.

A property can still pin its own budget, and a pinned one draws the same count
under every profile. So a budget written beside a property is read here too,
and must be a fraction of the profile's rather than a number.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import settings
from hypothesis.database import DirectoryBasedExampleDatabase

from _tree import ROOT, read_workflow

# A repository check: it reads the suite's own `conftest.py` and the workflow,
# neither of which ships in a wheel.
pytestmark = pytest.mark.repository

TESTS = ROOT / "tests"

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
    """The ``ci`` profile is derandomized on purpose; ``dev`` keeps the default.

    ``dev`` is asked too, because it is what runs when a caller sets no profile,
    and on a runner it inherits Hypothesis's own derandomized ``ci`` settings.
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
    workflow = read_workflow()
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

    The ``ci`` profile is derandomized: its draw is a function of the test's
    source, so every push asks the same examples until the test is edited. Only
    the nightly lane draws at random, and only in the files its deep suite
    names. A file left off that list draws the same examples on every push and
    none at night, so where it holds the walk to pydantic-core or jsonschema,
    the tree's one independent oracle is asked the same questions every time.
    """
    drawn = {
        path.relative_to(ROOT).as_posix()
        for path in TESTS.glob("test_*.py")
        if re.search(r"^\s*@given\(", path.read_text(encoding="utf-8"), re.MULTILINE)
    }
    assert drawn, "no test file draws a property"
    workflow = read_workflow()
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


def _fixed_budgets(tree: ast.Module) -> list[int]:
    """Give the lines where a `settings(...)` call gives `max_examples` a number.

    A name bound to a number at module scope is a number too: moving the figure
    into a constant keeps it the same under every profile.
    """
    numbers = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    return sorted(
        keyword.value.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).split(".")[-1] == "settings"
        for keyword in node.keywords
        if keyword.arg == "max_examples"
        and (
            isinstance(keyword.value, ast.Constant)
            or (isinstance(keyword.value, ast.Name) and keyword.value.id in numbers)
        )
    )


def test_no_property_pins_its_example_budget_to_a_number() -> None:
    """A budget beside a property moves with the profile, or it hides from it.

    A leaf that pins its own budget draws that many examples under every
    profile, so the deep lane asks it what the merge gate asks, and a reading of
    the profiles above says nothing about that leaf.
    """
    offenders = {
        path.name: found
        for path in sorted(TESTS.glob("test_*.py"))
        if (found := _fixed_budgets(ast.parse(path.read_text(encoding="utf-8"))))
    }
    assert not offenders, (
        f"properties drawing a fixed number of examples: {offenders}. Give the "
        "budget as a fraction of `settings().max_examples`, which is the "
        "profile's."
    )


def test_a_budget_is_read_as_a_number_however_it_is_spelled() -> None:
    # The readings the check turns on: a literal, a module constant and a
    # dotted call are numbers, and a fraction of the profile's is not.
    literal = ast.parse("@settings(max_examples=50)\ndef test_x(): pass\n")
    assert _fixed_budgets(literal) == [1]
    constant = ast.parse("N = 50\n@settings(max_examples=N)\ndef test_x(): pass\n")
    assert _fixed_budgets(constant) == [2]
    dotted = ast.parse("@hypothesis.settings(max_examples=5)\ndef test_x(): pass\n")
    assert _fixed_budgets(dotted) == [1]
    scaled = ast.parse(
        "N = max(1, settings().max_examples // 10)\n"
        "@settings(max_examples=N)\ndef test_x(): pass\n"
    )
    assert _fixed_budgets(scaled) == []
