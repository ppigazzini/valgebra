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
import subprocess
import sys
from pathlib import Path

import pytest

# A repository check: it reads the suite's own `conftest.py`, which ships in no
# wheel.
pytestmark = pytest.mark.repository

TESTS = Path(__file__).resolve().parent

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
