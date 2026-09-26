"""A published example reads clean under the checkers a reader runs, or says why.

`scripts/run_doc_examples.py` runs every fenced `python` block and reads the
exit code, which is the example's truth and not its reading. A reader copies the
block into a project that runs ty, pyright or ruff, and what those report on it
is the first thing that reader sees: an import a later edit left unused, a
return typed `str` handing back an `object`, an ignore comment written in one
checker's dialect that the other two read as noise. Six of those stood in the
pages while every example ran green.

`scripts/check_doc_examples.py` reads the same blocks with the three checkers
and holds what they report to its ledger of expected rows -- the diagnostics
that *are* the example, each with its reason. This runs it, and holds the ledger
to its own shape:

* every diagnostic is a row and every row is reported, in one run;
* every row names a page the runner reaches, a count, and a reason that is a
  sentence about the example rather than a shrug;
* every rule an example is exempt from says why;
* each checker reported something, so a parser that read nothing cannot pass
  as a clean tree.

LEDGER: every checker diagnostic on a published example is expected with a reason
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections import Counter
    from types import ModuleType

# A repository check: it reads the tree's pages and runs the tree's checkers,
# none of which ships in a wheel. The checkers are the dev group's, and the PyPy
# lane installs only what the product suite reads: there this has nothing to
# run and says so.
for _checker in ("ty", "pyright", "ruff"):
    pytest.importorskip(_checker)

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "scripts" / "check_doc_examples.py"


def checker() -> ModuleType:
    """Import the script, so its ledger is what this reads."""
    spec = importlib.util.spec_from_file_location(
        "valgebra_doc_example_checkers", CHECKER
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def found() -> Counter[tuple[str, str, str]]:
    """Read the blocks once: three subprocesses for the whole module."""
    count, diagnostics = checker().read()
    assert count >= 100, f"the runner yields {count} blocks"
    return diagnostics


def test_every_diagnostic_is_expected_and_every_row_reported(
    found: Counter[tuple[str, str, str]],
) -> None:
    wrong = checker().problems(found)
    assert not wrong, "\n".join(wrong)


@pytest.mark.parametrize("name", ["ty", "pyright", "ruff"])
def test_each_checker_reported_something(
    found: Counter[tuple[str, str, str]], name: str
) -> None:
    """A parser that read nothing would pass the ledger vacuously."""
    assert any(key[0] == name for key in found), sorted(found)


def test_every_row_names_a_reached_page_once_with_a_reason() -> None:
    module = checker()
    reached = {path.relative_to(ROOT).as_posix() for path in module.runner().DOCS}
    keys = [(row.checker, row.page, row.rule) for row in module.EXPECTED]
    assert len(keys) == len(set(keys)), "two rows for one checker, page and rule"
    for row in module.EXPECTED:
        assert row.page in reached, f"{row.page} is a page the runner does not list"
        assert row.count >= 1, row
        assert len(row.reason) > 40, f"{row.page} {row.rule}: {row.reason!r}"


def test_every_exemption_says_why() -> None:
    module = checker()
    for rule, reason in module.EXEMPT.items():
        assert len(reason) > 25, f"{rule}: {reason!r}"


def test_the_release_is_the_one_ty_reads_the_suite_at() -> None:
    """One source for the release, so the two cannot drift apart."""
    release = checker().newest_release()
    assert release.count(".") == 1
    assert f'python-version = "{release}"' in (ROOT / "pyproject.toml").read_text(
        "utf-8"
    )
