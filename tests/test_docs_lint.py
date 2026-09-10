"""The documentation lint must fail on every claim it exists to catch.

A gate that cannot be shown to fail is not evidence. Each of the four rules is
driven against a synthetic tree here -- no repository state is edited -- so the
lint's own behaviour is tested rather than assumed, and the index rule is driven
in both of its directions.

The rules it cannot check are named in `docs/dev/12-writing.md`, not here: a
real symbol attributed to the wrong file, a list with the wrong count, and a
behaviour described as absent from a build that has it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
LINT = ROOT / "scripts" / "docs_lint.py"


def _load() -> ModuleType:
    """Import the lint by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("docs_lint", LINT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lint = _load()


def test_a_dead_internal_link_fails(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("see [it](gone.md)\n")
    assert lint.check_links(page, page.read_text())

    (tmp_path / "there.md").write_text("here\n")
    page.write_text("see [it](there.md)\n")
    assert not lint.check_links(page, page.read_text())


def test_a_url_and_an_anchor_are_not_links_to_resolve(tmp_path: Path) -> None:
    # The rule is about paths in this tree. An external URL, a mailto and a bare
    # anchor are all outside it, and treating them as paths would make the check
    # fire on every reference page.
    page = tmp_path / "page.md"
    text = "[a](https://example.com) [b](mailto:x@example.com) [c](#heading)\n"
    page.write_text(text)
    assert not lint.check_links(page, text)


def test_an_anchor_on_a_real_page_resolves_to_the_page(tmp_path: Path) -> None:
    # The anchor itself is NOT verified -- a link to a heading that no longer
    # exists passes. Pinned so the boundary is on the record.
    (tmp_path / "there.md").write_text("# heading\n")
    page = tmp_path / "page.md"
    page.write_text("[c](there.md#no-such-heading)\n")
    assert not lint.check_links(page, page.read_text())


def test_a_named_path_that_does_not_exist_fails() -> None:
    assert lint.check_named_paths("crates/valgebra-py/src/check/walk.rs") == []
    assert lint.check_named_paths("see crates/valgebra-py/src/nope.rs")


def test_a_gitignored_path_is_skipped() -> None:
    # A generated path is absent from a fresh clone and present on a machine that
    # has run the tool that writes it. Asking git rather than the filesystem is
    # what makes the verdict the same on both -- a check that answers differently
    # is measuring the machine, and this one did, and it broke a lane.
    assert lint.check_named_paths("run it over fuzz/corpus/simplify") == []
    # The exemption is not blanket: a path git does not ignore is still checked.
    assert lint.check_named_paths("see fuzz/src/nope.rs")


def test_the_ignore_question_survives_a_platform_line_ending() -> None:
    # The paths reach git over a pipe. Text mode translates "\n" to the platform
    # line ending on write, so on Windows git received every path with a
    # trailing carriage return and matched none of them -- the check reported
    # paths as missing that `.gitignore` names, and only the Windows lane saw it.
    # Binary mode with NUL separators removes the translation.
    many = ["fuzz/corpus/simplify", "fuzz/corpus/decision", "scripts/docs_lint.py"]
    ignored = lint.ignored_paths(many)
    assert "fuzz/corpus/simplify" in ignored
    assert "fuzz/corpus/decision" in ignored
    assert "scripts/docs_lint.py" not in ignored
    # No result may carry stray whitespace, which is what a translated newline
    # would leave behind.
    assert all(path == path.strip() for path in ignored)


def test_the_ignore_question_is_asked_of_git_not_the_filesystem() -> None:
    # Pinned directly, because the difference is invisible on a machine where the
    # generated directory happens to exist.
    assert "fuzz/corpus/simplify" in lint.ignored_paths(["fuzz/corpus/simplify"])
    assert lint.ignored_paths(["scripts/docs_lint.py"]) == set()
    assert lint.ignored_paths([]) == set()


def test_a_placeholder_path_is_skipped() -> None:
    # Without this a page could not write a shape at all.
    assert lint.check_named_paths("crates/<name>/src/lib.rs") == []
    assert lint.check_named_paths("scripts/*.py") == []
    assert lint.check_named_paths("tests/test_....py") == []


def test_naming_the_untracked_surface_fails() -> None:
    assert lint.check_internal_reference("see the notes under __DEV/")
    assert not lint.check_internal_reference("see the untracked working area")


def test_naming_an_internal_note_or_a_milestone_code_fails() -> None:
    """The half of the internal surface written as prose rather than as a path.

    A page or a message saying "REPORT-31 asked for this" or "what M19 left
    open" points at something no reader outside the working area can open. The
    path check cannot see it -- there is no path -- so the names are held
    directly, and the sentence has to carry its own reason instead.
    """
    for names_one in (
        "REPORT-31 asked for the closed core",
        "the plan in REPORT-35 section 5",
        "what M19 left open",
        "M33.2's exit criterion",
        "recorded in 2-MILESTONES",
        "the workflow in PROMPT.md",
        "see 5-THEORY for the citation",
        "ITERATION-74 records the slice",
    ):
        assert lint.check_internal_reference(names_one), names_one

    for stands_alone in (
        "the decision procedure asked for a closed core",
        "the lowering left the class representation open",
        "the exit criterion was not met",
        "the release checklist records it",
    ):
        assert not lint.check_internal_reference(stands_alone), stands_alone


def test_a_budget_number_in_prose_fails() -> None:
    budget = json.loads(
        (ROOT / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
    )
    recorded = int(budget["core_workload_irefs"])
    numbers = lint.gate_numbers()
    assert lint.check_pinned_numbers(f"the budget is {recorded:,}", numbers)
    assert lint.check_pinned_numbers(f"the budget is {recorded}", numbers)
    assert not lint.check_pinned_numbers("the budget is in perf_budget.json", numbers)


def test_a_comparison_multiplier_that_names_no_interpreter_fails() -> None:
    unqualified = "## Speed\n\nA check is 5x faster than pydantic here.\n"
    assert lint.check_comparison_claims(unqualified)
    qualified = "## Speed\n\nOn CPython 3.14 a check is 5x faster than pydantic here.\n"
    assert not lint.check_comparison_claims(qualified)
    # The interpreter belongs in the section that states the figure: a page that
    # names one three headings earlier leaves the reader of this claim without it.
    elsewhere = (
        "## Method\n\nMeasured on CPython 3.14.\n\n"
        "## Speed\n\nA check is 5x faster than pydantic here.\n"
    )
    assert lint.check_comparison_claims(elsewhere)
    # A multiplier that compares nothing to another checker is prose, not a claim
    # this rule is about, and a fenced example is not prose at all.
    assert not lint.check_comparison_claims("## Scale\n\nA union of 5x members.\n")
    assert not lint.check_comparison_claims("## Speed\n\n```text\n5x pydantic\n```\n")


def test_the_shipped_tree_is_clean() -> None:
    # The gate's own subject. Run as a subprocess so the exit code is the
    # assertion, which is what a lane reads.
    result = subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(LINT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_the_three_exit_codes_are_distinct() -> None:
    assert (lint.EXIT_OK, lint.EXIT_FAIL, lint.EXIT_CANNOT_RUN) == (0, 1, 2)


@pytest.mark.parametrize("relative", ["docs", "docs/dev"])
def test_each_index_is_held_in_both_directions(relative: str) -> None:
    # Driven against the real sets, since the check reads a fixed location. Both
    # directions are exercised by the negative controls recorded in the commit
    # that added them; here the standing state is asserted clean and non-empty.
    assert lint.check_index(relative) == []
    pages = {p.name for p in (ROOT / relative).glob("*.md")}
    assert "README.md" in pages
    assert len(pages) >= 10, f"{relative} holds only {sorted(pages)}"


def test_a_test_gated_item_does_not_hide_the_bounds_below_it() -> None:
    # The scan reads what a file *defines*, so it stops at the file's test
    # module. Stopping at the first `#[cfg(test)]` instead hid every bound
    # under a test-gated re-export or helper -- silently, since a table that
    # reads no constants reports no problems about them.
    hidden = (
        "const ABOVE: usize = 1;\n"
        "#[cfg(test)]\n"
        "pub(crate) fn helper() {}\n"
        "const BELOW: usize = 2;\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    const FIXTURE: usize = 3;\n"
        "}\n"
    )
    read = lint.before_the_test_module(hidden)
    assert "ABOVE" in read
    assert "BELOW" in read, "a test-gated helper hid the bound below it"
    assert "FIXTURE" not in read, "a fixture is not a bound"
    assert {name for name, _ in lint.BOUND.findall(read)} == {"ABOVE", "BELOW"}


def test_the_bounds_ledger_is_held_in_both_directions() -> None:
    # Driven against the real tree, since the check reads a fixed page. The
    # value comparison is the half a name check would miss, so it is the one
    # exercised here with a control: the row is edited on disk in a copy of the
    # page's text, not in the tree.
    assert lint.check_bounds_ledger() == []
    page = (ROOT / "docs" / "dev" / "00-architecture.md").read_text(encoding="utf-8")
    rows = lint.BOUND_ROW.findall(page)
    assert len(rows) >= 20, f"the table lists only {len(rows)} bounds"
    where, name, value = rows[0]
    assert (ROOT / where).exists(), where
    assert f"const {name}" in (ROOT / where).read_text(encoding="utf-8")
    assert value
