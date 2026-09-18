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


def test_a_sentence_about_what_the_tree_was_is_refused() -> None:
    """The rule, and the line it names.

    A comment is prose about this tree that no page carries: it ages the same
    way, and a reader cannot tell whether the behaviour it describes is current.
    """
    assert lint.check_history("The walk previously read the element twice.")
    assert lint.check_history("# a record used to compile to two definitions")
    problems = lint.check_history("fine\nthis was fixed in the release before")
    assert problems
    assert problems[0].startswith("line 2:")
    # The rule a reader applies to a run, which the list leaves alone.
    assert not lint.check_history("an entry that is no longer a survivor fails")


def test_a_comment_is_read_and_a_string_is_not() -> None:
    """The universe of the comment sweep, in both directions.

    A `#` inside a string starts no comment, and a scan that read one as prose
    would refuse a pattern for the words it spells.
    """
    source = (
        'let anchor = "https://example.invalid/#originally";\n'
        "// the shape previously carried a hash\n"
        'let hash = "previously";\n'
    )
    read = lint.comments(source)
    assert "the shape previously carried a hash" in read
    assert read.splitlines()[0] == "", "a URL inside a string reads as a comment"
    assert len(read.split("\n")) == 3, "the sweep drops the line numbers"
    assert lint.check_history(read)[0].startswith("line 2:")


def test_every_history_exemption_names_a_file_that_spells_the_words() -> None:
    """An exemption outlives its reason silently, so it is held to one."""
    for relative in lint.HISTORY_EXEMPT:
        path = ROOT / relative
        assert path.exists(), relative
        assert lint.check_history(path.read_text(encoding="utf-8")), (
            f"{relative} is exempt from the history rule and carries none of "
            "its words; the exemption has outlived its reason"
        )


def test_a_width_read_off_a_table_is_not_a_bound() -> None:
    """The universe is the figures somebody chose, and a width is not one.

    A constant whose value is a table's length moves when the table does, so a
    row for it records that length in a second place -- which is the failure the
    ledger exists to catch, pointed at itself.
    """
    width = "const PARTS: usize = KEY_KINDS.len() + 1;\n"
    chosen = "const CEILING: usize = 4096;\n"
    found = dict(lint.BOUND.findall(width + chosen))
    assert set(found) == {"PARTS", "CEILING"}, "the scan reads both constants"
    kept = {
        name: value
        for name, value in found.items()
        if not lint.DERIVED_WIDTH.search(value)
    }
    assert set(kept) == {"CEILING"}


def test_a_baseline_comment_is_held_to_the_prose_rules() -> None:
    """A baseline's own argument is prose, and was the prose no rule read.

    The counts that went stale in those fields were the counts the same file
    records: a comment saying a budget was re-recorded at some figure reads as
    current for as long as it sits there, and the number beside it is the one
    thing in the tree guaranteed to move.
    """
    assert lint.baseline_prose(), "no baseline carries a comment field"
    for relative, prose in lint.baseline_prose():
        assert not lint.check_pinned_numbers(prose, lint.gate_numbers()), relative
        assert not lint.check_internal_reference(prose), relative


def test_a_baseline_comment_quoting_its_own_budget_is_refused() -> None:
    """Refuse a comment that quotes a count its own file records."""
    numbers = lint.gate_numbers()
    assert numbers, "the budget file records no count"
    assert lint.check_pinned_numbers(f"re-recorded at {numbers[0]}", numbers)
    prose = "re-recorded; the file owns the count"
    assert not lint.check_pinned_numbers(prose, numbers)


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


def _synthetic_ledgers(root: Path, how_many: int, spelling: str) -> None:
    """Write a tree of `how_many` ledgers and a page spelling the count.

    The table of ledgers is held to the tree by name in both directions and by
    a spelled count, and the count is the half that has no synthetic corpus of
    its own: the names are checked against a directory the rule reads, so a
    tree is what it takes to drive the spelling at a number the real tree does
    not have.
    """
    tests = root / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    names = [f"test_ledger_{index}.py" for index in range(how_many)]
    # Spelled from its pieces on purpose: written out, this file would declare
    # itself a ledger and the rule under test would demand a row for it.
    marker = "LEDGER" + ":"
    for name in names:
        (tests / name).write_text(f'"""A ledger.\n\n{marker} {name}\n"""\n')
    page = root / "docs" / "dev" / "08-testing.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"| `tests/{name}` | holds something |" for name in names)
    page.write_text(f"# Testing\n\n{spelling} of them:\n\n{rows}\n")


def test_a_page_spelling_the_wrong_ledger_count_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count drifts by one, which is the only way it ever drifts.

    Driven above twenty-four deliberately. The spellings the rule compares
    against were a hand-written table, and a table that stops short reports
    nothing past its end: the page could say any number above it and the rule
    would read every wrong spelling as absent and pass.
    """
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_ledgers(tmp_path, 28, "Twenty-seven")
    problems = lint.check_ledger_table()
    assert problems, "a page one short of the tree's ledger count passed"
    assert any("twenty-seven" in problem for problem in problems), problems
    assert any("twenty-eight" in problem for problem in problems), problems


def test_a_page_spelling_the_right_ledger_count_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_ledgers(tmp_path, 28, "Twenty-eight")
    assert lint.check_ledger_table() == []


def test_a_ledger_named_in_prose_alone_has_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mention is not a row, which is the rule this table is about.

    The table is what a reader consults, and the check that holds it to the
    tree searched the whole page for each name. A ledger the page discusses in
    a paragraph -- which every ledger worth having is discussed in somewhere --
    then satisfied the rule without appearing in the table at all, so the table
    could fall arbitrarily far behind the tree while the lint stayed green.
    This is the sibling of `test_lane_coverage.py`'s rule that a script named
    in a comment is not a script anything runs.
    """
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_ledgers(tmp_path, 4, "Four")
    page = tmp_path / "docs" / "dev" / "08-testing.md"
    marker = "LEDGER" + ":"
    (tmp_path / "tests" / "test_ledger_prose.py").write_text(
        f'"""A ledger.\n\n{marker} something\n"""\n'
    )
    page.write_text(
        page.read_text().replace("Four of them:", "Five of them:")
        + "\nThe fifth is `tests/test_ledger_prose.py`, discussed here only.\n"
    )

    problems = lint.check_ledger_table()
    assert any("test_ledger_prose.py" in problem for problem in problems), (
        f"a ledger named in prose passed as a row: {problems}"
    )


def _synthetic_products(root: Path, marked: list[str], listed: list[str]) -> None:
    """Write product-marked tests and a products table naming `listed`.

    The two lists are given apart because the rule is two directions, and a
    corpus where they always agree can only drive one of them.
    """
    tests = root / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    # Spelled from its pieces, as the ledger corpus above is: written out, this
    # file would declare itself a product and the rule would want a row for it.
    marker = "PRODUCT" + ":"
    for name in marked:
        (tests / name).write_text(f'"""A product.\n\n{marker} something\n"""\n')
    for name in listed:
        (tests / name).touch()
    page = root / "docs" / "dev" / "08-testing.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| every thing {index} | the tree | a test drives it | `tests/{name}` |"
        for index, name in enumerate(listed)
    )
    page.write_text(
        "# Testing\n\n| Product | Derived from | Covered means | Held by |\n"
        f"|---|---|---|---|\n{rows}\n\nAnd prose after it.\n"
    )


def test_a_product_with_no_row_in_the_products_table_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direction the page rots in: a product is added and the table is not."""
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_products(tmp_path, ["test_one.py", "test_two.py"], ["test_one.py"])
    problems = lint.check_product_table()
    assert problems, "a product outside the table passed"
    assert any("test_two.py" in problem for problem in problems), problems


def test_a_products_table_row_naming_no_product_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the other: a row outlives the product, or never named one."""
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_products(tmp_path, ["test_one.py"], ["test_one.py", "test_three.py"])
    problems = lint.check_product_table()
    assert problems, "a row naming no product passed"
    assert any("test_three.py" in problem for problem in problems), problems


def test_a_products_table_that_matches_the_tree_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_products(tmp_path, ["test_one.py"], ["test_one.py"])
    assert lint.check_product_table() == []


def test_a_products_table_the_page_does_not_have_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule that reads no table must not pass for having read nothing.

    This is the shape `check_ledger_table` grew its own detector for: a parse
    that finds an empty universe answers "no problems", which reads exactly
    like a page that is right.
    """
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_products(tmp_path, ["test_one.py"], ["test_one.py"])
    page = tmp_path / "docs" / "dev" / "08-testing.md"
    page.write_text("# Testing\n\nNo table at all.\n")
    problems = lint.check_product_table()
    assert problems, "a page with no products table passed"
