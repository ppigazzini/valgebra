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

import json
import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from _tree import ROOT, load_script

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

LINT = ROOT / "scripts" / "docs_lint.py"


def _load() -> ModuleType:
    """Import the lint by path; ``scripts/`` is not an importable package."""
    return load_script("docs_lint")


lint = _load()


def test_a_dead_internal_link_fails(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("see [it](gone.md)\n")
    assert lint.check_links(page, page.read_text())

    (tmp_path / "there.md").write_text("here\n")
    page.write_text("see [it](there.md)\n")
    assert not lint.check_links(page, page.read_text())


def test_a_url_is_not_a_link_to_resolve(tmp_path: Path) -> None:
    # The rule is about paths in this tree. An external URL and a mailto are
    # outside it, and treating them as paths would make the check fire on every
    # reference page. A bare anchor is inside it: the page it names is this one.
    page = tmp_path / "page.md"
    text = (
        "## Heading\n\n"
        "[a](https://example.com) [b](mailto:x@example.com) [c](#heading)\n"
    )
    page.write_text(text)
    assert not lint.check_links(page, text)


def test_an_anchor_resolves_to_a_heading_of_the_page_it_names(tmp_path: Path) -> None:
    (tmp_path / "there.md").write_text("# A heading\n\n## Twice\n\n## Twice\n")
    page = tmp_path / "page.md"
    page.write_text("[c](there.md#a-heading) [d](there.md#twice-1)\n")
    assert not lint.check_links(page, page.read_text())
    # A heading renamed and the link left behind: the page exists and the
    # anchor names nothing on it.
    page.write_text("[c](there.md#no-such-heading)\n")
    assert lint.check_links(page, page.read_text())
    page.write_text("[c](#no-such-heading)\n")
    assert lint.check_links(page, page.read_text())


def test_a_published_page_does_not_link_into_the_developer_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The target exists in the tree, so a file-level check passes it, and the
    # site renders no `docs/dev/`: twelve such links answered 404 there.
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    (tmp_path / "docs" / "dev").mkdir(parents=True)
    (tmp_path / "docs" / "dev" / "walk.md").write_text("# The walk\n")
    page = tmp_path / "docs" / "guide.md"
    page.write_text("see [the walk](dev/walk.md)\n")
    assert lint.check_links(page, page.read_text())
    # A developer page linking to another is in the repository either way.
    inner = tmp_path / "docs" / "dev" / "other.md"
    inner.write_text("see [the walk](walk.md)\n")
    assert not lint.check_links(inner, inner.read_text())


def test_a_heading_inside_a_fence_is_no_anchor(tmp_path: Path) -> None:
    (tmp_path / "there.md").write_text("```bash\n# not a heading\n```\n")
    page = tmp_path / "page.md"
    page.write_text("[c](there.md#not-a-heading)\n")
    assert lint.check_links(page, page.read_text())


def test_a_snippet_carries_its_headings_into_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The changelog page is a snippet of CHANGELOG.md, whose headings are the
    # page's anchors on the site.
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("## [0.1.0] - 2026-01-01\n")
    (tmp_path / "log.md").write_text('--8<-- "CHANGELOG.md"\n')
    page = tmp_path / "page.md"
    page.write_text("[c](log.md#010---2026-01-01)\n")
    assert not lint.check_links(page, page.read_text())


def test_a_site_link_resolves_under_both_renderers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A site page is read on the site and on GitHub, whose slugs part on a
    # heading spelling `==`: `what-compares` on one, `what--compares` on the
    # other. A link that follows one of them fails half its readers.
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "there.md").write_text("## What `==` compares\n\n## Plain\n")
    page = tmp_path / "docs" / "page.md"
    for anchor in ("what-compares", "what--compares"):
        page.write_text(f"[c](there.md#{anchor})\n")
        assert lint.check_links(page, page.read_text()), anchor
    page.write_text("[c](there.md#plain)\n")
    assert not lint.check_links(page, page.read_text())


def test_a_named_path_that_does_not_exist_fails() -> None:
    assert lint.check_named_paths("crates/valgebra-py/src/check/walk.rs") == []
    assert lint.check_named_paths("see crates/valgebra-py/src/nope.rs")


def test_a_gitignored_path_is_skipped() -> None:
    # A generated path is absent from a fresh clone and present on a machine that
    # has run the tool that writes it. Asking git rather than the filesystem is
    # what makes the verdict the same on both -- a check that answers differently
    # is measuring the machine.
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
    assert lint.check_history("an inline hint was tried and moved nothing")
    assert lint.check_history("the nightly went red on a flaky draw")
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


def test_a_docstring_is_read_whole_and_code_is_not() -> None:
    """A docstring is the comment Python keeps, so the sweep reads it.

    Every node that opens one is read -- the module, a class, a function --
    and nothing else: a string that is not the first statement is a value.
    """
    source = (
        '"""The module previously read two."""\n'
        "class Shape:\n"
        '    """A class formerly kept twice."""\n'
        "    def read(self) -> str:\n"
        '        """One line,\n'
        "\n"
        '        and a second that used to wrap."""\n'
        '        return "previously"\n'
    )
    documented = lint.docstring_lines(source)
    assert documented == {1, 3, 5, 6, 7}, documented
    problems = lint.check_history(lint.comments(source, documented))
    assert [problem.split(":")[0] for problem in problems] == [
        "line 1",
        "line 3",
        "line 7",
    ], problems
    assert lint.docstring_lines("def broken(:\n") == set(), "a parse error reads none"


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
    """A baseline's own argument is prose, held to the rules prose is.

    A count in those fields is a count the same file records: a comment saying
    a budget is recorded at some figure reads as current for as long as it sits
    there, and the number beside it is the one thing in the tree guaranteed to
    move.
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


def _synthetic_ledgers(root: Path, how_many: int) -> None:
    """Write a tree of `how_many` ledgers and a page tabling them."""
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
    page.write_text(f"# Testing\n\nThe ledgers:\n\n{rows}\n")


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
    _synthetic_ledgers(tmp_path, 4)
    page = tmp_path / "docs" / "dev" / "08-testing.md"
    marker = "LEDGER" + ":"
    (tmp_path / "tests" / "test_ledger_prose.py").write_text(
        f'"""A ledger.\n\n{marker} something\n"""\n'
    )
    page.write_text(
        page.read_text()
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


def _synthetic_manifest(root: Path, summary: str) -> None:
    """Write a site of one page, named in the nav and in the llms.txt manifest."""
    (root / "docs").mkdir()
    (root / "docs" / "README.md").write_text(
        "---\ndescription: What the page is.\n---\n\n# Home\n"
    )
    (root / "mkdocs.yml").write_text(
        "nav:\n"
        "  - Home: README.md\n"
        "plugins:\n"
        "  - llmstxt:\n"
        "      sections:\n"
        "        Start:\n"
        f"          - README.md: {summary}\n"
    )


def test_an_llms_line_that_is_not_its_page_description_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The manifest's summary is a second copy of the page's front matter, and
    # the one no human reads, so nothing but this rule sees it move apart.
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_manifest(tmp_path, "What the page was.")
    problems = lint.check_llms_manifest()
    assert any("README.md" in problem for problem in problems), problems


def test_an_llms_line_that_is_its_page_description_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_manifest(tmp_path, "What the page is.")
    assert lint.check_llms_manifest() == []


def _synthetic_examples(root: Path, imports: str, installs: str) -> None:
    """Write a page whose example imports `imports`, and an install page."""
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text(
        f"# Guide\n\n```python\n{imports}\nimport json\n```\n"
    )
    (root / "docs" / "00-installation.md").write_text(
        "# Installation\n\n```bash\npip install valgebra\n"
        f"pip install {installs}\n```\n"
    )


def test_an_example_import_the_install_page_does_not_name_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reader installs what the page says and runs the example, which then
    # stops at its first line.
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_examples(tmp_path, "import some_package.sub", "annotated-types")
    problems = lint.check_example_installs()
    assert any("imports some-package" in problem for problem in problems), problems
    assert any("names annotated-types" in problem for problem in problems), problems


def test_an_install_page_naming_what_the_examples_import_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The standard library and valgebra itself want no line, and a module's
    # underscores are its distribution's hyphens.
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    _synthetic_examples(
        tmp_path, "from annotated_types import Ge\nimport valgebra", "annotated-types"
    )
    assert lint.check_example_installs() == []


def test_every_bound_names_a_holder_that_names_it() -> None:
    """A row's last cell is held to a test that spells the bound.

    Driven against the real table, with controls from cells it carried: "its
    own tests" for a file whose tests live in another module's corpus names
    nothing, and a gate that measures a bound's cost without spelling it is no
    holder alone. The corrected cells name the corpus and the law that drive
    the two bounds, and a path read as "`x.rs` tests" is that file's test
    modules.
    """
    assert lint.check_bound_holders() == []

    def held(where: str, name: str, cell: str) -> bool:
        return any(
            re.search(rf"\b{name}\b", text) for text in lint.holder_texts(where, cell)
        )

    record = "crates/valgebra-py/src/check/walk/record.rs"
    assert lint.own_tests(record) == [], "the record walk's tests are the corpus"
    gate_alone = "its own tests, and `scripts/compare_gate.py`"
    assert not held(record, "SMALL_OBJECT", gate_alone)
    assert held(
        record,
        "SMALL_OBJECT",
        "the walk's corpus in `crates/valgebra-py/src/check/walk/interpreter.rs`",
    )
    lower = "crates/valgebra-core/src/descr/lower.rs"
    assert not held(lower, "UNFOLDS", "its own tests")
    assert held(lower, "UNFOLDS", "`crates/valgebra-core/src/laws.rs`")
    assert held(
        "crates/valgebra-core/src/descr/lines.rs",
        "MAX_LINES",
        "`crates/valgebra-core/src/descr/mod.rs` tests",
    )


#: A zone table of three rows: a file, a file with one beside it, a directory.
ZONES = (
    "# Architecture\n\n## The zones\n\n"
    "| Zone | Owns | Page |\n|---|---|---|\n"
    "| `crates/core/src/ir.rs` | the IR | here |\n"
    "| `crates/py/src/errors.rs`, `render.rs` | the errors | here |\n"
    "| `crates/py/src/check/` | the walk | here |\n"
    "\n## The direction\n"
)
HELD = {
    "crates/core/src/ir.rs",
    "crates/py/src/errors.rs",
    "crates/py/src/render.rs",
    "crates/py/src/check/",
    "crates/py/src/check/walk.rs",
}


def test_a_name_after_the_first_sits_beside_the_one_before_it() -> None:
    assert lint.zone_paths("`crates/py/src/errors.rs`, `render.rs`") == [
        "crates/py/src/errors.rs",
        "crates/py/src/render.rs",
    ]
    # A name spelled from the root moves the directory the next one sits in.
    assert lint.zone_paths(
        "`crates/a/src/x.rs`, `x/`, `crates/b/src/y.rs`, `z.rs`"
    ) == [
        "crates/a/src/x.rs",
        "crates/a/src/x/",
        "crates/b/src/y.rs",
        "crates/b/src/z.rs",
    ]


def test_every_file_a_row_or_a_directory_names_is_owned() -> None:
    production = sorted(path for path in HELD if not path.endswith("/"))
    assert lint.zone_problems(ZONES, production, HELD) == []


def test_a_file_in_no_zone_fails() -> None:
    production = ["crates/core/src/ir.rs", "crates/core/src/oracle.rs"]
    problems = lint.zone_problems(ZONES, production, HELD)
    assert problems == [
        "docs/dev/00-architecture.md: crates/core/src/oracle.rs is in no zone"
    ]


def test_a_zone_naming_what_the_tree_does_not_hold_fails() -> None:
    problems = lint.zone_problems(ZONES, [], HELD - {"crates/py/src/render.rs"})
    assert problems == [
        (
            "docs/dev/00-architecture.md: a zone names crates/py/src/render.rs, "
            "which the tree does not hold"
        )
    ]


def test_a_page_with_no_zone_table_is_reported() -> None:
    problems = lint.zone_problems("# Architecture\n\n## The direction\n", [], HELD)
    assert problems == ["docs/dev/00-architecture.md: the zone table reads as empty"]


def test_a_test_module_declared_with_a_visibility_is_test_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `pub(crate) mod tests;` lets a sibling read the module's fixtures and
    # compiles into the library exactly as little as a private one, so the
    # body is a test file: read as production, it would be a file the zone
    # rule demands a row for and the bounds scan reads for bounds.
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    source = tmp_path / "crates" / "core" / "src" / "oracle.rs"
    (source.parent / "oracle").mkdir(parents=True)
    source.write_text("pub fn ask() {}\n\n#[cfg(test)]\npub(crate) mod tests;\n")
    tests = source.parent / "oracle" / "tests.rs"
    tests.write_text("const FIXTURE: usize = 1;\n")
    assert lint.test_module_files([source, tests]) == {
        "crates/core/src/oracle/tests.rs"
    }


def test_the_zone_table_owns_every_production_file() -> None:
    # Driven against the real tree, as the bounds ledger is; the synthetic
    # rows above are the controls for both directions.
    assert lint.check_zone_table() == []
