"""The documentation lint must fail on every claim it exists to catch.

A gate that cannot be shown to fail is not evidence. Each of the four rules is
driven against a synthetic tree here -- no repository state is edited -- so the
lint's own behaviour is tested rather than assumed, and the index rule is driven
in both of its directions.

The rules it cannot check are named in `docs/dev/11-writing.md`, not here: a
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


def test_a_placeholder_path_is_skipped() -> None:
    # Without this a page could not write a shape at all.
    assert lint.check_named_paths("crates/<name>/src/lib.rs") == []
    assert lint.check_named_paths("scripts/*.py") == []
    assert lint.check_named_paths("tests/test_....py") == []


def test_naming_the_untracked_surface_fails() -> None:
    assert lint.check_internal_reference("see the notes under __DEV/")
    assert not lint.check_internal_reference("see the untracked working area")


def test_a_budget_number_in_prose_fails() -> None:
    budget = json.loads(
        (ROOT / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
    )
    recorded = int(budget["core_workload_irefs"])
    numbers = lint.gate_numbers()
    assert lint.check_pinned_numbers(f"the budget is {recorded:,}", numbers)
    assert lint.check_pinned_numbers(f"the budget is {recorded}", numbers)
    assert not lint.check_pinned_numbers("the budget is in perf_budget.json", numbers)


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


def test_the_developer_index_is_held_in_both_directions() -> None:
    # Driven against the real set, since the check reads a fixed location. Both
    # directions are exercised by the negative controls recorded in the commit
    # that added them; here the standing state is asserted clean and non-empty.
    assert lint.check_dev_index() == []
    pages = {p.name for p in (ROOT / "docs" / "dev").glob("*.md")}
    assert "README.md" in pages
    assert len(pages) >= 10, f"the developer set holds only {sorted(pages)}"
