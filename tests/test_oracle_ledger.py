"""The decision page names every question the core asks, and asks no other.

`LeafRelations` is the whole of what `valgebra-core` cannot decide alone, so the
page that documents the decision has to list it. A list maintained by hand drifts
the moment a question is added: two of the trait's ten reached the tree with the
readings that rest on them and never reached the page, and the page went on
saying there were five.

The other half of the drift is worse than an omission. The page claimed a
mutation of one oracle default "cannot be killed by any test", which is true only
where *every* call site collapses the declined answer into the one the mutant
returns -- a property of the call sites, not of the default. `leaf_subtype` grew
a site that reads `Some(false)` as a refutation and left that class; the page did
not notice, and `.cargo/mutants.toml` had already recorded the mutant as killed.
A page and the gate that holds it said opposite things.

So both directions are held here: every method is on the page, every name on the
page is a method, and the page's table of unkillable defaults is the sweep's
exclusion list.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "dev" / "02-decision.md"
TRAIT = ROOT / "crates" / "valgebra-core" / "src" / "decision.rs"
SWEEP = ROOT / ".cargo" / "mutants.toml"

# This file reads the tree rather than the library: it holds a shipped page to
# the trait and to the sweep's configuration, and exercises no schema.
pytestmark = pytest.mark.repository

# A trait method: `fn name(` at the trait's indentation, default-bodied or not.
_METHOD = re.compile(r"^    fn ([a-z_][a-z0-9_]*)\s*\(", re.MULTILINE)
# A page row: a list item opening with the method name in backticks, em-dashed.
_ROW = re.compile(r"^- `([a-z_][a-z0-9_]*)` [-—]", re.MULTILINE)
# The unkillable table's first column, which may name more than one question.
_TABLE_ROW = re.compile(
    r"^\| (`[a-z_, `]+`) \| `([^`]+)` \| `([^`]+)` \|", re.MULTILINE
)


def _trait_source() -> str:
    """Read the `LeafRelations` trait body, and nothing after it."""
    source = TRAIT.read_text(encoding="utf-8")
    start = source.index("pub trait LeafRelations")
    # The trait ends at the first line that closes a block at column zero.
    end = source.index("\n}\n", start)
    return source[start:end]


def _methods() -> set[str]:
    return set(_METHOD.findall(_trait_source()))


def _rows() -> list[str]:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index("## What the core cannot decide alone")
    return _ROW.findall(page[start : page.index("\n## ", start + 1)])


def test_the_trait_is_read_at_all() -> None:
    # A parser that matches nothing makes every assertion below vacuous, which
    # is the failure mode a ledger has: it passes loudest when it is broken.
    methods = _methods()
    assert len(methods) >= 8, f"the trait scan found only {sorted(methods)}"
    assert "leaf_subtype" in methods
    assert len(_rows()) >= 8, "the page scan found no oracle list"


def test_every_question_the_core_asks_is_on_the_page() -> None:
    missing = sorted(_methods() - set(_rows()))
    assert not missing, f"{PAGE.name} does not name {missing}"


def test_every_question_the_page_names_is_one_the_core_asks() -> None:
    # The direction that catches a rename: the page keeps the old name and reads
    # as complete, while the question it describes no longer exists.
    unknown = sorted(set(_rows()) - _methods())
    assert not unknown, f"{PAGE.name} names {unknown}, which `LeafRelations` has not"


def test_the_page_lists_each_question_once() -> None:
    rows = _rows()
    assert len(rows) == len(set(rows)), f"duplicated rows: {sorted(rows)}"


def test_an_unkillable_default_is_one_the_sweep_excludes() -> None:
    """The page's table of unkillable defaults is the sweep's exclusion list.

    A default is unkillable only where every call site reads the declined answer
    and the mutant's answer alike. The page states which three questions are in
    that class and which answer each; `.cargo/mutants.toml` excludes exactly
    those mutants. Either file drifting from the other is the failure this test
    exists for, and it has happened once.
    """
    page = PAGE.read_text(encoding="utf-8")
    sweep = SWEEP.read_text(encoding="utf-8")
    rows = _TABLE_ROW.findall(page)
    assert rows, "the page's table of unkillable defaults did not parse"

    claimed: set[tuple[str, str]] = set()
    for questions, _reading, answer in rows:
        for question in re.findall(r"`([a-z_]+)`", questions):
            claimed.add((question, answer))

    assert claimed, "the table named no question"
    for question, answer in sorted(claimed):
        assert question in _methods(), (
            f"the table names {question}, which is not a question"
        )
        # The exclusion spells the mutant `cargo mutants --list` offers, with
        # the parentheses escaped twice: once for the regex the sweep reads the
        # entry as, and once more by TOML's own string escaping.
        escaped = answer.replace("(", r"\\(").replace(")", r"\\)")
        wanted = f"replace LeafRelations::{question} -> Option<bool> with {escaped}"
        assert wanted in sweep, (
            f"the page calls {question}'s {answer} default unkillable, "
            f"and the sweep does not exclude it"
        )


def test_a_question_the_sweep_excuses_is_one_the_page_explains() -> None:
    sweep = SWEEP.read_text(encoding="utf-8")
    excluded = re.findall(
        r"replace LeafRelations::([a-z_]+) -> Option<bool> with (\w+)", sweep
    )
    assert excluded, "no oracle default is excluded; this test has nothing to hold"
    page = PAGE.read_text(encoding="utf-8")
    table = page[
        page.index("| question |") : page.index("\n\n", page.index("| question |"))
    ]
    for question, _answer in excluded:
        assert f"`{question}`" in table, (
            f"the sweep excuses {question}'s default; the page's table does not name it"
        )


@pytest.mark.parametrize("reading", ["leaf_subtype"])
def test_a_question_that_left_the_unkillable_class_is_swept(reading: str) -> None:
    # `leaf_subtype` was in the table until a call site read `Some(false)` as a
    # refutation. The page says so in prose; this holds the consequence.
    sweep = SWEEP.read_text(encoding="utf-8")
    assert f"replace LeafRelations::{reading} -> Option<bool>" not in sweep, (
        f"{reading}'s default is excluded again; the page says its mutant dies"
    )
    page = PAGE.read_text(encoding="utf-8")
    table = page[
        page.index("| question |") : page.index("\n\n", page.index("| question |"))
    ]
    assert f"`{reading}`" not in table, f"{reading} is back in the unkillable table"


# A reading named on the page as one that refutes, in backticks at a list item.
_READING = re.compile(r"^- `([a-z_][a-z0-9_]*)` [-—]", re.MULTILINE)


def _readings() -> list[str]:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index("## What a refutation stands on")
    return _READING.findall(page[start : page.index("\n## ", start + 1)])


def test_every_reading_the_page_names_exists() -> None:
    """A rule named on the page is a function in the tree.

    The direction a rename breaks: the page goes on describing a reading by a
    name nothing answers to, and reads as current because every sentence around
    it is still true. Six readings reached `decision.rs` in one day without the
    page; this is what keeps the page from drifting the other way afterwards.
    """
    source = TRAIT.read_text(encoding="utf-8")
    named = _readings()
    assert len(named) >= 5, f"the refutation section lists only {named}"
    missing = [name for name in named if f"fn {name}(" not in source]
    assert not missing, (
        f"{PAGE.name} names {missing}, which `decision.rs` does not define"
    )


def test_every_rule_that_refutes_is_on_the_page() -> None:
    """The other direction: a reading that returns `Relation::Fails` is listed.

    Read off `unstructured`, which is where a pair with no rule of its own is
    handed to the readings that refute -- so its body is the list, and the page
    is held to it rather than to a count someone maintains.
    """
    source = TRAIT.read_text(encoding="utf-8")
    body = source[source.index("    fn unstructured(") :]
    body = body[: body.index("\n    }\n")]
    asked = set(re.findall(r"self\.([a-z_]+)\([^)]*\)\s*\{", body))
    missing = sorted(asked - set(_readings()))
    assert not missing, (
        f"`unstructured` asks {missing}, and the page does not name them"
    )
