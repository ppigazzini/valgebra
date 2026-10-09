"""The decision page names every question the core asks, and asks no other.

`LeafRelations` is the whole of what `valgebra-core` cannot decide alone, so the
page that documents the decision has to list it. A list maintained by hand drifts
the moment a question is added: two of the trait's ten reached the tree with the
readings that rest on them and never reached the page, and the page went on
saying there were five.

The sweep is the other half. A mutation of an oracle default is excusable only
where no test can tell the default's `None` from the mutant's answer, and that
argument rests on the call sites: a question that grows a reader of the other
answer turns an excused mutant into a decided one, which `leaf_subtype` did
with a refutation and `no_int_between` with a value. So the argument is not
made at all. Every default declines, `the_default_oracle_declines_every_question`
asserts it of each, and a default that answers fails there whatever reads it.

So these are held here: every method is on the page, every name on the page is
a method, every default that declines is asserted to, and the sweep excuses
none.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _toml import load

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "dev" / "02-decision.md"
TRAIT = ROOT / "crates" / "valgebra-core" / "src" / "oracle.rs"
#: The structural rules, where a pair no rule decides is handed to the
#: readings that refute.
RULES = ROOT / "crates" / "valgebra-core" / "src" / "decision.rs"
#: The readings a pair with no rule of its own is given, which is where every
#: refutation the page lists is defined.
READINGS = ROOT / "crates" / "valgebra-core" / "src" / "decision" / "readings.rs"
SWEEP = ROOT / ".cargo" / "mutants.toml"
#: Where every default is asserted to decline.
PIN = ROOT / "crates" / "valgebra-core" / "src" / "oracle" / "tests.rs"
PIN_TEST = "the_default_oracle_declines_every_question"

# This file reads the tree rather than the library: it holds a shipped page to
# the trait and to the sweep's configuration, and exercises no schema.
pytestmark = pytest.mark.repository

# A trait method: `fn name(` at the trait's indentation, default-bodied or not.
_METHOD = re.compile(r"^    fn ([a-z_][a-z0-9_]*)\s*\(", re.MULTILINE)
# A page row: a list item opening with the method name in backticks, em-dashed.
_ROW = re.compile(r"^- `([a-z_][a-z0-9_]*)` [-—]", re.MULTILINE)
# A trait method whose default body is `None`: a signature, which holds no
# brace and no semicolon, then a body of that one word. The semicolon is what
# stops a required method -- `leaf_subtype`, which has no body -- from running
# on into the next method's and answering for it.
_DECLINING = re.compile(
    r"^    fn ([a-z_][a-z0-9_]*)\s*\([^{;]*\{\s*None\s*\}", re.MULTILINE
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


def _declining_defaults() -> set[str]:
    return set(_DECLINING.findall(_trait_source()))


def _pinned() -> set[str]:
    """Read the questions the pin test asks the default oracle."""
    source = PIN.read_text(encoding="utf-8")
    start = source.index(f"fn {PIN_TEST}()")
    body = source[start : source.index("\n}\n", start)]
    return set(re.findall(r"\boracle\.([a-z_][a-z0-9_]*)\(", body))


def test_every_declining_default_is_held_to_its_decline() -> None:
    """A default that declines is asserted to, so one that answers fails.

    That is what lets the sweep take every default: a mutation replacing one
    with an answer fails the pin whichever answer it is and whatever reads it.
    A question added with a declining default and no row in the pin is killed
    only where a decision happens to read the answer the mutant gives, which
    is the argument the sweep stopped resting on.
    """
    declining = _declining_defaults()
    assert len(declining) >= 8, f"the trait scan found only {sorted(declining)}"
    pinned = _pinned()
    assert pinned, f"`{PIN_TEST}` asks no question; the scan read nothing"
    missing = sorted(declining - pinned)
    assert not missing, f"`{PIN_TEST}` does not assert that {missing} decline"


def test_the_sweep_excuses_no_oracle_default() -> None:
    exclusions = load(SWEEP)["exclude_re"]
    assert exclusions, "the sweep's exclusions did not parse"
    excused = sorted(
        {
            question
            for entry in exclusions
            for question in re.findall(r"LeafRelations::([a-z_]+)", entry)
        }
    )
    assert not excused, (
        f"the sweep excuses the default of {excused}; every default declines and "
        f"`{PIN_TEST}` kills one that answers, so none is excused"
    )


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
    source = READINGS.read_text(encoding="utf-8")
    named = _readings()
    assert len(named) >= 5, f"the refutation section lists only {named}"
    missing = [name for name in named if f"fn {name}(" not in source]
    assert not missing, (
        f"{PAGE.name} names {missing}, which `{READINGS.name}` does not define"
    )


def test_every_rule_that_refutes_is_on_the_page() -> None:
    """The other direction: a reading that returns `Relation::Fails` is listed.

    Read off `unstructured`, which is where a pair with no rule of its own is
    handed to the readings that refute -- so its body is the list, and the page
    is held to it rather than to a count someone maintains.
    """
    source = RULES.read_text(encoding="utf-8")
    body = source[source.index("    fn unstructured(") :]
    body = body[: body.index("\n    }\n")]
    asked = set(re.findall(r"self\.([a-z_]+)\([^)]*\)\s*\{", body))
    missing = sorted(asked - set(_readings()))
    assert not missing, (
        f"`unstructured` asks {missing}, and the page does not name them"
    )
