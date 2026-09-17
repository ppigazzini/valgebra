"""A cache under coinduction is revertible, or it is not there.

The subtyping decision is coinductive: it assumes a goal, unfolds, and reads a
pair already on the trail as a local success. Two groups state the hazard that
comes with it independently. Castagna's record paper: "if one of the 'or'
clauses fails, then this **invalidates all the hypotheses we added to check
it**". Castagna & Duboc, for arrows: "a naive implementation may have to
backtrack and, therefore, **unroll all the memoized solutions found in the
current run**".

So a table over goals may not be an ordinary map. It has to be persistent, so a
failed disjunct can roll it back, or hold only results that rested on no open
hypothesis -- and a memo that is neither turns an assumption the procedure
abandoned into a cached falsehood, which is a wrong `True` from a relation whose
whole contract is that a `True` is a proof.

There is no memo in the tree, and the obligation is vacuous until one arrives.
That is exactly when a reader is least likely to remember it: the decision not
to memoise rests on a measured count (`decision/goal_tests.rs`), and a later
reader who moves that count adds the table without meeting the condition.

This holds the hazard in the tree rather than in memory. A map keyed by a pair
of schemas inside the coinductive procedure fails here until the sentence that
discharges §14.3 is written beside it -- not as a promise that the sentence is
true, which nothing mechanical can check, but so that writing one is a
deliberate act a reviewer sees rather than an omission nobody does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
#: The coinductive procedure and the rules it is split into. The descriptor is
#: not under this rule: it holds no hypothesis, so a cache there reverts nothing.
PROCEDURE = (
    ROOT / "crates" / "valgebra-core" / "src" / "decision.rs",
    *(ROOT / "crates" / "valgebra-core" / "src" / "decision").glob("*.rs"),
)

#: A map declaration, however it is spelled. The key is what makes it a memo:
#: a pair of schemas is a subtyping goal, which is the thing the trail assumes.
_MAP = re.compile(
    # The key is a tuple -- `(Schema, Schema)`, which is a goal -- or a single
    # type that may carry its own arguments. Taken as one group either way, so
    # the comma inside a tuple does not end it.
    r"\b(?:Fx)?(?:Hash|BTree)Map\s*<\s*(?P<key>\([^)]*\)|[^,<>]+(?:<[^>]*>)?)\s*,",
    re.MULTILINE,
)

#: What discharges the obligation, in the comment above the declaration. A
#: persistent table, a rollback, or a restriction to closed results.
_DISCHARGED = re.compile(
    r"\brevert|\brevertible|\broll(?:ed|s)?\s+back|\brollback"
    r"|\bpersistent\b|\bno\s+open\s+hypothesis\b",
    re.IGNORECASE,
)

#: How much of the text above a declaration counts as "beside it".
_BESIDE = 24


def _is_goal_keyed(key: str) -> bool:
    """Whether a map's key is a subtyping goal: a pair of schemas."""
    return key.count("Schema") >= 2 or bool(re.search(r"Goal|SubtypeKey", key))


def _without_test_items(text: str) -> str:
    """Blank every `#[cfg(test)]` item, keeping the offsets of what is left.

    A table keyed by a goal is not a memo when a test owns it: the goal counter
    that holds §14.2's number is exactly that shape, and it answers no query.
    Blanked rather than removed so the lines above a real declaration are still
    the lines above it.
    """
    out = list(text)
    for marker in re.finditer(r"#\[cfg\(test\)\]", text):
        opening = text.find("{", marker.end())
        semicolon = text.find(";", marker.end())
        # `#[cfg(test)] mod tests;` declares a sibling file and opens no block.
        if opening == -1 or (semicolon != -1 and semicolon < opening):
            continue
        depth, at = 0, opening
        while at < len(text):
            if text[at] == "{":
                depth += 1
            elif text[at] == "}":
                depth -= 1
                if depth == 0:
                    break
            at += 1
        for index in range(marker.start(), min(at + 1, len(text))):
            if out[index] != "\n":
                out[index] = " "
    return "".join(out)


def _memos(text: str) -> list[tuple[str, str]]:
    """Every goal-keyed map in `text`, with the lines above it."""
    text = _without_test_items(text)
    found: list[tuple[str, str]] = []
    for match in _MAP.finditer(text):
        if not _is_goal_keyed(match.group("key")):
            continue
        above = text[: match.start()].splitlines()[-_BESIDE:]
        found.append((match.group(0), "\n".join(above)))
    return found


# THEORY: a-memo-is-revertible-or-absent
def test_the_detector_finds_a_memo_and_reads_the_sentence() -> None:
    """The parse is the whole test, so it is shown to work on both answers.

    A guard that silently matches nothing passes on the tree it was meant to
    watch, which is the failure `tests/test_ledger_plants.py` exists to rule out
    for the ledgers. This one carries its own plant, because what it reads is a
    shape the tree does not have yet.
    """
    undischarged = """
    /// The goals already decided, so a repeat is answered from the table.
    cache: FxHashMap<(Schema, Schema), Relation>,
    """
    assert len(_memos(undischarged)) == 1, "the detector missed a goal-keyed map"
    assert not _DISCHARGED.search(_memos(undischarged)[0][1])

    discharged = """
    /// The goals already decided. Persistent, so a failed disjunct can roll
    /// back the hypotheses it added: see the obligation on the theory page.
    cache: FxHashMap<(Schema, Schema), Relation>,
    """
    assert len(_memos(discharged)) == 1
    assert _DISCHARGED.search(_memos(discharged)[0][1])

    # And a map that is not keyed by a goal is not this obligation's business.
    by_name = """
    /// The fields a record declares, by name.
    fields: FxHashMap<Arc<str>, Schema>,
    """
    assert not _memos(by_name), "a map keyed by a name is not a goal memo"

    # Nor is one a test owns: the goal counter is this shape and answers no
    # query, which is why the scan reads past a `#[cfg(test)]` item.
    counted = """
    #[cfg(test)]
    pub(crate) mod goals {
        thread_local! {
            static ASKED: RefCell<Option<FxHashMap<(Schema, Schema), u32>>> =
                const { RefCell::new(None) };
        }
    }
    """
    assert not _memos(counted), "a test-side counter is not a memo"


# THEORY: a-memo-is-revertible-or-absent
def test_a_memo_without_a_revert_condition_fails_placement() -> None:
    """A table over goals in the coinductive procedure says how it reverts.

    Vacuous today, and that is the point: the sentence is owed by whatever memo
    is ever built, and a vacuous guard is what makes the debt outlive the reader
    who measured it away.
    """
    undischarged: list[str] = []
    for path in PROCEDURE:
        text = path.read_text(encoding="utf-8")
        for declaration, above in _memos(text):
            if not _DISCHARGED.search(above):
                rel = path.relative_to(ROOT)
                undischarged.append(f"{rel}: {declaration.strip()}")

    assert not undischarged, (
        "a map keyed by a subtyping goal, with no sentence above it saying how a "
        "backtracked hypothesis is rolled back:\n" + "\n".join(undischarged) + "\n"
        "A memo under coinduction is persistent, or holds only results that "
        "rested on no open hypothesis. Write which, above the declaration."
    )
