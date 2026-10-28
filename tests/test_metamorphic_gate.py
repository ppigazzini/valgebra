"""The metamorphic gate must fail on every way a refactor can change meaning.

A gate that cannot be shown to fail is not evidence. The gate's three relations
are driven here from recordings alone -- no built extension, no corpus run -- so
each refusal is exercised on the recording shapes that produce it.

The one worth stating is the third. "Decisions only widen" is satisfied
perfectly by a build that proves every relation, so a widening is not evidence
of anything until a value fails to contradict it. The gate searches the corpus
for that value, and the tests below drive both outcomes: a widening the values
support, and one they refute.

LEDGER: every relation the metamorphic gate holds can be driven to fail
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from _tree import load_script

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    return load_script("metamorphic_gate")


gate = _load_gate()

# A membership recording over two schemas and two values, and the decisions that
# go with it: `narrow` is the singleton `{1}`, `wide` is every int.
MEMBERSHIP = {
    "narrow @ one": "y",
    "narrow @ two": "n",
    "wide @ one": "y",
    "wide @ two": "y",
}
VALUES = {"one": 1, "two": 2}


def test_an_unchanged_recording_holds_every_relation() -> None:
    assert gate.moved_membership(MEMBERSHIP, MEMBERSHIP) == []
    decisions = {"narrow <= wide": "s", "wide <= narrow": "n", "empty wide": "u"}
    assert gate.lost_decisions(decisions, decisions) == []
    assert gate.contradicted_decisions(decisions, decisions) == []
    assert gate.unwitnessed_widenings(decisions, decisions, MEMBERSHIP, VALUES) == []


def test_a_value_that_moved_fails() -> None:
    moved = {**MEMBERSHIP, "narrow @ two": "y"}
    assert gate.moved_membership(MEMBERSHIP, moved) == ["narrow @ two: n -> y"]


def test_a_value_that_moved_the_other_way_fails() -> None:
    # Both directions, because a refactor that starts rejecting is as much a
    # semantic change as one that starts accepting.
    moved = {**MEMBERSHIP, "narrow @ one": "n"}
    assert gate.moved_membership(MEMBERSHIP, moved) == ["narrow @ one: y -> n"]


def test_a_raised_error_is_a_verdict_like_any_other() -> None:
    # The walk answers `recursion_limit` on purpose, so a build that accepts
    # what the reference refuses has moved membership.
    reference = {**MEMBERSHIP, "narrow @ one": "!ValidationError"}
    assert gate.moved_membership(reference, MEMBERSHIP) == [
        "narrow @ one: !ValidationError -> y"
    ]


def test_a_lost_proof_fails() -> None:
    reference = {"narrow <= wide": "s"}
    measured = {"narrow <= wide": "u"}
    assert gate.lost_decisions(reference, measured) == [
        "narrow <= wide: decided by the reference, not by this build"
    ]


def test_a_lost_refutation_fails() -> None:
    # A refutation is an answer a caller acts on, and a boolean reads it and a
    # decline as one `False`: a recording of booleans cannot see it go.
    reference = {"wide <= narrow": "n", "empty narrow": "n"}
    measured = {"wide <= narrow": "u", "empty narrow": "u"}
    assert gate.lost_decisions(reference, measured) == [
        "empty narrow: decided by the reference, not by this build",
        "wide <= narrow: decided by the reference, not by this build",
    ]


def test_a_contradicted_decision_fails_in_both_directions() -> None:
    # Both builds claim soundness, so one of them is wrong whichever way the
    # answer turned.
    reference = {"narrow <= wide": "s", "wide <= narrow": "n"}
    measured = {"narrow <= wide": "n", "wide <= narrow": "s"}
    assert gate.contradicted_decisions(reference, measured) == [
        "narrow <= wide: s on the reference, n on this build",
        "wide <= narrow: n on the reference, s on this build",
    ]


def test_a_widening_is_not_a_lost_decision() -> None:
    # The relation is one-directional by soundness: gaining an answer is what
    # a change to the procedure does on purpose.
    reference = {"wide <= narrow": "u", "narrow <= wide": "u"}
    measured = {"wide <= narrow": "n", "narrow <= wide": "s"}
    assert gate.lost_decisions(reference, measured) == []
    assert gate.contradicted_decisions(reference, measured) == []


def test_a_widening_the_values_support_passes() -> None:
    # `narrow <= wide` is true, so no corpus value is in `narrow` and outside
    # `wide`. The gate records the widening and does not fail on it.
    reference = {"narrow <= wide": "u"}
    measured = {"narrow <= wide": "s"}
    assert gate.unwitnessed_widenings(reference, measured, MEMBERSHIP, VALUES) == []


def test_a_widening_a_value_refutes_fails() -> None:
    # `wide <= narrow` is false, and the corpus has the witness: `two` is in
    # `wide` and not in `narrow`. This is the case that stops a build proving
    # everything from reading as a clean widening.
    reference = {"wide <= narrow": "u"}
    measured = {"wide <= narrow": "s"}
    assert gate.unwitnessed_widenings(reference, measured, MEMBERSHIP, VALUES) == [
        "wide <= narrow: two is in wide and not in narrow"
    ]


def test_a_widening_no_corpus_value_reaches_is_not_refuted() -> None:
    # The honest limit of the search: a corpus with no witness cannot refute a
    # claim, and the gate does not pretend otherwise. This is why the corpus is
    # the gate's real subject -- the search is only as strong as the values in
    # it.
    reference = {"wide <= narrow": "u"}
    measured = {"wide <= narrow": "s"}
    thin = {"wide @ one": "y", "narrow @ one": "y"}
    assert gate.unwitnessed_widenings(reference, measured, thin, {"one": 1}) == []


def test_an_emptiness_proof_a_member_refutes_fails() -> None:
    # An emptiness row has no `A <= B` to parse, and its witness is any member
    # of the schema: `one` is in `narrow`, so `narrow` is not empty.
    reference = {"empty narrow": "u"}
    measured = {"empty narrow": "s"}
    assert gate.unwitnessed_widenings(reference, measured, MEMBERSHIP, VALUES) == [
        "empty narrow: one is in narrow"
    ]


def test_an_emptiness_proof_no_member_refutes_passes() -> None:
    reference = {"empty narrow": "u"}
    measured = {"empty narrow": "s"}
    hollow = {"narrow @ one": "n", "narrow @ two": "n"}
    assert gate.unwitnessed_widenings(reference, measured, hollow, VALUES) == []


def test_the_reference_the_gate_ships_with_describes_a_commit() -> None:
    # A reference recorded outside a checkout carries `unknown`, which makes the
    # comparison unattributable: a reader cannot tell what the tree is being
    # held against.
    import json  # noqa: PLC0415

    reference = json.loads(gate.REFERENCE_FILE.read_text(encoding="utf-8"))
    assert len(reference["commit"]) == 40, "the reference names no commit"
    assert reference["membership"], "the reference carries no membership verdicts"
    assert reference["decisions"], "the reference carries no decisions"


def test_the_corpus_asks_both_answers_of_both_relations() -> None:
    # A corpus every schema accepts, or one no relation holds between, compares
    # two builds on a question with one answer. Both verdicts must appear in
    # both recordings for the comparison to have any resolution at all.
    import json  # noqa: PLC0415

    reference = json.loads(gate.REFERENCE_FILE.read_text(encoding="utf-8"))
    for name, recorded, wanted in (
        ("membership", reference["membership"], {"y", "n"}),
        ("decisions", reference["decisions"], {"s", "n"}),
    ):
        answers = set(recorded.values())
        assert wanted <= answers, f"the {name} corpus only ever answers {answers}"
