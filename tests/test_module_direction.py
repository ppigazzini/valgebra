"""The definition does not import the optimisation.

Two representations decide the three relations, and the shipped pages state an
order between them: `crates/valgebra-core/src/descr/` is the **definition** --
a schema denotes a set, each kind carries a representation closed under the
three operations, and `a <= b` is `a & ~b` admitting no value -- while
`decision.rs` is the **fast path**, a structural procedure asked first only
because building a set costs about two orders of magnitude more than a rule
that already answers.

A dependency graph matching that order has the definition below the
optimisation, importing nothing from it. The tree ran it both ways: every
`descr/` module wrote `use crate::decision::{Kind, Verdict}` for the partition
it is indexed by and the answer it returns, while `decision.rs` imported
`descr::lower` for the fallback it asks when its rules decline.

What travelled the wrong way was never a rule. `Kind` is the partition a
descriptor's components are an array over, `Region` the summary derived from
it, and `Verdict` and `Relation` the two three-valued answers both deciders
give. Each belongs to the frame the pair shares, and none to the half that
happened to define it.

So the frame sits in `kind.rs` and `verdict.rs`, below both, and this holds the
direction. The reverse edge stays allowed -- an optimisation may depend on the
definition it optimises -- which is what makes this a direction rather than a
separation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "crates" / "valgebra-core" / "src"
DESCR = CORE / "descr"

# This file reads the tree rather than the library: it holds a module graph, and
# exercises no schema.
pytestmark = pytest.mark.repository

#: Any path into the structural procedure, as a `use` or a fully-qualified call.
_DECISION = re.compile(r"\bcrate::decision\b")


def _descr_sources() -> list[Path]:
    return sorted(DESCR.glob("*.rs"))


def test_the_descriptor_modules_are_read_at_all() -> None:
    # A scan that finds no files makes every assertion below vacuous, which is
    # the failure mode a ledger has: it passes loudest when it is broken.
    sources = _descr_sources()
    assert len(sources) >= 10, f"the descriptor scan found only {sources}"
    assert (DESCR / "mod.rs") in sources


def test_the_definition_imports_nothing_from_the_optimisation() -> None:
    offenders: list[str] = []
    for source in _descr_sources():
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _DECISION.search(line):
                offenders.append(f"{source.name}:{number}: {line.strip()}")
    assert not offenders, (
        "the descriptor reaches into the structural procedure:\n"
        + "\n".join(offenders)
        + "\nThe partition and the answer types live in `kind.rs` and "
        "`verdict.rs`, below both deciders; import them from there."
    )


def test_the_frame_is_where_the_definition_can_reach_it() -> None:
    # The other direction of the same claim: the modules the descriptor was
    # emptied into exist and hold what it needs. Without this the test above
    # passes by the frame having been deleted.
    kind = (CORE / "kind.rs").read_text(encoding="utf-8")
    verdict = (CORE / "verdict.rs").read_text(encoding="utf-8")
    assert "pub enum Kind" in kind
    assert "struct Region" in kind
    assert "enum Regions" in kind
    assert "pub enum Verdict" in verdict
    assert "pub enum Relation" in verdict
    # And the frame does not reach back into either decider.
    for name, source in (("kind.rs", kind), ("verdict.rs", verdict)):
        assert not _DECISION.search(source), f"{name} imports the optimisation"


def test_the_optimisation_may_still_depend_on_the_definition() -> None:
    """The reverse edge is a direction, not a separation.

    `decision.rs` asks the descriptor where its own rules decline, so it imports
    the lowering. A test that forbade both directions would be asking for two
    unrelated crates rather than for an order between two readings of one
    question.
    """
    decision = (CORE / "decision.rs").read_text(encoding="utf-8")
    assert "use crate::descr::lower::" in decision, (
        "the fast path no longer asks the definition; if that is deliberate, "
        "this test is what says the edge was allowed"
    )
