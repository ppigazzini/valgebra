"""The definition does not import the optimization.

Two sound deciders answer the three relations, and the shipped pages state an
order between them: `crates/valgebra-core/src/descr/` is the **definition** for
every schema it can build -- a schema denotes a set, each kind carries a
representation closed under the three operations, and `a <= b` is `a & ~b`
admitting no value -- while `decision.rs` is the structural procedure, asked
first because building a set costs about three orders of magnitude more than a
rule that already answers, and alone in reaching a recursive schema, a
predicate or a hooked class.

A dependency graph matching that order has the definition below the
procedure, importing nothing from it: the definition is what the procedure
hands a pair it declines, and built from a rule it would carry that rule's
defect into the answer given where the rule declined.

What both halves name is not a rule. `Kind` is the partition a descriptor's
components are an array over, `Region` the summary derived from it, and
`Verdict` and `Relation` the two three-valued answers both deciders give. Each
belongs to the frame the pair shares, and none to either half.

So the frame sits in `kind.rs`, `verdict.rs` and `oracle.rs`, below both, and
this holds the direction. The reverse edge stays allowed -- an optimization may
depend on the definition it optimizes -- which is what makes this a direction
rather than a separation.

`ir.rs` is under the same rule and for a stronger reason: it is the term, which
both readings are readings *of*. Its constructors apply the lattice laws while a
schema is built -- a join carrying a member together with its complement is the
top -- and the one statement of that law is in `oracle.rs`, which the emptiness
decision reads from there too. Stated inside a decider it would be an edge from
the term to a reading of it.

LEDGER: the definition, the term and the membership walk import nothing from
the structural procedure
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from _tree import ROOT, load_script

if TYPE_CHECKING:
    from pathlib import Path

CORE = ROOT / "crates" / "valgebra-core" / "src"
DESCR = CORE / "descr"

#: The scripts' one reading of which Rust file is test code.
lib = load_script("_lib")

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


# THEORY: the-definition-imports-nothing
def test_the_definition_imports_nothing_from_the_optimization() -> None:
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


# THEORY: the-definition-imports-nothing
def test_the_term_imports_nothing_from_the_optimization() -> None:
    """The constructors apply lattice laws, not decisions.

    A law about the term is a law wherever it is read, so it sits below every
    reader. The edge this refuses is `ir.rs` naming `crate::decision` to fold
    `A | ~A` at construction.
    """
    for name in ("ir.rs", "ir/transform.rs", "ir/intern.rs"):
        source = CORE / name
        assert source.exists(), f"{name} is gone; this rule names a file that moved"
        offenders = [
            f"{name}:{number}: {line.strip()}"
            for number, line in enumerate(
                source.read_text(encoding="utf-8").splitlines(), start=1
            )
            if _DECISION.search(line) and "has_disjoint_pair" not in line
        ]
        assert not offenders, (
            "the term reaches into the structural procedure:\n"
            + "\n".join(offenders)
            + "\nA lattice law both readings apply belongs in `oracle.rs`, below "
            "both; import it from there."
        )


#: The parts of the set representation that hold sets of numbers, automata and
#: unions of summands for any alphabet: the theory, which reads no value model.
_KERNELS = (
    "automaton.rs",
    "budget.rs",
    "floats.rs",
    "integers.rs",
    "interval.rs",
    "polar.rs",
    "symbolic.rs",
    "values.rs",
)
#: What a line of code names when it reads Python's values: the node set, the
#: partition, the host's questions and the class snapshot.
_VALUE_MODEL = re.compile(
    r"\b(?:Schema|Kind|LeafRelations|Constants|Class)\b|\bcrate::(?:ir|kind|oracle)\b"
)


# THEORY: the-kernels-hold-no-value-model
def test_no_kernel_names_a_fact_of_the_value_model() -> None:
    """A kernel is generic because it cannot see what its letters are.

    A universe is a set below the top (`docs/dev/10-theory.md`), so no second
    instance of these modules exists to prove them generic. This is what says
    they are: a fact about Python's values reaching one fails here.
    """
    offenders: list[str] = []
    for name in _KERNELS:
        source = DESCR / name
        assert source.exists(), f"{name} is gone; this rule names a file that moved"
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            code = line.split("//", 1)[0]
            if _VALUE_MODEL.search(code):
                offenders.append(f"{name}:{number}: {line.strip()}")
    assert not offenders, (
        "a kernel of the set representation reads the value model:\n"
        + "\n".join(offenders)
        + "\nA fact about Python's values belongs where a schema is lowered, in "
        "`descr/lower.rs` or the component that holds that kind."
    )


def _names_the_value_model(source: Path) -> bool:
    return any(
        _VALUE_MODEL.search(line.split("//", 1)[0])
        for line in source.read_text(encoding="utf-8").splitlines()
    )


# THEORY: the-kernels-hold-no-value-model
def test_every_module_free_of_the_value_model_is_a_kernel() -> None:
    """The other direction: the list names every module the claim is true of.

    A list that only had to be a subset of the truth could shrink to nothing and
    keep the test above green.
    """
    free = {
        source.name
        for source in _descr_sources()
        if not lib.is_test_module(source) and not _names_the_value_model(source)
    }
    assert free == set(_KERNELS), (
        "the kernels are the descriptor modules that read no value model: "
        f"{sorted(free ^ set(_KERNELS))} stands on one side of that only"
    )


def test_the_frame_is_where_the_definition_can_reach_it() -> None:
    # The other direction of the same claim: the frame modules the descriptor
    # reads exist and hold what it needs. Without this the test above
    # passes by the frame having been deleted.
    kind = (CORE / "kind.rs").read_text(encoding="utf-8")
    verdict = (CORE / "verdict.rs").read_text(encoding="utf-8")
    assert "pub enum Kind" in kind
    assert "struct Region" in kind
    assert "enum Regions" in kind
    assert "pub enum Verdict" in verdict
    assert "pub enum Relation" in verdict
    # The third frame module: what neither reading can decide alone, and the
    # two lattice laws all three apply.
    oracle = (CORE / "oracle.rs").read_text(encoding="utf-8")
    assert "pub trait LeafRelations" in oracle
    assert "fn has_complementary_pair_within" in oracle
    assert "fn denotes_a_set_within" in oracle
    assert not _DECISION.search(oracle), "oracle.rs imports the optimization"
    # And the frame does not reach back into either decider.
    for name, source in (("kind.rs", kind), ("verdict.rs", verdict)):
        assert not _DECISION.search(source), f"{name} imports the optimization"


# THEORY: the-definition-imports-nothing
def test_the_optimization_may_still_depend_on_the_definition() -> None:
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


#: The binding's membership walk: the files that decide whether a value belongs.
WALK = ROOT / "crates" / "valgebra-py" / "src" / "check"

#: A call into the decision procedure, by any of the names it is reached under.
#: `is_empty` is deliberately absent: `Vec::is_empty` is the same spelling and a
#: different question, so the schema's own emptiness is caught by the module
#: path rather than by the method name.
_SUBTYPING = re.compile(
    r"\bis_subtype_of\b|\bsubtype_relation\b|\bsubtype_steps\b"
    r"|\brelation_to\b|\bcrate::decision\b|\bvalgebra_core::decision\b"
)


def _walk_sources() -> list[Path]:
    return sorted(path for path in WALK.rglob("*.rs") if not lib.is_test_module(path))


# THEORY: no-arrow-type
def test_the_membership_walk_never_consults_subtyping() -> None:
    """Membership is decided without asking whether one schema is inside another.

    This is what lets the naive interpretation stand. A set-theoretic type
    system with arrows cannot read a type as a set of values directly: the
    meaning of an arrow is a set of functions, a function is checked by its
    type, and the type needs the subtyping relation the interpretation was
    supposed to define. The circle is broken by the universal model and the
    extensional interpretation, which is the apparatus this project does not
    have.

    It escapes because a value's membership here is structural: the walk reads
    the value and the node and answers, and it never asks the decision
    procedure anything. So the interpretation is the naive one, and the
    theorems that close the circle are machinery nothing needs.

    A call from the walk into the procedure would not be a bug on its own --
    it would be the claim above becoming false, and every argument resting on
    it going with it.
    """
    sources = _walk_sources()
    # A scan that finds no files makes the assertion vacuous.
    assert len(sources) >= 5, f"the walk scan found only {sources}"
    assert any(path.name == "walk.rs" for path in sources), (
        "the scan missed the walk itself"
    )

    offenders: list[str] = []
    for source in sources:
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _SUBTYPING.search(line):
                offenders.append(f"{source.name}:{number}: {line.strip()}")
    assert not offenders, (
        "the membership walk reaches into the decision procedure:\n"
        + "\n".join(offenders)
        + "\nMembership is structural, and that is what keeps the naive "
        "interpretation available. A walk that asks about inclusion makes the "
        "circularity argument in `docs/dev/10-theory.md` false."
    )
