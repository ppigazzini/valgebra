"""The fuzz soak bounds one allocation, and bounds memory per batch.

A libFuzzer run has two memory ceilings and they answer different questions.
`-rss_limit_mb` watches the *process*, which under the sanitizer grows with the
number of executions whatever the target does, because freed memory is
quarantined rather than returned. `-malloc_limit_mb` watches one *allocation*,
which is the thing a bound in this tree actually promises -- the automaton
builder's eight megabytes, the product's edge count -- and the thing worth
reddening a lane.

Leaving both unnamed gives the second the value of the first, so the only check
that ran was the one that cannot name a defect: the soak crossed the unnamed
2,048 MB default partway through its own budget and reported an out-of-memory
against whichever input was in flight, which replayed in 25 ms.

Held in both directions: the step must carry a fork mode, so the process ceiling
bounds a batch rather than the whole run, and it must name the allocation
ceiling rather than inherit one. A step that drops either fails here.

LEDGER: the fuzz soak names its allocation ceiling and forks its batches
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# A repository check: it reads the workflow, which ships in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: The step that runs the soak, by its `name:` in the workflow.
SOAK = "Fuzz the decision procedures"

#: The largest single allocation the soak may make, in megabytes. Above every
#: bound the code can ask for -- the regex builder's is eight -- and far below
#: the process ceiling it replaces, so it fires on a runaway allocation and on
#: nothing else.
ALLOCATION_CEILING_MB = 64


def _soak_step() -> str:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in spec["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == SOAK:
                return str(step["run"])
    message = f"the workflow has no {SOAK!r} step"
    raise AssertionError(message)


def test_the_soak_names_its_allocation_ceiling() -> None:
    """An unnamed `-malloc_limit_mb` is the process ceiling wearing its name."""
    step = _soak_step()
    found = re.search(r"-malloc_limit_mb=(\d+)", step)
    assert found, (
        "the soak does not name -malloc_limit_mb, so the only memory check it "
        "runs is the process one, which the sanitizer's own growth trips"
    )
    assert int(found.group(1)) == ALLOCATION_CEILING_MB, (
        f"the soak bounds one allocation at {found.group(1)} MB; this file "
        f"records {ALLOCATION_CEILING_MB}. Moving it is a decision with an "
        "argument, so move both."
    )


def test_the_soak_bounds_memory_per_batch() -> None:
    """Without a fork the process ceiling bounds the whole run, which grows."""
    step = _soak_step()
    assert re.search(r"-fork=\d+", step), (
        "the soak does not fork, so its memory ceiling applies to a process "
        "whose resident size grows with every execution it has done"
    )


def test_the_soak_still_has_a_time_budget() -> None:
    """The flags above are additions, not a replacement for the budget."""
    assert re.search(r"-max_total_time=\d+", _soak_step())


def test_the_generator_draws_every_node_the_ir_has() -> None:
    """A fuzzer explores the shapes it can build and no others.

    The generator is the target's universe, so a variant it never emits is one
    no soak has ever reached however long it runs -- and the laws it checks are
    laws about the fragment it draws rather than about the IR. Held to the
    `Schema` enum, which is where a node is added.

    Three variants are emitted through a name other than their own -- the top
    through its constant, and the two container kinds through the constructors
    that build them -- so each is listed with what reaches it. The top carries a
    *spelling* beside the set, and one arm covers both because a fuzz target
    reads verdicts rather than spellings. Anything else absent is a gap.
    """
    ir = (ROOT / "crates" / "valgebra-core" / "src" / "ir.rs").read_text(
        encoding="utf-8"
    )
    body = ir[ir.index("pub enum Schema") :]
    body = body[: body.index("\n}")]
    variants = set(re.findall(r"^    ([A-Z]\w+)", body, re.MULTILINE))
    assert len(variants) >= 15, sorted(variants)

    generator = (ROOT / "fuzz" / "src" / "lib.rs").read_text(encoding="utf-8")
    # The constructors count as well as the bare variants: a sequence is built
    # through `Schema::set` and `Schema::frozen_set`, which name no variant.
    emitted = set(re.findall(r"Schema::(\w+)", generator))
    #: Variants the generator reaches through a constructor rather than by name,
    #: each with the constructor that builds it.
    through = {
        "Anything": ("ANYTHING",),
        "Coll": ("set", "frozen_set"),
        "Seq": ("Seq",),
    }
    for variant, names in through.items():
        if any(name in emitted for name in names):
            emitted.add(variant)

    missing = sorted(variants - emitted)
    assert not missing, (
        f"the fuzz generator emits no {missing}. A variant it cannot build is "
        "one no soak reaches, so every law the target checks is a law about "
        "the fragment it draws."
    )


def test_the_generator_draws_a_reference_wider_than_its_table() -> None:
    """Both a reference that resolves and one that does not are reached.

    A `Ref` is an index into the definitions table the target carries, and the
    two cases are decided differently: one unfolds, and one becomes the
    polarity's cut, which is what a decider meets after a pruned build. Drawing
    the index from the table's own width would reach only the first.
    """
    generator = (ROOT / "fuzz" / "src" / "lib.rs").read_text(encoding="utf-8")
    assert "Schema::Ref(" in generator
    table = re.search(r"let n = count\(u, (\d+)\)\?;\s*\n\s*let mut defs", generator)
    assert table is not None, "the target builds no definitions table"
    index = re.search(r"Schema::Ref\(DefIx::new\(usize::from\(.*?% (\d+)\)", generator)
    assert index is not None, "the reference index is not drawn from a bound"
    assert int(index.group(1)) > int(table.group(1)), (
        "the reference index is drawn no wider than the table is built, so "
        "every reference resolves and the cut is never reached"
    )
