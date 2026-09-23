"""The coverage lanes measure the code that ships, and say so in numbers.

A coverage figure is read as "how much of this library is exercised". Both
lanes computed it over the test code too. The binding's four interpreter
corpora are 4,048 lines that run by construction -- a corpus is a table and a
loop -- so they arrive at 100% and lift the figure for the code around them.
The binding read 98.15% of lines with them and 97.01% without; regions, which
count each arm of a branch rather than each line, read 96.24% with and 93.61%
without. Two and a half points of the number described the tests.

So the scope is written down here, in both directions:

* every corpus file in the tree is one the lanes exclude, so a corpus added
  tomorrow does not quietly lift the figure;
* every exclusion the lanes carry names something that exists, so a file that
  moves takes its exclusion with it rather than leaving a pattern matching
  nothing.

A third scope is read here for the same reason, one measurement over. Coverage
counts the lines a suite *reached*; it says nothing about what those suites
assert. `tests/test_denotation.py` is the one that holds the walk to an
independent statement of the denotation -- a predicate sharing none of the
frontend -- and its reach is its generator. A refinement kind the generator
never builds is an arm nothing independent reads, and a generator that never
builds a shape reports no failure about it, so the kinds are read from
`Constraint` in `ir.rs` and held to the leaves that file lists.

And both lanes are held to enforcing a **region** floor beside the line floor.
A line is covered when any part of it ran, so a line with two arms counts as
covered having taken one; the region figure is the one that notices the arm
nobody reached, and it is three and a half points lower on the binding.

LEDGER: every coverage lane names its scope, and the scope is the tree's
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: The jobs that enforce a coverage floor, and the crate each one measures.
LANES = {"rust-coverage": "valgebra-core", "binding-coverage": "valgebra-py"}

#: What makes a file test code rather than shipped code.
#:
#: `laws.rs` and `index_laws.rs` are the core's property suites; an
#: `interpreter.rs` is a binding corpus driving real Python values through a
#: module. Both are compiled into the crate rather than into a `tests/`
#: directory -- a corpus needs the crate's private items -- which is why they
#: reach a coverage report at all.
#:
#: A file whose name ends in `tests.rs` is the same thing one module down.
#: `docs/dev/08-testing.md` states the rule that puts it there -- a test module
#: longer than a screen lives in a sibling file -- and `test_module_placement.py`
#: holds the tree to it, so the tree has twenty-odd of them and every one is
#: compiled into the crate for the same reason a corpus is. Counted as shipped
#: code they lift the figure for the code around them, which is the whole defect
#: this ledger is about; a suite is measured by whether it *kills mutants*, not
#: by how much of itself it executes.
CORPUS = re.compile(r"(?:^|/)(?:laws|index_laws|interpreter)\.rs$|tests\.rs$")


def _lane_scripts() -> dict[str, str]:
    """Give the script of each coverage lane's measuring step."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    found = {}
    for name in LANES:
        job = workflow["jobs"][name]
        scripts = [
            step["run"] for step in job["steps"] if isinstance(step.get("run"), str)
        ]
        measuring = [text for text in scripts if "--fail-under" in text]
        assert len(measuring) == 1, f"{name}: {len(measuring)} steps enforce a floor"
        found[name] = measuring[0]
    return found


def _ignored(lane: str) -> re.Pattern[str]:
    """Give the pattern a lane excludes files by, compiled.

    The argument rather than the step's text: a step is mostly prose, and the
    word `interpreter` appears in it three times about the *embedded* one. A
    check reading the whole script found it there and passed on a tree with the
    exclusion removed, which the planted defect is what caught.
    """
    script = _lane_scripts()[lane]
    # The argument is quoted and may be wrapped, so the continuations go first.
    flat = script.replace("\\\n", " ")
    # A lane names its scope once and spends it twice -- the crate floor and
    # the per-file one -- so the shell variable it is bound to is where the
    # scope is written. The literal argument is read too, for a lane that
    # passes it inline.
    found = re.search(r"^\s*ignore='([^']*)'", flat, re.MULTILINE) or re.search(
        r"--ignore-filename-regex\s+'([^']*)'", flat
    )
    assert found, f"{lane} enforces a floor without naming a scope"
    return re.compile(found.group(1))


def _corpus_files() -> set[str]:
    """Give every corpus file in the tree, as a path from the root.

    Spelled with forward slashes on every platform. The pattern these are
    matched against is the one the lane passes to `cargo llvm-cov`, which is
    written the way a coverage report spells a path -- and a path built from
    `Path` parts on Windows is separated by backslashes, which that pattern
    matches nowhere. Every file then reads as one the lane fails to exclude,
    and the ledger reddens on Windows alone while saying the scope is wrong.
    """
    return {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "crates").rglob("*.rs")
        if CORPUS.search(path.name)
    }


def test_the_tree_has_corpus_files_to_exclude() -> None:
    """The parse is a detector, so it must be shown to have read something."""
    corpora = _corpus_files()
    # Spelled the way the exclusion pattern is, on every platform: a path
    # separated the other way matches that pattern nowhere, and every row below
    # would fail on one operating system and pass on the rest.
    assert not any("\\" in name for name in corpora), sorted(corpora)
    assert all(name.startswith("crates/") for name in corpora), sorted(corpora)
    assert len(corpora) >= 5, sorted(corpora)
    assert any("laws.rs" in name for name in corpora)
    assert sum("interpreter.rs" in name for name in corpora) >= 4
    # And the sibling test modules, which are the bulk of it.
    assert sum(name.endswith("tests.rs") for name in corpora) >= 15, sorted(corpora)


@pytest.mark.parametrize("lane", sorted(LANES))
def test_a_lane_excludes_every_corpus_file_of_the_crate_it_measures(
    lane: str,
) -> None:
    """A corpus counted as shipped code lifts the figure for the code around it."""
    ignored = _ignored(lane)
    crate = LANES[lane]
    mine = sorted(name for name in _corpus_files() if crate in name)
    assert mine, f"{lane} measures {crate}, which has no corpus file"
    missing = [name for name in mine if not ignored.search(name)]
    assert not missing, (
        f"{lane} counts these corpus files as shipped code: {missing}. Add each "
        "to the step's --ignore-filename-regex, and re-record the floor in the "
        "same change with the number it measures."
    )


@pytest.mark.parametrize("lane", sorted(LANES))
def test_a_lane_enforces_a_region_floor_beside_its_line_floor(lane: str) -> None:
    """A line floor alone passes a branch with an arm nobody reached.

    A line is covered when any part of it ran, so `if a { x } else { y }` on
    one line counts as covered having taken one arm. The region figure counts
    the arms, and is the lower of the two on both crates.
    """
    script = _lane_scripts()[lane]
    assert "--fail-under-lines" in script, lane
    assert "--fail-under-regions" in script, (
        f"{lane} enforces a line floor and no region floor, so a branch with an "
        "unreached arm passes it."
    )


@pytest.mark.parametrize("lane", sorted(LANES))
def test_a_floor_is_a_number_the_lane_can_reach(lane: str) -> None:
    """A floor above the measurement is a lane that cannot pass.

    Read as a bound rather than as the measurement: the figure moves with every
    commit, and a test asserting today's number fails on the one that adds a
    line. What is held is that somebody wrote a percentage rather than a
    fraction or a count.
    """
    script = _lane_scripts()[lane]
    floors = [int(found) for found in re.findall(r"--fail-under-\w+ (\d+)", script)]
    assert len(floors) == 2, f"{lane}: {floors}"
    for floor in floors:
        assert 50 <= floor <= 100, f"{lane}: {floor} is not a percentage"


def test_a_lane_records_the_branch_number() -> None:
    """A region floor still passes a branch arm nobody reached.

    A region is a span the compiler emits, and a two-armed branch inside one
    contributes regions for both arms only where the arms are separate spans.
    Measured on the core, the shipped scope reads 98% of lines, 97% of regions
    and 90% of branches -- so eight points of arms sit under a floor that both
    other figures pass.

    `cargo llvm-cov` has no `--fail-under-branches`, so the number is recorded
    and ratcheted the way the mutation baseline records survivors: a lane
    measures it, `scripts/branch_coverage.py` compares it against the recorded
    floor, and the floor only ever moves up with the measurement.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    measuring = {
        name: [
            step["run"]
            for step in job.get("steps", [])
            if isinstance(step.get("run"), str)
        ]
        for name, job in workflow["jobs"].items()
        if any(
            isinstance(step.get("run"), str) and "--branch" in step["run"]
            for step in job.get("steps", [])
        )
    }
    assert measuring, (
        "no lane measures branch coverage, so the arms a region floor passes "
        "are counted by nothing"
    )
    # The measurement and the ratchet are separate steps, and they have to be
    # in one job: a figure measured where nothing reads it is printed rather
    # than held.
    assert any(
        any("branch_coverage.py" in text for text in steps)
        for steps in measuring.values()
    ), "a lane measures branches and no step in it ratchets the number"
    floor = ROOT / "scripts" / "branch_coverage.json"
    assert floor.exists(), f"{floor.name} records no floor"
    recorded = json.loads(floor.read_text(encoding="utf-8"))["floor"]
    assert 50 <= recorded <= 100, recorded


#: How the testing page spells the count of tests the GIL stands down.
_GIL_SKIPS = re.compile(
    r"\*\*Some arms only a free-threaded interpreter reaches.*?"
    r"(?P<count>\w+)\s+tests\s+skip\s+for\s+that\s+reason",
    re.DOTALL,
)

#: The English numbers the page writes a small count in.
_WRITTEN = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
    "Six": 6,
    "Seven": 7,
    "Eight": 8,
}


def _gil_skipped_tests() -> list[str]:
    """Every test the running interpreter's lock stands down, by name.

    Read from the decorators rather than by running the suite: a count taken
    from a run would be a count of *items*, which the parametrised one
    multiplies. A test is counted by the *condition* it skips on -- the running
    interpreter holding its lock, `_gil_enabled()` -- because that is the case
    the page's paragraph is about: two threads that cannot overlap, so the arm
    a moving container drives is never reached. A test that skips on how the
    interpreter was *built* stands down for another reason and drives no walk
    arm, however its reason is worded.
    """
    found = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not decorator.args:
                    continue
                condition = decorator.args[0]
                if (
                    isinstance(condition, ast.Call)
                    and isinstance(condition.func, ast.Name)
                    and condition.func.id == "_gil_enabled"
                ):
                    found.append(f"{path.name}::{node.name}")
    return found


def test_the_page_counts_the_tests_the_gil_stands_down() -> None:
    """The figure the free-threaded paragraph rests on is the tree's.

    The paragraph argues that some arms are executed by the python matrix and
    measured by no lane, and the tests that stand down under a lock are the
    evidence for it. A count nothing reads is a count that drifts, and this one
    had: the page said six where the tree skips three.

    Here rather than beside the tests, because the claim is about what a
    coverage lane measures -- which is this file's subject -- and because the
    arms it names are the ones no floor in this file covers.
    """
    page = (ROOT / "docs" / "dev" / "08-testing.md").read_text(encoding="utf-8")
    stated = _GIL_SKIPS.search(page)
    assert stated, "the testing page states no count of the tests the GIL skips"

    skipped = _gil_skipped_tests()
    # The parse is the detector: no decorator found would agree with a page
    # claiming none, having read nothing.
    assert skipped, "no test names the GIL as its reason to stand down"
    assert _WRITTEN[stated["count"]] == len(skipped), (
        f"the page says {stated['count'].lower()} tests skip under the lock and "
        f"the tree skips {len(skipped)}: {skipped}"
    )


def test_no_coverage_lane_runs_a_free_threaded_interpreter() -> None:
    """The other half of the same paragraph: those arms are measured nowhere.

    The claim is the reason the arms are accepted rather than driven, so it has
    to fail the day it stops being true. A free-threaded leg added to either
    coverage lane makes the paragraph wrong in the direction that matters -- it
    would be claiming a hole the tree no longer has -- and a reader would have
    no way to tell.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for name in LANES:
        job = workflow["jobs"][name]
        versions = [
            step["with"]["python-version"]
            for step in job["steps"]
            if isinstance(step.get("with"), dict) and "python-version" in step["with"]
        ]
        assert not any("t" in str(version) for version in versions), (
            f"{name} names a free-threaded interpreter ({versions}), so the "
            "testing page's paragraph about arms no lane measures is stale"
        )


# --- The denotation generator's scope, held to the algebra -------------------

IR = ROOT / "crates" / "valgebra-core" / "src" / "ir.rs"
DENOTATION = ROOT / "tests" / "test_denotation.py"

_CONSTRAINTS = re.compile(
    r"^pub enum Constraint \{$(.*?)^\}$", re.DOTALL | re.MULTILINE
)
_VARIANT = re.compile(r"^    ([A-Z][A-Za-z]*)[ ({,]", re.MULTILINE)

#: The leaves the denotation generator draws a refinement from, read as the
#: kind each names. Read from the source rather than imported: a product test
#: module is not this one's to import, and the list is written to be read --
#: each entry leads with the kind's own name.
_REFINED_LEAF = re.compile(r'^    \(\n        "([A-Z][A-Za-z]*)",$', re.MULTILINE)

#: The one kind no independent predicate can be written for. A `Predicate`
#: constraint runs arbitrary Python, so a predicate mirroring it would *be* it,
#: and what the walk owes there is to call it and read the answer --
#: `tests/test_refinements.py` drives that directly.
UNORACLED = frozenset({"Predicate"})


def _constraint_kinds() -> set[str]:
    """Give the constraint kinds the algebra carries, read from the tree."""
    body = _CONSTRAINTS.search(IR.read_text(encoding="utf-8"))
    assert body, "ir.rs has no Constraint enum this ledger reads"
    found = set(_VARIANT.findall(body.group(1)))
    # The scan is the detector: an empty set would pass having read nothing.
    assert len(found) >= 9, f"the scan found only {sorted(found)}"
    return found


def _oracled_kinds() -> set[str]:
    """Give the kinds the denotation generator builds a leaf from."""
    found = set(_REFINED_LEAF.findall(DENOTATION.read_text(encoding="utf-8")))
    assert found, "the denotation generator lists no refinement leaf"
    return found


def test_every_constraint_kind_is_one_the_denotation_generator_builds() -> None:
    """A kind no case builds is an arm nothing independent reads.

    The suites that drive the refinement arms -- the constraint matrix, the
    published boundary's rows -- assert answers a person wrote beside the code
    that gives them. The denotation suite is the one that does not: its
    predicate states the set again, from the pages, over values it did not
    choose. A kind outside its generator has no such reading, and nothing says
    so, because a generator that never builds a shape reports no failure about
    it.
    """
    missing = sorted(_constraint_kinds() - UNORACLED - _oracled_kinds())
    assert not missing, (
        f"constraint kinds no denotation case is built from: {missing}. Add a "
        "leaf to `_REFINED` in tests/test_denotation.py with the predicate "
        "that states its denotation independently, or name it in `UNORACLED` "
        "with the reason none can be written."
    )


_SCHEMA = re.compile(r"^pub enum Schema \{$(.*?)^\}$", re.DOTALL | re.MULTILINE)
_NODE_DRAWN = re.compile(r'^    "([A-Z][A-Za-z]*)": "', re.MULTILINE)

#: The one variant no compiled validator holds: the marker `recursive` uses
#: while a definition is being built, resolved to a `Ref` before a validator
#: is returned. No shape a caller writes leaves one in the tree.
UNDRAWN = frozenset({"SelfRef"})


def _schema_variants() -> set[str]:
    """Give the node kinds the algebra carries, read from the tree."""
    body = _SCHEMA.search(IR.read_text(encoding="utf-8"))
    assert body, "ir.rs has no Schema enum this ledger reads"
    found = set(_VARIANT.findall(body.group(1)))
    assert len(found) >= 19, f"the scan found only {sorted(found)}"
    return found


def _drawn_nodes() -> set[str]:
    """Give the variants the denotation generator says it builds a case from."""
    found = set(_NODE_DRAWN.findall(DENOTATION.read_text(encoding="utf-8")))
    assert found, "the denotation generator lists no node it draws"
    return found


def test_every_schema_variant_is_one_the_denotation_generator_builds() -> None:
    """A node the generator never builds is an arm nothing independent reads.

    The same claim the constraint ledger above makes, one enum over: the
    denotation suite is the one layer that states each node's set from the
    page rather than from the code, and a variant outside its generator has
    no such reading. A generator that never builds a shape reports no failure
    about it, so the list of what it builds is held to the enum.
    """
    missing = sorted(_schema_variants() - UNDRAWN - _drawn_nodes())
    assert not missing, (
        f"schema variants no denotation case is built from: {missing}. Add a "
        "shape to `_specs` in tests/test_denotation.py with the predicate that "
        "states its denotation independently, list the variant in "
        "`_NODES_DRAWN`, or name it in `UNDRAWN` with the reason none can be."
    )
    unknown = sorted(_drawn_nodes() - _schema_variants())
    assert not unknown, f"denotation shapes naming no variant: {unknown}"
    assert _schema_variants() >= UNDRAWN, "an excuse with no subject"


def test_every_leaf_the_denotation_generator_lists_is_a_kind() -> None:
    """The other direction: a leaf named for a kind the algebra dropped."""
    unknown = sorted(_oracled_kinds() - _constraint_kinds())
    assert not unknown, (
        f"denotation leaves naming no constraint kind: {unknown}. A kind the "
        "algebra drops takes its leaf with it."
    )
