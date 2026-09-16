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

And both lanes are held to enforcing a **region** floor beside the line floor.
A line is covered when any part of it ran, so a line with two arms counts as
covered having taken one; the region figure is the one that notices the arm
nobody reached, and it is three and a half points lower on the binding.

LEDGER: every coverage lane names its scope, and the scope is the tree's
"""

from __future__ import annotations

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
CORPUS = re.compile(r"(?:^|/)(?:laws|index_laws|interpreter)\.rs$")


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
    found = re.search(r"--ignore-filename-regex\s+'([^']*)'", flat)
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
