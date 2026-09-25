"""Every ledger ships with the defect that trips it.

A ledger is an enumerated list held to the tree in both directions, and the
count is kept on the testing page rather than here. Their worth rests entirely
on failing when the tree stops matching the list -- and one of them could
not. `tests/test_local_gate.py` filtered its steps with ``not runnable(name)
and name not in NEEDS_A_RUNNER``, which is ``X and not X``: the list it built
was empty for every possible workflow, so the assertion passed on a tree that
had already broken the claim. The comment above it said the clause was true by
construction, and the `assert` stayed.

Reading a ledger cannot tell you whether it can fail. Running it against a tree
that breaks its claim can, so that is what this does: for each ledger, plant the
defect it exists to catch in a throwaway copy of the tree, run that ledger there,
and require it to fail. A ledger with no plant here is a ledger nobody has shown
to work.

The copy is the tracked files only, made once and repaired between rows, so a
plant cannot leak into the next one or into the tree being audited.

LEDGER: every ledger fails on the defect it exists to catch
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# The repository checks are not the product suite: this file reads the tree,
# copies it, and runs pytest inside the copy, none of which ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent


class Plant(NamedTuple):
    """A defect, the ledger that must catch it, and the files it touches."""

    ledger: str
    """The ledger's test file, relative to the tree root."""
    touches: tuple[str, ...]
    """Every path the plant writes, so the copy can be repaired after it."""
    apply: Callable[[Path], None]
    """Break the claim, given the root of the copy."""


def _name_the_working_area(tree: Path) -> None:
    """Rewrite the tip's message so it points at a note no reader can open.

    Spelled from its pieces rather than written out, so this file does not
    itself carry the string the docs lint refuses.
    """
    # Written as a join on purpose: spelled out, this file would carry the very
    # reference the docs lint refuses, and fail it.
    note = "-".join(("REPORT", "99"))  # noqa: FLY002
    # An identity of its own: a clone inherits no committer, and this must not
    # depend on whether the machine running it has one configured.
    subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [  # noqa: S607 - git is on the path of every machine that clones this
            "git",
            "-C",
            str(tree),
            "commit",
            "--amend",
            "--no-verify",
            "-m",
            f"test: a planted message\n\n{note} asked for this.",
        ],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "plant",
            "GIT_AUTHOR_EMAIL": "plant@example.invalid",
            "GIT_COMMITTER_NAME": "plant",
            "GIT_COMMITTER_EMAIL": "plant@example.invalid",
        },
    )


def _cite_an_orphan(tree: Path) -> None:
    """Point the reference corpus at a commit no branch reaches.

    The commit is written here rather than looked for: an orphan is what a
    rewrite leaves behind, and a tree that already carries one is a tree whose
    ledger has something to report without this row's help.
    """
    written = _plant_git(tree, "write-tree").strip()
    orphan = _plant_git(tree, "commit-tree", written, "-m", "an orphan, planted")
    record = tree / "scripts" / "metamorphic_reference.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["commit"] = orphan.strip()
    record.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _plant_git(tree: Path, *args: str) -> str:
    """Run git in the copy under an identity of the plant's own.

    A clone inherits no committer, so writing an object cannot depend on
    whether the machine running the suite has one configured.
    """
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(tree), *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "plant",
            "GIT_AUTHOR_EMAIL": "plant@example.invalid",
            "GIT_COMMITTER_NAME": "plant",
            "GIT_COMMITTER_EMAIL": "plant@example.invalid",
        },
    )
    return done.stdout


def _edit(tree: Path, relative: str, old: str, new: str) -> None:
    """Replace `old` once in `relative`, refusing if it is not there.

    A plant that silently applies to nothing reads as "the ledger missed it",
    which is the failure mode this whole file exists to rule out.
    """
    path = tree / relative
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        message = f"the plant did not land: {old[:60]!r} in {relative}"
        raise AssertionError(message)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _write(tree: Path, relative: str, text: str) -> None:
    path = tree / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


#: A `run:` step no developer's machine can fill in and nobody has excused.
RUNNER_ONLY_STEP = """      - name: A step only a runner can fill in
        run: echo "${{ github.sha }}"
      - name: Audit the workflows"""

#: A pull-request job the `ci` aggregator does not wait for.
UNREQUIRED_JOB = """jobs:
  planted:
    runs-on: ubuntu-latest
    steps:
      - run: echo planted
"""

PLANTS = (
    Plant(
        "tests/test_module_placement.py",
        ("crates/valgebra-core/src/descr/budget.rs",),
        # The drift the bar exists to catch, in its smallest form: the shortest
        # inline test module in the tree, padded past a hundred lines. Nothing
        # about it fails to compile and no test changes its answer, which is
        # exactly why a ledger has to be the thing that notices.
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/descr/budget.rs",
            "mod tests {",
            "mod tests {\n" + "    // one case at a time\n" * 60,
        ),
    ),
    Plant(
        # The attribute gone from a crate root: `unsafe` compiles again, every
        # value answers exactly as it did, and the soundness page goes on
        # resting on a property nothing enforces.
        "tests/test_crate_attributes.py",
        ("crates/valgebra-core/src/lib.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/lib.rs",
            "#![forbid(unsafe_code)]",
            "// the attribute, planted away",
        ),
    ),
    Plant(
        # The other half of the same ledger: the attribute stands and a file
        # below it writes the keyword anyway. Behind a visibility, which is the
        # shape a scan anchored on the line's opening reads past.
        "tests/test_crate_attributes.py",
        ("crates/valgebra-core/src/kind.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/kind.rs",
            "    /// Whether a value can have both this kind and `other`.",
            "    pub unsafe fn planted() {}\n\n"
            "    /// Whether a value can have both this kind and `other`.",
        ),
    ),
    Plant(
        # A public name the suite never mentions: the state every method starts
        # in, and the one a stub grows a line for without anyone noticing.
        "tests/test_use_case_ledger.py",
        ("python/valgebra/_valgebra.pyi",),
        lambda tree: _edit(
            tree,
            "python/valgebra/_valgebra.pyi",
            "    def is_empty(self) -> bool: ...",
            "    def is_empty(self) -> bool: ...\n"
            "    def unnamed_by_any_test(self) -> None: ...",
        ),
    ),
    Plant(
        # A bound declared and driven by nothing: the state every bound starts
        # in, and the one a limits page cannot see from its own prose.
        "tests/test_bound_ledger.py",
        ("crates/valgebra-core/src/descr/lower.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/descr/lower.rs",
            "pub const UNFOLDS: u32 = 1;",
            "pub const UNFOLDS: u32 = 1;\npub const MAX_PLANTED_BOUND: u32 = 1;",
        ),
    ),
    Plant(
        # A load-bearing result whose `HELD-BY:` names a test the tree does not
        # have: the shape a rename leaves behind, and the one the page cannot
        # detect on its own.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/10-theory.md",
            "HELD-BY: test_walk_matches_denotation",
            "HELD-BY: a_test_this_tree_does_not_have, test_walk_matches_denotation",
        ),
    ),
    Plant(
        # The same ledger, on the citation half. The two checks that resolve a
        # `SOURCE:` line against the argument stand down wherever the argument
        # is not in the clone, which is every clone but the author's -- so the
        # line's *form* is what a runner can hold, and a line nobody can read
        # would otherwise be malformed on the page and skipped in the lane.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/10-theory.md",
            'SOURCE: \u00a713.2 "**The laws hold by construction**"',
            "SOURCE: the section on the lattice",
        ),
    ),
    Plant(
        # An obligation added to the page with neither a `HELD-BY:` nor an
        # `OWED:` line: a sentence, which is the state every result starts in
        # and the one the ledger exists to refuse.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/10-theory.md",
            "### Obligations\n",
            "### Obligations\n\n**A planted obligation nothing holds.** Every"
            " arm returns before the budget is read. **[OBLIGATION]**\n",
        ),
    ),
    Plant(
        "tests/test_closure_ledger.py",
        ("crates/valgebra-core/src/ir.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/ir.rs",
            "pub enum Schema {",
            "pub enum Schema {\n    Invented(u8),",
        ),
    ),
    Plant(
        "tests/test_completeness_ledger.py",
        ("tests/test_completeness_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_completeness_ledger.py",
            "DECIDED = [",
            "DECIDED = [\n"
            '    pytest.param("equivalent", int, str, id="planted:int==str"),',
        ),
    ),
    Plant(
        "tests/test_completeness_probe.py",
        ("tests/test_completeness_probe.py",),
        lambda tree: _edit(
            tree,
            "tests/test_completeness_probe.py",
            "ACCEPTED: dict[str, str] = {",
            'ACCEPTED: dict[str, str] = {\n    "planted gap": "",',
        ),
    ),
    Plant(
        "tests/test_build_surfaces.py",
        ("planted/Cargo.toml",),
        lambda tree: _write(
            tree,
            "planted/Cargo.toml",
            '[package]\nname = "planted"\nversion = "0.0.0"\nedition = "2024"\n',
        ),
    ),
    Plant(
        "tests/test_code_table.py",
        ("crates/valgebra-py/src/check/walk/scalar.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/check/walk/scalar.rs",
            "            code: LITERAL_ERROR.as_str(),",
            '            code: "literal_error",',
        ),
    ),
    Plant(
        "tests/test_doc_examples.py",
        ("CONTRIBUTING.md",),
        lambda tree: _write(
            tree,
            "CONTRIBUTING.md",
            (tree / "CONTRIBUTING.md").read_text(encoding="utf-8")
            + "\n```python\nraise SystemExit('an example no lane runs')\n```\n",
        ),
    ),
    Plant(
        "tests/test_version_gates.py",
        ("crates/valgebra-py/src/build/interpreter.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/build/interpreter.rs",
            '(Since(11), "typing.Never", "nothing"),',
            '(Since(99), "typing.Never", "nothing"),',
        ),
    ),
    Plant(
        "tests/test_feature_lanes.py",
        ("crates/valgebra-py/Cargo.toml",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/Cargo.toml",
            "pytest-sweep = []",
            "pytest-sweep = []\nplanted-tests = []",
        ),
    ),
    Plant(
        "tests/test_lane_coverage.py",
        ("scripts/planted_gate.py",),
        lambda tree: _write(
            tree, "scripts/planted_gate.py", '"""A gate in no lane."""\n'
        ),
    ),
    Plant(
        "tests/test_contract_inventory.py",
        ("scripts/planted_gate.py",),
        lambda tree: _write(
            tree, "scripts/planted_gate.py", '"""A gate with no contract row."""\n'
        ),
    ),
    # Not a roll entry, though the roll is what this ledger is *about*. Its
    # roll checks measure from the tag of the released version the page names,
    # and stand down where that tag is not in the clone -- a shallow checkout,
    # and the window between the release bump and the tag it is pushed with,
    # during which the roll is empty and there is no entry to take away. What
    # runs in every state is the claim that keeps the rest from running
    # nowhere: one lane checks out the whole history. Planting that is planting
    # a defect this ledger catches whenever it is asked.
    Plant(
        "tests/test_changelog_ledger.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "fetch-depth: ${{ (matrix.os == 'ubuntu-latest'",
            "fetch-depth: ${{ (matrix.os == 'macos-latest'",
        ),
    ),
    Plant(
        "tests/test_local_gate.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "      - name: Audit the workflows",
            RUNNER_ONLY_STEP,
        ),
    ),
    # The same ledger again, on the half of the gate that is not a workflow
    # step: the floor interpreter it builds. Written down rather than read, the
    # number is right today and wrong the morning the floor moves -- and wrong
    # in the direction that keeps passing, since the gate goes on building a
    # release nothing supports and reporting the suite green on it.
    Plant(
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            "    return versions[0]",
            '    return "3.10"',
        ),
    ),
    # The other half of the same claim, and the half a reader acts on: the leg
    # that takes the history says so in its name. Pointed at a leg that does not
    # take it, the name is still there and still specific -- and it now names
    # the one result of nine that skipped everything it advertises. A ledger
    # that holds the depth and not the name leaves that state green.
    Plant(
        "tests/test_changelog_ledger.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "\n      (matrix.os == 'ubuntu-latest'",
            "\n      (matrix.os == 'macos-latest'",
        ),
    ),
    Plant(
        # The same ledger, the other direction it grew: a job the gate cannot
        # reach carries its reason by *name*, and a rename leaves the reason
        # naming nothing while the real job drops out of the count.
        "tests/test_local_gate.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "\n  wheel:\n",
            "\n  wheel-renamed:\n",
        ),
    ),
    Plant(
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        # A job the merge gate does not wait on: it runs, it can go red, and it
        # blocks nothing, which is what a gate over other gates exists to catch.
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "        bench,\n        bench-free-threaded,\n",
            "        bench,\n",
        ),
    ),
    Plant(
        # A supported release gone from the python job. The release still
        # builds, its classifier still promises it, and every other leg is
        # green, which is why a calendar has to be the thing that notices.
        "tests/test_python_lifecycle.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            '"3.12", "3.13", "3.14", "3.14t"',
            '"3.12", "3.14", "3.14t"',
        ),
    ),
    Plant(
        "tests/test_lane_interpreters.py",
        (".github/workflows/ci.yml",),
        # A lane that installs an interpreter and names none, which is the
        # shape the ledger's version column exists to refuse.
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "      - uses: $/.github/actions/setup-uv\n        with:\n"
            '          python-version: "3.12"\n'
            "      - run: uv sync --locked --no-install-project --group bench\n",
            "      - uses: $/.github/actions/setup-uv\n"
            "      - run: uv sync --locked --no-install-project --group bench\n",
        ),
    ),
    Plant(
        "tests/test_clock_ledger.py",
        ("tests/test_records.py",),
        # A fourth test reading the clock, in a file that reads none: the ledger
        # closes the list, so a new one is a new argument rather than a new line.
        lambda tree: _edit(
            tree,
            "tests/test_records.py",
            "import pytest\n",
            "import pytest\nimport time\n\n\n"
            "def test_a_planted_clock() -> None:\n"
            "    started = time.perf_counter()\n"
            "    assert time.perf_counter() >= started\n",
        ),
    ),
    Plant(
        "tests/test_commit_messages.py",
        (),
        # The subject is a message rather than a file, and the stage is a real
        # clone, so the plant writes one: the tip's message gains a name from
        # the working area. Nothing in the tree changes, which is why this row
        # touches no path and repairs itself by restoring the message.
        _name_the_working_area,
    ),
    Plant(
        "tests/test_fuzz_lane.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "-max_total_time=360 -fork=1 -malloc_limit_mb=64",
            "-max_total_time=360",
        ),
    ),
    Plant(
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(tree, ".github/workflows/ci.yml", "jobs:\n", UNREQUIRED_JOB),
    ),
    Plant(
        "tests/test_sweep_skips.py",
        ("crates/valgebra-core/src/decision/budget_tests.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/decision/budget_tests.rs",
            # A named test rather than the bare attribute: the file carries more
            # than one, and a plant anchored on a string the file may repeat is
            # one that stops landing when a test is added beside it.
            "#[test]\nfn an_exhausted_budget_refuses_to_spend",
            "// SWEEP-SKIP: planted, and no --skip names it\n"
            "#[test]\nfn an_exhausted_budget_refuses_to_spend",
        ),
    ),
    Plant(
        "tests/test_suite_partition.py",
        ("tests/test_planted.py",),
        lambda tree: _write(
            tree,
            "tests/test_planted.py",
            "def test_planted() -> None:\n    assert True\n",
        ),
    ),
    Plant(
        "tests/test_mutation_scope.py",
        ("crates/valgebra-py/src/planted.rs",),
        lambda tree: _write(
            tree,
            "crates/valgebra-py/src/planted.rs",
            "pub fn planted() -> u8 {\n    1\n}\n",
        ),
    ),
    Plant(
        "tests/test_harness_conditionals.py",
        ("crates/valgebra-py/src/render.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/render.rs",
            "use std::cell::RefCell;",
            '#[cfg(feature = "interpreter-tests")]\n'
            "pub fn planted() {}\n\n"
            "use std::cell::RefCell;",
        ),
    ),
    Plant(
        "tests/test_floor_names.py",
        ("tests/test_enums.py",),
        # The form itself, from the lane it reddened: a name that reaches
        # `typing` one release above the floor, imported where importing the
        # module runs it.
        lambda tree: _edit(
            tree,
            "tests/test_enums.py",
            "import sys",
            "import sys\nfrom typing import NotRequired",
        ),
    ),
    Plant(
        "tests/test_metamorphic_gate.py",
        ("scripts/metamorphic_gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/metamorphic_gate.py",
            '    """List the pairs whose verdict changed.',
            '    """List the pairs whose verdict changed.\n\n'
            "    Planted: reports none.\n"
            '    """\n    return []\n    _unreachable = """',
        ),
    ),
    Plant(
        "tests/test_cited_commits.py",
        ("scripts/metamorphic_reference.json",),
        _cite_an_orphan,
    ),
    Plant(
        "tests/test_typed_consumer.py",
        ("tests/typing/consumer.py",),
        # The failure the ledger is for: a name checked by assignment alone,
        # which `Any` satisfies. Planted as the weaker reading rather than as a
        # deletion, because that is the shape it arrives in.
        lambda tree: _edit(
            tree,
            "tests/typing/consumer.py",
            "    assert_type(schema.is_valid(value), bool)",
            "    valid: bool = schema.is_valid(value)\n    del valid",
        ),
    ),
    Plant(
        "tests/test_coverage_scope.py",
        (".github/workflows/ci.yml",),
        # The failure the ledger is for: a corpus counted as shipped code,
        # which lifts the figure for the code around it. Planted by dropping
        # the exclusion, because that is how it arrives -- a new corpus is
        # written and nobody adds it to the pattern.
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "|/interpreter\\.rs$|tests\\.rs$'",
            "'",
        ),
    ),
    Plant(
        # A file excused from the ordinary sweep because the Python suite covers
        # it, and examined by no sweep that runs the Python suite. The exclusion
        # still reads as coverage and the file is measured by nothing, which is
        # the exact state the second configuration exists to make impossible --
        # and which nothing else in the tree would notice.
        "tests/test_pytest_sweep_scope.py",
        (".cargo/mutants-pytest.toml",),
        lambda tree: _edit(
            tree,
            ".cargo/mutants-pytest.toml",
            '    "crates/valgebra-py/src/render.rs",\n',
            "",
        ),
    ),
    Plant(
        # A numbered result cited under no work: the page names a lemma and a
        # reader has no way to say whose. The shelf direction cannot see it --
        # every paper is still there and every row still resolves -- so only the
        # attribution direction does.
        "tests/test_citation_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/10-theory.md",
            "# The theory\n",
            "# The theory\n\n"
            "A planted result cited as Theorem 9.9, above every work the page "
            "names, so nothing says whose theorem it is.\n",
        ),
    ),
    Plant(
        # The other end of the theory ledger: a claim renamed on the page,
        # which leaves every marker naming it pointing at nothing while still
        # reading like a pointer. The `HELD-BY:` direction does not see it --
        # the tests it names all still exist -- so only the reverse one does.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/10-theory.md",
            "**[OBLIGATION: the-budget-declines]**",
            "**[OBLIGATION: the-budget-stops]**",
        ),
    ),
    Plant(
        # The failure the ledger is for: a documented outcome the suite reaches
        # and never pins. The call runs, so the sweep reports the arm covered
        # and the coverage lane reports the line run -- and nothing holds the
        # method to the answer its own docstring promises a caller.
        "tests/test_surface_outcomes.py",
        ("tests/test_skeleton.py",),
        lambda tree: _edit(
            tree,
            "tests/test_skeleton.py",
            "    assert Validator(int).validate(3) is None\n"
            "    assert Validator(int).validate(3, fail_fast=True) is None\n",
            "    Validator(int).validate(3)\n"
            "    Validator(int).validate(3, fail_fast=True)\n",
        ),
    ),
    Plant(
        # The failure the ledger is for: a spelling the page teaches and the
        # suite never writes. The tests pick their own forms, so a row added to
        # a table is a promise nothing checks -- and the quiet half is that a
        # form the frontend stops reading is read as a literal, which denotes
        # the annotation object and admits nothing a caller has.
        "tests/test_form_ledger.py",
        ("docs/03-schema-language.md",),
        lambda tree: _edit(
            tree,
            "docs/03-schema-language.md",
            "| `bool` | `{True, False}` |\n",
            "| `bool` | `{True, False}` |\n| `complex` | every `complex` instance |\n",
        ),
    ),
    Plant(
        # The failure the ledger is for: an entry added to the published
        # boundary and driven by nothing. The page is what a caller reads to
        # learn what the relations decide, and it was maintained by hand
        # against a procedure that moves -- so a promise it makes and the tree
        # breaks, or a decline it states and the tree decides, reads the same
        # as a page nobody has checked.
        "tests/test_boundary_ledger.py",
        ("docs/15-decidability.md",),
        lambda tree: _edit(
            tree,
            "docs/15-decidability.md",
            "- **Sets and frozensets.** By element inclusion.\n",
            "- **Sets and frozensets.** By element inclusion.\n"
            "- **A relation nobody drives.** Planted.\n",
        ),
    ),
    Plant(
        # The failure the ledger is for: a constraint added to the algebra and
        # put to no kind. A marker the frontend cannot ask of a base builds a
        # schema admitting nothing and reporting itself inhabited, which is
        # findable only by a caller who tries every value -- so the product of
        # the markers with the kinds is what has to be driven, and a column
        # nobody wrote a cell for is the state this refuses.
        "tests/test_constraint_matrix.py",
        ("crates/valgebra-core/src/ir.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/ir.rs",
            "pub enum Constraint {\n",
            "pub enum Constraint {\n"
            "    /// A constraint this ledger has no cell for.\n"
            "    Planted,\n",
        ),
    ),
    Plant(
        # The failure the ledger is for: a node added to the algebra and asked
        # against nothing. The relation suites each pick the pairs they are
        # about, so a variant nobody wrote a pair for is a variant whose whole
        # column is whatever the procedure happens to answer -- which is the
        # state the product of the node set with itself exists to refuse.
        "tests/test_relation_ledger.py",
        ("crates/valgebra-core/src/ir.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/ir.rs",
            "pub enum Schema {\n",
            "pub enum Schema {\n"
            "    /// A node this ledger has no pair for.\n"
            "    Planted,\n",
        ),
    ),
    Plant(
        "tests/test_frontend_refusals.py",
        ("crates/valgebra-py/src/build/classes.rs",),
        # The failure the ledger is for: a refusal reworded into a sentence no
        # row reads. Planted in the Rust rather than in a row, because that is
        # the direction the drift runs -- a message is edited and the tests
        # keep passing on the exception's type.
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/build/classes.rs",
            '"a Protocol must be @runtime_checkable to be used as a schema"',
            '"this class does not name a set"',
        ),
    ),
)


def _tracked() -> list[str]:
    listing = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(ROOT), "ls-files", "-z"],  # noqa: S607
        capture_output=True,
        check=True,
        text=True,
    )
    return [name for name in listing.stdout.split("\0") if name]


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Clone the tree once, overlay the working files, repair between plants.

    A **clone** rather than a copy of the files, because one ledger reads the
    history: the changelog roll is measured from the last release tag, and over
    a directory of files it skips itself rather than judging the plant. The
    clone is local, so its objects are hardlinked and cost nothing.

    The tracked working files are then laid over the checkout, so the tree the
    plants run against is the one being audited rather than the last commit.
    """
    copy = tmp_path_factory.mktemp("tree") / "tree"
    subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "clone", "--quiet", "--local", "--no-hardlinks", str(ROOT), str(copy)],  # noqa: S607
        check=True,
        capture_output=True,
    )
    for name in _tracked():
        source = ROOT / name
        if not source.is_file():
            continue  # a submodule or a path removed since the index was written
        target = copy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    yield copy
    shutil.rmtree(copy, ignore_errors=True)


def _repair(tree: Path, plant: Plant) -> None:
    """Undo a plant, so the next row starts from the tree the audit describes."""
    for name in plant.touches:
        source, target = ROOT / name, tree / name
        if source.is_file():
            shutil.copy2(source, target)
        else:
            target.unlink(missing_ok=True)


def _judged_nothing(output: str) -> bool:
    """Whether the ledger skipped the half that would have caught the plant.

    A ledger reading git history skips those tests in a shallow clone -- which is
    where `scripts/gate.py` runs the suite, and the reason this project has a
    local gate at all. A skip is not a detection and it is not a miss either: the
    plant is unjudgeable there, and reporting it as a failure would redden the
    lane over a clone shape rather than over the tree.
    """
    summary = output.strip().splitlines()[-1] if output.strip() else ""
    return "skipped" in summary and "failed" not in summary


@pytest.mark.parametrize(
    "plant", PLANTS, ids=[plant.ledger.split("/")[-1] for plant in PLANTS]
)
def test_a_ledger_fails_on_the_defect_it_exists_to_catch(
    plant: Plant, tree: Path
) -> None:
    plant.apply(tree)
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
            [
                sys.executable,
                "-m",
                "pytest",
                plant.ledger,
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=tree,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        _repair(tree, plant)
    if _judged_nothing(result.stdout):
        pytest.skip(
            f"{plant.ledger} skips itself in this clone, so the plant cannot be "
            "judged: it reads history a shallow checkout does not carry"
        )
    assert result.returncode != 0, (
        f"{plant.ledger} passed on a tree that breaks its claim. A ledger that "
        f"cannot fail is not evidence.\n{result.stdout[-2000:]}"
    )
    assert "failed" in result.stdout or "error" in result.stdout, (
        f"{plant.ledger} exited non-zero without a failing test, which is a "
        f"broken run rather than a detection.\n{result.stdout[-2000:]}"
    )


def test_every_ledger_carries_a_plant() -> None:
    """A ledger with no row above is one nobody has shown to work."""
    marked = {
        f"tests/{path.name}"
        for path in (ROOT / "tests").glob("test_*.py")
        if "LEDGER:" in path.read_text(encoding="utf-8")
    }
    assert marked, "no LEDGER marker found in any test"
    planted = {plant.ledger for plant in PLANTS}
    # This file is a ledger over the others; its own plant would be circular.
    missing = sorted(marked - planted - {"tests/test_ledger_plants.py"})
    assert not missing, (
        f"ledgers with no planted defect: {missing}. Add a row to PLANTS that "
        "breaks the claim, or the ledger is an assertion nobody has run against "
        "a tree that violates it."
    )
