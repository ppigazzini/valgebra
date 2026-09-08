"""Every ledger ships with the defect that trips it.

A ledger is an enumerated list held to the tree in both directions, and the
project has fourteen. Their worth rests entirely on failing when the tree stops
matching the list -- and one of them could not. `tests/test_local_gate.py`
filtered its steps with ``not runnable(name) and name not in NEEDS_A_RUNNER``,
which is ``X and not X``: the list it built was empty for every possible
workflow, so the assertion passed on a tree that had already broken the claim.
The comment above it said the clause was true by construction, and the `assert`
stayed.

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
            "ACCEPTED: dict[str, str] = {}",
            'ACCEPTED: dict[str, str] = {"planted gap": ""}',
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
    Plant(
        "tests/test_changelog_ledger.py",
        ("CHANGELOG.md",),
        lambda tree: _edit(
            tree,
            "CHANGELOG.md",
            "- fix: report a list that resizes under the walk, as a dict already is\n",
            "",
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
            "#[test]",
            "// SWEEP-SKIP: planted, and no --skip names it\n#[test]",
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
