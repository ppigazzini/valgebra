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
that breaks its claim can, so that is what this does: plant a defect a ledger
exists to catch in a throwaway copy of the tree, run the tests the plant names
there, and require each of them to fail. One plant proves the tests it names and
no others, so every test function of every ledger file is named by a plant or
excused with the reason no plant fits it. A ledger with no plant here is a ledger
nobody has shown to work.

The copy is the tracked files only, made once and repaired between rows, so a
plant cannot leak into the next one or into the tree being audited.

LEDGER: every ledger fails on the defect it exists to catch
"""

from __future__ import annotations

import ast
import datetime
import functools
import json
import os
import re
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
    trips: tuple[str, ...]
    """The ledger's test functions the plant must fail, each by name."""


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


def _replace_all(tree: Path, relative: str, old: str, new: str) -> None:
    """Replace every `old` in `relative`, refusing if there is none.

    For a drift that renames a word wherever a file carries it, which `_edit`
    refuses as ambiguous.
    """
    path = tree / relative
    text = path.read_text(encoding="utf-8")
    if old not in text:
        message = f"the plant did not land: {old[:60]!r} in {relative}"
        raise AssertionError(message)
    path.write_text(text.replace(old, new), encoding="utf-8")


def _delete(tree: Path, relative: str) -> None:
    """Remove `relative`, refusing if it is not there to remove."""
    path = tree / relative
    if not path.is_file():
        message = f"the plant did not land: {relative} is not a file"
        raise AssertionError(message)
    path.unlink()


def _move(tree: Path, old: str, new: str) -> None:
    """Move `old` to `new`, as a file moved without its readers is.

    Both paths go in the plant's `touches`: the repair copies `old` back and
    unlinks `new`, which the audited tree does not carry.
    """
    source, target = tree / old, tree / new
    if not source.is_file():
        message = f"the plant did not land: {old} is not a file"
        raise AssertionError(message)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)


#: A release tag the plant writes on a commit no branch reaches, and the loose
#: ref it leaves, which `touches` names so the repair deletes it.
PLANTED_TAG = "v0.0.99"


def _tag_an_orphan(tree: Path) -> None:
    """Tag a commit no branch reaches, the state a rewrite below a tag leaves."""
    written = _plant_git(tree, "write-tree").strip()
    orphan = _plant_git(tree, "commit-tree", written, "-m", "an orphan, planted")
    _plant_git(tree, "tag", PLANTED_TAG, orphan.strip())


#: Two values of one set compared by a lying `>=`: an `int` whose `__ge__`
#: answers `False` against zero, which no sound decision can foresee.
LYING_VALUE = '''
class _LiesAtZero(int):
    """An `int` whose `>=` refuses zero: a value the trust base excludes."""

    def __ge__(self, other: object) -> bool:
        return other != 0 and int.__ge__(self, other)


_SEED: list[Any] = [
    _LiesAtZero(5),
'''


def _move(tree: Path, old: str, new: str) -> None:
    """Move `old` to `new` in the copy, as a rename the rest of the tree missed.

    Both paths go in the plant's `touches`: the repair copies `old` back and
    unlinks `new`, which the audited tree does not carry.
    """
    source = tree / old
    if not source.is_file():
        message = f"the plant did not land: {old} is not in the copy"
        raise AssertionError(message)
    shutil.move(source, tree / new)


def _widen_the_inventory(tree: Path) -> None:
    """Give every row of the contract inventory a fourth cell.

    A column added to the table is how its parse comes to read nothing: the
    ledger takes a row of three cells, and every row then has four.
    """
    path = tree / "CONTRIBUTING.md"
    text = path.read_text(encoding="utf-8")
    head, found, rest = text.partition("## Contract inventory")
    if not found:
        message = "the plant did not land: CONTRIBUTING.md has no inventory"
        raise AssertionError(message)
    section, cut, tail = rest.partition("\n## ")
    widened = re.sub(r"^(\|.*\|)$", r"\1 planted |", section, flags=re.MULTILINE)
    if widened == section:
        message = "the plant did not land: the inventory has no table row"
        raise AssertionError(message)
    path.write_text(head + found + widened + cut + tail, encoding="utf-8")


def _empty_the_smoke_matrix(tree: Path) -> None:
    """Leave the release's smoke matrix with no row, as a job being rewritten does."""
    path = tree / ".github" / "workflows" / "release.yml"
    text = path.read_text(encoding="utf-8")
    emptied, count = re.subn(
        r"^        wheel:\n(?:^          .*\n)+(?=    runs-on: \$\{\{ matrix\.wheel)",
        "        wheel: []\n",
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        message = f"the plant did not land: {count} smoke matrices matched"
        raise AssertionError(message)
    path.write_text(emptied, encoding="utf-8")


#: Three tests of the budget, marked out of the sweep each with a reason.
_BUDGET_TESTS = (
    "an_exhausted_budget_refuses_to_spend",
    "the_budget_declines_on_every_subtyping_path",
    "a_budgeted_equivalence_query_decides_the_same_or_declines",
)


def _skip_three_more(tree: Path) -> None:
    """Mark three more tests out of the mutation sweep, each with its reason."""
    for name in _BUDGET_TESTS:
        _edit(
            tree,
            "crates/valgebra-core/src/decision/budget_tests.rs",
            f"#[test]\nfn {name}(",
            "// SWEEP-SKIP: planted, a bound this case proves outlasts the sweep\n"
            f"#[test]\nfn {name}(",
        )


#: Repository modules enough to outnumber the product half, with room for it to
#: grow: the suite holds 76 product modules against 53 repository ones.
_PLANTED_REPOSITORY_MODULES = tuple(
    f"tests/test_planted_repository_{index:02d}.py" for index in range(60)
)


def _outnumber_the_product_suite(tree: Path) -> None:
    """Add repository checks until they outnumber the product tests."""
    for name in _PLANTED_REPOSITORY_MODULES:
        _write(
            tree,
            name,
            '"""A planted repository check."""\n\n'
            "import pytest\n\n"
            "pytestmark = pytest.mark.repository\n",
        )


#: Written after the page's first guiding claim, which carries no holding line
#: of its own, so an `OWED:` there belongs to it and to nothing else.
_AFTER_A_GUIDING_CLAIM = "exact there ([02-decision.md](02-decision.md)).\n"


def _owe(tree: Path, line: str) -> None:
    """Add an `OWED:` line to the theory page's first guiding claim."""
    _edit(
        tree,
        "docs/dev/10-theory.md",
        _AFTER_A_GUIDING_CLAIM,
        f"{_AFTER_A_GUIDING_CLAIM}\n{line}\n",
    )


def _edits(tree: Path, relative: str, *pairs: tuple[str, str]) -> None:
    """Apply several `_edit`s to one file, in order, each refusing if absent."""
    for old, new in pairs:
        _edit(tree, relative, old, new)


def _rename_in(tree: Path, relatives: tuple[str, ...], old: str, new: str) -> None:
    """Replace every `old` in each of `relatives`, refusing a file without one.

    `_edit` replaces once, which is right for a line and wrong for a name: a
    helper renamed is renamed at every call site, and a plant that renamed one
    would plant a file that does not compile rather than the drift.
    """
    for relative in relatives:
        path = tree / relative
        text = path.read_text(encoding="utf-8")
        if old not in text:
            message = f"the plant did not land: {old!r} in {relative}"
            raise AssertionError(message)
        path.write_text(text.replace(old, new), encoding="utf-8")


#: The changelog's roll, as `tests/test_changelog_ledger.py` reads it.
_ROLL = re.compile(r"<!--\s*changelog-roll\s*\n(.*?)-->", re.DOTALL)


def _roll(tree: Path) -> tuple[str, int, list[tuple[int, int, bool]]]:
    """Read the changelog, where its roll starts, and each roll line.

    A line is `(start, end, names a commit)`, offsets into the text, where a
    line names a commit when its subject is a `feat`/`fix` commit since the
    release the page names. The range is read the way the ledger reads it, so a
    clone without the tag -- a shallow one, or the release window -- skips the
    plant rather than failing it: the ledger skips there too, and a plant
    against a ledger that judges nothing is the unjudgeable case.
    """
    text = (tree / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.search(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
    block = _ROLL.search(text)
    assert released is not None, "the changelog names no released version"
    assert block is not None, "the changelog carries no roll"
    try:
        log = _plant_git(tree, "log", "--format=%s", f"v{released.group(1)}..HEAD")
    except subprocess.CalledProcessError:
        pytest.skip("the tag the roll is measured from is not in this clone")
    visible = {
        subject
        for subject in log.splitlines()
        if re.match(r"(feat|fix)(\(|!|:)", subject)
    }
    lines = []
    at = block.start(1)
    for line in block.group(1).splitlines(keepends=True):
        entry = line.strip()
        if entry.startswith("- "):
            subject = entry[2:].split(" -- ")[0].strip()
            lines.append((at, at + len(line), subject in visible))
        at += len(line)
    return text, block.start(1), lines


def _unroll_a_commit(tree: Path) -> None:
    """Drop the roll line of the newest commit it names: a fix landed unrolled."""
    text, _, lines = _roll(tree)
    named = [(start, end) for start, end, names in lines if names]
    if not named:
        pytest.skip("the roll names no commit of this release, so none can go missing")
    start, end = named[-1]
    (tree / "CHANGELOG.md").write_text(text[:start] + text[end:], encoding="utf-8")


def _roll_a_renamed_subject(tree: Path) -> None:
    """Put a line naming no commit before one that does: a subject a rebase renamed.

    Before the last matching line, because a line after it is the pending tail
    -- the commit being made -- which the ledger reads as not stale on purpose.
    """
    text, _, lines = _roll(tree)
    named = [start for start, _, names in lines if names]
    if not named:
        pytest.skip("the roll names no commit of this release, so none can be stale")
    first = lines[0][0]
    stale = "- fix: a subject a rebase renamed\n"
    (tree / "CHANGELOG.md").write_text(
        text[:first] + stale + text[first:], encoding="utf-8"
    )


def _empty_the_roll(tree: Path) -> None:
    """Drop every roll line while the release has commits to roll."""
    text, _, lines = _roll(tree)
    if not any(names for _, _, names in lines):
        pytest.skip("nothing moved this release, so an empty roll is the right one")
    kept, cursor = [], 0
    for start, end, _ in lines:
        kept.append(text[cursor:start])
        cursor = end
    kept.append(text[cursor:])
    (tree / "CHANGELOG.md").write_text("".join(kept), encoding="utf-8")


#: The lifecycle ledger, its clock, and the rows its calendar reads.
_LIFECYCLE = "tests/test_python_lifecycle.py"


_NOW = "TODAY = datetime.datetime.now(datetime.timezone.utc).date()"


_RELEASE_ROW = re.compile(
    r'^    Release\((\d+), \d+, _day\("(\d{4}-\d{2}-\d{2})"\)', re.MULTILINE
)


def _newest_alpha(tree: Path) -> tuple[int, datetime.date]:
    """Give the newest row's minor release and its first alpha."""
    minor, day = _RELEASE_ROW.findall((tree / _LIFECYCLE).read_text(encoding="utf-8"))[
        -1
    ]
    return int(minor), datetime.date.fromisoformat(day)


def _set_the_clock(tree: Path, day: datetime.date) -> None:
    _edit(tree, _LIFECYCLE, _NOW, f'TODAY = datetime.date.fromisoformat("{day}")')


def _reach_the_next_alpha(tree: Path) -> None:
    """Move the clock to the newest row's first alpha, with no lane for it.

    The drift is the calendar itself: the day a release's first alpha lands, the
    `python` job has to run it forgiven, and no commit makes that day come. A leg
    the tree has already added is dropped, so the plant stands on either side of
    the edit that answers it.
    """
    minor, alpha = _newest_alpha(tree)
    _set_the_clock(tree, alpha)
    workflow = tree / ".github" / "workflows" / "ci.yml"
    text = workflow.read_text(encoding="utf-8")
    workflow.write_text(re.sub(rf', "3\.{minor}t?"', "", text), encoding="utf-8")


def _outlive_the_table(tree: Path) -> None:
    """Move the clock past the day the release after the newest row reaches alpha."""
    _, alpha = _newest_alpha(tree)
    _set_the_clock(tree, alpha + datetime.timedelta(days=373))


def _shift_a_floor_statement(tree: Path) -> None:
    """Raise ruff's target by one release, leaving every other floor where it is."""
    path = tree / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    found = re.search(r'^target-version = "py3(\d+)"$', text, re.MULTILINE)
    assert found is not None, "the plant did not land: ruff's target-version"
    raised = f'target-version = "py3{int(found.group(1)) + 1}"'
    path.write_text(text.replace(found.group(0), raised), encoding="utf-8")


def _lower_the_newest_statement(tree: Path) -> None:
    """Lower ty's environment by one release, as a bump that missed it leaves it."""
    path = tree / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    found = re.search(r'^python-version = "3\.(\d+)"$', text, re.MULTILINE)
    assert found is not None, "the plant did not land: ty's python-version"
    lowered = f'python-version = "3.{int(found.group(1)) - 1}"'
    path.write_text(text.replace(found.group(0), lowered), encoding="utf-8")


_REFERENCE = "scripts/metamorphic_reference.json"


def _record_outside_a_checkout(tree: Path) -> None:
    """Write the reference's commit as a recording outside a checkout writes it."""
    path = tree / _REFERENCE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["commit"] = "unknown"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _prove_nothing(tree: Path) -> None:
    """Record every decision unproven: a corpus asking a question with one answer."""
    path = tree / _REFERENCE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decisions"] = dict.fromkeys(payload["decisions"], "n")
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


#: The frontend files a refusal is written through the `not_implemented` helper.
_REFUSING = (
    "crates/valgebra-py/src/build.rs",
    "crates/valgebra-py/src/build/generics.rs",
    "crates/valgebra-py/src/build/refine.rs",
    "crates/valgebra-py/src/build/classes.rs",
)


#: The fuzz soak's floor, the comparison the soak's `RIG FAULT` exit rests on.
_SOAK_FLOOR = (
    '          if [ "$rate" -le 0 ] || [ "$seconds" -lt "$MIN_SECONDS" ]; then\n'
    '            echo "RIG FAULT: ${seconds}s at ${rate} exec/s is beneath the" >&2\n'
    '            echo "floor: this run found nothing because it ran nothing." >&2\n'
    "            exit 2\n"
    "          fi\n"
)


def _import_an_unshipped_module(tree: Path) -> None:
    """Import a module from a test file, and date it as shipped from the floor."""
    _edit(
        tree,
        "tests/floor_names.json",
        '    "base64": {\n      "since": "3.10"\n    },',
        '    "_planted_stdlib": {\n      "since": "3.10"\n    },\n'
        '    "base64": {\n      "since": "3.10"\n    },',
    )
    _write(tree, "tests/_planted_import.py", "import _planted_stdlib\n")


#: The linux x86_64 build row's interpreter list, as `release.yml` spells it.
LINUX_ROW = re.compile(r"(target: x86_64, pgo: true, interpreter: )(python3\.\d+ )")


def _release_yml(tree: Path) -> Path:
    return tree / ".github" / "workflows" / "release.yml"


def _drop_a_release(tree: Path) -> None:
    """Take the first release off the linux x86_64 row, whichever it is."""
    path = _release_yml(tree)
    text, count = LINUX_ROW.subn(r"\1", path.read_text(encoding="utf-8"), count=1)
    assert count == 1, "the plant did not land: no linux x86_64 build row"
    path.write_text(text, encoding="utf-8")


def _build_an_unclassified_release(tree: Path) -> None:
    """Add a release no classifier names to the linux x86_64 row."""
    path = _release_yml(tree)
    text, count = LINUX_ROW.subn(
        r"\1python3.99 \2", path.read_text(encoding="utf-8"), count=1
    )
    assert count == 1, "the plant did not land: no linux x86_64 build row"
    path.write_text(text, encoding="utf-8")


def _unclassify_every_release(tree: Path) -> None:
    """Drop every `Python :: 3.N` classifier, leaving the ledger nothing to read."""
    path = tree / "pyproject.toml"
    text, count = re.subn(
        r'^    "Programming Language :: Python :: 3\.\d+",\n',
        "",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert count, "the plant did not land: no release classifier"
    path.write_text(text, encoding="utf-8")


#: The marker the bound ledger reads, spelled from its pieces: the ledger reads
#: every test file, this one included, and would take a plant's marker for a
#: test driving the bound.
BOUND_MARKER = "# " + "BOUND"


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
        # A package the suite reads, gone from the PyPy leg's list: the rows
        # reading it skip there, and nothing else in the lane goes red.
        "tests/test_suite_installs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            ' "typing-extensions>=4.16.0"\n      - name: pytest',
            "\n      - name: pytest",
        ),
        trips=("test_every_hand_written_install_names_what_the_suite_reads",),
    ),
    Plant(
        # A seed file at the path of a source file with no property test in it,
        # which is where a test module's seeds stay when the module moves.
        # Every test passes, and the seeds are replayed by nothing.
        "tests/test_proptest_seeds.py",
        ("crates/valgebra-core/proptest-regressions/lib.txt",),
        lambda tree: _write(
            tree,
            "crates/valgebra-core/proptest-regressions/lib.txt",
            "cc 0000000000000000000000000000000000000000000000000000000000000000"
            " # planted\n",
        ),
        trips=("test_every_seed_file_sits_beside_a_property_test",),
    ),
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
        trips=("test_no_inline_test_module_is_longer_than_a_screen",),
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
        trips=("test_every_crate_root_forbids_unsafe_code",),
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
        trips=("test_no_source_file_writes_unsafe",),
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
        trips=(
            "test_every_use_case_is_named_by_the_suite_or_accepted",
            "test_the_count_is_reported",
            "test_the_lane_prints_a_figure_for_each_product",
            "test_the_page_carries_the_figure_each_product_has",
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
        trips=("test_every_declared_bound_is_driven_or_accepted",),
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
        trips=(
            "test_every_held_test_carries_the_marker",
            "test_every_named_test_addresses_one_definition",
            "test_every_named_test_exists",
            "test_the_names_are_test_functions_rather_than_helpers",
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
        trips=("test_every_source_line_is_one_this_ledger_can_read",),
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
            " arm returns before the budget is read."
            " **[OBLIGATION: a-planted-obligation]**\n",
        ),
        trips=("test_every_claim_names_a_test_held_or_owed",),
    ),
    Plant(
        # A claim tagged with no id: no test can name it, so the parse refuses
        # the page, and a test says so rather than the module's collection.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/10-theory.md",
            "### Obligations\n",
            "### Obligations\n\n**A planted obligation nothing names.** Every"
            " arm returns before the budget is read. **[OBLIGATION]**\n",
        ),
        trips=("test_the_page_parses",),
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
        trips=("test_every_variant_is_a_generator_a_representative_or_a_marker",),
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
        trips=("test_decision_decides_true_relations",),
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
        trips=(
            "test_every_ledger_entry_carries_a_reason",
            "test_no_ledger_entry_is_stale",
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
        trips=("test_every_manifest_is_a_member_or_detached_with_a_reason",),
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
        trips=("test_no_violation_is_built_from_a_code_spelled_out",),
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
        trips=(
            "test_every_example_block_asserts_what_it_shows",
            "test_every_example_in_a_tracked_page_is_run_or_marked",
            "test_the_runner_reaches_every_example_the_lane_reports",
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
        trips=("test_every_gate_has_an_enforced_lane_on_each_side",),
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
        trips=("test_every_feature_is_run_by_a_lane_that_is_not_a_measurement",),
    ),
    Plant(
        "tests/test_lane_coverage.py",
        ("scripts/planted_gate.py",),
        lambda tree: _write(
            tree, "scripts/planted_gate.py", '"""A gate in no lane."""\n'
        ),
        trips=("test_every_script_runs_in_a_lane",),
    ),
    Plant(
        "tests/test_contract_inventory.py",
        ("scripts/planted_gate.py",),
        lambda tree: _write(
            tree, "scripts/planted_gate.py", '"""A gate with no contract row."""\n'
        ),
        trips=("test_every_gate_script_has_a_row",),
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
        trips=(
            "test_the_leg_that_takes_the_history_says_so_in_its_name",
            "test_the_workflow_runs_this_ledger_with_full_history",
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
        trips=("test_every_merge_gate_step_is_planned_or_excused_by_name",),
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
        trips=("test_the_gate_builds_the_floor_interpreter_the_matrix_names",),
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
        trips=("test_the_leg_that_takes_the_history_says_so_in_its_name",),
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
        trips=(
            "test_a_job_left_out_whole_is_one_the_workflow_has_and_says_why",
            "test_a_job_named_unreached_is_one_the_workflow_has",
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
        trips=("test_every_job_is_required_by_the_gate",),
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
        trips=("test_the_matrix_blocks_on_every_supported_release_and_no_other",),
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
        trips=("test_no_lane_lets_the_image_choose_its_interpreter",),
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
        trips=("test_only_the_argued_tests_read_a_clock",),
    ),
    Plant(
        "tests/test_commit_messages.py",
        (),
        # The subject is a message rather than a file, and the stage is a real
        # clone, so the plant writes one: the tip's message gains a name from
        # the working area. Nothing in the tree changes, which is why this row
        # touches no path; the repair puts the tip back, as it does for every row.
        _name_the_working_area,
        trips=("test_no_commit_message_names_the_working_area",),
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
        trips=("test_the_soak_names_its_allocation_ceiling",),
    ),
    Plant(
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(tree, ".github/workflows/ci.yml", "jobs:\n", UNREQUIRED_JOB),
        trips=("test_every_job_is_required_by_the_gate",),
    ),
    Plant(
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        # A sweep cut wider in its matrix and not in its divisor: the shards past
        # the old count select no mutant and pass having swept nothing.
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            '20 --shard "${{ matrix.shard }}/24"',
            '20 --shard "${{ matrix.shard }}/12"',
        ),
        trips=("test_a_sharded_sweep_covers_every_shard",),
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
        trips=("test_every_marked_test_is_actually_skipped",),
    ),
    Plant(
        "tests/test_suite_partition.py",
        ("tests/test_planted.py",),
        lambda tree: _write(
            tree,
            "tests/test_planted.py",
            "def test_planted() -> None:\n    assert True\n",
        ),
        trips=("test_every_test_file_falls_on_one_side_of_the_line",),
    ),
    Plant(
        "tests/test_mutation_scope.py",
        ("crates/valgebra-py/src/planted.rs",),
        lambda tree: _write(
            tree,
            "crates/valgebra-py/src/planted.rs",
            "pub fn planted() -> u8 {\n    1\n}\n",
        ),
        trips=("test_every_binding_file_is_swept_or_excluded_by_name",),
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
        trips=("test_every_feature_site_is_test_only_or_named",),
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
        trips=("test_no_module_reads_a_name_outside_the_releases_that_have_it",),
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
        trips=(
            "test_a_raised_error_is_a_verdict_like_any_other",
            "test_a_value_that_moved_fails",
            "test_a_value_that_moved_the_other_way_fails",
        ),
    ),
    Plant(
        "tests/test_cited_commits.py",
        ("scripts/metamorphic_reference.json",),
        _cite_an_orphan,
        trips=("test_every_cited_commit_is_reachable",),
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
        trips=("test_every_declared_name_is_put_through_assert_type",),
    ),
    Plant(
        "tests/test_checker_readings.py",
        ("python/valgebra/_valgebra.pyi",),
        # The failure the ledger is for: a reading that moves and nobody
        # notices. Planted from the stub's side, which is how the tree causes
        # it: the typed `union` overload goes, and ty's reading of two typed
        # validators falls from their union to `object`.
        lambda tree: _edit(
            tree,
            "python/valgebra/_valgebra.pyi",
            "@overload\ndef union(*schemas: Validator[_S]) -> Validator[_S]: ..."
            "  # pyright: ignore[reportOverlappingOverload]\n"
            "@overload\ndef union(*schemas: object)",
            "def union(*schemas: object)",
        ),
        trips=("test_a_reading_is_the_one_recorded",),
    ),
    Plant(
        "tests/test_doc_example_checkers.py",
        ("README.md",),
        # The failure the ledger is for: a diagnostic on a published example
        # that no row expects. Planted as the shape it arrives in -- an import
        # a later edit of the block left unused.
        lambda tree: _edit(
            tree,
            "README.md",
            "from valgebra import ValidationError, Validator\n",
            "import os\n\nfrom valgebra import ValidationError, Validator\n",
        ),
        trips=("test_every_diagnostic_is_expected_and_every_row_reported",),
    ),
    Plant(
        "tests/test_release_smoke.py",
        (".github/workflows/release.yml",),
        # The failure the ledger is for: a wheel set built and never run. Planted
        # by pointing the PyPy smoke row at the CPython set, which is how it
        # arrives -- a copied row whose artifact name nobody edited.
        lambda tree: _edit(
            tree,
            ".github/workflows/release.yml",
            "artifact: wheels-ubuntu-latest-x86_64-pypy73,",
            "artifact: wheels-ubuntu-latest-x86_64-native,",
        ),
        trips=(
            "test_every_built_wheel_set_is_smoked_or_excused",
            "test_pypy_is_built_plain",
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
        trips=("test_a_lane_excludes_every_corpus_file_of_the_crate_it_measures",),
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
        trips=("test_every_file_excused_to_pytest_is_examined_there",),
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
        trips=("test_every_numbered_result_is_attributed_to_a_work",),
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
        trips=(
            "test_every_held_test_carries_the_marker",
            "test_every_marked_test_is_one_its_claim_names",
            "test_every_marker_names_a_claim_the_page_carries",
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
        trips=("test_every_documented_return_is_asserted_somewhere",),
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
        trips=("test_every_tabulated_form_has_a_row_and_every_row_a_form",),
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
        trips=(
            "test_every_row_is_the_kind_its_list_calls_for",
            "test_the_universe_is_the_page_in_both_directions",
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
        trips=(
            "test_every_cell_narrows_its_base_or_is_refused",
            "test_the_universe_is_the_two_enums_in_both_directions",
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
        trips=("test_the_universe_is_the_ir_variants_in_both_directions",),
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
            '"{} is a generic Protocol, which names one set per type argument: \\',
            '"{} names no set: \\',
        ),
        trips=("test_every_refusal_is_matched_by_a_test_or_accepted",),
    ),
    Plant(
        # An assumption typed without its bolded lead-in: the trust base keeps
        # the sentence and the marker scan stops seeing it as an assumption.
        "tests/test_boundary_ledger.py",
        ("docs/14-soundness.md",),
        lambda tree: _edit(
            tree,
            "docs/14-soundness.md",
            "- **The crates contain no `unsafe`.** Both crate roots carry",
            "- The crates contain no `unsafe`. Both crate roots carry",
        ),
        trips=("test_every_assumption_leads_with_the_sentence_it_assumes",),
    ),
    Plant(
        # An assumption added to the trust base with no test naming it: a
        # sentence the suite cannot be wrong about.
        "tests/test_boundary_ledger.py",
        ("docs/14-soundness.md",),
        lambda tree: _edit(
            tree,
            "docs/14-soundness.md",
            "- **Predicate refinements are opaque.**",
            "- **A planted assumption no test shows the cost of.** Planted.\n"
            "- **Predicate refinements are opaque.**",
        ),
        trips=("test_every_assumption_names_a_test_that_shows_its_cost",),
    ),
    Plant(
        # An assumption's sentence rewritten on the page, which leaves the
        # `# TRUST:` marker carrying the old wording and naming nothing.
        "tests/test_boundary_ledger.py",
        ("docs/14-soundness.md",),
        lambda tree: _edit(
            tree,
            "docs/14-soundness.md",
            "- **The crates contain no `unsafe`.**",
            "- **No crate contains `unsafe`.**",
        ),
        trips=("test_every_trust_marker_names_an_assumption_the_page_states",),
    ),
    Plant(
        # A decided entry whose row is written the wrong way round: the page
        # promises an answer the procedure does not give for that pair.
        "tests/test_boundary_ledger.py",
        ("tests/test_boundary_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_boundary_ledger.py",
            '"Sets and frozensets.": Decides(set[bool], set[int], "subset"),',
            '"Sets and frozensets.": Decides(set[int], set[bool], "subset"),',
        ),
        trips=("test_every_entry_of_the_boundary_answers_as_the_page_says",),
    ),
    Plant(
        # A cache-key pattern narrowed to the top of each crate's sources: every
        # nested module becomes an input uv reinstalls the previous build over.
        "tests/test_build_surfaces.py",
        ("pyproject.toml",),
        lambda tree: _edit(
            tree,
            "pyproject.toml",
            '    { file = "crates/**/*.rs" },',
            '    { file = "crates/*/src/*.rs" },',
        ),
        trips=("test_every_build_input_is_covered_by_a_uv_cache_key",),
    ),
    Plant(
        # A detached surface whose reason is a word: a hole in every local gate
        # recorded with no argument for it.
        "tests/test_build_surfaces.py",
        ("tests/test_build_surfaces.py",),
        lambda tree: _edit(
            tree,
            "tests/test_build_surfaces.py",
            '"why": (\n'
            '            "libFuzzer needs a nightly toolchain and the sanitizer'
            ' flags; making "\n'
            '            "it a workspace member would put nightly on the stable'
            " gates' path.\"\n"
            "        ),",
            '"why": "nightly",',
        ),
        trips=("test_every_detached_surface_carries_its_reason",),
    ),
    Plant(
        # The lane's build command rewritten, so no workflow runs the command
        # the detached entry names: a crate nothing compiles.
        "tests/test_build_surfaces.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "cargo +${{ env.FUZZ_NIGHTLY }} fuzz build --target",
            "cargo +${{ env.FUZZ_NIGHTLY }} fuzz check --target",
        ),
        trips=("test_every_detached_surface_is_built_by_a_lane",),
    ),
    Plant(
        # A cache-key pattern with a typo, which matches nothing and reads as
        # coverage.
        "tests/test_build_surfaces.py",
        ("pyproject.toml",),
        lambda tree: _edit(
            tree,
            "pyproject.toml",
            '    { file = "crates/**/*.rs" },',
            '    { file = "crates/**/*.rs" },\n    { file = "crates/**/*.rss" },',
        ),
        trips=("test_no_cache_key_pattern_is_dead",),
    ),
    Plant(
        # The detached crate removed, its entry left behind naming nothing.
        "tests/test_build_surfaces.py",
        ("fuzz/Cargo.toml",),
        lambda tree: _delete(tree, "fuzz/Cargo.toml"),
        trips=("test_no_detached_entry_is_stale",),
    ),
    Plant(
        # uv's default key dropped from an explicit list, which replaces the
        # default rather than adding to it.
        "tests/test_build_surfaces.py",
        ("pyproject.toml",),
        lambda tree: _edit(
            tree, "pyproject.toml", '    { file = "pyproject.toml" },\n', ""
        ),
        trips=("test_pyproject_is_named_among_the_cache_keys",),
    ),
    Plant(
        # The contributor gate losing the line that compiles the detached
        # crate: a local run passes a change that breaks the fuzz lane. Trips
        # only with the ledger reading the gate's block rather than the whole
        # page, where the contract inventory's row still names the command.
        "tests/test_build_surfaces.py",
        ("CONTRIBUTING.md",),
        lambda tree: _edit(
            tree,
            "CONTRIBUTING.md",
            "cargo test\ncargo check --manifest-path fuzz/Cargo.toml --all-targets\n",
            "cargo test\n",
        ),
        trips=("test_the_local_gate_names_every_detached_surface",),
    ),
    Plant(
        # The page's section references respelled in words: the parse reads no
        # section citation, and every rule over them passes on none.
        "tests/test_citation_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _replace_all(tree, "docs/dev/10-theory.md", "§", "section "),
        trips=("test_the_page_cites_numbered_results",),
    ),
    Plant(
        # A release tag on a commit no branch reaches: what a rewrite below the
        # tag leaves, with every diff of the tree unchanged.
        "tests/test_cited_commits.py",
        (f".git/refs/tags/{PLANTED_TAG}",),
        _tag_an_orphan,
        trips=("test_every_release_tag_is_an_ancestor_of_this_branch",),
    ),
    Plant(
        # JSON skipped as content hashes: the reference corpus, the file whose
        # commit is most exposed to a rewrite, drops out of the scan.
        "tests/test_cited_commits.py",
        ("tests/test_cited_commits.py",),
        lambda tree: _edit(
            tree,
            "tests/test_cited_commits.py",
            '_NOT_COMMITS = ("uv.lock", "Cargo.lock", ".png", ".ico", ".svg")',
            '_NOT_COMMITS = ("uv.lock", "Cargo.lock", ".png", ".ico", ".svg", ".json")',
        ),
        trips=("test_the_scan_reads_the_tree_at_all",),
    ),
    Plant(
        # An argued test renamed: its argument outlives it and excuses nothing.
        "tests/test_clock_ledger.py",
        ("tests/test_equivalence.py",),
        lambda tree: _edit(
            tree,
            "tests/test_equivalence.py",
            "def test_two_wide_literal_sets_are_decided_as_sets_not_pair_by_pair(",
            "def test_wide_literal_sets_are_decided_as_sets_not_pair_by_pair(",
        ),
        trips=("test_every_argued_test_exists_and_reads_a_clock",),
    ),
    Plant(
        # A code declared in the table and reported nowhere.
        "tests/test_code_table.py",
        ("crates/valgebra-py/src/codes.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/codes.rs",
            'pub(crate) const DICT_TYPE: Code = Code("dict_type");',
            'pub(crate) const DICT_TYPE: Code = Code("dict_type");\n'
            'pub(crate) const PLANTED_ERROR: Code = Code("planted_error");',
        ),
        trips=("test_every_name_the_table_declares_is_written_somewhere",),
    ),
    Plant(
        # The table moved to another file: every rule above reads an empty
        # table and passes.
        "tests/test_code_table.py",
        ("crates/valgebra-py/src/codes.rs", "crates/valgebra-py/src/code.rs"),
        lambda tree: _move(
            tree, "crates/valgebra-py/src/codes.rs", "crates/valgebra-py/src/code.rs"
        ),
        trips=("test_the_binding_names_its_codes_in_one_table",),
    ),
    Plant(
        # The derivation going back to a path scan beside the table.
        "tests/test_code_table.py",
        ("scripts/use_case_ledger.py",),
        lambda tree: _edit(
            tree,
            "scripts/use_case_ledger.py",
            "\nCODE_TABLE = ",
            '\n_EMITTERS = ("crates/valgebra-py/src/check/walk.rs",)\nCODE_TABLE = ',
        ),
        trips=("test_the_derived_universe_is_the_two_tables",),
    ),
    Plant(
        # The value universe capped, the thinness the seed's comment warns of:
        # fewer values than the probe's floor.
        "tests/test_completeness_probe.py",
        ("tests/test_completeness_probe.py",),
        lambda tree: _edit(
            tree,
            "tests/test_completeness_probe.py",
            "VALUES: list[Any] = _universe(SCHEMAS)",
            "VALUES: list[Any] = _universe(SCHEMAS)[:40]",
        ),
        trips=(
            "test_the_probe_actually_compared_something",
            "test_a_refutation_names_a_value_outside",
        ),
    ),
    Plant(
        # A suspected gap's entry deleted while the gap stands.
        "tests/test_completeness_probe.py",
        ("tests/test_completeness_probe.py",),
        lambda tree: _edit(
            tree,
            "tests/test_completeness_probe.py",
            '    "Plain <= ~int": _OPEN_WORLD.format(kind="`int`"),\n',
            "",
        ),
        trips=("test_every_suspected_gap_is_on_the_ledger",),
    ),
    Plant(
        # A value that refutes a decided inclusion. A sound procedure has none,
        # so the plant is a value the trust base excludes -- an `int` whose `>=`
        # lies -- and both routes that decide `Ge(1) <= Ge(0)` are refuted by it.
        "tests/test_completeness_probe.py",
        ("tests/test_completeness_probe.py",),
        lambda tree: _edit(
            tree,
            "tests/test_completeness_probe.py",
            "\n_SEED: list[Any] = [\n",
            LYING_VALUE,
        ),
        trips=(
            "test_no_decided_relation_is_refuted_by_a_value",
            "test_neither_decider_is_unsound_where_they_disagree",
        ),
    ),
    Plant(
        # The list of accepted gaps grown past what a review reads.
        "tests/test_completeness_probe.py",
        ("tests/test_completeness_probe.py",),
        lambda tree: _edit(
            tree,
            "tests/test_completeness_probe.py",
            "ACCEPTED: dict[str, str] = {\n",
            "ACCEPTED: dict[str, str] = {\n"
            + "".join(
                f'    "planted {n}": "an accepted gap planted to grow the list",\n'
                for n in range(5)
            ),
        ),
        trips=("test_the_ledger_is_serialisable_for_a_report",),
    ),
    Plant(
        # A fixpoint schema added to the universe, bringing a disagreement the
        # recorded count does not carry.
        "tests/test_completeness_probe.py",
        ("tests/test_completeness_probe.py",),
        lambda tree: _edit(
            tree,
            "tests/test_completeness_probe.py",
            '    ("mu t.None|{a:int,next:t}", _LINKED),\n',
            '    ("mu t.None|{a:int,next:t}", _LINKED),\n'
            '    ("{a:int,next:mu t}", _v({"a": int, "next": _LINKED})),\n',
        ),
        trips=("test_the_two_deciders_are_measured_against_each_other",),
    ),
    Plant(
        # A row written to the numeric tower: `float` admitting an `int`, which
        # the frontend does not read and the page does not say.
        "tests/test_form_ledger.py",
        ("tests/test_form_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_form_ledger.py",
            '"`float`": Reads(float, 1.5, 1),',
            '"`float`": Reads(float, 1, "a"),',
        ),
        trips=("test_a_form_the_tables_read_admits_and_refuses",),
    ),
    Plant(
        # A refusal the row words differently from the frontend.
        "tests/test_form_ledger.py",
        ("tests/test_form_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_form_ledger.py",
            '"a tuple literal": Refuses(lambda: (int, str), '
            '"a tuple literal is not a schema"),',
            '"a tuple literal": Refuses(lambda: (int, str), '
            '"a tuple is not a schema"),',
        ),
        trips=("test_a_form_the_table_refuses_says_so_at_build",),
    ),
    Plant(
        # A form read as a literal of itself, stood in for by the row's spec:
        # the fallback admits the annotation and nothing a caller holds.
        "tests/test_form_ledger.py",
        ("tests/test_form_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_form_ledger.py",
            '"`float`": Reads(float, 1.5, 1),',
            '"`float`": Reads(1.5, 2.5, 1),',
        ),
        trips=("test_a_read_form_is_not_a_literal_of_itself",),
    ),
    Plant(
        # The markers table's header reworded, so the parse reads one page.
        "tests/test_form_ledger.py",
        ("docs/05-refinements.md",),
        lambda tree: _edit(
            tree,
            "docs/05-refinements.md",
            "| Marker | Constraint | Failure code |",
            "| Refinement | Constraint | Failure code |",
        ),
        trips=("test_the_universe_is_the_pages_own_tables",),
    ),
    Plant(
        # An excuse left for a script that is gone.
        "tests/test_lane_coverage.py",
        ("tests/test_lane_coverage.py",),
        lambda tree: _edit(
            tree,
            "tests/test_lane_coverage.py",
            "EXCUSED: dict[str, str] = {}",
            'EXCUSED: dict[str, str] = {"planted_gone.py": "a script removed '
            'long ago, and an excuse nobody deleted"}',
        ),
        trips=("test_no_excuse_is_stale",),
    ),
    Plant(
        # The wrapper naming a default interpreter, which every lane naming
        # none inherits.
        "tests/test_lane_interpreters.py",
        (".github/actions/setup-uv/action.yml",),
        lambda tree: _edit(
            tree,
            ".github/actions/setup-uv/action.yml",
            '      one itself.\n    required: false\n    default: ""\n',
            '      one itself.\n    required: false\n    default: "3.12"\n',
        ),
        trips=("test_the_wrapper_still_defaults_to_letting_uv_choose",),
    ),
    Plant(
        # A classifier stating an implementation no lane runs the suite on.
        "tests/test_lane_interpreters.py",
        ("pyproject.toml",),
        lambda tree: _edit(
            tree,
            "pyproject.toml",
            '    "Programming Language :: Python :: Implementation :: PyPy",\n',
            '    "Programming Language :: Python :: Implementation :: PyPy",\n'
            '    "Programming Language :: Python :: Implementation :: GraalPy",\n',
        ),
        trips=("test_every_stated_implementation_runs_the_suite_somewhere",),
    ),
    Plant(
        # A classifier dropped while its lane keeps running.
        "tests/test_lane_interpreters.py",
        ("pyproject.toml",),
        lambda tree: _edit(
            tree,
            "pyproject.toml",
            '    "Programming Language :: Python :: Implementation :: PyPy",\n',
            "",
        ),
        trips=("test_no_lane_runs_the_suite_on_an_implementation_nobody_states",),
    ),
    Plant(
        # A gate script answering "could not run" with a code of its own.
        "tests/test_local_gate.py",
        ("scripts/coverage_gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/coverage_gate.py",
            "EXIT_CANNOT_RUN = 2",
            "EXIT_CANNOT_RUN = 3",
        ),
        trips=("test_a_gate_script_answers_in_the_three_code_vocabulary",),
    ),
    Plant(
        # The offline substitution dropped: the plan runs the online form.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            "            if name in NETWORK:\n"
            "                filled = NETWORK[name][1]\n",
            "",
        ),
        trips=("test_a_network_step_runs_its_offline_form",),
    ),
    Plant(
        # An excuse recorded with no reason.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            '    "Name the merge base": "reads the event payload the runner provides",',
            '    "Name the merge base": "",',
        ),
        trips=("test_every_excuse_carries_a_reason",),
    ),
    Plant(
        # A merge-gate job added with no `run:` step: no step to excuse, no
        # line in `UNREACHED`, and a closing line that reads complete.
        "tests/test_local_gate.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edits(
            tree,
            ".github/workflows/ci.yml",
            ("        zizmor,\n", "        zizmor,\n        planted-job,\n"),
            (
                "\n  ci:\n",
                (
                    "\n  planted-job:\n    runs-on: ubuntu-latest\n    steps:\n"
                    "      - uses: actions/checkout@v4\n\n  ci:\n"
                ),
            ),
        ),
        trips=("test_every_merge_gate_job_is_planned_excused_or_named_unreached",),
    ),
    Plant(
        # A step renamed in the workflow while its offline row keeps the name.
        "tests/test_local_gate.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "name: Audit the fuzz workspace dependency tree",
            "name: Audit the fuzz crate's dependency tree",
        ),
        trips=("test_every_network_row_stands_for_a_step_the_gate_runs",),
    ),
    Plant(
        # A stand-in written for a step the gate already runs.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            "STANDINS: dict[str, tuple[str, str]] = {}",
            'STANDINS: dict[str, tuple[str, str]] = {"pytest": ("uv run --no-sync '
            'pytest -q", "nothing: the gate runs the step")}',
        ),
        trips=("test_every_stand_in_is_for_a_step_the_gate_excuses",),
    ),
    Plant(
        # An excused step renamed in the workflow, its excuse left under the
        # old name.
        "tests/test_local_gate.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "name: Record the competitive ratios",
            "name: Record the ratios against pydantic",
        ),
        trips=("test_no_excuse_has_outlived_its_step",),
    ),
    Plant(
        # A stand-in whose command a planned step runs outright, the shape the
        # corpora left when they gained a lane of their own.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            "STANDINS: dict[str, tuple[str, str]] = {}",
            "STANDINS: dict[str, tuple[str, str]] = {\n"
            '    "Measure binding coverage via the Python suite and Rust unit tests"'
            ": (\n"
            '        "cargo test -p valgebra-py --features interpreter-tests",\n'
            '        "the coverage figure",\n'
            "    ),\n"
            "}",
        ),
        trips=("test_no_stand_in_repeats_a_step_the_gate_runs",),
    ),
    Plant(
        # The floor environment pointed at the caller's, the reverted experiment.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            'venv = ROOT / "target" / f"gate-floor-{floor}"',
            'venv = ROOT / ".venv"',
        ),
        trips=("test_the_floor_build_does_not_write_the_callers_environment",),
    ),
    Plant(
        # `--allow-existing` dropped: every run after the first fails there.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            'f"uv venv --allow-existing --python {shlex.quote(floor)} "',
            'f"uv venv --python {shlex.quote(floor)} "',
        ),
        trips=("test_the_floor_build_runs_a_second_time",),
    ),
    Plant(
        # The floor running the whole suite rather than the product one.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            "\"uv run --no-sync pytest -q -p no:cacheprovider -m 'not repository'\",",
            '"uv run --no-sync pytest -q -p no:cacheprovider",',
        ),
        trips=("test_the_floor_interpreter_runs_the_product_suite",),
    ),
    Plant(
        # The clone taken whole, which is the local clone the gate replaces.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            '            "--depth",\n            "1",\n            "--no-tags",\n',
            '            "--no-tags",\n',
        ),
        trips=("test_the_gate_runs_in_a_shallow_clone_with_no_tags",),
    ),
    Plant(
        # The base read as the remote's tip rather than the merge base: on a
        # branch behind its remote, two unrelated trees are compared.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            '(resolved := git_output("merge-base", ref, "HEAD", cwd=tree))',
            '(resolved := git_output("rev-parse", ref, cwd=tree))',
        ),
        trips=("test_the_instruction_gate_measures_against_an_ancestor",),
    ),
    Plant(
        # A mode spelled as the measurer does not define it.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree, "scripts/gate.py", '"--decision-matrix",', '"--decision-matrices",'
        ),
        trips=("test_the_instruction_gate_names_modes_the_measurer_has",),
    ),
    Plant(
        # The instruction gate dropped from the plan without a word: neither run
        # nor excused, on a machine with valgrind or without.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            '    """Give the instruction-gate steps to run, and the ones excused'
            ' with a reason."""\n',
            '    """Give the instruction-gate steps to run, and the ones excused'
            ' with a reason."""\n    return [], []\n',
        ),
        trips=("test_the_instruction_gate_runs_here_or_says_why",),
    ),
    Plant(
        # The corpora's feature renamed in the lane that runs them, so no
        # planned step links the embedded interpreter.
        "tests/test_local_gate.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "timeout 300 cargo test -p valgebra-py --features interpreter-tests",
            "timeout 300 cargo test -p valgebra-py --features interpreter-test",
        ),
        trips=("test_the_interpreter_backed_binding_tests_are_in_the_plan",),
    ),
    Plant(
        # The suite excused as a runner's step.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/gate.py",
            "NEEDS_A_RUNNER = {\n",
            'NEEDS_A_RUNNER = {\n    "pytest": "the whole suite, minutes per run",\n',
        ),
        trips=("test_the_python_suite_is_one_of_the_steps_it_runs",),
    ),
    Plant(
        # A step handed the runner's environment without the interpreter the
        # workspace tests link.
        "tests/test_local_gate.py",
        ("scripts/gate.py",),
        lambda tree: _edit(
            tree, "scripts/gate.py", "        **interpreter_env(),\n", ""
        ),
        trips=(
            "test_the_workspace_test_step_is_handed_the_interpreter_on_the_loader_path",
        ),
    ),
    Plant(
        # The sweep wrapper moved out of `tests/`: no integration target left.
        "tests/test_mutation_scope.py",
        ("crates/valgebra-py/tests/pytest_sweep.rs",),
        lambda tree: _delete(tree, "crates/valgebra-py/tests/pytest_sweep.rs"),
        trips=("test_an_integration_target_is_not_a_subject",),
    ),
    Plant(
        # The by-hand recipes losing the shrink bound the lanes set.
        "tests/test_mutation_scope.py",
        ("docs/dev/07-tooling-ci.md",),
        lambda tree: _replace_all(
            tree,
            "docs/dev/07-tooling-ci.md",
            "export PROPTEST_MAX_SHRINK_TIME=1000\n",
            "",
        ),
        trips=("test_every_variable_a_sweep_lane_sets_is_one_the_recipe_sets",),
    ),
    Plant(
        # An accepted survivor whose function is gone.
        "tests/test_mutation_scope.py",
        ("scripts/mutation_baseline.json",),
        lambda tree: _edit(
            tree,
            "scripts/mutation_baseline.json",
            '  "_accepted": {\n',
            '  "_accepted": {\n'
            '    "crates/valgebra-core/src/planted.rs: replace planted -> u8 with 0":'
            ' "a survivor of a function the tree no longer has",\n',
        ),
        trips=("test_no_accepted_survivor_has_outlived_its_subject",),
    ),
    Plant(
        # An excluded file deleted, its exclusion left behind.
        "tests/test_mutation_scope.py",
        ("crates/valgebra-py/src/exception.rs",),
        lambda tree: _delete(tree, "crates/valgebra-py/src/exception.rs"),
        trips=("test_no_exclusion_names_a_file_that_is_gone",),
    ),
    Plant(
        # An `exclude_re` entry for a mutant no function offers.
        "tests/test_mutation_scope.py",
        (".cargo/mutants.toml",),
        lambda tree: _edit(
            tree,
            ".cargo/mutants.toml",
            "exclude_re = [\n",
            'exclude_re = [\n    "crates/valgebra-core/src/planted\\\\.rs: .*",\n',
        ),
        trips=("test_no_excused_mutant_has_outlived_its_subject",),
    ),
    Plant(
        # The configured multiplier moved away from the one the lanes pass.
        "tests/test_mutation_scope.py",
        (".cargo/mutants.toml",),
        lambda tree: _edit(
            tree,
            ".cargo/mutants.toml",
            "timeout_multiplier = 20.0",
            "timeout_multiplier = 5.0",
        ),
        trips=("test_the_configured_timeout_is_the_one_the_lanes_run_under",),
    ),
    Plant(
        # The config's argument rewritten without the feature that makes the
        # walk reachable from `cargo test`.
        "tests/test_mutation_scope.py",
        (".cargo/mutants.toml",),
        lambda tree: _edit(
            tree, ".cargo/mutants.toml", "`interpreter-tests`", "embedded-interpreter"
        ),
        trips=("test_the_scope_carries_its_reason",),
    ),
    Plant(
        # A broad glob that swallows the walk.
        "tests/test_mutation_scope.py",
        (".cargo/mutants.toml",),
        lambda tree: _edit(
            tree,
            ".cargo/mutants.toml",
            "exclude_globs = [\n",
            'exclude_globs = [\n    "crates/valgebra-py/src/check/**",\n',
        ),
        trips=("test_the_walk_is_not_excluded",),
    ),
    Plant(
        # A seed file truncated to its header: a file the scan counts and no
        # run replays a case from.
        "tests/test_proptest_seeds.py",
        ("crates/valgebra-core/proptest-regressions/laws.txt",),
        lambda tree: _write(
            tree,
            "crates/valgebra-core/proptest-regressions/laws.txt",
            "# Seeds for failure cases proptest has generated in the past.\n",
        ),
        trips=("test_the_scan_reads_the_seed_files_that_are_there",),
    ),
    Plant(
        # An excuse short enough to be a shrug.
        "tests/test_typed_consumer.py",
        ("tests/test_typed_consumer.py",),
        lambda tree: _edit(
            tree,
            "tests/test_typed_consumer.py",
            '    "__hash__": (\n'
            '        "a hash is reached through `hash()`, whose return type comes'
            ' from the "\n'
            '        "builtin rather than from this stub, so the row would assert'
            ' the "\n'
            '        "builtin\'s signature"\n'
            "    ),",
            '    "__hash__": "builtin",',
        ),
        trips=("test_every_accepted_reason_is_a_sentence",),
    ),
    Plant(
        # The version gate dropped from the import, so the floor's checker reads
        # a name `typing` does not have there.
        "tests/test_typed_consumer.py",
        ("tests/typing/consumer.py",),
        lambda tree: _edit(
            tree,
            "tests/typing/consumer.py",
            "if sys.version_info >= (3, 11):\n"
            "    from typing import Never, assert_type\n"
            "else:  # the floor, where `typing` carries neither yet\n"
            "    from typing_extensions import Never, assert_type\n",
            "from typing import Never, assert_type\n",
        ),
        trips=("test_the_consumer_reads_assert_type_on_the_floor_too",),
    ),
    Plant(
        # Every row put through a wrapper of `assert_type`: the checker still
        # asks, and the detector finds no call.
        "tests/test_typed_consumer.py",
        ("tests/typing/consumer.py",),
        lambda tree: _replace_all(
            tree, "tests/typing/consumer.py", "    assert_type(", "    check_type("
        ),
        trips=("test_the_consumer_uses_assert_type_rather_than_assignment_alone",),
    ),
    Plant(
        # A method renamed in the stub, which the detector's named rows miss.
        "tests/test_typed_consumer.py",
        ("python/valgebra/_valgebra.pyi",),
        lambda tree: _edit(
            tree,
            "python/valgebra/_valgebra.pyi",
            "    def relation_to(",
            "    def relation(",
        ),
        trips=("test_the_stub_declares_a_surface_to_check",),
    ),
    Plant(
        # An accepted bound renamed in the source with its excuse left behind:
        # the entry names a bound the tree no longer declares.
        "tests/test_bound_ledger.py",
        ("crates/valgebra-core/src/descr/regular.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/descr/regular.rs",
            "pub const BUILD_SIZE_LIMIT: usize = 8 * 1024 * 1024;",
            "pub const BUILD_BYTES_LIMIT: usize = 8 * 1024 * 1024;",
        ),
        trips=("test_every_accepted_bound_exists_and_has_a_reason",),
    ),
    Plant(
        # A test comes to drive the accepted bound and the excuse stays: an
        # excuse outliving the gap it excused.
        "tests/test_bound_ledger.py",
        ("tests/test_adversarial_bounds.py",),
        lambda tree: _edit(
            tree,
            "tests/test_adversarial_bounds.py",
            f"{BOUND_MARKER}: MAX_MARKER_TYPES",
            f"{BOUND_MARKER}: MAX_MARKER_TYPES\n{BOUND_MARKER}: BUILD_SIZE_LIMIT",
        ),
        trips=("test_no_accepted_bound_is_driven",),
    ),
    Plant(
        # A bound renamed in the binding while the Python test driving it keeps
        # the old name in its marker: a marker pointing at nothing.
        "tests/test_bound_ledger.py",
        ("crates/valgebra-py/src/render.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/render.rs",
            "const MAX_RENDER_DEPTH: usize = 200;",
            "const MAX_RENDER_NESTING: usize = 200;",
        ),
        trips=("test_every_marker_names_a_declared_bound",),
    ),
    Plant(
        # A fixture added under `readings/` with no row to read it: a reading
        # held by nothing.
        "tests/test_checker_readings.py",
        ("tests/typing/readings/planted.py",),
        lambda tree: _write(
            tree,
            "tests/typing/readings/planted.py",
            '"""A fixture no row reads."""\n\n'
            "from typing_extensions import reveal_type\n\n"
            "from valgebra import Validator\n\n"
            "reveal_type(Validator(int))\n",
        ),
        trips=("test_the_fixtures_and_the_rows_are_the_same_set",),
    ),
    Plant(
        # A second `reveal_type` in a fixture, which makes its cell a merge of
        # two readings rather than one.
        "tests/test_checker_readings.py",
        ("tests/typing/readings/newtype.py",),
        lambda tree: _edit(
            tree,
            "tests/typing/readings/newtype.py",
            "reveal_type(Validator(UserId))\n",
            "reveal_type(Validator(UserId))\nreveal_type(UserId)\n",
        ),
        trips=("test_a_fixture_reveals_one_expression",),
    ),
    Plant(
        # A row whose three cells agree, kept as a fixture after a checker
        # release closed the disagreement: it belongs in the consumer.
        "tests/test_checker_readings.py",
        ("tests/test_checker_readings.py",),
        lambda tree: _edit(
            tree,
            "tests/test_checker_readings.py",
            'mypy="Validator[object]",\n        pyright="Validator[HasX]",',
            'mypy="Validator[HasX]",\n        pyright="Validator[HasX]",',
        ),
        trips=("test_a_row_is_a_disagreement",),
    ),
    Plant(
        # ty asked for the output format the parse does not read: every line it
        # prints misses the pattern, and the checker reads as reading nothing.
        "tests/test_checker_readings.py",
        ("tests/test_checker_readings.py",),
        lambda tree: _edit(
            tree,
            "tests/test_checker_readings.py",
            '"--output-format",\n            "concise",',
            '"--output-format",\n            "full",',
        ),
        trips=("test_a_checker_read_every_fixture",),
    ),
    Plant(
        # A kind added to the partition, and no base for it here.
        "tests/test_constraint_matrix.py",
        ("crates/valgebra-core/src/kind.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/kind.rs",
            "    Dict,\n}\n",
            "    Dict,\n    /// A kind this ledger has no base for.\n    Range,\n}\n",
        ),
        trips=("test_every_kind_has_a_base_a_caller_writes",),
    ),
    Plant(
        # A column added to the inventory: every row has four cells, the parse
        # takes rows of three, and the checks below it compare nothing.
        "tests/test_contract_inventory.py",
        ("CONTRIBUTING.md",),
        _widen_the_inventory,
        trips=("test_the_inventory_has_rows",),
    ),
    Plant(
        # A source of truth moved, and its row still naming the old path.
        "tests/test_contract_inventory.py",
        ("CONTRIBUTING.md",),
        lambda tree: _edit(
            tree,
            "CONTRIBUTING.md",
            "| supply chain (Rust) | `deny.toml` |",
            "| supply chain (Rust) | `.cargo/deny.toml` |",
        ),
        trips=("test_every_row_names_a_real_source_of_truth",),
    ),
    Plant(
        # A rerun command naming a script by a name it does not have.
        "tests/test_contract_inventory.py",
        ("CONTRIBUTING.md",),
        lambda tree: _edit(
            tree,
            "CONTRIBUTING.md",
            "`uv run --no-sync python scripts/metamorphic_gate.py` |",
            "`uv run --no-sync python scripts/metamorphic_check.py` |",
        ),
        trips=("test_every_script_a_row_invokes_exists",),
    ),
    Plant(
        # A rerun cell written as prose rather than as a command.
        "tests/test_contract_inventory.py",
        ("CONTRIBUTING.md",),
        lambda tree: _edit(
            tree,
            "CONTRIBUTING.md",
            "| `uv.lock` | `uv export --locked --format requirements-txt",
            "| `uv.lock` | uv export --locked --format requirements-txt",
        ),
        trips=("test_every_row_carries_all_three_cells",),
    ),
    Plant(
        # A corpus under a name the pattern does not know: the detector's count
        # of interpreter corpora is what notices it has gone.
        "tests/test_coverage_scope.py",
        (
            "crates/valgebra-py/src/oracle/interpreter.rs",
            "crates/valgebra-py/src/oracle/corpus.rs",
        ),
        lambda tree: _move(
            tree,
            "crates/valgebra-py/src/oracle/interpreter.rs",
            "crates/valgebra-py/src/oracle/corpus.rs",
        ),
        trips=("test_the_tree_has_corpus_files_to_exclude",),
    ),
    Plant(
        # The binding lane's region floor dropped, leaving the line floor alone.
        "tests/test_coverage_scope.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "--fail-under-lines 96 --fail-under-regions 93",
            "--fail-under-lines 96",
        ),
        trips=("test_a_lane_enforces_a_region_floor_beside_its_line_floor",),
    ),
    Plant(
        # A floor written as a fraction, which the lane reads as a percentage it
        # always clears.
        "tests/test_coverage_scope.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "--fail-under-lines 98 --fail-under-regions 97",
            "--fail-under-lines 0.98 --fail-under-regions 97",
        ),
        trips=("test_a_floor_is_a_number_the_lane_can_reach",),
    ),
    Plant(
        # The branch figure measured and the step ratcheting it dropped: a
        # number printed rather than held.
        "tests/test_coverage_scope.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "      - name: Hold the figure to its recorded floor\n"
            "        run: uv run --no-sync python scripts/branch_coverage.py"
            " branches.json\n",
            "",
        ),
        trips=("test_a_lane_records_the_branch_number",),
    ),
    Plant(
        # The testing page's count of the tests the lock stands down, off by one.
        "tests/test_coverage_scope.py",
        ("docs/dev/08-testing.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/08-testing.md",
            "cannot be the one that moves it. Three tests",
            "cannot be the one that moves it. Four tests",
        ),
        trips=("test_the_page_counts_the_tests_the_gil_stands_down",),
    ),
    Plant(
        # A free-threaded interpreter given to the binding's coverage lane, which
        # makes the page's paragraph about arms no lane measures stale.
        "tests/test_coverage_scope.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            '          python-version: "3.12"\n      # Binding coverage combines',
            '          python-version: "3.14t"\n      # Binding coverage combines',
        ),
        trips=("test_no_coverage_lane_runs_a_free_threaded_interpreter",),
    ),
    Plant(
        # A constraint kind added to the algebra and to no denotation leaf.
        "tests/test_coverage_scope.py",
        ("crates/valgebra-core/src/ir.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/ir.rs",
            "pub enum Constraint {\n",
            "pub enum Constraint {\n"
            "    /// A constraint the denotation generator has no leaf for.\n"
            "    Planted,\n",
        ),
        trips=("test_every_constraint_kind_is_one_the_denotation_generator_builds",),
    ),
    Plant(
        # A constraint kind renamed in the algebra, and its leaf left naming the
        # old one.
        "tests/test_coverage_scope.py",
        ("crates/valgebra-core/src/ir.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/ir.rs",
            "    /// `value % pool[i] == 0`: a numeric multiple of the operand.\n"
            "    MultipleOf(OperandIx),",
            "    /// `value % pool[i] == 0`: a numeric multiple of the operand.\n"
            "    StepOf(OperandIx),",
        ),
        trips=("test_every_leaf_the_denotation_generator_lists_is_a_kind",),
    ),
    Plant(
        # A node added to the algebra that no denotation shape is built from.
        "tests/test_coverage_scope.py",
        ("crates/valgebra-core/src/ir.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/ir.rs",
            "pub enum Schema {\n",
            "pub enum Schema {\n"
            "    /// A node the denotation generator never builds.\n"
            "    Planted,\n",
        ),
        trips=("test_every_schema_variant_is_one_the_denotation_generator_builds",),
    ),
    Plant(
        # A page renamed, and the runner still listing it by the old name.
        "tests/test_doc_examples.py",
        ("scripts/run_doc_examples.py",),
        lambda tree: _edit(
            tree,
            "scripts/run_doc_examples.py",
            '    ROOT / "CHANGELOG.md",',
            '    ROOT / "CHANGES.md",',
        ),
        trips=("test_every_page_the_runner_lists_is_one_the_tree_tracks",),
    ),
    Plant(
        # An entry for a feature the manifest does not declare: the excuse a
        # removed feature leaves behind.
        "tests/test_harness_conditionals.py",
        ("tests/test_harness_conditionals.py",),
        lambda tree: _edit(
            tree,
            "tests/test_harness_conditionals.py",
            "PRODUCTION_FEATURES: dict[str, str] = {}",
            "PRODUCTION_FEATURES: dict[str, str] = {\n"
            '    "abi3": "a feature the manifest dropped while its entry stayed,'
            ' planted",\n}',
        ),
        trips=("test_no_ledger_entry_is_stale",),
    ),
    Plant(
        # An entry naming a declared feature with a word where an argument
        # belongs.
        "tests/test_harness_conditionals.py",
        ("tests/test_harness_conditionals.py",),
        lambda tree: _edit(
            tree,
            "tests/test_harness_conditionals.py",
            "PRODUCTION_FEATURES: dict[str, str] = {}",
            'PRODUCTION_FEATURES: dict[str, str] = {"interpreter-tests": "test-only"}',
        ),
        trips=("test_every_ledger_entry_carries_a_reason",),
    ),
    Plant(
        # The manifest's sentence reworded until it no longer says the claim the
        # arrangement rests on.
        "tests/test_harness_conditionals.py",
        ("crates/valgebra-py/Cargo.toml",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/Cargo.toml",
            "# The feature never reaches the shipped wheel,",
            "# The feature is off in the shipped wheel,",
        ),
        trips=("test_the_manifest_says_the_feature_never_ships",),
    ),
    Plant(
        # The ratchet pointed at another sweep's baseline.
        "tests/test_pytest_sweep_scope.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "python3 scripts/mutation_gate.py --baseline pytest \\",
            "python3 scripts/mutation_gate.py --baseline walk \\",
        ),
        trips=("test_a_lane_ratchets_what_the_sweep_finds",),
    ),
    Plant(
        # The sweep's condition dropped, which puts a suite run per mutant on
        # every merge.
        "tests/test_pytest_sweep_scope.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "    if: github.event_name == 'schedule' || github.event_name =="
            " 'workflow_dispatch'\n    runs-on: ubuntu-latest\n    # Six shards",
            "    runs-on: ubuntu-latest\n    # Six shards",
        ),
        trips=("test_a_lane_runs_the_sweep_off_the_merge_path",),
    ),
    Plant(
        # An examined file renamed in the tree and not in the configuration.
        "tests/test_pytest_sweep_scope.py",
        ("crates/valgebra-py/src/render.rs", "crates/valgebra-py/src/repr.rs"),
        lambda tree: _move(
            tree, "crates/valgebra-py/src/render.rs", "crates/valgebra-py/src/repr.rs"
        ),
        trips=("test_every_examined_file_is_in_the_tree",),
    ),
    Plant(
        # A file the pytest sweep examines added to the walk sweep as well.
        "tests/test_pytest_sweep_scope.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "            --file crates/valgebra-py/src/codes.rs \\\n"
            "            --features interpreter-tests -j 4",
            "            --file crates/valgebra-py/src/codes.rs \\\n"
            "            --file crates/valgebra-py/src/render.rs \\\n"
            "            --features interpreter-tests -j 4",
        ),
        trips=("test_nothing_is_examined_under_pytest_that_a_second_already_proves",),
    ),
    Plant(
        # The excuse reworded until it no longer names what measures it.
        "tests/test_pytest_sweep_scope.py",
        (".cargo/mutants.toml",),
        lambda tree: _edit(
            tree,
            ".cargo/mutants.toml",
            "`.cargo/mutants-pytest.toml` examines",
            "the pytest configuration examines",
        ),
        trips=("test_the_exclusion_points_at_what_measures_it",),
    ),
    Plant(
        # The wrapper's feature turned on by default.
        "tests/test_pytest_sweep_scope.py",
        ("crates/valgebra-py/Cargo.toml",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/Cargo.toml",
            "pytest-sweep = []\n",
            'pytest-sweep = []\ndefault = ["pytest-sweep"]\n',
        ),
        trips=("test_the_feature_is_declared_and_off_by_default",),
    ),
    Plant(
        # The gate's baseline renamed, so `--baseline pytest` names nothing.
        "tests/test_pytest_sweep_scope.py",
        ("scripts/mutation_gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/mutation_gate.py",
            '    "pytest": ROOT / "scripts" / "mutation_baseline_pytest.json",',
            '    "suite": ROOT / "scripts" / "mutation_baseline_pytest.json",',
        ),
        trips=("test_the_pytest_baseline_is_registered_and_recorded",),
    ),
    Plant(
        # The variable handed to the build unchanged, which every worker then
        # builds into at once.
        "tests/test_pytest_sweep_scope.py",
        ("crates/valgebra-py/tests/pytest_sweep.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/tests/pytest_sweep.rs",
            "    let venv = worker_venv(&base, &at);",
            "    let venv = base.clone();",
        ),
        trips=("test_the_workers_of_a_sweep_do_not_share_one_environment",),
    ),
    Plant(
        # The wrapper warning and carrying on where the variable is unset, which
        # reports every mutant caught while running no suite.
        "tests/test_pytest_sweep_scope.py",
        ("crates/valgebra-py/tests/pytest_sweep.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/tests/pytest_sweep.rs",
            'std::env::var("VALGEBRA_SWEEP_VENV") else {\n        panic!(',
            'std::env::var("VALGEBRA_SWEEP_VENV") else {\n        eprintln!(',
        ),
        trips=("test_the_wrapper_refuses_rather_than_measuring_nothing",),
    ),
    Plant(
        # A build row renamed, and the smoke row still downloading the old
        # artifact.
        "tests/test_release_smoke.py",
        (".github/workflows/release.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/release.yml",
            "variant: freethreaded }",
            "variant: nogil }",
        ),
        trips=("test_every_smoke_row_names_a_wheel_set_the_release_builds",),
    ),
    Plant(
        # The smoke matrix emptied: every check over it compares nothing.
        "tests/test_release_smoke.py",
        (".github/workflows/release.yml",),
        _empty_the_smoke_matrix,
        trips=("test_the_matrices_were_read",),
    ),
    Plant(
        # The suite made a row's choice: the rows that do not say so ship on
        # an import alone.
        "tests/test_release_smoke.py",
        (".github/workflows/release.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/release.yml",
            "      - name: Run the product suite on the built wheel\n",
            "      - name: Run the product suite on the built wheel\n"
            "        if: ${{ matrix.wheel.suite }}\n",
        ),
        trips=("test_every_smoke_runs_the_product_suite",),
    ),
    Plant(
        # The import's warnings left as warnings: a deprecation at import time
        # reaches a user's log rather than failing the release.
        "tests/test_release_smoke.py",
        (".github/workflows/release.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/release.yml",
            '"$bin/python" -W error -c',
            '"$bin/python" -c',
        ),
        trips=("test_every_smoke_import_fails_on_a_warning",),
    ),
    Plant(
        # A classified release dropped from a platform's row: every installer
        # there builds from source.
        "tests/test_release_matrix.py",
        (".github/workflows/release.yml",),
        _drop_a_release,
        trips=("test_every_platform_builds_every_release_the_classifiers_name",),
    ),
    Plant(
        # A wheel built for a release the classifiers do not name.
        "tests/test_release_matrix.py",
        (".github/workflows/release.yml",),
        _build_an_unclassified_release,
        trips=("test_no_platform_builds_a_release_the_classifiers_do_not_name",),
    ),
    Plant(
        # A gap recorded for a platform the release does not build for.
        "tests/test_release_matrix.py",
        ("tests/test_release_matrix.py",),
        lambda tree: _edit(
            tree,
            "tests/test_release_matrix.py",
            "GAPS: dict[tuple[str, str], str] = {\n",
            "GAPS: dict[tuple[str, str], str] = {\n"
            '    ("ubuntu-latest sparc64", "3.12"): "a platform no row builds",\n',
        ),
        trips=("test_every_gap_is_one_the_matrix_has",),
    ),
    Plant(
        # A musllinux row that names its interpreters, its excuse left behind.
        "tests/test_release_matrix.py",
        (".github/workflows/release.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/release.yml",
            "target: x86_64, manylinux: musllinux_1_2 }",
            "target: x86_64, manylinux: musllinux_1_2, interpreter: python3.12 }",
        ),
        trips=("test_every_row_names_its_interpreters_or_says_why",),
    ),
    Plant(
        # The classifiers gone: every check over them compares nothing.
        "tests/test_release_matrix.py",
        ("pyproject.toml",),
        _unclassify_every_release,
        trips=("test_the_matrix_and_the_classifiers_were_read",),
    ),
    Plant(
        # A crate that stops packaging a directory it tracks: the build from
        # source reads what the archive left out.
        "tests/test_sdist.py",
        ("crates/valgebra-core/Cargo.toml",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/Cargo.toml",
            "publish = false\n",
            'publish = false\nexclude = ["benches/"]\n',
        ),
        trips=("test_the_sdist_carries_every_build_input",),
    ),
    Plant(
        # A scratch file beside the package, untracked and unignored: it ships
        # to everyone who builds from source.
        "tests/test_sdist.py",
        ("python/valgebra/_scratch.py",),
        lambda tree: _write(tree, "python/valgebra/_scratch.py", "PLANTED = True\n"),
        trips=("test_the_sdist_carries_nothing_else",),
    ),
    Plant(
        # A job every push runs, accepted by the gate as `skipped`.
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "          needs.rust-lint.result != 'success' ||\n",
            "          (needs.rust-lint.result != 'success' &&"
            " needs.rust-lint.result != 'skipped') ||\n",
        ),
        trips=("test_a_job_a_push_does_not_run_may_be_skipped_and_no_other_may",),
    ),
    Plant(
        # A file added to the nightly binding sweep and not to the push lane's.
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "            --file crates/valgebra-py/src/codes.rs \\\n"
            "            --features interpreter-tests -j 4",
            "            --file crates/valgebra-py/src/codes.rs \\\n"
            "            --file crates/valgebra-py/src/render.rs \\\n"
            "            --features interpreter-tests -j 4",
        ),
        trips=("test_both_binding_sweeps_read_the_same_files",),
    ),
    Plant(
        # The merge counting one shard fewer than its sweep is cut into.
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            '          SHARDS: "6"',
            '          SHARDS: "5"',
        ),
        trips=("test_every_merge_counts_the_shards_its_sweep_is_cut_into",),
    ),
    Plant(
        # A job the gate waits on whose result its condition no longer reads.
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "          needs.rust-msrv.result != 'success' ||\n",
            "",
        ),
        trips=("test_every_need_is_read_by_the_condition",),
    ),
    Plant(
        # The free-threaded leg dropped from the python job while the
        # classifiers still promise free threading.
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            'python-version: ["3.10", "3.11", "3.12", "3.13", "3.14", "3.14t", "3.15"]',
            'python-version: ["3.10", "3.11", "3.12", "3.13", "3.14", "3.15"]',
        ),
        trips=("test_every_supported_interpreter_runs_on_every_event",),
    ),
    Plant(
        # A file the binding sweep lists that its trigger no longer matches: the
        # drift that let a change to the walk land with the sweep skipped.
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "|equality|oracle|codes)\\.rs$' changed.txt",
            "|equality|oracle)\\.rs$' changed.txt",
        ),
        trips=("test_the_binding_sweep_triggers_on_every_file_it_sweeps",),
    ),
    Plant(
        # A job renamed, and the gate still waiting on the old name.
        "tests/test_required_jobs.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree, ".github/workflows/ci.yml", "  fuzz-build:\n", "  fuzz-compile:\n"
        ),
        trips=("test_the_gate_names_no_job_the_workflow_lacks",),
    ),
    Plant(
        # A test module subscripting a typing form with a validator, which on
        # the floor is a collection error that takes the module with it.
        "tests/test_suite_partition.py",
        ("tests/test_planted_form.py",),
        lambda tree: _write(
            tree,
            "tests/test_planted_form.py",
            '"""A typing form over a validator."""\n\n'
            "from typing import Annotated\n\n"
            "from valgebra import union\n\n"
            "PLANTED = Annotated[union(int, str), 0]\n",
        ),
        trips=("test_no_typing_form_is_subscripted_with_a_validator",),
    ),
    Plant(
        # `--strict-markers` dropped, so a misspelt marker is accepted and the
        # boundary drawn by it moves silently.
        "tests/test_suite_partition.py",
        ("pyproject.toml",),
        lambda tree: _edit(
            tree,
            "pyproject.toml",
            'addopts = ["-ra", "--strict-markers", "--strict-config"]',
            'addopts = ["-ra", "--strict-config"]',
        ),
        trips=("test_the_marker_is_registered",),
    ),
    Plant(
        # The repository checks grown past the product tests.
        "tests/test_suite_partition.py",
        _PLANTED_REPOSITORY_MODULES,
        _outnumber_the_product_suite,
        trips=("test_the_product_suite_is_the_larger_half",),
    ),
    Plant(
        # A marker dropped from a test the sweeps still skip.
        "tests/test_sweep_skips.py",
        ("crates/valgebra-py/src/check/walk/interpreter.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/check/walk/interpreter.rs",
            "// SWEEP-SKIP: this case exists to prove a bound, so a mutation that"
            " removes\n// the bound makes it run without end.",
            "// This case exists to prove a bound, so a mutation that removes\n"
            "// the bound makes it run without end.",
        ),
        trips=("test_every_skip_names_a_marked_test",),
    ),
    Plant(
        # A marker whose argument is cut to a phrase.
        "tests/test_sweep_skips.py",
        ("crates/valgebra-py/src/check/walk/interpreter.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/check/walk/interpreter.rs",
            "// SWEEP-SKIP: this case exists to prove a bound, so a mutation that"
            " removes\n",
            "// SWEEP-SKIP: a bound\n",
        ),
        trips=("test_the_marker_carries_a_reason",),
    ),
    Plant(
        # A skip left out of the documented rerun command: following it returns
        # no verdict for that mutant.
        "tests/test_sweep_skips.py",
        ("CONTRIBUTING.md",),
        lambda tree: _edit(
            tree,
            "CONTRIBUTING.md",
            " --skip subtyping_terminates_on_a_distributed_tower` |",
            "` |",
        ),
        trips=("test_the_documented_rerun_command_carries_the_same_skips",),
    ),
    Plant(
        # Three more tests marked out of the sweep, which is how the list grows
        # quietly past what the sweep can still judge.
        "tests/test_sweep_skips.py",
        ("crates/valgebra-core/src/decision/budget_tests.rs",),
        _skip_three_more,
        trips=("test_the_skips_are_few",),
    ),
    Plant(
        # A blank line lost between a holding line and the claim after it: the
        # claim is read as more names for the one before, and its tag is a tag
        # the page carries and the parse does not.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/10-theory.md",
            "a_callback_inside_a_resolved_body_refuses_the_reference\n\n"
            "**Stone's representation theorem (1936).**",
            "a_callback_inside_a_resolved_body_refuses_the_reference\n"
            "**Stone's representation theorem (1936).**",
        ),
        trips=("test_the_page_carries_tagged_claims",),
    ),
    Plant(
        # A debt addressed to a moment rather than a test name.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _owe(
            tree, "OWED: the next release -- whichever commit proves the lowering"
        ),
        trips=("test_an_owed_claim_names_the_test_and_the_reason",),
    ),
    Plant(
        # A claim added as owed without the recorded figure moving with it.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _owe(
            tree,
            "OWED: the_scalar_lowering_is_a_set_algebra -- the lowering has no "
            "test naming this result yet",
        ),
        trips=("test_what_is_owed_only_shrinks",),
    ),
    Plant(
        # A debt paid and left standing: the test it is owed already exists.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _owe(
            tree,
            "OWED: test_the_page_carries_tagged_claims -- the test lands with "
            "the scalar lowering",
        ),
        trips=("test_an_owed_test_does_not_already_exist",),
    ),
    Plant(
        # A held claim cut down to a single test.
        "tests/test_theory_ledger.py",
        ("docs/dev/10-theory.md",),
        lambda tree: _edit(
            tree,
            "docs/dev/10-theory.md",
            "HELD-BY: test_walk_matches_denotation, test_node_admits_its_denotation",
            "HELD-BY: test_walk_matches_denotation",
        ),
        trips=("test_a_claim_is_held_by_more_than_its_own_restatement",),
    ),
    Plant(
        # One reader spelling a path its own way, which on Windows makes every
        # comparison between the two a mismatch.
        "tests/test_theory_ledger.py",
        ("tests/test_theory_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_theory_ledger.py",
            "found.setdefault(identifier, []).append((_relative(path), name))",
            "found.setdefault(identifier, []).append((str(path), name))",
        ),
        trips=("test_the_readers_spell_a_path_the_same_way",),
    ),
    Plant(
        # The failure the roll is for: a `fix` that landed with no line on it.
        # Taken from the newest commit the roll names, so the plant reads the
        # roll the tree has rather than a subject that a release empties.
        "tests/test_changelog_ledger.py",
        ("CHANGELOG.md",),
        _unroll_a_commit,
        trips=("test_every_visible_commit_is_on_the_roll",),
    ),
    Plant(
        # The other direction: a line whose subject a rebase renamed, left
        # standing among the lines that match.
        "tests/test_changelog_ledger.py",
        ("CHANGELOG.md",),
        _roll_a_renamed_subject,
        trips=("test_no_roll_entry_is_stale",),
    ),
    Plant(
        # The detector's case: a roll emptied while the release has commits,
        # which the two checks above read as a roll with nothing wrong in it.
        "tests/test_changelog_ledger.py",
        ("CHANGELOG.md",),
        _empty_the_roll,
        trips=("test_the_roll_is_not_empty_while_the_surface_moves",),
    ),
    Plant(
        # A variant dropped from the IR while its column keeps it.
        "tests/test_closure_ledger.py",
        ("crates/valgebra-core/src/ir.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/ir.rs",
            "reads it.\n    Bytes,\n",
            "reads it.\n",
        ),
        trips=("test_no_column_names_a_variant_that_is_gone",),
    ),
    Plant(
        # A generator moved into the markers without leaving its column: one
        # variant claimed twice, which the both-directions check reads as fine.
        "tests/test_closure_ledger.py",
        ("tests/test_closure_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_closure_ledger.py",
            'MARKERS = {"Ref", "SelfRef"}',
            'MARKERS = {"Ref", "SelfRef", "Refine"}',
        ),
        trips=("test_the_columns_do_not_overlap",),
    ),
    Plant(
        # A derivation that stopped being the representative's: `Literal[1]` is
        # an `int`, so the union is not `bool` any more.
        "tests/test_closure_ledger.py",
        ("tests/test_closure_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_closure_ledger.py",
            '"Bool": (Validator(bool), union(Literal[True], Literal[False])),',
            '"Bool": (Validator(bool), union(Literal[True], Literal[1])),',
        ),
        trips=("test_a_representative_denotes_what_it_stands_for",),
    ),
    Plant(
        # A checker whose output changed shape: the reader keys on a form the
        # checker no longer prints, reads nothing, and a clean page and a
        # broken parser read alike.
        "tests/test_doc_example_checkers.py",
        ("scripts/check_doc_examples.py",),
        lambda tree: _edit(
            tree,
            "scripts/check_doc_examples.py",
            '_RUFF_LINE = re.compile(r"^(?P<file>.+?):\\d+:\\d+: (?P<rule>\\S+)")',
            '_RUFF_LINE = re.compile(r"^(?P<file>.+?):\\d+:\\d+:'
            ' error\\[(?P<rule>\\S+)\\]")',
        ),
        trips=("test_each_checker_reported_something",),
    ),
    Plant(
        # An expected diagnostic kept with a shrug for a reason.
        "tests/test_doc_example_checkers.py",
        ("scripts/check_doc_examples.py",),
        lambda tree: _edit(
            tree,
            "scripts/check_doc_examples.py",
            'Expected("ty", "README.md", "invalid-assignment", 1, _MUTATION),',
            'Expected("ty", "README.md", "invalid-assignment", 1, "a'
            ' known diagnostic"),',
        ),
        trips=("test_every_row_names_a_reached_page_once_with_a_reason",),
    ),
    Plant(
        "tests/test_doc_example_checkers.py",
        ("scripts/check_doc_examples.py",),
        lambda tree: _edit(
            tree,
            "scripts/check_doc_examples.py",
            '"FBT": "an example passes a bool where the page shows one",',
            '"FBT": "not relevant",',
        ),
        trips=("test_every_exemption_says_why",),
    ),
    Plant(
        # A feature dropped from the manifest while its excuse stays.
        "tests/test_feature_lanes.py",
        ("crates/valgebra-py/Cargo.toml",),
        lambda tree: _edit(
            tree, "crates/valgebra-py/Cargo.toml", "pytest-sweep = []\n", ""
        ),
        trips=("test_no_excuse_is_stale",),
    ),
    Plant(
        # A name no row dates: a misspelling, which the rule above skips as
        # third-party rather than refusing.
        "tests/test_floor_names.py",
        ("tests/test_enums.py",),
        lambda tree: _edit(
            tree,
            "tests/test_enums.py",
            "import sys",
            "import sys\nfrom typing import NotRequierd",
        ),
        trips=("test_every_name_read_is_one_the_table_dates",),
    ),
    Plant(
        # An import added without re-running the table: the module is the
        # standard library's and no row dates it, so no rule judges it.
        "tests/test_floor_names.py",
        ("tests/test_enums.py",),
        lambda tree: _edit(
            tree, "tests/test_enums.py", "import sys", "import sys\nimport zoneinfo"
        ),
        trips=("test_every_stdlib_module_imported_is_one_the_table_dates",),
    ),
    Plant(
        # The import that reddened the floor leg, at module scope again.
        "tests/test_floor_names.py",
        ("tests/test_enums.py",),
        lambda tree: _edit(
            tree, "tests/test_enums.py", "import sys", "import sys\nimport tomllib"
        ),
        trips=(
            "test_no_module_imports_a_stdlib_module_outside_the_releases_that_ship_it",
        ),
    ),
    Plant(
        # A row for a name `typing` carries on no release, as a misspelt row is:
        # read by no source, so only the interpreter can say it is wrong, and
        # every release says so rather than the one a removal date would name.
        "tests/test_floor_names.py",
        ("tests/floor_names.json",),
        lambda tree: _edit(
            tree,
            "tests/floor_names.json",
            '      "ABCMeta": {\n        "since": "3.10"\n      },',
            '      "ABCMeta": {\n        "since": "3.10"\n      },\n'
            '      "ABCMetaclass": {\n        "since": "3.10"\n      },',
        ),
        trips=("test_the_table_agrees_with_this_interpreter",),
    ),
    Plant(
        # A module the tree imports, dated as shipped from the floor, which no
        # release ships: every interpreter reports it absent.
        "tests/test_floor_names.py",
        ("tests/floor_names.json", "tests/_planted_import.py"),
        _import_an_unshipped_module,
        trips=("test_the_stdlib_rows_agree_with_this_interpreter",),
    ),
    Plant(
        # The helper the frontend refuses through, renamed: the scan keys on
        # its name, and a third of the refusals is what it would read.
        "tests/test_frontend_refusals.py",
        _REFUSING,
        lambda tree: _rename_in(tree, _REFUSING, "not_implemented(", "refuse("),
        trips=("test_the_universe_is_read_from_the_frontend",),
    ),
    Plant(
        # This file's own marker, deleted: a repository check read as the
        # product suite, whose strings would hold refusals with work about the
        # tree. The file's filter spells the marker too, so a text search
        # excludes it either way; this trips once the marker is read from the
        # assignment.
        "tests/test_frontend_refusals.py",
        ("tests/test_frontend_refusals.py",),
        lambda tree: _edit(
            tree,
            "tests/test_frontend_refusals.py",
            "\npytestmark = pytest.mark.repository\n",
            "\n",
        ),
        trips=("test_the_product_suite_is_what_is_searched",),
    ),
    Plant(
        "tests/test_frontend_refusals.py",
        ("tests/test_frontend_refusals.py",),
        lambda tree: _edit(
            tree,
            "tests/test_frontend_refusals.py",
            "ACCEPTED: dict[str, str] = {}",
            'ACCEPTED: dict[str, str] = {"build.rs:1": "a later pass"}',
        ),
        trips=("test_every_accepted_reason_is_a_sentence",),
    ),
    Plant(
        # The fork dropped from the soak's command. The step's comment still
        # names `-fork=1`, so this trips only once the ledger reads commands.
        "tests/test_fuzz_lane.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "-max_total_time=360 -fork=1 -malloc_limit_mb=64",
            "-max_total_time=360 -malloc_limit_mb=64",
        ),
        trips=("test_the_soak_bounds_memory_per_batch",),
    ),
    Plant(
        "tests/test_fuzz_lane.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(
            tree,
            ".github/workflows/ci.yml",
            "-max_total_time=360 -fork=1",
            "-fork=1",
        ),
        trips=("test_the_soak_still_has_a_time_budget",),
    ),
    Plant(
        # The floor's comparison deleted. The words it is read by stay in the
        # step's other lines, so this trips only once the ledger reads the
        # comparison itself.
        "tests/test_fuzz_lane.py",
        (".github/workflows/ci.yml",),
        lambda tree: _edit(tree, ".github/workflows/ci.yml", _SOAK_FLOOR, ""),
        trips=("test_the_soak_has_a_floor_beneath_its_budget",),
    ),
    Plant(
        # A node added to the IR that the generator never learns to build.
        "tests/test_fuzz_lane.py",
        ("crates/valgebra-core/src/ir.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/src/ir.rs",
            "pub enum Schema {\n",
            "pub enum Schema {\n    /// A node no generator draws.\n    Planted,\n",
        ),
        trips=("test_the_generator_draws_every_node_the_ir_has",),
    ),
    Plant(
        # The reference index drawn from the table's own width: every
        # reference resolves, and the cut is never reached.
        "tests/test_fuzz_lane.py",
        ("fuzz/src/lib.rs",),
        lambda tree: _edit(
            tree,
            "fuzz/src/lib.rs",
            "u.arbitrary::<u8>()? % 6)",
            "u.arbitrary::<u8>()? % 3)",
        ),
        trips=("test_the_generator_draws_a_reference_wider_than_its_table",),
    ),
    Plant(
        # The soundness direction read backwards: a gained proof reported as a
        # lost one, and a lost one passed.
        "tests/test_metamorphic_gate.py",
        ("scripts/metamorphic_gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/metamorphic_gate.py",
            'if reference[key] == "y" and measured[key] == "n"',
            'if reference[key] == "n" and measured[key] == "y"',
        ),
        trips=(
            "test_a_lost_proof_fails",
            "test_a_widening_is_not_a_lost_proof",
        ),
    ),
    Plant(
        # A condition reading one recording: every proof the reference holds is
        # reported lost, whatever this build proves.
        "tests/test_metamorphic_gate.py",
        ("scripts/metamorphic_gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/metamorphic_gate.py",
            'if reference[key] == "y" and measured[key] == "n"',
            'if reference[key] == "y"',
        ),
        trips=("test_an_unchanged_recording_holds_every_relation",),
    ),
    Plant(
        # The witness searched for on the wrong side of the claim.
        "tests/test_metamorphic_gate.py",
        ("scripts/metamorphic_gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/metamorphic_gate.py",
            'if inside == "y" and outside == "n":',
            'if inside == "n" and outside == "y":',
        ),
        trips=(
            "test_a_widening_a_value_refutes_fails",
            "test_a_widening_the_values_support_passes",
        ),
    ),
    Plant(
        # The search failing closed: a widening no value reaches reported as
        # refuted, which is the gate pretending the corpus said something.
        "tests/test_metamorphic_gate.py",
        ("scripts/metamorphic_gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/metamorphic_gate.py",
            "                break\n    return refuted",
            "                break\n"
            "        else:\n"
            '            refuted.append(f"{key}: no corpus value reaches it")\n'
            "    return refuted",
        ),
        trips=("test_a_widening_no_corpus_value_reaches_is_not_refuted",),
    ),
    Plant(
        # The emptiness guard dropped, so an emptiness row is unpacked as a
        # subtyping claim.
        "tests/test_metamorphic_gate.py",
        ("scripts/metamorphic_gate.py",),
        lambda tree: _edit(
            tree,
            "scripts/metamorphic_gate.py",
            "        if claim is None:\n            continue\n",
            "",
        ),
        trips=("test_an_emptiness_row_is_not_read_as_a_subtyping_claim",),
    ),
    Plant(
        "tests/test_metamorphic_gate.py",
        (_REFERENCE,),
        _record_outside_a_checkout,
        trips=("test_the_reference_the_gate_ships_with_describes_a_commit",),
    ),
    Plant(
        "tests/test_metamorphic_gate.py",
        (_REFERENCE,),
        _prove_nothing,
        trips=("test_the_corpus_asks_both_answers_of_both_relations",),
    ),
    Plant(
        # The inline pattern with its multiline flag dropped: `^` anchors at the
        # start of the file only, and the scan reads no module at all.
        "tests/test_module_placement.py",
        ("tests/test_module_placement.py",),
        lambda tree: _edit(
            tree,
            "tests/test_module_placement.py",
            '\\{$", re.MULTILINE\n)',
            '\\{$"\n)',
        ),
        trips=("test_the_scan_reads_the_modules_that_are_there",),
    ),
    Plant(
        # The calendar reaching the newest row's first alpha with no leg run
        # forgiven for it -- what the day itself does, with no commit.
        "tests/test_python_lifecycle.py",
        (_LIFECYCLE, ".github/workflows/ci.yml"),
        _reach_the_next_alpha,
        trips=("test_a_release_in_development_runs_forgiven",),
    ),
    Plant(
        # A year past the newest row with no row added for the next release.
        "tests/test_python_lifecycle.py",
        (_LIFECYCLE,),
        _outlive_the_table,
        trips=("test_the_table_knows_the_release_in_development",),
    ),
    Plant(
        # A classifier for a release the schedule table never dated.
        "tests/test_python_lifecycle.py",
        ("pyproject.toml",),
        lambda tree: _edit(
            tree,
            "pyproject.toml",
            '    "Programming Language :: Python :: 3.12",',
            '    "Programming Language :: Python :: 3.9",\n'
            '    "Programming Language :: Python :: 3.12",',
        ),
        trips=("test_every_release_the_tree_names_has_a_row",),
    ),
    Plant(
        "tests/test_python_lifecycle.py",
        ("pyproject.toml",),
        _shift_a_floor_statement,
        trips=("test_every_statement_of_the_floor_names_the_oldest_supported_release",),
    ),
    Plant(
        "tests/test_python_lifecycle.py",
        ("pyproject.toml",),
        _lower_the_newest_statement,
        trips=("test_every_statement_of_the_newest_release_names_it",),
    ),
    Plant(
        # A supported release whose classifier was never added.
        "tests/test_python_lifecycle.py",
        ("pyproject.toml",),
        lambda tree: _edit(
            tree,
            "pyproject.toml",
            '    "Programming Language :: Python :: 3.15",\n',
            "",
        ),
        trips=("test_the_classifiers_name_the_supported_releases",),
    ),
    Plant(
        # A month read as inclusive the wrong way: the floor's security support
        # runs one month past the one its PEP gives.
        "tests/test_python_lifecycle.py",
        (_LIFECYCLE,),
        lambda tree: _edit(
            tree,
            _LIFECYCLE,
            'Release(10, 619, _day("2020-10-05"), _day("2021-08-03"), (2026, 10)),',
            'Release(10, 619, _day("2020-10-05"), _day("2021-08-03"), (2026, 11)),',
        ),
        trips=("test_the_calendar_moves_the_floor_and_admits_the_next_release",),
    ),
    Plant(
        # The meet the table's own comment warns against: `int & ~str` is `int`,
        # so two columns name one set.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_relation_ledger.py",
            '    "Intersection": intersection(int, complement(bool)),',
            '    "Intersection": intersection(int, complement(str)),',
        ),
        trips=("test_the_representatives_are_that_many_different_sets",),
    ),
    Plant(
        # A predicate standing for the refinement node: its opacity is what every
        # decline in its row and column would then report.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_relation_ledger.py",
            '    "Refine": Annotated[int, at.Ge(0)],',
            '    "Refine": Annotated[int, at.Predicate(bool)],',
        ),
        trips=("test_a_representative_denotes_a_set_the_procedure_can_read",),
    ),
    Plant(
        # A decline whose reason was dropped.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_relation_ledger.py",
            '    ("Instance", "Complement"): THE_CLASS_ORDER_IS_OPEN,\n',
            "",
        ),
        trips=("test_every_pair_is_proved_refuted_or_declined_with_a_reason",),
    ),
    Plant(
        # The corpus trimmed of the only value refuting two inclusions.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(tree, "tests/test_relation_ledger.py", "    -1,\n", ""),
        trips=("test_a_refutation_carries_a_value_the_walk_checks",),
    ),
    Plant(
        # The witness helper with its two schemas transposed: every proof is
        # then searched for a value on the wrong side of it.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_relation_ledger.py",
            "if built[subject].is_valid(value) and not"
            " built[supertype].is_valid(value)",
            "if built[supertype].is_valid(value) and not"
            " built[subject].is_valid(value)",
        ),
        trips=("test_a_proof_is_not_refuted_by_the_corpus",),
    ),
    Plant(
        # An incompleteness filed as a necessity: a decline the corpus refutes,
        # moved to the table of the ones nothing decides.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edits(
            tree,
            "tests/test_relation_ledger.py",
            ('    ("Anything", "Ref"): A_FIXPOINT_IS_LOWERED_ONCE,\n', ""),
            (
                '    ("AttrRecord", "Complement"): THE_CLASS_ORDER_IS_OPEN,\n',
                (
                    '    ("AttrRecord", "Complement"): THE_CLASS_ORDER_IS_OPEN,\n'
                    '    ("Anything", "Ref"): THE_CLASS_ORDER_IS_OPEN,\n'
                ),
            ),
        ),
        trips=("test_a_decline_is_recorded_as_the_kind_it_is",),
    ),
    Plant(
        # A reason kept for a pair the procedure decides.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_relation_ledger.py",
            '    ("AttrRecord", "Complement"): THE_CLASS_ORDER_IS_OPEN,\n',
            '    ("AttrRecord", "Complement"): THE_CLASS_ORDER_IS_OPEN,\n'
            '    ("Int", "Anything"): THE_CLASS_ORDER_IS_OPEN,\n',
        ),
        trips=("test_no_reason_outlives_the_pair_it_excuses",),
    ),
    Plant(
        # The corpus trimmed of every value of one kind.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(
            tree, "tests/test_relation_ledger.py", '    b"",\n    b"x",\n', ""
        ),
        trips=("test_the_corpus_reaches_every_representative",),
    ),
    Plant(
        # The one row driving a hook the binding reads, dropped as redundant.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_relation_ledger.py",
            '    "an abstract base with a registration": Hooked(\n'
            '        _Registered, 1, "a", "__subclasscheck__"\n'
            "    ),\n",
            "",
        ),
        trips=("test_every_hook_the_binding_reads_has_a_row",),
    ),
    Plant(
        # A row with its two sides transposed. The walk and the class agree on
        # both values, so the equality with `isinstance` alone passes it.
        "tests/test_relation_ledger.py",
        ("tests/test_relation_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_relation_ledger.py",
            'Hooked(_Hooked, 1, "a", "__instancecheck__")',
            'Hooked(_Hooked, "a", 1, "__instancecheck__")',
        ),
        trips=("test_the_walk_answers_where_the_relation_declines",),
    ),
    Plant(
        # An excuse left for a tool the group dropped.
        "tests/test_suite_installs.py",
        ("tests/test_suite_installs.py",),
        lambda tree: _edit(
            tree,
            "tests/test_suite_installs.py",
            '    "ty": "type-checks, once, on CPython",\n',
            '    "ty": "type-checks, once, on CPython",\n'
            '    "pytest-xdist": "spreads the suite over cores, which'
            ' no lane asks for",\n',
        ),
        trips=("test_every_excused_tool_is_in_the_dev_group",),
    ),
    Plant(
        # A pin respelled so the scan's key stops matching: the release's
        # installs drop out of the ledger, which reads the rest as all there is.
        "tests/test_suite_installs.py",
        (".github/workflows/release.yml",),
        lambda tree: _replace_all(
            tree, ".github/workflows/release.yml", '"pytest>=', '"pytest~='
        ),
        trips=("test_the_scan_reads_the_installs_that_are_there",),
    ),
    Plant(
        # A method the stub ships before the binding documents it.
        "tests/test_surface_outcomes.py",
        ("python/valgebra/_valgebra.pyi",),
        lambda tree: _edit(
            tree,
            "python/valgebra/_valgebra.pyi",
            "    def is_empty(self) -> bool: ...",
            "    def is_empty(self) -> bool: ...\n"
            "    def is_disjoint(self, other: object, /) -> bool: ...",
        ),
        trips=("test_the_documented_surface_is_the_shipped_one",),
    ),
    Plant(
        # A `Returns:` value reworded out of its backticks, which the reading
        # keys on.
        "tests/test_surface_outcomes.py",
        ("crates/valgebra-py/src/validator.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/validator.rs",
            "    ///     `None` if `obj` is a member of the schema's set.",
            "    ///     Nothing if `obj` is a member of the schema's set.",
        ),
        trips=("test_the_universe_is_read_out_of_the_binding",),
    ),
    Plant(
        # A docstring that grows a raise nobody drives.
        "tests/test_surface_outcomes.py",
        ("crates/valgebra-py/src/validator.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/validator.rs",
            "    ///     `True` if the schema denotes the empty set, else `False`.\n"
            "    fn is_empty(",
            "    ///     `True` if the schema denotes the empty set, else `False`.\n"
            "    ///\n"
            "    /// Raises:\n"
            "    ///     OverflowError: If the decision outruns its budget.\n"
            "    fn is_empty(",
        ),
        trips=("test_every_documented_raise_is_asserted_somewhere",),
    ),
    Plant(
        # The gap closed and the excuse kept: a test that drives the accepted
        # cell, beside the reason it cannot be driven.
        "tests/test_surface_outcomes.py",
        ("tests/test_planted_outcome.py",),
        lambda tree: _write(
            tree,
            "tests/test_planted_outcome.py",
            "import pytest\n\nfrom valgebra import Validator\n\n\n"
            "def test_planted() -> None:\n"
            "    with pytest.raises(ValueError):\n"
            "        Validator(int).simplify()\n",
        ),
        trips=("test_no_reason_outlives_the_gap_it_excuses",),
    ),
    Plant(
        # A code named in a paragraph and asserted by no test.
        "tests/test_use_case_ledger.py",
        ("tests/test_planted_prose.py",),
        lambda tree: _write(
            tree,
            "tests/test_planted_prose.py",
            '"""A paragraph about "object_type", which no test here asserts."""\n',
        ),
        trips=("test_a_cell_is_covered_by_code_rather_than_by_prose",),
    ),
    Plant(
        "tests/test_use_case_ledger.py",
        ("scripts/use_case_ledger.json",),
        lambda tree: _edit(
            tree,
            "scripts/use_case_ledger.json",
            '"recursion": "a reference reports through the definition it names, so '
            "the walk carries the definition's own code rather than this one\",",
            '"recursion": "not reached",',
        ),
        trips=("test_every_accepted_reason_is_a_sentence",),
    ),
    Plant(
        # A marker left behind by a rename.
        "tests/test_use_case_ledger.py",
        ("tests/test_planted_marker.py",),
        lambda tree: _write(
            tree,
            "tests/test_planted_marker.py",
            "# USE-CASE: Validator.renamed_away\n",
        ),
        trips=("test_every_marker_names_a_cell_that_exists",),
    ),
    Plant(
        # A code the matrix stops driving: its row deleted.
        "tests/test_use_case_ledger.py",
        ("tests/test_error_matrix.py",),
        lambda tree: _edit(
            tree,
            "tests/test_error_matrix.py",
            '    "string_pattern_mismatch": Case(\n'
            '        Annotated[str, Regex(r"\\\\d+")],\n'
            '        "ab",\n'
            '        nested=({"a": Annotated[str, Regex(r"\\\\d+")]},'
            ' {"a": "ab"}, ("a",)),\n'
            "        document='\"ab\"',\n"
            "    ),\n",
            "",
        ),
        trips=("test_every_reachable_code_is_driven_by_the_error_matrix",),
    ),
    Plant(
        # A row for a code the tree no longer writes.
        "tests/test_use_case_ledger.py",
        ("tests/test_error_matrix.py",),
        lambda tree: _edit(
            tree,
            "tests/test_error_matrix.py",
            '    "json_invalid":'
            ' "test_a_document_the_parser_refuses_reports_the_parse",\n',
            '    "json_invalid":'
            ' "test_a_document_the_parser_refuses_reports_the_parse",\n'
            '    "renamed_away":'
            ' "test_a_document_the_parser_refuses_reports_the_parse",\n',
        ),
        trips=("test_the_error_matrix_drives_no_code_the_walk_cannot_report",),
    ),
    Plant(
        # A test the matrix names, renamed.
        "tests/test_use_case_ledger.py",
        ("tests/test_error_matrix.py",),
        lambda tree: _edit(
            tree,
            "tests/test_error_matrix.py",
            '"json_invalid": "test_a_document_the_parser_refuses_reports_the_parse"',
            '"json_invalid": "test_a_document_the_parser_refuses"',
        ),
        trips=("test_every_test_the_matrix_names_exists",),
    ),
    Plant(
        # A code renamed in the library and kept in the hand-built corpus.
        "tests/test_use_case_ledger.py",
        ("crates/valgebra-core/tests/error_snapshots.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-core/tests/error_snapshots.rs",
            'violation("int_type", vec![], "int", "\'x\'"),',
            'violation("integer_type", vec![], "int", "\'x\'"),',
        ),
        trips=("test_the_snapshot_corpus_pins_codes_the_walk_can_write",),
    ),
    Plant(
        # A code whose matrix row goes, while the fail-fast helper still names
        # it: held in one mode of four.
        "tests/test_use_case_ledger.py",
        ("tests/test_error_matrix.py",),
        lambda tree: _edit(
            tree,
            "tests/test_error_matrix.py",
            '    "json_invalid":'
            ' "test_a_document_the_parser_refuses_reports_the_parse",\n',
            "",
        ),
        trips=("test_no_code_is_evidenced_by_the_fail_fast_helper_alone",),
    ),
    Plant(
        # This file's own marker, respelled: the repository check read as the
        # product suite. The probe string sits in the module docstring, which
        # the search cuts, so this trips only once the probe is code.
        "tests/test_use_case_ledger.py",
        ("tests/test_use_case_ledger.py",),
        lambda tree: _edit(
            tree,
            "tests/test_use_case_ledger.py",
            "\npytestmark = pytest.mark.repository\n",
            "\npytestmark = [pytest.mark.repository]\n",
        ),
        trips=("test_the_product_suite_is_what_is_searched",),
    ),
    Plant(
        # The package's re-export respelled absolute, which the surface reading
        # keys on as `_markers`: `Regex` leaves the universe unseen.
        "tests/test_use_case_ledger.py",
        ("python/valgebra/__init__.py",),
        lambda tree: _edit(
            tree,
            "python/valgebra/__init__.py",
            "from ._markers import Regex",
            "from valgebra._markers import Regex",
        ),
        trips=("test_the_universe_is_read_from_the_tree",),
    ),
    Plant(
        # A corpus test comparing the release itself rather than through the
        # helper, which this ledger reads.
        "tests/test_version_gates.py",
        ("crates/valgebra-py/src/build/interpreter.rs",),
        lambda tree: _edit(
            tree,
            "crates/valgebra-py/src/build/interpreter.rs",
            "if !Since(12).met(py) {",
            "if py.version_info() < (3, 12) {",
        ),
        trips=("test_a_corpus_spells_its_release_where_this_can_read_it",),
    ),
    Plant(
        # The corpus file moved: the listed path is gone, the scan skips it, and
        # the rule below is quantified over nothing.
        "tests/test_version_gates.py",
        (
            "crates/valgebra-py/src/build/interpreter.rs",
            "crates/valgebra-py/src/build/corpus.rs",
        ),
        lambda tree: _move(
            tree,
            "crates/valgebra-py/src/build/interpreter.rs",
            "crates/valgebra-py/src/build/corpus.rs",
        ),
        trips=("test_the_tree_has_version_gates_and_the_workflow_has_lanes",),
    ),
)


#: The test builds the input that breaks its claim -- a synthetic tree, table,
#: script or string -- and asserts the verdict on it, so it is its own plant.
OWN_PLANT = "builds the violating input itself and asserts the verdict on it"

#: The test skips wherever the gitignored notes are absent, which is every clone,
#: so a plant run in one judges nothing.
NOT_IN_A_CLONE = "reads notes a clone does not carry, so a plant there judges nothing"

#: The test holds an answer of the installed extension. The clone imports the
#: extension built from the tree under audit, so no plant in its source moves it.
INSTALLED_ANSWER = (
    "holds an answer of the installed extension, which no plant in the clone's "
    "source moves"
)


#: Ledger test functions no plant names, each with the reason, keyed
#: ``tests/<file>::<function>``. A ledger file holds more than its list: a test
#: that builds the violating input itself is its own plant, and one that reads
#: material a clone does not carry cannot be judged in one.
def _excuse(ledger: str, reason: str, *names: str) -> dict[str, str]:
    """Key each of a ledger file's `names` as the census reads it, with `reason`."""
    return {f"tests/{ledger}::{name}": reason for name in names}


UNPLANTED: dict[str, str] = {
    **_excuse(
        "test_changelog_ledger.py",
        OWN_PLANT,
        "test_a_line_outside_the_universe_is_never_stale",
    ),
    **_excuse(
        "test_citation_ledger.py",
        NOT_IN_A_CLONE,
        "test_every_cited_work_is_a_paper_the_shelf_holds",
        "test_every_numbered_result_is_one_its_paper_states",
        "test_the_table_and_the_shelf_name_the_same_papers",
    ),
    **_excuse(
        "test_citation_ledger.py",
        OWN_PLANT,
        "test_the_reader_of_a_statement_tells_it_from_a_mention",
    ),
    **_excuse(
        "test_cited_commits.py",
        OWN_PLANT,
        "test_the_check_would_see_an_orphan",
        "test_the_check_would_see_an_orphaned_tag",
    ),
    **_excuse(
        "test_closure_ledger.py",
        INSTALLED_ANSWER,
        "test_a_generator_is_not_reachable_by_the_obvious_derivation",
    ),
    **_excuse(
        "test_commit_messages.py",
        OWN_PLANT,
        "test_the_names_it_refuses_are_the_ones_it_matches",
    ),
    **_excuse(
        "test_completeness_ledger.py",
        INSTALLED_ANSWER,
        "test_a_literal_is_not_always_a_singleton",
        "test_a_record_missing_a_required_key_refutes_however_open_it_is",
        "test_a_spec_literal_type_is_not_a_singleton_either",
        "test_a_table_is_decided_however_it_was_written",
        "test_a_table_missing_a_member_refutes_at_every_size",
        "test_emptiness_claims_hold_on_the_universe",
        "test_frontend_rejects_non_value_objects",
        "test_integer_discreteness_rule_does_not_over_fire",
        "test_subtype_claims_hold_on_the_universe",
        "test_value_literals_still_build",
        "test_widening_a_table_by_a_member_is_decided_at_every_size",
        "test_widening_a_table_is_decided_at_the_sizes_a_table_reaches",
    ),
    **_excuse(
        "test_completeness_probe.py",
        INSTALLED_ANSWER,
        "test_every_predicate_is_the_proof_answer_of_a_relation",
        "test_the_three_answers_agree_with_the_two",
    ),
    **_excuse(
        "test_constraint_matrix.py",
        INSTALLED_ANSWER,
        "test_a_bound_of_another_kind_is_refused_however_ordered_the_base_is",
        "test_no_constraint_builds_a_schema_that_admits_nothing_and_says_otherwise",
    ),
    **_excuse(
        "test_crate_attributes.py",
        OWN_PLANT,
        "test_the_scan_reads_the_keyword_rather_than_the_word",
    ),
    **_excuse(
        "test_doc_examples.py",
        OWN_PLANT,
        "test_the_marker_is_read_as_a_comment",
    ),
    **_excuse(
        "test_feature_lanes.py",
        OWN_PLANT,
        "test_a_sweep_and_a_lint_are_not_test_runs",
    ),
    **_excuse(
        "test_floor_names.py",
        OWN_PLANT,
        "test_the_reader_sees_a_module_the_floor_does_not_ship",
        "test_the_reader_sees_a_name_a_release_takes_away",
        "test_the_reader_sees_the_two_forms_that_reddened_the_lanes",
    ),
    **_excuse(
        "test_frontend_refusals.py",
        OWN_PLANT,
        "test_a_pattern_that_holds_everything_is_not_evidence",
    ),
    **_excuse(
        "test_harness_conditionals.py",
        OWN_PLANT,
        "test_the_test_module_detector_distinguishes_the_two_shapes",
    ),
    **_excuse(
        "test_lane_coverage.py",
        OWN_PLANT,
        "test_a_comment_does_not_count_as_a_driver",
        "test_a_longer_name_does_not_satisfy_a_shorter_one",
        "test_the_stale_excuse_rules_fire",
    ),
    **_excuse(
        "test_local_gate.py",
        OWN_PLANT,
        "test_a_forced_colour_variable_does_not_reach_a_step",
        "test_a_step_only_a_runner_can_fill_in_is_unaccounted",
        "test_an_env_expression_the_workflow_defines_is_filled_in",
    ),
    **_excuse(
        "test_mutation_scope.py",
        OWN_PLANT,
        "test_the_glob_matcher_distinguishes_the_shapes_it_is_used_with",
    ),
    **_excuse(
        "test_relation_ledger.py",
        INSTALLED_ANSWER,
        "test_a_class_that_answers_membership_itself_is_declined",
    ),
    **_excuse(
        "test_suite_partition.py",
        OWN_PLANT,
        "test_an_import_is_read_from_the_syntax_and_not_the_text",
        "test_the_validator_head_is_read_and_a_nested_one_is_not",
    ),
    **_excuse(
        "test_theory_ledger.py",
        NOT_IN_A_CLONE,
        "test_every_cited_source_is_a_section_the_argument_has",
        "test_every_deviation_row_quotes_the_paragraph_restating_it",
        "test_every_deviation_the_argument_tables_is_restated_here",
        "test_every_result_the_argument_carries_is_restated_here",
    ),
    **_excuse(
        "test_use_case_ledger.py",
        OWN_PLANT,
        "test_a_call_with_no_assertion_is_named_and_not_asserted",
    ),
    **_excuse(
        "test_version_gates.py",
        OWN_PLANT,
        "test_a_job_that_forgives_itself_outright_enforces_no_interpreter",
        "test_the_helper_is_the_one_place_the_comparison_is_read_from",
    ),
}


def _ledgers() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "tests").glob("test_*.py")
        if "LEDGER:" in path.read_text(encoding="utf-8")
    )


def _test_functions(path: Path) -> set[str]:
    """Name the test functions pytest collects from `path`, as node-id suffixes."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names |= {
                f"{node.name}::{member.name}"
                for member in node.body
                if isinstance(member, ast.FunctionDef)
                and member.name.startswith("test_")
            }
    return names


def _working_files() -> list[str]:
    """Name the files the audited tree holds: tracked, and new but not ignored.

    A file added and not yet committed is part of the tree being audited; a
    clone without it runs every plant against a tree that is not that one.
    """
    listing = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [  # noqa: S607 - git is on the path of every machine that clones this
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
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
    clone is local, and costs a copy of the object store.

    The working files are then laid over the checkout, so the tree the
    plants run against is the one being audited rather than the last commit.
    """
    copy = tmp_path_factory.mktemp("tree") / "tree"
    subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "clone", "--quiet", "--local", "--no-hardlinks", str(ROOT), str(copy)],  # noqa: S607
        check=True,
        capture_output=True,
    )
    for name in _working_files():
        source = ROOT / name
        if not source.is_file():
            continue  # a submodule or a path removed since the index was written
        target = copy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    yield copy
    shutil.rmtree(copy, ignore_errors=True)


@functools.cache
def _audited_head() -> str:
    """Name the commit the clone starts from: the audited tree's own `HEAD`."""
    return _plant_git(ROOT, "rev-parse", "HEAD").strip()


def _repair(tree: Path, plant: Plant) -> None:
    """Undo a plant, so the next row starts from the tree the audit describes.

    The files it names come back from the audited tree, and so does the tip: a
    plant that rewrites a commit touches no path, and without the reset every
    later row would run on the history it left.
    """
    for name in plant.touches:
        source, target = ROOT / name, tree / name
        if source.is_file():
            shutil.copy2(source, target)
        else:
            target.unlink(missing_ok=True)
    _plant_git(tree, "reset", "--quiet", "--soft", _audited_head())


def _judged_nothing(output: str) -> bool:
    """Whether every test the plant names skipped itself rather than ran.

    A ledger reading git history skips those tests in a shallow clone -- which is
    where `scripts/gate.py` runs the suite, and the reason this project has a
    local gate at all. A skip is not a detection and it is not a miss either: the
    plant is unjudgeable there, and reporting it as a failure would redden the
    lane over a clone shape rather than over the tree. Only a run in which
    nothing else happened is one: a test that ran and passed is a miss.
    """
    summary = output.strip().splitlines()[-1] if output.strip() else ""
    return "skipped" in summary and not any(
        verdict in summary for verdict in ("passed", "failed", "error")
    )


def _failed(output: str, ledger: str) -> set[str]:
    """Name the test functions of `ledger` a `-rfE` summary reports failed or errored.

    Read per test, so a plant is credited only with the tests it names: a
    collection error names no test, and a failure elsewhere in the file is
    another plant's.
    """
    return {
        match.group(1)
        for match in re.finditer(
            rf"^(?:FAILED|ERROR) {re.escape(ledger)}::([\w:]+?)(?:\[| |$)",
            output,
            re.MULTILINE,
        )
    }


@pytest.mark.parametrize(
    "plant", PLANTS, ids=[plant.ledger.split("/")[-1] for plant in PLANTS]
)
def test_a_ledger_fails_on_the_defect_it_exists_to_catch(
    plant: Plant, tree: Path
) -> None:
    assert plant.trips, f"a plant for {plant.ledger} names no test it must fail"
    plant.apply(tree)
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
            [
                sys.executable,
                "-m",
                "pytest",
                *(f"{plant.ledger}::{name}" for name in plant.trips),
                "-q",
                "-rfE",
                "-p",
                "no:cacheprovider",
            ],
            cwd=tree,
            capture_output=True,
            text=True,
            check=False,
            # No bytecode: two plants writing one file to the same length within
            # a second leave a cached module whose mtime and size both match,
            # and the second run reads the first plant's code.
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    finally:
        _repair(tree, plant)
    if _judged_nothing(result.stdout):
        pytest.skip(
            f"{plant.ledger} skips the tests the plant names in this clone, so "
            "the plant cannot be judged: they read what a clone does not carry"
        )
    missed = sorted(set(plant.trips) - _failed(result.stdout, plant.ledger))
    assert not missed, (
        f"{plant.ledger} passed {missed} on a tree that breaks their claim. A "
        f"test that cannot fail is not evidence.\n{result.stdout[-2000:]}"
    )


def test_every_ledger_carries_a_plant() -> None:
    """A ledger with no row above is one nobody has shown to work."""
    marked = {f"tests/{path.name}" for path in _ledgers()}
    assert marked, "no LEDGER marker found in any test"
    planted = {plant.ledger for plant in PLANTS}
    # This file is a ledger over the others; its own plant would be circular.
    missing = sorted(marked - planted - {"tests/test_ledger_plants.py"})
    assert not missing, (
        f"ledgers with no planted defect: {missing}. Add a row to PLANTS that "
        "breaks the claim, or the ledger is an assertion nobody has run against "
        "a tree that violates it."
    )


def test_every_ledger_test_is_planted_or_excused() -> None:
    """A plant per test function, since one plant proves one test.

    A ledger file holds several independent assertions, and a plant that trips
    the first says nothing of the rest: each could be an ``X and not X`` and the
    file's one row would stay green. So every test function a ledger file
    collects is named by a plant's `trips`, or excused above with the reason no
    plant fits it.
    """
    planted = {f"{plant.ledger}::{name}" for plant in PLANTS for name in plant.trips}
    collected = {
        f"tests/{path.name}::{name}"
        for path in _ledgers()
        if path.name != "test_ledger_plants.py"
        for name in _test_functions(path)
    }
    assert collected, "no test function found in any ledger"
    bare = sorted(collected - planted - UNPLANTED.keys())
    assert not bare, (
        f"{len(bare)} ledger tests no plant names and nothing excuses:\n"
        + "\n".join(bare)
    )
    stale = sorted((planted | UNPLANTED.keys()) - collected)
    assert not stale, f"plants or excuses naming no ledger test: {stale}"
    both = sorted(planted & UNPLANTED.keys())
    assert not both, f"excused and planted at once: {both}"
