"""A version gate is a claim about the lanes, so the lanes have to stand on both sides.

A gate naming a release says two things at once: *below this, the thing is not
there*, and *at or above it, this is what happens*. Each half is a test only if
some lane runs an interpreter on that side of the line. A gate at a release no
lane reaches is code that runs nowhere and passes; a gate below the floor is a
guard always taken, which reads as caution and asserts nothing.

Both halves were untested until the binding's corpora gained a lane. The four
rows that named `typing.Required` and the star inside a subscript had never run
on the floor, because the only lane passing `interpreter-tests` measured
coverage on one interpreter -- so the release those rows *needed* was a fact
about the tree that nothing read. Writing the release onto each row fixed the
rows and left the same hole one level up: nothing said the release was one the
matrix installs.

Two spellings, one rule:

* Python spells a gate `sys.version_info < (3, n)`, read here from the syntax
  tree so a `skipif` split across lines is one comparison rather than two lines;
* the binding's corpora spell one `Since(n)`, which is the release a corpus row
  needs. That spelling is the whole of it: a bare `version_info` comparison in a
  corpus is a release this ledger cannot read, so it fails rather than passing
  over one.

A lane whose failure the workflow forgives is not a side to stand on. The 3.15
leg runs under `continue-on-error`, because a prerelease that breaks is news
rather than a defect -- so a gate whose only interpreter above it is that one is
a gate nothing enforces.

LEDGER: every release a version gate names has an enforced lane on each side
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

# A repository check: it reads the tree's own sources and the workflow, neither
# of which ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: The release a corpus row needs, as the corpora spell one.
_SINCE = re.compile(r"\bSince\((\d{1,2})\)")

#: The comparison the corpora must not spell for themselves, because a release
#: written here is one this ledger reads and one written any other way is not.
_RAW = re.compile(r"version_info\s*\(\s*\)\s*[<>=!]")

#: Where the binding keeps the tables that read a live interpreter.
CORPORA = (
    "crates/valgebra-py/src/build/interpreter.rs",
    "crates/valgebra-py/src/check/walk.rs",
    "crates/valgebra-py/src/oracle/interpreter.rs",
    "crates/valgebra-py/src/build.rs",
)


class Gate(ast.NodeVisitor):
    """Collect every release a module's `sys.version_info` comparisons name."""

    def __init__(self) -> None:
        self.releases: set[int] = set()

    def visit_Compare(self, node: ast.Compare) -> None:
        if "version_info" in ast.dump(node.left):
            for other in node.comparators:
                if isinstance(other, ast.Tuple) and len(other.elts) >= 2:
                    major, minor = other.elts[0], other.elts[1]
                    if (
                        isinstance(major, ast.Constant)
                        and major.value == 3
                        and isinstance(minor, ast.Constant)
                        and isinstance(minor.value, int)
                    ):
                        self.releases.add(minor.value)
        self.generic_visit(node)


#: Where a Python source can gate on a release. The shipped package first,
#: because a gate there is the one a caller runs into; globbed rather than
#: listed, since the direction that matters is a gate arriving somewhere nobody
#: thought to look.
SOURCES = ("tests/*.py", "python/**/*.py", "scripts/*.py")


def _python_gates() -> dict[int, list[str]]:
    """Give each release a source gates on, with the files that name it."""
    found: dict[int, list[str]] = {}
    paths = sorted(path for pattern in SOURCES for path in ROOT.glob(pattern))
    for path in paths:
        if path.name == Path(__file__).name:
            continue
        gate = Gate()
        gate.visit(ast.parse(path.read_text(encoding="utf-8")))
        for release in gate.releases:
            found.setdefault(release, []).append(path.name)
    return found


def _rust_gates() -> dict[int, list[str]]:
    """Give each release a corpus row needs, with the file that names it."""
    found: dict[int, list[str]] = {}
    for relative in CORPORA:
        path = ROOT / relative
        if not path.exists():
            continue
        for release in _SINCE.findall(path.read_text(encoding="utf-8")):
            found.setdefault(int(release), []).append(relative)
    return found


def _lanes() -> dict[str, bool]:
    """Give every interpreter the workflow installs, and whether it is enforced.

    A leg the workflow forgives is one whose red is not a merge gate, so the
    value is what the lane's `continue-on-error` says about that version rather
    than whether the lane exists.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    lanes: dict[str, bool] = {}
    for job in (workflow.get("jobs") or {}).values():
        strategy = job.get("strategy") or {}
        matrix = strategy.get("matrix") or {}
        versions = [str(value) for value in matrix.get("python-version", [])]
        versions += [
            str(entry["python-version"])
            for entry in matrix.get("include", [])
            if "python-version" in entry
        ]
        for step in job.get("steps") or []:
            with_ = step.get("with") if isinstance(step, dict) else None
            if isinstance(with_, dict) and "python-version" in with_:
                named = str(with_["python-version"])
                if named and "matrix." not in named:
                    versions.append(named)
        forgiven = job.get("continue-on-error", False)
        for version in versions:
            lanes[version] = lanes.get(version, False) or _enforced(forgiven, version)
    return lanes


def _enforced(forgiven: object, version: str) -> bool:
    """Whether a job's `continue-on-error` leaves this interpreter's red a gate.

    Two shapes, and reading only the first gets the second backwards. A job may
    forgive *one* leg with an expression naming its version, which is how the
    prerelease runs; it may also forgive itself outright with a literal, and a
    check looking for the version inside that literal finds nothing and calls
    the lane enforced -- the answer that hides a whole job nobody's merge waits
    on.
    """
    if forgiven is True or str(forgiven).strip().lower() == "true":
        return False
    return f"'{version}'" not in str(forgiven)


def _minor(version: str) -> int | None:
    """Read a lane's interpreter as its minor release, or `None` where it has no 3.x."""
    found = re.fullmatch(r"3\.(\d{1,2})t?", version)
    return int(found.group(1)) if found else None


def test_the_tree_has_version_gates_and_the_workflow_has_lanes() -> None:
    """Each half is read from the tree, so each is shown to have been read.

    Asked of the two spellings apart rather than of their union: a parse that
    reads none of the corpora would leave the rule below quantified over
    nothing and passing, which is the shape of a check that has stopped
    checking.
    """
    assert _python_gates(), "no test module gates on a release"
    assert _rust_gates(), "no corpus row names the release it needs"
    lanes = _lanes()
    assert len([v for v in lanes if _minor(v)]) >= 3, f"the workflow lists {lanes}"


@pytest.mark.parametrize("spelling", ["python", "rust"])
def test_every_gate_has_an_enforced_lane_on_each_side(spelling: str) -> None:
    gates = _python_gates() if spelling == "python" else _rust_gates()
    lanes = _lanes()
    enforced = sorted(
        minor
        for version, kept in lanes.items()
        if kept and (minor := _minor(version)) is not None
    )
    every = sorted(minor for version in lanes if (minor := _minor(version)) is not None)

    unreached = sorted(
        f"3.{release} ({', '.join(sorted(set(where)))})"
        for release, where in gates.items()
        if not any(lane >= release for lane in enforced)
    )
    assert not unreached, (
        f"releases gated on with no enforced lane at or above them: {unreached}. "
        "The guarded code runs on no lane whose red is a merge gate, so it is "
        "asserted by nothing."
    )

    ungated = sorted(
        f"3.{release} ({', '.join(sorted(set(where)))})"
        for release, where in gates.items()
        if not any(lane < release for lane in every)
    )
    assert not ungated, (
        f"releases gated on with no lane below them: {ungated}. The guard is "
        "taken on no interpreter, so it states a difference nothing reads."
    )


def _compares_outside_the_helper(source: str) -> list[int]:
    """Give the line numbers comparing `version_info` outside `impl Since`.

    The helper is the one place the comparison belongs, so the block it sits in
    is skipped and everything else is reported. The block ends where a brace
    returns to the first column, which is where `rustfmt` puts the end of a
    top-level item.
    """
    found = []
    inside = False
    for index, line in enumerate(source.splitlines(), start=1):
        if line.startswith("impl Since"):
            inside = True
        elif inside and line.startswith("}"):
            inside = False
        elif not inside and _RAW.search(line):
            found.append(index)
    return found


def test_a_corpus_spells_its_release_where_this_can_read_it() -> None:
    """The corpora name a release one way, because the other way is unreadable.

    A bare `py.version_info()` comparison is a gate this ledger walks past: it
    reads `Since(n)` and nothing else, so a corpus that compares for itself
    would carry a release no lane is held to -- which is the hole the rule
    above exists to close, reopened one file down.

    The helper itself compares, once, which is what makes the rest of the
    corpus able to say `Since(11)` and mean it.
    """
    raw = [
        f"{relative}:{line}"
        for relative in CORPORA
        if (path := ROOT / relative).exists()
        for line in _compares_outside_the_helper(path.read_text(encoding="utf-8"))
    ]
    assert not raw, (
        f"corpus lines comparing `version_info` directly: {raw}. Spell the "
        "release as `Since(n)`, which is the form this ledger reads and the "
        "form that says what the row needs."
    )


def test_the_helper_is_the_one_place_the_comparison_is_read_from() -> None:
    """The skip is a block, not a file: a second comparison below it is found."""
    inside = "impl Since {\n    fn met(self, py: Python) -> bool {\n"
    inside += "        py.version_info() >= (3, self.0)\n    }\n}\n"
    assert _compares_outside_the_helper(inside) == []
    assert _compares_outside_the_helper(
        inside + "fn other(py: Python) -> bool {\n    py.version_info() < (3, 12)\n}\n"
    ) == [7]


def test_a_job_that_forgives_itself_outright_enforces_no_interpreter() -> None:
    """The two shapes of `continue-on-error`, read apart."""
    assert _enforced(False, "3.14")
    assert _enforced("", "3.14")
    # One leg named, which is how the prerelease is forgiven.
    assert not _enforced("${{ matrix.python-version == '3.15' }}", "3.15")
    assert _enforced("${{ matrix.python-version == '3.15' }}", "3.14")
    # And the whole job, which names no version to look for.
    assert not _enforced(True, "3.14")
    assert not _enforced("true", "3.14")
