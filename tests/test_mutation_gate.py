"""The mutation ratchet must fail on a new survivor and pass on a known one.

A gate that cannot be shown to fail is not evidence. These drive
``scripts/mutation_gate.py`` against synthetic ``mutants.out`` fixtures so the
ratchet's pass/fail behaviour is itself tested, without running a real sweep.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "mutation_gate.py"


def _run(  # noqa: PLR0913 - a fixture per file the gate reads, named at each call
    work: Path,
    missed: list[str],
    baseline: list[str],
    *,
    extra: list[str] | None = None,
    accepted: dict[str, str] | None = None,
    timed_out: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out = work / "mutants.out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "caught.txt").write_text("some/file.rs:1:1: caught mutant\n")
    (out / "missed.txt").write_text("".join(line + "\n" for line in missed))
    if timed_out is not None:
        (out / "timeout.txt").write_text("".join(line + "\n" for line in timed_out))
    (work / "scripts").mkdir(exist_ok=True)
    (work / "scripts" / "mutation_gate.py").write_text(GATE.read_text())
    recorded: dict[str, object] = {"survivors": baseline}
    if accepted is not None:
        recorded["_accepted"] = accepted
    (work / "scripts" / "mutation_baseline.json").write_text(
        json.dumps(recorded) + "\n"
    )
    return subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(work / "scripts" / "mutation_gate.py"), *(extra or [])],
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
    )


def test_an_accepted_note_naming_no_survivor_fails(tmp_path: Path) -> None:
    """An excuse must not outlive the thing it excuses.

    The accepted set carries a hand-written argument per survivor, and nothing
    read those arguments: a note whose mutant a test started killing, or one
    spelled the way no survivor line spells it, sat in the file and the gate
    passed. Both were true of this tree at once.
    """
    survivor = "crates/x/src/a.rs: replace + with - in f"
    result = _run(
        tmp_path,
        missed=["crates/x/src/a.rs:10:5: replace + with - in f"],
        baseline=[survivor],
        accepted={"a.rs: replace + with - in f": "spelled as no survivor is"},
    )
    assert result.returncode == 1
    assert "ACCEPTED WITHOUT A SURVIVOR" in result.stdout


def test_an_accepted_note_matching_a_survivor_passes(tmp_path: Path) -> None:
    survivor = "crates/x/src/a.rs: replace + with - in f"
    result = _run(
        tmp_path,
        missed=["crates/x/src/a.rs:10:5: replace + with - in f"],
        baseline=[survivor],
        accepted={survivor: "equivalent: it cannot change an answer"},
    )
    assert result.returncode == 0
    assert "no new survivors" in result.stdout


def test_a_new_survivor_fails_the_gate(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        missed=["crates/x/src/a.rs:10:5: replace + with - in f"],
        baseline=[],
    )
    assert result.returncode == 1
    assert "NEW SURVIVOR" in result.stdout


def test_a_baselined_survivor_passes(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        missed=["crates/x/src/a.rs:10:5: replace + with - in f"],
        # Same identity, different line: the ratchet keys on file + mutation,
        # not on the drifting line:col.
        baseline=["crates/x/src/a.rs: replace + with - in f"],
    )
    assert result.returncode == 0
    assert "no new survivors" in result.stdout


def test_a_killed_baseline_survivor_fails_until_the_baseline_shrinks(
    tmp_path: Path,
) -> None:
    # An improvement, and still a failure: the baseline is an accepted hole, and
    # an accepted hole the tree no longer has must not stay standing. Left there
    # it silently re-accepts a future survivor with the same identity.
    result = _run(
        tmp_path,
        missed=[],
        baseline=["crates/x/src/a.rs: replace + with - in f"],
    )
    assert result.returncode == 1
    assert "STALE BASELINE ENTRY" in result.stdout


def test_an_empty_output_dir_refuses_to_pass(tmp_path: Path) -> None:
    (tmp_path / "mutants.out").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "mutation_gate.py").write_text(GATE.read_text())
    (tmp_path / "scripts" / "mutation_baseline.json").write_text('{"survivors": []}\n')
    result = subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(tmp_path / "scripts" / "mutation_gate.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    # A broken detector must not read as a clean tree, and must be
    # distinguishable from one: exit 2 is "could not run", never "failed".
    assert result.returncode == 2
    assert "did not run" in result.stderr


def test_a_baseline_matching_the_sweep_exactly_passes(tmp_path: Path) -> None:
    # The only shape that passes: every survivor accepted, and every accepted
    # entry still a survivor.
    result = _run(
        tmp_path,
        missed=["crates/x/src/a.rs:9:3: replace f -> bool with true"],
        baseline=["crates/x/src/a.rs: replace f -> bool with true"],
    )
    assert result.returncode == 0
    assert "no new survivors" in result.stdout


def test_a_missing_output_directory_is_could_not_run(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "mutation_gate.py").write_text(GATE.read_text())
    (tmp_path / "scripts" / "mutation_baseline.json").write_text('{"survivors": []}\n')
    result = subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(tmp_path / "scripts" / "mutation_gate.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "is absent" in result.stderr


def test_a_missing_baseline_is_could_not_run(tmp_path: Path) -> None:
    # No baseline is not "the baseline holds" and not "a survivor appeared": the
    # ratchet has nothing to compare against and says so with its own code.
    out = tmp_path / "mutants.out"
    out.mkdir()
    (out / "caught.txt").write_text("some/file.rs:1:1: caught mutant\n")
    (out / "missed.txt").write_text("")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "mutation_gate.py").write_text(GATE.read_text())
    result = subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(tmp_path / "scripts" / "mutation_gate.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "no core baseline" in result.stderr


def test_a_partial_sweep_opts_out_of_the_expiry_direction(tmp_path: Path) -> None:
    # An `--in-diff` sweep never generates most of the baseline, so every
    # untouched accepted survivor is absent by construction. Without --new-only
    # that reads as a whole baseline gone stale; with it, only a NEW survivor
    # fails.
    baseline = [
        "crates/x/src/a.rs: replace + with - in f",
        "crates/x/src/b.rs: replace + with - in g",
    ]
    partial = _run(tmp_path, missed=[], baseline=baseline, extra=["--new-only"])
    assert partial.returncode == 0
    assert "in this diff" in partial.stdout

    full = _run(tmp_path, missed=[], baseline=baseline)
    assert full.returncode == 1
    assert "STALE BASELINE ENTRY" in full.stdout

    # A new survivor still fails, which is the whole point of the lane.
    fresh = _run(
        tmp_path,
        missed=["crates/x/src/c.rs:1:1: replace + with - in h"],
        baseline=baseline,
        extra=["--new-only"],
    )
    assert fresh.returncode == 1
    assert "NEW SURVIVOR" in fresh.stdout


def test_re_recording_keeps_the_reasons_beside_the_set(tmp_path: Path) -> None:
    # The argument for why a survivor is accepted lives in a note beside the set.
    # A re-record that dropped it would leave the accepted set with no reason
    # behind it, which is the failure the whole "written reason" rule exists to
    # prevent.
    out = tmp_path / "mutants.out"
    out.mkdir()
    (out / "caught.txt").write_text("x.rs:1:1: caught\n")
    (out / "missed.txt").write_text("a.rs:1:1: replace f -> bool with true\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "mutation_gate.py").write_text(GATE.read_text())
    baseline = tmp_path / "scripts" / "mutation_baseline.json"
    baseline.write_text(
        json.dumps({"_why": "the argument", "survivors": []}) + "\n",
    )
    result = subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(tmp_path / "scripts" / "mutation_gate.py"), "--update"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    recorded = json.loads(baseline.read_text())
    assert recorded["_why"] == "the argument"
    assert recorded["survivors"] == ["a.rs: replace f -> bool with true"]


def _tracked_tree(work: Path, files: list[str]) -> None:
    """Make `work` a checkout whose index lists exactly `files`.

    Staged rather than committed: `git ls-files` reads the index, and staging
    needs no committer -- which is the point, since a runner has none.
    """
    subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        ["git", "-C", str(work), "init", "--quiet"],  # noqa: S607
        check=True,
    )
    for name in files:
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// a file the baseline names\n")
    subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        ["git", "-C", str(work), "add", *files],  # noqa: S607
        check=True,
    )


def test_an_accepted_survivor_for_a_moved_file_fails(tmp_path: Path) -> None:
    """A baseline keyed by path goes stale when the path moves, and says so here.

    Eight entries moved with the frontend's surfaces and nothing said so until
    the sweep read every one of that file's survivors as new -- nine minutes
    into a shard, with the mutants listed and no hint that what changed was the
    path. A path is cheaper to check than a sweep and is checked before one.
    """
    _tracked_tree(tmp_path, ["crates/x/src/b.rs"])
    result = _run(
        tmp_path,
        missed=[],
        baseline=["crates/x/src/a.rs: replace + with - in f"],
    )
    assert result.returncode == 1
    assert "ACCEPTED FOR A FILE THAT IS NOT HERE" in result.stdout
    assert "crates/x/src/a.rs" in result.stdout


def test_an_accepted_survivor_for_a_tracked_file_passes(tmp_path: Path) -> None:
    """The other direction: the same entry, re-keyed to where the code went."""
    _tracked_tree(tmp_path, ["crates/x/src/b.rs"])
    result = _run(
        tmp_path,
        missed=["crates/x/src/b.rs:10:5: replace + with - in f"],
        baseline=["crates/x/src/b.rs: replace + with - in f"],
    )
    assert result.returncode == 0, result.stdout


def test_a_timeout_is_a_rig_fault_rather_than_a_survivor(tmp_path: Path) -> None:
    """A mutant whose experiment never finished is a claim about the runner.

    `docs/dev/13-glossary.md` says a rig fault is "a run that produced no
    verdict -- a timeout ... Neither a pass nor a failure, and reported as
    itself", and `07-tooling-ci.md` says the same of a mutant whose experiment
    cannot finish. The gate folded `timeout.txt` into the survivors, so an
    overloaded runner read as a hole in the tests -- and the two are repaired in
    opposite directions.
    """
    result = _run(tmp_path, [], [], timed_out=["some/file.rs:1:1: a slow mutant"])
    assert result.returncode == 2, result.stdout + result.stderr
    said = result.stdout + result.stderr
    assert "rig fault" in said, said
    assert "1 mutant" in said or "1 mutation" in said, said


def test_a_timeout_is_not_recorded_as_an_accepted_survivor(tmp_path: Path) -> None:
    """And `--update` must not bake one into the baseline.

    That is the reading that outlives the run: a timeout accepted once is a
    permanent excuse for a mutant nobody ever judged, and the ratchet then fails
    the day the runner is fast enough to judge it.
    """
    result = _run(
        tmp_path,
        ["some/file.rs:1:1: a real survivor"],
        [],
        extra=["--update"],
        timed_out=["some/file.rs:2:2: a slow mutant"],
    )
    assert result.returncode == 2, result.stdout + result.stderr
    recorded = json.loads((tmp_path / "scripts" / "mutation_baseline.json").read_text())
    assert recorded["survivors"] == [], recorded


def test_a_survivor_still_fails_the_ratchet_beside_a_clean_timeout_file(
    tmp_path: Path,
) -> None:
    """An empty `timeout.txt` is not a fault, and the survivor direction holds."""
    result = _run(tmp_path, ["some/file.rs:1:1: a real survivor"], [], timed_out=[])
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NEW SURVIVOR" in result.stdout, result.stdout
