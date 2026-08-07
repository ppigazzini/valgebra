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

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "mutation_gate.py"


def _run(
    work: Path,
    missed: list[str],
    baseline: list[str],
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out = work / "mutants.out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "caught.txt").write_text("some/file.rs:1:1: caught mutant\n")
    (out / "missed.txt").write_text("".join(line + "\n" for line in missed))
    (work / "scripts").mkdir(exist_ok=True)
    (work / "scripts" / "mutation_gate.py").write_text(GATE.read_text())
    (work / "scripts" / "mutation_baseline.json").write_text(
        json.dumps({"survivors": baseline}) + "\n"
    )
    return subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(work / "scripts" / "mutation_gate.py"), *(extra or [])],
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
    )


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


def test_the_three_exit_codes_are_distinct() -> None:
    # The vocabulary itself: a caller can dispatch on the code, which it cannot
    # if "could not run" and "failed" share one.
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("mutation_gate_codes", GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.EXIT_OK, module.EXIT_FAIL, module.EXIT_CANNOT_RUN) == (0, 1, 2)


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
