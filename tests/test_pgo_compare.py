"""What a profile buys is reported per shape, or not reported at all.

The lane that reads a profiled build against a plain one exists because a single
figure for "what PGO buys" is true of one shape and false of the next. Two
things make its table worth reading: it names what the profile *costs* beside
what it buys, and it refuses a pair of readings whose ratio would be a statement
about something other than the profile.

Both are driven here from readings written by hand -- no wheels, no timer -- so
each refusal is exercised rather than assumed. A comparison that cannot be shown
to refuse is a comparison that prints whatever it is given.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

# The repository checks are not the product suite: this file reads the tree and
# the scripts, neither of which ships in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "pgo_compare.py"

ENVIRONMENT = "cpython3.12-gil-valgebra-0.0.11"


def _load() -> ModuleType:
    """Import the script by path; ``scripts/`` is not an importable package.

    The directory joins the search path first: the script reads its shapes from
    `compare_gate`, which is a sibling module rather than an installed one.
    Registered in `sys.modules` before it runs, because the dataclass it defines
    reads its own module back out of there while it is being built.
    """
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("pgo_compare", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pgo = _load()


def _reading(label: str, **nanoseconds: float) -> Any:
    """Build a reading of one build, as the script's own dataclass.

    `Any` because the module is loaded by path: the class exists at run time and
    has no name a checker can resolve, which is the same reason `gate` is read
    dynamically in the sibling suites.
    """
    return pgo.Reading(
        label=label,
        environment=ENVIRONMENT,
        extension=f"/wheels/{label}/_valgebra.so",
        nanoseconds=dict(nanoseconds),
    )


def test_a_reading_survives_the_trip_through_a_file(tmp_path: Path) -> None:
    """The lane writes a reading and a second invocation reads it.

    They are different processes on different wheels -- that is the point of the
    comparison -- so the file is the whole of what one hands the other.
    """
    written = _reading("plain", scalar=12.5, wide_record=980.0)
    path = tmp_path / "plain.json"
    written.write(path)
    assert pgo.Reading.read(path) == written
    # Readable as data, not only by this script: the lane uploads it, and what
    # is uploaded is read by whoever decides what the matrix does.
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["label"] == "plain"
    assert loaded["nanoseconds"]["scalar"] == pytest.approx(12.5)


def test_the_table_names_what_the_profile_costs(capsys: pytest.CaptureFixture) -> None:
    """A gain on one shape and a loss on another are one reading, not two.

    The loss is the half a reader is tempted to leave out: a table of gains
    alone is the single figure this lane exists to refuse, and the shape that
    loses here -- the accepting walk over a wide record -- is the one most
    callers spend their time in.
    """
    plain = _reading("plain", scalar=10.0, large_array=100.0, wide_record=100.0)
    profiled = _reading("pgo", scalar=10.1, large_array=70.0, wide_record=125.0)

    assert pgo.report(plain, profiled) == pgo.EXIT_OK
    printed = capsys.readouterr().out
    assert "large_array (-30%)" in printed
    assert "wide_record (+25%)" in printed
    # A shape inside the band is named as neither, since a move smaller than the
    # machine's own is not a reading about the profile.
    assert "scalar" not in printed.split("the profile buys:")[1]


def test_a_comparison_across_environments_is_refused(
    capsys: pytest.CaptureFixture,
) -> None:
    """A ratio between two builds cancels the machine and not the interpreter.

    A global lock changes what a per-element loop costs, and those are the
    shapes a profile serves best, so a table across two interpreters would
    report the interpreters. A different valgebra is a different binary, which
    is the same failure a version later.
    """
    plain = _reading("plain", scalar=10.0)
    elsewhere = pgo.Reading(
        label="pgo",
        environment="cpython3.14-freethreaded-valgebra-0.0.11",
        extension="/wheels/pgo/_valgebra.so",
        nanoseconds={"scalar": 7.0},
    )
    assert pgo.report(plain, elsewhere) == pgo.EXIT_CANNOT_RUN
    assert "different environments" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("left", "right", "why"),
    [
        ({"scalar": 10.0}, {"build": 7.0}, "share no shape"),
        ({"scalar": 0.0}, {"scalar": 7.0}, "timer that did not run"),
    ],
)
def test_a_reading_that_decides_nothing_is_refused(
    left: dict[str, float],
    right: dict[str, float],
    why: str,
    capsys: pytest.CaptureFixture,
) -> None:
    """Two readings with nothing in common, and one with nothing in it.

    Each would otherwise print a table: an empty one, or one whose ratio is a
    division by a time nothing took.
    """
    plain = _reading("plain", **left)
    profiled = _reading("pgo", **right)
    assert pgo.report(plain, profiled) == pgo.EXIT_CANNOT_RUN
    assert why in capsys.readouterr().out


def test_the_shapes_are_the_comparison_gate_s_own() -> None:
    """One definition of the seven shapes, read by both lanes.

    A second set would drift from the first by a shape, and the two tables would
    then describe workloads that are not the same while looking as though they
    were.
    """
    assert Path(pgo.compare_gate.__file__) == SCRIPTS / "compare_gate.py"
    assert "def shapes(" in (SCRIPTS / "compare_gate.py").read_text(encoding="utf-8")


def test_the_script_measures_or_refuses_and_has_no_third_answer() -> None:
    """A measurement is not a gate, and its exit codes say which it is.

    The gate vocabulary's middle code means "this ran and the answer is no".
    What these numbers mean is a decision a page carries, so there is no answer
    of that shape here -- and a script that defined one would invite a caller to
    read a verdict into a reading.
    """
    assert (pgo.EXIT_OK, pgo.EXIT_CANNOT_RUN) == (0, 2)
    assert not hasattr(pgo, "EXIT_FAIL")


def test_the_usage_refuses_an_invocation_that_asks_for_nothing(
    capsys: pytest.CaptureFixture,
) -> None:
    """Neither recording nor comparing is not a run that measured nothing."""
    assert pgo.main([]) == pgo.EXIT_CANNOT_RUN
    assert "usage" in capsys.readouterr().out.lower()
