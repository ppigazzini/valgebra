"""Competitive performance gate: valgebra against pydantic, by ratio.

The headline performance claim is that valgebra is as fast as pydantic on the
check it shares. This gate measures that claim across a matrix of realistic
schema/payload shapes and compares each shape's *ratio* of per-call time
(valgebra / pydantic) against a recorded baseline. A ratio cancels the
runner's absolute speed -- if the machine is slow, both libraries are slow in
proportion -- so it survives the shared-runner noise that an absolute wall-clock
budget cannot. Each per-call time is the minimum over many repeats, the stable
estimator that scheduling jitter inflates but never deflates.

**The pydantic side is the entry point a caller holds,** not pydantic-core:
``TypeAdapter.validate_python`` in strict mode, called through a lambda. That
is a Python frame and a keyword argument a direct ``SchemaValidator`` call does
not pay, and on the scalar shape it is most of pydantic's time -- against
``SchemaValidator`` that shape reads about 1.0, where the ceiling against the
adapter is 0.30, on one box with a release build and no profile. A shape
walking a container spends its time inside the core on both sides, and the
frame moves its ratio little.

**The ceiling is a claim, not a measurement.** Each shape carries the ratio it
must stay under, chosen with headroom over what the shape measures and written
down as what the project says of itself -- "at most this much of pydantic's
time here". A recorded measurement would be a fourth number that travels badly: the
two libraries respond differently to a PGO build and to an interpreter version,
so a ratio recorded in one environment is not the ratio of another. The gap is
the interpreter rather than the box: on a single machine a schema nested
twenty-five deep takes 0.13 to 0.14 of pydantic's time under CPython 3.12 and
3.14 and 0.26 under the free-threaded build, where every read of an element out
of a mutable container takes that container's lock. A claim
does not move when the environment does, and changing one is an edit somebody
argues for.

So this gate is the coarse tripwire: it catches ceding ground to pydantic,
on any machine, with no re-recording. The fine-grained work is
``scripts/perf_gate.py --against``, which compares a change to its own merge
base under cachegrind, within its ``RELATIVE_TOLERANCE``.

**A ceiling a shape passes by a wide margin stops measuring it.** The JSON
document sits under a ceiling of 1.00 and clears it by a third, so its ratio
moves by a tenth or more with nothing red until somebody re-runs the gate and
reads the number. So beside each ceiling -- the claim -- the file carries
a *recorded* ratio and a per-shape tolerance, which is the ratchet the mutation
sweep and the instruction gate already have: a shape drifting past what it last
measured is red even while it is under what the project promises.

A recorded ratio belongs to an environment as much as to the two libraries, so
the block names the one it was taken in -- interpreter, global lock, and the
pydantic-core it was measured against, because a faster pydantic-core raises
every ratio without anything here changing. Run somewhere else, the ratchet says
so and judges the ceilings alone rather than reporting a drift it cannot read.

**The recorded ratios are a window's, not a run's.** The bench lane reads one
tree on several classes of machine, further apart than a shape's tolerance, so
a ratio recorded from one run fires on the next run that lands on another
class. Each recorded ratio is the median of a window of the lane's readings,
and each tolerance is widened, where it has to be, to half again the widest
reading above that median: no reading of the window fires against the floor it
set. The window is kept in the block beside the ratios it gave.

**The valgebra half of each shape is counted as well as timed.**
``--count SHAPE N`` calls that half N times and prints how many calls did the
work, with no pydantic imported: ``scripts/perf_gate.py --profiled`` runs it
under cachegrind against a profile-guided wheel built at a change and at its
base, which reads to the instruction what this gate's clock reads to a few
percent.

Usage:
    python scripts/compare_gate.py            # ceilings, and the ratchet where it arms
    python scripts/compare_gate.py --update   # one run's ratios, for a window
    python scripts/compare_gate.py --record-window WINDOW.json   # arm from a window
    python scripts/compare_gate.py --count json_document 100     # one half, counted

Requires the ``bench`` dependency group (pydantic) and the built extension;
``--count`` the built extension alone.

Three outcomes, three exit codes: **0** every shape is under its ceiling and has
not drifted, **1** a shape is over one, has drifted past its tolerance, or the
shape sets disagree, **2** the gate **could not run** -- a missing dependency, an
unreadable ceiling file. A gate that could not run has proven nothing and must
not read as one that passed.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import timeit
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from _lib import EXIT_CANNOT_RUN, EXIT_FAIL, EXIT_OK, ROOT, arguments

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping, Sequence


CEILING_FILE = ROOT / "scripts" / "perf_compare.json"

# Per-shape call budget: repeats of `number` calls; the minimum per-call time is
# kept. Cheap shapes need more calls per repeat to rise above timer granularity.
REPEATS = 7


class Shape(TypedDict):
    valgebra: Callable[[object], object]
    pydantic: Callable[[object], object]
    data: object
    number: int


def provenance() -> str | None:
    """Say which extension is being timed, and refuse one that is not optimised.

    A binary nobody meant to measure reads as a regression: an editable install
    that silently falls back to a debug build, or a profiling build swapped in
    for symbols, gives figures off by a factor of ten or more that look exactly
    like one. A timing run has to say what it timed.

    The module that is *loaded* is the one asked, not a file matching a glob: a
    package directory can hold an extension per interpreter, and naming the
    wrong one is the same failure a step later.

    The size is the tell for the build. A release extension is a couple of
    megabytes; a debug or profiling one carries its symbols and runs an order of
    magnitude larger, so a threshold between them separates the two without
    asking the compiler what it did.
    """
    from valgebra import _valgebra  # noqa: PLC0415

    loaded = getattr(_valgebra, "__file__", None)
    if loaded is None:
        return "the extension reports no file: nothing to name"
    extension = Path(loaded)
    size = extension.stat().st_size
    print(f"extension: {extension} ({size / 1_048_576:.1f} MiB)")
    if getattr(_valgebra, "_debug_build", False):
        return (
            f"{extension.name} was built with debug assertions, so it is a "
            "`maturin develop` build rather than a release one; a figure from "
            "it is an order of magnitude off and is not comparable with "
            "anything here. Rebuild with `--release`, or install the wheel."
        )
    return None


def comparison_versions() -> str:
    """Read the versions the figures are measured against, rather than write them."""
    from importlib import metadata  # noqa: PLC0415

    named = []
    for package in ("valgebra", "pydantic", "pydantic-core", "jsonschema"):
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError:
            found = "absent"
        named.append(f"{package} {found}")
    return ", ".join(named)


def _building(build: Callable[[], object]) -> Callable[[object], object]:
    """Time a compile, and answer the rig check the accept shapes answer.

    Both libraries compile a schema ahead of the hot path and a caller waits for
    it at import, so a regression here is one a user times. `warm_up` asks every
    valgebra side for `True`, which is what says the work happened rather than
    an exception being swallowed.
    """

    def call(_: object) -> bool:
        build()
        return True

    return call


def _reporting(
    validate: Callable[[object], object], failure: type[BaseException]
) -> Callable[[object], object]:
    """Time the explain path: the walk that says which field, not the bool.

    A value that fails is the whole point, so the callable returns `True` only
    when the report was actually built -- a shape whose payload started passing
    would take the accept path and read as a speed-up.
    """

    def call(value: object) -> bool:
        try:
            validate(value)
        except failure:
            return True
        return False

    return call


class Corpus(NamedTuple):
    """The types and payloads both libraries are asked about, built once.

    A shape's two sides must be asked the same question, and the counted half
    the question the timed one is: the instruction gate reads a shape through
    `valgebra_sides` without pydantic, so the types and values live here rather
    than beside either library's validators.
    """

    array: list[int]
    record_fields: dict[str, type]
    record: dict[str, int]
    wide_record: type
    nested: object
    nested_value: object
    json_records: object
    json_document: str
    wrong_record: dict[str, object]


def corpus() -> Corpus:
    record_fields = {f"f{i}": int for i in range(50)}
    record_data = {f"f{i}": i for i in range(50)}

    nested: object = int
    nested_value: object = 0
    for _ in range(25):
        nested = list[nested]  # type: ignore[valid-type]
        nested_value = [nested_value]

    # A JSON document of records: the shape a request path carries, and the one
    # where both libraries parse and check in a single pass.
    json_fields = {
        "id": int,
        "name": str,
        "email": str,
        "tags": list[str],
        "meta": dict[str, str],
    }
    json_record = TypedDict("JsonRecord", json_fields)  # type: ignore[operator]
    return Corpus(
        array=list(range(10_000)),
        record_fields=record_fields,
        record=record_data,
        wide_record=TypedDict("Wide", record_fields),  # type: ignore[operator]
        nested=nested,
        nested_value=nested_value,
        json_records=list[json_record],  # type: ignore[valid-type]
        json_document=json.dumps(
            [
                {
                    "id": i,
                    "name": "Ada",
                    "email": "a@b.c",
                    "tags": ["x", "y"],
                    "meta": {"k": "v"},
                }
                for i in range(200)
            ]
        ),
        # One field of fifty holds the wrong type, so both libraries walk again
        # to say which one.
        wrong_record={**record_data, "f7": "not an int"},
    )


def valgebra_sides(given: Corpus) -> dict[str, tuple[Callable[..., object], object]]:
    """Each shape's valgebra half and the payload it is called with.

    Every validator is built here, once, as the schema is compiled ahead of the
    hot path in both libraries; the build shape is the one whose call is the
    compile.
    """
    from valgebra import ValidationError, Validator  # noqa: PLC0415

    fields = given.record_fields
    return {
        "scalar": (Validator(int).is_valid, 42),
        "large_array": (Validator(list[int]).is_valid, given.array),
        "wide_record": (Validator(fields).is_valid, given.record),
        "deep_nesting": (Validator(given.nested).is_valid, given.nested_value),
        "json_document": (
            Validator(given.json_records).is_valid_json,
            given.json_document,
        ),
        "build": (_building(lambda: Validator(fields)), None),
        "error_report": (
            _reporting(Validator(fields).validate, ValidationError),
            given.wrong_record,
        ),
    }


def shapes() -> dict[str, Shape]:
    # Imported here, not at module scope: pydantic is a benchmark-only dependency
    # and valgebra is the built extension, so a module-level import would make
    # this file unimportable on every lane that has neither -- including the one
    # that drives `judge` to prove this gate can fail.
    from pydantic import TypeAdapter  # noqa: PLC0415
    from pydantic import ValidationError as PydanticError  # noqa: PLC0415

    given = corpus()
    ours = valgebra_sides(given)

    # Build every validator and adapter exactly once -- both libraries compile
    # the schema ahead of the hot path, so the per-call comparison must too.
    def strict(adapter: TypeAdapter) -> Callable[[object], object]:
        return lambda v: adapter.validate_python(v, strict=True)

    wide = given.wide_record
    theirs: dict[str, tuple[Callable[..., object], int]] = {
        "scalar": (strict(TypeAdapter(int)), 200_000),
        "large_array": (strict(TypeAdapter(list[int])), 200),
        "wide_record": (strict(TypeAdapter(wide)), 2_000),
        "deep_nesting": (strict(TypeAdapter(given.nested)), 20_000),
        "json_document": (TypeAdapter(given.json_records).validate_json, 200),
        "build": (_building(lambda: TypeAdapter(wide)), 200),
        "error_report": (
            _reporting(TypeAdapter(wide).validate_python, PydanticError),
            2_000,
        ),
    }
    return {
        name: Shape(
            valgebra=ours[name][0],
            pydantic=pydantic,
            data=ours[name][1],
            number=number,
        )
        for name, (pydantic, number) in theirs.items()
    }


def count(shape: str, calls: int) -> int:
    """Call one shape's valgebra half `calls` times, and say how many did the work.

    The reading the instruction gate takes under cachegrind: the count is the
    workload's checksum, which equals `calls` only where every call answered
    `True` -- what `warm_up` asks of each half before it is timed. A shape this
    gate does not define is refused with exit 2, as an unknown flag is.
    """
    sides = valgebra_sides(corpus())
    if shape not in sides:
        print(f"compare_gate: no shape {shape!r}; the shapes are {sorted(sides)}")
        return EXIT_CANNOT_RUN
    call, data = sides[shape]
    done = 0
    for _ in range(calls):
        done += call(data) is True
    print(done)
    return EXIT_OK


def per_call_ns(call: Callable[[object], object], data: object, number: int) -> float:
    timer = timeit.Timer(lambda: call(data))
    best = min(timer.repeat(repeat=REPEATS, number=number))
    return best / number * 1e9


def gil_enabled() -> bool:
    """Whether this interpreter runs under a global lock.

    The free-threaded build is a separate performance environment, not the same
    one running slower: every read of an element out of a mutable container
    takes that container's lock, so a shape whose cost is per-element pays a
    price no interpreter with a global lock pays. It is a supported lane, so it
    is held to ceilings it can hold, recorded beside the shared ones.

    A build too old to answer runs under a global lock, which is the safe read:
    it selects the tighter set.
    """
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def _prepare() -> tuple[dict[str, float], dict[str, Shape]] | None:
    """Read the ceilings and build the shapes, or report why neither happened.

    Both are preconditions rather than verdicts: an unreadable ceiling file and
    a missing benchmark dependency each mean the comparison did not take place,
    which must not read as "did not regress". `None` is the caller's signal to
    exit 2.
    """
    try:
        recorded = json.loads(CEILING_FILE.read_text(encoding="utf-8"))
        ceilings = {name: float(v) for name, v in recorded["ceilings"].items()}
        if not gil_enabled():
            ceilings |= {
                name: float(v)
                for name, v in recorded.get("free_threaded_ceilings", {}).items()
            }
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"compare_gate: cannot read the ceilings: {err}")
        return None
    try:
        built = shapes()
    except ImportError as err:
        # No pydantic, or no built extension: there is nothing to compare
        # against.
        print(f"compare_gate: cannot build the comparison shapes: {err}")
        return None
    return ceilings, built


def warm_up(shapes: dict[str, Shape]) -> bool:
    """Warm each shape, and confirm the gate is timing the ACCEPT path.

    Public because it is a refusal, and a refusal that cannot be driven from a
    test is not evidence.

    First-touch effects (lazy imports, the allocator) would otherwise skew the
    first shape measured. The membership assertion is the rig check: a
    correctness regression that made valgebra reject the data would take the
    fast reject path and read as a speed-up.
    """
    for name, shape in shapes.items():
        if shape["valgebra"](shape["data"]) is not True:
            print(f"payload for shape {name!r} is not accepted by valgebra")
            return False
        shape["pydantic"](shape["data"])
    return True


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Read the command line before measuring anything; an unknown flag exits 2."""
    parser = arguments(__doc__)
    # One mode a run: a recording, a window, a count and a comparison are four
    # things to do, and naming two would leave one of them silently undone.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--update", action="store_true", help="record the ratios and the ceilings"
    )
    mode.add_argument(
        "--record-window",
        type=Path,
        metavar="WINDOW",
        help="arm the ratchet from a JSON window of the lane's readings",
    )
    mode.add_argument(
        "--count",
        nargs=2,
        metavar=("SHAPE", "CALLS"),
        help="call one shape's valgebra half CALLS times and print how many did "
        "the work",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = ()) -> int:
    args = _arguments(argv)
    if args.count is not None:
        shape, calls = args.count
        if not calls.isdigit():
            print(f"compare_gate: {calls!r} is not a number of calls")
            return EXIT_CANNOT_RUN
        return count(shape, int(calls))
    if args.record_window is not None:
        _record_window(args.record_window)
        return EXIT_OK
    return _compare(update=args.update)


def _compare(*, update: bool) -> int:
    """Time every shape on both sides, report it, and judge the ratios."""
    prepared = _prepare()
    if prepared is None:
        return EXIT_CANNOT_RUN
    ceilings, shapes = prepared
    wrong_build = provenance()
    if wrong_build is not None:
        print(f"compare_gate: {wrong_build}")
        return EXIT_CANNOT_RUN
    print(f"measured against: {comparison_versions()}")
    if not warm_up(shapes):
        return EXIT_FAIL

    measured: dict[str, float] = {}
    rows: list[tuple[str, float, float, float]] = []
    for name, shape in shapes.items():
        vg = per_call_ns(shape["valgebra"], shape["data"], shape["number"])
        pyd = per_call_ns(shape["pydantic"], shape["data"], shape["number"])
        measured[name] = vg / pyd
        rows.append((name, vg, pyd, vg / pyd))

    over, disagree = judge(measured, ceilings)
    stored = _recorded_block()
    moved, unarmed = drifted(measured, stored)
    _report(rows, ceilings, stored, over, moved)
    if update:
        _record(measured)
        return EXIT_OK
    if unarmed is not None:
        print(f"\nratchet not armed: {unarmed}")
    return _verdict(measured, ceilings, stored, Findings(over, disagree, moved))


def _report(
    rows: list[tuple[str, float, float, float]],
    ceilings: dict[str, float],
    stored: dict[str, object],
    over: list[str],
    moved: list[str],
) -> None:
    """Print one line per shape: what it measured, what it promises, what it was."""
    ratios = stored.get("ratios", {})
    ratios = ratios if isinstance(ratios, dict) else {}
    width = max(len(name) for name, *_ in rows)
    print(
        f"{'shape':<{width}}  {'valgebra':>12}  {'pydantic':>12}  "
        f"{'ratio':>7}  {'ceiling':>8}  {'recorded':>9}"
    )
    for name, vg, pyd, ratio in rows:
        ceiling = ceilings.get(name)
        shown = f"{ceiling:.2f}" if ceiling is not None else "-"
        was = ratios.get(name)
        last = f"{float(was):.3f}" if was is not None else "-"
        status = ""
        if name in over:
            status = "  OVER CEILING"
        elif name in moved:
            status = "  DRIFTED"
        print(
            f"{name:<{width}}  {vg:>10.1f}ns  {pyd:>10.1f}ns  "
            f"{ratio:>7.3f}  {shown:>8}  {last:>9}{status}"
        )


class Findings(NamedTuple):
    """What the three readings of one run found, before any of them is ranked."""

    over: list[str]
    """Shapes past the ceiling the project claims."""
    disagree: bool
    """Whether a shape has no ceiling, or a ceiling no shape."""
    moved: list[str]
    """Shapes past what they last measured, by more than their own spread."""


def _verdict(
    measured: dict[str, float],
    ceilings: dict[str, float],
    stored: dict[str, object],
    found: Findings,
) -> int:
    """Turn the three readings into one exit code, loudest failure first."""
    over, disagree, moved = found
    # Every measured shape must carry a ceiling and vice versa: a shape added
    # without one would pass unchecked, and a ceiling for a shape that is gone
    # is a claim about nothing.
    if disagree:
        missing = ", ".join(sorted(set(measured) - set(ceilings))) or "none"
        stale = ", ".join(sorted(set(ceilings) - set(measured))) or "none"
        print(
            f"\nshapes and ceilings disagree (shape with no ceiling: {missing}; "
            f"ceiling with no shape: {stale})"
        )
        return EXIT_FAIL
    if over:
        print(f"\nOVER CEILING on: {', '.join(over)}")
        print("valgebra ceded ground to pydantic here, or the ceiling was")
        print("always wrong. Both are edits somebody argues for.")
        return EXIT_FAIL
    if moved:
        ratios = stored["ratios"]
        tolerance = stored["tolerance"]
        print(f"\nDRIFTED past its recorded ratio: {', '.join(moved)}")
        for name in moved:
            print(
                f"  {name}: {measured[name]:.3f} against "
                f"{float(ratios[name]):.3f} + {float(tolerance[name]):.3f}"
            )
        print("Still under the ceiling, which is why this is read separately: a")
        print("shape may lose ground for a long time without reaching a claim.")
        print("Re-record with --update and an argument, or find what moved.")
        return EXIT_FAIL
    print(f"\nOK: all {len(measured)} shapes under their ceilings.")
    return EXIT_OK


def _recorded_block() -> dict[str, object]:
    try:
        stored = json.loads(CEILING_FILE.read_text(encoding="utf-8")).get(
            "recorded", {}
        )
    except (OSError, ValueError):  # pragma: no cover - _prepare refuses first
        return {}
    return stored if isinstance(stored, dict) else {}


def _record(measured: dict[str, float]) -> None:
    """Write this run's ratios, the reading a window of the lane takes in.

    The floor the file carries is a window's (`--record-window`), and a
    recording committed from one run fails the test that holds the ratios to
    the window beside them: one run lands on one class of machine. The
    tolerance is left alone: it is the shape's spread across runs, which one
    run cannot see, and lowering it silently would make the ratchet fire on
    noise. Widening or narrowing it is an edit with its own argument, like a
    ceiling.
    """
    document = json.loads(CEILING_FILE.read_text(encoding="utf-8"))
    block = document.setdefault("recorded", {})
    block["environment"] = environment()
    block["ratios"] = {
        name: round(ratio, 3) for name, ratio in sorted(measured.items())
    }
    block.setdefault("tolerance", {})
    block.setdefault("unratcheted", {})
    CEILING_FILE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"\nrecorded {len(measured)} ratios for {block['environment']}")


def window_record(
    runs: Sequence[Mapping[str, float]], tolerance: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    """Read the ratios and tolerances a window of readings arms the ratchet with.

    For each shape the file judges: its ratio is the median of the window, and
    its tolerance the larger of the one carried and half again the widest
    reading above that median, rounded up to a two-hundredth. A shape is
    judged where it carries a tolerance, so one argued out of the ratchet is
    not recorded at all.
    """
    ratios: dict[str, float] = {}
    widened: dict[str, float] = {}
    for name, carried in sorted(tolerance.items()):
        readings = [run[name] for run in runs]
        median = statistics.median(readings)
        needed = math.ceil(round(1.5 * (max(readings) - median) * 200, 6)) / 200
        ratios[name] = round(median, 3)
        widened[name] = round(max(float(carried), needed), 3)
    return ratios, widened


def _record_window(path: Path) -> None:
    """Arm the ratchet from the window at `path`, and keep the window beside it.

    The window is a list of runs, each naming where it was read and carrying
    every shape's ratio from that run's table. The environment is this
    interpreter's, so the window is recorded in one matching the lane's.
    """
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    document = json.loads(CEILING_FILE.read_text(encoding="utf-8"))
    block = document.setdefault("recorded", {})
    ratios, tolerance = window_record(
        [run["ratios"] for run in runs], block.get("tolerance", {})
    )
    block["environment"] = environment()
    block["ratios"] = ratios
    block["tolerance"] = tolerance
    block["window"] = runs
    CEILING_FILE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    where = block["environment"]
    print(f"armed {len(ratios)} ratios from {len(runs)} runs for {where}")


def environment() -> str:
    """Name the environment a recorded ratio belongs to.

    Three things move every ratio without anything in this repository changing:
    the interpreter's minor version, whether it has a global lock, and the
    version of the library on the other side of the comparison. Naming them is
    what lets a recorded number be compared at all -- and what stops it being
    compared where it means nothing.

    The build is not in here because :func:`provenance` already refuses a
    non-release extension outright, and because whether the wheel was
    PGO-trained is not a question the loaded module answers; the recording is
    made on the lane that builds it the same way every time.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        core = version("pydantic-core")
    except PackageNotFoundError:  # pragma: no cover - the prepare step refuses first
        core = "absent"
    lock = "gil" if gil_enabled() else "freethreaded"
    release = f"cpython{sys.version_info.major}.{sys.version_info.minor}"
    return f"{release}-{lock}-pydantic_core-{core}"


def drifted(
    measured: dict[str, float], recorded: dict[str, object]
) -> tuple[list[str], str | None]:
    """Shapes that measure worse than they last did, by more than their spread.

    Returns the drifted shapes and, where the ratchet does not arm, the reason
    -- which is not a failure: a recorded ratio read in the wrong environment is
    a number about somewhere else, and judging against it would be the error
    this file exists to avoid twice over.

    Split from ``main`` so the verdict can be driven, and shown to fail, without
    running a timer or pydantic.
    """
    if not recorded or not recorded.get("ratios"):
        return [], "no ratios are recorded; `_how_recorded` in the file says how"
    was = recorded.get("environment")
    now = environment()
    if was != now:
        return [], f"recorded on {was}, running on {now}"
    ratios = recorded.get("ratios", {})
    tolerance = recorded.get("tolerance", {})
    if not isinstance(ratios, dict) or not isinstance(tolerance, dict):
        return [], "the recorded block is malformed"
    return sorted(
        name
        for name, ratio in measured.items()
        if name in ratios
        and ratio > float(ratios[name]) + float(tolerance.get(name, 0.0))
    ), None


def judge(
    measured: dict[str, float], ceilings: dict[str, float]
) -> tuple[list[str], bool]:
    """Decide the verdict from measured ratios alone, with no timing involved.

    Returns the shapes over their ceiling and whether the shape sets disagree.
    Extracted from ``main`` so the gate's decision can be driven -- and shown to
    fail -- in a test, without running pydantic or a timer.
    """
    shapes_disagree = set(measured) != set(ceilings)
    over = [
        name
        for name, ratio in sorted(measured.items())
        if name in ceilings and ratio > ceilings[name]
    ]
    return over, shapes_disagree


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
