"""Competitive performance gate: valgebra against pydantic-core, by ratio.

The headline performance claim is that valgebra is pydantic-core-class on the
check it shares. This gate measures that claim across a matrix of realistic
schema/payload shapes and compares each shape's *ratio* of per-call time
(valgebra / pydantic-core) against a recorded baseline. A ratio cancels the
runner's absolute speed -- if the machine is slow, both libraries are slow in
proportion -- so it survives the shared-runner noise that an absolute wall-clock
budget cannot. Each per-call time is the minimum over many repeats, the stable
estimator that scheduling jitter inflates but never deflates.

**The ceiling is a claim, not a measurement.** Each shape carries the ratio it
must stay under, chosen with headroom over what the shape measures and written
down as what the project says of itself -- "at most this much of pydantic-core
here". A recorded measurement would be a fourth number that travels badly: the
two libraries respond differently to a PGO build, an interpreter version and a
cache size, so ratios measured on one machine are 1.7x ratios measured on
another (`large_array`, 0.52 recorded on the bench runner against 0.88 on a
developer's box). A claim does not move when the machine does, and changing one
is an edit somebody argues for.

So this gate is the coarse tripwire: it catches ceding ground to pydantic-core,
on any machine, with no re-recording. The fine-grained work is
``scripts/perf_gate.py --against``, which compares a change to its own merge
base under cachegrind at 2%.

Usage:
    python scripts/compare_gate.py            # check ratios against the ceilings

Requires the ``bench`` dependency group (pydantic) and the built extension.

Three outcomes, three exit codes: **0** every shape is under its ceiling, **1** a
shape is over one or the shape sets disagree, **2** the gate **could not run** --
a missing dependency, an unreadable ceiling file. A gate that could not run has
proven nothing and must not read as one that passed.
"""

from __future__ import annotations

import json
import sys
import timeit
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

# Three outcomes, three exit codes; see the module docstring.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent
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

    Two readings during this gate's own history were of a binary nobody meant to
    measure: an editable install that silently fell back to a debug build, and a
    profiling build swapped in for symbols. Both produced figures off by a factor
    of ten or more, and both looked exactly like a regression. A timing run has
    to say what it timed.

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


def _shapes() -> dict[str, Shape]:
    # Imported here, not at module scope: pydantic is a benchmark-only dependency
    # and valgebra is the built extension, so a module-level import would make
    # this file unimportable on every lane that has neither -- including the one
    # that drives `judge` to prove this gate can fail.
    from pydantic import TypeAdapter  # noqa: PLC0415
    from pydantic import ValidationError as PydanticError  # noqa: PLC0415

    from valgebra import ValidationError, Validator  # noqa: PLC0415

    array_data = list(range(10_000))
    record_fields = {f"f{i}": int for i in range(50)}
    record_data = {f"f{i}": i for i in range(50)}

    def nested_type(depth: int) -> object:
        schema: object = int
        for _ in range(depth):
            schema = list[schema]  # type: ignore[valid-type]
        return schema

    def nested_value(depth: int) -> object:
        value: object = 0
        for _ in range(depth):
            value = [value]
        return value

    nested_t = nested_type(25)
    nested_v = nested_value(25)

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
    json_doc = json.dumps(
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
    )
    wide_record_type = TypedDict("Wide", record_fields)  # type: ignore[operator]
    # One field of fifty holds the wrong type, so both libraries walk again to
    # say which one.
    wrong_record = {**record_data, "f7": "not an int"}

    # Build every validator and adapter exactly once -- both libraries compile
    # the schema ahead of the hot path, so the per-call comparison must too.
    def strict(adapter: TypeAdapter) -> Callable[[object], object]:
        return lambda v: adapter.validate_python(v, strict=True)

    return {
        "scalar": Shape(
            valgebra=Validator(int).is_valid,
            pydantic=strict(TypeAdapter(int)),
            data=42,
            number=200_000,
        ),
        "large_array": Shape(
            valgebra=Validator(list[int]).is_valid,
            pydantic=strict(TypeAdapter(list[int])),
            data=array_data,
            number=200,
        ),
        "wide_record": Shape(
            valgebra=Validator(record_fields).is_valid,
            pydantic=strict(TypeAdapter(wide_record_type)),
            data=record_data,
            number=2_000,
        ),
        "deep_nesting": Shape(
            valgebra=Validator(nested_t).is_valid,
            pydantic=strict(TypeAdapter(nested_t)),
            data=nested_v,
            number=20_000,
        ),
        "json_document": Shape(
            valgebra=Validator(list[json_record]).is_valid_json,
            pydantic=TypeAdapter(list[json_record]).validate_json,
            data=json_doc,
            number=200,
        ),
        "build": Shape(
            valgebra=_building(lambda: Validator(record_fields)),
            pydantic=_building(lambda: TypeAdapter(wide_record_type)),
            data=None,
            number=200,
        ),
        "error_report": Shape(
            valgebra=_reporting(Validator(record_fields).validate, ValidationError),
            pydantic=_reporting(
                TypeAdapter(wide_record_type).validate_python, PydanticError
            ),
            data=wrong_record,
            number=2_000,
        ),
    }


def _per_call_ns(call: Callable[[object], object], data: object, number: int) -> float:
    timer = timeit.Timer(lambda: call(data))
    best = min(timer.repeat(repeat=REPEATS, number=number))
    return best / number * 1e9


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
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"compare_gate: cannot read the ceilings: {err}")
        return None
    try:
        shapes = _shapes()
    except ImportError as err:
        # No pydantic, or no built extension: there is nothing to compare
        # against.
        print(f"compare_gate: cannot build the comparison shapes: {err}")
        return None
    return ceilings, shapes


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


def main() -> int:
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
        vg = _per_call_ns(shape["valgebra"], shape["data"], shape["number"])
        pyd = _per_call_ns(shape["pydantic"], shape["data"], shape["number"])
        measured[name] = vg / pyd
        rows.append((name, vg, pyd, vg / pyd))

    over, disagree = judge(measured, ceilings)
    width = max(len(name) for name in shapes)
    print(
        f"{'shape':<{width}}  {'valgebra':>12}  {'pydantic':>12}  "
        f"{'ratio':>7}  {'ceiling':>8}"
    )
    for name, vg, pyd, ratio in rows:
        ceiling = ceilings.get(name)
        shown = f"{ceiling:.2f}" if ceiling is not None else "-"
        status = "  OVER CEILING" if name in over else ""
        print(
            f"{name:<{width}}  {vg:>10.1f}ns  {pyd:>10.1f}ns  "
            f"{ratio:>7.3f}  {shown:>8}{status}"
        )

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
        print("valgebra ceded ground to pydantic-core here, or the ceiling was")
        print("always wrong. Both are edits somebody argues for.")
        return EXIT_FAIL
    print(f"\nOK: all {len(measured)} shapes under their ceilings.")
    return EXIT_OK


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
    sys.exit(main())
