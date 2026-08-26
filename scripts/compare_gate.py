"""Competitive performance gate: valgebra against pydantic-core, by ratio.

The headline performance claim is that valgebra is pydantic-core-class on the
check it shares. This gate measures that claim across a matrix of realistic
schema/payload shapes and compares each shape's *ratio* of per-call time
(valgebra / pydantic-core) against a recorded baseline. A ratio cancels the
runner's absolute speed -- if the machine is slow, both libraries are slow in
proportion -- so it survives the shared-runner noise that an absolute wall-clock
budget cannot. Each per-call time is the minimum over many repeats, the stable
estimator that scheduling jitter inflates but never deflates.

The policy: a shape fails when valgebra's ratio rises past the recorded baseline
by more than the tolerance -- valgebra became materially slower relative to
pydantic-core, whether through its own regression or by ceding ground. The
tolerance is generous so only a gross regression trips the merge gate; the
recorded numbers, not the gate, are the fine-grained record (see
``docs/11-performance.md``).

Usage:
    python scripts/compare_gate.py            # check ratios against the baseline
    python scripts/compare_gate.py --update   # re-record the baseline ratios

Requires the ``bench`` dependency group (pydantic) and the built extension.

Three outcomes, three exit codes: **0** every shape is within tolerance, **1** a
shape regressed or the shape sets disagree, **2** the gate **could not run** --
a missing dependency, an unreadable baseline. A gate that could not run has
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
BASELINE_FILE = ROOT / "scripts" / "perf_compare.json"

# Per-shape call budget: repeats of `number` calls; the minimum per-call time is
# kept. Cheap shapes need more calls per repeat to rise above timer granularity.
REPEATS = 7


class Shape(TypedDict):
    valgebra: Callable[[object], object]
    pydantic: Callable[[object], object]
    data: object
    number: int


def _shapes() -> dict[str, Shape]:
    # Imported here, not at module scope: pydantic is a benchmark-only dependency
    # and valgebra is the built extension, so a module-level import would make
    # this file unimportable on every lane that has neither -- including the one
    # that drives `judge` to prove this gate can fail.
    from pydantic import TypeAdapter  # noqa: PLC0415

    from valgebra import Validator  # noqa: PLC0415

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
            pydantic=strict(TypeAdapter(TypedDict("Wide", record_fields))),  # type: ignore[operator]
            data=record_data,
            number=2_000,
        ),
        "deep_nesting": Shape(
            valgebra=Validator(nested_t).is_valid,
            pydantic=strict(TypeAdapter(nested_t)),
            data=nested_v,
            number=20_000,
        ),
    }


def _per_call_ns(call: Callable[[object], object], data: object, number: int) -> float:
    timer = timeit.Timer(lambda: call(data))
    best = min(timer.repeat(repeat=REPEATS, number=number))
    return best / number * 1e9


def _prepare() -> tuple[dict, float, dict[str, Shape]] | None:
    """Read the baseline and build the shapes, or report why neither happened.

    Both are preconditions rather than verdicts: an unreadable baseline and a
    missing benchmark dependency each mean the comparison did not take place,
    which must not read as "did not regress". `None` is the caller's signal to
    exit 2.
    """
    try:
        baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        tolerance = float(baseline["tolerance"])
    except (OSError, ValueError, KeyError) as err:
        print(f"compare_gate: cannot read the baseline: {err}")
        return None
    try:
        shapes = _shapes()
    except ImportError as err:
        # No pydantic, or no built extension: there is nothing to compare
        # against.
        print(f"compare_gate: cannot build the comparison shapes: {err}")
        return None
    return baseline, tolerance, shapes


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
    update = "--update" in sys.argv[1:]
    prepared = _prepare()
    if prepared is None:
        return EXIT_CANNOT_RUN
    baseline, tolerance, shapes = prepared
    recorded: dict[str, float] = baseline.get("ratios", {})
    if not warm_up(shapes):
        return EXIT_FAIL

    measured: dict[str, float] = {}
    rows: list[tuple[str, float, float, float]] = []
    for name, shape in shapes.items():
        vg = _per_call_ns(shape["valgebra"], shape["data"], shape["number"])
        pyd = _per_call_ns(shape["pydantic"], shape["data"], shape["number"])
        ratio = vg / pyd
        measured[name] = ratio
        rows.append((name, vg, pyd, ratio))

    width = max(len(name) for name in shapes)
    header = f"{'shape':<{width}}  {'valgebra':>12}  {'pydantic':>12}  {'ratio':>7}  {'baseline':>9}"  # noqa: E501
    print(header)
    failures: list[str] = []
    for name, vg, pyd, ratio in rows:
        base = recorded.get(name)
        base_str = f"{base:.3f}" if base is not None else "-"
        status = ""
        if not update and base is not None:
            ceiling = base * (1 + tolerance)
            if name in judge({name: ratio}, {name: base}, tolerance)[0]:
                status = f"  REGRESSION (> {ceiling:.3f})"
                failures.append(name)
        row = f"{name:<{width}}  {vg:>10.1f}ns  {pyd:>10.1f}ns  {ratio:>7.3f}  {base_str:>9}{status}"  # noqa: E501
        print(row)

    if update:
        baseline["ratios"] = {name: round(measured[name], 4) for name in shapes}
        BASELINE_FILE.write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nrecorded {len(measured)} baseline ratios (tolerance {tolerance:.0%})")
        return EXIT_OK

    # Every measured shape must have a baseline and vice versa: a shape added
    # without re-recording, or a stale baseline key, would otherwise pass the gate
    # unchecked rather than being measured against a recorded ceiling.
    if set(measured) != set(recorded):
        missing = ", ".join(sorted(set(measured) - set(recorded))) or "none"
        stale = ", ".join(sorted(set(recorded) - set(measured))) or "none"
        print(
            "\nbaseline shapes do not match measured shapes; re-record with "
            f"--update (missing baseline: {missing}; stale baseline: {stale})"
        )
        return EXIT_FAIL

    if failures:
        print(f"\nREGRESSION on: {', '.join(failures)}")
        return EXIT_FAIL
    print(f"\nOK: all shapes within {tolerance:.0%} of the recorded ratio.")
    return EXIT_OK


def judge(
    measured: dict[str, float], recorded: dict[str, float], tolerance: float
) -> tuple[list[str], bool]:
    """Decide the verdict from measured ratios alone, with no timing involved.

    Returns the regressing shapes and whether the shape sets disagree. Extracted
    from ``main`` so the gate's decision can be driven -- and shown to fail -- in
    a test, without running pydantic or a timer.
    """
    shapes_disagree = set(measured) != set(recorded)
    failures = [
        name
        for name, ratio in sorted(measured.items())
        if name in recorded and ratio > recorded[name] * (1 + tolerance)
    ]
    return failures, shapes_disagree


if __name__ == "__main__":
    sys.exit(main())
