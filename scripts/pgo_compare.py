"""Time each comparison shape on a profiled build and on a plain one.

Whether a profile pays is a question **per shape**, not per project. Profile-
guided optimisation arranges what fat LTO has left to arrange, so a shape whose
cost is one hot loop over one element type is laid out straight and a shape whose
cost is many warm paths -- fifty key lookups, each dispatching on its field's own
schema -- can be laid out worse. A single figure for "what PGO buys" therefore
states something true of one shape and false of the next, which is the claim this
exists to stop anyone making.

The shapes are `scripts/compare_gate.py`'s own, so the table below and the
competitive ratios speak of the same seven workloads rather than two sets that
drift apart.

A reading carries the environment it was taken in. A ratio between two builds
cancels the machine and not the interpreter: a global lock changes what a
per-element loop costs, and a different valgebra is a different binary. A
comparison across either is refused rather than printed.

A shape counts as moved when it moves past **its own** recorded spread, which
`scripts/perf_compare.json` carries per shape because the spreads differ by an
order of magnitude between them. A single threshold would read noise as a
finding on one shape and hide a real move on another.

Each wheel is timed in its own environment, and the two readings are compared:

    uv run --group bench maturin build --release       --out plain
    uv run --group bench maturin build --release --pgo --out profiled
    uv venv .venv-plain && uv pip install --python .venv-plain/bin/python plain/*.whl
    uv venv .venv-pgo   && uv pip install --python .venv-pgo/bin/python profiled/*.whl
    .venv-plain/bin/python scripts/pgo_compare.py --record plain.json --label plain
    .venv-pgo/bin/python   scripts/pgo_compare.py --record pgo.json   --label pgo
    python scripts/pgo_compare.py --compare plain.json pgo.json

The comparison reports what the profile buys and what it costs. It does not
decide: whether the release matrix keeps `pgo: true` is a decision about which
shapes the project serves, and `docs/11-performance.md` is where that decision
and its reason are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import compare_gate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# A measurement, not a gate: it reports what it read, or refuses to read. The
# gate vocabulary's middle code says "this ran and the answer is no", and there
# is no answer of that shape here -- what the numbers mean is a decision a page
# carries, not a verdict this script reaches.
EXIT_OK = 0
EXIT_CANNOT_RUN = 2

#: The spread a shape with no recorded one is read against. `error_report` is
#: that shape: it times the path that raises and formats a Python exception, and
#: it spreads about a third of its own value, which is why the competitive gate
#: records an argument for it rather than a number.
DEFAULT_BAND = 0.33


def bands() -> dict[str, float]:
    """Read each shape's spread across runs of one build, as the lane measured it.

    A difference smaller than that spread is the machine, not the profile. The
    figures are the competitive gate's own recording rather than a threshold
    chosen here: a band nobody measured would turn noise into a finding on
    whichever shape it was too tight for.
    """
    try:
        recorded = json.loads(compare_gate.CEILING_FILE.read_text(encoding="utf-8"))
        spread = recorded["recorded"]["tolerance"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return {str(name): float(value) for name, value in spread.items()}


@dataclass(frozen=True, slots=True)
class Reading:
    """One build's per-shape times, with the environment they belong to."""

    label: str
    environment: str
    extension: str
    nanoseconds: dict[str, float]

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Reading:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            label=str(loaded["label"]),
            environment=str(loaded["environment"]),
            extension=str(loaded["extension"]),
            nanoseconds={str(k): float(v) for k, v in loaded["nanoseconds"].items()},
        )


def environment() -> str:
    """Name the environment a reading belongs to.

    The interpreter's minor version and its lock, because both change what a
    per-element loop costs, and the valgebra version, because two builds of
    different sources are not two builds. The library on the other side of the
    competitive comparison is *not* in here: no figure this script takes runs
    it, so a difference there would refuse a comparison that means something.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        built = version("valgebra")
    except PackageNotFoundError:  # pragma: no cover - provenance refuses first
        built = "absent"
    lock = "gil" if compare_gate.gil_enabled() else "freethreaded"
    release = f"cpython{sys.version_info.major}.{sys.version_info.minor}"
    return f"{release}-{lock}-valgebra-{built}"


def take_reading(label: str) -> Reading | None:
    """Time every shape on the installed extension, or say why nothing was timed.

    The refusals are the gate's own: an extension that is not a release build
    reads an order of magnitude slow, and a shape whose payload is rejected
    takes the fast reject path and reads as a speed-up. Neither is a figure, so
    neither is reported as one.
    """
    refusal = compare_gate.provenance()
    if refusal is not None:
        print(f"pgo_compare: {refusal}")
        return None
    try:
        built = compare_gate.shapes()
    except ImportError as err:
        print(f"pgo_compare: cannot build the shapes: {err}")
        return None
    if not compare_gate.warm_up(built):
        return None
    from valgebra import _valgebra  # noqa: PLC0415

    return Reading(
        label=label,
        environment=environment(),
        extension=str(getattr(_valgebra, "__file__", "unknown")),
        nanoseconds={
            name: compare_gate.per_call_ns(
                shape["valgebra"], shape["data"], shape["number"]
            )
            for name, shape in built.items()
        },
    )


def report(plain: Reading, profiled: Reading) -> int:
    """Print what the profile buys, per shape, and name what it costs.

    The ratio is the profiled time over the plain one, so under one is a gain
    and over one is what the layout cost. Both are printed: a table that showed
    only the gains would be the single figure this file exists to refuse.
    """
    if plain.environment != profiled.environment:
        print(
            "pgo_compare: the two readings are from different environments "
            f"({plain.environment} and {profiled.environment}); a ratio between "
            "them is a statement about the difference between them"
        )
        return EXIT_CANNOT_RUN
    shared = sorted(plain.nanoseconds.keys() & profiled.nanoseconds.keys())
    if not shared:
        print("pgo_compare: the two readings share no shape")
        return EXIT_CANNOT_RUN

    if any(plain.nanoseconds[name] <= 0 for name in shared):
        print("pgo_compare: a reading of zero is a timer that did not run")
        return EXIT_CANNOT_RUN

    print(f"environment: {plain.environment}")
    print(f"{plain.label}: {plain.extension}")
    print(f"{profiled.label}: {profiled.extension}")
    spread = bands()
    print(f"\n{'shape':<16}{'plain ns':>12}{'profiled ns':>14}{'ratio':>9}{'band':>8}")
    ratios: dict[str, float] = {}
    for name in shared:
        before, after = plain.nanoseconds[name], profiled.nanoseconds[name]
        ratios[name] = after / before
        band = spread.get(name, DEFAULT_BAND)
        print(
            f"{name:<16}{before:>12.1f}{after:>14.1f}"
            f"{ratios[name]:>9.3f}{band * 100:>7.1f}%"
        )

    def moved(direction: Callable[[float, float], bool]) -> str:
        named = [
            f"{name} ({(ratios[name] - 1) * 100:+.0f}%)"
            for name in shared
            if direction(ratios[name], spread.get(name, DEFAULT_BAND))
        ]
        return ", ".join(named) if named else "no shape past its own spread"

    print(f"\nthe profile buys:  {moved(lambda ratio, band: ratio < 1 - band)}")
    print(f"the profile costs: {moved(lambda ratio, band: ratio > 1 + band)}")
    print(
        "\nA shape inside its band moved by less than this machine does between "
        "runs of one build. What the release matrix does with the ones outside "
        "it is a decision about the shapes the project serves, and "
        "docs/11-performance.md carries that decision with its reason."
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument(
        "--record",
        type=Path,
        metavar="PATH",
        help="time the installed extension and write the reading here",
    )
    parser.add_argument(
        "--label",
        default="build",
        help="what this build is, as the table names it (plain, pgo)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("PLAIN", "PROFILED"),
        help="compare two readings, per shape",
    )
    args = parser.parse_args(argv)

    if args.compare:
        try:
            plain, profiled = (Reading.read(path) for path in args.compare)
        except (OSError, ValueError, KeyError, TypeError) as err:
            print(f"pgo_compare: cannot read a reading: {err}")
            return EXIT_CANNOT_RUN
        return report(plain, profiled)

    if args.record is None:
        parser.print_usage()
        return EXIT_CANNOT_RUN
    reading = take_reading(args.label)
    if reading is None:
        return EXIT_CANNOT_RUN
    reading.write(args.record)
    print(f"pgo_compare: {args.label} written to {args.record}")
    for name, nanoseconds in sorted(reading.nanoseconds.items()):
        print(f"  {name:<16}{nanoseconds:>12.1f} ns")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
