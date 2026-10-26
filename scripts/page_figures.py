"""Re-read every wall-clock figure the performance and JSON pages state.

`docs/11-performance.md` and `docs/07-json.md` state figures measured on one
machine over the repeats the performance page's Method section names. This
takes each of them again, on the checkout as it stands or on a commit, and
prints it under the name of the page section that states it. It edits no page:
a figure goes into its page by hand, once read, because a run something
disturbed reads slow and only a reader can tell that from a figure that moved.

A run is three stages, and `--stage` runs one alone:

- **build** unpacks the tree into one directory, makes an environment of the
  bench group for each interpreter, and builds each a profile-guided wheel from
  that directory, the training's hash seed fixed. CPython 3.14 also gets a
  plain wheel and a profile-guided one with PyO3's reference pool compiled in,
  for the two pairs the page states. Every build reads the same source path,
  since a build path moves the profiled layout.
- **measure** takes every reading `MEASURE` names, one at a time, the plain,
  profiled and pool wheels installed in turn for the pairs. Nothing else may
  run while it does.
- **report** prints every figure from the readings.

    uv run --no-sync python scripts/page_figures.py --python-314 /path/to/python

The work directory keeps the environments, the wheels and the readings, so a
report can be printed again from one set of readings.

It answers in two of the gates' three codes: 0 for a report and 2 for none --
an interpreter, a tool or a build missing, a reading that failed, or a run cut
short. The middle one says "this ran and the answer is no", and no answer here
has that shape.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import timeit
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import compare_gate
import perf_gate
from _lib import EXIT_CANNOT_RUN, EXIT_OK, ROOT, arguments

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Where a run builds and reads, unless `--work` names another place.
WORK = ROOT / "target" / "page-figures"

#: The interpreters the pages are measured on, by the label each file carries.
#: CPython 3.14 is the one the performance page's baseline names; the other two
#: are the builds `uv` installs, as the page's interpreter table says.
LABELS = ("3.14", "3.12", "3.14t")

#: How many readings each figure is taken over, as the performance page's
#: Method section states them. The comparison gate on CPython 3.14, the
#: benchmarks and the floor are each read in two sets, apart in the run, so a
#: run whose second half reads apart from its first shows it.
MEASURE = {
    "gate-3.14": 9,
    "gate-3.12": 12,
    "gate-3.14t": 7,
    "benchmarks": 5,
    "floor": 3,
    "techniques": 2,
    "pairs": 6,
}

#: The readings each label's comparison gate holds once a run is whole.
GATE_RUNS = {
    "3.14": 2 * MEASURE["gate-3.14"],
    "3.12": MEASURE["gate-3.12"],
    "3.14t": MEASURE["gate-3.14t"],
}

#: The criterion benchmarks the core micro-benchmark table names.
CRITERION = ("reindexed_record_width64", "with_records_open_depth32")

#: The shapes of the end-to-end table and of the JSON page's table, as
#: `benches/` groups them, and the comparison shapes the interpreter table
#: reads, each in its page's row order.
COMPARED = ("large_array", "wide_record", "deep_nesting")
JSON_SHAPES = ("record", "nested", "array")
INTERPRETER_ROWS = ("large_array", "wide_record", "deep_nesting", "scalar")

#: The two pages, as a report names them above the figures each carries.
PERFORMANCE = "docs/11-performance.md"
JSON_PAGE = "docs/07-json.md"

#: A crate reading the JSON document with jiter alone: the tree its walk reads,
#: and the pull parser building nothing. Written into the work directory with
#: the jiter the tree's lock file pins, and built with the release profile's
#: link-time settings and no profile, as the JSON page's table says.
PROBE_MAIN = """\
use std::hint::black_box;
use std::time::Instant;

use jiter::{Jiter, JsonValue};

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn skip_all(bytes: &[u8]) {
    let mut jiter = Jiter::new(bytes);
    jiter.next_skip().unwrap();
    jiter.finish().unwrap();
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(path).unwrap();
    let rounds = 2000;
    let (mut tree, mut pull) = (Vec::new(), Vec::new());
    for _ in 0..15 {
        let t = Instant::now();
        for _ in 0..rounds {
            black_box(JsonValue::parse(black_box(&bytes), false).unwrap());
        }
        tree.push(t.elapsed().as_secs_f64() / f64::from(rounds) * 1e6);
        let t = Instant::now();
        for _ in 0..rounds {
            skip_all(black_box(&bytes));
        }
        pull.push(t.elapsed().as_secs_f64() / f64::from(rounds) * 1e6);
    }
    let (tree, pull) = (median(tree), median(pull));
    println!("{{\\"tree_us\\": {tree:.2}, \\"pull_us\\": {pull:.2}}}");
}
"""


def _cannot_run(message: str) -> NoReturn:
    print(f"page_figures: no report: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CANNOT_RUN)


# --- the readings, each taken by a process of the environment it reads ---


def read_gate() -> dict[str, Any]:
    """Time every comparison shape on both sides, as the comparison gate does."""
    wrong = compare_gate.provenance()
    if wrong:
        _cannot_run(wrong)
    shapes = compare_gate.shapes()
    if not compare_gate.warm_up(shapes):
        _cannot_run("a comparison shape did not answer as the gate expects")
    out: dict[str, Any] = {"python": sys.version.split()[0]}
    for name, shape in shapes.items():
        data, number = shape["data"], shape["number"]
        ours = compare_gate.per_call_ns(shape["valgebra"], data, number)
        theirs = compare_gate.per_call_ns(shape["pydantic"], data, number)
        out[name] = {"vg": ours, "pyd": theirs, "ratio": ours / theirs}
    return out


def _best_ns(statement: str, names: dict[str, object]) -> float:
    """Time a statement as the floor is stated: the best of fifteen runs of 1e6."""
    runs = timeit.repeat(statement, globals=names, repeat=15, number=1_000_000)
    return min(runs) / 1_000_000 * 1e9


def read_floor() -> dict[str, float]:
    """Time the cheapest door and what the page sets beside it."""
    # The bench environment's wheel, which the orchestrating process never loads.
    from valgebra import Validator, anything  # noqa: PLC0415

    def empty(_: object) -> None:
        return None

    names = {"v": Validator(int), "top": Validator(anything), "f": empty}
    return {
        "in": _best_ns("1 in v", names),
        "is_valid": _best_ns("v.is_valid(1)", names),
        "anything": _best_ns("top.is_valid(1)", names),
        "isinstance": _best_ns("isinstance(1, int)", names),
        "empty_call": _best_ns("f(1)", names),
    }


def read_json_split(document: Path) -> dict[str, float]:
    """Time the gate's JSON document checked and parsed with no walk, in one process.

    The parse moves with the heap a process starts from, so both figures are
    read in one process, alternately. The document is written out for the probe.
    """
    from valgebra import Validator, anything  # noqa: PLC0415

    given = compare_gate.corpus()
    text = given.json_document
    checked = Validator(given.json_records)
    parsed = Validator(anything)
    if not (checked.is_valid_json(text) and parsed.is_valid_json(text)):
        _cannot_run("the comparison gate's JSON document is refused")
    document.write_text(text, encoding="utf-8")

    def median_us(call: Callable[[], object]) -> float:
        bests = (min(timeit.repeat(call, repeat=7, number=200)) for _ in range(5))
        return statistics.median(best / 200 * 1e6 for best in bests)

    rounds = [
        (
            median_us(lambda: checked.is_valid_json(text)),
            median_us(lambda: parsed.is_valid_json(text)),
        )
        for _ in range(3)
    ]
    return {
        "checked_us": statistics.median(r[0] for r in rounds),
        "parse_only_us": statistics.median(r[1] for r in rounds),
    }


def read_techniques() -> dict[str, float]:
    """Time the costs the techniques section states of this tree, not of a pair."""
    from typing import Annotated  # noqa: PLC0415

    from annotated_types import Ge  # noqa: PLC0415

    from valgebra import ValidationError, Validator, recursive, union  # noqa: PLC0415

    fields = {f"f{i}": Annotated[int, Ge(0)] for i in range(50)}
    build = min(timeit.repeat(lambda: Validator(fields), repeat=15, number=200)) / 200
    # The value the adversarial-bounds suite builds: a refusal at the bottom of
    # twenty thousand levels, each of which the report summarizes.
    schema = Validator(recursive(lambda node: union(int, [node])))
    deep: object = "not an int"
    for _ in range(20_000):
        deep = [deep]

    def report() -> None:
        try:
            schema.validate(deep)
        except ValidationError:
            return
        _cannot_run("the deep value was admitted")

    reported = min(timeit.repeat(report, repeat=7, number=10)) / 10
    return {"build_50_refined_us": build * 1e6, "deep_report_us": reported * 1e6}


def take_reading(kind: str, document: Path | None) -> int:
    """Print one reading as a JSON line; the process is the environment's."""
    if kind == "gate":
        found: dict[str, Any] = read_gate()
    elif kind == "floor":
        found = read_floor()
    elif kind == "techniques":
        found = read_techniques()
    elif document is None:
        _cannot_run("the JSON split needs --document")
    else:
        found = read_json_split(document)
    print(json.dumps(found))
    return EXIT_OK


# --- the build ---


def _run(argv: Sequence[str], cwd: Path, log: Path, env: dict[str, str]) -> None:
    """Run a build step into a log of its own, and refuse the run if it fails."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as sink:
        done = subprocess.run(
            argv, cwd=cwd, env=env, stdout=sink, stderr=subprocess.STDOUT, check=False
        )
    if done.returncode != 0:
        _cannot_run(f"`{' '.join(argv[:3])}` failed; {log} holds its output")


def build_environment() -> dict[str, str]:
    """Give the environment a build runs in: the caller's, less its compiler flags.

    A `RUSTFLAGS` the caller set would replace the flags `.cargo/config.toml`
    gives every build, which compile PyO3's reference pool out, and the wheel
    would not be the one the release ships.
    """
    env = dict(os.environ)
    for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "VIRTUAL_ENV"):
        env.pop(name, None)
    return env


def python_of(work: Path, label: str) -> Path:
    return work / f"env-{label}" / "bin" / "python"


def build_wheel(work: Path, label: str, kind: str) -> None:
    """Build one wheel of the unpacked tree into `wheel-<kind>-<label>`.

    `kind` is `pgo`, `plain`, or `pool`: the profile-guided build with PyO3's
    reference pool compiled back in, by an empty `RUSTFLAGS`, which replaces the
    flags `.cargo/config.toml` sets. Each kind's target directory is emptied
    first, as `perf_gate.build_profiled` empties its own: a profile is a file
    the compiler flags name by path, so a dependency an earlier run optimized by
    its profile would read as current to this one.
    """
    env_dir = work / f"env-{label}"
    out = work / f"wheel-{kind}-{label}"
    target = work / f"target-{kind}-{label}"
    for path in (out, target):
        shutil.rmtree(path, ignore_errors=True)
    env = build_environment() | {
        "VIRTUAL_ENV": str(env_dir),
        "PATH": f"{env_dir / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "CARGO_TARGET_DIR": str(target),
        "PYTHONHASHSEED": "0",
    }
    if kind == "pool":
        env["RUSTFLAGS"] = ""
    argv = [str(env_dir / "bin" / "maturin"), "build", "--release"]
    argv += ["--pgo"] if kind in {"pgo", "pool"} else []
    argv += ["--out", str(out), "--interpreter", str(python_of(work, label))]
    _run(argv, work / "tree", work / "logs" / f"{out.name}.log", env)
    if len(list(out.glob("*.whl"))) != 1:
        _cannot_run(f"the {kind} build for {label} left no single wheel in {out}")


def jiter_version(lock: Path) -> str:
    """Name the jiter the lock file pins, which the probe builds against."""
    found = re.search(r'name = "jiter"\nversion = "([^"]+)"', lock.read_text("utf-8"))
    if found is None:
        _cannot_run(f"{lock} pins no jiter")
    return found.group(1)


def probe_manifest(version: str) -> str:
    """Write the probe's manifest: one exact jiter, the release profile's linking."""
    return (
        '[package]\nname = "jiter-probe"\nversion = "0.0.0"\nedition = "2024"\n'
        "publish = false\n\n[dependencies]\n"
        f'jiter = {{ version = "={version}", default-features = false }}\n\n'
        '[profile.release]\nlto = "fat"\ncodegen-units = 1\n\n[workspace]\n'
    )


def _step(text: str) -> None:
    print(f"{time.strftime('%T')} {text}", flush=True)


def build(work: Path, tree: str, pythons: dict[str, Path]) -> None:
    """Unpack `tree` and build every environment, wheel and tool the readings use."""
    source = work / "tree"
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir(parents=True)
    perf_gate.unpack(tree, source)
    if not (source / "scripts" / "page_figures.py").is_file():
        _cannot_run(f"{tree} carries no scripts/page_figures.py to read with")
    (work / "tree.txt").write_text(f"{tree}\n", encoding="utf-8")
    uv = shutil.which("uv") or _cannot_run("uv is not on PATH")
    for label, python in pythons.items():
        _step(f"environment and profiled wheel, {label}")
        sync = [uv, "sync", "--locked", "--no-install-project", "--group", "bench"]
        env = build_environment() | {
            "UV_PROJECT_ENVIRONMENT": str(work / f"env-{label}")
        }
        log = work / "logs" / f"env-{label}.log"
        _run([*sync, "--python", str(python)], source, log, env)
        build_wheel(work, label, "pgo")
    for kind in ("plain", "pool"):
        _step(f"{kind} wheel, 3.14")
        build_wheel(work, "3.14", kind)
    _step("jiter probe and criterion")
    probe = work / "jiter-probe"
    (probe / "src").mkdir(parents=True, exist_ok=True)
    manifest = probe_manifest(jiter_version(source / "Cargo.lock"))
    (probe / "Cargo.toml").write_text(manifest, encoding="utf-8")
    (probe / "src" / "main.rs").write_text(PROBE_MAIN, encoding="utf-8")
    cargo = build_environment() | {"CARGO_TARGET_DIR": str(work / "target-probe")}
    _run(["cargo", "build", "--release"], probe, work / "logs" / "probe.log", cargo)
    cargo["CARGO_TARGET_DIR"] = str(work / "target-criterion")
    log = work / "logs" / "criterion.log"
    _run(["cargo", "bench", "--no-run", "--bench", "core"], source, log, cargo)


# --- the measurement ---


class Session:
    """The readings of one run, each appended to its file under `results/`."""

    def __init__(self, work: Path) -> None:
        self.work = work
        self.tree = work / "tree"
        self.results = work / "results"
        shutil.rmtree(self.results, ignore_errors=True)
        self.results.mkdir(parents=True)
        shutil.copy(work / "tree.txt", self.results / "tree.txt")

    def install(self, label: str, kind: str) -> None:
        wheel = next((self.work / f"wheel-{kind}-{label}").glob("*.whl"))
        perf_gate.install(wheel, python_of(self.work, label))

    def run(self, label: str, argv: Sequence[str]) -> str:
        """Run `argv` under the interpreter of `label`, in the tree; give its output."""
        done = subprocess.run(
            [str(python_of(self.work, label)), "-s", "-E", *argv],
            cwd=self.tree,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            _cannot_run(f"{' '.join(argv)} on {label}:\n{done.stdout}{done.stderr}")
        return done.stdout

    def reading(self, label: str, kind: str, *extra: str) -> str:
        """Take one reading in the environment of `label`, and give its line."""
        script = str(self.tree / "scripts" / "page_figures.py")
        output = self.run(label, [script, "--reading", kind, *extra])
        return output.strip().splitlines()[-1]

    def keep(self, name: str, label: str, kind: str, times: int) -> None:
        with (self.results / f"{name}.jsonl").open("a", encoding="utf-8") as sink:
            for _ in range(times):
                sink.write(self.reading(label, kind) + "\n")

    def benches(self, which: str) -> None:
        for run in range(1, MEASURE["benchmarks"] + 1):
            for bench in ("compare", "json"):
                out = self.results / f"{bench}-{which}-{run}.json"
                pytest = ["-m", "pytest", "-q", "-p", "no:cacheprovider"]
                self.run(
                    "3.14",
                    [*pytest, f"benches/bench_{bench}.py", f"--benchmark-json={out}"],
                )

    def pairs(self, name: str, kinds: tuple[str, str]) -> None:
        """Read two wheels in turn, each reading after one it discards."""
        for _ in range(MEASURE["pairs"]):
            for kind in kinds:
                self.install("3.14", kind)
                self.reading("3.14", "gate")
                self.keep(f"{name}-{kind}", "3.14", "gate", 1)

    def outside(
        self, argv: Sequence[str], file: str, env: dict[str, str] | None = None
    ) -> None:
        """Run a tool of no environment's, in the tree, and keep its output."""
        done = subprocess.run(
            argv, cwd=self.tree, capture_output=True, text=True, check=False, env=env
        )
        if done.returncode != 0:
            _cannot_run(f"{argv[0]} failed:\n{done.stdout}{done.stderr}")
        (self.results / file).write_text(done.stdout, encoding="utf-8")


def measure(work: Path) -> None:
    """Take every reading, one at a time, each set of a figure apart from its other."""
    run = Session(work)
    for label in LABELS:
        run.install(label, "pgo")
        run.reading(label, "gate")
    _step("comparison gate on 3.14, first set")
    run.keep("gate-3.14", "3.14", "gate", MEASURE["gate-3.14"])
    _step("benchmarks, first set")
    run.benches("first")
    _step("comparison gate on 3.12 and 3.14t")
    run.keep("gate-3.12", "3.12", "gate", MEASURE["gate-3.12"])
    run.keep("gate-3.14t", "3.14t", "gate", MEASURE["gate-3.14t"])
    _step("floor, first set")
    run.keep("floor", "3.14", "floor", MEASURE["floor"])
    _step("the JSON document's parts, jiter alone and criterion")
    document = run.results / "document.json"
    split = run.reading("3.14", "json-split", "--document", str(document))
    (run.results / "json-split.json").write_text(split + "\n", encoding="utf-8")
    probe = work / "target-probe" / "release" / "jiter-probe"
    run.outside([str(probe), str(document)], "jiter.json")
    cargo = build_environment() | {"CARGO_TARGET_DIR": str(work / "target-criterion")}
    criterion = ["cargo", "bench", "-q", "--bench", "core", "--", "|".join(CRITERION)]
    run.outside(criterion, "criterion.txt", cargo)
    _step("technique figures on 3.14 and 3.12")
    for label in ("3.14", "3.12"):
        run.keep(f"techniques-{label}", label, "techniques", MEASURE["techniques"])
    _step("benchmarks, second set")
    run.benches("second")
    _step("comparison gate on 3.14, second set")
    run.keep("gate-3.14", "3.14", "gate", MEASURE["gate-3.14"])
    _step("plain and profiled wheels in turn")
    run.pairs("pgo-pair", ("plain", "pgo"))
    _step("pool and profiled wheels in turn")
    run.pairs("pool-pair", ("pool", "pgo"))
    run.install("3.14", "pgo")
    _step("floor, second set")
    run.keep("floor", "3.14", "floor", MEASURE["floor"])


# --- the report ---


def read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        _cannot_run(f"no {path.name} in {path.parent}: run --stage measure")
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def _section(page: str, title: str) -> list[str]:
    return ["", f"{page}, {title}"]


def bench_medians(results: Path, bench: str) -> dict[tuple[str, str], list[float]]:
    """Give each benchmark cell's per-run medians in microseconds, by group and side."""
    cells: dict[tuple[str, str], list[float]] = {}
    paths = sorted(
        path
        for which in ("first", "second")
        for path in results.glob(f"{bench}-{which}-*.json")
    )
    if len(paths) != 2 * MEASURE["benchmarks"]:
        _cannot_run(f"{len(paths)} bench_{bench}.py runs in {results}")
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8"))["benchmarks"]:
            key = (row["group"], row["params"]["lib"])
            cells.setdefault(key, []).append(row["stats"]["median"] * 1e6)
    return cells


def cell(runs: list[float]) -> str:
    """State a table cell as the page does: the median, and the runs' half-range."""
    half = (max(runs) - min(runs)) / 2
    return f"{statistics.median(runs):.4g} +/- {half:.2g} us"


def gains(
    first: list[dict[str, Any]], second: list[dict[str, Any]], shape: str
) -> list[float]:
    """Give the time the second wheel saves on `shape`, in percent, pair by pair."""
    return [
        (a[shape]["vg"] - b[shape]["vg"]) / a[shape]["vg"] * 100
        for a, b in zip(first, second, strict=True)
    ]


def criterion_medians(text: str) -> dict[str, str]:
    """Read each named benchmark's median estimate from criterion's output.

    A name too long for its column is printed on a line of its own and its
    estimate on the next, so an estimate belongs to the last name read.
    """
    found: dict[str, str] = {}
    name = None
    for line in text.splitlines():
        name = next((n for n in CRITERION if line.strip().startswith(n)), name)
        estimate = re.search(r"time:\s+\[\S+ \S+ (\S+ \S+) \S+ \S+\]", line)
        if estimate and name is not None:
            found[name] = estimate.group(1)
    return found


def gate_readings(results: Path) -> dict[str, list[dict[str, Any]]]:
    """Read every interpreter's comparison-gate readings, refusing a run cut short."""
    gate = {label: read_lines(results / f"gate-{label}.jsonl") for label in LABELS}
    for label, runs in gate.items():
        if len(runs) != GATE_RUNS[label]:
            _cannot_run(f"{len(runs)} gate readings on {label}, of {GATE_RUNS[label]}")
    return gate


def report_profile(results: Path) -> list[str]:
    plain, profiled = (
        read_lines(results / f"pgo-pair-{k}.jsonl") for k in ("plain", "pgo")
    )
    out = _section(
        PERFORMANCE, f"Baseline matrix: what the profile buys, {len(plain)} pairs"
    )
    for shape in (name for name in profiled[0] if name != "python"):
        saved = gains(plain, profiled, shape)
        out.append(
            f"  {shape:14} {statistics.median(saved):5.1f}%"
            f"  from {min(saved):5.1f} to {max(saved):5.1f}"
        )
    return [*out, "  (a positive figure: the profiled wheel takes less time)"]


def report_floor(results: Path) -> list[str]:
    floor = read_lines(results / "floor.jsonl")
    out = _section(PERFORMANCE, f"The cheapest door: median of {len(floor)} readings")
    for name in floor[0]:
        each = " ".join(f"{r[name]:.1f}" for r in floor)
        out.append(
            f"  {name:11} {statistics.median(r[name] for r in floor):5.1f} ns ({each})"
        )
    return out


def report_results(results: Path, gate: dict[str, list[dict[str, Any]]]) -> list[str]:
    compared = bench_medians(results, "compare")
    out = _section(PERFORMANCE, "Results: median +/- half-range of ten runs")
    for shape in COMPARED:
        sides = [compared[shape, s] for s in ("valgebra", "pydantic", "jsonschema")]
        ours = statistics.median(sides[0])
        minimums = 1 / statistics.median(r[shape]["ratio"] for r in gate["3.14"])
        out += [
            f"  {shape:14} " + " | ".join(cell(runs) for runs in sides),
            (
                f"  {'':14} {statistics.median(sides[1]) / ours:.2f}x pydantic,"
                f" {statistics.median(sides[2]) / ours:.0f}x jsonschema,"
                f" the gate's minimums {minimums:.2f}x"
            ),
        ]
    return out


def report_interpreters(gate: dict[str, list[dict[str, Any]]]) -> list[str]:
    out = _section(PERFORMANCE, "One of those margins: each ratio's median")
    out.append(f"  {'':14} " + " ".join(f"{label:>7}" for label in LABELS))
    for shape in INTERPRETER_ROWS:
        ratios = [
            statistics.median(r[shape]["ratio"] for r in gate[label])
            for label in LABELS
        ]
        out.append(f"  {shape:14} " + " ".join(f"{ratio:7.3f}" for ratio in ratios))
    out.append(
        f"  {'runs':14} " + " ".join(f"{len(gate[label]):7}" for label in LABELS)
    )
    return out


def report_races(gate: dict[str, list[dict[str, Any]]]) -> list[str]:
    out = _section(PERFORMANCE, "The closest races, on 3.12: each run's ratio")
    for shape in gate["3.12"][0]:
        if shape == "python":
            continue
        ratios = sorted(r[shape]["ratio"] for r in gate["3.12"])
        each = " ".join(f"{x:.3f}" for x in ratios)
        spread = ratios[-1] - ratios[0]
        out.append(
            f"  {shape:14} median {statistics.median(ratios):.3f}"
            f" spread {spread:.4f} ({each})"
        )
    scalar = [r["scalar"]["vg"] for r in gate["3.14"]]
    margin = 1 / statistics.median(r["scalar"]["ratio"] for r in gate["3.14"])
    out.append(
        f"  the scalar on 3.14: {statistics.median(scalar):.1f} ns median,"
        f" {min(scalar):.1f} to {max(scalar):.1f}, {margin:.2f}x"
    )
    return out


def report_techniques(results: Path) -> list[str]:
    text = (results / "criterion.txt").read_text(encoding="utf-8")
    criterion = criterion_medians(text)
    out = _section(PERFORMANCE, "Core micro-benchmarks, one run")
    out += [f"  {name:28} {criterion.get(name, 'not read')}" for name in CRITERION]
    pooled, unpooled = (
        read_lines(results / f"pool-pair-{k}.jsonl") for k in ("pool", "pgo")
    )
    without = statistics.median(r["scalar"]["vg"] for r in unpooled)
    with_pool = statistics.median(r["scalar"]["vg"] for r in pooled)
    out += _section(
        PERFORMANCE, "Crossing into Rust: the scalar, without the pool and with it"
    )
    out.append(f"  {without:.1f} ns against {with_pool:.1f} ns")
    out += _section(
        PERFORMANCE, "Building a validator, and Reporting a failure: the best reading"
    )
    for label in ("3.14", "3.12"):
        found = read_lines(results / f"techniques-{label}.jsonl")
        build = min(r["build_50_refined_us"] for r in found)
        deep = min(r["deep_report_us"] for r in found)
        out.append(
            f"  {label:5} fifty refined fields compile in {build:.1f} us; the report on"
            f" a value nested twenty thousand deep takes {deep:.0f} us"
        )
    return out


def report_json(results: Path) -> list[str]:
    parsed = bench_medians(results, "json")
    out = _section(JSON_PAGE, "Performance: median of ten runs")
    for shape in JSON_SHAPES:
        walk = statistics.median(parsed[shape, "valgebra-json"])
        loads = statistics.median(parsed[shape, "valgebra-loads"])
        out.append(
            f"  {shape:8} is_valid_json {walk:.4g} us, json.loads + is_valid"
            f" {loads:.4g} us, {loads / walk:.2f}x"
        )
    (split,) = read_lines(results / "json-split.json")
    (probe,) = read_lines(results / "jiter.json")
    walked = split["checked_us"] - split["parse_only_us"]
    tree, pull = probe["tree_us"], probe["pull_us"]
    out += _section(JSON_PAGE, "Performance: where the document's time goes")
    return [
        *out,
        (
            f"  is_valid_json {split['checked_us']:.1f} us; the parse and the call"
            f" {split['parse_only_us']:.1f} us; the walk {walked:.1f} us"
        ),
        f"  jiter's tree {tree:.1f} us; its pull parser {pull:.1f} us",
    ]


def report(results: Path) -> list[str]:
    """State every figure the two pages carry, under the section that carries it."""
    gate = gate_readings(results)
    tree = (results / "tree.txt").read_text(encoding="utf-8").strip()
    return [
        f"the tree read: {tree}",
        *report_profile(results),
        *report_floor(results),
        *report_results(results, gate),
        *report_interpreters(gate),
        *report_races(gate),
        *report_techniques(results),
        *report_json(results),
    ]


# --- the command line ---


def _found(version: str) -> Path | None:
    """Find the `uv`-installed interpreter of `version`, which the page names."""
    uv = shutil.which("uv")
    if uv is None:
        return None
    done = subprocess.run(
        [uv, "python", "find", "--managed-python", version],
        capture_output=True,
        text=True,
        check=False,
    )
    return Path(done.stdout.strip()) if done.returncode == 0 else None


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = arguments(__doc__)
    parser.add_argument(
        "--python-314",
        type=Path,
        help="the CPython 3.14 the performance page's baseline names; "
        "required to build",
    )
    parser.add_argument(
        "--python-312", type=Path, help="CPython 3.12 (default: the one uv installs)"
    )
    parser.add_argument(
        "--python-314t",
        type=Path,
        help="the free-threaded CPython 3.14 (default: the one uv installs)",
    )
    parser.add_argument(
        "--rev", help="the commit to read (default: the checkout as it stands)"
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=WORK,
        help="where the environments, wheels and readings are kept "
        "(default: target/page-figures)",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "build", "measure", "report"),
        default="all",
        help="run one stage alone (default: all three)",
    )
    # What a process of a bench environment is started with to take one reading.
    parser.add_argument(
        "--reading",
        choices=("gate", "floor", "techniques", "json-split"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--document", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _arguments(argv)
    if args.reading is not None:
        return take_reading(args.reading, args.document)
    work = args.work.resolve()
    if args.stage in {"all", "build"}:
        if args.python_314 is None:
            _cannot_run("--python-314 names the interpreter the baseline is read on")
        pythons = {
            "3.14": args.python_314,
            "3.12": args.python_312 or _found("3.12"),
            "3.14t": args.python_314t or _found("3.14t"),
        }
        found = {label: path for label, path in pythons.items() if path is not None}
        if len(found) != len(LABELS):
            _cannot_run(f"no interpreter for {sorted(set(LABELS) - set(found))}")
        tree = perf_gate.resolve_rev(args.rev) if args.rev else perf_gate.snapshot()
        build(work, tree, found)
    if args.stage in {"all", "measure"}:
        if not (work / "tree.txt").is_file():
            _cannot_run(f"nothing is built in {work}: run --stage build")
        measure(work)
    if args.stage in {"all", "report"}:
        print("\n".join(report(work / "results")))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
