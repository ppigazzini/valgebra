"""Read what a build costs on each shape, and where the cost sits.

Three readings of the shapes `scripts/perf_shapes.py` holds, taken in
environments this script makes, each holding the standard library and one
valgebra: a release from PyPI, a wheel on disk, or a revision built as the
release builds it.

- `count`: instructions per call under cachegrind. On a build with the GIL,
  two environments named at one length compare to the instruction, so a
  difference is the code's; the performance page says why the length, and how
  a free-threaded build's small shapes read. It sees no cache, branch
  predictor or alignment.
- `cycles`: user-mode cycles, instructions, branch mispredicts and frontend
  stalls per call, from the processor's own counters through
  `perf_event_open`. What the machine did, so it sees what `count` cannot, and
  carries the machine's noise: read it interleaved, against a control taken in
  the same run, and claim nothing from it alone.
- `symbols`: which functions -- and, on builds that carry line tables, which
  source lines -- took the instructions one build spends over another on a
  shape, and with `--per N` the instructions run at least `N` times a call,
  disassembled: an element's or a level's path, side by side. A release wheel
  is stripped, so the extension's functions are named only between builds
  made here.

```bash
uv run --no-sync python scripts/perf_inspect.py env last --python PY --release 0.0.19
uv run --no-sync python scripts/perf_inspect.py env head --python PY --build
uv run --no-sync python scripts/perf_inspect.py count last head
uv run --no-sync python scripts/perf_inspect.py cycles last head --shapes deep records
uv run --no-sync python scripts/perf_inspect.py symbols last head --shape deep --per 25
```

Every reading is against the first environment named. Environments live under
`target/perf-inspect/`, and `--build` unpacks each tree there into a directory
of one length and builds it with the training's hash seed fixed, so two builds
differ in their sources and nothing else. A build keeps its symbols, and
`--lines` adds line tables, which leave the code as it is.

`count` exits 1 where `--tolerance` is given and a shape rose past it, and 2
where a reading cannot be taken: no valgrind, no counter, no environment.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import page_figures
import perf_gate
from _lib import EXIT_CANNOT_RUN, EXIT_FAIL, EXIT_OK, ROOT, arguments

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

WORK = ROOT / "target" / "perf-inspect"
SHAPES_SCRIPT = ROOT / "scripts" / "perf_shapes.py"

#: The two call counts a cachegrind reading is the difference of: the
#: interpreter's start-up and the shape's construction cancel between them.
CALLS = (200, 600)


def _cannot_run(message: str) -> NoReturn:
    print(f"perf_inspect: {message}")
    raise SystemExit(EXIT_CANNOT_RUN)


def env_dir(name: str) -> Path:
    return WORK / "envs" / name


def env_python(name: str) -> Path:
    python = env_dir(name) / "bin" / "python"
    if not python.exists():
        _cannot_run(f"no environment {name!r}; make it with `env {name} --python ...`")
    return python


def worker(name: str) -> list[str]:
    """Return the command that runs `perf_shapes.py` in environment `name`.

    `-s` keeps the user's site directory out. The script's own directory goes
    first on the path, as for every script, which is how it finds `_lib`; no
    script there is named for a standard-library module. `-I` would keep both
    out and also ignore `PYTHONHASHSEED`, which every reading fixes.
    """
    return [str(env_python(name)), "-s", str(SHAPES_SCRIPT)]


def tree_of(rev: str | None) -> str:
    """Name the tree `rev` holds, or the checkout as it stands where it is None."""
    if rev is None:
        return perf_gate.snapshot()
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{rev}^{{tree}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        _cannot_run(f"no tree at {rev!r}")
    return resolved.stdout.strip()


def build_wheel(tree: str, python: Path, *, lines: bool) -> Path:
    """Build the profile-guided wheel of `tree` for `python`, and return it.

    The tree is unpacked under a twelve-character name and built into an
    emptied target directory: the layout follows the length of the path a
    build reads its sources from, and a profile is a file the compiler flags
    name by path, so a reused directory would read one build's profile as
    current to the next.
    """
    if shutil.which("maturin") is None:
        _cannot_run("maturin is not on PATH, so no profiled wheel can be built")
    tag = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import sys, sysconfig; print(sys.implementation.cache_tag"
                " + ('t' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    label = f"{tree[:12]}-{tag}{'-lines' if lines else ''}"
    source = WORK / "trees" / tree[:12]
    if not source.exists():
        # Unpacked beside its place and renamed into it, so a build started
        # beside this one never reads a tree half written.
        partial = source.with_name(f"{source.name}.{os.getpid()}")
        partial.mkdir(parents=True)
        perf_gate.unpack(tree, partial)
        try:
            partial.rename(source)
        except OSError:
            shutil.rmtree(partial)
    target, out = WORK / "build" / label, WORK / "wheels" / label
    for path in (target, out):
        shutil.rmtree(path, ignore_errors=True)
    env = {
        **page_figures.build_environment(),
        "CARGO_TARGET_DIR": str(target),
        "PYTHONHASHSEED": "0",
        "CARGO_PROFILE_RELEASE_STRIP": "false",
    }
    if lines:
        env["CARGO_PROFILE_RELEASE_DEBUG"] = "line-tables-only"
    built = subprocess.run(
        [
            "maturin",
            "build",
            "--release",
            "--pgo",
            "-i",
            str(python),
            "--out",
            str(out),
        ],
        cwd=source,
        env=env,
        check=False,
    )
    wheels = sorted(out.glob("*.whl"))
    if built.returncode != 0 or len(wheels) != 1:
        _cannot_run(f"the profiled build of {tree[:12]} for {python} left no wheel")
    return wheels[0]


def make_env(args: Any) -> int:
    """Make environment `name` for `python`, holding the valgebra the flags name."""
    name, python = args.name, Path(args.python)
    where = env_dir(name)
    shutil.rmtree(where, ignore_errors=True)
    where.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(python), "-m", "venv", "--without-pip", str(where)], check=True)
    inner = where / "bin" / "python"
    if args.release:
        if shutil.which("uv") is None:
            _cannot_run("uv is not on PATH, so no release can be installed")
        spec = f"valgebra=={args.release}"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(inner),
                "--no-deps",
                "--only-binary",
                "valgebra",
                "--index-url",
                "https://pypi.org/simple/",
                spec,
            ],
            check=True,
        )
        source = spec
    elif args.wheel:
        perf_gate.install(Path(args.wheel), inner)
        source = args.wheel
    else:
        rev = args.build if isinstance(args.build, str) else None
        tree = tree_of(rev)
        perf_gate.install(build_wheel(tree, python, lines=args.lines), inner)
        tables = " with line tables" if args.lines else ""
        source = f"tree {tree} ({rev or 'the checkout'}){tables}"
    (where / "source.txt").write_text(f"{source}\n{python}\n", encoding="utf-8")
    print(f"perf_inspect: {name} holds {source}")
    return EXIT_OK


def cachegrind(command: Sequence[str], out: str = "/dev/null") -> int:
    """Return the instructions `command` executes under cachegrind."""
    run = subprocess.run(
        [
            "valgrind",
            "--tool=cachegrind",
            "--cache-sim=no",
            f"--cachegrind-out-file={out}",
            *command,
        ],
        capture_output=True,
        text=True,
        env=perf_gate.workload_environment(os.environ),
        check=False,
    )
    found = perf_gate.IREFS.search(run.stderr)
    if run.returncode != 0 or found is None:
        _cannot_run(
            f"cachegrind could not count {' '.join(command[-3:])}: {run.stderr[-400:]}"
        )
    return int(found.group(1).replace(",", ""))


def per_call(low: int, high: int, calls: tuple[int, int] = CALLS) -> float:
    """Return the instructions one call costs, from the counts at two call counts."""
    return (high - low) / (calls[1] - calls[0])


def change(base: float, value: float) -> float:
    """Return `value` against `base`, in per cent."""
    return 100 * (value / base - 1)


def count(names: Sequence[str], shapes: Sequence[str], tolerance: float | None) -> int:
    """Print instructions per call per shape, each environment against the first."""
    if shutil.which("valgrind") is None:
        _cannot_run("valgrind is not on PATH")
    commands = {name: worker(name) for name in names}
    jobs = [(name, shape, n) for name in names for shape in shapes for n in CALLS]
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        counted = dict(
            zip(
                jobs,
                pool.map(
                    lambda j: cachegrind([*commands[j[0]], "run", j[1], str(j[2])]),
                    jobs,
                ),
                strict=True,
            )
        )
    base, rest = names[0], names[1:]
    print(f"{'shape':<16}{base:>14}" + "".join(f"{name:>14}" for name in rest))
    risen = []
    for shape in shapes:
        cost = {
            name: per_call(
                counted[name, shape, CALLS[0]], counted[name, shape, CALLS[1]]
            )
            for name in names
        }
        cells = [change(cost[base], cost[name]) for name in rest]
        risen += [
            f"{shape} {name} {cell:+.2f}%"
            for name, cell in zip(rest, cells, strict=True)
            if tolerance is not None and cell > tolerance
        ]
        print(
            f"{shape:<16}{cost[base]:>14.1f}"
            + "".join(f"{cell:>+13.2f}%" for cell in cells)
        )
    if risen:
        print(f"perf_inspect: past {tolerance}%: {', '.join(risen)}")
        return EXIT_FAIL
    return EXIT_OK


def cycles(
    names: Sequence[str], shapes: Sequence[str], rounds: int, cpu: int | None
) -> int:
    """Print median cycles per call per shape over interleaved rounds.

    Each round starts the order at the next environment, so each takes every
    place in turn and none sits on the warmer half of a run more often than
    another; the cell is the median of the rounds against the first
    environment's, and beside it the rounds in which it read slower.
    """
    pin = (
        ["taskset", "-c", str(cpu)]
        if cpu is not None and shutil.which("taskset")
        else []
    )
    commands = {name: worker(name) for name in names}
    readings: dict[tuple[str, str], list[dict[str, float]]] = collections.defaultdict(
        list
    )
    for round_ in range(rounds):
        turn = round_ % len(names)
        for name in [*names[turn:], *names[:turn]]:
            for shape in shapes:
                run = subprocess.run(
                    [*pin, *commands[name], "counters", shape, "auto", "8"],
                    capture_output=True,
                    text=True,
                    env=perf_gate.workload_environment(os.environ),
                    check=False,
                )
                if run.returncode != 0:
                    _cannot_run(run.stdout.strip() or run.stderr.strip())
                readings[name, shape].append(json.loads(run.stdout))
    base, rest = names[0], names[1:]
    print(f"{'shape':<16}{base:>14}" + "".join(f"{name:>20}" for name in rest))
    for shape in shapes:
        first = [r["cycles"] for r in readings[base, shape]]
        cells = []
        for name in rest:
            these = [r["cycles"] for r in readings[name, shape]]
            slower = sum(1 for a, b in zip(first, these, strict=True) if b > a)
            cells.append(
                f"{change(statistics.median(first), statistics.median(these)):>+12.1f}%"
                f" {slower:>2}/{rounds:<3}"
            )
        print(f"{shape:<16}{statistics.median(first):>14.0f}" + "".join(cells))
    return EXIT_OK


_CRATE_PATH = re.compile(r".*?/(crates/|\.cargo/registry/src/[^/]+/)")


def cachegrind_costs(
    text: str, *, by_line: bool
) -> collections.Counter[tuple[str, str]]:
    """Read a cachegrind output file into instructions per function or per line.

    A file's path is cut to the part from its crate down, so two trees
    unpacked under different names name one source file alike.
    """
    costs: collections.Counter[tuple[str, str]] = collections.Counter()
    file = function = ""
    for line in text.splitlines():
        if line.startswith("fl="):
            file = _CRATE_PATH.sub(r"\1", line[3:].strip())
        elif line.startswith("fn="):
            function = re.sub(r"::h[0-9a-f]{16}$", "", line[3:].strip())
        elif line[:1].isdigit():
            where, *events = line.split()
            if events:
                costs[(file, where) if by_line else (file, function)] += int(events[0])
    return costs


def callgrind_instructions(text: str, binary: str) -> collections.Counter[int]:
    """Read a callgrind `--dump-instr=yes` file into instructions per address.

    Callgrind compresses its output: `ob=(n) name` names an object once and
    `ob=(n)` reuses it, a position is absolute (`0x...`), relative (`+n`,
    `-n`) or the last one (`*`), and the cost line after `calls=` is the
    inclusive cost of a call, which belongs to the callee. Only the object
    whose path holds `binary` is kept.
    """
    costs: collections.Counter[int] = collections.Counter()
    names: dict[str, str] = {}
    current, address, skip = "", 0, False
    for line in text.splitlines():
        named = re.match(r"(c?ob)=\((\d+)\)\s*(.*)", line)
        if named:
            kind, number, path = named.groups()
            if path:
                names[number] = path
            if kind == "ob":
                current = names.get(number, "")
            continue
        if line.startswith("calls="):
            skip = True
            continue
        if not line or line[0] not in "0123456789+-*" or "=" in line.split()[0]:
            continue
        position, *rest = line.split()
        address = _position(position, address)
        if skip:
            skip = False
        elif binary in current and len(rest) > 1:
            costs[address] += int(rest[1])
    return costs


def _position(field: str, last: int) -> int:
    """Return the address a callgrind position field names, given the last one."""
    if field.startswith("0x"):
        return int(field, 16)
    if field[0] in "+-":
        step = int(field[1:], 16) if field[1:].startswith("0x") else int(field[1:])
        return last + step if field[0] == "+" else last - step
    return last


def _extension(name: str) -> Path:
    found = sorted(env_dir(name).glob("lib/*/site-packages/valgebra/_valgebra*.so"))
    if not found:
        _cannot_run(f"environment {name!r} holds no extension")
    return found[0]


def _disassembly(binary: Path, addresses: Iterable[int]) -> dict[int, str]:
    wanted = sorted(addresses)
    text = subprocess.run(
        [
            "objdump",
            "-d",
            "--no-show-raw-insn",
            "-C",
            f"--start-address={wanted[0]:#x}",
            f"--stop-address={wanted[-1] + 16:#x}",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = {}
    for line in text.splitlines():
        match = re.match(r"\s*([0-9a-f]+):\s+(.*)", line)
        if match:
            found[int(match[1], 16)] = re.sub(r"\s+", " ", match[2].strip())
    return found


def _source_line(binary: Path, address: int) -> str:
    frames = subprocess.run(
        ["addr2line", "-i", "-e", str(binary), hex(address)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    for frame in frames:
        if "/crates/" in frame:
            return frame.split("/crates/", 1)[1].split(" ")[0]
    return "?"


#: The share of a call's named count an instruction must run to be on the
#: path: below one, since the last element or level can leave by another branch.
HOT = 0.95


def symbols(names: Sequence[str], shape: str, *, lines: bool, per: int | None) -> int:
    """Print where the second environment spends instructions the first does not."""
    if shutil.which("valgrind") is None:
        _cannot_run("valgrind is not on PATH")
    (WORK / "profiles").mkdir(parents=True, exist_ok=True)
    if per is None:
        _moved(names, shape, lines=lines)
    else:
        for name in names:
            _hot_path(name, shape, lines=lines, per=per)
    return EXIT_OK


def _moved(names: Sequence[str], shape: str, *, lines: bool) -> None:
    """Print the functions or lines whose instructions per call moved most."""
    moved: list[dict[tuple[str, str], float]] = []
    for name in names:
        files = [WORK / "profiles" / f"{name}-{shape}-{n}.cg" for n in CALLS]
        for n, path in zip(CALLS, files, strict=True):
            cachegrind([*worker(name), "run", shape, str(n)], str(path))
        low, high = (
            cachegrind_costs(f.read_text(encoding="utf-8"), by_line=lines)
            for f in files
        )
        moved.append({k: per_call(low[k], high[k]) for k in low.keys() | high.keys()})
    first, second = moved
    keys = sorted(
        first.keys() | second.keys(),
        key=lambda k: -abs(second.get(k, 0) - first.get(k, 0)),
    )
    print(f"total {sum(first.values()):.0f} -> {sum(second.values()):.0f} a call")
    for key in keys[:30]:
        a, b = first.get(key, 0.0), second.get(key, 0.0)
        if abs(b - a) < 1:
            break
        where = f"{key[0]}:{key[1]}" if lines else f"{key[1][:90]}  [{key[0][-40:]}]"
        print(f"{a:>10.1f} {b:>10.1f} {b - a:>+9.1f}  {where}")


def _hot_path(name: str, shape: str, *, lines: bool, per: int) -> None:
    """Print the extension's instructions run `per` times a call, disassembled."""
    binary = _extension(name)
    counts = []
    for n in CALLS:
        path = WORK / "profiles" / f"{name}-{shape}-{n}.out"
        subprocess.run(
            [
                *("valgrind", "--tool=callgrind", "--dump-instr=yes"),
                f"--callgrind-out-file={path}",
                *worker(name),
                *("run", shape, str(n)),
            ],
            capture_output=True,
            env=perf_gate.workload_environment(os.environ),
            check=True,
        )
        text = path.read_text(encoding="utf-8")
        counts.append(callgrind_instructions(text, "_valgebra"))
    low, high = counts
    each = {a: per_call(low[a], high[a]) / per for a in low.keys() | high.keys()}
    hot = sorted(a for a, v in each.items() if v >= HOT)
    if not hot:
        print(f"== {name}: no instruction of the extension runs {per} times a call")
        return
    listing = _disassembly(binary, hot)
    total = sum(each[a] for a in hot)
    print(f"== {name}: {total:.1f} instructions at each of {per} a call")
    for address in hot:
        site = f"  {_source_line(binary, address)}" if lines else ""
        text = listing.get(address, "?")
        print(f"{each[address]:>6.2f} {address:#09x} {text:<56}{site}")


def _arguments(argv: Sequence[str]) -> Any:
    parser = arguments(__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    env = steps.add_parser("env", help="make an environment holding one valgebra")
    env.add_argument("name")
    env.add_argument("--python", required=True, help="the interpreter it runs on")
    holds = env.add_mutually_exclusive_group(required=True)
    holds.add_argument("--release", help="a version on PyPI, as 0.0.19")
    holds.add_argument("--wheel", help="a wheel file")
    holds.add_argument(
        "--build", nargs="?", const=True, help="a revision, or the checkout"
    )
    env.add_argument("--lines", action="store_true", help="a --build with line tables")
    every = shape_names()
    reading = steps.add_parser("count", help="instructions per call, under cachegrind")
    reading.add_argument("names", nargs="+")
    reading.add_argument("--shapes", nargs="+", default=every, help="default: all")
    reading.add_argument(
        "--tolerance", type=float, help="exit 1 past this rise, in per cent"
    )
    timing = steps.add_parser("cycles", help="hardware counters per call, interleaved")
    timing.add_argument("names", nargs="+")
    timing.add_argument("--shapes", nargs="+", default=every, help="default: all")
    timing.add_argument("--rounds", type=int, default=8)
    timing.add_argument("--cpu", type=int, default=2, help="the CPU to pin to")
    where = steps.add_parser("symbols", help="the functions or lines a shape moved")
    where.add_argument("names", nargs=2)
    where.add_argument("--shape", required=True)
    where.add_argument("--lines", action="store_true", help="by source line")
    where.add_argument(
        "--per", type=int, help="the hot path run this many times a call"
    )
    return parser.parse_args(argv)


def shape_names() -> list[str]:
    """Read the shape table's names without importing valgebra here."""
    text = SHAPES_SCRIPT.read_text(encoding="utf-8")
    body = text.split("def _shapes()", 1)[1].split("\n\n\n", 1)[0]
    return re.findall(r'^        "(\w+)": ', body, flags=re.MULTILINE)


def main(argv: Sequence[str] = ()) -> int:
    args = _arguments(argv or sys.argv[1:])
    match args.step:
        case "env":
            return make_env(args)
        case "count":
            return count(args.names, args.shapes, args.tolerance)
        case "cycles":
            return cycles(args.names, args.shapes, args.rounds, args.cpu)
        case _:
            return symbols(args.names, args.shape, lines=args.lines, per=args.per)


if __name__ == "__main__":
    raise SystemExit(main())
