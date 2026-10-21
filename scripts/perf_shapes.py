"""The shapes `scripts/perf_inspect.py` reads, run inside the environment it reads.

Each shape is one call a caller makes -- a scalar checked, a list of records, a
document parsed and checked, a validator built -- with the value it is made on.
This file runs in the interpreter being measured, whose environment holds the
standard library and valgebra and nothing else, so it imports nothing more than
`_lib`, which imports the standard library alone: a dependency would be a
second thing the readings differ by. It uses only the API every release since
0.0.18 exports, so one table reads a release and a change alike.

```bash
python -s scripts/perf_shapes.py list
python -s scripts/perf_shapes.py run SHAPE CALLS
python -s scripts/perf_shapes.py counters SHAPE CALLS|auto REPEATS
```

`run` builds the shape and calls it `CALLS` times; under cachegrind, two runs
differ by the calls between them and nothing else. `counters` reads cycles,
instructions, branch mispredicts and frontend stalls around `CALLS` calls
through `perf_event_open`, in user mode alone, and prints the fewest of each
over `REPEATS` readings as JSON. It exits 2 where the kernel gives no counter,
as a virtual machine or a container commonly does not.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import json
import os
import platform
import struct
import sys
import time
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from _lib import EXIT_CANNOT_RUN, EXIT_OK, arguments

import valgebra as vg

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class _User(TypedDict):
    id: int
    name: str
    tags: list[str]


# Fifty fields, built rather than spelled out; a checker reads the functional
# form's fields only from a dict literal.
_Wide = TypedDict("_Wide", {f"f{i}": int for i in range(50)})  # ty: ignore[invalid-argument-type]


@dataclasses.dataclass
class _Point:
    x: int
    y: int
    label: str


@runtime_checkable
class _HasName(Protocol):
    name: str


class _Named:
    def __init__(self) -> None:
        self.name = "n"


class _AtLeast:
    """An order bound, carried as `annotated_types.Ge` carries one.

    It carries that package's module, because the frontend reads a constraint
    off that vocabulary, and this file depends on nothing beyond valgebra.
    """

    __module__ = "annotated_types"

    def __init__(self, ge: int) -> None:
        self.ge = ge


def _refusing(validate: Callable[..., object]) -> Callable[..., None]:
    def call(value: object) -> None:
        with contextlib.suppress(vg.ValidationError):
            validate(value)

    return call


def _deep(levels: int) -> tuple[object, object]:
    schema: object = int
    value: object = 1
    for _ in range(levels):
        schema = list[schema]  # type: ignore[valid-type]
        value = [value]
    return schema, value


def _tree(depth: int) -> object:
    return 1 if depth == 0 else [_tree(depth - 1), 2]


_RECORDS = [{"id": i, "name": f"n{i}", "tags": ["a", "b"]} for i in range(200)]


def _shapes() -> dict[str, Callable[[], tuple[Callable[..., object], object]]]:
    v = vg.Validator
    deep_schema, deep_value = _deep(25)
    recursive = vg.recursive(lambda node: int | list[node])  # ty: ignore[invalid-type-form]
    return {
        "scalar": lambda: (v(int).is_valid, 5),
        "union": lambda: (v(int | str | None).is_valid, "x"),
        "literal": lambda: (v(Literal["a", "b", "c", "d"]).is_valid, "d"),
        "membership": lambda: (v(int).__contains__, 5),
        "list_int": lambda: (v(list[int]).is_valid, list(range(1000))),
        "list_str": lambda: (v(list[str]).is_valid, [str(i) for i in range(1000)]),
        "list_nullable": lambda: (
            v(list[int | None]).is_valid,
            [None if i % 3 else i for i in range(1000)],
        ),
        "list_union": lambda: (
            v(list[int | str]).is_valid,
            [i if i % 2 else str(i) for i in range(1000)],
        ),
        "set_int": lambda: (v(set[int]).is_valid, set(range(1000))),
        "tuple_fixed": lambda: (v(tuple[int, str, float]).is_valid, (1, "a", 2.0)),
        "tuple_var": lambda: (v(tuple[int, ...]).is_valid, tuple(range(1000))),
        "dict_str_int": lambda: (
            v(dict[str, int]).is_valid,
            {str(i): i for i in range(100)},
        ),
        "dict_two_clauses": lambda: (
            v({str: int, int: bool}).is_valid,
            {**{f"k{i}": i for i in range(50)}, **dict.fromkeys(range(50), True)},
        ),
        "record": lambda: (v(_User).is_valid, _RECORDS[0]),
        "records": lambda: (v(list[_User]).is_valid, _RECORDS),
        "records_extra": lambda: (
            v(list[_User]).is_valid,
            [{**r, "extra": 1} for r in _RECORDS],
        ),
        "wide_record": lambda: (v(_Wide).is_valid, {f"f{i}": i for i in range(50)}),
        "dataclass": lambda: (v(_Point).is_valid, _Point(1, 2, "p")),
        "protocol": lambda: (v(_HasName).is_valid, _Named()),
        "refined": lambda: (
            v(list[Annotated[int, _AtLeast(0)]]).is_valid,
            list(range(1000)),
        ),
        "pattern": lambda: (
            v(list[Annotated[str, vg.Regex(r"[0-9a-f]{8}")]]).is_valid,
            ["deadbeef"] * 200,
        ),
        "recursive": lambda: (recursive.is_valid, _tree(12)),
        "deep": lambda: (v(deep_schema).is_valid, deep_value),
        "validate": lambda: (v(_User).validate, _RECORDS[0]),
        "explain_scalar": lambda: (_refusing(v(int).validate), "x"),
        "explain_list": lambda: (_refusing(v(list[int]).validate), [*range(999), "x"]),
        "explain_record": lambda: (
            _refusing(v(_User).validate),
            {"id": "x", "name": 1, "tags": [1]},
        ),
        "json_ints": lambda: (
            v(list[int]).is_valid_json,
            json.dumps(list(range(10_000))),
        ),
        "json_records": lambda: (v(list[_User]).is_valid_json, json.dumps(_RECORDS)),
        "json_union": lambda: (
            v(list[int | str | None]).is_valid_json,
            json.dumps([i if i % 3 else None for i in range(2000)]),
        ),
        "json_deep": lambda: (v(deep_schema).is_valid_json, json.dumps(deep_value)),
        "json_reject": lambda: (
            v(list[int]).is_valid_json,
            json.dumps([*range(999), "x"]),
        ),
        "load_records": lambda: (v(list[_User]).load, json.dumps(_RECORDS)),
        "build_record": lambda: (v, _User),
        "build_union": lambda: (v, int | str | list[int] | dict[str, int] | None),
        "relation": lambda: (v(list[int]).is_subtype_of, v(list[int | str])),
    }


SHAPES = _shapes()

#: `perf_event_open`'s number on the architectures the wheels ship for.
_PERF_EVENT_OPEN = {"x86_64": 298, "aarch64": 241}

#: The hardware events read, as the kernel's generic event numbers.
EVENTS = {"cycles": 0, "instructions": 1, "branch_misses": 5, "frontend_stalls": 7}

_ENABLE, _DISABLE, _RESET = 0x2400, 0x2401, 0x2403


def _counter(libc: ctypes.CDLL, number: int, event: int) -> int:
    """Open one user-mode hardware counter on this process, disabled; return it.

    The attribute block is the kernel's `perf_event_attr` at its 128-byte size:
    the event, then the flags word -- disabled, and the kernel and hypervisor
    excluded, which a `perf_event_paranoid` of 2 requires of an unprivileged
    process.
    """
    attr = bytearray(128)
    flags = 1 | (1 << 5) | (1 << 6)
    struct.pack_into("<IIQQQQQ", attr, 0, 0, len(attr), event, 0, 0, 0, flags)
    block = (ctypes.c_char * len(attr)).from_buffer(attr)
    fd = libc.syscall(number, block, 0, -1, -1, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return fd


def calls_for(check: Callable[..., object], value: object) -> int:
    """Return how many calls of `check` on `value` take about twenty milliseconds.

    Long enough that the reads around a reading are a rounding error beside it,
    short enough that a reading rarely meets an interrupt.
    """
    calls = 1
    while True:
        start = time.perf_counter()
        for _ in range(calls):
            check(value)
        if time.perf_counter() - start >= 0.02:  # noqa: PLR2004
            return calls
        calls *= 2


def counters(shape: str, calls: int | None, repeats: int) -> dict[str, float]:
    """Return the fewest of each event per call over `repeats` readings of `calls`.

    `None` reads as many calls as [`calls_for`] finds. The shape is called that
    many times first, so a reading starts warm. The fewest, not the median: a
    reading is disturbed upward only, by whatever the machine did beside it.
    """
    number = _PERF_EVENT_OPEN.get(platform.machine())
    if number is None:
        print(f"perf_shapes: no perf_event_open number for {platform.machine()}")
        raise SystemExit(EXIT_CANNOT_RUN)
    libc = ctypes.CDLL(None, use_errno=True)
    check, value = SHAPES[shape]()
    calls = calls or calls_for(check, value)
    for _ in range(calls):
        check(value)
    try:
        fds = {name: _counter(libc, number, event) for name, event in EVENTS.items()}
    except OSError as error:
        print(f"perf_shapes: the kernel gives no hardware counter here ({error})")
        raise SystemExit(EXIT_CANNOT_RUN) from error
    fewest = dict.fromkeys(EVENTS, float("inf"))
    for _ in range(repeats):
        for request in (_RESET, _ENABLE):
            for fd in fds.values():
                libc.ioctl(fd, request, 0)
        for _ in range(calls):
            check(value)
        for fd in fds.values():
            libc.ioctl(fd, _DISABLE, 0)
        for name, fd in fds.items():
            (count,) = struct.unpack("<Q", os.read(fd, 8))
            fewest[name] = min(fewest[name], count / calls)
    return fewest


def main(argv: Sequence[str] = ()) -> int:
    """Run the reading the command line names."""
    parser = arguments(__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("list", help="print the shapes' names")
    run = steps.add_parser("run", help="build a shape and call it CALLS times")
    run.add_argument("shape", choices=SHAPES)
    run.add_argument("calls", type=int)
    read = steps.add_parser("counters", help="read the hardware counters per call")
    read.add_argument("shape", choices=SHAPES)
    read.add_argument("calls", help="a count, or `auto` for twenty milliseconds")
    read.add_argument("repeats", type=int)
    args = parser.parse_args(argv or sys.argv[1:])
    if args.step == "list":
        print(" ".join(SHAPES))
    elif args.step == "run":
        check, value = SHAPES[args.shape]()
        for _ in range(args.calls):
            check(value)
    else:
        calls = None if args.calls == "auto" else int(args.calls)
        print(json.dumps(counters(args.shape, calls, args.repeats)))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
