---
description: Install the package and its build toolchain.
---

# Installation

valgebra publishes prebuilt wheels to PyPI, so the common path needs no Rust
toolchain. Building from source is the fallback for development or an
unsupported platform; it is a Rust extension built with
[maturin](https://www.maturin.rs/), which requires a Rust toolchain.

## From PyPI

With **Python** 3.10 or newer:

```bash
pip install valgebra
# or
uv add valgebra
```

valgebra has no runtime dependencies. The examples in these pages import three
packages it does not:

- [annotated-types](https://pypi.org/project/annotated-types/), whose markers
  most refinement examples are written with. valgebra reads the markers
  structurally, never by importing that package.
- [typing-extensions](https://pypi.org/project/typing-extensions/), for the
  forms an example spells from it where an older Python's `typing` lacks them.
- [pytest](https://pypi.org/project/pytest/), which the examples showing a
  refusal use.

Install them to run the examples as printed:

```bash
pip install annotated-types typing-extensions pytest
```

Wheels are published for Linux (manylinux, x86_64 and aarch64), macOS (Intel and
Apple silicon) and Windows x64 for every supported CPython, 3.10 through 3.15,
Windows arm64 from 3.12, free-threaded CPython 3.14 and 3.15 on Linux, macOS
and Windows x64, and [PyPy 3.11](#pypy) on Linux. musllinux (x86_64 and
aarch64) gets a wheel for each interpreter its build image carries: in 0.0.17,
CPython 3.10 through 3.14, 3.14t and PyPy 7.3, the set `UNNAMED` in
`tests/test_release_matrix.py` records for each musllinux row. Alpine's
CPython 3.15 installs from the source distribution. Free-threaded support
starts at 3.14t; the earlier 3.13 free-threaded build is not a target
([below](#free-threaded-cpython)).

## From source

Building from source additionally requires:

- A stable **Rust** toolchain (edition 2024, MSRV 1.88) via
  [rustup](https://rustup.rs/).
- [**uv**](https://docs.astral.sh/uv/) (recommended) for the environment and the
  build.

```bash
git clone https://github.com/ppigazzini/valgebra && cd valgebra
uv sync                 # create .venv and install the dev dependencies
uv run --no-sync maturin develop --uv  # build the Rust extension into the venv
```

## Verify it works

```python
import valgebra
from valgebra import Validator

print(valgebra.__version__)
assert Validator(int).is_valid(7)
```

## PyPy

PyPy 3.11 is a target on Linux. Wheels are published for PyPy 7.3 and
for PyPy 8.0 on manylinux x86_64 and aarch64 — two, because PyPy 8.0 changed
the ABI tag and an installer matches it exactly, so a wheel for one does not
install on the other — and for the PyPy the musl build image carries on
musllinux. Every push builds the extension against PyPy, imports it, and runs
the whole suite there. Both halves
are needed, because the C API PyPy offers is not CPython's: an extension there
runs through `cpyext`, which carries the limited API and not every static type
object CPython exports, so naming one of those links fine and fails at `import`
— and `cpyext` also *answers* differently, which no import can show. The PyPy
wheels are plain release builds, where the CPython wheels are profile-guided:
a profiled extension runs out of the native stack budget `cpyext` sizes from
the recursion limit before the walk reaches its own depth bound, and the
process dies where the plain build reports the bound. The release runs the
product suite on every wheel set it ships but the musllinux ones, on each set's
floor, newest and free-threaded interpreters -- the `smoke` matrix in
`.github/workflows/release.yml` names them -- and each PyPy wheel on the PyPy it
is built for, not only on the one the push lane builds; the musllinux wheels
on the musl CPython of each release, in Alpine containers, but for the
free-threaded and PyPy ones no Alpine image carries
(`docs/dev/09-releasing.md` in the repository). There is
no PyPy wheel for macOS or Windows, where the source distribution is the
install.

One promise this page makes holds differently there. A validator releases the
classes, enums and predicates its schema names when nothing else holds them, and
on PyPy it cannot: `cpyext` builds a `PyTypeObject` proxy for every class an
extension is shown and never frees it, so a class is kept alive from the first
validator that reads it, whatever that validator does afterwards. A long-running
process that compiles a schema over a *throwaway* class per request grows there
and does not on CPython. Nothing in valgebra can change this, and the suite's
three lifetime cases say so on PyPy rather than claiming to pass.

And one call CPython refuses, PyPy answers. `object.__new__(Validator)` makes a
validator whose schema was never compiled, which on PyPy 8.0 admits every
value. CPython's `object.__new__` refuses a class with a constructor of its
own; PyPy's does not ask, and no code of valgebra's runs to refuse it. Build a
validator by calling `Validator`.

## Free-threaded CPython

On a free-threaded interpreter a validator is immutable and shares no mutable
walk state, so object validation runs in parallel with the interpreter lock
disabled. The JSON path is the one exception: its string parser draws on a
process-wide interned-string cache guarded by a lock, so concurrent
`validate_json`/`load` calls serialize briefly on that cache even though the walk
itself does not.

Three things read differently there, each measured on 3.14t:

- **Threads checking one shared value contend for it.** The walk takes each
  container's own lock to read an element, so eight threads validating the same
  `dict[str, list[int]]` ran at about a third of one thread's throughput, where
  eight threads each holding their own copy ran at three and a half times it.
  Give each thread its own value, or check a shared value once.
- **Forking a process whose threads are inside a walk can hang the child.** A
  child forked while other threads walked a shared container hung on that
  container's lock in 16 to 26 of 60 forks; the interpreters with a lock never
  did. CPython already warns against forking a threaded process; start worker
  processes with the `spawn` or `forkserver` method.
- **A record's field names are interned for good.** A validator interns the
  names of the fields it reads, which is what makes a field lookup one pointer
  comparison, and the free-threaded build makes an interned string immortal: a
  process that builds a validator per request over field names it has not seen
  before keeps them all, about 190 bytes for each 100-character name. The
  builds with a lock free them with the last validator that held them.
