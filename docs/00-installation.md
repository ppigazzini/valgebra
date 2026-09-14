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

valgebra has no runtime dependencies. It reads
[annotated-types](https://pypi.org/project/annotated-types/) markers
structurally, never by importing that package — but most of the refinement
examples in these pages are *written* with it, so install it too if you want to
run them as printed:

```bash
pip install annotated-types
```

Wheels are published for Linux (manylinux and musllinux, x86_64 and aarch64),
macOS (Intel and Apple silicon), Windows, and free-threaded CPython 3.14 where
the release image exposes a `cp314t` interpreter. Free-threaded support starts at
3.14t; the earlier 3.13 free-threaded build is not a target.

**PyPy 3.11 is a target, on Linux.** Four wheels are published for it —
manylinux and musllinux, x86_64 and aarch64 — and every push builds the
extension against PyPy, imports it, and runs the whole suite there. Both halves
are needed, because the C API PyPy offers is not CPython's: an extension there
runs through `cpyext`, which carries the limited API and not every static type
object CPython exports, so naming one of those links fine and fails at `import`
— and `cpyext` also *answers* differently, which no import can show. There is no
PyPy wheel for macOS or Windows, where the source distribution is the install.

One promise this page makes holds differently there. A validator releases the
classes, enums and predicates its schema names when nothing else holds them, and
on PyPy it cannot: `cpyext` builds a `PyTypeObject` proxy for every class an
extension is shown and never frees it, so a class is kept alive from the first
validator that reads it, whatever that validator does afterwards. A long-running
process that compiles a schema over a *throwaway* class per request grows there
and does not on CPython. Nothing in valgebra can change this, and the suite's
three lifetime cases say so on PyPy rather than claiming to pass.

On a free-threaded interpreter a validator is immutable and shares no mutable
walk state, so object validation runs in parallel with the interpreter lock
disabled. The JSON path is the one exception: its string parser draws on a
process-wide interned-string cache guarded by a lock, so concurrent
`validate_json`/`load` calls serialize briefly on that cache even though the walk
itself does not.

## From source

Building from source additionally requires:

- A stable **Rust** toolchain (edition 2024, MSRV 1.88) via
  [rustup](https://rustup.rs/).
- [**uv**](https://docs.astral.sh/uv/) (recommended) for the environment and the
  build.

```bash
git clone https://github.com/ppigazzini/valgebra && cd valgebra
uv sync                 # create .venv and install the dev dependencies
uv run maturin develop  # build the Rust extension into the venv
```

## Verify it works

```python
import valgebra
from valgebra import Validator

print(valgebra.__version__)
assert Validator(int).is_valid(7)
```
