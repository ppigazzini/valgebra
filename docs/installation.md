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

Wheels are published for Linux (manylinux and musllinux, x86_64 and aarch64),
macOS (Intel and Apple silicon), Windows, and free-threaded CPython 3.14 where
the release image exposes a `cp314t` interpreter. Free-threaded support starts at
3.14t; the earlier 3.13 free-threaded build is not a target.

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
