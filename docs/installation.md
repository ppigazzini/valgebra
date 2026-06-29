---
description: Install the package and its build toolchain.
---

# Installation

valgebra is not yet published to PyPI. Until the first release, install it from
source. It is a Rust extension built with [maturin](https://www.maturin.rs/), so
a Rust toolchain is required to build it.

## Requirements

- A stable **Rust** toolchain (edition 2024, MSRV 1.88) via
  [rustup](https://rustup.rs/).
- **Python** 3.10 or newer.
- [**uv**](https://docs.astral.sh/uv/) (recommended) for the environment and the
  build.

## From source

```bash
git clone https://github.com/ppigazzini/valgebra && cd valgebra
uv sync                 # create .venv and install the dev dependencies
uv run maturin develop  # build the Rust extension into the venv
```

Verify it works:

```python
from valgebra import Validator

assert Validator(int).is_valid(7)
```

## After the first release

Once valgebra is on PyPI it will install as a prebuilt wheel, with no Rust
toolchain required:

```bash
pip install valgebra
```

Wheels will be published for Linux (manylinux and musllinux, x86_64 and aarch64),
macOS (Intel and Apple silicon), Windows, and free-threaded CPython 3.14 where
the release image exposes a `cp314t` interpreter.
