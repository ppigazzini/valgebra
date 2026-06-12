# valgebra

**Fast runtime validation through a complete Boolean algebra of schemas.**

A schema denotes a *set of Python values*. Validation is membership: a validator
checks whether the object you already hold belongs to the set — no copy, no
coercion. Schemas compile once into a Rust validator tree and the hot path
crosses into Rust exactly once per call.

> [!WARNING]
> **Pre-alpha.** valgebra is under active development. The walking skeleton
> compiles a single schema (`int`) through the Rust core; the schema IR, the
> typing-annotation frontend, the Boolean algebra, recursive schemas, the JSON
> input path, and the performance program are planned. There is no PyPI release.

## Status

`validator(schema)` builds an immutable validator. `is_valid` returns a bool
membership test against the compiled Rust tree:

```python
from valgebra import validator

validator(int).is_valid(7)   # True
```

**Planned:** the full schema IR (scalars, literals, sequences, tuples, sets,
mappings, records), the typing-annotation frontend, the combinator algebra
(`union`, `intersect`, `complement`), recursive schemas, structured error
reporting, the JSON path, and the performance program.

## Install

Not yet published to PyPI. Build from source:

```bash
git clone <repository-url> valgebra && cd valgebra
uv sync                 # create .venv and install dev dependencies
uv run maturin develop  # build the Rust extension into the venv
uv run python -c "from valgebra import validator; print(validator(int).is_valid(7))"
```

Requirements: stable Rust (edition 2024, MSRV 1.88), Python >= 3.10, and
[`uv`](https://docs.astral.sh/uv/).

## Contributing

Development standards, the build-health gate, and the project's non-negotiable
rules live in [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
