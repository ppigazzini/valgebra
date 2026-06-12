# valgebra

**Fast runtime validation through a complete Boolean algebra of schemas.**

A schema denotes a *set of Python values*. Validation is membership: a validator
checks whether the object you already hold belongs to the set — no copy, no
coercion. Schemas compile once into a Rust validator tree and the hot path
crosses into Rust exactly once per call.

> [!WARNING]
> **Pre-alpha.** valgebra is under active development. The validator engine
> compiles the structural schema language through the Rust core; the
> typing-annotation frontend, the Boolean algebra, recursive schemas, the JSON
> input path, and the performance program are planned. There is no PyPI release.

## Status

`validator(schema)` builds an immutable validator with `validate` (raises),
`is_valid` (returns a bool), and `cast` (validates, returns the object). A
failure raises `ValidationError` carrying a machine-readable `code`, the `path`
to the offending value, the `expected` label, and a `value` summary.

Schemas are written as compact native forms — the scalar types, `None`, and
`object` for the top; `[T]` for a list, `(A, B)` for a fixed tuple, `{T}` for a
set, `{KeyType: ValueType}` for a mapping; an all-string-key dict for a closed
record (a trailing `"?"` marks an optional key); and any constant for an
exact-value literal:

```python
from valgebra import ValidationError, validator

validator(int).is_valid(42)                       # scalars
validator([int]).is_valid([1, 2, 3])              # list
validator((int, str)).is_valid((1, "a"))          # fixed tuple
validator({int}).is_valid({1, 2, 3})              # set
validator({str: int}).is_valid({"a": 1})          # mapping
validator("red").is_valid("red")                  # literal

user = validator({"name": str, "age?": int})      # record, strict by default
assert user.is_valid({"name": "Ada"})
assert not user.is_valid({"name": "Ada", "x": 1})  # closed: no extra keys

# errors carry a stable code and a path to the offending value
try:
    validator({"user": {"name": str}}).validate({"user": {"name": 5}})
except ValidationError as err:
    assert err.code == "string_type"
    assert err.path == ("user", "name")
```

Semantics follow the real value sets: `bool` is a subtype of `int` (so
`True`/`False` are valid `int`s), while `float` is disjoint from `int`; literals
are typed singletons, so `Literal[1]`, `Literal[True]`, and `Literal[1.0]` stay
distinct.

**Planned:** the typing-annotation frontend (`list[T]`, `X | Y`, `Literal`,
`TypedDict`, `Annotated` refinements), the combinator algebra (`union`,
`intersect`, `complement`), recursive schemas, aggregated error reporting, the
JSON path, and the performance program.

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
