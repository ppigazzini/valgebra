# valgebra

**Fast runtime validation through a complete Boolean algebra of schemas.**

A schema denotes a *set of Python values*. Validation is membership: `validate`
checks whether the object you already hold belongs to the set — no copy, no
coercion. Schemas compile once into a Rust validator tree and the hot path
crosses into Rust exactly once per call.

> [!WARNING]
> **Pre-alpha.** valgebra is under active development. The validator engine, the
> typing-annotation frontend, and the Boolean algebra (`union`, `intersect`,
> `complement`) with its law-justified simplifier work today; recursive schemas,
> the JSON input path, and the performance program are planned, so the speed
> claim is unproven. There is no PyPI release.

## Why valgebra

valgebra is a **contracts-and-checking** tool, not a parsing framework. Reach
for it when you want to *check an object you already hold* against a composable,
inspectable contract — cheaply enough to run on every request. For ingesting
untrusted input into typed models with coercion and defaults, use **pydantic**;
for the fastest deserialization into structs, use **msgspec**.

## Quickstart

Contracts are written as standard typing annotations:

```python
from typing import Annotated, Literal, TypedDict

import annotated_types as at

from valgebra import validator

assert validator(int).is_valid(42)                        # scalars
assert validator(list[int]).is_valid([1, 2, 3])           # generics
assert validator(dict[str, int]).is_valid({"a": 1})
assert validator(tuple[int, ...]).is_valid((1, 2, 3))
assert validator(int | None).is_valid(None)               # unions and Optional
assert validator(Literal["red", "green"]).is_valid("red")  # literals

# refinements via Annotated and annotated-types
adult = validator(Annotated[int, at.Ge(18), at.Le(150)])
assert adult.is_valid(21)
assert not adult.is_valid(5)


# TypedDicts, dataclasses, NamedTuples, Enums, and Protocols all compile
class User(TypedDict):
    name: str
    age: Annotated[int, at.Ge(0)]


assert validator(User).is_valid({"name": "Ada", "age": 36})
```

`validator(schema)` builds an immutable validator with `validate` (raises),
`is_valid` (returns a bool), and `cast` (validates, returns the object). A
failure raises `ValidationError` carrying a machine-readable `code`, the `path`
to the offending value, the `expected` label, and a `value` summary.

valgebra also accepts compact native forms — a dict literal is a closed record
(a trailing `"?"` marks an optional key), and any constant is an equality
literal. A compiled schema prints back as the annotation that produced it:

```python
from valgebra import ValidationError, validator

user = validator({"name": str, "age?": int})         # record, strict by default
assert user.is_valid({"name": "Ada"})
assert not user.is_valid({"name": "Ada", "x": 1})  # closed: no extra keys

# errors carry a stable code and a path to the offending value
try:
    validator({"user": {"name": str}}).validate({"user": {"name": 5}})
except ValidationError as err:
    assert err.code == "string_type"
    assert err.path == ("user", "name")

assert repr(validator(list[dict[str, int]])) == "list[dict[str, int]]"
```

Semantics follow the real value sets: `bool` is a subtype of `int` (so
`True`/`False` are valid `int`s), while `float` is disjoint from `int` (an
`int` is not a `float`); literals are typed singletons, so `Literal[1]`,
`Literal[True]`, and `Literal[1.0]` stay distinct.

## The Boolean algebra

Union, intersection, and complement compose any schema — annotations, native
forms, or other compiled validators — into a complete, lawful Boolean lattice.
`anything` is the top (every value) and `nothing` is the bottom (no value):

```python
from typing import Annotated

import annotated_types as at

from valgebra import (
    anything,
    complement,
    cond,
    ifthen,
    intersect,
    nothing,
    union,
)

assert union(int, str).is_valid("x")              # a value in either set
assert intersect(int, complement(bool)).is_valid(5)  # ints that are not bools
assert not intersect(int, complement(bool)).is_valid(True)
assert complement(nothing).is_valid(5)            # the complement of bottom is top

# conditional shapes derived from the algebra: "if it is an int, it must be >= 0"
assert not ifthen(int, Annotated[int, at.Ge(0)]).is_valid(-1)
assert ifthen(int, Annotated[int, at.Ge(0)]).is_valid("x")  # not an int: admitted
assert cond(
    (str, Annotated[str, at.MinLen(1)]),
    (int, Annotated[int, at.Ge(0)]),
    default=nothing,
).is_valid("ok")
```

The refinement "an int that is not a bool" is `intersect(int, complement(bool))`
— expressed with the algebra, never baked into the primitives.

### `Any` versus `anything`

`Any` is the gradual dynamic type; `anything` is the top of the lattice. At
runtime both admit every value, but they are different: `anything` obeys the
lattice laws (`complement(anything)` is `nothing`, `intersect(anything, s)` is
`s`), while `Any` is an atom the simplifier never rewrites, preserving
"deliberately unchecked" as distinct from "checked: all values admitted".

`simplify(validator)` reduces a schema by the lattice laws while admitting
exactly the same values:

```python
from valgebra import complement, simplify, union

assert repr(simplify(complement(complement(int)))) == "int"
assert repr(simplify(union(int, int))) == "int"
```

```python
# PLANNED: recursive schemas via the lazy fixpoint are not implemented yet.
from valgebra import lazy, union, validator

json_value = lazy(lambda j: union(None, bool, int, float, str, [j], {str: j}))
assert json_value.is_valid({"a": [1, "x", {"b": None}]})
```

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

## Design at a glance

- **Schemas denote sets; validation is membership.** Subtyping is set
  inclusion, equivalence is mutual inclusion. *(In place.)*
- **Check, don't parse.** `validate` and `is_valid` never copy or coerce; `cast`
  is the explicit, separate conversion mode. *(In place.)*
- **One boundary crossing.** The validator tree runs entirely in Rust; Python
  predicates are a documented slow path, never a silent fallback. *(In place.)*
- **A lawful algebra.** Union, intersection, and complement form a Boolean
  lattice whose laws are property-tested in both Rust and Python, and a
  law-justified `simplify` reduces a schema without changing its value set.
  *(In place.)*

## Contributing

Development standards, the build-health gate, and the project's non-negotiable
rules live in [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
uv run maturin develop
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
