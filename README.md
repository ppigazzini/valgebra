# valgebra

**Fast runtime validation through a closed Boolean algebra of schemas.**

📖 **Documentation: <https://ppigazzini.github.io/valgebra/>**

A schema denotes a *set of Python values*. Validation is membership: `validate`
checks whether the object you already hold belongs to the set — no copy, no
coercion. Schemas compile once into a Rust validator tree and the hot path
crosses into Rust exactly once per call.

> [!WARNING]
> **Pre-alpha.** valgebra is under active development. The validator engine, the
> typing-annotation frontend, the Boolean algebra (`union`, `intersection`,
> `complement`) with its law-justified simplifier, recursive schemas (`recursive`),
> [JSON input](docs/json.md), and the benchmarked
> [performance program](docs/performance.md) work today. There is no PyPI
> release.

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

from valgebra import Validator

assert Validator(int).is_valid(42)                        # scalars
assert Validator(list[int]).is_valid([1, 2, 3])           # generics
assert Validator(dict[str, int]).is_valid({"a": 1})
assert Validator(tuple[int, ...]).is_valid((1, 2, 3))
assert Validator(int | None).is_valid(None)               # unions and Optional
assert Validator(Literal["red", "green"]).is_valid("red")  # literals

# refinements via Annotated and annotated-types
adult = Validator(Annotated[int, at.Ge(18), at.Le(150)])
assert adult.is_valid(21)
assert not adult.is_valid(5)


# TypedDicts, dataclasses, NamedTuples, Enums, and Protocols all compile
class User(TypedDict):
    name: str
    age: Annotated[int, at.Ge(0)]


assert Validator(User).is_valid({"name": "Ada", "age": 36})
assert not Validator(User).is_valid({"name": "Ada", "age": -1})  # field bound holds
```

`Validator(schema)` builds an immutable validator with `validate` (raises),
`is_valid` (returns a bool), and `ensure` (validates, returns the object). A
failure raises `ValidationError` carrying a machine-readable `code`, the `path`
to the offending value, the `expected` label, and a `value` summary.

valgebra also accepts compact native forms — a dict literal is a closed record
(a trailing `"?"` marks an optional key), and any constant is an equality
literal. A compiled schema prints back as the annotation that produced it:

```python
from valgebra import ValidationError, Validator

user = Validator({"name": str, "age?": int})         # record, closed by default
assert user.is_valid({"name": "Ada"})
assert not user.is_valid({"name": "Ada", "x": 1})  # closed: no extra keys

# errors carry a stable code and a path to the offending value
try:
    Validator({"user": {"name": str}}).validate({"user": {"name": 5}})
except ValidationError as err:
    assert err.code == "string_type"
    assert err.path == ("user", "name")

assert repr(Validator(list[dict[str, int]])) == "list[dict[str, int]]"
```

Semantics follow the real value sets: `bool` is a subtype of `int` (so
`True`/`False` are valid `int`s), while `float` is disjoint from `int` (an
`int` is not a `float`); literals are typed singletons, so `Literal[1]`,
`Literal[True]`, and `Literal[1.0]` stay distinct.

## The Boolean algebra

Union, intersection, and complement compose any schema — annotations, native
forms, or other compiled validators — into a closed, lawful Boolean lattice.
`anything` is the top (every value) and `nothing` is the bottom (no value):

```python
from typing import Annotated

import annotated_types as at

from valgebra import (
    complement,
    intersection,
    nothing,
    union,
)

assert union(int, str).is_valid("x")              # a value in either set
assert intersection(int, complement(bool)).is_valid(5)  # ints that are not bools
assert not intersection(int, complement(bool)).is_valid(True)
assert complement(nothing).is_valid(5)            # the complement of bottom is top

# conditional shapes are composed, not built in: "if it is an int it must be >= 0"
# is "(int and >= 0) or (not an int)" — a union of two intersections.
if_int_non_negative = union(
    intersection(int, Annotated[int, at.Ge(0)]),
    complement(int),
)
assert not if_int_non_negative.is_valid(-1)
assert if_int_non_negative.is_valid("x")  # not an int: admitted
```

The refinement "an int that is not a bool" is `intersection(int, complement(bool))`
— expressed with the algebra, never baked into the primitives.

### `Any` versus `anything`

`Any` is the gradual dynamic type; `anything` is the top of the lattice. At
runtime both admit every value, but they are different: `anything` obeys the
lattice laws (`complement(anything)` is `nothing`, `intersection(anything, s)` is
`s`), while `Any` is an atom the simplifier never rewrites, preserving
"deliberately unchecked" as distinct from "checked: all values admitted".

`validator.simplify()` reduces a schema by the lattice laws while admitting
exactly the same values:

```python
from valgebra import complement, union

assert repr(complement(complement(int)).simplify()) == "int"
assert repr(union(int, int).simplify()) == "int"
```

### Comparing schemas as sets

Because schemas denote sets, a compiled validator can be compared with another —
`is_subtype_of` is set inclusion, `is_equivalent` is mutual inclusion, and `is_empty`
detects an unsatisfiable schema. The answers are sound (a positive answer is a
proof) and decide a wide fragment; what stays conservative is mapped in the
[decidability boundary](docs/decidability.md).

```python
from typing import Annotated

import annotated_types as at

from valgebra import complement, intersection, union, Validator

assert Validator(bool).is_subtype_of(int)                       # bool is a subtype of int
assert union(bool, int).is_equivalent(int)                      # bool | int is just int
assert Validator(Annotated[int, at.Ge(0)]).is_subtype_of(int)   # a refinement <= its base
assert intersection(int, complement(int)).is_empty()            # an unsatisfiable schema
```

## Recursive schemas

`recursive` ties a fixpoint: the builder receives a placeholder standing for the
schema being defined and returns its body. The recursive reference must occur
under a structural constructor (a list, tuple, set, dict, record, or object) so
membership stays decidable; a non-contractive body is rejected when the
validator is built.

```python
from valgebra import recursive, union, Validator

# the recursive JSON value: a fixpoint over the structural constructors
json_value = recursive(
    lambda j: union(None, bool, int, float, str, [j], {str: j}),
)
assert json_value.is_valid({"a": [1, "x", {"b": None}], "c": [True, 3.5]})
assert not json_value.is_valid({"a": object()})

# a binary tree, then composed into a larger schema
tree = recursive(lambda t: {"value": int, "left?": t, "right?": t})
assert tree.is_valid({"value": 1, "left": {"value": 2}})
assert Validator([json_value]).is_valid([1, {"k": [None, 2]}])
```

A value that contains itself is rejected (`recursion_loop`) rather than looped
on, and recursion past a fixed depth fails cleanly (`recursion_limit`) instead
of exhausting the stack.

## JSON input

`validate_json` and `is_valid_json` validate JSON source directly, parsing on
the Rust path. The parsed document runs the same walk as a native object, so a
JSON document is judged exactly as `json.loads` of it would be — same decision,
same errors — and parsing in Rust is faster than parse-then-validate:

```python
from valgebra import Validator

users = Validator({"name": str, "age?": int})
users.validate_json('{"name": "Ada", "age": 36}')        # passes
assert Validator(list[int]).is_valid_json(b"[1, 2, 3]")  # str or bytes
```

The JSON-to-Python value mapping, the object-path consistency contract, and the
malformed-input behavior are on the [JSON page](docs/json.md).

## Performance

Schemas compile once; the hot path crosses into Rust a single time per call and
checks membership with no copy or coercion. On a synthetic benchmark over a
2023-era mobile CPU (the profile-guided-optimized release wheel, the same build
the release ships), validating a passing value is much faster than a strict
pydantic `TypeAdapter` on a deeply nested list (~7x), a 50-field record (~2x),
and a large `list[int]` (~1.9x), and far faster than pure-Python jsonschema
throughout. The comparison is not apples-to-apples — pydantic also constructs
output, jsonschema is pure Python, and pydantic's PyPI wheels are PGO-built too —
and the numbers are a single machine class.

The full methodology, the recorded matrix with versions and machine class, the
honest limits, and the deterministic instruction-count CI regression gate are in
the [performance page](docs/performance.md). Reproduce with `cargo bench` and
`uv run --group bench pytest benches/`.

## Install

Not yet published to PyPI. Build from source:

```bash
git clone <repository-url> valgebra && cd valgebra
uv sync                 # create .venv and install dev dependencies
uv run maturin develop  # build the Rust extension into the venv
uv run python -c "from valgebra import Validator; print(Validator(int).is_valid(7))"
```

Requirements: stable Rust (edition 2024, MSRV 1.88), Python >= 3.10, and
[`uv`](https://docs.astral.sh/uv/).

## Versioning and releases

valgebra follows [Semantic Versioning](https://semver.org/). While the version
is below `1.0` the public API may change between minor versions; every change is
recorded in [CHANGELOG.md](CHANGELOG.md), and where practical a deprecated form
keeps working for at least one minor release with a documented replacement.

Releases are dispatch-driven: a manual workflow run builds the wheel matrix
(Linux manylinux and musllinux, macOS, and Windows, including a free-threaded
build) and the sdist, and publishes them to PyPI through trusted publishing with
PEP 740 attestations only when the run names a publish target and confirms the
version. No tag push publishes. The package version is read from the workspace
manifest, so a release bumps `Cargo.toml` to the version the run confirms.

## Design at a glance

- **Schemas denote sets; validation is membership.** Subtyping is set
  inclusion, equivalence is mutual inclusion. *(In place.)*
- **Check, don't parse.** `validate` and `is_valid` never copy or coerce;
  `ensure` is the explicit, separate value-returning mode. *(In place.)*
- **One boundary crossing.** The validator tree runs entirely in Rust; Python
  predicates are a documented slow path, never a silent fallback. *(In place.)*
- **A lawful algebra.** Union, intersection, and complement form a Boolean
  lattice whose laws are property-tested in both Rust and Python, and a
  law-justified `simplify` reduces a schema without changing its value set. The
  set-theoretic model and its references are in the
  [foundations](docs/foundations.md). *(In place.)*
- **Immutable and thread-safe.** A compiled validator never mutates after it is
  built, so one validator can be shared and used from many threads at once,
  including on free-threaded (no-GIL) CPython. *(In place.)*

## Contributing

[ARCHITECTURE.md](ARCHITECTURE.md) maps the components and the path a value
takes through them. Development standards, the build-health gate, and the
project's non-negotiable rules live in [AGENTS.md](AGENTS.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

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
