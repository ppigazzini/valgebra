# valgebra

**A fast runtime validation library for Python.** Describe the shape you expect
with ordinary type annotations, then check whether a value matches — no copying,
no coercion, no rebuilding the object. The schema compiles to a Rust validator, so
the check is cheap enough to run on every request.

📖 **Documentation: <https://ppigazzini.github.io/valgebra/>**

🤖 **For AI assistants and coding agents:** the documentation is also published as
[`llms.txt`](https://ppigazzini.github.io/valgebra/llms.txt) (a curated manifest)
and [`llms-full.txt`](https://ppigazzini.github.io/valgebra/llms-full.txt) (the
full text, including the API reference) per the [llmstxt.org](https://llmstxt.org)
convention.

> [!WARNING]
> **Pre-alpha**, not yet on PyPI. The API works today but may change; build from
> source for now.

## What it's for

Reach for valgebra when you **already hold a Python object** — a parsed request
body, a config dict, an LLM tool-call argument, a function input — and need to
*check* it against a contract on the hot path, without turning it into something
else. A schema is a plain type annotation; checking it is asking whether the value
belongs to the set the annotation denotes.

```python
from valgebra import ValidationError, Validator

is_user = Validator({"name": str, "age": int})

assert is_user.is_valid({"name": "Ada", "age": 36})
assert not is_user.is_valid({"name": "Ada", "age": "unknown"})

# validate() raises a structured error pointing at the offending value
try:
    is_user.validate({"name": "Ada", "age": "unknown"})
except ValidationError as err:
    assert err.code == "int_type"
    assert err.path == ("age",)
```

`Validator(schema)` builds an immutable, thread-safe validator with `validate`
(raises), `is_valid` (returns a bool), and `ensure` (validates and returns the
object).

## Install

Not yet published to PyPI — build from source. Requires
[`uv`](https://docs.astral.sh/uv/), Python ≥ 3.10, and stable Rust (edition 2024,
MSRV 1.88):

```bash
git clone <repository-url> valgebra && cd valgebra
uv sync                 # create .venv and install dev dependencies
uv run maturin develop  # build the Rust extension into the venv
```

## What you can express

**Standard typing annotations** — generics, unions, literals, refinements, and
typed containers like `TypedDict`, `dataclass`, `NamedTuple`, and `Enum`:

```python
from typing import Annotated, Literal, TypedDict

import annotated_types as at

from valgebra import Validator

assert Validator(list[int]).is_valid([1, 2, 3])
assert Validator(int | None).is_valid(None)
assert Validator(Literal["red", "green"]).is_valid("red")

adult = Validator(Annotated[int, at.Ge(18)])          # refinements via Annotated
assert adult.is_valid(21) and not adult.is_valid(5)


class User(TypedDict):
    name: str
    age: Annotated[int, at.Ge(0)]


assert Validator(User).is_valid({"name": "Ada", "age": 36})
```

**A Boolean algebra of schemas.** Schemas are *sets of values*, so they compose
with union, intersection, and complement, and can be compared and simplified as
algebra — see the [algebra guide](docs/algebra.md):

```python
from valgebra import Validator, complement, intersection, union

non_bool_int = intersection(int, complement(bool))    # an int that is not a bool
assert non_bool_int.is_valid(5) and not non_bool_int.is_valid(True)

assert Validator(bool).is_subtype_of(int)             # compare schemas as sets
assert union(bool, int).is_equivalent(int)            # same set, different syntax
```

**[Recursive schemas](docs/recursion.md)** (`recursive`) for trees and JSON-like
data, and **[JSON input](docs/json.md)** validated directly on the Rust path:

```python
from valgebra import Validator

assert Validator(list[int]).is_valid_json(b"[1, 2, 3]")   # parse + check in Rust
```

## How it compares

valgebra **checks an object you already hold**; it never coerces or constructs.
That makes it different from the tools you might already use:

- **pydantic** parses untrusted input into typed models *with coercion and
  defaults*. Use it for ingestion; use valgebra to check a value you already have.
- **msgspec** is the fastest path for *deserializing* bytes into structs.
- **jsonschema** validates against the JSON Schema standard; valgebra validates
  against Python types and a set-theoretic algebra instead.

On a synthetic benchmark (the PGO release wheel) a passing check is faster than a
strict pydantic `TypeAdapter` — roughly 2× on a 50-field record and a large
`list[int]`, ~7× on deep nesting — and far faster than pure-Python jsonschema. The
comparison is not apples-to-apples and is gated against regression in CI; see the
[performance page](docs/performance.md) for the method, the matrix, and the limits.

## Why valgebra

- **Check, don't parse.** `validate`/`is_valid` never copy or coerce; `ensure` is
  the separate, explicit value-returning mode.
- **Schemas are sets; validation is membership.** Subtyping is set inclusion and
  equivalence is mutual inclusion — a single idea behind the whole model. The
  decisions are sound and decide a wide fragment, staying deliberately
  conservative beyond it ([foundations](docs/foundations.md),
  [decidability](docs/decidability.md), [soundness argument](docs/soundness.md)).
- **One boundary crossing.** Tree walks, key lookups, and bound checks run in
  Rust; a comparison against a Python object — a literal, a refinement predicate,
  or an instance or attribute check — is the documented step into Python, never a
  silent fallback.
- **Immutable and thread-safe** by design. Free-threaded (no-GIL) CPython is a
  monitored target, not a supported platform yet.

## Project

- **Versioning** follows [SemVer](https://semver.org/); changes are recorded in
  [CHANGELOG.md](CHANGELOG.md). Releases are dispatch-driven and published to PyPI
  through trusted publishing — no tag push publishes.
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md)
  cover the build-health gate and the project's rules;
  [ARCHITECTURE.md](ARCHITECTURE.md) maps the components.
- **Security**: the load-bearing property is *soundness of acceptance* — an
  accepted value really belongs to the schema's set. Report issues privately per
  [SECURITY.md](SECURITY.md). valgebra is pre-alpha and unaudited.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Contributions are dual-licensed as
above unless you state otherwise.
