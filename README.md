# valgebra

**A closed, irreducible Boolean algebra of schemas for Python.** A schema denotes
a *set of Python values*, and validating asks whether a value you already hold is
a member — no copy, no coercion. `union`, `intersection`, `complement`,
refinement, and fixpoints are the only primitives; they **close** into a lattice
whose laws are property-tested, and every other pattern is **derived** from them
by composition rather than bundled as a special combinator. The schema compiles
to a Rust validator, so a check is cheap enough to run on every request.

📖 **Documentation: <https://ppigazzini.github.io/valgebra/>**

🤖 **For AI assistants and coding agents:** the documentation is also published as
[`llms.txt`](https://ppigazzini.github.io/valgebra/llms.txt) (a curated manifest)
and [`llms-full.txt`](https://ppigazzini.github.io/valgebra/llms-full.txt) (the
full text, including the API reference) per the [llmstxt.org](https://llmstxt.org)
convention.

> [!WARNING]
> **Pre-alpha.** Published to PyPI; the API works today but may change before a
> stable `0.1.0` release.

## Schemas are sets; the operators are *or*, *and*, *not*

The everyday case looks like any validator — a type annotation is a schema, and
checking it asks whether a value belongs to the set the annotation denotes:

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

What sets valgebra apart starts when you treat schemas *as the sets they denote*.
Because membership is Boolean, `union`, `intersection`, and `complement` are
exactly *or*, *and*, and *not*, and they compose any schema into a lattice:

```python
from valgebra import Validator, complement, intersection, union

non_bool_int = intersection(int, complement(bool))  # an int that is not a bool
assert non_bool_int.is_valid(5)
assert not non_bool_int.is_valid(True)

# Schemas are first-class values you can compare as sets — soundly.
assert Validator(bool).is_subtype_of(int)  # subtyping is set inclusion
assert union(bool, int).is_equivalent(int)  # same set, different syntax
assert intersection(int, complement(int)).is_empty()  # provably no value
```

## What makes it peculiar

- **A real, closed Boolean algebra.** Schemas compose with `union`,
  `intersection`, and `complement` into a lattice with `anything` as top and
  `nothing` as bottom. Every Boolean law — associativity, idempotence, absorption,
  distributivity, De Morgan, double negation — is *property-tested against the
  membership relation*, not asserted.
- **Irreducible: only the generators ship.** valgebra bundles no `conditional`,
  no `at_least_one`, no `one_of`. Those are **derived** by composition (see
  [below](#everything-else-is-derived)). A named wrapper for a one-line
  composition would make a standard library, not a schema algebra.
- **Schemas are comparable values.** `is_subtype_of` (inclusion),
  `is_equivalent` (mutual inclusion), and `is_empty` (unsatisfiable) form a
  **sound** decision procedure: a `True` is always correct, and the procedure
  decides a wide fragment completely and stays conservative beyond it — never a
  wrong answer. Keep `is_equivalent` (semantic) distinct from `==` (syntactic
  shape).
- **A law-justified simplifier.** `simplify` reduces a schema to a lattice normal
  form that admits **exactly the same values** — it never changes a schema's
  meaning.
- **`anything` is not `Any`.** Both admit every value at runtime, but in the
  algebra `anything` is the lattice top (it obeys the laws) while `Any` is the
  gradual dynamic type — an atom the simplifier never rewrites. "Checked: all
  values admitted" stays distinct from "deliberately unchecked".
- **Check, don't parse.** `validate`/`is_valid` never copy or coerce; the proof
  is about the object you keep, not a reconstructed copy. `ensure` is the
  separate, explicit value-returning mode.
- **Typing-first.** Standard annotations are the primary notation, read through
  the typing spec's own introspection. Union has the operator typing already
  uses, `|`; intersection and complement stay spelled out because typing has no
  operator for them and valgebra invents none.

```python
from typing import Any

from valgebra import Validator, anything, complement, union

# A law-justified simplifier: a lattice normal form with the same value set.
assert repr(complement(complement(int)).simplify()) == "int"  # double negation
assert repr(union(int, int).simplify()) == "int"  # idempotence

# `anything` is the lattice top; `Any` means "deliberately unchecked".
assert repr(complement(anything).simplify()) == "nothing"  # top obeys the laws
assert repr(Validator(Any).simplify()) == "Any"  # left untouched

# Union has typing's `|`; intersection and complement stay spelled out.
assert (Validator(int) | str | None).is_equivalent(union(int, str, None))
```

## Everything else is derived

Because the algebra is closed, the patterns other libraries ship as built-in
combinators are one-line compositions here. "If it is an `int`, it must be
non-negative" is a `union` of two intersections — no `implies` primitive exists,
you derive it:

```python
from typing import Annotated

import annotated_types as at

from valgebra import anything, complement, intersection, union


def implies(condition, then, otherwise=anything):
    return union(
        intersection(condition, then),
        intersection(complement(condition), otherwise),
    )


non_negative_if_int = implies(int, Annotated[int, at.Ge(0)])
assert non_negative_if_int.is_valid(5)
assert not non_negative_if_int.is_valid(-1)
assert non_negative_if_int.is_valid("not an int")  # not an int: admitted
```

The same handful of operators derives first-matching-case dispatch, key
cardinality ("at least one of these keys", "exactly one", "not both"),
length-bounded lists, and conditional records. The recipes — each runnable and
explained — live in the **[Boolean algebra guide](docs/04-algebra.md#composition-recipes)**.

## Recursive schemas and JSON

Recursive (`recursive`) schemas describe trees and JSON-like data, and JSON input
is validated directly on the Rust path — parsed and checked in one pass, never
materialized into an untyped object graph first:

```python
from valgebra import Validator

assert Validator(list[int]).is_valid_json(b"[1, 2, 3]")  # parse + check in Rust
```

## Two ways to use it

### As a contract in your codebase

Reach for valgebra when you **already hold a Python object** — a parsed request
body, a config dict, an LLM tool-call argument, a function input — and need to
*check* it against a composable, inspectable contract on the hot path, cheaply
enough to run on every request or every agent turn. Because the algebra is
closed, a subsystem's contract is the *intersection* of its parts' contracts, an
exclusion is a *complement*, and a migration is "old schema *or* new" — contracts
refactor like code instead of decaying into opaque predicate functions.

Start at the [tutorial](docs/01-tutorial.md); the [algebra guide](docs/04-algebra.md)
covers composition.

### As a harness over a codebase that has none

A schema is an ordinary annotation, so valgebra reads the annotations a codebase
**already has** and answers questions about them the interpreter cannot. Nothing
is added to the code under study: the schemas live in the script asking the
question, and valgebra stays a development dependency. This is the mode that
suits an agent working on a codebase it did not write.

The sharpest question it can settle is whether a contract the code *implies* is
one anything *enforces*. A parameter used as a divisor must not be zero; one
whose attribute is read must not be `None`. The body states that by using the
value that way, and `declared ∧ ¬implied` is exactly the set of values that pass
the type check and break the function:

```python
from valgebra import Validator, complement, intersection


def unenforced(declared: object, implied: object) -> list[object]:
    """Values the declaration admits and the body cannot survive."""
    breaking = intersection(Validator(declared), complement(Validator(implied)))
    return [p for p in (None, 0, "", []) if breaking.is_valid(p)]


# `def make_grid(columns: int)` whose body computes `idx // columns`:
# the annotation is correct, the code typechecks, and zero breaks it.
assert unenforced(int, complement(Validator(0))) == [0]
```

Others in the same shape — a branch its own annotation makes unreachable, a union
arm another already covers, an annotation that admits nothing, an override that
narrows its base, and which arms of a union a test run never reached:

```python
from valgebra import Validator, intersection, union

# `if isinstance(key, bytes)` inside `def __setitem__(self, key: str, ...)`.
# The branch is dead, so the code and its annotation disagree about what arrives.
assert intersection(Validator(str), Validator(bytes)).is_empty()

# `bool | int` is `int`: bool is a subclass, so the arm adds nothing.
assert union(bool, int).is_equivalent(int)

# `except (OSError, TimeoutError)` names one class that contains the other.
assert Validator(TimeoutError).is_subtype_of(OSError)
```

**[Inspecting a codebase](docs/09-inspection.md)** is the full set of recipes, with
what each cannot see.

Neither mode requires the other. A codebase can adopt valgebra as a runtime
contract, or never import it in its own source and still be studied with it.

## How it compares

valgebra **checks an object you already hold**; it never coerces or constructs.
That makes it different from the tools you might already use:

- **pydantic** parses untrusted input into typed models *with coercion and
  defaults*, and guarantees the type of what it *returns*. Use it for ingestion;
  use valgebra to check a value you already hold — including one pydantic built.
- **msgspec** is the fastest path for *deserializing* bytes into structs, and
  it checks on that path only. A `Struct` constructor validates nothing, and
  `convert` hands back a `Struct` you already hold without re-examining it —
  there is no membership call to ask instead. Use it to decode; use valgebra
  to check what you decoded.
- **jsonschema** validates against the JSON Schema standard; valgebra validates
  against Python types and a set-theoretic algebra instead.

### The difference is not only speed

The benchmark below measures one operation: how long a passing check takes.
Three differences do not appear in it, because each is a verdict rather than a
duration. [`tests/test_pydantic_boundary.py`](tests/test_pydantic_boundary.py)
runs all three libraries over the same values and asserts every claim here.

**An object you already hold is re-examined.** Handed a value that is already an
instance of the target class, `TypeAdapter.validate_python` returns it without
checking its fields, and `msgspec.convert` returns it under `strict` and
`from_attributes` alike. pydantic's re-check is a *model config*
(`revalidate_instances`), so reaching it requires pydantic to own the class
declaration — for a dataclass declared elsewhere `TypeAdapter` raises
`PydanticUserError` rather than ignoring the setting. msgspec has no such
setting: its checking runs on the decode path, from untyped input. valgebra
reads the schema off the class, and every call asks the same membership
question.

**A value stays checkable after it changes.** A check that runs at construction
answers about the value handed to the constructor. Membership is a call you
repeat on the same object, with no rebuild:

```python
from dataclasses import dataclass

from valgebra import Validator


@dataclass
class Config:
    lr: float
    steps: int


is_config = Validator(Config)  # derived from the dataclass, never redeclared
config = Config(lr=0.1, steps=10)
assert is_config.is_valid(config)

config.steps = "ten"  # something mutates it later
assert not is_config.is_valid(config)  # the same question, asked again
```

**Schemas answer questions with no value involved.** `is_subtype_of`,
`is_equivalent`, `is_empty` and `simplify` take no value at all, so a contract
no value satisfies is decided from the schemas: `intersection(int, str)` reports
empty without a test case reaching it. Neither `TypeAdapter` nor
`msgspec.inspect` exposes any of the three, and the suite asserts that by
reading both surfaces, so the claim fails if either library grows one.
`complement` is the other half: a set defined by what it *excludes* is a schema
here, and a Python predicate — carried and run, but not reasoned about —
elsewhere.

Where the boundary runs the other way: pydantic in `strict` mode rejects without
coercing on the ingestion path, msgspec reaches a held value through
`convert(to_builtins(value), type=T)` — a check that costs building two objects
and discarding them — and valgebra deep-checks neither a pydantic `BaseModel`
nor a msgspec `Struct`: each reaches the frontend as a bare class, denoting the
set of its instances. Check either one's fields through a mapping view of them.

On a synthetic benchmark (the PGO release wheel) a passing check is faster than a
strict pydantic `TypeAdapter` — roughly 2× on a 50-field record and a large
`list[int]`, ~7× on deep nesting — and far faster than pure-Python jsonschema. The
comparison is not apples-to-apples and is gated against regression in CI; see the
[performance page](docs/11-performance.md) for the method, the matrix, and the limits.

## Install

valgebra ships prebuilt wheels to PyPI, so installing it needs no Rust toolchain
(Python ≥ 3.10):

```bash
pip install valgebra
# or
uv add valgebra
```

To build from source instead — for development or an unsupported platform —
requires [`uv`](https://docs.astral.sh/uv/) and stable Rust (edition 2024, MSRV
1.88):

```bash
git clone https://github.com/ppigazzini/valgebra && cd valgebra
uv sync                 # create .venv and install dev dependencies
uv run maturin develop  # build the Rust extension into the venv
```

## Why valgebra (in one screen)

- **Schemas are sets; validation is membership.** Subtyping is set inclusion and
  equivalence is mutual inclusion — sound, deciding a wide fragment and staying
  deliberately conservative beyond it ([foundations](docs/13-foundations.md),
  [decidability](docs/15-decidability.md), [soundness argument](docs/14-soundness.md)).
- **A closed, irreducible algebra.** Five primitives generate everything; the
  laws are property-tested and a law-justified simplifier exploits them.
- **Check, don't parse.** `validate`/`is_valid` never copy or coerce; `ensure` is
  the explicit value-returning mode.
- **One boundary crossing.** Tree walks, key lookups, and bound checks run in
  Rust; a comparison against a Python object — a literal, a refinement predicate,
  or an instance or attribute check — is the documented step into Python, never a
  silent fallback.
- **Immutable and thread-safe** by design. Free-threaded (no-GIL) CPython 3.14 is
  supported with a dedicated `cp314t` wheel where the release image exposes that
  interpreter.

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
