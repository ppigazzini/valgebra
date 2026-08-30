---
description: What valgebra is, how it is positioned, and the index to this guide.
---

# valgebra

**Fast runtime validation through a closed Boolean algebra of schemas.**

A schema denotes a *set of Python values*. Validation is membership: you ask
whether the object you already hold belongs to the set — no copy, no coercion.
Schemas compile once into a Rust validator tree, and the hot path crosses into
Rust exactly once per call.

```python
from typing import Annotated, TypedDict

import annotated_types as at

from valgebra import Validator


class User(TypedDict):
    name: str
    age: Annotated[int, at.Ge(0)]


users = Validator(User)
assert users.is_valid({"name": "Ada", "age": 36})
assert not users.is_valid({"name": "Ada", "age": -1})
```

!!! warning "Pre-alpha"
    valgebra is under active development and published to PyPI. The APIs
    described here work today but may change before a stable `0.1.0` release.
    See the [release notes](12-changelog.md) for what is built.

## What valgebra is for

valgebra is a **contracts-and-checking** tool, not a parsing framework. Reach for
it when you want to *check an object you already hold* against a composable,
inspectable contract — cheaply enough to run on every request or every agent
turn.

For ingesting untrusted input into typed models with coercion and defaults, use
**pydantic**; for the fastest deserialization into structs, use **msgspec**.
Neither answers the membership question: both check while *building* a value,
and hand back one they are given without re-examining it. valgebra occupies the
niche neither covers: a closed, lawful **algebra** of schemas (union,
intersection, complement, refinement, fixpoints) with **check-only** semantics,
on a Rust core.

That is a difference in job as well as in speed, and the difference in job has
consequences a benchmark does not measure. Both parsers check while building a
value from untyped input: handed a value that is already an instance of the
target class, each returns it without checking its fields, and neither
re-examines a value it built when something mutates it later. A membership question can be asked again, of the
same object, as many times as the contract needs. And because a schema here is a
*value* in an algebra rather than a class declaration, `is_subtype_of`,
`is_equivalent` and `is_empty` ask about the schemas themselves, with no value
involved. `tests/test_pydantic_boundary.py` pins each of these against pydantic,
including the case that runs the other way: a pydantic `BaseModel` reaches
valgebra as a bare class and is not deep-checked.

## What makes it different

- **Schemas denote sets; validation is membership.** Subtyping is set inclusion
  and equivalence is mutual inclusion — decided soundly over a wide fragment and
  deliberately conservative beyond it. The whole model is one idea.
- **A real Boolean algebra.** `union`, `intersection`, and `complement` compose any
  schema into a lattice whose laws are property-tested, with a law-justified
  [simplifier](04-algebra.md) that never changes a schema's value set.
- **Typing-first.** Standard annotations are the primary notation, read through
  the typing spec's own introspection.
- **Check, don't parse.** `validate` and `is_valid` never copy or coerce; `ensure`
  is the explicit, separate conversion mode.
- **Few boundary crossings.** Tree walks, key lookups, and bound checks run in
  Rust; a comparison against a Python object — a literal, a refinement predicate,
  or an instance or attribute check — is the documented step into Python, never a
  silent fallback.
- **JSON on the Rust path.** `validate_json` parses and validates JSON in Rust,
  consistent with the object path.
- **Immutable and thread-safe** by design. Free-threaded CPython 3.14 is
  supported with a dedicated `cp314t` wheel where the release image exposes that
  interpreter.

## The set

Read in order, or jump: each page owns one subject and says what it does not
cover.

| Page | Covers |
|---|---|
| [00-installation.md](00-installation.md) | installing the package and its build toolchain |
| [01-tutorial.md](01-tutorial.md) | a guided path from a scalar schema to an inspected failure |
| [02-quickstart.md](02-quickstart.md) | the condensed tour, for a reader who knows the domain |
| [03-schema-language.md](03-schema-language.md) | every schema form, with its denotation as a set of values |
| [04-algebra.md](04-algebra.md) | the Boolean lattice, and the law-justified simplifier |
| [05-refinements.md](05-refinements.md) | constraints and predicate refinements over a base schema |
| [06-recursion.md](06-recursion.md) | the fixpoint for self-referential schemas |
| [07-json.md](07-json.md) | parsing and validating JSON on the Rust path |
| [08-error-model.md](08-error-model.md) | the exception type, the codes, and the violation path |
| [09-inspection.md](09-inspection.md) | interrogating a codebase that has no schemas of its own |
| [10-limits.md](10-limits.md) | the resource bounds the validator enforces |
| [11-performance.md](11-performance.md) | the measured benchmarks, and what they do not claim |
| [12-changelog.md](12-changelog.md) | every released version, and what to do on upgrade |
| [13-foundations.md](13-foundations.md) | the denotational frame and the theory it is sourced from |
| [14-soundness.md](14-soundness.md) | why an accept is never wrong, node by node |
| [15-decidability.md](15-decidability.md) | what is decided exactly, and what stays conservative |
| [16-api.md](16-api.md) | everything callable |

## Where to start

- New here? [Installation](00-installation.md), then the
  [quickstart](02-quickstart.md).
- Writing schemas? The [schema language](03-schema-language.md) covers every
  form with its denotation, [refinements](05-refinements.md) the constraints,
  and [recursive schemas](06-recursion.md) the fixpoint.
- Composing them? The [Boolean algebra](04-algebra.md).
- Studying a codebase with no schemas of its own? [Inspecting a
  codebase](09-inspection.md) asks what its annotations imply, and what nothing
  enforces.
- Wondering what it decides? The [decidability boundary](15-decidability.md),
  and the [foundations](13-foundations.md) behind it.
