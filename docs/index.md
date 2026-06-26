---
description: Overview, positioning against pydantic and msgspec, and a first example.
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
    valgebra is under active development and is not yet published to PyPI. The
    APIs described here work today; see the
    [changelog](https://github.com/ppigazzini/valgebra/blob/main/CHANGELOG.md)
    for what is built.

## What valgebra is for

valgebra is a **contracts-and-checking** tool, not a parsing framework. Reach for
it when you want to *check an object you already hold* against a composable,
inspectable contract — cheaply enough to run on every request or every agent
turn.

For ingesting untrusted input into typed models with coercion and defaults, use
**pydantic**; for the fastest deserialization into structs, use **msgspec**.
valgebra occupies the niche neither covers: a closed, lawful **algebra** of
schemas (union, intersection, complement, refinement, fixpoints) with
**check-only** semantics, on a Rust core.

## What makes it different

- **Schemas denote sets; validation is membership.** Subtyping is set inclusion
  and equivalence is mutual inclusion — decided soundly over a wide fragment and
  deliberately conservative beyond it. The whole model is one idea.
- **A real Boolean algebra.** `union`, `intersection`, and `complement` compose any
  schema into a lattice whose laws are property-tested, with a law-justified
  [simplifier](algebra.md) that never changes a schema's value set.
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
- **Immutable and thread-safe** by design. Free-threaded CPython is a monitored
  target, not a supported platform yet.

## Where to go next

- New here? Start with [installation](installation.md) and the
  [quickstart](quickstart.md).
- Writing schemas? The [schema language](schema-language.md) reference covers
  every form with its denotation; [refinements](refinements.md) covers
  constraints, and [recursive schemas](recursion.md) the `recursive` fixpoint.
- Composing them? See the [Boolean algebra](algebra.md).
- What does it decide? The [decidability boundary](decidability.md) maps what
  subtyping, equivalence, and emptiness answer exactly and what stays
  conservative; the [foundations](foundations.md) record the theory.
- Everything callable is in the [API reference](api.md).
