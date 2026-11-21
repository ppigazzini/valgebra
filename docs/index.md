---
description: What valgebra is, how it is positioned, and where to start.
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

A set answers two questions: whether a value you hold belongs to it, and whether every value
of one set belongs to another, `subset`, `not_subset` or `undecided`.
[How it works](explanation/how-it-works.md) follows a schema from its annotation to both
answers, with a diagram for each step.

!!! warning "Pre-alpha"
    valgebra is under active development and published to PyPI. The APIs
    described here work, and every `0.0.x` release may break them, saying where
    in its notes. See the [release notes](changelog.md) for what is
    built and what counts as a break.

## What valgebra is for

valgebra is a **contracts-and-checking** tool, not a parsing framework. Reach for
it when you want to *check an object you already hold* against a composable,
inspectable contract — cheaply enough to run on every request or every agent
turn.

For ingesting untrusted input into typed models with coercion and defaults, use
**pydantic**; for decoding bytes into structs, use **msgspec**.
Neither answers the membership question: both check while *building* a value,
and by default hand back one they are given without re-examining it. valgebra
occupies the niche neither covers: a closed, lawful **algebra** of schemas
(union, intersection, complement, refinement, fixpoints) with **check-only**
semantics, on a Rust core.

That is a difference in job as well as in speed, and the difference in job has
consequences a benchmark does not measure. Both parsers check while building a
value from untyped input: handed a value that is already an instance of the
target class, each returns it without checking its fields by default. pydantic
re-checks one only for a class configured to ask for it (`revalidate_instances`,
read off the class rather than the call), and checks a later mutation only at
the assignment it is configured to watch (`validate_assignment`); msgspec's
checking runs on the decode path alone. valgebra checks every field of a held
instance, and asks the class alone where the caller writes `instance_of`. A
membership question can be asked again, of the same object, as many times as
the contract needs. And because a schema here is a *value* in
an algebra rather than a class declaration, `is_subtype_of`,
`is_equivalent` and `is_empty` ask about the schemas themselves, with no value
involved. `tests/test_pydantic_boundary.py` pins each of these against pydantic,
including the case that runs the other way: a pydantic `BaseModel` reaches
valgebra as a bare class and is not deep-checked.

## What makes it different

- **Schemas denote sets; validation is membership.** Subtyping is set inclusion
  and equivalence is mutual inclusion — decided soundly over a wide fragment and
  deliberately conservative beyond it. The whole model is one idea.
- **A real Boolean algebra.** `union`, `intersection`, and `complement` compose
  any schema into a lattice whose laws are property-tested, and a schema is built
  in the [lattice normal form](guides/algebra.md): `repr` shows it and `==` compares
  it, so no later pass is needed to reach it.
- **Typing-first.** Standard annotations are the primary notation, read through
  the typing spec's own introspection.
- **Check, don't parse.** `validate` and `is_valid` never copy or coerce; `ensure`
  is the explicit, separate mode that hands back the object it checked,
  unchanged.
- **Few boundary crossings.** Tree walks, key lookups, and bound checks run in
  Rust; a comparison against a Python object — a literal, a refinement predicate,
  or an instance or attribute check — is the documented step into Python, never a
  silent fallback.
- **JSON on the Rust path.** `validate_json` parses and validates JSON in Rust,
  consistent with the object path.
- **Immutable and thread-safe** by design. Free-threaded CPython 3.14 and 3.15
  are supported, with `cp314t` and `cp315t` wheels for Linux, macOS and Windows
  x64.

## Where to start

- New here? [Installation](get-started/installation.md), then the
  [quickstart](get-started/quickstart.md).
- Have a task in mind? The [recipes](get-started/recipes.md) each do one end to end.
- Writing schemas? The [schema language](guides/schema-language.md) covers every
  form with its denotation, [refinements](guides/refinements.md) the constraints,
  and [recursive schemas](guides/recursion.md) the fixpoint.
- Composing them? The [Boolean algebra](guides/algebra.md).
- Type-checking code that uses them? [Static checkers](guides/static-checking.md).
- Studying a codebase with no schemas of its own? [Inspecting a
  codebase](guides/inspection.md) asks what its annotations imply, and what nothing
  enforces.
- Holding another format's values? [Several value universes](explanation/universes.md) writes
  JSON's and TOML's as sets, and reads arrays by what they expose.
- Wondering what it decides? The [decidability boundary](explanation/decidability.md),
  and the [foundations](explanation/foundations.md) behind it.
- Wondering whether it does something at all? [What it cannot
  do](explanation/boundaries.md) lists the deliberate boundaries and says which are the
  product's and which are a runtime check's.
