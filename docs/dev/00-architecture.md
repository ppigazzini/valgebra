# Architecture

Two Rust crates and a Python package. This page owns the split between them, the
direction their dependencies run, and the two invariants the compiler holds.

## The zones

| Zone | Owns | Page |
|---|---|---|
| `crates/valgebra-core/src/ir.rs` | the schema IR: the node set, and what each node denotes | [01-schema-ir.md](01-schema-ir.md) |
| `crates/valgebra-core/src/decision.rs` | emptiness, subtyping, equivalence, disjointness | [02-decision.md](02-decision.md) |
| `crates/valgebra-core/src/simplify.rs` | the membership-preserving lattice normalisation | [02-decision.md](02-decision.md) |
| `crates/valgebra-py/src/build.rs` | typing annotations and native forms into the IR | [03-frontend.md](03-frontend.md) |
| `crates/valgebra-py/src/check/` | the membership walk | [04-walk.md](04-walk.md) |
| `crates/valgebra-py/src/errors.rs`, `render.rs` | the Python exception and the annotation render | [05-errors.md](05-errors.md) |
| `python/valgebra/` | the re-export package a user imports | — |

`crates/valgebra-core` is pure Rust. `crates/valgebra-py` is the PyO3 binding.
`python/valgebra/` re-exports the compiled extension and adds no logic.

## The direction

`valgebra-py` depends on `valgebra-core`. Nothing depends on `valgebra-py`, which
is a `cdylib` and has no downstream Rust consumer.

Within the binding, `crates/valgebra-py/src/validator.rs` and
`crates/valgebra-py/src/build.rs` depend on each other, and that pair is the only
cycle in either crate. It is a dependency rather than a placement: the frontend
needs the validator because an already compiled validator is itself a schema
description, and the validator needs the frontend to compile one. Every other
shared type lives in a leaf module its users import directly —
`crates/valgebra-py/src/exception.rs`, `crates/valgebra-py/src/check/ctx.rs` —
with the aggregator re-exporting, so no call site spells a longer path.

The cycle costs nothing the build can see. rustc's compilation unit is the crate,
and the shipped artifact is one `.so` statically linking `valgebra-core`, so at
the granularity the build has this is one node.

## Two invariants the compiler holds

**`#![forbid(unsafe_code)]` in both crates.** Zero `unsafe` blocks, and a future
one fails the build. The security policy's no-unsafe guarantee is a fact of the
compiler rather than a sentence in a document.

**`valgebra-core` has no `pyo3` dependency.** The core owns the IR, its
denotation and the decision procedure, and cannot see a Python object. That is
what lets the decision procedure be tested against a Rust value model with no
interpreter in the loop, and it is checkable in one line of
`crates/valgebra-core/Cargo.toml`.

The second invariant is why the core asks the binding about anything it cannot
decide alone, through the `LeafRelations` trait: a class hierarchy and a concrete
value are Python facts, and the core takes them as answers rather than reaching
for them. [02-decision.md](02-decision.md) owns that boundary.

## How a value flows

```
  a typing annotation
        |  build.rs                        crates/valgebra-py/src/build.rs
        v
  Schema + a constants pool + definitions  crates/valgebra-core/src/ir.rs
        |  simplify (optional)             crates/valgebra-core/src/simplify.rs
        v
  a compiled Validator                     crates/valgebra-py/src/validator.rs
        |  member()                        crates/valgebra-py/src/check/walk.rs
        v
  a bool, or a Violation list              crates/valgebra-core/src/violation.rs
```

Compilation happens once. A validator never changes after it is built and is safe
to share across threads.

The constants pool is one `Vec<Py<PyAny>>` holding four kinds of object — a
literal's constant, a class, a comparison operand, a user predicate — addressed
by four distinct index types. [06-type-design.md](06-type-design.md) owns why.

## What the core does not contain

No Python. No coercion: validation is a membership test on the object the caller
already holds, and no value is copied or converted on the accept path. No I/O
except the JSON parse, which `jiter` owns and which validates in place without
materialising Python objects first.

## Where to look next

A change to what a schema *means* is [01-schema-ir.md](01-schema-ir.md) and
[06-type-design.md](06-type-design.md). A change to what is *decidable* is
[02-decision.md](02-decision.md). A change to what a value *matches* is
[04-walk.md](04-walk.md), and it is the file where soundness is decided.
