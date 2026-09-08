# Architecture

Two Rust crates and a Python package. This page owns the split between them, the
direction their dependencies run, and the two invariants the compiler holds.

## The zones

| Zone | Owns | Page |
|---|---|---|
| `crates/valgebra-core/src/ir.rs` | the schema IR: the node set, and what each node denotes | [01-schema-ir.md](01-schema-ir.md) |
| `crates/valgebra-core/src/decision.rs` | emptiness, subtyping, equivalence, disjointness | [02-decision.md](02-decision.md) |
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
        |  checked and pruned              crates/valgebra-py/src/validator.rs
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

## Every bound, and what holds it

Nothing here is unbounded. A schema is built under limits, a walk descends under
one, a decision spends a budget, and every representation in the descriptor
refuses past a size rather than returning a set it cannot hold. They accumulated
one at a time, in eight files, and the list was nowhere -- so a reader could not
tell a measured number from a guessed one, and a new bound cost nothing to add.

The rule now is: **no bound without a gate that measures it.** Adding one means
adding a row here and a test that reaches it. The table is held to the tree in
both directions by `scripts/docs_lint.py`, values included, so a number that
moves in the source and not here fails, and a row naming a constant that is gone
fails too.

Each row says which of three kinds its bound is, because they are not the same
sort of thing and only one of them is a defect.

* **limit** -- past it this representation holds no sound answer, so the
  operation refuses rather than approximating. A set too wide is complemented
  into one too narrow, which is why rounding is never the alternative. These are
  the bounds the algebra asks for.
* **shape** -- a bound on a value or a schema a caller wrote: how deep it nests,
  how many members it has, how much of it an error message prints. A caller can
  see these and work within them.
* **debt** -- a budget on *work*, standing in for a termination argument that is
  already available. Regularity bounds the number of distinct subtyping goals, so
  a memo over shared nodes terminates by a theorem; these budgets exist because
  the nodes are not shared, and each names the work that removes it.

A bound that is `debt` carries the change that retires it in its own doc
comment. A `limit` or a `shape` carries the reason it is where it is.

| where | bound | value | kind | what it stops | what measures it |
|---|---|---|---|---|---|
| `crates/valgebra-py/src/validator.rs` | `MAX_SCHEMA_DEPTH` | `128` | shape | nesting in a constructed schema | `tests/test_adversarial_bounds.py` |
| `crates/valgebra-py/src/validator.rs` | `MAX_DEFINITIONS` | `128` | shape | recursive definitions chained in one schema | `tests/test_adversarial_bounds.py` |
| `crates/valgebra-py/src/validator.rs` | `MAX_SCHEMA_NODES` | `100_000` | shape | a schema that is shallow and exponentially wide | `tests/test_adversarial_bounds.py` |
| `crates/valgebra-py/src/build.rs` | `MAX_BUILD_DEPTH` | `crate::validator::MAX_SCHEMA_DEPTH + 1` | shape | the frontend descending past what `checked` will accept, so the schema past the bound is built and refused by name | `tests/test_adversarial_bounds.py` |
| `crates/valgebra-py/src/render.rs` | `MAX_RENDER_DEPTH` | `200` | shape | `repr` overflowing the stack on a chain of definitions | `tests/test_adversarial_bounds.py` |
| `crates/valgebra-py/src/check/ctx.rs` | `MAX_WALK_DEPTH` | `512` | shape | a walk overflowing the smallest thread stack a platform gives | its own tests, and `tests/test_adversarial_bounds.py` |
| `crates/valgebra-py/src/check/walk.rs` | `MAX_RECURSION_DEPTH` | `128` | shape | a pathologically deep *value* overflowing the stack | its own tests, and `tests/test_adversarial_bounds.py` |
| `crates/valgebra-py/src/validator.rs` | `MAX_ENUM_MEMBERS` | `512` | shape | one relation turning into a membership question per member of an enumeration | its own tests, and `tests/test_algebra_closure.py` |
| `crates/valgebra-py/src/check/walk.rs` | `CLOSEST_BRANCH_PROBE_LIMIT` | `64` | shape | the error path's second walk costing the branch count | `tests/test_union_messages.py` |
| `crates/valgebra-py/src/check/walk.rs` | `UNION_LABEL_LIMIT` | `64` | shape | a union naming a thousand labels in one `expected` | its own tests |
| `crates/valgebra-core/src/decision.rs` | `DECISION_BUDGET` | `1_000_000` | debt | one query spending unbounded work before answering conservatively | its own tests, and `tests/test_decision_adversarial.py` |
| `crates/valgebra-core/src/descr/lower.rs` | `BUDGET` | `64` | debt | the schema nodes one lowering reads | its own tests, and `crates/valgebra-core/benches/core.rs` |
| `crates/valgebra-core/src/descr/lower.rs` | `DEPTH` | `5` | debt | the nesting one lowering descends, which is the exponential | its own tests, and `crates/valgebra-core/benches/core.rs` |
| `crates/valgebra-core/src/descr/lower.rs` | `WORK` | `1024` | debt | the multiplying work one build spends before refusing | its own tests, and `crates/valgebra-core/benches/core.rs` |
| `crates/valgebra-core/src/descr/lines.rs` | `MAX_LINES` | `256` | limit | the lines one kind carries, which a meet multiplies and a complement doubles | `crates/valgebra-core/src/descr/mod.rs` tests |
| `crates/valgebra-core/src/descr/sets.rs` | `MAX_LINES` | `256` | limit | the lines a set lattice holds | its own tests |
| `crates/valgebra-core/src/descr/maps.rs` | `MAX_ATOMS` | `256` | limit | the atoms a map union holds | its own tests |
| `crates/valgebra-core/src/descr/maps.rs` | `PARTS` | `KEY_KINDS.len() + 1` | limit | nothing -- it is the key-kind partition's width, listed because it is a file-scope integer constant and the check that reads this table cannot tell the two apart | its own tests |
| `crates/valgebra-core/src/descr/records.rs` | `MAX_ATOMS` | `256` | limit | the atoms a record union holds, which a complement multiplies | its own tests |
| `crates/valgebra-core/src/descr/symbolic.rs` | `MAX_STATES` | `4096` | limit | a product of two automata multiplying past memory | its own tests |
| `crates/valgebra-core/src/descr/symbolic.rs` | `MAX_ROW` | `MAX_STATES` | limit | one row of a product growing past the alternatives a shape has | its own tests |
| `crates/valgebra-py/src/errors.rs` | `SUMMARY_CHARS` | `80` | shape | a value summary built in full and then cut, so a huge repr is paid for and thrown away | its own tests |
| `crates/valgebra-core/src/descr/symbolic.rs` | `MAX_EDGES` | `1 << 16` | limit | a table inside both dimensions and still too large: 4,096 states each with a 4,096-wide row is sixteen million edges | its own tests |
| `crates/valgebra-core/src/descr/regular.rs` | `MAX_STATES` | `4096` | limit | a pattern product doubling the exponent twice | its own tests |
| `crates/valgebra-core/src/descr/regular.rs` | `BUILD_SIZE_LIMIT` | `8 * 1024 * 1024` | limit | a pattern whose determinisation is exponential, which reaches `MAX_STATES` only after the table it refuses has been built | its own tests |
| `crates/valgebra-core/src/descr/integers.rs` | `MAX_PERIOD` | `4096` | limit | a step set holding one interval set per residue, and the period two steps share | its own tests |

Two of them are the same number for different reasons, and the difference
matters when one moves: `MAX_SCHEMA_DEPTH` is what a *caller* may build, and
`DEPTH` is what a *lowering* will descend. A schema at the first is refused by
the second, and answered by the structural rules instead.

## Where to look next

A change to what a schema *means* is [01-schema-ir.md](01-schema-ir.md) and
[06-type-design.md](06-type-design.md). A change to what is *decidable* is
[02-decision.md](02-decision.md). A change to what a value *matches* is
[04-walk.md](04-walk.md), and it is the file where soundness is decided.
