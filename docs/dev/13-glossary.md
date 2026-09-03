# Glossary

The words this set uses without stopping to define them. Every entry names the
file or symbol that owns the thing, so a rename dates the entry.

## The product

| Term | Means |
|---|---|
| **denotation** | the set of Python values a schema admits. Every node in `crates/valgebra-core/src/ir.rs` has one written down, and validation is membership in the root node's set |
| **membership** | the question the walk answers: is this value in that set. Never a conversion — the value is not copied or coerced |
| **check-only** | valgebra's semantics: it decides membership on the object the caller already holds. A validator that returns a *new* value is doing something else |
| **the algebra** | union, intersection, complement, refinement and fixpoints over value sets, closed under all of them |
| **atom** | a node with no schema inside it: the scalars, `Literal`, `Instance`, the lattice bounds |
| **the closure** | the sets reachable by combining the atoms with union, intersection and complement. A proposed node either denotes a set already in it — and is redundant — or extends the algebra; [01-schema-ir.md](01-schema-ir.md) owns the test |
| **carrier** | the Python class holding a structural node's shape: `list` for a `Seq`, `dict` for a `KeyedMap`, the `class_index` for an `Attrs`. Fixed three different ways, which is why widening one is a different size of change per node |
| **minimality** | the property that makes "the algebra" a claim: the smallest node set whose closure is consistent and complete for the domain. A node is admitted because the domain is unreachable without it, never because it is convenient |
| **the lattice bounds** | `Anything` (top, every value) and `Nothing` (bottom, no value) |
| **the gradual atom** | `Dynamic`, which the user spells `typing.Any`. It admits every value at runtime and is a *distinct* atom to the algebra, so the simplifier does not rewrite it by the lattice laws |
| **region** | one part of the mutually disjoint partition of the value universe that `Region` computes over. Six scalar regions plus one non-scalar remainder |
| **pool** | the validator's `Vec<Py<PyAny>>`, holding four kinds of object addressed by four index types ([06-type-design.md](06-type-design.md)) |
| **definition** | an entry in the validator's definitions table; the target of a `Ref` back edge, produced by `recursive` |
| **contractive** | a recursive definition whose every self-reference sits under a structural constructor. `Schema::occurs_unguarded` decides it |
| **the walk** | `member` in `crates/valgebra-py/src/check/walk.rs`. There is one, and it serves both input paths and all three modes |
| **violation** | the structured failure: a stable code, a path, an expected label and a value summary ([05-errors.md](05-errors.md)) |

## The decision procedure

| Term | Means |
|---|---|
| **sound** | a `true` is a proof. Every relation here is sound, and a `false` means "not proven" rather than "false" |
| **complete** | every true relation is decided. valgebra is complete on a published fragment and conservative elsewhere; `docs/15-decidability.md` states the line |
| **conservative** | the answer a procedure gives when it cannot decide: the one that claims less |
| **opaque** | a schema whose region is unknown, so the scalar rules do not apply. Any combination containing one is opaque |
| **the oracle** (in the core) | `LeafRelations`, the trait through which the decision procedure asks the bindings about a class or a value |
| **the budget** (in the core) | `DECISION_BUDGET`, the work ceiling one top-level query may spend before returning the conservative answer |
| **the ledger** (of completeness) | `tests/test_completeness_ledger.py`, enumerated relations the procedure must *decide*, failing in both directions |

## Verification

| Term | Means |
|---|---|
| **gate** | a step that **asserts** and exits non-zero when the assertion breaks. A step that only builds, measures or records is not one |
| **lane** | one independently driven run: a CI job, or one target inside a step that drives several |
| **the oracle** (in testing) | a judge of a claim that does not go through the code under test. The denotation predicate, pydantic-core, jsonschema |
| **survivor** | a mutation of the source the tests did not notice. A signal about the tests, never about the mutation |
| **equivalent mutant** | a mutation that provably cannot change any result, so no test can kill it. Excluded with its argument, never counted as a gap |
| **rig fault** | a run that produced no verdict — a timeout, an empty corpus, a mutation whose experiment cannot finish. Neither a pass nor a failure, and reported as itself |
| **ratchet** | a committed floor that may only move one way. The mutation baselines are ratchets; a budget is not |
| **budget** (of instructions) | a committed two-sided band a measurement is held to, in `scripts/perf_budget.json` |
| **ledger** (of a list) | an enumerated list held to the tree in both directions, so an entry that stops being true fails and a subject with no entry fails too. Each carries a `LEDGER:` marker; there are twelve ([08-testing.md](08-testing.md)) |
| **excused** | named on a ledger with the reason it is a hole. An excuse expires in its own direction: one that stops being true fails |
| **suspected gap** | a relation the procedure answers `False` that no value in the probe's universe refutes, so it looks true and was not seen. Suspected because the universe is finite ([08-testing.md](08-testing.md)) |
| **probe** | an instrument that *searches* for a defect rather than checking an enumerated list of them. `tests/test_completeness_probe.py` is the one here, and it exists because a list can only confirm the rules it was built from |
| **witness** | a value that settles a relation by example: one inside the subtype and outside the supertype disproves inclusion. A `False` with no witness is the probe's subject |
| **detached surface** | a `Cargo.toml` outside the root workspace, which no workspace-wide command reaches ([07-tooling-ci.md](07-tooling-ci.md)) |

## Four collisions, and both senses are live

Say which one you mean.

| Term | One sense | The other |
|---|---|---|
| **gate** | a CI step that asserts | the local build-health command set, which is a preview of the merge gate rather than a single check |
| **oracle** | an independent judge in a test | `LeafRelations`, the trait the decision procedure asks about a class or a value |
| **budget** | the committed instruction count a workload is held to | `DECISION_BUDGET`, the work ceiling one decision query may spend |
| **ledger** | a list held to the tree in both directions | the completeness ledger, which is that shape but about *relations* rather than about files |

## Words this set avoids

**"Fast"** without a number and the command that produced it. **"Should"** where
a gate decides. **"Simply"**, which is never true of the thing it precedes. And
**"validate"** in the sense of converting — valgebra checks; a library that
returns a new value is doing a different thing, and the distinction is the
product.
