# Glossary

The words this set uses without stopping to define them. Every entry names the
file or symbol that owns the thing, or, for a concept no symbol owns, the page
that defines it -- so a rename dates the entry.

## The product

| Term | Means |
|---|---|
| **denotation** | the set of Python values a schema admits. Every node in `crates/valgebra-core/src/ir.rs` has one written down, and validation is membership in the root node's set |
| **membership** | the question the walk answers: is this value in that set. Never a conversion — the value is not copied or coerced. `member` in `crates/valgebra-py/src/check/walk.rs` answers it |
| **check-only** | valgebra's semantics: it decides membership on the object the caller already holds. A validator that returns a *new* value is doing something else ([04-walk.md](04-walk.md)) |
| **the algebra** | union, intersection, complement, refinement and fixpoints over value sets, closed under all of them ([01-schema-ir.md](01-schema-ir.md)) |
| **atom** | a `Schema` variant in `crates/valgebra-core/src/ir.rs` with no schema inside it: the scalars, `Literal`, `Instance`, the lattice bounds |
| **the closure** | the sets reachable by combining the atoms with union, intersection and complement. A proposed node either denotes a set already in it — and is redundant — or extends the algebra; [01-schema-ir.md](01-schema-ir.md) owns the test |
| **carrier** | the Python class holding a structural node's shape: `list` or `tuple` for a `Seq`, by its `SeqKind`; `set` or `frozenset` for a `Coll`, by its `CollKind`; `dict` for a `KeyedMap`, by its denotation. An `AttrRecord` has none — a class is a separate `Instance` atom met with the record — so the carriers are fixed three ways and widening one is a different size of change per node ([01-schema-ir.md](01-schema-ir.md)) |
| **layout** | the class whose instance layout a class carries: itself where its own `__slots__` add a slot or it is one of the builtins, else the nearest ancestor that does, else none. Python builds a class deriving from two others only where one layout extends the other, so two classes carrying layouts neither of which extends the other share no instance; a class carrying none, a plain class, is disjoint from nothing. `layout_of` in `crates/valgebra-py/src/oracle.rs` reads it, and `Class` in `crates/valgebra-core/src/descr/classes.rs` carries it |
| **minimality** | the property that makes "the algebra" a claim: the smallest node set whose closure is consistent and complete for the domain. A node is admitted because the domain is unreachable without it, never because it is convenient ([01-schema-ir.md](01-schema-ir.md); `tests/test_closure_ledger.py` holds every variant to it) |
| **the lattice bounds** | `Anything` (top, every value) and `Nothing` (bottom, no value), variants in `crates/valgebra-core/src/ir.rs`, exported as `anything` and `nothing` by `crates/valgebra-py/src/lib.rs` |
| **the spelling** | how the top was written — `typing.Any` or `anything` — carried by `Anything` as a `Spelling` (`crates/valgebra-core/src/ir.rs`), written back by `render`, and read by the sharing table in `ir/intern.rs` so it never collapses the two. It is not part of the set: two schemas differing only in it are equal, and `Spelling` compares, orders and hashes alike, so every rule keyed on those — which is every rule — sees one value |
| **region** | one part of the mutually disjoint partition of the value universe that `Region` in `crates/valgebra-core/src/kind.rs` computes over. Six scalar regions plus one non-scalar remainder |
| **clause** | one key-schema/value-schema pair of a mapping, `MapClause` in `crates/valgebra-core/src/ir.rs`. A dict literal's `{str: int}` is one; an open `TypedDict`'s reading carries a `str` clause for the keys its fields do not name |
| **pool** | the validator's `Vec<Py<PyAny>>` (`Validator::literals` in `crates/valgebra-py/src/validator.rs`), holding four kinds of object addressed by four index types ([06-type-design.md](06-type-design.md)) |
| **definition** | an entry in the validator's definitions table (`Validator::definitions` in `crates/valgebra-py/src/validator.rs`); the target of a `Ref` back edge, produced by `recursive` |
| **contractive** | a recursive definition whose every self-reference sits under a structural constructor. `Schema::occurs_unguarded_under` in `crates/valgebra-core/src/ir.rs` decides it over the whole definitions table |
| **the walk** | `member` in `crates/valgebra-py/src/check/walk.rs`, with the leaf arms in `walk/scalar.rs` and the container arms in `walk/record.rs` and `walk/sequence.rs`. There is one, and it serves both input paths and all three modes |
| **the trail** | `Trail` in `crates/valgebra-py/src/check/ctx.rs`: the `(value, definition)` pairs the walk is inside, innermost last. A level enters its pair before walking the definition and leaves it after, so a value reached from inside itself meets its own pair and is refused as cyclic |
| **violation** | the structured failure: a stable code, a path, an expected label and a value summary. `Violation` in `crates/valgebra-core/src/violation.rs` ([05-errors.md](05-errors.md)) |

## The decision procedure

| Term | Means |
|---|---|
| **sound** | a `true` is a proof. Every relation here is sound over values that answer comparisons as their builtin does, and a `false` means "not proven" rather than "false"; `docs/14-soundness.md` pins the one case a lying comparison breaks |
| **complete** | every true relation is decided. valgebra is complete on a published fragment and conservative elsewhere; `docs/15-decidability.md` states the line |
| **conservative** | the answer a procedure gives when it cannot decide: the one that claims less ([02-decision.md](02-decision.md)) |
| **opaque** | a schema whose region is unknown, so the scalar rules do not apply. Any combination containing one is opaque: `Regions::Unknown` in `crates/valgebra-core/src/kind.rs`, which absorbs both lattice operations |
| **the rules** | `crates/valgebra-core/src/decision.rs` and `decision/`, the structural recursion over the schema tree. The fast path: it answers first, and where it declines the descriptor is asked |
| **the descriptor** | `crates/valgebra-core/src/descr/`, the set representation a schema is lowered into. One `Lines` per kind and one for the values of no listed kind, each closed under union, intersection and complement, so a relation comes out of the sets rather than out of a rule about the shape |
| **kind** | one of the eleven parts of the value universe the descriptor splits it into, plus a remainder: `Kind` in `crates/valgebra-core/src/kind.rs`. Disjoint but for `bool`, whose values are also `int`s. Coarser than a region, and defined over every value rather than the scalars alone |
| **line** | one summand of a kind's normal form: a structure met with a lattice of object guards. `Lines` in `crates/valgebra-core/src/descr/lines.rs` is a set of them with a negation flag, which is what closes a kind under complement |
| **bounds** (of a build) | `descr::lower::Bounds`, the three ceilings a lowering is held to — schema nodes read, nesting descended, units of multiplying work spent. Distinct from the budget, which is the rules' own step counter. [00-architecture.md](00-architecture.md) lists every bound in the tree |
| **the oracle** (in the core) | `LeafRelations` in `crates/valgebra-core/src/oracle.rs`, the trait through which every reading asks the bindings about a class or a value -- the decision procedure, the descriptor, and the constructors that apply the lattice laws. `crates/valgebra-py/src/oracle.rs` holds the implementation that answers |
| **verdict** | `Verdict` in `crates/valgebra-core/src/verdict.rs`: what a schema's emptiness is proven to be -- empty, inhabited, or neither. The three-valued reading a refutation needs; a bool of it turns *unknown* into *inhabited* |
| **the budget** (in the core) | `DECISION_BUDGET` in `crates/valgebra-core/src/decision.rs`, the work ceiling one top-level query may spend before returning the conservative answer |
| **declined** | what a reading says when it has no answer: the rules decline and the descriptor is asked, and a descriptor that declines leaves the relation `Unknown` (`Relation` in `crates/valgebra-core/src/verdict.rs`). Never a "no" -- a decline claims nothing about the relation |
| **refuted** | a relation disproved by a value: `Relation::Fails` in `crates/valgebra-core/src/verdict.rs`, which asserts that some value of the subject lies outside the other schema. A refutation stands on a witness, so a reading that cannot name one declines instead |
| **lowering** | turning a schema into the set it denotes -- `descr::lower`, which walks the tree and builds one `Descr`. The descriptor's entry point, and the step a `Bounds` holds |
| **unfolding** | replacing a `Ref` with the definition it names, once: `Schema::unfolded` in `crates/valgebra-core/src/ir/transform.rs`. The fixpoint reading a relation over a recursive schema needs: the subject is unfolded to widen and the supertype to narrow, which is what `Polarity`, in the same file, names ([02-decision.md](02-decision.md)) |
| **goal** | one pair a decision is asked about, subject and supertype together, as the recursion carries it. The unit a trail of seen goals would memoise, and the unit `DECISION_BUDGET` charges ([02-decision.md](02-decision.md)) |
| **the readings** | the two ways a relation is answered -- the rules and the descriptor -- and, in `decision/readings.rs`, the structural cases the rules decide by reading a schema's shape rather than its set |
| **the ledger** (of completeness) | `tests/test_completeness_ledger.py`, enumerated relations the procedure must *decide*, failing in both directions |

## Verification

| Term | Means |
|---|---|
| **gate** | a step that **asserts** and exits non-zero when the assertion breaks. A step that only builds, measures or records is not one ([07-tooling-ci.md](07-tooling-ci.md)) |
| **lane** | one independently driven run: a CI job in `.github/workflows/ci.yml`, or one target inside a step that drives several ([07-tooling-ci.md](07-tooling-ci.md)) |
| **the oracle** (in testing) | a judge of a claim that does not go through the code under test. The denotation predicate, pydantic-core, jsonschema ([08-testing.md](08-testing.md)) |
| **sweep** | one cargo-mutants run over a scope -- the core, the binding's walk, or the binding under the Python suite -- whose survivors `scripts/mutation_gate.py` holds to that scope's baseline in `scripts/`; a push sweeps the core's and the walk's files a change touches, and the nightly lanes each whole scope |
| **survivor** | a mutation of the source the tests did not notice. A signal about the tests, never about the mutation; `scripts/mutation_gate.py` counts them |
| **equivalent mutant** | a mutation that provably cannot change any result, so no test can kill it. Excluded with its argument in `exclude_re` of `.cargo/mutants.toml`, never counted as a gap |
| **rig fault** | a run that produced no verdict — a timeout, an empty corpus, a mutation whose experiment cannot finish. Neither a pass nor a failure, and reported as itself ([07-tooling-ci.md](07-tooling-ci.md); `scripts/mutation_gate.py` reports a mutant's) |
| **ratchet** | a committed floor that may only move one way. The mutation baselines (`scripts/mutation_baseline.json` and its `_walk` and `_pytest` siblings) are ratchets; a budget is not |
| **budget** (of instructions) | a committed two-sided band a measurement is held to, in `scripts/perf_budget.json` |
| **ledger** (of a list) | an enumerated list held to the tree in both directions, so an entry that stops being true fails and a subject with no entry fails too. Each carries a `LEDGER:` marker, and [08-testing.md](08-testing.md) tables them |
| **excused** | named on a ledger with the reason it is a hole. An excuse expires in its own direction: one that stops being true fails ([08-testing.md](08-testing.md)) |
| **suspected gap** | a relation the procedure answers `False` that no value in the probe's universe refutes, so it looks true and was not seen. Suspected because the universe is finite ([08-testing.md](08-testing.md)) |
| **probe** | an instrument that *searches* for a defect rather than checking an enumerated list of them. `tests/test_completeness_probe.py` is the one here, and it exists because a list can only confirm the rules it was built from |
| **finite set** | a union of nothing but literals, read as the set of constants it denotes: inclusion between two is membership of every constant of one in the other, decided by search over the canonical order the constructor leaves and a remap restores (`finite_set` in `crates/valgebra-core/src/decision/literals.rs`, which `decision.rs` calls; `mapped_member_set` in `ir/transform.rs`). A list not in that order is walked as a list |
| **the field cache** | the one-entry memo in `keyed_map_subtype` (`crates/valgebra-core/src/decision/records.rs`) and `linear_subtype` (`decision/products.rs`): a record's fields or a tuple's positions carrying one schema ask one goal each, in order, and the rule remembers the last pair it was asked. A table over the whole query is refused: the commit that added the cache carries the measurement |
| **witness** | a value that settles a relation by example: one inside the subtype and outside the supertype disproves inclusion. A `False` with no witness is the probe's subject ([08-testing.md](08-testing.md)) |
| **detached surface** | a `Cargo.toml` outside the root workspace, which no workspace-wide command reaches ([07-tooling-ci.md](07-tooling-ci.md)) |

## Collisions, and both senses are live

Say which one you mean.

| Term | One sense | The other |
|---|---|---|
| **gate** | a CI step that asserts | the local build-health command set, which is a preview of the merge gate rather than a single check |
| **oracle** | an independent judge in a test | `LeafRelations`, the trait the decision procedure asks about a class or a value |
| **close** | of the algebra: the combinators' results stay inside the schemas, so they **close** into a lattice (the README's first paragraph) | `close`, the transform that refuses the key-type region no clause claims, so a record admits only the keys it declares |
| **budget** | the committed instruction count a workload is held to | `DECISION_BUDGET`, the work ceiling one decision query may spend. Not `Bounds`, which holds a *build* rather than a query |
| **ledger** | a list held to the tree in both directions | the completeness ledger, which is that shape but about *relations* rather than about files |
| **snapshot** | a recorded expected output a test compares against, held by `syrupy` under `tests/__snapshots__` | the copy a container walk takes of a value's storage before reading it, so a `__len__` that lies or a mutation mid-walk cannot change what was measured |
| **floor** | the oldest interpreter release the package supports, which `ci.yml` names and `scripts/gate.py` builds beside the caller's | a committed minimum a figure may not fall below: a coverage scope's, or a ratchet's |
| **region** | one part of the scalar partition `Region` computes | the key-type region a mapping's clauses leave unclaimed, which `open` frees and `close` refuses |
| **witness** | the value a probe looks for: one inside the subtype and outside the supertype, which disproves inclusion | the value a *refutation* stands on, which the `witnessed` guard reads against the subject's own emptiness before believing a mismatch |

## Words this set avoids

**"Fast"** without a number and the command that produced it. **"Should"** where
a gate decides. **"Simply"**, which is never true of the thing it precedes. And
**"validate"** in the sense of converting — valgebra checks; a library that
returns a new value is doing a different thing, and the distinction is the
product.
