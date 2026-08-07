# The decision procedure

`crates/valgebra-core/src/decision.rs` owns emptiness, subtyping, equivalence and
disjointness. `crates/valgebra-core/src/simplify.rs` owns the
membership-preserving normalisation that uses them.

## Sound, not complete, and the direction matters

Every relation is **sound**: a `true` is a proof, a `false` means "not proven".
`is_subtype_of(a, b)` returning true asserts set inclusion; returning false
asserts nothing beyond the procedure's reach.

That asymmetry is deliberate and it is what makes the procedure safe to extend.
Adding a rule can only move answers from false to true, so no extension can make
an accept wrong. The published boundary is `docs/decidability.md`; the enumerated
relations the procedure is required to *decide* are
`tests/test_completeness_ledger.py`, which fails in both directions —
a relation that regresses to conservatism fails, and a closed hole whose ledger
entry survives fails too.

The ledger only holds relations someone thought to write down, so it is one half.
`tests/test_completeness_probe.py` is the other: it searches for pairs answered
`false` that no value in a wide universe refutes, and fails when one appears that
is not written down with a reason. An enumerated list can only confirm the rules
it was built from; a search can report a rule nobody wrote.

## The scalar fragment is exact, through a region partition

The value universe is cut into mutually disjoint regions, and a Boolean
combination of scalar atoms denotes a set the lattice operations compute
exactly. On that fragment emptiness and subtyping are decided completely:
subtyping **is** set inclusion between two region sets, and nothing else.

`Region` carries the operations — `union`, `intersect`, `complement`,
`is_empty`, `subset_of` — rather than leaving `|`, `&` and `!` at the call sites.
That is a correctness decision as much as a clarity one: the one-character
difference between a right and a wrong operator sat inside folds no test could
distinguish it in, and concentrating them into five methods put each in a place a
test reaches. [06-type-design.md](06-type-design.md) records what that moved.

Six scalar regions and one non-scalar remainder partition the universe. The
remainder is what keeps emptiness sound: the meet of all six scalar complements
is the non-empty non-scalar region, not the empty set. A test asserts the six are
non-empty and pairwise disjoint, because either half alone is satisfied by a
region that collapsed to nothing.

Off that fragment a schema's region is `None` — opaque — and every combination
containing one is opaque too. The gradual `Dynamic`, literals, instances,
refinements, content-bearing containers and references are all opaque.

## What the core cannot decide alone

A class hierarchy and a concrete value are Python facts, and `valgebra-core`
cannot see Python ([00-architecture.md](00-architecture.md)). The `LeafRelations`
trait is how it asks:

- `leaf_subtype` — is this literal a member of that set, is this class a subclass
  of that one;
- `compare` — order two pooled refinement bounds;
- `no_int_between` — does the open interval between two bounds admit no integer.

`NoLeafRelations` is the core's default and decides nothing. **Its `None` and a
`Some(false)` are the same conservative verdict** at both call sites —
`leaf_subtype(..).unwrap_or(false)` and `no_int_between(..) == Some(true)` — which
is what the defaults are for, and which is why a mutation replacing either with
`Some(false)` cannot be killed by any test.

## The budget, and what exhausting it means

Subtyping distributes over unions and intersections; emptiness recurses the
structural fragment. Without interning to share equal subtrees there is no cheap
memo, so a deeply nested Boolean combination can demand work exponential in its
depth. The procedure bounds its own work with a counter, threaded through a whole
top-level query so the two directions of an equivalence share it and the bound
cannot be spent twice or escaped through a side door.

Exhaustion returns the conservative answer. That is sound by the contract above,
and the ceiling is far above any schema a real annotation produces — only an
adversarial one reaches it.

**Two tests exist to prove that bound and they leave the mutation sweep**, because
a mutation that removes the bound makes them run without end. They are marked in
their own source and the ledger holds the marks to the sweep's skip list;
[07-tooling-ci.md](07-tooling-ci.md) owns that rule.

## The simplifier is not a decision procedure

`simplify` applies the lattice laws: flattening, identities, De Morgan,
deduplication, and the scalar-region collapse. It preserves membership — the set
does not change — and it is checked against a value oracle rather than against
itself.

It is not complete and cannot be. It runs the region check and the
complementary-pair check, so it collapses what those decide; it does not run the
full procedure, which is why `is_empty` on a structural schema can be true where
`simplify` leaves the schema standing. That split is deliberate: `simplify` is on
the compile path and the decision procedure is not, so a rule that costs is
allowed in one and not the other.

## The bounds are decided by emptiness, not by the atom

`∅ ⊆ B` for every `B`, and `A ⊆ U` for every `A`. Both are decided by asking
whether a schema **denotes** the empty set — not by matching the `Nothing` or
`Anything` atom — so a record with an uninhabited required field, a cancelling
intersection, and a union that covers the universe are all recognised.

Stating a bound over the atom is a rule that confirms itself: the pattern matches
only the shape the rule is written for, so nothing else is ever the subject. That
is how this held a gap for months while every instrument stayed green — the
fuzzer asserted `Nothing ≤ b` with the atom hardcoded on the left, the property
suite examined only the consequences of a `true`, and the region check upstream
decides a scalar right-hand side correctly, so the difference was invisible
unless the other side was a container, a record, an instance or the gradual atom.

Both laws are now stated over the property, in the fuzz targets and in the core
property suite, and the enumerated cases are in the completeness ledger.

## Two more places a shape stood in for the question

The bounds were not the only ones, and the same search found the rest.

**A complement on the right had no arm at all.** `A ⊆ ¬B` was decided only when
`A` was itself a complement, by contraposition. There is no shape on the right to
recurse into, so the structural arms had nothing to say and the answer fell
through to `false` — a container was never seen inside the complement of a
scalar. It is one question: `A ⊆ ¬B` exactly when `A ∩ B = ∅`, which emptiness
already answers through kind disjointness and the scalar regions.

**The closed record had a branch of its own.** The keyed-map rule dispatched on
`defaults.is_empty()` into a rule that required every field to meet a like-named
field of the supertype, so a field the supertype covers through a catch-all read
as undecided — although the general branch beside it already decided exactly
that. A second branch for the pure mapping computed what the general one
computes. Both are gone; one rule serves every shape a keyed map takes.

The pattern in all four: a branch keyed on a shape answers a narrower question
than the general rule beside it, and reads as a deliberate special case because
it has a comment. Prefer one rule that asks the question.

## The limit

Read `docs/decidability.md` for the published fragment. What it does not decide,
all of it honestly conservative and none of it a soundness question:

- a mixed keyed map where the supertype declares a **required** field the subtype
  lacks in the general case;
- the split of a language across a union of branches;
- whether a catch-all keyed by literals covers a declared field name — the key is
  matched against the `Str` and `Anything` atoms rather than asked whether it
  admits the name, and the name is a bare `String` the core cannot pool;
- a literal against the complement of a scalar, which wants the value oracle the
  core already has but does not ask here.

The first two would be decided by the interning and automata engine the theory
names ([09-theory.md](09-theory.md)). The last two are ordinary work: each has a
route, and each is on the probe's ledger so it cannot quietly become permanent.
