# The decision procedure

Two representations answer the three relations, and the order between them is a
cost measurement rather than a preference.

`crates/valgebra-core/src/descr/` is the **definition**. A schema denotes a set;
each kind carries a representation closed under union, intersection and
complement; and `a <= b` is `a & ~b` admitting no value. Everything it can hold,
it decides.

`crates/valgebra-core/src/decision.rs` is the **fast path**. It recurses over the
schema tree, matching shapes and applying rules. It answers first, and the
descriptor answers where it declines -- where it *declines*, not where it says
no: a schema the rules prove inhabited is not lowered, because a sound second
reading cannot overturn a proof and lowering one determinises automata and takes
products to fail to.

That ordering is not a second opinion. Both are asked the same question and
both answer in three values -- proved, refuted, or neither -- so the descriptor
is asked only where the rules reach the third, and the pair gives whichever of
them decides. The rules are an optimisation of a relation the descriptor defines
-- [01-schema-ir.md](01-schema-ir.md) records that decision under "Which
representation decides" -- and the reason they are worth having is measured:
building a set representation costs about two orders of magnitude more than a
rule that already answers, which is what a relation the rules *refute* stops
paying.

**A refutation stands on a value.** The descriptor's is direct: it proves the
difference `a & ~b` holds one. A rule's is a mismatch of shapes -- two arities
that cannot align, a key one side requires and the other does not declare --
and the value it names is implicit, *some* value of the subject shaped the way
the subject says. A subject with no value names none, and the empty set is
below every set including the shape it can never take, so a rule's refutation is
read against the subject's own emptiness before it is believed: proved inhabited
it refutes, proved empty it establishes the opposite, and undecided it decides
nothing and the descriptor is asked.

**The reading is taken where the refutation is made**, at every level of a
query and not once at its top. A composition carries a part's refutation up,
and the part is a subject of its own: a list of an element with no value is the
empty list, which is below a list of anything, and the mismatch its element
reports is about no value. The subject one level up says nothing about that --
it has the empty list, whatever its element admits. The shapes whose own form
names a value are read without a descent, which is what keeps the cost of
reading at every level near the cost of reading once: a scalar atom, a
container that admits an empty one, a keyed map that requires no key, a union
with such a member. `laws.rs` holds that shallow reading to the value corpus,
and `decision/tests.rs` pins its edge.

Both halves are gated. `scripts/perf_gate.py --decision` measures the relations
that hold; `--decision-refute` measures the ones a rule refutes, which is the
path the first workload never walks and where work therefore costs nothing any
budget holds. `Validator.relation_to` is the boundary's reading of the same
three values, and `tests/test_completeness_probe.py` holds a reported refutation
to naming a value its universe contains.

`crates/valgebra-core/src/simplify.rs` is the pass that used to normalise a
term afterwards. It is **deprecated** and goes in the next minor version: a
schema is built in the lattice normal form, so the reduction it promises is the
schema a caller already holds, and the folds it adds beyond the laws are
decisions the three relations make better.

## Why every rule stays

A rule earns its place by reaching a shape the descriptor refuses, or by
answering a common one far more cheaply. The first is the load-bearing half, and
the refusals are wide: the descriptor holds no cycle, so **every recursive
schema** is the rules' alone; it holds no predicate and no class whose metaclass
answers `isinstance`; and it refuses any schema past the three bounds a build is
held to.

That is not an argument, it is a measurement. Deleting the float arm of the
region partition takes out two lattice laws and a membership law. Deleting the
contravariance arm -- `~A <= ~B` is `B <= A` -- takes out *reflexivity*, because
a complemented recursive schema has no other route. Deleting the sequence arm of
the emptiness fold takes out the rule that a sequence is empty when a prefix
element is. And a full mutation sweep of the file leaves **one** survivor and one
timeout, both accepted in `scripts/mutation_baseline.json` with the argument for
each: 281 mutants of 311 are caught, and no arm is dead weight.

A rule the descriptor also decides is invisible through the public relation, so a
test about a rule's *scope* has to ask the rule. `decision.rs`'s own test module
carries `by_the_rules` and `empty_by_the_rules` for that, and every test about
what a rule reaches goes through them.

## Sound, not complete, and the direction matters

Every relation is **sound**: a `true` is a proof, a `false` means "not proven".
`is_subtype_of(a, b)` returning true asserts set inclusion; returning false
asserts nothing beyond the procedure's reach.

That asymmetry is deliberate and it is what makes the procedure safe to extend.
Adding a rule can only move answers from false to true, so no extension can make
an accept wrong. The published boundary is `docs/15-decidability.md`; the enumerated

A **refutation** is a claim too, and it stands on a value: the subject has one
that the other schema rejects. The query reads the subject's emptiness once,
at the top (`witnessed`), and believes a refutation only where the subject is
proven inhabited. A rule that refutes on a *part* of the subject reads the
part's own verdict the same way, in three values: a sequence whose repeated
element the rules cannot read is not refuted against a fixed length, because
the element may admit no value and the sequence is then its prefix alone
(`linear_subtype`). A bool would make *unknown* count as *inhabited*; the
verdict is what keeps the claim a claim.
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

Two partitions of the value universe live in `decision.rs`, and they are not
rivals: `Kind` is the eleven-part one the descriptor's components are indexed
by, and `Region` is a seven-bit *summary* derived from it (`Kind::region`) --
six scalar bits and one for everything else. The rules reason in the summary
because that is all a bitset needs to decide a scalar; the descriptor reasons in
the partition because a component per kind is what closes each under complement.
A change to one is a change to the other, in that direction only.

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
containing one is opaque too. Literals, instances, refinements,
content-bearing containers and references are all opaque.

## What the core cannot decide alone

A class hierarchy and a concrete value are Python facts, and `valgebra-core`
cannot see Python ([00-architecture.md](00-architecture.md)). The `LeafRelations`
trait is how it asks:

- `leaf_subtype` — is this literal a member of that set, is this class a subclass
  of that one;
- `literal_sets_disjoint` — do these two *sets* of constants share a value. The
  same relation `literals_disjoint` answers for one pair, asked of every pair at
  once: the core compares a union with a union member by member, which is
  quadratic in the oracle, and two twenty-thousand-member literal unions were
  four hundred million calls and six seconds. An implementor that can hash its
  constants answers in one pass; the default declines and the member walk
  stands;
- `compare` — order two pooled refinement bounds;
- `no_int_between` — does the open interval between two bounds admit no integer.

`NoLeafRelations` is the core's default and decides nothing. **Its `None` and a
`Some(false)` are the same conservative verdict** at both call sites —
`leaf_subtype(..).unwrap_or(false)` and `no_int_between(..) == Some(true)` — which
is what the defaults are for, and which is why a mutation replacing either with
`Some(false)` cannot be killed by any test.

## When a class is the union of the values it lists

`leaf_subtype` answers one question the class hierarchy alone cannot: whether an
enumeration is the union of its members. It is, when **every instance of the
class is one of the values `list(cls)` yields** — and that is four conditions,
not the three first written down:

| condition | the value that stands against dropping it |
|---|---|
| it is an `Enum` | — |
| it is **not** a `Flag` | `P.A \| P.B`, an instance of `P` that `list(P)` never yields |
| it has **at least one member** | a subclass's member: a memberless enum is still subclassable |
| its members compare by identity | an `IntEnum` member *is* the integer, so two of them are not two values |

The first three were read as "the members are fixed at class creation, a class
with any cannot be subclassed, and every instance is one of them". The middle
clause is true and the outer two are not: a flag's `|` operator makes instances
after the fact, and a class with *no* members is exactly the case the
no-subclassing rule does not cover. Both were decided as unions, so
`Validator(Flag) <= Literal[*Flag]` and `Validator(EmptyEnum) <= nothing` were
`True` with a value refuting each.

A class failing any condition stays the `isinstance` atom it already was. That
costs completeness and nothing else: membership is unaffected, and the relations
simply stay undecided. `tests/test_enums.py` holds each refused kind to the value
that would refute the union reading, so a kind read as its members again fails
there.

## The two routes to a subtype answer, and where they part

`a <= b` is `a & ~b` admitting no value, and there are two procedures that reach
it: the structural rules, and the emptiness of the difference through the
descriptor. Both are sound and neither subsumes the other, so `is_subtype_of`
asks the rules and then the descriptor, and `is_empty` asks its own pair.

They do not always agree, and the disagreement is *incompleteness* rather than a
wrong answer: `a.is_subtype_of(b)` can be `True` while
`intersection(a, complement(b)).is_empty()` is `False`. Every case measured is a
**fixpoint**. The rules carry a coinductive hypothesis -- a goal already being
proven on the path is assumed -- which decides a recursive schema against itself
and against its own body; the emptiness route unfolds a reference once and asks
the descriptor, which settles what kinds a body admits and not what a fixpoint
equals.

`tests/test_completeness_probe.py` measures the gap over the survey corpus and
holds it at **three pairs**, each carrying a fixpoint. That is a ratchet rather
than a target: a change making either route less complete widens it, and a
disagreement over anything but a fixpoint is a new fact that fails on arrival.
The same test checks the emptiness route for soundness against the value
universe, which the survey beside it does not walk.

## Four bounds, each measured

The rules bound their own work with a step counter, and a build is held to three
more: the schema nodes it will read, the nesting it will descend, and the units
of multiplying work it may spend. They are named together as
`descr::lower::Bounds`, they are rows in the table of every bound in the tree
([00-architecture.md](00-architecture.md)), and
`crates/valgebra-core/benches/core.rs` carries the workload each number was set
from -- a bound whose figure lives only in a comment
cannot be re-derived on another machine, and cannot fail when the shape it guards
against changes.

What the numbers say. The four relations the descriptor decides and the rules do
not build in 49 to 188 microseconds, which is the room the bounds must leave. A
record nested behind a list grows 1.7 microseconds, 198 microseconds, 1.5
milliseconds, 7.2 milliseconds at depths 0, 2, 4 and 6 -- nesting is the
exponential, and the nesting bound is what catches it. A union of four such
records minus a union of its siblings costs 9.0 milliseconds unheld and 1.35
microseconds held, from the nesting bound alone. On the shapes reachable today
that bound refuses first, so the work allowance is the one that remains for a
schema that is shallow and wide.

Held this way, the whole widening costs the decision path eleven percent. Asked
*first* instead, it cost seventeen hundred times the workload's budget, which is
why the rules answer first.

## The budget, and what exhausting it means

Subtyping distributes over unions and intersections; emptiness recurses the
structural fragment. A deeply nested Boolean combination can therefore demand
work exponential in its depth unless a goal already decided can be recognised
when it comes back — and recognising one cheaply needs *identity*, not structure.
The IR gives that: two subtrees built alike are one allocation
([01-schema-ir.md](01-schema-ir.md)), so a goal is named by a pair of pointers
and a memo over goals has the cheap key it wants. What a memo does not yet have
is its soundness argument, and that is not about keys: an answer reached under
the coinductive hypothesis the trail carries is not an answer without it, so
what may be remembered is a goal decided with an empty trail and nothing else.
The procedure bounds its own work with a counter until one is written, threaded
through a whole top-level query so the two directions of an equivalence share it
and the bound cannot be spent twice or escaped through a side door.

One shape needs neither the memo nor the counter. A union of nothing but
literals denotes a **finite set of values**, and inclusion between two finite
sets is membership of every value of one in the other -- a walk of the two
tables, exact in both directions, where distributing one against the other is a
decision step per pair. The rule reads a table only in the canonical order its
constructor leaves it in, and its refutation is the oracle's: two constants at
two pool positions are two values only where the bindings can compare them.

And one repetition is removed rather than bounded. A record whose fields carry
one schema, or a tuple whose positions do, asks the *same goal* once per field
or position -- in one loop, in order, so the repeat is the entry before this
one. Each of those two rules remembers the last pair it was asked and the
answer it gave, which is a memo of one entry where a table over the whole query
would be. That is not an accident of size: a table was measured against a
workload whose goals repeat and two whose goals do not, in four designs, and
every one of them charged the queries with nothing to remember more than it
saved the queries with something.

Part of the argument the counter stands in for is already in the code: the trail
holds each `(subject, supertype)` pair it is deciding, and a pair that comes back
returns against the hypothesis rather than unfolding again, which is what makes a
recursive schema decide at all. What it does not cover is the goals a rule
*builds*: deciding a fixed-length sequence against a union expands the branches
and constructs a sequence per expansion, and those are not subterms of anything
the query was handed, so the set the trail draws from is not obviously finite.
That is the gap between this counter and a theorem, and it is why the bound is
carried as debt rather than as a limit ([00-architecture.md](00-architecture.md)
groups the kinds).

The honest thing to say about the ceiling meanwhile is what it is measured to
reach. The ceiling is a million steps. Records nested six deep with
union-of-literal fields decide in 1,834; a union of two hundred literals against
one of three hundred, in 403; a fixed tuple of unions against the union of all
its expansions, at the width where the right-hand side is a thousand nodes, in
7,377. The step count grows with the size of the query rather than exponentially
in its depth on every shape that has been probed, and the build limits cap that
size, so nothing yet constructed comes within two orders of magnitude of the
ceiling.

Exhaustion returns the conservative answer. That is sound by the contract above,
and the numbers say it is a ceiling no real annotation reaches — only an
adversarial one, if one exists, would.

**Two tests exist to prove that bound and they leave the mutation sweep**, because
a mutation that removes the bound makes them run without end. They are marked in
their own source and the ledger holds the marks to the sweep's skip list;
[07-tooling-ci.md](07-tooling-ci.md) owns that rule.

## Construction is not a decision procedure

The constructors apply the lattice laws -- flattening, identities,
deduplication, the two complement laws -- so no schema reaches a rule in a shape
those describe. They stop there, and the line is the cost: a law that needs a
*containment* to see is a decision, and running one wherever a schema is built
is the price this design refuses everywhere else.

Absorption is the law on the far side of that line. `A | (A & B)` is `A` exactly
when `A` contains `A & B`, so the two are one set and two terms: `is_equivalent`
proves it and `==` does not. `is_empty` on a structural schema is true in places
construction leaves standing, for the same reason.

(`simplify` was a pass that folded a little more than construction does. It is
deprecated and goes in the next minor version; what it still folds beyond the
laws is listed in `docs/04-algebra.md`.)

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
unless the other side was a container, a record or an instance.

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

Read `docs/15-decidability.md` for the published fragment and
`tests/test_completeness_ledger.py` for the relations it declines, each a strict
expected failure. None of them is a soundness question. Six shapes account for
almost all of them, and naming the shapes is more useful than naming the cases:

- **emptiness never asks an inclusion.** `is_empty` decides the regions, the
  complement law and the bounds directly and never asks whether one member of a
  meet is below another, so `list[bool] & ~list[int]` is not decided empty
  although the inclusion under it is decided;
- **a container meet is not met componentwise**, so a meet that denotes the empty
  container is not recognised as one;
- **a kind is a region bit, not a set of its values**, so a finite kind is not the
  union of its members and a negated literal has nowhere to go inside one;
- **a bound is compared against a bound**, so a base that is not itself a
  refinement reaches one only through the value oracle, and a length bound is
  opaque to the shape it bounds;
- **a map's domain is its field list as written**, and a key type is matched
  against the `Str` and `Anything` atoms rather than asked whether it admits a
  name;
- **a lossy arm is reached before a lossless one**: a union on the right
  distributes before a reference is unfolded or a refinement drops to its base.

The first four are the same diagnosis in four places — no kind has a
representation closed under complement, so a negated atom of that kind falls
through — and that is the representation the theory names
([10-theory.md](10-theory.md)). The last is an ordering, and the fifth is
ordinary work.
