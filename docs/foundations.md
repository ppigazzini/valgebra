---
description: The denotational frame and the sourced set-theoretic and lattice theory.
---

# Foundations

This page records the theory valgebra rests on: what a schema *means*, why the
combinators form a real Boolean algebra, and where the algebra decides
relationships versus where it stays deliberately conservative. It is the
reference behind the claims the rest of the docs make — "a closed, lawful
lattice", "subtyping is set inclusion", "a law-justified simplifier" — so each
is backed rather than asserted. The [soundness argument](soundness.md) takes the
next step: why an accept is never wrong, node by node.

## Schemas denote sets; validation is membership

A schema denotes a **set of Python values**. Validating a value is deciding
whether it is a member of that set — nothing is copied or coerced. This is the
*denotational* view: the meaning of a schema is its value-set `[[s]]`, and every
other relationship is defined from it.

- **Subtyping is set inclusion.** `s` is a subtype of `t` exactly when
  `[[s]] ⊆ [[t]]`.
- **Equivalence is mutual inclusion.** `s` and `t` are equivalent when
  `[[s]] = [[t]]` — they accept the same values, whatever their syntax.

Because meaning is a set, the connectives are the set operations, and they obey
the set-algebra laws by construction rather than by convention.

## A Boolean algebra of schemas

`union`, `intersection`, and `complement` are set union, intersection, and
complement; `anything` is the top (every value) and `nothing` is the bottom (no
value). Schemas under these operations form a **Boolean lattice**: a bounded,
distributive, complemented lattice. Every Boolean-algebra law therefore holds —
commutativity, associativity, idempotence, absorption, identities,
distributivity, De Morgan, and double negation — and valgebra property-tests each
against the membership relation rather than asserting it (see the
[algebra guide](algebra.md)).

`simplify` rewrites a schema by these laws while admitting **exactly the same
values**. That soundness — simplification never changes the value-set — is the
single invariant the simplifier is held to.

## Semantic (set-theoretic) subtyping

Treating types as sets of values, with full union, intersection, and negation
and subtyping as inclusion, is **semantic subtyping**, developed by Frisch,
Castagna, and Benzaken. valgebra is a runtime membership checker built on that
model rather than a static type system, but it inherits the model's payoff: the
combinators are not ad-hoc primitives, they are the Boolean operations on
value-sets, and refinements like "an int that is not a bool" are simply
`intersection(int, complement(bool))`.

The same line of work models the structural forms valgebra uses:

- **Sequences as regular-expression types.** A list or tuple schema is a regular
  expression over element schemas — the regular-tree-type approach from XDuce and
  CDuce. One node expresses fixed tuples, variadic tuples, and prefix-plus-tail
  lists uniformly.
- **Maps as keyed-default functions.** A dict, record, or map is named fields
  plus key-schema-keyed default clauses — the set-theoretic model of records and
  maps as quasi-constant functions.

## Gradual `Any` versus the lattice top

`Any` (the gradual dynamic type) and `anything` (the lattice top) both admit
every value at runtime, but they are different objects in the algebra. `anything`
obeys the laws — `complement(anything)` is `nothing`. `Any` is the **gradual**
dynamic type: an atom the simplifier never rewrites, so `intersection(Any, s)` is
not collapsed and "deliberately unchecked" stays distinct from "checked: all
values admitted". This follows the treatment of the gradual `?` under union and
intersection types, where the dynamic type is an interval, not the top.

## What the algebra decides, and the conservative frontier

Deciding whether two arbitrary set-theoretic types are equal — equivalently,
whether a type is empty — is decidable but **EXPTIME-complete**. valgebra does
not need that decision to validate: membership is answered directly by the walk,
not by reducing the schema. So the simplifier implements the **soundly decidable
fragment** and is honest about the rest:

- **Folded by the simplifier.** The complement laws (`X ∩ ¬X = ⊥`,
  `X ∪ ¬X = ⊤`) for any `X` except the gradual `Any`, and disjointness of the
  scalar fragment. So `intersection(int, complement(int)).simplify()` is
  `nothing` and `intersection(int, str).simplify()` is `nothing`. It never treats
  `Any` as the top, so a deliberately-unchecked schema is preserved.
- **Decided by the comparison operators.** `is_subtype_of`, `is_equivalent`, and
  `is_empty` decide a wider fragment than the simplifier folds — class and literal
  inclusion, refinements (including the emptiness of contradictory bounds like
  `Ge(10) & Le(0)`), sequences, sets, records and mappings, and recursion at its
  greatest fixpoint. The [decidability boundary](decidability.md) lists exactly
  what is decided and what stays conservative.
- **Conservative.** A predicate refinement is opaque, and a narrow decidable tail
  and the runtime-undecidable constructs remain (the boundary records them). Every
  answer is sound: `is_empty` never reports a non-empty schema as empty, and a
  subtype is never claimed unless it provably holds.

The decision realizes the set-theoretic emptiness test directly rather than via a
full type-automaton construction; that construction would only widen the few
conservative cases the boundary records, never change a membership decision.

## References

The essential reading, in the order it maps onto valgebra:

1. **Frisch, Castagna & Benzaken — "Semantic Subtyping: Dealing
   Set-Theoretically with Function, Union, Intersection, and Negation Types",
   *JACM* 55(4), 2008.** [doi:10.1145/1391289.1391293](https://doi.org/10.1145/1391289.1391293).
   The foundation: types as sets, subtyping as inclusion, full Boolean
   connectives.
2. **Gesbert, Genevès & Layaïda — "A Logical Approach to Deciding Semantic
   Subtyping", *TOPLAS* 38(1), 2015.** The decision procedure and its
   EXPTIME-completeness — why the full emptiness decision is deferred.
3. **Hosoya, Vouillon & Pierce — "Regular Expression Types for XML", *TOPLAS*
   27(1), 2005.** Regular-tree types — the model behind sequences as one regex
   node.
4. **Castagna — "Typing Records, Maps, and Structs", *ICFP* 2023.**
   [doi:10.1145/3607838](https://doi.org/10.1145/3607838). Records and maps as
   keyed-default functions.
5. **Castagna & Lanvin — "Gradual Typing with Union and Intersection Types",
   *PACMPL* 1(ICFP), 2017**, and **Castagna, Lanvin, Petrucciani & Siek —
   "Gradual Typing: A New Perspective", *PACMPL* 3(POPL), 2019.** The gradual
   dynamic type under set-theoretic connectives — why `Any` is not the top.

A current synthesis is Castagna, "Programming with Union, Intersection, and
Negation Types", 2024 ([arXiv:2111.03354](https://arxiv.org/abs/2111.03354)).
