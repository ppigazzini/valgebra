# Soundness argument

This page argues, in a form a reader can check, why **an accept is never wrong**:
if valgebra reports a value valid, the value really is in the schema's set. It is
a written argument backed by adversarial tests, not a machine-checked proof; the
[honest limits](#what-this-argument-assumes) say exactly what is taken on trust.

## The claim

Write `⟦S⟧` for the set of Python values a schema `S` denotes (its
[denotation](foundations.md)). valgebra rests on three soundness statements:

1. **Membership is exact.** For every schema `S` and value `x`, the walk accepts
   `x` if and only if `x ∈ ⟦S⟧`. The "only if" half is *soundness of acceptance*
   — the property downstream code relies on; the "if" half is completeness of the
   check.
2. **Simplification preserves meaning.** `⟦simplify(S)⟧ = ⟦S⟧`.
3. **Decisions are sound.** If `is_subtype_of(A, B)` is `True` then `⟦A⟧ ⊆ ⟦B⟧`;
   if `is_empty(S)` is `True` then `⟦S⟧ = ∅`. The converses are *not* claimed —
   the decision is deliberately conservative.

There is no separate specification the implementation could disagree with: the
denotation *is* the meaning, so soundness is the statement that the Rust walk
computes `x ∈ ⟦S⟧`. The argument is therefore a node-by-node check that the
walk's accept condition is, line for line, the membership condition of `⟦S⟧`.

## Why membership is exact: structural induction

The walk recurses on the structure of `S`. Take as induction hypothesis that the
walk is exact on every strict sub-schema; then check each node. The denotation
and the walk's accept condition coincide at every one:

```text
S                accepts x  ⟺  x ∈ ⟦S⟧, by:
---------------  -----------------------------------------------------------
Anything         always                         (⟦Anything⟧ = all values)
Nothing          never                          (⟦Nothing⟧ = ∅)
Bool/Int/...     isinstance(x, T)               (the scalar region)
Literal(c)       type(x) is type(c) and x == c  (typed singleton)
Union(A_i)       some A_i accepts x             (∃: set union)
Intersection     every A_i accepts x            (∀: set intersection)
Complement(A)    A does not accept x            (¬: set complement)
Refine(B, c_j)   B accepts x and every c_j      (base ∩ constraints)
Seq(kind, r)     x is a kind whose elements     (regular language over
                 match the regex r                element denotations)
Set/FrozenSet(A) every element accepts A        (homogeneous container)
KeyedMap(f, d)   fields present-and-match, and   (named fields ∩ keyed
                 every other key matches a       default clauses)
                 default clause
Instance(C)      isinstance(x, C)               (class extension)
```

For the Boolean nodes the equivalence is the definition of the set operation, so
the step is immediate given the hypothesis on the children. For the structural
nodes (`Seq`, `KeyedMap`, `Set`) the walk evaluates the children exactly by
hypothesis and combines them by the same connective the denotation uses. The
scalar and `Instance` leaves reduce to `isinstance`, which is Python's own
membership test for those sets, and `Literal` adds the same-type guard that keeps
`Literal[1]`, `Literal[True]`, and `Literal[1.0]` distinct.

### Recursion terminates and stays exact

A `recursive` schema is a guarded (contractive) fixpoint: every back-edge sits
under a structural constructor. Membership unfolds that fixpoint against the
value in hand. Because the value is a **finite** Python object, the unfolding is
finite, and two guards keep it so:

- an object-identity guard rejects a value that contains itself
  (`recursion_loop`) rather than looping, and
- a depth bound rejects a value nested past the limit (`recursion_limit`) rather
  than overflowing the stack (see [resource limits](limits.md)).

On the inhabitants — the finite values — the guarded unfolding accepts exactly
the members, which is why a coinductive *comparison* at the greatest fixpoint
never contradicts a membership answer (see [recursion](recursion.md)).

## Why simplification preserves meaning

Every rewrite `simplify` performs is a law of the Boolean algebra of sets —
flattening associative nodes, dropping identities and duplicates, pushing
complement to negation-normal form, folding `X ∩ ¬X` to `⊥` and `X ∪ ¬X` to `⊤`,
and using scalar disjointness — each of which holds of the *sets*, so it cannot
change `⟦S⟧`. The simplifier is held to this one invariant and to nothing
stronger: it is a lattice normal form, not a decision, so membership relations
are read off the decision procedures, never off simplified structure.

## Why the decisions are sound (and only sound)

`is_subtype_of` applies structural inclusion rules, each a valid set inclusion:
`A ⊆ B₁ ∪ … ∪ Bₙ` when `A ⊆ some Bᵢ`, `A₁ ∩ … ∩ Aₙ ⊆ B` when some `Aᵢ ⊆ B`, the
contrapositive for complement, componentwise inclusion for the structural forms,
and the coinductive rule for recursion (assume the goal on the current path —
sound for inclusion at the greatest fixpoint). A leaf the rules cannot relate is
handed to an oracle that returns `False` when it cannot prove the relation. Every
rule preserves "the conclusion holds whenever the premises do", so a `True` is a
proof. `is_empty` and `is_equivalent` are derived (`is_empty(S)` is
`S ⊆ Nothing` via the same rules; `is_equivalent` is mutual inclusion), so they
inherit the soundness.

The conservatism is the price: when a rule does not fire and the oracle declines,
the answer is `False` — "not proven", not "disproven". This is why the
[decidability boundary](decidability.md) maps where `False` is exact and where it
is conservative, and why the docs say *closed algebra, conservative decision*.

## How the argument is mechanized

The argument is checked, not just asserted, by four independent test layers:

- **Denotation oracle.** Each node's `⟦S⟧` is written as a reference predicate
  over a value generator, and the walk is property-tested to agree with it — this
  is the membership-exactness claim, checked on generated values.
- **Algebra laws.** Every law `simplify` relies on is property-tested against the
  membership relation, in Rust (proptest) and Python (Hypothesis).
- **External ground truth.** The same schemas and values run through
  pydantic-core and jsonschema; a divergence is a bug or a documented intentional
  difference — an independent check that the *reference predicates themselves* are
  right, closing the single-author blind spot.
- **Coverage-guided fuzzing.** libFuzzer drives the decision procedures over the
  whole IR, asserting the sound order laws (reflexivity, the lattice bounds,
  equivalence as mutual inclusion); the same invariants gate every merge.

These are partial mechanization — adversarial, independent, and coverage-guided —
in place of a fully formal proof.

## What this argument assumes

The soundness is relative to a small, explicit trust base:

- `isinstance` and the PyO3 conversions report Python's own membership faithfully.
- The JSON parser (jiter) yields the value `json.loads` would, so the JSON path's
  denotation matches the object path's.
- The crates contain no `unsafe`, so there is no memory-safety obligation beyond
  the compiler's.
- **Predicate refinements are opaque.** A `Predicate` constraint runs arbitrary
  Python; valgebra checks that it returned truthy, and the soundness of *that*
  leaf is the caller's. Regex constraints are matched natively and related only
  by syntactic identity.

Within that base, an accept is a claim that `x ∈ ⟦S⟧`, justified node by node
above and exercised by the four test layers — which is what "rock solid" can
honestly mean before a machine-checked proof and outside review exist.
