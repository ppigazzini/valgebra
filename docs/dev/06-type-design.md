# The value domain

The types in this codebase are not decoration over integers. Each exists because
a quantity has a structure, and the structure is what the type carries. This page
states that structure, and closes with what these types do **not** promise.

## The premise: a type is a proof that travels

Both crates forbid `unsafe` ([00-architecture.md](00-architecture.md)), so the
type system is not a safety net over an escape hatch. When the code knows
something, the place to put it is where the compiler can see it: a doc comment
saying "this index is always in the pool" is a proof that has evaporated, and a
`ConstIx` is the same proof still there at every call site.

The papers this rests on are in [09-theory.md](09-theory.md).

## What it buys

Stated before the design, because a design that lists structure without stating
its yield is asking to be taken on faith.

**One pool, four kinds of object.** The validator holds a single
`Vec<Py<PyAny>>` carrying a literal's constant, a class, a comparison operand and
a user predicate. Every one of those was a bare `usize` into the same `Vec`, so
any of them reached any of them — and because the spaces do not merely share a
type but share the *table*, a payload used against the wrong kind retrieves a
real Python object of the wrong kind. The failure is a plausible wrong verdict,
never a panic.

**A fifth space that is not the pool.** `Schema::Ref` addresses the definitions
table. Before the split it was the same `usize` as the four above.

**Each of these is now a compile error**, and each was broken on purpose to
confirm the compiler rejects it:

| the swap | what it did instead of failing |
|---|---|
| the two shifts transposed at `Schema::shifted` | pool indices moved by the definitions offset |
| a class index where a literal's constant belongs | `isinstance` against a pooled number |
| a definition index where a pool index belongs | a schema read as a Python object |
| a pool shift applied to a length bound | a length bound moved by a pool length |

**It cost nothing measurable.** The core workload's instruction count is
identical to the instruction across the split, and the binding walk moved under a
hundredth of a percent. Five newtypes over `usize`, all `#[repr(transparent)]`,
carried and consumed one at a time: the free shape.

## The maps

Every arrow is a named function. A value crosses a boundary by calling something,
and the call is where a reader looks.

### The pool: four spaces, one table

```
  ConstIx    -- const_at     --> a literal's constant
  ClassIx    -- class_at     --> a class, for isinstance
  OperandIx  -- operand_at   --> a comparison or multiple-of operand
  PredIx     -- predicate_at --> a user predicate
```

Four accessors for four questions. Beneath them one private `pool_slot` takes a
bare `usize` — the single place an index space stops being tracked. The frontend
mints through the mirror-image four (`intern_const`, `intern_class`,
`intern_operand`, `intern_predicate`), so an index acquires its meaning at the
line that decides what the object is being pooled *as*
([03-frontend.md](03-frontend.md)).

`ClassIx` covers both `Schema::Instance` and `Schema::Attrs`: they address the
same kind of object and no call site carries one of each, so a fifth type would
be a distinction with no swap behind it.

### The shifts

```
  PoolShift -- shifted --> ConstIx, ClassIx, OperandIx, PredIx
  DefShift  -- shifted --> DefIx
```

`Schema::shifted` takes one of each. Transposing them no longer compiles, and the
constraint arms that must not take a pool shift cannot: a length is a `usize` and
has no `shifted(PoolShift)`.

### The region set

`Region` is a set of value-universe regions with `union`, `intersect`,
`complement`, `is_empty` and `subset_of`. Subtyping on the scalar-decidable
fragment **is** `subset_of`, and says so.

The raw operators left the folds entirely. That is worth more than it reads: a
`|` where a `^` belongs is a one-character defect sitting inside a fold no test
could distinguish it in, and concentrating the five operations into five one-line
methods put each somewhere a five-line test reaches.

### The modes

`WalkMode` is `Explain`, `ExplainFailFast`, `Fast` — three states where a pair of
booleans admitted four. `Guarded` and `Openness` name what a positional `bool`
used to carry at the guardedness check and the record constructor. `SeqArity` is
`Exactly(n)` or `AtLeast(n)`, so the schema's arity is one argument rather than a
length and a flag beside the value's own length.

**The discriminant order of `WalkMode` is load-bearing.** Written naively the
sealed mode measured 2.6% *worse* than the two booleans it replaced, because
neither predicate the walk asks per node compiled to a single comparison. Ordered
so that explaining is "at most `ExplainFailFast`" and stopping at the first
failure is "at least" it, the same three variants measure 1.4% *better*. The type
was not the variable; the discriminant assignment was. A test pins both
predicates over every variant so a reordering fails rather than silently costing.

## Adding a type

1. Say which set it denotes, and give it constructors that are the only way in.
2. Give it the algebra the quantity actually has and no more. An operator added
   because it is convenient will be used where it should not be.
3. Do not give it `From<the underlying integer>`. A conversion should be a place
   a reader can see.
4. **Make the mutation fail.** Break the code on purpose in the way the type
   exists to stop, and check the compiler rejects it. A type that has not been
   seen to reject something is a claim, not a guarantee.
5. Run `python scripts/perf_gate.py` and `python scripts/perf_gate.py --binding`.
   The direction is not predictable from the source.
6. Add a row here. A type added without one makes this page quietly wrong.

## What a compile error does NOT stop

A page that omits its own boundary invites over-trust.

**A wrong index that is in range.** Every index type here is a newtype over an
integer, not a refinement over a range. `ConstIx` stops a class reaching the
literal path; it does not stop the *wrong* constant reaching it. The pool is
trusted data the frontend built, and that stays a property the builder holds.

**A transposition between two arguments of the same type.** This is the largest
residual hazard in the tree and it is worth naming precisely:

| site | transposing it gives |
|---|---|
| `Schema::mapping(key, value)` | `dict[V, K]` — a valid schema, wrong |
| `located(_, key, _, expected, summary)` | the two halves of an error message |
| `compare(left, right)` | the inverse ordering |
| `literal_matches(value, literal)` | a literal tested against a value |
| `is_multiple_of(value, operand)` | the reciprocal test |
| `predicate_passes(predicate, value)` | a value called on a predicate |

`Schema::mapping` is the sharpest: the frontend mints one from `dict[K, V]`'s two
type arguments and either order typechecks and validates real values. The
technique that closes such a pair elsewhere — moving the discriminator into the
value so no call site carries one to transpose — does not apply, because a key
schema and a value schema are genuinely two schemas. What does apply is giving
the constructor one named argument instead of two positional ones: a struct
literal cannot be transposed.

Note also that the last three take the value in **different positions**. All
three are `&Bound<'_, PyAny>`, so a reader who has just read two of them has the
wrong prior for the third.

**Overflow.** The index shifts are a bare `+`. A pool index plus a pool length
cannot realistically overflow a `usize`, but nothing states that intent and
`Cargo.toml` sets no `overflow-checks`, so release inherits wrapping by accident
rather than by decision. The intended saturations elsewhere **are** spelled —
`saturating_add` on the node-count sum, `saturating_sub` on the required-field
counter and the union probe's depth, `checked_sub` in the decision budget — which
is the discipline; what is missing is the profile that makes it pay.

**Cost.** A newtype is free in *layout* and not always free in *codegen*. The one
place it was not free here was a branch the walk takes per node, which is why
step 5 above is not optional.
