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

The papers this rests on are in [10-theory.md](10-theory.md).

## What it buys

Stated before the design, because a design that lists structure without stating
its yield is asking to be taken on faith.

**One pool, four kinds of object.** The validator holds a single
`Vec<Py<PyAny>>` carrying a literal's constant, a class, a comparison operand and
a user predicate. Carried as a bare `usize` into the same `Vec`, any of them
would reach any of them — and because the spaces do not merely share a
type but share the *table*, a payload used against the wrong kind retrieves a
real Python object of the wrong kind. The failure is a plausible wrong verdict,
never a panic.

**A fifth space that is not the pool.** `Schema::Ref` addresses the definitions
table, where the four above are each their own.

**Each of these is a compile error**, and each is a swap that, without the
types, compiles into a wrong answer:

| the swap | what it would do instead of failing |
|---|---|
| the two shifts transposed at `Schema::shifted` | pool indices moved by the definitions offset |
| a class index where a literal's constant belongs | `isinstance` against a pooled number |
| a definition index where a pool index belongs | a schema read as a Python object |
| a pool shift applied to a length bound | a length bound moved by a pool length |

**It costs nothing measurable.** The core workload reads the same instruction
count with the index types as with a bare `usize`, to the instruction, and the
binding walk differs by under a hundredth of a percent. Seven newtypes over
`usize` -- five index spaces and two shifts -- all `#[repr(transparent)]`,
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

Four accessors for four questions, in the walk
(`crates/valgebra-py/src/check/walk.rs`). Beneath them one private `pool_slot`
takes a bare `usize`, which is the one place the walk stops tracking an index
space. The walk is not the only reader: the
oracle (`crates/valgebra-py/src/oracle.rs`), `==` (`equality.rs`) and the
per-validator precompute (`check/index.rs`) each open an index with `get()` at
their own sites, and every such call is a place the space stops being tracked,
which is why `get` is the one way out of an index type. The frontend
mints through the mirror-image four (`intern_const`, `intern_class`,
`intern_operand`, `intern_predicate`), so an index acquires its meaning at the
line that decides what the object is being pooled *as*
([03-frontend.md](03-frontend.md)).

`ClassIx` addresses `Schema::Instance`, the only node holding a
class: an attribute record carries fields and no carrier, and a class with
declared attributes is the meet of the two.

### The shifts

```
  PoolShift -- shifted --> ConstIx, ClassIx, OperandIx, PredIx
  DefShift  -- shifted --> DefIx
```

`Schema::shifted` takes one of each, so transposing them does not compile, and
the constraint arms that must not take a pool shift cannot: a length is a
`usize` and a pattern a `String`, and neither has `shifted(PoolShift)`.

### The region set

`Region` is a set of value-universe regions with `union`, `intersect`,
`complement`, `is_empty` and `subset_of`. Subtyping on the scalar-decidable
fragment **is** `subset_of`, and says so.

No fold spells the raw bit operators; each reads one of the five methods. That
is worth more than it reads: a `|` where a `^` belongs is a one-character defect
sitting inside a fold no test could distinguish it in, and concentrating the five
operations into five one-line methods puts each somewhere a five-line test
reaches.

`Regions`, beside it in `crates/valgebra-core/src/kind.rs`, is that set or
`Unknown`, for a schema that is not scalar-decidable. It is a monoid under each
lattice operation with `Unknown` absorbing both, and naming the absorbing
element is what lets a fold over members stop at it: past an opaque member no
later one can change the result.

`Kind`, in the same file, is the partition the descriptor is indexed by: the
type a value's `type(x)` is, so two schemas carrying different kinds share no
value, `bool` and `int` aside. It is public because a `Literal`'s kind is a fact
about a pooled constant only the bindings can read, and they answer in this
vocabulary through `LeafRelations::literal_kind`. `Kind::ALL` lists every kind,
and a test counts it against the variants rather than a comment promising it is
complete.

`BoolSet` (`crates/valgebra-core/src/descr/mod.rs`) is the two booleans as a
subset: `bool` has exactly two values, so a finite set over it is exact, and a
two-bit set is the smallest thing closed under the three operations.

### The answers

```
  Verdict   = Empty  | Inhabited | Unknown
  Relation  = Holds  | Fails     | Unknown
```

Both live in `crates/valgebra-core/src/verdict.rs`, and both exist because the
thing they answer has **three** states while a `bool` has two. A schema is proven
empty, proven inhabited, or neither -- an opaque leaf the core cannot read, or a
descent the work bound stopped. A relation is proved, refuted by a value, or
neither. Collapsing either onto a `bool` does not lose a rare case; it loses the
*contract*, which is "a positive answer is a proof, a negative is no **or not yet
proven**". A `false` that means both cannot say which it is, and a bail-out
becomes indistinguishable from a decision.

Three consequences follow, and they are why this is a type rather than a comment:

- **The reduction is named once per answer.** `Relation::holds` is the one
  place a relation becomes the `bool` `is_subtype_of` and `is_equivalent`
  return, and it reads `matches!(self, Relation::Holds)`. Emptiness becomes the
  `bool` `is_empty` returns in one match, in `Schema::is_empty_with`
  (`crates/valgebra-core/src/decision/emptiness.rs`), which reads only
  `Verdict::Empty` as `true` and asks the descriptor where the rules answer
  `Unknown`. So only a proof is `true`, at one place per answer rather than at
  every call site.
- **The combinators propagate it.** `Relation::and` and `Relation::or_else`
  carry `Unknown` through a conjunction and a disjunction, so a rule that
  declines one conjunct cannot have its decline read as a refutation by the rule
  above it.
- **It reaches the caller.** `Validator.relation_to` maps the three to
  `"subset"`, `"not_subset"` and `"undecided"`, which is what lets a test assert
  that a pair is *declined* rather than merely not proved
  ([02-decision.md](02-decision.md)).

The public `is_subtype_of` still answers `True`/`False`, because that is the
contract a caller is held to. The three-valued type is what makes the distinction
a property of the procedure instead of a sentence in a document.

```
  Reach     = Anything | AnyValue | Missing | Unread
```

`Reach` (`crates/valgebra-core/src/descr/classes.rs`) is what one attribute of a
*direct* instance of a class can be: missing or any value, any value, never
there, or code's answer. Four values rather than two flags, because the fourth
is not a pair of answers. Two flags would spell `Unread` as "never missing and
never a value", which reads as a fact about the instance -- no direct instance
carries the field -- and a proof of emptiness could stand on it, where a subclass
may carry the field and what code returns is not read. `Attributes` holds what the
bindings read off a class -- each name as a `Member`, in one `Namespace` per
class on the `__mro__`, the `Hook`, whether there is a `__dict__` -- and its
constructors are the only way in; `Attributes::reach` is the one place the
four answers are derived, so both deciders read one function.

### The modes

`WalkMode` is `Explain`, `ExplainFailFast`, `Fast` — three states where a pair of
booleans admitted four. `Guarded` and `Openness` name what a positional `bool`
would carry at the guardedness check and the record constructor. `SeqArity` is
`Exactly(n)` or `AtLeast(n)`, so the schema's arity is one argument rather than a
length and a flag beside the value's own length.

`Polarity` (`crates/valgebra-core/src/ir/transform.rs`) is `Widen` or `Narrow`,
the side of an inclusion a schema is read on: an unfolding cuts the reference no
finite representation holds to the top on the subject's side and to the bottom on
the supertype's. It is named rather than spelled as a `bool` because the two
members are what the soundness of a difference turns on, and a call site reading
`true` says nothing about which side that is.

`Entered` (`crates/valgebra-py/src/check/ctx.rs`) is what entering a reference at
a value found: the level is `Open`, the pair is already on the trail (`Cycle`, a
value that contains itself), or the trail stands at `MAX_RECURSION_DEPTH`
(`Full`) and no level was opened.

**The discriminant order of `WalkMode` is load-bearing.** Written naively the
sealed mode measured 2.6% *worse* than the two booleans it replaced, because
neither predicate the walk asks per node compiled to a single comparison. Ordered
so that explaining is "at most `ExplainFailFast`" and stopping at the first
failure is "at least" it, the same three variants measure 1.4% *better*. The type
was not the variable; the discriminant assignment was. A test pins both
predicates over every variant so a reordering fails rather than silently costing.

### The guards and the vocabulary

`Allowance` (`crates/valgebra-core/src/descr/budget.rs`) is a descriptor build's
work allowance for as long as it is held. The build is the guard's scope, and it
ends where the guard drops -- including where the scope unwinds, so a panicking
build gives back what it found rather than leaving the next one short. It is
`#[must_use]`, because a dropped one arms nothing.

`Code` (`crates/valgebra-py/src/codes.rs`) is a failure code as
`ValidationError.code` reports one. It is a newtype rather than a `&'static str`
so that the constants in that file are the only way to write one: a helper taking
a `Code` cannot be handed a string that looks like a code, and the compiler is
what says so. `tests/test_code_table.py` holds every name there to a site that
writes it.

## Adding a type

1. Say which set it denotes, and give it constructors that are the only way in.
2. Give it the algebra the quantity actually has and no more. An operator added
   because it is convenient will be used where it should not be.
3. Do not give it `From<the underlying integer>`. A conversion should be a place
   a reader can see.
4. **Make the mutation fail.** Break the code on purpose in the way the type
   exists to stop, and check the compiler rejects it. A type that has not been
   seen to reject something is a claim, not a guarantee.
5. Run `python scripts/perf_gate.py --against HEAD` and the same with
   `--binding`. The direction is not predictable from the source.
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
| `MapClause::of(key, value)` | a clause with its key and value sets swapped — a valid schema, wrong |
| `located(_, key, _, expected, summary)` | the two halves of an error message |
| `compare(left, right)` | the inverse ordering |
| `literal_matches(value, literal)` | a literal tested against a value |
| `is_multiple_of(value, operand)` | the reciprocal test |
| `predicate_passes(value, predicate)` | a value called on a predicate |

`MapClause::of` is the sharpest: either order typechecks and validates real
values. The technique that closes such a pair elsewhere — moving the
discriminator into the value so no call site carries one to transpose — does not
apply, because a key schema and a value schema are genuinely two schemas. What
does apply is a struct literal, which cannot be transposed: `Schema::mapping`
takes one `MapClause { key, value }`, the frontend spells every clause that way,
and `MapClause::of` is crate-private, called at the one pass that rebuilds a
clause from its parts.

The last three take the value first, and all three are `&Bound<'_, PyAny>`, so
an operand passed first typechecks.

**Overflow, though the policy makes the shape safe.** Overflow is a program
error and which behaviour a build gets is a per-profile choice, so `Cargo.toml`
states it: dev and test trap, release wraps. That is what makes the discipline
pay — every intended saturation in this tree is spelled, so a bare `+` that
wraps is a defect the test profile catches rather than a shape somebody meant,
and a test drives that rather than reading it off the manifest.

The index shifts are a bare `+` and cannot overflow: both terms are bounded by a
live pool's length, so their sum is bounded by the length of the pool they are
combined into. A `debug_assert` states it, because a wrapped index would read a
real object of the wrong kind and the release profile wraps.

**Cost.** A newtype is free in *layout* and not always free in *codegen*. The
place it is not free here is a branch the walk takes per node -- the `WalkMode`
discriminant order above -- which is why step 5 above is not optional.
