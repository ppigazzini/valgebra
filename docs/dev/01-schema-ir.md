# The schema IR

`crates/valgebra-core/src/ir.rs` owns the node set and the pure structural
operations over it: construction, index shifting, self-reference resolution, and
the guardedness check.

## A node denotes a set of Python values

That is the whole frame, and every variant's doc comment states its set.
`Schema::Int` denotes the `int` instances; `Schema::Union` denotes the union of
its members' sets; validation is membership in the set the root node denotes.

Two consequences a reader needs before touching this file:

**Subtyping is subset inclusion, so `bool` is a subtype of `int`.** Python makes
`bool` a subclass of `int`, so `True` is an integer and `Schema::Int` admits it.
No value is carved out. `Schema::Float` is disjoint from `Schema::Int` for the
mirror reason: `int` does not subclass `float`.

**A `Literal` is a *typed* singleton.** Python's `==` conflates across types
(`1 == True == 1.0`), so equality alone would make `Literal[1]` admit `True` and
`1.0`. Membership requires `type(x) is type(c)` as well, which is what keeps the
typing spec's distinction between `Literal[1]`, `Literal[True]` and
`Literal[1.0]`. The same-type test runs **before** `==`, so a value of another
type never reaches the comparison — which is why a raising `__eq__` is only
observable from an object of the pooled constant's own type.

## What fixes a node's carrier, and why it differs per node

A structural node denotes values of a *shape* held by some *carrier* — a Python
class. The three structural nodes fix their carrier three different ways, and
the difference decides how large a change is that widens one. One of the three
was widened, by taking the carrier out:

| node | how the carrier is fixed | widening it means |
|---|---|---|
| `Seq { container: SeqKind, shape }` | a **parameter** — `SeqKind` is `List \| Tuple` | another variant in an existing enum |
| `Coll { container: CollKind, element }` | a **parameter**, as above — `CollKind` is `Set \| FrozenSet` | another variant in an existing enum |
| `KeyedMap { fields, defaults }` | **the denotation** — the doc comment reads "Denotes dicts…", and there is no carrier field | giving the node a carrier it does not have |
| `AttrRecord { fields }` | **not fixed at all** — a record carries no carrier, and a class is a separate `Instance` atom met with it | nothing to widen; the two halves are already separate sets |

Read that table before answering "can valgebra express `Mapping[K, V]`" or "can
it express *any object* whose `.a` is an `int`". The first is still no, and the
second became yes the day the class came out of the node — which took a
denotation, a membership rule and a set of relations, not a flag. The three
nodes are not one mechanism seen three times, and what one of them cost to widen
is the measure of the other two.

## Whether to add a variant

Before the mechanics below, the admission test. **The node set is a minimal
generating set plus the representatives the normal form names** — that is the
definition, not a preference, and it is what makes "the algebra" a claim rather
than a collection.

Two columns, and `tests/test_closure_ledger.py` holds every variant to one of
them:

- a **generator** denotes a set no combination of the others reaches. `Int`,
  `Seq`, `Literal`, `Union`, `Complement` and the rest are here, and admitting a
  new one is the argument this section is about.
- a **representative** denotes a set the generators do reach, and is kept because
  the normal form has to name it. `Nothing` is `complement(anything)`, `Bool` is
  `Literal[True] | Literal[False]`, `NoneType` is `Literal[None]`, and
  `Intersection` is De Morgan of the other two — each is checked against its
  derivation, in both directions, by that ledger.

The distinction is what keeps the claim honest. Read as "no node denotes a set
another reaches", the sentence was simply false of five variants; read as this,
it is a statement a test settles.

So a proposed node is one of exactly two things:

1. **Its set is already in the closure of the existing atoms.** Then it is
   redundant: write it as a combination and add nothing.
2. **Its set is not.** Then it is an extension, and the case for it has to be
   that the algebra is *incomplete for its domain* without it — not that it
   would be convenient, and not that the code already computes something like it
   internally.

There is no third case. "It would be convenient", "a downstream user wants it",
and "the machinery is nearly there" are all case 2 arguing under another name,
and the only honest way to make them is to say which part of the domain is
unreachable without the node.

**The domain is the second half of the test**, and it is a real question rather
than a formality: valgebra exposes its algebra through standard Python typing
syntax, so a set that syntax can express and valgebra cannot is a completeness
gap, while a set only a bespoke combinator could name is not. Whoever owns the
algebra decides where that line sits; a page here records the decision, and
[10-theory.md](10-theory.md) already shows the form such a decision takes — the
map clauses are unordered where the paper's are ordered, written down as "a
deliberate narrowing".

### What does not count as evidence

Three arguments look like they settle case 1 and do not:

- **"The behaviour already looks right."** A `Validator` probe shows what the
  walk does. The walk can agree with a denotation by accident, and a node's
  denotation is what the doc comment in this file says.
- **"The implementation already computes it."** `build_object` computed
  per-attribute checks on its way to a node that denoted instances of a class,
  and for as long as that was the only node, "any object with these attributes"
  was not in the closure. It took splitting the node — a commit, a denotation, a
  membership rule — to put it there. A computation on the way to a set is not
  that set.
- **"It only changes which values a constructor sees, not the algebra."**
  Subtyping is defined from the denotation (`[[s ∧ ¬t]] = ∅`), so changing which
  values inhabit a constructor changes the subtyping relation. There is no lever
  that separates the two.

## Refused, and the test each one fails

These do not go in. Each carries the test it fails, so a proposal starts here
rather than at the beginning.

**A carrier for `KeyedMap`, so `Mapping[K, V]` and `Sequence[T]` build.**
Refused. It is a constructor extension, not an encoding change: `KeyedMap`
denotes dicts and has no carrier field, so supplying one changes the
constructor. Beyond the admission test, two costs specific to this one:

- The model `KeyedMap` is built from is recorded in
  [10-theory.md](10-theory.md) as *records and maps as quasi-constant
  functions* — named fields plus a key-typed default. No carrier appears in that
  description. Whether the source paper treats a nominal carrier is **not
  established here**, and its title names structs, so do not assume it does not.
  What follows is only that this project has no recorded reading of a
  carrier-indexed map: an absence of guidance, not a permission and not a
  prohibition.
- The decision procedures this project sources decide maps **without** a
  carrier: Elixir's `Module.Types.Descr`, and the negated-map-atom decomposition
  for deciding a keyed map under negation. A carrier-indexed map is outside the
  fragment with a published decision, which is the part the completeness claim
  rests on.

**A spelling for the carrier-free attribute record.** Not refused, and not
opened: the node is in the closure — `AttrRecord` denotes exactly "any object
whose `.a` is an `int`" — but no annotation builds one on its own, because the
frontend only ever meets one with the class it read it from. Giving it a
spelling is a surface question, and it belongs where the surface is decided
rather than here.

**Records keyed by a non-string.** Refused. A record's fields are named by
strings, and arbitrary keys are a different labelling than the record model
assumes.

### Arguments for the carrier change, and why each fails

| argument | why it fails |
|---|---|
| "the behaviour already looks right" | a probe shows the walk, not the denotation |
| "the implementation already computes it" | `build_object` computes attribute checks *inside* a node whose denotation includes the class |
| "it only changes which values a constructor sees" | subtyping is defined from the denotation, so that *is* the algebra |
| "the sources name no carrier, so adding one is free" | an absence of guidance is not permission. It also is not a refusal — the refusal rests on the admission test, not on this row |
| "a class carrier is unsound because `register()` mutates it" | false — class relations are re-decided on every call, never cached |
| "a downstream package needs it, and pays to route around it" | a consumer's requirement is a fact about the consumer. It is the loudest argument for growth and the weakest: every package built on valgebra will want the node that would make its own job easier |

Two rows need separating from the rest.

The `register()` row is an argument *against* the change, and it is also wrong,
so it is not a reason to refuse. The refusal rests on the admission test and the
two costs above, and on nothing else.

The downstream row is the one to watch, because it arrives with evidence
attached — a measured cost, a real user, a working reproduction — and none of
that bears on whether the set belongs in the algebra. **valgebra is not extended
to make a consumer's job easier.** A package that cannot express something with
the algebra as it stands has found a fact about itself; if the same limitation
also fails the admission test on its own terms, the test is what carries it, and
the consumer is at most the reason someone looked.

The asymmetry is the point: a consumer can always route around a missing node —
with a predicate, a conversion, its own translation — at a cost it measures and
accepts. The algebra cannot route around a node that should not have been added.

## Adding a variant

The compiler forces the exhaustive matches; the doc comment on `Schema` carries
the checklist of what it forces. Read it there rather than here: a second list
drifts by one entry and reads exactly like one that has not.

Two things the compiler does not force, and both are held by a test:

- **A representative in the node matrix.** `tests/test_node_matrix.py` reads the
  variants out of this file and fails when one carries no row, so a node cannot
  arrive without a case that exercises it.
- **A case in the denotation oracle.** `tests/test_denotation.py` pairs each
  generated schema with an independent Python predicate; a node it never draws is
  a node the oracle does not check.

## What the payloads address

Five payloads are integers addressing something the validator holds, and four of
them address the **same** constants pool: a literal's constant, a class, a
comparison operand, a user predicate. Each has its own type, and
[06-type-design.md](06-type-design.md) owns why and what a crossing would cost.

`Constraint::MinLen` and `Constraint::MaxLen` sit in the same enum and are **not**
pool indices at all — they carry the length inline. That is what the type says: a
length has no `shifted(PoolShift)`, so the arms that must not take a pool shift
cannot.

## Composition, and the two shifts

Two validators combine by concatenating their constants pools and their
definitions tables. The second schema's indices move past the first's lengths,
which `Schema::shifted` does — one shift per index space, and they are distinct
types so a caller cannot transpose them.

`Schema::reindexed` is the same operation where the second pool is *interned*
into the first rather than appended, so identity-shared constants collapse to one
slot. It is the one the binding actually calls; `shifted` is reached from the
tests and the fuzz targets.

Being the same operation, it is one walk: `Remap` names the difference — append
by a distance, or intern through a table — and the walk asks each index space how
it moves. A definitions index moves the same way under both, which is what makes
one walk enough. The two were separate walks, identical but for the leaf action,
and a payload site reached by one and missed by the other was a wrong index that
neither the types nor an exhaustive `match` could see: the compiler forces an arm
per variant and cannot check that the arm moved anything. `Schema::map_children`
holds the other half of that argument — it is the single place each variant's
child schemas are written down, so a walk that only descends inherits the child
set instead of restating it. `Schema::remapped_by` takes no wildcard on purpose:
a future variant carrying a pooled index must be a compile error there rather
than a node that silently keeps an index into the wrong pool.

The laws in `crates/valgebra-core/src/lib.rs` hold both entry points to moving
every payload by its own space's distance, counted through an enumeration written
in the test module rather than reached for in the IR — a check that judges the
walk against something other than itself.

## Recursion, and why the guardedness check answers what it does

`recursive` builds a fixpoint. The body is compiled with a `SelfRef` marker,
which `resolve_self` rewrites to a `Ref` into the definitions table before the
validator is returned — so no compiled schema holds a `SelfRef`, and the walk
treats one as a non-member if it ever sees one.

A definition is admitted only when it is **contractive**: every occurrence of the
self-reference sits under a structural constructor. `Schema::occurs_unguarded`
decides that, and its shape is worth stating because it is not obvious from
reading it.

**`Guarded::Yes` is absorbing.** The only arm that can answer true demands
`Guarded::No`, and the algebraic combinators pass the guard through unchanged, so
nothing below a structural constructor is ever reported unguarded however deeply
it nests. Each structural arm therefore answers false for every input — the same
answer the match's default gives. The arms are written out because they state
*which* constructors guard; they compute nothing.

Two properties in `crates/valgebra-core/src/lib.rs` pin that rather than leaving
it as a comment: the guard absorbs under every structural constructor, and read
from the top the check agrees exactly with "the reference is reachable through
algebraic combinators alone".

## Which laws construction settles

All of them: a schema is built in the lattice normal form. Members of a union or
a meet are flattened, ordered and deduplicated; the two identities are applied
(`A | nothing` is `A`, `A & anything` is `A`) and the two absorbing elements too;
absorption removes a member that contains another; `~~A` is `A`, `A | ~A` is the
top, and `A & ~A` is the bottom.

The three complement laws differ from the rest in what they need. `~~A` holds of
anything, because a complement is evaluated by negating what is under it, so
negating twice asks the same question once -- even of an atom that answers by
running code. The other two ask `A` *twice*, and a predicate or a class with an
`isinstance` hook may answer differently each time; those two folds are asked of
the atom first, and decline for one that is not a set.

**Why the normal form rather than a free term.** A constructor that folds some
laws and not others is neither: `union(int, int)` rendered `int | int` and
compared unequal to `int` while `union(int, complement(int))` rendered
`anything`, so `==` was equality of nothing in particular and `repr` showed a
shape no rule was written for. Normalising at construction makes `==` equality of
a canonical form -- which is what the API reference already calls it -- and makes
the shape every rule downstream may assume the shape it gets.

The first holds of anything. A complement is evaluated by negating what is under
it, so negating twice asks the same question once — even of an atom that answers
by running code. The other two do not: `A | ~A` and `A & ~A` ask `A` *twice*, and
a predicate or a class with an `isinstance` hook may answer differently each
time. Those two folds are therefore asked of the atom first, and decline for one
that is not a set.

**Both sides or neither.** A law folded in `union` and left standing in
`intersection` is two answers to one question, and the simplifier already folded
the meet — so the constructors disagreed with each other and with it. Stating a
law once, in the place a schema is built, is what lets a rule downstream assume
no such shape reaches it.

That assumption is why a transform that *descends* must refold. Opening the
records in `{a: int} | ~{a: int}` maps both sides to one schema beside its own
complement: a shape construction promises never survives it. Reindexing is the
exception and stays raw — it relabels pool slots, and a relabelling that changed
the shape would not be one.

The cost is that `repr`, `==` and the reported error follow the schema as
*built*, not as written: `intersection(int, complement(int))` is `nothing` and
reports `no_match`. That is recorded in
[08-error-model.md](../08-error-model.md), where a reader meets it.

## What a length is

`MinLen` and `MaxLen` on a `list` or a `tuple` read the number of elements the
value **holds**. Every other kind answers `__len__`.

The rule exists because a value must have *one* length. A sequence schema walks
the storage a `list` or `tuple` holds, so `[int, int]` counts elements; a length
marker used to call `__len__`, which a subclass may override to say anything. A
value with two lengths belongs to `Annotated[list[int], MinLen(5)]` and not to
the five-element shape, though both were written to mean the same narrowing —
and a set whose membership depends on which half of a schema asks is not a set.
Reading the storage in both places is what makes `MinLen(1)` on `list[T]` the
same set as a one-element prefix, which is what lets the algebra hold sequence
lengths structurally at all.

The choice is between *what the value reports* and *what the walk can see*, and
only the second is available to both halves. `__len__` remains the length of a
`str`, `bytes`, `set` and `dict`, because the walk reads those through it too:
the rule is not "distrust `__len__`", it is "one length, and it is the one the
walk already uses for that kind".

## What `Any` is

`typing.Any` is the top, spelled. The decision layer reads it as `Anything`;
the term keeps the spelling so that `repr` can give it back.

A runtime validator asks one question of a schema — does this value belong —
and to that question `Any` answers yes for every value. The walk always said
so, with one arm for both. Gradual typing (Siek & Taha)
holds the dynamic type apart from the top for a *second* question, consistency,
which a static checker asks at every site where a value crosses between typed
and untyped code. A validator has no such site and never asks it. Keeping the
atom for a question nobody asks costs a decision that disagrees with the walk:
`complement(Any)` admits nothing at runtime, so `intersection(Any,
complement(Any))` is the empty set, and an `is_empty` that declines to say so
is not conservative about a set — it is describing a set the library does not
check against.

So `Any` is not a variant. Building it yields `Anything` carrying a
`Spelling`; `render` reads it and writes `Any`; every relation and every law
reads the top. The flag is a term fact like a field name, and it is not a set:
two schemas that differ only in it are equal. That is not a rule asking to be
followed — `Spelling` implements `PartialEq`, `Ord` and `Hash` by hand so every
spelling compares and hashes alike, and a rule that tried to branch on it would
find two values it cannot tell apart. Nothing in the core reads it; `render` in
the bindings does, and it is the only thing that may. What is lost is the ability to say "this
branch was deliberately not checked" *inside the algebra*; what is kept is that
the reader sees it in `repr`, which is where they looked for it.


## What a `TypedDict` denotes

A `TypedDict` class denotes the set its typing spec assigns to it: an **open**
record, unless the class says `closed=True` or gives `extra_items` (PEP 728,
now in the spec). The dict-literal form `{"a": int}` stays **closed**.

The reason is who wrote the spelling. `Validator(TD)` reads an annotation whose
meaning is fixed elsewhere, and reading it as a different set — closed, when the
spec says open — is a deviation a caller has no way to see: the class carries
no mark of it. The dict-literal form is this library's own, and a schema written
as a *shape* means that shape. A user who wants an open shape writes the
open marker; a user who wants a closed `TypedDict` writes `closed=True`. Both
sets are spellable both ways, so the choice fixes defaults only, and each
default is the one its author's spec gives.

`ReadOnly` is stripped: it constrains writers, and a value has no writers.
`total`, `Required` and `NotRequired` set key optionality as today.
The frontend reads every `TypedDict` closed today; this is the rule it moves
to, and the map milestone that closes the negative set is where it moves.

## Where a class and an attribute record go

A value's class, and the attributes it must carry, narrow a value **within its
kind**; they are not a kind. The descriptor therefore holds them as lines of
the kind's component: each component is a small union of lines
`structure ∧ classes ∧ attrs`, complemented as a DNF inside the kind, and a
class beside a builtin kind is one more line of that kind.

The two other shapes were weighed and fail one test each. A DNF over the whole
descriptor loses the kind partition — every operation becomes a DNF operation
over every kind, and the cheap disjointness across kinds, which is most of what
the partition buys, goes with it. Scoping classes to the `other` kind, which is
what the descriptor does today, fails the value it was built for: a dataclass
that subclasses `int` has an integer kind and a class, and a component that
cannot hold both is one that cannot decide it. Per-kind lines keep the
partition, split a line's complement into at most three atoms, and are the
shape the record atom already has, so the code exists once.

It waited on one thing, and that has landed: guards held by handle, so a
component may be a union without multiplying the memory of every automaton that
holds one.

What is *not* a precondition, and once read as one, is the descriptor replacing
the decision procedure. That caution is about **consulting** a DNF descriptor
beside a procedure that already answers — paying twice for one verdict — and it
is the rule that withdrew the shadowing widening. It says nothing about building
the representation, which decides nothing a caller can reach and is checked
against membership over generated values like every other part of the
descriptor.

The half that waited on the object pool -- lowering an `Instance`, which needs
the bindings to say which classes a class derives from -- no longer does. The
pool answers three questions now: what an operand is, what a literal names, and
what order a class carries. See "How a class reaches the core" below.

## What a class with attributes is, on the surface

An object schema is the meet of an instance atom and an attribute record, and
the surface does not change: `repr(Validator(Pt))` stays `Pt`, and a value that
is not an instance reports one violation, `instance_type`.

The meet is right in the algebra — it is what the attribute form *is* — and
left alone it would change two things a user sees: `render` would write it as
`intersection(Pt, object(x=int))`, and the walk's explain mode, which collects
every failed member of a meet, would add attribute violations about a value that
is not an instance. Neither is information: the first is the algebra's spelling
of a thing the user spelled as `Pt`, and the second reports attributes of an
object that has no reason to have them.

So the surface is preserved by construction. `Schema::object_class` recognises
the pair — exactly one instance atom beside exactly one attribute record, other
members tolerated — and `render` and a union's branch label both read the class
out of it, which is why the shape is recognised once rather than matched at each
of them. The walk stops collecting once a member of a meet has failed *at the
meet's own path*: a member that rejects the value itself has settled it, and a
member that fails inside the value leaves the others meaningful. That second
rule is not specific to classes — it is the one a refinement already applies
between a base and its constraints, which is why `Annotated[int, Gt(0)]` does
not report a bound on a string — and it is recorded in
[08-error-model.md](../08-error-model.md). The error snapshot pins both.

## How a class reaches the core

The core holds no Python objects, so a schema names each one by an index into the
validator's table and the lowering asks the bindings to read it. `Constants` is
that question, in three parts: what a comparison operand is, what a literal
names, and -- since a class is a set the descriptor holds -- what order a class
carries.

A class answers as a snapshot: an id, the ids of the classes its `__mro__` lists,
and a layout tag. The snapshot is taken once, at lowering, rather than by asking
`issubclass` again later, because `abc.ABC.register` rewrites the subclass
relation after a schema is built and a relation that moves is not an order to
reason in. A class whose metaclass answers `isinstance` or `issubclass` itself is
declined outright, on the test the decision procedure already applied: two
occurrences of one such class can disagree, so `A ∧ ¬A` is not empty and the law
that says it is must not fire.

The layout tag is how disjointness is *proved* rather than guessed. Python
refuses `class C(int, str)` -- "multiple bases have instance lay-out conflict" --
so two classes built on different builtins share no value, and no class can
derive from both. A class built on no builtin lays down no layout of its own and
takes the plain tag, which conflicts with nothing: `class Both(Plain, MyStr)`
builds, so a plain class and a `str` subclass do share values. Everything else is
undecided, which is the honest answer -- two unrelated classes may still meet in
a subclass nobody has written yet.

An operand is read by its **exact** type. A subclass of `int` carries its own
`__eq__` and its own `__hash__`, and the sets the descriptor holds are Python's
equality on the builtin scalars alone, so a subclass instance is a value the pool
declines rather than one it kinds wrongly. Declining refuses the lowering, which
leaves the schema to the decision procedure; kinding it wrongly would put a value
in a set it is not in.

## What it costs to build a descriptor

`a ≤ b` is `a ∧ ¬b = ∅`, and the descriptor answers that question for schemas the
decision procedure declines. Whether it may be *asked* is a cost question, and
the cost was measured on the shapes the two disagree about.

The wins are cheap. Each of the differences the descriptor decides and the rules
do not -- a container meet, a double complement, one regular language inside
another -- builds in 130 to 330 microseconds.

Building is exponential in nesting depth. A record nested behind a list, at
depths 0, 2, 4, 6 and 8: 7 microseconds, 280 microseconds, 1.8 milliseconds, 8
milliseconds, 37 milliseconds. Bounding the depth does not bound the cost,
because breadth multiplies too: a union of four records at depth three builds in
2.7 milliseconds, and that union minus a union of its siblings spends **345
milliseconds** and then *refuses*, because the result exceeded the line bound.

That last number is the shape of the problem. `MAX_LINES`, `MAX_ATOMS` and
`MAX_STATES` -- rows in the table of every bound in the tree
([00-architecture.md](00-architecture.md)) -- bound the descriptor a build may
**produce**; nothing bounds the work a build may **do**, so refusing costs as
much as succeeding and a caller
cannot buy safety by being asked to accept less. This is what the nightly fuzzer
reported as an out-of-memory that four separate bound reductions did not move,
and what took its throughput from three million runs to two thousand three
hundred: not a bound set too high, but a quantity with no bound at all.

So the decision is:

**The descriptor may be asked only under a budget on the work a build does, and
a build that would exceed it refuses before spending it.** The budget is spent
where the recursion is -- the three closure operations, and a guard's own meet,
join and complement -- so a build that is about to be expensive stops while it is
still cheap. Outside a build there is no budget and nothing is charged, which is
what leaves the laws and the tests measuring the algebra rather than the meter.

Two things follow, and they are the reason this is recorded rather than assumed.
The descriptor **does not replace the decision procedure**. A schema whose build
runs out of budget still has to be decided, as does a recursive one, which no
finite descriptor holds. And the descriptor is asked **after** the rules, not
before: it can only turn "not proved" into "proved", so both orders give the same
answers, and building a descriptor beside a verdict the rules already reached is
work whose result is discarded.

There is a second bound, and the allowance does not replace it. An allowance
bounds what a build spends once it has started; it cannot bound what starting
costs. A lowering builds an automaton at every sequence node and a powerset at
every set node, and none of that is a product to charge for. Held to 1024 units
and nothing else, one `is_empty` over a record nested eight deep costs **more
than the structural rules spend on the entire decision workload**: seventeen
hundred times its instruction budget, which is not a budget to re-record but a
workload no lane can run.

So a build that will not pay for itself is refused *before* it is walked, and
**nesting** is what says which. Depth is the exponential -- 7 microseconds, 280
microseconds, 1.8 milliseconds, 8 milliseconds, 37 milliseconds at depths 0, 2,
4, 6 and 8 -- while breadth is not, a record of sixteen fields building in 13
microseconds. Every relation the descriptor decides and the rules do not nests
five deep or less; the shapes that blow up nest ten and deeper. Bounded at five,
the whole widening costs the decision path **eleven percent**, and the
validation path, which no relation is on, is unchanged.

What it does cost is the fuzzer. The decision target runs at 139 executions a
second against 10,544, over five times the covered features and twice the
resident memory: every input the fuzzer generates that the rules decline now
builds two descriptors. That is a testing-capability cost rather than a
user-facing one, and the route out of it is a target that asks the rules
directly for the high-volume properties, with the descriptor on a target of its
own. Until there is one, the nightly run buys the reach back with time: the
decision target's budget is 360 seconds rather than 180
(`.github/workflows/ci.yml`), because the throughput is CPU-bound and nothing
else moves it.

## What a bare builtin class denotes

`list` and `list[object]` admit exactly the same values -- every list, a
subclass instance included -- and neither was decided below the other. The two
were different sorts of thing: `list[object]` is a sequence node in the `List`
kind, and `list` was an `isinstance` atom in no kind at all, so the two never met
on a line.

**A bare builtin container is its kind.** The frontend maps `list`, `tuple`,
`set`, `frozenset` and `dict` to the kind's own top -- `list[anything]`,
`tuple[anything, ...]`, `set[anything]`, `frozenset[anything]`,
`dict[anything, anything]` -- which is the set the typing spec assigns an
unparameterised generic, and the set the walk already checked. The two spellings
build one schema and compare as one. `str`, `bytes`, `int`, `float` and `bool`
were already their kinds; this is the rest of that rule.

**A class laying down a builtin layout goes on that kind's line.** Every
instance of a `str` subclass is a `str`, so the class constrains a value *within*
the `Str` kind rather than instead of it, and putting it there is exact: it is
where `MyInt ≤ int` and `MyStr ≤ str` are decided. The layout is what says which
kind, and it is the layout the class lays down rather than the one it inherits,
because a subclass keeps it.

**A class laying down no layout of its own goes on every kind, and must.**
`class Both(Plain, MyStr)` builds and its instances are strings, so a plain
class's instances are not confined to the kindless slot -- a subclass may add any
layout. Placing such a class narrowly would be the one direction that is unsound:
claiming a value does not exist. It stays on every line, which is what it was.

## Which representation decides

Two decide, and the order between them is a cost measurement rather than a
preference.

The **descriptor** is the definition: a schema denotes a set, `a ≤ b` is
`a ∧ ¬b = ∅`, and each kind carries a representation closed under the three
operations so that question can be asked literally. Everything it can hold, it
decides.

The **structural rules** are the fast path. Building a descriptor costs about
two orders of magnitude more than a rule that already answers, which is measured
under "What it costs to build a descriptor" above; a shape the rules settle in
nanoseconds is not worth a set representation. So the rules answer first, and the
descriptor answers where they decline.

What that ordering is *not* is a second opinion. Both are asked of the same
question and the descriptor can only turn "not proved" into "proved", so the pair
gives the descriptor's answers wherever the descriptor can build -- the rules
never overturn one. The rules are an optimisation of a relation the descriptor
defines, and the page that describes the decision says so in that order.

**A rule earns its place by answering a shape the descriptor refuses, or by
answering a common one far more cheaply.** A rule that only repeats what the
descriptor decides is dead weight that the mutation sweep can no longer see
through the public relations, and it goes. Each rule that stays is named on
[02-decision.md](02-decision.md) with which of the two reasons it has.

**Every bound is measured.** A numeric bound in the decision -- the nodes a
lowering reads, the nesting it descends, the work a build spends, the steps the
rules take -- is admitted only with a workload in the tree that reproduces the
number it was set from. A bound whose number lives only in a comment cannot be
re-derived on another machine, and cannot fail when the shape it guards against
changes. The three a lowering can run out of are named together as `Bounds`, and
`crates/valgebra-core/benches/core.rs` measures each against the shapes it was
set from, with
`Bounds::UNHELD` to show what a build costs without them.

That measurement corrected a guess. The descriptor was thought to allocate a
component per kind, eleven of them, to fill one; it does not -- an empty
component is an empty `Vec`, which allocates nothing, and a whole descriptor is
384 bytes moved by value. The cost is in the products, which is what the three
bounds already hold.

## Which whole-schema operations stay

`open`, `close` and `ensure` stay; `simplify` goes.

The test is the one this page states for a node, applied to a method: a method is
justified by what neither typing nor the algebra can express. `open` and `close`
rewrite **every record a schema declares, at any depth, inside its recursive
definitions** -- the sets they produce are spellable one at a time
(`{"a": int, anything: anything}`), and the traversal is not. They are the
whole-schema operations the contract admits, and they are functions on sets: two
records denoting one set open to one set, which is a law with a test.

`ensure` is `validate(x); return x` and is kept for what it reads as, typed as
the identity it is. That is a judgement about the surface rather than about the
algebra, and it is the only one here.

**And the operator surface stops at `|`.** It is there because Python's own type
syntax writes a union that way -- `int | str` is a union before this library
sees it, and `__ror__` is what lets `None | validator` work -- so answering `|`
is answering the language. `&` and `~` would be second spellings of
`intersection` and `complement`, which the ship-versus-recipe rule refuses:
shorter is not a reason. A validator does not pickle either, and the refusal
says what to send instead ([docs/17-boundaries.md](../17-boundaries.md)).

`simplify` was the lattice normal form of a term the constructors left
un-normalised. Once construction settles the laws -- see the next section -- the
schema a caller holds *is* that normal form, `repr` shows it and `==` compares
it, and `simplify` is the identity under another name. It is deprecated rather
than removed at once, because a method that quietly starts returning its
argument is worse than one that says it is going.

## The limit

The IR is a tree with back edges, not a graph with sharing. Two structurally
equal subtrees are two allocations, and nothing interns them, so the structural
rules carry a work budget instead of a memo table ([02-decision.md](02-decision.md)).

What that costs is smaller than it was. Those rules are the fast path now, not
the whole answer: a shape they run out of budget on is asked again of the sets it
denotes, and the descriptor interns the guard behind each object line, so sharing
exists where a set is built even though it does not exist in the tree.

The limit that remains is the one no memo reaches. A cycle has no finite set
representation here, so a recursive schema is decided by the rules or not at
all, and that fragment -- not the missing sharing -- is what stands between this
procedure and a complete one.
