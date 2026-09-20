# The schema frontend

`crates/valgebra-py/src/build.rs` turns a Python schema description into the IR.
It is the only place a Python object becomes a pooled index, and the only place
that decides what an annotation means.

The file is the dispatch, the pool and the guard; each section below that names
a *surface* is a module beside it:

| module | the section it follows |
|---|---|
| `build/refine.rs` | How `Annotated` metadata is read |
| `build/classes.rs` | What a class declares |
| `build/generics.rs` | What a parametrized form says |

## How `Annotated` metadata is read

`parse_constraint` reads a marker by **attribute protocol**, never by name: the
frontend imports `typing`, `types`, `enum`, `collections.abc` and `builtins`,
and never `annotated_types`. A marker carrying `pattern`, `min_length`,
`max_length` or `multiple_of` contributes the matching constraint, so any
library's marker of that shape works.

**A class is never a marker**, and is refused before any attribute is read. A
marker *class* exposes descriptors where an instance exposes values: `at.Ge(0)`
carries `ge = 0`, while `at.Ge` carries the slot descriptor that reads it, and
taking that for a bound builds a comparison no value is ordered against. Calling
one is the same trap a step later — `Kilograms(1.5)` constructs a unit marker
rather than answering whether `1.5` belongs. Either way the schema ends up
denoting nothing, so a class is metadata this frontend does not recognise.

The rest are read in this order:

1. a marker that is itself callable — the marker becomes the predicate;
2. otherwise, a marker carrying a callable `func` — its `.func` becomes the
   predicate.

Metadata matching neither is ignored, which the typing spec requires of any
consumer for metadata it does not recognise.

**A constraint no value of the base can answer is refused where it is
written.** Reading a length off an `int` raises, and the walk reads a raise as
a non-member, so `Annotated[int, MinLen(1)]` would compile to a set that
admits nothing and says nothing about why. The rule that says which bases can
answer which constraints is the core's `carries` module, with three answers:
every value can, no value can, or the base does not say. Only the second is a
refusal. A union answers for its members, so `Annotated[int | str,
MinLen(1)]` is the non-empty strings; an intersection, a complement and a
reference do not say; and an order bound is asked about the pair, since
Python orders a value only within its own group. The laws' buildable fragment
draws its refinements through the same module, so the pairs the generator
draws are the pairs the frontend builds by construction rather than by two
tables kept alike.

**A name is a handle, and an absence is not an exception.** Every attribute the
protocol asks for is asked by an interned `PyString` the interpreter already
holds, because text would be decoded into a fresh string and hashed before the
lookup could begin, once per name per marker. And a marker carries one or two
of the ten names and not the rest, so absence is the common answer, and giving
it by *raising* costs an exception built, thrown and dropped — four hundred of
them to compile fifty fields. Which names a marker can carry is a property of
its type (`Ge` is a `slots` dataclass, so `Ge.ge` is the descriptor
that reads the slot and `Ge.gt` does not exist), so the type is read once and
its answer kept, and a marker that keeps its values in a dictionary of its own
is read from that dictionary. A type with a `__getattr__` hook answers for
names neither holds, and is asked for everything.

**Callability is how `annotated_types` tells its two marker shapes apart**, so
the order is its rule rather than a heuristic. `Not` defines `__call__` because
calling is what applies the negation; `Predicate` deliberately does not, and
carries its callable on `.func`. Both carry a `.func`, so reading that attribute
first would strip `Not` of its negation — and a `functools.partial` of its bound
arguments, since it has one too.

A class is excluded from the second arm although it is callable, because calling
one **constructs** rather than asks. `Kilograms(1.5)` builds a unit marker; it
does not answer whether `1.5` belongs, and a constructor that rejects the value
would leave the schema uninhabited. Every other callable — a function, a lambda,
a bound method, an object with `__call__` — is asked.

No other library in the ecosystem reads a bare callable as a constraint:
pydantic requires `AfterValidator`, beartype `Is[...]`, msgspec `Meta(...)`, and
`annotated_types` supplies `Predicate`. The typing spec leaves the choice to the
consumer, so this arm is a deviation rather than a defect — and the class
exclusion is where the deviation stops being ergonomic and starts being a trap.

### What the order buys

Three markers turn on it, and the wrong order is wrong for two:

| marker | callable | `.func` | read as |
|---|---|---|---|
| `annotated_types.Not(f)` | yes | yes | the marker — calling it applies the negation |
| `functools.partial(eq, 1)` | yes | yes | the marker — calling it supplies the bound arguments |
| `annotated_types.Predicate(f)` | no | yes | `.func` — the marker raises if called |

`tests/test_refinements.py` holds one row each, because a marker of a shape the
suite never receives is a defect nothing reports: `Not` inverted every verdict
under it while the whole suite passed.

## The dispatch order is load-bearing

`build_schema` tries forms in an order chosen for the common path, not for
symmetry:

1. `None` — before anything, since it is the most common leaf.
2. `typing.Any` — before the type-object branch, because on 3.11+ `Any` is
   itself a class and would otherwise be taken for an ordinary type.
3. `Never`/`NoReturn` — the lattice bottom, absent on older Pythons and skipped
   there.
4. **A plain type or class** — a scalar, `object`, a TypedDict, a dataclass, an
   enum, a protocol. Taken before the typing introspection below because a type
   never has a typing origin, so this skips a `get_origin` call per scalar node.
5. `Annotated[T, ...]` — the refinement metadata.
6. Anything with a typing origin — `list[int]`, `dict[K, V]`, `tuple[...]`,
   `X | Y`, `Literal`.
7. PEP 695 aliases, `NewType`, native list and dict literals.
8. An already-compiled validator, whose pool is interned into this one.
9. Anything else — an exact-value literal.

Moving a branch earlier is a behaviour change, not a refactor. `Any` above the
type branch is the sharp one.

## What a class declares

Dispatch step 4 takes any plain type, and what it builds depends on what the
class *says about itself* rather than on what it is called. The order is the
order of the questions, and each is asked of an attribute the runtime fills in:

1. **A builtin scalar** — `bool`, `int`, `float`, `str`, `bytes`, `NoneType` —
   is its own node.
2. **A bare container class** is its kind: `list` admits every list, which is
   the set `list[object]` names and the set the typing spec assigns an
   unparameterised generic. Read as an `isinstance` atom it would be a
   different sort of thing from the sequence node beside it, and neither
   spelling would be decided below the other.
3. **`object`** is the lattice top, and a bare `Union`/`Optional` — a class on
   some Pythons — is refused, because it is a form rather than a value.
4. **A `TypedDict`** declares itself by carrying `__required_keys__`, and
   becomes a keyed map. Its fields come from `get_type_hints`, not from the raw
   `__annotations__`: under `from __future__ import annotations` those are
   strings, so a `NotRequired[...]` is invisible to the class's own key sets and
   every optional key compiled required. A qualifier on the resolved hint wins
   over the key sets, and a qualifier may wrap another.
5. **An enum** is an instance check against the enumeration class.
6. **A dataclass or a `NamedTuple`** is an instance check *plus* a deep check
   of each declared field, so a value of the right class with a field of the
   wrong type is not a member. A field the hints do not carry is unannotated —
   a `collections` namedtuple's are — and the class's own instance check is the
   whole of it.
7. **A `Protocol`** validates by `isinstance` and must be `@runtime_checkable`
   to be a schema at all; one that is not is refused rather than silently
   admitting everything.
8. **Any other class** names its instances: the remaining builtins, the
   `collections.abc` ABCs, and every user class, uniformly.

`dataclasses.is_dataclass` is the one question that costs an import, so it is
asked of a handle held after the first class that asks and *only* after one
asks: importing `dataclasses` pulls `inspect`, `copy` and `functools` in with
it, and the tracked objects they leave behind are walked by every later garbage
collection.

## What a parametrized form says

Dispatch step 6 takes anything with a typing origin, and reads the origin
before the arguments. The origins are compared by identity against the forms
resolved once at import — `typing` is imported once, not once per node — and a
form this frontend does not know is a refusal rather than a guess.

`Union` and `X | Y` are the same origin in two spellings and build the same
node. `Literal` interns each argument as a constant, and refuses a list, a dict
or a set: the typing spec allows `None`, an enum member, or an `int`, `bool`,
`str` or `bytes` value, and a container there would be read as a schema of its
own rather than as a constant. The refusal names the spelling that was meant.

A `tuple` reads its arguments as a *shape*: `tuple[int, str]` is a fixed
sequence of two, `tuple[int, ...]` is a homogeneous one, and `tuple[()]` is the
empty tuple. `Unpack[Ts]` and `*tuple[int, ...]` are the same unpacking in two
spellings. A shape is a fixed prefix and then a repeating tail, so nothing may
follow the tail and a tuple cannot begin with `...`; both are refused by naming
the spelling that was meant.

Five origins are read as containers — `list`, `set`, `frozenset`, `dict` and
`tuple` — and their arguments become element and key types. A parametrized
`collections.abc` generic is **refused**, not read: `Sequence[int]` names a
protocol whose instances a check cannot enumerate without consuming them, and
the bare abstract class is available as an `isinstance` atom instead
([the API page](../16-api.md) records the refusal).
`dict[K, V]`'s two are read by name rather than by position, because the two
transposed is `dict[V, K]`, which typechecks and validates real values.

The two **native literals** are read here too, because they answer the same
question: `[A, B]` is the fixed-length list, which `typing` cannot spell, and a
`dict` literal is a record — string keys are named fields, with `"key?"` for an
optional one, and any other key is a schema governing the rest. A tuple literal
and a set literal are refused rather than accepted, since `tuple[A, B]` and
`set[T]` already spell them and two spellings for one set is a fork. A key
narrowed by a constraint is refused where "Three rejections that belong at
compile time" says why.

## Where an index acquires its meaning

One pool holds four kinds of object, so the slot means nothing until the frontend
says what it pooled. That decision is at the mint, not at the read:

```rust
lits.intern_const(obj)      // the constant of a typed singleton
lits.intern_class(ty)       // an isinstance atom
lits.intern_operand(bound)  // a comparison or multiple-of operand
lits.intern_predicate(func) // a user callback
```

Each returns its own index type. The private `intern` beneath them deduplicates
by object identity through an address-keyed map, so compiling a wide
`Literal[...]` or merging many validators stays linear rather than quadratic. The
address key is stable because every interned value is kept alive by the pool.

## Three rejections that belong at compile time

**A typing construct that carries no runtime value** — a `TypeVar`, a
`ParamSpec`, a bare `Final` or `ClassVar` — is refused rather than interned as a
literal. Interning it would produce a schema that admits only objects equal to
the TypeVar, which is almost nothing, and the user would see a validation failure
instead of a compile error.

**A zero divisor.** `MultipleOf(0)` is unsatisfiable and checking it would divide
by zero at validation time, so the error is raised where the schema is built.

**A map key narrowed by a constraint.** A clause's key says which keys it
governs, and a map reads that as whole *kinds* of key or as the constants a
`Literal` names. `Annotated[str, MinLen(2)]` is neither: it names part of a kind,
and two such clauses can overlap without either containing the other. Overlapping
key domains are a different theory from the one these maps are built on --
Castagna's §4.5 says that with them "there would not be any difference between
record types and an intersection of function types" -- and this library reads
overlapping clauses disjunctively, which answers a question the two theories do
not agree on. The narrowing is the one user-visible removal of the map work.

The same principle governs the regex: the pattern is compiled and anchored at
build time, so an invalid expression fails at construction rather than at first
validation.

## Recursion needs an explicit fixpoint

A class whose own type appears in its fields is recursive, and the frontend
refuses to chase it — a depth guard bounds `build_schema` and returns a message
naming `recursive(...)` as the way to express it. The bound exists so that case
fails cleanly instead of overflowing the native stack.

## The limit

**The frontend decides meaning; nothing checks it against the typing spec
mechanically.** A form compiled to the wrong node is a defect no gate in this
tree catches — the differential lane compares against pydantic-core and
jsonschema over the fragment where the semantics agree, with the divergences
enumerated in `tests/test_differential.py`, and that is the closest thing to an
external judge.

**A transposition inside one call is not closed by a type.** `Schema::mapping`
takes a key schema and a value schema, both `Schema`; `dict[K, V]` compiled with
them swapped typechecks and validates real values.
[06-type-design.md](06-type-design.md) records that as the sharpest residual
hazard in the tree.
