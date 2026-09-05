# The schema frontend

`crates/valgebra-py/src/build.rs` turns a Python schema description into the IR.
It is the only place a Python object becomes a pooled index, and the only place
that decides what an annotation means.

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

## Two rejections that belong at compile time

**A typing construct that carries no runtime value** — a `TypeVar`, a
`ParamSpec`, a bare `Final` or `ClassVar` — is refused rather than interned as a
literal. Interning it would produce a schema that admits only objects equal to
the TypeVar, which is almost nothing, and the user would see a validation failure
instead of a compile error.

**A zero divisor.** `MultipleOf(0)` is unsatisfiable and checking it would divide
by zero at validation time, so the error is raised where the schema is built.

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
