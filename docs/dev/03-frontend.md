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

1. a marker carrying a callable `func` — its `.func` becomes the predicate;
2. otherwise, a marker that is itself callable — the marker becomes the
   predicate.

Metadata matching neither is ignored, which the typing spec requires of any
consumer for metadata it does not recognise.

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

### What that order costs

The order is wrong for every marker that is *both* callable and carrying
`.func`, and two common ones are:

| marker | means | read as | verdict |
|---|---|---|---|
| `annotated_types.Not(f)` | the values where `f` is false | `f` | **every verdict inverted** |
| `functools.partial(eq, 1)` | equals 1 | `operator.eq` | raises on one argument, so the schema denotes **∅** |

`annotated_types` distinguishes the two cases itself: `Not` defines `__call__`
so a consumer calls it, and `Predicate` deliberately does not and carries its
callable on `.func`. Reading `.func` first therefore inverts `Not` and empties
`partial`, and is correct only for `Predicate`.

**Swapping the two arms does not go in**, and not because the swap looks wrong:
it addresses both rows, by the discriminator `annotated_types` encodes. It stays
out because it rests on an open question:

> Is a bare callable in metadata *meant* to be a constraint?

No shipped page says it is, and a downstream package builds every one of its
predicates on it. The swap promotes that unstated rule to the frontend's
**first** test, so the swap and the question are one decision. Landing the swap
alone answers the question by accident, and an accident that ships is a
contract.

Check the first row against a build of the current source — `True False`, where
`Not` means `False True`:

```bash
maturin develop --uv
uv run python -c "import annotated_types as at; from typing import Annotated; \
from valgebra import Validator; v = Validator(Annotated[int, at.Not(lambda x: x % 2 == 0)]); \
print(v.is_valid(2), v.is_valid(3))"
```

Nothing in the suite covers it. The tests reach for `Ge`, `Gt`, `Le`, `Lt`,
`Interval`, `Len`, `MinLen`, `MaxLen`, `MultipleOf` and `Predicate`, and the
whole suite passes with `Not` inverted — the gap is in the markers the suite is
given, not in what it checks.

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
lits.intern_class(ty)       // an isinstance atom, or an attribute record
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
