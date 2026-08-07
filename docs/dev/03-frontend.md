# The schema frontend

`crates/valgebra-py/src/build.rs` turns a Python schema description into the IR.
It is the only place a Python object becomes a pooled index, and the only place
that decides what an annotation means.

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
