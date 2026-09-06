---
description: What a runtime validator cannot do, and why each boundary is where it is.
---

# What a validator cannot do

Every entry here is a *deliberate* boundary rather than an unfinished feature.
They come from two places: what a runtime check can observe at all, and what
this library decided not to be. Knowing which is which tells you whether to wait
for it, work around it, or use another tool — so each says which it is.

The neighbouring pages cover two nearby questions: [resource
limits](10-limits.md) is what a *value* cannot make the validator do, and the
[decidability boundary](15-decidability.md) is what the comparison operators
cannot yet prove. This page is about the shape of the product.

## It does not convert

A validator answers whether the object you already hold is a member of a set. It
never copies, coerces, or returns a different value: `"1"` is not an `int`, a
missing key is not filled from a default, and no field is renamed on the way
through. `ensure` returns its argument — the same object, so `is` holds.

This is the product decision the rest of the library rests on. A tool that
converts is answering a different question, and there are good ones; reach for
one when the input is a wire format you need to *become* a domain object.

```python
from valgebra import Validator

numbers = Validator(int)
value = 1
assert numbers.ensure(value) is value  # the same object, not a copy
assert not numbers.is_valid("1")  # a string that looks like one is not one
```

## It does not read the future of a value

The check is a decision about a value at one moment. A value that is mutated
afterwards is not re-checked, and nothing is frozen: validating a list says the
list's elements belonged to the set when they were read.

An observable consequence: a container that changes *during* a walk is reported
rather than silently half-checked ([error model](08-error-model.md)), because
the alternative is an answer about a value that never existed.

## It cannot look inside a callable

`Callable[[int], str]` names a function's domain and return, and neither is
observable at runtime: no check can ask an arbitrary function what it accepts
without calling it, and calling it is not a membership test. A callable schema
is therefore an `isinstance` check for *being* callable, and the argument types
are not checked at all.

```python
from typing import Callable

from valgebra import Validator

callables = Validator(Callable[[int], str])
assert callables.is_valid(len)  # any callable belongs
assert not callables.is_valid(3)
```

## It cannot see a generic's arguments on a value

Python erases them. A `list` at runtime is a list of whatever it holds, so
`list[int]` is checked by reading the elements — which is why it works — while
`Sequence[int]` and `Mapping[str, int]` are refused: those name *protocols*
whose instances need not be enumerable without consuming them, and a check that
consumed an iterator would change the value it was asked about.

There is no variance and no type variable, for the same reason: a `TypeVar` is a
statement about a *relationship between* uses in a program, and a runtime check
sees one value.

## A predicate is a black box

A `Predicate` refinement runs your callable. That makes it as expressive as
Python — and opaque to the algebra: the complement of a predicate is not a set
the library can reason about, two predicates are not compared, and a schema
carrying one is answered conservatively by every relation
([decidability](15-decidability.md)). Predicates are the escape hatch, and using
one is choosing expressiveness over decidability.

The same holds of a class whose metaclass answers `isinstance` by running code:
what it admits is not a set that stands still, so the complement laws are not
applied to it.

## It does not generate, infer, or export

- **No schema inference from values or code.** A schema is written, not guessed.
- **No serialization.** Nothing here turns a value into JSON or back; the JSON
  path *validates during parsing* ([JSON input](07-json.md)) and produces the
  same accept/reject decision as the object path, not a serializer.
- **No JSON Schema import or export.** The two describe different value
  universes — JSON has no `bytes`, no `tuple`, and no class identity — so a
  translation would be lossy in both directions rather than merely absent.
- **No static checking.** valgebra runs; a type checker does not run it. The two
  are complementary, and the [foundations](13-foundations.md) page says where
  their models agree and where they part.

## It does not enforce ordering between separate checks

Two validators are two questions. A constraint that relates *two* values — this
field is greater than that one, this id exists in that table — is not a set of
values in the algebra's sense unless the relation is written inside one schema,
where a predicate can see both. Cross-value invariants belong in the code that
holds both values.

## What follows from all of this

The boundaries are what buys the guarantees. Because nothing converts, an accept
says something about the object you hold rather than about a copy. Because the
algebra is closed, `union`, `intersection` and `complement` of schemas are
schemas, and the relations between them are decidable on a published fragment.
Both would go if the library grew the abilities above.
