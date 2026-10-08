---
description: The recursive fixpoint for self-referential schemas.
---

# Recursive schemas

`recursive` ties a fixpoint: the builder it receives is given a placeholder standing
for the schema being defined, and returns the body. The recursive reference must
occur under a structural constructor (a list, tuple, set, dict, record, or
object) so membership stays decidable; a non-contractive body is rejected when
the validator is built.

## A recursive JSON value

```python
from valgebra import recursive, union

json_value = recursive(
    lambda j: union(None, bool, int, float, str, [j], {str: j}),
)
assert json_value.is_valid({"a": [1, "x", {"b": None}], "c": [True, 3.5]})
assert not json_value.is_valid({"a": object()})
```

## A tree, then composed

A `recursive` schema is an ordinary validator and composes like any other:

```python
from valgebra import Validator, recursive

tree = recursive(lambda t: {"value": int, "left?": t, "right?": t})
assert tree.is_valid({"value": 1, "left": {"value": 2}})

forest = Validator([tree])
assert forest.is_valid([{"value": 1}, {"value": 2, "right": {"value": 3}}])
```

## A `type` alias is a fixpoint too

A PEP 695 alias that names itself is the standard typing spelling of a recursive
schema, and it builds one. The alias is the binder: it is reached again while its
own body is read, and the schema it builds is the schema the explicit call
builds — the two are one set, and `is_equivalent` says so.

```python
from valgebra import Validator, recursive, union

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None

alias = Validator(Json)
assert alias.is_valid({"a": [1, "x", {"b": None}]})
assert alias.is_equivalent(
    recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
)
```

Mutual recursion works the same way, since each alias binds its own fixpoint:
`type Branch = list[Leaf]` beside `type Leaf = int | Branch` is two definitions
naming each other. An alias that names itself **outside** a structural
constructor — `type Bad = int | Bad` — is refused when the validator is built,
for the reason the next section gives: it denotes no set a value settles.

The syntax is Python 3.12 and later. On 3.10 and 3.11, write the fixpoint with
`recursive`.

## A generic alias, applied

A generic alias is read applied to its arguments: `Pair[int]` from
`type Pair[T] = tuple[T, T]` is `tuple[int, int]`, the body with the argument in
place of its parameter. A parameter given no argument takes its default, which
may name an earlier parameter, and an alias whose every parameter has one reads
bare as its defaults. The runtime counts no arguments, so a surplus or a
missing one is refused by the alias's name, and so is a bare alias with a
parameter no default stands for, since PEP 695 reads that as `Any`, "which is
rarely the intent". A parameter's bound or constraints are a checker's to hold
and are not read: `Bounded[str]` from `type Bounded[T: int] = list[T]` is
`list[str]`. A `ParamSpec` or a `TypeVarTuple` parameter stands for no single
type and is refused.

A recursive generic alias ties one fixpoint per list of arguments it meets.
`Tree[int]` is a fresh object at every read, and the body it substitutes to
names an equal `Tree[int]`, which is the back edge:

```python
from valgebra import Validator

type Pair[T] = tuple[T, T]
type Tree[T] = T | list[Tree[T]]

assert Validator(Pair[int]) == Validator(tuple[int, int])
ints = Validator(Tree[int])
assert ints.is_valid([1, [2, [3]]])
assert not ints.is_valid([1, ["a"]])
```

An alias that applies itself to an argument nesting its own parameter --
`type Nest[T] = T | list[Nest[list[T]]]` -- meets a new list of arguments at
every unfolding, so no fixpoint ties it, and it is refused before it is built.
mypy refuses the same shape at its definition.

## Why classes need it

A class whose own type appears in a field is recursive in the same way, and a
class definition has no place to tie the fixpoint, so building a validator for
one is refused with a message pointing here -- a dataclass, a `NamedTuple` and a
`TypedDict` alike. `recursive` spells the shape as containers instead: the
record below admits the nested dicts a recursive `TypedDict` describes, and
refuses every instance of the dataclass, which is not a dict. No form spells
"an instance of `Node` whose `next` is the fixpoint", so a self-referential
dataclass or `NamedTuple` cannot be deep-checked here.

```python
from __future__ import annotations

from dataclasses import dataclass

from valgebra import recursive


@dataclass
class Node:
    value: int
    next: Node | None = None


node = recursive(lambda n: {"value": int, "next?": n})
assert node.is_valid({"value": 1, "next": {"value": 2}})
assert not node.is_valid(Node(1, Node(2)))  # an instance is not a dict
```

## Soundness guarantees

Recursion is bounded so it always terminates cleanly:

- A value that **contains itself** is rejected with `recursion_loop` rather than
  looping forever (an object-identity guard).
- A value nested **past a fixed depth** fails with `recursion_limit` rather than
  overflowing the native stack: 128 levels of unfolding, and 384 levels of
  descent in total, which is the bound a deep definition body reaches first
  ([limits](10-limits.md)). Under `recursive(lambda t: union(int, [t]))`, 127
  lists nested around an `int` are a member and 128 are not: the `int` at the
  bottom is the 128th unfolding.
- A **non-contractive** body — one whose recursive reference is not under a
  structural constructor — is rejected when the validator is built, not at
  validation time.

```python
from valgebra import recursive, union

cyclic: list[object] = []
cyclic.append(cyclic)
assert not recursive(lambda s: union(int, [s])).is_valid(cyclic)  # recursion_loop
```

## Recursion in the decision procedure

Recursive schemas also take part in [subtyping, equivalence, and
emptiness](15-decidability.md). Equirecursive schemas compare at their greatest
fixpoint — a coinductive comparison that assumes a goal already being proven on
the current path — so a recursive schema is a subtype of itself and two
structurally identical recursive schemas are equivalent, and a recursive schema
with no base case is detected as uninhabited.

These are views of one definition, not separate definitions. *Membership* unfolds
the definition against a finite value — a value is in the set when its finite
unfolding matches. That is not a choice between fixpoints: values are finite, so
each unfolding asks about strictly smaller values and the set is *defined* by
that induction. Guardedness is what makes it well founded, which is why a
non-contractive body is refused rather than resolved somehow. *Inclusion* uses
the greatest fixpoint coinductively, which is the sound way to relate two such
definitions without unfolding forever. On the finite values the two agree, so a
subtype result never contradicts membership.

*Emptiness* asks the opposite question and takes the **least** fixpoint: a
reference reached again while resolving it demands an infinite unfolding, and no
finite value supplies one, so that occurrence is uninhabited. This is why a
mandatory self-reference with no base case is empty — under the greatest fixpoint
it would be inhabited by infinite trees, and valgebra validates finite Python
values. Contractivity is what keeps the two consistent: over the finite values a
guarded definition names one set, so what inclusion relates and what emptiness
counts are the same set.

```python
from valgebra import Validator, recursive, union

json_value = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
assert Validator(json_value).is_subtype_of(json_value)  # reflexive across the fixpoint
assert recursive(lambda t: {"value": int, "next": t}).is_empty()  # no base case
assert not recursive(
    lambda t: union(None, {"next": t})
).is_empty()  # a base case exists
```
