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
from valgebra import recursive, Validator

tree = recursive(lambda t: {"value": int, "left?": t, "right?": t})
assert tree.is_valid({"value": 1, "left": {"value": 2}})

forest = Validator([tree])
assert forest.is_valid([{"value": 1}, {"value": 2, "right": {"value": 3}}])
```

## Why classes need it

A class whose own type appears in a field is recursive in the same way, but a
class definition has no place to tie the fixpoint. Compiling such a class
directly is rejected with a message pointing here; model it with `recursive` instead:

```python
from valgebra import recursive

# instead of a self-referential @dataclass Node, write the shape with recursive:
node = recursive(lambda n: {"value": int, "next?": n})
assert node.is_valid({"value": 1, "next": {"value": 2}})
```

## Soundness guarantees

Recursion is bounded so it always terminates cleanly:

- A value that **contains itself** is rejected with `recursion_loop` rather than
  looping forever (an object-identity guard).
- A value nested **past a fixed depth** fails with `recursion_limit` rather than
  overflowing the native stack.
- A **non-contractive** body — one whose recursive reference is not under a
  structural constructor — is rejected when the validator is built, not at
  validation time.

```python
from valgebra import recursive, union

cyclic = []
cyclic.append(cyclic)
assert not recursive(lambda s: union(int, [s])).is_valid(cyclic)  # recursion_loop
```

## Recursion in the decision procedure

Recursive schemas also take part in [subtyping, equivalence, and
emptiness](decidability.md). Equirecursive schemas compare at their greatest
fixpoint — a coinductive comparison that assumes a goal already being proven on
the current path — so a recursive schema is a subtype of itself and two
structurally identical recursive schemas are equivalent, and a recursive schema
with no base case is detected as uninhabited.

These are two views of one definition, not two definitions. *Membership* unfolds
the unique guarded (contractive) fixpoint against a finite value — a value is in
the set when its finite unfolding matches. *Comparison* uses the greatest
fixpoint coinductively, which is the sound way to relate two such definitions
without unfolding forever. On the inhabitants the two agree: the greatest
fixpoint admits exactly the values the guarded unfolding accepts, so a subtype
result never contradicts membership.

```python
from valgebra import recursive, union, Validator

json_value = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
assert Validator(json_value).is_subtype_of(json_value)  # reflexive across the fixpoint
assert recursive(lambda t: {"value": int, "next": t}).is_empty()  # no base case
assert not recursive(lambda t: union(None, {"next": t})).is_empty()  # a base case exists
```
