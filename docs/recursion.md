# Recursive schemas

`lazy` ties a fixpoint: the builder it receives is given a placeholder standing
for the schema being defined, and returns the body. The recursive reference must
occur under a structural constructor (a list, tuple, set, dict, record, or
object) so membership stays decidable; a non-contractive body is rejected when
the validator is built.

## A recursive JSON value

```python
from valgebra import lazy, union

json_value = lazy(
    lambda j: union(None, bool, int, float, str, [j], {str: j}),
)
assert json_value.is_valid({"a": [1, "x", {"b": None}], "c": [True, 3.5]})
assert not json_value.is_valid({"a": object()})
```

## A tree, then composed

A `lazy` schema is an ordinary validator and composes like any other:

```python
from valgebra import lazy, validator

tree = lazy(lambda t: {"value": int, "left?": t, "right?": t})
assert tree.is_valid({"value": 1, "left": {"value": 2}})

forest = validator([tree])
assert forest.is_valid([{"value": 1}, {"value": 2, "right": {"value": 3}}])
```

## Why classes need it

A class whose own type appears in a field is recursive in the same way, but a
class definition has no place to tie the fixpoint. Compiling such a class
directly is rejected with a message pointing here; model it with `lazy` instead:

```python
from valgebra import lazy

# instead of a self-referential @dataclass Node, write the shape with lazy:
node = lazy(lambda n: {"value": int, "next?": n})
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
from valgebra import lazy, union

cyclic = []
cyclic.append(cyclic)
assert not lazy(lambda s: union(int, [s])).is_valid(cyclic)  # recursion_loop
```
