---
description: Resource limits and the bounds the validator enforces.
---

# Resource limits

A validator runs against untrusted values, so every recursive descent and every
error-reporting probe is bounded. A pathological input meets a gated limit and is
rejected cleanly; it never overflows the native stack, raises a Python
`RecursionError`, or hangs. The limits bound work driven by the *value* — the
untrusted part. A schema's own size (the width of a union, the number of declared
fields) is written by the developer and is trusted.

## The bounds

- **Schema build depth.** A schema nested past 100 levels is rejected when the
  validator is compiled, not at validation time. A self-referential class is the
  usual cause; model it with [`recursive`](recursion.md) instead.
- **Schema construction size.** Every way of growing a schema — the `Validator`
  constructor, the `|` operator, `union`, `intersection`, `complement`,
  `recursive`, and `simplify` — is bounded at construction, so no sequence of
  calls can build a schema that overflows the stack or exhausts memory on a later
  walk. Three bounds apply, and passing any one raises `ValueError`:
    - **depth** — at most 128 levels of structural nesting (a chain built in a
      loop, such as repeatedly wrapping a validator in a list);
    - **definitions** — at most 128 recursive definitions (a chain of distinct
      `recursive` schemas, which the depth measure alone cannot see because a
      back edge counts as a leaf);
    - **nodes** — at most 100,000 total schema nodes (a shallow but exponentially
      wide schema, such as combining a validator with itself in a loop, which
      doubles its node count each step).

  A real schema stays far under all three. Structural recursion belongs in
  [`recursive`](recursion.md), whose back edge does not count toward the depth.
- **Value-walk depth.** A value nested past a fixed depth fails with
  `recursion_limit` rather than recursing into the native stack. This holds on
  both the object path and the JSON path; an over-deep JSON document is rejected
  by the parser as `json_invalid`.
- **Self-reference.** A value that contains itself is caught by an
  object-identity guard and fails with `recursion_loop` rather than looping
  forever.
- **Closest-branch probe.** When a value misses a wide union, the error report
  searches only a bounded number of branches for the closest match, so building
  the explanation stays bounded regardless of how the value is shaped.

## Rejection is clean, not catastrophic

```python
from valgebra import ValidationError, Validator, recursive, union

schema = Validator(recursive(lambda j: union(int, [j])))

# A value nested far past the walk depth: a clean error, not a crash.
deep = 0
for _ in range(5000):
    deep = [deep]
assert not schema.is_valid(deep)
try:
    schema.validate(deep)
except ValidationError as error:
    assert error.code == "recursion_limit"

# A value that contains itself: caught as a loop.
cyclic = []
cyclic.append(cyclic)
assert not schema.is_valid(cyclic)

# An over-deep JSON document: rejected by the parser.
assert not schema.is_valid_json("[" * 5000 + "1" + "]" * 5000)
```

Growing a schema in an unbounded loop is stopped at construction, before the
growing schema can overflow the stack or exhaust memory on its next check:

```python
from valgebra import Validator

composed = Validator(int)
try:
    for _ in range(1000):
        composed = composed | str
except ValueError as error:
    assert "too deep" in str(error)
```

The worst-case timing of these shapes is measured by the adversarial benchmark
and the bounds are correctness-tested, so each limit is an enforced, exercised
guarantee rather than a comment.
