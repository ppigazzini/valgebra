# Resource limits

A validator runs against untrusted values, so every recursive descent and every
error-reporting probe is bounded. A pathological input meets a gated limit and is
rejected cleanly; it never overflows the native stack, raises a Python
`RecursionError`, or hangs. The limits bound work driven by the *value* — the
untrusted part. A schema's own size (the width of a union, the number of declared
fields) is written by the developer and is trusted.

## The bounds

- **Schema build depth.** A schema nested past a fixed depth is rejected when the
  validator is compiled, not at validation time. A self-referential class is the
  usual cause; model it with [`recursive`](recursion.md) instead.
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

The worst-case timing of these shapes is measured by the adversarial benchmark
and the bounds are correctness-tested, so each limit is an enforced, exercised
guarantee rather than a comment.
