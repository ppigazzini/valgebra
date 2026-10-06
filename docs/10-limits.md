---
description: Resource limits and the bounds the validator enforces.
---

# Resource limits

A validator runs against untrusted values, so every recursive descent and every
error-reporting probe is bounded. A pathological input meets a gated limit and is
rejected cleanly; on a thread with the stack its platform gives one, it never
overflows the native stack, raises a Python `RecursionError`, or hangs. The limits bound work driven by the *value* — the
untrusted part. A schema's own size (the width of a union, the number of declared
fields) is written by the developer and is trusted.

## The bounds

- **Schema build depth.** The frontend descends at most one level past the
  construction bound while compiling (`MAX_BUILD_DEPTH` in
  `crates/valgebra-py/src/build.rs`) and refuses anything deeper with
  `NotImplementedError`, whether or not it reaches a leaf. A self-referential
  class is the shape that gets there, because its field type names the class;
  model it with [`recursive`](06-recursion.md) instead. An annotation exactly one
  level too deep is refused by the construction bound below, with a
  `ValueError` that names it; a deeper one meets the frontend's bound first.
- **Schema construction size.** Every way of growing a schema — the `Validator`
  constructor, the `|` operator, `union`, `intersection`, `complement`,
  `recursive`, and the record transforms — is bounded at construction, so no sequence of
  calls can build a schema that overflows the stack or exhausts memory on a later
  walk. Three bounds apply, and passing any one raises `ValueError`:
    - **depth** — at most `MAX_SCHEMA_DEPTH` levels of structural nesting,
      128 (a chain built in a loop, such as repeatedly wrapping a validator in
      a set or a union). Every node counts one level, containers and the leaf
      they end in included, so 127 nested lists around an `int` are the deepest
      chain it admits and the 128th list is refused. A refinement counts one
      more on top of whatever it narrows, since the set it denotes is a node of
      its own: pinning a length on each list of such a chain admits 63 refined
      lists and refuses the 64th. Which marker the refinement carries makes no
      difference — the level belongs to the node;
    - **definitions** — at most 128 recursive definitions (a chain of distinct
      `recursive` schemas, which the depth measure alone cannot see because a
      back edge counts as a leaf);
    - **nodes** — at most 100,000 total schema nodes (a shallow but exponentially
      wide schema, such as combining a validator with itself in a loop, which
      doubles its node count each step).

  A real schema stays far under all three. Structural recursion belongs in
  [`recursive`](06-recursion.md), whose back edge does not count toward the depth.

  The three numbers are importable, so code that sizes a schema against them
  reads them rather than repeating them.

```python
from valgebra import MAX_DEFINITIONS, MAX_SCHEMA_DEPTH, MAX_SCHEMA_NODES

assert (MAX_SCHEMA_DEPTH, MAX_DEFINITIONS, MAX_SCHEMA_NODES) == (128, 128, 100_000)
```
- **Value-walk depth.** Two bounds hold the walk inside the stack, and reaching
  either fails with `recursion_limit`: at most 128 levels of **recursive
  unfolding**, and at most 384 levels of **descent** in total. The second is what
  binds for a deep definition, because a recursive definition descends its whole
  body once per level of the value — so the frames a value can ask for are the
  product of the two, not either one. The smallest recursive body, a reference
  under a union around one container such as `recursive(lambda t: union(int,
  [t]))`, opens three levels an unfolding, so it is walked to the unfolding
  bound: every value nested within it is a member.

    The walk fits a **1 MiB** thread stack, the least any thread CPython
  creates has. The shipped wheels are built with profile-guided optimization,
  whose inlining makes a level cost 2.5 to 3 KiB on the dict and union shapes,
  and the dearest walk, explaining a value refused at the bound, needs between
  832 and 896 KiB there. The release smoke runs the deepest walks on a 1 MiB
  thread on every wheel it ships. A thread given less can run out before the bound does, and the
  process ends with it; a thread an embedding host starts at the 512 KiB
  pthread default on macOS is one. Give a thread that validates deep values
  1 MiB or more. This holds on both the object path and the JSON path; an
  over-deep JSON document is rejected by the parser as `json_invalid`.

    The parser has a bound of its own — a couple of hundred levels of arrays and
    objects — and it sits **between** the two: wider than the unfolding bound and
    narrower than the descent one. A document therefore has three regions rather
    than two. Inside the unfolding bound it is a member. Past it and inside the
    parser's, the walk is what refuses, and the code is `recursion_limit`. Past
    the parser's, the text stops being a document before the walk sees it, and
    the code is `json_invalid`. The descent bound is not reachable through a
    document at all, because the parser refuses first — it binds on the object
    path, where there is no parser. `tests/test_adversarial_bounds.py` holds the
    three regions and their order, which is what keeps this paragraph a
    description of the tree rather than of two numbers that have since moved
    past each other.
- **Relation depth.** A relation -- `relation_to`, `is_subtype_of`,
  `is_equivalent`, `is_disjoint_from`, `is_empty` -- holds at most 512 levels of
  its own recursion, and past them answers undecided, which a relation may
  always answer. Two recursive schemas whose bodies nest 100 and 99 lists around
  the back edge prove nothing about each other until the two cycles realign,
  9,900 levels down, and a long chain of definitions asks its emptiness as deep:
  each overflowed the stack before the bound. A level costs about 1 KiB on the
  shipped wheels, so the deepest relation fits in half of a 1 MiB thread, and
  the release smoke runs it there.
- **Self-reference.** A value that contains itself is caught by an
  object-identity guard and fails with `recursion_loop` rather than looping
  forever.
- **Union error reporting.** When a value misses a wide union, the error report
  is bounded in two independent ways: it searches only a bounded number of
  branches for the closest match, and `expected` names only a bounded number of
  **labels** before truncating with `...`. The two counts differ — a branch that
  is itself a union, such as a wide `Literal[...]`, contributes one branch and
  many labels — so each carries its own bound. With `fail_fast` the explanation
  costs a fail-fast walk of each branch it searches, whatever the size of the
  value. Without it every failure of the chosen branch is reported, so the
  report grows with the value; a service answering untrusted input with the
  report asks for it with `fail_fast`.

## Rejection is clean, not catastrophic

```python
from valgebra import ValidationError, Validator, recursive, union

schema = Validator(recursive(lambda j: union(int, [j])))

# A value nested far past the walk depth: a clean error, not a crash.
deep: object = 0
for _ in range(5000):
    deep = [deep]
assert not schema.is_valid(deep)
raised = False
try:
    schema.validate(deep)
except ValidationError as error:
    raised = True
    assert error.code == "recursion_limit"
assert raised

# A value that contains itself: caught as a loop.
cyclic: list[object] = []
cyclic.append(cyclic)
assert not schema.is_valid(cyclic)

# An over-deep JSON document: rejected by the parser.
assert not schema.is_valid_json("[" * 5000 + "1" + "]" * 5000)
```

Growing a schema in an unbounded loop is stopped at construction, before the
growing schema can overflow the stack or exhaust memory on its next check:

```python
from valgebra import Validator

composed: Validator = Validator(int)
raised = False
try:
    for _ in range(1000):
        composed = Validator([composed])
except ValueError as error:
    raised = True
    assert "too deep" in str(error)
assert raised
```

The worst-case timing of these shapes is measured by the adversarial benchmark
and the bounds are correctness-tested, so each limit is an enforced, exercised
guarantee rather than a comment.
