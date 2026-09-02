# The membership walk

`crates/valgebra-py/src/check/walk.rs` decides whether a value belongs to a
schema's set. It is where soundness is decided, and it is the file to read first
when an accept looks wrong.

## One walk, two input paths, three modes

There is **one** `member` function. It was two — a fast one and an explaining one
— and the two drifted; fusing them removed the class of defect where the answer
depended on which walk ran.

It runs over a `Value`, which is either a borrowed Python object or a borrowed
parsed JSON value. That is what keeps the object path and the in-place JSON path
membership-equivalent by construction rather than by a test that compares them.

`WalkMode` names what the walk is for:

| Mode | Reports |
|---|---|
| `Fast` | membership only: nothing allocated, no path built, every composite short-circuits |
| `Explain` | a violation for each independent failure — every field, element and entry |
| `ExplainFailFast` | the first violation only |

Three modes, and the type says three. The pair of independent booleans this
replaced admitted a fourth combination that no caller produced. The
discriminants are ordered so both predicates the walk asks per node are a single
comparison, and a test pins both over every variant — that order is load-bearing
and measured, not cosmetic ([06-type-design.md](06-type-design.md)).

## The comparison-raises policy

Membership reads a value through Python operations that can raise: `__eq__` for a
literal, a rich comparison for a bound, `isinstance` for a class, `getattr` for
an attribute, `__mod__` for a multiple-of, `__len__` for a length.

**One rule across every such site: a value whose comparison, instance check or
attribute access raises an ordinary exception is a non-member.** A value that
cannot answer "are you in this set?" is not in it. This matches pydantic-core.

The one carve-out is a **user predicate**, whose raised error surfaces as a
distinct `predicate_error` rather than folding, so a buggy predicate is visible
instead of silently rejecting everything.

## The one error never folded

A **fatal interpreter signal** is not an answer to "are you in this set?" — the
interpreter is unwinding. `is_fatal` classifies it as two disjoint cases:

- a base exception that is not an ordinary exception: `KeyboardInterrupt`,
  `SystemExit`, `GeneratorExit`;
- `MemoryError` and `RecursionError`, which **are** ordinary exceptions, so the
  base-exception test alone misses them, and which mean the interpreter cannot
  continue.

The first such signal is recorded; the walk then short-circuits — every later
`member` call returns at once — and the entry point re-raises it. A `Cell` mirror
of "has a signal been seen" is read per node with a plain load, so the fast path
does not take a `RefCell` borrow on every step.

Each disjunct needs its own test case: a mutation collapsing the classifier to
one of them is invisible to a corpus that only raises `KeyboardInterrupt`.

## The union reports the closest branch

When no branch of a union matches, dumping every branch's errors buries the one
the value was closest to. The walk instead reports the branch that **descended
furthest** into the value before failing, measured as the greatest path depth
past the union's own location, with a tie keeping the earliest branch so the
choice is deterministic. When no branch makes progress — every branch a flat type
mismatch — it falls back to one union error.

The probe aggregates **regardless of fail-fast**, so the whole of the closest
branch is reported even to a caller that asked to stop at the first violation.
That is the point: the caller asked for less noise, not for less of the one
branch that matters.

## What is precomputed, and what correctness may not depend on

Three per-validator indexes are built once on first use and keyed by the address
of a node's own buffer:

- declared-field lookups per record;
- value sets for unions whose members are all literals;
- compiled patterns per regex source.

**Correctness never depends on one being present.** A node absent from an index
falls back to building the map, scanning the branches, or recompiling the
pattern. The literal-union plan is consulted only on the membership path and only
for an exact int or str — an explain walk, a non-literal union, another value
type and a JSON value all fall through to the linear scan, which stays the one
source of truth for behaviour.

## Recursion is guarded by value identity

`check_ref` records `(object id, definition index)` on the path. A value that
contains itself fails with `recursion_loop` rather than looping, and a chain
deeper than the unfolding bound fails with `recursion_limit` rather than
descending.

Counting unfoldings is not counting frames, and the walk needs both. Every level
takes one native frame, and an unfolding descends the *whole definition body*, so
a body at the construction depth bound turns 128 unfoldings into thousands of
frames — more stack than a thread has. `Ctx::descend` therefore counts the levels
themselves and refuses past `MAX_WALK_DEPTH`, with the same `recursion_limit` the
unfolding bound gives, because it is the same fact about the value. The count is
a depth rather than a total because the level is released when the frame that
took it returns, so a wide value pays for its widest child and not for all of
them.

## The limit

**JSON has no tuple.** `json.loads` produces a list, so the JSON path has no
tuple arm and a tuple schema rejects an array whatever its elements are. It is
the one place the two input paths deliberately decide differently, and it is
pinned by a case.

**The walk is where an accept can be wrong, and a line floor does not see that.**
Its adequacy is measured by mutation, over a value corpus in the file itself
under the embedded-interpreter feature; [08-testing.md](08-testing.md) owns what
that measures and what it skips.
