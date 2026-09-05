---
description: Using valgebra to interrogate a codebase that has no schemas of its own.
---

# Inspecting a codebase

A schema is an ordinary annotation, so valgebra reads the annotations a codebase
**already has** — and answers questions about them the interpreter cannot. Nothing
is added to the code under study: the schemas live in the script asking the
question, and valgebra stays a development dependency.

This page is the recipes. Each is one question, asked of one or two annotations,
answered by set reasoning rather than by reading. They suit an agent working on a
codebase it did not write, and they need no adoption from that codebase.

For using valgebra as a contract *in* your own code, see the
[tutorial](01-tutorial.md) and the [algebra guide](04-algebra.md).

## The shape of every question

Three relations do the work, and each answers a different kind of question:

| Relation | Asks |
|---|---|
| `is_valid` | is this value in that set? |
| `is_subtype_of` | is every value of this set in that one? |
| `is_empty` | does this set contain anything at all? |

The recipes below are compositions of those three over sets the codebase already
describes.

## A contract the code implies and nothing enforces

The strongest question, because it finds bugs rather than untidiness.

A parameter used as a divisor must not be zero. One whose attribute is read must
not be `None`. One passed to `len` must be sized. The body **states** these by
using the value that way — and if the declared type admits a violating value with
no guard on the path, the function is broken for that value.

That is one set difference: `declared ∧ ¬implied` is exactly the set of values
that pass the type check and break the code.

```python
from valgebra import Validator, complement, intersection, union


def unenforced(declared: object, implied: object) -> list[object]:
    """Values the declaration admits and the body cannot survive."""
    breaking = intersection(Validator(declared), complement(Validator(implied)))
    probes = [None, 0, 0.0, "", b"", [], {}, False, -1]
    return [p for p in probes if breaking.is_valid(p)]


# `def make_grid(columns: int)` whose body computes `idx // columns`.
# The body implies "not zero"; the annotation admits zero.
assert unenforced(int, complement(Validator(0))) == [0]

# `def f(cfg: object)` whose body reads `cfg.model`. Any attribute read implies
# "not None", and `object` admits None.
assert unenforced(object, complement(Validator(None))) == [None]

# A declaration that already excludes the breaking values has nothing to report.
assert unenforced(str, complement(Validator(None))) == []
```

An **unannotated** parameter is the same question with the top on the left, which
is why the absence of a contract is the loudest answer rather than a silent one:

```python
from valgebra import anything, complement, intersection, Validator

nothing_declared = intersection(anything, complement(Validator(None)))
assert not nothing_declared.is_valid(None)  # the body needs non-None
assert anything.is_valid(None)  # and nothing stops None arriving
```

A precondition the code writes down — `if n < 1: raise ValueError` — is a
refinement schema, and valgebra holds it as one. Any other function using that
value the same way without the check admits exactly what the first one rejects:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator, complement, intersection

stated = Validator(Annotated[int, at.Ge(1)])  # what `if n < 1: raise` leaves
unchecked = Validator(int)  # what a sibling function accepts
gap = intersection(unchecked, complement(stated))
assert gap.is_valid(0)
assert not gap.is_valid(5)
```

## A branch the annotation makes unreachable

An `isinstance` test asks for a value in both sets at once. When the intersection
is empty the branch is dead — and the interesting case is not that the branch is
wasted, but that the code and its annotation **disagree about what arrives**.

```python
from valgebra import Validator, intersection

# `def __setitem__(self, key: str, ...)` whose body does `if isinstance(key, bytes)`.
assert intersection(Validator(str), Validator(bytes)).is_empty()
```

One of the two is wrong, and neither can be read alone. The dual question is a
test that can never fail:

```python
from valgebra import Validator

# `def f(sequence: bytes)` guarded by `if not isinstance(sequence, bytes): raise`.
assert Validator(bytes).is_subtype_of(bytes)  # the guard can never fire
```

## A union arm another arm already covers

`bool` is a subclass of `int`, so `bool | int` is `int`. Only inclusion between
the arms says so; a type checker accepts the annotation as written.

```python
from valgebra import Validator, union

assert Validator(bool).is_subtype_of(int)  # the `bool` arm adds nothing
assert union(bool, int).is_equivalent(int)
assert union(object, None).is_equivalent(object)
```

## An exception a sibling already catches

`except (OSError, TimeoutError)` names two classes where one contains the other,
because `TimeoutError` became an `OSError` subclass in Python 3.3. Inside one
tuple that is redundancy; across two clauses it is a handler that never runs.

```python
from valgebra import Validator

assert Validator(TimeoutError).is_subtype_of(OSError)
assert Validator(ModuleNotFoundError).is_subtype_of(ImportError)
```

## An annotation that admits nothing

An unsatisfiable annotation accepts no value at all, and no test that feeds it
valid input will ever say so.

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator, complement, intersection

assert Validator(Annotated[int, at.Gt(10), at.Lt(5)]).is_empty()
assert intersection(int, complement(int)).is_empty()
assert intersection(int, str).is_empty()  # disjoint kinds
```

## An override that narrows what its base accepts

Liskov, decided rather than reviewed: an override must accept everything the base
accepts, so its parameter set is a **superset**.

```python
from valgebra import Validator

base_accepts = Validator(str | None)
override_accepts = str
assert not base_accepts.is_subtype_of(override_accepts)  # the override narrows
```

## What a workload actually passed

Static reading sees literals. A workload sees values, and the same three relations
answer three more questions of them: whether a value arrived that the declaration
rejects, whether the declaration is wider than anything observed, and **which arms
of a union the workload never reached**.

The last has no analogue in a coverage report: a line report says the branch ran,
and cannot say which members of `int | str` ever arrived.

```python
from valgebra import Validator, union

declared = Validator(int | str)
observed = [1, 2, 3]  # collected by a tracer over a real run

assert all(declared.is_valid(value) for value in observed)  # no value escapes
seen = union(*[Validator(type(value)) for value in observed])
assert seen.is_subtype_of(int | str)
assert not declared.is_subtype_of(seen)  # wider than anything observed
assert not any(Validator(str).is_valid(value) for value in observed)  # `str` untested
```

## Reading a negative answer

`is_valid` is exact: `False` means the value is not a member.

`is_subtype_of`, `is_equivalent` and `is_empty` are **sound, not complete**. A
`True` is a proof; a `False` means *not proven*, which is not the same as
disproven. A recipe that reads `not a.is_subtype_of(b)` as "a narrowing happened"
reports a change that may not have occurred.

The [decidability boundary](15-decidability.md) states where the answers are exact.
The places that bite an inspection script are a literal union of about a
thousand members and a deeply nested Boolean combination.

## What this cannot see

- **A codebase that declares nothing** is invisible to the static recipes; only
  the workload recipes reach it.
- **Names must resolve.** An annotation naming a project type cannot be compiled
  without importing the module, and importing runs it. Resolving with the
  module's own namespace reaches every annotation and executes the code; reading
  the source reaches only what the standard library names. Pick deliberately.
- **A declaration is not a missing return.** A `Protocol` method, an `@overload`
  and a docstring-only stub all reach the end of their body without returning.
- **`int` where `float` is declared is conformant.** The typing spec grants an
  implicit promotion; valgebra decides `isinstance`, where the two are disjoint.
  An inspection script must classify that apart rather than report it.
- **A finding needs a witness.** These recipes answer questions about sets; a
  concrete value from the offending set is what makes an answer checkable by a
  human, and one that cannot produce a value is not yet a finding.
- **A compiled schema does not enumerate itself.** There is no call returning a
  validator's declared keys, its per-field clauses, or its arms. Every question
  above is asked by *comparing a schema to another schema* — that is the whole
  interface, and it is enough for the questions on this page. `repr` renders the
  annotation that produces a schema and is meant for a human to read; parsing it
  to recover structure is not supported and will break. To check that a schema
  and a class declare the same fields, write the field set down once and build
  both from it, rather than reading it back out of either.
