---
description: Constraints and predicate refinements over a base schema.
---

# Refinements

A refinement narrows a base type to the subset satisfying one or more
constraints. Write it with `Annotated[T, ...markers]`; the base `T` is checked
first, then each constraint. valgebra reads the
[annotated-types](https://pypi.org/project/annotated-types/) markers
structurally, so it has no runtime dependency on that library.

That is a fact about valgebra, not about your environment: the **examples** on
this page import `annotated_types`, and `pip install valgebra` does not bring it
in. Install it alongside — `pip install annotated-types` — or write the same
constraint with `valgebra.Regex`, which needs nothing beyond valgebra.

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

adult = Validator(Annotated[int, at.Ge(18), at.Le(150)])
assert adult.is_valid(21)
assert not adult.is_valid(5)
```

Refinements built from bound and length markers also take part in the
[decision procedure](15-decidability.md): a refinement is a subtype of its base and
of a looser refinement, and a bound conjunction that cannot be satisfied is
detected as empty.

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

assert Validator(Annotated[int, at.Ge(0)]).is_subtype_of(int)  # refinement <= base
assert Validator(Annotated[int, at.Ge(0), at.Le(10)]).is_subtype_of(
    Annotated[int, at.Ge(0)]  # a tighter bound is a subtype of a looser one
)
assert Validator(Annotated[int, at.Ge(10), at.Le(0)]).is_empty()  # no such int
```

A predicate marker is checked at validation time, and nothing is inferred from it
— its satisfiability is undecidable in general, so two refinements relate through
a predicate only when they carry the same one. A decision query may still call it:
asking whether a literal is a subtype of a refinement asks whether that literal's
value belongs to it, which runs the predicate. A slow or side-effecting predicate
is one `is_subtype_of` pays for.

## Supported markers

| Marker | Constraint | Failure code |
| --- | --- | --- |
| `Ge(n)` | `value >= n` | `greater_than_equal` |
| `Gt(n)` | `value > n` | `greater_than` |
| `Le(n)` | `value <= n` | `less_than_equal` |
| `Lt(n)` | `value < n` | `less_than` |
| `MinLen(n)` | `len(value) >= n` | `too_short` |
| `MaxLen(n)` | `len(value) <= n` | `too_long` |
| `MultipleOf(n)` | `value % n == 0` | `multiple_of` |
| `Regex(p)` | the string fully matches the regex `p` | `string_pattern_mismatch` |
| `Predicate(f)` | `f(value)` is truthy | `predicate_failed` |

`Regex` is valgebra's own marker (`from valgebra import Regex`), since
`annotated-types` defines none for strings. The match is **anchored** — the whole
string must match, like `re.fullmatch` — and runs natively in Rust with a
linear-time engine (no catastrophic backtracking), so unlike a `Predicate` it
stays on the fast path and never crosses into Python per value. An invalid
pattern is rejected when the validator is built, not at first use. A compiled
`re.Pattern` works as metadata too:

```python
import re
from typing import Annotated

from valgebra import Regex, Validator

oid = Validator(Annotated[str, Regex(r"[0-9a-f]{24}")])
assert oid.is_valid("0123456789abcdef01234567")
assert not oid.is_valid("0123456789abcdef0123456X")  # not hex
assert not oid.is_valid("0123")  # not the full 24 characters

assert Validator(Annotated[str, re.compile(r"\d+")]).is_valid("123")
```

### The dialect is Rust's, not `re`'s

Running natively is what buys the linear-time guarantee, and it is also what
makes the dialect the Rust engine's. The two languages are close but not equal,
and **a pattern both engines accept can denote different sets**. Compiling
successfully is therefore not a test of which language a pattern is in. Three
places the two engines part, which a ported pattern should be checked against —
this is where they differ, not a complete audit of what a pattern denotes:

```python
import re
from typing import Annotated

from valgebra import Regex, Validator


def admits(pattern: str, text: str) -> bool:
    return Validator(Annotated[str, Regex(pattern)]).is_valid(text)


# POSIX bracket expressions. Python has no such class, so it reads a character
# class followed by a literal `]` and needs two characters.
assert admits(r"[[:alpha:]]", "a")
assert re.fullmatch(r"[[:alpha:]]", "a") is None
assert re.fullmatch(r"[[:alpha:]]", "a]") is not None

# Case folding. Both fold the ASCII pair; only Python folds the Turkish one.
assert admits("(?i)i", "I")
assert not admits("(?i)i", "\u0131")  # dotless i
assert re.fullmatch("(?i)i", "\u0131") is not None

# Property escapes. `\p{...}` is a pattern only this engine accepts.
assert admits(r"\p{L}+", "ab")
```

The third is the loud case: `re.compile(r"\p{L}+")` raises, so a pattern that
works here fails there and a reader finds out at once. The first two are the
quiet ones — both engines build the pattern and answer differently — and they
are the reason a library holding itself to `re`'s decisions cannot adopt `Regex`
behind a fallback that triggers on compile failure.

### Agreement with `re` is not ASCII

Checking a pattern against the list above tells you the two engines agree. It
does not tell you the pattern denotes what you meant, and the character classes
are where that bites: `\d`, `\w` and `\s` are **Unicode-aware** here, exactly as
they are in `re`. The two engines agree, and a reader who wanted digits gets
every decimal digit Unicode defines.

```python
import re
from typing import Annotated

from valgebra import Regex, Validator

year = Validator(Annotated[str, Regex(r"\d{4}")])
assert year.is_valid("٢٠٢٦")  # Arabic-Indic digits are decimal digits
assert re.fullmatch(r"\d{4}", "٢٠٢٦") is not None  # `re` agrees

assert Validator(Annotated[str, Regex(r"\w")]).is_valid("é")
assert Validator(Annotated[str, Regex(r"\s")]).is_valid("\u00a0")  # no-break space
```

Write the ASCII set when you mean the ASCII set. `[0-9]` is the portable
spelling; `(?-u:\d)` is the same thing with the Unicode flag turned off for that
group, and is this engine's syntax rather than `re`'s:

```python
from typing import Annotated

from valgebra import Regex, Validator

assert not Validator(Annotated[str, Regex(r"[0-9]{4}")]).is_valid("٢٠٢٦")
assert not Validator(Annotated[str, Regex(r"(?-u:\d){4}")]).is_valid("٢٠٢٦")
assert Validator(Annotated[str, Regex(r"[0-9]{4}")]).is_valid("2026")
```

This is the shape that reaches production: a timestamp, an identifier or a
version pattern written with `\d` admits strings the parser downstream rejects,
and every engine involved agreed the pattern was fine.


`MultipleOf(n)` requires a nonzero divisor: no value is a multiple of zero, so
`MultipleOf(0)` is an unsatisfiable constraint and is rejected with a `ValueError`
when the validator is built, rather than rejecting every value at check time.

The compound markers `Interval` and `Len` expand to the bounds they carry, so
`Interval(ge=0, le=10)` contributes `Ge(0)` and `Le(10)`, and `Len(2, 4)`
contributes `MinLen(2)` and `MaxLen(4)`:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

assert Validator(Annotated[int, at.Interval(ge=0, le=10)]).is_valid(5)
assert not Validator(Annotated[int, at.Interval(ge=0, le=10)]).is_valid(11)
assert Validator(Annotated[str, at.Len(2, 4)]).is_valid("abc")
assert not Validator(Annotated[str, at.Len(2, 4)]).is_valid("a")

assert Validator(Annotated[int, at.MultipleOf(3)]).is_valid(9)
assert not Validator(Annotated[int, at.MultipleOf(3)]).is_valid(5)
```

## Predicates: the slow path

A `Predicate` runs an arbitrary Python callable. It is the one *refinement*
constraint that leaves Rust for a caller's own code — literals, instance and
attribute checks, and comparison bounds also compare against Python objects, but
against fixed operators, not arbitrary callables — so it is a **documented slow
path**, never a silent fallback. Use it for checks the markers cannot express:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

even = Validator(Annotated[int, at.Predicate(lambda x: x % 2 == 0)])
assert even.is_valid(4)
assert not even.is_valid(3)
```

A predicate that *raises* is reported distinctly, as `predicate_error` rather
than an ordinary failure, so a buggy predicate is not mistaken for a rejected
value.

A bare callable is a predicate too, without the wrapper:

```python
from typing import Annotated

from valgebra import Validator

positive = Validator(Annotated[int, lambda value: value > 0])
assert positive.is_valid(1)
assert not positive.is_valid(-1)
```

`Predicate` is the portable spelling — it is what pydantic, msgspec and cattrs
read — so prefer it in an annotation other tools also consume. The bare form is
valgebra's own convenience, and it excludes a class for the reason above.

### A bare callable is metadata only

The bare form is read **only** in `Annotated` metadata position, and reading it
there at all is valgebra's own convenience: the libraries that share this
metadata channel each require a wrapper — pydantic `AfterValidator`, beartype
`Is[...]`, msgspec `Meta(...)`, `annotated_types` `Predicate`. The typing spec
leaves each consumer to say what its own metadata means, so this arm reaches
exactly as far as the metadata position and no further.

A schema language that reads a top-level callable as a predicate is expressing a
different rule for a position `Annotated` metadata does not cover; valgebra's
rule for that position is the one below.

Passed as a schema on its own, a callable is not a predicate. It is an object
the frontend has no other reading for, so it takes the
[fallback literal](03-schema-language.md#anything-unrecognized-is-a-literal)
form and denotes the one function object:

```python
from typing import Annotated, TypedDict

from valgebra import Validator, intersection


class Record(TypedDict):
    kind: str


def kind_is_known(value):
    return value["kind"] in {"a", "b"}


checked = intersection(Record, kind_is_known)  # NOT a refinement of Record
assert not checked.is_valid({"kind": "a"})  # a dict is not that function
assert not checked.is_empty()  # nor is emptiness a warning: see below

refined = Validator(Annotated[Record, kind_is_known])  # the refinement
assert refined.is_valid({"kind": "a"})
assert not refined.is_valid({"kind": "z"})
```

The first schema admits nothing, and nothing reports it. `is_empty` returning
`False` is not a claim that the set is inhabited — a negative answer from any
decision is "no, or not yet proven" (see the
[decidability boundary](15-decidability.md#the-contract)), and the meet of a
record with a literal is one it does not decide. So the failure mode is a schema
that silently rejects every value. Write the refinement as `Annotated`, and the
callable narrows the base rather than replacing it.

`annotated_types.Not` wraps a predicate and denotes the values it rejects:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

odd = Validator(Annotated[int, at.Not(lambda value: value % 2 == 0)])
assert odd.is_valid(3)
assert not odd.is_valid(2)
```

A marker that is itself callable is **called**, which is what applies `Not`'s
negation and what keeps a `functools.partial`'s bound arguments. Only a marker
that is not callable is taken apart by its `.func`, which is the shape
`Predicate` has.

## On classes

Refinements declared on a `TypedDict`, dataclass, or `NamedTuple` field are
enforced — the constraint travels with the field:

```python
from typing import Annotated, TypedDict

import annotated_types as at

from valgebra import Validator


class Account(TypedDict):
    balance: Annotated[int, at.Ge(0)]


assert Validator(Account).is_valid({"balance": 100})
assert not Validator(Account).is_valid({"balance": -1})
```

## Unrecognized markers

Per the typing spec, metadata valgebra does not recognize as a constraint is
ignored — so non-constraint `Annotated` metadata (documentation strings, unit
markers) is harmless and carries no membership meaning.

A **class** is among what is ignored. A marker carries its values on an
instance — `Ge(0)` holds `ge = 0` — so the class itself holds no value to read
and calling it constructs rather than asks. A unit or documentation marker
written as a class therefore carries no constraint, exactly as one written as an
instance does not.

```python
from typing import Annotated

from valgebra import Validator

assert repr(Validator(Annotated[int, "a documentation note"])) == "int"
```
