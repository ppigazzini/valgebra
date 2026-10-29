---
description: The full public surface, generated from the package docstrings.
---

# API reference

The full public surface of the `valgebra` package. Every name is re-exported
from the top-level `valgebra` namespace.

## Compiling and checking

::: valgebra.Validator

## Combinators

::: valgebra.union

::: valgebra.intersection

::: valgebra.complement

::: valgebra.instance_of

The whole-schema transforms `open` and `close` are methods on the compiled
validator (`Validator.open`/`close`), documented above.

What `open` and `close` move is the **key-type region no clause claims**:
opening frees it and closing refuses it, and neither touches a region a clause
already claims. A clause short of every value claims its keys; the clauses
admitting every value are read as one clause over the union of their keys, and
where that clause covers every key the others leave over, its part over keys
another clause claims stays and the rest is the region, so `{str: anything, complement(str): anything}` closes to `{}` as
`dict[anything, anything]` does. A closed record claims none, which is why
opening one admits every key its fields do not name. A `dict[str, int]` claims the `str` region, so
opening it keeps `str` keys mapping to integers and frees the rest, and an open
`TypedDict` claims it too, through the `str: anything` clause its reading
carries.

Opening a mapping writes the freed region as a clause keyed by the complement
of the keys it claims. Where those are whole kinds -- `dict[str, int]` -- the
relations decide the result like any other map. Where they are literals --
`dict[Literal["a"], int]` -- the complement is part of `str`, which the set
representation has no part for, and a relation about the result may answer
`"undecided"` ([decidability](15-decidability.md)). Membership is exact either
way.

A fixed-length list is the native `[A, B]` literal (see the [schema
language](03-schema-language.md)).

## Refinement markers

::: valgebra.Regex

## Recursion

::: valgebra.recursive

## Operators on a validator

A compiled validator carries an operator surface as well as its methods. These
are written here rather than generated, because CPython supplies its own text for
the dunder behind a type slot and that is what introspection would show.

| Operator | Method | Meaning |
| --- | --- | --- |
| `obj in validator` | `__contains__` | Membership: the operator form of `is_valid`, so a check reads as the set test it is. Python spells membership `in` as it spells a union `\|`, which is why both are operators and intersection and complement are not. |
| `a \| b` | `__or__`, `__ror__` | The union of the two schemas. `\|` is the operator typing already uses for a union; intersection and complement have no typing operator and stay named calls. The reflected form is what makes `None \| validator` work. |
| `a == b` | `__eq__` | Equality of the **normal form**: the schema trees, recursive definitions, and pooled constants all match, after the lattice laws construction settles — so a difference of member order, of a repeat, or of an identity is not one, and neither is the order the `recursive` definitions were met in: a reference is read through to the definition it names ([the algebra page](04-algebra.md#subtyping-equivalence-and-emptiness)). Ask `is_equivalent` for the semantic question — whether two schemas denote the same set, which needs a containment rather than a law. |
| `hash(validator)` | `__hash__` | Consistent with `==`, so a validator is a dict key or a set member. It digests the schema shape, each definition a reference names, and each pooled constant's own hash, so `Literal[1]` and `Literal[2]` hash apart; a constant with no hash contributes nothing, so an unhashable constant cannot break it. Computed once and kept, as a `frozenset` keeps its hash. |
| `repr(validator)` | `__repr__` | A rendering of the schema as an expression that builds it. |

`copy.copy` and `copy.deepcopy` both return the validator itself: it is
immutable and cannot be subclassed, so its copy is the object, as the standard
library's `copy` gives back a `tuple` or a `frozenset`.

`repr` is a **rendering**, not a serialization. What it gives back is an
expression that builds the same schema — a recursive schema as the `recursive`
call it is, its back edge bound to a name the rest of the render does not print,
so a class named `X` in the body is not read back as the back edge; an open
record as the catch-all entry it carries, the nullary product as `tuple[()]`, a
union holding a record or a list display as the `union(...)` call, since `|`
between two dicts is Python's merge, and a key and a constant by Python's own
`repr`, whole — so it can be pasted into a session that holds the names it
prints and read back. A refinement marker prints by its bare name, `Ge(0)` rather than
`at.Ge(0)`, so that session imports each marker by name, as
`from annotated_types import Ge` does, rather than through the
`import annotated_types as at` these pages write. Seven things it cannot render
as an expression, and one of them reads back quietly: a
**class**, which is an object rather than syntax and appears as its name -- or
as `instance_of(Name)` where it stands alone and its name would read back as
the class met with the fields it declares; a
**predicate**, which is a function and appears as `Predicate(...)`, and which
the frontend refuses where it is built; a **constant whose `repr` is not an
expression of it**, which is a syntax error where it is parsed -- an enum
member's own `<Color.RED: 1>`, an `int` past the interpreter's digit limit
printed as `<unrepresentable>`, and a value of a subclass of `int`, `float`,
`complex`, `str` or `bytes` whose `repr` is the builtin's spelling of it, which
would read back as the builtin and prints as `<MyStr: 'a'>` instead; a **meet of two
classes that each declare attributes**, which flattens to the two classes
alone, printed as `instance_of(...)`, and their two attribute records, and a
record standing apart from its class prints
as `object(x=int)`, a form no constructor spells -- the schema does not record
which class each record came from, so the render cannot fold them back, and a
`Protocol` prints the same way, as the record of its members; a
**required key whose name ends in `?`**, which no dict display spells -- the
dict literal reads every trailing `?` as optional -- and which prints marked,
`<required: 'a?'>`, a syntax error where it is parsed:
`TypedDict("TD", {"a?": int})` renders `{<required: 'a?'>: int, str: anything}`
rather than the display of a record whose `a` is optional; a **class met with a
declaration it did not make**, the quiet one:
`intersection(MyTuple, tuple[int, str])` prints as `MyTuple`,
`intersection(list[int], Plain)` as `Plain`, and `intersection(Plain, P)` for a
protocol `P` as `Plain`, because a `NamedTuple` and a dataclass are built as
exactly that meet and the schema does not record whether the shape or the
record was the class's own declaration -- and so a generic dataclass or named
tuple applied to its arguments prints as its name, `Box[int]` as `Box`, which
reads back as the class's defaults or is refused where a parameter has none;
and a **schema past the renderer's own depth bound**, which gives up and prints
`<...>`. The bound is
within reach: no *single* annotation can be written deep enough, since the
frontend refuses past `MAX_SCHEMA_DEPTH` and the renderer's bound sits above
it, but a chain of recursive definitions composes — the render descends into
each in turn, so a hundred shallow links reach a depth one annotation cannot.
The mark is deliberately not an ellipsis: `...` reads as a schema inside a
subscript, so a truncated render would parse and hand back a *different*
validator with nothing to say it had been cut. The bound is on depth, not on
length: a render is as long as the schema unfolded, and a chain of definitions
each naming the last twice prints the last once per path to it -- sixteen
links of a hundred nodes print megabytes. A REPL, a debugger, `logging` with
`%r` and pytest's assertion rewriting all ask for it. Do not parse a repr: it
is for a person to read, and [inspection](09-inspection.md) says how to ask a
schema questions instead.

## Lattice bounds

`anything` and `nothing` are the two bounds of the lattice, and both are
`Validator` instances rather than schema forms you construct.

- `anything` — the top: every Python value is a member, so `anything.is_valid(x)`
  is `True` for every `x`. It is the identity of `intersection` and the
  absorbing element of `union`.
- `nothing` — the bottom: no value is a member, and `nothing.is_empty()` is
  `True`. It is the identity of `union` and the absorbing element of
  `intersection`.

`nothing` is also where the third answer about emptiness is asked for:
`is_empty` is a predicate and reports the proof, while
`relation_to(nothing)` reports which of proof, refutation and decline the
procedure reached ([the boundary](15-decidability.md)).

Both are ordinary validators: they compose with the combinators, compare with
`is_subtype_of`, and are what the constructors fold to — `intersection(int,
complement(int))` **is** `nothing` ([the algebra](04-algebra.md)). `Any` is the
same set and the same schema as `anything`, differing
only in what `repr` gives back; see
[`Any` versus `anything`](04-algebra.md#any-versus-anything).

```python
from valgebra import Validator, anything, complement, intersection, nothing, union

assert anything.is_valid(object())
assert nothing.is_empty()
assert Validator(int).is_subtype_of(anything)
assert nothing.is_subtype_of(int)
assert intersection(int, anything).is_equivalent(int)
assert union(int, nothing).is_equivalent(int)
assert complement(anything).is_equivalent(nothing)
assert Validator(int).relation_to(nothing) == "not_subset"  # a value says so
```

## What raises, and when

A failure at **build** time — when the schema is compiled — is a different kind
of event from a failure at **validation** time, and it raises a different
exception. `ValidationError` is only ever the second. Building refuses with one
of three, and which one says what went wrong. A class whose annotation Python
cannot evaluate raises what evaluating it raises, before any of them: a
`NameError` for a name its module does not define when the validator is built,
such as an import moved under `if TYPE_CHECKING:` ([static
checking](18-static-checking.md)).

| Raised | When | Example |
| --- | --- | --- |
| `NotImplementedError` | The spec names a form with no decidable runtime membership. | `Sequence[int]`, `Mapping[str, int]`, a `TypeVar`, `Final`, `ClassVar` |
| `NotImplementedError` | A callable that is not a class, or the annotation `Validator[T]`, stands where a schema goes. The message names the spelling that was meant: `Annotated[T, fn]` or `Literal[fn]` for a callable, the schema or a compiled validator for `Validator[T]`. | `Validator(fn)`, `intersection(Record, fn)`, `list[fn]`, `Validator(Validator[int])` |
| `NotImplementedError` | A class is reached again, under the type arguments it is applied to, while its own fields are being built. | a self-referential dataclass, `TypedDict` or protocol, whose field type names the class |
| `ValueError` | A schema crosses a size bound: depth, definitions, or nodes, whether a sequence of calls built it or an annotation spells it. | growing a schema in a loop with `\|`, `union` or `intersection`; opening one near the node bound; `list[list[...]]` nested past `MAX_SCHEMA_DEPTH` |
| `ValueError` | A marker's value cannot denote a set. | `MultipleOf(0)` |
| `ValueError` | A `recursive` body is not contractive — its back edge is not under a structural constructor. | `recursive(lambda s: s)` |
| `TypeError` | An argument is the wrong Python type for the call. | `validate_json(123)`, `load(123)` |

The form row and the class row are **different refusals** that share a
class. The first is about the *form* and no depth would help it; the second is
a class whose field type names the class again, which never bottoms out and is
refused where it is reached rather than at a depth. The `ValueError` depth row
is one bound, `MAX_SCHEMA_DEPTH`, on a schema a sequence of calls built and on
one an annotation spells -- the frontend descends one level past it, so the
message names the bound and the schema past it is refused rather than
half-built -- and `MAX_SCHEMA_DEPTH` is importable so code that sizes a schema
reads it rather than repeating it. The [resource limits](10-limits.md) guide covers all three
sizes.

```python
from collections.abc import Sequence
from dataclasses import dataclass

from valgebra import Validator, recursive


@dataclass
class Node:
    next: "Node"


for spec in (Sequence[int], Node):
    try:
        Validator(spec)
        raise AssertionError("expected a rejection")
    except NotImplementedError:
        pass

try:
    Validator(recursive(lambda schema: schema))
    raise AssertionError("expected a rejection")
except ValueError:
    pass

try:
    Validator(int).validate_json(123)
    raise AssertionError("expected a rejection")
except TypeError:
    pass
```

Every one of these is raised before any value is checked, so a schema that
compiles is a schema whose every membership question is answerable.

## Errors

### `ValidationError`

Raised by `validate`, `validate_json`, `load` and `ensure` when a value is not a
member of the schema's set. It subclasses `Exception`, and carries a structured,
machine-readable model as well as a message. The attributes are set on the
instance, so they are written here rather than generated.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `errors` | `tuple[dict[str, object], ...]` | One item per independent failure, each a JSON-serializable dict with the keys `code`, `path`, `message`, `expected` and `value`. `json.dumps(err.errors)` is the JSON form of the whole report. One call reports every failure unless `fail_fast=True` stops the walk at the first. |
| `code` | `str` | The first item's stable failure code, such as `int_type`, `missing_key` or `literal_error`. |
| `path` | `tuple[str \| int, ...]` | The first item's location from the root, string keys and integer indices, empty at the root. |
| `message` | `str` | The first item's rendered one-line message. `str(err)` summarizes every failure rather than only this one. |
| `expected` | `str` | A short label of the set the first item expected, such as `int`. |
| `value` | `str` | A repr-style summary of the first item's offending value. |

The [error model](08-error-model.md) guide gives the path format and the
properties a code carries, and the files that declare the codes;
`tests/test_error_matrix.py` drives every code a report can carry, because a
list written out beside the walk drifts from it.

```python
import json

from valgebra import ValidationError, Validator

raised = False
try:
    Validator({"a": int, "b": str}).validate({"a": "x", "b": 1})
except ValidationError as err:
    raised = True
    assert err.code == "int_type"
    assert err.path == ("a",)
    assert [item["path"] for item in err.errors] == [("a",), ("b",)]
    assert json.dumps(err.errors)  # the whole report is JSON-serializable
assert raised
```

## Construction limits

Three module constants give the bounds every schema-growing call is checked
against, so a caller sizing a schema reads the number rather than repeating it.
[Resource limits](10-limits.md) says what each one bounds and why.

| name | value | bounds |
| --- | --- | --- |
| `MAX_SCHEMA_DEPTH` | 128 | levels of structural nesting |
| `MAX_DEFINITIONS` | 128 | recursive definitions in one schema |
| `MAX_SCHEMA_NODES` | 100,000 | total schema nodes |

```python
from valgebra import MAX_SCHEMA_DEPTH, Validator, complement

schema: Validator = Validator(int)
for _ in range(MAX_SCHEMA_DEPTH - 1):
    schema = complement(schema)
# An odd number of complements is the complement of `int`, so `1` is outside.
assert schema.is_valid(1) is ((MAX_SCHEMA_DEPTH - 1) % 2 == 0)
```

## Package version

`valgebra.__version__` is the distribution version as a string. It comes from
the Cargo workspace manifest, which is what maturin derives the wheel's metadata
from — so it matches the built wheel and never drifts from a hand-maintained
literal. Taken from the compiled extension rather than read back out of the
installed metadata, because the metadata reader pulls modules of its own into
`import valgebra` for a string the manifest already carries.
`tests/test_version.py` bounds what the import pulls in, and holds the two
versions to each other.

```python
import valgebra

assert isinstance(valgebra.__version__, str)
assert valgebra.__version__
```

## What is public

Every name above is public and is reached from the top-level `valgebra`
namespace. `valgebra.__all__` lists the schema surface — `Validator`, the
combinators, `Regex`, `recursive`, the two lattice bounds, the three
construction limits and `ValidationError` — and `__version__`, the one name in
it that is metadata rather than part of the algebra.

The compiled extension underneath, `valgebra._valgebra`, is private: its layout,
its module name, and which names it carries are free to change in any release.
Import from `valgebra`.
