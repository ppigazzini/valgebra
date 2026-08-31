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

The whole-schema transforms `simplify` (reduce by the lattice laws), `open`, and
`close` (a record's key set) are methods on the compiled validator
(`Validator.simplify`/`open`/`close`), documented above. A fixed-length list is
the native `[A, B]` literal (see the [schema language](03-schema-language.md)).

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
| `obj in validator` | `__contains__` | Membership: the operator form of `is_valid`, so a check reads as the set test it is. |
| `a \| b` | `__or__`, `__ror__` | The union of the two schemas. `\|` is the operator typing already uses for a union; intersection and complement have no typing operator and stay named calls. The reflected form is what makes `None \| validator` work. |
| `a == b` | `__eq__` | **Syntactic** equality: the schema trees, recursive definitions, and pooled constants all match. Ask `is_equivalent` for the semantic question — whether two schemas denote the same set however they are spelled. |
| `hash(validator)` | `__hash__` | Consistent with `==`, so a validator is a dict key or a set member. It digests the schema shape and definitions only, never the pooled constants, so an unhashable constant cannot break it. |
| `repr(validator)` | `__repr__` | The annotation expression that produces the schema. |

`copy.copy` and `copy.deepcopy` both return an equivalent validator; a validator
is immutable, so the copy shares the pool rather than duplicating it.

## Lattice bounds

`anything` and `nothing` are the two bounds of the lattice, and both are
`Validator` instances rather than schema forms you construct.

- `anything` — the top: every Python value is a member, so `anything.is_valid(x)`
  is `True` for every `x`. It is the identity of `intersection` and the
  absorbing element of `union`.
- `nothing` — the bottom: no value is a member, and `nothing.is_empty()` is
  `True`. It is the identity of `union` and the absorbing element of
  `intersection`.

Both are ordinary validators: they compose with the combinators, compare with
`is_subtype_of`, and appear as the reduced form the [simplifier](04-algebra.md)
produces. `anything` is distinct from `Any`, which is the gradual atom and is
exempt from the complement law; see the
[decidability boundary](15-decidability.md).

```python
from valgebra import Validator, anything, complement, intersection, nothing, union

assert anything.is_valid(object())
assert nothing.is_empty()
assert Validator(int).is_subtype_of(anything)
assert nothing.is_subtype_of(int)
assert intersection(int, anything).is_equivalent(int)
assert union(int, nothing).is_equivalent(int)
assert complement(anything).is_equivalent(nothing)
```

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

The [error model](08-error-model.md) guide lists every code and the path format.

```python
import json

from valgebra import ValidationError, Validator

try:
    Validator({"a": int, "b": str}).validate({"a": "x", "b": 1})
except ValidationError as err:
    assert err.code == "int_type"
    assert err.path == ("a",)
    assert [item["path"] for item in err.errors] == [("a",), ("b",)]
    assert json.dumps(err.errors)  # the whole report is JSON-serializable
```

## Package version

`valgebra.__version__` is the installed distribution version as a string. It is
read from the package metadata maturin derives from the Cargo workspace
manifest, so it always matches the built wheel and never drifts from a
hand-maintained literal.

```python
import valgebra

print(valgebra.__version__)
```

## What is public

Every name above is public and is reached from the top-level `valgebra`
namespace. `valgebra.__all__` lists the schema surface — `Validator`, the
combinators, `Regex`, `recursive`, the two bounds, and `ValidationError`;
`__version__` is public too and is not in it, being metadata rather than part of
the algebra.

The compiled extension underneath, `valgebra._valgebra`, is private: its layout,
its module name, and which names it carries are free to change in any release.
Import from `valgebra`.
