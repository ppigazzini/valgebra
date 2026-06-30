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
the native `[A, B]` literal (see the [schema language](schema-language.md)).

## Refinement markers

::: valgebra.Regex

## Recursion

::: valgebra.recursive

## Lattice bounds

::: valgebra.anything

::: valgebra.nothing

## Errors

::: valgebra.ValidationError

## Package version

`valgebra.__version__` is the installed distribution version as a string. It is
read from the package metadata maturin derives from the Cargo workspace
manifest, so it always matches the built wheel and never drifts from a
hand-maintained literal.

```python
import valgebra

print(valgebra.__version__)
```
