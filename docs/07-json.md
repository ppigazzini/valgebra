---
description: Parsing and validating JSON on the Rust path.
---

# JSON input

A compiled validator validates JSON source directly, parsing on the Rust path:

```python
from valgebra import Validator

users = Validator({"name": str, "age?": int})

users.validate_json('{"name": "Ada", "age": 36}')  # passes, returns None
assert users.is_valid_json('{"name": "Ada"}')  # optional key absent
assert not users.is_valid_json('{"name": 5}')  # name is not a str

# bytes input is accepted too
assert Validator(list[int]).is_valid_json(b"[1, 2, 3]")
```

`validate_json(data, *, fail_fast=False)` mirrors `validate`: it raises
`ValidationError` on failure and aggregates every independent failure by default.
`is_valid_json(data)` mirrors `is_valid`: it returns a bool and never raises.
Both accept a JSON `str` or `bytes`.

When you need the data, not just the verdict, `load` validates and **returns the
parsed value**, so it is not parsed twice:

```python
from valgebra import Validator

users = Validator({"name": str, "age?": int})
record = users.load('{"name": "Ada", "age": 36}')
assert record == {"name": "Ada", "age": 36}
```

`load(data, *, fail_fast=False)` raises `ValidationError` on malformed JSON or a
non-member, exactly as `validate_json` does, and otherwise returns the parsed
object.

## Same decisions as the object path

The JSON path parses the document and runs the **same** validation walk as it
would against the equivalent native object. So validating a JSON document is
exactly validating `json.loads` of that document — the same accept/reject
decision, the
same error codes, and the same paths:

```python
import json

from valgebra import Validator

v = Validator(list[dict[str, int]])
doc = '[{"a": 1}, {"b": "x"}]'

assert v.is_valid_json(doc) == v.is_valid(json.loads(doc))
```

Two documents are held to the JSON grammar where `json.loads` is looser, so the
two paths part on those and nowhere else: the non-standard float tokens (below),
and an escape naming a **lone surrogate**. `"\ud800"` is a half of a pair that
encodes no character, and the parser reports `json_invalid` where `json.loads`
builds a `str` the object path admits.

```python
import json

from valgebra import ValidationError, Validator

text = Validator(str)
assert text.is_valid(json.loads(r'"\ud800"'))  # the object path admits it
assert not text.is_valid_json(r'"\ud800"')  # the JSON grammar does not

try:
    text.validate_json(r'"\ud800"')
except ValidationError as error:
    assert error.code == "json_invalid"
```

This equivalence is locked by tests over a corpus spanning the JSON value model.

## JSON-to-Python value mapping

Parsing uses jiter (the parser pydantic-core uses) with the standard JSON model,
so a document maps to Python values exactly as the standard library's `json`
module produces them:

| JSON | Python | Matches schema |
| --- | --- | --- |
| `null` | `None` | `None` |
| `true` / `false` | `bool` | `bool` (and `int`, since `bool` is a subtype) |
| number, no fraction or exponent (`42`) | `int` | `int`, not `float` |
| number with fraction or exponent (`4.2`, `1e3`) | `float` | `float`, not `int` |
| string | `str` | `str` |
| array | `list` | `list[...]` and fixed lists; **not** `tuple[...]`, which is a different container |
| object | `dict` | records and mappings |

Two consequences follow from valgebra's value-set semantics:

```python
from valgebra import Validator

# JSON 42 is an int, and float is disjoint from int, so it is not a float
assert not Validator(float).is_valid_json("42")
assert Validator(float).is_valid_json("42.0")

# JSON true is a bool, and bool is a subtype of int
assert Validator(int).is_valid_json("true")
```

`Infinity`, `-Infinity`, and `NaN` are not valid JSON. The parser rejects these
tokens as malformed, even though Python's own `json.loads` accepts them as an
extension. This is deliberately stricter: a document is held to the JSON grammar,
so a float special can only enter through the object path (where `float('inf')`
is an ordinary member), never through `validate_json`. A number too large for a
machine integer still parses to a Python `int`, and an overflowing float literal
such as `1e400` is standard JSON and parses to `inf`.

```python
from valgebra import Validator

is_float = Validator(float)
# The non-standard tokens are rejected, though json.loads would accept them.
assert not is_float.is_valid_json("Infinity")
assert not is_float.is_valid_json("NaN")
# The object path, in contrast, admits the corresponding float special.
assert is_float.is_valid(float("inf"))
# An overflowing literal is valid JSON and parses to infinity.
assert is_float.is_valid_json("1e400")
```

## Malformed JSON

Unparseable input never reaches the validation walk. `validate_json` reports it
through the same structured error model as any other failure — a single `errors`
item coded `json_invalid` carrying the parser's diagnostic — and `is_valid_json`
treats it as a non-member:

```python
from valgebra import ValidationError, Validator

v = Validator(int)
assert not v.is_valid_json("{ not json")

try:
    v.validate_json("{ not json")
except ValidationError as err:
    assert err.code == "json_invalid"
```

A non-`str`, non-`bytes` argument is a `TypeError`, not a validation failure.

## Performance

`is_valid_json` parses with jiter and validates the parsed JSON value **in
place**: no intermediate Python objects are built for the structure it walks, so
membership of a large array or a deep document is decided in Rust. A comparison
against a Python object — a literal, a refinement predicate, or an instance or
attribute check — is the documented step back into Python (detailed below). The
same walk runs over either input source — a Python object or a JSON value — so
the two paths stay equivalent. On the benchmark machine (AMD Ryzen 7 PRO 7840U,
WSL2, CPython 3.14.7 built as the [performance page](11-performance.md)
records, jiter 0.16, the PGO release wheel — the same profile the release
ships), per-call median on a passing document:

| Shape | `is_valid_json` | `json.loads` + `is_valid` | speedup |
| --- | --- | --- | --- |
| Record, 50 int fields | 2.5 us | 6.2 us | ~2.5x |
| List of 200 small records | 40.5 us | 39.9 us | ~1.0x |
| `list[int]`, 10,000 elements | 71 us | 462 us | ~6.5x |

Avoiding materialization helps where the document is large or scalar-heavy: the
10,000-element array is six times faster than parse-then-validate, and the wide
record two and a half. The middle row is the shape where it stops paying -- two
hundred small mappings are two hundred dict walks either way, and the object
path reaches each of them through a walk that has been made cheaper than the
parse it avoids. Measure your own documents rather than reading a rule off
these three.

Where the middle shape's time goes is measured, because it is the shape where
valgebra is closest to pydantic-core. On the competitive gate's document -- two
hundred records of five fields, one of them a list and one a mapping, 17 KB --
`is_valid_json` reads about 90 us on the machine above, and the parts are:

| part | per call | how it was measured |
| --- | --- | --- |
| `jiter::JsonValue::parse`, the tree the walk reads | 64 us | the parse alone, in Rust, release |
| the walk over that tree | 26 us | `is_valid_json` less the parse |
| jiter's pull parser over the same bytes, building nothing | 21 us | the parse alone, in Rust, release |

The tree costs three times the parse that builds nothing: the cost is the
tree's containers -- a vector per array and per object, each behind its own
allocation -- and not the scanning. pydantic-core validates from the parser's
events and builds no tree, which is why it reads this shape in the time
valgebra reads the tree.

Two readings would remove that cost, and the page states both as the limits
they are.

**A tree of the walk's own** -- one fixed-size node per value in one vector,
strings as ranges into one buffer -- pays no allocation per container and pays
a push per *value* instead. It is a gain exactly while a document holds a
container per few dozen scalars: this one holds one per four and parses **2.4x
faster** that way, while an array of ten thousand bare numbers holds one per
ten thousand and parses slower. A representation that wins the document and
loses the array is a trade and not an improvement, so the reading in place is
the one that never loses.

**A walk over the parser's events**, building nothing at all, would pay the
pull parse and the walk's own reading -- about half of today's call on this
shape. It needs a value the walk can read *twice*: a union tries its members
against one value, an intersection every member, a complement the inner
schema, and a pull parser has moved on. So it is a walk with a buffer for the
rules that backtrack rather than a walk with no tree, and it is a JSON arm for
every container rule rather than an edit to one. Until it exists, the document
shape is a tree the walk reads once and drops.
`benches/bench_json.py` measures a strict `TypeAdapter.validate_json` over the
same three shapes; that column is not recorded above, so read the comparison
from the benchmark rather than from this page.

Nodes that compare against a Python object — literals, refinements, instance and
object checks, and predicates — materialize just the value at that node, since
the comparison runs in Python. The `validate_json` explain path still
materializes the whole document (it reports Python-level value summaries in its
errors); only the `is_valid_json` fast path is fully in place.
