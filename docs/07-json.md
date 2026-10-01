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
`is_valid_json(data)` mirrors `is_valid`: it returns a bool, and raises only
where a comparison raises a fatal interpreter signal.
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

The two paths part only where the parser is stricter than `json.loads`. A
`bytes` argument is read as UTF-8 alone, where `json.loads` detects a UTF-8
byte-order mark or a UTF-16 or UTF-32 encoding and decodes it (below). And four
kinds of document are held to the JSON grammar or to the parser's limits: the
non-standard float tokens (below), a document nested past the parser's
recursion limit (below), an integer longer than the parser's digit limit once
the interpreter's own limit is lifted (below), and an escape naming a **lone
surrogate**. `"\ud800"` is a half of a pair that encodes no character, and the
parser reports `json_invalid` where `json.loads` builds a `str` the object path
admits.

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
machine integer still parses to a Python `int`, up to jiter's digit limit: 4300
digits in the jiter the lockfile pins, which is also the interpreter's default
`sys.get_int_max_str_digits()`. Past it the parser reports `json_invalid`
(`number out of range`) whatever the interpreter's limit is set to, where
`json.loads` follows that setting. An overflowing float literal such as `1e400`
is standard JSON and parses to `inf`.

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

A non-`str`, non-`bytes` argument is a `TypeError` from `validate_json` and
`load`, not a validation failure; `is_valid_json` answers `False` for it, as it
does for anything that is not a document.

**A leading byte-order mark makes the input malformed.** RFC 8259 forbids a
sender to add one and lets a parser ignore one rather than treat it as an error;
jiter refuses it, so a document exported by a spreadsheet or written by a
Windows editor is refused at column 1 with `json_invalid` — reading as an error
about a character nobody can see. `json.loads` refuses one at the start of a
`str` as well, and decodes it away from `bytes`, so on `bytes` the two paths
part here. Strip it before validating:

```python
from valgebra import Validator

v = Validator({"a": int})
carrying = '\ufeff{"a": 1}'
assert not v.is_valid_json(carrying)
assert v.is_valid_json(carrying.lstrip("\ufeff"))
```

**`bytes` are read as UTF-8.** A document encoded as UTF-16 or UTF-32 is
malformed input, where `json.loads` detects the encoding and decodes it; decode
such bytes to a `str` before validating.

**A document nested past the parser's own recursion limit is malformed input
too**, not a deep document the walk then refuses: jiter stops at a couple of
hundred levels of arrays and objects, and stops on both readings alike, so a
document either is a document for both or is one for neither.
`tests/test_json_semantics.py` holds the two to each other across that
boundary rather than pinning where it falls, which is jiter's to move. A
document inside the limit but deeper than the walk descends is a different
refusal, reported by the walk's own depth guard.

## Performance

`is_valid_json` parses with jiter and validates the parsed JSON value **in
place**: no intermediate Python objects are built for the structure it walks, so
membership of a large array or a deep document is decided in Rust. A comparison
against a Python object — a literal, a refinement predicate, or an instance or
attribute check — is the documented step back into Python (detailed below). The
same walk runs over either input source — a Python object or a JSON value — so
the two paths stay equivalent. On the benchmark machine (AMD Ryzen 7 PRO 7840U,
WSL2, CPython 3.14.7 built as the [performance page](11-performance.md)
records, the jiter the lockfile pins, the PGO release wheel — the same profile the release
ships), per-call median on a passing document:

| Shape | `is_valid_json` | `json.loads` + `is_valid` | speedup |
| --- | --- | --- | --- |
| Record, 50 int fields | 1.18 us | 5.92 us | ~5.0x |
| List of 200 small mappings | 25.7 us | 36.8 us | ~1.4x |
| `list[int]`, 10,000 elements | 72 us | 447 us | ~6.2x |

`benches/bench_json.py` times both columns, and a strict
`TypeAdapter.validate_json` over the same three shapes beside them; that third
column is not recorded here, so read the comparison from the benchmark rather
than from this page:

```bash
uv run --no-sync --group bench pytest benches/bench_json.py
```

Avoiding materialization helps where the document is large or scalar-heavy: the
10,000-element array is six times faster than parse-then-validate, and the wide
record five. The middle row is the shape where it pays least, by about two
fifths -- two hundred small mappings are two hundred container walks either
way. Measure your own documents rather than reading a rule off these three.

Where the middle shape's time goes is measured, because it is the shape where
valgebra is closest to pydantic-core. On the competitive gate's document -- two
hundred records of five fields, one of them a list and one a mapping, 17 KB --
`is_valid_json` reads about 70 us on the machine above in the PGO wheel, and
the parts are:

| part | per call | how it was measured |
| --- | --- | --- |
| the parse into the tree the walk reads, and the call | 54 us | `Validator(anything).is_valid_json`, which parses and admits without a walk |
| the walk over that tree | 16 us | `is_valid_json` less the row above |
| `jiter::JsonValue::parse` alone | 64 us | in Rust, release, no profile |
| jiter's pull parser over the same bytes, building nothing | 23 us | in Rust, release, no profile |

The first two rows are one process. The parse moves with the heap a process
starts from -- one process in seventeen read it two fifths dearer -- and the
walk does not. Read in one build, the tree costs nearly three times the parse that
builds nothing: the cost is the tree's containers -- a vector per array and per
object, each behind its own allocation -- and not the scanning. pydantic-core
validates from the parser's events and builds no tree, and the comparison
gate's `json_document` reads it at 1.8 times valgebra's whole call on CPython
3.14, so the tree is most of valgebra's call and the call is still the
shorter.

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

**A walk over the parser's events** is refused
([dev/04-walk.md](dev/04-walk.md)): a union, a meet, a complement and a
refinement read their value again, which a pull parser has moved past, and a
stream answers differently from the tree on documents the tests hold. The
document shape is a tree the walk reads once and drops.

Nodes that compare against a Python object — literals, refinements, instance and
object checks, and predicates — materialize just the value at that node, since
the comparison runs in Python. `validate_json` and `load` materialize the whole
document, since each hands back what a caller reads -- a report of Python-level
value summaries, or the value itself; only `is_valid_json` is fully in place.
