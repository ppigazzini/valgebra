# JSON input

A compiled validator validates JSON source directly, parsing on the Rust path:

```python
from valgebra import validator

users = validator({"name": str, "age?": int})

users.validate_json('{"name": "Ada", "age": 36}')        # passes, returns None
assert users.is_valid_json('{"name": "Ada"}')            # optional key absent
assert not users.is_valid_json('{"name": 5}')            # name is not a str

# bytes input is accepted too
assert validator(list[int]).is_valid_json(b"[1, 2, 3]")
```

`validate_json(data, *, fail_fast=False)` mirrors `validate`: it raises
`ValidationError` on failure and aggregates every independent failure by default.
`is_valid_json(data)` mirrors `is_valid`: it returns a bool and never raises.
Both accept a JSON `str` or `bytes`.

## Same decisions as the object path

The JSON path parses the document into a Python value and runs the **same**
validation walk as a native object. So validating a JSON document is exactly
validating `json.loads` of that document — the same accept/reject decision, the
same error codes, and the same paths:

```python
import json

from valgebra import validator

v = validator(list[dict[str, int]])
doc = '[{"a": 1}, {"b": "x"}]'

assert v.is_valid_json(doc) == v.is_valid(json.loads(doc))
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
| array | `list` | `list[...]`, fixed lists, tuples |
| object | `dict` | records and mappings |

Two consequences follow from valgebra's value-set semantics:

```python
from valgebra import validator

# JSON 42 is an int, and float is disjoint from int, so it is not a float
assert not validator(float).is_valid_json("42")
assert validator(float).is_valid_json("42.0")

# JSON true is a bool, and bool is a subtype of int
assert validator(int).is_valid_json("true")
```

`Infinity` and `NaN` are not valid JSON and are rejected as malformed; whole
numbers too large for a machine integer still parse to a Python `int`.

## Malformed JSON

Unparseable input never reaches the validation walk. `validate_json` reports it
through the same structured error model as any other failure — a single `errors`
item coded `json_invalid` carrying the parser's diagnostic — and `is_valid_json`
treats it as a non-member:

```python
from valgebra import ValidationError, validator

v = validator(int)
assert not v.is_valid_json("{ not json")

try:
    v.validate_json("{ not json")
except ValidationError as err:
    assert err.code == "json_invalid"
```

A non-`str`, non-`bytes` argument is a `TypeError`, not a validation failure.

## Performance

`validate_json` parses with jiter and validates in one boundary crossing, faster
than parsing with the standard library and then validating. On the benchmark
machine (Intel i7-3770K, WSL2, CPython 3.14.5, jiter 0.15, valgebra release
build), per-call median on a passing document:

| Shape | `validate_json` | `json.loads` + `is_valid` | speedup |
| --- | --- | --- | --- |
| Record, 50 int fields | 8.2 us | 16.5 us | ~2.0x |
| List of 200 small records | 59 us | 77 us | ~1.3x |
| `list[int]`, 10,000 elements | 515 us | 1,030 us | ~2.0x |

The speedup comes from jiter being the faster parser; the validation step is the
same walk either way. The path currently materializes the parsed document into
Python objects before validating. Validating jiter values in place, without
materializing — which is what lets pydantic-core edge ahead on some shapes — is
a recorded future optimization, not yet implemented.
