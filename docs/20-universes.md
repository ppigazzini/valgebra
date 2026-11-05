---
description: JSON's and TOML's values, and arrays, as sets below the top, in one algebra with no vocabulary of their own.
---

# Several value universes

A schema denotes a set of Python values. What another format holds is a set of Python
values too, since it parses to them: JSON's values are what `json.loads` returns,
TOML's are what `tomllib` returns. So a format is a set you can write, and asking
about its values is a meet with that set. The algebra takes no parameter for a format,
and there is no node, keyword or mode for one.

Below any set `u`, the sets are a Boolean algebra again, with `u` for its top and
`intersection(u, complement(s))` for the complement of `s`. That is the one thing to
remember, and everything on this page is that sentence applied.

## The JSON values

The values a document parses to are a set like any other, so the sets of JSON
values are the sets below it, and asking inside it takes a meet and no new
vocabulary. Its complement of `s` is the JSON values outside `s`:

```python
from valgebra import complement, intersection, recursive, union

number = intersection(int | float, complement(bool))
json_value = recursive(
    lambda value: union(None, bool, number, str, [value], {str: value})
)
assert json_value.is_valid({"a": [1, 2.5, None]})
assert not json_value.is_valid((1, 2))  # a tuple is no JSON value
assert not number.is_valid(True)  # and a JSON boolean is no number

# Bytes are outside the universe, so the universe's "not a string" refuses them
# where a bare complement admits them.
assert complement(str).is_valid(b"x")
assert not intersection(json_value, complement(str)).is_valid(b"x")

# Relations asked inside it are relations of the meet.
assert intersection(json_value, bytes).is_empty()
assert intersection(json_value, list[int]).is_subtype_of(list[int | float])
```

JSON Schema's `integer` is a number with no fractional part, however it is
written, so `1.0` is one and `True` is not. Its set is a union of the two
spellings, and a step reads the floats exactly, so a range and its complement are
decided:

```python
from typing import Annotated

import annotated_types as at

from valgebra import complement, intersection, union

integer = intersection(union(int, Annotated[float, at.MultipleOf(1)]), complement(bool))
assert integer.is_valid(1) and integer.is_valid(1.0)
assert not integer.is_valid(1.5) and not integer.is_valid(True)


def between(lo: int, hi: int):
    return intersection(
        integer,
        union(
            Annotated[int, at.Ge(lo), at.Le(hi)],
            Annotated[float, at.Ge(lo), at.Le(hi)],
        ),
    )


rest = intersection(between(1, 1000), complement(between(1, 30)))
assert rest.relation_to(between(31, 1000)) == "subset"
assert between(31, 1000).relation_to(rest) == "subset"
```

## One number, however it is written

A JSON number is one number however it is written, and Python holds `1` and `1.0`
as two values. A set of JSON values admits both or neither, so a constant is a union
of its spellings, and a `bool` is no spelling of a number. Every set you build from
such pieces keeps that, through a union, a meet or a complement, and that is what makes
a relation between two of them an answer about JSON; `Literal[1]` alone is a set of
Python values.

A bound written with an exponent is the integer it names where it names one: `1e23`
parses to the double below `10**23`, and reading it as a double would refuse the
integer. Read the schema with `json.loads(text, parse_float=Decimal)` to see which
number was written:

```python
import json
from decimal import Decimal
from typing import Annotated

import annotated_types as at

from valgebra import Validator, union

one = union(1, 1.0)
assert one.is_valid(1)
assert one.is_valid(1.0)
assert not one.is_valid(True)

written = json.loads('{"maximum": 1e23}', parse_float=Decimal)["maximum"]
assert written == 10**23
exact = Validator(Annotated[int | float, at.Le(int(written))])
assert exact.is_valid(10**23)
assert not exact.is_valid(10**23 + 1)
assert not Validator(Annotated[int | float, at.Le(1e23)]).is_valid(10**23)
```

## TOML's values

What `tomllib` returns has no `None`, keeps an `int`
and a `float` apart, and holds dates and times, so `1.0` is an integer over JSON's
values and a float over TOML's, and the algebra is the same one:

```python
import datetime as dt
import tomllib
from typing import Annotated

import annotated_types as at

from valgebra import complement, intersection, recursive, union

toml_value = recursive(
    lambda value: union(
        bool,
        intersection(int, complement(bool)),
        float,
        str,
        dt.datetime,
        dt.date,
        dt.time,
        [value],
        {str: value},
    )
)
document = tomllib.loads("released = 2020-01-02\nversion = 1.0")
assert toml_value.is_valid(document)
assert not toml_value.is_valid(None)
assert intersection(toml_value, bytes).is_empty()

toml_integer = intersection(int, complement(bool))
json_integer = intersection(
    union(int, Annotated[float, at.MultipleOf(1)]), complement(bool)
)
assert not toml_integer.is_valid(document["version"])
assert json_integer.is_valid(document["version"])
assert toml_integer.relation_to(json_integer) == "subset"
```

## Arrays

An array is an instance of a class, so a universe of arrays needs nothing new either:
`np.ndarray[S, np.dtype[T]]` is the class met with a record of its `shape` and its
dtype. [What a validator cannot do](17-boundaries.md#it-reads-an-array-through-what-the-array-exposes)
says what that reads, and how the same spelling reads JAX's and PyTorch's. No element
is read, and a set over the elements is a predicate.

## Another language's schemas

This package reads no JSON Schema document. A schema language is a frontend over this
algebra, and a keyword in it denotes a set per universe: `{"type": "integer"}` holds
`1.0` over JSON's values and not over TOML's, where `1.0` is a float. So what a keyword
means is the frontend's to say, and what it asks of the algebra is the two things above:
a top to meet, and leaves that hold every spelling of one value.
