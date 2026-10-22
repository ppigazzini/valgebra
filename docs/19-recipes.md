---
description: Tasks done end to end -- a response in a test, a schema against its last version, a configuration file, a model's tool call, a field of a FastAPI request.
---

# Recipes

Each recipe is one task done with a value already in hand: a response checked
in a test, a schema held to its previous version, a configuration file read
and checked, a model's tool call checked before the tool runs, a field of a
web request checked against a shape another field chooses. Each says what it
cannot do, and each block runs as written.

## A response in a test

`validate` raises with the path and the code of every failure, and pytest
prints the message, so a test that a response belongs is one call:

```python
from typing import TypedDict

import pytest

from valgebra import ValidationError, Validator


class User(TypedDict):
    id: int
    name: str


USERS = Validator(User)

USERS.validate({"id": 7, "name": "Ada"})
with pytest.raises(ValidationError, match=r"at name: expected str, got None") as caught:
    USERS.validate({"id": 7, "name": None})
assert caught.value.errors[0]["code"] == "string_type"
```

A test written `assert payload in USERS` reads better and says less: where it
fails, pytest shows the schema and not the reason. A hook in `conftest.py` adds
the reason to every such failure. Membership and `validate` answer alike, so a
value the `in` test refuses is one `validate` raises on, and the hook's last
line is that contract broken:

```python
from valgebra import ValidationError, Validator


def pytest_assertrepr_compare(op: str, left: object, right: object) -> list[str] | None:
    """Say why a value is not in a schema, where `assert value in schema` fails."""
    if op != "in" or not isinstance(right, Validator):
        return None
    try:
        right.validate(left)
    except ValidationError as error:
        return [f"{left!r} is not in the schema", *str(error).splitlines()]
    raise AssertionError("validate admitted a value the `in` test refused")
```

```text
>       assert payload in USERS
E       AssertionError: assert {'id': 7, 'name': None} is not in the schema
E         at name: expected str, got None [string_type]
```

## A schema against its last version

Whether a new version of a schema breaks its callers is a question about two
sets, and which set must hold the other turns on who reads the value. A
**request** is written by an old client and read by the new server, so every
old value must be a new one: the old schema is a subtype of the new. A
**response** is written by the new server and read by an old client, so the new
schema must be a subtype of the old. Widening an enum is the case
[Habib et al.](https://arxiv.org/abs/1911.12651) use to tell the two apart:
fine for a request, a break for a response.

```python
from typing import Literal

from valgebra import Validator

CategoryV1 = Literal["staff", "wires", "freelance", "other"]
CategoryV2 = Literal["staff", "wires", "freelance", "stock", "handout", "other"]

# A request field: every value an old client sends is still accepted.
assert Validator(CategoryV1).is_subtype_of(CategoryV2)
# A response field: "stock" is a value an old client refuses.
assert Validator(CategoryV2).relation_to(CategoryV1) == "not_subset"
```

A record follows the same rule, and a `TypedDict` is
[open](03-schema-language.md#a-typeddict-is-open-a-dict-literal-is-closed):
the old version admits any key it does not name, with any value.

```python
from typing_extensions import NotRequired, TypedDict

from valgebra import Validator


class UserV1(TypedDict):
    id: int
    name: str


class UserV2(TypedDict):
    id: int
    name: str
    email: NotRequired[str]


class ClosedUserV1(TypedDict, closed=True):
    id: int
    name: str


# A response may gain a field: an old client already admits any extra key.
assert Validator(UserV2).is_subtype_of(UserV1)
# A request may not, even an optional one: an old client could send an
# `email` that is not a str, and the new version refuses it.
assert Validator(UserV1).relation_to(UserV2) == "not_subset"
# A closed old version sent no `email` at all, so the new field is compatible.
assert Validator(ClosedUserV1).is_subtype_of(UserV2)
```

**Assert a compatibility with `is_subtype_of`, and a break with
`relation_to(...) == "not_subset"`.** A `False` from `is_subtype_of` is "no, or
not yet proven", so a test asserting it passes on a question the procedure
declines; `"not_subset"` is a refutation, a value of the one outside the other.
Where the procedure declines, `relation_to` answers `"undecided"` and both
assertions fail, so the test is red rather than green on a guess. Which
relations decline is the [decidability boundary](15-decidability.md#the-contract).

## A configuration file

A configuration file is a document whose every entry should be checked before
any is used. `load` parses the text, checks it, and returns the parsed value,
and a failure lists every entry that is wrong, with its path. A closed record
-- `closed=True`, from PEP 728 -- also refuses a key it does not name, which is
how a misspelled setting is caught rather than ignored:

```python
from typing import Annotated, Literal

import annotated_types as at
from typing_extensions import NotRequired, TypedDict

from valgebra import ValidationError, Validator


class Server(TypedDict, closed=True):
    host: str
    port: Annotated[int, at.Ge(1), at.Le(65535)]
    log_level: NotRequired[Literal["debug", "info", "warning"]]


SETTINGS = Validator(dict[str, Server])

text = """{
  "web": {"host": "10.0.0.2", "port": 8080},
  "admin": {"host": "127.0.0.1", "port": 0, "log_level": "verbose"},
  "jobs": {"host": "10.0.0.5", "port": 9000, "timeout": 30}
}"""

refused = False
try:
    SETTINGS.load(text)
except ValidationError as error:
    refused = True
    found = [(item["path"], item["code"]) for item in error.errors]
    assert found == [
        (("admin", "log_level"), "union_error"),
        (("admin", "port"), "greater_than_equal"),
        (("jobs", "timeout"), "extra_forbidden"),
    ]
assert refused

config = SETTINGS.load('{"web": {"host": "10.0.0.2", "port": 8080}}')
assert config == {"web": {"host": "10.0.0.2", "port": 8080}}
```

A TOML file is the same check on the dict `tomllib.loads` returns, on Python
3.11 and later, and a YAML file on the dict a YAML library's safe loader
returns: `SETTINGS.validate(data)` raises as `load` does.

**It checks and does not convert.** A port written `"8080"` is refused --
`at port: expected int, got '8080' [int_type]` -- where a parsing library would
read it as the number. Convert first, with the library that owns the format,
and check what it returns.

## A model's tool call

A model that calls a tool writes the arguments itself, as JSON, and nothing
about the model makes them right. `load` parses and checks them in one call,
and its error names every wrong argument and why, which is what goes back to
the model as the tool's result. A closed record also refuses an argument the
tool does not take, which a model invents as readily as a wrong value:

```python
from typing import Literal

from typing_extensions import NotRequired, TypedDict

from valgebra import ValidationError, Validator


class GetWeather(TypedDict, closed=True):
    city: str
    unit: Literal["celsius", "fahrenheit"]
    days: NotRequired[int]


GET_WEATHER = Validator(GetWeather)

arguments = '{"city": "Turin", "unit": "kelvin", "days": 3.0, "country": "IT"}'
refused = False
try:
    GET_WEATHER.load(arguments)
except ValidationError as error:
    refused = True
    result = str(error)  # the tool's result, sent back to the model
    assert result.splitlines() == [
        "3 validation errors:",
        "at days: expected int, got 3.0 [int_type]",
        (
            "at unit: expected one of: the literal 'celsius', the literal "
            "'fahrenheit', got 'kelvin' [union_error]"
        ),
        "at country: expected no unexpected key, got 'country' [extra_forbidden]",
    ]
assert refused

call = GET_WEATHER.load('{"city": "Turin", "unit": "celsius"}')
assert call == {"city": "Turin", "unit": "celsius"}
```

Arguments a client library hands over already parsed, as a dict, are
`GET_WEATHER.validate(arguments)`, which raises the same error. A JSON number
written `3.0` is a float, so `days` refuses it: the model is told, rather than
the value converted.

**The tool's schema is written twice.** valgebra does not render a JSON Schema,
so the parameters the model is shown are written by hand, or by another
library, beside the `TypedDict` that checks what comes back, and nothing holds
the two to one shape: a parameter added to one is added to the other by hand.

## A field of a FastAPI request

The request body stays FastAPI's, a pydantic model, which parses it. What
valgebra checks is a field the model leaves free because its shape depends on
another field: an event whose `data` is read by its `kind`, from a table of
kinds the application adds to. One exception handler answers every refusal in
the application as FastAPI answers a body pydantic refuses: a `422` whose
`detail` lists the errors, each with its path and code.

```python
import asyncio
from typing import Any, Literal

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing_extensions import TypedDict

from valgebra import ValidationError, Validator


class Event(BaseModel):
    kind: str
    data: dict[str, Any]


class OrderPaid(TypedDict, closed=True):
    order: int
    amount: int
    currency: Literal["EUR", "USD"]


class OrderRefunded(TypedDict, closed=True):
    order: int
    reason: str


EVENT_DATA: dict[str, Validator[Any]] = {
    "order.paid": Validator(OrderPaid),
    "order.refunded": Validator(OrderRefunded),
}

app = FastAPI()


@app.exception_handler(ValidationError)
async def refused(_request: Request, error: ValidationError) -> JSONResponse:
    """Answer a refusal with a 422 that lists the errors, as FastAPI does."""
    return JSONResponse({"detail": jsonable_encoder(error.errors)}, status_code=422)


@app.post("/events")
def receive(event: Event) -> dict[str, str]:
    schema = EVENT_DATA.get(event.kind)
    if schema is None:
        raise HTTPException(422, detail=f"no event of kind {event.kind!r}")
    schema.validate(event.data)
    return {"accepted": event.kind}


refund = Event(kind="order.refunded", data={"order": 7, "reason": "late"})
assert receive(refund) == {"accepted": "order.refunded"}

paid = {"order": 7, "amount": "12.00", "currency": "GBP"}
with pytest.raises(ValidationError) as caught:
    receive(Event(kind="order.paid", data=paid))
assert [(item["path"], item["code"]) for item in caught.value.errors] == [
    (("amount",), "int_type"),
    (("currency",), "union_error"),
]
answer = asyncio.run(refused(Request({"type": "http"}), caught.value))
assert answer.status_code == 422
```

The endpoint and the handler are called here as the functions they are.
Posted to `/events`, the same payment is answered `422`, with each error's
path sent as a JSON list.
