---
description: Tasks done end to end -- a response in a test, a value whose type a checker does not know, a schema against its last version, stored data against a new schema, a configuration file, a model's tool call, a response from an API the code does not own, a field of a FastAPI request, a message routed by its shape.
---

# Recipes

Each recipe is one task done end to end with a value already in hand. Each says
what it cannot do, and each block runs as written.

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

## A value whose type the checker does not know

A value from `json.loads`, a cache, a queue or a JSON column reaches the code
typed `Any` or `object`, and a checker reads nothing about it: a misspelled key
is found when the line runs. `is_valid` is a `TypeGuard`, so where it answers
`True` a checker reads the value as the schema's type, and `ensure` is the
raising form, which returns the value it was given, typed:

```python
import json
from typing import Literal, TypedDict

from valgebra import Validator


class Order(TypedDict):
    id: int
    status: Literal["open", "paid"]
    total: int


ORDERS = Validator(Order)


def charge(order: Order) -> str:
    return f"charging {order['total']} for order {order['id']}"


def charge_cached(cache: dict[str, object], key: str) -> str | None:
    cached = cache.get(key)
    if not ORDERS.is_valid(cached):
        return None
    return charge(cached)  # a checker reads `cached` as an `Order` here


cache: dict[str, object] = {"order:7": {"id": 7, "status": "open", "total": 1200}}
assert charge_cached(cache, "order:7") == "charging 1200 for order 7"
assert charge_cached(cache, "order:8") is None

data = json.loads('{"id": 8, "status": "paid", "total": 300}')
order = ORDERS.ensure(data)  # an `Order` to a checker, and the same object
assert order is data
```

`ensure` is `typing.cast` with the check: `cast(Order, data)` tells the
checker what nothing has checked. A schema narrows where a checker reads it as
a type, as it reads a class such as a `TypedDict`, a dataclass or
`list[Order]`; `int | None`, a `Literal` and an `Annotated` form read as
`Validator[object]` and narrow nothing
([what each spelling reads as](18-static-checking.md#a-validators-type-is-its-sets)).
And a `False` narrows nothing: past a refusal the value reads as it did before
([a `True` narrows](18-static-checking.md#a-true-narrows-and-a-false-does-not)).

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

## Stored data against a new schema

Data stored under one version of a schema is read under the next, so a
migration is the request case of [the recipe above](#a-schema-against-its-last-version):
every stored value must be a new one. Where the old schema is a subtype of the
new, that is proven of every value the old schema admits, and no row needs
reading. Where `relation_to` answers `"not_subset"`, some old value is refused,
though not necessarily one the table holds, and the rows are read:
`is_valid_json` parses each stored document and checks it
[in place](07-json.md#performance), building no dict for it, and
`validate_json` gives the reasons for the ones it refuses:

```python
import sqlite3
from typing import Literal, TypedDict

from valgebra import ValidationError, Validator


class ProfileV1(TypedDict):
    name: str
    plan: Literal["free", "pro", "legacy"]


class ProfileV2(TypedDict):
    name: str
    plan: Literal["free", "pro", "legacy", "team"]


class ProfileV3(TypedDict):
    name: str
    plan: Literal["free", "pro", "team"]


STORED = Validator(ProfileV1)
# A plan added: every stored profile is a new one, and no row is read.
assert STORED.is_subtype_of(ProfileV2)
# A plan retired: a stored profile may be refused, so the rows are read.
assert STORED.relation_to(ProfileV3) == "not_subset"

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY, doc TEXT)")
db.executemany(
    "INSERT INTO profiles (doc) VALUES (?)",
    [
        ('{"name": "Ada", "plan": "pro"}',),
        ('{"name": "Bob", "plan": "legacy"}',),
        ('{"name": "Cy", "plan": "free"}',),
    ],
)

NEW = Validator(ProfileV3)


def why(doc: str) -> list[tuple[object, object]]:
    """Give the path and code of every reason `NEW` refuses a stored document."""
    try:
        NEW.validate_json(doc)
    except ValidationError as error:
        return [(item["path"], item["code"]) for item in error.errors]
    raise AssertionError("validate_json admitted a document is_valid_json refused")


rows = db.execute("SELECT id, doc FROM profiles").fetchall()
refused = {row_id: why(doc) for row_id, doc in rows if not NEW.is_valid_json(doc)}
assert refused == {2: [(("plan",), "union_error")]}
```

**The proof is about the old schema, not the table.** It covers the stored rows
only where each was checked against the old schema when it was written; a row
another writer put there, or one older than the check, is covered by no proof,
and only the scan reads it. Where `relation_to` answers `"undecided"`, read the
rows too. A column a driver hands back decoded, as a dict, is checked with
`is_valid` and `validate` instead.

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

## A response from an API the code does not own

An API the code does not own changes its responses when its provider decides
to. A check that raises stops the service at the first change, and no check
lets the change reach the line that reads the field. `ensure` checks every response and
returns it typed; where it refuses, each failure is logged by its path and
code, and the code carries on with the last response that belonged:

```python
import logging
from typing import TypedDict

from valgebra import ValidationError, Validator

log = logging.getLogger("rates")


class Rate(TypedDict):
    currency: str
    rate: float


RATES = Validator(list[Rate])
last_good: list[Rate] = [{"currency": "USD", "rate": 1.08}]

response = [{"currency": "USD", "rate": "1.09"}, {"code": "GBP", "rate": 0.86}]
try:
    rates = RATES.ensure(response)
except ValidationError as error:
    for item in error.errors:
        log.warning("rates drifted at %s [%s]", item["path"], item["code"])
    rates = last_good
assert rates == last_good
```

```text
rates drifted at (0, 'rate') [float_type]
rates drifted at (1, 'currency') [missing_key]
```

A response that belongs is walked once and builds no error: the walk that says
where runs, and the error is built, only for a response that drifted
([reporting a failure](11-performance.md#reporting-a-failure)). The log line
carries each failure's path and code and not its message, since the message
renders the value it refused, and a response's values can be data a log should
not keep; a code is stable, so a count by path and code is a metric.

**An open record reports what was removed or changed, not what was added.** The
second rate renamed `currency` to `code`, which reads as `currency` missing:
that is the drift that breaks the line reading the field, and a closed record
reports an added field too. A JSON number written without a point is an `int`,
which a `float` field refuses
([the JSON value mapping](07-json.md#json-to-python-value-mapping)), so where a
provider writes whole numbers bare, the field is `int | float`.

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

## A message routed by its shape

A consumer reading messages of several kinds hands each to a handler, and where
no field names the kind, the handler is the one whose schema the message is
in. Two schemas that overlap make that answer depend on the order the handlers
were added, since a message in both goes to whichever comes first.
`intersection(a, b).is_empty()` proves that no value is in both, so a route is
refused unless its schema is proven apart from every other's, and whether every
message the producer sends has a route is `is_subtype_of`:

```python
from collections.abc import Callable
from typing import Any

from typing_extensions import NotRequired, TypedDict

from valgebra import Validator, intersection, union


class Payment(TypedDict, closed=True):
    order: int
    amount: int


class Refund(TypedDict, closed=True):
    order: int
    reason: str


class Shipment(TypedDict, closed=True):
    order: int
    carrier: str


ROUTES: dict[Validator[Any], Callable[[Any], str]] = {}


def route(schema: Validator[Any], handler: Callable[[Any], str]) -> None:
    """Send `handler` the messages in `schema`, which no other route's may share."""
    for known in ROUTES:
        if not intersection(known, schema).is_empty():
            msg = f"{schema!r} may admit a message {known!r} admits"
            raise ValueError(msg)
    ROUTES[schema] = handler


def dispatch(message: object) -> str:
    """Hand `message` to the one route whose schema it is in."""
    for schema, handler in ROUTES.items():
        if schema.is_valid(message):
            return handler(message)
    msg = f"no route takes {message!r}"
    raise LookupError(msg)


def on_payment(message: Payment) -> str:
    return f"order {message['order']}: {message['amount']} paid"


def on_refund(message: Refund) -> str:
    return f"order {message['order']}: refunded, {message['reason']}"


def on_shipment(message: Shipment) -> str:
    return f"order {message['order']}: shipped by {message['carrier']}"


route(Validator(Payment), on_payment)
route(Validator(Refund), on_refund)
assert dispatch({"order": 7, "amount": 1200}) == "order 7: 1200 paid"


class Adjustment(TypedDict, closed=True):
    order: int
    amount: int
    note: NotRequired[str]


def on_adjustment(message: Adjustment) -> str:
    return f"order {message['order']}: adjusted by {message['amount']}"


refused = False
try:
    route(Validator(Adjustment), on_adjustment)
except ValueError:
    refused = True
assert refused
# A payment is an adjustment too, since `note` is optional.
assert {"order": 7, "amount": 1200} in Validator(Adjustment)

SENT = union(Payment, Refund, Shipment)  # every message the producer sends
assert SENT.relation_to(union(*ROUTES)) == "not_subset"  # a shipment has no route
route(Validator(Shipment), on_shipment)
assert SENT.is_subtype_of(union(*ROUTES))
```

**Handlers told apart by their keys take closed records.** An open `TypedDict`
admits every key it does not name, so two open records overlap wherever one's
keys can sit beside the other's: open, `Payment` and `Refund` both admit
`{"order": 7, "amount": 1200, "reason": "late"}`, and the second route is
refused. An optional field does the same between closed ones, as `Adjustment`
does above.

**A refused route is not a proven overlap.** `is_empty` answers `True` only on a
proof, and `False` both where the two schemas meet and where the procedure
declines ([the decidability boundary](15-decidability.md#the-contract)), so the
router refuses what it cannot prove apart, which is the safe side; a value both
admit, as the block names, shows the overlap is real. Where one field names the
kind, a table keyed by it is the router, as the FastAPI recipe's is, and needs
no proof.
