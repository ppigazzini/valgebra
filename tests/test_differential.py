"""Differential ground truth against independent validators.

The denotation oracles in the rest of the suite are written by the same hand as
the implementation, so a shared misconception about the semantics is invisible to
them. This module is the independent check: it runs the *same* schema and value
through valgebra and through a mature external validator, and asserts they agree
-- or that the disagreement is one of a small, enumerated set of **documented
intentional differences**, each tied to a deliberate semantic choice valgebra
makes. A divergence outside that set fails the gate as a valgebra bug.

Two oracles cover the two membership paths:

- **pydantic-core** (``TypeAdapter.validate_python`` in strict mode) for the
  object path. Strict mode disables coercion, so a success is the closest
  analogue to valgebra's check-only membership; it still constructs a value, so
  it does strictly more work, but the accept/reject verdict is comparable.
- **jsonschema** (``Draft202012Validator.is_valid``) for the JSON path. It is a
  pure check with no coercion against the JSON Schema 2020-12 data model.

The intentional differences, each localized to the case that exercises it:

- **bool is a subtype of int.** valgebra follows Python's runtime truth
  (``isinstance(True, int)``), so a bool satisfies ``int``. pydantic strict and
  JSON Schema both treat booleans as disjoint from the integers.
- **int and float are disjoint scalar regions.** valgebra distinguishes them by
  the Python type, so an ``int`` is not a ``float`` and vice versa. pydantic
  strict admits an int where a float is expected, and JSON Schema's ``number``
  subsumes integers while ``integer`` matches any integral-valued number
  (``1.0``, ``1e2``).
- **Literal membership is by exact value and type.** valgebra accepts only a
  value whose type and value match a literal member. pydantic matches a
  ``Literal`` by ``==``, so ``1`` satisfies ``Literal[True]`` and ``True``/``1.0``
  satisfy ``Literal[1]``.

Numeric and boolean traps are exercised only at the scalar cases, where the
predicate that classifies them is exact; the container and structural cases run
on values free of those traps, so they must agree exactly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import Validator

if TYPE_CHECKING:
    from collections.abc import Callable

# The oracles live in the optional ``bench`` dependency group. The dedicated CI
# lane installs that group and sets this variable, so a missing oracle there is a
# hard failure rather than a silently skipped gate; locally, the module skips
# cleanly when the group is absent.
_REQUIRED = os.environ.get("VALGEBRA_REQUIRE_DIFFERENTIAL") == "1"
try:
    from jsonschema import Draft202012Validator
    from pydantic import TypeAdapter
except ImportError:  # pragma: no cover - exercised by dependency presence, not branch
    if _REQUIRED:
        raise
    pytest.skip(
        "differential oracles (pydantic, jsonschema) are not installed",
        allow_module_level=True,
    )


def _never(_value: object, _vg: bool, _oracle: bool) -> bool:  # noqa: FBT001
    """No divergence is intentional: valgebra and the oracle must agree."""
    return False


def _bool_as_int(value: object, vg: bool, oracle: bool) -> bool:  # noqa: FBT001
    """Allow a bool that valgebra admits as an int but the oracle rejects."""
    return isinstance(value, bool) and vg and not oracle


def _int_as_float(value: object, vg: bool, oracle: bool) -> bool:  # noqa: FBT001
    """Allow an int the oracle admits as a float but valgebra rejects."""
    return type(value) is int and not vg and oracle


def _literal_equality(members: tuple[object, ...]) -> Callable[..., bool]:
    """Pydantic accepts a Literal member by ``==``; valgebra by exact type+value.

    The classification is narrow: a value that matches a member by both exact
    type and value is one valgebra must accept, so its rejection would be a real
    bug rather than equality coercion and is *not* treated as intentional.
    """

    def classify(value: object, vg: bool, oracle: bool) -> bool:  # noqa: FBT001
        if vg or not oracle:
            return False
        exact = any(type(value) is type(m) and value == m for m in members)
        return not exact

    return classify


@dataclass(frozen=True)
class ObjectCase:
    """A valgebra schema, the pydantic type to compare it against, and values."""

    name: str
    schema: object
    oracle: object
    values: tuple[object, ...]
    intentional: Callable[..., bool] = field(default=_never)


# Scalars carry the full value bank so every cross-type rejection is checked;
# the documented numeric and boolean differences surface here, where the
# per-case predicate classifies them exactly.
_SCALARS: tuple[object, ...] = (
    0,
    1,
    -3,
    True,
    False,
    1.0,
    0.0,
    3.5,
    "x",
    "",
    b"x",
    b"",
    None,
    [1, 2],
    {"a": 1},
    (1, 2),
)

OBJECT_CASES: tuple[ObjectCase, ...] = (
    ObjectCase("int", int, int, _SCALARS, _bool_as_int),
    ObjectCase("str", str, str, _SCALARS),
    ObjectCase("bytes", bytes, bytes, _SCALARS),
    ObjectCase("float", float, float, _SCALARS, _int_as_float),
    ObjectCase("none", type(None), type(None), _SCALARS),
    ObjectCase("bool", bool, bool, _SCALARS),
    ObjectCase(
        "list[int]",
        list[int],
        list[int],
        ([1, 2], [], [1, "a"], "x", (1, 2), {"a": 1}, [1.0]),
    ),
    ObjectCase(
        "dict[str,int]",
        dict[str, int],
        dict[str, int],
        ({"a": 1}, {}, {"a": "x"}, {1: 2}, [1], {"a": 1.0}),
    ),
    ObjectCase(
        "tuple[int,str]",
        tuple[int, str],
        tuple[int, str],
        ((1, "a"), [1, "a"], (1, 2), (1,), ("a", 1)),
    ),
    ObjectCase(
        "optional[int]", int | None, int | None, (None, 1, "x", 1.0), _bool_as_int
    ),
    ObjectCase(
        "str|int", str | int, str | int, ("x", 1, True, None, 1.0), _bool_as_int
    ),
    ObjectCase(
        "literal[1,2]",
        Literal[1, 2],
        Literal[1, 2],
        (1, 2, 3, True, 1.0, "1"),
        _literal_equality((1, 2)),
    ),
    ObjectCase(
        "literal[a,b]",
        Literal["a", "b"],
        Literal["a", "b"],
        ("a", "b", "c", 1),
        _literal_equality(("a", "b")),
    ),
)

_OBJECT_ORACLES = {case.name: TypeAdapter(case.oracle) for case in OBJECT_CASES}
_OBJECT_VALIDATORS = {case.name: Validator(case.schema) for case in OBJECT_CASES}


def _pydantic_accepts(adapter: TypeAdapter, value: object) -> bool:
    try:
        adapter.validate_python(value, strict=True)
    except Exception:  # noqa: BLE001 - any validation failure is a reject verdict
        return False
    return True


@pytest.mark.parametrize(
    ("case", "value"),
    [
        pytest.param(case, value, id=f"{case.name}-{value!r}")
        for case in OBJECT_CASES
        for value in case.values
    ],
)
def test_object_path_matches_pydantic(case: ObjectCase, value: object) -> None:
    vg = _OBJECT_VALIDATORS[case.name].is_valid(value)
    oracle = _pydantic_accepts(_OBJECT_ORACLES[case.name], value)
    if vg == oracle:
        return
    assert case.intentional(value, vg, oracle), (
        f"undocumented divergence on {case.name} for {value!r}: "
        f"valgebra={vg}, pydantic-strict={oracle}"
    )


def _json_int(doc: str, vg: bool, oracle: bool) -> bool:  # noqa: FBT001
    value = json.loads(doc)
    if isinstance(value, bool) and vg and not oracle:
        return True  # bool subtypes int in valgebra; JSON Schema excludes it
    # JSON Schema "integer" matches any integral-valued number; valgebra keeps
    # the parsed float out of int.
    return isinstance(value, float) and not vg and oracle


def _json_number(doc: str, vg: bool, oracle: bool) -> bool:  # noqa: FBT001
    value = json.loads(doc)
    # JSON Schema "number" subsumes the integers; valgebra keeps int out of float.
    return isinstance(value, int) and not isinstance(value, bool) and not vg and oracle


@dataclass(frozen=True)
class JsonCase:
    """A valgebra schema, the JSON Schema to compare it against, and documents."""

    name: str
    schema: object
    oracle: dict
    docs: tuple[str, ...]
    intentional: Callable[..., bool] = field(default=_never)


# Numeric documents exercise the int/float and bool differences at the scalar
# cases; the container cases run on documents free of those traps.
_NUMERIC_DOCS: tuple[str, ...] = (
    "0",
    "1",
    "-3",
    "true",
    "false",
    "1.0",
    "0.0",
    "3.5",
    "1e2",
    "null",
    '"x"',
    "[1,2]",
    '{"a":1}',
)

JSON_CASES: tuple[JsonCase, ...] = (
    JsonCase("integer", int, {"type": "integer"}, _NUMERIC_DOCS, _json_int),
    JsonCase("number", float, {"type": "number"}, _NUMERIC_DOCS, _json_number),
    JsonCase("string", str, {"type": "string"}, _NUMERIC_DOCS),
    JsonCase("boolean", bool, {"type": "boolean"}, _NUMERIC_DOCS),
    JsonCase("null", type(None), {"type": "null"}, _NUMERIC_DOCS),
    JsonCase(
        "array",
        list[int],
        {"type": "array", "items": {"type": "integer"}},
        ("[1,2]", "[]", "[1,2,3]", '[1,"a"]', '"x"', "{}"),
    ),
    JsonCase(
        "object",
        dict[str, int],
        {"type": "object", "additionalProperties": {"type": "integer"}},
        ('{"a":1}', "{}", '{"a":"x"}', "[1]", '"x"'),
    ),
)

_JSON_ORACLES = {case.name: Draft202012Validator(case.oracle) for case in JSON_CASES}
_JSON_VALIDATORS = {case.name: Validator(case.schema) for case in JSON_CASES}


@pytest.mark.parametrize(
    ("case", "doc"),
    [
        pytest.param(case, doc, id=f"{case.name}-{doc}")
        for case in JSON_CASES
        for doc in case.docs
    ],
)
def test_json_path_matches_jsonschema(case: JsonCase, doc: str) -> None:
    vg = _JSON_VALIDATORS[case.name].is_valid_json(doc)
    oracle = _JSON_ORACLES[case.name].is_valid(json.loads(doc))
    if vg == oracle:
        return
    assert case.intentional(doc, vg, oracle), (
        f"undocumented divergence on {case.name} for {doc}: "
        f"valgebra={vg}, jsonschema={oracle}"
    )


# A trap-free alphabet: printable, no lone surrogates, so json.dumps and the
# valgebra JSON parser agree on the encoding and the comparison is about the
# structural schema, not Unicode edge cases (those are covered elsewhere).
_TEXT = st.text(
    alphabet=st.characters(
        min_codepoint=0x20, max_codepoint=0xFFFF, categories=("L", "N", "P", "Zs")
    ),
    max_size=8,
)
# Mixing strings with integers makes a list[str] reject path that both the oracle
# and valgebra must agree on, without touching the numeric tower (the schema is
# string, so an int is simply not a member).
_LIST_ELEMS = st.one_of(_TEXT, st.integers(), st.none())

_STR_LIST_SCHEMA = {"type": "array", "items": {"type": "string"}}
_STR_LIST_VALIDATOR = Validator(list[str])
_STR_DICT_SCHEMA = {"type": "object", "additionalProperties": {"type": "string"}}
_STR_DICT_VALIDATOR = Validator(dict[str, str])
_STR_LIST_ADAPTER = TypeAdapter(list[str])


@given(value=st.lists(_LIST_ELEMS, max_size=6))
def test_string_list_agrees_with_both_oracles(value: list[object]) -> None:
    doc = json.dumps(value)
    vg_obj = _STR_LIST_VALIDATOR.is_valid(value)
    vg_json = _STR_LIST_VALIDATOR.is_valid_json(doc)
    jsonschema_ok = Draft202012Validator(_STR_LIST_SCHEMA).is_valid(value)
    pydantic_ok = _pydantic_accepts(_STR_LIST_ADAPTER, value)
    assert vg_obj == vg_json, f"object and JSON paths disagree on {value!r}"
    assert vg_obj == jsonschema_ok, f"valgebra vs jsonschema on {value!r}"
    assert vg_obj == pydantic_ok, f"valgebra vs pydantic-strict on {value!r}"


@given(value=st.dictionaries(_TEXT, st.one_of(_TEXT, st.integers()), max_size=6))
def test_string_dict_agrees_with_jsonschema(value: dict[str, object]) -> None:
    doc = json.dumps(value)
    vg_obj = _STR_DICT_VALIDATOR.is_valid(value)
    vg_json = _STR_DICT_VALIDATOR.is_valid_json(doc)
    jsonschema_ok = Draft202012Validator(_STR_DICT_SCHEMA).is_valid(value)
    assert vg_obj == vg_json, f"object and JSON paths disagree on {value!r}"
    assert vg_obj == jsonschema_ok, f"valgebra vs jsonschema on {value!r}"
