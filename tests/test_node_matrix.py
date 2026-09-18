"""Node coverage matrix.

For a representative schema of every IR node kind, the frontend builds it, the
decision procedure handles it (reflexive, self-equivalent, emptiness
terminates), and its rendered form is stable under simplification. The
sequence-regex shapes are checked under both the list and tuple containers, so a
capability reachable for one container but not the other -- the asymmetry class
of hole -- fails here rather than shipping silently.

PRODUCT: every schema node, in every walk mode
"""

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import GenericAlias
from typing import Annotated, Any

import annotated_types as at
import pytest

from valgebra import (
    ValidationError,
    Validator,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

# `simplify` is deprecated and these exercise it deliberately: the folds it
# still performs are its own, and they are checked until it goes.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Validator.simplify is deprecated:DeprecationWarning"
)

ROOT = Path(__file__).resolve().parent.parent


class _Klass:
    pass


@dataclass
class _Record:
    x: int


def _pt_tuple(*args: object) -> GenericAlias:
    """Return a prefix-plus-tail tuple schema, built at runtime."""
    return GenericAlias(tuple, args)


# Every IR node kind, keyed by its label, with a representative schema.
_NODES: dict[str, object] = {
    "Anything": object,
    "Any": Any,
    "Nothing": nothing,
    "NoneType": None,
    "Bool": bool,
    "Int": int,
    "Float": float,
    "Str": str,
    "Bytes": bytes,
    "Literal": 1,
    "Seq:list-homogeneous": list[int],
    "Seq:list-prefixtail": [int, int, ...],
    "Seq:tuple-fixed": tuple[int, str],
    "Seq:tuple-homogeneous": tuple[int, ...],
    "Seq:tuple-prefixtail": _pt_tuple(int, int, ...),
    "Coll": set[int],
    "KeyedMap:record": {"x": int},
    "KeyedMap:mapping": {str: int},
    "Union": union(int, str),
    "Intersection": intersection(int, complement(str)),
    "Complement": complement(int),
    "Instance": _Klass,
    "Object": _Record,
    "Refine": Annotated[int, at.Ge(0)],
    "Recursive": recursive(lambda t: union(None, {"next": t})),
}


@pytest.mark.parametrize("spec", list(_NODES.values()), ids=list(_NODES))
def test_every_node_is_reachable_and_handled(spec: object) -> None:
    compiled = Validator(spec)  # the frontend builds it
    assert isinstance(compiled.is_empty(), bool)  # emptiness terminates
    assert compiled.is_subtype_of(spec)  # reflexivity
    assert compiled.is_equivalent(spec)  # self-equivalence
    # The rendered form is stable under simplification.
    assert repr(compiled.simplify()) == repr(compiled.simplify().simplify())


# Each sequence shape must be reachable via both containers; the list and tuple
# forms of one shape are unrelated, since the container is part of the type.
_SHAPES: dict[str, tuple[object, object]] = {
    "homogeneous": (list[int], tuple[int, ...]),
    "fixed": ([int, str], tuple[int, str]),
    "prefixtail": ([int, int, ...], _pt_tuple(int, int, ...)),
}


@pytest.mark.parametrize(
    ("listed", "tupled"), list(_SHAPES.values()), ids=list(_SHAPES)
)
def test_sequence_shapes_reach_both_containers(listed: object, tupled: object) -> None:
    list_form = Validator(listed)
    tuple_form = Validator(tupled)
    assert list_form.is_subtype_of(listed)  # the list form builds and is reflexive
    assert tuple_form.is_subtype_of(tupled)  # the tuple form builds and is reflexive
    assert not list_form.is_subtype_of(tupled)  # a list is not a tuple
    assert not tuple_form.is_subtype_of(listed)  # a tuple is not a list


# An independent denotation for each node kind: hand-written members and
# non-members, written from the *meaning* of the node, not read off the
# implementation. Reflexivity and self-equivalence above are symmetric in a defect
# that hits both sides; this catches a node that admits the wrong set. `Anything`
# and `Any` have no non-member; `Nothing` has no member.
_MEMBERSHIP: dict[str, tuple[list[object], list[object]]] = {
    "Anything": ([1, "a", None, object()], []),
    "Any": ([1, "a", None], []),
    "Nothing": ([], [1, "a", None]),
    "NoneType": ([None], [1, "a", False]),
    "Bool": ([True, False], [1, "a", None]),
    "Int": ([1, 0, True], ["a", 1.5, None]),  # bool is a subset of int
    "Float": ([1.5, 0.0], [1, "a", True]),
    "Str": (["a", ""], [1, b"x", None]),
    "Bytes": ([b"x", b""], ["a", 1]),
    "Literal": ([1], [2, True, 1.0, "1"]),  # typed singleton: not True, not 1.0
    "Seq:list-homogeneous": ([[], [1, 2]], [["a"], [1, "a"], (1, 2), 1]),
    "Seq:list-prefixtail": ([[1], [1, 2], [1, 2, 3]], [[], ["a"], [1, "a"], (1, 2)]),
    "Seq:tuple-fixed": ([(1, "a")], [(1, 2), (1,), [1, "a"]]),
    "Seq:tuple-homogeneous": ([(), (1, 2)], [("a",), [1], (1, "a")]),
    "Seq:tuple-prefixtail": ([(1,), (1, 2), (1, 2, 3)], [(), ("a",), (1, "a"), [1, 2]]),
    "Coll": ([set(), {1, 2}], [{"a"}, [1], frozenset({1})]),
    "KeyedMap:record": ([{"x": 1}], [{"x": "a"}, {}, 1]),
    "KeyedMap:mapping": ([{}, {"a": 1, "b": 2}], [{"a": "x"}, [1]]),
    "Union": ([1, "a", True], [1.5, None, b"x"]),
    "Intersection": ([1, True], ["a", 1.5]),  # int and not str
    "Complement": (["a", 1.5, None], [1, True]),  # not int
    "Refine": ([0, 1, 5], [-1, "a"]),  # int >= 0
    "Recursive": ([None, {"next": None}, {"next": {"next": None}}], [1, {"next": 1}]),
}


# THEORY: denotational-semantics
@pytest.mark.parametrize("label", list(_MEMBERSHIP))
def test_node_admits_its_denotation(label: str) -> None:
    compiled = Validator(_NODES[label])
    members, non_members = _MEMBERSHIP[label]
    for value in members:
        assert compiled.is_valid(value), f"{label} should admit {value!r}"
    for value in non_members:
        assert not compiled.is_valid(value), f"{label} should reject {value!r}"


# --- The same table, through every entry point. --------------------------------
#
# The rows above ask `is_valid`, which is one of six ways a caller asks. A defect
# reachable through only one of them is invisible to a suite that asks through
# one: the explaining walk takes a level where the fast one did not, the JSON
# path parses before it walks, and `ensure` and `load` answer with the value
# rather than with a verdict. So each row runs through all of them and they are
# held to one answer -- which is what `docs/dev/04-walk.md` puts first.


def _object_path(compiled: Validator, value: object) -> dict[str, bool]:
    """Membership as each object entry point reports it."""
    answers = {"is_valid": compiled.is_valid(value), "in": value in compiled}
    answers.update(
        _verdicts(
            (
                ("validate", lambda: compiled.validate(value)),
                (
                    "validate_fail_fast",
                    lambda: compiled.validate(value, fail_fast=True),
                ),
                ("ensure", lambda: compiled.ensure(value)),
            )
        )
    )
    return answers


def _verdicts(calls: Iterable[tuple[str, Callable[[], object]]]) -> dict[str, bool]:
    """Run each call, reporting whether it accepted rather than what it raised.

    One place rather than one per entry point: the loop is what a linter reads
    as a cost and what a reader reads as the rule -- an entry point accepts or
    raises, and nothing else is an answer.
    """
    return {name: _accepts(call) for name, call in calls}


def _accepts(call: Callable[[], object]) -> bool:
    """Whether this entry point accepted, rather than what it raised."""
    try:
        call()
    except ValidationError:
        return False
    return True


def _round_trips(value: object) -> str | None:
    """Give the JSON document naming this value, where one names it.

    A tuple writes as an array and reads back as a list, a set writes as
    nothing at all, and a float may not survive its own text. Where the
    document does not name the value the two paths are being asked about two
    values, so the row is not a disagreement and is not compared.
    """
    try:
        text = json.dumps(value)
    except (TypeError, ValueError):
        return None
    back = json.loads(text)
    if type(back) is not type(value) or back != value:
        return None
    return text


def _json_path(compiled: Validator, text: str) -> dict[str, bool]:
    """Membership as each JSON entry point reports it."""
    answers = {
        "is_valid_json": compiled.is_valid_json(text),
        "is_valid_json_bytes": compiled.is_valid_json(text.encode()),
    }
    answers.update(
        _verdicts(
            (
                ("validate_json", lambda: compiled.validate_json(text)),
                (
                    "validate_json_fail_fast",
                    lambda: compiled.validate_json(text, fail_fast=True),
                ),
                ("load", lambda: compiled.load(text)),
                ("load_bytes", lambda: compiled.load(text.encode())),
            )
        )
    )
    return answers


@pytest.mark.parametrize("label", list(_MEMBERSHIP))
def test_every_entry_point_gives_one_answer(label: str) -> None:
    """Every way of asking reports the membership the denotation gives."""
    compiled = Validator(_NODES[label])
    members, non_members = _MEMBERSHIP[label]
    for expected, values in ((True, members), (False, non_members)):
        for value in values:
            answers = _object_path(compiled, value)
            assert set(answers.values()) == {expected}, (label, value, answers)
            text = _round_trips(value)
            if text is None:
                continue
            answers = _json_path(compiled, text)
            assert set(answers.values()) == {expected}, (label, text, answers)


def test_the_json_comparison_reaches_the_nodes_it_can() -> None:
    """A row skipped for every value would compare nothing at all.

    The guard on the comparison above is a `continue`, which is the shape that
    passes by checking nothing. This says how much of the table it does reach.
    """
    reached = {
        label
        for label, (members, non_members) in _MEMBERSHIP.items()
        if any(_round_trips(value) is not None for value in [*members, *non_members])
    }
    assert len(reached) >= 12, sorted(reached)
    for label in ("Int", "Str", "Seq:list-homogeneous", "KeyedMap:record", "Union"):
        assert label in reached


def test_ensure_hands_back_the_object_it_was_given() -> None:
    """The one entry point that answers with the value rather than a verdict."""
    value = [1, 2]
    assert Validator(list[int]).ensure(value) is value
    with pytest.raises(ValidationError):
        Validator(list[int]).ensure(["a"])


def test_load_hands_back_the_parsed_value() -> None:
    """And the one that answers with what the document named."""
    assert Validator(list[int]).load("[1, 2]") == [1, 2]
    assert Validator(list[int]).load(b"[1, 2]") == [1, 2]
    with pytest.raises(ValidationError):
        Validator(list[int]).load('["a"]')


def test_membership_table_covers_every_node() -> None:
    # The instance/object nodes need live class instances, added here; every other
    # node kind must carry an independent membership case.
    klass, record = _Klass(), _Record(1)
    assert Validator(_NODES["Instance"]).is_valid(klass)
    assert not Validator(_NODES["Instance"]).is_valid(object())
    assert Validator(_NODES["Object"]).is_valid(record)
    assert not Validator(_NODES["Object"]).is_valid(object())
    covered = set(_MEMBERSHIP) | {"Instance", "Object"}
    assert covered == set(_NODES), (
        f"node kinds without a denotation case: {set(_NODES) - covered}"
    )


# --- The other direction: the table must cover the IR, not only itself. --------
#
# The assertions above hold every entry in `_NODES` to being real and exercised.
# That is one direction, and it is the one a hand-written table satisfies while
# quietly missing a node nobody added a row for. The universe is therefore read
# out of the tree rather than listed a second time here: a `Schema` variant that
# reaches the walk and has no row fails below, at the commit that adds it.

_IR_SOURCE = ROOT / "crates" / "valgebra-core" / "src" / "ir.rs"

# Labels whose node kind is not their own name, and the one variant with no row.
_LABEL_TO_VARIANT = {
    # `typing.Any` is a spelling of the top, not a node of its own: it compiles
    # to `Anything` carrying the spelling `repr` gives back.
    "Any": "Anything",
    # A class with declared attributes compiles to `Instance ∧ AttrRecord`, so the
    # row exercises both variants and is named for the one only it reaches.
    "Object": "AttrRecord",
    "Recursive": "Ref",  # a fixpoint's back edge is what `recursive` compiles to
}
# `SelfRef` is a build-time marker resolved to a `Ref` before a validator is
# returned, so no compiled schema holds one and no row can represent it.
_NOT_IN_A_COMPILED_SCHEMA = {"SelfRef"}


def _ir_variants() -> set[str]:
    """Read the `Schema` variant names from the IR rather than restating them."""
    source = _IR_SOURCE.read_text(encoding="utf-8")
    start = source.index("pub enum Schema {")
    body = source[start : source.index("\n}\n", start)]
    names = set()
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("//", "/", "#")) or not stripped:
            continue
        match = re.match(r"([A-Z][A-Za-z0-9]*)\s*[({,]", stripped)
        if match:
            names.add(match.group(1))
    return names


# THEORY: the-ir-matches-its-producers
def test_the_node_table_covers_every_ir_variant() -> None:
    variants = _ir_variants()
    # The parse itself is a detector, so it must be shown to have read something:
    # a table that "covers" an empty universe covers nothing.
    assert len(variants) >= 20, f"the IR parse found only {sorted(variants)}"
    assert "KeyedMap" in variants
    assert "Complement" in variants

    covered = {_LABEL_TO_VARIANT.get(label, label.split(":")[0]) for label in _NODES}
    missing = variants - covered - _NOT_IN_A_COMPILED_SCHEMA
    assert not missing, f"IR variants with no row in the node table: {sorted(missing)}"

    # And no row names a variant the IR no longer has, so a rename cannot leave a
    # row silently testing nothing.
    stale = covered - variants
    assert not stale, f"node table rows naming no IR variant: {sorted(stale)}"

    # The excuse expires in its own direction: a variant excused from the table
    # that the IR has dropped is an excuse with no subject.
    assert variants >= _NOT_IN_A_COMPILED_SCHEMA
