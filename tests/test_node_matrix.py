"""Node coverage matrix.

For a representative schema of every IR node kind, the frontend builds it, the
decision procedure handles it (reflexive, self-equivalent, emptiness
terminates), and its rendered form is stable under simplification. The
sequence-regex shapes are checked under both the list and tuple containers, so a
capability reachable for one container but not the other -- the asymmetry class
of hole -- fails here rather than shipping silently.

PRODUCT: every schema node, in every walk mode
"""

import copy
import json
import pickle
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
    "Coll:frozenset": frozenset[int],
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
    assert repr(compiled.simplify()) == repr(compiled.simplify().simplify())  # ty: ignore[deprecated]


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
    "Coll:frozenset": ([frozenset(), frozenset({1, 2})], [frozenset({"a"}), [1], {1}]),
    "KeyedMap:record": ([{"x": 1}], [{"x": "a"}, {}, 1]),
    "KeyedMap:mapping": ([{}, {"a": 1, "b": 2}], [{"a": "x"}, [1]]),
    "Union": ([1, "a", True], [1.5, None, b"x"]),
    "Intersection": ([1, True], ["a", 1.5]),  # int and not str
    "Complement": (["a", 1.5, None], [1, True]),  # not int
    "Refine": ([0, 1, 5], [-1, "a"]),  # int >= 0
    # The instance and attribute nodes, with live instances. A dataclass
    # instance whose field is mistyped is the case that separates the record
    # from the class: the class admits it and the attribute record does not.
    "Instance": ([_Klass()], [1, "a", None, object()]),
    "Object": ([_Record(1)], [_Record("a"), 1, {"x": 1}, object()]),  # ty: ignore[invalid-argument-type]
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


#: The entry points the two loops above ask, named as the stub names them.
#: `in` is `__contains__`, and a mode or a bytes spelling is the same entry.
_ASKED_ENTRY_POINTS = {
    "is_valid",
    "__contains__",
    "validate",
    "ensure",
    "is_valid_json",
    "validate_json",
    "load",
}


def _validator_class_body() -> str:
    """Give the stub's `Validator` class, up to the next top-level statement."""
    stub = ROOT / "python" / "valgebra" / "_valgebra.pyi"
    text = stub.read_text(encoding="utf-8")
    found = re.search(r"^class Validator\b", text, re.MULTILINE)
    assert found is not None, "the stub declares no Validator class"
    start = found.start()
    after = re.search(r"^(?:def |class |[A-Za-z_]+:)", text[start + 1 :], re.MULTILINE)
    return text[start : start + 1 + after.start()] if after else text[start:]


def _entry_points_in_the_stub() -> set[str]:
    """Give every method of the stub's validator that takes a value or text."""
    body = _validator_class_body()
    found = set()
    for match in re.finditer(r"def (\w+)\(self, (\w+): ", body):
        name, parameter = match.groups()
        if parameter in {"obj", "data"}:
            found.add(name)
    return found


def test_the_entry_point_loops_ask_every_entry_point_the_stub_declares() -> None:
    """The columns of the entry-point run are read from the stub.

    The loops above name seven ways of asking, by hand. A method added to the
    stub that takes a value would be an eighth the table never runs, so the
    hand list is held to the stub in both directions.
    """
    declared = _entry_points_in_the_stub()
    assert len(declared) >= 7, sorted(declared)
    asked = {
        name.removesuffix("_fail_fast")
        .removesuffix("_bytes")
        .replace("in", "__contains__")
        if name == "in"
        else name.removesuffix("_fail_fast").removesuffix("_bytes")
        for name in [*_object_path(Validator(int), 1), *_json_path(Validator(int), "1")]
    }
    assert asked == _ASKED_ENTRY_POINTS
    assert declared == _ASKED_ENTRY_POINTS, (
        f"entry points the stub declares and the run does not ask: "
        f"{sorted(declared - _ASKED_ENTRY_POINTS)}; asked and not declared: "
        f"{sorted(_ASKED_ENTRY_POINTS - declared)}"
    )


#: The relations and operators the stub declares that take no value, each
#: with the answer every node must give when asked about itself.
_SELF_ANSWERS: dict[str, Callable[[Validator, object], bool]] = {
    "is_subtype_of": lambda v, spec: v.is_subtype_of(spec),
    "is_equivalent": lambda v, spec: v.is_equivalent(spec),
    "relation_to": lambda v, spec: v.relation_to(spec) == "subset",
    "is_empty": lambda v, spec: v.is_empty() is (spec is nothing),
    "open": lambda v, _spec: v.is_subtype_of(v.open()),
    "close": lambda v, _spec: v.close().is_subtype_of(v),
    "simplify": lambda v, _spec: v.simplify().is_equivalent(v),  # ty: ignore[deprecated]
    "__eq__": lambda v, spec: v == Validator(spec),
    "__hash__": lambda v, spec: hash(v) == hash(Validator(spec)),
    "__or__": lambda v, _spec: (v | nothing).is_equivalent(v),
    "__ror__": lambda v, _spec: (nothing | v).is_equivalent(v),
    "__copy__": lambda v, _spec: copy.copy(v) == v,
    "__deepcopy__": lambda v, _spec: copy.deepcopy(v) == v,
    "__reduce__": lambda v, _spec: _refuses_to_pickle(v),
}


def _refuses_to_pickle(v: Validator) -> bool:
    try:
        pickle.dumps(v)
    except TypeError:
        return True
    return False


@pytest.mark.parametrize("label", list(_NODES))
def test_every_relation_and_operator_answers_for_every_node(label: str) -> None:
    """The operator-by-node product, each cell with its answer asserted.

    Every relation and operator the stub declares that asks about a schema
    rather than a value is put to every node representative, and the answer
    it must give of a schema against itself is asserted: reflexivity for the
    relations, identity for the operators, refusal for pickling. A node an
    operator answers differently for is a cell this fails by name.
    """
    spec = _NODES[label]
    compiled = Validator(spec)
    for name, answer in _SELF_ANSWERS.items():
        assert answer(compiled, spec), f"{name} over {label}"


def test_the_self_answers_are_every_schema_method_the_stub_declares() -> None:
    """The rows of the product are read from the stub, in both directions."""
    # A method of a validator takes `self`; the constructor and the subscript
    # take the class.
    declared = set(re.findall(r"def (\w+)\(\s*self\b", _validator_class_body()))
    asks_about_a_schema = declared - _entry_points_in_the_stub()
    assert asks_about_a_schema == set(_SELF_ANSWERS), (
        f"declared and not asked: {sorted(asks_about_a_schema - set(_SELF_ANSWERS))}; "
        f"asked and not declared: {sorted(set(_SELF_ANSWERS) - asks_about_a_schema)}"
    )


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
    # Every node kind carries an independent membership case, the instance and
    # attribute nodes included, so every row runs through every entry point.
    covered = set(_MEMBERSHIP)
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
