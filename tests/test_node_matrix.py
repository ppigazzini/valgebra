"""Node coverage matrix.

For a representative schema of every IR node kind, the frontend builds it, the
decision procedure handles it (reflexive, self-equivalent, emptiness
terminates), and its rendered form is stable under simplification. The
sequence-regex shapes are checked under both the list and tuple containers, so a
capability reachable for one container but not the other -- the asymmetry class
of hole -- fails here rather than shipping silently.
"""

from dataclasses import dataclass
from types import GenericAlias
from typing import Annotated, Any

import annotated_types as at
import pytest

from valgebra import (
    Validator,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)


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
    "Set": set[int],
    "FrozenSet": frozenset[int],
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
    "Set": ([set(), {1, 2}], [{"a"}, [1], frozenset({1})]),
    "FrozenSet": ([frozenset(), frozenset({1})], [{1}, [1]]),
    "KeyedMap:record": ([{"x": 1}], [{"x": "a"}, {}, 1]),
    "KeyedMap:mapping": ([{}, {"a": 1, "b": 2}], [{"a": "x"}, [1]]),
    "Union": ([1, "a", True], [1.5, None, b"x"]),
    "Intersection": ([1, True], ["a", 1.5]),  # int and not str
    "Complement": (["a", 1.5, None], [1, True]),  # not int
    "Refine": ([0, 1, 5], [-1, "a"]),  # int >= 0
    "Recursive": ([None, {"next": None}, {"next": {"next": None}}], [1, {"next": 1}]),
}


@pytest.mark.parametrize("label", list(_MEMBERSHIP))
def test_node_admits_its_denotation(label: str) -> None:
    compiled = Validator(_NODES[label])
    members, non_members = _MEMBERSHIP[label]
    for value in members:
        assert compiled.is_valid(value), f"{label} should admit {value!r}"
    for value in non_members:
        assert not compiled.is_valid(value), f"{label} should reject {value!r}"


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
