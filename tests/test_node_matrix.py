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
    complement,
    fixed_sequence,
    intersection,
    nothing,
    recursive,
    simplify,
    union,
    validator,
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
    compiled = validator(spec)  # the frontend builds it
    assert isinstance(compiled.is_empty(), bool)  # emptiness terminates
    assert compiled.is_subtype_of(spec)  # reflexivity
    assert compiled.is_equivalent(spec)  # self-equivalence
    # The rendered form is stable under simplification.
    assert repr(simplify(compiled)) == repr(simplify(simplify(compiled)))


# Each sequence shape must be reachable via both containers; the list and tuple
# forms of one shape are unrelated, since the container is part of the type.
_SHAPES: dict[str, tuple[object, object]] = {
    "homogeneous": (list[int], tuple[int, ...]),
    "fixed": (fixed_sequence(int, str), tuple[int, str]),
    "prefixtail": ([int, int, ...], _pt_tuple(int, int, ...)),
}


@pytest.mark.parametrize(
    ("listed", "tupled"), list(_SHAPES.values()), ids=list(_SHAPES)
)
def test_sequence_shapes_reach_both_containers(listed: object, tupled: object) -> None:
    list_form = validator(listed)
    tuple_form = validator(tupled)
    assert list_form.is_subtype_of(listed)  # the list form builds and is reflexive
    assert tuple_form.is_subtype_of(tupled)  # the tuple form builds and is reflexive
    assert not list_form.is_subtype_of(tupled)  # a list is not a tuple
    assert not tuple_form.is_subtype_of(listed)  # a tuple is not a list
