"""Completeness ledger for the decision procedure.

The decision procedure is sound everywhere and complete on part of its domain.
This module measures the incompleteness directly, rather than letting it hide
behind a soundness-only suite.

Two mechanisms:

- A parametrized ledger of relations that hold by construction. Each case
  asserts the procedure decides a true relation. Cases the procedure decides are
  gates: a regression that makes one conservative fails the build. Cases it does
  not decide are marked ``xfail(strict=True)``, so closing one fails the strict
  mark and forces its entry out of the ledger and off the boundary the
  decidability page publishes. A new true relation the procedure declines, added
  here without a mark, fails outright.
- A finite-universe soundness fuzzer. For random schema pairs, a claimed subtype
  never accepts a value the supertype rejects across a region-complete universe.

LEDGER: every relation the procedure must decide is decided, and every relation
it declines is a strict expected failure that fails the day it is decided.

Every relation is written the way a caller writes it. A case built from the other
operand -- a wider union built by adding a member to the narrower one, rather
than written out -- shares its constants with it, and the shortcuts the procedure
takes are keyed on exactly that, so such a case measures the spelling instead of
the relation.
"""

import dataclasses
import enum
from typing import (
    Annotated,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    NoReturn,
    Optional,
    TypeVar,
    Union,
)

import annotated_types as at
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from valgebra import (
    Regex,
    Validator,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

# A recursive schema, reused below to record a reflexivity hole.
_RECURSIVE = recursive(lambda t: union(None, {"value": int, "next": t}))


class _Pair(NamedTuple):
    """A named tuple, whose instances are two-tuples of ints."""

    left: int
    right: int


# A region-complete value universe whose numbers straddle the bounds the ledger
# uses, so a subset relation over it reflects the true relation on those cases.
_UNIVERSE = [
    None,
    True,
    False,
    -5,
    -1,
    0,
    1,
    5,
    10,
    11,
    100,
    "",
    "a",
    "ab",
    b"",
    b"x",
    -1.5,
    1.5,
    [],
    [1],
    [1, 2],
    ["a"],
    (1,),
    (1, 2),
    {1},
    frozenset({1}),
    {"k": 1},
    {5: True},
    {"k": 1, 5: True},
    {"a": 1},
    {"a": 1, "k": "x"},
    {"a": 1, "b": "y"},
    {"a": 1, "b": "y", "k": "z"},
]


def _accepted(spec: object) -> frozenset[int]:
    """Return the universe indices a schema accepts -- its denotation on `U`."""
    compiled = Validator(spec)
    return frozenset(i for i, value in enumerate(_UNIVERSE) if compiled.is_valid(value))


# --- The ledger: relations that hold by construction -------------------------
#
# Each tuple is (operation, left, right). For "subtype" and "equivalent" the
# right side is the comparison schema; for "empty" it is None. Every relation
# listed is true set-theoretically.

_GE0 = Annotated[int, at.Ge(0)]
_GE0_LE10 = Annotated[int, at.Ge(0), at.Le(10)]
_GE10_LE0 = Annotated[int, at.Ge(10), at.Le(0)]
_GT0 = Annotated[int, at.Gt(0)]
_LT1 = Annotated[int, at.Lt(1)]


@dataclasses.dataclass
class _Animal:
    name: str


@dataclasses.dataclass
class _Dog(_Animal):
    breed: str


def _check(operation: str, left: object, right: object) -> None:
    compiled = Validator(left)
    if operation == "subtype":
        assert compiled.is_subtype_of(right)
    elif operation == "equivalent":
        assert compiled.is_equivalent(right)
    elif operation == "empty":
        assert compiled.is_empty()
    else:  # pragma: no cover - guards against a typo in a case
        msg = f"unknown operation {operation!r}"
        raise AssertionError(msg)


_DECIDED = [
    pytest.param("subtype", bool, int, id="bool<=int"),
    pytest.param("subtype", 1, int, id="Literal[1]<=int"),
    pytest.param("subtype", int, union(int, str), id="int<=int|str"),
    pytest.param("subtype", list[bool], list[int], id="list[bool]<=list[int]"),
    pytest.param("subtype", {"x": int}, {"x": int, "y?": str}, id="{x}<={x,y?}"),
    pytest.param(
        "subtype", [bool, int, ...], [int, int, ...], id="[bool,int,...]<=[int,int,...]"
    ),
    pytest.param("empty", intersection(int, str), None, id="empty:int&str"),
    # The lattice bounds, decided by EMPTINESS rather than by the shape of the
    # atom. A schema that denotes the empty set without being spelled `nothing`
    # is below everything; one that covers the universe without being spelled
    # `anything` is above everything. Both were enumerated only for the atoms,
    # which is a rule confirming itself -- and the region check upstream decides
    # a scalar right-hand side correctly, so these want a container, a record or
    # the gradual atom on the other side to reach the arm at all.
    pytest.param(
        "subtype", intersection(int, complement(int)), list[int], id="empty<=list[int]"
    ),
    pytest.param(
        "subtype", intersection(int, complement(int)), {"a": int}, id="empty<={a:int}"
    ),
    pytest.param("subtype", {"a": NoReturn}, list[int], id="{a:Never}<=list[int]"),
    pytest.param(
        "subtype", list[int], union(int, complement(int)), id="list[int]<=universal"
    ),
    pytest.param(
        "subtype", {"a": int}, union(int, complement(int)), id="{a:int}<=universal"
    ),
    pytest.param(
        "equivalent",
        intersection(int, complement(int)),
        nothing,
        id="int&~int==nothing",
    ),
    pytest.param("equivalent", union(bool, int), int, id="bool|int==int"),
    pytest.param("equivalent", intersection(int, int), int, id="int&int==int"),
    # A refinement is a subtype of its base, and a refinement with more bound or
    # length constraints is a subtype of one with fewer (equal bounds share a
    # pool index, so nested bounds decide by syntactic containment).
    pytest.param("subtype", _GE0, int, id="refine:Ge(0)<=int"),
    pytest.param("subtype", _GE0_LE10, _GE0, id="refine:Ge(0)Le(10)<=Ge(0)"),
    # Bound entailment: a tighter bound is a subtype of a looser one even when the
    # bound values differ, decided through the ordering oracle rather than a
    # verbatim constraint match.
    pytest.param(
        "subtype",
        Annotated[int, at.Ge(5)],
        Annotated[int, at.Ge(0)],
        id="refine:Ge(5)<=Ge(0)",
    ),
    pytest.param(
        "subtype",
        Annotated[int, at.Gt(5)],
        Annotated[int, at.Ge(0)],
        id="refine:Gt(5)<=Ge(0)",
    ),
    pytest.param(
        "subtype",
        Annotated[int, at.Le(0)],
        Annotated[int, at.Le(5)],
        id="refine:Le(0)<=Le(5)",
    ),
    pytest.param(
        "subtype",
        Annotated[str, at.MinLen(5)],
        Annotated[str, at.MinLen(2)],
        id="refine:MinLen(5)<=MinLen(2)",
    ),
    pytest.param(
        "subtype",
        Annotated[str, at.MaxLen(2)],
        Annotated[str, at.MaxLen(5)],
        id="refine:MaxLen(2)<=MaxLen(5)",
    ),
    # A bound conjunction whose lower bound exceeds its upper bound is empty,
    # whether the bounds sit on one refinement or across an intersection.
    pytest.param("empty", _GE10_LE0, None, id="empty:Ge(10)Le(0)"),
    pytest.param(
        "empty",
        intersection(_GE0, Annotated[int, at.Lt(0)]),
        None,
        id="empty:Ge(0)&Lt(0)",
    ),
    # An integer-discrete open interval whose endpoints are ordered but adjacent
    # in the integers admits no value: there is no `int` strictly between 0 and 1.
    pytest.param(
        "empty", Annotated[int, at.Gt(0), at.Lt(1)], None, id="empty:int-open-(0,1)"
    ),
    # The endpoints need not be integers themselves; the interval still skips
    # every integer, whether the bounds are strict (open) or inclusive (closed).
    pytest.param(
        "empty",
        Annotated[int, at.Gt(0.5), at.Lt(0.9)],
        None,
        id="empty:int-open-(0.5,0.9)",
    ),
    pytest.param(
        "empty",
        Annotated[int, at.Ge(0.5), at.Le(0.9)],
        None,
        id="empty:int-closed-[0.5,0.9]",
    ),
    # An interval that skips every integer is empty whichever way its bounds are
    # spelled: on one refinement, or across an intersection whose members bound
    # the meet to the integers. `bool` is bounded to the integers too.
    pytest.param(
        "empty", intersection(_GT0, _LT1), None, id="empty:int-open-(0,1)-across-a-meet"
    ),
    pytest.param(
        "empty", Annotated[bool, at.Gt(0), at.Lt(1)], None, id="empty:bool-open-(0,1)"
    ),
    # An attribute schema is a subtype of one over a base class it carries every
    # attribute of: the nominal question is the one the leaf oracle answers for an
    # isinstance atom.
    pytest.param("subtype", _Dog, _Animal, id="attrs:Dog<=Animal"),
    # An attribute schema is its class's isinstance atom narrowed by an attribute
    # record, so it is below the atom a bare class compiles to.
    pytest.param("subtype", _Dog, complement(complement(_Dog)), id="attrs:Dog<=~~Dog"),
    # An intersection that mixes a recursive reference with a union is a subtype
    # of itself: reflexivity holds even when the meet contains its own supertype.
    pytest.param(
        "subtype",
        intersection(_RECURSIVE, union(int, str)),
        intersection(_RECURSIVE, union(int, str)),
        id="reflexive:intersection(rec,union)",
    ),
    # A mapping is a subtype of one with more clauses subsuming its own; a closed
    # record is a subtype of an open map that declares its fields.
    pytest.param(
        "subtype", {str: int}, {str: int, int: bool}, id="map:{str}<={str,int}"
    ),
    pytest.param("subtype", {}, {str: int}, id="map:{}<={str:int}"),
    # A record mixed with a catch-all narrows field-wise and clause-wise.
    pytest.param(
        "subtype", {"a": bool, str: bool}, {"a": int, str: int}, id="map:mixed-narrow"
    ),
    # A mixed map with an extra field covered by the supertype's catch-all.
    pytest.param(
        "subtype",
        {"a": int, "b": str, str: bytes},
        {"a": int, str: object},
        id="map:mixed-extra-field-covered",
    ),
    # The supertype declares an *optional* field the subtype lacks; the subtype's
    # catch-all value type fits it, so the relation decides.
    pytest.param(
        "subtype",
        {"a": int, str: int},
        {"a": int, "b?": int, str: int},
        id="map:b-extra-optional-covered",
    ),
    # The subtype is a pure mapping whose catch-all covers the supertype's optional
    # field and catch-all alike.
    pytest.param(
        "subtype",
        {str: int},
        {"b?": int, str: int},
        id="map:pure<=mixed-optional",
    ),
    # A closed record against a catch-all mapping. One rule serves every keyed-map
    # shape: a branch of its own for the closed record read a field the supertype
    # covered through a catch-all as undecided, though the general rule beside it
    # already decided exactly that.
    pytest.param("subtype", {"x": int}, {str: int}, id="map:closed-record<=mapping"),
    pytest.param(
        "subtype", {"x": int}, {str: object}, id="map:closed-record<=wide-mapping"
    ),
    # Inclusion in a complement, which is `A ∩ B = ∅` and nothing structural: a
    # complement offers no shape on the right to recurse into.
    pytest.param(
        "subtype", list[int], complement(int), id="complement:list[int]<=~int"
    ),
    pytest.param(
        "subtype", dict[str, int], complement(str), id="complement:dict<=~str"
    ),
    pytest.param("subtype", int, complement(str), id="complement:int<=~str"),
]

# Known decision-completeness misses: true relations the procedure declines.
#
# Each is marked strict, so the day a rule decides one the mark fails and the
# entry leaves both this list and the conservative half of the decidability page.
# They are grouped by what the procedure is missing rather than by node kind,
# because the grouping is the diagnosis.
_MISSED = pytest.mark.xfail(strict=True, reason="the procedure declines this relation")

_LEDGERED = [
    # Emptiness has no structural-inclusion rule, so a meet with a complement is
    # not decided even where the inclusion under it is.
    pytest.param(
        "empty",
        intersection(list[bool], complement(list[int])),
        None,
        id="empty:list[bool]&~list[int]",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        intersection(union(list[int], str), complement(str)),
        list[int],
        id="(list[int]|str)&~str<=list[int]",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        intersection(Literal["a", "b"], complement(Validator(Literal["a"]))),
        Literal["b"],
        id="L[a,b]&~L[a]<=L[b]",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        {"a": union(int, str)},
        union(Validator({"a": int}), Validator({"a": str})),
        id="{a:int|str}<={a:int}|{a:str}",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        tuple[int],
        complement(tuple[str]),
        id="tuple[int]<=~tuple[str]",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        tuple[int, str],
        complement(tuple[str, int]),
        id="tuple[int,str]<=~tuple[str,int]",
        marks=_MISSED,
    ),
    # A container meet is not intersected componentwise, so a meet that is the
    # empty container is not recognised as one.
    pytest.param(
        "subtype",
        intersection(list[int], list[str]),
        [],
        id="list[int]&list[str]<=[]",
        marks=_MISSED,
    ),
    pytest.param(
        "equivalent",
        intersection(set[int], set[str]),
        set[nothing],  # ty: ignore[invalid-type-form]
        id="set[int]&set[str]==set[nothing]",
        marks=_MISSED,
    ),
    # A scalar kind is a region bit rather than a set of its values, so a finite
    # kind is not the union of its members and a negated literal has nowhere to go.
    pytest.param(
        "subtype", bool, Literal[True, False], id="bool<=L[True,False]", marks=_MISSED
    ),
    pytest.param(
        "subtype",
        Literal[1],
        intersection(int, complement(bool)),
        id="L[1]<=int&~bool",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        int,
        union(intersection(int, complement(Validator(Literal[1]))), Literal[1]),
        id="int<=(int&~L[1])|L[1]",
        marks=_MISSED,
    ),
    # A refinement's bounds are compared, and the values under them are not, so a
    # base that is not itself a refinement does not reach them.
    pytest.param(
        "subtype", bool, Annotated[int, at.Ge(0)], id="bool<=int&Ge(0)", marks=_MISSED
    ),
    pytest.param(
        "subtype",
        _GT0,
        Annotated[int, at.Ge(1)],
        id="int&Gt(0)<=int&Ge(1)",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        intersection(_GT0, Annotated[int, at.Lt(10)]),
        Annotated[int, at.Gt(0), at.Lt(10)],
        id="meet-of-refinements<=joint-refinement",
        marks=_MISSED,
    ),
    pytest.param(
        "empty", Annotated[bool, at.Ge(2)], None, id="empty:bool&Ge(2)", marks=_MISSED
    ),
    # A length bound is opaque to the shape it bounds.
    pytest.param(
        "empty",
        Annotated[tuple[int, int], at.MinLen(3)],
        None,
        id="empty:2-tuple&MinLen(3)",
        marks=_MISSED,
    ),
    pytest.param(
        "empty",
        Annotated[list[nothing], at.MinLen(1)],  # ty: ignore[invalid-type-form]
        None,
        id="empty:list[nothing]&MinLen(1)",
        marks=_MISSED,
    ),
    pytest.param(
        "empty",
        recursive(lambda t: Annotated[list[t], at.MinLen(1)]),  # ty: ignore[invalid-type-form]
        None,
        id="empty:mu-t.list[t]&MinLen(1)",
        marks=_MISSED,
    ),
    # A pattern and a divisor are related only by being written alike, so neither
    # the language nor the divisibility is read.
    pytest.param(
        "subtype",
        Annotated[int, at.MultipleOf(4)],
        Annotated[int, at.MultipleOf(2)],
        id="MultipleOf(4)<=MultipleOf(2)",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        Annotated[str, Regex("a")],
        Annotated[str, Regex("a|b")],
        id="Regex(a)<=Regex(a|b)",
        marks=_MISSED,
    ),
    # A map's domain is the field list as written, and a key type is matched
    # against the string atom rather than asked whether it admits the name.
    pytest.param(
        "subtype",
        {"a": int},
        {Literal["a"]: int},
        id="{a:int}<=dict[L[a],int]",
        marks=_MISSED,
    ),
    pytest.param(
        "equivalent", {"a?": nothing}, {}, id="{a?:nothing}=={}", marks=_MISSED
    ),
    pytest.param(
        "equivalent",
        dict[str, nothing],  # ty: ignore[invalid-type-form]
        {},
        id="dict[str,nothing]=={}",
        marks=_MISSED,
    ),
    pytest.param(
        "equivalent",
        dict[nothing, int],  # ty: ignore[invalid-type-form]
        {},
        id="dict[nothing,int]=={}",
        marks=_MISSED,
    ),
    pytest.param(
        "empty",
        intersection({"a": int}, dict[str, str]),
        None,
        id="empty:{a:int}&dict[str,str]",
        marks=_MISSED,
    ),
    pytest.param(
        "empty",
        intersection({"a": int}, dict[int, str]),
        None,
        id="empty:{a:int}&dict[int,str]",
        marks=_MISSED,
    ),
    pytest.param(
        "equivalent",
        intersection({"a?": int}, {"b?": int}),
        {},
        id="{a?:int}&{b?:int}=={}",
        marks=_MISSED,
    ),
    # An attribute schema and the shape its instances have are unrelated.
    pytest.param(
        "subtype",
        _Pair,
        tuple[int, int],
        id="NamedTuple<=tuple[int,int]",
        marks=_MISSED,
    ),
    # A union on the right is tried branch by branch before a reference is
    # unfolded or a refinement drops to its base, so neither reaches the arm that
    # would decide it.
    pytest.param(
        "subtype",
        _RECURSIVE,
        union(None, {"value": int, "next": _RECURSIVE}),
        id="mu-t<=its-own-body",
        marks=_MISSED,
    ),
    pytest.param(
        "subtype",
        # `int | str` rather than `union(int, str)`: the two build the same
        # schema, and `Annotated` takes a *type*, which 3.10 checks as the form
        # is built -- so a runtime validator there raises at import and takes
        # the whole module out of collection.
        Annotated[int | str, at.Ge(0)],
        union(int, str),
        id="refinement-of-a-union<=the-union",
        marks=_MISSED,
    ),
    # The complement laws are settled by the constructors, so a shape they do not
    # reach is not decided: a respelling, and a recursive definition whose two
    # occurrences are separate definitions.
    pytest.param(
        "subtype",
        Validator(list[int]),
        complement(union(complement(Validator(list[int])), nothing)),
        id="A<=~(~A|nothing)",
        marks=_MISSED,
    ),
    pytest.param(
        "empty",
        intersection(_RECURSIVE, complement(_RECURSIVE)),
        None,
        id="empty:mu-t&~mu-t",
        marks=_MISSED,
    ),
]


@pytest.mark.parametrize(("operation", "left", "right"), _DECIDED + _LEDGERED)
def test_decision_decides_true_relations(
    operation: str, left: object, right: object
) -> None:
    _check(operation, left, right)


# Enum widening at the sizes a real annotation reaches. A literal union is the
# shape that grows: an error-code table, a currency list, a set of tags.


def _codes(members: int, *, extra: bool = False) -> Validator:
    """Build a literal union of `members` codes, pooling its own constants."""
    codes = [f"code_{index:05d}" for index in range(members)]
    if extra:
        codes.append("extra")
    return union(*[Validator(code) for code in codes])


@pytest.mark.parametrize("members", [8, 64, 256, 512])
def test_widening_a_table_is_decided_at_the_sizes_a_table_reaches(
    members: int,
) -> None:
    # Two tables written out separately, which is how a codebase with a schema in
    # two modules has them: neither is built from the other, so they share no
    # constant and the containment shortcut does not fire.
    assert _codes(members).is_subtype_of(_codes(members, extra=True))


@pytest.mark.parametrize("members", [1024, 4096])
@pytest.mark.xfail(
    strict=True,
    reason="two independently written tables distribute over each other, and the "
    "product of the member counts spends the decision budget",
)
def test_widening_two_separate_tables_stops_at_the_budget(members: int) -> None:
    assert _codes(members).is_subtype_of(_codes(members, extra=True))


@pytest.mark.parametrize("members", [1024, 4096])
def test_widening_a_table_by_a_member_is_decided_at_every_size(members: int) -> None:
    # The wider table built *from* the narrower one shares its constants, so the
    # narrow union is a branch of the wide one and containment settles it without
    # distributing. This is the shape the shortcut is for, and it is not the shape
    # above.
    narrow = _codes(members)
    assert narrow.is_subtype_of(union(narrow, Validator("extra")))


# The integer-discreteness rule must fire only where the base is integer-discrete
# and an integer genuinely fails to fit. These controls keep it from over-firing:
# a dense float base, an interval that still contains an integer, and an inclusive
# bound that lands on one.
_NON_EMPTY = [
    pytest.param(Annotated[float, at.Gt(0), at.Lt(1)], id="float-open-(0,1)"),
    # Across a meet, and on a boolean base, the rule must still fire only where an
    # integer genuinely fails to fit.
    pytest.param(
        intersection(_GT0, Annotated[int, at.Lt(2)]),
        id="int-open-(0,2)-across-a-meet-has-1",
    ),
    pytest.param(
        intersection(Annotated[float, at.Gt(0)], Annotated[float, at.Lt(1)]),
        id="float-open-(0,1)-across-a-meet",
    ),
    pytest.param(Annotated[bool, at.Ge(0), at.Le(1)], id="bool-closed-[0,1]-has-both"),
    pytest.param(Annotated[int, at.Gt(0), at.Lt(2)], id="int-open-(0,2)-has-1"),
    pytest.param(Annotated[int, at.Gt(0.5), at.Lt(1.5)], id="int-open-(0.5,1.5)-has-1"),
    pytest.param(Annotated[int, at.Ge(0), at.Le(0)], id="int-closed-[0,0]-has-0"),
    pytest.param(Annotated[int, at.Ge(0), at.Lt(1)], id="int-half-[0,1)-has-0"),
    pytest.param(Annotated[int, at.Gt(0), at.Le(1)], id="int-half-(0,1]-has-1"),
]


@pytest.mark.parametrize("spec", _NON_EMPTY)
def test_integer_discreteness_rule_does_not_over_fire(spec: object) -> None:
    # A false `is_empty` would be unsound: it would license dropping a value the
    # schema in fact admits. The float case is the key guard — the rule generalizes
    # to dense bases only at the cost of soundness.
    assert not Validator(spec).is_empty()


# --- Frontend integrity: non-value objects are rejected -----------------------
#
# A construct carrying no runtime value is rejected, not interned as a literal
# that silently accepts almost nothing.

_T = TypeVar("_T")

_REJECTED = [
    pytest.param(_T, id="TypeVar"),
    pytest.param(list[_T], id="list[TypeVar]"),
    pytest.param(Final, id="Final"),
    pytest.param(ClassVar, id="ClassVar"),
    pytest.param(Union, id="bare-Union"),
    pytest.param(Optional, id="bare-Optional"),
    pytest.param(Literal, id="bare-Literal"),
]


@pytest.mark.parametrize("schema", _REJECTED)
def test_frontend_rejects_non_value_objects(schema: object) -> None:
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        Validator(schema)


def test_value_literals_still_build() -> None:
    # The rejection above does not over-reach: genuine constant values still build
    # as typed literals.
    class Color(enum.Enum):
        RED = 1

    sentinel = object()
    assert Validator(1).is_valid(1)
    assert Validator("a").is_valid("a")
    assert Validator(Color.RED).is_valid(Color.RED)
    assert Validator(sentinel).is_valid(sentinel)
    assert not Validator(sentinel).is_valid(object())


# --- Finite-universe soundness fuzzer ----------------------------------------

_RECURSIVE_FAMILY = [
    _RECURSIVE,
    recursive(lambda t: union(int, [t])),
    recursive(lambda t: union(None, bool, int, str, [t], {str: t})),
]
_atoms = st.sampled_from(
    [int, str, bool, float, bytes, None, _GE0, _GE0_LE10, 0, 1, "a", *_RECURSIVE_FAMILY]
)


def _compose(children: st.SearchStrategy) -> st.SearchStrategy:
    pair = st.tuples(children, children)
    return st.one_of(
        children.map(lambda c: [c]),
        children.map(lambda c: {str: c}),
        pair.map(lambda p: {str: p[0], int: p[1]}),  # multi-clause mapping
        pair.map(lambda p: {"a": p[0], str: p[1]}),  # record mixed with a catch-all
        pair.map(lambda p: {"a": p[0], "b": p[1], str: p[0]}),  # two fields + catch-all
        pair.map(lambda p: union(p[0], p[1])),
        pair.map(lambda p: intersection(p[0], p[1])),
        children.map(complement),
    )


_schemas = st.recursive(_atoms, _compose, max_leaves=8)


@given(left=_schemas, right=_schemas)
def test_subtype_claims_hold_on_the_universe(left: object, right: object) -> None:
    # The soundness direction of the differential: a claimed subtype accepts no
    # universe value the claimed supertype rejects. A violation is a real
    # unsoundness, not a conservatism.
    try:
        compiled = Validator(left)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        # Reject an unbuildable spec through assume so Hypothesis counts it.
        assume(False)
        return
    if compiled.is_subtype_of(right):
        assert _accepted(left) <= _accepted(right)


@given(spec=_schemas)
def test_emptiness_claims_hold_on_the_universe(spec: object) -> None:
    # A schema reported empty accepts nothing in the universe. The converse does
    # not hold over a finite universe, so only this sound direction is asserted.
    try:
        compiled = Validator(spec)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        # Reject an unbuildable spec through assume so Hypothesis counts it.
        assume(False)
        return
    if compiled.is_empty():
        assert not _accepted(spec)


class _Partial:
    """A value whose equality admits some others of its own class."""

    __slots__ = ("accepts", "tag")

    def __init__(self, tag: str, accepts: frozenset[str]) -> None:
        self.tag, self.accepts = tag, accepts

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Partial) and other.tag in self.accepts

    def __hash__(self) -> int:
        return 0


def test_a_literal_is_not_always_a_singleton() -> None:
    """Why the entry above cannot be closed by comparing the two constants.

    `Literal` admits any constant, not only the types the typing spec allows in
    `Literal[...]`, and equality is the value's own. Two such literals can share
    a member while neither contains the other, so "not mutually subtypes" does
    not imply "disjoint" — and a rule reading it that way would report a meet
    empty that a value belongs to, which `is_empty` must never do.
    """
    left = Validator(_Partial("a", frozenset({"a"})))
    right = Validator(_Partial("b", frozenset({"b"})))
    shared = _Partial("s", frozenset({"a", "b"}))

    assert left.is_valid(shared)
    assert right.is_valid(shared)
    assert not left.is_subtype_of(right)
    assert not right.is_subtype_of(left)

    # Sound today, and the reason the narrow rule must be gated on the
    # constant's type rather than on the subtype relation.
    assert not intersection(left, right).is_empty()


def test_a_spec_literal_type_is_not_a_singleton_either() -> None:
    """Why narrowing the rule to `Literal[...]`'s own types does not rescue it.

    An enum member is a constant the typing spec admits in `Literal[...]`, and
    its equality is ordinary user code. A rule that fired on "both constants
    have a type `Literal[...]` admits, and they are distinct" would report this
    meet empty, and a value belongs to it.
    """

    class Loose(enum.Enum):
        A = "a"
        B = "b"

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Loose)

        def __hash__(self) -> int:
            return 0

    left, right = Validator(Loose.A), Validator(Loose.B)
    assert Loose.A is not Loose.B
    assert left.is_valid(Loose.B)
    assert right.is_valid(Loose.A)
    assert not intersection(left, right).is_empty()
