import dataclasses
import enum
from typing import Annotated, Any, Literal

import annotated_types as at
import pytest

from valgebra import (
    MAX_SCHEMA_DEPTH,
    Regex,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (int, "int"),
        (str, "str"),
        (bool, "bool"),
        (float, "float"),
        (bytes, "bytes"),
        (None, "None"),
        (object, "anything"),
        (Any, "Any"),
        (list[int], "list[int]"),
        (set[int], "set[int]"),
        (frozenset[int], "frozenset[int]"),
        (dict[str, int], "dict[str, int]"),
        (tuple[int, str], "tuple[int, str]"),
        (tuple[int, ...], "tuple[int, ...]"),
        (tuple[str, int, ...], "tuple[str, int, ...]"),  # ty: ignore[invalid-type-form]
        (list[dict[str, int]], "list[dict[str, int]]"),
        (int | str, "int | str"),
        (Literal["a"], "Literal['a']"),
        (Literal["a", "b"], "Literal['a'] | Literal['b']"),
        ({"name": str, "age?": int}, "{'age?': int, 'name': str}"),
        (Annotated[int, at.Ge(0)], "Annotated[int, Ge(0)]"),
        (Annotated[str, at.MinLen(1)], "Annotated[str, MinLen(1)]"),
        # The nullary product. Python spells the empty subscript `tuple[()]`;
        # `tuple[]` is not an expression at all.
        (tuple[()], "tuple[()]"),
    ],
)
def test_repr_renders_the_annotation(schema: object, expected: str) -> None:
    assert repr(Validator(schema)) == expected


# The namespace the rendered form is evaluated in to re-parse it. It holds every
# name a round-trippable repr can mention.
_ROUNDTRIP_NS = {
    "int": int,
    "str": str,
    "bool": bool,
    "float": float,
    "bytes": bytes,
    "list": list,
    "set": set,
    "frozenset": frozenset,
    "tuple": tuple,
    "dict": dict,
    "Any": Any,
    "Literal": Literal,
    "Annotated": Annotated,
    "Ge": at.Ge,
    "Gt": at.Gt,
    "Le": at.Le,
    "Lt": at.Lt,
    "MinLen": at.MinLen,
    "MaxLen": at.MaxLen,
    # The package's own marker, which a pattern refinement renders as a call
    # to. It is not `annotated_types`', so a caller reading the repr of one
    # reaches for this name and not a third party's.
    "Regex": Regex,
    "anything": anything,
    "nothing": nothing,
    "union": union,
    "intersection": intersection,
    "complement": complement,
    "recursive": recursive,
}

# The round-trippable subset: every form whose repr re-parses to the same schema.
# Two are excluded by design and no more: a class renders only its name and a
# predicate renders as `Predicate(...)`, and neither is an expression that
# rebuilds what it names -- a class is an object, and a predicate is a callable.
# Everything else here is, the recursive form and the open record included.
ROUNDTRIP_SCHEMAS = [
    int,
    bool,
    float,
    bytes,
    None,
    nothing,
    object,
    Any,
    list[int],
    set[int],
    frozenset[int],
    dict[str, int],
    tuple[int, str],
    tuple[int, ...],
    tuple[str, int, ...],  # ty: ignore[invalid-type-form]
    list[dict[str, int]],
    int | str,
    Literal["a"],
    Literal["a", "b"],
    {"name": str, "age?": int},
    Annotated[int, at.Ge(0)],
    Annotated[str, at.MinLen(1)],
    union(int, str),
    intersection(int, complement(bool)),
    complement(int),
    # The three forms 21.4 fixed: each rendered as something that was not an
    # expression, or was one that built a different schema.
    tuple[()],
    recursive(lambda t: int | list[t]),  # ty: ignore[invalid-type-form]
    Validator({"name": str}).open(),
    recursive(lambda t: {"value": int, "left?": t}),
    # The package's own marker. It renders as a call and the call rebuilds it,
    # so it belongs in this list rather than beside the two exclusions -- and
    # the namespace below has to carry the name, or a repr a caller reads is
    # one they cannot evaluate.
    Annotated[str, Regex(r"[a-z]+")],
    Annotated[str, Regex(r"\\d+"), at.MinLen(2)],
]


def test_repr_of_class_and_recursive_forms() -> None:
    # A class renders as its name, which names the class rather than rebuilding
    # it -- the one form that stays a rendering, because a class is an object
    # and not an expression.
    class Color(enum.Enum):
        RED = 1

    @dataclasses.dataclass
    class Point:
        x: int

    assert repr(Validator(Color)) == "Color"
    assert repr(Validator(Point)) == "Point"
    # A class with attributes is a meet of its class atom and a record of them,
    # and a later meet flattens beside that pair: the class is still named, and
    # the other members render as themselves.
    assert repr(Validator(intersection(Point, int))) == "intersection(Point, int)"
    assert (
        repr(recursive(lambda s: {"v": int, "n?": s}))
        == "recursive(lambda X: {'n?': X, 'v': int})"
    )


# A spread of values to witness that two validators accept the same set, rather
# than only that a repr string is stable.
_WITNESS_VALUES = [
    None,
    True,
    False,
    0,
    1,
    -1,
    1.5,
    "x",
    "",
    b"x",
    b"",
    [1],
    [1, "a"],
    [],
    {1},
    {"k": 1},
    (1,),
    (1, "a"),
]


@pytest.mark.parametrize("schema", ROUNDTRIP_SCHEMAS)
def test_repr_round_trips_through_eval(schema: object) -> None:
    # repr is a fixpoint on this subset: rendering, re-parsing, and rendering
    # again yields the same string, so the printed form really does reconstruct
    # the schema.
    rendered = repr(Validator(schema))
    rebuilt = Validator(eval(rendered, dict(_ROUNDTRIP_NS)))  # noqa: S307
    assert repr(rebuilt) == rendered
    # The fixpoint alone would pass for a wrong-but-stable repr; require the
    # reconstructed validator to accept exactly the same values as the original,
    # so the printed form preserves meaning, not just its own shape.
    original = Validator(schema)
    for value in _WITNESS_VALUES:
        assert rebuilt.is_valid(value) == original.is_valid(value)


# --- The nullary combinators repr as their identities -------------------------
#
# A union of no members denotes the bottom and a meet of none denotes the top.
# The repr is the annotation that rebuilds the validator, so it names the
# identity rather than printing the empty join of no members.


@pytest.mark.parametrize(
    ("built", "expected"),
    [(union(), "nothing"), (intersection(), "anything")],
)
def test_a_nullary_combinator_reprs_as_its_identity(
    built: Validator, expected: str
) -> None:
    assert repr(built) == expected


def test_a_nullary_combinator_denotes_its_identity() -> None:
    assert not union().is_valid(1)
    assert intersection().is_valid(1)


def test_the_binders_a_nested_fixpoint_names_are_distinct() -> None:
    """Each `recursive` in a repr binds a name of its own, and re-parses.

    A fixpoint's back edge renders as the lambda's own parameter, so nested
    ones need names that do not collide: two `X`es would make the inner one
    shadow the outer, and the rendered expression would build a schema where
    the outer back edge points at the inner fixpoint. Three letters, then a
    numbered spelling, which is what keeps the supply from running out.
    """

    def nest(depth: int) -> Validator:
        if depth == 0:
            return Validator(int)
        return recursive(lambda inner, d=depth: {"v": nest(d - 1), "s?": inner})

    three = repr(nest(3))
    assert "lambda X:" in three
    assert "lambda Y:" in three
    assert "lambda Z:" in three
    # Past the letters the names are numbered rather than repeated.
    four = repr(nest(4))
    assert "lambda T3:" in four
    assert four.count("lambda X:") == 1

    # And each re-parses to the schema it came from, which is the property the
    # distinct names exist for.
    for depth in (1, 2, 3, 4):
        schema = nest(depth)
        rebuilt = Validator(eval(repr(schema), dict(_ROUNDTRIP_NS)))  # noqa: S307
        assert rebuilt == schema, repr(schema)


def test_no_schema_a_caller_can_build_renders_the_truncation_mark() -> None:
    """The renderer's depth bound is past the one a schema can be built at.

    `docs/16-api.md` names `...` as what a repr deeper than the renderer's own
    bound shows. A caller cannot reach it: the frontend refuses to compile a
    schema nested past `MAX_SCHEMA_DEPTH` first, so the two bounds are ordered
    and the truncating arm is unreachable through any annotation. Asserted at
    the deepest schema that compiles, so a change to either bound that reversed
    the order fails here rather than making a repr silently lossy.
    """
    deepest: object = int
    for _ in range(MAX_SCHEMA_DEPTH - 1):
        deepest = list[deepest]  # type: ignore[valid-type]
    rendered = repr(Validator(deepest))
    assert "..." not in rendered
    assert rendered.startswith("list[")

    # One past what compiles is a refusal rather than a truncated rendering.
    with pytest.raises(ValueError, match="too deep"):
        Validator(list[deepest])  # type: ignore[valid-type]


def test_a_union_renders_its_literals_after_the_sets_they_sit_beside() -> None:
    """A union of a kind and some of its values renders in the lattice's order.

    The members are not kept as written -- a schema is built in the normal
    form, so two spellings of one union are one schema and print alike. What a
    reader sees is the sets first and the literals after, whatever order the
    call used, and each literal on its own rather than gathered into one
    `Literal[...]`: they are separate members of the union, and the rendering
    says so.
    """
    written_after = union(int, 1, 2, 3)
    written_among = union(1, 2, int, 3)
    assert repr(written_after) == "int | Literal[1] | Literal[2] | Literal[3]"
    assert repr(written_among) == repr(written_after)
    assert written_among == written_after
