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

#: What the renderer prints where it gives up: past its depth bound, and for the
#: transient self-reference a compiled validator never holds. Spelled here so the
#: tests below read it from one place, as `docs/16-api.md` names it in one place.
TRUNCATED = "<...>"


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
    # Patterns the two languages spell differently. A control character is
    # `\u{7}` to Rust and `\x07` to Python, and the first is a syntax error in
    # the second; a quote picks the quoting. The pattern is a string the render
    # prints, so it has to be printed the way the reader's interpreter reads it.
    Annotated[str, Regex("\x07+")],
    Annotated[str, Regex("a'b")],
    Annotated[str, Regex("\u00e9+")],
]


def test_a_render_that_is_not_an_expression_refuses_rather_than_rebuilding() -> None:
    """The three renders that cannot be read back are each refused, not quiet.

    Everything else `repr` gives back is an expression that builds the same
    schema, so a form that is *not* one has to fail where it is read rather than
    build something else. The pages list these three beside the class name --
    which is not an expression either, and is the one that names its subject
    rather than failing.

    Each fails in its own place: a cut constant and a given-up render are syntax
    errors where they are parsed, and a predicate parses to a call the frontend
    refuses where the schema is built.
    """
    long_constant = Validator(Literal["x" * 500])  # ty: ignore[invalid-type-form]
    with pytest.raises(SyntaxError):
        eval(repr(long_constant), dict(_ROUNDTRIP_NS))  # noqa: S307

    shape = "a sequence in a union"
    given_up = Validator(_chain(shape, _CHAIN_SHAPES[shape][1]))
    with pytest.raises(SyntaxError):
        eval(repr(given_up), dict(_ROUNDTRIP_NS))  # noqa: S307

    predicate = Validator(Annotated[int, at.Predicate(lambda value: value > 0)])
    namespace = dict(_ROUNDTRIP_NS) | {"Predicate": at.Predicate}
    parsed = eval(repr(predicate), namespace)  # noqa: S307
    with pytest.raises(NotImplementedError):
        Validator(parsed)


def test_a_pattern_prints_as_python_spells_it() -> None:
    r"""The rendered marker is the marker's own repr, character for character.

    A pattern is a string, and a rendered schema claims to read back. Rust's
    debug spelling is not Python's -- it writes `\u{7}` where Python writes
    `\x07`, and double quotes where Python's repr picks single -- so a repr
    carrying a control character was a syntax error to the interpreter it was
    printed for, and one carrying none still disagreed with the marker beside
    it about how a string is written.
    """
    for pattern in ("[a-z]+", "\x07+", "a'b", "\u00e9+", '"'):
        marker = Regex(pattern)
        rendered = repr(Validator(Annotated[str, marker]))
        assert rendered == f"Annotated[str, {marker!r}]", rendered


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


def test_plain_nesting_never_reaches_the_renderer_s_bound() -> None:
    """One annotation's nesting is bounded below the renderer's own bound.

    The frontend refuses to compile a schema nested past `MAX_SCHEMA_DEPTH`, and
    the renderer's bound sits above it, so no *single* annotation can be written
    deep enough to truncate. Asserted at the deepest schema that compiles, so a
    change to either bound that reversed the order fails here.

    This is a fact about one annotation's depth and nothing more. A chain of
    definitions composes to a render deeper than either bound, which is what the
    test below is about.
    """
    deepest: object = int
    for _ in range(MAX_SCHEMA_DEPTH - 1):
        deepest = list[deepest]  # type: ignore[valid-type]
    rendered = repr(Validator(deepest))
    assert TRUNCATED not in rendered
    assert rendered.startswith("list[")

    # One past what compiles is a refusal rather than a truncated rendering.
    with pytest.raises(ValueError, match="too deep"):
        Validator(list[deepest])  # type: ignore[valid-type]


#: Each link shape of a chain of definitions, and the chain length at which its
#: render first gives up.
#:
#: The renderer's bound counts *levels*, and a link is worth as many levels as
#: its body has nodes -- so the length that reaches the bound is a property of
#: the shape, and every shape is listed with its own. The numbers are the edge
#: rather than a comfortable length either side of it: a bound is only tested at
#: the step across it, and either half alone passes for a bound one off.
#:
#: The shapes are chosen to reach the bound through *different* render arms --
#: a union beside a sequence, a meet under a complement, a keyed mapping -- so
#: an arm that stopped counting its own level shows up as a number that moved.
_CHAIN_SHAPES = {
    "a sequence in a union": (
        lambda inner: recursive(
            lambda body, i=inner: list[body] | i  # ty: ignore[invalid-type-form]
        ),
        100,
    ),
    "a meet under a complement": (
        lambda inner: recursive(
            lambda body, i=inner: intersection(
                list[body],  # ty: ignore[invalid-type-form]
                complement(i),
            )
        ),
        67,
    ),
    "a keyed mapping": (
        lambda inner: recursive(lambda body, i=inner: {"a": body, "b": i}),
        101,
    ),
}


def _chain(shape: str, links: int) -> object:
    """Build a chain of `links` definitions of this shape, innermost `int`."""
    link, _ = _CHAIN_SHAPES[shape]
    schema: object = int
    for _ in range(links):
        schema = link(schema)
    return schema


@pytest.mark.parametrize("shape", sorted(_CHAIN_SHAPES))
def test_a_chain_one_link_short_of_the_bound_renders_whole(shape: str) -> None:
    """The longest chain that fits renders whole, and re-parses.

    A single annotation cannot reach the renderer's bound -- the frontend
    refuses past `MAX_SCHEMA_DEPTH`, which is lower -- but a chain composes:
    the render descends into each definition in turn, so a hundred shallow
    links reach a depth no one annotation can be written to. The bound is
    therefore a behaviour rather than dead code, and this is the side of it
    where nothing is lost.
    """
    schema = _chain(shape, _CHAIN_SHAPES[shape][1] - 1)
    rendered = repr(Validator(schema))
    assert TRUNCATED not in rendered
    rebuilt = Validator(eval(rendered, dict(_ROUNDTRIP_NS)))  # noqa: S307
    assert rebuilt == Validator(schema)


@pytest.mark.parametrize("shape", sorted(_CHAIN_SHAPES))
def test_one_link_further_truncates_and_says_so(shape: str) -> None:
    """Past the bound the render gives up, and the mark is not valid Python.

    Every other render re-parses to the schema it came from. A truncated one
    cannot -- what is past the bound is not in the string -- so the mark has to
    be something no reader, and no `eval`, mistakes for a schema. An ellipsis is
    valid Python inside a subscript, so `list[...]` parsed, built and handed
    back a validator that was *not* the one printed: a lossy render with nothing
    to say it was lossy.
    """
    rendered = repr(Validator(_chain(shape, _CHAIN_SHAPES[shape][1])))
    assert TRUNCATED in rendered
    with pytest.raises(SyntaxError):
        eval(rendered, dict(_ROUNDTRIP_NS))  # noqa: S307


def test_past_the_bound_a_longer_chain_renders_the_same() -> None:
    """The bound is what makes the render finite, so past it nothing grows.

    A render that kept descending would reach the native stack; the whole
    reason the mark exists is that the walk stops. Two chains of different
    lengths, both past the bound, print the same string -- which is also what a
    depth that stops counting would break, since nothing would ever stop.
    """
    fits = _CHAIN_SHAPES["a sequence in a union"][1] - 1
    near = repr(Validator(_chain("a sequence in a union", fits + 3)))
    far = repr(Validator(_chain("a sequence in a union", fits + 21)))
    assert TRUNCATED in near
    assert near == far

    # Where it stops, spelled out. The bound counts levels rather than links,
    # and a link is two of them, so the render gives up part-way through one:
    # past the last whole link two more print their `recursive(` header and
    # nothing under it. A bound one level off prints one header fewer, which is
    # the whole difference between reading the bound as "past" and as "at" --
    # and a truncated render that is merely *a* truncated render tells those
    # two apart not at all.
    assert near.count("recursive(") == fits + 2
    assert near.count(TRUNCATED) == 2


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
