import dataclasses
import json
import math
import sys
import threading
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Regex,
    ValidationError,
    Validator,
    anything,
    complement,
    instance_of,
    intersection,
    recursive,
    union,
)

json_value = recursive(
    lambda j: union(None, bool, int, float, str, [j], {str: j}),
)


def _is_json(value: object) -> bool:
    """Independent reference denotation for `json_value`, by structural recursion."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json(v) for k, v in value.items())
    return False


# A value generator that reaches both members and non-members (bytes and tuples
# are foreign to the JSON schema, and a non-string dict key is foreign too).
_json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.text(max_size=3)
    | st.binary(max_size=2),
    lambda child: (
        st.lists(child, max_size=3)
        | st.tuples(child, child)
        | st.dictionaries(st.text(max_size=2) | st.integers(), child, max_size=3)
    ),
    max_leaves=8,
)


@given(value=_json_values)
def test_recursive_membership_matches_a_reference_denotation(value: object) -> None:
    # The recursive membership walk agrees with an independent recursive predicate
    # on members and non-members alike -- the denotation oracle for recursion.
    assert json_value.is_valid(value) == _is_json(value)


def test_recursive_json_value_accepts_nested_data() -> None:
    assert json_value.is_valid({"a": [1, "x", {"b": None}], "c": [True, 3.5]})
    assert json_value.is_valid([1, 2, 3])
    assert json_value.is_valid("leaf")


def test_recursive_json_value_rejects_a_foreign_leaf() -> None:
    assert not json_value.is_valid({"a": object()})


def test_recursive_tree() -> None:
    tree = recursive(lambda t: {"value": int, "left?": t, "right?": t})
    assert tree.is_valid({"value": 1, "left": {"value": 2}})
    assert tree.is_valid({"value": 1, "left": {"value": 2}, "right": {"value": 3}})
    assert not tree.is_valid({"value": "x"})
    assert not tree.is_valid({"value": 1, "left": {"value": "y"}})


def test_recursion_composes_into_larger_schemas() -> None:
    assert Validator([json_value]).is_valid([1, {"k": [None, 2]}])
    assert not Validator([json_value]).is_valid([object()])


def test_mutual_recursion_through_nested_builders() -> None:
    schema = recursive(lambda x: union(int, [x]))
    assert schema.is_valid(1)
    assert schema.is_valid([1, [2], [[3]]])
    assert not schema.is_valid([1, "x"])


# THEORY: guarded-recursion
def test_non_contractive_body_is_rejected() -> None:
    with pytest.raises(ValueError, match="contractive"):
        recursive(lambda r: union(int, r))


def test_an_inner_definition_may_name_the_one_being_built() -> None:
    # An inner fixpoint whose body names the outer variable is a definition that
    # refers back across the nesting. The marker for the outer one lands in the
    # inner definition rather than in the outer body, so resolving only the body
    # would leave it dangling -- and a dangling marker matches no value, which
    # reads as an ordinary non-member.
    schema = recursive(
        lambda outer: union(
            None, {"child": recursive(lambda inner: union(outer, [inner]))}
        )
    )
    assert schema.is_valid(None)
    assert schema.is_valid({"child": None})
    assert schema.is_valid({"child": [None]})
    assert schema.is_valid({"child": [[None]]})
    assert schema.is_valid({"child": {"child": None}})
    assert not schema.is_valid({"child": 5})


@pytest.mark.parametrize(
    "builder",
    [
        # X = ~X, which no set satisfies.
        pytest.param(
            lambda outer: recursive(lambda _inner: complement(outer)),
            id="complement-across-the-nesting",
        ),
        # X = X | list[Y], where the occurrence of X is not under a constructor.
        pytest.param(
            lambda outer: recursive(lambda inner: union(outer, [inner])),
            id="union-across-the-nesting",
        ),
    ],
)
def test_an_unguarded_occurrence_behind_a_definition_is_rejected(
    builder: Callable[[Validator], object],
) -> None:
    # Contractivity is a property of the system of definitions, not of one body:
    # the occurrence sits behind a `Ref`, which a walk over the body alone reads
    # as a leaf.
    with pytest.raises(ValueError, match="contractive"):
        recursive(builder)


def test_a_placeholder_kept_past_its_builder_is_refused() -> None:
    # The placeholder is an ordinary validator, so nothing stops a caller keeping
    # it; what it stands for stops existing when the builder returns. Using one
    # afterwards builds a schema whose marker resolves to nothing, which would
    # otherwise be a validator that silently admits no value.
    kept: list[Validator] = []

    def keep(placeholder: Validator) -> object:
        kept.append(placeholder)
        return union(None, [placeholder])

    recursive(keep)
    escaped = kept[0]
    for build in (
        lambda: Validator(escaped),
        lambda: union(escaped, int),
        lambda: Validator(list[escaped]),  # ty: ignore[invalid-type-form]
    ):
        with pytest.raises(ValueError, match="unresolved recursive placeholder"):
            build()


def test_a_placeholder_is_not_read_as_a_later_definition_self_reference() -> None:
    # The refusal above works by asking whether the placeholder's marker names a
    # definition being built right now. Inside a *later* `recursive` builder the
    # answer for that builder's own marker is yes, so the two must not share one:
    # if they did, the escaped placeholder would resolve to this definition and
    # build a schema the caller never wrote -- silently, since every check the
    # binding makes would pass.
    kept: list[Validator] = []

    def keep(placeholder: Validator) -> object:
        kept.append(placeholder)
        return union(None, [placeholder])

    recursive(keep)
    escaped = kept[0]

    with pytest.raises(ValueError, match="unresolved recursive placeholder"):
        recursive(lambda _fresh: union(None, [escaped]))


def test_a_placeholder_does_not_cross_a_thread_into_another_definition() -> None:
    # The same claim where a per-thread marker would break it. Two threads each
    # opening their first definition would be handed the same marker, and this
    # thread's escaped placeholder would answer to the other thread's open
    # definition. The markers are drawn from one counter for the whole process,
    # so they cannot collide and the refusal holds wherever the placeholder goes.
    kept: list[Validator] = []

    def keep(placeholder: Validator) -> object:
        kept.append(placeholder)
        return union(None, [placeholder])

    recursive(keep)
    escaped = kept[0]
    outcome: list[object] = []

    def build_elsewhere() -> None:
        try:
            outcome.append(recursive(lambda _fresh: union(None, [escaped])))
        except ValueError as err:
            outcome.append(err)

    worker = threading.Thread(target=build_elsewhere)
    worker.start()
    worker.join()
    assert isinstance(outcome[0], ValueError), (
        "an escaped placeholder built a schema on another thread"
    )
    assert "unresolved recursive placeholder" in str(outcome[0])


# Leaves whose set is the one constant they name, so a relation of one to the
# finished fixpoint is decided by that constant's membership.
_CONSTANTS: tuple[object, ...] = (1, 2, "a", "b", True, None, b"x")
_BODY_KINDS: tuple[object, ...] = (int, str, bool, bytes)


def test_a_question_inside_a_builder_proves_nothing_of_the_placeholder() -> None:
    # The placeholder stands for a definition the builder has not returned yet,
    # so no answer about it is an answer about that set. Read as a set that
    # admits nothing, `Literal[1]` would be refuted below the definition and
    # proved below its complement, and a list of it proved empty beside a list
    # of the definition -- while the finished set below holds 1.
    asked: dict[tuple[str, str], object] = {}

    def body(placeholder: Validator) -> object:
        for constant in _CONSTANTS:
            leaf = Validator(Literal[constant] if constant is not None else None)  # ty: ignore[invalid-type-form]
            name = repr(constant)
            asked[name, "<= X"] = leaf.relation_to(placeholder)
            asked[name, "<= ~X"] = leaf.relation_to(complement(placeholder))
            asked[name, "& X empty"] = intersection(leaf, placeholder).is_empty()
            asked[name, "list & X empty"] = intersection(
                Validator(list[leaf]),  # ty: ignore[invalid-type-form]
                Annotated[list[Any], at.MinLen(1)],
                Validator(list[placeholder]),  # ty: ignore[invalid-type-form]
            ).is_empty()
        asked["X", "1 in X"] = placeholder.is_valid(1)
        asked["X", "1 in ~X"] = complement(placeholder).is_valid(1)
        return union(int, str, [placeholder])

    finished = recursive(body)
    assert finished.is_valid(1)
    assert finished.is_valid([1, ["a"]])
    decided = {
        key: answer
        for key, answer in asked.items()
        if answer not in ("undecided", False)
    }
    assert not decided, f"definite answers about an open placeholder: {decided}"


def test_a_walk_of_an_open_placeholder_stops_rather_than_refusing() -> None:
    # A stop refuses the value from a set and from its complement alike, and the
    # report names it, as the depth bound's stop does.
    reports: list[str] = []

    def body(placeholder: Validator) -> object:
        for schema in (placeholder, complement(placeholder), union(int, placeholder)):
            with pytest.raises(ValidationError) as caught:
                schema.validate("x")
            reports.append(str(caught.value.errors[0]["code"]))
        return union(None, [placeholder])

    recursive(body)
    assert reports == ["unresolved_recursion"] * 3


@given(
    constant=st.sampled_from(_CONSTANTS),
    held=st.lists(st.sampled_from(_CONSTANTS + _BODY_KINDS), min_size=1, max_size=3),
    container=st.sampled_from(("list", "record", "tuple")),
)
def test_a_definite_answer_inside_a_builder_holds_of_the_finished_set(
    constant: object, held: list[object], container: str
) -> None:
    # LAW: an answer the decision gives about the placeholder inside its builder
    # is true of the fixpoint the builder returns. The subject is one constant,
    # so its relation to the finished set is that constant's membership.
    leaf = Validator(Literal[constant] if constant is not None else None)  # ty: ignore[invalid-type-form]
    asked: dict[str, object] = {}

    def body(placeholder: Validator) -> object:
        asked["<= X"] = leaf.relation_to(placeholder)
        asked["<= ~X"] = leaf.relation_to(complement(placeholder))
        asked["& X empty"] = intersection(leaf, placeholder).is_empty()
        branch = {
            "list": Validator(list[placeholder]),  # ty: ignore[invalid-type-form]
            "record": Validator({"k": placeholder}),
            "tuple": Validator(tuple[placeholder]),  # ty: ignore[invalid-type-form]
        }[container]
        members = [
            Validator(Literal[item])  # ty: ignore[invalid-type-form]
            if isinstance(item, (int, str, bytes))
            else Validator(item)
            for item in held
        ]
        return union(branch, *members)

    finished = recursive(body)
    inside = finished.is_valid(constant)
    if asked["<= X"] == "subset" or asked["<= ~X"] == "not_subset":
        assert inside, asked
    refuted = asked["<= X"] == "not_subset" or asked["<= ~X"] == "subset"
    if refuted or asked["& X empty"]:
        assert not inside, asked


def test_self_containing_value_is_rejected_as_a_loop() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValidationError) as info:
        json_value.validate(cyclic)
    # The cycle guard fires on the self-reference; pin the exact code rather than
    # accept a generic union miss, which would hide a regression.
    assert info.value.code == "recursion_loop"


def test_deeply_nested_value_fails_cleanly() -> None:
    chain = recursive(lambda c: union(None, [c]))
    value: object = None
    for _ in range(500):
        value = [value]
    # A value deeper than the recursion bound is rejected, not a crash.
    assert not chain.is_valid(value)


def test_recursive_schema_renders_as_the_call_that_builds_it() -> None:
    # Finite, and rebuildable: the back edge is the lambda's own parameter
    # rather than an ellipsis, which is a dict value the frontend reads as
    # `Literal[Ellipsis]`.
    tree = recursive(lambda t: {"value": int, "left?": t})
    assert repr(tree) == "recursive(lambda X: {'left?': X, 'value': int})"
    assert recursive(lambda x: {"value": int, "left?": x}) == tree


# --- Whole-schema transforms reach the definitions table ----------------------
#
# A recursive validator's root is a single back edge and every record, union and
# refinement it declares lives in the definitions table. A transform that rewrites
# only the root is a no-op on exactly the schemas that carry the most structure,
# and a property asserting that a transform *preserves* membership passes
# vacuously against one. Each case below asserts the change, not its absence.


def test_open_admits_undeclared_keys_inside_a_recursive_definition() -> None:
    node = recursive(lambda n: Validator({"a": int, "next": union(None, n)}))
    extra = {"a": 1, "next": None, "undeclared": 9}
    assert not node.is_valid(extra)
    assert node.open().is_valid(extra)


def test_open_reaches_a_record_nested_under_the_back_edge() -> None:
    node = recursive(lambda n: Validator({"a": int, "next": union(None, n)}))
    nested = {"a": 1, "next": {"a": 2, "next": None, "undeclared": 9}}
    assert not node.is_valid(nested)
    assert node.open().is_valid(nested)


def test_close_removes_the_catch_all_from_a_recursive_definition() -> None:
    node = recursive(lambda n: Validator({"a": int, "next": union(None, n)})).open()
    extra = {"a": 1, "next": None, "undeclared": 9}
    assert node.is_valid(extra)
    assert not node.close().is_valid(extra)


def test_a_recursive_definition_is_built_in_the_normal_form() -> None:
    node = recursive(
        lambda n: Validator({"a": union(int, int, int), "n": union(None, n)})
    )
    assert repr(node).count("int") == 1


# PEP 695 is 3.12+ syntax, so the alias is written in a source string rather than
# in this module: a `type` statement here would be a SyntaxError on 3.11 before
# any skip could apply.
def _defined(source: str, **names: object) -> dict[str, Any]:
    namespace: dict[str, Any] = dict(names)
    exec(source, namespace)  # noqa: S102 - the syntax under test, not user input
    return namespace


def _alias(source: str, name: str) -> object:
    return _defined(source)[name]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_a_recursive_type_alias_is_the_fixpoint_it_writes() -> None:
    """`type X = ... X ...` is the standard spelling of a recursive schema.

    The alias is the binder: it is reached again while its own body is being
    read, and there is no lambda to carry the fixpoint. So it builds the same
    schema the explicit `recursive` call builds, which is what this asserts --
    equivalence rather than a repr, because the two spellings are one set.
    """
    alias = _alias(
        "type Json = None | bool | int | float | str | list[Json] | dict[str, Json]",
        "Json",
    )
    compiled = Validator(alias)

    assert compiled.is_equivalent(json_value)
    assert compiled.is_valid({"a": [1, "x", {"b": None}]})
    assert not compiled.is_valid({"a": {1: 2}})


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_an_alias_that_never_names_itself_builds_what_it_always_did() -> None:
    # The fixpoint is tied only where the knot exists: an ordinary alias is the
    # schema its value builds, with no definition and nothing to resolve.
    assert repr(Validator(_alias("type Plain = int | str", "Plain"))) == "int | str"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_an_alias_naming_itself_outside_a_constructor_is_refused() -> None:
    # `type X = int | X` names a set no value settles, which is the same refusal
    # an unguarded `recursive` gets, for the same reason.
    with pytest.raises(ValueError, match="not contractive"):
        Validator(_alias("type Bad = int | Bad", "Bad"))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_two_aliases_may_name_each_other() -> None:
    # Mutual recursion is two definitions, and each alias is the binder of its
    # own: the pair is a system, not one fixpoint seen twice.
    source = "type Branch = list[Leaf]\ntype Leaf = int | Branch"
    branch = Validator(_alias(source, "Branch"))

    assert branch.is_valid([1, [2]])
    assert not branch.is_valid([1, "x"])


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_two_aliases_spelled_alike_are_one_definition() -> None:
    # Each alias binds a fixpoint of its own, and two that spell one body name
    # one definition, so a union of them is one member, as a union of one
    # schema with itself is; and one alias read twice in a schema is one
    # definition, as it is when the two reads are composed.
    source = "type Left = int | list[Left]\ntype Right = int | list[Right]"
    left, right = _alias(source, "Left"), _alias(source, "Right")
    joined = Validator(left | right)  # ty: ignore[unsupported-operator]
    assert joined == Validator(left)
    assert repr(joined) == "recursive(lambda X: int | list[X])"
    twice = Validator(list[left] | dict[str, left])  # ty: ignore[invalid-type-form]
    assert twice == union(list[left], dict[str, left])  # ty: ignore[invalid-type-form]


_BOXES = """
from dataclasses import dataclass

@dataclass
class Box[T]:
    item: T

type Tree = int | Box[Tree]
"""


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_a_class_reached_again_inside_an_alias_is_tied_by_the_alias() -> None:
    """`Box[Tree]` reaches `Box[Tree]` again inside `Tree`, which it opened.

    Read once more there, the class's fields reach `Tree` while it is open,
    which is the alias's back edge, so the build ends, and it is the set the
    annotation writes: the boxes holding a tree.
    """
    defined = _defined(_BOXES)
    box, tree = defined["Box"], defined["Tree"]
    boxes = Validator(box[tree])
    assert boxes.is_valid(box(1))
    assert boxes.is_valid(box(box(1)))
    assert not boxes.is_valid(box("s"))
    assert not boxes.is_valid(box(box("s")))
    assert not boxes.is_valid(1)
    assert boxes.relation_to(tree) == "subset"
    assert Validator(tree).relation_to(boxes) == "not_subset"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_a_class_naming_itself_through_an_alias_is_checked_in_depth() -> None:
    linked = _defined(
        "from dataclasses import dataclass\n"
        "type Rest = Node | None\n"
        "@dataclass\n"
        "class Node:\n"
        "    value: int\n"
        "    rest: Rest\n"
    )["Node"]
    nodes = Validator(linked)
    assert nodes.is_valid(linked(1, linked(2, None)))
    assert not nodes.is_valid(linked(1, linked("a", None)))
    assert not nodes.is_valid(linked(1, 2))
    # With no base case the alias ties a fixpoint no finite value is in, as the
    # same body written with `recursive` does.
    loop = _defined(
        "from dataclasses import dataclass\n"
        "type Again = Loop\n"
        "@dataclass\n"
        "class Loop:\n"
        "    again: Again\n"
    )["Loop"]
    assert Validator(loop).is_empty()


@dataclasses.dataclass
class _Cell:
    nxt: "_Cell | None" = None


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type aliases")
def test_a_class_reached_again_with_no_alias_opened_since_is_refused() -> None:
    # An alias open around the whole class ties nothing inside it: the class
    # names itself with no alias between, and is refused where it is reached.
    outer = _defined("type Outer = list[_Cell]", _Cell=_Cell)["Outer"]
    with pytest.raises(
        NotImplementedError, match="reached again while its own fields are being built"
    ):
        Validator(outer)


def test_a_fixpoint_is_decided_against_the_kinds_its_body_admits() -> None:
    """A meet with a kind the body never names is empty, and provably so.

    The rules read a fixpoint's *shape* and a set representation holds no cycle,
    so neither answers this pair of questions alone: `bytes` and a JSON value
    share no value, and neither can say it. Unfolding the body once puts the
    kinds it admits in front of the representation, with a bound standing where
    the reference was.
    """
    json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))

    assert intersection(bytes, json).is_empty()
    assert intersection(tuple, json).is_empty()
    assert intersection(set[int], json).is_empty()
    assert Validator(bytes).is_subtype_of(complement(json))
    assert json.is_subtype_of(anything)
    # And the values still decide the same way, which is what the proofs are
    # about: the unfolding is a reasoning step, not a change of schema.
    assert json.is_valid({"a": [1, "x", None]})
    assert not json.is_valid(b"x")
    assert not json.is_valid((1,))


# THEORY: two-fixpoints-one-procedure, a-reference-denotes-its-definition
def test_the_unfolding_is_sound_in_both_directions() -> None:
    """The over- and under-approximation must each stay on its own side.

    A positive occurrence is unfolded to a *superset* and a negative one to a
    subset, which is what keeps `a & ~b` sound over a recursive schema. Getting
    the two the wrong way round would prove a containment that does not hold,
    and the pairs below are the ones that would show it: each is a true
    non-relation over a fixpoint, and a wrong polarity turns one into a proof.
    """
    json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    ints = recursive(lambda t: union(int, [t]))

    # `json` is not below the integer tree, and the tree is below `json`.
    assert ints.is_subtype_of(json)
    assert not json.is_subtype_of(ints)
    # A kind the body *does* admit is not disjoint from it, however deep.
    assert not intersection(str, json).is_empty()
    assert not intersection(list[int], json).is_empty()
    assert not intersection(json, complement(str)).is_empty()
    # And a witness for each: what the decisions say, the walk agrees with.
    assert json.is_valid("x")
    assert not ints.is_valid("x")
    assert json.is_valid([[1]])
    assert ints.is_valid([[1]])


# THEORY: a-reference-denotes-its-definition
def test_a_reference_denotes_the_definition_it_names() -> None:
    """A fixpoint and its own unfolding are one set, in both directions.

    This is the *equirecursive* reading, and it is a choice: an isorecursive
    one would make the two distinct types with a coercion between them, and
    every relation here would then be about which side of that coercion a value
    sat on. A reference denoting its definition is what lets a caller write the
    body out by hand and get the same schema.

    Both directions, because one alone is the weaker claim a widening would
    also satisfy: unfolding to a superset proves the fixpoint below its body
    and says nothing about the way back.
    """
    chain = recursive(lambda t: union(None, {"next": t}))
    unfolded = union(None, {"next": chain})
    assert chain.is_subtype_of(unfolded)
    assert unfolded.is_subtype_of(chain)
    assert chain.is_equivalent(unfolded)

    # And the walk agrees, which is what makes the relation a claim about
    # values rather than about two spellings the rules happen to fold alike.
    for value in (None, {"next": None}, {"next": {"next": None}}):
        assert chain.is_valid(value)
        assert unfolded.is_valid(value)
    for outside in ({"next": 1}, {"other": None}, 1):
        assert not chain.is_valid(outside)
        assert not unfolded.is_valid(outside)


# THEORY: the-depth-bound-reports-itself
def test_every_entry_point_reports_the_bound_it_reaches() -> None:
    """A value deeper than the walk descends is refused, never a crash.

    The two paths reach different bounds and each names its own. The object
    walk unfolds to its depth limit and reports `recursion_limit`; the JSON
    path hands the document to the parser first, and the parser's own nesting
    bound is the lower of the two, so a document that deep is refused as
    unreadable before the walk sees it.

    Every entry point, because the bound is the one place where a missing
    guard is a segmentation fault rather than a wrong answer: a path that
    forgot it would take the interpreter down with it, and the suite would
    report a crash rather than a failure.
    """
    schema = recursive(lambda t: union(int, [t]))
    deep: list[object] = []
    cursor = deep
    for _ in range(400):
        nested: list[object] = []
        cursor.append(nested)
        cursor = nested

    # The object path: an answer from the two that give one, and the bound's
    # own code from the two that raise.
    assert schema.is_valid(deep) is False
    for call in (schema.validate, schema.ensure):
        with pytest.raises(ValidationError) as caught:
            call(deep)
        assert caught.value.errors[0]["code"] == "recursion_limit"

    # The JSON path: the parser's bound, which is the lower one, and which the
    # limits page names beside the walk's.
    document = json.dumps(deep)
    assert schema.is_valid_json(document) is False
    for reads in (schema.validate_json, schema.load):
        with pytest.raises(ValidationError) as caught:
            reads(document)
        assert caught.value.errors[0]["code"] == "json_invalid"


@dataclasses.dataclass
class _Tagged:
    tag: Literal["a", "b"]


#: Leaves whose union of literals the deciding walk answers from a table, and
#: an `int` beside them whose union it walks branch by branch.
_BOUNDARY_LEAVES: dict[str, tuple[object, object]] = {
    "Literal[a, b]": (Literal["a", "b"], "a"),
    "Literal[1, 2]": (Literal[1, 2], 2),
    "tuple[Literal, ...]": (tuple[Literal["a", "b"], ...], ("a",)),
    "set[Literal]": (set[Literal["a", "b"]], {"a"}),
    "a dataclass of a Literal": (_Tagged, _Tagged("a")),
    "int": (int, 1),
}


def _boundary_answers(validator: Validator, value: object) -> set[bool]:
    """Return the answers five entry points give about `value`."""
    answers = {validator.is_valid(value), value in validator}
    for call in (
        validator.validate,
        lambda v: validator.validate(v, fail_fast=True),
        validator.ensure,
    ):
        answers.add(_accepts(call, value))
    return answers


def _accepts(call: Callable[[object], object], value: object) -> bool:
    """Return whether an entry point that raises on a refusal accepts `value`."""
    try:
        call(value)
    except ValidationError:
        return False
    return True


def _stops_at_the_bound(validator: Validator, value: object) -> bool:
    """Return whether the walk refuses `value` at the depth bound."""
    try:
        validator.validate(value, fail_fast=True)
    except ValidationError as error:
        return error.code == "recursion_limit"
    return False


@pytest.mark.parametrize("leaf", list(_BOUNDARY_LEAVES))
def test_every_entry_point_gives_one_answer_at_the_depth_bound(leaf: str) -> None:
    """At the walk's depth bound `is_valid` and `validate` give one answer.

    A union of literals is decided by a table in the deciding walk and branch
    by branch in the explaining one, and a table answering without the level
    a branch opens would make a value whose leaf sits one level past the bound
    a member to `is_valid` and `in` and refused with `recursion_limit` by
    `validate`, `ensure` and a fail-fast `validate`. The levels swept are the
    ones on either side of the first the walk refuses, whatever that bound is.
    """
    schema, member = _BOUNDARY_LEAVES[leaf]
    # The leaf sits in a tuple so a union of literals keeps its own table rather
    # than flattening into the tree's, and three lists a level reach the walk's
    # bound before the reference trail's.
    tree = recursive(lambda t: union([[[t]]], tuple[schema]))  # ty: ignore[invalid-type-form]

    def nested(depth: int) -> object:
        value: object = (member,)
        for _ in range(depth):
            value = [value]
        return value

    for outer in range(6):
        validator = Validator(tree)
        for _ in range(outer):
            validator = Validator([validator])
        # A value of the tree is three lists a level inside the outer ones.
        first = next(
            k
            for k in range(1, 700)
            if _stops_at_the_bound(validator, nested(3 * k + outer))
        )
        for level in range(max(first - 2, 0), first + 2):
            answers = _boundary_answers(validator, nested(3 * level + outer))
            assert len(answers) == 1, (outer, level, answers)


#: `T = int | list[T] | list[T | str]`: two list branches read a list's
#: element through one definition, and the walk keeps the second reading.
_TWO_LISTS = Validator(
    recursive(lambda t: union(int, list[t], list[union(t, str)]))  # ty: ignore[invalid-type-form]
)


def _lists_around(leaf: object, depth: int) -> object:
    """Return `leaf` inside `depth` one-element lists."""
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


# THEORY: metamorphic-testing
@pytest.mark.parametrize(("inner", "outer"), [(100, 60), (120, 10), (90, 40), (40, 40)])
def test_a_shared_value_answers_as_its_copy_past_the_bounds(
    inner: int, outer: int
) -> None:
    """A value holding one sub-object at two depths answers as the copy sharing nothing.

    The first list branch reads the sub-object near the root, and the second
    finds what the walk kept. Met again near the bound, the walk has less of
    the bounds left than the kept walk used, and walks it as the copy is
    walked, so sharing moves no answer of any entry point.
    """
    near = _lists_around(1, inner)
    shared = [near, _lists_around(near, outer), "s"]
    copy = [_lists_around(1, inner), _lists_around(_lists_around(1, inner), outer), "s"]
    answers = _boundary_answers(_TWO_LISTS, shared)
    assert answers == _boundary_answers(_TWO_LISTS, copy)
    assert len(answers) == 1


@pytest.mark.parametrize("depth", [60, 96, 100, 110, 118, 130])
def test_a_document_past_the_bounds_answers_alike_on_each_json_path(depth: int) -> None:
    """A JSON document the second list branch reads again past the bound answers alike.

    `[[...["s", "s"]...], "s"]`: the first list branch refuses each level at
    its `"s"`, and the second reads the inner list again with less of the
    bounds left, which a document reaches with no sharing at all.
    """
    value: object = "s"
    for _ in range(depth):
        value = [value, "s"]
    doc = json.dumps(value)
    try:
        _TWO_LISTS.validate_json(doc)
    except ValidationError:
        explained = False
    else:
        explained = True
    loaded = _TWO_LISTS.is_valid(json.loads(doc))
    assert {_TWO_LISTS.is_valid_json(doc), explained} == {loaded}


# THEORY: a-cycle-is-caught-by-identity
# TRUST: The values are the finite ones.
def test_a_cycle_is_caught_by_identity_rather_than_by_equality() -> None:
    """The guard reads `id`, so two equal values are not one cycle.

    A value that contains itself has no finite unfolding and is refused with
    `recursion_loop`. What the guard must *not* do is refuse a value that
    merely repeats: a list holding two equal sublists is an ordinary finite
    value, and reading equality instead of identity would report it as a cycle
    -- a refusal of a value every reading of the schema admits.

    The two halves are the same guard from either side, and a guard keyed on
    equality passes the first and fails the second.
    """
    schema = recursive(lambda t: union(int, [t]))

    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert schema.is_valid(cyclic) is False
    with pytest.raises(ValidationError) as caught:
        schema.validate(cyclic)
    assert caught.value.errors[0]["code"] == "recursion_loop"

    # Two sublists that are equal and are not the same object, twice over: a
    # repeat at one level, and the same repeat one level down.
    repeated = [[1], [1]]
    assert repeated[0] == repeated[1]
    assert repeated[0] is not repeated[1]
    assert schema.is_valid(repeated)
    assert schema.is_valid([repeated, repeated])


def test_a_body_that_never_names_the_fixpoint_is_that_body() -> None:
    """`recursive(lambda t: A)` with no `t` in `A` is `A`.

    A fixpoint whose body never names it is its own set, so the build returns
    the body rather than a definition no reference reaches beside a reference
    to it: equal to the body, rendered as it, and the same set.
    """
    for body in (int, [int], {"a": str}, union(int, str)):
        unnamed = recursive(lambda _t, body=body: body)
        assert unnamed == Validator(body)
        assert repr(unnamed) == repr(Validator(body))
    named = recursive(lambda t: union(int, [t]))
    assert repr(named).startswith("recursive(")
    # An inner fixpoint that never names the outer one is the outer body, so
    # the result is the inner fixpoint; one that names it makes the outer a
    # fixpoint too, whose values reach it through the inner.
    inner = recursive(lambda _t: recursive(lambda u: union(int, [u])))
    assert inner == recursive(lambda u: union(int, [u]))
    through = recursive(lambda t: recursive(lambda u: union(None, [u], {"next": t})))
    assert repr(through).startswith("recursive(lambda X: recursive(")
    assert through.is_valid({"next": [{"next": None}]})
    assert not through.is_valid({"next": 1})


_NESTED_LISTS = recursive(lambda t: [t])
_NESTED_LISTS_SPLIT = recursive(
    lambda t: union(
        Annotated[list[t], at.MaxLen(3)],  # ty: ignore[invalid-type-form]
        Annotated[list[t], at.MinLen(2)],  # ty: ignore[invalid-type-form]
    )
)


class _CarriesNestedLists(Protocol):
    x: _NESTED_LISTS  # ty: ignore[invalid-type-form]


class _CarriesNestedListsSplit(Protocol):
    x: _NESTED_LISTS_SPLIT  # ty: ignore[invalid-type-form]


class _Carrier:
    """A value of no kind the set reading names, carrying `x`."""

    def __init__(self, x: object) -> None:
        self.x = x


class _ListCarrier(list[int]):
    """A value of the list kind, carrying `x`: a protocol names no kind."""

    def __init__(self, x: object, items: list[int] | None = None) -> None:
        super().__init__(items or [])
        self.x = x


# THEORY: a-cut-reference-proves
def test_a_cut_beneath_an_attribute_record_refutes_no_value() -> None:
    """A protocol holds values of every kind, so a cut beneath one reaches them all.

    The two fields are one set, the second split over two length bounds, and a
    value carrying a nested list as `x` is in both protocols or neither. The
    list standing between the record and the cut says what `x` is and nothing
    of the kind of the value holding it, which may be a list itself.
    """
    plain = Validator(_CarriesNestedLists)
    split = Validator(_CarriesNestedListsSplit)
    for carried in ([], [[]], [[], []], [[], [], [], []], [[[], [], [], []]]):
        for carrier in (_Carrier(carried), _ListCarrier(carried)):
            assert plain.is_valid(carrier), carried
            assert split.is_valid(carrier), carried
    for carried in (1, [1], [[], 1]):
        for carrier in (_Carrier(carried), _ListCarrier(carried)):
            assert not plain.is_valid(carrier), carried
            assert not split.is_valid(carrier), carried
    assert _NESTED_LISTS.relation_to(_NESTED_LISTS_SPLIT) != "not_subset"
    assert plain.relation_to(split) != "not_subset"
    assert split.relation_to(plain) != "not_subset"


_LISTS_NOT_NONE = recursive(
    lambda t: intersection(
        union(Validator([]), Validator(list[t])),  # ty: ignore[invalid-type-form]
        complement(None),
    )
)


# THEORY: a-cut-reference-proves
@pytest.mark.parametrize(
    ("subject", "other", "witness"),
    [
        (_CarriesNestedLists, int, _Carrier([])),
        (_CarriesNestedLists, list[int], _Carrier([])),
        (anything, _CarriesNestedLists, 1),
        (_LISTS_NOT_NONE, str, []),
        (
            Validator(list[complement(_NESTED_LISTS)]),  # ty: ignore[invalid-type-form]
            Validator([]),
            [1],
        ),
        (
            intersection(_CarriesNestedLists, instance_of(_ListCarrier)),
            list[_CarriesNestedLists],
            _ListCarrier([], [1]),
        ),
    ],
    ids=[
        "protocol-int",
        "protocol-list",
        "anything-protocol",
        "meet",
        "complement",
        "list-subclass",
    ],
)
def test_a_value_below_every_cut_refutes_at_any_kind(
    subject: object, other: object, witness: object
) -> None:
    """The difference cut the other way lies inside the real one.

    Each subject holds a value with its reference taken as nothing, at a kind the
    cut reaches, and each supertype read with its reference taken as everything
    leaves that value out. So the value is a value of the real difference, and the
    relation refutes on it where the widened difference names nothing real. A list
    subclass carrying the attribute is a list of the meet whichever items it holds,
    so an item carrying none is what a list of carriers refuses.
    """
    held, outside = Validator(subject), Validator(other)
    assert held.is_valid(witness)
    assert not outside.is_valid(witness)
    assert held.relation_to(outside) == "not_subset"


# THEORY: a-cut-reference-proves
def test_a_value_of_a_kind_no_cut_reaches_refutes_a_fixpoint() -> None:
    """A refinement's outside is named beside a fixpoint of containers.

    The set reading unfolds a reference once and cuts it beneath the list, the
    mapping, the record or the tuple that holds it, which changes values of
    that kind alone: a value of the refinement's kind outside it is outside the
    fixpoint, and the relation says so. Where the subject's values are of the
    container's own kind, the cut reaches them, and an inclusion that holds is
    still proved.
    """
    shapes = {
        "list": lambda leaf: recursive(lambda t: union(leaf, [t])),
        "dict": lambda leaf: recursive(lambda t: union(leaf, {str: t})),
        "record": lambda leaf: recursive(lambda t: union(leaf, {"k": t})),
        "tuple": lambda leaf: recursive(
            lambda t: union(leaf, tuple[t, ...])  # ty: ignore[invalid-type-form]
        ),
    }
    rows = [
        (int, Annotated[int, at.Ge(0)], -1),
        (int, Annotated[int, at.MultipleOf(2)], 1),
        (str, Annotated[str, at.MinLen(1)], ""),
        (str, Annotated[str, Regex("a")], "b"),
        (float, Annotated[float, at.Ge(0)], -0.5),
        (bool, Literal[True], False),
    ]
    for name, fixpoint in shapes.items():
        for kind, leaf, witness in rows:
            supertype = fixpoint(leaf)
            assert Validator(kind).is_valid(witness)
            assert not supertype.is_valid(witness), (name, witness)
            answer = Validator(kind).relation_to(supertype)
            assert answer == "not_subset", (name, kind, leaf)
        assert Validator(list[str]).relation_to(shapes["list"](str)) == "subset"
        assert Validator(str).relation_to(fixpoint(str)) == "subset", name


def test_a_constant_inside_a_record_of_patterns_is_outside_its_complement() -> None:
    """A set inside `c` that holds a value is outside `¬c`, however `c` is proved.

    The record is the one a translation from JSON Schema writes for an object of a
    named property and a property pattern, met with the JSON values, and `not`
    over it is its complement met with them again. The constant lies inside the
    record by a proof the structural rules do not make, and the refutation below
    the complement stands on the constant itself.
    """
    json_values = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    key = intersection(str, Annotated[str, Regex("(?s:.*)(?:^v)(?s:.*)")])
    record = intersection(
        Validator({"foo?": json_values, key: json_values}), json_values
    )
    constant = Validator({"foo": Literal[1], "vroom": Literal[2]})
    assert constant.relation_to(record) == "subset"
    assert constant.relation_to(complement(record)) == "not_subset"
    assert (
        constant.relation_to(intersection(json_values, complement(record)))
        == "not_subset"
    )
    assert record.is_valid({"foo": 1, "vroom": 2})


def test_a_set_is_below_the_excluded_middle_of_a_set_it_meets() -> None:
    """`U <= B | (U & ~B)` holds for every `U` and every set `B`.

    The supertype is the one a translation from JSON Schema writes for `anyOf
    [B, not B]`, `U` the JSON values, and it is decided by splitting `U` on `B`:
    the half inside `B` is below `B`, and the half outside it is the second
    member. With no recursion the split decides it alike, `B` holding a
    refinement met with `U`.
    """
    json_values = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    two = Validator(Literal[2])
    assert (
        json_values.relation_to(union(two, intersection(json_values, complement(two))))
        == "subset"
    )
    flat = Validator(union(None, bool, int, float, str, list, dict))
    big = union(int, intersection(Annotated[float, at.Ge(2)], flat))
    assert flat.relation_to(union(big, intersection(flat, complement(big)))) == "subset"
    # One set short of the middle, and the half outside it is refuted.
    assert (
        flat.relation_to(union(big, intersection(str, complement(big)))) == "not_subset"
    )


def test_a_conditional_is_below_its_excluded_middle_split_twice_over() -> None:
    """`U <= X | (U & ~X)` for the conditional a translation from JSON Schema writes.

    `if: {exclusiveMaximum: 0}, else: {multipleOf: 2}` is `(N & U) | (E & ~N & U)`,
    each keyword guarded by the values that are no number. The supertype negates
    two sets: the split on the first decides nothing, and asked of their parts its
    halves spend the steps the split on the second proves with, so every set's
    halves are asked of the rules and the sets first.
    """
    real = Annotated[float, at.Ge(-math.inf)]
    json_values = recursive(lambda j: union(None, bool, int, str, [j], {str: j}, real))
    no_number = complement(intersection(union(int, real), complement(bool)))

    def guarded(constraint: object) -> object:
        number = union(Annotated[float, constraint], Annotated[int, constraint])
        return union(intersection(number, complement(bool)), no_number)

    negative, even = guarded(at.Lt(0)), guarded(at.MultipleOf(2))
    conditional = union(
        intersection(negative, json_values),
        intersection(even, complement(negative), json_values),
    )
    middle = union(intersection(complement(conditional), json_values), conditional)
    assert json_values.relation_to(middle) == "subset"


def test_a_subject_of_finitely_many_values_is_related_by_each() -> None:
    """A subject of constants is related by the membership of each of its values.

    The spelling is the one a translation from JSON Schema writes for an array
    constant, a one-item list met with its length, nested: `[[[["1"]]]]` is not a
    list of lists of numbers, on the string at the bottom, which no rule reads
    through four exact lengths. A predicate in the supertype leaves the pair
    undecided, since its code may answer two calls differently.
    """

    def one(item: object) -> Validator[Any]:
        return intersection(list[item], Annotated[list[Any], at.Len(1, 1)])  # ty: ignore[invalid-type-form]

    word = one(one(one(one(Literal["1"]))))
    numbers = Validator(list[list[list[list[int | float]]]])
    assert word.relation_to(numbers) == "not_subset"
    assert not word.is_subtype_of(numbers)
    texts = Validator(list[list[list[list[str]]]])
    assert word.relation_to(texts) == "subset"
    assert word.is_subtype_of(texts)
    checked = Validator(
        list[list[list[list[Annotated[int, at.Predicate(lambda _: True)]]]]]
    )
    assert word.relation_to(checked) == "undecided"


# THEORY: a-cut-reference-proves
def test_a_recursive_set_is_refuted_on_a_value_two_unfoldings_down() -> None:
    """The JSON values are outside the values whose lists hold at most two items.

    It is `{"maxItems": 2}` as a translation from JSON Schema writes it, met with
    the JSON values. One unfolding cuts every list to the empty one, which the
    supertype holds; `[1, 1, 1]` lies two down, and the subject unfolded twice
    lies inside it, so the value refutes. The float leaf leaves out NaN.
    """
    real = Annotated[float, at.Ge(-math.inf)]
    json_values = recursive(lambda j: union(None, bool, int, real, str, [j], {str: j}))
    short = union(complement(list), Annotated[list, at.MaxLen(2)])
    assert json_values.relation_to(intersection(short, json_values)) == "not_subset"
    assert json_values.is_valid([1, 1, 1])
    assert not Validator(short).is_valid([1, 1, 1])
