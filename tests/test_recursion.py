from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import ValidationError, Validator, complement, recursive, union

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
        lambda: Validator(list[escaped]),
    ):
        with pytest.raises(ValueError, match="unresolved recursive placeholder"):
            build()


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


def test_recursive_schema_renders_finitely() -> None:
    tree = recursive(lambda t: {"value": int, "left?": t})
    assert repr(tree) == "{'value': int, 'left?': ...}"


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


def test_simplify_reduces_a_recursive_definition() -> None:
    node = recursive(
        lambda n: Validator({"a": union(int, int, int), "n": union(None, n)})
    )
    assert repr(node.simplify()).count("int") == 1
