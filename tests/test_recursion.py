import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import ValidationError, Validator, recursive, union

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


def test_self_containing_value_is_rejected_as_a_loop() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValidationError) as info:
        json_value.validate(cyclic)
    assert info.value.code in {"recursion_loop", "union_error"}


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
