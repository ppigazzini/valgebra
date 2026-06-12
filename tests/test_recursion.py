import pytest

from valgebra import ValidationError, lazy, union, validator

json_value = lazy(
    lambda j: union(None, bool, int, float, str, [j], {str: j}),
)


def test_recursive_json_value_accepts_nested_data() -> None:
    assert json_value.is_valid({"a": [1, "x", {"b": None}], "c": [True, 3.5]})
    assert json_value.is_valid([1, 2, 3])
    assert json_value.is_valid("leaf")


def test_recursive_json_value_rejects_a_foreign_leaf() -> None:
    assert not json_value.is_valid({"a": object()})


def test_recursive_tree() -> None:
    tree = lazy(lambda t: {"value": int, "left?": t, "right?": t})
    assert tree.is_valid({"value": 1, "left": {"value": 2}})
    assert tree.is_valid({"value": 1, "left": {"value": 2}, "right": {"value": 3}})
    assert not tree.is_valid({"value": "x"})
    assert not tree.is_valid({"value": 1, "left": {"value": "y"}})


def test_recursion_composes_into_larger_schemas() -> None:
    assert validator([json_value]).is_valid([1, {"k": [None, 2]}])
    assert not validator([json_value]).is_valid([object()])


def test_mutual_recursion_through_nested_builders() -> None:
    schema = lazy(lambda x: union(int, [x]))
    assert schema.is_valid(1)
    assert schema.is_valid([1, [2], [[3]]])
    assert not schema.is_valid([1, "x"])


def test_non_contractive_body_is_rejected() -> None:
    with pytest.raises(ValueError, match="contractive"):
        lazy(lambda r: union(int, r))


def test_self_containing_value_is_rejected_as_a_loop() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValidationError) as info:
        json_value.validate(cyclic)
    assert info.value.code in {"recursion_loop", "union_error"}


def test_deeply_nested_value_fails_cleanly() -> None:
    chain = lazy(lambda c: union(None, [c]))
    value: object = None
    for _ in range(500):
        value = [value]
    # A value deeper than the recursion bound is rejected, not a crash.
    assert not chain.is_valid(value)


def test_recursive_schema_renders_finitely() -> None:
    tree = lazy(lambda t: {"value": int, "left?": t})
    assert repr(tree) == "{'value': int, 'left?': ...}"
