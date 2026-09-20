"""A union has to name what its branches accept.

A value that matches no branch of a union is reported at the union's own
location, and `expected` is the only part of that report naming the alternatives.
The commonest shape a field has is a set of permitted values, so `expected` is
where a caller looks for them -- and if it names the *kinds* of the branches
instead, every caller writes the list out again by hand.

`docs/08-error-model.md` states the rule these tests hold: `expected` names the
set.
"""

from __future__ import annotations

import enum
from typing import Literal

import pytest

from valgebra import ValidationError, Validator, recursive, union


class _Backend(enum.Enum):
    TORCH = "torch"
    JAX = "jax"


def _expected(spec: object, value: object) -> str:
    with pytest.raises(ValidationError) as info:
        Validator(spec).validate(value)
    return str(info.value.errors[0]["expected"])


@pytest.mark.parametrize(
    ("spec", "value", "wanted"),
    [
        # A literal union is the enum-shaped field; the members are the point.
        (Literal["torch", "jax"], "tensorflow", ["'torch'", "'jax'"]),
        # A single-constant Literal is a union of one, and must not say "union".
        (union(Literal["only"], int), 1.5, ["'only'", "int"]),
        # An Enum branch names the class, as it does when it fails alone.
        (union(_Backend, Literal["cpu"]), "arcfase", ["_Backend", "'cpu'"]),
        # Mixed scalars already worked; kept so the change is shown not to regress.
        (int | str, 1.5, ["int", "str"]),
    ],
)
# PROMISE: What a union's `expected` says
def test_a_union_names_what_its_branches_accept(
    spec: object, value: object, wanted: list[str]
) -> None:
    expected = _expected(spec, value)
    for token in wanted:
        assert token in expected, expected


def test_a_branch_is_named_as_it_would_name_itself_alone() -> None:
    """The label a branch carries in a union is the one it carries by itself."""
    alone = _expected(Literal["torch"], "tensorflow")
    inside = _expected(union(Literal["torch"], int), "tensorflow")
    assert alone in inside, (alone, inside)


def test_a_wide_union_does_not_produce_an_unbounded_message() -> None:
    """The report stays bounded however wide the union is."""
    wide = union(*[Validator(f"code_{index:04d}") for index in range(500)])
    expected = _expected(wide, "absent")
    assert len(expected) < 2000, len(expected)
    assert expected.endswith("...")


def test_the_label_bound_is_not_the_branch_bound() -> None:
    """A branch that is itself a union contributes many labels, not one.

    A nested union flattens into the label list, and `Literal[...]` builds one
    of its constants -- so a two-branch union can carry a hundred labels. The
    two counts are bounded separately because they are separate quantities.
    """
    hundred = union(*[Validator(f"c{index}") for index in range(100)])
    spec = union(hundred, int)  # two branches
    expected = _expected(spec, 1.5)
    # Sixty-four labels, of which `int` is one: the members of a join are held in
    # the normal form's order, and a kind sorts before a literal.
    assert expected.count("the literal") == 63
    assert expected.startswith("one of: int, ")
    assert expected.endswith("...")


# `CLOSEST_BRANCH_PROBE_LIMIT` in `crates/valgebra-py/src/check/walk.rs`. Not
# published, so it is written here and this test is what holds the two together:
# raise the constant and the first case stops falling back.
CLOSEST_BRANCH_PROBE_LIMIT = 64


@pytest.mark.parametrize(
    ("before", "explained"),
    [
        (CLOSEST_BRANCH_PROBE_LIMIT - 1, True),
        (CLOSEST_BRANCH_PROBE_LIMIT, False),
    ],
)
def test_the_closest_branch_probe_stops_at_its_cap(
    before: int, *, explained: bool
) -> None:
    """The probe re-walks the first branches only, and says so by falling back.

    A union that matches nothing is explained through its *closest* branch --
    the one that descended furthest -- and finding it costs a second walk per
    branch. That pass is capped, so a wide union's error stays linear in the cap
    rather than in the branch count, and a branch past the cap is not the one
    reported however deep it would have got.

    The branches are named so the order is the one the assertion needs: a union
    is built in a normal form, so its members are sorted rather than left in the
    order they were written.
    """
    deep = {"zz": {"b": int}}
    spec = union(*({f"a{at:03d}": int} for at in range(before)), deep)
    with pytest.raises(ValidationError) as info:
        Validator(spec).validate({"zz": {"b": "not an int"}})
    paths = [error["path"] for error in info.value.errors]
    assert (("zz", "b") in paths) is explained, paths


# BOUND: MAX_LABEL_UNFOLDS
def test_a_union_label_follows_four_references_and_then_names_the_kind() -> None:
    """A branch label follows a chain of definitions to a depth, not forever.

    A reference contributes the label of the definition it names, and a
    definition may name another. The chain is followed four references deep,
    which is far enough for the definitions a caller writes, and past it the
    node's own kind is the honest answer: a label is prose, and a label that
    followed a cycle would be a walk with no end.
    """
    chain = recursive(lambda t: union(int, list[t]))  # ty: ignore[invalid-type-form]

    def wrapped(depth: int) -> Validator:
        schema = chain
        for _ in range(depth):
            schema = recursive(lambda t, inner=schema: union(inner, {"n": t}))
        return Validator(union(schema, bytes))

    def expected(schema: Validator) -> str:
        with pytest.raises(ValidationError) as caught:
            schema.validate("x")
        return caught.value.expected

    # Three wrappers and the chain are four references: every one is followed,
    # and the chain's own branches are named.
    assert expected(wrapped(3)) == "one of: bytes, dict, dict, dict, int, list"
    # A fourth wrapper puts the chain a fifth reference away, which is not
    # followed: its label is the kind of a reference, `value`.
    assert expected(wrapped(4)) == "one of: bytes, dict, dict, dict, dict, value"
    # And a longer chain reads the same, since nothing past the bound is read.
    assert expected(wrapped(9)) == expected(wrapped(4))
