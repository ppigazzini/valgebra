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

from valgebra import ValidationError, Validator, union


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
    assert expected.count("the literal") == 64  # sixty-four labels
    assert expected.endswith("...")
