"""What a union says when no branch admits the value.

Two rules, both from `docs/08-error-model.md`. A union reports the **closest**
branch -- the one that descended furthest -- and where no branch descends at all
it reports one summary naming the branches. And that summary names each branch
"as each would name itself alone", which is the sentence the rows here hold.

The summary is the part to distrust, because it stands in for whatever it could
not read. Two branch kinds had no name of their own in it and fell back to the
node's bare kind, and a branch whose failure sat at the union's own depth was
counted as no progress -- so a walk that ran out of levels, or found a value
inside itself, was reported as "this union matched nothing" with the reason
dropped.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

import annotated_types as at
import pytest

from valgebra import (
    ValidationError,
    Validator,
    complement,
    intersection,
    recursive,
    union,
)


def _errors(schema: object, value: object) -> list[dict]:
    with pytest.raises(ValidationError) as caught:
        Validator(schema).validate(value)
    return list(caught.value.errors)


def _cyclic_dict() -> dict:
    cycle: dict = {}
    cycle["s"] = cycle
    return cycle


def _nested(levels: int) -> object:
    value: object = 0
    for _ in range(levels):
        value = [value]
    return value


# PROMISE: Unions report the closest branch
def test_a_branch_names_itself_as_it_would_alone() -> None:
    """The page's own words, for the two branch kinds that had no name."""
    # A complement alone says what it excludes.
    alone = _errors(complement(str), "zz")
    assert alone[0]["expected"] == "not str"
    # And it says the same inside a union.
    inside = _errors(union(int, complement(str)), "zz")
    assert inside[0]["code"] == "union_error"
    assert "not str" in inside[0]["expected"]
    assert "complement" not in inside[0]["expected"]


def test_a_recursive_branch_names_what_it_admits() -> None:
    """A reference contributes the definition's branches, not the word `value`."""
    tree = recursive(lambda t: union(None, {"n": t}))
    inside = _errors(union(int, tree), 1.5)
    assert inside[0]["code"] == "union_error"
    expected = inside[0]["expected"]
    assert "None" in expected
    assert "dict" in expected
    assert "value" not in expected


def test_a_branch_the_walk_could_not_answer_keeps_its_reason() -> None:
    """A walk out of levels is not a value that matched no branch."""
    edge = recursive(lambda t: union(int, [[[[t]]]]))
    errors = _errors(edge, _nested(600))
    assert [item["code"] for item in errors] == ["recursion_limit"]


def test_a_value_inside_itself_keeps_its_reason_under_a_union() -> None:
    """The same rule, for the guard that refuses a cycle."""
    schema = recursive(lambda t: dict[str, union(int, t)])  # ty: ignore[invalid-type-form]
    errors = _errors(schema, _cyclic_dict())
    assert [item["code"] for item in errors] == ["recursion_loop"]
    # And where the branch descends before it meets the cycle, the progress rule
    # already reported it: the two agree on the answer.
    cycle: list = []
    cycle.append(cycle)
    nested = recursive(lambda t: union(int, [t]))
    assert [item["code"] for item in _errors(nested, cycle)] == ["recursion_loop"]


def test_a_predicate_that_raises_inside_a_union_keeps_its_code() -> None:
    """A buggy predicate stays visible, which is why its code is its own."""

    def boom(_value: object) -> bool:
        raise ValueError("no")

    schema = union(int, Annotated[str, at.Predicate(boom)])
    errors = _errors(schema, "x")
    assert [item["code"] for item in errors] == ["predicate_error"]


def test_a_union_with_no_progress_still_reports_one_summary() -> None:
    """The control: a flat mismatch on every branch is one `union_error`."""
    errors = _errors(union(int, str), 1.5)
    assert [item["code"] for item in errors] == ["union_error"]
    assert errors[0]["expected"] == "one of: int, str"


def test_a_fail_fast_report_walks_no_branch_past_its_first_failure() -> None:
    """Refusing a union costs a fail-fast walk of each branch, not the value's size.

    A predicate on the elements after a branch's first failure never runs under
    `fail_fast`, on either path; a full report runs it on each, since it reports
    each.
    """
    calls: list[object] = []

    def counts(value: object) -> bool:
        calls.append(value)
        return True

    schema = Validator(union(int, [Annotated[int, at.Predicate(counts)]]))
    value = ["x", *range(1_000)]
    with pytest.raises(ValidationError) as caught:
        schema.validate(value, fail_fast=True)
    assert [item["path"] for item in caught.value.errors] == [(0,)]
    with pytest.raises(ValidationError):
        schema.load(json.dumps(value), fail_fast=True)
    assert calls == []
    with pytest.raises(ValidationError):
        schema.validate(value)
    assert len(calls) == 1_000


def test_a_meet_branch_is_named_by_what_its_members_admit() -> None:
    """A branch names the set it expects; the word `intersection` names none."""
    errors = _errors(union(str, intersection(int, complement(bool))), 1.5)
    assert errors[0]["expected"] == "one of: str, int and not bool"


def test_the_closest_branch_is_reported_where_one_descends() -> None:
    """The control the summary stands in for."""
    errors = _errors(union(int, {"a": int}), {"a": "x"})
    assert errors[0]["path"] == ("a",)
    assert errors[0]["code"] == "int_type"


def test_the_record_branch_that_admitted_the_most_is_reported() -> None:
    """Of the record branches, the one whose deciding walk got furthest.

    Each refused record branch was explained and the one whose first failure
    lay deepest was kept, the earliest on a tie -- so a value tagged for the
    second branch, refused at its own field, was reported against the first
    branch's tag. One record branch is explained, the one whose deciding walk
    admitted the most fields, and the tag is read first.
    """
    branches = union(
        {"t": Literal["x"], "z": int},
        {"t": Literal["y"], "z": str, "u": int},
    )
    errors = _errors(branches, {"t": "y", "z": 1.5, "u": 1})
    assert [(e["code"], e["path"]) for e in errors] == [("string_type", ("z",))]


def test_a_tag_read_first_counts_toward_how_far_its_branch_got() -> None:
    """A field read before the declared order counts for its branch.

    `type` is read before `left`, so the `y` branch admits its tag before it
    finds `left` missing, and the `x` branch refuses at the tag: one field
    against none, although `left` comes first in either branch.
    """
    branches = union(
        {"left": list[int], "type": Literal["x"]},
        {"left": list[int], "type": Literal["y"]},
    )
    errors = _errors(branches, {"type": "y"})
    assert [(e["code"], e["path"]) for e in errors] == [("missing_key", ("left",))]


def test_a_record_branch_read_by_its_entries_is_ranked_by_what_it_admitted() -> None:
    """A record whose clause reads values is decided by scanning its entries.

    The scan admitted no field by name, so it counted as admitting nothing and
    lost to any record that admitted one: the open record below admits `a` and
    refuses only `l`, and the closed one admits `a` and lacks `b`. Counting the
    entries a scan admits ranks the two alike, and the union's first is kept.
    """
    branches = union({"a": int, "b": str}, {"a": int, str: int})
    first = _errors(branches, {"a": 1, "l": "b"})[0]
    assert (first["code"], first["path"]) == ("int_type", ("l",))


def test_the_branch_probe_has_a_width_and_the_answer_does_not() -> None:
    """A branch past the probe's width decides membership and is not described.

    The cap is what keeps building an error for a pathologically wide union
    affordable. It is a property of the *report*: a value matching a branch past
    it is still a member, and one that matches none is still refused.
    """

    def wide(count: int) -> object:
        branches = [Literal[i] for i in range(count)]  # ty: ignore[invalid-type-form]
        return union(*branches, {"a": int})

    # Inside the width, the record branch is probed and named.
    near = _errors(wide(10), {"a": "x"})
    assert near[0]["path"] == ("a",)
    # Past it, the union reports its summary rather than pinpointing a branch.
    far = _errors(wide(200), {"a": "x"})
    assert far[0]["code"] == "union_error"
    assert far[0]["path"] == ()
    # And membership is unaffected in both directions.
    assert Validator(wide(200)).is_valid({"a": 1}) is True
    assert Validator(wide(200)).is_valid(199) is True
    assert Validator(wide(200)).is_valid("nope") is False


def test_the_summary_names_a_bounded_number_of_branches() -> None:
    """A wide union's summary is bounded and says it was cut."""
    branches = [Literal[i] for i in range(200)]  # ty: ignore[invalid-type-form]
    errors = _errors(union(*branches), "nope")
    assert errors[0]["code"] == "union_error"
    assert errors[0]["expected"].endswith(", ...")
