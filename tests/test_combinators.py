import itertools
import json
import sys
from typing import Annotated, Literal, NoReturn

import annotated_types as at
import pytest

from valgebra import (
    MAX_SCHEMA_NODES,
    Regex,
    ValidationError,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)


def test_union_admits_any_branch() -> None:
    schema = union(int, str)
    assert schema.is_valid(1)
    assert schema.is_valid("x")
    assert not schema.is_valid(1.0)


def test_intersect_requires_every_member() -> None:
    schema = intersection(int, complement(bool))
    assert schema.is_valid(5)
    assert not schema.is_valid(True)  # an int, but also a bool
    assert not schema.is_valid("x")


def test_complement_inverts_membership() -> None:
    schema = complement(int)
    assert schema.is_valid("x")
    assert not schema.is_valid(5)


def test_lattice_bounds() -> None:
    assert anything.is_valid(object())
    assert anything.is_valid(None)
    assert not nothing.is_valid(5)
    # the complement of bottom is the top
    assert complement(nothing).is_valid(5)
    assert not complement(anything).is_valid(5)


def test_typing_native_bound_spellings() -> None:
    # `object` is the top spelling; `NoReturn` (3.8+) is a bottom spelling.
    assert Validator(object).is_equivalent(anything)
    assert Validator(NoReturn).is_equivalent(nothing)
    assert Validator(NoReturn).is_empty()
    assert not Validator(NoReturn).is_valid(5)
    assert not Validator(NoReturn).is_valid(None)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="typing.Never was added in 3.11")
def test_never_is_the_bottom() -> None:
    from typing import Never  # noqa: PLC0415 -- 3.11+ only, so import is gated

    assert Validator(Never).is_equivalent(nothing)
    assert Validator(Never).is_empty()
    assert not Validator(Never).is_valid(5)


def test_combinators_compose_over_compiled_validators() -> None:
    inner = Validator(list[int])
    schema = union(inner, str)
    assert schema.is_valid([1, 2, 3])
    assert schema.is_valid("x")
    assert not schema.is_valid(1.0)


def test_composition_preserves_pooled_literals() -> None:
    schema = union(Literal["a"], int)
    assert schema.is_valid("a")
    assert schema.is_valid(7)
    assert not schema.is_valid("b")


def test_complement_failure_reports_unexpected_match() -> None:
    with pytest.raises(ValidationError) as info:
        complement(int).validate(5)
    assert info.value.code == "unexpected_match"


#: The codes that say the walk stopped rather than that the value is outside
#: a set (`docs/08-error-model.md`).
STOPS = (
    "recursion_limit",
    "recursion_loop",
    "mutated_during_validation",
    "predicate_error",
)


def _raises(_: object) -> NoReturn:
    raise ValueError("a predicate with a bug")


def _nested(depth: int) -> object:
    """Return an integer `depth` lists deep."""
    value: object = 0
    for _ in range(depth):
        value = [value]
    return value


def _stops(schema: Validator, value: object) -> bool:
    """Whether the schema's walk stops on `value` at a bound."""
    try:
        schema.validate(value, fail_fast=True)
    except ValidationError as error:
        return error.code == "recursion_limit"
    return False


def _stopping(code: str) -> tuple[Validator, object]:
    """Return a schema and a fresh value its walk stops on with `code`.

    The depth is the walk's own bound: three lists a level reach it before the
    reference trail's, which one list a level reaches first.
    """
    lists = Validator(recursive(lambda t: union(int, [t])))
    if code == "recursion_limit":
        return Validator(recursive(lambda t: union(int, [[[t]]]))), _nested(1000)
    if code == "recursion_loop":
        loop: list[object] = []
        loop.append(loop)
        return lists, loop
    if code == "mutated_during_validation":
        moved = {"a": 1, "b": 2}

        def grow(_: object) -> bool:
            moved.setdefault("c", 3)
            return True

        grows = Annotated[int, at.Predicate(grow)]
        return Validator({"a": grows, "b": int, "c?": int}), moved
    return Validator(Annotated[int, at.Predicate(_raises)]), 1


@pytest.mark.parametrize("code", STOPS)
def test_a_complement_refuses_a_value_its_inner_walk_stopped_on(code: str) -> None:
    """A walk that stopped said nothing about the set, so its negation says nothing.

    The inner walk answers `False` for a value it stopped on, as it does for
    one outside its set, and the complement read the two alike: it admitted a
    value past the depth bound, a value inside itself, a value that moved and
    a value whose predicate raised. Each is refused with the stop's code, at
    every entry point, where the inner schema's own walk reports the same code.
    """
    inner, value = _stopping(code)
    with pytest.raises(ValidationError) as caught:
        inner.validate(value, fail_fast=True)
    assert caught.value.code == code

    inner, value = _stopping(code)
    assert complement(inner).is_valid(value) is False
    inner, value = _stopping(code)
    assert (value in complement(inner)) is False
    for fail_fast in (True, False):
        inner, value = _stopping(code)
        with pytest.raises(ValidationError) as caught:
            complement(inner).validate(value, fail_fast=fail_fast)
        assert caught.value.code == code
    inner, value = _stopping(code)
    with pytest.raises(ValidationError) as caught:
        complement(inner).ensure(value)
    assert caught.value.code == code


def test_a_complement_refuses_a_value_that_moves_under_its_inner_scan() -> None:
    """The same for a container read by scanning its entries.

    A record whose clause reads undeclared keys with their values is read by
    walking every entry, and so is a list: a predicate on an entry that grows
    the container moves it under that scan rather than between two lookups.
    """
    moved: dict[str, int] = {}

    def grow(_: object) -> bool:
        moved.setdefault("c", 3)
        return True

    record = Validator({"a": int, str: Annotated[int, at.Predicate(grow)]})
    moved = {"a": 1, "b": 2}
    assert complement(record).is_valid(moved) is False
    moved = {"a": 1, "b": 2}
    with pytest.raises(ValidationError) as caught:
        complement(record).validate(moved)
    assert caught.value.code == "mutated_during_validation"

    grown: list[int] = []

    def append(_: object) -> bool:
        grown.append(3)
        return True

    sequence = Validator(list[Annotated[int, at.Predicate(append)]])
    grown = [1, 2]
    assert complement(sequence).is_valid(grown) is False
    grown = [1, 2]
    with pytest.raises(ValidationError) as caught:
        complement(sequence).validate(grown)
    assert caught.value.code == "mutated_during_validation"


@pytest.mark.parametrize("code", ["recursion_limit", "predicate_error"])
def test_a_complement_refuses_a_document_its_inner_walk_stopped_on(code: str) -> None:
    """The same on the JSON path, for the two stops a document can carry.

    A parsed document holds no cycle and is not shared with a predicate that
    could move it, so the depth bound and a raising predicate are the two.
    The depth is the first the walk refuses, which the parser's own nesting
    bound -- the lower of the two over a document -- lets through.
    """
    inner, value = _stopping(code)
    if code == "recursion_limit":
        # One list a level: the trail's bound, which a document reaches inside
        # the parser's nesting bound, where three a level do not.
        inner = Validator(recursive(lambda t: union(int, [t])))
        depth = next(d for d in range(1, 1000) if _stops(inner, _nested(d)))
        value = _nested(depth)
    document = json.dumps(value)
    assert complement(inner).is_valid_json(document) is False
    for reads in (complement(inner).validate_json, complement(inner).load):
        with pytest.raises(ValidationError) as caught:
            reads(document)
        assert caught.value.code == code


def test_a_stopped_branch_does_not_block_a_branch_that_decides() -> None:
    """A union admits a value one branch decides, whatever another branch did.

    The complement's refusal stands on the inner walk having decided nothing;
    a union beside it that does decide the value keeps its answer, and a meet
    beside it keeps its refusal.
    """
    inner, value = _stopping("recursion_limit")
    assert union(complement(inner), list).is_valid(value)
    assert not intersection(list, complement(inner)).is_valid(value)
    # A stop beside a complement that does decide is kept for a complement
    # above both: the union stopped in its first branch and refused in the
    # second, which decides nothing about the value. The list branch is the
    # union's first, as it holds them.
    with pytest.raises(ValidationError) as caught:
        complement(union([inner], complement(list))).validate(value)
    assert caught.value.code == "recursion_limit"


def test_a_meet_proved_empty_admits_no_value_past_the_bound() -> None:
    """`[T] & ~T` is empty, and the walk admits nothing into it.

    `T` is the finite nested lists and `[T]` its unfolding, which the relation
    proves the same set, so the meet is empty. A list one level past the depth
    `T` is walked to is one `[T]` reads and `T` stops on, and the complement
    read that stop as "not in `T`": every entry point admitted it.
    """
    lists = recursive(lambda t: [t])
    assert Validator(lists).is_equivalent([lists])
    meet = intersection([lists], complement(lists))
    walked = Validator(lists)
    depth = next(d for d in range(1, 1000) if _stops(walked, _nested_lists(d)))
    value = _nested_lists(depth)
    assert meet.is_valid(value) is False
    assert (value in meet) is False
    for call in (meet.validate, meet.ensure):
        with pytest.raises(ValidationError) as caught:
            call(value)
        assert caught.value.code == "recursion_limit"
    document = json.dumps(value)
    assert meet.is_valid_json(document) is False
    with pytest.raises(ValidationError) as caught:
        meet.load(document)
    assert caught.value.code == "recursion_limit"


def _nested_lists(depth: int) -> object:
    """Return an empty list `depth` lists deep."""
    value: object = []
    for _ in range(depth):
        value = [value]
    return value


def test_intersect_with_an_annotation() -> None:
    schema = intersection(int, complement(Literal[0]))
    assert schema.is_valid(1)
    assert not schema.is_valid(0)


def test_a_predicate_does_not_fold_against_its_own_complement() -> None:
    """The complement laws are laws about sets, and a predicate is not one.

    `A & ~A = nothing` holds because a value is in `A` or it is not, once. A
    predicate is user code and the two occurrences of `A` are two calls, so one
    that does not answer from the value alone answers them differently. The
    witness below is admitted by the meet, and the dual rejects a value the top
    would have to admit -- so folding either would claim emptiness of a schema
    that admits values.

    Pinned rather than left to the reader of the law, because "`A & ~A` is
    empty" is exactly what somebody reading `docs/04-algebra.md` would add.
    `docs/dev/01-schema-ir.md` carries the refusal.
    """
    flip = itertools.count()

    def alternating(_: object) -> bool:
        return next(flip) % 2 == 0

    predicate = Validator(Annotated[int, at.Predicate(alternating)])
    assert predicate == predicate  # noqa: PLR0124 - the identity is the subject

    meet = intersection(predicate, complement(predicate))
    assert not meet.is_empty(), "a predicate is not a set, so the law does not apply"
    # The witness: some call lands inside a meet the fold would call empty.
    assert any(meet.is_valid(1) for _ in range(8))

    join = union(predicate, complement(predicate))
    assert not join.is_equivalent(anything)
    # And the dual witness: some call falls outside a join the fold would call
    # the top.
    assert not all(join.is_valid(1) for _ in range(8))


def test_a_pattern_does_fold_against_its_own_complement() -> None:
    """The contrast that makes the refusal about predicates, not about markers.

    A pattern is a function of the string, so the two occurrences agree and the
    law holds.
    """
    pattern = Validator(Annotated[str, Regex("a+")])
    assert intersection(pattern, complement(pattern)).is_empty()
    assert union(pattern, complement(pattern)).is_equivalent(anything)


def test_every_outcome_the_constructors_document_is_driven() -> None:
    """The four names a caller reaches most, each raise they promise driven.

    `union`, `intersection`, `complement` and `recursive` are how a schema the
    annotation syntax cannot spell is written, so their refusals are the ones a
    caller meets first -- and the outcomes ledger had no cell for any of them
    until the blocks naming them existed. Each row below is one of those cells.

    The forms with no set are the schema-language page's own: a set literal and
    a tuple literal, which read like schemas and denote the collection object
    rather than a set of values.
    """
    # Written out rather than looped over the three: a cell is a `(name,
    # outcome)` pair, and a loop drives the name the loop variable holds, which
    # the ledger reading the syntax tree cannot see.
    with pytest.raises(NotImplementedError, match="is not a schema"):
        union({1, 2})
    with pytest.raises(NotImplementedError, match="is not a schema"):
        union((1, 2))
    with pytest.raises(NotImplementedError, match="is not a schema"):
        intersection({1, 2})
    with pytest.raises(NotImplementedError, match="is not a schema"):
        intersection((1, 2))
    with pytest.raises(NotImplementedError, match="is not a schema"):
        complement({1, 2})
    with pytest.raises(NotImplementedError, match="is not a schema"):
        complement((1, 2))

    # A schema past the node ceiling, built the way a loop builds one. Each
    # record is two nodes, so a union of two-fifths of the ceiling is within it
    # and a join of two such unions over disjoint names is not.
    span = MAX_SCHEMA_NODES * 2 // 5
    left = union(*[Validator({f"a{i}": int}) for i in range(span)])
    right = union(*[Validator({f"b{i}": int}) for i in range(span)])
    with pytest.raises(ValueError, match="too large"):
        union(left, right)
    with pytest.raises(ValueError, match="too large"):
        intersection(left, right)
    with pytest.raises(ValueError, match="too large"):
        complement(union(left, right))

    # `recursive` takes a callable, reads its body as a schema, and refuses a
    # body whose reference is not under a structural constructor.
    with pytest.raises(TypeError):
        recursive(3)  # ty: ignore[invalid-argument-type]
    with pytest.raises(NotImplementedError, match="is not a schema"):
        recursive(lambda t: {t, 1})
    with pytest.raises(ValueError, match="contractive"):
        recursive(lambda t: union(t, int))


def test_emptiness_has_a_third_answer_and_where_to_ask_for_it() -> None:
    """`is_empty` answers in two, and the question it reduces to answers in three.

    A relation reports which of proof, refutation and decline it reached;
    `is_empty` gives a `bool`, so "proved inhabited" and "not proved empty"
    arrive as one `False`. The two are different claims about a schema -- one
    names a value, the other names a limit -- and a caller choosing whether to
    trust a schema wants to tell them apart.

    Emptiness is `s <= nothing`, so the third answer is a relation away, and
    asking for it is the documented route rather than a trick: the reduction is
    what the decidability page states and what the procedure runs.
    """
    predicate = Validator(Annotated[int, at.Predicate(lambda value: value > 0)])
    # Nothing proves this schema empty, and nothing proves it inhabited either.
    assert predicate.is_empty() is False
    assert predicate.relation_to(nothing) == "undecided"

    # The other two answers, which the same `False` would have flattened.
    assert Validator(int).is_empty() is False
    assert Validator(int).relation_to(nothing) == "not_subset"
    crossing = Validator(Annotated[int, at.Gt(0), at.Lt(1)])
    assert crossing.is_empty() is True
    assert crossing.relation_to(nothing) == "subset"
