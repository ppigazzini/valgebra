"""Adversarial inputs meet their resource bounds instead of exhausting one.

A validator processes untrusted values, so every recursive walk and every
error-reporting probe is bounded by a gated limit rather than the input. These
tests drive each pathological shape and assert the bound engages: the operation
terminates with a graceful verdict or a specific error code, never a native
stack overflow, a Python ``RecursionError``, or an unbounded hang. The limits
themselves are:

- schema build recursion (deeply nested annotation),
- value-walk recursion (deeply nested value, on both the object and JSON paths),
- the object-identity loop guard (a value that contains itself),
- the closest-branch probe cap (error reporting over a wide union).

Worst-case *timing* on these shapes is recorded by ``benches/bench_adversarial``;
here the guarantee is that the guard fires, which is what keeps a hostile input
from turning into denial of service. Union width is part of the developer-written
schema, not untrusted input, so the bounds here are about the value, not the
schema's declared size.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from typing import TYPE_CHECKING, Annotated, Literal

import pytest
from annotated_types import MultipleOf

if TYPE_CHECKING:
    from collections.abc import Callable

from valgebra import (
    Regex,
    ValidationError,
    Validator,
    complement,
    intersection,
    recursive,
    union,
)
from valgebra._valgebra import (
    MAX_DEFINITIONS,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_NODES,
)

# `simplify` is deprecated and these exercise it deliberately: the folds it
# still performs are its own, and they are checked until it goes.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Validator.simplify is deprecated:DeprecationWarning"
)


def _run_construction_loop(body: str) -> subprocess.CompletedProcess[str]:
    """Run a schema-construction loop in a fresh interpreter.

    The loop must raise ``ValueError`` once a construction bound trips. Running
    it in a subprocess turns a native stack overflow or memory blow-up into a
    non-zero exit code the caller can assert against, instead of taking the whole
    test session down with it.
    """
    program = textwrap.dedent(
        """
        from valgebra import Validator, union, intersection, complement, recursive
        _BOUND_MESSAGES = ("too deep", "too many", "too large")
        try:
        {body}
        except ValueError as exc:
            assert any(m in str(exc) for m in _BOUND_MESSAGES)
            print("RAISED")
        else:
            print("NO ERROR")
        """
    ).format(body=textwrap.indent(textwrap.dedent(body), "    "))
    return subprocess.run(  # noqa: S603 -- fixed interpreter, in-repo program
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _nested_annotation(depth: int) -> object:
    schema: object = int
    for _ in range(depth):
        schema = list[schema]  # type: ignore[valid-type]
    return schema


def _nested_value(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = [value]
    return value


def test_build_depth_guard_rejects_an_overdeep_schema() -> None:
    # A reasonably nested annotation compiles; one past the build-depth guard is
    # rejected at compile time with a clean exception, not a stack overflow.
    assert Validator(_nested_annotation(50)).is_valid(_nested_value(50))
    with pytest.raises((NotImplementedError, ValidationError, ValueError)):
        Validator(_nested_annotation(1000))


def _compose_in_a_loop(compose: Callable[[object], object]) -> None:
    schema: object = Validator(int)
    for _ in range(1000):
        schema = compose(schema)


@pytest.mark.parametrize(
    "compose",
    [
        lambda s: Validator([s]) | str,
        lambda s: union(Validator([s]), str),
        lambda s: intersection(Validator([s]), str),
        lambda s: complement(Validator([s])),
    ],
)
def test_composition_depth_guard_rejects_unbounded_nesting(
    compose: Callable[[object], object],
) -> None:
    # Each combinator call grows the schema by one nesting level. Past the
    # depth guard the call raises a clean ValueError instead of letting the next
    # clone, decision, or render walk overflow the native stack. Every combinator
    # family — the `|` operator, union, intersection, and complement — is bounded
    # the same way.
    #
    # Each step wraps the subject in a list first, because a *repeated* join or
    # meet of the same two members does not grow at all: a schema is built in the
    # lattice normal form, so `s | str` twice is `s | str`. That is the subject of
    # the test below, and it is why the growing shape here has to be one the
    # normal form keeps.
    with pytest.raises(ValueError, match="too deep"):
        _compose_in_a_loop(compose)


def test_a_repeated_composition_does_not_grow():
    # Idempotence, the identities and the complement laws are settled where the
    # schema is built, so a loop that re-applies the same step reaches a fixed
    # point instead of a bound. There is nothing to guard, which is a better
    # answer than guarding it: the set stops growing, so the schema does.
    v = Validator(int)
    for _ in range(1000):
        v = v | str
    assert v == Validator(int) | str

    v = Validator(int)
    for _ in range(1000):
        v = union(v, v)
    assert v == Validator(int)

    # `~~A` is `A`, so complementing in a loop oscillates between two shapes.
    v = Validator(int)
    for _ in range(1000):
        v = complement(v)
    assert v == Validator(int)  # an even count cancels away entirely
    assert repr(complement(v)) == "complement(int)"


@pytest.mark.parametrize(
    ("door", "loop"),
    [
        ("constructor list literal", "v = Validator([v])"),
        ("constructor list[...]", "v = Validator(list[v])"),
        ("union operator", "v = Validator([v]) | str"),
        ("complement of a list", "v = complement(Validator([v]))"),
        ("recursive body", "v = recursive(lambda s, prev=v: [prev, s])"),
    ],
)
def test_every_construction_door_rejects_unbounded_depth(door: str, loop: str) -> None:
    # The depth guard lives at schema construction, not only at the combinators,
    # so no public way of growing a schema in a loop can overflow the stack. Each
    # door is driven past the bound in a subprocess; a clean ValueError exits 0,
    # a native stack overflow would exit with a signal.
    result = _run_construction_loop(
        f"v = Validator(int)\nfor _ in range(50_000):\n    {loop}"
    )
    assert result.returncode == 0, f"{door}: crashed with rc={result.returncode}"
    assert "RAISED" in result.stdout, f"{door}: {result.stdout} {result.stderr}"


def test_self_combination_rejects_before_exhausting_memory() -> None:
    # Combining a validator with two *different* growing copies of itself doubles
    # its node count each step while its depth barely grows, so only the
    # node-count bound catches it. The subprocess must reject cleanly, never OOM.
    # `union(v, v)` is `v`, so the two copies have to differ for the count to
    # double at all.
    result = _run_construction_loop(
        "v = Validator(int)\n"
        "for _ in range(60):\n"
        "    v = union(Validator([v]), Validator({'k': v}))"
    )
    assert result.returncode == 0, f"crashed with rc={result.returncode}"
    assert "RAISED" in result.stdout, f"{result.stdout} {result.stderr}"


def test_chained_definitions_reject_before_overflowing_render() -> None:
    # A chain of distinct recursive definitions is invisible to the per-tree depth
    # measure (a Ref is a leaf) but overflows the render/decision walk one frame
    # per link, so the definition-count bound is what catches it.
    result = _run_construction_loop(
        "v = Validator(int)\n"
        "for _ in range(50_000):\n"
        "    v = recursive(lambda s, prev=v: [prev, s])"
    )
    assert result.returncode == 0, f"crashed with rc={result.returncode}"
    assert "RAISED" in result.stdout, f"{result.stdout} {result.stderr}"


def test_a_schema_at_the_depth_limit_still_works() -> None:
    # A schema right at the depth limit still builds, validates, decides
    # emptiness, and reprs without a crash: the guard rejects only past the bound,
    # not at it. The limit is imported, not hard-coded, so it cannot silently
    # drift away from the tested edge.
    deep = Validator(int)
    for _ in range(MAX_SCHEMA_DEPTH - 1):
        deep = Validator([deep])
    assert deep.is_valid(_nested_value(MAX_SCHEMA_DEPTH - 1))
    assert not deep.is_empty()
    assert isinstance(repr(deep), str)
    # One more step crosses the bound and is rejected.
    with pytest.raises(ValueError, match="too deep"):
        Validator([deep])


def test_the_published_bounds_are_positive() -> None:
    # The bounds a caller sizes schemas against are exported and sane.
    assert MAX_SCHEMA_DEPTH > 0
    assert MAX_DEFINITIONS > 0
    assert MAX_SCHEMA_NODES > MAX_SCHEMA_DEPTH


def test_deeply_nested_object_hits_the_recursion_limit() -> None:
    schema = Validator(recursive(lambda j: union(int, [j])))
    deep = _nested_value(5000)
    # is_valid swallows the bound as a non-membership; validate names it.
    assert not schema.is_valid(deep)
    with pytest.raises(ValidationError) as info:
        schema.validate(deep)
    assert info.value.code == "recursion_limit"


def test_a_deep_body_reaches_the_walk_bound_rather_than_the_stack() -> None:
    # The walk descends one native frame per level, and a recursive definition
    # descends its whole body once per level of the value: the frames a value can
    # demand are the *product* of the unfolding bound and the body's depth, not
    # either alone. A body 60 records deep unfolded 127 times asks for thousands
    # of frames, which is a bound the walk holds rather than a stack it exhausts.
    body_depth, unfoldings = 60, 127

    def wrap(inner: object, levels: int) -> object:
        for _ in range(levels):
            inner = {"c": inner}
        return inner

    schema = Validator(recursive(lambda j: union(None, wrap(j, body_depth))))
    value: object = None
    for _ in range(unfoldings):
        value = wrap(value, body_depth)

    assert not schema.is_valid(value)
    with pytest.raises(ValidationError) as info:
        schema.validate(value)
    assert info.value.code == "recursion_limit"


def test_a_recursive_value_at_the_unfolding_bound_still_validates() -> None:
    # The walk bound sits above what the published unfolding bound asks of a
    # linked list, so bounding the descent refuses only the shapes that would
    # have exhausted the stack, not the recursion the schema language is for.
    schema = Validator(recursive(lambda t: union(None, {"next": t})))
    node: object = None
    # One short of the published 128 levels of recursion, spelled out because the
    # bound is documentation rather than a name a caller imports.
    for _ in range(127):
        node = {"next": node}
    assert schema.is_valid(node)


def test_deeply_nested_json_is_rejected_cleanly() -> None:
    schema = Validator(recursive(lambda j: union(int, [j])))
    document = "[" * 5000 + "1" + "]" * 5000
    assert not schema.is_valid_json(document)
    with pytest.raises(ValidationError) as info:
        schema.validate_json(document)
    assert info.value.code == "json_invalid"


def test_self_referential_value_is_caught_as_a_loop() -> None:
    schema = Validator(recursive(lambda j: union(int, [j])))
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert not schema.is_valid(cyclic)
    with pytest.raises(ValidationError) as info:
        schema.validate(cyclic)
    # The value's self-reference is caught by the cycle guard, not a generic union
    # miss: pin the exact code so a regression to `union_error` is visible.
    assert info.value.code == "recursion_loop"


def test_wide_union_membership_is_decided_and_bounded() -> None:
    wide = union(*[Literal[i] for i in range(5000)])  # ty: ignore[invalid-type-form]
    # The value-driven work (the linear scan and the capped closest-branch probe)
    # terminates with the right verdict for both a member and a non-member.
    assert wide.is_valid(4999)
    assert not wide.is_valid(10_000)
    with pytest.raises(ValidationError) as info:
        wide.validate(10_000)
    # One aggregated union failure: the probe cap keeps the report from rewalking
    # every branch of the union.
    assert info.value.code == "union_error"
    assert len(info.value.errors) == 1


def test_hostile_dict_keys_are_handled() -> None:
    mapping = Validator(dict[str, int])
    assert mapping.is_valid({str(i): i for i in range(100_000)})
    assert mapping.is_valid({"k" * 1_000_000: 1})
    assert not mapping.is_valid({"bad": "not an int"})


# --- Every whole-schema transform is bounded ----------------------------------
#
# The construction bounds are enforced at one choke point, and a transform that
# rebuilds a validator without reaching it hands the next caller a schema already
# past the ceiling. Opening a closed record adds a catch-all clause, so `open` is
# a growth path and not a size-preserving rebuild.

_TRANSFORMS = ["open", "close", "simplify", "__copy__"]


@pytest.mark.parametrize("transform", _TRANSFORMS)
def test_no_whole_schema_transform_escapes_the_node_bound(transform: str) -> None:
    # A union of closed records, sized so the schema is within the node ceiling
    # and opening it is not: a record is two nodes and a catch-all clause adds two
    # more, so `n` records span `2n + 1` nodes closed and `4n + 1` open, and
    # `MAX_SCHEMA_NODES // 3` sits between the two. Rejecting is as correct as
    # staying within the bound; handing back an oversized validator is not.
    records = MAX_SCHEMA_NODES // 3
    within = union(*[Validator({f"f{i}": int}) for i in range(records)])
    try:
        rebuilt = getattr(within, transform)()
    except ValueError:
        return
    # Composing the result reports its size, so a validator past the ceiling is
    # visible here even though the transform accepted it.
    try:
        union(rebuilt, Validator(int))
    except ValueError as error:
        message = str(error)
        assert "too large" not in message, (
            f"{transform}() returned a validator already past the node bound: {message}"
        )


def test_a_pattern_whose_determinisation_explodes_answers_in_bounded_time() -> None:
    """A subtype question must not be a way to exhaust the process's memory.

    `(a|b)*a(a|b){k}` doubles its DFA states for every `k`: the automaton has to
    remember the last `k` letters to know whether an `a` sat `k` back. The
    descriptor bounds the automaton it keeps, but that bound is checked after
    the regex engine has built the whole dense table -- so `k = 20` spent six
    seconds and 668 MB reaching a refusal, and `k = 25` aborted the interpreter
    on a four-gigabyte allocation. A caller who lets a user supply a pattern had
    handed that user the process.

    The engine is now given the size limit, so the refusal happens where the
    memory would be spent. Timed rather than merely answered: a bound that only
    stops the allocation would still leave the question taking minutes.
    """
    started = time.perf_counter()
    for exponent in (16, 20, 25, 40):
        pattern = Annotated[str, Regex(f"(a|b)*a(a|b){{{exponent}}}")]
        # Undecided is the honest answer: the descriptor refused the pattern, so
        # no relation over it is proved either way.
        assert not Validator(pattern).is_subtype_of(Annotated[str, Regex("(a|b)*")])
        # And membership is unaffected -- the walk runs the regex, not the
        # automaton, so the schema still accepts and rejects.
        assert Validator(pattern).is_valid("a" * (exponent + 1))
        assert not Validator(pattern).is_valid("b")
    elapsed = time.perf_counter() - started
    assert elapsed < 30, f"four relations took {elapsed:.1f}s; the bound is not biting"


def test_a_pattern_that_stays_small_is_still_decided() -> None:
    """The bound is on the table, not on the shape or the length of the source."""
    small = Annotated[str, Regex("(a|b)*a(a|b){4}")]
    assert Validator(small).is_subtype_of(Annotated[str, Regex("(a|b)*")])
    long_but_simple = Annotated[str, Regex("a" * 2000)]
    assert Validator(long_but_simple).is_subtype_of(Annotated[str, Regex("a*")])


def test_explaining_a_deep_value_does_not_scale_with_its_size() -> None:
    """A summary is bounded while it is built, not built and then cut.

    The walk stops at `MAX_WALK_DEPTH` levels, so an error over a deeply nested
    value names a bounded path -- but every violation along the way summarised
    the value it was about, and a summary was the value's *whole* repr cut to
    eighty characters afterwards. A 20,000-deep list has a 40,000-character
    repr, built once per level and discarded, which was twelve seconds for a
    single error against twenty microseconds for `is_valid` on the same value.

    Timed against depth rather than against a constant: the defect was a cost
    that grew with the size of the value, and only a comparison across sizes
    fails if it comes back.
    """
    schema = Validator(recursive(lambda node: union(int, [node])))

    def explain(depth: int) -> float:
        value: object = "not an int"
        for _ in range(depth):
            value = [value]

        def once() -> float:
            started = time.perf_counter()
            with pytest.raises(ValidationError):
                schema.validate(value)
            return time.perf_counter() - started

        # The fastest of three, for the reason the sibling bound in
        # `tests/test_equivalence.py` takes the fastest of five: a loaded
        # machine only ever adds to a reading.
        return min(once() for _ in range(3))

    small, large = explain(2_000), explain(20_000)
    # Ten times the value, and the work must not follow it. Generous, because a
    # loaded machine moves a millisecond around: quadratic was a factor of 40.
    assert large < small * 5 + 0.05, (
        f"explaining a 20,000-deep value took {large:.3f}s against "
        f"{small:.3f}s for a 2,000-deep one; the summary is scaling with the value"
    )


def test_a_bounded_summary_still_names_a_small_value_exactly() -> None:
    """The bound must not cost the messages that were already readable."""

    def message(schema: object, value: object) -> str:
        with pytest.raises(ValidationError) as info:
            Validator(schema).validate(value)
        return str(info.value.errors[0]["message"])

    assert message(int, [1, 2, 3]) == "expected int, got [1, 2, 3] [int_type]"
    assert message(int, {"a": 1}) == "expected int, got {'a': 1} [int_type]"
    assert message(int, (1, 2)) == "expected int, got (1, 2) [int_type]"
    assert message(int, "text") == "expected int, got 'text' [int_type]"
    # And a value too large to print is cut rather than printed.
    assert len(message(int, list(range(10_000)))) < 200


def test_two_steps_that_meet_past_the_period_bound_are_refused() -> None:
    """A pair of ordinary steps must not be a way to crash or to lie.

    A `MultipleOf` lowers to a set held as one interval set per residue, so its
    period is the step, and the representation caps that period. The cap was on
    one step and not on two: `MultipleOf(64)` and `MultipleOf(81)` are each far
    inside it and meet at 5,184, which is past it. Asking whether their meet was
    empty then materialised a table at the largest period the representation
    holds -- the multiples of something else -- and answered from it, or, on a
    build with debug assertions (which is what `maturin develop` produces),
    raised a panic across the language boundary that no caller catches as a
    validation failure.

    Composition is now bounded like the automaton components are: past the
    shared period the operation refuses, the descriptor becomes unbuildable and
    the relation stays undecided. Membership is unaffected either way, because
    the walk runs the modulo rather than the set.
    """

    def step(n: int) -> Validator:
        return Validator(Annotated[int, MultipleOf(n)])

    for left, right, shared in [
        (64, 81, 5184),
        (4093, 4096, 16764928),
        (3, 4096, 12288),
    ]:
        met = intersection(step(left), step(right))
        # Undecided, not empty: the two steps do share their multiples, and a
        # `True` here would be the wrong verdict the refusal exists to avoid.
        assert not met.is_empty(), f"{left} and {right} meet at {shared}"
        assert met.is_valid(shared)
        assert not met.is_valid(shared + left)
        assert not met.is_valid(shared + right)

    # A pair whose shared period is inside the bound still answers, so the
    # refusal costs only what it must.
    inside = intersection(step(63), step(64))
    assert inside.is_valid(4032)
    assert not inside.is_valid(63)
    assert not inside.is_empty()
    # And a step against itself is decided, whatever the step.
    for n in (2, 64, 81, 4096):
        assert step(n).is_subtype_of(step(n))
        assert intersection(step(n), complement(step(n))).is_empty()
