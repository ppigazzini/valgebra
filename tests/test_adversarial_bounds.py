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

import json
import subprocess
import sys
import textwrap
import time
import types
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
    _debug_build,
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


# BOUND: MAX_BUILD_DEPTH
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


def test_a_repeated_composition_does_not_grow() -> None:
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


def test_the_definition_bound_admits_its_own_number_and_refuses_one_past() -> None:
    """The bound is `more than MAX_DEFINITIONS`, at the number itself and past it.

    A bound stated as a limit has two edges and only one of them is a refusal.
    Nothing drove either: the suite built chains far past the number, where a
    comparison off by one still refuses, so a bound reading `>=` -- which would
    refuse the largest schema a caller is promised -- or `==` -- which would
    admit everything above it -- passed unnoticed. Both edges are asked here.
    """

    def chain(links: int) -> object:
        # Distinct recursive definitions, which is what the bound counts: each
        # `recursive` call adds one, and nesting them keeps the depth small.
        schema: object = recursive(lambda t: union(None, [t]))
        for _ in range(links - 1):
            inner = schema
            schema = recursive(lambda t, inner=inner: union(inner, [t]))
        return schema

    # The number itself is the largest a caller is promised, and it builds.
    assert Validator(chain(MAX_DEFINITIONS)).is_valid(None)
    # One past it is refused, by the message that names the bound.
    with pytest.raises(ValueError, match="too many recursive definitions"):
        Validator(chain(MAX_DEFINITIONS + 1))


def test_the_published_bounds_are_positive() -> None:
    # The bounds a caller sizes schemas against are exported and sane.
    assert MAX_SCHEMA_DEPTH > 0
    assert MAX_DEFINITIONS > 0
    assert MAX_SCHEMA_NODES > MAX_SCHEMA_DEPTH


# BOUND: MAX_RECURSION_DEPTH
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


def test_a_recursive_meet_of_records_is_decided_rather_than_overflowing() -> None:
    # The key both records require names the fixpoint, so deciding the meet
    # asks the meet of the key's types, and that unfolds the fixpoint into the
    # same meet. The decision reads the second unfolding as the cycle it is. A
    # child process, because the failure this guards is the stack giving out,
    # and that takes the interpreter with it.
    program = textwrap.dedent(
        """
        from valgebra import Validator, intersection, recursive, union
        def meet(t):
            return intersection({"a": union(t, int)}, {"a": union(t, str)})
        node = Validator(recursive(meet))
        print(node.is_empty(), node.is_subtype_of(int))
        """
    )
    result = subprocess.run(  # noqa: S603 -- fixed interpreter, in-repo program
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # The meet holds no finite value, so any answer either gives is sound: the
    # claim is that both finish.
    assert len(result.stdout.split()) == 2, result.stdout


#: The stack a thread validating deep values needs, as the limits page states
#: it: 1 MiB, the least any thread CPython creates has. The release smoke runs
#: this file on every wheel it ships, which is where the figure is held. An
#: unoptimized build spends several times a release build's stack a level --
#: its explaining walk of a refused value at the bound needs 2 MiB -- and no
#: wheel is one.
_THREAD_STACK = (4 if _debug_build else 1) * 1024 * 1024


def _on_a_thread(program: str) -> subprocess.CompletedProcess[str]:
    """Run `program`'s `run` in a fresh interpreter, on a thread of that stack.

    A child process, because the failure this guards is the stack giving out,
    and that takes the interpreter with it. `run` prints what it found, so a
    thread that raised -- which leaves the process's return code at 0 -- is
    caught by its missing output.
    """
    source = textwrap.dedent(program) + textwrap.dedent(
        f"""
        import threading
        threading.stack_size({_THREAD_STACK})
        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
        """
    )
    return subprocess.run(  # noqa: S603 -- fixed interpreter, in-repo program
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


# BOUND: MAX_WALK_DEPTH
def test_the_deepest_walk_fits_the_documented_stack() -> None:
    """Every walk up to the depth bound runs on a 1 MiB thread.

    The smallest recursive body, whose unfolding bound refuses first, and a
    body of three records, which meets the walk's own bound first. For each:
    the deepest member, a value refused at the deepest level under the bound,
    whose explaining walk is the dearest there is, and a value past the bound.
    The profile-guided wheel needed 1.5 MiB for these at a bound of 512.
    """
    result = _on_a_thread(
        """
        from valgebra import ValidationError, Validator, recursive, union

        def wrapped(levels, leaf, wrap):
            for _ in range(levels):
                leaf = wrap(leaf)
            return leaf

        def code(schema, value):
            try:
                schema.validate(value)
            except ValidationError as error:
                return error.code
            return "member"

        shapes = {
            "lists": (recursive(lambda t: union(int, [t])), lambda v: [v]),
            "records": (
                recursive(lambda t: union(int, {"a": {"b": {"c": t}}})),
                lambda v: {"a": {"b": {"c": v}}},
            ),
        }

        def run():
            for name, (schema, wrap) in shapes.items():
                schema = Validator(schema)
                first = next(
                    n for n in range(1, 400)
                    if not schema.is_valid(wrapped(n, 0, wrap))
                )
                print(
                    name,
                    schema.is_valid(wrapped(first - 1, 0, wrap)),
                    code(schema, wrapped(first - 1, "leaf", wrap)),
                    code(schema, wrapped(first + 20, 0, wrap)),
                )
            lists = Validator(shapes["lists"][0])
            print("document", lists.is_valid_json("[" * 127 + "0" + "]" * 127))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "lists True union_error recursion_limit",
        "records True union_error recursion_limit",
        "document True",
    ], result.stderr


# BOUND: MAX_DECISION_DEPTH
def test_a_relation_fits_the_documented_stack() -> None:
    """A relation the trail cannot close quickly runs on a 1 MiB thread.

    Two fixpoints nesting 100 and 99 lists around the back edge meet their
    hypothesis only after 9,900 levels of goals, and a chain of 96 definitions
    a hundred and ten one-tuples deep asks its emptiness that deep. Each
    overflowed the main thread's 8 MiB before the decision bounded its own
    depth; past the bound each declines. A pair whose cycles meet under it is
    decided.
    """
    result = _on_a_thread(
        """
        from valgebra import Validator, recursive

        def nest(leaf, levels):
            for _ in range(levels):
                leaf = list[leaf]
            return leaf

        def cycles(p, q):
            return (
                recursive(lambda s: nest(s, p)),
                recursive(lambda s: nest(s, q)),
            )

        def chain(leaf):
            schema = Validator(leaf)
            for _ in range(96):
                def body(s, before=schema):
                    inner = tuple[before, list[s]]
                    for _ in range(110):
                        inner = tuple[inner]
                    return inner
                schema = recursive(body)
            return schema

        def run():
            a, b = cycles(100, 99)
            print(a.relation_to(b), a.is_equivalent(b))
            a, b = cycles(21, 20)
            print(a.relation_to(b))
            ints, words = chain(int), chain(int | str)
            print(ints.relation_to(words), ints.is_empty())
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "undecided False",
        "subset",
        "undecided False",
    ], result.stderr


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


def test_the_unfolding_bound_admits_127_lists_and_refuses_128() -> None:
    # The pages name the boundary by its value rather than by the count alone:
    # around an `int` at the bottom, which is itself an unfolding, 127 lists are
    # a member of the smallest recursive body and 128 are refused.
    schema = Validator(recursive(lambda t: union(int, [t])))

    def nested(lists: int) -> object:
        value: object = 0
        for _ in range(lists):
            value = [value]
        return value

    assert schema.is_valid(nested(127))
    with pytest.raises(ValidationError) as info:
        schema.validate(nested(128))
    assert info.value.code == "recursion_limit"


def test_deeply_nested_json_is_rejected_cleanly() -> None:
    schema = Validator(recursive(lambda j: union(int, [j])))
    document = "[" * 5000 + "1" + "]" * 5000
    assert not schema.is_valid_json(document)
    with pytest.raises(ValidationError) as info:
        schema.validate_json(document)
    assert info.value.code == "json_invalid"


# THEORY: the-depth-bound-reports-itself
def test_the_parser_and_the_walk_bound_a_document_in_that_order() -> None:
    """Which bound a deep document reaches, and which one it reaches first.

    `docs/10-limits.md` says the code names which bound was reached:
    `recursion_limit` while the parser could still read the document,
    `json_invalid` once it could not. That is a claim about the *order* of two
    bounds in different crates -- the walk's unfolding bound and jiter's own
    recursion limit -- and it holds only while the parser's is the wider of the
    two. Neither is a number a caller imports, so nothing but this row would
    notice the day one moved past the other and the page started describing a
    sequence the tree does not have.

    The walk's total-descent bound is a third number and is *not* reachable
    through a document: the parser refuses first, which is why the page reads it
    against the unfolding bound rather than against the descent one.
    """
    schema = Validator(recursive(lambda j: union(int, [j])))

    def document(levels: int) -> str:
        return "[" * levels + "1" + "]" * levels

    def code(levels: int) -> str | None:
        try:
            schema.validate_json(document(levels))
        except ValidationError as error:
            return error.code
        return None

    # Inside the published unfolding bound, the document is a member. One
    # short of it, for the reason the row above spells: the outermost array is
    # itself an unfolding.
    assert code(127) is None
    # Past it and inside the parser's, the walk is what refuses.
    assert code(128) == "recursion_limit"
    assert code(200) == "recursion_limit"
    # Past the parser's, the document stops being one before the walk sees it.
    assert code(201) == "json_invalid"

    # And the ordering that makes the sentence true: the parser reads further
    # than the walk unfolds, so there is a band where the walk is the answer.
    assert code(128) != code(201)


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


# BOUND: CLOSEST_BRANCH_PROBE_LIMIT
def _colliding_keys(count: int) -> list[str]:
    """Return `count` keys of 32 ASCII bytes that all hash alike in `rustc-hash`.

    `rustc-hash` 2.x hashes a 32-byte string as `multiply_mix(SEED2 ^ middle,
    t ^ tail)`, with `t` a function of the first sixteen bytes, so a tail equal
    to `t` zeroes the product whatever the middle eight bytes are. The crate is
    unseeded, so the constants below are every process's and the keys are
    computed here, as a document's author could compute them: the first
    sixteen-digit head whose `t` is printable will do.
    """
    # The two constants the head's mix reads; the seed beside the middle bytes
    # drops out, because the product it enters is zero.
    seed1, prevent = 0x243F6A8885A308D3, 0xA4093822299F31D0
    word = (1 << 64) - 1
    # Printable ASCII without the two bytes a JSON string would escape.
    allowed = {*range(0x20, 0x22), *range(0x23, 0x5C), *range(0x5D, 0x7F)}
    for attempt in range(10**6):
        head = f"{attempt:016d}".encode()
        product = (seed1 ^ int.from_bytes(head[:8], "little")) * (
            prevent ^ int.from_bytes(head[8:], "little")
        )
        tail = ((product & word) ^ (product >> 64)).to_bytes(8, "little")
        if all(byte in allowed for byte in tail):
            break
    return [(head + f"{i:08d}".encode() + tail).decode() for i in range(count)]


def test_a_documents_colliding_keys_are_read_in_linear_time() -> None:
    """A parsed object's keys are not trusted to hash apart.

    The undeclared keys of a parsed object wider than eight entries go through
    a table keyed by the document's own keys, and that table hashed with
    `rustc-hash`, which is unseeded: 20,000 keys computed to collide, 0.74 MB,
    took half a second where as many ordinary keys took 2 ms, and four times as
    long for each doubling. The table is keyed per process now, so the keys
    collide in the document only.
    """
    schema = Validator(dict[str, int])

    def document(keys: list[str]) -> str:
        return "{" + ",".join(json.dumps(key) + ":1" for key in keys) + "}"

    count = 20_000
    colliding = document(_colliding_keys(count))
    ordinary = document([f"k{i:030d}x" for i in range(count)])
    assert len(colliding) == len(ordinary)
    slow = _fastest(lambda: schema.is_valid_json(colliding))
    fast = _fastest(lambda: schema.is_valid_json(ordinary))
    assert schema.is_valid_json(colliding)
    # Generous, because a loaded machine moves a millisecond around: the
    # colliding document read 250 times slower than the ordinary one.
    assert slow < fast * 5 + 0.05, (
        f"{count:,} colliding keys took {slow:.3f}s against {fast:.3f}s for "
        "ordinary ones; the table is hashing the document's keys predictably"
    )


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


def test_opening_past_the_node_bound_refuses() -> None:
    """Opening a validator near the ceiling raises rather than handing one back.

    The row below reads a transform that *may* refuse or may stay within the
    bound, and passes either way -- which is right for a transform whose result
    depends on the shape, and holds nothing about the refusal itself. `open` at
    this size does refuse, and the docstring promises the `ValueError`, so the
    promise is asserted rather than tolerated.

    A record is two nodes and a catch-all clause adds two more, so a union of
    `MAX_SCHEMA_NODES // 3` records spans well under the ceiling closed and well
    over it open.
    """
    records = MAX_SCHEMA_NODES // 3
    within = union(*[Validator({f"f{i}": int}) for i in range(records)])
    with pytest.raises(ValueError, match="too large"):
        within.open()


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


def test_a_schema_of_exactly_the_node_ceiling_builds() -> None:
    """The ceiling is a limit, not the first refused size.

    The refusal reads "past the limit", and a schema *at* the limit is not past
    it. Read one off, the message contradicts itself -- "spans 100000 nodes,
    past the limit of 100000" -- and a caller sizing against the published
    number is refused at the size the number told them was allowed.

    Sized from the bound rather than written out: a record is two nodes, a
    union node is one, and `int` is one, so `n` records beside an `int` span
    `2n + 2`. The count is the whole point, so it is derived from
    `MAX_SCHEMA_NODES` and moves with it.
    """
    records = (MAX_SCHEMA_NODES - 2) // 2
    at_the_ceiling = [Validator({f"f{i}": int}) for i in range(records)]
    union(*at_the_ceiling, Validator(int))

    # One record further is two nodes past it, and refused by name.
    with pytest.raises(ValueError, match="too large"):
        union(*at_the_ceiling, Validator({"last": int}), Validator(int))


def test_an_annotation_of_exactly_the_node_ceiling_builds() -> None:
    """The constructor's door holds the bound where the combinators' does.

    One `Validator(spec)` call reads a whole annotation, and the frontend holds
    what it builds to the node bound as it reads. A flat `tuple` of `n` `int`s
    spans `n + 1` nodes, so the tuple at the ceiling builds and one element
    more is refused by name. Spelled as the alias object, since the length is
    the bound's and no checker reads a type form computed at run time.
    """
    Validator(types.GenericAlias(tuple, (int,) * (MAX_SCHEMA_NODES - 1)))
    with pytest.raises(ValueError, match="too large"):
        Validator(types.GenericAlias(tuple, (int,) * MAX_SCHEMA_NODES))


@pytest.mark.skipif(
    sys.platform == "win32", reason="the high-water mark is read from `resource`"
)
def test_a_class_named_from_many_places_is_refused_before_it_is_built() -> None:
    """A shared part is refused at the node bound, not after it is built whole.

    Four records a level, each holding the union of the level below: 48
    classes, and a schema that multiplies by four with each level -- 134
    million nodes at twelve, gigabytes of them built whole. Held as it is
    read, the build stops a step past the bound, and six levels, within it,
    build. A child process, because the failure this guards is memory, and
    the high-water mark is the process's.
    """
    program = textwrap.dedent(
        """
        import resource
        from typing import Literal, TypedDict
        from valgebra import Validator
        def levels(depth):
            below = int
            for level in range(depth):
                tagged = [
                    TypedDict(f"L{level}{tag}", {"type": Literal[tag], "left": below})
                    for tag in "abcd"
                ]
                below = tagged[0] | tagged[1] | tagged[2] | tagged[3]
            return below
        Validator(levels(6))
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            Validator(levels(12))
        except ValueError as error:
            assert "too large" in str(error), error
        else:
            raise AssertionError("twelve levels built")
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before)
        """
    )
    result = subprocess.run(  # noqa: S603 -- fixed interpreter, in-repo program
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # `ru_maxrss` is in kibibytes on Linux and in bytes on macOS.
    unit = 1 if sys.platform == "darwin" else 1024
    assert int(result.stdout) < 100 * 1024 * 1024 // unit, result.stdout


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


def _fastest(call: Callable[[], object]) -> float:
    """Return the fastest of three timed calls; a loaded machine only adds."""

    def once() -> float:
        started = time.perf_counter()
        call()
        return time.perf_counter() - started

    return min(once() for _ in range(3))


def test_a_tag_after_a_recursive_field_refuses_its_branch_first() -> None:
    """A union of records told apart by a tag is linear, whatever the tag is called.

    Fields are stored sorted by name, so `type` beside `left` was read after
    the child, and each branch the tag refuses walked the whole child first:
    four branches to the power of the depth, a second at depth ten and no
    answer at twenty, on a valid value of a few hundred bytes. The deciding
    walk reads a field decided by its own value first. Both entry points, as
    the JSON path reads a record by the same plan.
    """
    tags = (Literal["a"], Literal["b"], Literal["c"], Literal["d"])
    schema = Validator(
        recursive(lambda t: union(int, *[{"left": t, "type": tag} for tag in tags]))
    )

    def chain(depth: int) -> object:
        value: object = 0
        for _ in range(depth):
            value = {"left": value, "type": "d"}
        return value

    for check in (schema.is_valid, lambda v: schema.is_valid_json(json.dumps(v))):
        small, large = (
            _fastest(lambda d=d, check=check: check(chain(d))) for d in (6, 12)
        )
        assert check(chain(12)) is True
        # Twice the depth: linear is a factor of two, and the defect was 4,096.
        assert large < small * 5 + 0.05, (small, large)


def test_explaining_a_refused_union_of_records_does_not_double_per_level() -> None:
    """A union of records explains one record branch, not each of them.

    Explaining every refused record branch walked the levels below once per
    branch, so the default report on a refused chain doubled with each level:
    a third of a second at eighteen levels of a value of three hundred bytes.
    The branch explained is the one whose deciding walk admitted the most.
    """
    schema = Validator(
        recursive(
            lambda t: union(
                None, {"z": t, "t": Literal["x"]}, {"z": t, "t": Literal["y"], "u": str}
            )
        )
    )

    def chain(depth: int) -> object:
        value: object = 3.5
        for _ in range(depth):
            value = {"z": value, "t": "x"}
        return value

    def explain(depth: int) -> Callable[[], object]:
        def refused() -> None:
            with pytest.raises(ValidationError):
                schema.validate(chain(depth))

        return refused

    small, large = _fastest(explain(8)), _fastest(explain(16))
    # Twice the depth: the report is quadratic at worst, a factor of four, and
    # the defect was 256.
    assert large < small * 6 + 0.05, (small, large)


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


# BOUND: MAX_MARKER_TYPES
def test_a_marker_type_past_the_cache_bound_is_still_read() -> None:
    """`MAX_MARKER_TYPES`: the cache stops growing; the reading stays right.

    Which attributes a refinement marker carries is a property of its type, so
    it is asked once per type and kept -- and a cache entry keeps its type
    alive, so a program that builds a marker class per call would grow it
    forever. Past the bound a type is read each time instead of remembered,
    which is the slower path and must not be a different answer.

    Built well past the bound on purpose: the rows below are the ones the cache
    cannot have seen.
    """
    # Four times the bound `docs/dev/00-architecture.md` records for it. The
    # value is not published -- it changes no answer, only how often one is
    # recomputed -- so it is spelled here, and a bound raised past this is a
    # bound whose test stops reaching the far side.
    past_the_cache = 4 * 256
    # Of the vocabulary's module, the one a constraint is read off.
    made = [
        type(
            f"Bound{n}", (), {"__slots__": (), "ge": n, "__module__": "annotated_types"}
        )()
        for n in range(past_the_cache)
    ]
    for n, marker in enumerate(made):
        schema = Validator(Annotated[int, marker])
        assert schema.is_valid(n), "the bound its type carries admits its own value"
        assert not schema.is_valid(n - 1), "and refuses the one below it"


# TRUST: `isinstance` and the PyO3 conversions report Python's own membership.
def test_a_lying_class_attribute_does_not_admit_a_value_to_a_builtin_kind() -> None:
    """A kind is read from the value's real type, not from what it claims.

    `isinstance` consults `__class__`, and a property can answer that with any
    type at all: an object declaring itself an `int` passes
    `isinstance(value, int)` while holding none of an integer's storage. A
    schema over a builtin kind that believed it would admit a value with
    nothing an integer's operations could read.

    `isinstance` reads the real type first and consults `__class__` only when
    that fails, which is what makes the second row below the sharper of the
    two: a genuine `int` subclass claiming to be a `str` is admitted to `int`
    by its type *and* to `str` by its claim. Python says it is both, which no
    value is. Reading the real type gives one answer to each kind.
    """

    class Impostor:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            return int

    class Disclaiming(int):
        @property
        def __class__(self) -> type:  # type: ignore[override]
            return str

    impostor = Impostor()
    assert isinstance(impostor, int)  # Python believes it
    assert not Validator(int).is_valid(impostor)

    # A genuine subclass that claims another kind is admitted to both by
    # Python: to `int` by its real type, and to `str` by its claim.
    disclaiming = Disclaiming(1)
    assert isinstance(disclaiming, int)
    assert isinstance(disclaiming, str)
    # It is an integer and it is not a string, and a schema says exactly that.
    assert Validator(int).is_valid(disclaiming)
    assert not Validator(str).is_valid(disclaiming)
    # Which is what keeps the two kinds disjoint: believing the claim would
    # put one value in both, and `int & str` is proved empty.
    assert Validator(intersection(int, str)).is_empty()

    # A container is read the same way: a claim is not storage.
    class ListImpostor:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            return list

    assert not Validator(list[int]).is_valid(ListImpostor())


def test_a_user_class_is_whatever_python_says_it_is() -> None:
    """Membership of a user class *is* `isinstance`, so a lie there is honoured.

    The boundary the row above sits on. A builtin kind has storage this library
    can read and a claim it can check against; a user class has neither, and
    `isinstance` is the definition of belonging to one rather than a reading of
    it. Overriding `__class__` is a documented way to write a proxy, and a
    proxy that says it is a `Target` is one every other consumer treats as one.
    """

    class Target:
        pass

    class Proxy:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            return Target

    proxy = Proxy()
    assert isinstance(proxy, Target)
    assert Validator(Target).is_valid(proxy)
