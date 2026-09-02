"""Metamorphic gate: two builds agree about values, and only grow more decisive.

A refactor that is not meant to change what valgebra means has no test of its
own. The suites check the tree against its specification, so a change that is
wrong in the *same* way on both sides of the refactor passes them both, and a
change that is right but unintended passes them too. What is missing is a
comparison against the tree as it was.

That comparison is metamorphic in the sense of Chen et al.: it relates the
outputs of two runs rather than checking one output against an expected value,
so it needs no oracle for what the answer should be. Two relations hold between
a reference build and the build under test:

* **Membership does not move.** For every schema and value in the corpus, the
  two builds accept and reject exactly the same pairs. Membership is the
  denotation, so a single flip is a semantic change -- which may be intended,
  but is never a refactor.
* **Decisions only widen.** A relation the reference proved must still be
  proven: the procedure is sound, so losing a `True` loses a proof. Gaining one
  is allowed, because the procedure is deliberately incomplete and every
  milestone after this one makes it decide more.

The second relation has an escape the first does not, and this gate closes it: a
build that answered `True` to everything would satisfy "only widens" perfectly.
So every new `True` is checked against the corpus values. A subtyping claim
`A <= B` with a corpus value in `A` and not in `B` is a false proof, and the
gate fails on it rather than recording the widening.

Usage:
    python scripts/metamorphic_gate.py            # compare against the reference
    python scripts/metamorphic_gate.py --record   # write the reference from this build

Three outcomes, three exit codes: **0** membership held and no decision was
lost, **1** a value moved, a proof was lost, or a new proof has a
counterexample, **2** the gate **could not run** -- no built extension, no
readable reference, or a corpus the reference does not describe. A gate that
could not run has proven nothing and must not read as one that passed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

# Three outcomes, three exit codes; see the module docstring.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_FILE = ROOT / "scripts" / "metamorphic_reference.json"


class Recording(TypedDict):
    """One build's answers to the whole corpus.

    `builds` records which spellings the frontend accepted at all, so a
    refusal that arrives or departs is visible without reading it out of the
    membership rows it silently removed.
    """

    commit: str
    builds: dict[str, bool]
    membership: dict[str, str]
    decisions: dict[str, str]


# How many failing pairs to print before stopping. A refactor that moves
# membership usually moves it for a whole class of values at once, and a
# thousand lines of the same flip says no more than a dozen.
REPORTED = 12


def _schemas() -> dict[str, object]:
    """Build the schema corpus, spelled the way a caller spells it.

    Built from the frontend's own syntax rather than from the IR, so the gate
    compares what a user can reach. A schema the frontend refuses is part of the
    corpus too -- the refusal is a verdict, and a refactor that starts or stops
    refusing has moved the boundary.
    """
    from types import GenericAlias  # noqa: PLC0415
    from typing import Annotated, Literal, TypedDict  # noqa: PLC0415

    from valgebra import (  # noqa: PLC0415
        Regex,
        anything,
        complement,
        intersection,
        nothing,
        recursive,
        union,
    )

    class Point(TypedDict):
        x: int
        y: int

    # `list[ref]` with a runtime schema inside it is a subscription a type
    # checker reads as a type expression and refuses. It is a value here, built
    # the way the interpreter builds `list[int]`, so it is spelled as one.
    def listed(element: object) -> object:
        return GenericAlias(list, (element,))

    def keyed(value: object) -> object:
        return GenericAlias(dict, (str, value))

    json_value = recursive(
        lambda ref: union(None, bool, int, float, str, listed(ref), keyed(ref))
    )

    scalars: dict[str, object] = {
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "bytes": bytes,
        "none": None,
        "anything": anything,
        "nothing": nothing,
    }
    lattice: dict[str, object] = {
        "int|str": union(int, str),
        "int|bool": union(int, bool),
        "int&str": intersection(int, str),
        "~int": complement(int),
        "~~int": complement(complement(int)),
        "~int&~str": intersection(complement(int), complement(str)),
        "(int|str)&~str": intersection(union(int, str), complement(str)),
        "int|~int": union(int, complement(int)),
        "int&~int": intersection(int, complement(int)),
    }
    literals: dict[str, object] = {
        "lit-1": Literal[1],
        "lit-true": Literal[True],
        "lit-1-2-3": Literal[1, 2, 3],
        "lit-a": Literal["a"],
        "lit-none": Literal[None],  # noqa: PYI061 -- the spelling is the subject
        "lit-1|lit-2": union(Literal[1], Literal[2]),
    }
    containers: dict[str, object] = {
        "list[int]": list[int],
        "list[list[int]]": list[list[int]],
        "list[int|str]": listed(union(int, str)),
        "list-bare": list,
        "set[int]": set[int],
        "frozenset[str]": frozenset[str],
        "tuple[int,str]": tuple[int, str],
        "tuple[int,...]": tuple[int, ...],
        "tuple-empty": tuple[()],
        "dict[str,int]": dict[str, int],
        "dict[str,list[int]]": dict[str, list[int]],
    }
    records: dict[str, object] = {
        "typeddict-point": Point,
        "record-xy": {"x": int, "y": int},
        "record-x": {"x": int},
        "record-x-str": {"x": str},
    }
    refinements: dict[str, object] = {
        "int-ge-0": Annotated[int, Ge(0)],
        "int-ge-0-le-9": Annotated[int, Ge(0), Le(9)],
        "int-ge-5-le-1": Annotated[int, Ge(5), Le(1)],
        "str-regex-a": Annotated[str, Regex("a+")],
        "list-minlen-2": Annotated[list[int], MinLen(2)],
    }
    recursive_schemas: dict[str, object] = {
        "json": json_value,
        "tree": recursive(lambda ref: union(int, listed(ref))),
    }
    return {
        **scalars,
        **lattice,
        **literals,
        **containers,
        **records,
        **refinements,
        **recursive_schemas,
    }


def _values() -> dict[str, object]:
    """Build the value corpus: one value per membership question worth asking.

    Every scalar region is represented, and so is each way a container can miss:
    the wrong element type, the wrong length, the wrong key type. The values are
    written out rather than generated, because a generated corpus that differs
    between two runs compares nothing.
    """
    return {
        "true": True,
        "false": False,
        "0": 0,
        "1": 1,
        "5": 5,
        "-1": -1,
        "0.0": 0.0,
        "1.5": 1.5,
        "nan": float("nan"),
        "inf": float("inf"),
        "empty-str": "",
        "a": "a",
        "aaa": "aaa",
        "b": "b",
        "bytes": b"a",
        "none": None,
        "empty-list": [],
        "list-1": [1],
        "list-1-2": [1, 2],
        "list-a": ["a"],
        "list-mixed": [1, "a"],
        "list-nested": [[1], [2]],
        "empty-set": set(),
        "set-1": {1},
        "frozenset-a": frozenset({"a"}),
        "empty-tuple": (),
        "tuple-1-a": (1, "a"),
        "tuple-1-2": (1, 2),
        "empty-dict": {},
        "dict-xy": {"x": 1, "y": 2},
        "dict-x": {"x": 1},
        "dict-x-str": {"x": "a"},
        "dict-xyz": {"x": 1, "y": 2, "z": 3},
        "dict-int-key": {1: 1},
        "json-object": {"a": [1, None, {"b": "c"}]},
    }


class Ge:
    """A lower bound, spelled the way `annotated_types` spells it.

    Written here rather than imported so the corpus depends on nothing but
    valgebra: a gate that cannot run without an optional dependency reports
    "could not run" on the lanes that lack it, which is the outcome that proves
    the least.
    """

    def __init__(self, ge: int) -> None:
        self.ge = ge


class Le:
    def __init__(self, le: int) -> None:
        self.le = le


class MinLen:
    def __init__(self, min_length: int) -> None:
        self.min_length = min_length


def _membership(schemas: dict[str, Any], values: dict[str, object]) -> dict[str, str]:
    """Every schema against every value: `y`, `n`, or the error the walk raised.

    A raised error is a verdict like any other. `recursion_limit` and
    `mutated_during_validation` are answers this library gives on purpose, and a
    refactor that turns one into an accept has moved membership as surely as one
    that flips a `False`.
    """
    verdicts = {}
    for schema_name, validator in schemas.items():
        if validator is None:
            continue
        for value_name, value in values.items():
            try:
                answer = "y" if validator.is_valid(value) else "n"
            except Exception as err:  # noqa: BLE001 -- the class is the verdict
                answer = f"!{type(err).__name__}"
            verdicts[f"{schema_name} @ {value_name}"] = answer
    return verdicts


def _decisions(schemas: dict[str, Any]) -> dict[str, str]:
    """Every ordered pair under subtyping, and every schema under emptiness."""
    verdicts = {}
    for name, validator in schemas.items():
        if validator is None:
            continue
        verdicts[f"empty {name}"] = "y" if validator.is_empty() else "n"
        for other_name, other in schemas.items():
            if other is None:
                continue
            answer = validator.is_subtype_of(other)
            verdicts[f"{name} <= {other_name}"] = "y" if answer else "n"
    return verdicts


def _build(specs: dict[str, object]) -> dict[str, Any]:
    """Build every corpus schema, keeping a refusal as the absence of one.

    A `None` here means the frontend declined the spelling. The membership and
    decision passes skip it, and the recorded build verdicts carry the refusal,
    so a spelling that starts or stops building is caught by the comparison
    rather than by a crash inside it.
    """
    from valgebra import Validator  # noqa: PLC0415

    def build(spec: object) -> Any:
        try:
            return Validator(spec)
        except Exception:  # noqa: BLE001 -- any refusal is the same verdict here
            return None

    return {name: build(spec) for name, spec in specs.items()}


def record() -> Recording | None:
    """Run the whole corpus through this build, or report why it could not."""
    try:
        specs = _schemas()
    except ImportError as err:
        print(f"metamorphic_gate: no built extension to measure: {err}")
        return None
    built = _build(specs)
    return Recording(
        commit=_commit(),
        builds={name: validator is not None for name, validator in built.items()},
        membership=_membership(built, _values()),
        decisions=_decisions(built),
    )


def _commit() -> str:
    """Name the commit this recording describes, or `unknown` outside a checkout."""
    try:
        result = subprocess.run(  # fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _keys_agree(name: str, reference: dict[str, str], measured: dict[str, str]) -> bool:
    """Refuse a comparison whose two sides ask different questions.

    A corpus that grew or shrank makes every "held" below meaningless: the
    relations are over the pairs both recordings carry, and a pair only one
    carries was never compared. This is the "could not run" case, not a failure
    -- the tree may be fine and the reference merely stale.
    """
    only_reference = sorted(set(reference) - set(measured))
    only_measured = sorted(set(measured) - set(reference))
    if not only_reference and not only_measured:
        return True
    print(f"metamorphic_gate: the {name} corpus does not match the reference")
    for key in only_reference[:REPORTED]:
        print(f"  reference only: {key}")
    for key in only_measured[:REPORTED]:
        print(f"  this build only: {key}")
    print("  re-record the reference (--record) if the corpus change is intended")
    return False


def moved_membership(reference: dict[str, str], measured: dict[str, str]) -> list[str]:
    """List the pairs whose verdict changed.

    Public because a gate that cannot be driven to fail from a test is not
    evidence, and this is the relation the gate exists to hold.
    """
    return [
        f"{key}: {reference[key]} -> {measured[key]}"
        for key in sorted(reference)
        if reference[key] != measured[key]
    ]


def lost_proofs(reference: dict[str, str], measured: dict[str, str]) -> list[str]:
    """List the decisions that were proven and no longer are.

    Soundness makes this one-directional. A `y` that became `n` lost a proof the
    reference had, which no refactor should do; an `n` that became `y` is the
    procedure deciding more, which every later milestone does on purpose.
    """
    return [
        f"{key}: proven by the reference, not by this build"
        for key in sorted(reference)
        if reference[key] == "y" and measured[key] == "n"
    ]


def _subtype_claim(key: str) -> tuple[str, str] | None:
    """Read `A <= B` back out of a decision key, or `None` for an emptiness row."""
    left, separator, right = key.partition(" <= ")
    return (left, right) if separator else None


def unwitnessed_widenings(
    reference: dict[str, str],
    measured: dict[str, str],
    membership: dict[str, str],
    values: dict[str, object],
) -> list[str]:
    """List the new `True`s a corpus value contradicts.

    "Only widens" is satisfied perfectly by a build that proves everything, so
    the widenings are checked rather than counted. `A <= B` is refuted by a
    value in `A` that is not in `B`, and the membership pass has already asked
    every one of those questions -- so the search is a lookup, and it uses this
    build's own answers rather than a second opinion about them.
    """
    refuted = []
    for key in sorted(reference):
        if reference[key] != "n" or measured[key] != "y":
            continue
        claim = _subtype_claim(key)
        if claim is None:
            continue
        left, right = claim
        for value_name in values:
            inside = membership.get(f"{left} @ {value_name}")
            outside = membership.get(f"{right} @ {value_name}")
            if inside == "y" and outside == "n":
                refuted.append(f"{key}: {value_name} is in {left} and not in {right}")
                break
    return refuted


def _report(title: str, failures: list[str]) -> None:
    print(f"{title}: {len(failures)}")
    for line in failures[:REPORTED]:
        print(f"  {line}")
    if len(failures) > REPORTED:
        print(f"  ... and {len(failures) - REPORTED} more")


def main(argv: list[str]) -> int:
    measured = record()
    if measured is None:
        return EXIT_CANNOT_RUN
    if "--record" in argv:
        REFERENCE_FILE.write_text(
            json.dumps(measured, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"metamorphic_gate: recorded at {measured['commit']}")
        return EXIT_OK

    try:
        reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
        reference_membership = reference["membership"]
        reference_decisions = reference["decisions"]
    except (OSError, ValueError, KeyError) as err:
        print(f"metamorphic_gate: cannot read the reference: {err}")
        return EXIT_CANNOT_RUN

    membership = measured["membership"]
    decisions = measured["decisions"]
    agreed = _keys_agree("membership", reference_membership, membership)
    agreed = _keys_agree("decision", reference_decisions, decisions) and agreed
    if not agreed:
        return EXIT_CANNOT_RUN

    moved = moved_membership(reference_membership, membership)
    lost = lost_proofs(reference_decisions, decisions)
    refuted = unwitnessed_widenings(
        reference_decisions, decisions, membership, _values()
    )
    widened = sum(
        1
        for key in reference_decisions
        if reference_decisions[key] == "n" and decisions[key] == "y"
    )

    print(f"reference: {reference['commit']}")
    print(f"corpus:    {len(membership)} value pairs, {len(decisions)} decisions")
    _report("membership moved", moved)
    _report("proofs lost", lost)
    _report("widenings refuted by a value", refuted)
    print(f"widenings: {widened}")
    if moved or lost or refuted:
        return EXIT_FAIL
    print("OK: membership held and no proof was lost.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
