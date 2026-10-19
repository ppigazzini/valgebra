"""Every design the architecture page refuses names the test that holds it.

A refusal written down without a holder is a verdict, and a verdict is what the
next proposal argues with. The section of `docs/dev/00-architecture.md` titled
"Refused, and what each one stands on" gives each refused proposal the rule the
tree keeps in its place and the figure or reason the rule stands on, and ends
each entry with the tests that hold it: a test that fails if the proposal
lands, or, where the rule stands on a figure, the test holding the figure.

This file reads the section and holds it both ways. Every entry names a holder
that exists, as a test function in the file it is named in; and every holder
defined here is named by an entry, so a check that outlives its refusal fails.
The holders a refusal has no other test for live below.

What it cannot judge is the reason. Whether a measurement still says what the
entry says it does is the reader's, and the entry names the gate or the page
that reproduces it so that the reader can ask.

LEDGER: every refusal on the architecture page names a test that holds it
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pytest

from _reason import is_a_reason
from _toml import load
from _tree import ROOT, WORKFLOW, load_script, runs

# This file reads the tree rather than the library: the page, the manifests and
# the module graph. It exercises no schema.
pytestmark = pytest.mark.repository

PAGE = ROOT / "docs" / "dev" / "00-architecture.md"
HEADING = "## Refused, and what each one stands on"

lint = load_script("docs_lint")

#: An entry of the section: a paragraph led by the proposal in bold.
_ENTRY = re.compile(r"^\*\*(.+?)\*\*(.*?)(?=^\*\*|\Z)", re.MULTILINE | re.DOTALL)
#: The sentence naming an entry's holders, which closes it.
_HELD_BY = re.compile(r"Held\s+by\s+(.*?)\.(?:\s|$)", re.DOTALL)
#: A backticked name in that sentence.
_NAME = re.compile(r"`([^`]+)`")


def _section() -> str:
    text = PAGE.read_text(encoding="utf-8")
    assert HEADING in text, f"{PAGE.name} has no section {HEADING!r}"
    return text.split(HEADING, 1)[1].split("\n## ", 1)[0]


def _entries() -> list[tuple[str, list[tuple[str, str]]]]:
    """Read each entry's proposal and the (test, file) pairs it is held by.

    A holder sentence lists names and the file after them: ``Held by `a` and
    `b` in `tests/x.py`, and `c` in `tests/y.py`.`` -- each name belongs to
    the next file named.
    """
    entries = []
    for proposal, body in _ENTRY.findall(_section()):
        held = _HELD_BY.search(body)
        pairs: list[tuple[str, str]] = []
        pending: list[str] = []
        for name in _NAME.findall(held.group(1) if held else ""):
            if name.endswith((".py", ".rs")):
                pairs += [(test, name) for test in pending]
                pending = []
            else:
                pending.append(name)
        entries.append((proposal.strip(), pairs))
    return entries


def _defines(path: Path, name: str) -> bool:
    keyword = "def" if path.suffix == ".py" else "fn"
    return (
        re.search(
            rf"^\s*{keyword} {re.escape(name)}\(",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        is not None
    )


def test_every_refusal_names_a_holder_that_exists() -> None:
    entries = _entries()
    # The section is the detector: one read as empty would pass naming nothing.
    assert len(entries) >= 10, f"the section reads only {len(entries)} entries"
    unheld = [proposal for proposal, pairs in entries if not pairs]
    assert not unheld, f"entries that name no holder: {unheld}"
    missing = [
        f"{proposal}: {test} in {file}"
        for proposal, pairs in entries
        for test, file in pairs
        if not (ROOT / file).is_file() or not _defines(ROOT / file, test)
    ]
    assert not missing, f"holders that are not tests in the file named: {missing}"


def test_every_holder_here_is_named_by_a_refusal() -> None:
    here = Path(__file__).name
    named = {
        test for _, pairs in _entries() for test, file in pairs if file.endswith(here)
    }
    ledger = {
        "test_every_refusal_names_a_holder_that_exists",
        "test_every_holder_here_is_named_by_a_refusal",
        "test_every_admitted_cycle_says_what_crosses_it",
    }
    defined = set(
        re.findall(
            r"^def (test_\w+)\(",
            (ROOT / "tests" / here).read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    orphans = sorted(defined - ledger - named)
    assert not orphans, f"holders no refusal on {PAGE.name} names: {orphans}"


# --- the module cycles -------------------------------------------------------


def _production() -> list[str]:
    """Give every production source file, spelled from the root."""
    sources = [
        path
        for path in lint.tracked_files("*.rs")
        if re.match(r"crates/[^/]+/src/", path.relative_to(ROOT).as_posix())
    ]
    test_only = lint.test_module_files(sources)
    return sorted(
        where
        for where in (source.relative_to(ROOT).as_posix() for source in sources)
        if where not in test_only
    )


def _code(path: str) -> str:
    """Read what a source file compiles into the library, comments left out.

    A doc link or a comment naming a module is no dependency, and a test
    module's imports are not the library's.
    """
    text = lint.before_the_test_module((ROOT / path).read_text(encoding="utf-8"))
    return re.sub(r"//[^\n]*", "", text)


def _module(path: str) -> tuple[str, tuple[str, ...]]:
    """Give the crate and the module path a source file is."""
    parts = PurePosixPath(path).parts
    names = list(parts[3:])
    last = names.pop()[: -len(".rs")]
    return parts[1], tuple(names + ([] if last in {"lib", "mod"} else [last]))


def _expand(tree: str) -> list[str]:
    """Expand a use tree into the paths at its leaves: `a::{b, c::{d}}`."""
    tree = re.sub(r"\s+as\s+\w+", "", tree)
    tree = re.sub(r"\s+", "", tree)
    paths: list[str] = []
    depth, start, parts = 0, 0, []
    for index, char in enumerate(tree):
        depth += {"{": 1, "}": -1}.get(char, 0)
        if char == "," and depth == 0:
            parts.append(tree[start:index])
            start = index + 1
    parts.append(tree[start:])
    for part in filter(None, parts):
        if "{" in part:
            head, rest = part.split("{", 1)
            paths += [head + leaf for leaf in _expand(rest[:-1])]
        else:
            paths.append(part)
    return paths


_USE = re.compile(r"\buse\s+([^;]+);")
_QUALIFIED = re.compile(r"\b(?:crate|super)(?:::\w+)+")


def _edges() -> dict[str, set[str]]:
    """Read which production file names which, through `crate::` and `super::`.

    A path is taken to the longest module it names, so `crate::check::ctx::Ctx`
    is an edge to `check/ctx.rs` and a re-export read from the crate root is an
    edge to `lib.rs`. Comments and the test module are not read: a doc link is
    no dependency, and a test's imports are not the library's.
    """
    files = _production()
    modules = {_module(path): path for path in files}
    edges: dict[str, set[str]] = {}
    for path in files:
        crate, here = _module(path)
        text = _code(path)
        named = [leaf for tree in _USE.findall(text) for leaf in _expand(tree)]
        named += _QUALIFIED.findall(_USE.sub("", text))
        reached: set[str] = set()
        for leaf in named:
            segments = leaf.split("::")
            if segments[0] == "crate":
                at, segments = (), segments[1:]
            elif segments[0] == "super":
                at = here[:-1]
                segments = segments[1:]
                while segments and segments[0] == "super":
                    at, segments = at[:-1], segments[1:]
            else:
                continue
            full = at + tuple(segments)
            target = next(
                modules[crate, full[:length]]
                for length in range(len(full), -1, -1)
                if (crate, full[:length]) in modules
            )
            if target != path:
                reached.add(target)
        edges[path] = reached
    return edges


def _reached(edges: dict[str, set[str]], start: str) -> set[str]:
    """Give every file `start` reaches by one edge or more."""
    seen: set[str] = set()
    pending = list(edges.get(start, ()))
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending += edges.get(node, ())
    return seen


def _cycles(edges: dict[str, set[str]]) -> set[frozenset[str]]:
    """Give each set of more than one file whose members all reach each other."""
    reached = {node: _reached(edges, node) for node in edges}
    return {
        frozenset({node} | {other for other in reached[node] if node in reached[other]})
        for node in edges
        if node in reached[node]
    }


CORE = "crates/valgebra-core/src/"
PY = "crates/valgebra-py/src/"

#: Every set of modules that import each other, and what crosses between them.
#: A cycle costs nothing the build can see -- the crate is the compilation unit
#: and the wheel one `.so` -- so the rule is not that there are none, but that
#: each is named, and a new one is a decision rather than a drift.
ADMITTED_CYCLES: dict[frozenset[str], str] = {
    frozenset(
        CORE + name
        for name in (
            "ir.rs",
            "oracle.rs",
            "descr/lower.rs",
            "descr/budget.rs",
            "descr/maps.rs",
            "descr/polar.rs",
            "descr/symbolic.rs",
            "descr/values.rs",
        )
    ): (
        "the constructors in ir.rs fold the complement law through oracle.rs, "
        "which reads the Schema it is asked about and the Constants a lowering "
        "is given; a lowering reads the term and builds the map atoms, whose "
        "values, polarity union and sequence automaton spend the Budget, which "
        "reads the WORK allowance descr/lower.rs defines"
    ),
    frozenset(PY + name for name in ("build.rs", "validator.rs", "render.rs")): (
        "a compiled validator is itself a schema description, so the frontend "
        "reads Validator and the validator compiles through the frontend; it "
        "renders through render.rs, which asks the frontend's declares_fields"
    ),
    frozenset(PY + name for name in ("check/walk.rs", "check/violation.rs")): (
        "the walk calls the violation builders, which call back the walk's "
        "record_if_fatal, a function each way"
    ),
    frozenset(PY + name for name in ("check/walk/record.rs", "check/walk/scalar.rs")): (
        "a record's walk reads a field through scalar_member and "
        "admitted_quietly, and a parsed container's length bound counts its "
        "keys through the record walk's distinct_keys, a function each way"
    ),
    frozenset(PY + name for name in ("build/classes.rs", "build/generics.rs")): (
        "what a class declares asks the typing introspection's "
        "is_field_qualifier, which asks field_name and is_truthy_attr back, a "
        "function each way"
    ),
}


def test_the_module_cycles_are_the_ones_the_page_admits() -> None:
    found = _cycles(_edges())
    new = sorted(sorted(cycle) for cycle in found - ADMITTED_CYCLES.keys())
    gone = sorted(sorted(cycle) for cycle in ADMITTED_CYCLES.keys() - found)
    assert not new, (
        f"modules that import each other and are not admitted: {new}. Name the "
        "cycle and what crosses it in ADMITTED_CYCLES and on the architecture "
        "page, or break it."
    )
    assert not gone, f"admitted cycles the tree does not hold: {gone}"


def test_every_admitted_cycle_says_what_crosses_it() -> None:
    vague = [
        sorted(cycle) for cycle, why in ADMITTED_CYCLES.items() if not is_a_reason(why)
    ]
    assert not vague, f"admitted cycles with no reason: {vague}"


# --- the walk dispatches by match ---------------------------------------------


def test_the_walk_dispatches_by_match() -> None:
    walk = sorted(
        path
        for path in _production()
        if path.startswith((PY + "check.rs", PY + "check/"))
    )
    assert walk, "the walk's production files read as none"
    boxed = [path for path in walk if re.search(r"\bdyn\s", _code(path))]
    assert not boxed, f"a trait object in the walk: {boxed}"
    lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    assert 'name = "enum_dispatch"' not in lock, "enum_dispatch is in Cargo.lock"


# --- no build targets the stable ABI -------------------------------------------


def test_no_build_targets_the_stable_abi() -> None:
    manifests = [ROOT / "Cargo.toml", *sorted((ROOT / "crates").glob("*/Cargo.toml"))]
    for manifest in manifests:
        text = manifest.read_text(encoding="utf-8")
        assert not re.search(r"abi3", text), f"{manifest.relative_to(ROOT)} names abi3"
    release = ROOT / ".github" / "workflows" / "release.yml"
    assert not re.search(r"\babi3", "\n".join(runs(release))), (
        "a release step names abi3"
    )
    script = (ROOT / "crates" / "valgebra-py" / "build.rs").read_text(encoding="utf-8")
    assert "pyo3_build_config::use_pyo3_cfgs()" in script, (
        "the binding's build script no longer re-emits the interpreter's flags, "
        "which is what lets the walk choose its readings per interpreter"
    )


# --- the deciders read the normal form construction builds ----------------------


def test_the_deciders_never_normalise_a_query() -> None:
    deciders = sorted(
        path
        for path in _production()
        if path.startswith((CORE + "decision", CORE + "descr/"))
    )
    assert len(deciders) > 10, f"the deciders read as {deciders}"
    normalising = [path for path in deciders if re.search(r"\bsimplify\b", _code(path))]
    assert not normalising, f"a decider names a normalizing pass: {normalising}"


# --- the gates are scripts ------------------------------------------------------


def test_the_workspace_is_the_two_crates() -> None:
    root = load(ROOT / "Cargo.toml")
    assert sorted(root["workspace"]["members"]) == [
        "crates/valgebra-core",
        "crates/valgebra-py",
    ]
    config = load(ROOT / ".cargo" / "config.toml")
    assert "alias" not in config, "a cargo alias names a task the scripts do not run"


# --- one lint rule for the workspace --------------------------------------------


def test_every_crate_takes_the_workspace_lints() -> None:
    root = load(ROOT / "Cargo.toml")
    assert "pedantic" in root["workspace"]["lints"]["clippy"]
    for member in root["workspace"]["members"]:
        manifest = load(ROOT / member / "Cargo.toml")
        assert manifest.get("lints") == {"workspace": True}, (
            f"{member} takes lints of its own, which is a second rule"
        )
    # A command, with its continuation lines joined: the restriction step
    # spreads the lints it denies over three.
    commands = [
        line.strip()
        for shell in runs(WORKFLOW)
        for line in shell.replace("\\\n", " ").splitlines()
        if line.strip().startswith("cargo clippy")
    ]
    assert any("-D warnings" in command for command in commands), (
        "no clippy step denies every warning"
    )
    # Every clippy step denies: all warnings, or the named restriction lints
    # the library targets are held to. A step allowing a lint is a second rule.
    lax = [
        command
        for command in commands
        if not re.search(r"-D (?:warnings|clippy::)", command) or " -A " in command
    ]
    assert not lax, f"a clippy step that is not the workspace's rule: {lax}"
