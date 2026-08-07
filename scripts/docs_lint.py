"""The mechanical half of documentation rot.

Reads every tracked Markdown file and fails on four things a reader cannot be
expected to catch by eye:

* **A dead internal link.** Any ``[text](target)`` that is not a URL, a
  ``mailto:`` or a bare ``#anchor`` must resolve, relative to the linking file or
  to the repository root.
* **A named path that does not exist.** A ``crates/...``, ``scripts/...``,
  ``tests/...`` or ``.github/...`` path written in prose is a claim about this
  tree. A path holding a placeholder (``*``, ``<``, ``>``, ``...``) is skipped,
  which is what lets ``crates/<name>/`` be written at all. A path ``.gitignore``
  names is skipped too: the repository decided not to carry it, and a doc naming
  one is usually documenting the tool that writes it. **The exemption is asked of
  git, not of the filesystem**, so the answer does not depend on whether the
  reader happens to have run that tool -- a check whose verdict changes with
  local build output is measuring the machine rather than the tree.
* **A reference into the internal surface.** ``__DEV/`` is gitignored, so a
  tracked file naming it leaves a reference that dangles for every reader but its
  author. The path check above cannot see this class: it exempts what
  ``.gitignore`` names, on the grounds that an ignored path is usually the output
  of the tool being documented.
* **A gate's number quoted in prose.** The instruction budgets live in
  ``scripts/perf_budget.json`` and move when the budget is re-recorded. A figure
  copied into a page is stale the next time it moves, and a stale number is worse
  than an absent one -- it tells a reader to hold the wrong invariant.

Three classes stay out of its reach, and they are the common ones: a real symbol
attributed to the wrong file, a list with the wrong count or order, and a
behaviour described as absent from a build that has it. **It cannot tell you a
sentence is false.** It buys the mechanical half so review can spend its
attention on the half that needs a reader.

Usage:
    python scripts/docs_lint.py

Three outcomes, three exit codes: 0 clean, 1 a claim is wrong, 2 the check could
not run.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent
INTERNAL = "__DEV"

# `[text](target)`, excluding image embeds, which carry the same rule but are
# matched by the same expression.
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# A path claim: one of the tree's real roots followed by a path component.
NAMED_PATH = re.compile(
    r"(?<![\w/.-])((?:crates|scripts|tests|benches|fuzz|python|docs|\.github)/[\w./-]+)"
)
PLACEHOLDER = re.compile(r"[*<>]|\.\.\.")
# A tree with fewer pages than this has not been read; the glob is the detector.
MIN_TRACKED_PAGES = 5


def tracked_markdown() -> list[Path]:
    """Every Markdown file git tracks. Untracked notes are not shipped prose."""
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"docs_lint: git ls-files failed: {result.stderr.strip()}")
        raise SystemExit(EXIT_CANNOT_RUN)
    return [ROOT / line for line in result.stdout.split() if line]


def check_links(path: Path, text: str) -> list[str]:
    problems = []
    for raw in LINK.findall(text):
        target = raw.split(" ")[0].strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if not (path.parent / target.split("#")[0]).resolve().exists():
            problems.append(f"dead link {target!r}")
    return problems


def ignored_paths(candidates: list[str]) -> set[str]:
    """Ask git which of these `.gitignore` names.

    Asked of git rather than of the filesystem on purpose: a generated path is
    absent from a fresh clone and present on a machine that has run the tool, and
    a check whose verdict differs between the two is measuring the machine.
    """
    if not candidates:
        return set()
    # NUL-separated, in binary mode. Text mode translates "\n" to the platform
    # line ending on write, so on Windows git received every path with a trailing
    # carriage return, matched none of them, and the check reported paths as
    # missing that `.gitignore` names. Bytes and `-z` remove both the newline
    # translation and any question about a path containing whitespace.
    result = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        cwd=ROOT,
        input=b"\0".join(path.encode() for path in candidates),
        capture_output=True,
        check=False,
    )
    # Exit 0 = some ignored, 1 = none ignored, anything else = it could not run.
    if result.returncode not in (0, 1):
        print(f"docs_lint: git check-ignore failed: {result.stderr.decode().strip()}")
        raise SystemExit(EXIT_CANNOT_RUN)
    return {chunk.decode() for chunk in result.stdout.split(b"\0") if chunk.strip()}


def check_named_paths(text: str, ignored: set[str] | None = None) -> list[str]:
    named = [
        raw.rstrip(".,;:)`")
        for raw in NAMED_PATH.findall(text)
        if not PLACEHOLDER.search(raw)
    ]
    if ignored is None:
        ignored = ignored_paths(named)
    return [
        f"named path does not exist: {path}"
        for path in named
        if path not in ignored and not (ROOT / path).exists()
    ]


def check_internal_reference(text: str) -> list[str]:
    if INTERNAL in text:
        return [
            (
                f"names the internal surface ({INTERNAL}/), which is gitignored: "
                "the reference dangles for every reader but its author"
            )
        ]
    return []


def gate_numbers() -> list[str]:
    """Collect the figures a gate owns, in the forms prose would write them."""
    budget = json.loads(
        (ROOT / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
    )
    numbers = []
    for key in ("core_workload_irefs", "binding_workload_irefs"):
        value = int(budget[key])
        numbers += [f"{value:,}", str(value)]
    return numbers


def check_pinned_numbers(text: str, numbers: list[str]) -> list[str]:
    return [
        f"quotes a number the perf budget owns ({n}); name the file instead"
        for n in numbers
        if n in text
    ]


def check_dev_index() -> list[str]:
    """Hold the developer set's index to the set, in both directions."""
    dev = ROOT / "docs" / "dev"
    if not dev.is_dir():
        return []
    index = dev / "README.md"
    if not index.exists():
        return ["docs/dev/ has no README.md index"]
    # An index row names a sibling page by bare filename. A link that walks out
    # of the directory is prose, not a row, and is checked by the link rule.
    listed = {
        target
        for target in LINK.findall(index.read_text(encoding="utf-8"))
        if "/" not in target and target.endswith(".md")
    }
    pages = {p.name for p in dev.glob("*.md")} - {"README.md"}
    problems = [
        f"docs/dev/{missing} is in no row of docs/dev/README.md"
        for missing in sorted(pages - listed)
    ]
    problems += [
        f"docs/dev/README.md lists {stale}, which does not exist"
        for stale in sorted(listed - pages)
    ]
    # The index is the detector; an empty set would pass having checked nothing.
    if not pages:
        problems.append("docs/dev/ holds no pages")
    return problems


def check_ledger_table() -> list[str]:
    """Hold the table of ledgers to the ledgers, in both directions.

    Every list in this repository that could rot is held to the tree both ways
    -- except, until this rule, the list *of* those lists. It carried a spelled
    count in two places and neither was held to anything, so both drifted: the
    table said five, the glossary said four, and the tree had six.

    A ledger declares itself with a ``LEDGER:`` marker in its own docstring, so
    the universe is read from the tests rather than restated here. The marker is
    explicit rather than inferred from prose: the phrasing varies between these
    files ("held in both directions", "held to the tree in both directions"), and
    a universe built by matching a sentence silently omits whichever file worded
    it differently -- which is the failure this rule exists to stop.
    """
    page = ROOT / "docs" / "dev" / "08-testing.md"
    tests = ROOT / "tests"
    if not page.exists() or not tests.is_dir():
        return []
    declared = {
        path.name
        for path in sorted(tests.glob("test_*.py"))
        if "LEDGER:" in path.read_text(encoding="utf-8")
    }
    text = page.read_text(encoding="utf-8")
    problems = [
        f"docs/dev/08-testing.md: tests/{name} is a ledger with no row"
        for name in sorted(declared)
        if name not in text
    ]
    listed = set(re.findall(r"`tests/(test_\w+\.py)`", text))
    problems += [
        f"docs/dev/08-testing.md: names tests/{name}, which does not exist"
        for name in sorted(listed)
        if not (tests / name).exists()
    ]
    # The count is prose beside the table, so it rots on its own schedule.
    spelled = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    want = spelled.get(len(declared))
    for doc in (page, ROOT / "docs" / "dev" / "12-glossary.md"):
        if not doc.exists():
            continue
        for wrong, word in spelled.items():
            if wrong == len(declared):
                continue
            spelling = rf"\b{word}\b of them|There are {word}\b"
            if re.search(spelling, doc.read_text(encoding="utf-8"), re.IGNORECASE):
                problems.append(
                    f"{doc.relative_to(ROOT)}: says {word} ledgers; there are "
                    f"{len(declared)} ({want})"
                )
    # The scan is the detector: no ledgers at all would pass having read nothing.
    if not declared:
        problems.append("no test declares itself a ledger")
    return problems


def main() -> int:
    files = tracked_markdown()
    if len(files) < MIN_TRACKED_PAGES:
        print(f"docs_lint: found only {len(files)} tracked Markdown files")
        return EXIT_CANNOT_RUN
    numbers = gate_numbers()
    # One `git check-ignore` for the whole set rather than one per file.
    every_named = [
        raw.rstrip(".,;:)`")
        for path in files
        for raw in NAMED_PATH.findall(path.read_text(encoding="utf-8"))
        if not PLACEHOLDER.search(raw)
    ]
    ignored = ignored_paths(every_named)

    failures: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        failures += [
            f"{rel}: {problem}"
            for problem in check_links(path, text)
            + check_named_paths(text, ignored)
            + check_internal_reference(text)
            + check_pinned_numbers(text, numbers)
        ]
    failures += check_dev_index()
    failures += check_ledger_table()

    if failures:
        for failure in failures:
            print(failure)
        print(f"\ndocs_lint: {len(failures)} problem(s) in {len(files)} files")
        return EXIT_FAIL
    print(f"docs_lint: {len(files)} tracked Markdown files, no problems")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
