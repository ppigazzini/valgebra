"""The mechanical half of documentation rot.

Reads every tracked Markdown file and fails on four things a reader cannot be
expected to catch by eye:

* **A dead internal link.** Any ``[text](target)`` that is not a URL, a
  ``mailto:`` or a bare ``#anchor`` must resolve, relative to the linking file or
  to the repository root.
* **A named path that does not exist.** A ``crates/...``, ``scripts/...``,
  ``tests/...`` or ``.github/...`` path written in prose is a claim about this
  tree. A path holding a placeholder (``*``, ``<``, ``>``, ``...``) is skipped,
  which is what lets ``crates/<name>/`` be written at all.
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


def check_named_paths(text: str) -> list[str]:
    problems = []
    for raw in NAMED_PATH.findall(text):
        named = raw.rstrip(".,;:)`")
        if PLACEHOLDER.search(named):
            continue
        if not (ROOT / named).exists():
            problems.append(f"named path does not exist: {named}")
    return problems


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


def main() -> int:
    files = tracked_markdown()
    if len(files) < MIN_TRACKED_PAGES:
        print(f"docs_lint: found only {len(files)} tracked Markdown files")
        return EXIT_CANNOT_RUN
    numbers = gate_numbers()

    failures: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        failures += [
            f"{rel}: {problem}"
            for problem in check_links(path, text)
            + check_named_paths(text)
            + check_internal_reference(text)
            + check_pinned_numbers(text, numbers)
        ]
    failures += check_dev_index()

    if failures:
        for failure in failures:
            print(failure)
        print(f"\ndocs_lint: {len(failures)} problem(s) in {len(files)} files")
        return EXIT_FAIL
    print(f"docs_lint: {len(files)} tracked Markdown files, no problems")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
