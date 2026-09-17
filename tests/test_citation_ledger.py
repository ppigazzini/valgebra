"""A numbered result the theory page cites is one the shelf can be opened at.

`docs/dev/10-theory.md` does not argue the theory; it records which result each
design decision rests on, so a reader can check the decision against a page of a
paper rather than against a recollection. That only works while the citation
resolves: "Lemma 6.5" is worth writing because somebody can open the paper at
Lemma 6.5, and worth nothing once nobody can say which paper.

Two lists make it resolvable, and neither held the other. The page names works
and cites numbered results of them. The shelf's own table names a work per
surface and the file that holds it. A work cited on the page and missing from
the table is a citation with no paper behind it; a file on the shelf with no
row is a paper nothing says the use of.

So both are read here. A numbered result is attributed to the **nearest work
named at or above it**, which is how a reader attributes it, and that work has
to be one the table names and a file the shelf holds.

The shelf is not redistributed, so a clone has none: these rows skip where it is
absent rather than failing, and say so. What does not skip is the direction that
lives entirely in the tree -- every citation attributable to some work at all --
because a numbered result under no work is unresolvable to every reader, with a
shelf or without one.

LEDGER: every numbered result the theory page cites names a work on the shelf
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
THEORY = ROOT / "docs" / "dev" / "10-theory.md"

#: Where the shelf lives. Assembled from its pieces rather than written, for the
#: reason `tests/test_commit_messages.py` assembles the same name: a tracked file
#: that spells it is the reference `scripts/docs_lint.py` exists to refuse, since
#: the directory reaches nobody who clones this repository.
SHELF = ROOT / ("__" + "DEV") / "papers"

#: A numbered result of a paper: the named kinds, and a section number. Both are
#: things a reader opens a paper *at*, which is what makes them citations rather
#: than prose.
_CITATION = re.compile(
    r"\b(?:Theorem|Definition|Lemma|Proposition|Corollary)\s+\d+(?:\.\d+)*"
    r"|§\d+(?:\.\d+)*"
)

#: A row of the shelf's table: the surface, the work, and the file holding it.
_ROW = re.compile(r"^\|(?P<surface>[^|]+)\|(?P<work>[^|]+)\|(?P<file>[^|]+)\|$")

#: The authors of a work, which is the table's owner cell up to the title. The
#: page names a work the same way, so this is what ties the two together.
_AUTHORS = re.compile(r"^(.*?),\s*\*")

#: A work as the *page* names it: a bold run opening with an author list. The
#: page opens a claim in bold too, so the run has to be told from a sentence,
#: and what tells them apart is that an author list is names and connectives
#: only -- "Frisch, Castagna & Benzaken" against "Subtyping is inclusion".
_NAMED = re.compile(r"\*\*([A-Z][A-Za-z\u2019'&.\- ]+?),")

#: The words an author list may carry that are not a name.
_CONNECTIVES = frozenset({"&", "and", "et", "al."})


def _table() -> list[tuple[str, str]]:
    """Read the shelf's table as `(authors, file)` pairs.

    Pairs rather than a mapping from authors: the shelf holds two of Castagna's
    papers, and a dict keyed by the author would drop one of them -- which reads
    as a file with no row, since the row is there and the parse lost it.
    """
    shelf = SHELF / "README.md"
    rows: list[tuple[str, str]] = []
    for line in shelf.read_text(encoding="utf-8").splitlines():
        row = _ROW.match(line.strip())
        if row is None:
            continue
        work, name = row["work"].strip(), row["file"].strip().strip("`")
        if not name.endswith(".pdf"):
            continue
        authors = _AUTHORS.match(work)
        if authors is None:
            continue
        rows.append((authors[1].strip(), name))
    return rows


def _files_by_author(rows: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Group the table by its author list, which is what the page names."""
    found: dict[str, list[str]] = {}
    for authors, name in rows:
        found.setdefault(authors, []).append(name)
    return found


def _works_the_page_names() -> list[str]:
    """Give the author lists the theory page itself writes.

    Read from the page rather than from the shelf, because attribution is what a
    reader does with the page in hand and a clone has no shelf to consult. The
    shelf then holds these names in its own direction below.
    """
    found: list[str] = []
    for run in _NAMED.findall(THEORY.read_text(encoding="utf-8")):
        words = run.split()
        if words and all(word in _CONNECTIVES or word[:1].isupper() for word in words):
            found.append(run.strip())
    return sorted(set(found))


def _citations(authors: list[str]) -> list[tuple[str, str | None]]:
    """Each numbered result on the page, with the work it is attributed to.

    Attributed to the nearest name at or above it, which is how the page reads:
    a paragraph opening with a work's authors carries every result it cites, and
    one that opens with none belongs to the work above it.
    """
    text = THEORY.read_text(encoding="utf-8")
    marks: list[tuple[int, str]] = [
        (match.start(), name)
        for name in authors
        for match in re.finditer(re.escape(name), text)
    ]
    marks.sort()
    found: list[tuple[str, str | None]] = []
    for cited in _CITATION.finditer(text):
        above = [name for at, name in marks if at < cited.start()]
        found.append((cited.group(0), above[-1] if above else None))
    return found


def _on_the_shelf() -> list[str]:
    return sorted(path.name for path in SHELF.glob("*.pdf"))


_ABSENT = "the shelf is not redistributed, so a clone has none to read"


def test_the_page_cites_numbered_results() -> None:
    """The parse is the detector, so it is shown to have read something.

    A ledger over an empty universe passes having checked nothing, which is the
    failure `tests/test_ledger_plants.py` rules out for the rest of them.
    """
    text = THEORY.read_text(encoding="utf-8")
    cited = _CITATION.findall(text)
    assert len(cited) >= 4, cited
    # And it reads both shapes: a named result and a section number.
    assert any(hit.startswith("§") for hit in cited), cited
    assert any(not hit.startswith("§") for hit in cited), cited


def test_every_numbered_result_is_attributed_to_a_work() -> None:
    """A result under no work is one no reader can resolve, shelf or no shelf.

    The direction that holds without the papers, which is why it does not skip:
    the page is the tracked record of what the design rests on, and "Lemma 6.5"
    with nothing above it saying whose lemma is a sentence that cannot be
    checked by anybody.
    """
    named = _works_the_page_names()
    # The scan is the detector: no names is no attribution for anything.
    assert len(named) >= 5, named
    orphaned = [cited for cited, work in _citations(named) if work is None]
    assert not orphaned, (
        f"numbered results with no work named above them: {orphaned}. Name the "
        "authors in the paragraph, or move the citation under the one it is of."
    )


def test_every_cited_work_is_a_paper_the_shelf_holds() -> None:
    """The work a citation resolves to is a file somebody can open."""
    if not SHELF.is_dir():
        pytest.skip(_ABSENT)
    by_author, held = _files_by_author(_table()), _on_the_shelf()
    named = [work for work in _works_the_page_names() if work in by_author]
    cited = {work for _, work in _citations(named) if work is not None}
    missing = sorted(
        f"{work} -> {name}"
        for work in cited
        for name in by_author[work]
        if name not in held
    )
    assert not missing, f"works the page cites whose file the shelf lacks: {missing}"


def test_the_table_and_the_shelf_name_the_same_papers() -> None:
    """Both directions over the shelf itself, so neither list drifts alone.

    A row whose file is gone is a citation that resolves to nothing; a file with
    no row is a paper the table cannot say the use of, which is the whole
    selection rule the shelf is chosen by.
    """
    if not SHELF.is_dir():
        pytest.skip(_ABSENT)
    table, held = _table(), _on_the_shelf()
    assert len(table) >= 10, sorted(table)
    rowed = {name for _, name in table}
    absent = sorted(name for name in rowed if name not in held)
    assert not absent, f"rows naming a file the shelf does not hold: {absent}"
    unrowed = sorted(set(held) - rowed)
    assert not unrowed, (
        f"papers on the shelf with no row saying what they are for: {unrowed}"
    )
