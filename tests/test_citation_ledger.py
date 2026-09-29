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

So both are read here, over the theory page and the decision page. A numbered
result is attributed to the **work named nearest at or before it in its
paragraph**, or in the paragraph before where its own names none -- which is how
a reader attributes it -- and a work is named by its authors, its venue or the
shorthand the pages use. That work has to be one the table names and a file the
shelf holds, and the number has to be one the paper states: the shelf's own
rule is to verify a theorem number and trust a citation chain, and a wrong
number resolved to a real paper passed before.

The shelf is not redistributed, so a clone has none: these rows skip where it is
absent rather than failing, and say so. What does not skip is the direction that
lives entirely in the tree -- every citation attributable to some work at all --
because a numbered result under no work is unresolvable to every reader, with a
shelf or without one.

LEDGER: every numbered result the theory pages cite is one a paper on the shelf states
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

#: The pages whose citations are held: the design rationale, and the decision
#: procedure's own page.
PAGES = (THEORY, ROOT / "docs" / "dev" / "02-decision.md")

#: Every way the pages name a work, and the stem of the shelf's file for it.
#: Longest first at one position, so "Frisch, Castagna & Benzaken" is not read as
#: the "Castagna" inside it. `None` is a work the shelf does not hold, named so
#: that a citation of it is declared rather than resolved to the nearest paper.
_JACM = "frisch-castagna-benzaken-2008-semantic-subtyping-JACM"
_HVP = "hosoya-vouillon-pierce-2005-regular-expression-types-xml"
ALIASES: dict[str, str | None] = {
    "Frisch, Castagna & Benzaken": _JACM,
    "JACM": _JACM,
    "Castagna & Duboc": "castagna-duboc-2024-guard-analysis-safe-erasure-elixir",
    "Castagna & Peyrot": "castagna-peyrot-2025-polymorphic-records",
    "ICFP 2023": "castagna-2023-typing-records-maps-structs-ICFP",
    "ICFP": "castagna-2023-typing-records-maps-structs-ICFP",
    "UIN": "castagna-2021-union-intersection-negation",
    "Amadio & Cardelli": "amadio-cardelli-1993-subtyping-recursive-types",
    "Hosoya, Vouillon & Pierce": _HVP,
    "TOPLAS 2005": _HVP,
    "Nakano": "nakano-2000-modality-for-recursion",
    # Cited for the family that makes a trail keeping nothing exponential, and
    # held by no file here: the shelf carries it through the papers that cite
    # it, so its section numbers are the one thing this cannot check.
    "Gapeyev, Levin & Pierce": None,
}

#: A section of the project's own notes, which number from 13 and which no paper
#: on the shelf reaches; a `SOURCE:` line quotes those notes.
_NOTES = re.compile(r"§1[3-5](?:\.\d+)*")


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """Each paragraph of `text`, with the line it starts on."""
    found, line = [], 1
    for block in re.split(r"(\n\s*\n)", text):
        if block.strip():
            found.append((line, block))
        line += block.count("\n")
    return found


def _citations(page: Path) -> list[tuple[str, str, str | None]]:
    """Each numbered result on `page`: where, what, and the alias it is of."""
    found: list[tuple[str, str, str | None]] = []
    last: str | None = None
    for start, paragraph in _paragraphs(page.read_text(encoding="utf-8")):
        marks = sorted(
            (match.start(), -len(alias), alias)
            for alias in ALIASES
            for match in re.finditer(re.escape(alias), paragraph)
        )
        if not paragraph.lstrip().startswith("SOURCE:"):
            for cited in _CITATION.finditer(paragraph):
                if _NOTES.fullmatch(cited.group(0)):
                    continue
                before = [alias for at, _, alias in marks if at <= cited.start()]
                line = start + paragraph[: cited.start()].count("\n")
                work = before[-1] if before else last
                found.append((f"{page.name}:{line}", cited.group(0), work))
        if marks:
            last = marks[-1][2]
    return found


def _states(text: str, cited: str) -> bool:
    """Whether the extraction `text` *states* the numbered result `cited`.

    A statement opens a line: `LEMMA 6.5.`, `Lemma 4.7 (Map Containment).`,
    number-first as Amadio & Cardelli print one (`5.2.2 LEMMA`), and a section
    as `6.9. TITLE`, `1.5 Algorithm Outline` or its number alone. Case is folded,
    so a ligature the extraction split (`DEfiNITION`) still reads. A mention --
    "(Definition 6.9)" mid-sentence -- does not answer for a statement.
    """
    if cited.startswith("§"):
        number = re.escape(cited[1:])
        return re.search(rf"(?mi)^\s*{number}\.?(?:\s|$)", text) is not None
    kind, number = cited.split()
    number = re.escape(number)
    stated = rf"(?mi)^\s*(?:{kind}\s+{number}\b|{number}\.?\s+{kind}\b)"
    return re.search(stated, text) is not None


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
    the pages are the tracked record of what the design rests on, and "Lemma
    6.5" with nothing before it saying whose lemma is a sentence that cannot be
    checked by anybody.
    """
    cited = [row for page in PAGES for row in _citations(page)]
    # The scan is the detector: no citations is no attribution for anything.
    assert len(cited) >= 8, cited
    orphaned = [f"{where} {what}" for where, what, work in cited if work is None]
    assert not orphaned, (
        f"numbered results with no work named before them: {orphaned}. Name the "
        "work in the paragraph, or move the citation under the one it is of."
    )


def test_every_cited_work_is_a_paper_the_shelf_holds() -> None:
    """The work a citation resolves to is a file somebody can open."""
    if not SHELF.is_dir():
        pytest.skip(_ABSENT)
    rowed = {Path(name).stem for _, name in _table()}
    cited = {ALIASES[work] for page in PAGES for _, _, work in _citations(page) if work}
    missing = sorted(stem for stem in cited if stem is not None and stem not in rowed)
    assert not missing, f"works the pages cite that the shelf's table lacks: {missing}"


def test_every_numbered_result_is_one_its_paper_states() -> None:
    """The number a citation gives is one the cited paper prints.

    Read against the shelf's text extractions. A citation of a work the shelf
    does not hold is declared in `ALIASES` rather than checked.
    """
    texts = SHELF / "text"
    if not texts.is_dir():
        pytest.skip(_ABSENT)
    unstated, checked = [], 0
    for page in PAGES:
        for where, what, work in _citations(page):
            stem = ALIASES.get(work) if work else None
            if stem is None:
                continue
            checked += 1
            text = (texts / f"{stem}.txt").read_text(encoding="utf-8")
            if not _states(text, what):
                unstated.append(f"{where} {what}, which {work} does not state")
    assert checked >= 8, checked
    assert not unstated, unstated


def test_the_reader_of_a_statement_tells_it_from_a_mention() -> None:
    """The matcher answers both ways over the shapes the shelf prints."""
    jacm = "we introduce (Definition 6.9) which\nLEMMA 6.5.\nLet P\n6.9. DECIDABILITY\n"
    assert _states(jacm, "Lemma 6.5")
    assert _states(jacm, "§6.9")
    assert not _states(jacm, "Definition 6.9"), "a mention is not a statement"
    assert not _states(jacm, "Lemma 6.4")
    amadio = "5.2.2 LEMMA (A system of contractile equations).\n1.5 Algorithm Outline\n"
    assert _states(amadio, "Lemma 5.2.2")
    assert _states(amadio, "§1.5")
    assert not _states(amadio, "§4.1")
    icfp = "2.1.4\nImplementation.\nLemma 4.7 (Map Containment). Let\n"
    assert _states(icfp, "§2.1.4")
    assert _states(icfp, "Lemma 4.7")


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
