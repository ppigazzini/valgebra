"""A result the tree cites is cited by label, at a number its paper prints.

`docs/dev/10-theory.md` records which result each design decision rests on, so
a reader can check the decision against a page of a paper rather than against a
recollection. That only works while a citation resolves: "Lemma 6.5" is worth
writing because somebody can open the paper at Lemma 6.5, and worth nothing once
nobody can say which paper. So a citation is a label and a locator,
`[FCB08, Lem. 6.5]`, the label a row of the page's table of works, and
`docs/dev/12-writing.md` says when one is owed and where it goes.

What this holds, over the files `SCOPE` names:

* every citation's label is a row of the table, and every row is cited, so the
  table neither names a work nobody cites nor misses one somebody does;
* every locator has one of the forms the writing rule lists, and none stands
  outside a label -- a `Lemma 4.7` with no label is the unresolvable citation
  the form exists to end -- and no label stands without one, which rustdoc
  reads as a link it cannot resolve;
* no citation opens a paragraph or a comment, and none stands before a colon,
  where Markdown reads it as a link definition and deletes the line;
* where the shelf is at hand, every locator names a statement its paper prints:
  the shelf's own rule is to verify a theorem number and trust a citation
  chain, and a wrong number resolved to a real paper passed before.

The shelf is not redistributed, so a clone has none: the rows that read it skip
where it is absent rather than failing, and say so. The rest live entirely in
the tree and hold in every clone.

LEDGER: every citation is a label the theory page tables, at a number its paper prints
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING, NamedTuple

import pytest

from _tree import ROOT

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.repository

THEORY = ROOT / "docs" / "dev" / "10-theory.md"

#: Where the shelf lives. Assembled from its pieces rather than written, for the
#: reason `tests/test_commit_messages.py` assembles the same name: a tracked file
#: that spells it is the reference `scripts/docs_lint.py` exists to refuse, since
#: the directory reaches nobody who clones this repository.
SHELF = ROOT / ("__" + "DEV") / "papers"

#: The files the citation rule holds, as globs from the root. A file joins when
#: its citations are written in the form, which is the commit that writes them.
SCOPE = ("docs/dev/*.md",)

#: Each label the table may carry, and the stem of the shelf's file for the
#: work. The page cannot name the file -- the shelf is not in a clone -- so the
#: pairing lives here, beside the check that reads it.
LABELS: dict[str, str] = {
    "AC93": "amadio-cardelli-1993-subtyping-recursive-types",
    "CD24": "castagna-duboc-2024-guard-analysis-safe-erasure-elixir",
    "Cas23": "castagna-2023-typing-records-maps-structs-ICFP",
    "Cas24": "castagna-2021-union-intersection-negation",
    "DV14": "dantoni-veanes-2014-minimization-symbolic-automata",
    "FCB08": "frisch-castagna-benzaken-2008-semantic-subtyping-JACM",
    "HVP05": "hosoya-vouillon-pierce-2005-regular-expression-types-xml",
    "JV20": "jhala-vazou-2020-refinement-types",
    "Nak00": "nakano-2000-modality-for-recursion",
}

#: A label as `alpha` builds it: up to four letters of the authors, a `+` past
#: four of them, the year's two digits, and a letter where two collide.
_LABEL = r"[A-Z][A-Za-z]{0,3}\+?\d{2}[a-z]?"

#: A citation: a label, a comma, and a locator up to the bracket.
_TOKEN = re.compile(rf"\[(?P<label>{_LABEL}), (?P<locator>[^\]\n]+)\]")

#: A label alone in brackets, which is a citation with no place in the work.
_BARE = re.compile(rf"\[(?P<label>{_LABEL})\](?!\()")

#: A citation broken across a line: the parse reads no citation there, so its
#: locator would escape every check below.
_BROKEN = re.compile(rf"\[(?P<label>{_LABEL}),[^\]\n]*\n")

#: The parts a locator is written in, comma-separated, each one of the forms the
#: writing rule lists: an equation or rule tag as printed, a section, a numbered
#: statement, a figure, an algorithm, an appendix, a footnote, and the page or
#: line that narrows the part before it.
_NUMBER = r"[0-9A-Z]+(?:\.[0-9]+)*"
_PART = re.compile(
    rf"\((?P<tag>[A-Za-z0-9][A-Za-z0-9-]*)\)"
    rf"|§(?P<section>\d+(?:\.\d+)*)"
    rf"|(?P<kind>Def|Thm|Lem|Prop|Cor|Rem|Fig|Alg|App|fn)\. (?P<number>{_NUMBER})"
    rf"|(?:p|l)\. \d+(?:-\d+)?"
)

#: The words a statement is printed under, for each abbreviation a locator uses.
_KINDS = {
    "Def": "Definition",
    "Thm": "Theorem",
    "Lem": "Lemma",
    "Prop": "Proposition",
    "Cor": "Corollary",
    "Rem": "Remark",
    "Alg": "Algorithm",
}

#: A locator written outside a label: a named statement with its number, a
#: section sign, or a footnote. What a reader opens a paper *at*, which is what
#: makes it a citation rather than prose.
_LOOSE = re.compile(
    r"\b(?:Theorem|Definition|Lemma|Proposition|Corollary|Remark|Figure|Algorithm"
    r"|Appendix)\s+[0-9A-Z]+(?:\.[0-9]+)*\b"
    r"|§\d+(?:\.\d+)*"
    r"|\bfootnote\s+\d+"
)

#: A section of the project's own notes, which number from 13 and which no paper
#: here reaches; a `SOURCE:` line quotes those notes, and the theory ledger holds
#: it.
_SOURCE = "SOURCE:"

#: The table's rows: the label, the work, what it owns, the copy read, and the
#: identifier, which is a DOI or a versioned arXiv number.
_ROW = re.compile(
    r"^\| `(?P<label>[^`]+)` \|(?P<work>[^|]+)\|(?P<owns>[^|]+)\|(?P<read>[^|]+)"
    r"\| \[(?P<identifier>doi:10\.\d+/\S+|arXiv:\d{4}\.\d{4,5}v\d+)\]"
    r"\(https://\S+\) \|$"
)


class Citation(NamedTuple):
    """One citation in a file the rule holds."""

    where: str
    label: str
    locator: str


@functools.cache
def _scoped() -> tuple[Path, ...]:
    """Give the files `SCOPE` names, in a stable order."""
    return tuple(sorted({path for glob in SCOPE for path in ROOT.glob(glob)}))


#: Code a page quotes, inline or fenced: an example of the form, or of the form
#: refused, which is spelled rather than used.
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def _text(path: Path) -> str:
    """Give the text of `path` the rule reads, with what it does not read blanked.

    A `SOURCE:` line quotes the project's notes by their own section numbers,
    which are not a paper's, so it is read by the theory ledger and not here;
    and quoted code spells a citation rather than making one. Blanked rather
    than dropped, so a line number still points at its line.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    text = "\n".join("" if line.startswith(_SOURCE) else line for line in lines)
    return _CODE.sub(lambda code: re.sub(r"[^\n]", " ", code[0]), text)


@functools.cache
def _citations() -> tuple[Citation, ...]:
    """Give every citation in the files the rule holds."""
    found: list[Citation] = []
    for path in _scoped():
        text = _text(path)
        for token in _TOKEN.finditer(text):
            line = text.count("\n", 0, token.start()) + 1
            where = f"{path.relative_to(ROOT)}:{line}"
            found.append(Citation(where, token["label"], token["locator"]))
    return tuple(found)


@functools.cache
def _table() -> dict[str, str]:
    """Give the theory page's table of works, each label with its identifier."""
    text = THEORY.read_text(encoding="utf-8")
    section = text.split("## Works cited by label", 1)[1].split("\n## ", 1)[0]
    rows: dict[str, str] = {}
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        row = _ROW.match(line)
        assert row is not None, f"a row of the table this ledger cannot read: {line}"
        assert row["label"] not in rows, f"a label two rows carry: {row['label']}"
        rows[row["label"]] = row["identifier"]
    return rows


def _parts(locator: str) -> list[re.Match[str]] | None:
    """Give the parts of `locator`, or `None` where one has no form the rule has."""
    parts = []
    for piece in locator.split(", "):
        part = _PART.fullmatch(piece)
        if part is None:
            return None
        parts.append(part)
    return parts


def _states(text: str, part: re.Match[str]) -> bool | None:
    """Whether the extraction `text` *prints* the statement `part` locates.

    `None` for a page or a line, which narrows the part before it and is not a
    statement of its own. A statement opens a line: `LEMMA 6.5.`, `Lemma 4.7
    (Map Containment).`, number-first as Amadio & Cardelli print one (`5.2.2
    LEMMA`), a section as `6.9. TITLE`, `1.5 Algorithm Outline` or its number
    alone, a figure by its caption, an appendix by its heading, a footnote by its
    number set against its first word, and an equation or a rule by its tag
    closing a line. Case is folded, so a ligature the extraction split
    (`DEfiNITION`) still reads. A mention -- "(Definition 6.9)" mid-sentence --
    does not answer for a statement.
    """
    if part["tag"] is not None:
        tag = re.escape(part["tag"])
        return re.search(rf"(?m)\({tag}\)\s*$", text) is not None
    if part["section"] is not None:
        number = re.escape(part["section"])
        return re.search(rf"(?m)^\s*{number}\.?(?:\s|$)", text) is not None
    kind, number = part["kind"], part["number"]
    if kind is None:
        return None
    escaped = re.escape(number)
    if kind == "Fig":
        stated = rf"(?mi)^\s*(?:Figure|Fig\.)\s+{escaped}[.:]"
    elif kind == "App":
        stated = rf"(?m)^\s*(?:Appendix\s+)?{escaped}\.?(?:\s|$)"
    elif kind == "fn":
        stated = rf"(?m)^{escaped}[A-Z]"
    else:
        word = _KINDS[kind]
        stated = rf"(?mi)^\s*(?:{word}\s+{escaped}\b|{escaped}\.?\s+{word}\b)"
    return re.search(stated, text) is not None


_ABSENT = "the shelf is not redistributed, so a clone has none to read"


def test_the_scope_cites_by_label() -> None:
    """The parse is the detector, so it is shown to have read something.

    A ledger over an empty universe passes having checked nothing, which is the
    failure `tests/test_ledger_plants.py` rules out for the rest of them. And it
    reads both shapes the pages use most: a section and a named statement.
    """
    cited = _citations()
    assert len(cited) >= 20, cited
    assert any(c.locator.startswith("§") for c in cited), cited
    assert any(not c.locator.startswith("§") for c in cited), cited


def test_every_label_is_a_row_and_every_row_is_cited() -> None:
    """The table and the citations name the same works, in both directions.

    A label with no row is a citation no reader can resolve, and a row nothing
    cites is a work the page lists for nothing. `LABELS` pairs each row with the
    shelf's file, so a row it lacks is one no statement check can reach.
    """
    rows = _table()
    assert len(rows) >= 5, rows
    cited = {c.label for c in _citations()}
    unknown = sorted(cited - set(rows))
    assert not unknown, f"labels cited with no row in the table: {unknown}"
    idle = sorted(set(rows) - cited)
    assert not idle, f"rows of the table nothing cites: {idle}"
    unpaired = sorted(set(rows) ^ set(LABELS))
    assert not unpaired, f"rows and LABELS name different works: {unpaired}"


def test_every_locator_has_a_form_the_rule_lists() -> None:
    """A locator is written in the parts the writing rule names, and no other.

    `Lemma 4.7` inside a label, `p.12` with no space or a `Section 4` are each
    a form a reader parses differently from the next one, which is the drift a
    fixed form ends.
    """
    malformed = [
        f"{c.where} [{c.label}, {c.locator}]"
        for c in _citations()
        if _parts(c.locator) is None
    ]
    for path in _scoped():
        text = _text(path)
        malformed += [
            f"{path.relative_to(ROOT)}:{text.count(chr(10), 0, hit.start()) + 1} "
            f"[{hit['label']}, broken across a line"
            for hit in _BROKEN.finditer(text)
        ]
    assert not malformed, f"locators in no form the rule lists: {malformed}"


def test_no_locator_stands_outside_a_label() -> None:
    """A numbered statement or a section is cited inside a label, or not at all.

    The direction that holds without the papers, which is why it does not skip:
    "Lemma 6.5" with nothing saying whose lemma is a sentence nobody can check.
    """
    loose: list[str] = []
    for path in _scoped():
        text = _TOKEN.sub(lambda token: " " * len(token[0]), _text(path))
        for hit in _LOOSE.finditer(text):
            line = text.count("\n", 0, hit.start()) + 1
            loose.append(f"{path.relative_to(ROOT)}:{line} {hit[0]!r}")
    assert not loose, (
        f"locators outside a label: {loose}. Write them as [Label, locator], "
        "with the label a row of the theory page's table."
    )


def test_no_label_stands_without_a_locator() -> None:
    """A label alone names a work and no place in it, and rustdoc reads a link."""
    bare: list[str] = []
    for path in _scoped():
        text = _text(path)
        for hit in _BARE.finditer(text):
            line = text.count("\n", 0, hit.start()) + 1
            bare.append(f"{path.relative_to(ROOT)}:{line} {hit[0]}")
    assert not bare, f"labels with no locator: {bare}"


def test_no_citation_opens_a_paragraph_or_stands_before_a_colon() -> None:
    """A citation follows the clause it supports, and is never a link definition.

    A paragraph opening on a citation puts the source before the claim, and a
    citation before a colon is `[label]: destination` to Markdown, which deletes
    the line -- mid-paragraph, in the site's renderer.
    """
    misplaced: list[str] = []
    for path in _scoped():
        text = _text(path)
        for paragraph in re.finditer(r"(?:^|\n\n)\s*(\[[^\]\n]*\])", text):
            if _TOKEN.fullmatch(paragraph[1]):
                line = text.count("\n", 0, paragraph.start(1)) + 1
                misplaced.append(f"{path.relative_to(ROOT)}:{line} opens on it")
        for token in _TOKEN.finditer(text):
            if text.startswith(":", token.end()):
                line = text.count("\n", 0, token.start()) + 1
                misplaced.append(f"{path.relative_to(ROOT)}:{line} before a colon")
    assert not misplaced, misplaced


def test_every_locator_is_one_its_paper_states() -> None:
    """The number a citation gives is one the cited paper prints.

    Read against the shelf's text extractions, each part of a locator that names
    a statement of its own.
    """
    texts = SHELF / "text"
    if not texts.is_dir():
        pytest.skip(_ABSENT)
    unstated, checked = [], 0
    for citation in _citations():
        stem = LABELS.get(citation.label)
        parts = _parts(citation.locator)
        if stem is None or parts is None:
            continue
        text = (texts / f"{stem}.txt").read_text(encoding="utf-8")
        for part in parts:
            stated = _states(text, part)
            if stated is None:
                continue
            checked += 1
            if not stated:
                unstated.append(f"{citation.where} [{citation.label}, {part[0]}]")
    assert checked >= 20, checked
    assert not unstated, f"locators the cited paper does not print: {unstated}"


def test_the_reader_of_a_statement_tells_it_from_a_mention() -> None:
    """The matcher answers both ways over the shapes the shelf prints."""

    def part(locator: str) -> re.Match[str]:
        parts = _parts(locator)
        assert parts is not None, locator
        return parts[0]

    jacm = (
        "we introduce (Definition 6.9) which\nLEMMA 6.5.\nLet P\n6.9. DECIDABILITY\n"
        "(5)\nFigure 3. The rules\n"
    )
    assert _states(jacm, part("Lem. 6.5"))
    assert _states(jacm, part("§6.9"))
    assert not _states(jacm, part("Def. 6.9")), "a mention is not a statement"
    assert not _states(jacm, part("Lem. 6.4"))
    assert _states(jacm, part("(5)"))
    assert not _states(jacm, part("(6)"))
    assert _states(jacm, part("Fig. 3"))
    assert not _states(jacm, part("Fig. 4"))
    amadio = "5.2.2 LEMMA (A system of contractile equations).\n1.5 Algorithm Outline\n"
    assert _states(amadio, part("Lem. 5.2.2"))
    assert _states(amadio, part("§1.5"))
    assert not _states(amadio, part("§4.1"))
    icfp = "2.1.4\nImplementation.\nLemma 4.7 (Map Containment). Let\nC.1\n15We could\n"
    assert _states(icfp, part("§2.1.4"))
    assert _states(icfp, part("Lem. 4.7"))
    assert _states(icfp, part("App. C.1"))
    assert _states(icfp, part("fn. 15"))
    assert not _states(icfp, part("fn. 16"))
    rules = "C1 ⊆ C2 (uninhabited)\nas (anyOf-l) states\n"
    assert _states(rules, part("(uninhabited)"))
    assert not _states(rules, part("(anyOf-l)")), "a tag mid-line is a mention"
    assert _states(icfp, part("l. 14")) is None, "a line narrows, it is no statement"


def test_the_table_and_the_shelf_name_the_same_papers() -> None:
    """Every label's file is on the shelf, and every paper there has a row.

    Both directions over the shelf's own table, so neither list drifts alone. A
    label whose file is gone is a citation that resolves to nothing; a file with
    no row in the shelf's table is a paper nothing says the use of, which is the
    whole selection rule the shelf is chosen by.
    """
    if not SHELF.is_dir():
        pytest.skip(_ABSENT)
    held = {path.stem for path in SHELF.glob("*.pdf")}
    missing = sorted(stem for stem in LABELS.values() if stem not in held)
    assert not missing, f"labels whose paper the shelf does not hold: {missing}"
    shelf = (SHELF / "README.md").read_text(encoding="utf-8")
    rowed = set(re.findall(r"`([a-z0-9-]+(?:-[A-Za-z]+)?)\.pdf`", shelf))
    assert len(rowed) >= 10, sorted(rowed)
    absent = sorted(name for name in rowed if name not in held)
    assert not absent, f"rows naming a file the shelf does not hold: {absent}"
    unrowed = sorted(held - rowed)
    assert not unrowed, (
        f"papers on the shelf with no row saying what they are for: {unrowed}"
    )
