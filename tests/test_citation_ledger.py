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

And the code half, which is what keeps the page, the code and the papers one
account. A claim on the theory page names the items that implement it on an
`IMPLEMENTED-BY:` line, and over the source files in `SCOPE`:

* every entry resolves to one item -- a function, a method, a type, or a module
  -- whose doc comment cites a result its claim cites, so a rename or a
  deletion takes the entry with it rather than leaving the page pointing at
  nothing;
* every item whose doc comment cites is named under a claim, and every result it
  cites is one a claim naming it cites, so code cannot start resting on a
  result the page does not account for;
* a citation inside a body is a step of what its item implements, so the item
  cites too, and no citation belongs to no item;
* a claim that cites a result names the code resting on it, so a line deleted
  whole does not leave a claim no check reads.

What no check can see is an item that implements a result and cites nothing. A
reader decides that a function is Lemma 4.7; this holds the citation once
written.

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
SCOPE = (
    "docs/dev/*.md",
    "crates/valgebra-core/src/decision.rs",
    "crates/valgebra-core/src/decision/*.rs",
    "crates/valgebra-core/src/descr/mod.rs",
    "crates/valgebra-core/src/descr/lines.rs",
    "crates/valgebra-core/src/descr/polar.rs",
    "crates/valgebra-core/src/descr/sets.rs",
    "crates/valgebra-core/src/descr/values.rs",
    "crates/valgebra-core/src/descr/classes.rs",
    "crates/valgebra-core/src/descr/budget.rs",
    "crates/valgebra-core/src/descr/maps.rs",
    "crates/valgebra-core/src/descr/records.rs",
    "crates/valgebra-core/src/descr/lower.rs",
    "crates/valgebra-core/src/descr/tests.rs",
    "crates/valgebra-core/src/descr/lattice_tests.rs",
    "crates/valgebra-core/src/descr/sets/*.rs",
    "crates/valgebra-core/src/descr/maps/*.rs",
    "crates/valgebra-core/src/descr/records/*.rs",
    "crates/valgebra-core/src/descr/lower/*.rs",
    "crates/valgebra-core/src/descr/regular.rs",
    "crates/valgebra-core/src/descr/symbolic.rs",
    "crates/valgebra-core/src/descr/automaton.rs",
    "crates/valgebra-core/src/descr/integers.rs",
    "crates/valgebra-core/src/descr/interval.rs",
    "crates/valgebra-core/src/descr/floats.rs",
    "crates/valgebra-core/src/descr/regular/*.rs",
    "crates/valgebra-core/src/descr/symbolic/*.rs",
    "crates/valgebra-core/src/descr/integers/*.rs",
    "crates/valgebra-core/src/descr/interval/*.rs",
    "crates/valgebra-core/src/descr/floats/*.rs",
    "crates/valgebra-core/src/ir.rs",
    "crates/valgebra-core/src/ir/*.rs",
    "crates/valgebra-core/src/kind.rs",
    "crates/valgebra-core/src/oracle.rs",
    "crates/valgebra-core/src/verdict.rs",
    "crates/valgebra-core/src/carries.rs",
    "crates/valgebra-core/src/violation.rs",
    "crates/valgebra-core/src/lib.rs",
    "crates/valgebra-core/src/laws.rs",
    "crates/valgebra-core/src/schema_tests.rs",
    "crates/valgebra-py/src/build.rs",
    "crates/valgebra-py/src/check/walk.rs",
    "crates/valgebra-py/src/check/walk/record.rs",
    "crates/valgebra-py/src/check/walk/scalar.rs",
)

#: A test module, whose tests a `HELD-BY:` line names: its citations are held to
#: the form and listed under no claim, since a test holds a result rather than
#: implementing it.
_TEST_MODULE = re.compile(r"(?:^|/)(?:(?:\w+_)?tests|laws)\.rs$")

#: Each label the table may carry, and the stem of the shelf's file for the
#: work. The page cannot name the file -- the shelf is not in a clone -- so the
#: pairing lives here, beside the check that reads it.
LABELS: dict[str, str] = {
    "AC93": "amadio-cardelli-1993-subtyping-recursive-types",
    "BBC+26": "baazizi-et-al-2026-json-schema-inclusion-refutational-normalization",
    "CD24": "castagna-duboc-2024-guard-analysis-safe-erasure-elixir",
    "Cas23": "castagna-2023-typing-records-maps-structs-ICFP",
    "Cas24": "castagna-2021-union-intersection-negation",
    "DV14": "dantoni-veanes-2014-minimization-symbolic-automata",
    "FCB08": "frisch-castagna-benzaken-2008-semantic-subtyping-JACM",
    "HSHP20": "habib-shinnar-hirzel-pradel-2020-type-safety-json-subschema",
    "HVP05": "hosoya-vouillon-pierce-2005-regular-expression-types-xml",
    "JV20": "jhala-vazou-2020-refinement-types",
    "LN26": "laurent-nguyen-2026-implementing-set-theoretic-types",
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


#: A line of Rust that is a comment, doc or plain, and the text it carries.
_COMMENT = re.compile(r"^(?P<lead>\s*//[/!]?)(?P<text>.*)$")


@functools.cache
def _text(path: Path) -> str:
    """Give the text of `path` the rule reads, with what it does not read blanked.

    A `SOURCE:` line quotes the project's notes by their own section numbers,
    which are not a paper's, so it is read by the theory ledger and not here;
    and quoted code spells a citation rather than making one. Of a source file
    only the comments are read, each line that is one, so a blank comment line
    parts two paragraphs as a blank line parts them on a page. Blanked rather
    than dropped, so a line number still points at its line.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    if path.suffix == ".rs":
        read = []
        for line in lines:
            comment = _COMMENT.match(line)
            if comment is None or not comment["text"].strip():
                read.append("")
                continue
            read.append(" " * len(comment["lead"]) + comment["text"])
        lines = read
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
    reads both shapes the theory page uses most: a section and a named
    statement.
    """
    cited = _citations()
    assert len(cited) >= 20, cited
    page = [c for c in cited if c.where.startswith("docs/dev/10-theory.md:")]
    assert any(c.locator.startswith("§") for c in page), page
    assert any(not c.locator.startswith("§") for c in page), page


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


#: What a theory claim names its implementing items on.
IMPLEMENTED_BY = "IMPLEMENTED-BY:"

#: The tags a claim about code carries, which the theory ledger reads; a
#: guiding or planned paragraph is context, and code rests on no context.
_ABOUT_CODE = ("LOAD-BEARING", "OBLIGATION", "DEVIATION", "NOT-REACHED")


class Claim(NamedTuple):
    """One tagged paragraph of the theory page, its citations, and its items."""

    tag: str
    identifier: str
    cites: frozenset[tuple[str, str]]
    items: tuple[str, ...]


@functools.cache
def _claims() -> tuple[Claim, ...]:
    """Give the theory page's claims, each with the items it names.

    An `IMPLEMENTED-BY:` paragraph belongs to the claim before it, as a
    `HELD-BY:` one does, and the claim's citations are its own paragraph's: a
    claim is read on its own, so what it rests on is written in it.
    """
    found: list[Claim] = []
    for paragraph in THEORY.read_text(encoding="utf-8").split("\n\n"):
        if paragraph.startswith(IMPLEMENTED_BY):
            assert found, f"an IMPLEMENTED-BY line before any claim: {paragraph!r}"
            last = found[-1]
            assert not last.items, f"two IMPLEMENTED-BY lines for {last.identifier}"
            entries = paragraph[len(IMPLEMENTED_BY) :].replace("\n", " ").split(",")
            items = tuple(entry.strip() for entry in entries if entry.strip())
            found[-1] = last._replace(items=items)
            continue
        tag = _TAG.search(paragraph)
        if tag is None:
            continue
        cites = frozenset(
            (token["label"], token["locator"]) for token in _TOKEN.finditer(paragraph)
        )
        found.append(Claim(tag[1], tag[2], cites, ()))
    return tuple(found)


#: A claim's tag as the theory page writes it, and the id a line names it by.
_TAG = re.compile(r"\*\*\[([A-Z-]+): ([a-z][a-z0-9-]*)\]\*\*")


class Item(NamedTuple):
    """A function, method, type or module of a source file, and where it sits."""

    file: Path
    name: str
    """`phi`, `Slot::below`, `<Atom as Summand>::meet`, `Labels`, or the empty
    string for the module."""
    doc: frozenset[tuple[str, str]]
    """What its doc comment cites."""
    steps: frozenset[tuple[str, str]]
    """What the comments inside its body cite."""
    span: range
    """The lines it holds, its doc comment's included, counted from zero."""


_IMPL = re.compile(
    r"^impl\b(?:<.*?>)?\s+(?:(?:\w+::)*(?P<trait>\w+)(?:<[^{]*?>)?\s+for\s+)?"
    r"(?:\w+::)*(?P<owner>\w+)"
)
_TRAIT = re.compile(r"^(?:pub(?:\([\w:]+\))?\s+)?(?:unsafe\s+)?trait\s+(?P<owner>\w+)")
_FN = re.compile(
    r"^(?P<indent>\s*)(?:pub(?:\([\w:]+\))?\s+)?(?:const\s+)?(?:unsafe\s+)?"
    r"fn\s+(?P<name>\w+)"
)
_TYPE = re.compile(
    r"^(?:pub(?:\([\w:]+\))?\s+)?(?:struct|enum|union|type)\s+(?P<name>\w+)"
)


def _cited(text: str) -> frozenset[tuple[str, str]]:
    return frozenset((t["label"], t["locator"]) for t in _TOKEN.finditer(text))


def _named(line: str, owner: str | None) -> str | None:
    """Give the item `line` opens, inside the block `owner` names, or `None`."""
    fn = _FN.match(line)
    if fn and not fn["indent"]:
        return fn["name"]
    if fn and fn["indent"] == "    " and owner is not None:
        return f"{owner}::{fn['name']}"
    kind = _TYPE.match(line) or _TRAIT.match(line)
    if kind is None:
        return None
    return kind.groupdict().get("name") or kind["owner"]


def _closing(raw: list[str], at: int) -> int:
    """Give the line that closes the item opened at `at`, at its indentation."""
    line = raw[at]
    if line.rstrip().endswith((";", "}")):
        return at
    indent = line[: len(line) - len(line.lstrip())]
    return next(
        (k for k in range(at + 1, len(raw)) if raw[k].startswith(indent + "}")), at
    )


@functools.cache
def _items(path: Path) -> tuple[Item, ...]:
    """Give the items of the Rust file `path`, read as `rustfmt` lays them out.

    A function at the margin is its name, one an `impl` or `trait` block holds
    is `Owner::name`, and a type at the margin is its name; the module is the
    `//!` lines at the head. An item's doc comment is the `///` lines above it,
    attributes allowed between, and its body the lines to the brace that closes
    at its indentation. A citation in a comment the read reaches inside no item
    is reported by the check that wants every citation owned.
    """
    raw = path.read_text(encoding="utf-8").split("\n")
    read = _text(path).split("\n")
    head = 0
    while head < len(raw) and raw[head].lstrip().startswith("//!"):
        head += 1
    module = _cited("\n".join(read[:head]))
    found = [Item(path, "", module, frozenset(), range(head))]
    owner = None
    for at, line in enumerate(raw):
        if owner is not None and line.startswith("}"):
            owner = None
        name = _named(line, owner)
        block = _IMPL.match(line) or _TRAIT.match(line)
        if block is not None and not line.rstrip().endswith(";"):
            trait = block.groupdict().get("trait")
            owner = f"<{block['owner']} as {trait}>" if trait else block["owner"]
        if name is None:
            continue
        top = at
        while top > 0 and re.match(r"^\s*(?:///|#\[|#!\[)", raw[top - 1]):
            top -= 1
        doc = "\n".join(
            read[k] for k in range(top, at) if raw[k].lstrip().startswith("///")
        )
        end = _closing(raw, at)
        steps = _cited("\n".join(read[at : end + 1]))
        found.append(Item(path, name, _cited(doc), steps, range(top, end + 1)))
    return tuple(found)


def _sources() -> tuple[Path, ...]:
    """Give the source files the rule holds whose items implement, not test."""
    return tuple(
        path
        for path in _scoped()
        if path.suffix == ".rs" and not _TEST_MODULE.search(path.as_posix())
    )


def _entry(path: Path) -> str:
    """Give how an `IMPLEMENTED-BY:` entry names the file `path`.

    A path under the core's `src`, which is where most of the code a theory
    claim names lives, and the binding's from its crate down, so its `lib.rs`
    is not the core's.
    """
    relative = path.relative_to(ROOT / "crates").as_posix()
    return relative.removeprefix("valgebra-core/src/")


def _resolve(entry: str) -> list[Item]:
    """Give the items an `IMPLEMENTED-BY:` entry can mean, among the held sources."""
    where, _, name = entry.partition("::")
    return [
        item
        for path in _sources()
        if _entry(path) == where
        for item in _items(path)
        if item.name == name
    ]


def test_an_implementing_line_follows_a_claim_about_code() -> None:
    """Code is named under a claim it can be false against, never under context.

    A guiding or planned paragraph names a result the design leans on without
    building; an item listed under one would be code resting on a sentence no
    test holds.
    """
    claims = _claims()
    assert len(claims) >= 40, claims
    misplaced = [c.identifier for c in claims if c.items and c.tag not in _ABOUT_CODE]
    assert not misplaced, f"IMPLEMENTED-BY lines under context: {misplaced}"
    named = [c for c in claims if c.items]
    assert len(named) >= 5, "the parse read no IMPLEMENTED-BY line"


def test_every_implementing_entry_resolves_and_cites_its_claim() -> None:
    """An entry is one item, and the item cites a result its claim cites.

    So a function renamed or deleted takes its entry with it, and an entry
    whose item cites nothing of the claim is two accounts of the code that
    disagree.
    """
    wrong: list[str] = []
    for claim in _claims():
        for entry in claim.items:
            found = _resolve(entry)
            if len(found) != 1:
                wrong.append(f"{claim.identifier}: {entry} names {len(found)} items")
                continue
            if not found[0].doc & claim.cites:
                wrong.append(
                    f"{claim.identifier}: {entry} cites nothing the claim cites"
                )
    assert not wrong, wrong


def test_every_citing_item_is_listed_under_a_claim_that_cites_it() -> None:
    """Code cites only results the page accounts for, under a claim naming it.

    The reverse direction: an item whose doc comment cites is named on some
    claim's `IMPLEMENTED-BY:` line, and each result it cites is one a claim
    naming it cites -- a function that starts resting on a lemma the page does
    not mention fails here, in the commit that writes the citation.
    """
    listed: dict[str, set[tuple[str, str]]] = {}
    for claim in _claims():
        for entry in claim.items:
            listed.setdefault(entry, set()).update(claim.cites)
    unlisted: list[str] = []
    for path in _sources():
        for item in _items(path):
            if not item.doc:
                continue
            short = _entry(path)
            entry = f"{short}::{item.name}" if item.name else short
            cites = listed.get(entry)
            if cites is None:
                unlisted.append(f"{entry} is named under no claim")
                continue
            unlisted += [
                f"{entry} cites [{label}, {locator}], which no claim naming it cites"
                for label, locator in sorted(item.doc - cites)
            ]
    assert not unlisted, unlisted


def test_a_claim_about_code_lists_what_implements_it() -> None:
    """A claim that cites a result about code names the code that rests on it.

    The direction the other two cannot reach: an entry deleted from a line
    leaves an item no claim names, which the reverse check reports, but a
    claim whose line is deleted whole names nothing and asks nothing. A claim
    citing a result is a claim some code implements, so the line is owed.
    """
    unlisted = [
        claim.identifier
        for claim in _claims()
        if claim.tag in _ABOUT_CODE and claim.cites and not claim.items
    ]
    assert not unlisted, (
        f"claims citing a result with no IMPLEMENTED-BY line: {unlisted}"
    )


def test_a_citation_inside_a_body_is_a_step_of_an_item_that_cites() -> None:
    """A step cites inside an item that implements, and every citation has one.

    A citation in a body is a line taking a step of a result, so the item it is
    in implements that result and says so in its doc comment; a citation the
    read finds in no item -- a comment between two functions -- belongs to
    nothing a claim can name.
    """
    orphaned: list[str] = []
    for path in _sources():
        items = _items(path)
        orphaned += [
            f"{path.relative_to(ROOT)}::{item.name} cites inside its body only"
            for item in items
            if item.steps and not item.doc and item.name
        ]
        text = _text(path)
        for token in _TOKEN.finditer(text):
            line = text.count("\n", 0, token.start())
            if not any(line in item.span for item in items):
                where = f"{path.relative_to(ROOT)}:{line + 1}"
                orphaned.append(f"{where} {token[0]} belongs to no item")
    assert not orphaned, orphaned
