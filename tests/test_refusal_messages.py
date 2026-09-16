"""The refusals a caller meets, read back one sentence at a time.

A frontend refusal is the whole of what a caller gets: the exception is
`NotImplementedError` or `ValueError` at every site, so the type says that an
annotation was refused and never which one. The message is what says why, and
what to write instead.

So each row here provokes one site and asserts its *sentence*. A reworded
message fails the row that reads it, which is what makes the wording a thing the
suite holds rather than a thing that drifts. `tests/test_frontend_refusals.py`
is the ledger over this file: it reads every refusal the frontend writes and
fails on one no row matches, so a site added tomorrow arrives here rather than
going unread.

Two sites share a sentence -- a `...` out of place is the same mistake whether
the list ends in one or not -- and both are provoked, because the ledger reads
the sites rather than the sentences.

Most rows carry a `ty: ignore`, and that is the point of them rather than a
concession: the annotation is one a type checker is right to refuse, and the
question here is what *this library* says when a caller writes it anyway. A row
with no suppression would be a row about an annotation nobody makes by mistake.
"""

from __future__ import annotations

import re
import typing
from typing import Annotated, Protocol, TypeVar

import annotated_types as at
import pytest

import valgebra as vg


class _Flagged:
    """A pattern marker carrying a flag, read the way a compiled one is.

    The frontend reads a marker's `pattern` and `flags`, so this stands in for
    a compiled pattern without compiling one. Written out because the two flags
    worth refusing cannot both be reached through `re.compile`: `re.LOCALE` is
    valid only on a bytes pattern, which is refused one step earlier for being
    bytes, and compiling with `re.DEBUG` runs the other engine's disassembler --
    which prints to stdout at import, and on one supported interpreter raises
    `IndexError` out of its own opcode table, taking collection down with it.
    """

    def __init__(self, pattern: str, flags: int) -> None:
        self.pattern = pattern
        self.flags = flags


class _Structural(Protocol):
    """A protocol with no `@runtime_checkable`, which `isinstance` refuses."""

    def method(self) -> None: ...


#: Each row: what the refusal is about, the annotation that provokes it, the
#: exception it raises, and a phrase from the message that is about *this*
#: refusal rather than about refusals in general.
REFUSALS: list[tuple[str, object, type[Exception], str]] = [
    # `build.rs`: the value fallthrough, where a typing object that is not a
    # type would otherwise be interned as a literal of itself and match nothing.
    (
        "a type variable is not a type",
        TypeVar("T"),  # ty: ignore[invalid-legacy-type-variable]
        NotImplementedError,
        "is a typing construct, not a value",
    ),
    # `build/generics.rs`: the arities. A mapping needs both halves, and a
    # container that takes one argument is not given two. `typing` guards the
    # qualifiers and `Unpack` at subscription, so the two callers that reach
    # this refusal are the containers.
    (
        "a mapping needs both halves",
        dict[str],  # ty: ignore[invalid-type-arguments]
        NotImplementedError,
        "needs a key type and a value type",
    ),
    (
        "a list takes one element type",
        list[int, str],  # ty: ignore[invalid-type-arguments]
        NotImplementedError,
        "takes exactly one type argument",
    ),
    (
        "a set takes one element type",
        set[int, str],  # ty: ignore[invalid-type-arguments]
        NotImplementedError,
        "a set schema is homogeneous",
    ),
    # A repeated tail repeats the element before it, so a `...` anywhere else
    # names no element. Both arms: the list that ends in one, and the list that
    # does not.
    (
        "a repeat before the end, in a list that ends in one",
        [int, ..., ...],
        NotImplementedError,
        "may appear only as the last element",
    ),
    (
        "a repeat before the end, in a fixed list",
        [int, ..., str],
        NotImplementedError,
        "may appear only as the last element",
    ),
    # `build/refine.rs`: a constraint asked of a base that cannot answer it.
    # Ignoring the marker would leave a schema admitting what it excludes.
    (
        "a length bound over a base with no length",
        Annotated[int, at.MinLen(1)],
        NotImplementedError,
        "values have no",
    ),
    # A flag the other engine understands and this one does not: carrying the
    # pattern without it would match a different set of strings.
    (
        "a pattern flag this engine does not carry",
        Annotated[str, _Flagged("x", re.DEBUG)],
        NotImplementedError,
        "cannot be carried into this pattern",
    ),
    (
        "a pattern flag about an alphabet this engine has no notion of",
        Annotated[str, _Flagged("x", re.LOCALE)],
        NotImplementedError,
        "cannot be carried into this pattern",
    ),
    # A pattern that is not text. `re` matches one against `bytes`; a pattern
    # constraint here matches the text of a `str`.
    (
        "a bytes pattern",
        Annotated[str, re.compile(b"x")],
        NotImplementedError,
        "cannot constrain a schema",
    ),
    (
        "a pattern that is not a pattern",
        Annotated[str, vg.Regex(123)],  # ty: ignore[invalid-argument-type]
        NotImplementedError,
        "cannot constrain a schema",
    ),
    # A length no value can have. The marker holds it; the schema cannot.
    (
        "a negative length",
        Annotated[str, at.MinLen(-1)],
        ValueError,
        "must be a length a value can have",
    ),
    # A marker from the vocabulary this frontend reads, naming a constraint it
    # does not check. Ignored, it would admit every value the marker excludes.
    (
        "a vocabulary marker this frontend does not check",
        Annotated[str, at.Timezone(None)],
        NotImplementedError,
        "is a constraint this frontend does not check",
    ),
    # `build/classes.rs`: a special form that is a class on some interpreters,
    # so it reaches the class path and is refused there rather than built into
    # an instance check that accepts nothing.
    (
        "a bare special form",
        typing.Union,
        NotImplementedError,
        "is a typing special form",
    ),
    # A protocol decides membership by `isinstance`, which a protocol that is
    # not runtime-checkable refuses to answer.
    (
        "a protocol that is not runtime-checkable",
        _Structural,
        NotImplementedError,
        "must be @runtime_checkable",
    ),
]


@pytest.mark.parametrize(
    ("annotation", "raises", "says"),
    [(row[1], row[2], row[3]) for row in REFUSALS],
    ids=[row[0] for row in REFUSALS],
)
def test_a_refused_annotation_says_which_refusal_it_is(
    annotation: object, raises: type[Exception], says: str
) -> None:
    """The message names this refusal, not refusals in general."""
    with pytest.raises(raises, match=says):
        vg.Validator(annotation)


def test_an_arity_refusal_names_the_annotation_that_was_written() -> None:
    """The message quotes the spelling, so a reader can find it in their source.

    `single_arg` served list, set and frozenset and said only "expected exactly
    one type argument": a count, with no mention of whose or of what to write.
    A caller with one long annotation had nothing to search for.
    """
    for annotation, spelling in [
        (list[int, str], "list[int, str]"),  # ty: ignore[invalid-type-arguments]
        (set[int, str], "set[int, str]"),  # ty: ignore[invalid-type-arguments]
        (frozenset[int, str], "frozenset[int, str]"),  # ty: ignore[invalid-type-arguments]
    ]:
        with pytest.raises(NotImplementedError) as caught:
            vg.Validator(annotation)
        message = str(caught.value)
        assert message.startswith(spelling), message
        assert "written with 2" in message, message


def test_every_refusal_names_what_to_write_instead() -> None:
    """A refusal that only says no leaves a caller with nothing to do.

    Every message here is two halves -- what was found, and what to write --
    and the second half is the one a reworded message loses first. Read as a
    length rather than as a wording, because which words carry it is the
    reviewer's question and `docs/03-schema-language.md` is where that answer
    is written.
    """
    for name, annotation, raises, _ in REFUSALS:
        with pytest.raises(raises) as caught:
            vg.Validator(annotation)
        message = str(caught.value)
        assert len(message) > 40, f"{name}: {message!r}"
        assert message[0] not in "\n ", f"{name}: {message!r}"


def test_a_refusal_is_raised_before_any_value_is_walked() -> None:
    """A refused annotation builds no validator, so nothing is half-compiled.

    The failure this guards is a refusal raised from the walk instead of the
    build: a caller would get a validator that refuses every value with a
    message about the annotation, which reads as the value being wrong.
    """
    for name, annotation, raises, _ in REFUSALS:
        with pytest.raises(raises):
            vg.Validator(annotation)
        # And the same annotation nested inside a container is refused at the
        # same place, rather than being reached only when a value arrives.
        with pytest.raises(raises) as caught:
            vg.Validator(list[annotation])  # type: ignore[valid-type]  # ty: ignore[invalid-type-form]
        assert str(caught.value), name
