"""The node set is minimal, and this is where that stops being a slogan.

`AGENTS.md` says valgebra is "the smallest set of schema nodes whose Boolean
closure is consistent and complete for its domain", and nothing checked it. Five
of the twenty-one variants denote sets the others already reach -- `Bool` is
`Literal[True] | Literal[False]`, `Nothing` is `complement(anything)` -- so read
literally the claim was false, and read charitably it was a claim nobody had
written down.

So the definition is restated and held: the node set is a **generating set plus
named representatives**. A generator denotes a set no combination of the others
reaches, and admitting one is the argument `docs/dev/01-schema-ir.md` describes.
A representative denotes a set the generators do reach, and earns its place by
being the form the normal form names -- `A & ~A` has to fold to *something*, and
`Nothing` is what it folds to. What is not allowed is a third kind: a variant
that is neither irreducible nor the canonical form of something.

Each column below is held to the tree in both directions. A variant in `ir.rs`
that no column names fails, a column naming a variant that is gone fails, and
every representative carries the derivation it stands for -- checked by
`is_equivalent`, in both directions, so a representative that stopped being one
fails here rather than in a reader's head.

LEDGER: every schema variant is a generator, a representative, or a marker
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

import annotated_types as at
import pytest

from valgebra import (
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    union,
)

# The value a `Literal` of it is built from; see `REPRESENTATIVES`.
_NONE = None

ROOT = Path(__file__).resolve().parent.parent
IR = ROOT / "crates" / "valgebra-core" / "src" / "ir.rs"

# The variants of `pub enum Schema`, read from the tree rather than restated.
ENUM = re.compile(r"^pub enum Schema \{$(.*?)^\}$", re.DOTALL | re.MULTILINE)
VARIANT = re.compile(r"^    ([A-Z][A-Za-z]*)[ ({,]", re.MULTILINE)

# A set no combination of the others reaches. Adding one extends the algebra,
# and the case for it is that the domain is unreachable without it.
GENERATORS = {
    # The lattice top. Every other set is written by narrowing it, and no
    # combination of the rest reaches it -- `A | ~A` is *folded* to it, which is
    # the fold naming this variant rather than deriving it.
    "Anything",
    # The scalar kinds that are not singletons.
    "Int",
    "Float",
    "Str",
    "Bytes",
    # The pooled atoms: a typed singleton, and the instances of a class.
    "Literal",
    "Instance",
    # The structural constructors. Each denotes a shape no Boolean combination
    # of the others expresses, which is the whole reason the algebra has them.
    "Seq",
    "Set",
    "FrozenSet",
    "KeyedMap",
    "AttrRecord",
    # A base narrowed by constraints a set-theoretic combination cannot state.
    "Refine",
    # The two irreducible connectives. Intersection is not here: De Morgan
    # derives it, and it is a representative below.
    "Union",
    "Complement",
}

# A set the generators reach, kept because the normal form needs a form to name.
# Each carries the derivation it stands for, checked below in both directions.
REPRESENTATIVES = {
    "Nothing": (nothing, complement(anything)),
    # `Literal[None]` is the derivation, and the linters rewrite that spelling
    # to `None` on sight -- which is the claim rather than a check of it. The
    # literal is built from the value instead, which is the same node.
    "NoneType": (Validator(None), Validator(Literal[_NONE])),
    "Bool": (Validator(bool), union(Literal[True], Literal[False])),
    "Intersection": (
        intersection(int, str),
        complement(union(complement(int), complement(str))),
    ),
}

# Not sets at all: a reference to a definition, and the marker for one being
# built. They denote what they name, which is why neither column above fits.
MARKERS = {"Ref", "SelfRef"}


def _variants() -> set[str]:
    body = ENUM.search(IR.read_text(encoding="utf-8"))
    assert body, "ir.rs has no `pub enum Schema`"
    found = set(VARIANT.findall(body.group(1)))
    # The scan is the detector: an empty variant set would pass both directions
    # having read nothing.
    assert len(found) >= 15, f"the variant scan found only {sorted(found)}"
    return found


def test_every_variant_is_a_generator_a_representative_or_a_marker() -> None:
    claimed = GENERATORS | set(REPRESENTATIVES) | MARKERS
    missing = sorted(_variants() - claimed)
    assert not missing, (
        f"schema variants this ledger does not account for: {missing}. Each is a "
        "generator (a set no combination of the others reaches), a representative "
        "(a set they do reach, kept because the normal form names it), or a "
        "build marker. A variant that is none of the three is a node the "
        "minimality claim does not survive."
    )


def test_no_column_names_a_variant_that_is_gone() -> None:
    claimed = GENERATORS | set(REPRESENTATIVES) | MARKERS
    stale = sorted(claimed - _variants())
    assert not stale, f"columns naming variants ir.rs no longer has: {stale}"


def test_the_columns_do_not_overlap() -> None:
    both = sorted((GENERATORS & set(REPRESENTATIVES)) | (GENERATORS & MARKERS))
    assert not both, f"variants claimed twice: {both}"


@pytest.mark.parametrize("name", sorted(REPRESENTATIVES))
def test_a_representative_denotes_what_it_stands_for(name: str) -> None:
    """The derivation is checked, not asserted: a representative is a shorthand.

    Both directions, because equivalence is mutual inclusion and a
    representative that had drifted would usually keep one of them.
    """
    kept, derived = REPRESENTATIVES[name]
    assert kept.is_equivalent(derived), f"{name} is not what it stands for"
    assert derived.is_equivalent(kept), f"{name} is not what it stands for"


def test_a_generator_is_not_reachable_by_the_obvious_derivation() -> None:
    """The other half of the claim, on the cases a reader would try.

    A generator earns its place by denoting a set the others do not reach, which
    is not a property a test can settle in general -- it is a statement about
    every combination. What a test can do is refuse the *specific* derivations
    that would make one redundant, so a variant does not sit in the generator
    column because nobody tried.
    """
    # A sequence is not a union of the sets its elements come from, and a set is
    # not its element type: the container is part of the value.
    assert not Validator(list[int]).is_equivalent(int)
    assert not Validator(set[int]).is_equivalent(list[int])
    assert not Validator(set[int]).is_equivalent(frozenset[int])
    # A record is not a mapping over the union of its field types.
    assert not Validator({"a": int}).is_equivalent(dict[str, int])
    # A refinement is not its base, and a bounded interval is not the union of
    # the literals inside it -- the second is a set the algebra reaches and the
    # first is not.
    non_empty = Validator(Annotated[str, at.MinLen(1)])
    assert not non_empty.is_equivalent(str)
    assert Validator(Annotated[int, at.Ge(0), at.Le(1)]).is_equivalent(
        union(Literal[0], Literal[1], Literal[True], Literal[False])
    )
    # The top is not any one kind, and the classes are not the scalars.
    assert not Validator(anything).is_equivalent(int)
    assert not Validator(int).is_equivalent(str)
