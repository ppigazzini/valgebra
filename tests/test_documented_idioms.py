"""Two claims the pages make, measured rather than asserted.

Both concern a reader arriving at the wrong conclusion for want of a sentence:
how much of the depth budget a refinement spends, and how to say that some keys
are constrained while the rest are free. Neither is a defect, so nothing else in
the suite would notice if the pages stopped saying it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import annotated_types as at
import pytest

from valgebra import Validator, anything, complement

_DOCS = Path(__file__).resolve().parent.parent / "docs"

# Far past any bound, so a shape that never fails is a failed test rather than a
# hung one.
_CEILING = 400


def _first_refused_depth(wrap: Any) -> int:
    """Return the depth at which the frontend first refuses ``wrap``'s chain."""
    for depth in range(1, _CEILING):
        schema: Any = int
        for _ in range(depth):
            schema = wrap(schema)
        try:
            Validator(schema)
        except ValueError:
            return depth
    pytest.fail(f"no depth below {_CEILING} was refused")
    return _CEILING  # pragma: no cover - pytest.fail does not return


def test_a_refinement_spends_a_level_of_the_depth_budget() -> None:
    """A refinement is a node, so it costs a level exactly as a container does.

    `{ x in [[base]] | constraints }` is a different set from `[[base]]`, which
    is why it is a node rather than a decoration on one. The consequence a reader
    meets is that pinning a length on a nested list gives up a third of the
    budget, and the marker chosen has nothing to do with it.
    """
    plain = _first_refused_depth(lambda t: list[t])
    refined = _first_refused_depth(lambda t: Annotated[list[t], at.Len(1, 1)])

    # A list costs two levels per wrapping; the refinement makes it three.
    assert plain == 64
    assert refined == 43

    # The marker is irrelevant — the level belongs to the node, not the class.
    assert _first_refused_depth(lambda t: Annotated[list[t], at.MinLen(1)]) == refined


def test_the_limits_page_names_the_refinement() -> None:
    """A budget a reader can exhaust is one the page accounts for."""
    page = (_DOCS / "10-limits.md").read_text(encoding="utf-8")
    assert "refinement" in page.lower(), (
        "docs/10-limits.md explains the list and tuple multiplier but not the "
        "refinement, so a reader who pins a length has nothing to attribute "
        "the lost depth to"
    )


def test_a_map_can_constrain_some_keys_and_free_the_rest() -> None:
    """The permissive clause is the complement of the constrained keys.

    `open` admits a clause matching *every* key, which subsumes a narrower one
    and frees everything. Taking the complement instead leaves the two clauses
    disjoint, so the disjunction never widens the keys that were constrained.
    """
    partly_open = Validator(
        {"name": str, str: int, complement(Validator(str)): anything}
    )

    assert partly_open.is_valid({"name": "a"})
    assert partly_open.is_valid({"name": "a", "count": 1})
    assert partly_open.is_valid({"name": "a", 7: object()})  # no clause claims it
    assert not partly_open.is_valid({"name": "a", "count": "not an int"})

    # What `open` does instead, for contrast: every key becomes free.
    fully_open = Validator({"name": str, str: int}).open()
    assert fully_open.is_valid({"name": "a", "count": "not an int"})


def test_the_schema_language_page_shows_the_idiom() -> None:
    """The composition is worth a page only if the page carries it."""
    page = (_DOCS / "03-schema-language.md").read_text(encoding="utf-8")
    assert "complement" in page, (
        "docs/03-schema-language.md states that clauses are a disjunction and that "
        "named fields take precedence, but never shows how to constrain some "
        "keys and leave the rest free"
    )
