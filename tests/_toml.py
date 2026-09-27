"""Read a TOML file this repository owns, with a parser.

`tomllib` is in the standard library from 3.11. Below it, `tomli` is the same
parser under its own name, and pytest requires it there, so every interpreter
the suite runs on has one.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # the floor, where pytest's own dependency supplies the parser
    import tomli as tomllib

if TYPE_CHECKING:
    from pathlib import Path


def load(path: Path) -> dict[str, Any]:
    """Parse `path` as TOML."""
    with path.open("rb") as handle:
        return tomllib.load(handle)
