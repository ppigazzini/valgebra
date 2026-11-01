"""Read a TOML file this repository owns, with the standard library's parser."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def load(path: Path) -> dict[str, Any]:
    """Parse `path` as TOML."""
    with path.open("rb") as handle:
        return tomllib.load(handle)
