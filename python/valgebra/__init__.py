from importlib.metadata import PackageNotFoundError, version

from ._markers import Regex
from ._valgebra import (
    MAX_DEFINITIONS,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_NODES,
    ValidationError,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

try:
    __version__ = version("valgebra")
except PackageNotFoundError:  # pragma: no cover - only when run uninstalled
    # The distribution metadata is absent only when the package runs from an
    # uninstalled source tree; the built wheel always carries it.
    __version__ = "0.0.0+unknown"

__all__ = [
    "MAX_DEFINITIONS",
    "MAX_SCHEMA_DEPTH",
    "MAX_SCHEMA_NODES",
    "Regex",
    "ValidationError",
    "Validator",
    "anything",
    "complement",
    "intersection",
    "nothing",
    "recursive",
    "union",
]
