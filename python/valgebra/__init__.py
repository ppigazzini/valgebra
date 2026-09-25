"""Check that a Python object belongs to the set of values a schema denotes.

A schema is a typing annotation, or a combinator from this package where an
annotation cannot spell the set. `Validator` compiles one once into a Rust
validator tree, and every check is a membership test on the object the caller
holds: nothing is copied or coerced.
"""

from ._markers import Regex
from ._valgebra import (
    MAX_DEFINITIONS,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_NODES,
    ValidationError,
    Validator,
    __version__,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

__all__ = [
    "MAX_DEFINITIONS",
    "MAX_SCHEMA_DEPTH",
    "MAX_SCHEMA_NODES",
    "Regex",
    "ValidationError",
    "Validator",
    "__version__",
    "anything",
    "complement",
    "intersection",
    "nothing",
    "recursive",
    "union",
]
