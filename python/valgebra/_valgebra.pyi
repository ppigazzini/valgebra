# Typed signatures for the compiled `valgebra._valgebra` extension.
#
# This stub carries the *types* of the public surface for type checkers and IDEs.
# The prose documentation lives on the compiled objects themselves (the Rust
# docstrings) and is rendered on the API reference page, so it has a single
# source and cannot drift from a hand-copied duplicate here. Import the public
# names from the top-level `valgebra` package, not from this module.
#
# Every parameter is positional-only, because the compiled functions take them
# that way: naming one in a call is a `TypeError` at runtime, and a stub that
# allowed it would type-check code that cannot run. The keyword-only `fail_fast`
# is the exception, and is written as one.

from collections.abc import Callable
from typing import NoReturn, TypeVar, final

_T = TypeVar("_T")

class ValidationError(Exception):
    # The structured, machine-readable error model. `errors` is a tuple of
    # per-failure items, each a JSON-serializable dict with the keys
    # `code`/`path`/`message`/`expected`/`value`; `json.dumps(err.errors)` is the
    # JSON output. The scalar attributes mirror the first item.
    #
    # An error built by hand reports no failures, so it carries the class
    # defaults -- empty strings and empty tuples -- rather than nothing at all.
    code: str
    path: tuple[str | int, ...]
    message: str
    expected: str
    value: str
    errors: tuple[dict[str, object], ...]

@final
class Validator:
    # `__new__` rather than `__init__`: the compiled type builds the validator
    # in its constructor, so that is where the parameter is, and a stub that put
    # it on `__init__` would describe a signature the runtime does not have.
    def __new__(cls, schema: object, /) -> Validator: ...
    def validate(self, obj: object, /, *, fail_fast: bool = ...) -> None: ...
    def is_valid(self, obj: object, /) -> bool: ...
    # The value-returning check is an identity: validation is a membership test
    # rather than a coercion, so the object that comes back is the one that went
    # in, and the annotation says so rather than widening it to `object`.
    def ensure(self, obj: _T, /) -> _T: ...
    def validate_json(self, data: str | bytes, /, *, fail_fast: bool = ...) -> None: ...
    def load(self, data: str | bytes, /, *, fail_fast: bool = ...) -> object: ...
    def is_valid_json(self, data: str | bytes, /) -> bool: ...
    def open(self) -> Validator: ...
    def close(self) -> Validator: ...
    # Deprecated: a schema is built in the lattice normal form, so this returns
    # the schema the caller already holds plus a handful of folds the three
    # relations decide better. Removed in the next minor version.
    def simplify(self) -> Validator: ...
    def is_empty(self) -> bool: ...
    def is_subtype_of(self, other: object, /) -> bool: ...
    def is_equivalent(self, other: object, /) -> bool: ...
    def __contains__(self, obj: object, /) -> bool: ...
    def __or__(self, other: object, /) -> Validator: ...
    def __ror__(self, other: object, /) -> Validator: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    # Raises `TypeError`: a validator holds the classes and callables its
    # schema names, so the schema is what travels. `NoReturn` rather than
    # `Never`, which the floor interpreter's `typing` does not carry.
    def __reduce__(self) -> NoReturn: ...
    def __copy__(self) -> Validator: ...
    def __deepcopy__(self, memo: object, /) -> Validator: ...

def union(*schemas: object) -> Validator: ...
def intersection(*schemas: object) -> Validator: ...
def complement(schema: object, /) -> Validator: ...
def recursive(builder: Callable[[Validator], object], /) -> Validator: ...

anything: Validator
nothing: Validator

MAX_SCHEMA_DEPTH: int
MAX_DEFINITIONS: int
MAX_SCHEMA_NODES: int

# The extension's own export list, which the package re-exports from. Written
# out rather than typed as `list[str]` so `stubtest` holds the stub to the
# module both ways: a name added to one and not the other fails here rather than
# passing quietly.
__all__ = [
    "MAX_DEFINITIONS",
    "MAX_SCHEMA_DEPTH",
    "MAX_SCHEMA_NODES",
    "ValidationError",
    "Validator",
    "anything",
    "complement",
    "intersection",
    "nothing",
    "recursive",
    "union",
]
