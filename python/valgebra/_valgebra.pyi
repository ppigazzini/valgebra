# Typed signatures for the compiled `valgebra._valgebra` extension.
#
# This stub carries the *types* of the public surface for type checkers and IDEs.
# The prose documentation lives on the compiled objects themselves (the Rust
# docstrings) and is rendered on the API reference page, which
# `scripts/docs_stubs.py` merges with this stub, so each has a single source
# and cannot drift from a hand-copied duplicate here. Import the public
# names from the top-level `valgebra` package, not from this module.
#
# Every parameter is positional-only, because the compiled functions take them
# that way: naming one in a call is a `TypeError` at runtime, and a stub that
# allowed it would type-check code that cannot run. The keyword-only `fail_fast`
# is one exception, and `__class_getitem__`'s `key`, which is PyO3's and takes a
# keyword too, is the other.

from collections.abc import Callable
from types import GenericAlias
from typing import Any, Generic, Literal, NoReturn, TypeGuard, final, overload

from typing_extensions import TypeVar, deprecated

# The set a validator denotes, as a checker reads it: `Validator[int]` is a
# validator whose members are ints. `Any` by default, so a bare `Validator` is
# the gradual reading and accepts every typed validator. Invariant, so the
# receiver overloads below can tell an untyped `Validator[object]` from a typed
# `Validator[int]`; a parameter meant to take any validator is written bare.
_T = TypeVar("_T", default=Any)
_S = TypeVar("_S")
_V = TypeVar("_V")

#: The distribution version, taken from the crate manifest the wheel is built
#: from rather than read back out of the installed metadata.
__version__: str
#: Whether this extension carries debug assertions, which is what separates a
#: `maturin develop` build from a release or PGO one. Read by the timing
#: harnesses, which refuse to quote a figure from a debug build.
_debug_build: bool

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
class Validator(Generic[_T]):
    # `__new__` rather than `__init__`: the compiled type builds the validator
    # in its constructor, so that is where the parameter is, and a stub that put
    # it on `__init__` would describe a signature the runtime does not have.
    #
    # The overloads say `object` wherever the static reading and the set part,
    # and a type only where every member of the set is a member of the type:
    # a compiled validator keeps its set; a class, parametrized or not, is at
    # most the instances it names; `None` is `None`. Anything else -- a union
    # written with `|`, a `Literal`, an `Annotated`, a native form, a constant --
    # is `object`, since no overload before `TypeForm` can read it as a type.
    @overload
    def __new__(cls, schema: Validator[_S], /) -> Validator[_S]: ...
    @overload
    def __new__(cls, schema: type[_S], /) -> Validator[_S]: ...
    @overload
    def __new__(cls, schema: None, /) -> Validator[None]: ...
    @overload
    def __new__(cls, schema: object, /) -> Validator[object]: ...
    # `Validator[int]` is a `types.GenericAlias` at runtime, which an annotation
    # evaluated at module level on 3.10 to 3.13 builds.
    def __class_getitem__(cls, key: object) -> GenericAlias: ...
    def validate(self, obj: object, /, *, fail_fast: bool = ...) -> None: ...
    # On an untyped validator the answer is a `bool`; on a typed one a `True`
    # narrows the argument to the set. `TypeGuard` rather than `TypeIs`: a
    # `False` narrows nothing, because this library's `float` refuses the `int`
    # a checker admits there.
    @overload
    def is_valid(self: Validator[object], obj: object, /) -> bool: ...
    @overload
    def is_valid(self, obj: object, /) -> TypeGuard[_T]: ...
    # The value-returning check returns its argument, since validation is a
    # membership test rather than a coercion: the argument's own type on an
    # untyped validator, the set's on a typed one.
    @overload
    def ensure(self: Validator[object], obj: _V, /) -> _V: ...
    @overload
    def ensure(self, obj: object, /) -> _T: ...
    def validate_json(self, data: str | bytes, /, *, fail_fast: bool = ...) -> None: ...
    @overload
    def load(
        self: Validator[object], data: str | bytes, /, *, fail_fast: bool = ...
    ) -> object: ...
    @overload
    def load(self, data: str | bytes, /, *, fail_fast: bool = ...) -> _T: ...
    def is_valid_json(self, data: str | bytes, /) -> bool: ...
    # `open` frees the key region no clause claims, so the opened set is not
    # within the annotation's; `close` and the copies keep or narrow it.
    def open(self) -> Validator[object]: ...
    def close(self) -> Validator[_T]: ...
    # A schema is built in the lattice normal form, so this returns the schema
    # the caller already holds plus a handful of folds the three relations
    # decide better. Removed in the next minor version.
    @deprecated("a schema is built in normal form; simplify folds nothing more")
    def simplify(self) -> Validator[_T]: ...
    def is_empty(self) -> bool: ...
    def is_subtype_of(self, other: object, /) -> bool: ...
    def relation_to(
        self, other: object, /
    ) -> Literal["subset", "not_subset", "undecided"]: ...
    def is_equivalent(self, other: object, /) -> bool: ...
    def __contains__(self, obj: object, /) -> bool: ...
    @overload
    def __or__(self, other: Validator[_S], /) -> Validator[_T | _S]: ...
    @overload
    def __or__(self, other: object, /) -> Validator[object]: ...
    @overload
    def __ror__(self, other: Validator[_S], /) -> Validator[_T | _S]: ...
    @overload
    def __ror__(self, other: object, /) -> Validator[object]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    # Raises `TypeError`: a validator holds the classes and callables its
    # schema names, so the schema is what travels. `NoReturn` rather than
    # `Never`, which the floor interpreter's `typing` does not carry.
    def __reduce__(self) -> NoReturn: ...
    def __copy__(self) -> Validator[_T]: ...
    def __deepcopy__(self, memo: object, /) -> Validator[_T]: ...

# A union reads as its members' type where a checker solves one -- validators of
# one type, for every checker -- and as `object` otherwise, which it always is
# when a member is not a validator. A meet and a complement are sets the static
# language does not spell, and a fixpoint is built by a call.
@overload
def union(*schemas: Validator[_S]) -> Validator[_S]: ...
@overload
def union(*schemas: object) -> Validator[object]: ...
def intersection(*schemas: object) -> Validator[object]: ...
def complement(schema: object, /) -> Validator[object]: ...
def recursive(builder: Callable[[Validator[Any]], object], /) -> Validator[object]: ...

anything: Validator[object]
# The bottom: `is_valid` never answers `True` and `ensure` never returns.
nothing: Validator[NoReturn]

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
    "__version__",
    "_debug_build",
    "anything",
    "complement",
    "intersection",
    "nothing",
    "recursive",
    "union",
]
