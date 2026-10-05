"""TypedDicts and dataclasses whose annotations are strings.

`from __future__ import annotations` (PEP 563) makes every annotation in this
module a string, which is what a large share of real code does. It lives in its
own module because the effect is per-module and cannot be turned on inside a
function.

`Required` and `NotRequired` reach `typing` in 3.11 and `ReadOnly` in 3.13, and
the builder asks `typing` itself for each marker -- a qualifier the running
interpreter does not carry is not one it can be asked about. Each class below is
guarded at the release that spells it, and the tests reading it skip beneath
that floor.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated, TypedDict

import typing_extensions
from annotated_types import Ge

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required

    class Deferred(TypedDict):
        """A class whose qualifiers CPython cannot see.

        `__required_keys__` is computed from the annotations as written, and here
        they are strings, so every key lands in it however it is qualified.
        """

        a: int
        b: NotRequired[list[str]]
        c: Required[str]

    class DeferredTotalFalse(TypedDict, total=False):
        """The other direction: `total=False` with one key pulled back to required."""

        a: int
        b: Required[str]


if sys.version_info >= (3, 13):
    from typing import ReadOnly

    class DeferredReadOnly(TypedDict):
        """A qualifier wrapping a qualifier, with both hidden behind a string."""

        a: int
        d: ReadOnly[NotRequired[int]]


class DeferredAnnotated(TypedDict):
    """A qualifier inside `Annotated`, behind a string.

    The resolved hint is `Annotated[NotRequired[int], ...]`, whose origin is the
    qualified type rather than the qualifier, so a search that stopped at the
    `Annotated` layer read the key from `__required_keys__` -- here, required.
    `typing_extensions`' qualifiers, so the class is read on the floor too.
    """

    a: int
    b: Annotated[typing_extensions.NotRequired[int], Ge(0)]
    c: Annotated[typing_extensions.ReadOnly[typing_extensions.NotRequired[str]], "doc"]


class DeferredAnnotatedTotalFalse(TypedDict, total=False):
    """The other direction: `Annotated[Required[T], ...]` in a `total=False` class."""

    a: int
    b: Annotated[typing_extensions.Required[int], Ge(0)]


@dataclass
class DeferredRecord:
    x: int
    y: str = "given"
