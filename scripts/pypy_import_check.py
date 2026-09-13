"""Import the extension on PyPy and build the forms that reach a type object.

A published wheel exists for PyPy, and the symbols an extension may name there
are not the symbols CPython offers. PyPy implements the C API through `cpyext`,
which carries the ones the limited API defines and not every static type object
CPython exports: `Py_GenericAliasType` is one it does not carry. Naming such a
symbol links here and fails on PyPy at **import** -- before any schema is built,
with `undefined symbol` and nothing else to go on.

That is a *link* property, not a behavioural one, so this is a link check. It
imports the extension and builds the annotation forms whose compilation most
plausibly reaches a type object the limited API does not define -- the
parametrized generics, the legacy aliases they are told apart from, the unions
in both spellings, and the containers. What each form *means* is held by the
suites, on CPython, where they run against every interpreter the matrix carries.

Run by the `pypy import` job of the CI workflow, against a wheel that job has
just built and installed. It exits non-zero on the first form that fails, so the
job's log names the form rather than the file.
"""

from __future__ import annotations

import sys
import typing

import valgebra as vg


def main() -> int:
    print(f"valgebra {vg.__version__} imported on {sys.implementation.name}")

    # The pair the parametrized check is about: a *bare* legacy alias is the
    # class it aliases, and a parametrization with no arguments is not. Telling
    # them apart is what reached for the type object this check exists for.
    for bare, native in ((typing.List, list), (typing.Tuple, tuple)):
        if repr(vg.Validator(bare)) != repr(vg.Validator(native)):
            print(f"FAIL: {bare} does not read as {native.__name__}")
            return 1

    checks: list[tuple[str, bool]] = [
        ("int accepts", vg.Validator(int).is_valid(1)),
        ("int refuses", not vg.Validator(int).is_valid("x")),
        ("list[int] accepts", vg.Validator(list[int]).is_valid([1, 2])),
        ("list[int] refuses", not vg.Validator(list[int]).is_valid(["x"])),
        ("dict[str, int]", vg.Validator(dict[str, int]).is_valid({"a": 1})),
        ("tuple[int, str]", vg.Validator(tuple[int, str]).is_valid((1, "a"))),
        ("set[int]", vg.Validator(set[int]).is_valid({1})),
        ("typing.Union", vg.Validator(typing.Union[int, str]).is_valid("a")),
        ("PEP 604 union", vg.Validator(int | str).is_valid(1)),
        ("typing.Optional", vg.Validator(typing.Optional[int]).is_valid(None)),
        ("typing.Literal", vg.Validator(typing.Literal["a"]).is_valid("a")),
        ("typing.Any", vg.Validator(typing.Any).is_valid(object())),
        ("bare list", vg.Validator(list).is_valid([1, "x"])),
        ("json", vg.Validator(list[int]).is_valid_json("[1, 2]")),
        ("a relation", vg.Validator(bool).is_subtype_of(int)),
    ]
    for name, held in checks:
        if not held:
            print(f"FAIL: {name}")
            return 1

    print(f"{len(checks) + 2} forms built and imported cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
