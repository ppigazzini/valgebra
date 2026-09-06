"""What the compiled surface promises about itself.

The stub says these things, and `stubtest` holds it to the runtime's
signatures. What it cannot say is why: that a validator names the public
package rather than the extension under it, that it cannot be subclassed, that
every argument is positional, and that an error built by hand still carries the
structured model's attributes. Each is a property a caller can rely on, so each
is asserted here rather than left to the stub alone.
"""

from __future__ import annotations

import copy

import pytest

import valgebra
from valgebra import ValidationError, Validator


def test_a_validator_names_the_package_a_user_imports_it_from() -> None:
    # `repr(type(v))`, a pickle of a subclass, and every traceback read this. The
    # extension module is an implementation detail the API reference reserves the
    # right to rename, so it is not the name to bake in.
    assert Validator.__module__ == "valgebra"
    assert ValidationError.__module__ == "valgebra"
    assert getattr(valgebra, Validator.__qualname__) is Validator


def test_a_validator_cannot_be_subclassed() -> None:
    # Every method reads a schema this type built. A subclass overriding one
    # would be a validator whose answers are not the algebra's, and the type
    # system's `@final` says so only to a type checker.
    with pytest.raises(TypeError):
        type("Sub", (Validator,), {})  # ty: ignore[subclass-of-final-class]


@pytest.mark.parametrize(
    ("name", "keyword"),
    [("Validator", "schema"), ("complement", "schema"), ("recursive", "builder")],
)
def test_every_module_level_argument_is_positional(name: str, keyword: str) -> None:
    # The stub writes them with `/`, and a stub that allowed a keyword would
    # type-check a call the runtime rejects. Reached through `getattr` because a
    # type checker reading this file would reject the call before it ran, which
    # is the stub being right rather than the runtime being tested.
    with pytest.raises(TypeError):
        getattr(valgebra, name)(**{keyword: int})


@pytest.mark.parametrize(
    ("method", "keyword"),
    [
        ("is_valid", "obj"),
        ("ensure", "obj"),
        ("validate", "obj"),
        ("is_subtype_of", "other"),
        ("is_equivalent", "other"),
        ("is_valid_json", "data"),
        ("validate_json", "data"),
        ("load", "data"),
    ],
)
def test_every_method_argument_is_positional(method: str, keyword: str) -> None:
    with pytest.raises(TypeError):
        getattr(Validator(int), method)(**{keyword: int})


def test_fail_fast_is_keyword_only() -> None:
    # The one keyword the surface takes, and it cannot be passed positionally.
    # Reached through a name for the reason above: a type checker reading the
    # direct call rejects it, which is the stub agreeing rather than the runtime
    # being tested.
    validate = "validate"
    with pytest.raises(TypeError):
        getattr(Validator(int), validate)("x", True)


def test_ensure_returns_the_object_it_was_given() -> None:
    # Validation is a membership test rather than a coercion, which is why the
    # stub types this as an identity: what comes back is the object that went in,
    # not a copy of it and not a widened view of it.
    value = [1, 2, 3]
    assert Validator(list[int]).ensure(value) is value


def test_an_error_built_by_hand_carries_the_model_s_attributes() -> None:
    # The structured model describes failures, and this error reports none -- so
    # it reads as empty rather than raising `AttributeError` for an attribute the
    # stub declares unconditionally.
    error = ValidationError("boom")
    assert str(error) == "boom"
    assert error.code == ""
    assert error.message == ""
    assert error.expected == ""
    assert error.value == ""
    assert error.path == ()
    assert error.errors == ()


def test_a_raised_error_shadows_the_defaults_with_its_failure() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int).validate("x")
    assert info.value.code == "int_type"
    assert info.value.errors[0]["code"] == "int_type"


def test_a_copy_is_the_same_validator() -> None:
    schema = Validator({"a": int})
    assert copy.copy(schema) == schema
    assert copy.deepcopy(schema) == schema
