"""The error model has to survive a process boundary.

`validate` raises `ValidationError`, and a worker that fails validation is
expected to deliver that failure to whatever started it. `pickle` locates a class
by its `__module__` and its qualified name, so an exception whose module string
names nothing cannot be serialized at all -- and a process pool, a task queue or
a subprocess test runner then reports a pickling error instead of the validation
result.

Nothing else in this suite exercises serialization, which is how a defect of that
shape stayed invisible while every other gate passed.
"""

from __future__ import annotations

import importlib
import pickle

import pytest

from valgebra import ValidationError, Validator


def _raised() -> ValidationError:
    """Raise a `ValidationError` carrying more than one item, the ordinary way."""
    with pytest.raises(ValidationError) as info:
        Validator({"a": int, "b": str}).validate({"a": "x", "b": 1})
    return info.value


def test_the_exception_type_is_importable_at_the_module_it_names() -> None:
    """`pickle` resolves a class by importing `__module__` and reading the name.

    This is the property the round trip below rests on, asserted separately so a
    failure says which half broke.
    """
    module = importlib.import_module(ValidationError.__module__)
    assert getattr(module, ValidationError.__name__) is ValidationError


def test_a_raised_error_survives_a_pickle_round_trip() -> None:
    error = _raised()
    restored = pickle.loads(  # noqa: S301 -- this test's own dumps, not input
        pickle.dumps(error)
    )

    assert isinstance(restored, ValidationError)
    assert str(restored) == str(error)
    # The structured model travels in the instance state, so every attribute the
    # error model documents has to arrive with it.
    assert restored.code == error.code
    assert restored.path == error.path
    assert restored.message == error.message
    assert restored.expected == error.expected
    assert restored.value == error.value
    assert restored.errors == error.errors


def test_the_aggregate_survives_and_stays_ordered() -> None:
    """Aggregation is the part a caller reads; order is part of the contract."""
    error = _raised()
    restored = pickle.loads(  # noqa: S301 -- this test's own dumps, not input
        pickle.dumps(error)
    )
    assert [item["path"] for item in restored.errors] == [("a",), ("b",)]
    assert [item["code"] for item in restored.errors] == ["int_type", "string_type"]
