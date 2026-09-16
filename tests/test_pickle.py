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

import base64
import importlib
import pickle
import subprocess
import sys
import textwrap

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


def test_the_error_crosses_a_real_process_boundary() -> None:
    """A round trip in one process shares the module already imported.

    `pickle.loads` in the same interpreter finds `ValidationError` in
    `sys.modules` and never imports anything, so a module string that names
    nothing resolves anyway and the round trip above passes. The failure this
    file exists to catch is exactly that, and only a second interpreter --
    which starts with nothing imported -- puts the question.

    A subprocess rather than `multiprocessing`: a forked child inherits the
    parent's modules, so it shares the same blind spot. This one is spawned
    with no valgebra imported, and unpickles the bytes the parent produced.
    """
    error = _raised()
    payload = base64.b64encode(pickle.dumps(error)).decode("ascii")
    script = textwrap.dedent(
        """
        import base64, pickle, sys
        assert "valgebra" not in sys.modules, "the child started with it loaded"
        restored = pickle.loads(base64.b64decode(sys.argv[1]))
        assert type(restored).__name__ == "ValidationError", type(restored)
        assert restored.code == "int_type", restored.code
        assert restored.path == ("a",), restored.path
        assert len(restored.errors) == 2, restored.errors
        print("ok")
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, "-c", script, payload],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_a_hand_built_error_crosses_the_same_boundary() -> None:
    """An error a caller constructs travels too, carrying nothing from a walk.

    A test runner that forwards failures builds one of these, and it takes the
    path the walk's own never does: no violation behind it, so the model is
    whatever the constructor was given.
    """
    built = ValidationError("nothing was checked")
    payload = base64.b64encode(pickle.dumps(built)).decode("ascii")
    script = textwrap.dedent(
        """
        import base64, pickle, sys
        restored = pickle.loads(base64.b64decode(sys.argv[1]))
        assert str(restored) == "nothing was checked", str(restored)
        print("ok")
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, "-c", script, payload],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
