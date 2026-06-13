"""The prefix-plus-tail and non-empty list forms.

A list schema `[A, ..., Z, ...]` denotes a fixed positional prefix followed by
zero or more elements matching the element before the trailing `...`. `[T, ...]`
is the prefix-free (homogeneous) case, and `[T, T, ...]` is the non-empty list.
"""

import pytest

from valgebra import ValidationError, validator


def test_prefix_then_repeated_tail() -> None:
    schema = validator([str, int, ...])  # a str, then zero or more ints
    assert schema.is_valid(["a"])
    assert schema.is_valid(["a", 1, 2, 3])
    assert not schema.is_valid([1])  # prefix must be a str
    assert not schema.is_valid([])  # the prefix is required
    assert not schema.is_valid(["a", "b"])  # the tail must be ints


def test_homogeneous_form_is_unchanged() -> None:
    schema = validator([int, ...])
    assert schema.is_valid([])
    assert schema.is_valid([1, 2, 3])
    assert not schema.is_valid(["a"])


def test_non_empty_list() -> None:
    schema = validator([int, int, ...])  # at least one int
    assert not schema.is_valid([])
    assert schema.is_valid([1])
    assert schema.is_valid([1, 2, 3])
    assert not schema.is_valid(["a"])


def test_too_short_reports_a_length_code() -> None:
    with pytest.raises(ValidationError) as info:
        validator([str, int, ...]).validate([])
    assert info.value.code == "list_length"
    assert "at least 1" in info.value.expected


def test_tail_element_failure_reports_its_index() -> None:
    with pytest.raises(ValidationError) as info:
        validator([str, int, ...]).validate(["a", 1, "b"])
    assert info.value.path == (2,)


def test_repr_round_trips_the_forms() -> None:
    assert repr(validator([str, int, ...])) == "[str, int, ...]"
    assert repr(validator([int, ...])) == "list[int]"
    assert repr(validator([int, int, ...])) == "[int, int, ...]"


def test_ellipsis_only_as_the_last_element() -> None:
    with pytest.raises(NotImplementedError):
        validator([int, ..., ...])
    with pytest.raises(NotImplementedError):
        validator([..., int])
