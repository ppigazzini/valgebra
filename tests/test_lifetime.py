"""A validator does not keep alive what it merely describes.

A schema names classes, enum members and predicates, and the validator holds a
reference to each so they cannot be collected under a walk. That makes the
natural spelling a reference cycle -- a class that keeps its own validator holds
the validator, and the validator holds the class -- and a cycle is only
collectable if the collector can *see* both edges.

It could not: the type was untracked, so every class that owned its validator
leaked, and a long-running process building validators per request grew without
bound. These hold the two halves of the fix, and the control that says the leak
was the cycle rather than the reference.
"""

from __future__ import annotations

import gc
import typing
import weakref
from typing import Annotated

import annotated_types as at

from valgebra import Validator


def _classes_alive(count: int, *, own_validator: bool) -> int:
    """Make `count` classes, optionally each holding its own validator."""
    gc.collect()
    watches: list[weakref.ref[type]] = []
    for _ in range(count):
        model = type("Model", (), {"__annotations__": {"a": int}})
        held = Validator(model)
        if own_validator:
            model.validator = held  # ty: ignore[unresolved-attribute]
        watches.append(weakref.ref(model))
        del model, held
    gc.collect()
    return sum(1 for watch in watches if watch() is not None)


def test_a_class_that_owns_its_validator_is_collected() -> None:
    assert _classes_alive(200, own_validator=True) == 0
    assert not gc.garbage, "a cycle the collector cannot break is uncollectable garbage"


def test_the_same_classes_without_the_cycle_are_collected_too() -> None:
    """The control: the leak was the cycle, not the reference."""
    assert _classes_alive(200, own_validator=False) == 0


def test_a_validator_is_tracked_by_the_collector() -> None:
    # The property the two above rest on: an untracked object is not examined,
    # so a cycle through one is never found however many times gc runs.
    assert gc.is_tracked(Validator(int))
    assert gc.is_tracked(Validator({"a": int}))


def test_a_predicate_reaching_back_to_its_class_is_collected() -> None:
    """The other way a validator reaches what made it: through the pool.

    A predicate is a Python callable held in the validator's constants pool, so
    a callable that closes over the class makes the same cycle one hop longer.

    `typing` memoises `Annotated[...]`, and that cache holds the marker, the
    callable and everything it closes over for the life of the process -- so
    this cycle stays alive whatever valgebra traverses. Clearing it is what
    leaves *this* claim under test rather than CPython's caching: without the
    traversal the classes stay alive after the clear too, which is the case the
    assertion distinguishes.
    """
    gc.collect()
    watches: list[weakref.ref[type]] = []
    for _ in range(100):
        model = type("Model", (), {"__annotations__": {"a": int}})

        def only_ints(value: object, held: type = model) -> bool:
            """Hold the class in a default, so the pool entry reaches it."""
            return isinstance(value, int)

        model.validator = Validator(  # ty: ignore[unresolved-attribute]
            Annotated[int, at.Predicate(only_ints)]
        )
        watches.append(weakref.ref(model))
        # The callable too: the last one made stays bound in this frame, and one
        # live default is one live class.
        del model, only_ints

    for cleanup in getattr(typing, "_cleanups", ()):
        cleanup()
    gc.collect()
    assert sum(1 for watch in watches if watch() is not None) == 0


def test_a_validator_can_be_weakly_referenced() -> None:
    """So a registry can hold one without keeping it alive."""
    validator = Validator({"a": int})
    watch = weakref.ref(validator)
    assert watch() is validator

    registry: weakref.WeakValueDictionary[str, Validator] = (
        weakref.WeakValueDictionary()
    )
    registry["row"] = validator
    assert registry["row"] is validator

    del validator
    gc.collect()
    assert watch() is None
    assert "row" not in registry


def test_a_live_validator_keeps_the_class_its_schema_names() -> None:
    """Traversal must not become a licence to free what the walk still reads."""
    model = type("Held", (), {"__annotations__": {"a": int}})
    watch = weakref.ref(model)
    validator = Validator(model)

    del model
    gc.collect()
    held = watch()
    assert held is not None, "the validator must keep what its schema names"
    assert validator.is_valid(held()), "and the schema still decides with it"
    del held

    del validator
    gc.collect()
    assert watch() is None, "and releases it when nothing else holds it"
