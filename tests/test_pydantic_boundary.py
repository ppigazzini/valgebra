"""Where valgebra's job and pydantic's stop being the same job.

pydantic owns parse-and-ingest: it turns untrusted input into a typed value and
guarantees the type of what it *returns*. valgebra owns check-and-contract: it
asks whether a value the caller already holds is a member of a set, and returns
a bool. The two are not ranked, and this module is not a benchmark -- every test
here passes or fails on a *verdict*, never on a duration.

What it pins down is the fragment where the difference is observable, so that
"different jobs" is a demonstrated claim rather than a slogan. Three groups:

- **The already-held object.** pydantic's default for an input that is already
  an instance of the target class is to pass it through without re-checking its
  fields, because for ingestion it has already done its work. That default is
  correct for its job and wrong for the case valgebra exists to serve, and it
  cannot be switched on from the adapter side for a class pydantic does not own.
- **The sets pydantic has no spelling for.** Negation is not in its schema
  language, so a set defined by exclusion has no model.
- **The questions about schemas.** Subtyping, equivalence and emptiness are
  relations between schemas rather than facts about a value. A model definition
  is not a value in an algebra, so pydantic has nowhere to put the question.

A fourth group states valgebra's own limit in the same currency, because a page
that only lists what the other tool cannot do is an advertisement:
``test_basemodel_is_an_isinstance_check`` records that a ``BaseModel`` reaches
valgebra as a bare class and is *not* deep-checked, and shows the mapping view
that is the actual bridge.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Literal, TypedDict

import pytest

from valgebra import Validator, complement, intersection, union

# The oracle lives in the optional ``bench`` dependency group, exactly as it does
# for the differential suite, and shares that suite's CI lane and flag: a missing
# pydantic there is a hard failure rather than a silently skipped gate, while
# locally the module skips cleanly when the group is absent.
_REQUIRED = os.environ.get("VALGEBRA_REQUIRE_DIFFERENTIAL") == "1"
try:
    import msgspec
    import msgspec.inspect
    import pydantic
    from msgspec import Struct
    from pydantic import BaseModel, ConfigDict, TypeAdapter
except ImportError:  # pragma: no cover - exercised by dependency presence, not branch
    if _REQUIRED:
        raise
    pytest.skip("the comparison oracles are not installed", allow_module_level=True)


@dataclass
class Config:
    """A plain stdlib dataclass, owned by neither library."""

    lr: float
    steps: int


def _unchecked(cls: type, **fields: object) -> object:
    """Build an instance whose fields bypass ``__init__``.

    A dataclass with a wrong-typed field is exactly what a long-lived object
    becomes after the code that mutates it gets something wrong, and it is the
    input both libraries are asked about below. Going through ``object.__new__``
    keeps the construction itself out of the comparison.
    """
    obj = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(obj, name, value)
    return obj


# --- The already-held object ------------------------------------------------


def test_ingestion_path_agrees() -> None:
    """On a mapping, both reject: this is the case pydantic is built for.

    The contrast in the next tests is not that pydantic misses a bad value; it
    is that the two libraries are asked *different questions*. Pinning the
    agreement first keeps that honest.
    """
    bad = {"lr": 0.1, "steps": "ten"}
    with pytest.raises(pydantic.ValidationError):
        TypeAdapter(Config).validate_python(bad)
    assert not Validator(Config).is_valid(bad)


def test_constructed_instance_is_rechecked_where_pydantic_passes_it_through() -> None:
    """A held instance is re-checked here; pydantic returns it unexamined.

    ``validate_python`` sees an instance of the target dataclass and, by
    default, treats it as already validated -- the wrong-typed ``steps`` reaches
    the caller unexamined, and the object it returns *is* the object it was
    given. valgebra asks the membership question the caller actually asked.
    """
    bad = _unchecked(Config, lr=0.1, steps="ten")

    returned = TypeAdapter(Config).validate_python(bad)
    assert returned is bad
    assert returned.steps == "ten"

    assert not Validator(Config).is_valid(bad)


def test_instance_revalidation_cannot_be_enabled_from_the_adapter() -> None:
    """The re-check switch belongs to the class, not to the adapter.

    ``revalidate_instances`` is a model config, so reaching it means pydantic
    must own the class definition. For a dataclass declared elsewhere -- a
    third-party type, or one that must stay a plain dataclass -- the adapter
    refuses the config outright rather than ignoring it.
    """
    with pytest.raises(pydantic.errors.PydanticUserError):
        TypeAdapter(Config, config=ConfigDict(revalidate_instances="always"))


def test_pydantic_rechecks_a_class_it_owns() -> None:
    """Given ownership of the class, pydantic does re-check. State it plainly.

    This is the other half of the previous test and the reason the difference is
    about *reach* rather than capability: the mechanism exists, and it is
    available exactly when the declaration is yours to change.
    """

    @pydantic.dataclasses.dataclass(config=ConfigDict(revalidate_instances="always"))
    class Owned:
        lr: float
        steps: int

    bad = _unchecked(Owned, lr=0.1, steps="ten")
    with pytest.raises(pydantic.ValidationError):
        TypeAdapter(Owned).validate_python(bad)


def test_mutation_after_construction_is_caught_on_the_next_check() -> None:
    """A value validated once and mutated later is invalid and unexamined.

    Construction-time validation states a fact about a moment. valgebra's check
    is a question you can ask again, so the same validator that admitted the
    object rejects it after the mutation without the object being rebuilt.
    """
    config = Config(lr=0.1, steps=10)
    is_config = Validator(Config)
    assert is_config.is_valid(config)

    config.steps = "ten"  # deliberately wrong: the mutation under test
    assert not is_config.is_valid(config)

    # pydantic's answer for a model it owns is validate_assignment, which moves
    # the check to the assignment rather than making it repeatable.
    class Owned(BaseModel):
        model_config = ConfigDict(validate_assignment=True)
        steps: int

    owned = Owned(steps=10)
    with pytest.raises(pydantic.ValidationError):
        owned.steps = "ten"  # deliberately wrong: the mutation under test


def test_membership_returns_a_bool_and_builds_nothing() -> None:
    """The check produces a verdict; the ingestion call produces a value.

    This is the semantic difference the performance page declines to read as a
    ranking: pydantic returns a new list because returning a validated value is
    its contract. valgebra has no output to build.
    """
    source = list(range(1000))

    assert TypeAdapter(list[int]).validate_python(source) is not source

    verdict = Validator(list[int]).is_valid(source)
    assert verdict is True


# --- Sets pydantic has no spelling for --------------------------------------


def test_complement_expresses_a_set_defined_by_exclusion() -> None:
    """Exclusion is a schema here and a Python predicate there.

    Negation is a generator of the algebra, so a set defined by what it excludes
    is a schema like any other -- composable, and comparable to the set it was
    carved out of. pydantic's schema language has no negation: the same set is
    reachable only by attaching a Python predicate to a ``str``, which the
    schema can carry and run but not reason about.
    """
    not_admin = intersection(str, complement(Literal["admin"]))
    assert not_admin.is_valid("bob")
    assert not not_admin.is_valid("admin")
    assert not not_admin.is_valid(1)

    # And the carved-out set relates to its carrier, strictly in one direction.
    assert not_admin.is_subtype_of(str)
    assert not Validator(str).is_subtype_of(not_admin)


def test_complement_of_a_record_is_a_schema() -> None:
    """Exclusion composes at any shape, not just at scalars."""
    not_a_point = complement({"x": int, "y": int})
    assert not_a_point.is_valid({"x": 1})
    assert not not_a_point.is_valid({"x": 1, "y": 2})


# --- Questions about schemas rather than values ------------------------------


def test_schemas_are_comparable_as_sets() -> None:
    """Inclusion, equivalence and emptiness are answered about the schemas.

    No value is involved in any of these three calls. They are relations between
    sets, decided soundly -- which is a question a model definition cannot be
    asked, because it is a declaration rather than a value in an algebra.
    """
    assert Validator(bool).is_subtype_of(int)
    assert union(bool, int).is_equivalent(int)
    assert intersection(int, complement(int)).is_empty()


def test_pydantic_exposes_no_such_relation() -> None:
    """Pin the absence against the adapter surface rather than asserting it.

    A ``TypeAdapter`` carries the schema, so it is where the question would live
    if it existed. Reading the surface keeps this test honest if pydantic ever
    grows one.
    """
    adapter = TypeAdapter(bool)
    for relation in ("is_subtype_of", "is_equivalent", "is_empty"):
        assert not hasattr(adapter, relation)


def test_an_unsatisfiable_contract_is_detected_before_any_value() -> None:
    """A contract no value can satisfy is a bug findable without a test case.

    ``is_empty`` decides it from the schema alone. Reached through values, the
    same bug is a test that passes for the wrong reason: every input is
    rejected, and nothing says the set was empty.

    The procedure is sound, not complete: it decides a wide fragment and answers
    ``False`` beyond it rather than guessing. Two disjoint literal sets are
    outside that fragment today, so this test uses disjoint scalar carriers,
    which are inside it.
    """
    impossible = intersection(int, str)
    assert impossible.is_empty()
    assert not impossible.is_valid(1)
    assert not impossible.is_valid("1")


# --- msgspec answers the same question from further away ---------------------


class MsgConfig(Struct):
    """The msgspec spelling of the same two fields."""

    lr: float
    steps: int


def test_msgspec_constructor_checks_nothing() -> None:
    """A ``Struct`` is built from whatever it is handed.

    msgspec's checking is on the decode path. The constructor is not on it, so
    the wrong-typed field reaches the object with no diagnostic at all -- there
    is no moment of validation to be after.
    """
    config = MsgConfig(lr=0.1, steps="ten")  # ty: ignore[invalid-argument-type]
    assert config.steps == "ten"


def test_msgspec_convert_returns_a_held_struct_unexamined() -> None:
    """``convert`` passes through an instance of the target type.

    This holds under ``strict`` and ``from_attributes`` alike: neither reaches
    the fields of a value that is already the right class. The argument that
    would change the answer does not exist, which is the difference from
    pydantic, where it exists and requires ownership of the declaration.
    """
    bad = MsgConfig(lr=0.1, steps="ten")  # ty: ignore[invalid-argument-type]
    for kwargs in ({}, {"strict": True}, {"from_attributes": True}):
        assert msgspec.convert(bad, type=MsgConfig, **kwargs) is bad


def test_msgspec_checks_on_the_decode_path() -> None:
    """The path msgspec owns rejects the same value, from untyped input."""
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"lr": 0.1, "steps": "ten"}, type=MsgConfig)
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"lr":0.1,"steps":"ten"}', type=MsgConfig)


def test_msgspec_reaches_a_held_value_through_a_round_trip() -> None:
    """The bridge exists, and it dismantles the value to cross.

    ``to_builtins`` does not check on the way out -- it accepts the wrong-typed
    field -- so the verdict comes from re-decoding the result. That is a check,
    reached by building two objects that are discarded.
    """
    bad = MsgConfig(lr=0.1, steps="ten")  # ty: ignore[invalid-argument-type]

    assert msgspec.to_builtins(bad) == {"lr": 0.1, "steps": "ten"}

    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(msgspec.to_builtins(bad), type=MsgConfig)


def test_neither_oracle_exposes_a_membership_predicate() -> None:
    """Read the surfaces rather than asserting the absence.

    The claim is that no call in either library answers "is this value a member
    of this type" without producing a value. Reading the module surface keeps
    the claim honest if either library grows one.
    """
    assert not [name for name in dir(msgspec) if name.startswith("is_")]
    assert not hasattr(msgspec, "validate")
    assert not hasattr(TypeAdapter(int), "is_valid")

    assert Validator(int).is_valid(1) is True


def test_msgspec_inspect_describes_a_type_without_relating_two() -> None:
    """``msgspec.inspect`` is introspection, not a decision procedure."""
    assert msgspec.inspect.type_info(MsgConfig).__class__.__name__ == "StructType"
    for relation in ("is_subtype_of", "is_equivalent", "is_empty"):
        assert not hasattr(msgspec.inspect, relation)


# --- valgebra's own limit at this boundary -----------------------------------


def test_basemodel_is_an_isinstance_check() -> None:
    """A ``BaseModel`` is not deep-checked by valgebra, and this records it.

    valgebra reads a dataclass, ``TypedDict`` and ``NamedTuple`` structurally,
    but a ``BaseModel`` is none of those: it reaches the frontend as a bare
    class, which denotes the set of its instances. So the wrong-typed field is
    admitted -- the schema is asking a question about the *class*, and the value
    is an instance of it. This is a gap, not a design.
    """

    class Model(BaseModel):
        steps: int

    model = Model(steps=10)
    model.__dict__["steps"] = "ten"  # bypass validate_assignment, which is off

    assert Validator(Model).is_valid(model)  # isinstance only

    # The bridge is a mapping view of the same data, which is deep-checked.
    class ModelFields(TypedDict):
        steps: int

    assert not Validator(ModelFields).is_valid(model.__dict__)


def test_dataclass_and_typeddict_are_deep_checked() -> None:
    """The contrast that makes the previous test a gap rather than a rule."""
    assert dataclasses.is_dataclass(Config)
    assert not Validator(Config).is_valid(_unchecked(Config, lr=0.1, steps="ten"))

    class Fields(TypedDict):
        lr: float
        steps: int

    assert not Validator(Fields).is_valid({"lr": 0.1, "steps": "ten"})
