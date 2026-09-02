"""Closure relations of the schema algebra.

Every assertion is true set-theoretically, and every one is a relation the
procedure decides or a fold the constructors perform. A true relation the
procedure declines belongs on the completeness ledger
(``tests/test_completeness_ledger.py``), where it is a strict expected failure
that fails the day it becomes decided; this file holds what does not need one.
"""

import enum
from typing import Any, Literal

from valgebra import Validator, anything, complement, intersection, nothing, union


def test_a_literal_is_disjoint_from_another_kind():
    assert Validator(Literal["a"]).is_subtype_of(complement(int))


def test_two_distinct_literals_share_no_value():
    assert intersection(Literal["a"], Literal["b"]).is_empty()


# --- A literal is a typed singleton, so its kind places it --------------------


def test_a_literal_int_is_disjoint_from_a_literal_bool():
    # `Literal[1]` requires `type(x) is int`, `Literal[True]` requires `bool`,
    # so they share no value even though `1 == True` in Python.
    assert intersection(Literal[1], Literal[True]).is_empty()


def test_an_enum_literal_meet_stays_conservative():
    # An enum member carries user-defined equality, so two of them may share a
    # value. The rule must decline rather than guess.
    class Colour(enum.Enum):
        RED = 1
        BLUE = 2

    assert not intersection(Literal[Colour.RED], Literal[Colour.BLUE]).is_empty()


def test_a_literal_is_still_a_member_of_its_own_kind():
    assert Validator(Literal["a"]).is_subtype_of(str)
    assert not Validator(Literal["a"]).is_subtype_of(int)


# --- Products decompose (JACM Lemma 6.5), and the shapes that must NOT --------


def test_a_product_splits_across_union_branches():
    assert Validator(tuple[int | str, int]).is_subtype_of(
        union(tuple[int, int], tuple[str, int])
    )


def test_a_product_splits_on_its_second_component():
    assert Validator(tuple[int, int | str]).is_subtype_of(
        union(tuple[int, int], tuple[int, str])
    )


def test_a_list_product_splits_too():
    assert Validator([int | str, int]).is_subtype_of(union([int, int], [str, int]))


def test_a_product_does_not_split_across_both_components():
    # (int, bytes) is in the left and in neither branch on the right.
    assert not Validator(tuple[int | str, int | bytes]).is_subtype_of(
        union(tuple[int, int], tuple[str, bytes])
    )


def test_a_product_is_not_below_branches_that_miss_it():
    assert not Validator(tuple[int, int]).is_subtype_of(
        union(tuple[str, int], tuple[int, str])
    )


def test_arity_does_not_mix():
    assert not Validator(tuple[int, int]).is_subtype_of(
        union(tuple[int], tuple[int, int, int])
    )


# --- The record meet, and the shapes it must NOT empty -----------------------


def test_a_record_is_below_the_complement_of_a_disjoint_record():
    assert Validator({"a": int}).is_subtype_of(complement({"a": str}))


def test_a_required_field_with_an_empty_meet_empties_the_record():
    assert intersection({"a": int}, {"a": str}).is_empty()


def test_required_on_one_side_is_enough():
    assert intersection({"a": int}, {"a?": str}).is_empty()


def test_two_optional_fields_do_not_empty_the_record():
    # The empty dict is in both, so the meet is inhabited.
    assert not intersection({"a?": int}, {"a?": str}).is_empty()


def test_two_pure_maps_never_meet_empty():
    # ICFP footnote 11: a meet of two mappings always contains `{}`.
    assert not intersection({str: int}, {str: str}).is_empty()


def test_a_compatible_meet_stays_inhabited():
    assert not intersection({"a": int}, {"a": bool}).is_empty()


def test_a_closed_record_admits_no_key_it_does_not_declare():
    # `{'a': int}` is closed, so no dict in it carries 'b'; `{'b': str}` requires
    # one. Nothing is in both.
    assert intersection({"a": int}, {"b": str}).is_empty()


def test_two_open_records_admit_each_other_s_keys():
    # Both carry a catch-all, so each admits the key the other requires and a
    # dict with both keys is in the meet.
    assert not intersection({"a": int, str: object}, {"b": str, str: object}).is_empty()


# --- The complement laws, settled where the schema is built -------------------
#
# `~~A` and `A | ~A` are folded by the constructors, so a comparison between the
# folded form and its operand asks the procedure nothing: the two are the same
# schema. What the fold does is the claim, so that is what is asserted; what the
# procedure decides when the fold does not reach a shape is on the completeness
# ledger.


def test_a_double_complement_is_the_schema_it_negates_twice():
    record = Validator({"a": int})
    assert complement(complement(record)) == record
    assert complement(complement(complement(complement(record)))) == record


def test_a_join_with_a_complement_is_the_top():
    assert union(list[int], complement(list[int])) == Validator(anything)


def test_the_fold_reaches_a_schema_built_through_the_constructors_only():
    # A respelling denotes the same set and is not folded, because the fold reads
    # structural equality; the procedure has no rule for the shape either, which
    # is the ledger entry.
    record = Validator({"a": int})
    respelled = union(record, complement(union(record, Validator(nothing))))
    assert respelled != Validator(anything)
    assert not Validator(anything).is_subtype_of(respelled)


def test_the_gradual_atom_is_exempt_from_the_fold():
    # `Any` is the dynamic type, not a set whose complement completes it.
    assert union(Any, complement(Any)) != Validator(anything)
    assert not Validator(anything).is_subtype_of(union(Any, complement(Any)))
