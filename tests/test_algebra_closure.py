"""R0: the closure relations the algebra must decide but does not.

Every assertion here is true set-theoretically. Each failure is a row in
`REPORT-30`; each is expected to fail until its milestone lands.
"""

import enum
from typing import Literal

from valgebra import Validator, complement, intersection


def test_a_literal_is_disjoint_from_another_kind():
    assert Validator(Literal["a"]).is_subtype_of(complement(int))


def test_two_distinct_literals_share_no_value():
    assert intersection(Literal["a"], Literal["b"]).is_empty()


# --- R4: products must decompose (JACM Lemma 6.5) ----------------------------


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


# --- R4: product decomposition, and the shapes it must NOT decide -------------
