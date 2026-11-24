//! An integer the descriptor holds exactly, whatever its width.
//!
//! A bound a schema names is an `int`, a `Fraction`, a `Decimal` or a `float`,
//! and Python compares each with an `int` exactly, so the integers a bound
//! admits end where the bound does: `Le(1e100)` admits every integer up to one
//! of 333 bits, and `Ge(2**121)` none below one of 122. An integer set's ends
//! are therefore as wide as the bounds it is built from.
//!
//! **A machine integer where one holds it, an arbitrary one past that.** The
//! ends a schema writes are almost all small, and the integer sets meet,
//! complement and divide them on every decision that reads a bounded `int`; a
//! `BigInt` at every end would allocate there. So an end is an `i128` wherever
//! one holds it and a shared `BigInt` only past `i128`'s range, the form kept
//! canonical so that one integer has one spelling: equality is the derived one,
//! and an end past the range orders against a machine one by its sign alone.
//! `num-bigint`, the arithmetic past the range, is the crate `jiter` and `PyO3`
//! already build for the binding; the wrapper is this module's, since no crate
//! holds the small case inline.

use core::cmp::Ordering;
use std::sync::Arc;

use num_bigint::{BigInt, Sign};
use num_traits::{FromPrimitive, ToPrimitive};

/// An integer of any width: `Small` wherever an `i128` holds it, `Big` only
/// past that range, so each integer has exactly one spelling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WideInt {
    /// An integer within `i128`'s range.
    Small(i128),
    /// An integer past `i128`'s range, shared rather than copied.
    Big(Arc<BigInt>),
}

impl WideInt {
    /// The integer `value`, spelled as the form requires: a machine integer
    /// where one holds it.
    #[must_use]
    pub fn from_big(value: BigInt) -> WideInt {
        match value.to_i128() {
            Some(small) => WideInt::Small(small),
            None => WideInt::Big(Arc::new(value)),
        }
    }

    /// The integer a float equals, or `None` for a float that is not one:
    /// fractional, infinite or `nan`. Every finite float past `2**53` is an
    /// integer, and it converts exactly.
    #[must_use]
    pub fn from_float(value: f64) -> Option<WideInt> {
        if !value.is_finite() || value.trunc().total_cmp(&value).is_ne() {
            return None;
        }
        BigInt::from_f64(value).map(WideInt::from_big)
    }

    /// This integer as a `BigInt`.
    #[must_use]
    pub fn to_big(&self) -> BigInt {
        match self {
            WideInt::Small(value) => BigInt::from(*value),
            WideInt::Big(value) => BigInt::clone(value),
        }
    }

    /// This integer as an `i128`, where one holds it.
    #[must_use]
    pub fn to_i128(&self) -> Option<i128> {
        match self {
            WideInt::Small(value) => Some(*value),
            WideInt::Big(_) => None,
        }
    }

    /// How many bits its magnitude takes: `0` for zero.
    #[must_use]
    pub fn bits(&self) -> u64 {
        match self {
            WideInt::Small(value) => u64::from(128 - value.unsigned_abs().leading_zeros()),
            WideInt::Big(value) => value.bits(),
        }
    }

    /// The integer after this one.
    #[must_use]
    pub fn succ(&self) -> WideInt {
        self.plus(1)
    }

    /// The integer before this one.
    #[must_use]
    pub fn pred(&self) -> WideInt {
        self.plus(-1)
    }

    /// This integer plus `offset`, exactly.
    #[must_use]
    pub fn plus(&self, offset: i64) -> WideInt {
        match self {
            WideInt::Small(value) => match value.checked_add(i128::from(offset)) {
                Some(sum) => WideInt::Small(sum),
                None => WideInt::from_big(BigInt::from(*value) + offset),
            },
            WideInt::Big(value) => WideInt::from_big(BigInt::clone(value) + offset),
        }
    }

    /// This integer divided by a positive `stride`, rounded towards negative
    /// infinity.
    #[must_use]
    pub fn div_floor(&self, stride: i64) -> WideInt {
        debug_assert!(stride > 0, "a stride is a positive step");
        match self {
            // Dividing by a stride of at least one never leaves the range.
            WideInt::Small(value) => WideInt::Small(value.div_euclid(i128::from(stride))),
            WideInt::Big(value) => {
                let stride = BigInt::from(stride);
                let quotient = BigInt::clone(value) / &stride;
                // `/` truncates towards zero, which is the floor only where
                // the division is exact or the dividend is not negative.
                let exact = &quotient * &stride == **value;
                WideInt::from_big(if value.sign() == Sign::Minus && !exact {
                    quotient - 1
                } else {
                    quotient
                })
            }
        }
    }

    /// This integer divided by a positive `stride`, rounded towards positive
    /// infinity.
    #[must_use]
    pub fn div_ceil(&self, stride: i64) -> WideInt {
        let floor = self.div_floor(stride);
        let back = match &floor {
            WideInt::Small(value) => value.checked_mul(i128::from(stride)).map(WideInt::Small),
            WideInt::Big(_) => None,
        }
        .unwrap_or_else(|| WideInt::from_big(floor.to_big() * stride));
        if back == *self { floor } else { floor.succ() }
    }

    /// Whether this integer equals a float under Python's `==`, which compares
    /// an `int` and a `float` exactly.
    #[must_use]
    pub fn equals_float(&self, float: f64) -> bool {
        WideInt::from_float(float).is_some_and(|whole| whole == *self)
    }
}

impl From<i64> for WideInt {
    fn from(value: i64) -> WideInt {
        WideInt::Small(i128::from(value))
    }
}

impl From<i128> for WideInt {
    fn from(value: i128) -> WideInt {
        WideInt::Small(value)
    }
}

impl From<u64> for WideInt {
    fn from(value: u64) -> WideInt {
        WideInt::Small(i128::from(value))
    }
}

impl From<bool> for WideInt {
    fn from(value: bool) -> WideInt {
        WideInt::Small(i128::from(value))
    }
}

impl Ord for WideInt {
    fn cmp(&self, other: &WideInt) -> Ordering {
        match (self, other) {
            (WideInt::Small(a), WideInt::Small(b)) => a.cmp(b),
            // A `Big` lies past `i128`'s range, so its sign places it.
            (WideInt::Small(_), WideInt::Big(big)) => match big.sign() {
                Sign::Minus => Ordering::Greater,
                _ => Ordering::Less,
            },
            (WideInt::Big(big), WideInt::Small(_)) => match big.sign() {
                Sign::Minus => Ordering::Less,
                _ => Ordering::Greater,
            },
            (WideInt::Big(a), WideInt::Big(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for WideInt {
    fn partial_cmp(&self, other: &WideInt) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests;
