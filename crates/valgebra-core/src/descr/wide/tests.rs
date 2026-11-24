use num_bigint::BigInt;
use proptest::prelude::*;

use super::WideInt;

/// Integers on both sides of `i128`'s range and at its edges, where the two
/// spellings meet.
fn integer() -> impl Strategy<Value = BigInt> {
    let edge = || {
        prop_oneof![
            Just(BigInt::from(i128::MAX)),
            Just(BigInt::from(i128::MIN)),
            Just(BigInt::from(i128::MAX) + 1),
            Just(BigInt::from(i128::MIN) - 1),
        ]
    };
    prop_oneof![
        any::<i64>().prop_map(BigInt::from),
        any::<i128>().prop_map(BigInt::from),
        (edge(), -3_i64..=3).prop_map(|(at, by)| at + by),
        (any::<i128>(), any::<i128>(), any::<bool>()).prop_map(|(high, low, negative)| {
            let wide: BigInt = (BigInt::from(high) << 128) + BigInt::from(low);
            if negative { -wide } else { wide }
        }),
    ]
}

proptest! {
    /// The spelling is the integer's: one past `i128`'s range is `Big`, every
    /// other `Small`, so the derived equality is the integers'.
    #[test]
    fn an_integer_has_one_spelling(value in integer()) {
        let wide = WideInt::from_big(value.clone());
        prop_assert_eq!(matches!(wide, WideInt::Small(_)), i128::try_from(&value).is_ok());
        prop_assert_eq!(wide.to_big(), value);
    }

    /// The order is the integers' order, across the two spellings.
    #[test]
    fn the_order_is_the_integers(a in integer(), b in integer()) {
        prop_assert_eq!(WideInt::from_big(a.clone()).cmp(&WideInt::from_big(b.clone())), a.cmp(&b));
    }

    /// A step and a shift are exact, past the range as within it.
    #[test]
    fn a_shift_is_exact(value in integer(), offset in any::<i64>()) {
        let wide = WideInt::from_big(value.clone());
        prop_assert_eq!(wide.plus(offset).to_big(), &value + offset);
        prop_assert_eq!(wide.succ().to_big(), &value + 1);
        prop_assert_eq!(wide.pred().to_big(), &value - 1);
    }

    /// Division by a positive stride rounds down and up as the integers do:
    /// the floor is the greatest `q` with `q * stride <= value`, the ceiling
    /// the least with `q * stride >= value`.
    #[test]
    fn a_division_rounds_each_way(value in integer(), stride in 1_i64..=i64::MAX) {
        let wide = WideInt::from_big(value.clone());
        let floor = wide.div_floor(stride).to_big();
        let ceil = wide.div_ceil(stride).to_big();
        prop_assert!(&floor * stride <= value && value < (&floor + 1) * stride);
        prop_assert!(&ceil * stride >= value && value > (&ceil - 1) * stride);
    }

    /// A float is the integer it equals, and no other.
    #[test]
    fn a_float_is_the_integer_it_equals(float in any::<f64>()) {
        let whole = WideInt::from_float(float);
        prop_assert_eq!(
            whole.is_some(),
            float.is_finite() && float.trunc().total_cmp(&float).is_eq()
        );
        if let Some(whole) = whole {
            prop_assert!(whole.equals_float(float));
            prop_assert!(!whole.succ().equals_float(float));
        }
    }
}

/// The bits of a magnitude, at the edges of each spelling.
#[test]
fn the_bits_count_the_magnitude() {
    assert_eq!(WideInt::from(0_i64).bits(), 0);
    assert_eq!(WideInt::from(1_i64).bits(), 1);
    assert_eq!(WideInt::from(-1_i64).bits(), 1);
    assert_eq!(WideInt::from(i128::MIN).bits(), 128);
    assert_eq!(WideInt::from_big(BigInt::from(1) << 1024).bits(), 1025);
    assert_eq!(
        WideInt::from_float(1e100).map(|whole| whole.bits()),
        Some(333)
    );
}
