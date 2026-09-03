//! Sets of floats: intervals over the ordered line, and a bit for `nan`.
//!
//! A float is not an integer with more digits, and the two ways it differs are
//! exactly what a set of floats has to carry.
//!
//! **`nan` is outside the order.** Every comparison with it is false, so it sits
//! in no interval and no interval excludes it. A bit beside the intervals is
//! what says whether the set holds it, and that bit is the reason
//! `Annotated[float, Ge(0)] | Annotated[float, Lt(0)]` is *not* `float`: the two
//! halves cover the whole ordered line and neither admits `nan`.
//!
//! **`-0.0` and `0.0` are one value.** They are two bit patterns that `==`
//! cannot tell apart, so a set that held one and not the other would be a set no
//! value can distinguish. Every endpoint is normalised on the way in, and the
//! negative zero never reaches the representation.
//!
//! The intervals carry their own inclusivity, because `Gt(0)` and `Ge(0)` differ
//! by one value and both are things a caller writes. Endpoints are floats, so
//! the infinities are *values* rather than open ends: `[-inf, inf]` is every
//! float but `nan`.

/// One interval of floats, with each end open or closed.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Span {
    lo: f64,
    lo_closed: bool,
    hi: f64,
    hi_closed: bool,
}

/// Reflexive for the same reason [`FloatSet`]'s is: an endpoint is never `nan`.
impl Eq for Span {}

/// A stable order on the representation, which is what [`FloatSet`]'s is built
/// from.
///
/// `total_cmp` rather than `partial_cmp`, so the order is total without an arm
/// that cannot happen, and it agrees with equality: normalisation leaves no
/// `-0.0`, so two endpoints compare equal here exactly when `==` says so.
impl Ord for Span {
    fn cmp(&self, other: &Span) -> core::cmp::Ordering {
        self.lo
            .total_cmp(&other.lo)
            .then(self.lo_closed.cmp(&other.lo_closed))
            .then(self.hi.total_cmp(&other.hi))
            .then(self.hi_closed.cmp(&other.hi_closed))
    }
}

impl PartialOrd for Span {
    fn partial_cmp(&self, other: &Span) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Span {
    /// The interval between two normalised endpoints.
    fn new(lo: f64, lo_closed: bool, hi: f64, hi_closed: bool) -> Span {
        Span {
            lo: normalise(lo),
            lo_closed,
            hi: normalise(hi),
            hi_closed,
        }
    }

    /// Whether this interval holds no float: the ends crossed, or they met at a
    /// point that at least one of them excludes.
    fn is_empty(self) -> bool {
        match self.lo.total_cmp(&self.hi) {
            core::cmp::Ordering::Greater => true,
            core::cmp::Ordering::Equal => !(self.lo_closed && self.hi_closed),
            core::cmp::Ordering::Less => false,
        }
    }

    fn holds(self, value: f64) -> bool {
        let above = if self.lo_closed {
            self.lo <= value
        } else {
            self.lo < value
        };
        let below = if self.hi_closed {
            value <= self.hi
        } else {
            value < self.hi
        };
        above && below
    }

    /// The floats in both intervals.
    fn intersect(self, other: Span) -> Span {
        // At a shared endpoint the *stricter* inclusivity wins, which is what
        // makes `[0, 1] & (0, 2]` start open rather than closed.
        let (lo, lo_closed) = match self.lo.total_cmp(&other.lo) {
            core::cmp::Ordering::Less => (other.lo, other.lo_closed),
            core::cmp::Ordering::Greater => (self.lo, self.lo_closed),
            core::cmp::Ordering::Equal => (self.lo, self.lo_closed && other.lo_closed),
        };
        let (hi, hi_closed) = match self.hi.total_cmp(&other.hi) {
            core::cmp::Ordering::Less => (self.hi, self.hi_closed),
            core::cmp::Ordering::Greater => (other.hi, other.hi_closed),
            core::cmp::Ordering::Equal => (self.hi, self.hi_closed && other.hi_closed),
        };
        Span {
            lo,
            lo_closed,
            hi,
            hi_closed,
        }
    }

    /// Whether `other` starts before this interval ends, or exactly where it
    /// ends with at least one of the two holding that point.
    ///
    /// The float line has no successor, so two intervals meeting at a point are
    /// one interval when either side holds it -- `[0, 1]` and `(1, 2]` are
    /// `[0, 2]` -- and two intervals when neither does, since the point itself
    /// is then a hole.
    fn reaches(self, other: Span) -> bool {
        match self.hi.total_cmp(&other.lo) {
            core::cmp::Ordering::Greater => true,
            core::cmp::Ordering::Equal => self.hi_closed || other.lo_closed,
            core::cmp::Ordering::Less => false,
        }
    }
}

/// `-0.0` read as the value it is equal to.
///
/// The two zeros are one value under `==`, so only one of them may reach the
/// representation: otherwise a set holding `0.0` would answer differently about
/// `-0.0`, which no float can tell apart from it.
#[allow(
    clippy::float_cmp,
    reason = "the comparison is the point: the two zeros are equal under `==` \
              and differ under `total_cmp`, and it is the first that decides \
              which values a set can tell apart"
)]
fn normalise(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// A set of floats.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatSet {
    /// Sorted, disjoint, non-touching intervals over the ordered floats.
    spans: Vec<Span>,
    /// Whether the set holds `nan`, which no interval can say.
    nan: bool,
}

/// Reflexive because `nan` never reaches a span: the endpoints are normalised
/// and the one value that is not equal to itself lives in the bit beside them,
/// where it is an ordinary `bool`.
impl Eq for FloatSet {}

/// Ordered so a float set can be a *guard*, which the sequence automaton sorts
/// to reach a canonical table.
///
/// The order is on the representation, not the sets -- no order on sets is
/// wanted here, only a stable one. It agrees with equality, which is what a
/// sort needs of it: `total_cmp` separates two endpoints exactly when `==`
/// does, because normalisation leaves no `-0.0` and no span holds `nan`.
impl Ord for FloatSet {
    fn cmp(&self, other: &FloatSet) -> core::cmp::Ordering {
        self.spans.cmp(&other.spans).then(self.nan.cmp(&other.nan))
    }
}

impl PartialOrd for FloatSet {
    fn partial_cmp(&self, other: &FloatSet) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl FloatSet {
    /// The empty set.
    #[must_use]
    pub fn empty() -> FloatSet {
        FloatSet {
            spans: Vec::new(),
            nan: false,
        }
    }

    /// Every float, `nan` included.
    #[must_use]
    pub fn all() -> FloatSet {
        FloatSet {
            spans: vec![Span::new(f64::NEG_INFINITY, true, f64::INFINITY, true)],
            nan: true,
        }
    }

    /// Just `nan`.
    #[must_use]
    pub fn nan() -> FloatSet {
        FloatSet {
            spans: Vec::new(),
            nan: true,
        }
    }

    /// The one-element set, or the empty set for `nan`.
    ///
    /// `Literal[c]` denotes the values equal to `c`, and `nan` is equal to
    /// nothing -- itself included. So a literal `nan` admits no value at all,
    /// which is a fact about Python's equality rather than a choice made here.
    #[must_use]
    pub fn just(value: f64) -> FloatSet {
        if value.is_nan() {
            FloatSet::empty()
        } else {
            FloatSet::from_span(Span::new(value, true, value, true))
        }
    }

    /// The floats at or above `bound`, or none where `bound` is `nan`.
    #[must_use]
    pub fn at_least(bound: f64) -> FloatSet {
        FloatSet::ordered(bound, true, f64::INFINITY, true)
    }

    /// The floats strictly above `bound`.
    #[must_use]
    pub fn above(bound: f64) -> FloatSet {
        FloatSet::ordered(bound, false, f64::INFINITY, true)
    }

    /// The floats at or below `bound`.
    #[must_use]
    pub fn at_most(bound: f64) -> FloatSet {
        FloatSet::ordered(f64::NEG_INFINITY, true, bound, true)
    }

    /// The floats strictly below `bound`.
    #[must_use]
    pub fn below(bound: f64) -> FloatSet {
        FloatSet::ordered(f64::NEG_INFINITY, true, bound, false)
    }

    /// An ordered interval, empty where either end is `nan`.
    ///
    /// Every comparison with `nan` is false, so a bound of `nan` admits no
    /// float: `Annotated[float, Ge(float("nan"))]` is the empty set, and reading
    /// it as an unbounded end would make it every float instead.
    fn ordered(lo: f64, lo_closed: bool, hi: f64, hi_closed: bool) -> FloatSet {
        if lo.is_nan() || hi.is_nan() {
            return FloatSet::empty();
        }
        FloatSet::from_span(Span::new(lo, lo_closed, hi, hi_closed))
    }

    fn from_span(span: Span) -> FloatSet {
        if span.is_empty() {
            FloatSet::empty()
        } else {
            FloatSet {
                spans: vec![span],
                nan: false,
            }
        }
    }

    /// Whether this set holds no float.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty() && !self.nan
    }

    /// Whether this set holds `value`.
    #[must_use]
    pub fn holds(&self, value: f64) -> bool {
        if value.is_nan() {
            return self.nan;
        }
        let value = normalise(value);
        self.spans.iter().any(|span| span.holds(value))
    }

    /// The floats in either set.
    #[must_use]
    pub fn union(&self, other: &FloatSet) -> FloatSet {
        FloatSet {
            spans: self.spans.iter().chain(&other.spans).copied().collect(),
            nan: self.nan || other.nan,
        }
        .canonical()
    }

    /// The floats in both sets.
    #[must_use]
    pub fn intersect(&self, other: &FloatSet) -> FloatSet {
        let mut spans = Vec::new();
        for mine in &self.spans {
            for theirs in &other.spans {
                let met = mine.intersect(*theirs);
                if !met.is_empty() {
                    spans.push(met);
                }
            }
        }
        FloatSet {
            spans,
            nan: self.nan && other.nan,
        }
        .canonical()
    }

    /// Every float this set does not hold.
    ///
    /// The ordered part complements by reading the gaps, each taking the
    /// inclusivity the neighbouring end did not; the `nan` bit flips on its own,
    /// which is what keeps the two halves independent.
    #[must_use]
    pub fn complement(&self) -> FloatSet {
        let mut spans = Vec::new();
        let mut lo = f64::NEG_INFINITY;
        let mut lo_closed = true;
        for span in &self.spans {
            let gap = Span {
                lo,
                lo_closed,
                hi: span.lo,
                hi_closed: !span.lo_closed,
            };
            if !gap.is_empty() {
                spans.push(gap);
            }
            lo = span.hi;
            lo_closed = !span.hi_closed;
        }
        let tail = Span {
            lo,
            lo_closed,
            hi: f64::INFINITY,
            hi_closed: true,
        };
        if !tail.is_empty() {
            spans.push(tail);
        }
        FloatSet {
            spans,
            nan: !self.nan,
        }
        .canonical()
    }

    /// Sort, drop the empties, and merge what meets.
    fn canonical(self) -> FloatSet {
        let mut spans: Vec<Span> = self.spans.into_iter().filter(|s| !s.is_empty()).collect();
        // A closed end sorts before an open one at the same point, so a merge
        // never has to look backwards for the wider start.
        spans.sort_by(|a, b| a.lo.total_cmp(&b.lo).then(b.lo_closed.cmp(&a.lo_closed)));
        let mut merged: Vec<Span> = Vec::with_capacity(spans.len());
        for span in spans {
            match merged.last_mut() {
                Some(last) if last.reaches(span) => {
                    let wider = match last.hi.total_cmp(&span.hi) {
                        core::cmp::Ordering::Less => (span.hi, span.hi_closed),
                        core::cmp::Ordering::Greater => (last.hi, last.hi_closed),
                        core::cmp::Ordering::Equal => (last.hi, last.hi_closed || span.hi_closed),
                    };
                    last.hi = wider.0;
                    last.hi_closed = wider.1;
                }
                _ => merged.push(span),
            }
        }
        FloatSet {
            spans: merged,
            nan: self.nan,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FloatSet;
    use proptest::prelude::*;

    /// The floats a law is checked over.
    ///
    /// Both zeros, both infinities, `nan`, and a point strictly inside every
    /// gap the generator's endpoints leave -- including the two unbounded ones,
    /// where a set like `(-inf, -2.0)` is inhabited by values no endpoint names.
    /// The three that make floats their own case are all here: a law that held
    /// over ordinary finite values alone would miss every one of them.
    fn universe() -> Vec<f64> {
        vec![
            f64::NEG_INFINITY,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            f64::INFINITY,
            f64::NAN,
        ]
    }

    /// Whether two sets hold the same floats. `nan` compares by membership, not
    /// by equality, which is the whole reason it needs a bit of its own.
    fn same(a: &FloatSet, b: &FloatSet) -> bool {
        universe().into_iter().all(|f| a.holds(f) == b.holds(f))
    }

    /// Sets built from endpoints inside the universe, so agreement on it is
    /// agreement everywhere.
    fn float_set() -> impl Strategy<Value = FloatSet> {
        let point = prop_oneof![
            Just(-2.0f64),
            Just(-1.0),
            Just(-0.0),
            Just(0.0),
            Just(1.0),
            Just(2.0),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
        ];
        let leaf = prop_oneof![
            Just(FloatSet::empty()),
            Just(FloatSet::all()),
            Just(FloatSet::nan()),
            point.clone().prop_map(FloatSet::just),
            point.clone().prop_map(FloatSet::at_least),
            point.clone().prop_map(FloatSet::above),
            point.clone().prop_map(FloatSet::at_most),
            point.prop_map(FloatSet::below),
        ];
        leaf.prop_recursive(4, 24, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a.union(&b)),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a.intersect(&b)),
                inner.prop_map(|a| a.complement()),
            ]
        })
    }

    proptest! {
        // A bounded shrink, so a broken invariant cannot turn a caught mutation
        // into a run that outlasts a sweep: every draw is larger under one, and
        // shrinking a counterexample redraws it thousands of times.
        #![proptest_config(ProptestConfig {
            max_shrink_time: 2_000,
            ..ProptestConfig::default()
        })]

        /// The order is total and agrees with equality, which is all a sort of
        /// guards asks of it. Written by hand here because an endpoint is a
        /// float, so it is the one component whose order needs a law.
        #[test]
        fn the_order_is_total_and_agrees_with_equality(a in float_set(), b in float_set()) {
            prop_assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
            // The spans carry the order the set's is built from, and a partial
            // order that declined a pair would leave the set's undefined there.
            prop_assert_eq!(a.spans.partial_cmp(&b.spans), Some(a.spans.cmp(&b.spans)));
            prop_assert_eq!(a.cmp(&b) == core::cmp::Ordering::Equal, a == b);
            prop_assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
        }

        /// The Boolean algebra, checked against the floats.
        #[test]
        fn the_lattice_laws_hold_of_the_floats(
            a in float_set(),
            b in float_set(),
            c in float_set(),
        ) {
            prop_assert!(same(&a.union(&b), &b.union(&a)));
            prop_assert!(same(&a.intersect(&b), &b.intersect(&a)));
            prop_assert!(same(&a.union(&b).union(&c), &a.union(&b.union(&c))));
            prop_assert!(same(
                &a.intersect(&b).intersect(&c),
                &a.intersect(&b.intersect(&c))
            ));
            prop_assert!(same(&a.union(&a.intersect(&b)), &a));
            prop_assert!(same(&a.intersect(&a.union(&b)), &a));
            prop_assert!(same(
                &a.intersect(&b.union(&c)),
                &a.intersect(&b).union(&a.intersect(&c))
            ));
        }

        /// The complement laws, and De Morgan both ways. The `nan` bit rides
        /// along: it is an ordinary two-element algebra beside the intervals.
        #[test]
        fn the_complement_laws_hold_of_the_floats(a in float_set(), b in float_set()) {
            prop_assert!(a.intersect(&a.complement()).is_empty());
            prop_assert!(same(&a.union(&a.complement()), &FloatSet::all()));
            prop_assert!(same(&a.complement().complement(), &a));
            prop_assert!(same(
                &a.union(&b).complement(),
                &a.complement().intersect(&b.complement())
            ));
            prop_assert!(same(
                &a.intersect(&b).complement(),
                &a.complement().union(&b.complement())
            ));
        }

        /// Holding the same floats is being equal, which is what the merging and
        /// the endpoint normalisation are for.
        #[test]
        fn holding_the_same_floats_is_being_equal(a in float_set(), b in float_set()) {
            prop_assert_eq!(same(&a, &b), a == b);
        }

        /// Emptiness is a decision.
        #[test]
        fn emptiness_agrees_with_the_floats(a in float_set()) {
            prop_assert_eq!(a.is_empty(), universe().into_iter().all(|f| !a.holds(f)));
        }
    }

    /// The two halves of the ordered line do not cover `float`, and the reason
    /// is the one value that sits in neither.
    ///
    /// This is what the `nan` bit exists to say. Without it the two halves would
    /// be the whole kind, and `float` would be decided equal to their union --
    /// a claim no float supports, because `nan` is a float and is in neither.
    #[test]
    fn the_ordered_halves_leave_nan_outside() {
        let halves = FloatSet::at_least(0.0).union(&FloatSet::below(0.0));
        assert!(!halves.holds(f64::NAN));
        assert!(halves.holds(0.0));
        assert!(halves.holds(f64::INFINITY));
        assert!(halves.holds(f64::NEG_INFINITY));
        assert_ne!(halves, FloatSet::all());
        // What is missing is exactly `nan`, and adding it closes the gap.
        assert_eq!(halves.union(&FloatSet::nan()), FloatSet::all());
        assert_eq!(halves.complement(), FloatSet::nan());
    }

    /// A literal `nan` admits nothing, because `nan` is equal to no value --
    /// itself included.
    #[test]
    fn a_literal_nan_is_the_empty_set() {
        assert!(FloatSet::just(f64::NAN).is_empty());
        assert!(!FloatSet::just(f64::NAN).holds(f64::NAN));
        // The set that *does* hold it is a different thing, and it is not a
        // literal: `float` holds `nan` because `nan` is a float.
        assert!(FloatSet::nan().holds(f64::NAN));
        assert!(FloatSet::all().holds(f64::NAN));
        // A bound of `nan` admits nothing either: every comparison with it is
        // false, so no float is at least it.
        assert!(FloatSet::at_least(f64::NAN).is_empty());
        assert!(FloatSet::below(f64::NAN).is_empty());
    }

    /// The two zeros are one value, so no set can hold one without the other.
    #[test]
    fn the_two_zeros_are_one_value() {
        assert_eq!(FloatSet::just(-0.0), FloatSet::just(0.0));
        assert!(FloatSet::just(0.0).holds(-0.0));
        assert!(FloatSet::just(-0.0).holds(0.0));
        // A strict bound at either zero excludes both, since they are the same
        // point on the line.
        assert!(!FloatSet::above(-0.0).holds(0.0));
        assert!(!FloatSet::above(0.0).holds(-0.0));
        assert_eq!(FloatSet::above(-0.0), FloatSet::above(0.0));
        assert_eq!(FloatSet::at_least(-0.0), FloatSet::at_least(0.0));
    }

    /// An open and a closed end differ by one value, which is the reason the
    /// inclusivity is carried rather than rounded to the nearest float.
    #[test]
    fn an_open_end_excludes_exactly_its_own_point() {
        assert!(FloatSet::at_least(1.0).holds(1.0));
        assert!(!FloatSet::above(1.0).holds(1.0));
        assert!(FloatSet::above(1.0).holds(1.5));
        // The two differ by the point alone.
        assert_eq!(
            FloatSet::at_least(1.0).intersect(&FloatSet::above(1.0).complement()),
            FloatSet::just(1.0)
        );
        // Meeting at a point that either side holds is one interval; meeting at
        // one that neither holds leaves a hole.
        assert_eq!(
            FloatSet::at_most(1.0).union(&FloatSet::above(1.0)),
            FloatSet::all().intersect(&FloatSet::nan().complement())
        );
        let holed = FloatSet::below(1.0).union(&FloatSet::above(1.0));
        assert!(!holed.holds(1.0));
        assert!(holed.holds(0.5) && holed.holds(1.5));
    }

    /// The infinities are floats, not open ends: a set can hold them, exclude
    /// them, and be bounded by them.
    #[test]
    fn the_infinities_are_values_of_the_set() {
        assert!(FloatSet::at_least(f64::NEG_INFINITY).holds(f64::NEG_INFINITY));
        assert!(FloatSet::at_least(f64::NEG_INFINITY).holds(f64::INFINITY));
        assert!(!FloatSet::above(f64::NEG_INFINITY).holds(f64::NEG_INFINITY));
        assert!(!FloatSet::below(f64::INFINITY).holds(f64::INFINITY));
        assert!(
            !FloatSet::just(f64::INFINITY)
                .complement()
                .holds(f64::INFINITY)
        );
        // The whole ordered line, reached from both ends, is everything but
        // `nan`.
        let line = FloatSet::at_least(f64::NEG_INFINITY);
        assert_eq!(
            line,
            FloatSet::all().intersect(&FloatSet::nan().complement())
        );
        assert_eq!(line.complement(), FloatSet::nan());
    }
}
