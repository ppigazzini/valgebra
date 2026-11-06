//! Sets of floats: closed intervals of floats, and a bit for `nan`.
//!
//! A float is not an integer with more digits, and the three ways it differs are
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
//! value can distinguish. Every endpoint is normalized on the way in, and the
//! negative zero never reaches the representation.
//!
//! **The floats are discrete.** Every float but the largest has a next one, and
//! nothing lies between the two: `Gt(1.0)` admits the floats from the one after
//! `1.0` up, which is what `Ge(math.nextafter(1.0, inf))` admits too. So an
//! interval holds both its ends, and an open end a caller writes is held as its
//! neighboring float. Two spellings of one set are then one interval, an
//! interval between two adjacent floats is empty, and two intervals with no float
//! between them are one. Endpoints are floats, so the infinities are *values*
//! rather than open ends: `[-inf, inf]` is every float but `nan`.
//!
//! **A step reads only the whole floats.** `x % n == 0` holds of a whole float
//! and of no fractional one, and of neither infinity, since `inf % n` is `nan`.
//! So a step splits the line into parts a bound cuts alike: the floats no step
//! holds, and the whole floats of each residue class. A set a step has touched
//! is one list of spans per part, and the three operations stay pointwise, part
//! by part, as they are residue by residue over the integers ([`IntSet`]). What
//! the floats add is that the whole ones run out: past `2^53` they stand a power
//! of two apart, so the next float of a class is found by solving a congruence
//! in each stretch of equally spaced floats ([`whole_at_or_above`]), and a span
//! holding no float of its part is read as empty.
//!
//! [`IntSet`]: super::integers::IntSet

use std::borrow::Cow;

use super::integers::MAX_PERIOD;

/// The floats from `lo` to `hi`, both held.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Span {
    lo: f64,
    hi: f64,
}

/// Reflexive for the same reason [`FloatSet`]'s is: an endpoint is never `nan`.
impl Eq for Span {}

/// A stable order on the representation, which is what [`FloatSet`]'s is built
/// from.
///
/// `total_cmp` rather than `partial_cmp`, so the order is total without an arm
/// that cannot happen, and it agrees with equality: normalization leaves no
/// `-0.0`, so two endpoints compare equal here exactly when `==` says so.
impl Ord for Span {
    fn cmp(&self, other: &Span) -> core::cmp::Ordering {
        self.lo
            .total_cmp(&other.lo)
            .then(self.hi.total_cmp(&other.hi))
    }
}

impl PartialOrd for Span {
    fn partial_cmp(&self, other: &Span) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The span holding no float: the one an open end at an infinity leaves, since
/// nothing lies above `inf` or below `-inf`.
const EMPTY: Span = Span {
    lo: f64::INFINITY,
    hi: f64::NEG_INFINITY,
};

impl Span {
    /// The floats between two endpoints, each held or not.
    ///
    /// An end that is not held is read as its neighboring float, which is exact
    /// rather than a rounding: no float lies between a float and the next one.
    ///
    /// Normalization is the invariant every other method here rests on: the
    /// order compares endpoints with `total_cmp` and equality compares them
    /// with `==`, and the two disagree on exactly one pair of values. `-0.0`
    /// reaching an endpoint would make `[0.0, -0.0]` a span that `is_empty`
    /// reads as crossed and `holds` reads as holding zero, which is not a
    /// canonicity cost but a wrong answer.
    fn new(lo: f64, lo_held: bool, hi: f64, hi_held: bool) -> Span {
        let (lo, hi) = (normalize(lo), normalize(hi));
        debug_assert!(
            !lo.is_nan() && !hi.is_nan(),
            "an endpoint is never `nan`: the one value outside the order is held \
             in the bit beside the spans"
        );
        let lo = if lo_held { Some(lo) } else { successor(lo) };
        let hi = if hi_held { Some(hi) } else { predecessor(hi) };
        let (Some(lo), Some(hi)) = (lo, hi) else {
            return EMPTY;
        };
        // The order and equality must agree on the endpoints, because one
        // decides whether a span is empty and the other whether two spans are
        // one. `-0.0` is the only pair they part on, and normalization is what
        // keeps it out; the comparison is the point of the assertion, so the
        // lint against comparing floats is allowed here by name.
        #[expect(
            clippy::float_cmp,
            reason = "the disagreement between `==` and `total_cmp` is what this reads"
        )]
        {
            debug_assert!(
                (lo.total_cmp(&hi) == core::cmp::Ordering::Equal) == (lo == hi),
                "normalization leaves the order and equality agreeing on {lo} and {hi}"
            );
        }
        Span { lo, hi }
    }

    /// Whether this interval holds no float: the ends crossed.
    fn is_empty(self) -> bool {
        self.lo.total_cmp(&self.hi) == core::cmp::Ordering::Greater
    }

    #[cfg(test)]
    fn holds(self, value: f64) -> bool {
        self.lo <= value && value <= self.hi
    }

    /// The floats in both intervals.
    fn intersect(self, other: Span) -> Span {
        Span {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        }
    }

    /// Whether `other` starts no later than the float after this interval's end,
    /// so no float lies between the two and they are one interval.
    ///
    /// `[0, 1]` and `[nextafter(1), 2]` meet that way although they share no
    /// float, which a reading of the floats as a continuum would keep apart --
    /// and would then answer `==` differently for two spellings of one set.
    fn reaches(self, other: Span) -> bool {
        successor(self.hi)
            .is_none_or(|next| other.lo.total_cmp(&next) != core::cmp::Ordering::Greater)
    }
}

/// The least float above `value`, or `None` above the largest.
fn successor(value: f64) -> Option<f64> {
    (value != f64::INFINITY).then(|| normalize(value.next_up()))
}

/// The greatest float below `value`, or `None` below the least.
fn predecessor(value: f64) -> Option<f64> {
    (value != f64::NEG_INFINITY).then(|| normalize(value.next_down()))
}

/// `-0.0` read as the value it is equal to.
///
/// The two zeros are one value under `==`, so only one of them may reach the
/// representation: otherwise a set holding `0.0` would answer differently about
/// `-0.0`, which no float can tell apart from it.
fn normalize(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// Which floats a list of spans speaks for, once a step has split the line.
#[derive(Debug, Clone, Copy)]
enum Part {
    /// The floats no step holds: the fractional ones and the two infinities.
    Stepless,
    /// The whole floats of one residue class.
    Residue { modulus: i64, residue: i64 },
}

/// `2^52`, from which every finite float is whole.
const ALL_WHOLE: f64 = 4_503_599_627_370_496.0;
/// `2^53`, up to which every integer is a float.
const EVERY_INTEGER: f64 = 9_007_199_254_740_992.0;
/// More stretches of equally spaced whole floats than there are: 971 binades past
/// `2^53` on each side of zero, and the integers between them. It says the search
/// ends where the ceiling does, which it reaches first.
const STRETCHES: usize = 2048;

impl Part {
    /// The least float of this part at or above `from`.
    fn at_or_above(self, from: f64) -> Option<f64> {
        match self {
            Part::Stepless => Some(stepless_at_or_above(from)),
            Part::Residue { modulus, residue } => whole_at_or_above(from, modulus, residue),
        }
    }

    /// The greatest float of this part at or below `from`, which is the least
    /// float at or above `-from` of the part's mirror.
    fn at_or_below(self, from: f64) -> Option<f64> {
        let mirror = match self {
            Part::Stepless => self,
            Part::Residue { modulus, residue } => Part::Residue {
                modulus,
                residue: modulus - residue,
            },
        };
        mirror.at_or_above(-from).map(|found| normalize(-found))
    }
}

/// The least float at or above `from` that is fractional or infinite.
///
/// A whole float's neighbor above is fractional below `2^52` and whole from
/// there on, so past that the next is `inf`, and below `-2^52` it is the first
/// float above `-2^52`.
fn stepless_at_or_above(from: f64) -> f64 {
    if from.is_infinite() || from.fract() != 0.0 {
        return from;
    }
    let next = from.next_up();
    if next.fract() != 0.0 {
        next
    } else if from.is_sign_negative() {
        (-ALL_WHOLE).next_up()
    } else {
        f64::INFINITY
    }
}

/// The least whole float at or above `from` congruent to `residue`.
///
/// Whole floats are equally spaced in stretches: one apart up to `2^53`, and a
/// power of two apart in each binade past it. Inside a stretch a float is
/// `t * spacing`, which makes this one linear congruence in `t`, so no float is
/// walked and the cost is one solve per stretch: a stretch whose spacing rules
/// the residue out takes a `gcd`, and one that does not answers. It ends at the
/// largest finite float.
///
/// A float `t * 2^j` shares that power of two with the modulus, so a residue
/// holding fewer twos than the modulus has no float once the spacing passes them,
/// and every whole float of its class lies within the [`ceiling`] that follows.
fn whole_at_or_above(from: f64, modulus: i64, residue: i64) -> Option<f64> {
    let ceiling = ceiling(modulus, residue);
    let mut at = from.ceil().max(-ceiling);
    for _ in 0..STRETCHES {
        if at > ceiling {
            return None;
        }
        let above = at.next_up();
        let gap = if above.is_finite() {
            above - at
        } else {
            at - at.next_down()
        };
        let spacing = gap.max(1.0);
        #[expect(
            clippy::float_cmp,
            reason = "a stretch's spacing is exactly one or a larger power of two"
        )]
        let last = if spacing == 1.0 {
            EVERY_INTEGER
        } else if at.is_sign_positive() {
            spacing * (EVERY_INTEGER - 1.0)
        } else {
            -spacing * ALL_WHOLE
        };
        if let Some(found) = in_stretch(at, last, spacing, modulus, residue) {
            return Some(normalize(found));
        }
        at = last.next_up();
    }
    None
}

/// The largest magnitude a whole float of a class can have: the spacing past the
/// twos the residue holds begins at `2^53` times them, and a class whose residue
/// holds as many as the modulus is read to the largest float.
fn ceiling(modulus: i64, residue: i64) -> f64 {
    let twos = (modulus | residue).trailing_zeros();
    if twos == modulus.trailing_zeros() {
        return f64::MAX;
    }
    #[expect(
        clippy::cast_possible_wrap,
        reason = "a modulus at most `MAX_PERIOD` holds a dozen twos"
    )]
    let twos = twos as i32;
    (EVERY_INTEGER * 2.0f64.powi(twos)).next_down()
}

/// The least `t * spacing` from `first` to `last` congruent to `residue`.
fn in_stretch(first: f64, last: f64, spacing: f64, modulus: i64, residue: i64) -> Option<f64> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a modulus is at most `MAX_PERIOD`, which a float holds exactly"
    )]
    let wide = modulus as f64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a stretch holds at most 2^53 floats each way, and a remainder is below the modulus"
    )]
    let (from, to, each) = (
        (first / spacing) as i64,
        (last / spacing) as i64,
        spacing.rem_euclid(wide) as i64,
    );
    let shared = gcd(each, modulus);
    if residue % shared != 0 {
        return None;
    }
    let period = modulus / shared;
    let wanted = (residue / shared) * inverse(each / shared, period);
    let found = from + (wanted - from).rem_euclid(period);
    #[expect(
        clippy::cast_precision_loss,
        reason = "`found` is inside the stretch, so it times the spacing is a float"
    )]
    (found <= to).then_some(found as f64 * spacing)
}

/// The greatest common divisor, with `gcd(0, m)` the modulus.
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The inverse of `a` modulo `period`, which it is coprime to; every value is
/// the inverse of every other modulo one.
fn inverse(a: i64, period: i64) -> i64 {
    let (mut low, mut high) = (a.rem_euclid(period), period);
    let (mut coefficient, mut other) = (1, 0);
    while low > 1 {
        let quotient = high / low;
        (high, low) = (low, high % low);
        (other, coefficient) = (coefficient, other - quotient * coefficient);
    }
    coefficient.rem_euclid(period)
}

/// The spans read in `part`: each end moved in to a float of the part, a span
/// holding none dropped, and two with no float of the part between them merged.
fn fitted(part: Part, spans: Vec<Span>) -> Vec<Span> {
    let mut fit: Vec<Span> = spans
        .into_iter()
        .filter_map(|span| {
            let lo = part.at_or_above(span.lo)?;
            let hi = part.at_or_below(span.hi)?;
            (lo <= hi).then_some(Span { lo, hi })
        })
        .collect();
    fit.sort();
    let mut merged: Vec<Span> = Vec::with_capacity(fit.len());
    for span in fit {
        match merged.last_mut() {
            Some(last)
                if successor(last.hi)
                    .and_then(|next| part.at_or_above(next))
                    .is_none_or(|next| next >= span.lo) =>
            {
                last.hi = last.hi.max(span.hi);
            }
            _ => merged.push(span),
        }
    }
    merged
}

/// The floats between two spans and past the outer two, in order: what a sorted
/// list of spans leaves of the line.
fn gaps(spans: &[Span]) -> Vec<Span> {
    let mut gaps = Vec::with_capacity(spans.len() + 1);
    // The least float no interval so far holds, or `None` once `inf` is.
    let mut next = Some(f64::NEG_INFINITY);
    for span in spans {
        if let (Some(lo), Some(hi)) = (next, predecessor(span.lo)) {
            gaps.push(Span { lo, hi });
        }
        next = successor(span.hi);
    }
    if let Some(lo) = next {
        gaps.push(Span {
            lo,
            hi: f64::INFINITY,
        });
    }
    gaps
}

/// The spans two lists share, unsorted.
fn shared_spans(mine: &[Span], theirs: &[Span]) -> Vec<Span> {
    let mut spans = Vec::new();
    for mine in mine {
        for theirs in theirs {
            let met = mine.intersect(*theirs);
            if !met.is_empty() {
                spans.push(met);
            }
        }
    }
    spans
}

/// The whole floats of a set a step has touched: their spans per residue.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Stepped {
    /// The period. At least one, and at most [`MAX_PERIOD`].
    modulus: i64,
    /// One list per residue, in that part's canonical form.
    classes: Vec<Vec<Span>>,
}

/// A set of floats.
#[derive(Debug, Clone)]
pub(crate) struct FloatSet {
    /// Sorted intervals over the ordered floats, with a float between each two
    /// that neither holds. Where a step has split the line they speak for
    /// [`Part::Stepless`], and otherwise for every ordered float.
    spans: Vec<Span>,
    /// Whether the set holds `nan`, which no interval can say.
    nan: bool,
    /// The whole floats, where a step reads them apart. Boxed because a set no
    /// step touched is nearly every set a relation builds, and it pays one
    /// discriminant for this.
    stepped: Option<Box<Stepped>>,
}

/// Two sets are equal when they hold the same floats, read at the period where
/// their residues line up; [`IntSet`](super::integers::IntSet) says why equality
/// lifts, and why a pair whose periods share none under the bound answers `false`
/// here as there, which can read two spellings of one set as two and cannot read
/// two sets as one.
impl PartialEq for FloatSet {
    fn eq(&self, other: &FloatSet) -> bool {
        if self.nan != other.nan {
            return false;
        }
        if self.stepped.is_none() && other.stepped.is_none() {
            return self.spans == other.spans;
        }
        common_period(self.period(), other.period())
            .is_some_and(|modulus| self.split_at(modulus) == other.split_at(modulus))
    }
}

/// Reflexive because `nan` never reaches a span: the endpoints are normalized
/// and the one value that is not equal to itself lives in the bit beside them,
/// where it is an ordinary `bool`.
impl Eq for FloatSet {}

/// Ordered so a float set can be a *guard*, which the sequence automaton sorts
/// to reach a canonical table.
///
/// The order is on the representation, not the sets -- no order on sets is
/// wanted here, only a stable one. It agrees with equality on the sets no step
/// touched, which is what a sort needs of it: `total_cmp` separates two
/// endpoints exactly when `==` does, because normalization leaves no `-0.0` and
/// no span holds `nan`. A set a step touched has spellings equality lifts and
/// the order reads apart, as an [`IntSet`](super::integers::IntSet) has.
impl Ord for FloatSet {
    fn cmp(&self, other: &FloatSet) -> core::cmp::Ordering {
        self.spans
            .cmp(&other.spans)
            .then(self.nan.cmp(&other.nan))
            .then_with(|| self.stepped.cmp(&other.stepped))
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
    pub(crate) fn empty() -> FloatSet {
        FloatSet {
            spans: Vec::new(),
            nan: false,
            stepped: None,
        }
    }

    /// Every float, `nan` included.
    #[must_use]
    pub(crate) fn all() -> FloatSet {
        FloatSet {
            spans: vec![Span::new(f64::NEG_INFINITY, true, f64::INFINITY, true)],
            nan: true,
            stepped: None,
        }
    }

    /// Just `nan`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn nan() -> FloatSet {
        FloatSet {
            spans: Vec::new(),
            nan: true,
            stepped: None,
        }
    }

    /// The one-element set, or the empty set for `nan`.
    ///
    /// `Literal[c]` denotes the values equal to `c`, the model's singleton
    /// [FCB08, §4.2] where `==` is identity, and `nan` is equal to nothing --
    /// itself included. So a literal `nan` admits no value at all,
    /// which is a fact about Python's equality rather than a choice made here.
    #[must_use]
    pub(crate) fn just(value: f64) -> FloatSet {
        if value.is_nan() {
            FloatSet::empty()
        } else {
            FloatSet::from_span(Span::new(value, true, value, true))
        }
    }

    /// The floats at or above `bound`, or none where `bound` is `nan`.
    #[must_use]
    pub(crate) fn at_least(bound: f64) -> FloatSet {
        FloatSet::ordered(bound, true, f64::INFINITY, true)
    }

    /// The floats strictly above `bound`.
    #[must_use]
    pub(crate) fn above(bound: f64) -> FloatSet {
        FloatSet::ordered(bound, false, f64::INFINITY, true)
    }

    /// The floats at or below `bound`.
    #[must_use]
    pub(crate) fn at_most(bound: f64) -> FloatSet {
        FloatSet::ordered(f64::NEG_INFINITY, true, bound, true)
    }

    /// The floats strictly below `bound`.
    #[must_use]
    pub(crate) fn below(bound: f64) -> FloatSet {
        FloatSet::ordered(f64::NEG_INFINITY, true, bound, false)
    }

    /// An ordered interval, empty where either end is `nan`.
    ///
    /// Every comparison with `nan` is false, so a bound of `nan` admits no
    /// float: `Annotated[float, Ge(float("nan"))]` is the empty set, and reading
    /// it as an unbounded end would make it every float instead.
    fn ordered(lo: f64, lo_held: bool, hi: f64, hi_held: bool) -> FloatSet {
        if lo.is_nan() || hi.is_nan() {
            return FloatSet::empty();
        }
        FloatSet::from_span(Span::new(lo, lo_held, hi, hi_held))
    }

    fn from_span(span: Span) -> FloatSet {
        if span.is_empty() {
            FloatSet::empty()
        } else {
            FloatSet {
                spans: vec![span],
                nan: false,
                stepped: None,
            }
        }
    }

    /// The finite floats `x` with `x % step == 0`, or `None` for a step past
    /// [`MAX_PERIOD`] or one no float is a multiple of.
    ///
    /// Python reads the remainder exactly and the sign of the step changes
    /// nothing, so the set is the residue zero of its magnitude. `inf % n` is
    /// `nan`, so the infinities are outside it, and so is `nan`.
    #[must_use]
    pub(crate) fn multiples_of(step: i64) -> Option<FloatSet> {
        let modulus = step
            .checked_abs()
            .filter(|m| (1..=MAX_PERIOD).contains(m))?;
        let every = Span::new(f64::NEG_INFINITY, true, f64::INFINITY, true);
        let classes = (0..modulus)
            .map(|residue| {
                if residue == 0 {
                    fitted(Part::Residue { modulus, residue }, vec![every])
                } else {
                    Vec::new()
                }
            })
            .collect();
        Some(FloatSet {
            spans: Vec::new(),
            nan: false,
            stepped: Some(Box::new(Stepped { modulus, classes })),
        })
    }

    /// Whether this set holds no float.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.spans.is_empty()
            && !self.nan
            && self
                .stepped
                .as_ref()
                .is_none_or(|stepped| stepped.classes.iter().all(Vec::is_empty))
    }

    /// Whether this set holds `value`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn holds(&self, value: f64) -> bool {
        if value.is_nan() {
            return self.nan;
        }
        let value = normalize(value);
        let spans = match &self.stepped {
            Some(stepped) if value.is_finite() && value.fract() == 0.0 => {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a modulus is at most `MAX_PERIOD`, which a float holds exactly"
                )]
                let wide = stepped.modulus as f64;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a remainder is a whole float below the modulus"
                )]
                let residue = value.rem_euclid(wide) as usize;
                stepped.classes.get(residue).map_or(&[][..], Vec::as_slice)
            }
            _ => self.spans.as_slice(),
        };
        spans.iter().any(|span| span.holds(value))
    }

    /// The period this set's whole floats are read at, one where no step split
    /// them.
    fn period(&self) -> i64 {
        self.stepped.as_ref().map_or(1, |stepped| stepped.modulus)
    }

    /// This set's stepless spans and its spans per residue, read at `modulus`,
    /// a multiple of its own period: the table the pointwise operations take.
    fn split_at(&self, modulus: i64) -> (Cow<'_, [Span]>, Cow<'_, [Vec<Span>]>) {
        if let Some(stepped) = &self.stepped
            && stepped.modulus == modulus
        {
            return (Cow::Borrowed(&self.spans), Cow::Borrowed(&stepped.classes));
        }
        let spans = match &self.stepped {
            Some(_) => Cow::Borrowed(self.spans.as_slice()),
            None => Cow::Owned(fitted(Part::Stepless, self.spans.clone())),
        };
        let classes = (0..modulus)
            .map(|residue| {
                let of_class = self
                    .stepped
                    .as_ref()
                    .map_or(self.spans.as_slice(), |stepped| {
                        let at = residue % stepped.modulus;
                        usize::try_from(at)
                            .ok()
                            .and_then(|at| stepped.classes.get(at))
                            .map_or(&[][..], Vec::as_slice)
                    });
                fitted(Part::Residue { modulus, residue }, of_class.to_vec())
            })
            .collect();
        (spans, Cow::Owned(classes))
    }

    /// `op` applied part by part, at the period the two sets share, or `None`
    /// past the bound on one: a rounded period describes a different set.
    fn stepwise(
        &self,
        other: &FloatSet,
        nan: bool,
        op: impl Fn(&[Span], &[Span]) -> Vec<Span>,
    ) -> Option<FloatSet> {
        let modulus = common_period(self.period(), other.period())?;
        let (my_spans, my_classes) = self.split_at(modulus);
        let (their_spans, their_classes) = other.split_at(modulus);
        let classes = my_classes
            .iter()
            .zip(their_classes.iter())
            .zip(0..)
            .map(|((mine, theirs), residue)| {
                fitted(Part::Residue { modulus, residue }, op(mine, theirs))
            })
            .collect();
        Some(FloatSet {
            spans: fitted(Part::Stepless, op(&my_spans, &their_spans)),
            nan,
            stepped: Some(Box::new(Stepped { modulus, classes })),
        })
    }

    /// Every float this set holds where there are at most `cap` and none is
    /// `nan`, and `None` otherwise.
    ///
    /// `nan` counts as more than any cap: no `nan` is equal to another value,
    /// itself included, so a set holding one holds as many unequal members as a
    /// caller asks for.
    pub(crate) fn values_up_to(&self, cap: usize) -> Option<Vec<f64>> {
        if self.nan {
            return None;
        }
        let mut values = Vec::new();
        let Some(stepped) = &self.stepped else {
            for span in &self.spans {
                let mut value = span.lo;
                loop {
                    if values.len() == cap {
                        return None;
                    }
                    values.push(value);
                    if value.total_cmp(&span.hi).is_ge() {
                        break;
                    }
                    value = successor(value)?;
                }
            }
            return Some(values);
        };
        let classes = stepped.classes.iter().zip(0..).map(|(spans, residue)| {
            let part = Part::Residue {
                modulus: stepped.modulus,
                residue,
            };
            (part, spans)
        });
        for (part, spans) in [(Part::Stepless, &self.spans)].into_iter().chain(classes) {
            for span in spans {
                let mut value = span.lo;
                loop {
                    if values.len() == cap {
                        return None;
                    }
                    values.push(value);
                    if value.total_cmp(&span.hi).is_ge() {
                        break;
                    }
                    value = successor(value).and_then(|next| part.at_or_above(next))?;
                }
            }
        }
        Some(values)
    }

    /// The floats in either set, or `None` where two steps share no period under
    /// the bound.
    #[must_use]
    pub(crate) fn union(&self, other: &FloatSet) -> Option<FloatSet> {
        let nan = self.nan || other.nan;
        if self.stepped.is_none() && other.stepped.is_none() {
            return Some(
                FloatSet {
                    spans: self.spans.iter().chain(&other.spans).copied().collect(),
                    nan,
                    stepped: None,
                }
                .canonical(),
            );
        }
        self.stepwise(other, nan, |mine, theirs| {
            mine.iter().chain(theirs).copied().collect()
        })
    }

    /// The floats in both sets, or `None` where two steps share no period under
    /// the bound.
    #[must_use]
    pub(crate) fn intersect(&self, other: &FloatSet) -> Option<FloatSet> {
        let nan = self.nan && other.nan;
        if self.stepped.is_none() && other.stepped.is_none() {
            return Some(
                FloatSet {
                    spans: shared_spans(&self.spans, &other.spans),
                    nan,
                    stepped: None,
                }
                .canonical(),
            );
        }
        self.stepwise(other, nan, shared_spans)
    }

    /// Every float this set does not hold.
    ///
    /// The ordered part complements by reading the gaps: from the float after
    /// one interval's end to the float before the next one's start, part by part
    /// where a step split the line. The `nan` bit flips on its own, which is what
    /// keeps the halves independent.
    #[must_use]
    pub(crate) fn complement(&self) -> FloatSet {
        let Some(stepped) = &self.stepped else {
            return FloatSet {
                spans: gaps(&self.spans),
                nan: !self.nan,
                stepped: None,
            }
            .canonical();
        };
        let modulus = stepped.modulus;
        let classes = stepped
            .classes
            .iter()
            .zip(0..)
            .map(|(spans, residue)| fitted(Part::Residue { modulus, residue }, gaps(spans)))
            .collect();
        FloatSet {
            spans: fitted(Part::Stepless, gaps(&self.spans)),
            nan: !self.nan,
            stepped: Some(Box::new(Stepped { modulus, classes })),
        }
    }

    /// Sort, drop the empties, and merge what no float separates.
    fn canonical(self) -> FloatSet {
        let mut spans: Vec<Span> = self.spans.into_iter().filter(|s| !s.is_empty()).collect();
        spans.sort();
        let mut merged: Vec<Span> = Vec::with_capacity(spans.len());
        for span in spans {
            match merged.last_mut() {
                Some(last) if last.reaches(span) => last.hi = last.hi.max(span.hi),
                _ => merged.push(span),
            }
        }
        FloatSet {
            spans: merged,
            nan: self.nan,
            stepped: None,
        }
    }
}

/// The least common multiple of two periods, or `None` past [`MAX_PERIOD`].
fn common_period(a: i64, b: i64) -> Option<i64> {
    Some(a / gcd(a, b) * b).filter(|modulus| *modulus <= MAX_PERIOD)
}

#[cfg(test)]
mod tests;
