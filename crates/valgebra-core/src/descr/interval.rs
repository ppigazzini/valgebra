//! Sets of integers as sorted disjoint intervals, with unbounded ends.
//!
//! The building block under the integer component of a
//! [`Descr`](super::Descr): an integer set that is a finite union of intervals,
//! each possibly reaching to infinity in one or both directions. `Ge(0)` is one
//! such interval, `Literal[3]` is a degenerate one, and their Boolean
//! combinations are what a refinement over `int` denotes.
//!
//! **Canonical.** The intervals are sorted, pairwise disjoint, and never
//! adjacent -- `[0, 3]` and `[4, 7]` merge into `[0, 7]`, because both denote
//! the same integers and a representation that kept them apart would make two
//! equal sets unequal. Every constructor and operation restores that form, so
//! equality of the representation is equality of the sets.
//!
//! **Two forms, one set.** An end is exact at any width ([`WideInt`]), and
//! nearly every end a schema writes is one an `i128` holds. A set whose ends
//! all are is held in that width ([`Narrow`]), its spans copied, compared and
//! dropped as machine integers; a set with an end past it is held in [`Span`]s.
//! The form is a function of the set -- narrow exactly where every end fits --
//! so the canonical form is still one representation a set, and an operation
//! whose answer leaves the width, the complement of a span ending at an end of
//! it or a preimage whose shift does, is taken in the wide form. Held wide
//! throughout, the ends cost the relations asked of the sets a percent of their
//! instructions: each span's ends dropped as a value that may own memory, a
//! sort's comparisons out of line, and each step by one a call.

use core::cmp::{Ordering, max, min};
use core::mem::ManuallyDrop;

use super::wide::WideInt;

/// One interval of integers, `lo..=hi`, where `None` is unbounded.
///
/// Inclusive at both ends, because the elements are integers: a half-open end
/// would be a second way to write the same set, and the point of the form is
/// that there is one.
///
/// **The ends are exact at any width** ([`WideInt`]). A caller's bound is as
/// wide as the number it is written with, and complementing a span needs the
/// integer just outside it: the complement of `Ge(-2**200)` begins at
/// `-2**200 - 1`. An end held in a fixed width would leave the integers past
/// it unrepresentable, and a complement or a step would drop or move them.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Span {
    lo: Option<WideInt>,
    hi: Option<WideInt>,
}

impl Span {
    /// Whether this span holds no integer, which is `lo > hi` and only possible
    /// when both ends are bounded.
    fn is_empty(&self) -> bool {
        matches!((&self.lo, &self.hi), (Some(lo), Some(hi)) if lo > hi)
    }

    fn holds(&self, value: &WideInt) -> bool {
        self.lo.as_ref().is_none_or(|lo| lo <= value)
            && self.hi.as_ref().is_none_or(|hi| value <= hi)
    }

    /// The integers in both spans.
    fn intersect(&self, other: &Span) -> Span {
        Span {
            lo: bound(self.lo.as_ref(), other.lo.as_ref(), max),
            hi: bound(self.hi.as_ref(), other.hi.as_ref(), min),
        }
    }

    /// Whether `other` starts at or before this span's end, counting adjacency:
    /// `[0, 3]` and `[4, 7]` have no integer between them, so they are one span
    /// and the canonical form says so.
    fn reaches(&self, other: &Span) -> bool {
        match (&self.hi, &other.lo) {
            (None, _) | (_, None) => true,
            (Some(hi), Some(lo)) => *lo <= hi.succ(),
        }
    }

    /// This span in the narrow form, where an `i128` holds both its ends.
    fn narrow(&self) -> Option<Narrow> {
        let lo = match &self.lo {
            None => None,
            Some(lo) => Some(lo.to_i128()?),
        };
        let hi = match &self.hi {
            None => None,
            Some(hi) => Some(hi.to_i128()?),
        };
        Some(Narrow { lo, hi })
    }
}

/// Combine two bounds, where `None` is the unbounded end.
///
/// `pick` is `max` for a lower bound and `min` for an upper one, which is the
/// only difference between the two cases: an unbounded lower end is negative
/// infinity, so the bounded one wins; an unbounded upper end is positive
/// infinity, likewise.
fn bound(
    a: Option<&WideInt>,
    b: Option<&WideInt>,
    pick: fn(WideInt, WideInt) -> WideInt,
) -> Option<WideInt> {
    match (a, b) {
        (Some(a), Some(b)) => Some(pick(a.clone(), b.clone())),
        (Some(only), None) | (None, Some(only)) => Some(only.clone()),
        (None, None) => None,
    }
}

/// One interval whose ends an `i128` holds: [`Span`]'s narrow form, which
/// nearly every set takes. It orders as the span with the same ends does, so a
/// set orders the same in either form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Narrow {
    lo: Option<i128>,
    hi: Option<i128>,
}

impl Narrow {
    fn is_empty(self) -> bool {
        matches!((self.lo, self.hi), (Some(lo), Some(hi)) if lo > hi)
    }

    fn holds(self, value: i128) -> bool {
        self.lo.is_none_or(|lo| lo <= value) && self.hi.is_none_or(|hi| value <= hi)
    }

    /// The integers in both spans.
    fn intersect(self, other: Narrow) -> Narrow {
        Narrow {
            lo: narrow_bound(self.lo, other.lo, max),
            hi: narrow_bound(self.hi, other.hi, min),
        }
    }

    /// Whether `other` starts at or before this span's end, counting adjacency.
    fn reaches(self, other: Narrow) -> bool {
        match (self.hi, other.lo) {
            (None, _) | (_, None) => true,
            // `other` starts at an `i128`, so it starts at or before an end at
            // the top of the width, which the saturated successor answers as
            // the exact one does.
            (Some(hi), Some(lo)) => lo <= hi.saturating_add(1),
        }
    }

    /// This span in the wide form.
    fn wide(self) -> Span {
        Span {
            lo: self.lo.map(WideInt::Small),
            hi: self.hi.map(WideInt::Small),
        }
    }
}

/// [`bound`] for the narrow form.
fn narrow_bound(a: Option<i128>, b: Option<i128>, pick: fn(i128, i128) -> i128) -> Option<i128> {
    match (a, b) {
        (Some(a), Some(b)) => Some(pick(a, b)),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

/// The wide form's spans, behind one pointer, released by hand.
///
/// The wide form's ends may share a `BigInt`, so dropping its spans is a loop
/// over their ends. Held as a vector in the wide variant, that loop was in
/// every set's drop, which was a call out of line on every set a table of them
/// released. Held in `ManuallyDrop`, the box has no drop code, and the `Drop`
/// below tests the pointer and releases a box out of line only where there is
/// one, as the walk's state holds what it rarely holds (`RareSlot` in the
/// binding's `check/ctx.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
struct WideSlot(ManuallyDrop<Option<Box<[Span]>>>);

impl Drop for WideSlot {
    #[inline]
    fn drop(&mut self) {
        if self.0.is_some() {
            self.release();
        }
    }
}

impl WideSlot {
    /// The spans this slot holds.
    fn get(&self) -> &[Span] {
        self.0.as_deref().unwrap_or_default()
    }

    /// Drop the box, out of line: only a set with an end past `i128` has one.
    #[cold]
    #[inline(never)]
    fn release(&mut self) {
        drop(self.0.take());
    }
}

/// A set's spans, held in the form its ends take. The wide slot is two words
/// and fits beside the narrow vector's capacity, so a set is the three words
/// one vector of spans was.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Spans {
    Narrow(Vec<Narrow>),
    Wide(WideSlot),
}

/// A set of integers: sorted, disjoint, non-adjacent spans, in one form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IntervalSet {
    spans: Spans,
}

impl Ord for IntervalSet {
    /// Order two sets by their spans, as one form would hold them: two narrow
    /// sets by their machine ends, and any other pair by every span widened,
    /// which orders a narrow span as its wide one.
    ///
    /// The pairs are named, not caught: the widened order answers a narrow pair
    /// too, so an arm a catch-all stood behind could go and no answer would
    /// move.
    fn cmp(&self, other: &IntervalSet) -> Ordering {
        match (&self.spans, &other.spans) {
            (Spans::Narrow(mine), Spans::Narrow(theirs)) => mine.cmp(theirs),
            (Spans::Narrow(_) | Spans::Wide(_), Spans::Wide(_))
            | (Spans::Wide(_), Spans::Narrow(_)) => self.wide_spans().cmp(other.wide_spans()),
        }
    }
}

impl PartialOrd for IntervalSet {
    fn partial_cmp(&self, other: &IntervalSet) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl IntervalSet {
    fn narrow(spans: Vec<Narrow>) -> IntervalSet {
        IntervalSet {
            spans: Spans::Narrow(spans),
        }
    }

    /// The set of canonical spans with an end past `i128`, in the wide form.
    fn wide(spans: Vec<Span>) -> IntervalSet {
        IntervalSet {
            spans: Spans::Wide(WideSlot(ManuallyDrop::new(Some(spans.into_boxed_slice())))),
        }
    }

    /// The set of canonical wide spans, in the form their ends take.
    fn of_wide(spans: Vec<Span>) -> IntervalSet {
        match spans.iter().map(Span::narrow).collect::<Option<Vec<_>>>() {
            Some(narrow) => IntervalSet::narrow(narrow),
            None => IntervalSet::wide(spans),
        }
    }

    /// Every span of this set, in the wide form.
    fn wide_spans(&self) -> impl Iterator<Item = Span> + '_ {
        let (narrow, wide): (&[Narrow], &[Span]) = match &self.spans {
            Spans::Narrow(spans) => (spans, &[]),
            Spans::Wide(slot) => (&[], slot.get()),
        };
        narrow
            .iter()
            .map(|span| span.wide())
            .chain(wide.iter().cloned())
    }

    /// How many spans this set holds.
    pub(crate) fn spans(&self) -> usize {
        match &self.spans {
            Spans::Narrow(spans) => spans.len(),
            Spans::Wide(slot) => slot.get().len(),
        }
    }

    /// The empty set.
    #[must_use]
    pub(crate) fn empty() -> IntervalSet {
        IntervalSet::narrow(Vec::new())
    }

    /// Every integer.
    #[must_use]
    pub(crate) fn all() -> IntervalSet {
        IntervalSet::narrow(vec![Narrow { lo: None, hi: None }])
    }

    /// The integers from `lo` to `hi` inclusive, with `None` unbounded.
    #[must_use]
    pub(crate) fn between(lo: Option<WideInt>, hi: Option<WideInt>) -> IntervalSet {
        let span = Span { lo, hi };
        if span.is_empty() {
            IntervalSet::empty()
        } else if let Some(narrow) = span.narrow() {
            IntervalSet::narrow(vec![narrow])
        } else {
            IntervalSet::wide(vec![span])
        }
    }

    /// The one-element set.
    #[must_use]
    pub(crate) fn just(value: i64) -> IntervalSet {
        let value = Some(i128::from(value));
        IntervalSet::narrow(vec![Narrow {
            lo: value,
            hi: value,
        }])
    }

    /// Whether this set holds no integer.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        // The wide form holds a span past `i128`, so only a narrow set is empty.
        matches!(&self.spans, Spans::Narrow(spans) if spans.is_empty())
    }

    /// Whether this set holds every integer: the one unbounded span, which
    /// is narrow. Read off the form, so the question builds no set to compare.
    #[must_use]
    pub(crate) fn is_all(&self) -> bool {
        matches!(&self.spans, Spans::Narrow(spans) if *spans == [Narrow { lo: None, hi: None }])
    }

    /// Whether this set holds `value`.
    #[must_use]
    pub(crate) fn holds(&self, value: i64) -> bool {
        match &self.spans {
            Spans::Narrow(spans) => spans.iter().any(|span| span.holds(i128::from(value))),
            Spans::Wide(slot) => {
                let value = WideInt::from(value);
                slot.get().iter().any(|span| span.holds(&value))
            }
        }
    }

    /// Every integer this set holds where there are at most `cap`, and `None`
    /// where there are more -- an unbounded span among them.
    pub(crate) fn values_up_to(&self, cap: usize) -> Option<Vec<WideInt>> {
        let mut values = Vec::new();
        match &self.spans {
            Spans::Narrow(spans) => {
                for span in spans {
                    let (Some(lo), Some(hi)) = (span.lo, span.hi) else {
                        return None;
                    };
                    for value in lo..=hi {
                        if values.len() == cap {
                            return None;
                        }
                        values.push(WideInt::Small(value));
                    }
                }
            }
            Spans::Wide(slot) => {
                for span in slot.get() {
                    let (Some(lo), Some(hi)) = (&span.lo, &span.hi) else {
                        return None;
                    };
                    let mut value = lo.clone();
                    while value <= *hi {
                        if values.len() == cap {
                            return None;
                        }
                        let next = value.succ();
                        values.push(value);
                        value = next;
                    }
                }
            }
        }
        Some(values)
    }

    /// The integers in either set.
    ///
    /// Merging is what restores the canonical form: the spans are gathered,
    /// sorted by their lower end, and folded so that any two that meet or touch
    /// become one.
    #[must_use]
    pub(crate) fn union(&self, other: &IntervalSet) -> IntervalSet {
        match (&self.spans, &other.spans) {
            (Spans::Narrow(mine), Spans::Narrow(theirs)) => {
                narrow_canonical(mine.iter().chain(theirs).copied().collect())
            }
            (Spans::Narrow(_) | Spans::Wide(_), Spans::Wide(_))
            | (Spans::Wide(_), Spans::Narrow(_)) => {
                wide_canonical(self.wide_spans().chain(other.wide_spans()).collect())
            }
        }
    }

    /// The integers in both sets.
    ///
    /// Pairwise: each span of one meets each span of the other in at most one
    /// interval, and what comes out is canonicalized rather than assumed
    /// ordered.
    #[must_use]
    pub(crate) fn intersect(&self, other: &IntervalSet) -> IntervalSet {
        if let (Spans::Narrow(mine), Spans::Narrow(theirs)) = (&self.spans, &other.spans) {
            let mut spans = Vec::new();
            for mine in mine {
                for theirs in theirs {
                    let met = mine.intersect(*theirs);
                    if !met.is_empty() {
                        spans.push(met);
                    }
                }
            }
            return narrow_canonical(spans);
        }
        let theirs: Vec<Span> = other.wide_spans().collect();
        let mut spans = Vec::new();
        for mine in self.wide_spans() {
            for theirs in &theirs {
                let met = mine.intersect(theirs);
                if !met.is_empty() {
                    spans.push(met);
                }
            }
        }
        wide_canonical(spans)
    }

    /// Every integer this set does not hold.
    ///
    /// Read off the gaps: the run below the first span, one between each
    /// adjacent pair, and the run above the last. The empty set has no span to
    /// skip, so the same walk gives every integer.
    #[must_use]
    pub(crate) fn complement(&self) -> IntervalSet {
        match &self.spans {
            Spans::Narrow(spans) => narrow_complement(spans)
                .unwrap_or_else(|| wide_complement(&self.wide_spans().collect::<Vec<_>>())),
            Spans::Wide(slot) => wide_complement(slot.get()),
        }
    }

    /// The set `{ k : offset + stride * k is in self }`, for a positive `stride`.
    ///
    /// The change of variable a modulus needs: a residue class is
    /// order-isomorphic to the integers, and this carries a set across that
    /// isomorphism. Each end divides, rounding *inward* -- up at the lower end,
    /// down at the upper -- because a `k` outside the rounded range maps to an
    /// integer outside the span.
    ///
    /// The arithmetic is exact at any width: a span's end minus the offset,
    /// divided by the stride, is the integer the residue class reads there, and
    /// any rounding past the exact value would name another class.
    #[must_use]
    pub(crate) fn preimage(&self, offset: i64, stride: i64) -> IntervalSet {
        debug_assert!(stride > 0, "a stride is a positive step");
        match &self.spans {
            Spans::Narrow(spans) => narrow_preimage(spans, offset, stride).unwrap_or_else(|| {
                wide_preimage(&self.wide_spans().collect::<Vec<_>>(), offset, stride)
            }),
            Spans::Wide(slot) => wide_preimage(slot.get(), offset, stride),
        }
    }
}

/// Sort, drop the empties, and merge what touches, in the narrow form.
fn narrow_canonical(spans: Vec<Narrow>) -> IntervalSet {
    let mut spans: Vec<Narrow> = spans.into_iter().filter(|s| !s.is_empty()).collect();
    // An unbounded lower end sorts first, which is where negative infinity
    // belongs.
    spans.sort_by_key(|span| (span.lo.is_some(), span.lo));
    let mut merged: Vec<Narrow> = Vec::with_capacity(spans.len());
    for span in spans {
        match merged.last_mut() {
            Some(last) if last.reaches(span) => {
                last.hi = match (last.hi, span.hi) {
                    (Some(a), Some(b)) => Some(max(a, b)),
                    // Either end unbounded above swallows the other.
                    _ => None,
                };
            }
            _ => merged.push(span),
        }
    }
    IntervalSet::narrow(merged)
}

/// Sort, drop the empties, and merge what touches, in the wide form, and hold
/// the answer in the form its ends take.
fn wide_canonical(spans: Vec<Span>) -> IntervalSet {
    let mut spans: Vec<Span> = spans.into_iter().filter(|s| !s.is_empty()).collect();
    spans
        .sort_by(|left, right| (left.lo.is_some(), &left.lo).cmp(&(right.lo.is_some(), &right.lo)));
    let mut merged: Vec<Span> = Vec::with_capacity(spans.len());
    for span in spans {
        match merged.last_mut() {
            Some(last) if last.reaches(&span) => {
                last.hi = match (last.hi.take(), span.hi) {
                    (Some(a), Some(b)) => Some(max(a, b)),
                    _ => None,
                };
            }
            _ => merged.push(span),
        }
    }
    IntervalSet::of_wide(merged)
}

/// The complement in the narrow form, or `None` where a gap's end leaves the
/// width: a span ending at either end of it has integers past that end.
fn narrow_complement(spans: &[Narrow]) -> Option<IntervalSet> {
    let mut gaps = Vec::new();
    // Where the next gap starts: `None` until the first span is seen, which
    // is the unbounded run below it.
    let mut gap_lo: Option<Option<i128>> = Some(None);
    for span in spans {
        if let (Some(lo), Some(start)) = (span.lo, gap_lo) {
            gaps.push(Narrow {
                lo: start,
                hi: Some(lo.checked_sub(1)?),
            });
        }
        // A run above a span starts after its end; one unbounded above
        // leaves nothing.
        gap_lo = match span.hi {
            Some(hi) => Some(Some(hi.checked_add(1)?)),
            None => None,
        };
    }
    if let Some(start) = gap_lo {
        gaps.push(Narrow {
            lo: start,
            hi: None,
        });
    }
    Some(narrow_canonical(gaps))
}

/// The complement in the wide form.
fn wide_complement(spans: &[Span]) -> IntervalSet {
    let mut gaps = Vec::new();
    let mut gap_lo: Option<Option<WideInt>> = Some(None);
    for span in spans {
        // The gap ends just below this span, wherever the span begins: the
        // ends are exact at any width, so the integer before one is too.
        if let (Some(lo), Some(start)) = (&span.lo, gap_lo.take()) {
            gaps.push(Span {
                lo: start,
                hi: Some(lo.pred()),
            });
        }
        gap_lo = span.hi.as_ref().map(|hi| Some(hi.succ()));
    }
    if let Some(start) = gap_lo {
        gaps.push(Span {
            lo: start,
            hi: None,
        });
    }
    wide_canonical(gaps)
}

/// The preimage in the narrow form, or `None` where an end shifted by the
/// offset leaves the width.
fn narrow_preimage(spans: &[Narrow], offset: i64, stride: i64) -> Option<IntervalSet> {
    let offset = i128::from(offset);
    let stride = i128::from(stride);
    let mut lifted = Vec::with_capacity(spans.len());
    for span in spans {
        lifted.push(Narrow {
            lo: match span.lo {
                Some(lo) => Some(div_ceil(lo.checked_sub(offset)?, stride)),
                None => None,
            },
            hi: match span.hi {
                Some(hi) => Some(div_floor(hi.checked_sub(offset)?, stride)),
                None => None,
            },
        });
    }
    Some(narrow_canonical(lifted))
}

/// The preimage in the wide form.
fn wide_preimage(spans: &[Span], offset: i64, stride: i64) -> IntervalSet {
    let shifted = |bound: &WideInt| bound.plus(-offset);
    let lifted = spans
        .iter()
        .map(|span| Span {
            lo: span.lo.as_ref().map(|lo| shifted(lo).div_ceil(stride)),
            hi: span.hi.as_ref().map(|hi| shifted(hi).div_floor(stride)),
        })
        .collect();
    wide_canonical(lifted)
}

/// `a / b` rounded towards positive infinity, for a positive `b`. A nonzero
/// remainder needs a `b` of at least two, so the floor is at most half of `a`
/// and the step past it cannot saturate.
fn div_ceil(a: i128, b: i128) -> i128 {
    let quotient = a.div_euclid(b);
    if a.rem_euclid(b) == 0 {
        quotient
    } else {
        quotient.saturating_add(1)
    }
}

/// `a / b` rounded towards negative infinity, for a positive `b`.
fn div_floor(a: i128, b: i128) -> i128 {
    a.div_euclid(b)
}

#[cfg(test)]
mod tests;
