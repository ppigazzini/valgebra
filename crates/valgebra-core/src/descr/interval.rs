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

use core::cmp::{max, min};

/// One interval of integers, `lo..=hi`, where `None` is unbounded.
///
/// Inclusive at both ends, because the elements are integers: a half-open end
/// would be a second way to write the same set, and the point of the form is
/// that there is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Span {
    lo: Option<i64>,
    hi: Option<i64>,
}

impl Span {
    /// Whether this span holds no integer, which is `lo > hi` and only possible
    /// when both ends are bounded.
    fn is_empty(self) -> bool {
        matches!((self.lo, self.hi), (Some(lo), Some(hi)) if lo > hi)
    }

    fn holds(self, value: i64) -> bool {
        self.lo.is_none_or(|lo| lo <= value) && self.hi.is_none_or(|hi| value <= hi)
    }

    /// The integers in both spans.
    fn intersect(self, other: Span) -> Span {
        Span {
            lo: bound(self.lo, other.lo, max),
            hi: bound(self.hi, other.hi, min),
        }
    }

    /// Whether `other` starts at or before this span's end, counting adjacency:
    /// `[0, 3]` and `[4, 7]` have no integer between them, so they are one span
    /// and the canonical form says so.
    fn reaches(self, other: Span) -> bool {
        match (self.hi, other.lo) {
            (None, _) | (_, None) => true,
            // No `lo` can exceed `i64::MAX`, so saturating at the top means the
            // two do not meet -- which is the answer, rather than a wrap to the
            // bottom of the range.
            (Some(hi), Some(lo)) => lo <= hi.saturating_add(1),
        }
    }
}

/// Combine two bounds, where `None` is the unbounded end.
///
/// `pick` is `max` for a lower bound and `min` for an upper one, which is the
/// only difference between the two cases: an unbounded lower end is negative
/// infinity, so the bounded one wins; an unbounded upper end is positive
/// infinity, likewise.
fn bound(a: Option<i64>, b: Option<i64>, pick: fn(i64, i64) -> i64) -> Option<i64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(pick(a, b)),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

/// A set of integers: sorted, disjoint, non-adjacent spans.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct IntervalSet {
    spans: Vec<Span>,
}

impl IntervalSet {
    /// The empty set.
    #[must_use]
    pub fn empty() -> IntervalSet {
        IntervalSet { spans: Vec::new() }
    }

    /// Every integer.
    #[must_use]
    pub fn all() -> IntervalSet {
        IntervalSet {
            spans: vec![Span { lo: None, hi: None }],
        }
    }

    /// The integers from `lo` to `hi` inclusive, with `None` unbounded.
    #[must_use]
    pub fn between(lo: Option<i64>, hi: Option<i64>) -> IntervalSet {
        let span = Span { lo, hi };
        if span.is_empty() {
            IntervalSet::empty()
        } else {
            IntervalSet { spans: vec![span] }
        }
    }

    /// The one-element set.
    #[must_use]
    pub fn just(value: i64) -> IntervalSet {
        IntervalSet::between(Some(value), Some(value))
    }

    /// Whether this set holds no integer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Whether this set holds `value`.
    #[must_use]
    pub fn holds(&self, value: i64) -> bool {
        self.spans.iter().any(|span| span.holds(value))
    }

    /// The integers in either set.
    ///
    /// Merging is what restores the canonical form: the spans are gathered,
    /// sorted by their lower end, and folded so that any two that meet or touch
    /// become one.
    #[must_use]
    pub fn union(&self, other: &IntervalSet) -> IntervalSet {
        let spans: Vec<Span> = self.spans.iter().chain(&other.spans).copied().collect();
        IntervalSet { spans }.canonical()
    }

    /// The integers in both sets.
    ///
    /// Pairwise: each span of one meets each span of the other in at most one
    /// interval, and what comes out is canonicalised rather than assumed
    /// ordered.
    #[must_use]
    pub fn intersect(&self, other: &IntervalSet) -> IntervalSet {
        let mut spans = Vec::new();
        for mine in &self.spans {
            for theirs in &other.spans {
                let met = mine.intersect(*theirs);
                if !met.is_empty() {
                    spans.push(met);
                }
            }
        }
        IntervalSet { spans }.canonical()
    }

    /// Every integer this set does not hold.
    ///
    /// Read off the gaps: the run below the first span, one between each
    /// adjacent pair, and the run above the last. The empty set has no span to
    /// skip, so the same walk gives every integer.
    #[must_use]
    pub fn complement(&self) -> IntervalSet {
        let mut spans = Vec::new();
        // Where the next gap starts: `None` until the first span is seen, which
        // is the unbounded run below it.
        let mut gap_lo: Option<Option<i64>> = Some(None);
        for span in &self.spans {
            // The gap ends just below this span. No integer sits below the
            // bottom of the range, so a span starting there leaves no gap --
            // which is what `checked_sub` returning `None` says, and what
            // saturating would have turned into a gap holding that integer.
            if let (Some(lo), Some(start)) = (span.lo, gap_lo)
                && let Some(hi) = lo.checked_sub(1)
            {
                spans.push(Span {
                    lo: start,
                    hi: Some(hi),
                });
            }
            gap_lo = match span.hi {
                // Likewise at the top: a span reaching the last integer leaves
                // no run above it, and neither does one unbounded above.
                Some(hi) => hi.checked_add(1).map(Some),
                None => None,
            };
        }
        if let Some(start) = gap_lo {
            spans.push(Span {
                lo: start,
                hi: None,
            });
        }
        IntervalSet { spans }.canonical()
    }

    /// Sort, drop the empties, and merge what touches.
    fn canonical(self) -> IntervalSet {
        let mut spans: Vec<Span> = self.spans.into_iter().filter(|s| !s.is_empty()).collect();
        // An unbounded lower end sorts first, which is where negative infinity
        // belongs.
        spans.sort_by_key(|span| (span.lo.is_some(), span.lo));
        let mut merged: Vec<Span> = Vec::with_capacity(spans.len());
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
        IntervalSet { spans: merged }
    }

    /// The set `{ k : offset + stride * k is in self }`, for a positive `stride`.
    ///
    /// The change of variable a modulus needs: a residue class is
    /// order-isomorphic to the integers, and this carries a set across that
    /// isomorphism. Each end divides, rounding *inward* -- up at the lower end,
    /// down at the upper -- because a `k` outside the rounded range maps to an
    /// integer outside the span.
    #[must_use]
    pub fn preimage(&self, offset: i64, stride: i64) -> IntervalSet {
        debug_assert!(stride > 0, "a stride is a positive step");
        let spans = self
            .spans
            .iter()
            .map(|span| Span {
                lo: span
                    .lo
                    .map(|lo| div_ceil(lo.saturating_sub(offset), stride)),
                hi: span
                    .hi
                    .map(|hi| div_floor(hi.saturating_sub(offset), stride)),
            })
            .collect();
        IntervalSet { spans }.canonical()
    }
}

/// `a / b` rounded towards positive infinity, for a positive `b`.
fn div_ceil(a: i64, b: i64) -> i64 {
    let quotient = a.div_euclid(b);
    if a.rem_euclid(b) == 0 {
        quotient
    } else {
        quotient.saturating_add(1)
    }
}

/// `a / b` rounded towards negative infinity, for a positive `b`.
fn div_floor(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

#[cfg(test)]
mod tests;
