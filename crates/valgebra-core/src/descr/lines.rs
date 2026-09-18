//! A kind's values as a union of lines.
//!
//! A class and an attribute record constrain a value *within* its kind rather
//! than instead of it: a dataclass deriving from `int` has an integer's
//! structure and a class, and a component that can hold only one of the two
//! cannot decide it. So a kind's values are a **union of lines**, each line a
//! structure met with the objects it may be:
//!
//! ```text
//! kind  =  ⋁ᵢ ( structureᵢ  ∧  classesᵢ ∧ attrsᵢ )
//! ```
//!
//! The kind partition stays above this, which is the point. Disjointness across
//! kinds is still free -- an `int` and a `str` share no value whatever classes
//! either carries -- and the DNF is *inside* one kind, over a handful of lines,
//! rather than over the eleven kinds at once. A meet is a pairwise product and a
//! complement splits each line into two, both bounded by [`MAX_LINES`].
//!
//! The alternative shapes each fail one test, recorded in
//! `docs/dev/01-schema-ir.md` under "Where a class and an attribute record go":
//! a DNF over the whole descriptor loses the partition, and scoping classes to
//! the kindless slot cannot describe a value that has both a builtin kind and a
//! class.

use std::sync::Arc;

use super::budget;
use super::records::RecordLattice;
use super::{Component, Descr, Op, Whole};
use std::borrow::Cow;

use crate::kind::Kind;
use crate::verdict::Verdict;

/// The most lines one kind may carry.
///
/// A meet multiplies the two sides' line counts and a complement doubles them,
/// so the union needs a bound for the same reason every other union here does:
/// past it there is no sound set to substitute, since one too wide is
/// complemented into one too narrow. The refusal reaches the caller instead.
pub(super) const MAX_LINES: usize = 256;

/// One line: the values of a kind whose structure holds and which are among the
/// objects the line admits.
///
/// `objects` carries the classes and the attributes together, because they
/// constrain the same value in the same way and the record lattice already
/// decides them as one ([`RecordLattice`]). A line with no class and no
/// attribute constraint has `objects` at its top, which is every object, and is
/// then exactly its structure.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Line {
    structure: Component,
    objects: RecordLattice<Arc<Descr>>,
}

impl Line {
    /// The line admitting every value of the structure's kind.
    fn everything(structure: Component) -> Line {
        Line {
            structure,
            objects: RecordLattice::all(),
        }
    }

    /// What is known about this line admitting a value of `kind`.
    ///
    /// A meet of the two halves: empty as soon as either is, inhabited only when
    /// both are proved so. A class the core cannot enumerate the subclasses of
    /// leaves the object half unknown, and the line with it.
    ///
    /// The kind is passed down rather than read off the structure, because the
    /// object half is asked *within* it: a constraint on objects sits on every
    /// kind's line, and a class that confines its instances to no kind gives no
    /// value of this one. `None` is the line of objects with no builtin kind.
    fn emptiness(&self, kind: Option<Kind>) -> Verdict {
        Verdict::every(
            [
                self.structure.emptiness(),
                self.objects.emptiness_of_kind(kind),
            ]
            .into_iter(),
        )
    }

    /// Whether this line is proved to hold nothing.
    ///
    /// Asked without a kind, and that is not a shortcut: the kind can only turn
    /// a *proof of a value* into an unknown, never a line into an empty one, so
    /// a line empty on one kind's terms is empty on every kind's. Dropping a
    /// line is the one decision that reads emptiness rather than inhabitance.
    fn is_empty(&self) -> bool {
        self.emptiness(None) == Verdict::Empty
    }

    /// The two operations, on two lines of the same kind.
    fn combine(&self, other: &Line, op: Op) -> Option<Line> {
        Some(Line {
            structure: self.structure.combine(&other.structure, op)?,
            objects: match op {
                Op::Union => self.objects.union(&other.objects)?,
                Op::Intersect => self.objects.intersect(&other.objects)?,
            },
        })
    }

    /// The values of the kind this line does not admit, as a union.
    ///
    /// `¬(s ∧ o)` is `¬s ∨ ¬o`: either the structure fails, or it holds and the
    /// object constraints do not. Two lines, which is why a complement doubles
    /// the count rather than exploding it.
    ///
    /// **Both halves end up within the kind**, and the fold below is what puts
    /// them there. A representation does not always say which kind it serves --
    /// one automaton carries the `str` words and the `bytes` ones -- and their
    /// universes are not the same set: every byte string is a `bytes`, and a
    /// `str` is one that encodes a sequence of code points. So the structure's
    /// raw complement holds words no value of the kind takes, and
    /// [`complement_lines`] seeds its fold with the kind's own top and meets
    /// each pair into it, which cuts them back. Cutting again here is the same
    /// meet taken twice.
    ///
    /// **Total.** A component's complement is a flag or a flip and a record
    /// lattice's is the same, so neither half can refuse; the fold below is
    /// where a complement runs out of room, and it is the product that does it.
    fn complement(&self, whole: Whole) -> [Line; 2] {
        [
            Line::everything(self.structure.complement()),
            Line {
                structure: whole.component(),
                objects: self.objects.complement(),
            },
        ]
    }
}

/// A kind's component: the union of its lines, or their complement.
///
/// No lines is the empty set and no lines *negated* is the whole kind, which is
/// what makes the two bounds cost nothing: a descriptor holding one kind is
/// eleven empty vectors and one line.
///
/// The polarity is the same device the record lattice carries, for the same
/// reason: a complement must be **total** -- the [`Guard`](super::Guard)
/// contract asks for one -- and the positive form of a complement can pass
/// [`MAX_LINES`]. Flipping a flag always succeeds; normalising back to a union
/// is attempted and, where it does not fit, the flag carries the set instead.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Lines {
    lines: Vec<Line>,
    /// Whether the lines are the values held or the values *not* held.
    negated: bool,
}

impl Lines {
    /// No value of the kind.
    pub(crate) fn bottom() -> Lines {
        Lines {
            lines: Vec::new(),
            negated: false,
        }
    }

    /// Every value of the kind whose structure this describes.
    pub(crate) fn everything(structure: Component) -> Lines {
        Lines::of(Line::everything(structure))
    }

    /// Every value of the kind that `objects` admits, whatever its structure.
    pub(crate) fn objects(whole: &Component, objects: RecordLattice<Arc<Descr>>) -> Lines {
        Lines::of(Line {
            structure: whole.clone(),
            objects,
        })
    }

    /// One line, or the bottom where that line is proved to hold nothing.
    ///
    /// **A `Lines` never carries a line proved empty.** Every operation drops
    /// them ([`tidy`]), so one built with one is a value no operation can return
    /// -- and `a ∪ a` would tidy it away and stop equalling `a`. Equality is what
    /// the lattice laws are asked in, so a form only a constructor can produce is
    /// a form that breaks them.
    fn of(line: Line) -> Lines {
        Lines {
            lines: if line.is_empty() {
                Vec::new()
            } else {
                vec![line]
            },
            negated: false,
        }
    }

    /// The lines of the values this holds, complementing a negated form.
    ///
    /// `whole` names the kind these lines are a part of: complementing *no*
    /// lines is the whole kind, and an empty union carries no line to read one
    /// off. It is a name rather than a set because this is the only place that
    /// wants the set, and only on the negated path -- so the eleven wholes an
    /// operation runs over are built where they are read and nowhere else.
    ///
    /// Borrowed where the list is already positive, which is the common case
    /// and the one asked most often: a component is asked whether it is empty
    /// once per kind per constraint, and copying a list to read whether it
    /// holds a line was the whole cost of asking.
    fn positive(&self, whole: Whole) -> Option<Cow<'_, [Line]>> {
        if self.negated {
            complement_lines(&self.lines, whole).map(Cow::Owned)
        } else {
            Some(Cow::Borrowed(&self.lines))
        }
    }

    /// What is known about this kind admitting a value.
    ///
    /// A union of lines, so the verdict is the union's: empty when every line is
    /// proved empty, inhabited as soon as one is. A negated form has to be
    /// expanded first, and a refusal there is *unknown* rather than inhabited --
    /// past the bound there is no union to read, so nothing is proved either way.
    pub(crate) fn emptiness(&self, whole: Whole) -> Verdict {
        match self.positive(whole) {
            Some(lines) => Verdict::any(lines.iter().map(|line| line.emptiness(whole.kind()))),
            None => Verdict::Unknown,
        }
    }

    /// Whether some line admits a value with this structure, class and
    /// attributes.
    ///
    /// The two halves are asked of the *same* line, which is what the shape is
    /// for: a value meeting one line's structure and another's class meets
    /// neither. The polarity flips the answer, as it does for the record atoms.
    pub(crate) fn admits(
        &self,
        structural: &dyn Fn(&Component) -> bool,
        objects: &dyn Fn(&RecordLattice<Arc<Descr>>) -> bool,
    ) -> bool {
        self.lines
            .iter()
            .any(|line| structural(&line.structure) && objects(&line.objects))
            != self.negated
    }

    /// Every value in either union, or in both.
    ///
    /// A union concatenates and a meet multiplies, which is where the bound
    /// bites.
    pub(crate) fn combine(&self, other: &Lines, op: Op, whole: Whole) -> Option<Lines> {
        // A meet against a negated side removes one of its lines at a time.
        // `⋀ᵢ¬Lᵢ` is `¬⋁ᵢLᵢ`, so both orders compute the same values; what they
        // differ in is the widest intermediate they ask [`MAX_LINES`] about.
        // Expanding the negation first multiplies every `¬Lᵢ` together with
        // nothing to narrow the product, which is a width the answer rarely
        // has -- and a bound reached under one spelling of a difference and
        // not another makes a relation's answer a property of how it was
        // written.
        if op == Op::Intersect && (self.negated || other.negated) {
            let mut lines = match (self.negated, other.negated) {
                (false, _) => self.lines.clone(),
                (true, false) => other.lines.clone(),
                // Two negated sides leave nothing positive to start from, so
                // the meet starts at the whole kind and both sides narrow it.
                (true, true) => vec![Line::everything(whole.component())],
            };
            for negated in [self, other].into_iter().filter(|side| side.negated) {
                for line in &negated.lines {
                    lines = product(&lines, &line.complement(whole))?;
                }
            }
            return Some(Lines {
                lines,
                negated: false,
            });
        }
        let mine = self.positive(whole)?;
        let theirs = other.positive(whole)?;
        let lines = match op {
            Op::Union => {
                let mut lines = mine.into_owned();
                // The right-hand list is appended where it lies: owning it
                // first buys a second allocation and a move, and the lines
                // are cloned into the result either way.
                match theirs {
                    Cow::Borrowed(rest) => lines.extend_from_slice(rest),
                    Cow::Owned(rest) => lines.extend(rest),
                }
                tidy(lines)?
            }
            Op::Intersect => product(&mine, &theirs)?,
        };
        Some(Lines {
            lines,
            negated: false,
        })
    }

    /// Every value of the kind this union does not admit.
    ///
    /// Total, which is what the [`Guard`](super::Guard) contract asks. The lines
    /// are rebuilt where the product fits, so the common forms stay comparable,
    /// and the polarity carries the rest.
    pub(crate) fn complement(&self, whole: Whole) -> Lines {
        // A negated union's complement is its own lines held positively; a
        // positive one's is De Morgan over them, where that fits, and the
        // same lines under the flipped flag where it does not.
        if self.negated {
            return Lines {
                lines: self.lines.clone(),
                negated: false,
            };
        }
        // Expanded only where the expansion is one product, which is what
        // keeps the cheap forms canonical -- complementing "no lines" gives
        // back exactly the whole kind rather than a second spelling of it.
        // Past that the negation is carried, and the meet it is headed for
        // removes one line at a time instead.
        if self.lines.len() > 1 {
            return Lines {
                lines: self.lines.clone(),
                negated: true,
            };
        }
        match complement_lines(&self.lines, whole) {
            Some(lines) => Lines {
                lines,
                negated: false,
            },
            None => Lines {
                lines: self.lines.clone(),
                negated: true,
            },
        }
    }
}

/// The lines a union of lines complements into, or `None` past [`MAX_LINES`].
///
/// De Morgan over the lines: `¬⋁ᵢ Lᵢ` is `⋀ᵢ ¬Lᵢ`, and each `¬Lᵢ` is the two
/// lines [`Line::complement`] gives. The fold starts from the whole kind, which
/// is what complementing no lines yields.
fn complement_lines(lines: &[Line], whole: Whole) -> Option<Vec<Line>> {
    let mut kept = vec![Line::everything(whole.component())];
    for line in lines {
        kept = product(&kept, &line.complement(whole))?;
    }
    Some(kept)
}

/// The lines of a meet, which is a meet of every pair.
///
/// Every pair charges the build's allowance, because this is the loop that
/// multiplies: the pairs are the product of the two counts, and a fold over
/// several unions raises that to a power. The line bound stops the result from
/// being too wide, and the allowance stops the *work* from being too much
/// before the width is known.
fn product(left: &[Line], right: &[Line]) -> Option<Vec<Line>> {
    let mut lines = Vec::new();
    for mine in left {
        for theirs in right {
            if !budget::spend() {
                return None;
            }
            if lines.len() >= MAX_LINES {
                // [`MAX_LINES`] bounds the *union*, and a union is only as wide
                // as it is once the lines proved empty and the repeats are
                // gone. Compacting here is what keeps the raw count from
                // standing in for that width, and the bound itself is
                // [`tidy`]'s: asked once, so a union as wide as the bound
                // builds whichever order its factors were multiplied in,
                // and one wider than it refuses whichever order they took.
                lines = tidy(lines)?;
            }
            lines.push(mine.combine(theirs, Op::Intersect)?);
        }
    }
    tidy(lines)
}

/// Drop the lines proved empty, merge the ones that differ only in structure,
/// put the rest in order, and refuse past the bound.
///
/// Dropping is not an optimisation: a line proved empty contributes no value to
/// the union, so removing it leaves the same set and keeps the count from
/// growing on shapes that describe nothing. A line that is merely *unknown*
/// stays, because it may yet hold a value.
///
/// Merging is distributivity, and it is what keeps a kind that constrains no
/// object at *one* line: `(s₁ ∧ o) ∨ (s₂ ∧ o)` is `(s₁ ∨ s₂) ∧ o`, so two lines
/// agreeing on their objects are one line over the joined structure. Without it
/// `bool` would be a line per boolean and `int` a line per interval, and two
/// descriptors admitting the same values would stop comparing equal. Where the
/// structure union passes its own bound the two lines stay apart, which is the
/// same set spelled longer.
///
/// The order is what makes two equal unions compare equal, as far as equality
/// here goes.
fn tidy(mut lines: Vec<Line>) -> Option<Vec<Line>> {
    // In place: the caller has just built this list and the result is the same
    // list shorter, so a second one of the same width is an allocation per
    // meet and per union. `kept` is how many of the front are keepers, and a
    // line that survives is swapped up to join them.
    let mut kept = 0;
    for at in 0..lines.len() {
        {
            // The keepers are `lines[..kept]`, which the split puts in `front`;
            // the line being read is the first of `back`. Continuing drops it,
            // falling through keeps it.
            let (front, back) = lines.split_at_mut(at);
            let Some(line) = back.first() else { break };
            if line.is_empty() {
                continue;
            }
            if let Some(keeper) = front
                .iter()
                .take(kept)
                .position(|held| held.objects == line.objects)
                && let Some(held) = front.get_mut(keeper)
                && let Some(joined) = held.structure.combine(&line.structure, Op::Union)
            {
                held.structure = joined;
                continue;
            }
        }
        lines.swap(kept, at);
        kept += 1;
    }
    lines.truncate(kept);
    lines.sort();
    lines.dedup();
    (lines.len() <= MAX_LINES).then_some(lines)
}

#[cfg(test)]
mod tests;
