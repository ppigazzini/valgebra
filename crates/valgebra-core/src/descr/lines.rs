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
//! the kindless slot -- which is where they were -- cannot describe a value that
//! has both a builtin kind and a class.

use std::sync::Arc;

use super::budget;
use super::records::RecordLattice;
use super::{Component, Descr, Op};
use crate::Kind;
use crate::decision::Verdict;

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
    fn complement(&self) -> [Line; 2] {
        [
            Line::everything(self.structure.complement()),
            Line {
                structure: self.structure.top_like(),
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
    /// `whole` is the kind these lines are a part of: complementing *no* lines
    /// is the whole kind, and an empty union carries no line to read one off.
    fn positive(&self, whole: &Component) -> Option<Vec<Line>> {
        if self.negated {
            complement_lines(&self.lines, whole)
        } else {
            Some(self.lines.clone())
        }
    }

    /// What is known about this kind admitting a value.
    ///
    /// A union of lines, so the verdict is the union's: empty when every line is
    /// proved empty, inhabited as soon as one is. A negated form has to be
    /// expanded first, and a refusal there is *unknown* rather than inhabited --
    /// past the bound there is no union to read, so nothing is proved either way.
    pub(crate) fn emptiness(&self, whole: &Component, kind: Option<Kind>) -> Verdict {
        match self.positive(whole) {
            Some(lines) => Verdict::any(lines.iter().map(|line| line.emptiness(kind))),
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
    pub(crate) fn combine(&self, other: &Lines, op: Op, whole: &Component) -> Option<Lines> {
        let mine = self.positive(whole)?;
        let theirs = other.positive(whole)?;
        let lines = match op {
            Op::Union => {
                let mut lines = mine;
                lines.extend(theirs);
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
    pub(crate) fn complement(&self, whole: &Component) -> Lines {
        let flipped = Lines {
            lines: self.lines.clone(),
            negated: !self.negated,
        };
        match flipped.positive(whole) {
            Some(lines) => Lines {
                lines,
                negated: false,
            },
            None => flipped,
        }
    }
}

/// The lines a union of lines complements into, or `None` past [`MAX_LINES`].
///
/// De Morgan over the lines: `¬⋁ᵢ Lᵢ` is `⋀ᵢ ¬Lᵢ`, and each `¬Lᵢ` is the two
/// lines [`Line::complement`] gives. The fold starts from the whole kind, which
/// is what complementing no lines yields.
fn complement_lines(lines: &[Line], whole: &Component) -> Option<Vec<Line>> {
    let mut kept = vec![Line::everything(whole.clone())];
    for line in lines {
        kept = product(&kept, &line.complement())?;
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
            if lines.len() >= MAX_LINES || !budget::spend() {
                return None;
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
fn tidy(lines: Vec<Line>) -> Option<Vec<Line>> {
    let mut kept: Vec<Line> = Vec::with_capacity(lines.len());
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some(at) = kept.iter().position(|held| held.objects == line.objects)
            && let Some(held) = kept.get_mut(at)
            && let Some(joined) = held.structure.combine(&line.structure, Op::Union)
        {
            held.structure = joined;
            continue;
        }
        kept.push(line);
    }
    kept.sort();
    kept.dedup();
    (kept.len() <= MAX_LINES).then_some(kept)
}
