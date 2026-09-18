//! Sets of sets, as a union of powerset lines.
//!
//! `set[T]` denotes the *powerset* of `T`: every set whose members all lie in
//! `T`. That one observation decides the kind, because a powerset is not closed
//! under two of the three operations. Meets are: `P(A) ∧ P(B) = P(A ∧ B)`, since
//! a set whose members are in both is a set of the meet. Joins and complements
//! are not -- `P(A) ∨ P(B)` holds the sets drawn wholly from `A` and those drawn
//! wholly from `B`, and no single powerset is that -- so the representation is a
//! union of **lines**, each one powerset minus finitely many others.
//!
//! Emptiness of a line is where the kind pays for itself. A set in
//! `P(T) ∧ ⋀ⱼ ¬P(Sⱼ)` is a subset of `T` that escapes every `Sⱼ`, and escaping
//! `Sⱼ` means holding a member outside it. Those members can be chosen
//! independently, one per `j`, and collected into a single set -- so the line is
//! inhabited exactly when every `T ∧ ¬Sⱼ` is. Contrapositively:
//!
//! > a line is empty exactly when some `Sⱼ` covers `T`.
//!
//! The empty set is the reason the rule reads that way and not the other. `∅` is
//! a member of every powerset, so `P(T)` is never empty however empty `T` is:
//! `set[nothing]` holds exactly one value, and it is not `nothing`. A line with
//! no subtracted powerset is therefore always inhabited, which the rule gives
//! for free -- there is no `j` to find.

use super::budget;
use super::symbolic::Guard;
use super::values::Values;
use crate::verdict::Verdict;

/// The most lines a union may hold.
///
/// A complement multiplies the lines, for the reason a product multiplies
/// states: the complement of a union is an intersection of complements, and each
/// one is itself a union. The bound is a limit of the representation rather than
/// an approximation -- past it there is no sound union to substitute, so the
/// operation refuses.
pub const MAX_LINES: usize = 256;

/// One powerset minus finitely many others.
///
/// `elements` is the set every member lies in and `minus` the powersets this
/// line excludes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Line<G> {
    elements: Values<G>,
    minus: Vec<Values<G>>,
}

impl<G: Guard> Line<G> {
    /// Whether no set satisfies this line, proved.
    fn is_empty(&self) -> bool {
        self.emptiness() == Verdict::Empty
    }

    /// What is known about a set satisfying this line.
    ///
    /// A subtracted powerset that covers the elements empties it. A guard that
    /// cannot answer whether it covers leaves the line *unknown* rather than
    /// inhabited -- the distinction the whole verdict exists to make, since a
    /// line kept is only a line not disproved.
    fn emptiness(&self) -> Verdict {
        let mut verdict = Verdict::Inhabited;
        for excluded in &self.minus {
            match excluded.covers(&self.elements) {
                Some(true) => return Verdict::Empty,
                Some(false) => {}
                None => verdict = Verdict::Unknown,
            }
        }
        verdict
    }

    /// Whether the set whose members are `members` satisfies this line.
    fn holds(&self, members: &[G::Value]) -> bool {
        members.iter().all(|value| self.elements.holds(value))
            && self
                .minus
                .iter()
                .all(|excluded| members.iter().any(|value| !excluded.holds(value)))
    }

    /// The same line with its subtractions in canonical shape.
    ///
    /// Two steps, each removing a way to write one line twice. A subtracted
    /// powerset only matters where it meets `elements`, so it is cut down to
    /// that; and one that another covers adds nothing, since excluding the
    /// larger already excludes the smaller. What is left is an antichain, kept
    /// in order so two ways of writing it compare equal.
    fn tidy(mut self) -> Option<Line<G>> {
        for excluded in &mut self.minus {
            *excluded = excluded.meet(&self.elements)?;
        }
        let mut kept: Vec<Values<G>> = Vec::with_capacity(self.minus.len());
        for excluded in self.minus {
            // Already said by something kept -- including by an equal entry,
            // which is how a pair that covers each other keeps one rather than
            // losing both.
            if kept
                .iter()
                .any(|other| other.covers(&excluded) == Some(true))
            {
                continue;
            }
            kept.retain(|other| excluded.covers(other) != Some(true));
            kept.push(excluded);
        }
        kept.sort();
        self.minus = kept;
        Some(self)
    }
}

/// The lines a union of powerset lines complements into, or `None` past
/// [`MAX_LINES`].
///
/// The complement of a union is the intersection of the complements, and one
/// line's complement is itself a union: a set fails `P(T) ∧ ⋀ⱼ ¬P(Sⱼ)` by
/// escaping `T`, or by falling inside one of the `Sⱼ` after all.
fn complement_lines<G: Guard>(lines: &[Line<G>]) -> Option<Vec<Line<G>>> {
    let mut whole = SetLattice::all_lines();
    for line in lines {
        whole = product(&whole, &escapes(line))?;
    }
    Some(whole)
}

/// The lines a set outside `line` belongs to: escaping `T`, or falling inside
/// one of the `Sⱼ` after all.
///
/// One statement of it, because the complement of a union and a meet against
/// one both remove a line and would otherwise each carry their own reading of
/// what being outside it means.
fn escapes<G: Guard>(line: &Line<G>) -> Vec<Line<G>> {
    let mut alternatives = vec![Line {
        elements: Values::Every,
        minus: vec![line.elements.clone()],
    }];
    alternatives.extend(line.minus.iter().map(|excluded| Line {
        elements: excluded.clone(),
        minus: Vec::new(),
    }));
    alternatives
}

/// The lines of a meet, which is a meet of every pair: a set in both is drawn
/// wholly from both, so the elements meet and the subtractions collect.
fn product<G: Guard>(left: &[Line<G>], right: &[Line<G>]) -> Option<Vec<Line<G>>> {
    let mut lines = Vec::new();
    for mine in left {
        for theirs in right {
            // The bound says how wide the result may be; the budget says how
            // much reaching one may cost, and a product is the step that
            // multiplies. The three sibling lattices charge here and this is
            // the fourth, so a build that has spent its allowance refuses in
            // every one of them rather than in three.
            if !budget::spend() {
                return None;
            }
            if lines.len() >= MAX_LINES {
                // [`MAX_LINES`] bounds the *union*, and a union is only as wide
                // as it is once the lines holding no set and the repeats are
                // gone. Compacting here is what keeps the raw count from
                // standing in for that width, and the bound itself is
                // [`tidy`]'s: asked once, so a union as wide as the bound
                // builds whichever order its factors were multiplied in,
                // and one wider than it refuses whichever order they took.
                lines = tidy(lines)?;
            }
            let mut minus = mine.minus.clone();
            minus.extend(theirs.minus.iter().cloned());
            lines.push(Line {
                elements: mine.elements.meet(&theirs.elements)?,
                minus,
            });
        }
    }
    tidy(lines)
}

/// Drop the lines that hold nothing, put the rest in order, and refuse a union
/// past the bound.
fn tidy<G: Guard>(lines: Vec<Line<G>>) -> Option<Vec<Line<G>>> {
    let mut kept: Vec<Line<G>> = Vec::with_capacity(lines.len());
    for line in lines {
        let line = line.tidy()?;
        if !line.is_empty() && !kept.contains(&line) {
            kept.push(line);
        }
    }
    if kept.len() > MAX_LINES {
        return None;
    }
    kept.sort();
    Some(kept)
}

/// A set of sets, held as a union of powerset lines and a polarity.
///
/// The polarity is what keeps `complement` total, which the [`Guard`] the
/// sequence automaton reads its letters through requires of it. Complementing a
/// union of lines is a *product* -- an intersection of complements, each itself
/// a union -- so doing it eagerly could pass the bound and have nowhere sound to
/// go. Flipping a flag cannot, and the product is paid for later by the
/// operation that needs the lines, where a refusal is already allowed. The byte
/// automaton keeps complement total the same way, by flipping its accepting
/// states rather than rebuilding.
///
/// **Not canonical, unlike the other components.** Two unions can hold the same
/// sets and stay unequal: `P(A ∪ B)` is also the union of `P(A)`, `P(B)` and the
/// line that subtracts both, and recognising that costs a search for coverings
/// this does not run. So the laws here are checked against the *sets* rather
/// than by equality of the forms -- the same weakening the sequence automaton
/// takes, and for a reason of the same kind.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SetLattice<G: Guard> {
    lines: Vec<Line<G>>,
    /// Whether the lines are the sets held or the sets *not* held.
    negated: bool,
}

impl<G: Guard> SetLattice<G> {
    /// The one line every set satisfies.
    fn all_lines() -> Vec<Line<G>> {
        vec![Line {
            elements: Values::Every,
            minus: Vec::new(),
        }]
    }

    /// No set at all -- not even the empty one.
    #[must_use]
    pub fn empty() -> SetLattice<G> {
        SetLattice {
            lines: Vec::new(),
            negated: false,
        }
    }

    /// Every set: the one line that subtracts nothing and bounds nothing.
    #[must_use]
    pub fn all() -> SetLattice<G> {
        SetLattice {
            lines: SetLattice::all_lines(),
            negated: false,
        }
    }

    /// The sets whose members all lie in `elements`.
    #[must_use]
    pub fn of(elements: G) -> SetLattice<G> {
        SetLattice {
            lines: vec![Line {
                elements: Values::Only(elements),
                minus: Vec::new(),
            }],
            negated: false,
        }
    }

    /// The lines of the sets this holds, complementing a negated form.
    fn positive(&self) -> Option<Vec<Line<G>>> {
        if self.negated {
            complement_lines(&self.lines)
        } else {
            Some(self.lines.clone())
        }
    }

    /// Whether this holds no set.
    ///
    /// The empty set inhabits every powerset, so a line with nothing subtracted
    /// is never empty however empty its elements. A negated form has to be
    /// expanded first, and a refusal there reads as *not* empty -- the safe
    /// direction, since claiming emptiness is the claim that can be wrong.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.emptiness() == Verdict::Empty
    }

    /// What is known about this holding a set.
    ///
    /// A negated form has to be expanded first, and a refusal there is
    /// *unknown* rather than inhabited: past the bound there is no union to
    /// read, so nothing has been proved either way.
    #[must_use]
    pub fn emptiness(&self) -> Verdict {
        match self.positive() {
            Some(lines) => Verdict::any(lines.iter().map(Line::emptiness)),
            None => Verdict::Unknown,
        }
    }

    /// Whether the set whose members are `members` is held.
    #[must_use]
    pub fn holds(&self, members: &[G::Value]) -> bool {
        self.lines.iter().any(|line| line.holds(members)) != self.negated
    }

    /// The sets in either, or `None` past [`MAX_LINES`].
    #[must_use]
    pub fn union(&self, other: &SetLattice<G>) -> Option<SetLattice<G>> {
        if self.negated || other.negated {
            // De Morgan: `A ∪ B` is `¬(¬A ∩ ¬B)`, and the meet is the operation
            // that can drop a line mid-way. Expanding the negation first asks
            // the bound about a union that is only an intermediate.
            let met = self.complement().intersect(&other.complement())?;
            return Some(met.complement());
        }
        let mut lines = self.lines.clone();
        lines.extend(other.lines.iter().cloned());
        Some(SetLattice {
            lines: tidy(lines)?,
            negated: false,
        })
    }

    /// The sets in both, or `None` past [`MAX_LINES`] or where a guard refuses.
    ///
    /// A negated side is removed one line at a time rather than rebuilt into a
    /// union first. `¬⋁ᵢLᵢ` is `⋀ᵢ¬Lᵢ`, so both orders compute this set; what
    /// they differ in is the widest intermediate they ask [`MAX_LINES`] about,
    /// and rebuilding first multiplies every `¬Lᵢ` together with nothing to
    /// narrow the product.
    #[must_use]
    pub fn intersect(&self, other: &SetLattice<G>) -> Option<SetLattice<G>> {
        let mut lines = match (self.negated, other.negated) {
            (false, false) => product(&self.lines, &other.lines)?,
            (false, true) => self.lines.clone(),
            (true, false) => other.lines.clone(),
            // Two negated sides leave nothing positive to start from, so the
            // meet starts at every set and both sides narrow it.
            (true, true) => SetLattice::all_lines(),
        };
        for negated in [self, other].into_iter().filter(|side| side.negated) {
            for line in &negated.lines {
                lines = product(&lines, &escapes(line))?;
            }
        }
        Some(SetLattice {
            lines,
            negated: false,
        })
    }

    /// The sets this does not hold.
    ///
    /// Total, which is what the [`Guard`] contract asks and what keeps a
    /// descriptor complementable. The lines are rebuilt where the product fits,
    /// so the common forms stay comparable -- complementing `every set` gives
    /// back exactly `no set` rather than a second spelling of it -- and the
    /// polarity carries the rest, where there is no bounded union to rebuild
    /// into.
    #[must_use]
    pub fn complement(&self) -> SetLattice<G> {
        let flipped = SetLattice {
            lines: self.lines.clone(),
            negated: !self.negated,
        };
        // Expanded only where the expansion is one product, which is what
        // keeps the cheap forms canonical -- complementing "every set"
        // gives back exactly "no set" rather than a second spelling of it.
        // Past that the negation is carried: rebuilding a wide union's
        // complement here spends the build's allowance on an intermediate that
        // the meet it is headed for would have pruned, and a meet against a
        // negated side removes one lines at a time instead.
        if self.lines.len() > 1 {
            return flipped;
        }
        match flipped.positive() {
            Some(lines) => SetLattice {
                lines,
                negated: false,
            },
            None => flipped,
        }
    }
}

#[cfg(test)]
mod tests;
