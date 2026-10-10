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
//! `Sⱼ` means holding a member outside it. So:
//!
//! > a line is empty when some `Sⱼ` covers `T`.
//!
//! The converse holds of mathematical sets, where one member per `j` can be
//! chosen and the members collected, and it fails of Python's. A Python set
//! holds no two members that compare equal, and `==` joins values the element
//! descriptor keeps apart -- `1` and `True`, `(1,)` and `(True,)` -- so the
//! members collected may be one. A set of `Literal[1, True]` outside both
//! `set[Literal[1]]` and `set[Literal[True]]` must hold `1` and `True`, and
//! `{1, True}` is `{1}`. So a line is read inhabited where one member escapes
//! every `Sⱼ`, a set of one, and otherwise where the letter says one member per
//! `Sⱼ` stays apart ([`Members::one_member_each`]).
//!
//! The empty set is the reason the rule reads that way and not the other. `∅` is
//! a member of every powerset, so `P(T)` is never empty however empty `T` is:
//! `set[nothing]` holds exactly one value, and it is not `nothing`. A line with
//! no subtracted powerset is therefore always inhabited, which the rule gives
//! for free -- there is no `j` to find.

use std::sync::Arc;

use super::polar::{PolarUnion, Summand};
use super::symbolic::Guard;
use super::values::Values;
use super::{Component, Descr, Whole};
use crate::kind::Kind;
use crate::verdict::Verdict;

/// The most lines a union may hold.
///
/// A complement multiplies the lines, for the reason a product multiplies
/// states: the complement of a union is an intersection of complements, and each
/// one is itself a union. The bound is a limit of the representation rather than
/// an approximation -- past it there is no sound union to substitute, so the
/// operation refuses.
pub const MAX_LINES: usize = 256;

/// What the set lattice asks of a letter beyond the algebra: which of its
/// distinct values a Python set keeps as one member.
pub trait Members: Guard {
    /// Whether one Python set can hold a member of each of `sets`, each of
    /// which holds a value.
    ///
    /// A member taken from each and the members collected is such a set, unless
    /// two of the members are equal under `==` and the set keeps one of them.
    /// The default is a letter that equates none of its distinct values.
    fn one_member_each(sets: &[Self]) -> Verdict {
        let _ = sets;
        Verdict::Inhabited
    }
}

/// One powerset minus finitely many others.
///
/// `elements` is the set every member lies in and `minus` the powersets this
/// line excludes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Line<G> {
    elements: Values<G>,
    minus: Vec<Values<G>>,
}

impl<G: Members> Line<G> {
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
        if self.minus.len() < 2 {
            return verdict;
        }
        // One member escaping every subtraction is a set of one, which no `==`
        // can collapse. Where it takes one member per subtraction, whether one
        // set holds them all is the letter's to say -- and a subtraction the
        // loop above could not read leaves its escape unknown, which leaves
        // that answer unknown too.
        let escapes_alone = self
            .minus
            .iter()
            .try_fold(Values::none(), |all, excluded| all.join(excluded))
            .and_then(|all| all.covers(&self.elements));
        if escapes_alone == Some(false) {
            Verdict::Inhabited
        } else {
            self.one_member_each()
        }
    }

    /// Whether one set holds a member of every `T ∧ ¬Sⱼ`.
    ///
    /// An escape holding every value offers a member no other can collide with,
    /// so only the ones a letter bounds are asked.
    fn one_member_each(&self) -> Verdict {
        let mut escapes = Vec::with_capacity(self.minus.len());
        for excluded in &self.minus {
            match self.elements.meet(&excluded.complement()) {
                Some(Values::Only(escape)) => escapes.push(escape),
                Some(Values::Every) => {}
                None => return Verdict::Unknown,
            }
        }
        G::one_member_each(&escapes)
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

/// The lines a set outside `line` belongs to: escaping `T`, or falling inside
/// one of the `Sⱼ` after all.
///
/// One statement of it, because the complement of a union and a meet against
/// one both remove a line and would otherwise each carry their own reading of
/// what being outside it means.
fn escapes<G: Members>(line: &Line<G>) -> Vec<Line<G>> {
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

impl<G: Members> Summand for Line<G> {
    /// Nothing: a line's universe is every set.
    type Within = ();
    type Complement = Vec<Line<G>>;
    const MAX: usize = MAX_LINES;

    /// The one line every set satisfies: it subtracts nothing and bounds
    /// nothing.
    fn top((): ()) -> Line<G> {
        Line {
            elements: Values::Every,
            minus: Vec::new(),
        }
    }

    /// A set in both lines is drawn wholly from both, so the elements meet and
    /// the subtractions collect.
    fn meet(&self, other: &Line<G>) -> Option<Line<G>> {
        let mut minus = self.minus.clone();
        minus.extend(other.minus.iter().cloned());
        Some(Line {
            elements: self.elements.meet(&other.elements)?,
            minus,
        })
    }

    /// A set fails `P(T) ∧ ⋀ⱼ ¬P(Sⱼ)` by escaping `T`, or by falling inside one
    /// of the `Sⱼ` after all ([`escapes`]).
    fn complement(&self, (): ()) -> Vec<Line<G>> {
        escapes(self)
    }

    /// Each line with its subtractions in canonical shape, without the ones
    /// holding no set, each once, and in order. Once by equality, for the
    /// reason the record atoms give: a letter's equality can be coarser than
    /// its order, so a sort need not put two equal lines side by side.
    fn compacted(lines: Vec<Line<G>>) -> Option<Vec<Line<G>>> {
        let mut kept: Vec<Line<G>> = Vec::with_capacity(lines.len());
        for line in lines {
            let line = line.tidy()?;
            if !line.is_empty() && !kept.contains(&line) {
                kept.push(line);
            }
        }
        kept.sort();
        Some(kept)
    }
}

/// A set of sets, held as a union of powerset lines and a polarity.
///
/// The polarity is what keeps `complement` total, which the [`Guard`] the
/// sequence automaton reads its letters through requires of it. Complementing a
/// union of lines is a *product* -- an intersection of complements, each itself
/// a union -- so doing it eagerly could pass the bound and have nowhere sound to
/// go. The device is the one every lattice built from parts shares, in
/// `descr/polar.rs`; the byte automaton keeps complement total the same way, by
/// flipping its accepting states rather than rebuilding.
///
/// **Not canonical, unlike the other components.** Two unions can hold the same
/// sets and stay unequal: `P(A ∪ B)` is also the union of `P(A)`, `P(B)` and the
/// line that subtracts both, and recognising that costs a search for coverings
/// this does not run. So the laws here are checked against the *sets* rather
/// than by equality of the forms -- the same weakening the sequence automaton
/// takes, and for a reason of the same kind.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SetLattice<G: Members>(PolarUnion<Line<G>>);

impl<G: Members> SetLattice<G> {
    /// No set at all -- not even the empty one.
    #[must_use]
    pub fn empty() -> SetLattice<G> {
        SetLattice(PolarUnion::of(Vec::new()))
    }

    /// Every set: the one line that subtracts nothing and bounds nothing.
    #[must_use]
    pub fn all() -> SetLattice<G> {
        SetLattice(PolarUnion::of(vec![Line::top(())]))
    }

    /// The sets whose members all lie in `elements`.
    #[must_use]
    pub fn of(elements: G) -> SetLattice<G> {
        SetLattice(PolarUnion::of(vec![Line {
            elements: Values::Only(elements),
            minus: Vec::new(),
        }]))
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
    #[must_use]
    pub fn emptiness(&self) -> Verdict {
        self.0.verdict((), Line::emptiness)
    }

    /// Whether the set whose members are `members` is held.
    #[must_use]
    pub fn holds(&self, members: &[G::Value]) -> bool {
        self.0.holds(|line| line.holds(members))
    }

    /// The sets in either, or `None` past [`MAX_LINES`].
    #[must_use]
    pub fn union(&self, other: &SetLattice<G>) -> Option<SetLattice<G>> {
        self.0.union(&other.0, ()).map(SetLattice)
    }

    /// The sets in both, or `None` past [`MAX_LINES`] or where a guard refuses.
    #[must_use]
    pub fn intersect(&self, other: &SetLattice<G>) -> Option<SetLattice<G>> {
        self.0.intersect(&other.0, ()).map(SetLattice)
    }

    /// The sets this does not hold.
    ///
    /// Total, which is what the [`Guard`] contract asks and what keeps a
    /// descriptor complementable.
    #[must_use]
    pub fn complement(&self) -> SetLattice<G> {
        SetLattice(self.0.complement(()))
    }

    /// The lines as held, whichever polarity reads them.
    #[cfg(test)]
    fn lines(&self) -> &[Line<G>] {
        self.0.summands()
    }

    /// Whether the lines are the sets *not* held.
    #[cfg(test)]
    const fn negated(&self) -> bool {
        self.0.is_negated()
    }
}

/// Python's `==` equates two distinct scalars of a descriptor in one way: a
/// `bool`, an `int` and a `float` holding one number -- `True == 1 == 1.0`.
/// That is the collision read here, and the one the map lattice reads between
/// two dict keys. A tuple or a frozenset compares its members, so `(1,) ==
/// (True,)` as well, but its kind holds its subclasses, and one whose `==` is
/// identity collides with nothing; a value of a kind outside the three is read
/// as that member.
///
/// A set offering such a value, or more numbers than there are sets to serve,
/// is read as offering a member nothing collides with. The rest offer a few
/// numbers each, and the members collected stay apart unless two of those sets
/// hold one number in two kinds; that pair is unknown rather than refuted,
/// since another choice of members may still keep them apart.
impl Members for Arc<Descr> {
    fn one_member_each(sets: &[Arc<Descr>]) -> Verdict {
        let mut few: Vec<Vec<(Kind, Number)>> = Vec::new();
        for set in sets {
            match numbers_offered(set, sets.len()) {
                Offer::Free => {}
                Offer::Numbers(numbers) => few.push(numbers),
                Offer::Unknown => return Verdict::Unknown,
            }
        }
        let collide = |a: &[(Kind, Number)], b: &[(Kind, Number)]| {
            a.iter().any(|(kind, number)| {
                b.iter()
                    .any(|(other_kind, other)| kind != other_kind && number.equals(*other))
            })
        };
        let apart = few.iter().enumerate().all(|(i, mine)| {
            few.get(i + 1..)
                .into_iter()
                .flatten()
                .all(|theirs| !collide(mine, theirs))
        });
        if apart {
            Verdict::Inhabited
        } else {
            Verdict::Unknown
        }
    }
}

/// A number as `==` reads it across the three kinds that hold one.
#[derive(Debug, Clone, Copy)]
enum Number {
    /// A `bool` or an `int`.
    Integral(i128),
    /// A `float`, never `nan`.
    Float(f64),
}

impl Number {
    /// Whether the two are equal under `==`, which compares an `int` and a
    /// `float` exactly rather than after rounding one to the other.
    fn equals(self, other: Number) -> bool {
        match (self, other) {
            (Number::Integral(a), Number::Integral(b)) => a == b,
            (Number::Float(a), Number::Float(b)) => a.total_cmp(&b).is_eq(),
            (Number::Integral(whole), Number::Float(float))
            | (Number::Float(float), Number::Integral(whole)) => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "an integral float converts exactly; one past the range saturates \
                              to a value no bound reaches"
                )]
                let round_trip = float as i128;
                float.trunc().total_cmp(&float).is_eq() && round_trip == whole
            }
        }
    }
}

/// What one set offers a Python set that must hold a member of several.
enum Offer {
    /// A member read as equal to nothing chosen for the others.
    Free,
    /// Only these numbers, each with the kind it is a value of.
    Numbers(Vec<(Kind, Number)>),
    /// Neither is proved.
    Unknown,
}

/// What `set` offers a Python set serving `serving` sets at once.
///
/// Read off the lines proved to hold a value, so a number listed is one some
/// value of `set` holds, and a line proved neither way leaves the offer unknown.
/// More numbers than there are sets to serve leaves one no other set's member
/// can take, which is a member as free as a value of another kind.
fn numbers_offered(set: &Descr, serving: usize) -> Offer {
    const NUMBER_KINDS: [Kind; 3] = [Kind::Bool, Kind::Int, Kind::Float];
    let mut confined = true;
    let others = set
        .kinds
        .iter()
        .zip(Kind::ALL)
        .filter(|(_, kind)| !NUMBER_KINDS.contains(kind))
        .map(|(lines, kind)| lines.emptiness(Whole::Kind(kind)))
        .chain([set.other.emptiness(Whole::Kindless)]);
    for verdict in others {
        match verdict {
            Verdict::Inhabited => return Offer::Free,
            Verdict::Empty => {}
            Verdict::Unknown => confined = false,
        }
    }
    if !confined {
        return Offer::Unknown;
    }
    let mut numbers = Vec::new();
    for kind in NUMBER_KINDS {
        let Some(structures) = set.component(kind).inhabited_structures(Whole::Kind(kind)) else {
            return Offer::Unknown;
        };
        for structure in structures {
            match structure {
                Component::Booleans(flags) => numbers.extend(
                    [false, true]
                        .into_iter()
                        .filter(|flag| flags.holds(*flag))
                        .map(|flag| (kind, Number::Integral(i128::from(flag)))),
                ),
                Component::Integers(integers) => match integers.values_up_to(serving) {
                    Some(values) => {
                        numbers.extend(values.into_iter().map(|v| (kind, Number::Integral(v))));
                    }
                    None => return Offer::Free,
                },
                Component::Floats(floats) => match floats.values_up_to(serving) {
                    Some(values) => {
                        numbers.extend(values.into_iter().map(|v| (kind, Number::Float(v))));
                    }
                    None => return Offer::Free,
                },
                _ => return Offer::Unknown,
            }
        }
    }
    Offer::Numbers(numbers)
}

#[cfg(test)]
mod tests;
