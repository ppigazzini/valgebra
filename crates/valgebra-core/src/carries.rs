//! Whether the values a base admits can answer a constraint.
//!
//! A constraint that cannot be asked of a value is not a narrowing: reading a
//! length off an `int` raises, and the walk reads a raise as a non-member, so
//! `Annotated[int, MinLen(1)]` names a set that admits nothing and says
//! nothing about why. The frontend refuses such a constraint where it is
//! written, and the laws' buildable fragment draws only the pairs the frontend
//! builds. Both read this module, so the rule is stated once and the two
//! cannot drift: a base the frontend refuses is one the generator never draws,
//! and a base it accepts is one the generator may.
//!
//! The rule reads the base's *node*. A literal's node says only "a literal",
//! so the frontend rewrites a literal to the kind of its constant before
//! asking; the generator has no pool and asks the node as it is, which answers
//! `Maybe` for a literal and so never draws a refinement over one.

use crate::ir::{Schema, SeqKind};

/// The three answers, because a refusal needs certainty.
///
/// A constraint is refused only where *no* value of the base can answer it,
/// since that is the case where the refinement denotes the empty set and the
/// marker was written to narrow rather than to empty. Where the base is opaque
/// -- a class, the gradual atom, a literal, a recursive reference -- the check
/// stands aside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carries {
    /// Every value of the base can answer the constraint.
    Yes,
    /// No value of the base can, so the refinement admits nothing.
    No,
    /// The base does not say.
    Maybe,
}

impl Carries {
    /// The answer for a base that is a union of the two.
    ///
    /// A member that can answer makes the constraint a narrowing of the union
    /// rather than an emptying of it: `Annotated[int | str, MinLen(1)]` is the
    /// non-empty strings, which is a set a reader can mean.
    #[must_use]
    pub fn or(self, other: Carries) -> Carries {
        match (self, other) {
            (Carries::Yes, _) | (_, Carries::Yes) => Carries::Yes,
            (Carries::Maybe, _) | (_, Carries::Maybe) => Carries::Maybe,
            (Carries::No, Carries::No) => Carries::No,
        }
    }
}

/// The group Python orders a value within.
///
/// Python orders numbers with numbers, text with text, bytes with bytes, lists
/// with lists, tuples with tuples and sets with sets, and raises across the
/// groups. An order bound is a question about the pair, so the rule needs the
/// operand's group beside the base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderGroup {
    /// Any `numbers.Number`, `bool` among them.
    Number,
    /// A `str`.
    Text,
    /// A `bytes`.
    Bytes,
    /// A `list`.
    List,
    /// A `tuple`.
    Tuple,
    /// A `set` or a `frozenset`: the two share inclusion as their order.
    Set,
}

/// Fold `answer` over the members of a union, and stand aside elsewhere.
///
/// An intersection or a complement narrows a set this check does not compute,
/// and a refinement's answer is its own base's. A plain base answers `None`
/// here and is read by the caller's own table.
pub fn carries_through(base: &Schema, answer: &impl Fn(&Schema) -> Carries) -> Option<Carries> {
    match base {
        Schema::Union(members) => Some(members.iter().map(answer).fold(Carries::No, Carries::or)),
        Schema::Refine { base, .. } => Some(answer(base)),
        Schema::Intersection(_) | Schema::Complement(_) | Schema::Ref(_) | Schema::SelfRef(_) => {
            Some(Carries::Maybe)
        }
        _ => None,
    }
}

/// Whether the base's values have a length.
#[must_use]
pub fn carries_length(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_length) {
        return answer;
    }
    match base {
        Schema::Str
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::Yes,
        Schema::NoneType | Schema::Bool | Schema::Int | Schema::Float => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are text a pattern can be matched against.
#[must_use]
pub fn carries_pattern(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_pattern) {
        return answer;
    }
    match base {
        Schema::Str => Carries::Yes,
        Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are numbers, which is what a divisor needs.
#[must_use]
pub fn carries_division(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_division) {
        return answer;
    }
    match base {
        Schema::Bool | Schema::Int | Schema::Float => Carries::Yes,
        Schema::NoneType
        | Schema::Str
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are ordered against an operand of `group`, where
/// `None` is an operand of no group at all.
///
/// Two kinds order against *no* group: `None`, which has no comparison, and
/// `dict`, whose values are unordered however ordered their keys are. A bound
/// over either names the empty set whatever the operand is, which is why they
/// answer before the group is read.
///
/// The question is about the **pair**. A base with an order of its own says
/// nothing on its own: a set is ordered by inclusion and `{1} >= 0` raises all
/// the same, so a rule reading only the base admits every mismatched bound and
/// builds the schema that admits nothing.
#[must_use]
pub fn carries_order(base: &Schema, group: Option<OrderGroup>) -> Carries {
    let by_group = |base: &Schema| carries_order(base, group);
    if let Some(answer) = carries_through(base, &by_group) {
        return answer;
    }
    let matches = match base {
        Schema::Bool | Schema::Int | Schema::Float => group == Some(OrderGroup::Number),
        Schema::Str => group == Some(OrderGroup::Text),
        Schema::Bytes => group == Some(OrderGroup::Bytes),
        // A sequence orders against a sequence of its own container: a list and
        // a tuple are two kinds to Python's comparison as much as to this one.
        Schema::Seq { container, .. } => match container {
            SeqKind::List => group == Some(OrderGroup::List),
            SeqKind::Tuple => group == Some(OrderGroup::Tuple),
        },
        Schema::Coll { .. } => group == Some(OrderGroup::Set),
        Schema::NoneType | Schema::KeyedMap { .. } => return Carries::No,
        _ => return Carries::Maybe,
    };
    if matches { Carries::Yes } else { Carries::No }
}

#[cfg(test)]
mod tests;
