//! The two three-valued answers, and what each is an answer *about*.
//!
//! [`Verdict`] is what can be established about one schema's set, and
//! [`Relation`] what can be established about a *relation* between two.
//!
//! Both sit below the two deciders rather than inside either, because both
//! answer in them: a procedure returning `false` for "no" and for "I ran out of
//! budget" alike returns something strictly weaker than a decision, and telling
//! those apart is what these two are for. `docs/dev/02-decision.md` describes
//! the pair that shares them.

/// What a decision could establish about a *relation* between two schemas.
///
/// The companion of [`Verdict`], which says what could be established about one
/// schema's set. A `bool` answer to "is this a subtype of that" conflates the
/// two things a caller most needs to tell apart: a value of the left that is
/// outside the right, which refutes the relation, and a rule that declined --
/// an oracle with no answer, a constructor pair no rule relates, a descent the
/// work bound stopped. Both read as `false`, so the conservative half of the
/// procedure is invisible from outside and cannot be counted, listed, or held
/// to a ledger.
///
/// The public relations still answer `bool`, because that is what the contract
/// promises: [`Relation::Holds`] is `true` and both other answers are `false`.
/// What the three values buy is that "not proven" is a value the tests can
/// count and the pages can enumerate.
///
/// A rule that reports [`Relation::Fails`] is claiming a proof. Where a rule is
/// sound but incomplete -- it establishes the relation when it fires and says
/// nothing when it does not -- the answer is [`Relation::Unknown`], and the
/// combinators below propagate that: a conjunction is unknown when a conjunct
/// is, and a disjunction is unknown when no disjunct holds and one is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// Proven: every value of the subject belongs to the other schema.
    Holds,
    /// Refuted: a value of the subject is outside the other schema.
    ///
    /// A witness is what makes this answer. The descriptor's reading produces
    /// one directly, by proving the difference holds a value; the structural
    /// rules produce one by finding a mismatch of shapes, which stands on
    /// *some* value of the subject and so is read against the subject's own
    /// emptiness before it is believed.
    Fails,
    /// Neither, and the procedure says so rather than answering `false`.
    Unknown,
}

impl Relation {
    /// The relation as the boundary reports it: only a proof is `true`.
    #[must_use]
    #[inline]
    pub fn holds(self) -> bool {
        matches!(self, Relation::Holds)
    }

    /// A relation from a decision that is exact in both directions.
    ///
    /// For a rule whose `false` is a refutation rather than a decline -- set
    /// inclusion between two region sets, an arity that cannot match. A rule
    /// that is sound one way and silent the other spells its own arms.
    #[must_use]
    #[inline]
    pub(crate) fn decided(held: bool) -> Relation {
        if held {
            Relation::Holds
        } else {
            Relation::Fails
        }
    }

    /// A relation from a rule that proves inclusion and declines otherwise.
    #[must_use]
    #[inline]
    pub(crate) fn proven(held: bool) -> Relation {
        if held {
            Relation::Holds
        } else {
            Relation::Unknown
        }
    }

    /// Both must hold, where the second is reached only if the first does.
    ///
    /// The first answer that is not a proof is the answer, which is not what
    /// [`Relation::all`] does with a list of conjuncts. The two differ because
    /// the operands do: `all` folds conjuncts of one claim about one pair, so
    /// any one of them refutes it; this chains *steps*, and the second is often
    /// a question the first has to have answered before it means anything --
    /// a repeated element against a tail, once the prefix is known to align.
    /// A refutation from a step whose premise is unproven is not a value.
    /// Reporting the weaker of the two is the safe direction.
    #[must_use]
    #[inline]
    pub(crate) fn and(self, other: impl FnOnce() -> Relation) -> Relation {
        match self {
            Relation::Holds => other(),
            answer => answer,
        }
    }

    /// The proof this answer carries, with its refutation dropped.
    ///
    /// For a step that is sound in one direction only. Reducing a refinement to
    /// its base is the case: a refinement is a *subset* of its base, so the
    /// base's inclusion carries it -- but the base's refutation does not, since
    /// the value that puts the base outside may be one the constraints exclude.
    /// `list[int]` is not below the empty list and `list[int]` of length at most
    /// zero is, and both are read from the same base.
    #[must_use]
    #[inline]
    pub(crate) fn proof_only(self) -> Relation {
        match self {
            Relation::Holds => Relation::Holds,
            _ => Relation::Unknown,
        }
    }

    /// The refutation this answer carries, with its proof dropped.
    ///
    /// For a rule that narrows the supertype: a value outside the wider set is
    /// outside the narrower one, so the refutation carries, and a value inside
    /// the wider set says nothing about the narrower.
    #[must_use]
    #[inline]
    pub(crate) fn refutation_only(self) -> Relation {
        match self {
            Relation::Fails => Relation::Fails,
            _ => Relation::Unknown,
        }
    }

    /// The refutation a mismatch carries, read against what the subject holds.
    ///
    /// A rule refutes by finding a mismatch -- two arities that cannot align, a
    /// key one side requires and the other does not guarantee -- and names the
    /// value that stands against the inclusion only implicitly: it is *some*
    /// value of the subject, shaped the way the subject says. A subject with no
    /// value names none, and the empty set is below every set, the shape it can
    /// never take included. So the mismatch is read against the subject: proved
    /// inhabited it refutes, proved empty it establishes the opposite, and
    /// undecided it decides nothing.
    #[must_use]
    #[inline]
    pub(crate) const fn of_mismatch(subject: Verdict) -> Relation {
        match subject {
            Verdict::Inhabited => Relation::Fails,
            Verdict::Empty => Relation::Holds,
            Verdict::Unknown => Relation::Unknown,
        }
    }

    /// The second decider, asked only where the first declines.
    ///
    /// Two procedures answer inclusion here: the structural rules and the
    /// difference the descriptor builds. A proof from either is a proof, and a
    /// *refutation* from either is a refutation -- so the only answer worth a
    /// second opinion is `Unknown`. Reaching for the second decider after the
    /// first has refuted would be asking a question already answered, and
    /// discarding the answer it gave.
    #[must_use]
    #[inline]
    pub(crate) fn or_else(self, second: impl FnOnce() -> Relation) -> Relation {
        match self {
            Relation::Unknown => second(),
            answer => answer,
        }
    }

    /// The inclusion a difference decides: `a <= b` is `a & ~b = {}`.
    ///
    /// The whole of the descriptor's reading, named where the two vocabularies
    /// meet. An empty difference proves the inclusion; a difference proved to
    /// hold a value refutes it, because that value is in `a` and outside `b`;
    /// and a difference the descriptor cannot decide decides nothing.
    #[must_use]
    #[inline]
    pub(crate) const fn of_difference(difference: Verdict) -> Relation {
        match difference {
            Verdict::Empty => Relation::Holds,
            Verdict::Inhabited => Relation::Fails,
            Verdict::Unknown => Relation::Unknown,
        }
    }

    /// Every item, with [`Relation::any`]'s propagation of a decline.
    ///
    /// The dual of `any`, and dual for the same reason. The items are conjuncts
    /// of one claim about one pair -- every member of a union below the same
    /// supertype, every field of a record against its counterpart -- so a
    /// refutation from any one of them is a value of the subject the supertype
    /// rejects, whatever the others answer. Reading the *first* answer that is
    /// not a proof made that depend on the order the conjuncts happened to be
    /// in: two records differing only in the order of their fields decided
    /// differently, one in a microsecond and one in two hundred. A decline is
    /// not an answer, so it cannot end the fold.
    ///
    /// A `Fails` still ends it, and no conjunct can produce one from a
    /// coinductive assumption -- an assumption yields `Holds` -- so the answer
    /// rests on a value rather than on the hypothesis being discharged.
    #[inline]
    pub(crate) fn all(items: impl IntoIterator<Item = Relation>) -> Relation {
        let mut declined = false;
        for answer in items {
            match answer {
                Relation::Holds => {}
                Relation::Unknown => declined = true,
                Relation::Fails => return Relation::Fails,
            }
        }
        if declined {
            Relation::Unknown
        } else {
            Relation::Holds
        }
    }

    /// Any item, with [`Relation::or`]'s propagation of a decline.
    #[inline]
    pub(crate) fn any(items: impl IntoIterator<Item = Relation>) -> Relation {
        let mut declined = false;
        for answer in items {
            match answer {
                Relation::Holds => return Relation::Holds,
                Relation::Unknown => declined = true,
                Relation::Fails => {}
            }
        }
        if declined {
            Relation::Unknown
        } else {
            Relation::Fails
        }
    }
}

/// What a decision could establish about a set.
///
/// A `bool` answer conflates two different things. `is_empty` returning `false`
/// means "not proven empty", which covers a schema proven to admit values and a
/// schema the procedure gave up on -- an opaque leaf, or a descent the work bound
/// stopped. The caller cannot tell them apart, and neither can an instrument
/// watching from outside, so a budget exhaustion at a realistic size reads as a
/// confident answer.
///
/// The public relations still answer `bool`, because that is what soundness
/// promises: `Unknown` and `Inhabited` both mean "not proven empty". What the
/// three values buy is that the difference is now *visible* -- to a test and to
/// a gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Proven to admit no value.
    Empty,
    /// Proven to admit at least one value.
    Inhabited,
    /// Neither: an opaque leaf the core cannot read, or a descent the work bound
    /// stopped before it reached one.
    Unknown,
}

impl Verdict {
    /// Whether this verdict proves emptiness. The reduction the public relations
    /// make, named once: `Unknown` is not a proof, so it answers with
    /// `Inhabited`.
    pub(crate) const fn is_empty(self) -> bool {
        matches!(self, Verdict::Empty)
    }

    /// The verdict for a value that must satisfy **every** part: a product, a
    /// meet of positions, a record's required fields.
    ///
    /// One empty part empties the whole, whatever the others are, so `Empty`
    /// absorbs. Otherwise every part must be proven inhabited for the whole to
    /// be, and one `Unknown` leaves it unknown. An empty iterator is `Inhabited`:
    /// nothing is required, so the empty value satisfies it.
    pub(crate) fn every(parts: impl Iterator<Item = Verdict>) -> Verdict {
        let mut verdict = Verdict::Inhabited;
        for part in parts {
            match part {
                Verdict::Empty => return Verdict::Empty,
                Verdict::Unknown => verdict = Verdict::Unknown,
                Verdict::Inhabited => {}
            }
        }
        verdict
    }

    /// The verdict for a value that may satisfy **any** part: a union.
    ///
    /// The dual of [`every`](Self::every). One inhabited part inhabits the whole,
    /// so `Inhabited` absorbs; every part must be proven empty for the whole to
    /// be. An empty iterator is `Empty`, which is what a union of no members
    /// denotes.
    pub(crate) fn any(parts: impl Iterator<Item = Verdict>) -> Verdict {
        let mut verdict = Verdict::Empty;
        for part in parts {
            match part {
                Verdict::Inhabited => return Verdict::Inhabited,
                Verdict::Unknown => verdict = Verdict::Unknown,
                Verdict::Empty => {}
            }
        }
        verdict
    }
}
