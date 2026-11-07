//! The structural rules: subtyping, equivalence and disjointness, decided by
//! matching constructors against each other.
//!
//! One module per surface beside this one: the emptiness every relation
//! reduces to, the readings a pair no rule decides is given, and four
//! constructor surfaces -- a refinement's bounds, sequences split across a
//! union, keyed maps and attribute records, and tables of constants. What only
//! a binding can answer is asked of the oracle at the crate root. What stays
//! here is the coinductive procedure that ties them together: the trail,
//! the hypothesis a recursive pair is proved under, the shape match, and the
//! hand-off to the set representation where the rules decline.

mod constraints;
mod emptiness;
mod lengths;
mod literals;
mod products;
mod readings;
mod records;

use std::cell::{Cell, OnceCell, RefCell};
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::descr::budget;
use crate::descr::lower::{
    Constants, WORK, an_empty_reading_stands, lower_unfolded, parts_a_cut_reaches, read_widened,
};
use crate::ir::{Constraint, DefIx, Members, Polarity, Schema, SeqShape};
use crate::kind::{Region, Regions};
use crate::unordered_pairs;
use crate::verdict::{Relation, Verdict};

use crate::oracle::{denotes_a_set_within, has_complementary_pair};
use constraints::constraint_entailed;
use emptiness::{ReadBack, class_with_attributes, narrowed_meet, the_rules_find_a_value};
use literals::{
    a_union_spilled_past_as_a_table, finite_meet_below, finite_set, finite_set_below,
    spills_past_a_table, the_constant_speaks_for_the_literal,
};
use products::{linear_subtype, seq_splits_across_union};
use readings::{outside_every_kind, refuted_by_a_multiple, tables_disjoint, tag_pairs};
use records::{
    a_shared_field_refuted_by_the_sets, attr_record_subtype, keyed_map_splits_across_union,
    keyed_map_subtype, keyed_meet_below,
};

pub use crate::oracle::{LeafRelations, NoLeafRelations};

/// The most decision steps one top-level query may take before it stops and
/// returns the conservative answer. Subtyping distributes over unions and
/// intersections and emptiness recurses the structural fragment, so a deeply
/// nested Boolean combination can demand work exponential in its depth.
///
/// **The bound is on cost; termination does not rest on it.** Every goal a
/// query asks is a pair drawn from a finite closure of its subterms -- the
/// subterms, their complements, meets of those, and unions of some members of
/// a union among them -- because the four rules that build a term build it
/// out of the subterms: [`seq_splits_across_union`] meets a component with the
/// complements of branch components, [`keyed_map_splits_across_union`] a field
/// with the complements of member fields, the union rule narrows a union to
/// the branches the subject can meet, and a meet or a complement folds to a
/// bound.
/// Every other step asks a smaller pair or unfolds a reference onto the trail,
/// where a pair that comes back is answered by hypothesis, so every path is
/// finite. `docs/dev/02-decision.md` writes the argument out, and
/// `decision::goal_tests` holds its premise. The closure is exponential in the
/// subterms, a meet per subset, so a query that terminates can still outlast
/// any caller, and that is what this ceiling is for.
///
/// One budget is threaded through a whole top-level query — subtyping, the
/// emptiness checks it calls into and the two split rules share it, and the two
/// directions of an equivalence share it — so no rule spends a second one. A
/// lowering is the one cost it does not count: where the rules can neither prove
/// nor refute `A <= ~B`, the `Complement` arm of `subtype_by_shape` asks the
/// set representation whether `B` is empty, and that question is bounded by the
/// lowering's own `Bounds` -- nodes read, nesting descended, multiplying work
/// -- once for each goal that reaches it.
///
/// **A memo is mostly not what would lower it.** Counted by
/// `decision::goal_tests` over the other workloads' shapes and over a record of
/// thirty-two fields sharing one inner schema, the goals a query *repeats*
/// number zero: the trail absorbs recursion, and the field and position caches
/// absorb the shape where one goal is asked once per field.
///
/// The relation matrix is where that stops holding. A meet against a union
/// repeats **four** goals per query, twice in that corpus and eight over it:
/// the union distribution asks the meet of each branch, and the meet rule asks
/// the class atom against the same thing once per member with no cache between
/// the two rules. What a table over goals would collect is a saving on that one
/// shape, measured rather than assumed.
///
/// The ceiling is for a schema built to blow up the decision -- the
/// distributed tower `subtyping_terminates_on_a_distributed_tower` drives to
/// it -- and there the query answers undecided, which the conservative
/// contract allows. A relation the rules have no rule for is undecided well
/// below it, and that is the procedure's incompleteness rather than this
/// bound's. A memo over goals that would share the work is not built: the
/// paragraphs above measure what it would save.
pub(crate) const DECISION_BUDGET: u32 = 1_000_000;

/// The steps a pair respelled may spend, of what its query has left, in a
/// question's first asking.
///
/// **This bound is cost.** A respelled question declines as dearly as the
/// plain one where it declines too, and drawing on the whole remainder turned a
/// pair that declined in a millisecond into one that declined in fifty. On the
/// shapes a translation from JSON Schema writes, a quarter of this decides all
/// but one in sixty of what this does.
pub(crate) const RESPELLING: u32 = 4096;

/// The steps a pair respelled may spend in a question's second asking
/// ([`asked_again`]).
///
/// **This bound is cost.** Answers sat a few steps under [`RESPELLING`], so a
/// sound reading added beside the others lost them, and a question that ends
/// undecided at an allowance is asked again under this one. Raising
/// [`RESPELLING`] itself loses answers the first asking gives: a respelling
/// charged to the query spends steps the readings after it decided with, and
/// at four times the figure a mapping met with a complement lost eleven of its
/// refutations. Sixteen times it regains what four does not: two answers a
/// step-free kinds reading moved past four times the figure.
pub(crate) const RESPELLING_AGAIN: u32 = 65_536;

/// The steps a goal read by the sets is charged ([`Schema::decided_by_its_leaves`]).
///
/// **This bound is cost.** No step counts a lowering, so this is what holds how
/// many of them one asking reads: sixty-four under [`RESPELLING`].
const LEAF_READING: u32 = 64;

thread_local! {
    /// The respelling allowance of the asking a question on this thread is in:
    /// [`RESPELLING`], or [`RESPELLING_AGAIN`] while [`asked_again`] asks a
    /// second time.
    static RESPELLING_NOW: Cell<u32> = const { Cell::new(RESPELLING) };
}

/// The steps a pair respelled may spend in the asking under way.
pub(super) fn respelling() -> u32 {
    RESPELLING_NOW.with(Cell::get)
}

/// Whether a root question's first asking, undecided under `budget` with
/// `met` bounds met before it began, is asked again ([`asked_again`]): a bound
/// was met since, the question has steps of its own left, and it is no second
/// asking already. Read at the root entries after the first asking, so a
/// question that decides pays one read of the bounds met and one test.
#[inline]
fn asks_again(met: u64, budget: &Budget) -> bool {
    bounds_met() != met && budget.left() > 0 && respelling() == RESPELLING
}

/// A root question asked once more, under [`RESPELLING_AGAIN`] and a budget of
/// its own, where its first asking under [`RESPELLING`] ended undecided at a
/// bound with steps of its own left ([`asks_again`]).
///
/// The pattern of [BBC+26, §6]'s fast-fail strategy: the cheap check first,
/// and the dearer one only "when the incomplete (anyOf-r) strategy is not
/// sufficient". Here the second is the same procedure with a larger
/// allowance, not a complete one. The first asking is the whole answer
/// wherever it decides, so the second takes no answer away. A question that
/// met no bound declines the same with more steps
/// (`a_query_no_bound_stopped_answers_alike_with_more_steps`), and one that
/// spent all of its own declines at that budget, which a second asking would
/// spend again, so neither is asked twice. A question asked inside a second
/// asking is asked once, under the allowance of the asking it is in.
///
/// Out of line and cold: the root entries ask the first time inline, as they
/// did with one asking, since a closure around the first asking cost the
/// decision workload 6% of its instructions.
#[cold]
#[inline(never)]
fn asked_again<T>(ask: impl FnOnce(&Budget) -> T) -> T {
    let _again = SecondAsking::open();
    ask(&Budget::new(DECISION_BUDGET))
}

/// A question's second asking under way on this thread, closed when dropped,
/// unwinding included.
struct SecondAsking;

impl SecondAsking {
    fn open() -> Self {
        RESPELLING_NOW.with(|now| now.set(RESPELLING_AGAIN));
        Self
    }
}

impl Drop for SecondAsking {
    fn drop(&mut self) {
        RESPELLING_NOW.with(|now| now.set(RESPELLING));
    }
}

/// The most levels of its own recursion one decision query holds open.
///
/// The step budget bounds what a query costs and never how deep it goes: two
/// recursive schemas whose bodies nest 100 and 99 lists around the back edge
/// meet the coinductive hypothesis only after `lcm(100, 99)` levels of goals,
/// each a chain of native frames, a few steps apiece. A goal level costs about
/// 1 KiB of native stack on the profiled wheel and an emptiness level less,
/// measured, so 512 of them fit in half of the 1 MiB that is the least stack
/// any thread `CPython` creates has. Past the bound the query answers undecided,
/// which the conservative contract always allows: a schema a caller writes
/// unfolds a handful of levels, and only one built to nest coprime cycles or a
/// long chain of definitions reaches it.
pub(crate) const MAX_DECISION_DEPTH: u32 = 512;

/// What one top-level query may spend: steps of work, and levels of its own
/// recursion held open on the stack.
///
/// One value for the two because every recursive call is handed the budget
/// already, so the depth travels with it. Counting a level on every goal that
/// reaches the rules and on every emptiness question costs the decision
/// workloads 2.5% to 3.1%; a count kept per thread read 6% to 11%, a
/// thread-local in a shared library being a call to find on every level, and
/// one taken only through `verdict_rec` read less and left a union's members
/// uncounted.
///
/// The steps are fuel [JV20, §10.5], a bound no theorem derives, so which
/// pairs a query decides turns on the path it takes: a query no bound stopped
/// answers alike with more steps, and one a bound stopped may not
/// (`docs/dev/10-theory.md`, "The decision is fuel").
pub(crate) struct Budget {
    steps: Cell<u32>,
    depth: Cell<u32>,
}

impl Budget {
    /// A query's budget of `steps`, with no level open.
    pub(crate) const fn new(steps: u32) -> Self {
        Self {
            steps: Cell::new(steps),
            depth: Cell::new(0),
        }
    }

    /// The steps not yet spent: what a pair respelled may draw on, and what the
    /// tests that pin a cost read.
    pub(crate) fn left(&self) -> u32 {
        self.steps.get()
    }

    /// `reading` run on at most `cap` of the steps left, and what it spent
    /// charged to the query: an allowance the query's total still bounds, on
    /// this budget, so the meets the query holds open stay open inside it
    /// (`OpenMeet`, in [`emptiness`], keys them by the budget).
    pub(super) fn within<T>(&self, cap: u32, reading: impl FnOnce() -> T) -> T {
        let left = self.steps.get();
        let allowance = left.min(cap);
        self.steps.set(allowance);
        let answer = reading();
        let spent = allowance.saturating_sub(self.steps.get());
        self.steps.set(left - spent);
        answer
    }
}

/// Run `descent` one level below the budget's current depth, or answer
/// `refused` where the level would pass [`MAX_DECISION_DEPTH`].
///
/// The goal recursion and the emptiness recursion each descend through this
/// once a level, and the product and record rules once for each branch they
/// narrow by; all of them share the count, since an emptiness question asked
/// deep inside a goal, and a goal asked at the bottom of a split, run on the
/// same stack. No guard closes the level:
/// a query that unwinds drops its budget with it, so the count is restored on
/// the return alone -- a guard's drop put an unwinding path on every goal and
/// read dearer than the count itself.
pub(super) fn descending<T>(budget: &Budget, refused: T, descent: impl FnOnce() -> T) -> T {
    let depth = budget.depth.get();
    if depth >= MAX_DECISION_DEPTH {
        a_bound_is_met();
        return refused;
    }
    budget.depth.set(depth + 1);
    let answer = descent();
    budget.depth.set(depth);
    answer
}

/// Whether `sub ⊆ sup` holds by membership alone: the subject is a branch of a
/// union supertype [BBC+26, (anyOf-r)], or the supertype is a member of a meet
/// subject.
///
/// Each is set containment read off the terms -- `X ⊆ X ∪ Y` and `X ∩ Y ⊆ X`
/// -- and reads nothing below the terms' first level: the reflexive case is the
/// one a rule-based inclusion proves in linear time [BBC+26, §6, p. 8]. So it is
/// never refused for want of a step. The rule for each shape asks it before
/// anything that spends -- a meet before it asks its members, a union
/// supertype before it asks its branches -- and a goal that reaches a spent
/// budget asks it rather than declining, since a placement that waits behind
/// a dear member is a placement the member's work starved.
#[inline]
fn placed_by_membership(sub: &Schema, sup: &Schema) -> bool {
    matches!(sup, Schema::Union(branches) if branches.contains(sub))
        || matches!(sub, Schema::Intersection(members) if members.contains(sup))
}

/// Whether `¬inner ⊆ sup` holds as the complements pushed one level in: the
/// complement of a union is the meet of the complements, and the complement of
/// a meet the union of them [FCB08, Lem. 6.2], and the pushed term is `sup`
/// itself or placed in it by membership.
///
/// One level, as a rule-based procedure pushes a negation lazily, no further
/// than the question in hand needs [BBC+26, §6.2, p. 9]. So `¬(X ∪ ¬Y)` is
/// `¬X ∩ Y`, which is below `Y` by membership, and `¬(X ∪ Y)` below `¬X ∩ ¬Y`
/// is the one term spelled twice. The two terms denote one set, so a placement
/// of the pushed one is a proof about the complement as written.
///
/// Only the placements, not the rules below them. Asked of every rule, the
/// pushed term is a new subject each arm reads again, and where the pair
/// declines it reads it for nothing -- a complement of a union holding a
/// recursive universe, asked below `str`, takes ten times as long to decline
/// -- for a few answers in ten thousand. Out of line, behind the caller's test
/// of the complement's shape, since it builds a term.
#[inline(never)]
fn pushed_complement_is_placed(inner: &Schema, sup: &Schema, defs: &[Schema]) -> bool {
    let pushed = match inner {
        Schema::Union(branches) => {
            Schema::meet_within(branches.iter().map(|b| b.clone().complement()), defs)
        }
        Schema::Intersection(members) => {
            Schema::union_within(members.iter().map(|m| m.clone().complement()), defs)
        }
        _ => return false,
    };
    pushed == *sup || placed_by_membership(&pushed, sup)
}

/// The union `members` denote, with two branches that split one set by another
/// merged into the set they split, or `None` where no two branches do.
///
/// `(R ∩ B) ∪ (R ∩ ¬B)` is `R`: the meet distributes over the join, and the
/// join of `B` and `¬B` is everything where `B` is a set [FCB08, Def. 4.1].
/// `B` is asked whether it is one, as the complement law asks, since a
/// predicate answers its two occurrences apart. Each branch is read as its conjuncts, a nested meet's
/// among them, since a meet is the set of what it holds however it is spelled:
/// the constructor flattens `B` into the branch where `B` is a meet, a rule may
/// not, and either way `Ge(0) ∩ NUMBER` is the conjuncts `Ge(0)`, `int | float`
/// and `¬bool`, which with `Ge(0) ∩ ¬NUMBER` merge into `Ge(0)`.
fn merged_split(members: &[Schema], cx: &Cx<'_>) -> Option<Schema> {
    for (at, holding) in members.iter().enumerate() {
        // The branch holding `¬B` holds a complement, at its top or in a meet
        // nested in it, and no other branch is read as that one.
        let Schema::Intersection(top) = holding else {
            continue;
        };
        if !top
            .iter()
            .any(|member| matches!(member, Schema::Complement(_) | Schema::Intersection(_)))
        {
            continue;
        }
        let holding = conjuncts_of(holding);
        for &complement in &holding {
            let Schema::Complement(split) = complement else {
                continue;
            };
            let parts = conjuncts_of(split);
            let shared: Vec<&Schema> = holding
                .iter()
                .copied()
                .filter(|member| *member != complement)
                .collect();
            let splits = |other: &Schema| {
                if !matches!(other, Schema::Intersection(_)) {
                    return false;
                }
                let other = conjuncts_of(other);
                other
                    .iter()
                    .all(|member| shared.contains(member) || parts.contains(member))
                    && shared.iter().all(|member| other.contains(member))
                    && parts.iter().all(|part| other.contains(part))
            };
            let Some(paired) = members
                .iter()
                .enumerate()
                .position(|(i, other)| i != at && splits(other))
            else {
                continue;
            };
            if !denotes_a_set_within(split, cx.oracle, cx.defs) {
                continue;
            }
            let merged = Schema::meet_within(shared.into_iter().cloned(), cx.defs);
            let rest = members
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != at && *i != paired)
                .map(|(_, member)| member.clone());
            return Some(Schema::union_within(rest.chain([merged]), cx.defs));
        }
    }
    None
}

/// The conjuncts of `schema`, a nested meet's own in its place: a schema that
/// is no meet is its own one conjunct.
fn conjuncts_of(schema: &Schema) -> Vec<&Schema> {
    let mut held = Vec::new();
    conjuncts(std::slice::from_ref(schema), &mut held);
    held
}

/// Push the conjuncts of `members` onto `into`, a nested meet's own in its
/// place.
fn conjuncts<'s>(members: &'s [Schema], into: &mut Vec<&'s Schema>) {
    for member in members {
        match member {
            Schema::Intersection(inner) => conjuncts(inner, into),
            other => into.push(other),
        }
    }
}

/// Whether `schema` holds every value of `kind` (`wholly`) or none of them, as
/// the kind readings answer it with no oracle: the laws hold the two readings
/// to membership.
#[cfg(test)]
pub(crate) fn kind_reading(
    schema: &Schema,
    kind: crate::Kind,
    defs: &[Schema],
    wholly: bool,
) -> bool {
    let bit = 1 << kind.position();
    if wholly {
        schema.kinds_held_whole(&NoLeafRelations, defs) & bit != 0
    } else {
        schema.kinds_possibly_held(&NoLeafRelations, defs) & bit == 0
    }
}

/// Spend one unit of `budget`; returns `false` when it is already exhausted, the
/// signal a budgeted decision uses to stop and report the conservative answer.
pub(super) fn spend(budget: &Budget) -> bool {
    if let Some(remaining) = budget.steps.get().checked_sub(1) {
        budget.steps.set(remaining);
        true
    } else {
        a_bound_is_met();
        false
    }
}

/// Spend `steps` units of `budget` at once where that many are left, and none
/// where they are not: what a verdict read back from a memo spends
/// ([`emptiness::ReadBack::AsRead`]). Returns whether they were spent.
pub(super) fn spend_at_once(budget: &Budget, steps: u32) -> bool {
    budget
        .steps
        .get()
        .checked_sub(steps)
        .map(|remaining| budget.steps.set(remaining))
        .is_some()
}

thread_local! {
    /// How many times a reading on this thread has stopped at a step or a
    /// depth bound ([`bounds_met`]): a step refused, a level refused, and a
    /// reading declined for want of a step.
    static BOUNDS_MET: Cell<u64> = const { Cell::new(0) };
}

/// Count a reading stopped at a step or a depth bound: where it answered
/// `Unknown` for want of a step or a level, which more of either might decide.
fn a_bound_is_met() {
    BOUNDS_MET.with(|met| met.set(met.get() + 1));
}

/// How many readings on this thread have stopped at a bound so far.
///
/// Read before and after a reading: where the count moved, its `Unknown` is a
/// function of the steps and the depth it was read with as well as of its
/// terms, and where it did not, a reading of the same terms with as many steps
/// and no deeper declines alike (`held_verdict`, in [`emptiness`]).
pub(super) fn bounds_met() -> u64 {
    BOUNDS_MET.with(Cell::get)
}

impl SeqShape {
    /// Whether every sequence this shape admits is also admitted by `other`.
    ///
    /// The whole rule is [`linear_subtype`]; this is where the elements are
    /// borrowed out of the two shapes. There is no alternation to distribute
    /// over and no shape to first prove linear, because a shape is the linear
    /// form.
    ///
    /// **Two equal shapes never arrive here.** The arm that calls this matches
    /// on the two containers being equal, and the caller above it answers
    /// `Holds` for two equal schemas -- so an equal container and an equal
    /// shape is a pair that returned one frame earlier. A reflexivity check
    /// here would be a second answer to a question already answered, and no
    /// input reaches it.
    fn shape_subtype(&self, other: &SeqShape, cx: &mut Cx<'_>) -> Relation {
        linear_subtype(
            &self.prefix,
            self.tail.as_deref(),
            &other.prefix,
            other.tail.as_deref(),
            cx,
        )
    }
}

/// Whether `schema` is `dict[anything, anything]`, the dict kind's top, which
/// holds every dict (`docs/dev/01-schema-ir.md`, "A bare builtin container is
/// its kind"). The kind walk does not read it whole: it is the reading a dict
/// key asks too, and no dict is a key. A list's and a set's tops are read by
/// the other readings where this one would be asked.
fn the_dicts_top(schema: &Schema) -> bool {
    match schema {
        Schema::KeyedMap { fields, defaults } => {
            matches!((&**fields, &**defaults), ([], [clause]) if clause.is_top())
        }
        _ => false,
    }
}

/// Each member of a union as one union holds them, a union nested in it read as
/// its members: `X ∪ (Y ∪ Z)` is `X ∪ Y ∪ Z`. The constructors flatten what
/// they build, so the nesting arrives in a term built raw, which the fuzz target
/// draws; a reading that picks a union's members out by shape -- a complement,
/// the dict kind's top -- finds one nested a union deep only through this.
///
/// A worklist rather than a recursion: each node is taken once and the term is
/// a finite tree, so the walk ends within its node count. The list allocates
/// only once a nested union is met, so a flat union is read as directly as its
/// own members are.
fn flat_members(members: &[Schema]) -> impl Iterator<Item = &Schema> {
    let mut nested: Vec<&Schema> = Vec::new();
    let mut top = members.iter();
    core::iter::from_fn(move || {
        loop {
            let member = nested.pop().or_else(|| top.next())?;
            match member {
                Schema::Union(inner) => nested.extend(inner.iter()),
                _ => return Some(member),
            }
        }
    })
}

/// A meet narrowed by the complements beside its unions ([`narrowed_meet`]),
/// the same set, below `other`; `Unknown` where nothing narrows.
///
/// Asked under an allowance of its own, [`RESPELLING`] steps of what the query
/// has left, as a respelled pair is: the meet arm asks it last, and a narrowed
/// meet that declines spent the steps the readings after the arm -- the
/// subject's own emptiness among them -- would have decided the pair with.
///
/// A narrowed meet asked inside another's question is asked with that
/// question's steps and its trail, and is given no allowance of its own
/// ([`NARROWING`]). Given one, its steps were counted by nothing above it, and
/// a fixpoint met with the complement of its own lists, `μt. (int | list[t] |
/// dict[str, t]) & ~list[t]`, asked below `μt. int | dict[str, t]`, narrowed
/// again at every goal its narrowed meet asked, each time with fresh steps and
/// a trail that had forgotten the goal: the query never ended. Asked on the
/// question's trail, the goal met again is the assumption it is, and the pair
/// is `subset`.
///
/// A narrowed meet asked outside another's question is asked once a query for
/// a supertype and an allowance, and its answer read back after
/// ([`ASKED_NARROWED`]). The question runs on a trail and a budget of its own
/// and nothing it spends is charged to the query, so its answer rests on no
/// hypothesis of the query's and a read-back leaves the query's steps where
/// asking again would: it changes how often the question runs and no answer.
/// Asked again, the repeats are what cost: a pair of recursive lists, tuples
/// and records asked its narrowed meets 6,009 times over seven questions,
/// 11.6 million steps no query counted, and declined in 527 ms; read back, it
/// declines in 49.
///
/// Out of line, since the meet arm it is asked from is inlined into the goal
/// recursion, and a meet with no union answers at the first look.
#[inline(never)]
fn narrowed_meet_below(members: &Members, other: &Schema, cx: &mut Cx<'_>) -> Relation {
    let query = cx.query();
    if NARROWING.with(Cell::get) == query {
        return narrowed_meet(members, cx).map_or(Relation::Unknown, |narrowed| {
            narrowed.is_subtype_rec(other, cx)
        });
    }
    let given = cx.budget.left().min(respelling());
    let key = (query, Arc::as_ptr(members).cast::<Schema>().addr(), given);
    let asked = ASKED_NARROWED.with_borrow(|asked| {
        asked.get(&key).and_then(|below| {
            below
                .iter()
                .find(|(_, supertype, _)| supertype == other)
                .map(|&(_, _, answer)| answer)
        })
    });
    if let Some(answer) = asked {
        return answer;
    }
    let answer = narrowed_meet(members, cx).map_or(Relation::Unknown, |narrowed| {
        let allowance = Budget::new(given);
        let _open = Narrowing::open(query);
        narrowed.is_subtype_rec(other, &mut cx.within_this_query(&allowance))
    });
    ASKED_NARROWED.with_borrow_mut(|asked| {
        if asked.len() >= READINGS_HELD {
            asked.clear();
        }
        asked
            .entry(key)
            .or_default()
            .push((Arc::clone(members), other.clone(), answer));
    });
    answer
}

/// The narrowed meet's question open on this thread, closed when dropped: the
/// query that was open before is open again.
struct Narrowing(u64);

impl Narrowing {
    fn open(query: u64) -> Self {
        Self(NARROWING.with(|open| open.replace(query)))
    }
}

impl Drop for Narrowing {
    fn drop(&mut self) {
        NARROWING.with(|open| open.set(self.0));
    }
}

impl Schema {
    /// Whether the descriptor proves every value of this schema is one of
    /// `other`, by proving the difference empty.
    ///
    /// `a ≤ b` is `a ∧ ¬b = ∅` [FCB08, §4.1], which is the whole of the test. A
    /// **widening and nothing else**, asked where
    /// [`denotes_no_value`](Self::denotes_no_value) is asked and for the same
    /// reasons.
    ///
    /// An empty left side is below every set, and it says so without lowering
    /// the right one at all -- otherwise a schema the descriptor proves empty
    /// would be below nothing whose descriptor it could not build, which is a
    /// pair of answers that contradict each other.
    pub(crate) fn descriptor_contained_in(
        &self,
        other: &Schema,
        pool: &dyn Constants,
        defs: &[Schema],
    ) -> Relation {
        #[cfg(test)]
        goal_tests::goals::read_by_the_sets();
        // A term of length bounds over one base is the lengths it holds, read
        // without a build ([`lengths`]).
        if let Some(answer) = self.related_by_lengths(other) {
            return answer;
        }
        // The two sides are unfolded in opposite directions, which is what makes
        // a difference over a recursive schema sound *in one direction*: the
        // left grows and the right shrinks, so a difference proved empty here
        // was empty before.
        //
        // **The other direction does not carry.** `self⁺ ∧ ¬other⁻` contains
        // the real difference and is not contained by it, so a value found in
        // the widened reading need be no value of `self ∧ ¬other` at all, and
        // the inhabited answer refutes nothing. Where a reference was cut, as a
        // finite approximation cuts a recursive type [AC93, §3.4], an empty
        // difference proves the inclusion and an inhabited one is a decline,
        // but at a kind no cut reaches: a cut beneath a constructor changes that
        // constructor's kind of values alone, the difference being read a kind
        // at a time [FCB08, (4)], so a value it holds at another kind is a
        // value of the real difference (`parts_a_cut_reaches`). An inhabited
        // reading refuting nothing there hands the pair to the reading the
        // other way round ([`Self::refuted_narrowed`]).
        //
        // A side this reading cannot lower, and a difference it cannot build,
        // are declines rather than refutations: nothing about the inclusion is
        // known from a set that was never constructed.
        //
        // **The difference is a build of its own**, under the allowance the two
        // lowerings each ran under. It is the dearest of the three: the sides
        // are built from a schema the node bound has already measured, and the
        // difference multiplies a complement against a meet, which is where the
        // lattices spend. Nothing else bounds it -- each lattice's width bound
        // refuses a wide *answer*, not a long search for a narrow one -- so
        // without this the cheapest thing a caller can ask for is bounded and
        // the dearest is not.
        //
        // **A literal key read as a label is the same kind of widening**, and
        // what it costs is the same half: an empty difference over a label on
        // the narrowed side may prove nothing ([`an_empty_reading_stands`]),
        // and is a decline where it does not. Asked only of an empty answer,
        // which is the one it can overturn.
        //
        // **A conjunct the subject's lowering refuses is left out**, as
        // [`read_widened`] reads it, which is a widening of the same kind:
        // the subject reads as a superset, and the label check reads the
        // members kept. A thinned subject that can only be inhabited answers
        // nothing against a supertype the lowering refuses, so where the
        // reading would be one, the supertype is lowered first, and once.
        //
        // `¬other` is what the difference meets, and a complement's is its
        // inner set, read on the widened side: `self ⊆ ¬inner` asks `self ∧
        // inner = ∅`, a double complement being the set [FCB08, Def. 4.1].
        // Complementing the lowered `¬inner` would rebuild that set
        // from a double negation, and where the negation expands -- a record
        // whose field spans two kinds expands into more atoms than it began
        // with -- the meet with the rebuilt form spends past the allowance a
        // meet with the inner set fits in: `{"t": int} ⊆ ¬({"t": str} ∪
        // {"t": bytes})` declines that way, while three field kinds, whose
        // complement stays negated, decide.
        //
        // **Each reading is decided where it is built**, and one that answers
        // nothing hands the question to the next ([`read_widened`]): an empty
        // difference the label check refuses over a whole subject is asked
        // again of the reading that leaves a scalar past the depth out. The
        // supertype is lowered once for all of them.
        let (read, side) = other.read_as_a_supertype();
        let theirs = OnceCell::new();
        let theirs = || {
            theirs
                .get_or_init(|| lower_unfolded(read, defs, side, pool))
                .as_ref()
        };
        let negate = side == Polarity::Narrow;
        // The parts of the universe a cut reaches, on either side: past them
        // the difference is the real one, read a kind at a time. A side with no
        // reference is cut nowhere, so it reaches none.
        let cut = OnceCell::new();
        let cut = || {
            *cut.get_or_init(|| parts_a_cut_reaches(self, defs) | parts_a_cut_reaches(read, defs))
        };
        let mut inhabited = false;
        read_widened(
            self,
            defs,
            pool,
            |kept| !the_rules_find_a_value(kept) || theirs().is_some(),
            |mine, kept| {
                // The schemas the subject's side reads as widened are itself,
                // or the conjuncts a thinned reading kept, collected only where
                // read.
                if mine.emptiness() == Verdict::Empty {
                    let stands = match kept {
                        None => an_empty_reading_stands(&[self], &[], defs, pool),
                        Some(kept) => an_empty_reading_stands(
                            &kept.iter().collect::<Vec<_>>(),
                            &[],
                            defs,
                            pool,
                        ),
                    };
                    return stands.then_some(Relation::Holds);
                }
                let theirs = theirs()?;
                let (emptiness, refuted) = budget::under(WORK, || {
                    let difference = if negate {
                        mine.minus(theirs)
                    } else {
                        mine.intersect(theirs)
                    };
                    difference.map(|difference| {
                        let emptiness = difference.emptiness();
                        // With no definitions nothing was unfolded: a
                        // reference none resolves is a set no lowering reads,
                        // so a difference built here holds none, and the one
                        // comparison keeps the cut's walk off every pair with
                        // no fixpoint.
                        let refuted = kept.is_none()
                            && emptiness == Verdict::Inhabited
                            && (defs.is_empty() || difference.holds_a_value_outside(cut()));
                        (emptiness, refuted)
                    })
                })?;
                inhabited |= emptiness == Verdict::Inhabited;
                let stands = || match (kept, negate) {
                    (None, true) => an_empty_reading_stands(&[self], &[read], defs, pool),
                    (None, false) => an_empty_reading_stands(&[self, read], &[], defs, pool),
                    (Some(kept), true) => an_empty_reading_stands(
                        &kept.iter().collect::<Vec<_>>(),
                        &[read],
                        defs,
                        pool,
                    ),
                    (Some(kept), false) => {
                        let widened: Vec<&Schema> = kept.iter().chain([read]).collect();
                        an_empty_reading_stands(&widened, &[], defs, pool)
                    }
                };
                let relation = if emptiness == Verdict::Empty && !stands() {
                    Relation::Unknown
                } else if refuted {
                    Relation::Fails
                } else {
                    Relation::proven(emptiness == Verdict::Empty)
                };
                (relation != Relation::Unknown).then_some(relation)
            },
        )
        .unwrap_or_else(|| {
            if inhabited {
                self.refuted_narrowed(read, side, defs, pool)
            } else {
                Relation::Unknown
            }
        })
    }

    /// `Fails` where a value of the subject outside the supertype is found in
    /// the reading that holds no other: the subject narrowed, the supertype
    /// widened.
    ///
    /// The finite approximation runs in both directions. Cut to the bottom where
    /// a schema is used positively and to the top under a complement, a
    /// recursive type lies below the type it cuts [AC93, §3.4], so `self⁻ ⊆
    /// self`, and the dual cut gives `other ⊆ other⁺`, so `¬other⁺ ⊆ ¬other`.
    /// Their meet lies inside the real difference, and a value of it refutes at
    /// any kind, a kind a cut reaches included. An empty reading proves nothing,
    /// which is the half the widened reading has. `read` and `side` are what
    /// [`Self::read_as_a_supertype`] answered, and `side` flips with the cut:
    /// `other` narrowed is read as `¬read⁺`, and a complement's inner set
    /// widened as `read⁻`.
    ///
    /// **Asked only of a pair the widened reading held a value of.** A pair with
    /// no reference is refuted or proved before it gets here, so the relations
    /// over no fixpoint pay for one comparison. Asked of every undecided pair
    /// holding a reference, it read 5% more instructions over drawn JSON-Schema
    /// translations, for 13 more answers in 22,500 draws. A value at a kind no
    /// cut reaches is in this reading too, the two sides reading alike there,
    /// and the difference already built names it: asking this reading for those
    /// as well read 1% to 3% more over recursive schemas, for the same answers.
    ///
    /// **One unfolding, as the widened reading.** A required field cut to the
    /// bottom empties its record, so a deeper unfolding could find a value this
    /// one does not, but the wider term spends [`BUDGET`](crate::descr::lower)
    /// first: two unfoldings left 17% more of a corpus of recursive lists and
    /// maps undecided.
    #[inline(never)]
    pub(crate) fn refuted_narrowed(
        &self,
        read: &Schema,
        side: Polarity,
        defs: &[Schema],
        pool: &dyn Constants,
    ) -> Relation {
        #[cfg(test)]
        goal_tests::goals::read_by_the_sets();
        let refuted = || {
            let mine = lower_unfolded(self, defs, Polarity::Narrow, pool)?;
            let theirs = lower_unfolded(read, defs, side.flipped(), pool)?;
            budget::under(WORK, || {
                let difference = if side == Polarity::Narrow {
                    mine.minus(&theirs)
                } else {
                    mine.intersect(&theirs)
                };
                difference.map(|difference| difference.emptiness() == Verdict::Inhabited)
            })
        };
        if refuted() == Some(true) {
            Relation::Fails
        } else {
            Relation::Unknown
        }
    }

    /// What the set representation lowers of this schema as a supertype, and on
    /// which side: a complement's inner set on the widened side, since `a ⊆
    /// ¬inner` asks `a ∧ inner = ∅`, and the schema itself on the narrowed side.
    pub(crate) fn read_as_a_supertype(&self) -> (&Schema, Polarity) {
        match self {
            Schema::Complement(inner) => (inner, Polarity::Widen),
            _ => (self, Polarity::Narrow),
        }
    }

    /// The decision steps [`is_subtype_of`](Self::is_subtype_of) spends, by the
    /// same argument as [`empty_steps`](Self::empty_steps).
    #[cfg(test)]
    pub(crate) fn subtype_steps(&self, other: &Schema) -> u32 {
        self.subtype_steps_under(other, &NoLeafRelations)
    }

    /// The same count where the rules can look a constant or a class up, for a
    /// rule whose work depends on what the oracle answers.
    #[cfg(test)]
    pub(crate) fn subtype_steps_under(&self, other: &Schema, oracle: &dyn LeafRelations) -> u32 {
        let budget = Budget::new(DECISION_BUDGET);
        self.is_subtype_rec(other, &mut Cx::new(oracle, &[], &budget));
        DECISION_BUDGET - budget.left()
    }

    /// Whether every value of `self` is also a value of `other` — set inclusion,
    /// the semantic-subtyping relation [Cas24, Def. 4]. Complete on the scalar
    /// fragment via `self ∧ ¬other = ∅`, and decided structurally past it by
    /// recursion on matching constructors (the lattice rules, set/frozenset
    /// element inclusion, and sequence inclusion on the prefix-and-tail form).
    /// Every rule is **sound** — it never reports a subtype it cannot justify
    /// — and conservative where it cannot decide: there it returns `false`
    /// rather than guess. A `false` is then asked again of the two *sets*,
    /// under a bound on what building them may cost, which decides what no
    /// rule about shapes reaches: a container meet, a complement nested inside
    /// another, one regular language inside another, a kind against its own
    /// literals, one step dividing another.
    ///
    /// A complement *directly* under a complement is not on that list and cannot
    /// be: [`Schema::complement`] folds `¬¬X` to `X` whichever schema it is
    /// given, so the shape reaches a comparison only from a term built without
    /// the constructors -- which the fuzz target does, and which is what the
    /// bound on a universal subject is for.
    ///
    /// The decision is bounded twice over: an adversarial schema that would take
    /// more than a fixed number of steps stops, and a descriptor too costly to
    /// build refuses. A `false` can therefore mean "not proven a subtype within
    /// the work bound". A real schema decides far inside both.
    #[must_use]
    #[inline]
    pub fn is_subtype_of(&self, other: &Schema) -> bool {
        self.is_subtype_of_under(other, &NoLeafRelations, &[])
    }

    /// [`is_subtype_of`](Self::is_subtype_of) with a [`LeafRelations`] oracle deciding
    /// the leaf relations the structural rules cannot (an `Instance` class or a
    /// `Literal` value), and the `defs` that resolve recursive references so
    /// subtyping is decided between recursive schemas too. The oracle's `None`
    /// and an unresolved reference both keep the conservative `false`.
    #[must_use]
    #[inline]
    pub fn is_subtype_of_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> bool {
        self.subtype_relation_under(other, oracle, defs).holds()
    }

    /// The inclusion in three values: proven, refuted, or neither.
    ///
    /// [`is_subtype_of_under`](Self::is_subtype_of_under) is this reduced to the
    /// two a boundary reports, and the reduction loses the distinction a caller
    /// most often wants: whether a `false` is a value outside the supertype or
    /// a question no rule here answers. Both deciders are asked, the second
    /// only where the first declines, and a refutation from either stands on a
    /// value of the subject -- which the rules' answer is read against, and
    /// which the set reading proves by finding the difference inhabited.
    #[must_use]
    #[expect(
        clippy::inline_always,
        reason = "measured: with its second asking, left to `#[inline]` the decision \
                  workload read 3.2% more instructions, inlined 0.96%"
    )]
    #[inline(always)]
    pub fn subtype_relation_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> Relation {
        let met = bounds_met();
        let budget = Budget::new(DECISION_BUDGET);
        let mut cx = Cx::new(oracle, defs, &budget);
        let first = self
            .is_subtype_rec(other, &mut cx)
            .or_else(|| self.decided_by_sets(other, &mut cx, true));
        if first == Relation::Unknown && asks_again(met, &budget) {
            return asked_again(|budget| {
                let mut cx = Cx::new(oracle, defs, budget);
                self.is_subtype_rec(other, &mut cx)
                    .or_else(|| self.decided_by_sets(other, &mut cx, true))
            });
        }
        first
    }

    /// The set representation of the pair, then both deciders asked of its
    /// parts, in the query the rules declined: what follows their decline.
    /// `by_branch` is whether a branch of a union supertype may prove the pair
    /// alone. Last, a string subject's texts put to the walk, which a meet or a
    /// refinement can carry and the shape test reads before the call.
    ///
    /// Out of line, and handed the query rather than what it holds, so the
    /// three public entries above it stay thin and are inlined into a caller's
    /// loop, which `#[inline]` on each asks for: left to the inliner they are
    /// not, and the call costs the decision workload 2% of its instructions,
    /// twice that through a wrapper. The relation's entry asks for it always:
    /// with its second asking beside the first, `#[inline]` alone left it out
    /// of the loop, at 3%.
    #[inline(never)]
    fn decided_by_sets(&self, other: &Schema, cx: &mut Cx<'_>, by_branch: bool) -> Relation {
        let below_nothing = self.below_a_supertype_with_no_value(other, cx);
        if below_nothing != Relation::Unknown {
            return below_nothing;
        }
        self.descriptor_contained_in(other, cx.oracle, cx.defs)
            .or_else(|| self.decided_by_parts(other, cx, by_branch))
            .or_else(|| {
                if by_branch {
                    self.decided_as_respelled(other, cx)
                } else {
                    Relation::Unknown
                }
            })
            .or_else(|| {
                if matches!(self, Schema::Refine { .. } | Schema::Intersection(_)) {
                    self.decided_by_its_texts(other, cx.oracle)
                } else {
                    Relation::Unknown
                }
            })
            .or_else(|| {
                if by_branch {
                    self.decided_by_its_leaves(other, cx)
                } else {
                    Relation::Unknown
                }
            })
            .or_else(|| self.below_by_emptiness(other, cx))
    }

    /// The rules asked once more, a goal they decline read by the sets where no
    /// reference stands in the way.
    ///
    /// The whole pair's set reading stops at what it cannot lower or at a cut,
    /// and it stops for all of the pair: `tuple[bool, R]` below `tuple[Annotated[int,
    /// Lt(8)], R]` holds a reference at one position, and the leaf at the other is
    /// a question the sets answer on their own. Both deciders are sound, so a
    /// part's answer composes through the rules as any other does. A reference on
    /// neither side is cut nowhere, so the reading is the set itself.
    ///
    /// Asked last of a pair everything else declined. A goal the rules decline
    /// is the common case on the way to an answer another rule gives, and asked
    /// of every such goal a lowering cost the relation matrix ninety-six times its
    /// instructions. The rules are asked here and never reach this again, so a
    /// query asks it a fixed number of times, under [`RESPELLING`]'s allowance.
    #[inline(never)]
    pub(super) fn decided_by_its_leaves(&self, other: &Schema, cx: &Cx<'_>) -> Relation {
        if !(self.holds_parts() || other.holds_parts()) {
            return Relation::Unknown;
        }
        let allowance = Budget::new(cx.budget.left().min(respelling()));
        let mut read = cx.within_this_query(&allowance);
        read.reads_leaves = true;
        self.is_subtype_rec(other, &mut read)
    }

    /// Whether a goal of this schema's can stand beneath a constructor or a
    /// reference, under the connectives. One that cannot is a leaf itself, whose
    /// whole pair the set reading has read.
    fn holds_parts(&self) -> bool {
        match self {
            Schema::Seq { .. }
            | Schema::Coll { .. }
            | Schema::KeyedMap { .. }
            | Schema::AttrRecord { .. }
            | Schema::Ref(_) => true,
            Schema::Union(members) | Schema::Intersection(members) => {
                members.iter().any(Schema::holds_parts)
            }
            Schema::Complement(inner) => inner.holds_parts(),
            _ => false,
        }
    }

    /// A goal the rules declined: what the sets answer of one with no reference
    /// on either side where the query reads its leaves, charged [`LEAF_READING`]
    /// steps, since no step counts a lowering. A difference the sets read inhabited
    /// holds the value a refutation stands on, so no witness is asked of it. Out of
    /// line, as a declined goal is the path this is on: the rules' answers read one
    /// test.
    #[inline(never)]
    fn a_goal_declined(&self, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        if !cx.reads_leaves || self.has_reference() || other.has_reference() {
            return Relation::Unknown;
        }
        if !spend_at_once(cx.budget, LEAF_READING) {
            a_bound_is_met();
            return Relation::Unknown;
        }
        self.descriptor_contained_in(other, cx.oracle, cx.defs)
    }

    /// A supertype proved empty is the bottom, so the pair is the subject's own
    /// emptiness: the dual of [BBC+26, (uninhabited)]. A reference is what no reading
    /// of the pair sees through: `μt. attr{y: t}` holds no finite value, `is_empty`
    /// says so at once, and the difference the set reading builds of `str` against it
    /// is read as holding values both ways, to a depth that doubles. A schema with no
    /// reference is read by the sets, with nothing cut, so it is not asked. `Unknown`
    /// where the subject's own emptiness is undecided, which leaves the pair to the
    /// readings after this one: answered here, they lost a hundred answers in 72,000
    /// drawn pairs.
    #[inline(never)]
    fn below_a_supertype_with_no_value(&self, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        if !matches!(
            other,
            Schema::Ref(_) | Schema::AttrRecord { .. } | Schema::KeyedMap { .. }
        ) || other.holds_a_value_shallowly()
            || !other.has_reference()
            || !other.is_empty_with(cx.oracle, cx.defs)
        {
            return Relation::Unknown;
        }
        self.is_subtype_rec(&Schema::Nothing, cx)
            .or_else(|| self.descriptor_contained_in(&Schema::Nothing, cx.oracle, cx.defs))
    }

    /// The two lattice bounds read once more, each under an emptiness query of its
    /// own, where every decider declined with the query's steps spent: an empty
    /// subject is below every set [BBC+26, (uninhabited)], and a subject below a
    /// complement is one sharing no value with its inner set [FCB08, §4.1]. The
    /// rules read both as they descend, under the query's steps, and a pair they
    /// spent every step on reaches the bounds with none left; `is_empty` asked of
    /// the same terms with steps of its own proves them, so the relation and the
    /// emptiness reading do not answer one question two ways. A query with steps
    /// left has asked the bounds already, and the readings asked again there cost a
    /// fifth of the relation matrix for nothing.
    #[inline(never)]
    fn below_by_emptiness(&self, other: &Schema, cx: &Cx<'_>) -> Relation {
        if cx.budget.left() > 0 {
            return Relation::Unknown;
        }
        if self.is_empty_with(cx.oracle, cx.defs) {
            return Relation::Holds;
        }
        match other {
            Schema::Complement(inner) => Relation::proven(
                Schema::meet_within([self.clone(), (**inner).clone()], cx.defs)
                    .is_empty_with(cx.oracle, cx.defs),
            ),
            _ => Relation::Unknown,
        }
    }

    /// The pair asked in the spellings the Boolean algebra gives it, where
    /// every decider declined it as written.
    ///
    /// `a ⊆ b` is `a ∧ ¬b = ∅` [FCB08, §4.1], so a difference holding every
    /// value of some kind refutes the pair: a kind the subject holds whole and
    /// the supertype holds none of. The rules read that for the universe and a
    /// complement alone (`spills_a_kind`), since a walk on every pair costs the
    /// pairs that never need it; here it is read for any subject, once the
    /// rules have declined. Against a complement the difference of `a ⊆ ¬c` is
    /// the meet `a ∧ c`, asked by the rules as a subject of its own, and the
    /// pair is `c ⊆ ¬a`, asked from its other side. Where both decline, the
    /// meet is read empty as `is_empty` reads a set (`empty_under`): a union a
    /// member at a time, each by the rules and by the set representation, so
    /// `anything ⊆ ¬s` is proved where `s` is read empty within these steps,
    /// where the rules alone stop at a member only the set representation
    /// empties.
    /// Against a union holding
    /// complements the difference is the same meet with the union's other
    /// members subtracted: `a ⊆ b ∨ ¬c` is `(a ∧ c) ∧ ¬b = ∅`, which is `a ∧ c
    /// ⊆ b`, exactly -- `anything ⊆ ¬dict ∨ x` asks whether `dict ⊆ x` -- where
    /// `c` denotes a set: the identity is excluded middle on `c`, which a
    /// predicate answering two calls apart does not obey, so `anything ⊆ p ∨ ¬p`
    /// is not `p ⊆ p` (`denotes_a_set_within`). Both are
    /// goals over the query's signed subterms, so the goals stay in the
    /// closure the budget is argued over. The difference against any other
    /// supertype is a meet the constructors build on every pair that declines,
    /// which the relation matrix pays a seventh of its instructions for, and is
    /// not asked.
    ///
    /// A meet against a meet is asked as the two together: `a ⊆ b` with
    /// `a ∧ b = ∅` holds only where `a` is empty, so a subject the rules find a
    /// value of is refuted where the meet of the two is proved empty. No rule
    /// reads the two together -- a meet subject is placed by its members and a
    /// meet supertype is asked a member at a time -- and two guarded records a
    /// translation from JSON Schema writes, told apart by a tag, are that pair.
    ///
    /// The rules asked again decline as dearly as they did, so they share
    /// [`RESPELLING`] steps of what the query has left rather than all of it.
    pub(super) fn decided_as_respelled(&self, other: &Schema, cx: &Cx<'_>) -> Relation {
        if readings::spills_a_kind_into(self, other, cx) {
            return Relation::Fails;
        }
        // A complement of a union is the meet of its branches' complements
        // [FCB08, Def. 4.1], whatever the supertype, so the pair is asked again
        // that way under an allowance of its own: `¬(¬X ∪ Y)` below `dict` is
        // `X ∩ ¬Y` below it, which a member of `X` places. The meet is of the
        // subject's own branches and their signs, so the goals a query asks
        // stay drawn from its terms.
        if let Schema::Complement(inner) = self
            && let Schema::Union(branches) = &**inner
        {
            let pushed = Schema::meet_within(
                branches.iter().map(|branch| branch.clone().complement()),
                cx.defs,
            );
            let allowance = Budget::new(cx.budget.left().min(respelling()));
            let answer = pushed.is_subtype_rec(other, &mut cx.within_this_query(&allowance));
            if answer != Relation::Unknown {
                return answer;
            }
        }
        let meets = matches!(
            (self, other),
            (Schema::Intersection(_), Schema::Intersection(_))
        );
        // The dict kind's top is the complement of its complement, `K = ¬¬K`,
        // and is asked as a complement member is: `a ⊆ b ∨ K` is `a ∧ ¬K ⊆ b`
        // [FCB08, §4.1]. The kinds the kind walk reads whole are settled by the
        // cover reading already, and this one it does not read.
        let negates = match other {
            Schema::Complement(_) => true,
            Schema::Union(members) => flat_members(members)
                .any(|member| matches!(member, Schema::Complement(_)) || the_dicts_top(member)),
            _ => false,
        };
        // A subject below a table of constants is one whose meet with the
        // table's complement holds no value, `a ⊆ T` being `a ∧ ¬T = ∅`
        // [FCB08, §4.1], read as `is_empty` reads that meet, so the two answer
        // alike where no oracle kinds the constants: `None ∧ (int ∪ 1)` meets
        // `¬1` in nothing and is below `1`.
        if finite_set(other).is_some() {
            let difference =
                Schema::meet_within([self.clone(), other.clone().complement()], cx.defs);
            let allowance = Budget::new(cx.budget.left().min(respelling()));
            let part = &mut cx.within_this_query(&allowance);
            if difference.empty_under(part)
                || (matches!(difference, Schema::Intersection(_))
                    && difference.below_the_set_it_complements(part).holds())
            {
                return Relation::Holds;
            }
        }
        if matches!(other, Schema::Nothing) {
            return match self {
                Schema::Intersection(_) => self.below_the_set_it_complements(cx),
                Schema::Complement(_) => self.complement_of_the_universe(cx),
                _ => Relation::Unknown,
            };
        }
        // The relation's two lattice bounds read those readings too: a meet or
        // a complement they read empty is below every supertype
        // [BBC+26, (uninhabited)], and a subject below a complement is asked
        // through its meet with the inner set (below), so `is_empty` and the
        // relation answer alike.
        if (matches!(self, Schema::Intersection(_))
            && self.below_the_set_it_complements(cx).holds())
            || (matches!(self, Schema::Complement(_))
                && self.complement_of_the_universe(cx).holds())
        {
            return Relation::Holds;
        }
        if !meets && !negates {
            return Relation::Unknown;
        }
        let allowance = Budget::new(cx.budget.left().min(respelling()));
        let cx = &mut cx.within_this_query(&allowance);
        if let Schema::Union(members) = other {
            let (complements, rest): (Vec<&Schema>, Vec<&Schema>) = flat_members(members)
                .partition(|member| {
                    matches!(member, Schema::Complement(inner)
                        if denotes_a_set_within(inner, cx.oracle, cx.defs))
                        || the_dicts_top(member)
                });
            if complements.is_empty() {
                return Relation::Unknown;
            }
            let inners = complements.into_iter().map(|member| match member {
                Schema::Complement(inner) => (**inner).clone(),
                top => top.clone().complement(),
            });
            let narrowed =
                Schema::meet_within(core::iter::once(self.clone()).chain(inners), cx.defs);
            let rest = Schema::union_within(rest.into_iter().cloned(), cx.defs);
            // Then as the difference itself [FCB08, §4.1], read as `is_empty` reads
            // it, which empties a member of the subject where the rules decline the
            // meet below the rest. That reading folds no predicate's complement, so
            // a predicate in the union is no excluded middle here either.
            return narrowed.is_subtype_rec(&rest, cx).or_else(|| {
                let difference =
                    Schema::meet_within([self.clone(), other.clone().complement()], cx.defs);
                Relation::proven(difference.empty_under(cx))
            });
        }
        let Schema::Complement(inner) = other else {
            let disjoint = Schema::meet_within([self.clone(), other.clone()], cx.defs)
                .is_empty_rec(cx)
                && self.verdict_rec(cx) == Verdict::Inhabited;
            return if disjoint {
                Relation::Fails
            } else {
                Relation::Unknown
            };
        };
        let meet = Schema::meet_within([self.clone(), (**inner).clone()], cx.defs);
        meet.is_subtype_rec(&Schema::Nothing, cx)
            .or_else(|| inner.is_subtype_rec(&self.clone().complement(), cx))
            .or_else(|| Relation::proven(meet.empty_under(cx)))
            .or_else(|| meet.below_the_set_it_complements(cx))
    }

    /// The pair asked of its parts, where both deciders declined it whole.
    ///
    /// A union subject is below a supertype where each member is, and outside
    /// it where one member is [BBC+26, (anyOf-l)], so each member is asked as
    /// a pair of its own, by both deciders and by its own parts. The rules
    /// distribute the union already; what this adds is the set representation
    /// asked of one member, where asked of the whole pair it is refused past
    /// its node bound: a union of seven tagged records against seven decides a
    /// record at a time.
    ///
    /// A union supertype is narrowed to the branches the subject shares a
    /// value with, as `below_a_union` narrows it, and the narrowed pair asked
    /// whole: a branch sharing no value with the subject decides nothing about
    /// the inclusion. That takes a reference out of a pair the set
    /// representation could otherwise only prove: `str` against a fixpoint of
    /// lists beside a pattern is the pattern's question alone, refuted at a
    /// string the pattern refuses. Where nothing narrows, a branch the set
    /// representation places the subject in proves the pair
    /// [BBC+26, (anyOf-r)], and a branch the rules refuted the subject below at
    /// the root is not lowered: both deciders are sound, so a containment the
    /// rules refuted is one the set representation never proves
    /// ([`Cx::refuted_at_the_root`]) -- for a subject that is not a member of a
    /// union being asked, since a union against a union would lower every
    /// member against every branch: two hundred patterns against two hundred
    /// spend three seconds that way, and decide nothing. A branch of one
    /// constant is not asked: it holds the subject only where the subject is that one
    /// value, and a subject of one value is inside every branch it shares the
    /// value with, so narrowing leaves that constant alone or beside branches
    /// that hold the subject too. Asked a constant at a time, a table lowers
    /// the subject once a constant: a millisecond each for a pattern of a
    /// thousand words, and a table of a thousand names decides nothing for it.
    ///
    /// A meet supertype is above a subject where each member is, and refuted
    /// where one member is, a meet being the intersection of its members'
    /// sets [FCB08, Def. 4.1], in Kleene's strong conjunction: a member refuted
    /// by either decider refutes the meet beside a member neither reads. The
    /// rules ask each member already, by their own readings alone, and what
    /// they answered at the root is read back ([`Cx::held_below`]); what this
    /// adds is the set representation asked of one member, where a member that
    /// runs a callback -- a predicate, a class whose `isinstance` is code --
    /// refuses the whole meet a lowering: `int` is outside `Ge(0)` at `-1`, and
    /// so outside `Ge(0)` met with a predicate. A meet of sets was read whole,
    /// and is not read again a member at a time, which would lower each member
    /// for an answer the whole reading did not find.
    ///
    /// A record against a record is asked a shared field at a time by the set
    /// representation, which the pair is past the node bound of where it is
    /// wide: a field refuted refutes the record on a subject that holds a
    /// value ([`a_shared_field_refuted_by_the_sets`]).
    ///
    /// The parts are asked in the query, so the rules' work stays bounded by
    /// its budget in total, and the lowerings are one per member, per
    /// narrowing, per shared field and per branch of the one subject that is
    /// not a constant, each under its own bounds. They are asked only while the
    /// query has a step left: the lowerings are counted by no step, and a pair
    /// the rules spent the budget on is a pair wide enough that reading it a
    /// part at a time costs its width in lowerings.
    fn decided_by_parts(&self, other: &Schema, cx: &mut Cx<'_>, by_branch: bool) -> Relation {
        // A query whose steps the pair spent asks no part. A part is a reading
        // of the set representation, which no step counts, and a pair the
        // rules spent every step on has as many parts as members and
        // branches: a union of four thousand records against another, or a
        // tuple against its two thousand corners, read a part at a time cost
        // seventy times what the decline did and answered nothing more.
        if cx.budget.left() == 0 {
            a_bound_is_met();
            return Relation::Unknown;
        }
        if let Schema::Union(members) = self {
            // The supertype is every member's, so it is lowered once: where it
            // refuses, every member's pair refuses on it, after a build of the
            // member and of the supertype up to its bound -- a millisecond a
            // member for sixty-four patterns. An empty member is below it all
            // the same, which is the one answer such a pair gives.
            let (read, side) = other.read_as_a_supertype();
            let lowers = lower_unfolded(read, cx.defs, side, cx.oracle).is_some();
            // What the rules answered of each member at the root is read back
            // rather than asked again ([`Schema::each_member_below_at_the_root`]).
            let held = cx.held_below(other, members);
            return Relation::all(members.iter().enumerate().map(|(at, member)| {
                match held {
                    Some(held) => Relation::proven(held >> at & 1 == 1),
                    None => member.is_subtype_rec(other, cx),
                }
                .or_else(|| {
                    if lowers {
                        member.descriptor_contained_in(other, cx.oracle, cx.defs)
                    } else {
                        Relation::proven(cx.denotes_no_value(member))
                    }
                })
                .or_else(|| member.decided_by_parts(other, cx, false))
            }));
        }
        if let (Schema::KeyedMap { fields: fa, .. }, Schema::KeyedMap { fields: fb, .. }) =
            (self, other)
        {
            return if a_shared_field_refuted_by_the_sets(fa, fb, cx) {
                Relation::of_mismatch(self.verdict_rec(cx))
            } else {
                Relation::Unknown
            };
        }
        if let Schema::Intersection(members) = other {
            if members
                .iter()
                .all(|member| denotes_a_set_within(member, cx.oracle, cx.defs))
            {
                return Relation::Unknown;
            }
            let held = cx.held_below(self, members);
            return Relation::all(members.iter().enumerate().map(|(at, member)| {
                match held {
                    Some(held) => Relation::proven(held >> at & 1 == 1),
                    None => self.is_subtype_rec(member, cx),
                }
                .or_else(|| self.descriptor_contained_in(member, cx.oracle, cx.defs))
            }));
        }
        let Schema::Union(branches) = other else {
            return Relation::Unknown;
        };
        let met: Vec<Schema> = branches
            .iter()
            .filter(|branch| !self.shares_no_value_with(branch, cx))
            .cloned()
            .collect();
        if met.len() == branches.len() {
            // What the rules refuted of each branch at the root is read back
            // ([`Schema::below_a_union`]): a containment the rules refuted is
            // one the set representation, sound as they are, never proves.
            let refuted = cx.refuted_at_the_root(self, branches);
            return Relation::proven(
                by_branch
                    && branches
                        .iter()
                        .enumerate()
                        .filter(|&(at, branch)| {
                            !matches!(branch, Schema::Literal(_))
                                && (at >= MEMBERS_HELD || refuted >> at & 1 == 0)
                        })
                        .any(|(_, branch)| {
                            self.descriptor_contained_in(branch, cx.oracle, cx.defs)
                                .holds()
                        }),
            );
        }
        let narrowed = Schema::union(met);
        self.is_subtype_rec(&narrowed, cx)
            .or_else(|| self.decided_by_sets(&narrowed, cx, by_branch))
    }

    /// The three-valued subtyping answer under an oracle, the definitions, and a
    /// work budget the caller owns.
    ///
    /// The structural procedure alone: the descriptor's own reading is a second
    /// decider the public relation asks after this one. Held here so a test can
    /// count what the rules decline rather than reading `false` for both halves
    /// of the conservative contract.
    #[cfg(test)]
    pub(crate) fn subtype_relation(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        budget: &Budget,
    ) -> Relation {
        self.is_subtype_rec(other, &mut Cx::new(oracle, defs, budget))
    }

    /// Whether this schema has a value by its own shape, read without descent.
    ///
    /// The witness guard runs on every refutation and at every level of one, so
    /// the shapes whose inhabitance is their own form answer here rather than
    /// through the general fold: a scalar atom is a value, a container that
    /// admits an empty one has that value whatever its elements say -- a map
    /// with no required field holds the empty dict [Cas23, fn. 11] -- and a
    /// union has one where a member does. `false` is "not seen from here" and
    /// costs the descent, never an answer -- and `laws.rs` holds this to the
    /// fold in the direction it claims.
    pub(crate) fn holds_a_value_shallowly(&self) -> bool {
        match self {
            Schema::Anything(_)
            | Schema::Bool
            | Schema::Int
            | Schema::Float
            | Schema::Str
            | Schema::Bytes
            | Schema::NoneType
            | Schema::Coll { .. } => true,
            Schema::Seq { shape, .. } => shape.prefix.is_empty(),
            // A map atom with no label required is empty only where its
            // default meets a wanted constraint in nothing, and an atom built
            // from a schema wants none [Cas23, (11)].
            Schema::KeyedMap { fields, .. } => fields.iter().all(|f| !f.required),
            Schema::Union(members) => members.iter().any(Schema::holds_a_value_shallowly),
            _ => false,
        }
    }

    /// One rule's answer about this subject, with a refutation believed only
    /// where the subject has a value to stand on.
    ///
    /// Every refutation the rules reach is a mismatch of shapes: a value of the
    /// subject the other schema rejects. A subject with no value is below every
    /// schema [BBC+26, (uninhabited)] and has no such value, so the reading is
    /// what turns a mismatch into a claim -- and it is taken about the subject
    /// of *this* comparison, at every level, because a composition carries a
    /// part's refutation up and the part is a subject of its own. A list of an
    /// empty element is the empty list and is below a list of anything; the
    /// mismatch its element reports is about no value.
    ///
    /// A union's refutation is read already. The rules refute a union only
    /// through a member, whose refutation was read against the member at its
    /// own level, through its exact regions, or through a constant it lists,
    /// and each of those is a value the union holds. Reading its emptiness
    /// again stops at the first member that reading cannot prove empty, which
    /// need not be the member refuted: a union of a pattern and a fixpoint
    /// holding `None` is refuted below the empty set by `None`, and the pattern
    /// read first is unknown to it.
    ///
    /// It costs the *proving* half of the decision surface 1.4%, measured by
    /// disabling it: sixteen instructions per query that never refutes, which
    /// is the price of reading a refutation honestly on the queries that do. An
    /// `#[inline]` recovers none of it -- the compiler is already free to,
    /// within the crate -- so the reading stands as the cost.
    fn witnessed(&self, answer: Relation, cx: &mut Cx<'_>) -> Relation {
        if answer == Relation::Fails {
            if self.holds_a_value_shallowly()
                || matches!(self, Schema::Union(_))
                || self.holds_a_kind_whole(cx)
            {
                return answer;
            }
            return Relation::of_mismatch(self.verdict_rec(cx));
        }
        answer
    }

    /// Whether this schema is a complement holding every value of a kind,
    /// which its inner schema holds none of: every kind is inhabited, so such a
    /// complement has a value to stand a refutation on.
    ///
    /// Read here, for the subject of a refutation, and not by emptiness: a
    /// complement read inhabited inside a record or a meet lets refutations
    /// stand that the union rule then narrows on, and that work starves a
    /// proof elsewhere in the query -- a guarded meet against its own split
    /// spends 835,074 of the budget's million steps without the reading and
    /// runs out with it.
    #[inline]
    fn holds_a_kind_whole(&self, cx: &Cx<'_>) -> bool {
        matches!(self, Schema::Complement(inner) if inner.misses_a_kind(cx))
    }

    /// The rules' answer about this pair, with every refutation read against
    /// the subject it is about. See [`witnessed`](Self::witnessed).
    fn is_subtype_rec(&self, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        // Every goal the recursion is asked reaches this, and reaches it after
        // the caches that answer a repeat without asking -- which is what makes
        // the count below a count of goals the procedure really re-derives.
        #[cfg(test)]
        goal_tests::goals::record(self, other);
        let answer = self.subtype_by_rules(other, cx);
        if answer == Relation::Unknown {
            return self.a_goal_declined(other, cx);
        }
        self.witnessed(answer, cx)
    }

    /// The rules' answer before the witness reads it: reflexivity, the scalar
    /// fragment, a goal already on the trail assumed to hold [AC93, §1.5], and
    /// the rules below for the rest.
    fn subtype_by_rules(&self, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        // Reflexivity, by identity and then by structure. The checks below are
        // ordered by what they cost, and each answers the whole query on its
        // own, so a cheaper one never hides a verdict a later one would reach.
        // Both come before the budget: they read the two terms, and a
        // reflexive pair refused for want of a step is an answer lost to the
        // work of rules that have nothing to say about it. The discriminants
        // are compared inline first, so a pair of two shapes -- most pairs --
        // pays no call: asked bare, the comparison cost the matrix and the
        // rules corpus 1.2% each, and guarded it costs neither anything.
        if core::ptr::eq(self, other)
            || (core::mem::discriminant(self) == core::mem::discriminant(other) && self == other)
        {
            return Relation::Holds;
        }
        // Bound the total work: the distribution rules below can demand effort
        // exponential in the schema depth, so once the shared budget is spent the
        // decision answers `Unknown` rather than running unbounded. A real
        // annotation decides in a few steps; only an adversarial schema reaches
        // the ceiling.
        if !spend(cx.budget) {
            // The work bound stopping a descent proves nothing about the pair,
            // except what the terms place on their own, which costs no step.
            return Relation::proven(placed_by_membership(self, other));
        }
        // Scalar fragment: exact via the region partition. Subtyping there is
        // set inclusion between the two region sets, and nothing else. A
        // non-scalar node yields `Unknown` from its own discriminant, so this
        // costs one match off the fragment.
        //
        // The subtype's own regions are read only once the supertype's are
        // known. Reading a region set is one match on an atom and a fold over
        // every member of a union, so on a Boolean schema it walks the subtree;
        // asking for a set that a mismatched partner has already made unusable
        // walks it for nothing. Both are pure, so which is asked first is free
        // to choose, and the answer is the same either way.
        let supertype_regions = other.region_set();
        if let Regions::Known(b) = supertype_regions
            && let Regions::Known(a) = self.region_set()
        {
            // Exact in both directions: on the scalar fragment the region sets
            // *are* the schemas, unions and complements of disjoint regions
            // read as the sets [FCB08, Def. 4.1], so a region outside the
            // supertype's set is a value outside it.
            return Relation::decided(a.subset_of(b));
        }
        // Coinductive hypothesis: a goal already being proven on this path is
        // assumed to hold, so two recursive types are compared at their greatest
        // fixpoint rather than unfolded forever. The scan is empty for a
        // non-recursive query; under recursion the stack holds one entry per
        // reference goal still being unfolded. Every recorded goal has a `Ref` on
        // one side, so the structural compare rejects a mismatched goal on the
        // discriminant before walking either subtree.
        if cx.trail.iter().any(|(a, b)| a == self && b == other) {
            #[cfg(test)]
            goal_tests::goals::met_again(self, other);
            return Relation::Holds;
        }
        // A level of the recursion, held while the rules descend: a pair
        // answered above opens none, and one past the depth bound proves
        // nothing, as one past the work bound does not.
        descending(cx.budget, Relation::Unknown, || {
            self.subtype_decide(other, supertype_regions, cx)
        })
    }

    /// Whether one of the lattice bounds settles `self ⊆ other`: `self` denotes
    /// the empty set [BBC+26, (uninhabited)], or `other` denotes the whole
    /// universe.
    ///
    /// Decided by emptiness rather than by matching the `Nothing` or `Anything`
    /// atom, so a record with an uninhabited required field and a union covering
    /// the universe are both recognized. Asking the atom alone is a rule that
    /// confirms itself: the pattern matches only the shape it is written for.
    ///
    /// The universe side asks whether the complement is empty, which is the
    /// same question one De Morgan step away [FCB08, §4.1] and needs no
    /// separate procedure.
    ///
    /// **Only the walk is here.** The cheap half of the universe bound -- the
    /// supertype's region set being the whole partition -- is a comparison the
    /// caller already holds, and [`subtype_decide`](Self::subtype_decide)
    /// answers on it before the rules run. A pair that reaches this one has
    /// been past that comparison, so repeating it reads a value known to be
    /// something else. What is left is the reading the comparison cannot do:
    /// the subject's own emptiness, and the complete walk over a union whose
    /// members cover the universe between them.
    fn bounds_the_pair(&self, other: &Schema, cx: &mut Cx<'_>) -> bool {
        if self.is_empty_rec(cx) {
            return true;
        }
        // The regions again, read without the stop the fast fold takes at the
        // first opaque member. `~bool | bool` covers the universe and so does
        // the same union with a set beside it, which the fast reading declines
        // because it stops on the set and a member behind it cannot reopen the
        // answer -- so the same three members read as the universe in one order
        // and as unknown in another.
        //
        // Reading on is what the fast fold cannot afford: it is asked of every
        // pair, and a union of a thousand literals is a thousand opaque members
        // to walk rather than one to stop at. Here the pair has had every rule
        // and none of them answered, so the walk is spent on a query that was
        // going to be declined, and a query that decides never reaches it.
        other.covers_the_universe() || other.union_complement_is_empty(cx)
    }

    /// Whether a union's complement holds no value, read one De Morgan step
    /// in: `¬(X₁ ∪ … ∪ Xₙ)` is `¬X₁ ∩ … ∩ ¬Xₙ`, the normal form of a negated
    /// union [FCB08, Lem. 6.2], and a meet is what emptiness decides by its
    /// members.
    ///
    /// The region walk above passes over a member the partition cannot read,
    /// and that member may be the one covering what the others leave out.
    /// `~None | ~set[Any]` is the universe -- no value is both `None` and a
    /// set -- yet `~set[Any]` has no region, so the walk sees every region but
    /// `None` and declines. Pushed through the complement, the same union is
    /// `None ∩ set[Any]`, a meet of two kinds that share no value, and the
    /// rules settle it without lowering anything.
    ///
    /// **A member that is itself a union is read as its members**, at any
    /// depth. `X ∪ (Y ∪ Z)` is `X ∪ Y ∪ Z`, so the two meets of complements are
    /// one set, and only the flat one shows the meet's own fold the pair it
    /// reads. `¬c ∪ (c)`, with `(c)` a union of one member, is the universe;
    /// taken whole, its complements are `c ∩ ¬(c)`, and the fold compares
    /// `¬(c)` with `c` and finds two spellings. Grouping holds of every set, a
    /// class with a hook among them, so which pairs cancel is the fold's own
    /// guard to say. The constructors flatten what they build, so the nesting
    /// arrives in a term built raw, which the fuzz target draws, and the walk
    /// above reads such a member as one region or as none.
    ///
    /// **The meet is the supertype's alone, so the set representation reads
    /// it where the rules leave it open** ([`Cx::denotes_no_value`]), as the
    /// complement rule reads its inner set alone. Two patterns no word
    /// matches both are a meet only the sets empty, and the descriptor route
    /// lowers the subject first, so a subject the set reading cannot lower --
    /// a literal the null oracle cannot place -- refuses the pair there: a
    /// union `is_empty` proves the universe, through the sets, bounded every
    /// set but that subject.
    ///
    /// Asked only of a union, only where every rule and the walk have
    /// declined, and only where the walk passed a member over, so a query
    /// that decides never builds the meet. Building it for every declined
    /// union cost the relation matrix a fifth of its instructions.
    fn union_complement_is_empty(&self, cx: &mut Cx<'_>) -> bool {
        let Schema::Union(members) = self else {
            return false;
        };
        // A walk that skipped no member read every one exactly, so its `false`
        // is the answer and the meet would only say it again.
        if members.iter().all(|m| m.region_set() != Regions::Unknown) {
            return false;
        }
        let complements = flat_members(members).map(|member| member.clone().complement());
        let meet = Schema::meet_within(complements, cx.defs);
        meet.is_empty_rec(cx) || cx.denotes_no_value(&meet)
    }

    /// Whether `self` and `other` share no value, which is what decides `self ⊆
    /// ¬other`.
    ///
    /// This is the semantic subtyping reduction `[[s ∧ ¬t]] = ∅` [FCB08, §4.1]
    /// applied where the structural arms have nothing to say: a complement
    /// offers no shape on the right to recurse into, so the question goes to
    /// emptiness, which already decides kind disjointness and the scalar
    /// regions. Without it a container is never seen below the complement of a
    /// scalar.
    fn shares_no_value_with(&self, other: &Schema, cx: &mut Cx<'_>) -> bool {
        self.verdict_beside(other, cx) == Verdict::Empty
    }

    /// Whether `self` and `other` share a value, as a verdict: the meet's, the
    /// question `self ⊆ ¬other` reduces to [FCB08, §4.1], which
    /// [`shares_no_value_with`](Self::shares_no_value_with) reads for its
    /// empty answer and the complement arm for both.
    fn verdict_beside(&self, other: &Schema, cx: &mut Cx<'_>) -> Verdict {
        // Kind disjointness reads two discriminants and settles most pairs: a
        // list never shares a value with an int, positives of two kinds meeting
        // in nothing [FCB08, (5)]. The general question below owns the rest,
        // and has to build the meet to ask it.
        if self.disjoint_with(other, cx.oracle) {
            return Verdict::Empty;
        }
        // Built through the meet constructor rather than as a raw node: the
        // emptiness rule that decides most pairs compares the *members* of one
        // intersection pairwise, and a member that is itself an intersection
        // hides its own members from it. Asked of `tuple[int, str]` against
        // `dict[str, int] & ~{"a": int}` as a raw node, the tuple and the dict
        // would never meet, so `A <= ~B` would decline for a pair whose meet
        // `is_empty` decides, and the two relations would disagree about one
        // question asked two ways. A query asks one such meet again at every
        // goal reaching the pair, so it is read once a query.
        emptiness::held_verdict(
            Schema::meet_within([self.clone(), other.clone()], cx.defs),
            ReadBack::AsRead,
            cx,
        )
    }

    /// Whether `self` shares no value with `part`, a part of the supertype: a
    /// branch of its union, a component of a product branch, a slot of a
    /// record branch.
    ///
    /// The meet reads `part` at a positive position, and `part` sits at a
    /// negative one of the question `self ∧ ¬T` a goal decides. A definition
    /// the emptiness recursion is resolving is read as empty [LN26, §5.1],
    /// which narrows a subject soundly and a supertype not at all: read inside `part`, it
    /// drops a branch the subject's values fall in and refutes what holds.
    /// So where a definition is being resolved the meet is not asked, and the
    /// two are told apart by the kinds each may hold, a reference read through
    /// its definition: an over-approximation of each set, and one that reads
    /// no hypothesis.
    fn shares_no_value_with_a_part_of_the_supertype(&self, part: &Schema, cx: &mut Cx<'_>) -> bool {
        if cx.visiting.is_empty() {
            return self.shares_no_value_with(part, cx);
        }
        self.disjoint_with(part, cx.oracle)
            || self.kinds_possibly_held(cx.oracle, cx.defs)
                & part.kinds_possibly_held(cx.oracle, cx.defs)
                == 0
    }

    /// Whether `self` reduces to something below `other` by a rule that reads
    /// only the left side: a reference unfolds to its definition, and a
    /// refinement drops to its base.
    ///
    /// Both rules are sound alone, and the `A ⊆ (Y ∪ Z)` rule beside them is
    /// *lossy* -- it commits to a single branch, so a subject that lands in the
    /// union only once it has been reduced gets no answer from it. A match picks
    /// one arm, but the relation is the disjunction of every sound rule that
    /// applies, so where both do, both are asked. That is what decides a
    /// recursive schema against its own body, and a refinement of a union
    /// against that union.
    ///
    /// The reference case records its goal before descending, the trail
    /// algorithm's step for a recursive type on the left [AC93, §1.5], so a
    /// cycle back to it meets the coinductive hypothesis rather than unfolding
    /// forever.
    fn left_reduces_below(&self, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        match self {
            Schema::Ref(id) => match cx.defs.get(id.get()) {
                Some(def) => cx.assuming((self.clone(), other.clone()), |cx| {
                    def.is_subtype_rec(other, cx)
                }),
                // A reference into a table that does not hold it: the schema
                // names a set this query cannot read, which is not a refutation.
                None => Relation::Unknown,
            },
            // A refinement carries its base's *proof* and not its refutation:
            // the constraints can exclude every value that would stand against
            // the inclusion, which is how a bounded-length list lands inside a
            // fixed-length one whose base it is nowhere near.
            Schema::Refine { base, .. } => base.is_subtype_rec(other, cx).proof_only(),
            // Nothing on the left reduces, so this rule has nothing to say --
            // which is not the same as the relation failing.
            _ => Relation::Unknown,
        }
    }

    /// `A ⊆ R` for a reference `R`: assume the goal and ask the definition.
    ///
    /// The coinductive rule [AC93, §1.5]. The pair goes on the trail so a goal
    /// that comes back is answered by the assumption, and the definition is
    /// what `R` denotes, so the answer about it is the answer about `R`.
    /// `Unknown` for a supertype that is no reference, or a reference the table
    /// does not resolve, which names a set this query cannot read.
    fn below_a_reference(&self, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        let Schema::Ref(id) = other else {
            return Relation::Unknown;
        };
        let Some(def) = cx.defs.get(id.get()) else {
            return Relation::Unknown;
        };
        cx.assuming((self.clone(), other.clone()), |cx| {
            self.is_subtype_rec(def, cx)
        })
    }

    /// `A ⊆ (Y ∪ Z)`: every rule that can place a subject inside a union.
    ///
    /// A branch equal to the subject settles it, which is set containment and
    /// is linear in the branches. Failing that, the subject may land in one
    /// branch [BBC+26, (anyOf-r)], or -- for a fixed-arity sequence -- split
    /// across them, which no single-branch rule sees, or be one a left-side
    /// rule reduces to something the union contains.
    ///
    /// Every rule here proves inclusion when it fires, so a union none of them
    /// places the subject in is a pair this procedure declines -- with one
    /// exception, which is the reduction below. A reference denotes exactly its
    /// definition, so a definition that holds a value outside the union names a
    /// value of the reference outside it, and that is a refutation rather than
    /// a decline. The arm for a non-union supertype already reads it that way;
    /// this reads it the same, which is the only difference a union makes.
    fn below_a_union(&self, other: &Schema, members: &Members, cx: &mut Cx<'_>) -> Relation {
        if members.contains(self) {
            return Relation::Holds;
        }
        // Each branch is asked once and the answer read for two things: a
        // branch that contains the subject settles the union, and a branch that
        // *refutes* is the answer the union inherits once the branches beside
        // it are gone. The second is what the narrowing below is for, and it is
        // why each answer is read in its three values rather than as `holds`
        // alone: whether any branch refuted is what decides to narrow.
        let refuted = if cx.budget.depth.get() == 1 && members.len() <= MEMBERS_HELD {
            match self.each_branch_at_the_root(members, cx) {
                Some(refuted) => refuted,
                None => return Relation::Holds,
            }
        } else {
            let mut refuted = false;
            for member in members.iter() {
                let answer = self.is_subtype_rec(member, cx);
                if answer.holds() {
                    return Relation::Holds;
                }
                refuted |= answer == Relation::Fails;
            }
            refuted
        };
        if members
            .iter()
            .filter(|member| matches!(member, Schema::Intersection(_)))
            .nth(1)
            .is_some()
        {
            let merged = self.below_a_merged_split(members, cx);
            if merged != Relation::Unknown {
                return merged;
            }
        }
        // The kind tested here and the rules out of line: a split is the rare
        // answer, and the union rule is inlined into the goal recursion, so
        // every union this path reaches would otherwise carry both rules.
        if matches!(self, Schema::Seq { .. } | Schema::KeyedMap { .. }) {
            let split = self.splits_across_union(members, cx);
            if split != Relation::Unknown {
                return split;
            }
            if readings::one_item_outside(self, members, cx) {
                return Relation::Fails;
            }
        }
        // A branch the subject shares no value with decides nothing about the
        // inclusion: `A ⊆ X ∪ Y` with `A ∩ X = ∅` is `A ⊆ Y`. Dropping such a
        // branch is what lets a *refutation* out -- the rules refute against a
        // single supertype, and a union but for one item, so
        // without it a chain of records beside a `None` branch is undecided
        // where the record branch alone refutes.
        //
        // Sound in both directions, by two facts: the narrowed union is a subset
        // of the original, so a `Holds` against it holds against the original,
        // and a `Fails` stands on a value of the subject outside every remaining
        // branch -- a value the dropped branches do not admit either, since they
        // share none with the subject.
        //
        // Asked with the cheap reading of disjointness first -- two
        // discriminants, which settles a `None` branch beside a record -- and
        // with the full one only where that left the question open. The full
        // reading builds a meet per branch, and the path it is asked on is the
        // one already headed for the set representation, which costs about
        // three orders of magnitude more.
        //
        // And asked at all only where a branch refutes, which is the answer
        // the narrowing carries back from the branch it leaves: a union every
        // branch of which *declines* narrows to a smaller union of declines.
        // One reading could answer there and does not -- every branch dropped,
        // which refutes on a value of the subject however its branches
        // answered -- and reaching for it costs a meet per branch on every
        // union in the language. Measured on the relation matrix: 36% of the
        // decision path asked of every union against 0.14% asked of the unions
        // a branch refutes. Where it leaves a pair undecided the set
        // representation is asked next, which is the route that pair takes
        // where no branch is dropped.
        if refuted {
            for full in [false, true] {
                let answer = self.below_the_branches_it_can_meet(members, full, cx);
                if answer != Relation::Unknown {
                    return answer;
                }
            }
        }
        // Asked once, and read twice: for the proof it may carry, and -- where
        // it carries none -- for the refutation, which only a reference's
        // reduction can give. A refinement's reduction drops its refutation
        // before this sees it, since the constraints may exclude the very value
        // that stood against the inclusion.
        let reduced = self.left_reduces_below(other, cx);
        if reduced.holds()
            // Last, and only for the one subject the oracle can answer about
            // here: an `Instance` whose *values* the bindings can enumerate
            // is below a union when each of them is, which is what makes an
            // enumeration and the union of its members one set. Asked here
            // because a union on the right never reaches the leaf arm, and
            // asked for nothing else because every other subject would pay a
            // call that always declines -- ten percent of the decision
            // workload, measured.
            || (matches!(self, Schema::Instance(_))
                && cx.oracle.leaf_subtype(self, other).unwrap_or(false))
        {
            return Relation::Holds;
        }
        reduced.refutation_only()
    }

    /// The subject against the union `members` denote with two branches that
    /// split one set by another merged into the set they split
    /// ([`merged_split`]): the same set, so its answer, either way, is the
    /// union's. `Unknown` where no two branches merge.
    ///
    /// Asked by the union arm and by the meet arm, which answers a meet subject
    /// on its own, each behind a count of the meets among the branches, two
    /// being where such a pair can stand: a split is the rare supertype, and
    /// both arms are on every union's path.
    #[inline(never)]
    fn below_a_merged_split(&self, members: &[Schema], cx: &mut Cx<'_>) -> Relation {
        match merged_split(members, cx) {
            Some(merged) => self.is_subtype_rec(&merged, cx),
            None => Relation::Unknown,
        }
    }

    /// `A ⊆ (Y ∪ Z)` where `A` lands in no single branch and splits across
    /// several: a fixed-length sequence across sequences of its arity
    /// ([`seq_splits_across_union`]), a keyed map across keyed maps
    /// ([`keyed_map_splits_across_union`]). `Unknown` for any other subject.
    #[inline(never)]
    fn splits_across_union(&self, members: &[Schema], cx: &mut Cx<'_>) -> Relation {
        match self {
            Schema::Seq { .. } => seq_splits_across_union(self, members, cx),
            Schema::KeyedMap { .. } => keyed_map_splits_across_union(self, members, cx),
            _ => Relation::Unknown,
        }
    }

    /// `A ⊆ X ∪ Y` where `A ∩ X = ∅`, which is `A ⊆ Y`, the union and the meet
    /// denoting the union and the intersection of the sets [FCB08, Def. 4.1].
    ///
    /// Answers `Unknown` where no branch is dropped, so a caller can ask twice
    /// -- once cheaply and once fully -- without the first answer standing in
    /// for the second.
    ///
    /// Sound in both directions, by two facts. The narrowed union is a subset
    /// of the original, so a `Holds` against it holds against the original.
    /// And a `Fails` stands on a value of the subject outside every remaining
    /// branch, which the dropped branches do not admit either -- they share no
    /// value with the subject at all.
    fn below_the_branches_it_can_meet(
        &self,
        members: &[Schema],
        full: bool,
        cx: &mut Cx<'_>,
    ) -> Relation {
        let mut meets = |member: &Schema| {
            if full {
                !self.shares_no_value_with_a_part_of_the_supertype(member, cx)
            } else {
                !self.disjoint_with(member, cx.oracle)
            }
        };
        // Asked before the branches are collected, because the common answer is
        // that none of them goes: a union that narrows to itself is every union
        // the subject meets at every branch, and building the list to discover
        // that is an allocation and a clone per branch on a path that returns
        // without reading either.
        if members.iter().all(&mut meets) {
            return Relation::Unknown;
        }
        let narrowed: Vec<Schema> = members.iter().filter(|m| meets(m)).cloned().collect();
        match &narrowed[..] {
            // Every branch dropped: the subject shares no value with the union
            // at all. That is a *refutation* where the subject holds a value --
            // the value is outside every branch, so it is outside the union --
            // and a proof where it holds none, since the empty set is below
            // everything. Reading it as "proved, or not proven" loses the
            // refutation, which a caller acts on as much as the proof:
            // `dict[str, int]` shares no value with either of `int` and `str`,
            // and the empty dict is the value that says so.
            //
            // Which of the two it is, is read where the goal returns
            // ([`witnessed`](Self::witnessed)), as every refutation is: the
            // subject is the goal's, and that reading answers a subject with a
            // value by its shape before it walks one. Read here as well, the
            // subject's emptiness would be walked twice, the second time on a
            // budget the first has spent.
            [] => Relation::Fails,
            [only] => self.is_subtype_rec(only, cx),
            _ => self.is_subtype_rec(&Schema::Union(narrowed.into()), cx),
        }
    }

    /// `(A ∩ B) ⊆ C`: a meet is below whatever one of its conjuncts is below.
    ///
    /// When `C` is a union the meet may instead land in one branch
    /// [BBC+26, (anyOf-r)], so that sound rule is tried here too -- ahead of
    /// the plain `_ ⊆ (Y ∪ Z)` rule, so a meet that contains its own supertype
    /// (a reference beside that union) decides, which is what lets such a meet
    /// be recognized as a subtype of itself.
    ///
    /// Both halves are sound and incomplete: a conjunct that does not contain
    /// the supertype says nothing about the meet, so a disjunction of them that
    /// finds no proof is unproven rather than refuted. One shape refutes past
    /// them: a class met with the attributes its instances carry, against a
    /// class the oracle places it outside or a supertype outside every kind it
    /// has.
    fn meet_below(&self, other: &Schema, members: &[Schema], cx: &mut Cx<'_>) -> Relation {
        // The placements by membership first, before a member is asked below
        // the supertype: a member dear enough to spend the budget would leave
        // the meet unplaced in a union that holds it.
        if placed_by_membership(self, other) {
            return Relation::Holds;
        }
        // A meet that may hold a value of no kind holds no value, and is below
        // everything [BBC+26, (uninhabited)]. It holds no kind a member cannot
        // hold, and a complement holds none its inner set holds whole
        // [FCB08, Def. 4.1] (`kinds_possibly_held`), so two members of kinds
        // apart, or a kind's complement beside a member of that kind, empty
        // it. Read before a member is asked below the supertype, which asks the
        // member whole -- a reference's definition walked, a pattern's tables
        // built -- where a member refuted says nothing of the meet: a meet of
        // patterns over `str` beside `~str` built them all to be placed by the
        // complement. Asked of a meet holding a complement, or below a
        // reference, alone: the walk asked of every meet subject cost the
        // relation matrix 3.03%, and of these two 0.60%.
        if (matches!(other, Schema::Ref(_))
            || members
                .iter()
                .any(|member| matches!(member, Schema::Complement(_))))
            && self.kinds_possibly_held(cx.oracle, cx.defs) == 0
        {
            return Relation::Holds;
        }
        // Every placement here proves and none refutes: a member that is *not*
        // below the supertype says nothing about the meet, which is a smaller
        // set than that member. So a pair none of them places is handed to the
        // reading every pair with no rule gets, rather than ending the match.
        let placed = Relation::proven(
            Relation::any(members.iter().map(|m| m.is_subtype_rec(other, cx))).holds()
                || matches!(other, Schema::Union(branches)
                    if Relation::any(
                        branches
                            .iter()
                            .map(|b| self.is_subtype_rec(b, cx)),
                    )
                    .holds()),
        );
        if placed.holds() {
            return placed;
        }
        // One shape does refute, and it is the meet a dataclass lowers to. A
        // class together with the attributes its instances carry is below
        // another class exactly as its class is: the value the meet holds is a
        // *direct* instance of `C` carrying those attributes, so `type(v) is C`
        // settles `isinstance(v, D)` through the order alone, and a class
        // deriving from both changes nothing about that value. The guard around
        // this rule reads whether the meet has it.
        if let Some((class, _)) = class_with_attributes(members) {
            if matches!(other, Schema::Instance(_))
                && cx.oracle.leaf_subtype(&Schema::Instance(class), other) == Some(false)
            {
                return Relation::Fails;
            }
            // And against a supertype every value of which has one kind: the
            // same direct instance either has that kind or does not, and the
            // oracle reads `type(v) is C` rather than the subtree beneath `C`,
            // which is what makes the answer a value rather than an open-world
            // guess.
            if outside_every_kind(other, class, cx) {
                return Relation::Fails;
            }
        }
        // A union splitting a set by another is that set, asked last: the
        // readings above answer the dataclass meets a union of kinds is asked
        // about, and a split is the rare supertype.
        if let Schema::Union(branches) = other
            && branches
                .iter()
                .filter(|branch| matches!(branch, Schema::Intersection(_)))
                .nth(1)
                .is_some()
        {
            let merged = self.below_a_merged_split(branches, cx);
            if merged != Relation::Unknown {
                return merged;
            }
        }
        placed
    }

    /// The structural subtyping decision: the lattice, recursion, and
    /// constructor-matching rules.
    ///
    /// Reached from [`subtype_by_rules`](Self::subtype_by_rules) after the
    /// coinductive, scalar and identity fast paths, which are the three that
    /// answer without reading a shape.
    fn subtype_decide(
        &self,
        other: &Schema,
        supertype_regions: Regions,
        cx: &mut Cx<'_>,
    ) -> Relation {
        // Two finite sets of constants, before the lattice rules distribute
        // them: inclusion between them is membership, and membership is a
        // lookup. What the distribution below would make of the same pair is a
        // decision step per pair of members, which is the product the budget
        // binds; this is a walk of the two lists, which is a sum.
        if let Some(supertype) = finite_set(other)
            && let Some(subject) = finite_set(self)
        {
            let answer = finite_set_below(subject, supertype, other, cx.oracle);
            if answer != Relation::Unknown {
                return answer;
            }
        }
        // `A ⊆ U`, read off the supertype's regions alone: the caller already
        // holds them for the scalar rule, so this is a comparison rather than a
        // walk, and it belongs ahead of the rules because a universe on the
        // right answers every pair. The other lattice bound is a walk and is
        // asked in `or_bounded`, after the rules have had the pair.
        if supertype_regions == Regions::Known(Region::ALL) {
            return Relation::Holds;
        }
        // A complemented union or meet is read pushed one level in, and taken
        // where the pushed term is the supertype or placed in it by
        // membership. Asked before the arms by shape, which would distribute
        // the supertype or unfold it and reach the complement only branch by
        // branch, where no branch alone holds it.
        if let Schema::Complement(inner) = self
            && matches!(&**inner, Schema::Union(_) | Schema::Intersection(_))
            && pushed_complement_is_placed(inner, other, cx.defs)
        {
            return Relation::Holds;
        }
        let answer = self.subtype_by_shape(other, cx);
        self.or_bounded(answer, other, cx)
    }

    /// A subject below a meet, a member at a time: `A ⊆ (Y ∩ Z)` iff `A ⊆ Y`
    /// and `A ⊆ Z`, a meet denoting the intersection of its members' sets
    /// [FCB08, Def. 4.1]. At the root of a query, what the rules answer of each
    /// member is kept ([`below_each_member_at_the_root`]); below it nothing is.
    ///
    /// [`below_each_member_at_the_root`]: Self::below_each_member_at_the_root
    fn below_each_member(&self, members: &Members, cx: &mut Cx<'_>) -> Relation {
        if cx.budget.depth.get() == 1 {
            return self.below_each_member_at_the_root(members, cx);
        }
        Relation::all(members.iter().map(|m| self.is_subtype_rec(m, cx)))
    }

    /// [`below_each_member`](Self::below_each_member) at the root of a query.
    ///
    /// There no hypothesis is open -- every stack of them is pushed by a level
    /// above the goal reading it -- so what the rules answer of a member is an
    /// answer about that pair. Where they decline the meet, the members they
    /// place the subject below are kept for the query ([`HELD`]), and the
    /// parts reading asks the set representation of the members left: asking
    /// the rules again of each one reads the answers this arm read, which on a
    /// meet of records is the cost of the whole query a second time.
    ///
    /// Out of line, with the depth tested by the caller: the goals below the
    /// root are most of a query's meets, and they keep nothing.
    #[inline(never)]
    fn below_each_member_at_the_root(&self, members: &Members, cx: &mut Cx<'_>) -> Relation {
        let keeps = members.len() <= MEMBERS_HELD;
        let mut placed = 0u64;
        let answer = Relation::all(members.iter().enumerate().map(|(at, member)| {
            let answer = self.is_subtype_rec(member, cx);
            if keeps && answer == Relation::Holds {
                placed |= 1 << at;
            }
            answer
        }));
        if keeps && answer == Relation::Unknown {
            let query = cx.query();
            HELD.with_borrow_mut(|held| {
                *held = Some((query, self.clone(), Arc::clone(members), placed, 0));
            });
        }
        answer
    }

    /// The branches of a union supertype asked at the root of a query, as the
    /// union rule asks them: `None` where one holds `self`, and otherwise
    /// whether one refuted it. A refutation stands on a value of the subject
    /// outside the branch, whatever is assumed, so the branches refuted are
    /// kept for the query ([`HELD`]), for the parts reading to read back rather
    /// than lower: both deciders are sound, and a containment the rules refuted
    /// is one the set representation never proves
    /// ([`Cx::refuted_at_the_root`]). Out of line, so the union rule the goal
    /// recursion inlines asks its branches as it did.
    #[inline(never)]
    fn each_branch_at_the_root(&self, members: &Members, cx: &mut Cx<'_>) -> Option<bool> {
        let mut refuted = 0u64;
        for (at, member) in members.iter().enumerate() {
            let answer = self.is_subtype_rec(member, cx);
            if answer.holds() {
                return None;
            }
            if answer == Relation::Fails {
                refuted |= 1 << at;
            }
        }
        if refuted != 0 {
            let query = cx.query();
            HELD.with_borrow_mut(|held| {
                *held = Some((query, self.clone(), Arc::clone(members), 0, refuted));
            });
        }
        Some(refuted != 0)
    }

    /// A union below `other`, member by member: a union is below a schema
    /// exactly where each of its members is [BBC+26, (anyOf-l)], so one member
    /// refuted refutes it.
    ///
    /// A table of constants below the complement of another is the two tables
    /// sharing no value, which the oracle answers for both at once, and a
    /// constant they share is a value of the subject outside the supertype. Distributed, each constant asks it against the whole other
    /// table, which is quadratic across the boundary.
    ///
    /// The members are asked until one refutes, or until one declines with the
    /// budget spent. A goal asked on a spent budget answers what the two terms
    /// place by membership and declines the rest ([`subtype_by_rules`]), and is
    /// never refuted, so past that decline the members left can neither prove
    /// the union nor refute it. Asked anyway, each scans the supertype's
    /// branches for itself: a union of four thousand records against another
    /// compares sixteen million pairs for the answer the first decline gave.
    ///
    /// [`subtype_by_rules`]: Self::subtype_by_rules
    fn union_below(&self, members: &[Schema], other: &Schema, cx: &mut Cx<'_>) -> Relation {
        if let Schema::Complement(inner) = other
            && matches!(members.first(), Some(Schema::Literal(_)))
            && matches!(**inner, Schema::Literal(_) | Schema::Union(_))
            && let Some(disjoint) = tables_disjoint(self, inner, cx.oracle)
        {
            return Relation::decided(disjoint);
        }
        // Two discriminated unions: a member whose tag no member of the other
        // pins shares no value with it, and is below its complement by that
        // tag alone, so only the members a pair names are read (`tag_pairs`).
        // Read a member at a time, every member against every member of the
        // other, two unions of sixteen hundred records took a quarter second.
        if let Schema::Complement(inner) = other
            && let Schema::Union(theirs) = &**inner
            && matches!(members.first(), Some(Schema::KeyedMap { .. }))
            && let Some(pairs) = tag_pairs(members, theirs, cx.oracle)
        {
            let mut named: Vec<usize> = pairs.into_iter().map(|(at, _)| at).collect();
            named.dedup();
            return Schema::each_below(
                named.into_iter().filter_map(|at| members.get(at)),
                other,
                cx,
            );
        }
        if cx.budget.depth.get() == 1
            && let Schema::Union(kept) = self
        {
            return Schema::each_member_below_at_the_root(kept, other, cx);
        }
        Schema::each_below(members, other, cx)
    }

    /// [`union_below`](Self::union_below) at the root of a query, where no
    /// hypothesis is open, so what the rules answer of a member is an answer
    /// about that pair: where they decline the union, the members they place
    /// below `other` are kept for the query ([`HELD`]), and the parts reading
    /// asks the set representation of the members left rather than the rules
    /// again, which would read the answers this arm read -- on a union of
    /// refined integers drawn eight deep, half of the query a second time.
    /// The meet supertype's arm keeps its members alike
    /// ([`below_each_member_at_the_root`](Self::below_each_member_at_the_root)).
    ///
    /// Out of line, with the depth tested by the caller.
    #[inline(never)]
    fn each_member_below_at_the_root(
        members: &Members,
        other: &Schema,
        cx: &mut Cx<'_>,
    ) -> Relation {
        let keeps = members.len() <= MEMBERS_HELD;
        let mut placed = 0u64;
        let mut declined = false;
        for (at, member) in members.iter().enumerate() {
            match member.is_subtype_rec(other, cx) {
                Relation::Fails => return Relation::Fails,
                Relation::Unknown => declined = true,
                Relation::Holds => {
                    if keeps {
                        placed |= 1 << at;
                    }
                }
            }
            if declined && cx.budget.left() == 0 {
                a_bound_is_met();
                return Relation::Unknown;
            }
        }
        if !declined {
            return Relation::Holds;
        }
        if keeps {
            let query = cx.query();
            HELD.with_borrow_mut(|held| {
                *held = Some((query, other.clone(), Arc::clone(members), placed, 0));
            });
        }
        Relation::Unknown
    }

    /// Whether each of `members` is below `other`, as the arm calling this
    /// states: `Fails` at the first refuted, `Unknown` where one declined, and
    /// `Unknown` at once where one declined with the query's steps spent.
    #[inline]
    fn each_below<'m>(
        members: impl IntoIterator<Item = &'m Schema>,
        other: &Schema,
        cx: &mut Cx<'_>,
    ) -> Relation {
        let mut declined = false;
        for member in members {
            match member.is_subtype_rec(other, cx) {
                Relation::Fails => return Relation::Fails,
                Relation::Unknown => declined = true,
                Relation::Holds => {}
            }
            if declined && cx.budget.left() == 0 {
                a_bound_is_met();
                return Relation::Unknown;
            }
        }
        if declined {
            Relation::Unknown
        } else {
            Relation::Holds
        }
    }

    /// A meet the member rule and the table reading leave, read by the rest of
    /// its readings: as one record below a map or a union of maps, as sharing
    /// no value with a complement's inner set, then as a pair with no rule of
    /// its own, then through a reference.
    ///
    /// The meet arm asks the narrowed meet after these, and never before:
    /// asked before them, a narrowed meet that declines has spent the steps
    /// they share, and asked in their place, its readings miss what the
    /// meet's own spelling gives them.
    fn a_meet_read_whole(&self, members: &[Schema], other: &Schema, cx: &mut Cx<'_>) -> Relation {
        match other {
            Schema::KeyedMap { .. } | Schema::Union(_) => keyed_meet_below(members, other, cx),
            Schema::Complement(inner) => match self.verdict_beside(inner, cx) {
                Verdict::Empty => Relation::Holds,
                Verdict::Inhabited => Relation::Fails,
                Verdict::Unknown => Relation::Unknown,
            },
            _ => Relation::Unknown,
        }
        .or_else(|| self.unstructured(other, cx))
        .or_else(|| self.below_a_reference(other, cx))
    }

    /// The arms that match a pair by the shapes on its two sides.
    ///
    /// They are rules of the kind a rule-based inclusion states: a union is
    /// below a schema where each member is [BBC+26, (anyOf-l)], a set below a
    /// set where its element is [FCB08, Lem. 6.30], and a schema below a
    /// complement where the two share no value [FCB08, §4.1].
    ///
    /// Split from its caller because the two lattice bounds around it are asked
    /// at different points and the whole read past what one function may be.
    /// No `#[inline]`: it moves no workload, since the compiler is already free
    /// to inline within the crate.
    fn subtype_by_shape(&self, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        match (self, other) {
            // (X ∪ Y) ⊆ Z iff X ⊆ Z and Y ⊆ Z [BBC+26, (anyOf-l)]; A ⊆ (Y ∩ Z)
            // iff A ⊆ Y and A ⊆ Z.
            (Schema::Union(members), _) => self.union_below(members, other, cx),
            (_, Schema::Intersection(members)) => self.below_each_member(members, cx),
            // A meet is placed by the member rule where a member is below the
            // supertype, and where none is -- the meet is a branch of a
            // definition, say -- the reference it may be asked against is
            // unfolded, as it is for any other subject: a reference denotes
            // exactly its definition, and the decomposition that runs first is
            // the cheap route rather than the only one.
            //
            // A meet of tables and complements of tables is a table, which is
            // what a record's tag narrows to, and is asked as one once the
            // member rule has not placed it, and a meet a complement beside its
            // union narrows is the narrowed meet, the same set, asked after
            // every other reading (`narrowed_meet_below`). A meet of keyed maps is one record,
            // their pointwise meet, asked below a map or a union of maps; and a
            // meet below a complement is a meet sharing no value with its inner,
            // asked as the arm below asks it of any other subject.
            (Schema::Intersection(members), _) => self
                .meet_below(other, members, cx)
                .or_else(|| finite_meet_below(members, other, cx.oracle))
                .or_else(|| self.a_meet_read_whole(members, other, cx))
                .or_else(|| narrowed_meet_below(members, other, cx)),
            // A table of constants is a union of literals, and holds no kind of
            // infinitely many values whole, which the branches read alone
            // cannot say; asked where the branches have declined.
            (_, Schema::Union(members)) => self
                .below_a_union(other, members, cx)
                .or_else(|| a_union_spilled_past_as_a_table(self, members, other, cx))
                .or_else(|| self.unstructured(other, cx)),
            // A refinement carrying no constraint denotes exactly its base, so
            // it is decided as its base on whichever side it sits.
            //
            // The constructors fold this away and the frontend never builds
            // one, so the node arrives only from a term built by hand -- which
            // the fuzz target does. Without these two arms the two spellings
            // would be decided differently: `Literal[1]` is below the
            // complement of an empty sequence, and would not be below the same
            // complement wrapped in a constraint-free refinement, although the
            // wrapper narrows nothing. A rule that reads the wrapper and not
            // what it says is a rule about the spelling.
            (Schema::Refine { base, constraints }, _) if constraints.is_empty() => {
                base.is_subtype_rec(other, cx)
            }
            (_, Schema::Refine { base, constraints }) if constraints.is_empty() => {
                self.is_subtype_rec(base, cx)
            }
            // Unfold a recursive reference -- after the lattice rules, so an
            // intersection or union meeting a reference decomposes first, which
            // lets a recursive member be compared against the reference rather
            // than the reference being unfolded past it. A union member reaches
            // this arm on its own; a meet the member rule does not place reaches
            // the same reading from its arm above.
            //
            // A reference the table does not hold names a set this query cannot
            // read, so no rule about the subject has anything to say -- and the
            // supertype's rule still does. A reference on the right denotes
            // exactly its definition, so a subject below the definition is below
            // the reference whatever the subject is: every set is below the
            // universe, one nobody can read included. A reference that resolves
            // reaches that rule through its definition's own descent, so it is
            // asked here only where the unfolding has no definition to descend
            // into, and a pair of declines is not decided twice.
            (Schema::Ref(id), _) if cx.defs.get(id.get()).is_none() => {
                self.below_a_reference(other, cx)
            }
            (Schema::Ref(_), _) => self.left_reduces_below(other, cx),
            (_, Schema::Ref(_)) => self.below_a_reference(other, cx),
            // Set and frozenset inclusion reduces to element inclusion: a
            // powerset of finite sets is inside another exactly where its
            // element is [FCB08, Lem. 6.30].
            (
                Schema::Coll {
                    container: a_kind,
                    element: a,
                },
                Schema::Coll {
                    container: b_kind,
                    element: b,
                },
            ) if a_kind == b_kind => {
                #[cfg(test)]
                let _crossing = goal_tests::goals::crossing();
                a.is_subtype_rec(b, cx)
            }
            // Same-kind sequence inclusion is language inclusion on the shapes.
            (
                Schema::Seq {
                    container: ka,
                    shape: sa,
                },
                Schema::Seq {
                    container: kb,
                    shape: sb,
                },
            ) if ka == kb => sa.shape_subtype(sb, cx),
            // Record and mapping inclusion.
            (
                Schema::KeyedMap {
                    fields: fa,
                    defaults: da,
                },
                Schema::KeyedMap {
                    fields: fb,
                    defaults: db,
                },
            ) => keyed_map_subtype(fa, da, fb, db, cx),
            // Record inclusion on attributes: width and depth, no class in it.
            // The nominal half of a dataclass is a separate conjunct, and the
            // lattice rules below relate the meet to the meet.
            (Schema::AttrRecord { fields: fa }, Schema::AttrRecord { fields: fb }) => {
                attr_record_subtype(fa, fb, cx)
            }
            // Complement is contravariant: ¬A ⊆ ¬B exactly when B ⊆ A. That
            // puts `B` in the subject, where only the rules read its emptiness,
            // so where they decline an empty `B` settles the pair as it does in
            // the arm below, read by the set representation alone: the
            // descriptor route lowers `¬A` first, and a predicate or a class in
            // `A` leaves it nothing to lower.
            (Schema::Complement(a), Schema::Complement(b)) => b
                .is_subtype_rec(a, cx)
                .or_else(|| Relation::proven(cx.denotes_no_value(b))),
            // `A ⊆ ¬B` is disjointness [FCB08, §4.1], which the meet `A ∧ B`
            // answers both ways: empty, the pair holds; inhabited, a value of
            // the meet is a value of `A` inside `B`, so outside `¬B`, and the
            // pair is refuted. `B` stands at a positive position of the meet,
            // and where a reference is being resolved the meet reads it there
            // as empty, a subset of the set it denotes, so a value it finds is
            // a value of the real meet. The rules are sound and incomplete,
            // so a meet they leave open is neither, and the opposite proof
            // is asked: `A ⊆ B` holding puts every value of `A` in `B`, which
            // needs `A` to have one, which the reading around this settles.
            //
            // Where both decline, an empty `B` still settles it: `¬B` is the
            // universe. The rules cannot empty a pattern beside a length, and
            // the set representation can, reading `B` alone -- where the
            // subject may have no descriptor at all, as a literal the oracle
            // cannot place has none.
            (_, Schema::Complement(inner)) => {
                match self.verdict_beside(inner, cx) {
                    Verdict::Empty => return Relation::Holds,
                    Verdict::Inhabited => return Relation::Fails,
                    Verdict::Unknown => {}
                }
                if self.is_subtype_rec(inner, cx).holds() {
                    Relation::of_mismatch(self.verdict_rec(cx))
                } else if self.spills_a_kind(other, cx) {
                    Relation::Fails
                } else if let Schema::Instance(_) = self
                    && let Some(below) = cx.oracle.leaf_subtype(self, other)
                {
                    // A class the bindings read as its values -- an enumeration
                    // that is the union of its members -- is below the
                    // complement exactly where each value is, and the meet the
                    // rule above asks reads the class as an atom. The union arm
                    // asks the same of a union supertype.
                    Relation::decided(below)
                } else {
                    Relation::proven(cx.denotes_no_value(inner))
                }
            }
            (
                Schema::Refine {
                    base: narrow_base,
                    constraints: narrow_cons,
                },
                Schema::Refine {
                    base: wide_base,
                    constraints: wide_cons,
                },
            ) => refinement_subtype(narrow_base, narrow_cons, wide_base, wide_cons, cx)
                // The constraint rule only proves, so a pair it leaves unproven is
                // handed to the reading every pair with no rule gets, rather than
                // ending the match here. A refinement against a *non*-refinement
                // already reaches that reading, and it is written for a refinement
                // supertype -- so without this a refinement on both sides would be
                // the one pair kept from it, and a list refinement against an
                // integer one would walk the sets to learn that a list is not an
                // integer.
                .or_else(|| self.unstructured(other, cx))
                // And the refutation that reads the two constraint lists together.
                .or_else(|| refuted_by_a_multiple(narrow_base, narrow_cons, wide_cons, cx.oracle)),
            // Against a non-refinement, a refinement inherits its base's
            // supertypes -- and inherits nothing else, so a pair this leaves
            // unproven is a pair with no rule of its own.
            (Schema::Refine { .. }, _) => self
                .left_reduces_below(other, cx)
                .or_else(|| self.unstructured(other, cx)),
            (_, Schema::Literal(_)) if spills_past_a_table(self, other, cx) => Relation::Fails,
            _ => self.unstructured(other, cx),
        }
    }

    /// An empty subject below `B` [BBC+26, (uninhabited)], asked where the rules
    /// declined.
    ///
    /// The remaining lattice bound, and the one that costs a walk: whether the
    /// *subject* is empty. It is asked last rather than first because a proof
    /// stands without it and a refutation is read against that same emptiness
    /// by [`witnessed`](Self::witnessed) on the way out -- so asking first
    /// would walk the subject on every pair the rules are about to decide
    /// anyway, which is a sixth of the repeating workload.
    ///
    /// The answer is identical either way. Both readings are sound, and an
    /// empty subject is below everything whichever of them says so.
    fn or_bounded(&self, answer: Relation, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        match answer {
            Relation::Unknown if self.bounds_the_pair(other, cx) => Relation::Holds,
            answer => answer,
        }
    }

    /// What a pair no structural rule decided is worth, asked in one place.
    ///
    /// Four readings, in this order. Five **refutations**, each naming a
    /// value of the subject outside the supertype: the empty sequence below a
    /// length bound (`shorter_than`), a region the subject reaches past the
    /// supertype's (`spills_past`), a kind whose builtin the supertype's class
    /// does not derive from (`outside_the_class`), a kind the subject holds
    /// whole and the supertype holds none of (`spills_a_kind`), and two
    /// schemas that share no value (`disjoint_with`) -- the cheap disjointness, two discriminants
    /// that cannot overlap, a list beside a tuple, a set beside a mapping. Then
    /// the **oracle**, the only reader of a class or a constant, whose `None`
    /// is the decline it says it is. Then the supertype's own shape: a
    /// **refinement** is a subset of its base, so a subject outside the base is
    /// outside the refinement -- the value that refutes the one refutes the
    /// other, and it is the same value. Its proof carries only where the
    /// subject's own lengths entail every constraint
    /// ([`lengths_held`](Self::lengths_held)), since being inside the base says
    /// nothing else about the constraints. Last, what a **length** settles
    /// ([`read_by_length`](Self::read_by_length)): a refinement whose length is
    /// at most zero is the empty container, a sequence or a set shape holds one
    /// longer than a supertype's maximum, and a record's required fields alone
    /// may be a length it refuses. Last because the pairs it reads are
    /// rare and every pair before them would pay for asking first: on the
    /// matrix workload, asked beside `shorter_than`, it cost 0.95% of the
    /// instructions, and asked here, 0.27%.
    ///
    /// The oracle is asked before the refinement because it *proves* where the
    /// refinement reading only refutes: a literal below a predicate refinement
    /// is settled by running the predicate on the constant, and a reading that
    /// answered first would take that proof away.
    ///
    /// Every arm above that answers for a shape and then *declines* ends here
    /// rather than ending the match, because a decline is not an answer: such a
    /// pair has had no rule, and this is what a pair with no rule is worth.
    ///
    /// **The five refutations are ordered by cost, which is free to choose.**
    /// Each answers `Fails` on its own and none of them proves, so a pair any
    /// two of them refute is refuted whichever ran first -- the order decides
    /// which reading names the value, never whether one is named. What it does
    /// decide is what an *unrefuted* pair pays, and that pair pays all five.
    /// Four of them are a discriminant test and an oracle call for every
    /// subject but the universe and a complement, which `spills_a_kind` reads
    /// a kind at a time; `disjoint_with` walks both subtrees and reads the kind
    /// tags, so it is asked last and the pairs the cheap four refute never
    /// reach it.
    fn unstructured(&self, other: &Schema, cx: &mut Cx<'_>) -> Relation {
        if self.shorter_than(other) {
            return Relation::Fails;
        }
        if self.spills_past(other, cx.oracle) {
            return Relation::Fails;
        }
        if self.outside_the_class(other, cx.oracle) {
            return Relation::Fails;
        }
        if self.spills_a_kind(other, cx) {
            return Relation::Fails;
        }
        if self.disjoint_with(other, cx.oracle) {
            return Relation::Fails;
        }
        // A proof walks a literal's constant, which speaks for the literal only
        // where nothing the supertype asks tells two equal values apart; a
        // proof it cannot speak for is no answer, and the pair reads as one
        // the oracle declined.
        let leaf = cx
            .oracle
            .leaf_subtype(self, other)
            .filter(|&holds| match self {
                Schema::Literal(constant) if holds => {
                    the_constant_speaks_for_the_literal(*constant, other, cx.oracle, cx.defs)
                }
                _ => true,
            });
        match leaf {
            Some(true) => Relation::Holds,
            Some(false) => Relation::Fails,
            None => match other {
                // The proof carries where the subject's own lengths entail every
                // bound ([`Schema::lengths_entail`]): a closed record below a
                // mapping, and a set below a set, are below the refinement
                // wherever their lengths are its lengths. Only those two shapes
                // hold lengths, so every other subject pays the one test.
                Schema::Refine { base, constraints } => {
                    let below = self.is_subtype_rec(base, cx);
                    if matches!(self, Schema::Coll { .. } | Schema::KeyedMap { .. })
                        && self.lengths_entail(constraints, cx.oracle)
                    {
                        below
                    } else {
                        below.refutation_only()
                    }
                }
                _ => Relation::Unknown,
            }
            // Last, and out of line behind the subject's discriminant: what a
            // length settles is read only of a pair everything above left open.
            .or_else(|| {
                if matches!(
                    self,
                    Schema::Refine { .. }
                        | Schema::Str
                        | Schema::Bytes
                        | Schema::Seq { .. }
                        | Schema::Coll { .. }
                        | Schema::KeyedMap { .. }
                ) {
                    self.read_by_length(other, cx)
                } else {
                    Relation::Unknown
                }
            }),
        }
    }

    /// Whether `self` and `other` denote the same set — mutual inclusion
    /// [Cas24, Def. 4].
    ///
    /// Like the relations it composes, the decision is bounded; a `false` can mean
    /// "not proven equivalent within the work bound" for an adversarial schema.
    #[must_use]
    pub fn is_equivalent(&self, other: &Schema) -> bool {
        self.is_equivalent_under(other, &NoLeafRelations, &[])
    }

    /// [`is_equivalent`](Self::is_equivalent) under a [`LeafRelations`] oracle and
    /// the recursive definitions.
    #[must_use]
    pub fn is_equivalent_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> bool {
        // Both inclusion directions share one budget, so equivalence cannot spend
        // twice the ceiling, and its verdict does not depend on which direction
        // happened to allocate a fresh allowance first.
        let met = bounds_met();
        let budget = Budget::new(DECISION_BUDGET);
        let first = self.equivalence_relation(other, oracle, defs, &budget);
        if first == Relation::Unknown && asks_again(met, &budget) {
            return asked_again(|budget| self.equivalence_relation(other, oracle, defs, budget))
                .holds();
        }
        first.holds()
    }

    /// The three-valued form of
    /// [`is_equivalent_under`](Self::is_equivalent_under), on the caller's
    /// budget.
    ///
    /// Equivalence is the meet of two inclusions [Cas24, Def. 4], taken in the
    /// vocabulary both of them answer in: mutual inclusion, and the
    /// conjunction's own short circuit, rather than two booleans that have each
    /// forgotten whether they were refuted or merely unproven. The public form
    /// is `holds()` of this, which reads a decline as "not proven" -- the
    /// conservative answer.
    ///
    /// The budget is a parameter, as it is on the rules' own entry the tests
    /// drive: a cell the function allocates for itself is a path no test can
    /// drive at the one value that matters, and "the budget declines, it never
    /// refutes" is a claim about this path too.
    pub(crate) fn equivalence_relation(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        budget: &Budget,
    ) -> Relation {
        let mut cx = Cx::new(oracle, defs, budget);
        let mut within = |sub: &Schema, sup: &Schema| {
            sub.is_subtype_rec(sup, &mut cx)
                .or_else(|| sub.decided_by_sets(sup, &mut cx, true))
        };
        within(self, other).and(|| within(other, self))
    }
}

/// What one decision query threads through both of its recursions: the
/// leaf-relation oracle, the definitions that resolve references, the budget,
/// and the two stacks of hypotheses the recursions hold open.
///
/// One per query, built where the query starts and handed down by `&mut`, so
/// every rule takes one parameter for all of it and a call passes a pointer.
/// The two directions of an equivalence share one, as they share the budget.
/// The meets the emptiness recursion unfolds a reference into are a third
/// stack, kept off it on a list `emptiness.rs` keys by the budget, for the
/// cost that list's comment gives.
///
/// **Each stack is a path's, and is given back as it was.** A frame that
/// pushes pops before it returns, so a stack holds exactly the hypotheses open
/// above the frame reading it. The goal recursion pushes the trail and nothing
/// else, and the emptiness recursion pushes `visiting`.
///
/// **A goal asked under `visiting` reads it at positive positions only.** The
/// emptiness recursion asks goals where it subtracts records from a record
/// (`a_difference_of_sets` in `emptiness.rs`), so a goal can meet a reference
/// being resolved. The hypothesis is read in one place, the reference arm of
/// the emptiness recursion, and never under a complement, whose arm reads its
/// inner's region alone; every verdict a goal reads is of its subject or of a
/// meet with its supertype's complement, positive in `S ∧ ¬T`. The one meet a
/// goal builds with a part of its supertype itself -- a branch, a component, a
/// slot the subject shares no value with -- reads that part at a negative
/// position, and is asked only where no definition is being resolved
/// ([`Schema::shares_no_value_with_a_part_of_the_supertype`]). So an answer
/// under the hypothesis is an answer about the schema with that reference's
/// positive occurrences read as empty: a proof of the smaller set, and a
/// refutation standing on one of its values, which is a value of the real one.
/// The definition's emptiness is that smaller schema's, since a value of least
/// height needs no value of the definition at a positive position below it, and
/// nothing decided under the hypothesis outlives the frame that pushed it.
pub(super) struct Cx<'a> {
    oracle: &'a dyn LeafRelations,
    defs: &'a [Schema],
    budget: &'a Budget,
    /// The `(subject, supertype)` goals the goal recursion is unfolding a
    /// reference under, innermost last: a goal that comes back is answered by
    /// hypothesis, which is the greatest fixpoint read coinductively.
    trail: Vec<(Schema, Schema)>,
    /// The definitions the emptiness recursion is resolving, innermost last: a
    /// reference reached again is read as empty, the least fixpoint read
    /// inductively.
    visiting: Vec<DefIx>,
    /// The number this query's set readings are memoized under, given at the
    /// first one; zero until then ([`Readings`]).
    query: Cell<u64>,
    /// Whether a goal the rules decline is read by the sets, where no reference
    /// stands in the way ([`Schema::decided_by_its_leaves`]).
    reads_leaves: bool,
}

/// What [`HELD`] keeps of the rules' answers at a query's root: the query's
/// number, the schema on the other side of the pair, the members, the members
/// the subject was placed below and the branches it was refuted below, one bit
/// a member each.
type Held = (u64, Schema, Members, u64, u64);

/// The widest meet whose members' answers at the root are kept ([`HELD`]), one
/// bit a member: past it the parts reading asks the rules again.
const MEMBERS_HELD: usize = u64::BITS as usize;

/// The answers the set representation gave the query running on this thread,
/// by the schemas asked and by the oracle and definitions they were read under.
///
/// A reading is a function of its schemas, the pool and the definitions, and
/// each build within it arms an allowance of its own and gives the enclosing
/// one back untouched ([`budget::under`]), so an answer read back here is the
/// answer a reading again would give: the memo changes no answer and no
/// allowance, only how often a build runs. A query asks one reading of every
/// goal its rules decline, and the goals repeat one meet many times over -- a
/// relation between two records nested past the depth the sets descend asks a
/// hundred and thirty-six readings of one meet -- so a reading is built once a
/// query rather than once a goal.
///
/// Each answer rests on no open hypothesis, which is what a memo inside the
/// coinductive procedure owes (`docs/dev/10-theory.md`, "A cache under
/// coinduction is revertible or absent"): a reading takes the pool and the definitions and
/// never the trail or the definitions being resolved, so no assumption a
/// failed disjunct abandons is in what it answers.
///
/// **Held beside the query, not on its context.** A query whose rules answer
/// is a few hundred instructions on the decision workload, and one pointer on
/// the context, never allocated, read 5.4% more there: building and dropping
/// it is that query's cost. A thread's queries run one at a time, so the memo
/// is the thread's, and an answer is held under the number of the query that
/// read it: a query is numbered at its first reading, a number is never given
/// twice on a thread, and a part of a query read under the same sets shares
/// its number, so an answer is read back by the query that read it and by no
/// other. A query a callback or a reading starts inside another has a number
/// of its own and leaves the outer one's answers where they are. Nothing is
/// asked as a query starts, so a query that reads nothing pays one word; the
/// memo is emptied when it holds [`READINGS_HELD`] answers, which bounds what
/// the finished queries' answers occupy and changes no answer. What is held is the answer and never a descriptor: a
/// descriptor of a few hundred atoms is what a reading builds to throw away.
///
/// Only emptiness is held. It is the reading the goals repeat -- the meet of a
/// record's kept members, the inner set of a complement -- while a pair placed
/// by the sets is asked once a query, where a memo is a hash and a copy of two
/// schemas for nothing.
type Readings = FxHashMap<(u64, Schema), bool>;

/// The most answers the memo holds before it is emptied.
///
/// A query that repeats a reading repeats it within a few hundred goals, and
/// the slowest relations of the record families hold a few dozen answers at
/// once; the bound is what keeps the finished queries' answers from piling up
/// on a thread that asks millions.
const READINGS_HELD: usize = 4096;

thread_local! {
    /// The readings of the queries on this thread, by query number.
    static READINGS: RefCell<Readings> = RefCell::new(Readings::default());
    /// The query whose narrowed meet is being asked on this thread, or zero
    /// where none is ([`narrowed_meet_below`]). Held beside the query for the
    /// reason [`Readings`] is, and by number, so a query a callback starts
    /// inside the question asks its own narrowed meets as a query of its own.
    static NARROWING: Cell<u64> = const { Cell::new(0) };
    /// The number the next query to read on this thread is given.
    static QUERIES: Cell<u64> = const { Cell::new(0) };
    /// The meet supertype the rules last declined at the root of a query on
    /// this thread ([`Schema::below_each_member_at_the_root`]): the query's
    /// number, the subject, the members, and the members the subject was
    /// placed below, one bit each; or the union subject they last declined
    /// there ([`Schema::each_member_below_at_the_root`]): the number, the
    /// supertype, the union's members, and the members placed below it; or
    /// the union supertype a branch of which they refuted there
    /// ([`Schema::each_branch_at_the_root`]): the number, the subject, the
    /// branches, and the branches refuted, in a field of their own. The root
    /// pair is one of the three, and each is read back by the pair it names;
    /// read by a pair it does not name, as the reverse direction of an
    /// equivalence can, a record names no member placed and no branch refuted,
    /// which asks the set representation as no record would. Held beside the
    /// query for the reason
    /// [`Readings`] is, and by number, so a query reads back only what it
    /// kept, and the members' storage is held, so no other meet's members are
    /// at that address while the record stands.
    static HELD: RefCell<Option<Held>> = const { RefCell::new(None) };
    /// The answers of the narrowed meets the queries on this thread asked
    /// outside another's question ([`narrowed_meet_below`]): by the query's
    /// number, the address of the meet's members and the allowance the
    /// question ran under, each supertype asked beside its answer. Held beside
    /// the query for the reason [`Readings`] is, and by number; the members'
    /// storage is held, so no other meet's members are at that address while
    /// the answer stands; and emptied at [`READINGS_HELD`] entries.
    static ASKED_NARROWED: RefCell<AskedNarrowed> = RefCell::new(AskedNarrowed::default());
}

/// The table of [`ASKED_NARROWED`].
type AskedNarrowed = FxHashMap<(u64, usize, u32), Vec<(Members, Schema, Relation)>>;

impl<'a> Cx<'a> {
    /// The context a query starts with: nothing assumed and nothing resolved.
    pub(super) fn new(
        oracle: &'a dyn LeafRelations,
        defs: &'a [Schema],
        budget: &'a Budget,
    ) -> Self {
        Self {
            oracle,
            defs,
            budget,
            trail: Vec::new(),
            visiting: Vec::new(),
            query: Cell::new(0),
            reads_leaves: false,
        }
    }

    /// The members of `members` the rules placed `subject` below at the root of
    /// this query, one bit each, where they declined that meet there; `None`
    /// where they did not ([`HELD`]).
    ///
    /// The record is taken as it is read: a query reads it once, after the
    /// rules, and nothing a finished query kept is left to occupy the thread.
    fn held_below(&self, subject: &Schema, members: &Members) -> Option<u64> {
        let query = self.query.get();
        HELD.with_borrow_mut(Option::take)
            .and_then(|(at, of, kept, placed, _)| {
                (at == query && Arc::ptr_eq(&kept, members) && of == *subject).then_some(placed)
            })
    }

    /// The branches of `members` the rules refuted `subject` below at the root
    /// of this query, one bit each; none where they kept no such record
    /// ([`HELD`]). Taken as it is read, as [`held_below`](Self::held_below)
    /// takes it.
    fn refuted_at_the_root(&self, subject: &Schema, members: &Members) -> u64 {
        let query = self.query.get();
        HELD.with_borrow_mut(Option::take)
            .filter(|(at, of, kept, _, _)| {
                *at == query && Arc::ptr_eq(kept, members) && of == subject
            })
            .map_or(0, |(_, _, _, _, refuted)| refuted)
    }

    /// A context for a part of this query, which reads the same sets and so
    /// shares its memo.
    fn within_this_query(&self, budget: &'a Budget) -> Self {
        let cx = Self::new(self.oracle, self.defs, budget);
        cx.query.set(self.query());
        cx
    }

    /// This query's number, given at its first reading.
    fn query(&self) -> u64 {
        if self.query.get() == 0 {
            self.query.set(QUERIES.with(|queries| {
                let next = queries.get() + 1;
                queries.set(next);
                next
            }));
        }
        self.query.get()
    }

    /// Whether the set representation proves `schema` holds no value, read once
    /// a query ([`Readings`]).
    fn denotes_no_value(&self, schema: &Schema) -> bool {
        let key = (self.query(), schema.clone());
        if let Some(known) = READINGS.with(|readings| readings.borrow().get(&key).copied()) {
            return known;
        }
        let answer = schema.denotes_no_value(self.oracle, self.defs);
        READINGS.with(|readings| {
            let mut readings = readings.borrow_mut();
            if readings.len() >= READINGS_HELD {
                // Emptied in place: the table grown is the one refilled.
                readings.clear();
            }
            readings.insert(key, answer);
        });
        answer
    }

    /// Decide `goal` with the goal itself on the trail, and take it off after.
    ///
    /// The trail is Amadio & Cardelli's, a set of pairs assumed while they are
    /// decided [AC93, §1.5], held as a stack of terms rather than of addresses
    /// [AC93, §4.4] and popped rather than kept.
    ///
    /// The two places a reference is unfolded assume here, and the test-side
    /// recorder is told how deep the trail stands, so the longest trail a query
    /// builds is a number a test reads rather than a sentence a page keeps.
    fn assuming(
        &mut self,
        goal: (Schema, Schema),
        decide: impl FnOnce(&mut Self) -> Relation,
    ) -> Relation {
        #[cfg(test)]
        let _open = goal_tests::goals::held_open(&goal.0, &goal.1);
        self.trail.push(goal);
        #[cfg(test)]
        goal_tests::goals::trail(self.trail.len());
        let answer = decide(self);
        self.trail.pop();
        answer
    }
}

/// Whether some two members are provably disjoint (distinct concrete kinds,
/// `bool ⊆ int` aside), so their intersection is empty: positives of two kinds
/// meet in nothing [FCB08, (5)]. This decides the structural-kind disjointness
/// (a list is never a set) the scalar region bitset cannot see.
fn has_disjoint_pair(members: &[Schema], oracle: &dyn LeafRelations) -> bool {
    unordered_pairs(members).any(|(a, b)| a.disjoint_with(b, oracle))
}

/// A refinement below another: the base narrows and every constraint holds.
///
/// A refinement is a subset of its base. Against another refinement the base
/// must subtype and every constraint of the supertype must hold of every
/// subtype value, Sub-Base's entailment [JV20, Fig. 3.4] without a solver:
/// either it appears verbatim, or it is entailed by the subtype's bounds (a
/// tighter lower, upper or length bound entails a looser one, decided through
/// the ordering oracle), or a length bound is entailed by the lengths a set or
/// a record base keeps to without writing them
/// ([`lengths_entail`](Schema::lengths_entail)). A bound the oracle cannot
/// compare and a non-order constraint stay on the verbatim path, so a
/// constraint neither written nor entailed leaves the pair unproven rather
/// than refuted.
fn refinement_subtype(
    narrow_base: &Schema,
    narrow_cons: &[Constraint],
    wide_base: &Schema,
    wide_cons: &[Constraint],
    cx: &mut Cx<'_>,
) -> Relation {
    narrow_base
        .is_subtype_rec(wide_base, cx)
        .proof_only()
        .and(|| {
            Relation::proven(wide_cons.iter().all(|constraint| {
                narrow_cons.contains(constraint)
                    || constraint_entailed(constraint, narrow_base, narrow_cons, cx.oracle)
                    || (matches!(narrow_base, Schema::Coll { .. } | Schema::KeyedMap { .. })
                        && narrow_base.lengths_entail(core::slice::from_ref(constraint), cx.oracle))
            }))
        })
}

#[cfg(test)]
mod budget_tests;

#[cfg(test)]
mod goal_tests;

#[cfg(test)]
mod tests;
