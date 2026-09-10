//! Lowering a schema to a descriptor.
//!
//! A schema is a *term*: a tree the walk reads and the printer renders. A
//! descriptor is a *set*. Lowering is the map between them, and the two live
//! side by side rather than one replacing the other -- the walk still decides
//! membership, and the descriptor is what the algebra reasons in.
//!
//! **The map is partial, and the type says so.** A form the descriptor cannot
//! yet hold lowers to `None` rather than to something close: a descriptor that
//! stood for a schema it does not denote would be complemented into one that is
//! wrong the other way, and a refusal is the only sound answer to give.
//!
//! The core holds no Python objects, so the constants a schema names are indices
//! into a pool the bindings keep. [`Constants`] is the way in: the caller reads
//! the pool and answers what a comparison operand or a literal carries.

use super::budget;
use super::classes::Class;
use super::floats::FloatSet;
use super::maps::{KEY_KINDS, Label};
use super::{BoolSet, Descr, integers::IntSet};
use crate::decision::Kind;
use crate::ir::{
    ClassIx, CollKind, ConstIx, Constraint, Field, MapClause, OperandIx, Schema, SeqKind, SeqShape,
};
use std::cell::Cell;

/// A pooled value, as far as a descriptor can read one.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// One of the two booleans.
    Boolean(bool),
    /// An integer, which is never a boolean: `bool` is its own kind here.
    Integer(i64),
    /// A float, `nan` included.
    Float(f64),
    /// A word, under the kind that reads it -- a `str` as its UTF-8 bytes.
    Word(Vec<u8>, Kind),
    /// `None`.
    NoneType,
    /// An instance of a pure class.
    Instance(Class),
}

/// What a lowering needs from the pool the core does not hold.
///
/// The core stays free of Python objects, so a schema names its constants by
/// index. Answering `None` is the honest reading of an operand this cannot see
/// -- an impure class, a callable, an object with a custom `__eq__` -- and it
/// refuses the lowering rather than guessing.
pub trait Constants {
    /// What a comparison or `MultipleOf` operand carries.
    ///
    /// Every question here defaults to `None`, the answer of a pool that cannot
    /// see this one. That is the conservative direction -- it refuses the
    /// lowering rather than describing a set the schema does not denote -- and it
    /// is what lets an oracle that holds no objects be written as an empty impl
    /// instead of three declines.
    fn operand(&self, _index: OperandIx) -> Option<Operand> {
        None
    }

    /// What a `Literal` names.
    fn constant(&self, _index: ConstIx) -> Option<Operand> {
        None
    }

    /// The class an `Instance` atom names, as the order snapshot the core reads.
    ///
    /// Only the bindings can walk an `__mro__`, and only they can tell a **pure**
    /// class from one whose metaclass answers `isinstance` by running code. A
    /// hooked class is not a set this algebra holds, so answering `None` for one
    /// refuses the lowering rather than describing a set the class does not have.
    fn class(&self, _index: ClassIx) -> Option<Class> {
        None
    }
}

/// A pool that knows nothing, for a caller that has none.
///
/// The core's own relations have no object table -- the constants live in the
/// bindings -- so a schema naming one refuses to lower here rather than being
/// guessed at. What still lowers is everything whose meaning is structural: the
/// kinds, the sequences, the sets, the three operations, and the length and
/// pattern constraints, which carry their operand inline.
pub struct NoConstants;

impl Constants for NoConstants {}

/// The nodes this will lower before refusing.
///
/// A schema is a tree the caller writes, and lowering both recurses over it and
/// *builds* at every node -- a sequence node determinises an automaton, a set
/// node takes a powerset. Without a bound a deep enough schema exhausts the
/// stack and a wide one exhausts the clock, which is why the procedure beside
/// this one carries a work budget too.
///
/// A refusal past the bound is the same refusal as for a form with nowhere to
/// land, and it reaches the same place: the caller decides the old way. It is a
/// bound on the *lowering*, not on the algebra -- the components have their own,
/// and those are limits of what they can represent rather than of what they will
/// spend.
///
/// Sized by what can be *built* rather than by what can be parsed. Complementing
/// a descriptor complements the guards inside it, so a chain of complements
/// deepens the descriptor as well as the schema, and the operations recurse
/// through that nesting: past roughly a hundred the stack goes rather than the
/// clock. An annotation anyone writes is orders of magnitude inside this.
///
/// **This bound is debt.** It counts work rather than limiting what the
/// representation can hold, and it is here for the reason the decision's own
/// budget is: a lowering repeats itself over structurally equal subtrees.
/// Construction shares those subtrees where they are built alike
/// (`ir/intern.rs`), which is what gives the memo that would replace this
/// ceiling a cheap key; the memo is the half not written.
pub const BUDGET: u32 = 64;

/// The set a schema denotes, or `None` where the descriptor cannot yet hold it.
///
/// # Errors
///
/// Refuses rather than approximating. The forms it refuses are the ones with no
/// component to land in -- a dict, an attribute record beside a builtin kind, a
/// recursive reference, the gradual `Any` -- and the ones whose operand the pool
/// could not read.
pub fn lower(schema: &Schema, pool: &dyn Constants) -> Option<Descr> {
    lower_within(Bounds::DEFAULT, schema, pool)
}

/// What a lowering may spend, in the three quantities it can run out of.
///
/// They are three because a build can be too big in three ways, and no one of
/// them implies another: a schema of a dozen nodes can nest ten deep, a shallow
/// one can be a hundred wide, and either can spend a third of a second inside a
/// product and then refuse anyway. Named together so a caller measuring one can
/// lift the other two, which is what the benchmark behind these numbers does --
/// a bound whose figure lives only in a comment cannot be re-derived on another
/// machine, and cannot fail when the shape it guards against changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    /// The schema nodes a lowering will read.
    pub nodes: u32,
    /// The schema nesting a lowering will descend.
    pub depth: u32,
    /// The units of multiplying work a build may spend.
    pub work: u64,
}

impl Bounds {
    /// The bounds every relation lowers under. See the three constants for what
    /// each number is measured from.
    pub const DEFAULT: Bounds = Bounds {
        nodes: BUDGET,
        depth: DEPTH,
        work: WORK,
    };

    /// Bounds no build reaches, for measuring what one costs unheld.
    ///
    /// Not a mode the library runs in: an unheld build is the thing the three
    /// numbers above exist to prevent, and this exists so a benchmark can show
    /// what they prevent.
    pub const UNHELD: Bounds = Bounds {
        nodes: u32::MAX,
        depth: u32::MAX,
        work: u64::MAX,
    };
}

/// The work a build may spend before it refuses.
///
/// [`BUDGET`] bounds the schema this reads; this bounds what reading it costs,
/// and the two are different quantities. A schema of a dozen nodes can spend a
/// third of a second and then refuse anyway, because the bounds on a descriptor
/// say how large a result may be and say nothing about the work of reaching one.
///
/// Sized from the shapes on both sides of the question. The differences the
/// descriptor decides and the structural rules do not -- a container meet, a
/// double complement, one regular language inside another -- each spend under
/// 256 units. The shapes that blow up spend tens or hundreds of thousands: a
/// record nested eight deep spends 22,806, and a union of four records nested
/// three deep minus a union of its siblings spends 170,597.
///
/// On the shapes reachable today [`DEPTH`] refuses first, so this number is
/// rarely what stops a build -- `lower_sibling_union_difference_held` in
/// `benches/core.rs` is refused for nesting, not for work. It is the bound that
/// remains when a shape is shallow and wide, which is the one nesting cannot
/// catch, and it is kept for that.
///
/// **This bound is debt.** It counts work rather than limiting what the
/// representation can hold, and it is here for the reason the decision's own
/// budget is: a lowering repeats itself over structurally equal subtrees.
/// Construction shares those subtrees where they are built alike
/// (`ir/intern.rs`), which is what gives the memo that would replace this
/// ceiling a cheap key; the memo is the half not written.
pub const WORK: u64 = 1024;

/// The schema nesting this will descend before refusing.
///
/// [`WORK`] bounds what a build spends once it has started, and it cannot bound
/// what starting costs: a lowering builds an automaton at every sequence node
/// and a powerset at every set node, and none of that is a product to charge
/// for. On the shapes a relation is asked about, that floor is around a hundred
/// times what the structural rules spend answering the whole question -- so a
/// build that will not pay for itself has to be refused *before* it is walked,
/// and depth is what says which.
///
/// Nesting is the exponential, and `benches/core.rs` measures it:
/// `lower_nested_records_depth{0,2,4,6}`, run under [`Bounds::UNHELD`], grows
/// 1.7 microseconds, 198 microseconds, 1.5 milliseconds, 7.2 milliseconds.
/// Breadth is not the exponential and is bounded by [`BUDGET`] instead.
///
/// Set from both sides. Every relation the descriptor decides and the rules do
/// not nests five deep or less -- `lower_container_meet`,
/// `lower_double_complement`, `lower_regular_language` and `lower_step_divides`
/// are those four, and each builds in 49 to 188 microseconds. The shapes that
/// blow up nest ten and deeper. `lower_sibling_union_difference_{unheld,held}`
/// is the pair that shows what the bound buys: 9.0 milliseconds against 1.35
/// microseconds, on one shape, from this number alone.
///
/// **This bound is debt.** It counts work rather than limiting what the
/// representation can hold, and it is here for the reason the decision's own
/// budget is: a lowering repeats itself over structurally equal subtrees.
/// Construction shares those subtrees where they are built alike
/// (`ir/intern.rs`), which is what gives the memo that would replace this
/// ceiling a cheap key; the memo is the half not written.
pub const DEPTH: u32 = 5;

/// [`lower`] under explicit bounds, for a caller measuring one of them.
///
/// # Errors
///
/// Refuses what [`lower`] refuses, and also a build that would exceed any of
/// `bounds`. That refusal is not a statement about the schema: the same schema
/// lowers under larger bounds. It is what makes the cost of asking bounded,
/// which is the only thing that makes asking safe.
pub fn lower_within(bounds: Bounds, schema: &Schema, pool: &dyn Constants) -> Option<Descr> {
    budget::under(bounds.work, || {
        descend(schema, pool, &Cell::new(bounds.nodes), bounds.depth)
    })
}

/// [`lower`] with the nodes left to spend, which every node spends one of.
fn descend(schema: &Schema, pool: &dyn Constants, budget: &Cell<u32>, depth: u32) -> Option<Descr> {
    budget.set(budget.get().checked_sub(1)?);
    let depth = depth.checked_sub(1)?;
    match schema {
        Schema::Anything(_) => Some(Descr::anything()),
        Schema::Nothing => Some(Descr::nothing()),
        Schema::NoneType => Some(Descr::of_kind(Kind::NoneType)),
        Schema::Bool => Some(Descr::of_kind(Kind::Bool)),
        // `bool` subclasses `int`, so every boolean is an integer. The two are
        // separate *kinds* here, which is what makes their components
        // independent -- so the schema that denotes both spells both.
        Schema::Int => Descr::of_kind(Kind::Int).union(&Descr::of_kind(Kind::Bool)),
        Schema::Float => Some(Descr::of_kind(Kind::Float)),
        Schema::Str => Some(Descr::of_kind(Kind::Str)),
        Schema::Bytes => Some(Descr::of_kind(Kind::Bytes)),
        Schema::Literal(index) => singleton(&pool.constant(*index)?),
        Schema::Seq { container, shape } => sequence(*container, shape, pool, budget, depth),
        Schema::Coll { container, element } => {
            let kind = match container {
                CollKind::Set => Kind::Set,
                CollKind::FrozenSet => Kind::FrozenSet,
            };
            Descr::set(&descend(element, pool, budget, depth)?, kind)
        }
        Schema::Union(members) => members.iter().try_fold(Descr::nothing(), |whole, member| {
            whole.union(&descend(member, pool, budget, depth)?)
        }),
        Schema::Intersection(members) => {
            members.iter().try_fold(Descr::anything(), |whole, member| {
                whole.intersect(&descend(member, pool, budget, depth)?)
            })
        }
        Schema::Complement(inner) => Some(descend(inner, pool, budget, depth)?.complement()),
        Schema::Refine { base, constraints } => refine(base, constraints, pool, budget, depth),
        // Every field narrows the same value, so the record is their meet. Each
        // field's type is a descriptor in its own right, which is what makes the
        // attribute half recursive; a field the schema does not require admits
        // the values that do not carry it at all.
        Schema::AttrRecord { fields } => {
            fields.iter().try_fold(Descr::anything(), |whole, field| {
                let ty = descend(&field.schema, pool, budget, depth)?;
                whole.intersect(&Descr::attribute(&field.name, &ty, !field.required))
            })
        }
        Schema::KeyedMap { fields, defaults } => map(fields, defaults, pool, budget, depth),
        Schema::Instance(index) => Some(Descr::instance_of(pool.class(*index)?)),
        // A reference is a cycle a finite descriptor has no room for.
        Schema::Ref(_) | Schema::SelfRef(_) => None,
    }
}

/// The dicts a keyed map spells.
///
/// A record's field and a mapping's literal key are one thing -- each names a
/// key and says what it maps to -- so both become labels. A clause whose key is
/// a *kind* opens that part of the key partition instead. What no clause opens
/// stays shut, which is what makes a record with no catch-all closed.
fn map(
    fields: &[Field],
    defaults: &[MapClause],
    pool: &dyn Constants,
    budget: &Cell<u32>,
    depth: u32,
) -> Option<Descr> {
    let mut labels: Vec<(Label, Descr, bool)> = Vec::with_capacity(fields.len());
    for field in fields {
        let ty = descend(&field.schema, pool, budget, depth)?;
        labels.push((Label::str(&field.name), ty, !field.required));
    }
    let mut opened: Vec<(Option<Kind>, Descr)> = Vec::new();
    for clause in defaults {
        let value = descend(&clause.value, pool, budget, depth)?;
        let (named, parts) = key_cover(&clause.key, pool)?;
        for label in named {
            // A clause names a key without requiring it: `dict[Literal["a"], V]`
            // admits the dict that carries no `a` at all.
            labels.push((label, value.clone(), true));
        }
        for part in parts {
            opened.push((part, value.clone()));
        }
    }
    Descr::keyed_map(labels, opened)
}

/// The keys a clause governs: the constants it names, and the parts of the key
/// partition it covers whole.
///
/// `None` where it is neither -- a key schema covering *part* of a kind, such as
/// a regex over strings, would need the default to hold a set of keys rather
/// than a part of the partition, and that is the theory of overlapping domains
/// the paper sets aside: "it is possible to define a theory for maps with
/// overlapping domains, but in that case, there would not be any difference
/// between record types and an intersection of function types whose codomain may
/// contain an undefined value". Refusing is what keeps the default a function.
fn key_cover(key: &Schema, pool: &dyn Constants) -> Option<(Vec<Label>, Vec<Option<Kind>>)> {
    let part = |kind: Kind| Some((Vec::new(), vec![Some(kind)]));
    match key {
        Schema::Nothing => Some((Vec::new(), Vec::new())),
        // Every part, the one for a key of no listed kind included.
        Schema::Anything(_) => Some((
            Vec::new(),
            KEY_KINDS.iter().copied().map(Some).chain([None]).collect(),
        )),
        Schema::NoneType => part(Kind::NoneType),
        Schema::Bool => part(Kind::Bool),
        Schema::Int => part(Kind::Int),
        Schema::Float => part(Kind::Float),
        Schema::Str => part(Kind::Str),
        Schema::Bytes => part(Kind::Bytes),
        Schema::Literal(index) => Some((vec![label_of(&pool.constant(*index)?)?], Vec::new())),
        Schema::Union(members) => {
            let mut labels = Vec::new();
            let mut parts = Vec::new();
            for member in members.iter() {
                let (mine, theirs) = key_cover(member, pool)?;
                labels.extend(mine);
                parts.extend(theirs);
            }
            Some((labels, parts))
        }
        _ => None,
    }
}

/// The constant a pooled operand is, where a map atom could name it as a key.
fn label_of(operand: &Operand) -> Option<Label> {
    match operand {
        Operand::NoneType => Some(Label::NoneType),
        Operand::Boolean(value) => Some(Label::Bool(*value)),
        Operand::Integer(value) => Some(Label::Int(*value)),
        Operand::Word(word, kind) => Some(Label::Word(word.clone(), *kind)),
        // A float is not a `Literal` the typing spec allows, and an instance is
        // not a constant.
        Operand::Float(_) | Operand::Instance(_) => None,
    }
}

/// The set holding one pooled value and nothing else.
fn singleton(constant: &Operand) -> Option<Descr> {
    match constant {
        Operand::Boolean(value) => Some(Descr::boolean(*value)),
        Operand::Integer(value) => Some(Descr::integer(*value)),
        Operand::Float(value) => Some(Descr::float(*value)),
        Operand::Word(word, kind) => Descr::word(word, *kind),
        Operand::NoneType => Some(Descr::of_kind(Kind::NoneType)),
        // A class is a set of objects, not one value; a literal naming an
        // instance also pins *which* instance, which the descriptor cannot say.
        Operand::Instance(_) => None,
    }
}

/// The sequences a shape spells, under the kind that reads them.
fn sequence(
    container: SeqKind,
    shape: &SeqShape,
    pool: &dyn Constants,
    budget: &Cell<u32>,
    depth: u32,
) -> Option<Descr> {
    let kind = match container {
        SeqKind::List => Kind::List,
        SeqKind::Tuple => Kind::Tuple,
    };
    let prefix: Option<Vec<Descr>> = shape
        .prefix
        .iter()
        .map(|element| descend(element, pool, budget, depth))
        .collect();
    let tail = match &shape.tail {
        Some(element) => Some(descend(element, pool, budget, depth)?),
        None => None,
    };
    Descr::sequence(&prefix?, tail.as_ref(), kind)
}

/// The base narrowed by every constraint, which is a meet.
///
/// Each constraint is a *set* here rather than a test, so narrowing is
/// intersection and the order the constraints are written in carries no meaning
/// -- which is what makes `Ge(0) ∧ Lt(0)` decide rather than run.
fn refine(
    base: &Schema,
    constraints: &[Constraint],
    pool: &dyn Constants,
    budget: &Cell<u32>,
    depth: u32,
) -> Option<Descr> {
    let mut narrowed = descend(base, pool, budget, depth)?;
    for constraint in constraints {
        narrowed = narrowed.intersect(&constrained(constraint, &narrowed, pool)?)?;
    }
    Some(narrowed)
}

/// The set one constraint denotes, read under the kinds the base admits.
///
/// A bound is a set of *numbers* and a length bound a set of *words*, so which
/// component a constraint lands in depends on what it is narrowing. A constraint
/// the descriptor cannot read -- a predicate, a bound on a float, a length bound
/// on a sequence -- refuses.
fn constrained(constraint: &Constraint, base: &Descr, pool: &dyn Constants) -> Option<Descr> {
    // A bound is a set of whole numbers, and `bool` is a kind of its own here,
    // so the set has to be spelled in both slots: `True` is `1` to every
    // comparison Python makes, and a bound that admits `1` admits it.
    //
    // **Refuses unless the base is whole numbers and nothing else**, for the
    // reason [`words`] refuses a length bound over more than words. A bound
    // orders whatever a value's type orders -- a float, a string, a date -- and
    // these two components speak for two of those. Narrowing a float base to a
    // set of integers gives a *smaller* set than the schema denotes, and a
    // smaller set has a larger complement, which is a subtype proof no value
    // supports.
    let integers = |set: IntSet| {
        let numbers = Descr::of_kind(Kind::Int).union(&Descr::of_kind(Kind::Bool))?;
        if !base.intersect(&numbers.complement())?.is_empty() {
            return None;
        }
        let mut descr = Descr::nothing();
        descr.integers(set.clone());
        descr.booleans(
            [false, true]
                .into_iter()
                .filter(|boolean| set.holds(i64::from(*boolean)))
                .fold(BoolSet::EMPTY, |held, boolean| {
                    held.union(BoolSet::just(boolean))
                }),
        );
        Some(descr)
    };
    // The same shape for a float base. A bound on a float was refused outright,
    // so every relation needing the descriptor and mentioning `Annotated[float,
    // Gt(0)]` stayed undecided -- which was every one of the eighteen the random
    // sweep reported as an undecided corpus-true subtype.
    //
    // **Refuses unless the base is floats and nothing else**, for the reason the
    // integer side refuses: narrowing a wider base to a set of floats gives a
    // smaller set than the schema denotes, and a smaller set has a larger
    // complement. `nan` is outside every interval, which is what `FloatSet`'s
    // constructors already say and what Python's own comparisons do.
    let floats = |set: FloatSet| {
        if !base
            .intersect(&Descr::of_kind(Kind::Float).complement())?
            .is_empty()
        {
            return None;
        }
        let mut descr = Descr::nothing();
        descr.floats(set);
        Some(descr)
    };
    // A bound's operand is whatever the caller wrote: `Annotated[float, Gt(0)]`
    // carries the *integer* zero, and it orders the floats all the same. So the
    // side a bound lands on is chosen by the **base**, not by the operand's own
    // type, and the operand is read as a number either way.
    let as_float = |index: &OperandIx| match pool.operand(*index) {
        Some(Operand::Float(value)) => Some(value),
        #[expect(
            clippy::cast_precision_loss,
            reason = "a bound past 2^53 rounds to the nearest float, which is the \
                      comparison Python makes for the same pair"
        )]
        Some(Operand::Integer(value)) => Some(value as f64),
        _ => None,
    };
    let base_is_floats = base
        .intersect(&Descr::of_kind(Kind::Float).complement())
        .is_some_and(|rest| rest.is_empty());
    match constraint {
        Constraint::Ge(index) | Constraint::Gt(index) if base_is_floats => {
            let bound = as_float(index)?;
            floats(if matches!(constraint, Constraint::Gt(_)) {
                FloatSet::above(bound)
            } else {
                FloatSet::at_least(bound)
            })
        }
        Constraint::Le(index) | Constraint::Lt(index) if base_is_floats => {
            let bound = as_float(index)?;
            floats(if matches!(constraint, Constraint::Lt(_)) {
                FloatSet::below(bound)
            } else {
                FloatSet::at_most(bound)
            })
        }
        Constraint::Ge(index) | Constraint::Gt(index) => {
            let Operand::Integer(bound) = pool.operand(*index)? else {
                return None;
            };
            let lo = if matches!(constraint, Constraint::Gt(_)) {
                bound.checked_add(1)?
            } else {
                bound
            };
            integers(IntSet::between(Some(lo), None))
        }
        Constraint::Le(index) | Constraint::Lt(index) => {
            let Operand::Integer(bound) = pool.operand(*index)? else {
                return None;
            };
            let hi = if matches!(constraint, Constraint::Lt(_)) {
                bound.checked_sub(1)?
            } else {
                bound
            };
            integers(IntSet::between(None, Some(hi)))
        }
        Constraint::MultipleOf(index) => {
            let Operand::Integer(step) = pool.operand(*index)? else {
                return None;
            };
            integers(IntSet::multiple_of(step)?)
        }
        Constraint::MinLen(least) => lengths(base, &format!(".{{{least},}}"), &|kind| {
            Descr::sequences_at_least(*least, kind)
        }),
        Constraint::MaxLen(most) => lengths(base, &format!(".{{0,{most}}}"), &|kind| {
            Descr::sequences_at_most(*most, kind)
        }),
        Constraint::Regex(pattern) => words(pattern, base),
        // A callback is a leaf the core cannot read, which is what makes it
        // opaque to the procedure beside this one too.
        Constraint::Predicate(_) => None,
    }
}

/// The values of the base's kinds whose length the bound admits.
///
/// A length is not a word's alone, and for two of the kinds that have one it is
/// a property the representation can state: a word's length is a pattern over
/// its alphabet, and a sequence's is "any element, that many times", which is
/// as regular as any other shape the automaton holds. Both are built here and
/// unioned, so a base admitting words *and* sequences is bounded on each.
///
/// **Refuses for a base admitting anything else.** A set and a dict have a
/// length their components do not count, and a value with no length at all
/// fails the bound by raising -- which the walk reads as a non-member and this
/// cannot express. Lowering the bound while ignoring those kinds would give a
/// set larger than the schema denotes, and a larger set has a smaller
/// complement: a subtype proof no value supports.
fn lengths(
    base: &Descr,
    pattern: &str,
    sequences: &dyn Fn(Kind) -> Option<Descr>,
) -> Option<Descr> {
    const WORDS: [Kind; 2] = [Kind::Str, Kind::Bytes];
    const SEQUENCES: [Kind; 2] = [Kind::List, Kind::Tuple];

    let mut counted = Descr::nothing();
    for kind in WORDS.into_iter().chain(SEQUENCES) {
        counted = counted.union(&Descr::of_kind(kind))?;
    }
    if !base.intersect(&counted.complement())?.is_empty() {
        return None;
    }
    let mut whole = Descr::nothing();
    for kind in WORDS {
        if !Descr::of_kind(kind).intersect(base)?.is_empty() {
            whole = whole.union(&Descr::pattern(pattern, kind)?)?;
        }
    }
    for kind in SEQUENCES {
        if !Descr::of_kind(kind).intersect(base)?.is_empty() {
            whole = whole.union(&sequences(kind)?)?;
        }
    }
    // The base lies within the counted kinds, so `whole` is empty exactly when
    // the base is -- and a bound over an empty base denotes the empty set, which
    // is an answer rather than a refusal.
    Some(whole)
}

/// The words a pattern matches, under whichever word kind the base admits.
///
/// A length bound counts what the kind's alphabet counts -- code points for a
/// `str`, bytes for `bytes` -- which is why the pattern is read under the kind
/// rather than compiled once.
///
/// **Refuses unless the base is words and nothing else.** A length is not a
/// word's alone: a list, a tuple, a set and a dict all have one, and a bound on
/// them is a constraint the word component cannot express. Lowering the bound as
/// if it only spoke about words would give a *smaller* set than the schema
/// denotes -- and a smaller set has a larger complement, which is a subtype
/// proof that no value supports.
fn words(pattern: &str, base: &Descr) -> Option<Descr> {
    let mut alphabets = Descr::nothing();
    for kind in [Kind::Str, Kind::Bytes] {
        alphabets = alphabets.union(&Descr::of_kind(kind))?;
    }
    if !base.intersect(&alphabets.complement())?.is_empty() {
        return None;
    }
    let mut whole = Descr::nothing();
    for kind in [Kind::Str, Kind::Bytes] {
        if Descr::of_kind(kind).intersect(base)?.is_empty() {
            continue;
        }
        whole = whole.union(&Descr::pattern(pattern, kind)?)?;
    }
    (!whole.is_empty()).then_some(whole)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{BUDGET, Bounds, Constants, DEPTH, Operand, lower, lower_within};
    use crate::decision::{Kind, Verdict};
    use crate::descr::classes::Class;
    use crate::descr::{Descr, Value};
    use crate::ir::{
        ClassIx, ConstIx, Constraint, Field, MapClause, Openness, OperandIx, Schema, SeqKind,
        SeqShape,
    };

    /// A pool that answers from a list, which is what the bindings do from the
    /// validator's object table.
    struct Pool(Vec<Operand>);

    impl Constants for Pool {
        fn operand(&self, index: OperandIx) -> Option<Operand> {
            self.0.get(index.get()).cloned()
        }

        fn constant(&self, index: ConstIx) -> Option<Operand> {
            self.0.get(index.get()).cloned()
        }

        /// A class reaches the core through the bindings, and these tests have
        /// none: a pooled `Instance` reads as one this pool cannot see.
        fn class(&self, index: ClassIx) -> Option<Class> {
            match self.0.get(index.get()) {
                Some(Operand::Instance(class)) => Some(class.clone()),
                _ => None,
            }
        }
    }

    fn empty_pool() -> Pool {
        Pool(Vec::new())
    }

    /// The kinds and the two ends map straight across.
    #[test]
    fn the_leaves_lower_to_the_sets_they_denote() {
        let pool = empty_pool();
        let leaf = |schema| lower(&schema, &pool).expect("a leaf lowers");

        assert_eq!(leaf(Schema::ANYTHING), Descr::anything());
        assert_eq!(leaf(Schema::Nothing), Descr::nothing());
        assert_eq!(leaf(Schema::Float), Descr::of_kind(Kind::Float));
        assert_eq!(leaf(Schema::Str), Descr::of_kind(Kind::Str));
    }

    /// `bool` subclasses `int`, so the schema that denotes every integer
    /// denotes both kinds.
    ///
    /// The one leaf that is not one kind. Keeping `Bool` its own kind is what
    /// makes the two components independent, and the price is that `int` says
    /// so here rather than being read off the name.
    #[test]
    fn the_integers_include_the_booleans() {
        let pool = empty_pool();
        let ints = lower(&Schema::Int, &pool).expect("int lowers");

        assert!(ints.admits(Value::integer(1)));
        assert!(ints.admits(Value::boolean(true)));
        assert!(!ints.admits(Value::float(1.0)));
        assert!(
            !lower(&Schema::Bool, &pool)
                .expect("bool lowers")
                .admits(Value::integer(1))
        );
    }

    /// The three operations lower to the three operations, which is the whole
    /// point of the map.
    #[test]
    fn the_operations_lower_to_the_operations() {
        let pool = empty_pool();
        let joined = lower(
            &Schema::Union(vec![Schema::Str, Schema::Float].into()),
            &pool,
        )
        .expect("a small union");
        let expected = Descr::of_kind(Kind::Str)
            .union(&Descr::of_kind(Kind::Float))
            .expect("a small union");
        assert_eq!(joined, expected);

        let barred =
            lower(&Schema::Complement(Arc::new(Schema::Str)), &pool).expect("a small complement");
        assert_eq!(barred, Descr::of_kind(Kind::Str).complement());
    }

    /// A refinement is a *meet of sets*, so the order the constraints are
    /// written in carries no meaning and an impossible pair decides.
    #[test]
    fn a_bound_pair_that_cannot_hold_lowers_to_the_empty_set() {
        let pool = Pool(vec![Operand::Integer(0)]);
        let refined = |constraints: Vec<Constraint>| {
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::Int),
                    constraints: constraints.into(),
                },
                &pool,
            )
            .expect("a small refinement")
        };

        let ge_then_lt = refined(vec![
            Constraint::Ge(OperandIx::new(0)),
            Constraint::Lt(OperandIx::new(0)),
        ]);
        let lt_then_ge = refined(vec![
            Constraint::Lt(OperandIx::new(0)),
            Constraint::Ge(OperandIx::new(0)),
        ]);
        assert_eq!(ge_then_lt.emptiness(), Verdict::Empty);
        assert_eq!(ge_then_lt, lt_then_ge, "the order says nothing");

        let non_negative = refined(vec![Constraint::Ge(OperandIx::new(0))]);
        assert!(non_negative.admits(Value::integer(0)));
        assert!(!non_negative.admits(Value::integer(-1)));
    }

    /// The relations a decision procedure cannot reach, decided on the sets.
    ///
    /// Each is a *value* fact the term layer has no rule for: a step divides
    /// another, a literal misses a kind, a kind is exhausted by its literals, a
    /// set is its own hole plus what fills it. Lowering them puts each in the
    /// kind set it denotes, where the answer is `a ∧ ¬b = ∅` and nothing else.
    #[test]
    fn the_value_relations_decide_on_the_sets() {
        let pool = Pool(vec![
            Operand::Integer(4),
            Operand::Integer(2),
            Operand::Integer(1),
            Operand::Boolean(true),
            Operand::Boolean(false),
        ]);
        let set = |schema| lower(&schema, &pool).expect("a small schema");
        let step = |at| {
            set(Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![Constraint::MultipleOf(OperandIx::new(at))].into(),
            })
        };
        let literal = |at| set(Schema::Literal(ConstIx::new(at)));
        let within = |a: &Descr, b: &Descr| {
            a.intersect(&b.complement())
                .expect("two small sets")
                .emptiness()
        };

        // A step is a subset of the steps it is a multiple of, and of no other.
        assert_eq!(within(&step(0), &step(1)), Verdict::Empty);
        assert_eq!(within(&step(1), &step(0)), Verdict::Inhabited);

        // `bool` is its own kind, so an integer literal is never a boolean.
        let no_bool = set(Schema::Complement(Arc::new(Schema::Bool)));
        assert_eq!(within(&literal(2), &no_bool), Verdict::Empty);

        // A kind with finitely many values is exhausted by naming them all.
        let both = set(Schema::Union(
            vec![
                Schema::Literal(ConstIx::new(3)),
                Schema::Literal(ConstIx::new(4)),
            ]
            .into(),
        ));
        assert_eq!(within(&set(Schema::Bool), &both), Verdict::Empty);
        assert_eq!(within(&both, &set(Schema::Bool)), Verdict::Empty);

        // A set is the hole punched in it, put back: `a = (a ∧ ¬v) ∨ v` for a
        // value `a` holds.
        let split = set(Schema::Union(
            vec![
                Schema::meet(vec![
                    Schema::Int,
                    Schema::Complement(Arc::new(Schema::Literal(ConstIx::new(2)))),
                ]),
                Schema::Literal(ConstIx::new(2)),
            ]
            .into(),
        ));
        assert_eq!(within(&set(Schema::Int), &split), Verdict::Empty);
        assert_eq!(within(&split, &set(Schema::Int)), Verdict::Empty);
    }

    /// A bound orders the booleans as well as the integers.
    ///
    /// `bool` is a kind of its own here, and `int` denotes both -- so a bound
    /// over `int` that spoke only for the integer component would give a
    /// *smaller* set than the schema denotes, and a smaller set has a larger
    /// complement. `True` is `1` to every comparison Python makes, so a bound
    /// admitting `1` admits it.
    #[test]
    fn a_bound_over_the_integers_orders_the_booleans_too() {
        let pool = Pool(vec![Operand::Integer(1)]);
        let bounded = |constraint| {
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::Int),
                    constraints: vec![constraint].into(),
                },
                &pool,
            )
            .expect("a small refinement")
        };

        let at_least_one = bounded(Constraint::Ge(OperandIx::new(0)));
        assert!(at_least_one.admits(Value::boolean(true)));
        assert!(!at_least_one.admits(Value::boolean(false)));
        assert!(at_least_one.admits(Value::integer(1)));

        let below_one = bounded(Constraint::Lt(OperandIx::new(0)));
        assert!(below_one.admits(Value::boolean(false)));
        assert!(!below_one.admits(Value::boolean(true)));

        // A bound over a **float** base lands in the float component instead,
        // with the operand read as a number: `Annotated[float, Ge(0)]` carries
        // the integer zero and orders the floats all the same. Narrowing a float
        // base to a set of *integers* is what would be unsound, and that is not
        // what happens.
        // The pool's only operand is the integer 1, so this is "float >= 1".
        let floats_at_least_one = lower(
            &Schema::Refine {
                base: Arc::new(Schema::Float),
                constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
            },
            &pool,
        )
        .expect("a bound over floats lowers");
        assert!(floats_at_least_one.admits(Value::float(1.0)));
        assert!(floats_at_least_one.admits(Value::float(1.5)));
        assert!(!floats_at_least_one.admits(Value::float(0.5)));
        assert!(!floats_at_least_one.admits(Value::float(-1.0)));
        // `nan` is outside every interval, which is the comparison Python makes.
        assert!(!floats_at_least_one.admits(Value::float(f64::NAN)));
        // And it is floats and nothing else: the integer 1 is not in it.
        assert!(!floats_at_least_one.admits(Value::integer(1)));

        // A base that is neither whole numbers nor floats alone still refuses,
        // for the reason both sides refuse: narrowing it to one component gives
        // a smaller set than the schema denotes, and a smaller set has a larger
        // complement.
        assert!(
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::ANYTHING),
                    constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
                },
                &pool,
            )
            .is_none()
        );
    }

    /// A bound over floats reads a float operand, and reads both directions.
    ///
    /// The test above writes the bound as `Annotated[float, Ge(1)]`, whose
    /// operand is the *integer* one; this writes the one a caller reaches for
    /// more often, `Ge(1.5)`, whose fractional part is the whole difference --
    /// read as an integer it would round, and `1.25` would land inside a set
    /// that excludes it. The upper direction is a separate arm from the lower,
    /// and a reading that carried only one of them refuses the other, leaving
    /// the relation undecided where the schema is perfectly ordinary.
    #[test]
    fn a_float_bound_is_read_as_a_float_in_both_directions() {
        let pool = Pool(vec![Operand::Float(1.5)]);
        let bounded = |constraint| {
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::Float),
                    constraints: vec![constraint].into(),
                },
                &pool,
            )
            .expect("a bound over floats lowers")
        };

        let at_least = bounded(Constraint::Ge(OperandIx::new(0)));
        assert!(at_least.admits(Value::float(1.5)));
        assert!(!at_least.admits(Value::float(1.25)), "1.5 is not 1");
        let above = bounded(Constraint::Gt(OperandIx::new(0)));
        assert!(above.admits(Value::float(1.75)) && !above.admits(Value::float(1.5)));

        let at_most = bounded(Constraint::Le(OperandIx::new(0)));
        assert!(at_most.admits(Value::float(1.5)) && at_most.admits(Value::float(0.0)));
        assert!(!at_most.admits(Value::float(1.75)));
        // `nan` is outside every interval, on this side as on the other.
        assert!(!at_most.admits(Value::float(f64::NAN)));
        let below = bounded(Constraint::Lt(OperandIx::new(0)));
        assert!(below.admits(Value::float(1.25)) && !below.admits(Value::float(1.5)));
    }

    /// A step is the constraint no union of intervals can spell, and it meets
    /// the bounds rather than being checked beside them.
    #[test]
    fn a_step_meets_the_bounds() {
        let pool = Pool(vec![Operand::Integer(2), Operand::Integer(1)]);
        let evens = lower(
            &Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![
                    Constraint::MultipleOf(OperandIx::new(0)),
                    Constraint::Ge(OperandIx::new(1)),
                ]
                .into(),
            },
            &pool,
        )
        .expect("a small refinement");

        assert!(evens.admits(Value::integer(2)) && evens.admits(Value::integer(4)));
        assert!(!evens.admits(Value::integer(0)), "the bound excludes it");
        assert!(!evens.admits(Value::integer(3)), "the step excludes it");
    }

    /// A sequence lowers through the one constructor its three spellings share.
    #[test]
    fn a_sequence_lowers_to_its_shape() {
        const ONE: &[Value] = &[Value::integer(1)];
        const TWO: &[Value] = &[Value::integer(1), Value::integer(1)];

        let pool = empty_pool();
        let of_ints = lower(
            &Schema::Seq {
                container: SeqKind::List,
                shape: SeqShape {
                    prefix: Vec::new().into(),
                    tail: Some(Arc::new(Schema::Int)),
                },
            },
            &pool,
        )
        .expect("a small sequence");

        assert!(of_ints.admits(Value::sequence(ONE, Kind::List)));
        assert!(of_ints.admits(Value::sequence(TWO, Kind::List)));
        assert!(
            !of_ints.admits(Value::sequence(ONE, Kind::Tuple)),
            "a tuple is not a list"
        );
    }

    /// A set lowers to the powerset of what it holds, hashability included.
    #[test]
    fn a_set_lowers_to_a_powerset() {
        const NOTHING: &[Value] = &[];

        let pool = empty_pool();
        let of_lists = lower(
            &Schema::set(Schema::Seq {
                container: SeqKind::List,
                shape: SeqShape {
                    prefix: Vec::new().into(),
                    tail: Some(Arc::new(Schema::Int)),
                },
            }),
            &pool,
        )
        .expect("a small set");

        // A list is unhashable, so the only member left is none at all.
        assert!(of_lists.admits(Value::sequence(NOTHING, Kind::Set)));
        assert_eq!(
            of_lists,
            Descr::set(&Descr::nothing(), Kind::Set).expect("a set kind")
        );
    }

    /// The schemas both the procedure and the descriptor understand, as a
    /// corpus to compare them over.
    fn leaves() -> Vec<Schema> {
        let seq = |tail| Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape {
                prefix: Vec::new().into(),
                tail: Some(Arc::new(tail)),
            },
        };
        vec![
            Schema::ANYTHING,
            Schema::Nothing,
            Schema::NoneType,
            Schema::Bool,
            Schema::Int,
            Schema::Float,
            Schema::Str,
            Schema::Bytes,
            seq(Schema::Int),
            seq(Schema::Str),
            Schema::set(Schema::Int),
        ]
    }

    /// The leaves, their complements, and every pair joined and met.
    ///
    /// Quadratic in the leaves, so the *containment* check below takes the
    /// leaves alone: it is quadratic again over whatever it is given, and a
    /// descriptor meet builds automata.
    fn corpus() -> Vec<Schema> {
        let leaves = leaves();
        let mut corpus = leaves.clone();
        for left in &leaves {
            for right in &leaves {
                corpus.push(Schema::Union(vec![left.clone(), right.clone()].into()));
                corpus.push(Schema::Intersection(
                    vec![left.clone(), right.clone()].into(),
                ));
            }
            corpus.push(Schema::Complement(Arc::new(left.clone())));
        }
        corpus
    }

    /// The descriptor and the procedure agree about emptiness wherever both
    /// decide it.
    ///
    /// The check the whole milestone is for: two representations of one meaning,
    /// asked the same question. Neither is taken as the oracle -- each has
    /// answers the other lacks, so the claim is only that they never *contradict*
    /// each other, and a proof from one is never met by the opposite proof from
    /// the other.
    #[test]
    fn the_descriptor_and_the_procedure_never_contradict_each_other() {
        let pool = empty_pool();
        for schema in corpus() {
            let Some(descr) = lower(&schema, &pool) else {
                continue;
            };
            if descr.emptiness() == Verdict::Empty {
                assert!(
                    schema.is_empty(),
                    "the descriptor proved {schema:?} empty and the procedure did not"
                );
            }
            if schema.is_empty() {
                assert_ne!(
                    descr.emptiness(),
                    Verdict::Inhabited,
                    "the procedure proved {schema:?} empty and the descriptor denied it"
                );
            }
        }
    }

    /// And about containment, which is the relation emptiness is asked for.
    #[test]
    fn the_two_agree_about_containment_where_both_decide() {
        let pool = empty_pool();
        let corpus: Vec<Schema> = leaves()
            .iter()
            .flat_map(|leaf| [leaf.clone(), Schema::Complement(Arc::new(leaf.clone()))])
            .collect();
        for left in &corpus {
            for right in &corpus {
                let (Some(a), Some(b)) = (lower(left, &pool), lower(right, &pool)) else {
                    continue;
                };
                let Some(difference) = a.intersect(&b.complement()) else {
                    continue;
                };
                if difference.emptiness() == Verdict::Empty {
                    assert!(
                        left.is_subtype_of(right),
                        "the descriptor made {left:?} a subtype of {right:?} and the \
                         procedure did not"
                    );
                }
            }
        }
    }

    /// A literal lowers to the singleton its pooled value names, under the kind
    /// that reads that value.
    ///
    /// The typing spec keeps `Literal[1]`, `Literal[True]` and `Literal["1"]`
    /// apart, and so does this: each lands in its own kind's component, so no
    /// two of them meet.
    #[test]
    fn a_literal_lowers_to_the_singleton_its_kind_reads() {
        let pool = Pool(vec![
            Operand::Integer(1),
            Operand::Boolean(true),
            Operand::Word(b"a".to_vec(), Kind::Str),
            Operand::NoneType,
            Operand::Float(1.5),
        ]);
        let literal =
            |slot| lower(&Schema::Literal(ConstIx::new(slot)), &pool).expect("a pooled constant");

        assert!(literal(0).admits(Value::integer(1)) && !literal(0).admits(Value::integer(2)));
        assert!(
            literal(1).admits(Value::boolean(true)) && !literal(1).admits(Value::boolean(false))
        );
        assert!(literal(2).admits(Value::word(b"a", Kind::Str)));
        assert!(literal(3).admits(Value::of_kind(Kind::NoneType)));
        assert!(literal(4).admits(Value::float(1.5)));

        // The three the spec keeps apart stay apart, because each is a
        // different kind's component.
        for (left, right) in [(0, 1), (0, 2), (1, 2)] {
            assert_eq!(
                literal(left)
                    .intersect(&literal(right))
                    .expect("two singletons")
                    .emptiness(),
                Verdict::Empty,
                "{left} and {right}"
            );
        }
    }

    /// A class the pool can see lowers to the set of its instances, carrying the
    /// order it was given: the subclass is a subtype, and the two ends are not.
    #[test]
    fn a_class_the_pool_knows_lowers_to_its_instances() {
        let animal = Class::new(0, Class::PLAIN, &[]);
        let dog = Class::new(1, Class::PLAIN, std::slice::from_ref(&animal));
        let pool = Pool(vec![
            Operand::Instance(animal),
            Operand::Instance(dog),
            Operand::Instance(Class::new(2, Class::PLAIN, &[])),
        ]);
        let instances = |at| lower(&Schema::Instance(ClassIx::new(at)), &pool).expect("a class");
        let (animals, dogs, others) = (instances(0), instances(1), instances(2));

        // `a ≤ b` is `a ∧ ¬b = ∅`, which is the whole of the subtyping test.
        let within = |a: &Descr, b: &Descr| {
            a.intersect(&b.complement())
                .expect("two classes")
                .emptiness()
        };
        assert_eq!(within(&dogs, &animals), Verdict::Empty);
        // The converse is not merely undecided: an animal that is not a dog is
        // a value the descriptor can name.
        assert_eq!(within(&animals, &dogs), Verdict::Inhabited);
        // Two unrelated classes share `object`, so nothing here proves them
        // disjoint -- only that neither contains the other.
        assert_eq!(
            animals.intersect(&others).expect("two classes").emptiness(),
            Verdict::Unknown
        );
        // A class is a set of objects and nothing else: no scalar is one.
        assert!(!animals.admits(Value::integer(0)));
    }

    /// A literal naming a class instance refuses: a class is a set of objects,
    /// and the literal pins *which* instance, which the descriptor cannot say.
    #[test]
    fn a_literal_naming_an_instance_refuses() {
        let pool = Pool(vec![Operand::Instance(Class::laid_out(1, 1))]);
        assert!(lower(&Schema::Literal(ConstIx::new(0)), &pool).is_none());
    }

    /// A length bound and a pattern are sets of *words*, and they lower under
    /// whichever word kind the base admits.
    #[test]
    fn the_word_constraints_lower_to_languages() {
        let pool = empty_pool();
        let refined = |base, constraints| {
            lower(
                &Schema::Refine {
                    base: Arc::new(base),
                    constraints,
                },
                &pool,
            )
        };

        let non_empty =
            refined(Schema::Str, vec![Constraint::MinLen(1)].into()).expect("a length bound");
        assert!(non_empty.admits(Value::word(b"a", Kind::Str)));
        assert!(!non_empty.admits(Value::word(b"", Kind::Str)));

        let short =
            refined(Schema::Str, vec![Constraint::MaxLen(1)].into()).expect("a length bound");
        assert!(
            short.admits(Value::word(b"", Kind::Str)) && short.admits(Value::word(b"a", Kind::Str))
        );
        assert!(!short.admits(Value::word(b"ab", Kind::Str)));

        let matching = refined(Schema::Str, vec![Constraint::Regex("a+".to_owned())].into())
            .expect("a pattern");
        assert!(matching.admits(Value::word(b"a", Kind::Str)));
        assert!(!matching.admits(Value::word(b"b", Kind::Str)));

        // Two of them meet rather than being checked one after the other, which
        // is what makes an impossible pair decide.
        let impossible = refined(
            Schema::Str,
            vec![Constraint::MinLen(2), Constraint::MaxLen(1)].into(),
        )
        .expect("two length bounds");
        assert_eq!(impossible.emptiness(), Verdict::Empty);
    }

    /// A word constraint on a base that is *more* than words refuses too.
    ///
    /// A length is not a word's alone -- a list, a set and a dict all have one --
    /// so a bound over `anything` constrains values the word component cannot
    /// speak about. Lowering it as if it only spoke about words would give a
    /// smaller set than the schema denotes, and a smaller set has a *larger*
    /// complement: `set[anything] <= ~Annotated[anything, MinLen(0)]` would be
    /// proved, with the empty set standing against it.
    #[test]
    fn a_length_bound_over_more_than_words_refuses() {
        let pool = empty_pool();
        assert!(
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::ANYTHING),
                    constraints: vec![Constraint::MinLen(0)].into(),
                },
                &pool,
            )
            .is_none()
        );

        // Narrowed to the words first, the same bound lowers.
        assert!(
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::Str),
                    constraints: vec![Constraint::MinLen(0)].into(),
                },
                &pool,
            )
            .is_some()
        );
    }

    /// A word constraint on a base with no words refuses.
    ///
    /// The bound has no component to land in: an integer has no length, so
    /// there is no language to meet the base with. Refusing says that; lowering
    /// to the empty set would claim the schema *denotes* nothing, which is a
    /// stronger statement than this map is entitled to make.
    #[test]
    fn a_length_bound_on_a_base_with_no_words_refuses() {
        let pool = empty_pool();
        assert!(
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::Int),
                    constraints: vec![Constraint::MinLen(1)].into(),
                },
                &pool,
            )
            .is_none()
        );
    }

    /// A schema past a bound refuses rather than spending without end.
    ///
    /// Lowering builds at every node, so a wide schema is a lot of work and a
    /// deep one is that work raised to a power. There are two bounds because
    /// there are two quantities: [`BUDGET`] counts the nodes and [`DEPTH`] the
    /// nesting, and each is asserted in both directions -- a bound that only
    /// ever refuses would pass half of this.
    ///
    /// Refusing is safe, because the caller decides the old way.
    #[test]
    fn a_schema_past_a_bound_refuses() {
        let pool = empty_pool();
        let wide = |members: usize| {
            Schema::Union(
                core::iter::repeat_n(Schema::Str, members)
                    .collect::<Vec<_>>()
                    .into(),
            )
        };
        assert!(
            lower(&wide(BUDGET as usize), &pool).is_none(),
            "too many nodes"
        );
        assert!(lower(&wide(BUDGET as usize / 2), &pool).is_some());

        let nested = |levels: u32| {
            (0..levels).fold(Schema::Str, |inner, _| Schema::Complement(Arc::new(inner)))
        };
        assert!(lower(&nested(DEPTH + 1), &pool).is_none(), "too deep");
        assert!(lower(&nested(DEPTH - 1), &pool).is_some());
    }

    /// A build that would cost too much refuses, and the same schema lowers
    /// under an allowance that covers it.
    ///
    /// The refusal is about the *work*, not about the schema: nothing here is a
    /// form the descriptor cannot hold. Which is why it is asserted in both
    /// directions -- an allowance that only ever refuses would pass half of it.
    #[test]
    fn a_build_past_its_allowance_refuses() {
        let pool = empty_pool();
        let meet = Schema::Intersection(
            vec![
                Schema::list(SeqShape::homogeneous(Schema::Int)),
                Schema::list(SeqShape::homogeneous(Schema::Str)),
            ]
            .into(),
        );

        assert!(
            lower_within(
                Bounds {
                    work: 0,
                    ..Bounds::DEFAULT
                },
                &meet,
                &pool
            )
            .is_none(),
            "nothing to spend"
        );
        assert!(lower_within(Bounds::DEFAULT, &meet, &pool).is_some());
        // A leaf takes no product, so it costs nothing and lowers on an empty
        // allowance: the budget bounds what multiplies, not what is read. `int`
        // is not one of those -- it is the union of two kinds -- which is why
        // this says `str`.
        assert!(
            lower_within(
                Bounds {
                    work: 0,
                    ..Bounds::DEFAULT
                },
                &Schema::Str,
                &pool
            )
            .is_some()
        );
        assert!(
            lower_within(
                Bounds {
                    work: 0,
                    ..Bounds::DEFAULT
                },
                &Schema::Int,
                &pool
            )
            .is_none(),
            "a union"
        );
    }

    /// The allowance a lowering is given is its own: an expensive one does not
    /// leave the next one poorer.
    #[test]
    fn a_build_does_not_spend_the_next_one_s_allowance() {
        let pool = empty_pool();
        let deep = (0..6).fold(
            Schema::record(
                vec![Field {
                    name: "leaf".into(),
                    schema: Schema::Int,
                    required: true,
                }],
                Openness::Closed,
            ),
            |inner, _| {
                Schema::record(
                    vec![Field {
                        name: "child".into(),
                        schema: Schema::list(SeqShape::homogeneous(inner)),
                        required: true,
                    }],
                    Openness::Closed,
                )
            },
        );

        for _ in 0..3 {
            let _ = lower(&deep, &pool);
            assert!(
                lower(&Schema::set(Schema::Int), &pool).is_some(),
                "the allowance came back"
            );
        }
    }

    /// The map is partial, and it refuses rather than approximating.
    ///
    /// Each of these has no component to land in, or none that would mean what
    /// the schema does: a dict has no map component, an attribute record beside
    /// a builtin kind wants a descriptor that is a union of lines, and a
    /// reference is a cycle.
    #[test]
    fn the_forms_with_nowhere_to_land_refuse() {
        let pool = empty_pool();
        for schema in [
            // A class needs the object pool to say what it derives from, and the
            // core has none.
            Schema::Instance(crate::ir::ClassIx::new(0)),
            Schema::Ref(crate::ir::DefIx::new(0)),
            // A key schema that is neither a kind nor a constant covers part of
            // a part, and the default is a function on the parts.
            Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![MapClause {
                    key: Schema::Refine {
                        base: Arc::new(Schema::Str),
                        constraints: vec![Constraint::MinLen(1)].into(),
                    },
                    value: Schema::Int,
                }]
                .into(),
            },
        ] {
            assert!(lower(&schema, &pool).is_none(), "{schema:?}");
        }
        // A refusal inside a form refuses the whole form rather than dropping
        // the part it could not read.
        assert!(
            lower(
                &Schema::Union(vec![Schema::Str, Schema::Ref(crate::ir::DefIx::new(0))].into()),
                &pool
            )
            .is_none()
        );
        // `Any` is the top, spelled, so it lowers to the top rather than
        // refusing: the spelling is not a set and the descriptor holds sets.
        assert_eq!(lower(&Schema::ANY, &pool), Some(Descr::anything()));
    }

    /// An attribute record lowers without the pool, because a field's name and
    /// its type are the whole of it.
    ///
    /// It is the half of an object schema the core can read: the class beside it
    /// needs the pool, and the two meet once the pool is in reach. The record
    /// narrows a value of *any* kind, so what lowers here is not scoped to the
    /// values that have no kind.
    #[test]
    fn an_attribute_record_lowers_to_the_values_carrying_it() {
        const CARRIED: &[(&str, Value)] = &[("a", Value::integer(1))];
        let pool = empty_pool();
        let field = |name: &str, schema, required| crate::ir::Field {
            name: name.into(),
            schema,
            required,
        };
        let record = Schema::AttrRecord {
            fields: vec![field("a", Schema::Int, true)].into(),
        };
        let lowered = lower(&record, &pool).expect("an attribute record lowers");
        assert!(lowered.admits(Value::object(CARRIED)));
        // And it narrows a value that also has a kind, which is the whole point
        // of holding the record on a line rather than beside the kinds.
        assert!(lowered.admits(Value::integer(7).carrying(CARRIED)));
        assert!(!lowered.admits(Value::integer(7)));
        // A field whose type admits nothing empties the record.
        let empty = Schema::AttrRecord {
            fields: vec![field("a", Schema::Nothing, true)].into(),
        };
        assert!(lower(&empty, &pool).expect("it lowers").is_empty());
    }

    /// A map lowers to the atom its keys spell, and the three rows the report
    /// lists as undecided fall out of the semantic `dom`.
    ///
    /// A label whose type its part's default already gives it says nothing, and
    /// the atom drops it -- so a key that must be absent from a closed record
    /// leaves the empty map, whichever of the three ways it was written.
    #[test]
    fn the_maps_that_name_nothing_are_the_empty_map() {
        let pool = empty_pool();
        let closed =
            |fields: Vec<crate::ir::Field>, defaults: Vec<crate::ir::MapClause>| Schema::KeyedMap {
                fields: fields.into(),
                defaults: defaults.into(),
            };
        let field = |name: &str, schema, required| crate::ir::Field {
            name: name.into(),
            schema,
            required,
        };
        let empty = lower(&closed(Vec::new(), Vec::new()), &pool).expect("`{}` lowers");

        // `{"a?": nothing}`: the key may be absent and holds nothing, which is
        // what the closed default already says, so the label is absorbed.
        let optional_nothing = closed(vec![field("a", Schema::Nothing, false)], Vec::new());
        assert_eq!(lower(&optional_nothing, &pool), Some(empty.clone()));

        // `dict[str, nothing]`: every `str` key maps into nothing, so there are
        // none, and no other part was opened.
        let no_str_values = closed(
            Vec::new(),
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Nothing,
            }],
        );
        assert_eq!(lower(&no_str_values, &pool), Some(empty.clone()));

        // `dict[nothing, int]`: no key at all is governed, so none is admitted.
        let no_keys = closed(
            Vec::new(),
            vec![MapClause {
                key: Schema::Nothing,
                value: Schema::Int,
            }],
        );
        assert_eq!(lower(&no_keys, &pool), Some(empty.clone()));

        // And the empty map is not the empty *set*: it holds one dict.
        assert!(!empty.is_empty());
        assert!(empty.admits(Value::dict(&[])));
    }

    /// A required field is required, and an open record admits the keys it does
    /// not name.
    #[test]
    fn a_record_lowers_closed_and_a_catch_all_opens_it() {
        const A_IS_INT: &[(Value, Value)] = &[(Value::word(b"a", Kind::Str), Value::integer(1))];
        const A_AND_B: &[(Value, Value)] = &[
            (Value::word(b"a", Kind::Str), Value::integer(1)),
            (Value::word(b"b", Kind::Str), Value::integer(2)),
        ];
        let pool = empty_pool();
        let field = crate::ir::Field {
            name: "a".into(),
            schema: Schema::Int,
            required: true,
        };
        let shut = lower(
            &Schema::KeyedMap {
                fields: vec![field.clone()].into(),
                defaults: Vec::new().into(),
            },
            &pool,
        )
        .expect("a closed record lowers");
        assert!(shut.admits(Value::dict(A_IS_INT)));
        assert!(!shut.admits(Value::dict(&[])), "the field is required");
        assert!(!shut.admits(Value::dict(A_AND_B)), "and nothing else is");

        let open = lower(
            &Schema::KeyedMap {
                fields: vec![field].into(),
                defaults: vec![MapClause::top()].into(),
            },
            &pool,
        )
        .expect("an open record lowers");
        assert!(open.admits(Value::dict(A_IS_INT)));
        assert!(open.admits(Value::dict(A_AND_B)), "a catch-all opens it");
        assert!(
            !open.admits(Value::dict(&[])),
            "the field is still required"
        );
    }

    /// Each key kind is its own part, and a clause opens the one its key names.
    ///
    /// The default is a function on the partition, so `dict[int, V]` says what an
    /// integer key maps to and leaves a string key forbidden -- the map was
    /// closed, and only the part the clause named was opened.
    #[test]
    fn a_clause_opens_the_part_its_key_names() {
        /// One key of each part, beside the schema that names that part.
        const KEYS: [(Kind, &[(Value, Value)]); 6] = [
            (
                Kind::NoneType,
                &[(Value::of_kind(Kind::NoneType), Value::integer(1))],
            ),
            (Kind::Bool, &[(Value::boolean(true), Value::integer(1))]),
            (Kind::Int, &[(Value::integer(7), Value::integer(1))]),
            (Kind::Float, &[(Value::float(1.5), Value::integer(1))]),
            (
                Kind::Str,
                &[(Value::word(b"a", Kind::Str), Value::integer(1))],
            ),
            (
                Kind::Bytes,
                &[(Value::word(b"a", Kind::Bytes), Value::integer(1))],
            ),
        ];
        let atom = |kind| match kind {
            Kind::NoneType => Schema::NoneType,
            Kind::Bool => Schema::Bool,
            Kind::Int => Schema::Int,
            Kind::Float => Schema::Float,
            Kind::Bytes => Schema::Bytes,
            _ => Schema::Str,
        };
        let pool = empty_pool();
        for (kind, _) in KEYS {
            let opened = lower(
                &Schema::KeyedMap {
                    fields: Vec::new().into(),
                    defaults: vec![MapClause {
                        key: atom(kind),
                        value: Schema::Int,
                    }]
                    .into(),
                },
                &pool,
            )
            .expect("a mapping lowers");
            for (other, entry) in KEYS {
                assert_eq!(
                    opened.admits(Value::dict(entry)),
                    other == kind,
                    "a {kind:?}-keyed map against a {other:?} key"
                );
            }
        }
    }

    /// A union of key schemas opens each part it names, and a `Literal` names a
    /// key rather than a part.
    #[test]
    fn a_union_opens_each_part_and_a_literal_names_one_key() {
        const A: &[(Value, Value)] = &[(Value::word(b"a", Kind::Str), Value::integer(1))];
        const B: &[(Value, Value)] = &[(Value::word(b"b", Kind::Str), Value::integer(1))];
        const ONE: &[(Value, Value)] = &[(Value::integer(1), Value::integer(1))];
        let pool = empty_pool();
        let either = lower(
            &Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![MapClause {
                    key: Schema::Union(vec![Schema::Str, Schema::Int].into()),
                    value: Schema::Int,
                }]
                .into(),
            },
            &pool,
        )
        .expect("a union of key kinds lowers");
        assert!(either.admits(Value::dict(A)));
        assert!(either.admits(Value::dict(ONE)));

        // A literal key is a label: the key it names is governed, and every
        // other key of that part is not.
        let named = Pool(vec![Operand::Word(b"a".to_vec(), Kind::Str)]);
        let one_key = lower(
            &Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![MapClause {
                    key: Schema::Literal(ConstIx::new(0)),
                    value: Schema::Int,
                }]
                .into(),
            },
            &named,
        )
        .expect("a literal key lowers");
        assert!(one_key.admits(Value::dict(A)));
        assert!(
            one_key.admits(Value::dict(&[])),
            "a clause does not require it"
        );
        assert!(!one_key.admits(Value::dict(B)), "and names no other key");
    }

    /// A literal key of any constant kind is a label, not only a `str` one.
    ///
    /// The descriptor names the key by the constant it is, so reading a *value's*
    /// key has to answer with the same constant -- and a key it cannot name is
    /// read through its part's default instead, which a closed map shuts. Each
    /// kind of constant is its own arm on both sides, and a missing one turns a
    /// named key into an anonymous one that the map then rejects.
    #[test]
    fn a_literal_key_of_every_constant_kind_names_its_key() {
        const NONE_KEY: &[(Value, Value)] = &[(Value::of_kind(Kind::NoneType), Value::integer(1))];
        const TRUE_KEY: &[(Value, Value)] = &[(Value::boolean(true), Value::integer(1))];
        const FALSE_KEY: &[(Value, Value)] = &[(Value::boolean(false), Value::integer(1))];
        const ONE_KEY: &[(Value, Value)] = &[(Value::integer(1), Value::integer(1))];
        const TWO_KEY: &[(Value, Value)] = &[(Value::integer(2), Value::integer(1))];
        const RAW_KEY: &[(Value, Value)] = &[(Value::word(b"a", Kind::Bytes), Value::integer(1))];

        /// One constant, a dict whose key it names, and one of the same part it
        /// does not.
        type Case = (
            Operand,
            &'static [(Value, Value)],
            &'static [(Value, Value)],
        );

        let cases: [Case; 4] = [
            (Operand::NoneType, NONE_KEY, ONE_KEY),
            (Operand::Boolean(true), TRUE_KEY, FALSE_KEY),
            (Operand::Integer(1), ONE_KEY, TWO_KEY),
            (Operand::Word(b"a".to_vec(), Kind::Bytes), RAW_KEY, ONE_KEY),
        ];
        for (constant, named, other) in cases {
            let pool = Pool(vec![constant.clone()]);
            let map = lower(
                &Schema::KeyedMap {
                    fields: Vec::new().into(),
                    defaults: vec![MapClause {
                        key: Schema::Literal(ConstIx::new(0)),
                        value: Schema::Int,
                    }]
                    .into(),
                },
                &pool,
            )
            .expect("a literal key lowers");
            assert!(map.admits(Value::dict(named)), "{constant:?} names its key");
            assert!(
                !map.admits(Value::dict(other)),
                "{constant:?} names no other"
            );
        }
    }

    /// An operand the pool cannot read refuses the constraint that names it.
    #[test]
    fn an_operand_the_pool_cannot_read_refuses() {
        let pool = Pool(vec![Operand::Float(0.5)]);
        assert!(
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::Int),
                    constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
                },
                &pool,
            )
            .is_none(),
            "a float bound is not an integer set"
        );
    }

    /// A length bound lands in the kinds the base has and in no other: the bound
    /// over a word admits no sequence, and the bound over a sequence no word.
    #[test]
    fn a_length_bound_stays_within_the_kinds_of_its_base() {
        const ONE: &[Value] = &[Value::integer(1)];
        let pool = empty_pool();
        let over_words = lower(
            &Schema::Refine {
                base: Arc::new(Schema::Str),
                constraints: vec![Constraint::MinLen(1)].into(),
            },
            &pool,
        )
        .expect("a bound over words lowers");
        assert!(over_words.admits(Value::word(b"a", Kind::Str)));
        assert!(!over_words.admits(Value::sequence(ONE, Kind::List)));

        let over_lists = lower(
            &Schema::Refine {
                base: Arc::new(Schema::list(SeqShape::homogeneous(Schema::ANYTHING))),
                constraints: vec![Constraint::MinLen(1)].into(),
            },
            &pool,
        )
        .expect("a bound over sequences lowers");
        assert!(over_lists.admits(Value::sequence(ONE, Kind::List)));
        assert!(!over_lists.admits(Value::word(b"a", Kind::Str)));
        assert!(!over_lists.admits(Value::sequence(ONE, Kind::Tuple)));
    }
}
