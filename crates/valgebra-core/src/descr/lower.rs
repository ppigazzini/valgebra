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
use crate::ir::{
    ClassIx, CollKind, ConstIx, Constraint, Field, MapClause, OperandIx, Polarity, Schema, SeqKind,
    SeqShape,
};
use crate::kind::Kind;
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
/// clock.
///
/// **This is the bound an ordinary annotation reaches first**, and it reaches
/// it by width. A record whose fields each take two types, against the union of
/// the records that fix every field, reads 22 nodes at two fields, 45 at three
/// and 96 at four -- so three fields decide and four are refused before they
/// are answered. `tests/test_completeness_ledger.py` carries all three as rows,
/// which is what makes these figures re-derivable: the row fails on the commit
/// that moves the number either way.
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

/// How many times a reference is unfolded before the descriptor is built.
///
/// One. A single unfolding puts the fixpoint's own body in front of the
/// representation, which settles every relation that turns on *what kinds* a
/// recursive schema admits -- a meet with a disjoint kind, an inclusion in a
/// wider union -- and that is the whole of what the structural rules cannot
/// read. Each further unfolding multiplies the schema this must build against
/// [`BUDGET`]'s 64 nodes, for relations nobody has asked for.
///
/// A bound like the three in [`Bounds`], and unlike them in one way worth
/// stating: those three say what a build may spend, and this says what it is
/// given to build *from*. A `Ref` has no component to land in, so without the
/// unfolding a recursive schema does not lower at all.
pub const UNFOLDS: u32 = 1;

/// [`lower`], resolving a recursive schema's references through `definitions`.
///
/// `polarity` is the side the schema is read on, and it is the caller's to
/// know: unfolding grows the set on one side and shrinks it on the other, so a
/// difference stays sound only when the two sides are unfolded in opposite
/// directions -- and only in the direction that proves it empty.
///
/// # Errors
///
/// Refuses what [`lower`] refuses. A reference `definitions` does not resolve
/// is one of those: the unfolding leaves it in place and no component holds it.
pub fn lower_unfolded(
    schema: &Schema,
    definitions: &[Schema],
    polarity: Polarity,
    pool: &dyn Constants,
) -> Option<Descr> {
    // The empty check first: it is one comparison, and it is true for every
    // schema that carries no fixpoint at all -- which is almost all of them, and
    // all of the ones on the workload the decision budget is measured over. The
    // walk that follows costs a pass over the tree, and paying it per relation
    // for a schema with no definitions was ten percent of that workload.
    if definitions.is_empty() || !schema.has_reference() {
        return lower(schema, pool);
    }
    lower(&schema.unfolded(definitions, UNFOLDS, polarity), pool)
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
/// complement nested inside another, one regular language inside another -- each
/// spend under
/// 256 units. The shapes that blow up spend tens or hundreds of thousands: a
/// record nested eight deep spends 22,806, and a union of four records nested
/// three deep minus a union of its siblings spends 170,597.
///
/// On the shapes reachable today [`DEPTH`] refuses first for anything deep, so
/// this number stops a build only where a shape is shallow and wide -- which is
/// the one nesting cannot catch, and what it is kept for.
///
/// **The figure is what a record split across a union of records costs**, which
/// is the widest shallow shape the sets are asked to decide. A two-field record
/// against the four records that fix both keys spends 1,173 units; the
/// three-field record against its eight corners spends 2,241. Both are
/// relations a caller writes and the rules decline, so both are the sets' to
/// answer, and a bound below them refuses an ordinary question. The blow-ups
/// the number exists to stop are an order of magnitude further out, so the
/// margin is real rather than nominal.
///
/// The field after that is outside on both counts: four fields against sixteen
/// corners spend 9,965 units and read 96 nodes against a [`BUDGET`] of 64, so
/// raising either bound alone leaves the refusal where it is. That is the width
/// of the decided fragment, and `docs/15-decidability.md` states it as one.
///
/// `tests/test_completeness_ledger.py` carries both shapes as rows, which is
/// what makes this figure re-derivable: the row fails on the commit that
/// lowers it.
///
/// **This bound is debt.** It counts work rather than limiting what the
/// representation can hold, and it is here for the reason the decision's own
/// budget is: a lowering repeats itself over structurally equal subtrees.
/// Construction shares those subtrees where they are built alike
/// (`ir/intern.rs`), which is what gives the memo that would replace this
/// ceiling a cheap key; the memo is the half not written.
pub const WORK: u64 = 4096;

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
/// A record's field and a mapping's literal key both name a key and say what
/// it maps to, so both become labels -- and they differ in one thing the atom
/// has to be told. A field is read *instead of* the clauses: a key a field
/// names is governed by the field alone. A literal-keyed clause is read
/// *beside* them: clauses are a disjunction, so a key the clause names is
/// admitted when it or any other clause covering that key admits the value,
/// and the label's type is the join of every clause that reads it. A clause
/// whose key is a *kind* opens that part of the key partition. What no clause
/// opens stays shut, which is what makes a record with no catch-all closed.
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
    // Every clause first -- what it names, what it opens, what it maps to --
    // because a named key's type is read off all of them.
    let mut clauses: Vec<(Vec<Label>, Vec<Option<Kind>>, Descr)> =
        Vec::with_capacity(defaults.len());
    for clause in defaults {
        let value = descend(&clause.value, pool, budget, depth)?;
        let (named, parts) = key_cover(&clause.key, pool)?;
        clauses.push((named, parts, value));
    }
    let mut opened: Vec<(Option<Kind>, Descr)> = Vec::new();
    for (index, (named, parts, value)) in clauses.iter().enumerate() {
        for label in named {
            // A field of this name is read instead of the clause.
            if fields.iter().any(|field| Label::str(&field.name) == *label) {
                continue;
            }
            // A clause names a key without requiring it: `dict[Literal["a"], V]`
            // admits the dict that carries no `a` at all. And the key is read
            // by every clause that covers it, so its type is the join.
            let mut ty = value.clone();
            for (other, (other_named, other_parts, other_value)) in clauses.iter().enumerate() {
                let covers = other != index
                    && (other_named.contains(label)
                        || other_parts.iter().any(|part| *part == Some(label.kind())));
                if covers {
                    ty = ty.union(other_value)?;
                }
            }
            labels.push((label.clone(), ty, true));
        }
        for part in parts {
            opened.push((*part, value.clone()));
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
        // A `bool` key **is** an `int` key: `{True: 1}` is a dict whose key is
        // an integer, and the walk admits it under `dict[int, V]`. Opening only
        // the `Int` part would name a set smaller than the schema denotes.
        Schema::Int => Some((Vec::new(), vec![Some(Kind::Int), Some(Kind::Bool)])),
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
        if !base.within(&[Kind::Int, Kind::Bool]) {
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
        if !base.within(&[Kind::Float]) {
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
    //
    // An integer operand past 2^53 has **no float equal to it**, and rounding it
    // to the nearest one moves the boundary whichever way that rounding went --
    // naming a set of floats the schema does not denote, in either direction. It
    // is read against its two neighbours instead: a float is `>= n` exactly when
    // it is at or above the smallest float above `n`, and `<= n` exactly when it
    // is at or below the largest float below `n`. Python compares an `int` with
    // a `float` exactly, and so does this.
    let float_bound = |constraint: &Constraint, index: &OperandIx| {
        float_bound(constraint, &pool.operand(*index)?)
    };
    let base_is_floats = base.within(&[Kind::Float]);
    match constraint {
        Constraint::Ge(index) | Constraint::Gt(index) if base_is_floats => {
            floats(float_bound(constraint, index)?)
        }
        Constraint::Le(index) | Constraint::Lt(index) if base_is_floats => {
            floats(float_bound(constraint, index)?)
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
        Constraint::MinLen(least) => {
            lengths(base, &|kind| Descr::words_at_least(*least, kind), &|kind| {
                Descr::sequences_at_least(*least, kind)
            })
        }
        Constraint::MaxLen(most) => {
            lengths(base, &|kind| Descr::words_at_most(*most, kind), &|kind| {
                Descr::sequences_at_most(*most, kind)
            })
        }
        Constraint::Regex(pattern) => words(pattern, base),
        // A callback is a leaf the core cannot read, which is what makes it
        // opaque to the procedure beside this one too.
        Constraint::Predicate(_) => None,
    }
}

/// Where an integer bound falls among the floats.
///
/// Every integer up to 2^53 is a float. Past that a bound may have *no* float
/// equal to it, and then the set it cuts is bounded by a neighbour rather than
/// by the bound itself -- which is a different set from the one the nearest
/// float would cut, in whichever direction that rounding went.
enum Straddle {
    /// A float equals the bound.
    Exact(f64),
    /// No float does; this is the smallest one above it.
    Above(f64),
    /// No float does; this is the largest one below it.
    Below(f64),
}

/// Which floats an integer bound cuts between.
///
/// `lower` picks the side: a lower bound is answered by the smallest float above
/// an inexact value, an upper bound by the largest below it, and both by the
/// value itself where it is exact.
fn straddle(value: i64, lower: bool) -> Straddle {
    #[expect(
        clippy::cast_precision_loss,
        reason = "the rounding is the thing measured: its direction picks the neighbour"
    )]
    let nearest = value as f64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an integral float inside the i64 range converts exactly"
    )]
    let round_trip = nearest as i128;
    match round_trip.cmp(&i128::from(value)) {
        core::cmp::Ordering::Equal => Straddle::Exact(nearest),
        // The nearest float overshot, so it *is* the one above.
        core::cmp::Ordering::Greater if lower => Straddle::Above(nearest),
        core::cmp::Ordering::Greater => Straddle::Below(nearest.next_down()),
        core::cmp::Ordering::Less if lower => Straddle::Above(nearest.next_up()),
        core::cmp::Ordering::Less => Straddle::Below(nearest),
    }
}

/// The floats a comparison bound admits, for a base that is the floats.
///
/// An integer operand past 2^53 has no float equal to it, and rounding it to the
/// nearest one moves the boundary whichever way that rounding went -- naming a
/// set of floats the schema does not denote, in either direction. It is read
/// against its neighbours instead: a float is `>= n` exactly when it is at or
/// above the smallest float above `n`, and `<= n` exactly when it is at or below
/// the largest float below `n`. Strictness costs nothing there, since no float
/// equals the bound for `>` or `<` to exclude. Python compares an `int` with a
/// `float` exactly, and so does this.
fn float_bound(constraint: &Constraint, operand: &Operand) -> Option<FloatSet> {
    let lower = matches!(constraint, Constraint::Ge(_) | Constraint::Gt(_));
    let strict = matches!(constraint, Constraint::Gt(_) | Constraint::Lt(_));
    let value = match *operand {
        Operand::Float(value) => Straddle::Exact(value),
        Operand::Integer(value) => straddle(value, lower),
        _ => return None,
    };
    // No float equals an inexact bound, so `>` and `>=` admit the same ones and
    // the neighbour carries what the strictness would have.
    Some(match value {
        Straddle::Exact(bound) if lower && strict => FloatSet::above(bound),
        Straddle::Exact(bound) if strict => FloatSet::below(bound),
        Straddle::Exact(bound) | Straddle::Above(bound) if lower => FloatSet::at_least(bound),
        Straddle::Exact(bound) | Straddle::Below(bound) => FloatSet::at_most(bound),
        // `lower` picks the variant `straddle` returns, so the two crossed
        // cases are unreachable and answer with the set for the side they name.
        Straddle::Above(bound) => FloatSet::at_least(bound),
    })
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
    words: &dyn Fn(Kind) -> Option<Descr>,
    sequences: &dyn Fn(Kind) -> Option<Descr>,
) -> Option<Descr> {
    const WORDS: [Kind; 2] = [Kind::Str, Kind::Bytes];
    const SEQUENCES: [Kind; 2] = [Kind::List, Kind::Tuple];

    if !base.within(&[Kind::Str, Kind::Bytes, Kind::List, Kind::Tuple]) {
        return None;
    }
    let mut whole = Descr::nothing();
    for kind in WORDS {
        if base.reaches(kind) {
            whole = whole.union(&words(kind)?)?;
        }
    }
    for kind in SEQUENCES {
        if base.reaches(kind) {
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
    if !base.within(&[Kind::Str, Kind::Bytes]) {
        return None;
    }
    let mut whole = Descr::nothing();
    for kind in [Kind::Str, Kind::Bytes] {
        if !base.reaches(kind) {
            continue;
        }
        whole = whole.union(&Descr::pattern(pattern, kind)?)?;
    }
    (!whole.is_empty()).then_some(whole)
}

#[cfg(test)]
mod tests;
