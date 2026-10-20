//! Equality of two compiled validators, modulo the constant pools they index.
//!
//! A schema's leaves name their constants by *pool slot*, and a slot is
//! construction order: `Literal[1, 2]` and `Literal[2, 1]` build the same two
//! nodes over pools that hold the same two values the other way round.
//! Comparing slot for slot answers "were these built the same way", and the
//! question `==` is asked is "are these the same schema" -- so the comparison
//! here reads *through* the slots to the values, and matches a union's members
//! and a refinement's constraints as the sets they are rather than as lists.
//!
//! A reference is the same accident one level up. A `Ref` names a recursive
//! definition by its index in the validator's table, and the table is numbered
//! in the order construction met the definitions, so `union(r1, r2)` and
//! `union(r2, r1)`, or a record whose two recursive fields were written in
//! either order, hold one schema under two numberings. A reference is read
//! through to the definition it names, and the comparison pairs the two
//! tables' definitions one to one ([`Renaming`]): equality up to a renaming of
//! the definitions, which is the equality terms with binders have everywhere
//! else -- a `recursive` definition binds its back edge, and the index is the
//! binder's name. Composing two validators places one's definitions in the
//! other's table by the same pairing ([`placement`]).
//!
//! What this does **not** do is decide anything. Two schemas that denote one set
//! by a theorem -- `int | ~int` and `anything`, a record and a mapping that
//! admit the same dicts -- are not equal here, and `is_equivalent` is the
//! question for that. This is the syntactic equality the constructors settle,
//! read modulo an accident of construction order.
//!
//! A constant is compared and hashed by its own `__eq__` and `__hash__`. One
//! raising an ordinary exception reads as unequal and adds nothing to the
//! digest; a fatal signal propagates, as it does from every other question
//! valgebra asks of user code.

use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use valgebra_core::{Attribute, Constraint, Field, MapClause, Schema};

use crate::errors::unless_fatal;

/// One side of a comparison: the schema's definitions and the pool it indexes.
pub(crate) struct Side<'a> {
    pub(crate) definitions: &'a [Schema],
    pub(crate) pool: &'a [Py<PyAny>],
}

/// Whether two compiled schemas are the same schema, modulo their pools and
/// the numbering of their definitions.
///
/// The definitions are compared where a reference reaches them, not as two
/// lists zipped by index: a definition no reference reaches denotes nothing
/// the schema does, and the validator drops it at construction
/// (`valgebra_core::pruned`), so the ones a comparison of the two trees pairs
/// are all there are.
///
/// Two tables in one order are read first, each definition against the one
/// at its own index: the renaming is the identity, every reference meets its
/// pair already made, and no body is compared inside another, so the reading
/// is as deep as one tree and as long as the two. Only where that fails is a
/// renaming searched for: the definitions are paired as composition places
/// them ([`Comparison::pair_definitions`]), every one of them or none, and the
/// two trees compared under the pairing.
pub(crate) fn schemas_equal(
    py: Python<'_>,
    left_schema: &Schema,
    left: &Side<'_>,
    right_schema: &Schema,
    right: &Side<'_>,
) -> PyResult<bool> {
    let comparison = |renaming: Renaming| Comparison {
        constants: Constants::ByValue(py),
        left,
        right,
        searching: !renaming.identity,
        renaming: RefCell::new(renaming),
        depth: Cell::new(0),
        steps: Cell::new(0),
    };
    if let Some(identity) = Renaming::identity(left, right) {
        let in_order = comparison(identity);
        if equal(&in_order, left_schema, right_schema)?
            && every(left.definitions.iter().zip(right.definitions), |(a, b)| {
                equal(&in_order, a, b)
            })?
        {
            return Ok(true);
        }
    }
    // With fewer than two definitions the identity is the only renaming, and
    // it has answered.
    if left.definitions.len() < 2 {
        return Ok(false);
    }
    // A renaming pairs every definition, so two tables of different lengths
    // have none.
    if left.definitions.len() != right.definitions.len() {
        return Ok(false);
    }
    if !same_bodies(py, left, right)? {
        return Ok(false);
    }
    let search = comparison(Renaming::between(left, right));
    search.pair_definitions()?;
    if search.renaming.borrow().to_right.contains(&None) {
        return Ok(false);
    }
    equal(&search, left_schema, right_schema)
}

/// Whether the two tables' bodies agree as multisets, each digested with its
/// references read as references and nothing more.
///
/// What any renaming needs: it pairs every definition with one whose body is
/// the same up to the references in it, which such a digest cannot see, and
/// the tables hold no definition a reference does not reach. Two tables that
/// fail this have no renaming to search for, so the search is not run.
fn same_bodies(py: Python<'_>, left: &Side<'_>, right: &Side<'_>) -> PyResult<bool> {
    let digests = |side: &Side<'_>| -> PyResult<Vec<u64>> {
        let bodies = Side {
            definitions: &[],
            pool: side.pool,
        };
        let mut digests = side
            .definitions
            .iter()
            .map(|body| {
                let mut hasher = DefaultHasher::new();
                hash_shape(py, body, &bodies, &mut hasher)?;
                Ok(hasher.finish())
            })
            .collect::<PyResult<Vec<u64>>>()?;
        digests.sort_unstable();
        Ok(digests)
    };
    Ok(digests(left)? == digests(right)?)
}

/// Which definition of `target` each definition of `inner` is, up to a
/// renaming of the definitions, where `target` holds one.
///
/// Both tables index one pool -- `inner` has been interned into it -- so a
/// constant is read by its slot, the rule composition interns by, and nothing
/// here runs Python. Each definition of `inner` is paired with the first
/// definition of `target` it equals, and a pairing carries the definitions its
/// body names with it, so what is paired is closed under reference: a
/// definition left unpaired is one to append, and its body may name a paired
/// one.
pub(crate) fn placement(inner: &[Schema], target: &[Schema]) -> Vec<Option<usize>> {
    let (inner_side, target_side) = (
        Side {
            definitions: inner,
            pool: &[],
        },
        Side {
            definitions: target,
            pool: &[],
        },
    );
    let comparison = Comparison {
        constants: Constants::BySlot,
        left: &inner_side,
        right: &target_side,
        renaming: RefCell::new(Renaming::between(&inner_side, &target_side)),
        depth: Cell::new(0),
        steps: Cell::new(0),
        searching: true,
    };
    // Reading constants by slot raises nothing, so an error cannot arrive;
    // were one to, the definitions left unpaired would be appended.
    let _paired = comparison.pair_definitions();
    comparison.renaming.into_inner().to_right
}

/// How a comparison reads the constant behind a pool slot.
enum Constants<'py> {
    /// Through the slot to the object, compared the way a literal reads one:
    /// the two sides index two pools, and `==` asks this.
    ByValue(Python<'py>),
    /// By the slot alone: the two sides index one pool, which holds one slot
    /// per constant by the rule composition interns by.
    BySlot,
}

/// One comparison: how it reads constants, the two sides, and the definitions
/// it has paired so far.
struct Comparison<'a, 'py> {
    constants: Constants<'py>,
    left: &'a Side<'a>,
    right: &'a Side<'a>,
    renaming: RefCell<Renaming>,
    /// How many levels of [`equal`] are open, held under
    /// [`MAX_COMPARISON_DEPTH`].
    depth: Cell<usize>,
    /// How many levels it has opened in all, held under
    /// [`MAX_COMPARISON_STEPS`].
    steps: Cell<usize>,
    /// Whether it makes pairs. The reading in one order makes none -- every
    /// reference meets its pair at its own index -- so it follows no reference
    /// into a body, is as deep as one tree, and has nothing to take back: it
    /// keeps neither the counts the two bounds read nor a mark per candidate.
    searching: bool,
}

/// The most levels of [`equal`] one comparison holds open at once.
///
/// A level is a native stack frame, and a reference that meets a definition
/// not yet paired compares that definition's body inside the frame that met
/// it, so the levels a comparison needs are the bodies such references reach,
/// added up: a tower of a hundred fixpoints, each naming the one around it and
/// nesting a hundred levels, asks ten thousand, which neither construction
/// bound caps on its own. Past the bound the comparison answers no -- two
/// schemas, the direction `==` errs in, and a definition composition appends
/// rather than places -- where it would otherwise overflow the thread it runs
/// on.
///
/// What it costs is measured on the wheel that ships: held at the bound through
/// a union at every level, the deepest frame chain a level takes, the `CPython`
/// 3.14 profile-guided wheel's comparison runs on a 462 KiB thread and not on
/// less, inside the 1 MiB that is the least stack any thread `CPython` creates
/// has (`test_a_comparison_fits_the_documented_stack`).
const MAX_COMPARISON_DEPTH: usize = 384;

/// The most levels of [`equal`] one comparison opens in all.
///
/// Matching a set's members tries a member against every candidate its digest
/// does not tell apart, and the digest reads a reference's definition one level
/// deep, so members that part only past a reference are each tried in turn,
/// every try a comparison of its own; nested, the tries multiply. Two equal
/// schemas read member by member open a level per node of each tree and each
/// definition, which the validator's node bound caps a side, so the bound is
/// four times that one -- room for every such reading -- and stops a search
/// that has spent it, answering no as the depth bound does. Written as a
/// number because the validator reads this module, and asserted against its
/// bound there.
pub(crate) const MAX_COMPARISON_STEPS: usize = 400_000;

/// One open level of a comparison, closed when dropped; a comparison that
/// makes no pairs counts none.
struct Level<'a>(Option<&'a Cell<usize>>);

impl Drop for Level<'_> {
    fn drop(&mut self) {
        if let Some(depth) = self.0 {
            depth.set(depth.get().saturating_sub(1));
        }
    }
}

impl Comparison<'_, '_> {
    /// Open a level, or refuse where [`MAX_COMPARISON_DEPTH`] are open or
    /// [`MAX_COMPARISON_STEPS`] have been.
    fn descend(&self) -> Option<Level<'_>> {
        if !self.searching {
            return Some(Level(None));
        }
        let (depth, steps) = (self.depth.get(), self.steps.get());
        if depth >= MAX_COMPARISON_DEPTH || steps >= MAX_COMPARISON_STEPS {
            return None;
        }
        self.depth.set(depth + 1);
        self.steps.set(steps + 1);
        Some(Level(Some(&self.depth)))
    }

    /// Whether the left side's slot `a` and the right side's slot `b` name one
    /// constant.
    fn same_constant(&self, a: usize, b: usize) -> PyResult<bool> {
        match self.constants {
            Constants::ByValue(py) => {
                objects_equal(py, self.left.pool.get(a), self.right.pool.get(b))
            }
            Constants::BySlot => Ok(a == b),
        }
    }

    /// Pair each left definition, in the order its table holds them, with the
    /// first free right definition it equals under the pairs made so far.
    ///
    /// The order is the one construction numbers a table in: a definition
    /// takes its index once its body is built, after every definition the body
    /// built, so a body names lower indices -- paired already -- and meets the
    /// higher ones only where an inner fixpoint names an outer one. A body is
    /// then read once and not inside another, which is what keeps a chain of
    /// definitions from adding its bodies' depths up; where a reference does
    /// meet one unpaired, its body is read where it is met, under the bounds
    /// [`MAX_COMPARISON_DEPTH`] and [`MAX_COMPARISON_STEPS`] set.
    ///
    /// The first it equals, not the one a later reading needs: in a table
    /// holding two definitions equal to each other up to a renaming, the
    /// pairing can take the other, and the answer is no, the direction `==`
    /// errs in. Construction leaves no such pair but where an alias's inner
    /// fixpoint names the alias: composing places the second definition on the
    /// first, and an alias definition already held is not added again.
    fn pair_definitions(&self) -> PyResult<()> {
        for definition in 0..self.left.definitions.len() {
            if !matches!(self.renaming.borrow().to_right.get(definition), Some(None)) {
                continue;
            }
            for candidate in 0..self.right.definitions.len() {
                let free = matches!(self.renaming.borrow().to_left.get(candidate), Some(None));
                if free && self.attempt_references(definition, candidate)? {
                    break;
                }
            }
        }
        Ok(())
    }

    /// [`references_equal`], taking back what a refusal paired, as
    /// [`attempt`](Self::attempt) does.
    fn attempt_references(&self, a: usize, b: usize) -> PyResult<bool> {
        let mark = self.renaming.borrow().paired.len();
        let matched = references_equal(self, a, b)?;
        if !matched {
            self.renaming.borrow_mut().undo_to(mark);
        }
        Ok(matched)
    }

    /// [`equal`], taking back every pair of definitions it made when it
    /// answers no: what a candidate that failed to match leaves behind must
    /// not decide what the next candidate may pair.
    fn attempt(&self, a: &Schema, b: &Schema) -> PyResult<bool> {
        if !self.searching {
            return equal(self, a, b);
        }
        let mark = self.renaming.borrow().paired.len();
        let matched = equal(self, a, b)?;
        if !matched {
            self.renaming.borrow_mut().undo_to(mark);
        }
        Ok(matched)
    }
}

/// Which right definition each left definition stands for, and back.
///
/// The pairing is one to one, so the renaming it states is a renaming, and two
/// definitions on one side never stand for one on the other. A pair is made
/// the first time a reference meets one, *before* the two bodies are compared,
/// which is what ends a cycle: the back edge inside a body meets the pair that
/// is being compared and answers yes, the coinductive reading of a recursive
/// definition. Kept in the order made, so a failed candidate's pairs can be
/// taken back ([`Comparison::attempt`]).
struct Renaming {
    /// Whether this is the identity, which pairs each index with itself and
    /// holds no table for it.
    identity: bool,
    to_right: Vec<Option<usize>>,
    to_left: Vec<Option<usize>>,
    paired: Vec<usize>,
}

/// What meeting a pair of references found.
enum Paired {
    /// The two were paired already: the same definition under the renaming.
    Before,
    /// The two are paired now, and their bodies are to be compared.
    Now,
    /// One of the two stands for another definition, so they are not one.
    Apart,
}

impl Renaming {
    fn between(left: &Side<'_>, right: &Side<'_>) -> Self {
        Self {
            identity: false,
            to_right: vec![None; left.definitions.len()],
            to_left: vec![None; right.definitions.len()],
            paired: Vec::new(),
        }
    }

    /// Every definition paired with the one at its own index, where the two
    /// tables are as long.
    fn identity(left: &Side<'_>, right: &Side<'_>) -> Option<Self> {
        (left.definitions.len() == right.definitions.len()).then(|| Self {
            identity: true,
            to_right: Vec::new(),
            to_left: Vec::new(),
            paired: Vec::new(),
        })
    }

    fn pair(&mut self, left: usize, right: usize) -> Paired {
        if self.identity {
            return if left == right {
                Paired::Before
            } else {
                Paired::Apart
            };
        }
        let (Some(to_right), Some(to_left)) = (self.to_right.get(left), self.to_left.get(right))
        else {
            // An index past its table names a definition the schema does not
            // have, which the frontend could not have built.
            return Paired::Apart;
        };
        match (*to_right, *to_left) {
            (Some(paired), _) if paired == right => Paired::Before,
            (None, None) => {
                set(&mut self.to_right, left, Some(right));
                set(&mut self.to_left, right, Some(left));
                self.paired.push(left);
                Paired::Now
            }
            _ => Paired::Apart,
        }
    }

    /// Take back every pair made after the first `mark`.
    fn undo_to(&mut self, mark: usize) {
        for left in self.paired.drain(mark..) {
            let right = self.to_right.get_mut(left).and_then(Option::take);
            if let Some(right) = right {
                set(&mut self.to_left, right, None);
            }
        }
    }
}

/// Write `value` at `at`, where the table has the slot.
fn set(table: &mut [Option<usize>], at: usize, value: Option<usize>) {
    if let Some(slot) = table.get_mut(at) {
        *slot = value;
    }
}

/// Whether `test` holds of every item, stopping at the first it fails or raises.
fn every<T>(
    items: impl IntoIterator<Item = T>,
    mut test: impl FnMut(T) -> PyResult<bool>,
) -> PyResult<bool> {
    for item in items {
        if !test(item)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether two pooled objects are one constant.
///
/// Identity first, so a validator equals itself even when it pools a value that
/// is not equal to itself -- a `nan` literal is the case, and comparing it by
/// `==` alone would make `v == v` false. Type then value after that, because the
/// literal rule is typed: `1` and `True` are equal in Python and name different
/// singletons here.
fn objects_equal(
    py: Python<'_>,
    left: Option<&Py<PyAny>>,
    right: Option<&Py<PyAny>>,
) -> PyResult<bool> {
    match (left, right) {
        (Some(a), Some(b)) => {
            let (a, b) = (a.bind(py), b.bind(py));
            Ok(a.is(b) || (a.get_type().is(b.get_type()) && unless_fatal(a.eq(b), py, false)?))
        }
        // A slot that is not in the pool is a schema the frontend could not have
        // built. Two of them are not evidence of anything, so they are not equal.
        _ => Ok(false),
    }
}

/// Whether every member of one list matches a distinct member of the other.
///
/// The lists are a union's members, a map's clauses or a refinement's
/// constraints: each is a *set* written as a list, and construction sorts them
/// by a key that reads a pool slot, so equal sets can be written in different
/// orders. Each member is looked for among the other list's members of its
/// digest, in the order they are written, and takes the first that equals it.
///
/// A digest consistent with equality -- a member and its match hash alike,
/// which [`hash_shape`] promises -- leaves the buckets as narrow as the members
/// are distinct, so two spellings of a wide `Literal` match with one comparison
/// a member whatever their orders, where a scan is quadratic in the width. A
/// digest answering one figure for every member leaves one bucket, which is the
/// search from the front, and the right cost for the short lists.
fn same_multiset_by<T>(
    items: &[T],
    others: &[T],
    mut equal_item: impl FnMut(&T, &T) -> PyResult<bool>,
    mut digest_item: impl FnMut(&T) -> PyResult<u64>,
    mut digest_other: impl FnMut(&T) -> PyResult<u64>,
) -> PyResult<bool> {
    if items.len() != others.len() {
        return Ok(false);
    }
    let mut buckets: FxHashMap<u64, Vec<&T>> = FxHashMap::default();
    for other in others {
        buckets.entry(digest_other(other)?).or_default().push(other);
    }
    for item in items {
        let Some(bucket) = buckets.get_mut(&digest_item(item)?) else {
            return Ok(false);
        };
        let mut found = None;
        for (at, other) in bucket.iter().enumerate() {
            #[cfg(test)]
            tests::scanned();
            if equal_item(item, other)? {
                found = Some(at);
                break;
            }
        }
        let Some(at) = found else {
            return Ok(false);
        };
        // In order, so the members left are searched as they are written.
        bucket.remove(at);
    }
    Ok(true)
}

/// [`same_multiset_by`] with one bucket: the search from the front, for a
/// map's clauses and a refinement's constraints, lists short enough that it is
/// their cost.
fn same_multiset<T>(
    items: &[T],
    others: &[T],
    equal_item: impl FnMut(&T, &T) -> PyResult<bool>,
) -> PyResult<bool> {
    same_multiset_by(items, others, equal_item, |_| Ok(0), |_| Ok(0))
}

/// [`same_multiset_by`] over a union's or a meet's members, each digested
/// through its own side's pool: the list a wide `Literal` makes long.
///
/// Two validators built the same way hold their members in one order, which
/// is the common case and needs no digest: the members are matched in place
/// first, and only those past the first one out of place are digested.
fn same_members(cx: &Comparison<'_, '_>, members: &[Schema], others: &[Schema]) -> PyResult<bool> {
    if members.len() != others.len() {
        return Ok(false);
    }
    let mut alike = members.len();
    for (at, (member, other)) in members.iter().zip(others).enumerate() {
        if !cx.attempt(member, other)? {
            alike = at;
            break;
        }
    }
    let (rest, other_rest) = (
        members.get(alike..).unwrap_or_default(),
        others.get(alike..).unwrap_or_default(),
    );
    // The pair the pass above refused is refused again unread: the attempt took
    // back what it paired, so a second reading starts where the first did and
    // answers what it did. Read again, a refusal past a reference -- where a
    // digest no longer tells two members apart -- is read twice at every level
    // of nesting above it, which is exponential in the nesting.
    let refused = (rest.first(), other_rest.first());
    let attempt = |a: &Schema, b: &Schema| {
        if let (Some(member), Some(other)) = refused
            && std::ptr::eq(a, member)
            && std::ptr::eq(b, other)
        {
            return Ok(false);
        }
        cx.attempt(a, b)
    };
    // A digest reads each constant's own hash, which is Python's to answer; a
    // comparison by slot reads none, and searches from the front.
    let Constants::ByValue(py) = cx.constants else {
        return same_multiset(rest, other_rest, attempt);
    };
    let digest = |member: &Schema, side: &Side<'_>| {
        let mut hasher = DefaultHasher::new();
        hash_shape(py, member, side, &mut hasher)?;
        Ok(hasher.finish())
    };
    same_multiset_by(
        rest,
        other_rest,
        attempt,
        |member| digest(member, cx.left),
        |member| digest(member, cx.right),
    )
}

fn fields_equal(cx: &Comparison<'_, '_>, a: &[Field], b: &[Field]) -> PyResult<bool> {
    // Ordered rather than matched: construction sorts a record's fields by name,
    // which is a key no pool slot reaches, so two equal records are already in
    // the same order.
    Ok(a.len() == b.len()
        && every(a.iter().zip(b), |(x, y)| {
            Ok(x.name == y.name && x.required == y.required && equal(cx, &x.schema, &y.schema)?)
        })?)
}

fn attributes_equal(cx: &Comparison<'_, '_>, a: &[Attribute], b: &[Attribute]) -> PyResult<bool> {
    // Ordered, for the reason a record's fields are.
    Ok(a.len() == b.len()
        && every(a.iter().zip(b), |(x, y)| {
            Ok(x.name == y.name && equal(cx, &x.schema, &y.schema)?)
        })?)
}

/// Whether two clauses are one, taking back the definitions a refusal paired,
/// as [`Comparison::attempt`] does: a clause is a candidate in a search.
fn clause_equal(cx: &Comparison<'_, '_>, a: &MapClause, b: &MapClause) -> PyResult<bool> {
    if !cx.searching {
        return Ok(equal(cx, &a.key, &b.key)? && equal(cx, &a.value, &b.value)?);
    }
    let mark = cx.renaming.borrow().paired.len();
    let matched = equal(cx, &a.key, &b.key)? && equal(cx, &a.value, &b.value)?;
    if !matched {
        cx.renaming.borrow_mut().undo_to(mark);
    }
    Ok(matched)
}

fn constraint_equal(cx: &Comparison<'_, '_>, a: &Constraint, b: &Constraint) -> PyResult<bool> {
    // The discriminants agree before anything is read: `Ge(0)` and `Le(0)` name
    // one constant and bound nothing alike.
    if core::mem::discriminant(a) != core::mem::discriminant(b) {
        return Ok(false);
    }
    let pooled = |i: usize, j: usize| cx.same_constant(i, j);
    match (a, b) {
        (Constraint::Ge(i), Constraint::Ge(j))
        | (Constraint::Gt(i), Constraint::Gt(j))
        | (Constraint::Le(i), Constraint::Le(j))
        | (Constraint::Lt(i), Constraint::Lt(j))
        | (Constraint::MultipleOf(i), Constraint::MultipleOf(j)) => pooled(i.get(), j.get()),
        (Constraint::Predicate(i), Constraint::Predicate(j)) => pooled(i.get(), j.get()),
        // A length and a pattern carry their operand inline, so the derived
        // comparison is the whole of it.
        (x, y) => Ok(x == y),
    }
}

/// Whether two references name one definition: read through the index to the
/// definition the first time the pair meets, and by the pairing after.
fn references_equal(cx: &Comparison<'_, '_>, a: usize, b: usize) -> PyResult<bool> {
    let met = cx.renaming.borrow_mut().pair(a, b);
    Ok(match met {
        Paired::Before => true,
        Paired::Apart => false,
        Paired::Now => match (cx.left.definitions.get(a), cx.right.definitions.get(b)) {
            (Some(x), Some(y)) => equal(cx, x, y)?,
            _ => false,
        },
    })
}

/// The comparison itself, one node at a time.
fn equal(cx: &Comparison<'_, '_>, left_schema: &Schema, right_schema: &Schema) -> PyResult<bool> {
    let Some(_level) = cx.descend() else {
        return Ok(false);
    };
    let recur = |a: &Schema, b: &Schema| equal(cx, a, b);
    Ok(match (left_schema, right_schema) {
        // The pooled leaves: read through the slot to the object it names.
        (Schema::Literal(a), Schema::Literal(b)) => cx.same_constant(a.get(), b.get())?,
        (Schema::Instance(a), Schema::Instance(b)) => cx.same_constant(a.get(), b.get())?,
        (Schema::Ref(a), Schema::Ref(b)) => references_equal(cx, a.get(), b.get())?,
        // The set-shaped lists: matched rather than zipped.
        (Schema::Union(a), Schema::Union(b))
        | (Schema::Intersection(a), Schema::Intersection(b)) => same_members(cx, a, b)?,
        (Schema::Complement(a), Schema::Complement(b)) => recur(a, b)?,
        (
            Schema::Coll {
                container: a_kind,
                element: a,
            },
            Schema::Coll {
                container: b_kind,
                element: b,
            },
        ) => a_kind == b_kind && recur(a, b)?,
        (
            Schema::Seq {
                container: a_kind,
                shape: a,
            },
            Schema::Seq {
                container: b_kind,
                shape: b,
            },
        ) => {
            // A sequence's elements are positional, so this one is a zip.
            a_kind == b_kind
                && a.prefix.len() == b.prefix.len()
                && every(a.prefix.iter().zip(b.prefix.iter()), |(x, y)| recur(x, y))?
                && match (&a.tail, &b.tail) {
                    (Some(x), Some(y)) => recur(x, y)?,
                    (None, None) => true,
                    _ => false,
                }
        }
        (
            Schema::KeyedMap {
                fields: a_fields,
                defaults: a_defaults,
            },
            Schema::KeyedMap {
                fields: b_fields,
                defaults: b_defaults,
            },
        ) => {
            fields_equal(cx, a_fields, b_fields)?
                && same_multiset(a_defaults, b_defaults, |x, y| clause_equal(cx, x, y))?
        }
        (Schema::AttrRecord { fields: a }, Schema::AttrRecord { fields: b }) => {
            attributes_equal(cx, a, b)?
        }
        (
            Schema::Refine {
                base: a_base,
                constraints: a,
            },
            Schema::Refine {
                base: b_base,
                constraints: b,
            },
        ) => recur(a_base, b_base)? && same_multiset(a, b, |x, y| constraint_equal(cx, x, y))?,
        // What carries no pool slot, no definition index and no set-shaped
        // list: the derived comparison is the whole of it -- the scalars, the
        // two bounds, and the build-time marker -- and it answers `false`
        // against another variant. Named rather than caught by `(a, b)`, so a variant that
        // carries a slot is a compile error here: the derived comparison reads
        // a slot's index, and two pools hold one constant at two indices.
        (
            a @ (Schema::Anything(_)
            | Schema::Nothing
            | Schema::NoneType
            | Schema::Bool
            | Schema::Int
            | Schema::Float
            | Schema::Str
            | Schema::Bytes
            | Schema::SelfRef(_)),
            b,
        ) => a == b,
        // A variant with an arm above, met by another variant.
        (
            Schema::Literal(_)
            | Schema::Instance(_)
            | Schema::Ref(_)
            | Schema::Union(_)
            | Schema::Intersection(_)
            | Schema::Complement(_)
            | Schema::Coll { .. }
            | Schema::Seq { .. }
            | Schema::KeyedMap { .. }
            | Schema::AttrRecord { .. }
            | Schema::Refine { .. },
            _,
        ) => false,
    })
}

/// Fold the constant a pool slot names into the digest, where it has a hash.
///
/// Equality reads the *value* behind a slot, so the hash must too or two
/// schemas that differ only in a constant land on one bucket: `Literal[1]`
/// through `Literal[1000]` would be one hash, and a dictionary keyed by
/// validators -- the reason `__hash__` exists -- would degenerate into a list.
///
/// [`objects_equal`] is `a is b`, or `type(a) is type(b)` and `a == b`, so
/// folding the constant's own hash beside its type's keeps the two consistent
/// for every type whose `__eq__` and `__hash__` agree -- which is the contract
/// Python's own dictionaries already require.
///
/// A constant with no hash contributes nothing and the shape stands alone,
/// which is what the one case needing it asks for: a validator is usable as a
/// key whatever it pools, and refusing to hash would be worse than a
/// collision.
fn hash_constant<H: Hasher>(
    py: Python<'_>,
    slot: usize,
    pool: &[Py<PyAny>],
    hasher: &mut H,
) -> PyResult<()> {
    let Some(object) = pool.get(slot) else {
        return Ok(());
    };
    let bound = object.bind(py);
    if let Some(hash) = unless_fatal(bound.hash().map(Some), py, None)? {
        hash.hash(hasher);
        if let Some(kind) = unless_fatal(bound.get_type().hash().map(Some), py, None)? {
            kind.hash(hasher);
        }
    }
    Ok(())
}

/// Digest a schema, reading the constants its pool slots name.
///
/// The companion of [`schemas_equal`]: every node equality separates is a node
/// this may separate, and the two are consistent where the constants behave.
/// What it does *not* read is a slot index or a definition index -- both are
/// construction order, and two spellings of one schema pool their constants in
/// different slots and number their definitions apart.
pub(crate) fn hash_shape<H: Hasher>(
    py: Python<'_>,
    schema: &Schema,
    side: &Side<'_>,
    hasher: &mut H,
) -> PyResult<()> {
    let pool = side.pool;
    core::mem::discriminant(schema).hash(hasher);
    match schema {
        // The lists whose order is not part of the schema fold commutatively,
        // so two spellings of one set hash alike.
        Schema::Union(members) | Schema::Intersection(members) => {
            members.len().hash(hasher);
            unordered(members, hasher, |member, one| {
                hash_shape(py, member, side, one)
            })?;
        }
        Schema::Complement(inner) => hash_shape(py, inner, side, hasher)?,
        Schema::Coll { container, element } => {
            container.hash(hasher);
            hash_shape(py, element, side, hasher)?;
        }
        Schema::Seq { container, shape } => {
            container.hash(hasher);
            shape.prefix.len().hash(hasher);
            for element in shape.prefix.iter() {
                hash_shape(py, element, side, hasher)?;
            }
            shape.tail.is_some().hash(hasher);
            if let Some(tail) = &shape.tail {
                hash_shape(py, tail, side, hasher)?;
            }
        }
        Schema::KeyedMap { fields, defaults } => {
            hash_fields(py, fields, side, hasher)?;
            defaults.len().hash(hasher);
            unordered(defaults, hasher, |clause, one| {
                hash_shape(py, &clause.key, side, one)?;
                hash_shape(py, &clause.value, side, one)
            })?;
        }
        Schema::AttrRecord { fields } => hash_attributes(py, fields, side, hasher)?,
        Schema::Refine { base, constraints } => {
            hash_shape(py, base, side, hasher)?;
            constraints.len().hash(hasher);
            unordered(constraints, hasher, |constraint, one| {
                core::mem::discriminant(constraint).hash(one);
                // The bounds a slot does not reach are part of the shape; the
                // ones it does are left to equality.
                match constraint {
                    Constraint::MinLen(n) | Constraint::MaxLen(n) => n.hash(one),
                    Constraint::Regex(pattern) => pattern.hash(one),
                    // A bound names its operand through the pool, and equality
                    // reads it, so `Ge(0)` and `Ge(1)` must not hash alike.
                    Constraint::Ge(i)
                    | Constraint::Gt(i)
                    | Constraint::Le(i)
                    | Constraint::Lt(i)
                    | Constraint::MultipleOf(i) => hash_constant(py, i.get(), pool, one)?,
                    Constraint::Predicate(i) => hash_constant(py, i.get(), pool, one)?,
                }
                Ok(())
            })?;
        }
        // A reference by the definition it names, read one level deep: the
        // body's own references stop there, which keeps the digest finite on a
        // cycle and blind to a numbering equality reads through.
        Schema::Ref(index) => {
            if let Some(body) = side.definitions.get(index.get()) {
                let shallow = Side {
                    definitions: &[],
                    pool,
                };
                hash_shape(py, body, &shallow, hasher)?;
            }
        }
        // The pooled leaves: the constant behind the slot, not the slot.
        Schema::Literal(index) => hash_constant(py, index.get(), pool, hasher)?,
        Schema::Instance(index) => hash_constant(py, index.get(), pool, hasher)?,
        // The scalars, the two bounds and the build-time marker: the
        // discriminant above is the whole of their shape. Named rather than
        // caught by `_`, so a variant that carries a slot or a child is a
        // compile error here and in `equal` alike.
        Schema::Anything(_)
        | Schema::Nothing
        | Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Str
        | Schema::Bytes
        | Schema::SelfRef(_) => {}
    }
    Ok(())
}

fn hash_fields<H: Hasher>(
    py: Python<'_>,
    fields: &[Field],
    side: &Side<'_>,
    hasher: &mut H,
) -> PyResult<()> {
    // Ordered, because construction orders them by name.
    fields.len().hash(hasher);
    for field in fields {
        field.name.hash(hasher);
        field.required.hash(hasher);
        hash_shape(py, &field.schema, side, hasher)?;
    }
    Ok(())
}

fn hash_attributes<H: Hasher>(
    py: Python<'_>,
    attributes: &[Attribute],
    side: &Side<'_>,
    hasher: &mut H,
) -> PyResult<()> {
    // Ordered, for the reason a record's fields are.
    attributes.len().hash(hasher);
    for attribute in attributes {
        attribute.name.hash(hasher);
        hash_shape(py, &attribute.schema, side, hasher)?;
    }
    Ok(())
}

/// Fold each item's own digest into one that does not depend on their order.
///
/// Addition rather than exclusive-or: a pair of equal items cancels under xor,
/// and while the lists here are deduplicated sets, a hash that turns two members
/// into none is a trap laid for the next person to relax that.
fn unordered<T, H: Hasher>(
    items: &[T],
    hasher: &mut H,
    mut each: impl FnMut(&T, &mut DefaultHasher) -> PyResult<()>,
) -> PyResult<()> {
    let mut total: u64 = 0;
    for item in items {
        let mut one = DefaultHasher::new();
        each(item, &mut one)?;
        total = total.wrapping_add(one.finish());
    }
    total.hash(hasher);
    Ok(())
}

#[cfg(test)]
mod tests;

// Needs a live interpreter: the comparison reads pooled objects, which is the
// half a pure-Rust test cannot reach.
#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter;
