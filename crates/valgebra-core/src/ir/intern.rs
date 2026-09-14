//! One node built twice is one node.
//!
//! A schema's payloads are shared handles -- a member list, a field list, a
//! clause list, a child node -- and two structurally equal schemas built apart
//! hold two of them. Nothing is wrong with that: the two denote the same set
//! and answer every question alike. What it costs is paid twice over. The
//! second handle is a second allocation, and every later question that reaches
//! both walks them as two trees, because equality between separate nodes is
//! structural and structural equality is a walk.
//!
//! This is the table the four bounds name. `DECISION_BUDGET` and the
//! descriptor's `BUDGET`, `DEPTH` and `WORK` each say in their own doc comment
//! that a lowering repeats itself over structurally equal subtrees *because
//! they are separate nodes*. A handle a second construction gets back rather
//! than makes is not a separate node, and the repetition it causes is not
//! repeated.
//!
//! ## What makes it sound
//!
//! Two handles may be collapsed into one when nothing can tell them apart. The
//! test used here is **stricter** than the crate's structural equality, which
//! is what makes the substitution safe rather than merely plausible:
//!
//! * a child is the same child when the two handles are the same allocation
//!   ([`Arc::ptr_eq`]), never when they are two allocations that compare equal;
//! * a spelling is the same spelling when it is the same variant, where
//!   [`Spelling`]'s own `PartialEq` calls every spelling equal to every other
//!   (it is not part of the set, and `repr` reads it) -- collapsing on that
//!   equality would hand a caller back the spelling it did not write.
//!
//! Under that test, "the same" means every field is the same value or the same
//! allocation, so a caller holding either handle cannot observe which it got.
//! The test is also what makes the table *bootstrap*: a child built through
//! this module is already shared, so two parents built over equal children hold
//! pointer-equal handles and the strict test sees them as equal. Depth is
//! reached by sharing the level below, not by a walk.
//!
//! ## What makes it cheap
//!
//! The hash is over the same fields the test compares -- a variant tag, an
//! index, a pointer -- so hashing a node is constant work whatever the depth of
//! the tree beneath it, and hashing a list is constant work whatever its width:
//! it reads [`SUMMARY`] entries and the length, and leaves the rest to the test
//! that decides. The crate's derived `Hash` is the whole subtree and is not
//! used here.
//!
//! The table is per thread, fixed at [`SLOTS`] entries, and holds [`Weak`]
//! handles. Fixed and direct-mapped: a collision evicts rather than grows, so
//! the table cannot become a record of how many schemas a process has ever
//! built. Weak, so that what a table entry keeps is the *block* a dropped list
//! was written in and never the tree it held: a validator its owner drops takes
//! its children with it on the spot, and the entry left behind is a miss the
//! next time that slot is read, and freed by the entry that replaces it.
//!
//! A strong entry was measured against this one, and it shares more: the core
//! workload rebuilds the same three trees two thousand times and throws each
//! away, so with strong entries the table answers every rebuild after the first
//! and the workload allocates **1.4 MB in 16,502 blocks** against 21.8 MB in
//! 190,427. It also reads *worse* -- the core workload 1.91% down against 4.46%,
//! the record walk 2.29% up against 1.28% down -- because what a rebuild saves
//! in allocation it pays back in the comparison that agrees to share, and
//! because a table that holds trees alive holds the cache they sit in. The
//! sharing this keeps is between schemas that are both alive, which is where
//! the question that reads both of them is asked.
//!
//! A miss costs the hash, an upgrade and the allocation it was going to make
//! anyway. That is the price of the table, and it is paid by the construction
//! of a schema nothing else has built -- which is what a first build of a
//! validator is, and is why the gate for this table is two workloads and not
//! one.

use std::cell::RefCell;
use std::hash::Hasher;
use std::sync::{Arc, Weak};

use rustc_hash::FxHasher;

use super::{Clauses, Field, Fields, MapClause, Members, Schema, SeqShape, Spelling};

/// Entries per table, one table per kind of handle.
///
/// Direct-mapped, so this is the whole of the table's memory: a thousand weak
/// handles a thread, evicted by collision. Large enough to hold the working set
/// of a schema being built -- a wide record's field list, the member lists of
/// the joins under it -- and small enough that the tables are a rounding error
/// beside the schemas they point at.
///
/// **What holds this number is `perf_gate.py --decision`** (measured
/// 2026-09-14): with the table answering nothing that workload reads **65.31%
/// higher**, because it asks two relations about a record built twice and the
/// sharing is what makes the second one free. Eight times the slots moves it
/// by nothing. The *core* workload is the other side of the trade and not
/// evidence for this number: it builds trees and drops them, so every probe
/// there finds a handle whose `Arc` is already gone -- 174,066 probes, no hit
/// -- and disabling the table reads 1.97% **cheaper**. Two percent on a
/// pipeline that cannot share, for sixty-five on one that can.
const SLOTS: usize = 1024;

/// Entries of a list the hash reads before it stops.
///
/// The hash chooses a slot; the sameness test decides. A summary of a list is
/// therefore as sound as a hash of all of it -- two lists it cannot tell apart
/// land in one slot and are separated there -- and it is what keeps a wide
/// record's construction from paying per field for a table that is about to
/// compare the fields anyway. A record whose first entries and length agree
/// with another's is the collision this trades for, and a collision costs one
/// comparison that fails.
///
/// **What holds this number is `perf_gate.py --core`** (measured 2026-09-14):
/// hashing every entry instead of four reads that workload **4.47% higher**,
/// and the decision workload 0.10%, because the core one is the one that
/// builds wide lists. It was recorded against the validator-building shape
/// when it landed, and that shape has since been rebuilt to read a Python
/// spelling once per iteration: it interns one list per build and repeats
/// none, so it no longer sees this number at all.
const SUMMARY: usize = 4;

/// The address a shared handle holds, as a number to hash.
///
/// Two handles are the same handle when these agree, which is the test
/// [`Arc::ptr_eq`] makes; taking it as a number is how the hash agrees with it.
fn address<T: ?Sized>(handle: &Arc<T>) -> usize {
    Arc::as_ptr(handle).cast::<()>() as usize
}

/// Hash a node by its own fields, not by the tree beneath it.
fn hash_node(schema: &Schema, into: &mut FxHasher) {
    into.write_u8(tag(schema));
    match schema {
        // The spelling is hashed as the variant it is, so the two spellings of
        // the top land in different slots and are never collapsed.
        Schema::Anything(spelling) => into.write_u8(u8::from(matches!(spelling, Spelling::Any))),
        Schema::Nothing
        | Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Str
        | Schema::Bytes => {}
        Schema::Literal(index) => into.write_usize(index.get()),
        Schema::Seq { container, shape } => {
            into.write_u8(*container as u8);
            hash_shape(shape, into);
        }
        Schema::Coll { container, element } => {
            into.write_u8(*container as u8);
            into.write_usize(address(element));
        }
        Schema::KeyedMap { fields, defaults } => {
            into.write_usize(address(fields));
            into.write_usize(address(defaults));
        }
        Schema::Union(members) | Schema::Intersection(members) => {
            into.write_usize(address(members));
        }
        Schema::Complement(inner) => into.write_usize(address(inner)),
        Schema::Instance(index) => into.write_usize(index.get()),
        Schema::AttrRecord { fields } => into.write_usize(address(fields)),
        Schema::Refine { base, constraints } => {
            into.write_usize(address(base));
            into.write_usize(address(constraints));
        }
        Schema::Ref(id) => into.write_usize(id.get()),
        Schema::SelfRef(token) => into.write_u64(*token),
    }
}

/// The variant, as the number the hash writes for it.
///
/// Written out rather than taken from [`std::mem::discriminant`], whose value
/// is deliberately opaque and not a number. A variant added and left out of
/// this match collides with `Anything` in the hash and is refused by the
/// sameness test below, which costs sharing and decides nothing wrongly.
fn tag(schema: &Schema) -> u8 {
    match schema {
        Schema::Anything(_) => 0,
        Schema::Nothing => 1,
        Schema::NoneType => 2,
        Schema::Bool => 3,
        Schema::Int => 4,
        Schema::Float => 5,
        Schema::Str => 6,
        Schema::Bytes => 7,
        Schema::Literal(_) => 8,
        Schema::Seq { .. } => 9,
        Schema::Coll { .. } => 10,
        Schema::KeyedMap { .. } => 11,
        Schema::Union(_) => 12,
        Schema::Intersection(_) => 13,
        Schema::Complement(_) => 14,
        Schema::Instance(_) => 15,
        Schema::AttrRecord { .. } => 16,
        Schema::Refine { .. } => 17,
        Schema::Ref(_) => 18,
        Schema::SelfRef(_) => 19,
    }
}

/// Hash a sequence shape by its two handles.
fn hash_shape(shape: &SeqShape, into: &mut FxHasher) {
    into.write_usize(address(&shape.prefix));
    match &shape.tail {
        Some(tail) => into.write_usize(address(tail)),
        None => into.write_u8(0),
    }
}

/// Whether two nodes are the same node: the same variant, the same values, and
/// the same allocations under it.
fn same_node(one: &Schema, two: &Schema) -> bool {
    match (one, two) {
        (Schema::Anything(left), Schema::Anything(right)) => {
            matches!(left, Spelling::Any) == matches!(right, Spelling::Any)
        }
        (Schema::Nothing, Schema::Nothing)
        | (Schema::NoneType, Schema::NoneType)
        | (Schema::Bool, Schema::Bool)
        | (Schema::Int, Schema::Int)
        | (Schema::Float, Schema::Float)
        | (Schema::Str, Schema::Str)
        | (Schema::Bytes, Schema::Bytes) => true,
        (Schema::Literal(left), Schema::Literal(right)) => left == right,
        (
            Schema::Seq {
                container: left,
                shape: one,
            },
            Schema::Seq {
                container: right,
                shape: two,
            },
        ) => left == right && same_shape(one, two),
        (
            Schema::Coll {
                container: left,
                element: one,
            },
            Schema::Coll {
                container: right,
                element: two,
            },
        ) => left == right && Arc::ptr_eq(one, two),
        (
            Schema::KeyedMap {
                fields: one,
                defaults: left,
            },
            Schema::KeyedMap {
                fields: two,
                defaults: right,
            },
        ) => Arc::ptr_eq(one, two) && Arc::ptr_eq(left, right),
        (Schema::Union(one), Schema::Union(two))
        | (Schema::Intersection(one), Schema::Intersection(two)) => Arc::ptr_eq(one, two),
        (Schema::Complement(one), Schema::Complement(two)) => Arc::ptr_eq(one, two),
        (Schema::Instance(left), Schema::Instance(right)) => left == right,
        (Schema::AttrRecord { fields: one }, Schema::AttrRecord { fields: two }) => {
            Arc::ptr_eq(one, two)
        }
        (
            Schema::Refine {
                base: one,
                constraints: left,
            },
            Schema::Refine {
                base: two,
                constraints: right,
            },
        ) => Arc::ptr_eq(one, two) && Arc::ptr_eq(left, right),
        (Schema::Ref(left), Schema::Ref(right)) => left == right,
        (Schema::SelfRef(left), Schema::SelfRef(right)) => left == right,
        _ => false,
    }
}

/// Whether two sequence shapes are the same shape, by the same test.
fn same_shape(one: &SeqShape, two: &SeqShape) -> bool {
    Arc::ptr_eq(&one.prefix, &two.prefix)
        && match (&one.tail, &two.tail) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            (None, None) => true,
            _ => false,
        }
}

/// A field is the same field when its name reads the same and its schema is the
/// same node.
///
/// The name is compared by what it says rather than by where it is: two records
/// built from two parsed annotations carry two `"id"` strings, and a field list
/// that shared nothing because of it would share nothing at all. `Arc<str>`
/// compares its bytes and takes the same allocation as an early yes, so the
/// common case here is a pointer test anyway.
fn same_field(one: &Field, two: &Field) -> bool {
    one.required == two.required && one.name == two.name && same_node(&one.schema, &two.schema)
}

/// Hash a field by what [`same_field`] compares.
fn hash_field(field: &Field, into: &mut FxHasher) {
    into.write(field.name.as_bytes());
    into.write_u8(u8::from(field.required));
    hash_node(&field.schema, into);
}

/// Whether two clauses are the same clause.
fn same_clause(one: &MapClause, two: &MapClause) -> bool {
    same_node(&one.key, &two.key) && same_node(&one.value, &two.value)
}

/// Hash a clause by what [`same_clause`] compares.
fn hash_clause(clause: &MapClause, into: &mut FxHasher) {
    hash_node(&clause.key, into);
    hash_node(&clause.value, into);
}

/// The slot a hash reads, which is the whole of the table's addressing.
fn slot_of(hash: u64) -> usize {
    // Folded to the table's width first, so what the cast carries is a slot
    // rather than a hash: `SLOTS` fits a `usize` on every target this builds
    // for, and the remainder is smaller than it.
    usize::try_from(hash % SLOTS as u64).unwrap_or(0)
}

/// One kind's table: weak handles in fixed slots, per thread.
struct Table<T: ?Sized>(Vec<Option<Weak<T>>>);

impl<T: ?Sized> Table<T> {
    fn new() -> Table<T> {
        Table((0..SLOTS).map(|_| None).collect())
    }

    /// The handle in `hash`'s slot, when it is the one wanted.
    ///
    /// Every step is asked rather than assumed, down to the slot: the table is
    /// as long as the slots a hash can name, and a table that is not says the
    /// same thing an empty slot says -- build it -- rather than ending the
    /// process over a cache.
    fn hit(&self, hash: u64, same: impl Fn(&T) -> bool) -> Option<Arc<T>> {
        let held = self.0.get(slot_of(hash))?.as_ref()?.upgrade()?;
        same(&held).then_some(held)
    }

    /// Leave a handle in `hash`'s slot for the next caller to find.
    fn put(&mut self, hash: u64, made: &Arc<T>) {
        if let Some(slot) = self.0.get_mut(slot_of(hash)) {
            *slot = Some(Arc::downgrade(made));
        }
    }
}

// The four tables, one per shape of shared payload.
//
// **Nothing called while a table is borrowed may reach back into one.** Each of
// the four functions below probes and stores under a single borrow -- the probe
// and the store are one question, and splitting them paid two thread-local
// lookups and two borrow pairs on the miss path, which is every call in a pass
// that builds new nodes. What that costs is the rule above: the rejected
// payload is built and dropped inside the borrow, so a `Drop` on `Schema`,
// `Field` or `MapClause` that interned anything would panic on a borrow already
// held. None does today; each drops handles and nothing else, and any that
// stops being true has to split its table's access again.
thread_local! {
    static NODES: RefCell<Table<Schema>> = RefCell::new(Table::new());
    static MEMBERS: RefCell<Table<[Schema]>> = RefCell::new(Table::new());
    static FIELDS: RefCell<Table<[Field]>> = RefCell::new(Table::new());
    static CLAUSES: RefCell<Table<[MapClause]>> = RefCell::new(Table::new());
}

/// The shared handle for a child node.
pub(super) fn node(schema: Schema) -> Arc<Schema> {
    let mut state = FxHasher::default();
    hash_node(&schema, &mut state);
    let hash = state.finish();
    NODES.with_borrow_mut(move |table| {
        if let Some(held) = table.hit(hash, |held| same_node(held, &schema)) {
            return held;
        }
        let made = Arc::new(schema);
        table.put(hash, &made);
        made
    })
}

/// The shared handle for a member list.
pub(super) fn members(items: &[Schema]) -> Members {
    let mut state = FxHasher::default();
    state.write_usize(items.len());
    for item in items.iter().take(SUMMARY) {
        hash_node(item, &mut state);
    }
    let hash = state.finish();
    MEMBERS.with_borrow_mut(|table| {
        let held = table.hit(hash, |held: &[Schema]| {
            held.len() == items.len()
                && held.iter().zip(items).all(|(one, two)| same_node(one, two))
        });
        if let Some(held) = held {
            return held;
        }
        let made = Members::from(items);
        table.put(hash, &made);
        made
    })
}

/// The shared handle for a field list.
///
/// The buffer is taken by reference and drained only on a miss: a hit leaves
/// the caller's fields where they are, and dropping them there costs the same
/// as the copy into a second list would have.
pub(super) fn fields(items: &mut Vec<Field>) -> Fields {
    let mut state = FxHasher::default();
    state.write_usize(items.len());
    for item in items.iter().take(SUMMARY) {
        hash_field(item, &mut state);
    }
    let hash = state.finish();
    FIELDS.with_borrow_mut(|table| {
        let held = table.hit(hash, |held: &[Field]| {
            held.len() == items.len()
                && held
                    .iter()
                    .zip(items.iter())
                    .all(|(one, two)| same_field(one, two))
        });
        if let Some(held) = held {
            return held;
        }
        let made = Fields::from_iter(items.drain(..));
        table.put(hash, &made);
        made
    })
}

/// The shared handle for a clause list, by the same rule as [`fields`].
pub(super) fn clauses(items: &mut Vec<MapClause>) -> Clauses {
    let mut state = FxHasher::default();
    state.write_usize(items.len());
    for item in items.iter().take(SUMMARY) {
        hash_clause(item, &mut state);
    }
    let hash = state.finish();
    CLAUSES.with_borrow_mut(|table| {
        let held = table.hit(hash, |held: &[MapClause]| {
            held.len() == items.len()
                && held
                    .iter()
                    .zip(items.iter())
                    .all(|(one, two)| same_clause(one, two))
        });
        if let Some(held) = held {
            return held;
        }
        let made = Clauses::from_iter(items.drain(..));
        table.put(hash, &made);
        made
    })
}

#[cfg(test)]
mod tests;
