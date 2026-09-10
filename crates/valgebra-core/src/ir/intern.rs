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
    if let Some(held) = NODES.with_borrow(|table| table.hit(hash, |held| same_node(held, &schema)))
    {
        return held;
    }
    let made = Arc::new(schema);
    NODES.with_borrow_mut(|table| table.put(hash, &made));
    made
}

/// The shared handle for a member list.
pub(super) fn members(items: &[Schema]) -> Members {
    let mut state = FxHasher::default();
    state.write_usize(items.len());
    for item in items.iter().take(SUMMARY) {
        hash_node(item, &mut state);
    }
    let hash = state.finish();
    let same = |held: &[Schema]| {
        held.len() == items.len() && held.iter().zip(items).all(|(one, two)| same_node(one, two))
    };
    if let Some(held) = MEMBERS.with_borrow(|table| table.hit(hash, same)) {
        return held;
    }
    let made = Members::from(items);
    MEMBERS.with_borrow_mut(|table| table.put(hash, &made));
    made
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
    let same = |held: &[Field]| {
        held.len() == items.len()
            && held
                .iter()
                .zip(items.iter())
                .all(|(one, two)| same_field(one, two))
    };
    if let Some(held) = FIELDS.with_borrow(|table| table.hit(hash, same)) {
        return held;
    }
    let made = Fields::from_iter(items.drain(..));
    FIELDS.with_borrow_mut(|table| table.put(hash, &made));
    made
}

/// The shared handle for a clause list, by the same rule as [`fields`].
pub(super) fn clauses(items: &mut Vec<MapClause>) -> Clauses {
    let mut state = FxHasher::default();
    state.write_usize(items.len());
    for item in items.iter().take(SUMMARY) {
        hash_clause(item, &mut state);
    }
    let hash = state.finish();
    let same = |held: &[MapClause]| {
        held.len() == items.len()
            && held
                .iter()
                .zip(items.iter())
                .all(|(one, two)| same_clause(one, two))
    };
    if let Some(held) = CLAUSES.with_borrow(|table| table.hit(hash, same)) {
        return held;
    }
    let made = Clauses::from_iter(items.drain(..));
    CLAUSES.with_borrow_mut(|table| table.put(hash, &made));
    made
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::ir::{ClassIx, CollKind, ConstIx, Constraint, Constraints, DefIx, SeqKind};

    fn field(name: &str, schema: Schema) -> Field {
        Field {
            name: name.into(),
            schema,
            required: true,
        }
    }

    fn clause(key: Schema, value: Schema) -> MapClause {
        MapClause { key, value }
    }

    /// The slot a node reads, which is what decides whether two nodes ever meet
    /// in the sameness test at all.
    fn slot(schema: &Schema) -> usize {
        let mut state = FxHasher::default();
        hash_node(schema, &mut state);
        slot_of(state.finish())
    }

    /// Two distinct nodes the table puts in one slot, found by searching.
    ///
    /// A slot is a hint and the sameness test is what decides, so the case that
    /// exercises the test is two nodes that land together and are not the same.
    /// The pair is searched for rather than written down: which two collide is a
    /// property of the hash, and a pair pinned here would go stale the moment it
    /// changed while the property it stands for held.
    fn colliding_pair(make: impl Fn(usize) -> Schema) -> (Schema, Schema) {
        let mut seen: HashMap<usize, Schema> = HashMap::new();
        for index in 0..100_000 {
            let candidate = make(index);
            match seen.get(&slot(&candidate)) {
                Some(first) => return (first.clone(), candidate),
                None => {
                    seen.insert(slot(&candidate), candidate);
                }
            }
        }
        panic!("no two of a hundred thousand nodes share a slot");
    }

    /// Whether a family has a distinct node for every index, or only the few a
    /// finite payload allows -- the top has two spellings and no more, so it is
    /// asked the questions a pair of nodes can answer and not the one that
    /// wants an unbounded supply of them.
    #[derive(PartialEq)]
    enum Endless {
        Yes,
        No,
    }

    /// Every family of node, as a rule that makes a distinct one per index.
    ///
    /// Two tests read it: one asks each family to be shared when it is built
    /// twice, which is what the table is for, and one asks two of a family that
    /// land in one slot to stay two nodes, which is what keeps it sound. A
    /// variant missing here is a variant neither question is asked about.
    #[expect(clippy::type_complexity, reason = "one list, read by three tests")]
    fn families() -> Vec<(&'static str, Box<dyn Fn(usize) -> Schema>, Endless)> {
        let base = |i: usize| Schema::Literal(ConstIx::new(i));
        let spelled = |i: usize| {
            Schema::Anything(if i.is_multiple_of(2) {
                Spelling::Top
            } else {
                Spelling::Any
            })
        };
        let refined = {
            // The constraints are carried rather than rebuilt, so what tells two
            // refinements apart is the base: a refinement over a freshly built
            // constraint list is a fresh node, which is what the sameness test
            // says and what this family is written to.
            let held = Constraints::from([Constraint::MinLen(3)]);
            move |i: usize| Schema::Refine {
                base: node(base(i)),
                constraints: held.clone(),
            }
        };
        vec![
            ("anything", Box::new(spelled), Endless::No),
            ("literal", Box::new(base), Endless::Yes),
            (
                "instance",
                Box::new(|i| Schema::Instance(ClassIx::new(i))),
                Endless::Yes,
            ),
            (
                "ref",
                Box::new(|i| Schema::Ref(DefIx::new(i))),
                Endless::Yes,
            ),
            (
                "self-ref",
                Box::new(|i| Schema::SelfRef(i as u64)),
                Endless::Yes,
            ),
            (
                "complement",
                Box::new(move |i| Schema::Complement(node(base(i)))),
                Endless::Yes,
            ),
            (
                "collection",
                Box::new(move |i| Schema::Coll {
                    container: CollKind::Set,
                    element: node(base(i)),
                }),
                Endless::Yes,
            ),
            (
                "sequence",
                Box::new(move |i| Schema::Seq {
                    container: SeqKind::List,
                    shape: SeqShape {
                        prefix: members(&[base(i)]),
                        tail: None,
                    },
                }),
                Endless::Yes,
            ),
            (
                "sequence tail",
                Box::new(move |i| Schema::Seq {
                    container: SeqKind::Tuple,
                    shape: SeqShape {
                        prefix: members(&[]),
                        tail: Some(node(base(i))),
                    },
                }),
                Endless::Yes,
            ),
            (
                "keyed map",
                Box::new(move |i| Schema::KeyedMap {
                    fields: fields(&mut vec![field("k", base(i))]),
                    defaults: clauses(&mut vec![clause(Schema::Str, Schema::Int)]),
                }),
                Endless::Yes,
            ),
            (
                "attribute record",
                Box::new(move |i| Schema::AttrRecord {
                    fields: fields(&mut vec![field("k", base(i))]),
                }),
                Endless::Yes,
            ),
            (
                "union",
                Box::new(move |i| Schema::Union(members(&[base(i), Schema::Int]))),
                Endless::Yes,
            ),
            (
                "meet",
                Box::new(move |i| Schema::Intersection(members(&[base(i), Schema::Int]))),
                Endless::Yes,
            ),
            ("refinement", Box::new(refined), Endless::Yes),
        ]
    }

    /// Two indices of a family the table puts in *different* slots, found by
    /// searching for the same reason [`colliding_pair`] searches.
    fn parted_pair(make: impl Fn(usize) -> Schema) -> (Schema, Schema) {
        let first = make(0);
        for index in 1..100_000 {
            let candidate = make(index);
            if slot(&candidate) != slot(&first) {
                return (first, candidate);
            }
        }
        panic!("a hundred thousand nodes of one family share one slot");
    }

    /// The substitution the table makes is invisible, so what a caller reads
    /// back is what it asked for -- and what a second caller asking for the
    /// same thing reads is the same allocation.
    #[test]
    fn a_list_built_twice_is_one_list_holding_what_was_built() {
        let make = || vec![field("a", Schema::Int), field("b", Schema::Str)];
        let one = fields(&mut make());
        let two = fields(&mut make());
        assert!(
            Arc::ptr_eq(&one, &two),
            "the second build is the first list"
        );
        assert_eq!(one.len(), 2);
        assert_eq!(&*one[0].name, "a");
        assert_eq!(one[1].schema, Schema::Str);
    }

    /// The same of the other three kinds of handle, each of which has its own
    /// table and its own sameness test.
    #[test]
    fn every_kind_of_handle_is_shared_by_a_second_build() {
        let members_one = members(&[Schema::Int, Schema::Str]);
        let members_two = members(&[Schema::Int, Schema::Str]);
        assert!(Arc::ptr_eq(&members_one, &members_two));

        let clauses_one = clauses(&mut vec![clause(Schema::Str, Schema::Int)]);
        let clauses_two = clauses(&mut vec![clause(Schema::Str, Schema::Int)]);
        assert!(Arc::ptr_eq(&clauses_one, &clauses_two));

        let node_one = node(Schema::Complement(node(Schema::Bytes)));
        let node_two = node(Schema::Complement(node(Schema::Bytes)));
        assert!(Arc::ptr_eq(&node_one, &node_two));

        let shape = |element| Schema::Seq {
            container: SeqKind::Tuple,
            shape: SeqShape {
                prefix: members(&[]),
                tail: Some(node(element)),
            },
        };
        assert!(Arc::ptr_eq(
            &node(shape(Schema::Float)),
            &node(shape(Schema::Float))
        ));
    }

    /// A handle is found again after another is built, which is the whole of
    /// what the hash is for: entries that differ take different slots, so one
    /// build does not evict the answer to the next.
    #[test]
    fn a_handle_is_found_again_after_another_is_built() {
        let one = fields(&mut vec![field("a", Schema::Int)]);
        let other = fields(&mut vec![field("b", Schema::Str)]);
        let again = fields(&mut vec![field("a", Schema::Int)]);
        assert!(!Arc::ptr_eq(&one, &other));
        assert!(
            Arc::ptr_eq(&one, &again),
            "the first list took its own slot"
        );

        let list = clauses(&mut vec![clause(Schema::Str, Schema::Int)]);
        let apart = clauses(&mut vec![clause(Schema::Int, Schema::Str)]);
        let back = clauses(&mut vec![clause(Schema::Str, Schema::Int)]);
        assert!(!Arc::ptr_eq(&list, &apart));
        assert!(Arc::ptr_eq(&list, &back));

        let held = members(&[Schema::Int]);
        let beside = members(&[Schema::Str]);
        let read = members(&[Schema::Int]);
        assert!(!Arc::ptr_eq(&held, &beside));
        assert!(Arc::ptr_eq(&held, &read));
    }

    /// Two nodes that land in one slot are two nodes, for every family.
    #[test]
    fn two_nodes_in_one_slot_are_two_nodes() {
        for (name, family, endless) in families() {
            if endless == Endless::No {
                // Two spellings and no more: a slot each, and never a pair that
                // meets in one. The sharing test below is what holds this
                // family's arm of the sameness test.
                continue;
            }
            let (one, two) = colliding_pair(&family);
            // Not `assert_ne!`: the two spellings of the top compare equal to
            // the algebra, and the point of the pair is that the table can tell
            // apart what the algebra does not.
            assert!(
                !same_node(&one, &two),
                "{name}: the search found one node twice"
            );
            let first = node(one.clone());
            let second = node(two.clone());
            assert!(
                !Arc::ptr_eq(&first, &second),
                "{name}: two nodes were collapsed into one"
            );
            assert_eq!(*first, one, "{name}");
            assert_eq!(*second, two, "{name}");
        }
    }

    /// An index of a family whose node takes a slot none of its parts take.
    ///
    /// A tree is put bottom-up, and a node landing on a part's slot evicts the
    /// part. The next build of the same tree then makes that part afresh at a
    /// new address, the node's hash -- which reads the address -- names a slot
    /// the first node is not in, and the build misses. That is the miss of a
    /// direct-mapped cache and not a defect, and which index takes it depends
    /// on where the allocator put the part, so a test claiming a hit picks an
    /// index the miss cannot happen to.
    fn settled(make: impl Fn(usize) -> Schema) -> (usize, Schema) {
        for index in 0..100_000 {
            let candidate = make(index);
            let own = slot(&candidate);
            if candidate.children().all(|part| slot(part) != own) {
                return (index, candidate);
            }
        }
        panic!("a hundred thousand nodes of one family each land on a part");
    }

    /// Every family is shared when it is built twice: the table answers for the
    /// whole node set and not for the variants somebody thought of.
    #[test]
    fn a_node_built_twice_is_one_node_in_every_family() {
        for (name, family, _) in families() {
            let (index, first) = settled(&family);
            let one = node(first);
            let two = node(family(index));
            assert!(
                Arc::ptr_eq(&one, &two),
                "{name}: the second build made a second node"
            );
        }
    }

    /// And a node is found again after another of its family is built, which is
    /// what the hash buys: two nodes that differ take two slots, so one build
    /// does not evict the answer to the next.
    #[test]
    fn a_node_is_found_again_after_another_of_its_family() {
        for (name, family, _) in families() {
            let (one, two) = parted_pair(&family);
            let first = node(one.clone());
            let _other = node(two);
            let again = node(one);
            assert!(
                Arc::ptr_eq(&first, &again),
                "{name}: the node lost its slot to one that differs from it"
            );
        }
    }

    /// The two spellings of the top denote the same set and compare equal, and
    /// `repr` gives back the one that was written. Collapsing them would hand a
    /// caller the other one.
    #[test]
    fn the_two_spellings_of_the_top_are_not_one_node() {
        let top = members(&[Schema::Anything(Spelling::Top)]);
        let any = members(&[Schema::Anything(Spelling::Any)]);
        assert!(!Arc::ptr_eq(&top, &any));
        assert!(matches!(top[0], Schema::Anything(Spelling::Top)));
        assert!(matches!(any[0], Schema::Anything(Spelling::Any)));
        // Equal to the algebra, which is what makes the test above the only
        // thing standing between a spelling and the wrong answer from `repr`.
        assert_eq!(top[0], any[0]);
    }

    /// The hash reads a bounded prefix, so two lists agreeing on their length
    /// and their first entries land in one slot. The test that decides is the
    /// whole list, and this is the case that shows it deciding.
    #[test]
    fn two_lists_alike_only_where_the_hash_reads_are_two_lists() {
        let prefix = || (0..SUMMARY).map(|i| Schema::Literal(ConstIx::new(i)));
        let one: Vec<Schema> = prefix().chain([Schema::Int]).collect();
        let two: Vec<Schema> = prefix().chain([Schema::Str]).collect();
        let held = members(&one);
        let second = members(&two);
        assert!(!Arc::ptr_eq(&held, &second));
        assert_eq!(held.last(), Some(&Schema::Int));
        assert_eq!(second.last(), Some(&Schema::Str));
        // And the first is still itself: sharing a slot does not rewrite it.
        assert_eq!(&*held, &one[..]);
    }

    /// The same for a field list, which differs in three ways the hash's
    /// summary may not have read: a name, a schema, and whether it is required.
    #[test]
    fn two_field_lists_alike_only_where_the_hash_reads_are_two_lists() {
        let prefix = || {
            (0..SUMMARY)
                .map(|i| field("same", Schema::Literal(ConstIx::new(i))))
                .collect::<Vec<_>>()
        };
        let with = |last: Field| {
            let mut list = prefix();
            list.push(last);
            list
        };
        let by_schema = fields(&mut with(field("last", Schema::Int)));
        let by_name = fields(&mut with(field("other", Schema::Int)));
        let by_requirement = fields(&mut with(Field {
            name: "last".into(),
            schema: Schema::Int,
            required: false,
        }));
        assert!(!Arc::ptr_eq(&by_schema, &by_name));
        assert!(!Arc::ptr_eq(&by_schema, &by_requirement));
        assert_eq!(&*by_schema[SUMMARY].name, "last");
        assert_eq!(&*by_name[SUMMARY].name, "other");
        assert!(by_schema[SUMMARY].required);
        assert!(!by_requirement[SUMMARY].required);
    }

    /// And for a clause list, whose two halves are both schemas: a key that
    /// differs and a value that differs each make two lists.
    #[test]
    fn two_clause_lists_alike_only_where_the_hash_reads_are_two_lists() {
        let prefix = || {
            (0..SUMMARY)
                .map(|i| clause(Schema::Literal(ConstIx::new(i)), Schema::Int))
                .collect::<Vec<_>>()
        };
        let with = |last: MapClause| {
            let mut list = prefix();
            list.push(last);
            list
        };
        let held = clauses(&mut with(clause(Schema::Str, Schema::Int)));
        let by_key = clauses(&mut with(clause(Schema::Bytes, Schema::Int)));
        let by_value = clauses(&mut with(clause(Schema::Str, Schema::Bool)));
        assert!(!Arc::ptr_eq(&held, &by_key));
        assert!(!Arc::ptr_eq(&held, &by_value));
        assert_eq!(held[SUMMARY].key, Schema::Str);
        assert_eq!(by_key[SUMMARY].key, Schema::Bytes);
        assert_eq!(by_value[SUMMARY].value, Schema::Bool);
    }

    /// A list is the length it was built with. Two lists of different lengths
    /// are two lists however much of a prefix they share.
    #[test]
    fn a_list_is_not_a_prefix_of_another() {
        let short: Vec<Schema> = (0..SUMMARY)
            .map(|i| Schema::Literal(ConstIx::new(i)))
            .collect();
        let long: Vec<Schema> = short.iter().cloned().chain([Schema::Int]).collect();
        let one = members(&short);
        let two = members(&long);
        assert!(!Arc::ptr_eq(&one, &two));
        assert_eq!(one.len(), SUMMARY);
        assert_eq!(two.len(), SUMMARY + 1);

        let short_fields: Vec<Field> = (0..SUMMARY)
            .map(|i| field("f", Schema::Literal(ConstIx::new(i))))
            .collect();
        let long_fields: Vec<Field> = short_fields
            .iter()
            .cloned()
            .chain([field("g", Schema::Int)])
            .collect();
        assert_eq!(fields(&mut short_fields.clone()).len(), SUMMARY);
        assert_eq!(fields(&mut long_fields.clone()).len(), SUMMARY + 1);

        let short_clauses: Vec<MapClause> = (0..SUMMARY)
            .map(|i| clause(Schema::Literal(ConstIx::new(i)), Schema::Int))
            .collect();
        let long_clauses: Vec<MapClause> = short_clauses
            .iter()
            .cloned()
            .chain([clause(Schema::Str, Schema::Int)])
            .collect();
        assert_eq!(clauses(&mut short_clauses.clone()).len(), SUMMARY);
        assert_eq!(clauses(&mut long_clauses.clone()).len(), SUMMARY + 1);
    }

    /// A node the table has seen is freed when its owner drops it: the entry
    /// left behind holds a weak handle, which keeps no tree alive.
    #[test]
    fn the_table_does_not_keep_a_dropped_node_alive() {
        let held = node(Schema::Coll {
            container: CollKind::Set,
            element: node(Schema::Bytes),
        });
        let watch = Arc::downgrade(&held);
        assert!(watch.upgrade().is_some());
        drop(held);
        assert!(
            watch.upgrade().is_none(),
            "the table's entry outlived the node it named"
        );
    }

    /// Sharing is by construction, so a child built through the table is the
    /// same child in both parents and the parents are one node in turn.
    #[test]
    fn sharing_a_child_shares_the_parent_over_it() {
        let one = node(Schema::Complement(node(Schema::Float)));
        let two = node(Schema::Complement(node(Schema::Float)));
        assert!(Arc::ptr_eq(&one, &two));
    }
}
