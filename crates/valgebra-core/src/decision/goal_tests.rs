//! The goals a query asks: how often it asks one twice, and what they are
//! drawn from.
//!
//! `DECISION_BUDGET`'s doc says a goal memo would mostly not lower it, because
//! over the decision workloads the goals a query *repeats* number zero, and
//! that termination does not rest on it, because every goal is a pair drawn
//! from a finite closure of the query's subterms. The recorder in
//! [`super::goals`] reads both, and this asks them of the shapes the claims
//! are about.
//!
//! The shapes are the workloads' own, written here rather than imported: an
//! example is a binary and a test cannot call into one. What is measured is the
//! shape, so each is spelled the way `examples/decision_repeat_workload.rs` and
//! `examples/decision_matrix_workload.rs` spell it, and a change to either
//! belongs in both.
//!
//! A repeat counted here is a goal the procedure really re-derives: the
//! recorder sits inside `is_subtype_rec`, which is past the caches that answer
//! a repeated field or position without asking again. So zero here is "the
//! caches absorb it", not "the goal was never reached".

use std::sync::Arc;

/// A test-side record of the goals one query asks.
///
/// [`DECISION_BUDGET`]'s doc rests on two readings of it: over the decision
/// workloads the goals a query *repeats* are zero, so a memo would mostly not
/// lower the ceiling, and every goal is drawn from the closure of the query's
/// subterms, so the ceiling bounds cost rather than standing in for
/// termination. This takes both. Compiled for this crate's own tests only, so
/// the procedure a caller runs carries no recorder and pays nothing for one.
///
/// A goal is the pair the recursion is asked about, and two are the same goal
/// when the pair is equal. Equality rather than the interned address: a query
/// builds goals of its own -- [`seq_splits_across_union`] meets a component
/// with a branch's complement -- and those are temporaries, so an address
/// freed and handed out again would read as a repeat that never happened.
/// Equality also counts at least as many repeats as identity can, which is the
/// safe direction for a claim that there are none.
pub(crate) mod goals {
    use std::cell::RefCell;

    use rustc_hash::FxHashMap;

    use crate::ir::Schema;

    thread_local! {
        /// The goals asked inside a [`counted`] call, and how often each was.
        static ASKED: RefCell<Option<FxHashMap<(Schema, Schema), u32>>> =
            const { RefCell::new(None) };
        /// The deepest the trail grew inside a [`counted`] call.
        static DEEPEST: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// How many pairs the set representation was asked on this thread.
        static READ_BY_THE_SETS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        /// How many constructors stand on the path to the frame running.
        static CROSSED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// The goals the path holds open, each with [`CROSSED`] as it was opened.
        static OPENED: RefCell<Vec<(Schema, Schema, usize)>> = const { RefCell::new(Vec::new()) };
    }

    /// A constructor on the path for as long as this is held: what is asked below
    /// it is about a value strictly inside the one above it.
    pub(crate) struct Crossing;

    /// Enter a rule whose goals are of a position, a key, a value or an attribute.
    pub(crate) fn crossing() -> Crossing {
        CROSSED.with(|crossed| crossed.set(crossed.get() + 1));
        Crossing
    }

    /// The crossing `schema` is, where a reading of it asks of its positions.
    pub(crate) fn crossing_of(schema: &Schema) -> Option<Crossing> {
        matches!(
            schema,
            Schema::Seq { .. } | Schema::KeyedMap { .. } | Schema::AttrRecord { .. }
        )
        .then(crossing)
    }

    impl Drop for Crossing {
        fn drop(&mut self) {
            CROSSED.with(|crossed| crossed.set(crossed.get() - 1));
        }
    }

    /// A goal opened on the path: the trail's, or `(s, nothing)` for the emptiness
    /// of `s` [LN26, §5.1], which the lists beside it hold.
    pub(crate) fn opened(sub: &Schema, sup: &Schema) {
        let at = CROSSED.with(std::cell::Cell::get);
        OPENED.with_borrow_mut(|open| open.push((sub.clone(), sup.clone(), at)));
    }

    /// The goal opened last, closed.
    pub(crate) fn closed() {
        OPENED.with_borrow_mut(|open| {
            open.pop();
        });
    }

    /// A goal open on the path for as long as this is held.
    pub(crate) struct HeldOpen;

    /// Open a goal until what this gives back is dropped.
    pub(crate) fn held_open(sub: &Schema, sup: &Schema) -> HeldOpen {
        opened(sub, sup);
        HeldOpen
    }

    impl Drop for HeldOpen {
        fn drop(&mut self) {
            closed();
        }
    }

    /// A goal met again, which reads as its assumption [AC93, §1.5]. That stands
    /// only where a constructor lies between the two: a hypothesis read at a smaller
    /// value is an induction on the value, and one read at the same value proves
    /// anything. Contractive definitions are what is held to give it, and this is
    /// where every query under test is held to it. A goal the path did not open is
    /// one a test wrote on the list.
    pub(crate) fn met_again(sub: &Schema, sup: &Schema) {
        let now = CROSSED.with(std::cell::Cell::get);
        let at = OPENED.with_borrow(|open| {
            open.iter()
                .rev()
                .find(|(a, b, _)| a == sub && b == sup)
                .map(|(_, _, at)| *at)
        });
        assert!(
            at.is_none_or(|at| at < now),
            "{sub:?} below {sup:?} is met again under no constructor"
        );
    }

    /// Count a pair put to the set representation
    /// ([`Schema::descriptor_contained_in`]).
    pub(crate) fn read_by_the_sets() {
        READ_BY_THE_SETS.with(|read| read.set(read.get() + 1));
    }

    /// How many pairs the set representation was asked on this thread so far:
    /// read before and after a reading, what moved is how many it lowered.
    pub(crate) fn pairs_read_by_the_sets() -> u64 {
        READ_BY_THE_SETS.with(std::cell::Cell::get)
    }

    /// What one counted query asked.
    ///
    /// Both numbers, because either alone can be read wrong: `repeated` is the
    /// claim, and `asked` is what says the recorder was wired to the procedure
    /// at all. A count of zero repeats over a query that asked nothing is the
    /// shape a silent recorder gives, and it reads exactly like a good answer.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Counts {
        /// Goals the recursion was asked, counting a repeat once per ask.
        pub(crate) asked: usize,
        /// How many of those asks were of a pair already asked.
        pub(crate) repeated: usize,
        /// The most pairs the trail held at once.
        pub(crate) deepest: usize,
    }

    /// Run `query` with its goals counted, and give back what it asked.
    ///
    /// One count per thread: two at once would each be about part of the
    /// other's query, and a number about part of a query is not the number the
    /// argument needs.
    pub(crate) fn counted<T>(query: impl FnOnce() -> T) -> (T, Counts) {
        let (answer, table) = tabled(query);
        let counts = Counts {
            asked: table.values().map(|times| *times as usize).sum(),
            repeated: table.values().map(|times| (times - 1) as usize).sum(),
            deepest: DEEPEST.with(std::cell::Cell::get),
        };
        (answer, counts)
    }

    /// Run `query`, and give back every distinct goal it asked, each once.
    pub(crate) fn recorded<T>(query: impl FnOnce() -> T) -> (T, Vec<(Schema, Schema)>) {
        let (answer, table) = tabled(query);
        (answer, table.into_keys().collect())
    }

    /// Run `query` with the table installed, and hand the table back: every
    /// goal it asked, with how often.
    pub(crate) fn tabled<T>(query: impl FnOnce() -> T) -> (T, FxHashMap<(Schema, Schema), u32>) {
        ASKED.with(|asked| {
            let mut slot = asked.borrow_mut();
            assert!(slot.is_none(), "a count is already running on this thread");
            *slot = Some(FxHashMap::default());
        });
        DEEPEST.with(|deepest| deepest.set(0));
        let answer = query();
        let table = ASKED
            .with(|asked| asked.borrow_mut().take())
            .expect("the table installed above is still there");
        (answer, table)
    }

    /// Record the trail's depth after a push, where a count is running.
    pub(crate) fn trail(depth: usize) {
        DEEPEST.with(|deepest| deepest.set(deepest.get().max(depth)));
    }

    /// Record one goal, where a count is running.
    pub(crate) fn record(subject: &Schema, other: &Schema) {
        ASKED.with(|asked| {
            if let Some(table) = asked.borrow_mut().as_mut() {
                *table.entry((subject.clone(), other.clone())).or_insert(0) += 1;
            }
        });
    }
}

use crate::{
    Attribute, ClassIx, Constraint, DefIx, Field, Kind, LeafRelations, Openness, OperandIx,
    Relation, Schema, SeqShape,
};
use goals::Counts;

use super::{Budget, DECISION_BUDGET};

/// Two classes that lay down no builtin layout, answered as the bindings answer
/// a plain Python class. The matrix workload's oracle, for the matrix's pairs:
/// an oracle that declines every class question leaves the class readings
/// unreachable, and a count over pairs nothing reaches is a count of nothing.
struct PlainClasses;

impl crate::descr::lower::Constants for PlainClasses {}

impl LeafRelations for PlainClasses {
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool> {
        match (sub, sup) {
            (Schema::Instance(a), Schema::Instance(b)) => Some(a == b),
            _ => None,
        }
    }

    fn class_admits_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        None
    }

    fn direct_instance_of_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        Some(false)
    }

    fn kind_derives_from(&self, _kind: Kind, _class: ClassIx) -> Option<bool> {
        Some(false)
    }

    fn atom_denotes_a_set(&self, atom: &Schema) -> Option<bool> {
        Some(matches!(atom, Schema::Instance(_)))
    }

    /// A plain class's instances carry a dictionary, and it defines none of
    /// the names a record here asks about.
    fn attribute_reach(
        &self,
        _class: ClassIx,
        _name: &str,
    ) -> Option<crate::descr::classes::Reach> {
        Some(crate::descr::classes::Reach::Anything)
    }
}

/// What one query asked, with the verdict beside it.
///
/// The verdict is carried out so a test can say the query it counted was the
/// query it meant: a pair that stopped being decided would otherwise read as a
/// happy zero.
fn ask(sub: &Schema, sup: &Schema, defs: &[Schema]) -> (Relation, Counts) {
    goals::counted(|| sub.subtype_relation_under(sup, &PlainClasses, defs))
}

/// The goals one query asks twice.
fn repeated(sub: &Schema, sup: &Schema, defs: &[Schema]) -> usize {
    ask(sub, sup, defs).1.repeated
}

/// `list[T]`.
fn list_of(element: Schema) -> Schema {
    Schema::list(SeqShape::homogeneous(element))
}

/// A list nested `depth` deep over `leaf`.
fn nested_lists(depth: usize, leaf: Schema) -> Schema {
    (0..depth).fold(leaf, |inner, _| list_of(inner))
}

/// A record whose fields all carry `element`.
fn repeating_record(width: usize, element: &Schema) -> Schema {
    Schema::record(
        (0..width)
            .map(|index| Field {
                name: format!("f{index}").into(),
                schema: element.clone(),
                required: true,
            })
            .collect(),
        Openness::Closed,
    )
}

/// A tuple whose positions all carry `element`.
fn repeating_tuple(width: usize, element: &Schema) -> Schema {
    Schema::tuple(SeqShape::fixed((0..width).map(|_| element.clone())))
}

/// A refinement carrying one constraint.
fn refined(base: Schema, constraint: Constraint) -> Schema {
    Schema::Refine {
        base: Arc::new(base),
        constraints: vec![constraint].into(),
    }
}

/// A class met with the attributes its instances carry.
fn dataclass(class: ClassIx, width: usize) -> Schema {
    let fields = (0..width)
        .map(|i| Attribute {
            name: format!("f{i}").into(),
            schema: Schema::Int,
        })
        .collect::<Vec<_>>();
    Schema::meet([
        Schema::Instance(class),
        Schema::AttrRecord {
            fields: fields.into(),
        },
    ])
}

/// A recursive record, reached through the reference that names it.
fn tree_defs() -> Vec<Schema> {
    vec![Schema::KeyedMap {
        fields: vec![
            Field {
                name: "value".into(),
                schema: Schema::Int,
                required: true,
            },
            Field {
                name: "left".into(),
                schema: Schema::Ref(DefIx::new(0)),
                required: false,
            },
        ]
        .into(),
        defaults: Vec::new().into(),
    }]
}

// THEORY: repeated-goals-are-counted, regularity-bounds-the-goals
/// The counter is wired to the procedure, so a zero from it is a reading.
///
/// The control every count below needs. A recorder attached to nothing reports
/// no repeats for every query, which reads exactly like a procedure that
/// repeats nothing -- so what is asserted here is that goals arrive at all.
/// Stated as `asked` rather than as `repeated` on purpose: a shape that repeats
/// may stop under a change to the rules, and a control that went away with it
/// would leave every number below unwitnessed.
#[test]
fn the_counter_sees_the_goals_a_query_asks() {
    let narrow = list_of(list_of(Schema::Int));
    let wide = list_of(list_of(Schema::union([Schema::Int, Schema::Str])));
    let (verdict, counts) = ask(&narrow, &wide, &[]);
    assert_eq!(verdict, Relation::Holds);
    assert!(
        counts.asked >= 3,
        "the counter saw {} goal(s)",
        counts.asked
    );
    assert_eq!(counts.repeated, 0);
}

/// The shape a memo is usually proposed for: one goal, once per field.
#[test]
fn a_record_whose_fields_share_a_schema_repeats_no_goal() {
    let element = nested_lists(2, Schema::Int);
    let wider = nested_lists(2, Schema::union([Schema::Int, Schema::Str]));
    let narrow = repeating_record(8, &element);
    let wide = repeating_record(8, &wider);

    // The proof: every field's element is inside the wider one, which is one
    // goal reached once per field, and the field cache answers it after the
    // first.
    assert_eq!(ask(&narrow, &wide, &[]).0, Relation::Holds);
    assert_eq!(repeated(&narrow, &wide, &[]), 0);
    // And the refutation, which walks the same repeated goal the other way.
    let (verdict, counts) = ask(&wide, &narrow, &[]);
    assert_eq!(counts.repeated, 0);
    assert_ne!(verdict, Relation::Holds);
}

/// The same repetition in the container whose elements are reached by position.
#[test]
fn a_tuple_whose_positions_share_a_schema_repeats_no_goal() {
    let element = nested_lists(2, Schema::Int);
    let wider = nested_lists(2, Schema::union([Schema::Int, Schema::Str]));
    let narrow = repeating_tuple(8, &element);
    let wide = repeating_tuple(8, &wider);

    assert_eq!(ask(&narrow, &wide, &[]).0, Relation::Holds);
    assert_eq!(repeated(&narrow, &wide, &[]), 0);
    let (verdict, counts) = ask(&wide, &narrow, &[]);
    assert_eq!(counts.repeated, 0);
    assert_ne!(verdict, Relation::Holds);
}

// THEORY: repeated-goals-are-counted
/// The width the page names, so the number it states is the number here.
#[test]
fn a_record_of_thirty_two_fields_sharing_one_schema_repeats_no_goal() {
    let element = nested_lists(2, Schema::Int);
    let wider = nested_lists(2, Schema::union([Schema::Int, Schema::Str]));
    let narrow = repeating_record(32, &element);
    let wide = repeating_record(32, &wider);
    assert_eq!(ask(&narrow, &wide, &[]).0, Relation::Holds);
    assert_eq!(repeated(&narrow, &wide, &[]), 0);
}

// THEORY: repeated-goals-are-counted
/// The matrix's own pairs, each counted on its own -- and two of them repeat.
///
/// The structural readings' slow set, which the three older workloads do not
/// carry: a refinement on both sides, two sequences read as stars, a class met
/// with its attributes, a complement, and a reference on the supertype's side.
/// Every pair is one query, so the number is per query and not a total over the
/// matrix -- a budget is per query too.
///
/// **A meet against a union repeats four goals.** The union distribution asks
/// the meet of each branch, and the meet rule then asks the class atom against
/// the same thing once per member with no cache between the two rules. That is
/// word for word the shape [`super::DECISION_BUDGET`]'s argument names as what
/// would reopen the question of a goal memo, and this workload -- written after
/// that argument -- is where it is in hand. The numbers are recorded rather
/// than rounded to a claim: a cache between those two rules would take them to
/// zero, and this table is what would say so.
#[test]
fn the_matrix_repeats_a_goal_only_where_a_meet_meets_a_union() {
    let long_list = refined(list_of(Schema::Int), Constraint::MinLen(2));
    let positive = refined(Schema::Int, Constraint::Ge(OperandIx::new(0)));
    let long_text = refined(Schema::Str, Constraint::MinLen(1));
    let longer_list = refined(list_of(Schema::Int), Constraint::MinLen(3));
    let bare_list = list_of(Schema::Int);
    let strings = list_of(Schema::Str);
    let nested = list_of(list_of(Schema::Int));
    let point = dataclass(ClassIx::new(0), 2);
    let kinds = Schema::union([Schema::Int, Schema::Str]);
    let optional = Schema::union([Schema::Int, Schema::NoneType]);
    let not_int = Schema::Complement(Arc::new(Schema::Int));
    let a_class = Schema::Instance(ClassIx::new(1));
    let own_class = Schema::Instance(ClassIx::new(0));
    let defs = tree_defs();
    let tree = Schema::Ref(DefIx::new(0));
    let none: &[Schema] = &[];

    let matrix: Vec<(&str, &Schema, &Schema, &[Schema], usize)> = vec![
        ("a bound against a sign", &long_list, &positive, none, 0),
        ("a sign against a bound", &positive, &long_list, none, 0),
        (
            "a bound below a looser one",
            &longer_list,
            &long_list,
            none,
            0,
        ),
        ("two element kinds apart", &long_list, &strings, none, 0),
        ("an element against a nesting", &long_list, &nested, none, 0),
        (
            "a bare sequence against a bound",
            &bare_list,
            &long_list,
            none,
            0,
        ),
        (
            "a word bound against a list one",
            &long_text,
            &long_list,
            none,
            0,
        ),
        (
            "a refinement below its own base",
            &long_list,
            &bare_list,
            none,
            0,
        ),
        // The two that repeat: a meet against a union, twice over.
        ("a class against a union of kinds", &point, &kinds, none, 4),
        ("a class against an optional", &point, &optional, none, 4),
        ("a class against one kind", &point, &Schema::Int, none, 0),
        (
            "a class against the class it is built on",
            &point,
            &own_class,
            none,
            0,
        ),
        ("a complement against a class", &not_int, &a_class, none, 0),
        ("a complement against a meet", &not_int, &point, none, 0),
        ("a class against a fixpoint", &point, &tree, &defs, 0),
        ("a sequence against a fixpoint", &bare_list, &tree, &defs, 0),
    ];

    let mut total = 0;
    for (name, sub, sup, defs, expected) in matrix {
        let (_, counts) = ask(sub, sup, defs);
        assert!(counts.asked > 0, "{name} asked no goal at all");
        assert_eq!(counts.repeated, expected, "{name}");
        total += counts.repeated;
    }
    // The matrix's whole number, so a pair that starts repeating somewhere else
    // while one of the two stops cannot cancel out row by row.
    assert_eq!(total, 8);
}

/// A fixed pair of unions: the subject the product rule splits across the
/// branches of [`product_split`], where no single branch contains it.
fn product_subject() -> Schema {
    Schema::tuple(SeqShape::fixed([
        Schema::union([Schema::Int, Schema::Str, Schema::Bytes]),
        Schema::union([Schema::Int, Schema::Str]),
    ]))
}

/// The union of pairs that covers [`product_subject`] only taken together.
fn product_split() -> Schema {
    let pair = |a: Schema, b: Schema| Schema::tuple(SeqShape::fixed([a, b]));
    Schema::union([
        pair(Schema::Int, Schema::Int),
        pair(Schema::Str, Schema::Int),
        pair(Schema::Bytes, Schema::Int),
        pair(
            Schema::union([Schema::Int, Schema::Str, Schema::Bytes]),
            Schema::Str,
        ),
    ])
}

/// Every distinct subtree of `schema`, and of every definition it can reach.
///
/// The set regularity is about: a schema is a finite tree with back edges, so
/// unfolding it forever reaches only these, and a goal the procedure asks is
/// a pair of them unless a rule built a term of its own.
fn subterms(schema: &Schema, defs: &[Schema]) -> Vec<Schema> {
    fn collect(schema: &Schema, out: &mut Vec<Schema>) {
        if !out.contains(schema) {
            out.push(schema.clone());
        }
        for child in schema.children() {
            collect(child, out);
        }
    }
    let mut out = Vec::new();
    collect(schema, &mut out);
    for body in defs {
        collect(body, &mut out);
    }
    out
}

/// The distinct goals a query asks, against the pairs of subterms the two
/// sides have.
fn distinct_against_subterm_pairs(sub: &Schema, sup: &Schema, defs: &[Schema]) -> (usize, usize) {
    let (_, counts) = ask(sub, sup, defs);
    let pairs = subterms(sub, defs).len() * subterms(sup, defs).len();
    (counts.asked - counts.repeated, pairs)
}

/// Whether `term` is in the closure the goals of a query over `subterms` are
/// drawn from.
///
/// A rule asks a pair of the children of the pair it was asked, or unfolds a
/// reference into the definition `subterms` holds, except in six places. The
/// product rule meets a component with the complement of a branch's
/// component, one conjunct per narrowing, so what it asks is a meet of
/// subterms and of their complements. The union rule asks the subject against
/// the branches it can meet, which is a union of some members of a union among
/// the subterms, at any depth ([`members_at_every_depth`]): the rules read a
/// nested union as its members, so `X ∪ (Y ∪ ¬c)` is respelled `X ∪ Y`, which
/// names members of two subterms, and a rule may keep one whole. A pair every
/// decider declined against a complement is asked as the meet of the subject
/// and the inner set below the bottom, and as the inner set below the
/// subject's complement. A sequence or a mapping below a union asks its item
/// against the union of what the branches hold at one position, which are
/// children of the branches ([`readings::one_item_outside`]). A meet of
/// sequences is asked as the one sequence of its items' meets
/// ([`one_sequence_of`]). And a complement or a meet folds to a bound, the top
/// or the bottom. Each of the six is a subset, a sign or a meet of the
/// subterms, so the closure is finite because they are.
fn in_closure(term: &Schema, subterms: &[Schema]) -> bool {
    let signed = |t: &Schema| {
        subterms.contains(t) || matches!(t, Schema::Complement(inner) if subterms.contains(inner))
    };
    let built = |t: &Schema| match t {
        Schema::Union(members) => members.iter().all(|m| one_sequence_of(m, subterms)),
        t => one_sequence_of(t, subterms),
    };
    match term {
        _ if signed(term) => true,
        Schema::Nothing | Schema::Anything(_) => true,
        Schema::Intersection(members) => members.iter().all(|m| signed(m) || built(m)),
        Schema::Union(members) => {
            members.iter().all(|member| subterms.contains(member))
                || subterms.iter().any(|union| match union {
                    Schema::Union(theirs) => {
                        let read = members_at_every_depth(theirs);
                        members.iter().all(|m| read.contains(&m))
                    }
                    _ => false,
                })
                || built(term)
        }
        _ => built(term),
    }
}

/// Whether `term` is a sequence a meet of sequences among `subterms` is asked as
/// ([`readings::a_meet_of_sequences_as_one`]): at each position the meet of the
/// items the subterms hold there, under bounds the subterms carry.
///
/// The product step of [LN26, §5.1], which pushes the intersection below the
/// constructor and argues that "only a finite number of such new types are
/// created": an item is a meet of signed subterms, a bound one a subterm
/// carries, and finitely many of each exist.
fn one_sequence_of(term: &Schema, subterms: &[Schema]) -> bool {
    let signed = |t: &Schema| {
        subterms.contains(t) || matches!(t, Schema::Complement(inner) if subterms.contains(inner))
    };
    let (shape, bounds) = match term {
        Schema::Seq { shape, .. } => (shape, &[][..]),
        Schema::Refine { base, constraints } => match &**base {
            Schema::Seq { shape, .. } => (shape, &constraints[..]),
            _ => return false,
        },
        _ => return false,
    };
    let item = |item: &Schema| match item {
        Schema::Nothing | Schema::Anything(_) => true,
        Schema::Intersection(members) => members.iter().all(signed),
        item => signed(item),
    };
    shape.prefix.iter().chain(shape.tail.as_deref()).all(item)
        && bounds.iter().all(|bound| {
            subterms.iter().any(
                |t| matches!(t, Schema::Refine { constraints, .. } if constraints.contains(bound)),
            )
        })
}

/// A union's members at every depth: each nested union itself, which a rule
/// asking a union of some members keeps whole, and the members the rules read
/// through it ([`flat_members`](super::flat_members)). A finite tree holds
/// finitely many.
fn members_at_every_depth(members: &[Schema]) -> Vec<&Schema> {
    let mut found = Vec::new();
    let mut pending: Vec<&Schema> = members.iter().collect();
    while let Some(member) = pending.pop() {
        found.push(member);
        if let Schema::Union(inner) = member {
            pending.extend(inner.iter());
        }
    }
    found
}

/// The goals a query asks that fall outside the closure of its subterms, and
/// how many of the goals inside it are built rather than found.
fn outside_the_closure(
    sub: &Schema,
    sup: &Schema,
    defs: &[Schema],
) -> (Vec<(Schema, Schema)>, usize) {
    let (_, asked) = goals::recorded(|| sub.subtype_relation_under(sup, &PlainClasses, defs));
    let found = [subterms(sub, defs), subterms(sup, defs)].concat();
    let built = asked
        .iter()
        .filter(|(a, b)| !found.contains(a) || !found.contains(b))
        .count();
    let outside = asked
        .into_iter()
        .filter(|(a, b)| !in_closure(a, &found) || !in_closure(b, &found))
        .collect();
    (outside, built)
}

/// A product over `kinds` against the products of every two of them, less
/// the last `missing`: the shape the product rule narrows most.
fn corners(kinds: &[Schema], missing: usize) -> (Schema, Schema) {
    let pair = |a: &Schema, b: &Schema| Schema::tuple(SeqShape::fixed([a.clone(), b.clone()]));
    let any = Schema::union(kinds.iter().cloned());
    let mut products: Vec<Schema> = kinds
        .iter()
        .flat_map(|a| kinds.iter().map(move |b| (a, b)))
        .map(|(a, b)| pair(a, b))
        .collect();
    products.truncate(products.len() - missing);
    (pair(&any, &any), Schema::union(products))
}

/// The shapes the budget is measured over, the recursive pairs the trail is
/// for, and the product rule's split, each with the definitions it reads.
fn measured_pairs() -> Vec<(Schema, Schema, Vec<Schema>)> {
    let element = nested_lists(2, Schema::Int);
    let wider = nested_lists(2, Schema::union([Schema::Int, Schema::Str]));
    let tree = tree_defs();
    let list_tree = vec![Schema::union([
        Schema::Int,
        list_of(Schema::Ref(DefIx::new(0))),
    ])];
    vec![
        (
            repeating_record(8, &element),
            repeating_record(8, &wider),
            vec![],
        ),
        (
            repeating_tuple(6, &element),
            repeating_tuple(6, &wider),
            vec![],
        ),
        (
            nested_lists(6, Schema::Bool),
            nested_lists(6, Schema::Int),
            vec![],
        ),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::union([Schema::Ref(DefIx::new(0)), Schema::Str]),
            tree.clone(),
        ),
        (Schema::Ref(DefIx::new(0)), Schema::Ref(DefIx::new(0)), tree),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::union([Schema::Int, list_of(Schema::ANYTHING)]),
            list_tree.clone(),
        ),
        (
            list_of(Schema::Ref(DefIx::new(0))),
            Schema::Ref(DefIx::new(0)),
            list_tree,
        ),
        (product_subject(), product_split(), vec![]),
    ]
}

// THEORY: regularity-bounds-the-goals
/// Every goal a query asks is a pair drawn from the closure of its subterms.
///
/// The closure is finite, so the distinct goals number at most its square, and
/// a recursion whose pairs are drawn from a finite set and whose trail cuts a
/// pair that comes back terminates without a budget. That is the argument
/// [FCB08, §6.9] makes over `N(A)`, which the closure stands in for -- finite by
/// the argument of [LN26, §5.1], since a narrowing builds new atoms and finitely
/// many -- and [HVP05, Thm. 4] makes over its automaton's states. What a test can hold is the premise: no rule builds a
/// term outside the closure. The product shapes build one narrowing per branch
/// and position, and the last shape a union of the branches its subject can
/// meet, so each is also held to have built something, which keeps the row
/// from passing on a shape that never reached the rule.
#[test]
fn every_goal_a_query_asks_is_drawn_from_the_closure_of_its_subterms() {
    let kinds = [Schema::Int, Schema::Str, Schema::Bytes, Schema::NoneType];
    let (product, all) = corners(&kinds, 0);
    let (_, all_but_one) = corners(&kinds, 1);
    assert_eq!(ask(&product, &all, &[]).0, Relation::Holds);
    assert_eq!(ask(&product, &all_but_one, &[]).0, Relation::Fails);
    // A branch refutes and another shares no value with the subject, so the
    // union rule asks the subject against the two it can meet.
    let words = list_of(Schema::union([Schema::Int, Schema::Str]));
    let split_by_kind =
        Schema::union([list_of(Schema::Int), list_of(Schema::Str), Schema::NoneType]);
    assert_eq!(ask(&words, &split_by_kind, &[]).0, Relation::Fails);
    // A record whose field takes the four kinds, against the records fixing
    // it: the record rule narrows the field by each member's.
    let record = |field: Schema| {
        Schema::record(
            vec![crate::ir::Field {
                name: "a".into(),
                schema: field,
                required: true,
            }],
            crate::ir::Openness::Closed,
        )
    };
    let split = record(Schema::union(kinds.iter().cloned()));
    let by_kind = Schema::union(kinds.iter().cloned().map(record));
    let by_kind_but_one = Schema::union(kinds.iter().skip(1).cloned().map(record));
    assert_eq!(ask(&split, &by_kind, &[]).0, Relation::Holds);
    assert_eq!(ask(&split, &by_kind_but_one, &[]).0, Relation::Fails);
    let built = [
        (product.clone(), all),
        (product, all_but_one),
        (product_subject(), product_split()),
        (words, split_by_kind),
        (split.clone(), by_kind),
        (split, by_kind_but_one),
    ];
    for (sub, sup) in &built {
        let (outside, built) = outside_the_closure(sub, sup, &[]);
        assert_eq!(outside, [], "{sub:?} <= {sup:?} asked outside the closure");
        assert!(built >= 1, "{sub:?} <= {sup:?} built no goal of its own");
    }
    for (sub, sup, defs) in measured_pairs() {
        let (outside, _) = outside_the_closure(&sub, &sup, &defs);
        assert_eq!(outside, [], "{sub:?} <= {sup:?} asked outside the closure");
    }
}

// THEORY: regularity-bounds-the-goals
/// The distinct goals a query asks, measured against the pairs of subterms the
/// two schemas have.
///
/// The closure bounds them by an argument, and its square is a bound on
/// nothing a caller waits for: a meet per subset of the subterms. This is the
/// number the shapes in hand reach, which is far below it. Each shape here
/// asks fewer distinct goals than its sides have subterm pairs, the product
/// rule's split included -- its narrowings are few and each is asked once --
/// and the row is what says so when a rule starts asking more.
#[test]
fn the_goals_a_query_asks_are_pairs_of_the_subterms() {
    assert_eq!(
        ask(&product_subject(), &product_split(), &[]).0,
        Relation::Holds
    );
    for (sub, sup, defs) in &measured_pairs() {
        let (distinct, bound) = distinct_against_subterm_pairs(sub, sup, defs);
        assert!(
            distinct >= 1,
            "the counter saw no goal for {sub:?} <= {sup:?}"
        );
        assert!(
            distinct <= bound,
            "{sub:?} <= {sup:?} asked {distinct} distinct goals over {bound} subterm pairs"
        );
    }
}

// THEORY: the-trail-holds-terms
/// The longest trail any recursive shape in hand builds is two pairs.
///
/// The deviation's cost is a scan over the trail at every recursive goal, and
/// the sentence that bounds the cost is a number the recorder reads: every
/// recursive pair the counter's shapes carry, and the pairs the fixpoint laws
/// are written over, are asked with the trail's depth recorded, and the
/// deepest is the figure the page states. Two definitions naming each other
/// through a list and its complement unfold both references, which is two,
/// and a list over one of them against a union holding it reaches two as
/// well. A shape whose trail grows past it fails here by name, which is what
/// turns the sentence into a claim the tree holds.
#[test]
fn the_longest_trail_any_recursive_shape_builds_is_two_pairs() {
    let tree = tree_defs();
    let list_tree = vec![Schema::union([
        Schema::Int,
        list_of(Schema::Ref(DefIx::new(0))),
    ])];
    let word_and_not = vec![
        Schema::union([Schema::Str, list_of(Schema::Ref(DefIx::new(0)))]),
        Schema::union([
            Schema::Int,
            list_of(Schema::Complement(Arc::new(Schema::Ref(DefIx::new(1))))),
        ]),
    ];
    let pairs: Vec<(Schema, Schema, Vec<Schema>)> = vec![
        (
            Schema::Ref(DefIx::new(0)),
            Schema::Ref(DefIx::new(0)),
            tree.clone(),
        ),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::union([Schema::Ref(DefIx::new(0)), Schema::Str]),
            tree,
        ),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::union([Schema::Int, list_of(Schema::ANYTHING)]),
            list_tree.clone(),
        ),
        (
            list_of(Schema::Ref(DefIx::new(0))),
            Schema::Ref(DefIx::new(0)),
            list_tree,
        ),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::Ref(DefIx::new(1)),
            word_and_not.clone(),
        ),
        (
            Schema::Ref(DefIx::new(1)),
            Schema::Ref(DefIx::new(0)),
            word_and_not.clone(),
        ),
        (
            list_of(Schema::Ref(DefIx::new(1))),
            Schema::union([Schema::Int, Schema::Ref(DefIx::new(1))]),
            word_and_not,
        ),
    ];
    let mut deepest = 0;
    for (sub, sup, defs) in &pairs {
        let (_, counts) = ask(sub, sup, defs);
        assert!(
            counts.asked >= 1,
            "the counter saw no goal for {sub:?} <= {sup:?}"
        );
        assert!(
            counts.deepest <= 2,
            "{sub:?} <= {sup:?} grew the trail to {} pairs",
            counts.deepest
        );
        deepest = deepest.max(counts.deepest);
    }
    assert_eq!(deepest, 2, "no shape reached the length the page states");
}

// THEORY: the-assumption-set-is-popped
/// A goal reached by two paths is derived twice, so a family whose every level
/// reaches the next twice doubles its work per level until the budget declines.
///
/// The family is the one Gapeyev, Levin & Pierce (JFP 2002) analyze for an
/// algorithm that keeps no proved pair across its calls, where the two
/// directions of an arrow reach the level below twice; the paper is not on the
/// shelf, so this describes it rather than citing a section. A second constructor does it here -- a
/// pair of `T` and `list[T]` -- and the goals stay few, two per level, while
/// the steps double. Seventeen levels decide; the eighteenth spends the whole
/// budget and is declined, which is the conservative answer.
#[test]
fn a_goal_reached_by_two_paths_is_derived_twice_until_the_budget_declines() {
    let level = |inner: Schema| Schema::tuple(SeqShape::fixed([inner.clone(), list_of(inner)]));
    let (mut sub, mut sup) = (Schema::Int, Schema::union([Schema::Int, Schema::Str]));
    let mut spent = Vec::new();
    for depth in 1..=18 {
        sub = level(sub);
        sup = level(sup);
        if depth <= 8 {
            let counts = ask(&sub, &sup, &[]).1;
            assert_eq!(
                counts.asked - counts.repeated,
                2 * depth + 1,
                "depth {depth}"
            );
        }
        let budget = Budget::new(DECISION_BUDGET);
        let answer = sub.subtype_relation(&sup, &PlainClasses, &[], &budget);
        spent.push((answer, DECISION_BUDGET - budget.left()));
    }
    for (depth, pair) in spent.windows(2).enumerate().take(16) {
        assert!(pair[1].1 >= 2 * pair[0].1, "depth {}: {pair:?}", depth + 2);
    }
    assert!(
        spent[..17]
            .iter()
            .all(|(answer, _)| *answer == Relation::Holds)
    );
    assert_eq!(spent[17], (Relation::Unknown, DECISION_BUDGET));
}

/// A union subject stops asking members once one declines with the budget
/// spent. A goal asked on a spent budget answers by the terms' membership or
/// declines, and is never refuted, so the members after that decline change
/// nothing, and asked anyway each scans the supertype's branches. With steps
/// left a decline is not the end: a member after it may refute the union.
#[test]
fn a_union_stops_at_a_decline_its_budget_cannot_follow() {
    let members = Schema::union([
        Schema::Int,
        Schema::Str,
        Schema::Bytes,
        Schema::Float,
        Schema::NoneType,
    ]);
    let lists = Schema::union([list_of(Schema::Int), list_of(Schema::Str)]);
    let (verdict, counts) =
        goals::counted(|| members.subtype_relation(&lists, &PlainClasses, &[], &Budget::new(1)));
    assert_eq!(verdict, Relation::Unknown);
    // The pair, which spends the one step, and its first member, which finds
    // none left: the four members after it are not asked.
    assert_eq!(counts.asked, 2);

    let pattern = |words: &str| refined(Schema::Str, Constraint::Regex(words.into()));
    let steps = || Budget::new(DECISION_BUDGET);
    assert_eq!(
        pattern("a").subtype_relation(&pattern("b"), &PlainClasses, &[], &steps()),
        Relation::Unknown,
        "the rules decline one pattern below another"
    );
    // Written raw, so the member the rules decline is asked first.
    let declined_first = Schema::Union(vec![pattern("a"), Schema::Int].into());
    assert_eq!(
        declined_first.subtype_relation(&pattern("b"), &PlainClasses, &[], &steps()),
        Relation::Fails
    );
}

// THEORY: regularity-bounds-the-goals
/// A supertype union is respelled through the union nested in it, inside the
/// closure.
///
/// The pair: `{"n?": Any}` below `(¬Any ∪ {}) ∪ R`, `R` a chain of `n`. One
/// unfolding of `R` holds the subject whole, so no reading of one unfolding
/// names a value outside it and the pair reaches the respelling, which takes
/// the complement out of the supertype's members read through the
/// nested union and asks the subject below `{} ∪ R`, which names a member of
/// each of two union subterms: a union of some members of the supertype read
/// flat, and so a goal the closure holds. It is a goal built rather than
/// found, which keeps the row from passing on a pair that never reached the
/// respelling.
#[test]
fn a_supertype_union_is_respelled_through_its_nested_union_inside_the_closure() {
    let empty = Schema::KeyedMap {
        fields: Vec::new().into(),
        defaults: Vec::new().into(),
    };
    let sub = Schema::KeyedMap {
        fields: vec![crate::ir::Field {
            name: "n".into(),
            schema: Schema::ANYTHING,
            required: false,
        }]
        .into(),
        defaults: Vec::new().into(),
    };
    let sup = Schema::Union(
        vec![
            Schema::Union(vec![Schema::Complement(Arc::new(Schema::ANYTHING)), empty].into()),
            Schema::Ref(DefIx::new(0)),
        ]
        .into(),
    );
    let defs = vec![Schema::Union(
        vec![
            Schema::NoneType,
            Schema::KeyedMap {
                fields: vec![crate::ir::Field {
                    name: "n".into(),
                    schema: Schema::Ref(DefIx::new(0)),
                    required: false,
                }]
                .into(),
                defaults: Vec::new().into(),
            },
        ]
        .into(),
    )];
    let (outside, built) = outside_the_closure(&sub, &sup, &defs);
    assert!(outside.is_empty(), "asked outside the closure: {outside:?}");
    assert!(built >= 1, "the respelling built no goal");
    // A nested union kept whole beside a member of the union holding it, as
    // the union rule asks it in the constructors' order: `None` below
    // `(¬c ∪ (bool)) ∪ R` asks `None` below `(bool) ∪ ¬c`.
    let grouped = Schema::Union(
        vec![
            Schema::Union(
                vec![
                    Schema::Complement(Arc::new(Schema::Literal(crate::ir::ConstIx::new(0)))),
                    Schema::Union(vec![Schema::Bool].into()),
                ]
                .into(),
            ),
            Schema::Ref(DefIx::new(0)),
        ]
        .into(),
    );
    let bools = vec![
        Schema::Union(vec![Schema::Bool, list_of(Schema::Ref(DefIx::new(0)))].into()),
        Schema::Union(vec![Schema::ANYTHING, list_of(Schema::Ref(DefIx::new(0)))].into()),
    ];
    let (outside, built) = outside_the_closure(&Schema::NoneType, &grouped, &bools);
    assert!(outside.is_empty(), "asked outside the closure: {outside:?}");
    assert!(built >= 1, "the union rule built no goal");
}

// THEORY: regularity-bounds-the-goals
/// A meet of sequences asked as the one sequence of its items' meets asks goals
/// inside the closure, in each of the three forms the reading builds: the one
/// sequence alone, a union of them where a union of sequences is distributed,
/// and a meet of one with a member no sequence is.
///
/// The subjects are the meets a translation from JSON Schema writes for
/// `items` beside `minItems`, below a list of records with a required key: the
/// pair the reading was written for, which the rules and the set
/// representation decline as written. Each row is also held to have built a
/// goal, which keeps it from passing on a pair that never reached the reading.
#[test]
fn a_meet_of_sequences_is_asked_as_one_sequence_inside_the_closure() {
    use crate::ir::{MapClause, Spelling};
    let json = Schema::Ref(DefIx::new(0));
    let a_map = || {
        Schema::keyed_map(
            Vec::new(),
            vec![MapClause {
                key: Schema::Str,
                value: json.clone(),
            }],
        )
    };
    let defs = vec![Schema::union([
        Schema::NoneType,
        Schema::Bool,
        Schema::Int,
        Schema::Float,
        Schema::Str,
        list_of(json.clone()),
        a_map(),
    ])];
    let empty = || Schema::list(SeqShape::fixed([]));
    let at_least_one = Schema::refine(
        list_of(Schema::Anything(Spelling::Any)),
        vec![Constraint::MinLen(1)],
    );
    let record = Schema::meet([
        a_map(),
        Schema::record(
            vec![Field {
                name: "t".into(),
                schema: Schema::Str,
                required: true,
            }],
            Openness::Closed,
        ),
    ]);
    let records = Schema::meet([
        list_of(json.clone()),
        Schema::union([empty(), list_of(record)]),
    ]);
    let one = Schema::meet([
        list_of(json.clone()),
        list_of(Schema::Str),
        at_least_one.clone(),
    ]);
    let a_union = Schema::meet([
        list_of(json.clone()),
        Schema::union([empty(), list_of(Schema::Str), list_of(Schema::Bytes)]),
    ]);
    let with_a_member = Schema::meet([
        list_of(json),
        Schema::union([empty(), list_of(Schema::Str)]),
        at_least_one,
        Schema::Complement(Arc::new(Schema::set(Schema::Int))),
    ]);
    for subject in [one, a_union, with_a_member] {
        let (outside, built) = outside_the_closure(&subject, &records, &defs);
        assert!(outside.is_empty(), "asked outside the closure: {outside:?}");
        assert!(built >= 1, "{subject:?} reached no reading that builds");
    }
}

/// The same bound, asked of drawn pairs over drawn definitions.
mod drawn {
    use super::{distinct_against_subterm_pairs, outside_the_closure};
    use crate::laws::{drawn_defs, recursive_schema};
    use proptest::prelude::*;

    proptest! {
        // A bounded shrink, so a broken bound cannot turn a caught mutation
        // into a run that outlasts a sweep: see `budget::law`.
        #![proptest_config(ProptestConfig {
            max_shrink_time: 2_000,
            ..ProptestConfig::default()
        })]

        // THEORY: regularity-bounds-the-goals
        /// The goals a drawn query asks are drawn from the closure of its
        /// subterms, and number no more than its subterm pairs, over drawn
        /// definitions.
        ///
        /// The chosen pairs above hold both on the shapes the budget is
        /// measured over; this holds them on whatever the recursive fragment
        /// draws, references into drawn definitions included, so a rule that
        /// starts building goals past the closure is caught on a shape nobody
        /// chose.
        #[test]
        fn the_goals_a_drawn_query_asks_are_pairs_of_the_subterms(
            sub in recursive_schema(),
            sup in recursive_schema(),
            defs in drawn_defs(),
        ) {
            let (distinct, pairs) = distinct_against_subterm_pairs(&sub, &sup, &defs);
            prop_assert!(distinct >= 1, "the counter saw no goal for {sub:?} <= {sup:?}");
            let (outside, _) = outside_the_closure(&sub, &sup, &defs);
            prop_assert!(
                outside.is_empty(),
                "{sub:?} <= {sup:?} asked {outside:?} outside the closure"
            );
            prop_assert!(
                distinct <= pairs,
                "{distinct} distinct goals over {pairs} subterm pairs for {sub:?} <= {sup:?}"
            );
        }
    }
}

// THEORY: recursive-subtyping
/// A goal met again under no constructor is refused by the recorder, on each of
/// the three lists a goal is held open on. Each table is one the build refuses, a
/// reference standing unguarded in its own definition, since no contractive one
/// reaches this: the rows say the recorder is wired to the hit.
#[should_panic(expected = "under no constructor")]
#[test]
fn a_goal_met_again_on_the_trail_stands_under_a_constructor() {
    let this = Schema::Ref(DefIx::new(0));
    let defs = [Schema::union([this.clone(), Schema::Int])];
    let budget = Budget::new(DECISION_BUDGET);
    let _ = this.subtype_relation(&Schema::Str, &super::NoLeafRelations, &defs, &budget);
}

// THEORY: recursive-subtyping
/// See [`a_goal_met_again_on_the_trail_stands_under_a_constructor`].
#[should_panic(expected = "under no constructor")]
#[test]
fn a_reference_met_again_stands_under_a_constructor() {
    let this = Schema::Ref(DefIx::new(0));
    let defs = [Schema::union([this.clone(), Schema::Int])];
    let budget = Budget::new(DECISION_BUDGET);
    let _ = this.verdict_rec(&mut super::Cx::new(&super::NoLeafRelations, &defs, &budget));
}

// THEORY: recursive-subtyping
/// See [`a_goal_met_again_on_the_trail_stands_under_a_constructor`].
#[should_panic(expected = "under no constructor")]
#[test]
fn a_meet_met_again_stands_under_a_constructor() {
    let budget = Budget::new(DECISION_BUDGET);
    let goal = vec![Schema::Int, Schema::Str];
    let _open = super::emptiness::OpenMeet::open(&budget, goal.clone());
    let _again = super::emptiness::OpenMeet::open(&budget, goal);
}

// THEORY: recursive-subtyping
/// A recursive pair meets its goal again under the constructor its definition
/// guards it with, and the recorder passes it: `μt. list[t]` below itself spelled
/// apart holds, and the emptiness of a list of itself is read through the tail.
#[test]
fn a_goal_met_again_under_a_constructor_passes() {
    let at = |index| Schema::Ref(DefIx::new(index));
    let defs = [
        Schema::list(SeqShape::homogeneous(at(0))),
        Schema::list(SeqShape::homogeneous(at(1))),
    ];
    let budget = Budget::new(DECISION_BUDGET);
    assert_eq!(
        at(0).subtype_relation(&at(1), &super::NoLeafRelations, &defs, &budget),
        Relation::Holds
    );
}
