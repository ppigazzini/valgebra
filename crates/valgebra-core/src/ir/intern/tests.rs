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

/// Build one nesting twice, over a leaf whose own handle does not share the
/// nesting's slot.
///
/// A node is hashed by the **address** of the handles it carries
/// ([`hash_node`]), so which slot a nesting lands in is a fact about this
/// process's allocator rather than about the schema. Where a leaf and the
/// nesting around it collide, interning the leaf evicts the nesting and the
/// second build shares nothing -- which is the direct-mapped table doing
/// exactly what [`SLOTS`] says it does, and not the sharing the rows below
/// are about. Written with one fixed leaf, those rows failed about one run
/// in a thousand, which is a red lane nobody can reproduce.
///
/// So the leaf is chosen rather than fixed: the first that lands clear of
/// its own nesting, searched the way [`parted_pair`] searches.
fn nesting_built_twice(wrap: &dyn Fn(Arc<Schema>) -> Schema) -> (Arc<Schema>, Arc<Schema>) {
    for leaf in [
        Schema::Bytes,
        Schema::Float,
        Schema::Str,
        Schema::Int,
        Schema::Bool,
        Schema::NoneType,
    ] {
        // Held for the length of the check, so the address the slot is
        // read from is the address the two builds below will hash.
        let inner = node(leaf.clone());
        if slot(&wrap(Arc::clone(&inner))) != slot(&leaf) {
            return (
                node(wrap(node(leaf.clone()))),
                node(wrap(node(leaf.clone()))),
            );
        }
    }
    panic!("every leaf shares a slot with its own nesting");
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

    let (node_one, node_two) = nesting_built_twice(&|inner| Schema::Complement(inner));
    assert!(Arc::ptr_eq(&node_one, &node_two));

    let (seq_one, seq_two) = nesting_built_twice(&|inner| Schema::Seq {
        container: SeqKind::Tuple,
        shape: SeqShape {
            prefix: members(&[]),
            tail: Some(inner),
        },
    });
    assert!(Arc::ptr_eq(&seq_one, &seq_two));
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
///
/// Over a leaf chosen clear of its own nesting, for the reason
/// [`nesting_built_twice`] carries: a node is hashed by the address of the
/// handles it holds, so a fixed leaf collides with the nesting around it in
/// some processes and not others, and where it does the inner build evicts
/// the outer. That is the table's own bound doing what it says, and this
/// row is about sharing.
#[test]
fn sharing_a_child_shares_the_parent_over_it() {
    let (one, two) = nesting_built_twice(&|inner| Schema::Complement(inner));
    assert!(Arc::ptr_eq(&one, &two));
}
