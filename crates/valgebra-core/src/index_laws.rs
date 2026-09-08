use std::sync::Arc;

use super::*;
use proptest::prelude::*;

/// One past the highest pool index the generator uses, so an interning table
/// over `0..POOL_LEN` covers every index a generated schema holds. `remap`
/// asserts that coverage in debug, so a table shorter than this fails the
/// test rather than silently keeping an index.
const POOL_LEN: usize = 8;

/// The largest shift the laws compose, small enough that two of them stay
/// far from overflowing and large enough to move every index off its slot.
const MAX_SHIFT: usize = 8;

fn field(schema: Schema) -> Field {
    Field {
        name: "f".into(),
        schema,
        required: true,
    }
}

/// Every index-carrying node, under every structural node that has to pass a
/// remap to its children.
fn indexed_schema() -> impl Strategy<Value = Schema> {
    let leaf = prop_oneof![
        Just(Schema::Int),
        Just(Schema::Literal(ConstIx::new(0))),
        Just(Schema::Literal(ConstIx::new(2))),
        Just(Schema::Instance(ClassIx::new(1))),
        Just(Schema::Ref(DefIx::new(0))),
        Just(Schema::Ref(DefIx::new(3))),
        // Not a pool index: a self-reference token names a `recursive`
        // definition being built, so no shift may touch it.
        Just(Schema::SelfRef(7)),
    ];
    leaf.prop_recursive(4, 48, 3, |inner| {
        prop_oneof![
            inner.clone().prop_map(Schema::set),
            inner.clone().prop_map(Schema::frozen_set),
            inner.clone().prop_map(|s| Schema::Complement(Arc::new(s))),
            proptest::collection::vec(inner.clone(), 1..3).prop_map(|m| Schema::Union(m.into())),
            proptest::collection::vec(inner.clone(), 1..3)
                .prop_map(|m| Schema::Intersection(m.into())),
            inner
                .clone()
                .prop_map(|s| Schema::list(SeqShape::homogeneous(s))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| Schema::tuple(SeqShape::prefix_tail([a], b))),
            proptest::collection::vec(inner.clone(), 0..3)
                .prop_map(|elements| Schema::list(SeqShape::fixed(elements))),
            (inner.clone(), inner.clone())
                .prop_map(|(key, value)| Schema::mapping(MapClause { key, value })),
            inner
                .clone()
                .prop_map(|s| Schema::record(vec![field(s)], Openness::Closed)),
            inner.clone().prop_map(|s| Schema::AttrRecord {
                fields: vec![field(s)].into(),
            }),
            inner.prop_map(|s| Schema::Refine {
                base: Arc::new(s),
                constraints: vec![
                    Constraint::Ge(OperandIx::new(5)),
                    Constraint::MultipleOf(OperandIx::new(5)),
                    Constraint::Predicate(PredIx::new(6)),
                    // Neither is a pool index; both must survive untouched.
                    Constraint::MinLen(1),
                    Constraint::Regex("x".to_owned()),
                ]
                .into(),
            }),
        ]
    })
}

/// The interning table that sends every pool slot `by` places along, so
/// reindexing through it is a shift and the two operations are comparable.
fn shift_table(by: usize) -> Vec<usize> {
    (0..POOL_LEN).map(|slot| slot + by).collect()
}

/// Every pool index the schema holds, in traversal order, and every
/// definitions index beside it.
///
/// This is a **second, independent** enumeration of the payload sites,
/// written here rather than reached for in the IR on purpose: a law that
/// collects the sites the way the remap visits them cannot notice a site
/// neither of them visits. The compiler holds it complete — the match takes no
/// wildcard, so a new variant stops the tests compiling until its payloads are
/// declared here too.
fn indices(schema: &Schema, pool: &mut Vec<usize>, defs: &mut Vec<usize>) {
    match schema {
        Schema::Anything(_)
        | Schema::Nothing
        | Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Str
        | Schema::Bytes
        | Schema::SelfRef(_) => {}
        Schema::Literal(index) => pool.push(index.get()),
        Schema::Instance(index) => pool.push(index.get()),
        Schema::Ref(index) => defs.push(index.get()),
        Schema::Coll { element: inner, .. } | Schema::Complement(inner) => {
            indices(inner, pool, defs);
        }
        Schema::Union(members) | Schema::Intersection(members) => {
            for member in members.iter() {
                indices(member, pool, defs);
            }
        }
        Schema::Seq { shape, .. } => {
            for element in shape.elements() {
                indices(element, pool, defs);
            }
        }
        Schema::KeyedMap { fields, defaults } => {
            for field in fields.iter() {
                indices(&field.schema, pool, defs);
            }
            for clause in defaults.iter() {
                indices(&clause.key, pool, defs);
                indices(&clause.value, pool, defs);
            }
        }
        Schema::AttrRecord { fields } => {
            for field in fields.iter() {
                indices(&field.schema, pool, defs);
            }
        }
        Schema::Refine { base, constraints } => {
            indices(base, pool, defs);
            for constraint in constraints.iter() {
                match constraint {
                    Constraint::Ge(index)
                    | Constraint::Gt(index)
                    | Constraint::Le(index)
                    | Constraint::Lt(index)
                    | Constraint::MultipleOf(index) => pool.push(index.get()),
                    Constraint::Predicate(index) => pool.push(index.get()),
                    Constraint::MinLen(_) | Constraint::MaxLen(_) | Constraint::Regex(_) => {}
                }
            }
        }
    }
}

proptest! {
    /// A shift of zero is the identity. The law that catches the opposite of
    /// a missed site: a payload moved by a shift nobody asked for.
    #[test]
    fn a_zero_shift_is_the_identity(schema in indexed_schema()) {
        prop_assert_eq!(
            schema.shifted(PoolShift::new(0), DefShift::new(0)),
            schema.clone()
        );
    }

    /// Shifting twice equals shifting once by the sum, in both index spaces
    /// independently. A site moved twice, or not at all, breaks this; a site
    /// moved by the wrong space's distance breaks it as soon as the two
    /// distances differ, which the generator's range makes likely.
    #[test]
    fn shifts_compose_by_addition(
        schema in indexed_schema(),
        pool_a in 0..MAX_SHIFT,
        pool_b in 0..MAX_SHIFT,
        defs_a in 0..MAX_SHIFT,
        defs_b in 0..MAX_SHIFT,
    ) {
        let twice = schema
            .shifted(PoolShift::new(pool_a), DefShift::new(defs_a))
            .shifted(PoolShift::new(pool_b), DefShift::new(defs_b));
        let once = schema.shifted(
            PoolShift::new(pool_a + pool_b),
            DefShift::new(defs_a + defs_b),
        );
        prop_assert_eq!(twice, once);
    }

    /// Every payload moves, and moves by its own space's distance.
    ///
    /// This is the law that catches a **missed** site, and the two above
    /// cannot: a payload the remap never touches satisfies both of them, since
    /// standing still is the identity and composes with itself. It catches one
    /// by counting the sites independently, so it holds however the remap is
    /// factored internally.
    #[test]
    fn every_index_moves_by_its_own_space(
        schema in indexed_schema(),
        pool_by in 1..MAX_SHIFT,
        defs_by in 1..MAX_SHIFT,
    ) {
        let (mut pool_before, mut defs_before) = (Vec::new(), Vec::new());
        indices(&schema, &mut pool_before, &mut defs_before);
        let shifted = schema.shifted(PoolShift::new(pool_by), DefShift::new(defs_by));
        let (mut pool_after, mut defs_after) = (Vec::new(), Vec::new());
        indices(&shifted, &mut pool_after, &mut defs_after);

        prop_assert_eq!(
            pool_after,
            pool_before.iter().map(|slot| slot + pool_by).collect::<Vec<_>>()
        );
        prop_assert_eq!(
            defs_after,
            defs_before.iter().map(|slot| slot + defs_by).collect::<Vec<_>>()
        );
    }

    /// Interning sends every pool payload through the table, and every
    /// definitions payload along by the offset. The same independent count as
    /// above, held against the other entry point.
    #[test]
    fn interning_sends_every_index_through_the_table(
        schema in indexed_schema(),
        pool_by in 1..MAX_SHIFT,
        defs_by in 1..MAX_SHIFT,
    ) {
        let table = shift_table(pool_by);
        let (mut pool_before, mut defs_before) = (Vec::new(), Vec::new());
        indices(&schema, &mut pool_before, &mut defs_before);
        let interned = schema.reindexed(&table, DefShift::new(defs_by));
        let (mut pool_after, mut defs_after) = (Vec::new(), Vec::new());
        indices(&interned, &mut pool_after, &mut defs_after);

        prop_assert_eq!(
            pool_after,
            pool_before.iter().map(|slot| table[*slot]).collect::<Vec<_>>()
        );
        prop_assert_eq!(
            defs_after,
            defs_before.iter().map(|slot| slot + defs_by).collect::<Vec<_>>()
        );
    }

    /// Interning through the identity table, with no definitions offset,
    /// leaves the schema alone.
    #[test]
    fn reindexing_through_the_identity_table_is_the_identity(
        schema in indexed_schema(),
    ) {
        prop_assert_eq!(
            schema.reindexed(&shift_table(0), DefShift::new(0)),
            schema.clone()
        );
    }

    /// Interning through a table that is itself a shift agrees with shifting.
    /// The two operations visit the same payload sites by construction, so a
    /// site one of them handles and the other walks past shows up here and
    /// nowhere else.
    #[test]
    fn reindexing_through_a_shifted_table_is_a_shift(
        schema in indexed_schema(),
        pool in 0..MAX_SHIFT,
        defs in 0..MAX_SHIFT,
    ) {
        prop_assert_eq!(
            schema.reindexed(&shift_table(pool), DefShift::new(defs)),
            schema.shifted(PoolShift::new(pool), DefShift::new(defs))
        );
    }
}
