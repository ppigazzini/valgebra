//! valgebra schema intermediate representation.
//!
//! A schema denotes a set of Python values; validation is membership. This
//! crate is pure Rust: it defines the IR, the denotation of every node, and the
//! structured [`Violation`] produced when membership fails. Inspecting a Python
//! object requires `PyO3`, so the validator walk itself lives in the bindings
//! crate; this crate is the stable, language-agnostic core.
//!
//! The crate forbids `unsafe`: the security policy's no-unsafe guarantee is
//! compiler-enforced here, not merely asserted, so a future `unsafe` block fails
//! the build instead of silently voiding it.
#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};

mod decision;
pub mod descr;
mod ir;
mod simplify;
mod violation;

pub use decision::{Kind, LeafRelations, NoLeafRelations, Verdict};
pub use ir::{
    ClassIx, ConstIx, Constraint, DefIx, DefShift, Field, Guarded, MapClause, Openness, OperandIx,
    PathSegment, PoolShift, PredIx, Schema, SeqKind, SeqShape,
};
pub use violation::Violation;

/// Fresh tokens for the transient [`Schema::SelfRef`] marker, so no two
/// `recursive` definitions ever resolve each other's self-references.
///
/// **Process-unique, deliberately.** The placeholder carrying a token is an
/// ordinary Python object, so a caller can keep one past the builder call that
/// gave it meaning and pass it into another -- on this thread or on any other.
/// The binding refuses that, by asking whether the token names a definition
/// currently being built; and that question is only sound while a token means
/// one definition across the whole process. A per-thread counter would hand two
/// threads the same first token, and one thread's escaped placeholder would then
/// answer to the other's open definition: not a refusal, but a silently
/// different schema. The counter is shared so that the tokens cannot collide.
static NEXT_SELF_TOKEN: AtomicU64 = AtomicU64::new(0);

/// Allocate a fresh self-reference token for a `recursive` definition.
///
/// `Relaxed` is the whole ordering this needs. The token is compared for
/// equality and never orders anything, so the one guarantee wanted from the
/// counter is that no two calls return the same value -- which `fetch_add`
/// gives under any ordering.
#[must_use]
pub fn fresh_self_token() -> u64 {
    NEXT_SELF_TOKEN.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// One representative of every `Schema` variant, each carrying a child where
    /// the variant can hold one.
    ///
    /// The traversal and the functor are held to each other over this list, so a
    /// variant missing here is a variant the agreement is not checked for.
    /// `tests/test_node_matrix.py` reads the variant list out of the IR and fails
    /// when one carries no row, which is what stops this list going stale.
    fn every_variant() -> Vec<Schema> {
        let field = |name: &str| Field {
            name: name.to_owned(),
            schema: Schema::Int,
            required: true,
        };
        vec![
            Schema::Anything,
            Schema::Dynamic,
            Schema::Nothing,
            Schema::NoneType,
            Schema::Bool,
            Schema::Int,
            Schema::Float,
            Schema::Str,
            Schema::Bytes,
            Schema::Literal(ConstIx::new(0)),
            Schema::Instance(ClassIx::new(0)),
            Schema::Ref(DefIx::new(0)),
            Schema::SelfRef(0),
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Str])),
            Schema::list(SeqShape::prefix_tail([Schema::Int], Schema::Str)),
            Schema::list(SeqShape::fixed([])),
            Schema::Set(Box::new(Schema::Int)),
            Schema::FrozenSet(Box::new(Schema::Int)),
            Schema::Complement(Box::new(Schema::Int)),
            Schema::Union(vec![Schema::Int, Schema::Str]),
            Schema::Intersection(vec![Schema::Int, Schema::Str]),
            Schema::record(vec![field("a")], Openness::Open),
            Schema::mapping(MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }),
            Schema::AttrRecord {
                fields: vec![field("a")],
            },
            Schema::Refine {
                base: Box::new(Schema::Int),
                constraints: vec![Constraint::MinLen(1)],
            },
        ]
    }

    /// `node_count` sizes the whole tree, and the binding's schema-size limit is
    /// the only consumer: an undercount admits a schema past the cap. Each arm
    /// carries a distinct total so a wrong operator cannot coincide with a right
    /// answer.
    #[test]
    fn node_count_totals_every_arm() {
        assert_eq!(Schema::Int.node_count(), 1);
        assert_eq!(Schema::Ref(DefIx::new(0)).node_count(), 1);
        assert_eq!(Schema::Complement(Box::new(Schema::Int)).node_count(), 2);
        assert_eq!(Schema::Set(Box::new(Schema::Str)).node_count(), 2);
        assert_eq!(Schema::FrozenSet(Box::new(Schema::Str)).node_count(), 2);
        // Union counts every member, not the deepest: three members, not one.
        assert_eq!(
            Schema::Union(vec![Schema::Int, Schema::Str, Schema::Bytes]).node_count(),
            4
        );
        assert_eq!(
            Schema::Intersection(vec![Schema::Int, Schema::Complement(Box::new(Schema::Str))])
                .node_count(),
            4
        );
        // A constraint is a node: base + one per constraint.
        assert_eq!(
            Schema::Refine {
                base: Box::new(Schema::Str),
                constraints: vec![Constraint::MinLen(1), Constraint::MaxLen(9)],
            }
            .node_count(),
            4
        );
        // The regex constructor is not itself a node; its element subtree is.
        assert_eq!(
            Schema::list(SeqShape::homogeneous(Schema::Complement(Box::new(
                Schema::Int
            ))))
            .node_count(),
            3
        );
        assert_eq!(
            Schema::list(SeqShape::fixed([Schema::Int, Schema::Str])).node_count(),
            3
        );
        assert_eq!(Schema::list(SeqShape::fixed([])).node_count(), 1);
        // A keyed map counts declared fields and both halves of every default.
        // Distinct field and default totals so neither the sum of the two nor
        // its factors coincide with a wrong operator.
        assert_eq!(
            Schema::KeyedMap {
                fields: vec![
                    Field {
                        name: "a".into(),
                        schema: Schema::Complement(Box::new(Schema::Int)),
                        required: true,
                    },
                    Field {
                        name: "b".into(),
                        schema: Schema::Str,
                        required: false,
                    },
                ],
                defaults: vec![
                    MapClause {
                        key: Schema::Str,
                        value: Schema::Bytes,
                    },
                    MapClause {
                        key: Schema::Int,
                        value: Schema::Complement(Box::new(Schema::Str))
                    },
                ],
            }
            .node_count(),
            // 1 map + (2 + 1) fields + (2 + 3) defaults
            9
        );
        assert_eq!(
            Schema::AttrRecord {
                fields: vec![
                    Field {
                        name: "a".into(),
                        schema: Schema::Int,
                        required: true
                    },
                    Field {
                        name: "b".into(),
                        schema: Schema::Str,
                        required: true
                    },
                ],
            }
            .node_count(),
            3
        );
    }

    /// `depth` bounds the native stack every recursive walk descends, so each
    /// structural arm must add exactly one level. The scalar and combinator arms
    /// are covered above; these are the container arms.
    #[test]
    fn depth_descends_every_container_arm() {
        assert_eq!(Schema::Set(Box::new(Schema::Int)).depth(), 2);
        assert_eq!(Schema::FrozenSet(Box::new(Schema::Int)).depth(), 2);
        // A sequence is one level, whichever shape it holds: the elements sit
        // directly in the shape, so reaching one is a single descent. While the
        // body was a regular expression the constructors above an element were
        // levels of their own, and the same three shapes measured 1, 3 and 4.
        assert_eq!(Schema::list(SeqShape::fixed([])).depth(), 1);
        assert_eq!(Schema::list(SeqShape::fixed([Schema::Int])).depth(), 2);
        assert_eq!(Schema::list(SeqShape::homogeneous(Schema::Int)).depth(), 2);
        assert_eq!(
            Schema::list(SeqShape::prefix_tail([Schema::Int], Schema::Str)).depth(),
            2
        );
        // The max over the elements, not the sum: a deeper element decides.
        assert_eq!(
            Schema::list(SeqShape::fixed([
                Schema::Int,
                Schema::Complement(Box::new(Schema::Str))
            ]))
            .depth(),
            3
        );
        assert_eq!(
            Schema::Refine {
                base: Box::new(Schema::Str),
                constraints: vec![]
            }
            .depth(),
            2
        );
        assert_eq!(
            Schema::KeyedMap {
                fields: vec![Field {
                    name: "a".into(),
                    schema: Schema::Complement(Box::new(Schema::Int)),
                    required: true,
                }],
                defaults: vec![],
            }
            .depth(),
            3
        );
        assert_eq!(
            Schema::AttrRecord {
                fields: vec![Field {
                    name: "a".into(),
                    schema: Schema::Int,
                    required: true
                }],
            }
            .depth(),
            2
        );
    }

    /// Combining two schemas concatenates their constant pools, so the right
    /// operand's pooled indices shift by the left pool's length. A constraint
    /// that fails to shift resolves to the WRONG pooled constant and silently
    /// compares against the wrong value; a length bound is not a pool index and
    /// must not move.
    #[test]
    fn shifted_remaps_pooled_constraint_operands_only() {
        let refined = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![
                Constraint::Ge(OperandIx::new(1)),
                Constraint::Gt(OperandIx::new(2)),
                Constraint::Le(OperandIx::new(3)),
                Constraint::Lt(OperandIx::new(4)),
                Constraint::MultipleOf(OperandIx::new(5)),
                Constraint::Predicate(PredIx::new(8)),
                Constraint::MinLen(6),
                Constraint::MaxLen(7),
            ],
        };
        let Schema::Refine { constraints, .. } =
            refined.shifted(PoolShift::new(10), DefShift::new(0))
        else {
            panic!("shifted a Refine into a non-Refine");
        };
        assert_eq!(
            constraints,
            vec![
                Constraint::Ge(OperandIx::new(11)),
                Constraint::Gt(OperandIx::new(12)),
                Constraint::Le(OperandIx::new(13)),
                Constraint::Lt(OperandIx::new(14)),
                Constraint::MultipleOf(OperandIx::new(15)),
                // A pooled predicate operand shifts like the numeric bounds.
                Constraint::Predicate(PredIx::new(18)),
                // Length bounds are counts, not pool indices: unmoved.
                Constraint::MinLen(6),
                Constraint::MaxLen(7),
            ]
        );
        // Pooled leaves shift by the pool; definition refs shift by defs.
        assert_eq!(
            Schema::Literal(ConstIx::new(1)).shifted(PoolShift::new(10), DefShift::new(3)),
            Schema::Literal(ConstIx::new(11))
        );
        assert_eq!(
            Schema::Instance(ClassIx::new(1)).shifted(PoolShift::new(10), DefShift::new(3)),
            Schema::Instance(ClassIx::new(11))
        );
        assert_eq!(
            Schema::Ref(DefIx::new(1)).shifted(PoolShift::new(10), DefShift::new(3)),
            Schema::Ref(DefIx::new(4))
        );
        // An attribute record has no index of its own; its fields carry theirs.
        let record = Schema::AttrRecord {
            fields: vec![Field {
                name: "a".into(),
                schema: Schema::Literal(ConstIx::new(2)),
                required: true,
            }],
        };
        let Schema::AttrRecord { fields } = record.shifted(PoolShift::new(10), DefShift::new(3))
        else {
            panic!("shifted an attribute record into another variant");
        };
        assert_eq!(fields[0].schema, Schema::Literal(ConstIx::new(12)));
    }

    /// A recursive body is well-formed only if every self-reference sits under a
    /// structural constructor; the algebraic combinators pass `guarded` through,
    /// so a reference under only a complement or a refinement is UNGUARDED. If
    /// this check misses one, an unguarded fixpoint is admitted and membership
    /// stops being decidable.
    #[test]
    fn occurs_unguarded_sees_through_the_algebraic_combinators() {
        // Bare: unguarded.
        assert!(Schema::Ref(DefIx::new(0)).occurs_unguarded(DefIx::new(0), Guarded::No));
        assert!(!Schema::Ref(DefIx::new(1)).occurs_unguarded(DefIx::new(0), Guarded::No));
        // Complement and Refine do NOT guard: the reference stays exposed.
        assert!(
            Schema::Complement(Box::new(Schema::Ref(DefIx::new(0))))
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        assert!(
            Schema::Refine {
                base: Box::new(Schema::Ref(DefIx::new(0))),
                constraints: vec![Constraint::MinLen(1)],
            }
            .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        assert!(
            Schema::Union(vec![Schema::Int, Schema::Ref(DefIx::new(0))])
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        assert!(
            Schema::Intersection(vec![Schema::Int, Schema::Ref(DefIx::new(0))])
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        // Nested combinators still pass it through.
        assert!(
            Schema::Complement(Box::new(Schema::Union(vec![
                Schema::Int,
                Schema::Ref(DefIx::new(0))
            ])))
            .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        // Structural constructors guard.
        assert!(
            !Schema::Set(Box::new(Schema::Ref(DefIx::new(0))))
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        assert!(
            !Schema::FrozenSet(Box::new(Schema::Ref(DefIx::new(0))))
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        assert!(
            !Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0))))
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        // A guarded reference under a combinator is still guarded.
        assert!(
            !Schema::Complement(Box::new(Schema::Set(Box::new(Schema::Ref(DefIx::new(0))))))
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
    }

    /// The element schema of a homogeneous (`[T, ...]`) sequence node.
    fn homogeneous_elem(schema: &Schema) -> &Schema {
        match schema {
            Schema::Seq { shape, .. } => shape.tail.as_deref().expect("homogeneous tail"),
            _ => panic!("not a sequence: {schema:?}"),
        }
    }

    #[test]
    fn violation_renders_root_message() {
        let v = Violation {
            code: "int_type",
            path: Vec::new(),
            expected: "int".to_owned(),
            value_summary: "'x'".to_owned(),
        };
        assert_eq!(v.location(), "");
        assert_eq!(v.to_string(), "expected int, got 'x' [int_type]");
    }

    #[test]
    fn violation_renders_nested_location() {
        let v = Violation {
            code: "string_type",
            path: vec![PathSegment::Key("name".to_owned()), PathSegment::Index(2)],
            expected: "str".to_owned(),
            value_summary: "5".to_owned(),
        };
        assert_eq!(v.location(), "name[2]");
        assert!(v.to_string().starts_with("at name[2]: expected str"));
    }

    #[test]
    fn labels_and_codes_for_every_variant() {
        let cases = [
            (Schema::Anything, "anything", "anything"),
            (Schema::Nothing, "nothing", "no_match"),
            (Schema::NoneType, "None", "none_type"),
            (Schema::Bool, "bool", "bool_type"),
            (Schema::Int, "int", "int_type"),
            (Schema::Float, "float", "float_type"),
            (Schema::Str, "str", "string_type"),
            (Schema::Bytes, "bytes", "bytes_type"),
            (Schema::Literal(ConstIx::new(0)), "literal", "literal_error"),
            (
                Schema::list(SeqShape::homogeneous(Schema::Int)),
                "list",
                "list_type",
            ),
            (
                Schema::tuple(SeqShape::fixed([Schema::Int])),
                "tuple",
                "tuple_type",
            ),
            (Schema::Set(Box::new(Schema::Int)), "set", "set_type"),
            (
                Schema::mapping(MapClause {
                    key: Schema::Str,
                    value: Schema::Int,
                }),
                "dict",
                "dict_type",
            ),
            (
                Schema::record(
                    vec![Field {
                        name: "k".to_owned(),
                        schema: Schema::Int,
                        required: true,
                    }],
                    Openness::Closed,
                ),
                "dict",
                "dict_type",
            ),
        ];
        for (schema, expected, code) in cases {
            assert_eq!(schema.expected(), expected, "expected for {schema:?}");
            assert_eq!(schema.error_code(), code, "code for {schema:?}");
        }
    }

    #[test]
    fn location_renders_keys_indices_and_their_mix() {
        let key_only = Violation {
            code: "x",
            path: vec![
                PathSegment::Key("a".to_owned()),
                PathSegment::Key("b".to_owned()),
            ],
            expected: String::new(),
            value_summary: String::new(),
        };
        assert_eq!(key_only.location(), "a.b");

        let index_only = Violation {
            code: "x",
            path: vec![PathSegment::Index(0), PathSegment::Index(3)],
            expected: String::new(),
            value_summary: String::new(),
        };
        assert_eq!(index_only.location(), "[0][3]");

        let mixed = Violation {
            code: "x",
            path: vec![
                PathSegment::Key("items".to_owned()),
                PathSegment::Index(2),
                PathSegment::Key("id".to_owned()),
            ],
            expected: "int".to_owned(),
            value_summary: "'x'".to_owned(),
        };
        assert_eq!(mixed.location(), "items[2].id");
        assert_eq!(
            mixed.to_string(),
            "at items[2].id: expected int, got 'x' [x]"
        );
    }

    #[test]
    fn mapping_and_record_share_the_dict_label() {
        let mapping = Schema::mapping(MapClause {
            key: Schema::Str,
            value: Schema::Int,
        });
        let record = Schema::record(Vec::new(), Openness::Closed);
        assert_eq!(mapping.expected(), record.expected());
        assert_eq!(mapping.error_code(), record.error_code());
    }

    /// Whether a record-shaped keyed map admits undeclared keys (has a default).
    fn record_is_open(schema: &Schema) -> bool {
        match schema {
            Schema::KeyedMap { defaults, .. } => !defaults.is_empty(),
            _ => panic!("not a keyed map: {schema:?}"),
        }
    }

    #[test]
    fn with_records_open_flips_every_record_in_the_tree() {
        let record = Schema::record(
            vec![Field {
                name: "k".to_owned(),
                schema: Schema::Int,
                required: true,
            }],
            Openness::Closed,
        );
        let schema = Schema::list(SeqShape::homogeneous(record));
        let opened = schema.with_records_open(Openness::Open);
        assert!(record_is_open(homogeneous_elem(&opened)));
        // strict flips it back.
        let closed = schema
            .with_records_open(Openness::Open)
            .with_records_open(Openness::Closed);
        assert!(!record_is_open(homogeneous_elem(&closed)));
    }

    /// A transform leaves the tree in the shape the constructors guarantee.
    ///
    /// Opening the records in `{a: int} | ~{a: int}` maps both sides to one
    /// schema beside its own complement -- a shape `union` folds away, and one a
    /// rule downstream is entitled never to meet. The descent refolds for that
    /// reason; reindexing does not, because a relabelling that changed the shape
    /// would not be one.
    #[test]
    fn with_records_open_refolds_a_pair_it_creates() {
        let closed = Schema::record(
            vec![Field {
                name: "a".to_owned(),
                schema: Schema::Int,
                required: true,
            }],
            Openness::Closed,
        );
        let pair = Schema::Union(vec![
            closed.clone(),
            Schema::Complement(Box::new(closed.with_records_open(Openness::Open))),
        ]);

        assert!(
            matches!(pair, Schema::Union(_)),
            "the two differ before the transform"
        );
        assert_eq!(
            pair.with_records_open(Openness::Open),
            Schema::Anything,
            "and are one schema and its complement after it"
        );
    }

    #[test]
    fn with_records_open_leaves_a_pure_mapping_closed() {
        // A KeyedMap with no declared fields is a mapping, not a record: opening
        // it must not graft a catch-all clause. The `!fields.is_empty()` guard is
        // what distinguishes the two, so an empty-field map keeps its own clauses
        // and gains none.
        let mapping = Schema::KeyedMap {
            fields: Vec::new(),
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        };
        let Schema::KeyedMap { fields, defaults } = mapping.with_records_open(Openness::Open)
        else {
            panic!("a mapping opened into a non-map");
        };
        assert!(fields.is_empty());
        assert_eq!(
            defaults,
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Int
            }]
        );
    }

    #[test]
    fn fresh_self_token_is_unique_per_call() {
        // Nested `recursive` definitions must not resolve each other's
        // self-references, which holds only if successive tokens differ. A
        // constant token would collide, so assert two calls disagree.
        assert_ne!(fresh_self_token(), fresh_self_token());
    }

    /// The three algebraic constructors own the identity of their own arity. A
    /// nullary union is the bottom and a nullary meet is the top, so no consumer
    /// of a member list receives a node it has to special-case -- which is what
    /// the render did not do.
    #[test]
    fn the_nullary_operations_are_their_identities() {
        assert_eq!(Schema::union([]), Schema::Nothing);
        assert_eq!(Schema::meet([]), Schema::Anything);
        // A non-empty list is kept as given: order and repeats are observable
        // through the render and structural equality, and the lattice laws are
        // `simplify`'s job.
        assert_eq!(
            Schema::union([Schema::Int, Schema::Int]),
            Schema::Union(vec![Schema::Int, Schema::Int])
        );
        assert_eq!(
            Schema::meet([Schema::Int]),
            Schema::Intersection(vec![Schema::Int])
        );
        assert_eq!(
            Schema::Int.complement(),
            Schema::Complement(Box::new(Schema::Int))
        );
    }

    /// The rebuilding walk and the reading traversal describe the same child set.
    /// `map_children` reconstructs a node and `children` reads it, so nothing but
    /// this holds one to the other when a variant gains a child schema -- and a
    /// measure that reads a different child set than the map writes is how two
    /// size measures came to disagree about a sequence.
    #[test]
    fn the_functor_and_the_traversal_describe_the_same_children() {
        for schema in every_variant() {
            let mapped = Cell::new(0usize);
            schema.map_children(&|child| {
                mapped.set(mapped.get() + 1);
                child.clone()
            });
            assert_eq!(
                mapped.get(),
                schema.children().count(),
                "map_children and children disagree on {schema:?}"
            );
        }
    }

    /// Both size measures read the same child set and differ only in what each
    /// node contributes on its own: a node is always one level, and always one
    /// node plus the constraints a refinement carries, which are payloads rather
    /// than child schemas. Neither measure can read a child the other misses.
    #[test]
    fn both_size_measures_read_the_shared_traversal() {
        for schema in every_variant() {
            let children: Vec<&Schema> = schema.children().collect();
            let deepest = children.iter().map(|c| c.depth()).max().unwrap_or(0);
            let total: usize = children.iter().map(|c| c.node_count()).sum();
            assert_eq!(schema.depth(), 1 + deepest);
            assert_eq!(schema.node_count(), schema.own_nodes() + total);
        }
    }

    /// The guard is a two-element lattice in which crossing a structural
    /// constructor absorbs: nothing below one is unguarded, however deeply it
    /// nests. The absorbing element was stated in a doc comment and threaded by
    /// hand through every structural arm.
    #[test]
    fn the_guard_join_absorbs_at_yes() {
        assert_eq!(Guarded::No.join(Guarded::No), Guarded::No);
        for other in [Guarded::No, Guarded::Yes] {
            assert_eq!(Guarded::Yes.join(other), Guarded::Yes);
            assert_eq!(other.join(Guarded::Yes), Guarded::Yes);
        }
    }

    /// Which constructors guard is a property of the node, stated once. A
    /// reference below a structural constructor is productive; below an algebraic
    /// one it is not, because the combinator does not consume an unfolding step.
    #[test]
    fn only_the_structural_constructors_guard_their_children() {
        for schema in every_variant() {
            let guards = matches!(
                schema,
                Schema::Seq { .. }
                    | Schema::Set(_)
                    | Schema::FrozenSet(_)
                    | Schema::KeyedMap { .. }
                    | Schema::AttrRecord { .. }
            );
            assert_eq!(
                schema.guards_children(),
                if guards { Guarded::Yes } else { Guarded::No },
                "{schema:?}"
            );
        }
    }

    #[test]
    fn schema_equality_is_structural() {
        assert_eq!(
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Schema::list(SeqShape::homogeneous(Schema::Int))
        );
        assert_ne!(
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Schema::list(SeqShape::homogeneous(Schema::Str))
        );
        assert_ne!(
            Schema::Literal(ConstIx::new(0)),
            Schema::Literal(ConstIx::new(1))
        );
    }

    #[test]
    fn resolve_self_replaces_only_the_matching_token() {
        let body = Schema::list(SeqShape::homogeneous(Schema::SelfRef(1)));
        let resolved = body.resolve_self(1, DefIx::new(3));
        assert_eq!(homogeneous_elem(&resolved), &Schema::Ref(DefIx::new(3)));
        assert!(matches!(
            Schema::SelfRef(2).resolve_self(1, DefIx::new(3)),
            Schema::SelfRef(2)
        ));
    }

    #[test]
    fn contractivity_requires_a_structural_guard() {
        assert!(
            !Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0))))
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        assert!(Schema::Ref(DefIx::new(0)).occurs_unguarded(DefIx::new(0), Guarded::No));
        assert!(
            Schema::Union(vec![Schema::Int, Schema::Ref(DefIx::new(0))])
                .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
        assert!(
            !Schema::list(SeqShape::homogeneous(Schema::Union(vec![
                Schema::Int,
                Schema::Ref(DefIx::new(0))
            ])))
            .occurs_unguarded(DefIx::new(0), Guarded::No)
        );
    }

    #[test]
    fn shifted_remaps_ref_by_the_definition_offset() {
        let shifted = Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0))))
            .shifted(PoolShift::new(7), DefShift::new(4));
        assert_eq!(homogeneous_elem(&shifted), &Schema::Ref(DefIx::new(4)));
        assert!(matches!(
            Schema::SelfRef(9).shifted(PoolShift::new(1), DefShift::new(1)),
            Schema::SelfRef(9)
        ));
    }

    #[test]
    fn reindexed_maps_pool_indices_through_the_table() {
        let schema = Schema::Union(vec![
            Schema::Literal(ConstIx::new(0)),
            Schema::Instance(ClassIx::new(1)),
            Schema::Refine {
                base: Box::new(Schema::Int),
                constraints: vec![Constraint::Ge(OperandIx::new(0)), Constraint::MinLen(2)],
            },
            Schema::Ref(DefIx::new(0)),
        ]);
        let reindexed = schema.reindexed(&[10, 11], DefShift::new(5));
        assert_eq!(
            reindexed,
            Schema::Union(vec![
                Schema::Literal(ConstIx::new(10)),  // 0 -> table[0] = 10
                Schema::Instance(ClassIx::new(11)), // 1 -> table[1] = 11
                Schema::Refine {
                    base: Box::new(Schema::Int),
                    // Ge index remaps through the table; MinLen is a length, untouched.
                    constraints: vec![Constraint::Ge(OperandIx::new(10)), Constraint::MinLen(2)],
                },
                Schema::Ref(DefIx::new(5)), // ref offset by def_offset = 5
            ])
        );
    }

    #[test]
    fn refine_delegates_label_and_code_to_its_base() {
        let refined = Schema::Refine {
            base: Box::new(Schema::Str),
            constraints: vec![Constraint::MinLen(1)],
        };
        assert_eq!(refined.expected(), "str");
        assert_eq!(refined.error_code(), "string_type");
    }

    /// The escaped-marker walk: which token is open is the caller's fact, and
    /// this brings the traversal that finds every marker to ask about.
    ///
    /// The binding refuses a validator carrying a marker no open definition
    /// claims, and that refusal is the whole reason a placeholder kept past its
    /// builder cannot build a schema whose back edge points at nothing. Tested
    /// here against the walk, so a marker the traversal fails to reach is a
    /// failure rather than a validator that silently admits no value.
    #[test]
    fn the_escaped_marker_walk_finds_a_marker_wherever_it_sits() {
        let open_none: &dyn Fn(u64) -> bool = &|_| false;
        let open_all: &dyn Fn(u64) -> bool = &|_| true;
        let open_seven: &dyn Fn(u64) -> bool = &|token| token == 7;

        // A bare marker is the shortest case, and the one an arm that only
        // recursed into children would miss: a marker has no children.
        assert!(Schema::SelfRef(7).has_escaped_self_ref(open_none));
        // Open is the other answer, and it is the common one: inside the builder
        // of the definition the marker stands for, every schema carries it.
        assert!(!Schema::SelfRef(7).has_escaped_self_ref(open_all));
        // Which token is which is the caller's fact, so two markers under one
        // predicate answer differently.
        assert!(!Schema::SelfRef(7).has_escaped_self_ref(open_seven));
        assert!(Schema::SelfRef(9).has_escaped_self_ref(open_seven));

        // A schema with no marker at all is the finished shape, and `recursive`
        // returns one: the marker it minted is a back edge by then.
        assert!(!Schema::Ref(DefIx::new(0)).has_escaped_self_ref(open_none));
        assert!(!Schema::Int.has_escaped_self_ref(open_none));

        // Buried, under each way a schema holds a child: an escaped marker
        // anywhere in the tree is one the validator must not carry.
        let buried = Schema::union([
            Schema::Int,
            Schema::list(SeqShape::prefix_tail(
                [Schema::Str],
                Schema::Set(Box::new(Schema::SelfRef(9))),
            )),
        ]);
        assert!(buried.has_escaped_self_ref(open_none));
        assert!(!buried.has_escaped_self_ref(open_all));
    }

    #[test]
    fn field_is_cloneable_and_carries_its_flag() {
        let field = Field {
            name: "n".to_owned(),
            schema: Schema::Int,
            required: false,
        };
        let copy = field.clone();
        assert_eq!(copy.name, "n");
        assert!(!copy.required);
        assert_eq!(copy.schema, Schema::Int);
    }

    #[test]
    fn a_shape_is_the_three_spellings_and_nothing_else() {
        // The three forms a caller can write, and what each one is. There is no
        // fourth: a shape is a prefix and an optional tail, so the question the
        // old `linear` answered -- is this regex one of the shapes the frontend
        // builds? -- has no way left to answer no.
        let homogeneous = SeqShape::homogeneous(Schema::Int);
        assert!(homogeneous.prefix.is_empty());
        assert_eq!(homogeneous.tail.as_deref(), Some(&Schema::Int));

        let fixed = SeqShape::fixed([Schema::Int, Schema::Str]);
        assert_eq!(fixed.prefix, vec![Schema::Int, Schema::Str]);
        assert!(fixed.tail.is_none());

        let prefixed = SeqShape::prefix_tail([Schema::Str], Schema::Int);
        assert_eq!(prefixed.prefix, vec![Schema::Str]);
        assert_eq!(prefixed.tail.as_deref(), Some(&Schema::Int));

        // The empty sequence, which `fixed` of nothing is and `Default` gives.
        assert_eq!(SeqShape::fixed([]), SeqShape::default());
        assert!(SeqShape::default().prefix.is_empty() && SeqShape::default().tail.is_none());
    }

    #[test]
    fn sequence_transforms_reach_the_prefix_and_the_tail() {
        // A shape with a Ref in its prefix and a SelfRef in its tail, so every
        // element a transform must reach is one it would miss by handling only
        // the other.
        let seq = Schema::list(SeqShape::prefix_tail(
            [Schema::Ref(DefIx::new(0))],
            Schema::SelfRef(7),
        ));

        // The Ref sits under the sequence guard, so it is not unguarded.
        assert!(!seq.occurs_unguarded(DefIx::new(0), Guarded::No));
        // simplify and with_records_open preserve the sequence shape.
        assert!(matches!(seq.simplify(), Schema::Seq { .. }));
        assert!(matches!(
            seq.with_records_open(Openness::Open),
            Schema::Seq { .. }
        ));

        // shifted moves the prefix's Ref by the definitions offset.
        let Schema::Seq { shape, .. } = seq.shifted(PoolShift::new(0), DefShift::new(5)) else {
            panic!("shape preserved")
        };
        assert_eq!(shape.prefix, vec![Schema::Ref(DefIx::new(5))]);

        // resolve_self rewrites the tail's SelfRef into a Ref.
        let Schema::Seq { shape, .. } = seq.resolve_self(7, DefIx::new(3)) else {
            panic!("shape preserved")
        };
        assert_eq!(shape.tail.as_deref(), Some(&Schema::Ref(DefIx::new(3))));
    }

    #[test]
    fn keyed_map_transforms_recurse_through_fields_and_defaults() {
        let schema = Schema::KeyedMap {
            fields: vec![Field {
                name: "f".to_owned(),
                schema: Schema::Ref(DefIx::new(0)),
                required: true,
            }],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::SelfRef(7),
            }],
        };
        // Both the field's Ref and the default's SelfRef sit under the map guard.
        assert!(!schema.occurs_unguarded(DefIx::new(0), Guarded::No));
        // shifted moves the field's Ref by the definitions offset.
        let Schema::KeyedMap { fields, .. } = schema.shifted(PoolShift::new(0), DefShift::new(5))
        else {
            panic!("shape preserved")
        };
        assert_eq!(fields[0].schema, Schema::Ref(DefIx::new(5)));
        // resolve_self rewrites the default clause's SelfRef into a Ref.
        let Schema::KeyedMap { defaults, .. } = schema.resolve_self(7, DefIx::new(3)) else {
            panic!("shape preserved")
        };
        assert_eq!(defaults[0].value, Schema::Ref(DefIx::new(3)));
    }

    fn not(s: Schema) -> Schema {
        Schema::Complement(Box::new(s))
    }

    #[test]
    fn simplify_decides_the_complement_laws() {
        // X ∩ ¬X = ⊥ and X ∪ ¬X = ⊤.
        assert_eq!(
            Schema::Intersection(vec![Schema::Int, not(Schema::Int)]).simplify(),
            Schema::Nothing
        );
        assert_eq!(
            Schema::Union(vec![Schema::Int, not(Schema::Int)]).simplify(),
            Schema::Anything
        );
        // The law is the complementary pair itself, not scalar-region coverage: an
        // opaque member has no region, so `X ∪ ¬X` here is decided only by finding
        // the pair, with the whole universe left unaccounted for by the bitset.
        let opaque = Schema::list(SeqShape::homogeneous(Schema::Int));
        assert_eq!(
            Schema::Union(vec![opaque.clone(), not(opaque)]).simplify(),
            Schema::Anything
        );
        // Disjoint basics and disjoint container kinds give an empty intersection.
        assert_eq!(
            Schema::Intersection(vec![Schema::Int, Schema::Str]).simplify(),
            Schema::Nothing
        );
        assert_eq!(
            Schema::Intersection(vec![
                Schema::list(SeqShape::homogeneous(Schema::Int)),
                Schema::Set(Box::new(Schema::Int)),
            ])
            .simplify(),
            Schema::Nothing
        );
        // bool ⊆ int, so their intersection is not empty.
        assert_ne!(
            Schema::Intersection(vec![Schema::Bool, Schema::Int]).simplify(),
            Schema::Nothing
        );
    }

    #[test]
    fn simplify_preserves_gradual_any_under_complement() {
        // The gradual `Any` must not be rewritten by the complement laws.
        assert_ne!(
            Schema::Intersection(vec![Schema::Dynamic, not(Schema::Dynamic)]).simplify(),
            Schema::Nothing
        );
        assert_ne!(
            Schema::Union(vec![Schema::Dynamic, not(Schema::Dynamic)]).simplify(),
            Schema::Anything
        );
    }

    #[test]
    fn disjoint_is_sound_for_the_decidable_fragment() {
        assert!(Schema::Int.disjoint(&Schema::Str));
        assert!(Schema::Int.disjoint(&Schema::Float));
        // Every concrete tag is disjoint from a distinct one.
        assert!(Schema::NoneType.disjoint(&Schema::Int));
        assert!(Schema::Bytes.disjoint(&Schema::Str));
        let list_int = Schema::list(SeqShape::homogeneous(Schema::Int));
        let tuple_empty = Schema::tuple(SeqShape::fixed([]));
        assert!(tuple_empty.disjoint(&list_int)); // tuple vs list
        assert!(
            Schema::FrozenSet(Box::new(Schema::Int)).disjoint(&Schema::Set(Box::new(Schema::Int)))
        );
        assert!(
            Schema::mapping(MapClause {
                key: Schema::Str,
                value: Schema::Int,
            })
            .disjoint(&Schema::Int)
        ); // dict vs int
        // Nothing is disjoint from everything.
        assert!(Schema::Nothing.disjoint(&Schema::Int));
        assert!(Schema::Int.disjoint(&Schema::Nothing));
        // Same tag is not disjoint: two list types share the empty list.
        assert!(!list_int.disjoint(&Schema::list(SeqShape::homogeneous(Schema::Str))));
        assert!(!Schema::Bool.disjoint(&Schema::Int)); // bool is a subtype of int
        assert!(!Schema::Int.disjoint(&Schema::Int));
        // Conservative where the core cannot decide soundly.
        assert!(!Schema::Literal(ConstIx::new(0)).disjoint(&Schema::Int));
        assert!(!Schema::Instance(ClassIx::new(0)).disjoint(&Schema::Int));
        assert!(!Schema::Dynamic.disjoint(&Schema::Int));
        // A refinement is disjoint exactly when its base is.
        let refined = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![Constraint::Ge(OperandIx::new(0))],
        };
        assert!(refined.disjoint(&Schema::Str));
        assert!(!refined.disjoint(&Schema::Int));
    }
}

#[cfg(test)]
mod laws {
    use super::*;
    use crate::decision::DECISION_BUDGET;
    use proptest::prelude::*;

    /// A small schema generator: atoms combined by union, intersection, and
    /// complement. Pool indices are arbitrary but consistent across a value.
    fn schema() -> impl Strategy<Value = Schema> {
        let atom = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Nothing),
            Just(Schema::Dynamic),
            Just(Schema::NoneType),
            Just(Schema::Bool),
            Just(Schema::Int),
            Just(Schema::Float),
            Just(Schema::Str),
            Just(Schema::Bytes),
            Just(Schema::Literal(ConstIx::new(0))),
            Just(Schema::Instance(ClassIx::new(1))),
        ];
        atom.prop_recursive(4, 24, 3, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    fn union(a: Schema, b: Schema) -> Schema {
        Schema::Union(vec![a, b])
    }
    fn intersection(a: Schema, b: Schema) -> Schema {
        Schema::Intersection(vec![a, b])
    }
    fn not(a: Schema) -> Schema {
        Schema::Complement(Box::new(a))
    }

    /// One representative value per distinguishable scalar region. The five
    /// container kinds and `OTHER` are indistinguishable to a scalar schema (no
    /// scalar atom touches them, and a complement includes them together), so a
    /// single `Other` sample stands for that whole class.
    #[derive(Clone, Copy)]
    enum Sample {
        None,
        Bool,
        Int,
        Float,
        Str,
        Bytes,
        Other,
    }

    const SAMPLES: [Sample; 7] = [
        Sample::None,
        Sample::Bool,
        Sample::Int,
        Sample::Float,
        Sample::Str,
        Sample::Bytes,
        Sample::Other,
    ];

    /// A reference membership predicate for the scalar fragment, independent of
    /// the region-set decision under test, used as its oracle.
    fn member(schema: &Schema, value: Sample) -> bool {
        match schema {
            Schema::Anything => true,
            Schema::Nothing => false,
            Schema::NoneType => matches!(value, Sample::None),
            Schema::Bool => matches!(value, Sample::Bool),
            Schema::Int => matches!(value, Sample::Bool | Sample::Int), // bool ⊆ int
            Schema::Float => matches!(value, Sample::Float),
            Schema::Str => matches!(value, Sample::Str),
            Schema::Bytes => matches!(value, Sample::Bytes),
            Schema::Union(members) => members.iter().any(|m| member(m, value)),
            Schema::Intersection(members) => members.iter().all(|m| member(m, value)),
            Schema::Complement(inner) => !member(inner, value),
            other => unreachable!("oracle is scalar-only, got {other:?}"),
        }
    }

    /// A generator over the scalar-decidable fragment: scalar atoms combined by
    /// union, intersection, and complement.
    fn scalar_schema() -> impl Strategy<Value = Schema> {
        let atom = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Nothing),
            Just(Schema::NoneType),
            Just(Schema::Bool),
            Just(Schema::Int),
            Just(Schema::Float),
            Just(Schema::Str),
            Just(Schema::Bytes),
        ];
        atom.prop_recursive(4, 24, 3, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    #[test]
    fn decides_scalar_emptiness_subtyping_and_equivalence() {
        // Multi-way emptiness the pairwise checks cannot reach.
        assert!(
            Schema::Intersection(vec![Schema::Int, not(Schema::Bool), not(Schema::Int)]).is_empty()
        );
        assert!(
            Schema::Intersection(vec![
                Schema::Union(vec![Schema::Int, Schema::Str]),
                not(Schema::Int),
                not(Schema::Str),
            ])
            .is_empty()
        );
        assert!(!Schema::Intersection(vec![Schema::Int, not(Schema::Bool)]).is_empty());
        // Subtyping, with bool ⊆ int.
        assert!(Schema::Bool.is_subtype_of(&Schema::Int));
        assert!(!Schema::Int.is_subtype_of(&Schema::Bool));
        assert!(!Schema::Float.is_subtype_of(&Schema::Int));
        // Equivalence between structurally different schemas: bool ∪ int = int.
        assert!(Schema::Union(vec![Schema::Bool, Schema::Int]).is_equivalent(&Schema::Int));
    }

    #[test]
    fn is_empty_and_subtype_are_sound_off_the_scalar_fragment() {
        // Non-scalar leaves are never decided empty.
        assert!(!Schema::Dynamic.is_empty());
        assert!(!Schema::Literal(ConstIx::new(0)).is_empty());
        assert!(!Schema::Instance(ClassIx::new(0)).is_empty());
        assert!(!Schema::Set(Box::new(Schema::Int)).is_empty());
        assert!(!Schema::list(SeqShape::homogeneous(Schema::Int)).is_empty());
        // A scalar mixed with a non-scalar leaf is undecidable here, so it is
        // never claimed empty (an instance could subclass the scalar's type).
        assert!(
            !Schema::Intersection(vec![Schema::Int, Schema::Instance(ClassIx::new(0))]).is_empty()
        );
        // The gradual `Any` is never collapsed.
        assert!(!Schema::Intersection(vec![Schema::Dynamic, not(Schema::Dynamic)]).is_empty());
        // Subtyping off the fragment is reflexive only.
        assert!(
            Schema::Instance(ClassIx::new(0)).is_subtype_of(&Schema::Instance(ClassIx::new(0)))
        );
        assert!(
            !Schema::Instance(ClassIx::new(0)).is_subtype_of(&Schema::Instance(ClassIx::new(1)))
        );
    }

    #[test]
    fn decides_structural_container_emptiness() {
        // A fixed sequence with an impossible element matches no sequence.
        let empty_pair = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Nothing]));
        assert!(empty_pair.is_empty());
        // A list or tuple that admits the empty sequence is never empty.
        assert!(!Schema::list(SeqShape::homogeneous(Schema::Nothing)).is_empty());
        assert!(!Schema::tuple(SeqShape::fixed([Schema::Int])).is_empty());
        // A set or frozenset is never empty: the empty collection is a member.
        assert!(!Schema::Set(Box::new(Schema::Nothing)).is_empty());
        assert!(!Schema::FrozenSet(Box::new(Schema::Nothing)).is_empty());
        // A keyed map is empty exactly when a required field is impossible.
        let field = |required| Field {
            name: "x".to_owned(),
            schema: Schema::Nothing,
            required,
        };
        assert!(
            Schema::KeyedMap {
                fields: vec![field(true)],
                defaults: Vec::new(),
            }
            .is_empty()
        );
        assert!(
            !Schema::KeyedMap {
                fields: vec![field(false)],
                defaults: Vec::new(),
            }
            .is_empty()
        );
        // A union is empty only when every member is.
        assert!(Schema::Union(vec![Schema::Nothing, empty_pair.clone()]).is_empty());
        assert!(!Schema::Union(vec![Schema::Int, empty_pair]).is_empty());
    }

    #[test]
    fn decides_structural_subtyping_between_containers() {
        let set = |s| Schema::Set(Box::new(s));
        let frozenset = |s| Schema::FrozenSet(Box::new(s));
        // Sets and frozensets reduce to element inclusion (bool ⊆ int).
        assert!(set(Schema::Bool).is_subtype_of(&set(Schema::Int)));
        assert!(!set(Schema::Int).is_subtype_of(&set(Schema::Bool)));
        assert!(frozenset(Schema::Bool).is_subtype_of(&frozenset(Schema::Int)));
        // Different container kinds are never subtypes.
        assert!(!set(Schema::Int).is_subtype_of(&frozenset(Schema::Int)));
        // Homogeneous sequences: list[bool] ⊆ list[int], not list[int] ⊆ list[str].
        let list = |r| Schema::list(r);
        let tuple = |r| Schema::tuple(r);
        assert!(
            list(SeqShape::homogeneous(Schema::Bool))
                .is_subtype_of(&list(SeqShape::homogeneous(Schema::Int)))
        );
        assert!(
            !list(SeqShape::homogeneous(Schema::Int))
                .is_subtype_of(&list(SeqShape::homogeneous(Schema::Str)))
        );
        // Fixed sequences compare pointwise; a tuple is not a list.
        assert!(
            tuple(SeqShape::fixed([Schema::Bool, Schema::Str]))
                .is_subtype_of(&tuple(SeqShape::fixed([Schema::Int, Schema::Str])))
        );
        assert!(
            !tuple(SeqShape::fixed([Schema::Int]))
                .is_subtype_of(&list(SeqShape::homogeneous(Schema::Int)))
        );
        // A fixed list is a subtype of a homogeneous list when each element is.
        assert!(
            list(SeqShape::fixed([Schema::Bool, Schema::Int]))
                .is_subtype_of(&list(SeqShape::homogeneous(Schema::Int)))
        );
        // Equivalence between structurally different container schemas.
        assert!(
            set(Schema::Union(vec![Schema::Bool, Schema::Int])).is_equivalent(&set(Schema::Int))
        );
    }

    #[test]
    fn decides_record_and_mapping_subtyping() {
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        let record = |fields| Schema::KeyedMap {
            fields,
            defaults: Vec::new(),
        };
        let mapping = |k, v| Schema::KeyedMap {
            fields: Vec::new(),
            defaults: vec![MapClause { key: k, value: v }],
        };

        // Width: a closed record with fewer keys is a subtype of one with more.
        let narrow = record(vec![field("x", Schema::Int, true)]);
        let wide = record(vec![
            field("x", Schema::Int, true),
            field("y", Schema::Str, false),
        ]);
        assert!(narrow.is_subtype_of(&wide));
        assert!(!wide.is_subtype_of(&narrow)); // wide admits key y; narrow (closed) forbids it
        // Depth: shared field schemas covary (bool ⊆ int).
        assert!(
            record(vec![field("x", Schema::Bool, true)]).is_subtype_of(&record(vec![field(
                "x",
                Schema::Int,
                true
            )]))
        );
        // Required: a field the supertype requires must be required in the subtype.
        let required = record(vec![field("x", Schema::Int, true)]);
        let optional = record(vec![field("x", Schema::Int, false)]);
        assert!(required.is_subtype_of(&optional));
        assert!(!optional.is_subtype_of(&required));
        // Mappings covary in key and value.
        assert!(
            mapping(Schema::Str, Schema::Bool).is_subtype_of(&mapping(Schema::Str, Schema::Int))
        );
        assert!(
            !mapping(Schema::Str, Schema::Int).is_subtype_of(&mapping(Schema::Str, Schema::Bool))
        );
        // A closed record is below a mapping whose catch-all covers each of its
        // keys: `{"x": int}` places an `int` at a `str` key and nothing else.
        assert!(narrow.is_subtype_of(&mapping(Schema::Str, Schema::Int)));
        // ...and the value type still has to hold.
        assert!(!narrow.is_subtype_of(&mapping(Schema::Str, Schema::Str)));
    }

    #[test]
    fn decides_sequence_subtyping_with_prefix_tail_and_alternation() {
        // A list `[head, tail*]`: a one-element fixed prefix then a repeated tail.
        let prefix_tail = |head, tail| Schema::list(SeqShape::prefix_tail([head], tail));
        // Prefix and tail covary (bool ⊆ int), in both positions.
        assert!(
            prefix_tail(Schema::Bool, Schema::Bool)
                .is_subtype_of(&prefix_tail(Schema::Int, Schema::Int))
        );
        assert!(
            !prefix_tail(Schema::Int, Schema::Int)
                .is_subtype_of(&prefix_tail(Schema::Int, Schema::Bool))
        );
        // A fixed-length list is a subtype of a prefix-and-tail one it fits.
        assert!(
            Schema::list(SeqShape::fixed([Schema::Bool, Schema::Int]))
                .is_subtype_of(&prefix_tail(Schema::Int, Schema::Int))
        );
        // ...and is NOT one it is too short for. `[int]` does not fit
        // `[int, int, int*]`: the supertype's fixed prefix is longer than the
        // subtype's whole length, so no alignment exists. The two halves of the
        // alignment test are a conjunction for exactly this case -- the element
        // comparisons that DO happen all succeed, so a disjunction there reports
        // a subtype relation that does not hold.
        assert!(
            !Schema::list(SeqShape::fixed([Schema::Int])).is_subtype_of(&Schema::list(
                SeqShape::prefix_tail([Schema::Int, Schema::Int], Schema::Int)
            ))
        );
        // Two fixed lists of the SAME length whose elements do not relate. Equal
        // lengths are necessary and not sufficient, so the length test and the
        // element test are a conjunction here too.
        assert!(
            !Schema::list(SeqShape::fixed([Schema::Int]))
                .is_subtype_of(&Schema::list(SeqShape::fixed([Schema::Str])))
        );
        assert!(
            Schema::list(SeqShape::fixed([Schema::Bool]))
                .is_subtype_of(&Schema::list(SeqShape::fixed([Schema::Int])))
        );
        // A union of sequences is a union of schemas, which the lattice rules
        // already distribute over: (bool* | int*) <= int*, and int* is in
        // neither branch of (bool* | str*).
        let alternation = |a, b| {
            Schema::union([
                Schema::list(SeqShape::homogeneous(a)),
                Schema::list(SeqShape::homogeneous(b)),
            ])
        };
        assert!(
            alternation(Schema::Bool, Schema::Int)
                .is_subtype_of(&Schema::list(SeqShape::homogeneous(Schema::Int)))
        );
        assert!(
            !Schema::list(SeqShape::homogeneous(Schema::Int))
                .is_subtype_of(&alternation(Schema::Bool, Schema::Str))
        );
    }

    #[test]
    fn decides_tuple_prefix_tail_distinctly_from_lists() {
        // The same prefix-plus-tail regex carried by the tuple container. The
        // decision procedure shares the regex with lists, so this pins that the
        // container is honoured throughout subtyping, emptiness, and equivalence.
        let tup = |head, tail| Schema::tuple(SeqShape::prefix_tail([head], tail));

        // Subtyping is covariant in both the prefix and the repeated tail.
        assert!(tup(Schema::Bool, Schema::Bool).is_subtype_of(&tup(Schema::Int, Schema::Int)));
        assert!(!tup(Schema::Int, Schema::Int).is_subtype_of(&tup(Schema::Int, Schema::Bool)));
        // A fixed-length tuple is a subtype of a prefix-and-tail one it fits.
        assert!(
            Schema::tuple(SeqShape::fixed([Schema::Bool, Schema::Int]))
                .is_subtype_of(&tup(Schema::Int, Schema::Int))
        );

        // The container is part of the type: a list is never a tuple, even with
        // an identical element regex.
        assert!(
            !Schema::list(SeqShape::prefix_tail([Schema::Int], Schema::Int))
                .is_subtype_of(&tup(Schema::Int, Schema::Int))
        );
        assert!(!tup(Schema::Int, Schema::Int).is_subtype_of(&Schema::list(
            SeqShape::prefix_tail([Schema::Int], Schema::Int)
        )));

        // Emptiness reasons about position: an uninhabited prefix empties the
        // whole tuple, but an uninhabited *tail* only forbids the repeats, so a
        // single-element tuple matching the prefix still inhabits it.
        assert!(tup(Schema::Nothing, Schema::Int).is_empty());
        assert!(!tup(Schema::Int, Schema::Nothing).is_empty());

        // Equivalence collapses a redundant union in the tail (bool ⊆ int).
        assert!(
            tup(Schema::Int, Schema::Union(vec![Schema::Bool, Schema::Int]))
                .is_equivalent(&tup(Schema::Int, Schema::Int))
        );
    }

    /// Refinement subtyping decides a supertype bound by *entailment* through the
    /// ordering oracle, not only a verbatim constraint match: a tighter lower,
    /// upper, or length bound is a subtype of a looser one. Soundness negatives
    /// confirm a looser bound is not a subtype of a tighter one, and a non-strict
    /// bound does not entail its strict form at the same value.
    #[test]
    fn refinement_subtyping_decides_bound_entailment() {
        use core::cmp::Ordering;
        struct ByIndex;
        impl LeafRelations for ByIndex {
            fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
                None
            }
            fn compare(&self, a: OperandIx, b: OperandIx) -> Option<Ordering> {
                Some(a.get().cmp(&b.get()))
            }
        }
        let refine = |constraints: Vec<Constraint>| Schema::Refine {
            base: Box::new(Schema::Int),
            constraints,
        };
        let sub = |a: Vec<Constraint>, b: Vec<Constraint>| {
            refine(a).is_subtype_of_under(&refine(b), &ByIndex, &[])
        };
        assert!(sub(
            vec![Constraint::Ge(OperandIx::new(5))],
            vec![Constraint::Ge(OperandIx::new(0))]
        ));
        assert!(sub(
            vec![Constraint::Gt(OperandIx::new(5))],
            vec![Constraint::Ge(OperandIx::new(0))]
        ));
        assert!(sub(
            vec![Constraint::Le(OperandIx::new(0))],
            vec![Constraint::Le(OperandIx::new(5))]
        ));
        assert!(sub(
            vec![Constraint::Lt(OperandIx::new(0))],
            vec![Constraint::Lt(OperandIx::new(5))]
        ));
        assert!(sub(
            vec![Constraint::MinLen(5)],
            vec![Constraint::MinLen(2)]
        ));
        assert!(sub(
            vec![Constraint::MaxLen(2)],
            vec![Constraint::MaxLen(5)]
        ));
        // Soundness negatives.
        assert!(!sub(
            vec![Constraint::Ge(OperandIx::new(0))],
            vec![Constraint::Ge(OperandIx::new(5))]
        ));
        assert!(!sub(
            vec![Constraint::Le(OperandIx::new(5))],
            vec![Constraint::Le(OperandIx::new(0))]
        ));
        assert!(!sub(
            vec![Constraint::Ge(OperandIx::new(5))],
            vec![Constraint::Gt(OperandIx::new(5))]
        ));
    }

    /// An integer-discrete refinement is empty when its bounds leave no integer
    /// between them, even though the endpoints themselves are ordered. The rule is
    /// gated on the integer base and on the value oracle: a dense base, or a core
    /// with no value oracle, keeps the interval conservatively non-empty.
    #[test]
    fn refinement_emptiness_decides_integer_adjacency() {
        use core::cmp::Ordering;
        // The pool index doubles as the integer bound value.
        struct ByValue;
        impl LeafRelations for ByValue {
            fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
                None
            }
            fn compare(&self, a: OperandIx, b: OperandIx) -> Option<Ordering> {
                Some(a.get().cmp(&b.get()))
            }
            fn no_int_between(
                &self,
                lo: OperandIx,
                lo_strict: bool,
                hi: OperandIx,
                hi_strict: bool,
            ) -> Option<bool> {
                let least = i64::try_from(lo.get()).unwrap() + i64::from(lo_strict);
                let greatest = i64::try_from(hi.get()).unwrap() - i64::from(hi_strict);
                Some(least > greatest)
            }
        }
        let refine = |base, constraints: Vec<Constraint>| Schema::Refine {
            base: Box::new(base),
            constraints,
        };
        // Gt(0) & Lt(1): the open interval (0, 1) holds no integer, so it is empty.
        assert!(
            refine(
                Schema::Int,
                vec![
                    Constraint::Gt(OperandIx::new(0)),
                    Constraint::Lt(OperandIx::new(1))
                ]
            )
            .is_empty_with(&ByValue, &[])
        );
        // Gt(0) & Lt(2): the integer 1 fits, so it is not empty.
        assert!(
            !refine(
                Schema::Int,
                vec![
                    Constraint::Gt(OperandIx::new(0)),
                    Constraint::Lt(OperandIx::new(2))
                ]
            )
            .is_empty_with(&ByValue, &[])
        );
        // A dense (float) base is not integer-discrete: the discreteness rule must
        // not fire, or it would unsoundly empty a populated interval.
        assert!(
            !refine(
                Schema::Float,
                vec![
                    Constraint::Gt(OperandIx::new(0)),
                    Constraint::Lt(OperandIx::new(1))
                ]
            )
            .is_empty_with(&ByValue, &[])
        );
        // With no value oracle the default `no_int_between` is `None`, so even an
        // integer base stays conservative.
        assert!(
            !refine(
                Schema::Int,
                vec![
                    Constraint::Gt(OperandIx::new(0)),
                    Constraint::Lt(OperandIx::new(1))
                ]
            )
            .is_empty()
        );
    }

    // Trapping on overflow is a property of the profile, not of the code, so
    // this exists only in the profiles that promise it. `cargo test --release`
    // builds with `overflow-checks = false` by the manifest's own decision, and
    // an unconditional `#[should_panic]` would read an honest release run as a
    // broken one. There is no stable `cfg(overflow_checks)`; the manifest sets
    // the checks and the assertions together in every profile, so this is the
    // condition it can be asked under.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn the_dev_and_test_profiles_trap_an_overflowing_add() {
        // The policy, driven rather than read off the manifest: `overflow-checks`
        // is on in dev and test, so a bare `+` that wraps is a defect those
        // profiles catch. Every intended saturation in this tree is spelled, so
        // nothing legitimate trips it.
        let _ = std::hint::black_box(usize::MAX) + std::hint::black_box(1);
    }

    #[test]
    fn an_intersection_is_empty_when_any_member_is() {
        // The member verdicts are folded with a disjunction, and the fold is the
        // ONLY thing that sees this case: an empty member whose region is opaque
        // (a sequence whose element language is empty) intersected with the top.
        // No region cancels, no pair is complementary or disjoint, and no bound
        // contradicts, so a fold that lost a member's verdict would report a
        // non-empty intersection.
        let empty_list = Schema::list(SeqShape::fixed([Schema::Nothing]));
        assert!(empty_list.is_empty());
        assert!(Schema::Intersection(vec![empty_list.clone(), Schema::Anything]).is_empty());
        // Order does not matter: the fold runs over every member.
        assert!(Schema::Intersection(vec![Schema::Anything, empty_list]).is_empty());
        // And an intersection of two inhabited members with an opaque region is
        // not reported empty, so the fold is not simply answering true.
        let list_of_int = Schema::list(SeqShape::homogeneous(Schema::Int));
        assert!(!Schema::Intersection(vec![list_of_int, Schema::Anything]).is_empty());
    }

    #[test]
    fn only_the_bottom_is_disjoint_from_itself() {
        // `disjoint` compares two schemas, and the pairwise scan over an
        // intersection's members deliberately skips the self-comparison. That
        // skip is only observable for a schema disjoint from ITSELF, and bottom
        // is the only one: every other kind carries a type tag equal to its own.
        assert!(Schema::Nothing.disjoint(&Schema::Nothing));
        for schema in [
            Schema::NoneType,
            Schema::Bool,
            Schema::Int,
            Schema::Float,
            Schema::Str,
            Schema::Bytes,
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Schema::tuple(SeqShape::fixed([Schema::Int])),
            Schema::Set(Box::new(Schema::Int)),
            Schema::FrozenSet(Box::new(Schema::Int)),
            Schema::mapping(MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }),
            Schema::Anything,
            Schema::Dynamic,
        ] {
            assert!(
                !schema.disjoint(&schema),
                "{schema:?} is disjoint from itself"
            );
        }
        // A refinement takes its base's disjointness, so it is not self-disjoint
        // either -- unless its base is bottom, which is the same one case.
        let refined = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![Constraint::MinLen(1)],
        };
        assert!(!refined.disjoint(&refined));
    }

    #[test]
    fn decision_arms_are_pinned_independently_of_the_python_suite() {
        // Each assertion fails under a specific mutation of a decision arm, so the
        // core's own unit tests catch a defect without relying on the Python layer.
        use core::cmp::Ordering;
        struct ByIndex;
        impl LeafRelations for ByIndex {
            fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
                None
            }
            fn compare(&self, a: OperandIx, b: OperandIx) -> Option<Ordering> {
                Some(a.get().cmp(&b.get()))
            }
        }
        let list = |element| Schema::list(SeqShape::homogeneous(element));
        // A mapping in one call, so an assertion still reads as one line.
        let map = |key, value| Schema::mapping(MapClause { key, value });
        // Short spellings of the pooled bounds, so an assertion still reads as
        // one line: the operand index names its space at the constructor.
        let ge = |n: usize| Constraint::Ge(OperandIx::new(n));
        let gt = |n: usize| Constraint::Gt(OperandIx::new(n));
        let le = |n: usize| Constraint::Le(OperandIx::new(n));
        let lt = |n: usize| Constraint::Lt(OperandIx::new(n));

        // Bottom-below and top-above on a non-scalar (region_set is None there, so
        // the dedicated arms decide it).
        assert!(Schema::Nothing.is_subtype_of(&list(Schema::Int)));
        assert!(list(Schema::Int).is_subtype_of(&Schema::Anything));
        // Below a complement, which is the emptiness reduction rather than a
        // structural arm: a list shares no value with an int, so it lies inside
        // the complement of int. Nothing structural can see this -- there is no
        // shape on the right to recurse into.
        let not = |s| Schema::Complement(Box::new(s));
        assert!(list(Schema::Int).is_subtype_of(&not(Schema::Int)));
        assert!(map(Schema::Str, Schema::Int).is_subtype_of(&not(Schema::Str)));
        // ...and it stays sound where the two do share values.
        assert!(!list(Schema::Int).is_subtype_of(&not(list(Schema::Bool))));
        assert!(!Schema::Bool.is_subtype_of(&not(Schema::Int)));
        // A meet is below a member of a join, and a conjunct decides a meet's
        // supertype.
        assert!(list(Schema::Bool).is_subtype_of(&Schema::Intersection(vec![
            list(Schema::Int),
            Schema::Anything
        ])));
        assert!(
            Schema::Intersection(vec![list(Schema::Bool), list(Schema::Int)])
                .is_subtype_of(&list(Schema::Int))
        );
        // Complement is contravariant, on a non-scalar so the region check does
        // not decide it before the complement arm.
        assert!(not(list(Schema::Int)).is_subtype_of(&not(list(Schema::Bool))));
        assert!(!not(list(Schema::Bool)).is_subtype_of(&not(list(Schema::Int))));
        // A schema is below the empty set exactly when it is empty, decided through
        // the oracle for a refinement with unsatisfiable bounds.
        assert!(
            Schema::Refine {
                base: Box::new(Schema::Int),
                constraints: vec![ge(10), le(0)],
            }
            .is_subtype_of_under(&Schema::Nothing, &ByIndex, &[])
        );

        // Refinement bounds: equal closed bounds are a singleton (not empty), and
        // a strict pair at the same value is empty; a length window that is exactly
        // satisfiable is not empty.
        let refine = |constraints| Schema::Refine {
            base: Box::new(Schema::Int),
            constraints,
        };
        assert!(!refine(vec![ge(5), le(5)]).is_empty_with(&ByIndex, &[]));
        assert!(refine(vec![gt(5), lt(5)]).is_empty_with(&ByIndex, &[]));
        assert!(!refine(vec![Constraint::MinLen(5), Constraint::MaxLen(5)]).is_empty());
        // An intersection's refinement bounds are joined: both sides are needed.
        assert!(
            Schema::Intersection(vec![refine(vec![ge(5)]), refine(vec![le(0)]),])
                .is_empty_with(&ByIndex, &[])
        );
        assert!(
            !Schema::Intersection(vec![refine(vec![ge(0)]), refine(vec![le(5)]),])
                .is_empty_with(&ByIndex, &[])
        );
    }

    /// The keyed-map arm of the same pinning, split out because the rule is a
    /// conjunction of four independent halves: field depth, required-coverage,
    /// the extra fields a catch-all must reach, and clause subsumption. Each
    /// assertion below fails under a mutation of exactly one of them.
    #[test]
    fn keyed_map_arms_are_pinned_independently_of_the_python_suite() {
        // Keyed maps: each branch's conjunction is needed -- a depth failure is not
        // rescued by the required-coverage holding.
        let map = |key, value| Schema::mapping(MapClause { key, value });
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        let closed = |fields| Schema::record(fields, Openness::Closed);
        // Closed record: a depth failure is not rescued by required-coverage.
        assert!(
            !closed(vec![field("x", Schema::Int, true)]).is_subtype_of(&closed(vec![field(
                "x",
                Schema::Str,
                true
            )]))
        );
        // Closed record: an optional field is not a subtype of the same field made
        // required (required-coverage must hold on top of width and depth).
        assert!(
            !closed(vec![field("x", Schema::Int, false)]).is_subtype_of(&closed(vec![field(
                "x",
                Schema::Int,
                true
            )]))
        );
        // Pure mapping: a clause is subsumed only when both key and value narrow;
        // a key mismatch is not rescued by the value matching.
        assert!(!map(Schema::Str, Schema::Int).is_subtype_of(&map(Schema::Bytes, Schema::Int)));
        // Mixed record-and-catch-all: required-coverage must hold there too.
        let mixed = |required| Schema::KeyedMap {
            fields: vec![field("x", Schema::Int, required)],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        };
        assert!(!mixed(false).is_subtype_of(&mixed(true)));
        // A pure mapping is not a subtype of a mixed map that requires a field it
        // lacks: the pure-mapping branch must need both sides field-free.
        assert!(
            !map(Schema::Str, Schema::Int).is_subtype_of(&Schema::KeyedMap {
                fields: vec![field("x", Schema::Int, true)],
                defaults: vec![MapClause {
                    key: Schema::Str,
                    value: Schema::Int
                }],
            })
        );
        // A mixed map is not a subtype of one with an extra field whose catch-all
        // would admit an incompatible value: the mixed rule needs matching field
        // names, so its guard needs both an equal count and a name match.
        assert!(
            !Schema::KeyedMap {
                fields: vec![field("x", Schema::Int, false)],
                defaults: vec![MapClause {
                    key: Schema::Str,
                    value: Schema::Int
                }],
            }
            .is_subtype_of(&Schema::KeyedMap {
                fields: vec![
                    field("x", Schema::Int, false),
                    field("z", Schema::Bool, false),
                ],
                defaults: vec![MapClause {
                    key: Schema::Str,
                    value: Schema::Int
                }],
            })
        );
    }

    /// A mixed record with a catch-all is a subtype of one that declares an extra
    /// *optional* field, when the catch-all's value type fits that field. The
    /// soundness negatives confirm a *required* extra field stays undecided (a
    /// catch-all never guarantees a key's presence) and an optional field the
    /// catch-all value does not fit is not a subtype.
    #[test]
    fn keyed_map_subtyping_decides_supertype_extra_field() {
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        let with_catch_all = |fields| Schema::KeyedMap {
            fields,
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        };
        let base = || with_catch_all(vec![field("x", Schema::Int, true)]);
        let plus_y = |schema, required| {
            with_catch_all(vec![
                field("x", Schema::Int, true),
                field("y", schema, required),
            ])
        };
        // Optional extra field whose type the catch-all value (int) fits.
        assert!(base().is_subtype_of(&plus_y(Schema::Int, false)));
        // Required extra field: decided FALSE, and correctly -- the subtype admits
        // a value with no such key, so a catch-all over the key space cannot stand
        // in for the field's presence. Not a gap; naming it one hides where the
        // real gap is.
        assert!(!base().is_subtype_of(&plus_y(Schema::Int, true)));
        // Optional extra field the catch-all value type does not fit.
        assert!(!base().is_subtype_of(&plus_y(Schema::Str, false)));
    }

    #[test]
    fn decides_refinement_subtyping_structurally() {
        let refine = |base, constraints: Vec<Constraint>| Schema::Refine {
            base: Box::new(base),
            constraints,
        };

        // A refinement is a subtype of its base, and of anything its base subtypes.
        assert!(
            refine(Schema::Bool, vec![Constraint::Ge(OperandIx::new(0))])
                .is_subtype_of(&Schema::Int)
        );
        // More constraints denote a smaller set: a superset of constraints (with
        // the supertype's constraints all present) is a subtype.
        assert!(
            refine(
                Schema::Int,
                vec![
                    Constraint::Ge(OperandIx::new(0)),
                    Constraint::Le(OperandIx::new(1))
                ]
            )
            .is_subtype_of(&refine(
                Schema::Int,
                vec![Constraint::Ge(OperandIx::new(0))]
            ))
        );
        // The looser refinement is not a subtype of the tighter one.
        assert!(
            !refine(Schema::Int, vec![Constraint::Ge(OperandIx::new(0))]).is_subtype_of(&refine(
                Schema::Int,
                vec![
                    Constraint::Ge(OperandIx::new(0)),
                    Constraint::Le(OperandIx::new(1))
                ]
            ))
        );
        // The base must still subtype: a refined int is not a str.
        assert!(
            !refine(Schema::Int, vec![Constraint::Ge(OperandIx::new(0))])
                .is_subtype_of(&Schema::Str)
        );
        // An empty base empties the refinement; an inhabited base does not (bound
        // contradictions need value comparison and stay conservative here).
        assert!(refine(Schema::Nothing, vec![Constraint::Ge(OperandIx::new(0))]).is_empty());
        assert!(
            !refine(
                Schema::Int,
                vec![
                    Constraint::Ge(OperandIx::new(0)),
                    Constraint::Le(OperandIx::new(0))
                ]
            )
            .is_empty()
        );
    }

    #[test]
    fn reindexed_remaps_pool_and_definition_indices() {
        // Composing a validator concatenates pools and definitions: `reindexed`
        // remaps each pooled index through the intern map and offsets each `Ref`.
        let schema = Schema::Union(vec![
            Schema::Literal(ConstIx::new(0)),
            Schema::Instance(ClassIx::new(1)),
            Schema::Ref(DefIx::new(0)),
            Schema::Set(Box::new(Schema::Literal(ConstIx::new(1)))),
        ]);
        // The second pool interned into the first: old 0 -> 5, old 1 -> 6.
        let lit_map = [5, 6];
        let remapped = schema.reindexed(&lit_map, DefShift::new(3));
        assert_eq!(
            remapped,
            Schema::Union(vec![
                Schema::Literal(ConstIx::new(5)),
                Schema::Instance(ClassIx::new(6)),
                Schema::Ref(DefIx::new(3)),
                Schema::Set(Box::new(Schema::Literal(ConstIx::new(6)))),
            ])
        );

        // `shifted` is the identity-map case: every index moves by a fixed offset.
        let shifted = schema.shifted(PoolShift::new(5), DefShift::new(3));
        assert_eq!(
            shifted,
            Schema::Union(vec![
                Schema::Literal(ConstIx::new(5)),
                Schema::Instance(ClassIx::new(6)),
                Schema::Ref(DefIx::new(3)),
                Schema::Set(Box::new(Schema::Literal(ConstIx::new(6)))),
            ])
        );
        // A constraint operand index is remapped too.
        let refined = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![Constraint::Ge(OperandIx::new(0))],
        };
        assert_eq!(
            refined.reindexed(&lit_map, DefShift::new(0)),
            Schema::Refine {
                base: Box::new(Schema::Int),
                constraints: vec![Constraint::Ge(OperandIx::new(5))],
            }
        );
    }

    #[test]
    fn simplify_canonicalizes_refinement_constraints() {
        let refine = |base, constraints: Vec<Constraint>| Schema::Refine {
            base: Box::new(base),
            constraints,
        };
        // A repeated constraint collapses (idempotence over the conjunction).
        assert_eq!(
            refine(
                Schema::Int,
                vec![
                    Constraint::Ge(OperandIx::new(0)),
                    Constraint::Ge(OperandIx::new(0))
                ]
            )
            .simplify(),
            refine(Schema::Int, vec![Constraint::Ge(OperandIx::new(0))])
        );
        // Constraint order does not matter: both spellings share one normal form.
        assert_eq!(
            refine(
                Schema::Int,
                vec![
                    Constraint::Le(OperandIx::new(1)),
                    Constraint::Ge(OperandIx::new(0))
                ]
            )
            .simplify(),
            refine(
                Schema::Int,
                vec![
                    Constraint::Ge(OperandIx::new(0)),
                    Constraint::Le(OperandIx::new(1))
                ]
            )
            .simplify()
        );
        // A refinement of a refinement flattens into one refinement over the base.
        assert_eq!(
            refine(
                refine(Schema::Int, vec![Constraint::Ge(OperandIx::new(0))]),
                vec![Constraint::Le(OperandIx::new(1))],
            )
            .simplify(),
            refine(
                Schema::Int,
                vec![
                    Constraint::Ge(OperandIx::new(0)),
                    Constraint::Le(OperandIx::new(1))
                ]
            )
        );
        // The base is simplified before the refinement is rebuilt.
        assert_eq!(
            refine(
                Schema::Union(vec![Schema::Int, Schema::Int]),
                vec![Constraint::Ge(OperandIx::new(0))],
            )
            .simplify(),
            refine(Schema::Int, vec![Constraint::Ge(OperandIx::new(0))])
        );
        // Canonicalization is idempotent.
        let once = refine(
            Schema::Int,
            vec![
                Constraint::Le(OperandIx::new(1)),
                Constraint::Ge(OperandIx::new(0)),
                Constraint::Ge(OperandIx::new(0)),
            ],
        )
        .simplify();
        assert_eq!(once.clone(), once.simplify());
    }

    #[test]
    fn a_transform_over_a_sequence_keeps_its_shape() {
        // The structure-preserving transforms map over elements without moving
        // one between the prefix and the tail. That is the whole invariant the
        // decision procedure needs: it reads a sequence's arity off the prefix
        // length and its unbounded part off the tail, and a transform that
        // shuffled them would change which lengths the schema admits.
        let shape = SeqShape::prefix_tail([Schema::Str], Schema::Int);
        let mapped = shape.map_elems(&|s| s.clone());
        assert_eq!(mapped, shape);

        let complemented = shape.map_elems(&|s| Schema::Complement(Box::new(s.clone())));
        assert_eq!(
            complemented.prefix,
            vec![Schema::Complement(Box::new(Schema::Str))]
        );
        assert_eq!(
            complemented.tail.as_deref(),
            Some(&Schema::Complement(Box::new(Schema::Int)))
        );
    }

    #[test]
    fn decides_multi_clause_mapping_subtyping() {
        let map = |clauses: Vec<MapClause>| Schema::keyed_map(Vec::new(), clauses);
        // A mapping is a subtype of one with more clauses that subsume its own.
        assert!(
            map(vec![MapClause {
                key: Schema::Str,
                value: Schema::Int
            }])
            .is_subtype_of(&map(vec![
                MapClause {
                    key: Schema::Str,
                    value: Schema::Int
                },
                MapClause {
                    key: Schema::Int,
                    value: Schema::Bool
                },
            ]))
        );
        // The reverse fails: the extra int-keyed clause is not covered.
        assert!(
            !map(vec![
                MapClause {
                    key: Schema::Str,
                    value: Schema::Int
                },
                MapClause {
                    key: Schema::Int,
                    value: Schema::Bool
                }
            ])
            .is_subtype_of(&map(vec![MapClause {
                key: Schema::Str,
                value: Schema::Int
            }]))
        );
        // A clause is subsumed only when both key and value narrow.
        assert!(
            map(vec![MapClause {
                key: Schema::Str,
                value: Schema::Bool
            }])
            .is_subtype_of(&map(vec![MapClause {
                key: Schema::Str,
                value: Schema::Int
            }]))
        );
        assert!(
            !map(vec![MapClause {
                key: Schema::Str,
                value: Schema::Int
            }])
            .is_subtype_of(&map(vec![MapClause {
                key: Schema::Str,
                value: Schema::Bool
            }]))
        );
    }

    /// The same rule where a map carries fields as well as clauses: the field
    /// half and the clause half must both hold, and each is checked against the
    /// other side's catch-all where the two field lists differ.
    #[test]
    fn decides_a_record_beside_its_catch_all() {
        // A closed record is a subtype of an open one that declares its fields.
        let closed = |fields| Schema::record(fields, Openness::Closed);
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        assert!(
            closed(vec![field("x", Schema::Int, true)]).is_subtype_of(&Schema::record(
                vec![field("x", Schema::Int, true)],
                Openness::Open
            ))
        );

        // A record mixed with a catch-all narrows field-wise and clause-wise when
        // the field names match; a widening field or value, or differing field
        // names, are not subtypes.
        let mixed = |value_field, value_default| Schema::KeyedMap {
            fields: vec![field("a", value_field, true)],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: value_default,
            }],
        };
        assert!(mixed(Schema::Bool, Schema::Bool).is_subtype_of(&mixed(Schema::Int, Schema::Int)));
        assert!(!mixed(Schema::Int, Schema::Int).is_subtype_of(&mixed(Schema::Int, Schema::Bool)));
        assert!(
            !mixed(Schema::Int, Schema::Bool).is_subtype_of(&mixed(Schema::Bool, Schema::Bool))
        );
        let mixed_b = Schema::KeyedMap {
            fields: vec![field("b", Schema::Int, true)],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        };
        assert!(!mixed(Schema::Int, Schema::Int).is_subtype_of(&mixed_b));

        // A mixed map with an extra field is a subtype when a supertype catch-all
        // over all string keys covers that field's value.
        let with_extra = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true), field("b", Schema::Str, true)],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Bytes,
            }],
        };
        let covering = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true)],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Anything,
            }],
        };
        assert!(with_extra.is_subtype_of(&covering));
        // The extra field is not covered when the catch-all value is too narrow,
        // even though the catch-all clauses subsume (so only the extra-field
        // coverage decides it -- the "extra" set must be the fields not shared).
        let extra_uncovered = Schema::KeyedMap {
            fields: vec![
                field("a", Schema::Int, true),
                field("b", Schema::Bytes, true),
            ],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        };
        let str_catch_all = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true)],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        };
        assert!(!extra_uncovered.is_subtype_of(&str_catch_all));
        // The catch-all key must admit the field name: an int-keyed catch-all does
        // not cover a string field name even when its value would.
        let extra_str = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true), field("b", Schema::Str, true)],
            defaults: vec![MapClause {
                key: Schema::Int,
                value: Schema::Int,
            }],
        };
        let int_catch_all = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true)],
            defaults: vec![MapClause {
                key: Schema::Int,
                value: Schema::Anything,
            }],
        };
        assert!(!extra_str.is_subtype_of(&int_catch_all));
        // The reverse direction -- the supertype declaring a *required* field the
        // subtype lacks -- is decided FALSE, and correctly: the subtype admits a
        // value without that key.
        assert!(!covering.is_subtype_of(&with_extra));
    }

    #[test]
    fn decides_refinement_bound_emptiness_with_an_ordering_oracle() {
        use core::cmp::Ordering;
        // A mock oracle that treats each pool index as its own value, so
        // comparing indices orders the bounds those indices stand for.
        struct ByIndex;
        impl LeafRelations for ByIndex {
            fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
                None
            }
            fn compare(&self, a: OperandIx, b: OperandIx) -> Option<Ordering> {
                Some(a.get().cmp(&b.get()))
            }
        }
        let refine = |constraints| Schema::Refine {
            base: Box::new(Schema::Int),
            constraints,
        };
        // A lower bound above the upper bound is empty.
        assert!(
            refine(vec![
                Constraint::Ge(OperandIx::new(10)),
                Constraint::Le(OperandIx::new(0))
            ])
            .is_empty_with(&ByIndex, &[])
        );
        // Equal bounds with one strict end are empty; both closed is a singleton.
        assert!(
            refine(vec![
                Constraint::Ge(OperandIx::new(5)),
                Constraint::Lt(OperandIx::new(5))
            ])
            .is_empty_with(&ByIndex, &[])
        );
        assert!(
            !refine(vec![
                Constraint::Ge(OperandIx::new(5)),
                Constraint::Le(OperandIx::new(5))
            ])
            .is_empty_with(&ByIndex, &[])
        );
        // A satisfiable range is not empty.
        assert!(
            !refine(vec![
                Constraint::Ge(OperandIx::new(0)),
                Constraint::Le(OperandIx::new(10))
            ])
            .is_empty_with(&ByIndex, &[])
        );
        // A length contradiction needs no value comparison.
        assert!(refine(vec![Constraint::MinLen(5), Constraint::MaxLen(3)]).is_empty());
        // Refinements with contradictory bounds across an intersection are empty.
        let intersection = Schema::Intersection(vec![
            refine(vec![Constraint::Ge(OperandIx::new(5))]),
            refine(vec![Constraint::Lt(OperandIx::new(5))]),
        ]);
        assert!(intersection.is_empty_with(&ByIndex, &[]));
        // Without an ordering oracle the numeric bounds stay conservative.
        assert!(
            !refine(vec![
                Constraint::Ge(OperandIx::new(10)),
                Constraint::Le(OperandIx::new(0))
            ])
            .is_empty()
        );
    }

    #[test]
    fn detects_uninhabited_recursive_schemas() {
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        // t = {value: int, next: t} — a mandatory self-reference, no base case:
        // no finite value satisfies it.
        let uninhabited = [Schema::KeyedMap {
            fields: vec![
                field("value", Schema::Int, true),
                field("next", Schema::Ref(DefIx::new(0)), true),
            ],
            defaults: Vec::new(),
        }];
        assert!(Schema::Ref(DefIx::new(0)).is_empty_under(&uninhabited));
        // t = None | {next: t} — a base case makes it inhabited.
        let inhabited = [Schema::Union(vec![
            Schema::NoneType,
            Schema::KeyedMap {
                fields: vec![field("next", Schema::Ref(DefIx::new(0)), true)],
                defaults: Vec::new(),
            },
        ])];
        assert!(!Schema::Ref(DefIx::new(0)).is_empty_under(&inhabited));
        // t = {next?: t} — an optional self-reference is inhabited by the empty map.
        let optional = [Schema::KeyedMap {
            fields: vec![field("next", Schema::Ref(DefIx::new(0)), false)],
            defaults: Vec::new(),
        }];
        assert!(!Schema::Ref(DefIx::new(0)).is_empty_under(&optional));
        // t = [t] — a list of itself is inhabited by the empty list.
        let list_of_self = [Schema::list(SeqShape::homogeneous(Schema::Ref(
            DefIx::new(0),
        )))];
        assert!(!Schema::Ref(DefIx::new(0)).is_empty_under(&list_of_self));
        // An unresolved reference stays conservative.
        assert!(!Schema::Ref(DefIx::new(9)).is_empty_under(&uninhabited));
        // Without the definitions, recursion is not resolved (no-arg is_empty).
        assert!(!Schema::Ref(DefIx::new(0)).is_empty());
    }

    #[test]
    fn decides_complement_subtyping_contravariantly() {
        let not = |s| Schema::Complement(Box::new(s));
        // ¬A ⊆ ¬B iff B ⊆ A: ¬int ⊆ ¬bool because bool ⊆ int.
        assert!(not(Schema::Int).is_subtype_of(&not(Schema::Bool)));
        assert!(!not(Schema::Bool).is_subtype_of(&not(Schema::Int)));
        // Reflexivity holds for a complement (regression: it failed before this
        // rule existed).
        assert!(not(Schema::Int).is_subtype_of(&not(Schema::Int)));
        assert!(
            not(Schema::Literal(ConstIx::new(0)))
                .is_subtype_of(&not(Schema::Literal(ConstIx::new(0))))
        );
    }

    #[test]
    fn decides_recursive_subtyping_coinductively() {
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        let list_of = |value, next| {
            Schema::Union(vec![
                Schema::NoneType,
                Schema::KeyedMap {
                    fields: vec![
                        field("value", value, true),
                        field("next", Schema::Ref(next), true),
                    ],
                    defaults: Vec::new(),
                },
            ])
        };
        // Two structurally identical recursive linked-list types are equivalent.
        let identical = [
            list_of(Schema::Int, DefIx::new(0)),
            list_of(Schema::Int, DefIx::new(1)),
        ];
        assert!(Schema::Ref(DefIx::new(0)).is_equivalent_under(
            &Schema::Ref(DefIx::new(1)),
            &NoLeafRelations,
            &identical
        ));
        // Depth covariance through the recursion: a bool-valued list is a subtype
        // of an int-valued one (bool ⊆ int), but not the reverse.
        let covary = [
            list_of(Schema::Bool, DefIx::new(0)),
            list_of(Schema::Int, DefIx::new(1)),
        ];
        assert!(Schema::Ref(DefIx::new(0)).is_subtype_of_under(
            &Schema::Ref(DefIx::new(1)),
            &NoLeafRelations,
            &covary
        ));
        assert!(!Schema::Ref(DefIx::new(1)).is_subtype_of_under(
            &Schema::Ref(DefIx::new(0)),
            &NoLeafRelations,
            &covary
        ));
    }

    /// A sample value for the subtyping oracle: a scalar, or a set whose element
    /// kinds are listed. Sets suffice to exercise the container rule without a
    /// regex matcher; sequence rules are covered by the unit test above.
    #[derive(Clone)]
    enum Val {
        Scalar(Sample),
        SetOf(Vec<Sample>),
    }

    fn samples_v() -> Vec<Val> {
        let mut values: Vec<Val> = SAMPLES.iter().map(|&s| Val::Scalar(s)).collect();
        values.push(Val::SetOf(vec![]));
        values.push(Val::SetOf(vec![Sample::Bool]));
        values.push(Val::SetOf(vec![Sample::Int]));
        values.push(Val::SetOf(vec![Sample::Str]));
        values.push(Val::SetOf(vec![Sample::Int, Sample::Str]));
        values
    }

    /// Reference membership for the scalar-and-set fragment, the oracle the
    /// structural subtyping decision is checked against.
    fn member_v(schema: &Schema, value: &Val) -> bool {
        match schema {
            Schema::Anything => true,
            Schema::Nothing => false,
            Schema::Set(element) => match value {
                Val::SetOf(elements) => elements.iter().all(|&e| member(element, e)),
                Val::Scalar(_) => false,
            },
            Schema::Union(members) => members.iter().any(|m| member_v(m, value)),
            Schema::Intersection(members) => members.iter().all(|m| member_v(m, value)),
            Schema::Complement(inner) => !member_v(inner, value),
            scalar => match value {
                Val::Scalar(sample) => member(scalar, *sample),
                Val::SetOf(_) => false,
            },
        }
    }

    /// A generator over scalars, sets of scalar schemas, and their Boolean
    /// combinations — the fragment the `member_v` oracle covers.
    fn scalar_or_set_schema() -> impl Strategy<Value = Schema> {
        let leaf = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Nothing),
            Just(Schema::NoneType),
            Just(Schema::Bool),
            Just(Schema::Int),
            Just(Schema::Float),
            Just(Schema::Str),
            Just(Schema::Bytes),
            scalar_schema().prop_map(|s| Schema::Set(Box::new(s))),
        ];
        leaf.prop_recursive(3, 16, 3, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    fn constraint() -> impl Strategy<Value = Constraint> {
        prop_oneof![
            (0usize..3).prop_map(|i| Constraint::Ge(OperandIx::new(i))),
            (0usize..3).prop_map(|i| Constraint::Le(OperandIx::new(i))),
            (0usize..8).prop_map(Constraint::MinLen),
            (0usize..8).prop_map(Constraint::MaxLen),
            Just(Constraint::Regex("a+".into())),
        ]
    }

    /// A generator over the whole structural fragment — sequences, sets, records,
    /// and refinements as well as scalars and Boolean combinations. The decision
    /// procedures stay conservative here, so this drives the *sound* invariants
    /// (termination, idempotent normalization, the order laws) rather than the
    /// value oracle, mirroring on the stable gate what the coverage-guided fuzz
    /// targets explore.
    fn structural_schema() -> impl Strategy<Value = Schema> {
        let leaf = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Dynamic),
            Just(Schema::Nothing),
            Just(Schema::NoneType),
            Just(Schema::Bool),
            Just(Schema::Int),
            Just(Schema::Float),
            Just(Schema::Str),
            Just(Schema::Bytes),
            (0usize..3).prop_map(|i| Schema::Literal(ConstIx::new(i))),
            (0usize..3).prop_map(|i| Schema::Instance(ClassIx::new(i))),
        ];
        leaf.prop_recursive(4, 32, 3, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Intersection),
                inner.clone().prop_map(|s| Schema::Complement(Box::new(s))),
                inner.clone().prop_map(|s| Schema::Set(Box::new(s))),
                inner.clone().prop_map(|s| Schema::FrozenSet(Box::new(s))),
                (inner.clone(), proptest::collection::vec(constraint(), 0..3)).prop_map(
                    |(base, constraints)| Schema::Refine {
                        base: Box::new(base),
                        constraints,
                    }
                ),
                inner.clone().prop_map(|s| Schema::Seq {
                    container: SeqKind::List,
                    shape: SeqShape::homogeneous(s),
                }),
                (inner.clone(), inner).prop_map(|(field, default)| Schema::KeyedMap {
                    fields: vec![Field {
                        name: "a".into(),
                        schema: field,
                        required: true,
                    }],
                    defaults: vec![MapClause {
                        key: Schema::Str,
                        value: default
                    }],
                }),
            ]
        })
    }

    /// A schema with a `Ref(0)` reachable somewhere inside it, for the
    /// guardedness property below: the reference is what the check looks for, so
    /// a generator that never produces one proves nothing.
    fn schema_holding_a_ref() -> impl Strategy<Value = Schema> {
        let leaf = prop_oneof![
            Just(Schema::Ref(DefIx::new(0))),
            Just(Schema::Int),
            Just(Schema::Str),
            Just(Schema::Anything),
        ];
        leaf.prop_recursive(4, 24, 3, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 1..3).prop_map(Schema::Union),
                prop::collection::vec(inner.clone(), 1..3).prop_map(Schema::Intersection),
                inner.clone().prop_map(|s| Schema::Complement(Box::new(s))),
                inner.clone().prop_map(|s| Schema::Refine {
                    base: Box::new(s),
                    constraints: vec![Constraint::MinLen(1)],
                }),
                inner.clone().prop_map(|s| Schema::Set(Box::new(s))),
                inner
                    .clone()
                    .prop_map(|s| Schema::list(SeqShape::homogeneous(s))),
                inner.prop_map(|s| Schema::record(
                    vec![Field {
                        name: "f".to_owned(),
                        schema: s,
                        required: true,
                    }],
                    Openness::Closed,
                )),
            ]
        })
    }

    proptest! {
        /// `Guarded::Yes` absorbs: once a structural constructor has been crossed,
        /// no reference below it is ever reported unguarded, however the algebraic
        /// combinators nest underneath.
        ///
        /// This is the argument that makes deleting one of `occurs_unguarded`'s
        /// structural arms an *equivalent* mutant rather than an untested one:
        /// every such arm answers false for every input, so the default answers
        /// the same. Pinned as a property rather than asserted in a comment, so a
        /// future arm that breaks the absorption fails here.
        #[test]
        fn structural_constructors_absorb_the_guard(s in schema_holding_a_ref()) {
            prop_assert!(!s.occurs_unguarded(DefIx::new(0), Guarded::Yes));
        }

        /// The same schema read from the top is unguarded exactly when some
        /// occurrence of the reference is reachable through algebraic combinators
        /// alone -- the observable half of the check, and the one a recursive
        /// definition's soundness rests on.
        #[test]
        fn a_reference_under_only_combinators_is_unguarded(s in schema_holding_a_ref()) {
            fn reachable_through_combinators(s: &Schema) -> bool {
                match s {
                    Schema::Ref(id) => *id == DefIx::new(0),
                    Schema::Union(es) | Schema::Intersection(es) => {
                        es.iter().any(reachable_through_combinators)
                    }
                    Schema::Complement(e) => reachable_through_combinators(e),
                    Schema::Refine { base, .. } => reachable_through_combinators(base),
                    _ => false,
                }
            }
            prop_assert_eq!(
                s.occurs_unguarded(DefIx::new(0), Guarded::No),
                reachable_through_combinators(&s)
            );
        }

        #[test]
        fn scalar_decision_matches_the_value_oracle(a in scalar_schema(), b in scalar_schema()) {
            let a_empty = SAMPLES.iter().all(|&v| !member(&a, v));
            prop_assert_eq!(a.is_empty(), a_empty);

            let a_sub_b = SAMPLES.iter().all(|&v| !member(&a, v) || member(&b, v));
            let b_sub_a = SAMPLES.iter().all(|&v| !member(&b, v) || member(&a, v));
            prop_assert_eq!(a.is_subtype_of(&b), a_sub_b);
            prop_assert_eq!(a.is_equivalent(&b), a_sub_b && b_sub_a);
        }

        /// The lattice bounds, stated over the PROPERTY rather than over the
        /// atoms. `Nothing ≤ b` and `a ≤ Anything` are the cases everything
        /// else already asserted, and asserting only those is a rule confirming
        /// itself: a schema that denotes the empty set without being spelled
        /// `Nothing` was never the subject. This generator produces such schemas
        /// constantly -- a cancelling intersection is two leaves away.
        ///
        /// This is a COMPLETENESS assertion, which the procedure does not make
        /// in general; it is assertable here because emptiness is decided on
        /// this fragment, so the premise is a proof rather than a guess.
        #[test]
        fn the_lattice_bounds_hold_over_emptiness_not_over_the_atoms(
            a in scalar_or_set_schema(),
            b in scalar_or_set_schema(),
        ) {
            if a.is_empty() {
                prop_assert!(a.is_subtype_of(&b), "empty {a:?} not below {b:?}");
            }
            if Schema::Complement(Box::new(b.clone())).is_empty() {
                prop_assert!(a.is_subtype_of(&b), "{a:?} not below universal {b:?}");
            }
        }

        #[test]
        fn structural_subtyping_is_sound(a in scalar_or_set_schema(), b in scalar_or_set_schema()) {
            prop_assert!(a.is_subtype_of(&a)); // reflexivity holds everywhere
            // Soundness: a claimed subtype never accepts a sample the supertype rejects.
            if a.is_subtype_of(&b) {
                for value in &samples_v() {
                    prop_assert!(!member_v(&a, value) || member_v(&b, value));
                }
            }
        }

        #[test]
        fn simplify_is_idempotent(a in schema()) {
            let once = a.simplify();
            prop_assert_eq!(once.clone(), once.simplify());
        }

        /// The laws, as statements about the *sets* two schemas denote.
        ///
        /// The properties below them compare two simplified schemas for
        /// structural equality, which is a property of the simplifier: it says
        /// the normal form does not depend on the order the members were
        /// written in. That is worth holding and it is not the law. The law is
        /// that the two sides admit the same values, and over the fragment the
        /// oracle decides exactly, that is what this checks.
        #[test]
        fn the_lattice_laws_hold_of_the_sets(
            a in scalar_or_set_schema(),
            b in scalar_or_set_schema(),
            c in scalar_or_set_schema(),
        ) {
            let same = |left: &Schema, right: &Schema| {
                samples_v().iter().all(|value| member_v(left, value) == member_v(right, value))
            };
            // Commutativity.
            prop_assert!(same(&union(a.clone(), b.clone()), &union(b.clone(), a.clone())));
            prop_assert!(same(
                &intersection(a.clone(), b.clone()),
                &intersection(b.clone(), a.clone()),
            ));
            // Associativity.
            prop_assert!(same(
                &union(a.clone(), union(b.clone(), c.clone())),
                &union(union(a.clone(), b.clone()), c.clone()),
            ));
            prop_assert!(same(
                &intersection(a.clone(), intersection(b.clone(), c.clone())),
                &intersection(intersection(a.clone(), b.clone()), c.clone()),
            ));
            // Idempotence.
            prop_assert!(same(&union(a.clone(), a.clone()), &a));
            prop_assert!(same(&intersection(a.clone(), a.clone()), &a));
            // The bounds.
            prop_assert!(same(&union(a.clone(), Schema::Nothing), &a));
            prop_assert!(same(&intersection(a.clone(), Schema::Anything), &a));
            prop_assert!(same(&union(a.clone(), Schema::Anything), &Schema::Anything));
            prop_assert!(same(&intersection(a.clone(), Schema::Nothing), &Schema::Nothing));
            // Distributivity, which the simplifier does not apply at all, so no
            // structural property below could state it.
            prop_assert!(same(
                &intersection(a.clone(), union(b.clone(), c.clone())),
                &union(intersection(a.clone(), b.clone()), intersection(a.clone(), c.clone())),
            ));
            prop_assert!(same(
                &union(a.clone(), intersection(b.clone(), c.clone())),
                &intersection(union(a.clone(), b.clone()), union(a.clone(), c.clone())),
            ));
            // Absorption, likewise.
            prop_assert!(same(&union(a.clone(), intersection(a.clone(), b.clone())), &a));
            prop_assert!(same(&intersection(a.clone(), union(a.clone(), b.clone())), &a));
        }

        /// The complement laws, as statements about the sets.
        #[test]
        fn the_complement_laws_hold_of_the_sets(
            a in scalar_or_set_schema(),
            b in scalar_or_set_schema(),
        ) {
            let same = |left: &Schema, right: &Schema| {
                samples_v().iter().all(|value| member_v(left, value) == member_v(right, value))
            };
            prop_assert!(same(&not(not(a.clone())), &a));
            prop_assert!(same(&union(a.clone(), not(a.clone())), &Schema::Anything));
            prop_assert!(same(&intersection(a.clone(), not(a.clone())), &Schema::Nothing));
            prop_assert!(same(
                &not(union(a.clone(), b.clone())),
                &intersection(not(a.clone()), not(b.clone())),
            ));
            prop_assert!(same(
                &not(intersection(a.clone(), b.clone())),
                &union(not(a.clone()), not(b.clone())),
            ));
        }

        /// Simplifying two sides of a law reaches one normal form. A property of
        /// the simplifier, not of the algebra: the law itself is above.
        #[test]
        fn union_and_intersection_commute(a in schema(), b in schema()) {
            prop_assert_eq!(union(a.clone(), b.clone()).simplify(), union(b.clone(), a.clone()).simplify());
            prop_assert_eq!(intersection(a.clone(), b.clone()).simplify(), intersection(b, a).simplify());
        }

        #[test]
        fn union_and_intersection_associate(a in schema(), b in schema(), c in schema()) {
            prop_assert_eq!(
                union(a.clone(), union(b.clone(), c.clone())).simplify(),
                union(union(a.clone(), b.clone()), c.clone()).simplify()
            );
            prop_assert_eq!(
                intersection(a.clone(), intersection(b.clone(), c.clone())).simplify(),
                intersection(intersection(a, b), c).simplify()
            );
        }

        #[test]
        fn idempotence(a in schema()) {
            prop_assert_eq!(union(a.clone(), a.clone()).simplify(), a.clone().simplify());
            prop_assert_eq!(intersection(a.clone(), a.clone()).simplify(), a.simplify());
        }

        #[test]
        fn identities(a in schema()) {
            prop_assert_eq!(union(a.clone(), Schema::Nothing).simplify(), a.clone().simplify());
            prop_assert_eq!(intersection(a.clone(), Schema::Anything).simplify(), a.clone().simplify());
            prop_assert_eq!(union(a.clone(), Schema::Anything).simplify(), Schema::Anything);
            prop_assert_eq!(intersection(a, Schema::Nothing).simplify(), Schema::Nothing);
        }

        #[test]
        fn double_negation(a in schema()) {
            prop_assert_eq!(not(not(a.clone())).simplify(), a.simplify());
        }

        #[test]
        fn de_morgan(a in schema(), b in schema()) {
            // Both forms: the complement of a join is the meet of the complements,
            // and the complement of a meet is the join of the complements.
            prop_assert_eq!(
                not(union(a.clone(), b.clone())).simplify(),
                intersection(not(a.clone()), not(b.clone())).simplify()
            );
            prop_assert_eq!(
                not(intersection(a.clone(), b.clone())).simplify(),
                union(not(a), not(b)).simplify()
            );
        }

        /// The strongest law check: simplification preserves membership, not just
        /// structural shape. Over the scalar-and-set fragment the `member_v` oracle
        /// decides exactly, so a simplified schema must admit each sample value
        /// exactly when the original does. This catches an unsound rewrite that the
        /// structural-equality laws above cannot, since they only compare two
        /// already-simplified forms. Refinements carry value-level bounds the
        /// kind-only samples cannot evaluate, so their membership preservation is
        /// covered by the Python suite over real values.
        #[test]
        fn simplify_preserves_membership(a in scalar_or_set_schema()) {
            let simplified = a.simplify();
            for value in &samples_v() {
                prop_assert_eq!(member_v(&simplified, value), member_v(&a, value));
            }
        }

        /// The sound invariants over the whole structural fragment: every
        /// procedure terminates without panicking, `simplify` reaches a fixpoint
        /// after one application, the order is reflexive with the lattice bounds
        /// above and below every schema, and equivalence is exactly mutual
        /// inclusion. These hold despite the conservatism, so a violation is a
        /// defect; this is the stable-toolchain mirror of the fuzz targets.
        #[test]
        fn structural_decision_invariants(a in structural_schema(), b in structural_schema()) {
            let once = a.simplify();
            prop_assert_eq!(once.clone(), once.simplify());
            prop_assert!(a.is_subtype_of(&a));
            prop_assert!(a.is_equivalent(&a));
            prop_assert!(a.is_subtype_of(&Schema::Anything));
            prop_assert!(Schema::Nothing.is_subtype_of(&a));
            let ab = a.is_subtype_of(&b);
            let ba = b.is_subtype_of(&a);
            prop_assert_eq!(a.is_equivalent(&b), ab && ba);
            let _ = a.is_empty();
        }

        #[test]
        fn complement_laws_on_the_scalar_fragment(a in scalar_schema()) {
            // On the decidable scalar fragment the complement laws hold exactly:
            // a meet its complement is empty, and a join its complement is the
            // universe. The decision procedure folds both to the lattice bounds.
            prop_assert!(intersection(a.clone(), not(a.clone())).is_empty());
            prop_assert!(not(union(a.clone(), not(a))).is_empty());
        }
    }

    /// A balanced tree of complements over unions: every level wraps two copies
    /// of the level below in a union and complements the result. Its node count
    /// doubles per level, so a single bottom-up simplification visits each node
    /// once and finishes in milliseconds, while a pass that re-normalises every
    /// member once per level it is nested under grows superlinearly on top of
    /// that and takes tens of seconds at this depth.
    fn complemented_tower(depth: usize) -> Schema {
        if depth == 0 {
            return Schema::Complement(Box::new(Schema::Int));
        }
        let child = complemented_tower(depth - 1);
        Schema::Complement(Box::new(Schema::Union(vec![child.clone(), child])))
    }

    /// A tower of intersections of unions, the shape whose subtyping decision
    /// re-explored shared subtrees before the goal memo. The leaves are sets so
    /// the scalar region fast path does not short-circuit the descent.
    fn intersection_of_unions_tower(depth: usize, leaf: Schema) -> Schema {
        let mut node = Schema::Set(Box::new(leaf));
        for _ in 0..depth {
            node = Schema::Intersection(vec![
                Schema::Union(vec![node.clone(), Schema::Set(Box::new(Schema::Str))]),
                Schema::Union(vec![node, Schema::Set(Box::new(Schema::Bytes))]),
            ]);
        }
        node
    }

    /// The simplifier stays within a single bottom-up pass: a deeply nested
    /// complemented tree is reduced by visiting each of its nodes once. A
    /// regression to re-normalising each member per nesting level visits them
    /// once per level instead, which is what this separates.
    ///
    /// The pass is measured in nodes visited rather than in seconds. One visit
    /// per node *is* the claim, and the count is the same number on every
    /// machine; a duration is a claim about the machine as much as about the
    /// pass, and it separates a linear pass from a quadratic one only where the
    /// machine is fast enough to notice.
    #[test]
    fn simplify_stays_linear_on_a_complemented_tower() {
        let schema = complemented_tower(18);
        // The duplicate union members collapse, so the reduced form is small;
        // the point is the work it took to get there.
        assert!(matches!(schema.simplify(), Schema::Complement(_)));
        let nodes = schema.node_count() as u64;
        let steps = schema.simplify_steps();
        assert!(
            steps <= nodes,
            "simplify visited {steps} nodes of a {nodes}-node tower; a single \
             bottom-up pass visits each at most once"
        );
    }

    /// A decision says which of the three things it established, and an exhausted
    /// budget is never mistaken for a proof.
    ///
    /// This is the whole point of the third value. `is_empty` answering `false`
    /// covers a schema proven to admit values and a schema the work bound
    /// stopped, and nothing outside the procedure could tell them apart -- so a
    /// budget hit at a realistic size read as a confident answer. It reads as
    /// `Unknown` now, and this fails if it ever reads as either proof.
    #[test]
    fn an_exhausted_budget_proves_nothing_in_either_direction() {
        // Proven, both ways, on the fragment the regions decide exactly.
        assert_eq!(Schema::Int.verdict(), Verdict::Inhabited);
        assert_eq!(Schema::Nothing.verdict(), Verdict::Empty);
        assert_eq!(
            intersection(Schema::Int, Schema::Str).verdict(),
            Verdict::Empty
        );
        assert_eq!(
            union(Schema::Int, Schema::Str).verdict(),
            Verdict::Inhabited
        );
        assert_eq!(
            intersection(Schema::Int, not(Schema::Int)).verdict(),
            Verdict::Empty
        );

        // Not proven, because the core cannot read the leaf. A literal's constant
        // may be `nan`, which is equal to nothing and denotes the empty set.
        assert_eq!(Schema::Literal(ConstIx::new(0)).verdict(), Verdict::Unknown);
        assert_eq!(
            Schema::Instance(ClassIx::new(0)).verdict(),
            Verdict::Unknown
        );

        // Not proven, because the budget stopped the descent. The same tower the
        // step-count test drives: it spends the whole allowance, so whatever the
        // fold would have concluded, it did not conclude it here.
        let tower = intersection_of_unions_tower(18, Schema::Int);
        assert_eq!(
            tower.empty_steps(),
            DECISION_BUDGET,
            "the tower drives the bound"
        );
        assert_eq!(
            tower.verdict(),
            Verdict::Unknown,
            "an exhausted budget must not read as a proof"
        );
        // And the public relation still answers the sound `false` for it, which
        // is what "Unknown reduces to not-proven-empty" means.
        assert!(!tower.is_empty());
    }

    /// The subtyping decision terminates on a deeply nested
    /// intersection-of-unions, where the union and intersection distribution
    /// rules re-explore the schema exponentially in its depth. The work budget
    /// stops the descent and returns the conservative answer instead of running
    /// for minutes; this guards against a regression that removes the bound.
    ///
    /// The bound is read as a step count rather than as a duration. What the
    /// budget promises is that no query spends more than `DECISION_BUDGET`
    /// steps, and that is the same number on a quiet laptop and on a loaded CI
    /// runner. A wall-clock assertion tests the machine alongside the algorithm
    /// and fails for reasons that have nothing to do with the code.
    // SWEEP-SKIP: this case exists to prove a bound, so a mutation that removes
    // the bound makes it run without end. It stays in the test lane and leaves
    // the mutation sweep, where a run that returns no verdict is a rig fault.
    #[test]
    fn subtyping_terminates_on_a_distributed_tower() {
        let narrow = intersection_of_unions_tower(18, Schema::Int);
        let wide = intersection_of_unions_tower(18, union(Schema::Int, Schema::Float));
        // The verdict on this adversarial shape may be conservative; the property
        // under test is that the decision stops rather than the answer.
        let steps = narrow.subtype_steps(&wide);
        assert!(
            steps <= DECISION_BUDGET,
            "is_subtype_of on a depth-18 distributed tower spent {steps} steps, \
             past the {DECISION_BUDGET}-step budget"
        );
        // The shape really does exhaust the budget: a bound this case never
        // reaches would pass with the bound removed.
        assert_eq!(
            steps, DECISION_BUDGET,
            "the tower stopped short of the budget, so it no longer drives it"
        );
    }

    /// Structural depth counts one level per nested constructor, takes the max
    /// over a node's children rather than the sum, and treats a `Ref` back edge
    /// as a leaf so a recursive schema has finite depth. The composition guard
    /// relies on this to bound the native stack every recursive walk descends.
    #[test]
    fn depth_counts_nesting_and_treats_refs_as_leaves() {
        assert_eq!(Schema::Int.depth(), 1);
        assert_eq!(Schema::Ref(DefIx::new(0)).depth(), 1);
        assert_eq!(Schema::Complement(Box::new(Schema::Int)).depth(), 2);
        assert_eq!(union(Schema::Int, Schema::Str).depth(), 2);
        // The max over members, not their sum: one branch is two deep.
        let branchy = union(Schema::Int, Schema::Complement(Box::new(Schema::Str)));
        assert_eq!(branchy.depth(), 3);
        // A left-nested tower grows by exactly one level per composition.
        let mut tower = Schema::Int;
        for _ in 0..10 {
            tower = union(tower, Schema::Str);
        }
        assert_eq!(tower.depth(), 11);
        // A list whose element is a recursive back edge is finite: the `Ref` is a
        // leaf, so the depth does not follow it into the definitions table.
        let recursive_list = Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0))));
        assert!(recursive_list.depth() < 10);
    }

    /// The work budget must not change a verdict a real schema needs, including
    /// under recursion: a recursive list of ints is a subtype of itself and of a
    /// wider recursive list, and the wider one is not a subtype of the narrower.
    #[test]
    fn budgeted_subtyping_decides_recursive_relations() {
        let int_list = Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::homogeneous(Schema::Ref(DefIx::new(0))),
        };
        let wide_list = Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::homogeneous(Schema::Ref(DefIx::new(1))),
        };
        let defs = vec![
            union(Schema::Int, int_list.clone()),
            union(union(Schema::Int, Schema::Str), wide_list.clone()),
        ];
        let oracle = NoLeafRelations;
        assert!(int_list.is_subtype_of_under(&int_list, &oracle, &defs));
        assert!(int_list.is_subtype_of_under(&wide_list, &oracle, &defs));
        assert!(!wide_list.is_subtype_of_under(&int_list, &oracle, &defs));
    }

    /// The shared budget threads through the subtype-into-bottom path (which calls
    /// the emptiness decision) and through both directions of equivalence without
    /// changing a real verdict, including an uninhabited recursive reference
    /// reached via `A ⊆ ∅`.
    #[test]
    fn the_shared_budget_decides_real_emptiness_and_equivalence() {
        // `A ⊆ ∅` reaches the budgeted emptiness check.
        assert!(intersection(Schema::Int, Schema::Str).is_subtype_of(&Schema::Nothing));
        assert!(!Schema::Int.is_subtype_of(&Schema::Nothing));
        // Equivalence runs both directions against one budget and still decides.
        assert!(union(Schema::Int, Schema::Str).is_equivalent(&union(Schema::Str, Schema::Int)));
        assert!(!Schema::Int.is_equivalent(&Schema::Str));
        // An uninhabited recursive reference is empty, decided through the
        // subtype-into-bottom crossing under the shared budget.
        let defs = vec![Schema::Ref(DefIx::new(0))];
        assert!(Schema::Ref(DefIx::new(0)).is_subtype_of_under(
            &Schema::Nothing,
            &NoLeafRelations,
            &defs
        ));
    }

    /// Querying a deep schema against bottom routes through the emptiness check,
    /// which shares the subtyping budget, so it stops promptly rather than running
    /// the decision unbounded down a side door.
    // SWEEP-SKIP: this case exists to prove a bound, so a mutation that removes
    // the bound makes it run without end. It stays in the test lane and leaves
    // the mutation sweep, where a run that returns no verdict is a rig fault.
    #[test]
    fn deep_subtype_into_bottom_terminates() {
        let deep = intersection_of_unions_tower(18, Schema::Int);
        let steps = deep.subtype_steps(&Schema::Nothing);
        assert!(
            steps <= DECISION_BUDGET,
            "subtype-into-bottom on a depth-18 tower spent {steps} steps, past \
             the {DECISION_BUDGET}-step budget"
        );
    }

    /// The bottom-up region the emptiness pass folds from a node's children
    /// detects a *collective* cancellation — one where no single member is empty
    /// but their regions cancel — through nesting. Two disjoint scalar unions
    /// cancel; wrapping the result deeper must not lose that.
    #[test]
    fn emptiness_folds_collective_region_cancellation_through_nesting() {
        // Neither member is empty alone; their regions are disjoint, so the
        // intersection is empty — decided by the folded region, not by a member.
        let cancel = intersection(
            union(Schema::Int, Schema::Str),
            union(Schema::Float, Schema::Bytes),
        );
        assert!(cancel.is_empty());
        // The same cancellation, buried under more Boolean structure, still folds
        // up: the region reaches zero at the outer intersection.
        let nested = intersection(
            intersection(cancel.clone(), Schema::Anything),
            not(Schema::NoneType),
        );
        assert!(nested.is_empty());
        // A complement chain over scalars keeps the non-scalar region, so it is not
        // empty — the fold must preserve that, not over-report empty.
        let surviving = intersection(not(Schema::Int), not(Schema::Str));
        assert!(!surviving.is_empty());
    }

    /// The emptiness decision derives each intersection's region from its children
    /// instead of re-walking the whole subtree at every level, so a deeply nested
    /// intersection is decided in work linear in its size. The pre-fix quadratic
    /// re-walk visited the subtree once per level, which is the growth this
    /// pins. Run on a large stack because the schema is intentionally
    /// left-nested to this depth.
    ///
    /// Linearity is asserted on the step count at two depths rather than on a
    /// duration. Doubling the depth doubles the steps of a linear pass and
    /// quadruples the steps of the re-walk, so the ratio separates the two
    /// exactly, on any machine and under any load. A wall-clock bound separates
    /// them only where the machine is fast enough, which is a property of the
    /// runner.
    #[test]
    fn emptiness_decides_a_deep_intersection_in_linear_time() {
        /// ¬Int ∩ ¬Str ∩ … : region-decidable, never empty (the non-scalar
        /// region survives), so the walk visits every level — the worst case.
        fn left_nested_complements(depth: usize) -> Schema {
            let mut deep = Schema::Complement(Box::new(Schema::Int));
            for _ in 0..depth {
                deep = Schema::Intersection(vec![deep, Schema::Complement(Box::new(Schema::Str))]);
            }
            deep
        }

        let worker = std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(|| {
                let shallow = left_nested_complements(10_000);
                let deep = left_nested_complements(20_000);
                assert!(!shallow.is_empty());
                assert!(!deep.is_empty());
                let (small, large) = (shallow.empty_steps(), deep.empty_steps());
                // Linear growth doubles the work; the quadratic re-walk would
                // quadruple it. Three is the midpoint that tells the two apart
                // and leaves room for the constant per-level overhead.
                assert!(
                    u64::from(large) < 3 * u64::from(small),
                    "doubling the depth took {small} steps to {large}; a linear \
                     fold roughly doubles, a per-level re-walk quadruples"
                );
            })
            .expect("spawn worker thread");
        worker.join().expect("deep emptiness worker panicked");
    }

    /// An attribute record with an uninhabited required attribute is empty: no
    /// value can carry that attribute, so the dataclass-style schema denotes
    /// nothing. The symmetric keyed-map rule already held; this closes the
    /// asymmetry.
    #[test]
    fn an_uninhabited_required_attribute_empties_the_schema() {
        let record = |schema| Schema::AttrRecord {
            fields: vec![Field {
                name: "x".into(),
                schema,
                required: true,
            }],
        };
        let empty_field = record(intersection(Schema::Int, Schema::Str));
        assert!(empty_field.is_empty());
        assert!(!record(Schema::Int).is_empty());
        // The class the frontend meets with the record does not rescue it: an
        // empty conjunct empties the meet.
        assert!(Schema::meet([Schema::Instance(ClassIx::new(0)), empty_field]).is_empty());
    }

    /// A class with declared attributes is read back out of the meet the
    /// frontend builds it as, so `repr` and a union's branch label can name the
    /// class the user wrote. A meet that is not that pair names no class, and a
    /// meet of two objects names neither.
    #[test]
    fn an_object_meet_knows_which_class_it_is() {
        let record = Schema::AttrRecord {
            fields: vec![Field {
                name: "x".into(),
                schema: Schema::Int,
                required: true,
            }],
        };
        let class = |index| Schema::Instance(ClassIx::new(index));
        let object = Schema::meet([class(3), record.clone()]);
        assert_eq!(object.object_class(), Some(ClassIx::new(3)));
        // A later meet may flatten other members in beside the pair; it is still
        // the same class.
        assert_eq!(
            Schema::meet([class(3), record.clone(), Schema::Int]).object_class(),
            Some(ClassIx::new(3))
        );
        // Each half alone, and a meet of two objects, name no single class.
        assert_eq!(class(3).object_class(), None);
        assert_eq!(record.object_class(), None);
        assert_eq!(
            Schema::meet([class(3), class(4), record.clone()]).object_class(),
            None
        );
        assert_eq!(
            Schema::meet([class(3), record.clone(), record]).object_class(),
            None
        );
    }

    /// Attribute records relate by width and depth: a record carrying every
    /// attribute of the supertype with a narrower schema is a subtype. The class
    /// is a separate conjunct, so a record over one class relates to a record
    /// over another -- and the meets stay conservative, because the nominal
    /// hierarchy is not decided in the core.
    #[test]
    fn attribute_records_subtype_by_width_and_depth() {
        let narrow = Schema::AttrRecord {
            fields: vec![
                Field {
                    name: "x".into(),
                    schema: Schema::Bool,
                    required: true,
                },
                Field {
                    name: "y".into(),
                    schema: Schema::Str,
                    required: true,
                },
            ],
        };
        let wide = Schema::AttrRecord {
            fields: vec![Field {
                name: "x".into(),
                schema: Schema::Int, // bool ⊆ int, and x is narrower; y is extra
                required: true,
            }],
        };
        assert!(narrow.is_subtype_of(&wide));
        assert!(!wide.is_subtype_of(&narrow)); // wide lacks y
        // Required-ness is part of the relation: a supertype that demands the
        // attribute is not satisfied by a subtype that only may carry it, and
        // the values with it missing are what separate them.
        let maybe_x = Schema::AttrRecord {
            fields: vec![Field {
                name: "x".into(),
                schema: Schema::Bool,
                required: false,
            }],
        };
        assert!(!maybe_x.is_subtype_of(&wide));
        assert!(narrow.is_subtype_of(&Schema::AttrRecord {
            fields: vec![Field {
                name: "x".into(),
                schema: Schema::Int,
                required: false,
            }],
        }));
        let object =
            |class, record: &Schema| Schema::meet([Schema::Instance(class), record.clone()]);
        // Same records, different classes: conservative, and the record half is
        // what the meet rule reaches for on the way there.
        assert!(!object(ClassIx::new(0), &narrow).is_subtype_of(&object(ClassIx::new(1), &wide)));
        assert!(object(ClassIx::new(0), &narrow).is_subtype_of(&wide));
    }

    /// A sequence whose repeated tail is empty only under the recursive
    /// definitions matches just its fixed prefix, so it is a subtype of the bare
    /// prefix sequence. The tail's emptiness is decided with the active context,
    /// not the public no-context check, which would miss it.
    #[test]
    fn a_tail_empty_under_defs_reduces_to_the_prefix() {
        // def 0 references only itself: an uninhabited recursive schema.
        let defs = vec![Schema::Ref(DefIx::new(0))];
        let with_phantom_tail = Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::prefix_tail([Schema::Int], Schema::Ref(DefIx::new(0))),
        };
        let just_int = Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::fixed([Schema::Int]),
        };
        let oracle = NoLeafRelations;
        // The phantom tail never repeats, so the two denote the same language.
        assert!(with_phantom_tail.is_subtype_of_under(&just_int, &oracle, &defs));
        assert!(just_int.is_subtype_of_under(&with_phantom_tail, &oracle, &defs));
    }

    /// The union-covers-the-universe fold is a live simplification, not dead code:
    /// the complement of a scalar carries the non-scalar region, so a complement
    /// beside a covering scalar reduces to the top even when no complementary or
    /// disjoint-complement pair is present.
    #[test]
    fn a_complement_plus_a_covering_scalar_is_the_universe() {
        // ¬bool covers everything except bools; int adds the bools back.
        let everything = union(not(Schema::Bool), Schema::Int);
        assert_eq!(everything.simplify(), Schema::Anything);
    }

    // -- An independent, value-aware denotation oracle ----------------------------
    //
    // The scalar oracle above models kinds, so it cannot tell `Literal[1]` from
    // `Literal[2]` or `Ge(0)` from `Ge(5)`. This oracle carries concrete values
    // and a fixed constant pool, so it decides membership for the whole
    // non-opaque fragment — literals, refinement bounds and lengths, sequences,
    // sets, and records — and is the ground truth `simplify` is checked against
    // over that fragment. It is a direct transcription of each node's denotation,
    // sharing no code with `simplify` or the decision procedure under test.

    /// A concrete Python-shaped value.
    #[derive(Clone, Debug)]
    enum Obj {
        None,
        Bool(bool),
        Int(i32),
        Float(f64),
        Str(&'static str),
        Bytes,
        List(Vec<Obj>),
        Tuple(Vec<Obj>),
        Set(Vec<Obj>),
        FrozenSet(Vec<Obj>),
        Map(Vec<(&'static str, Obj)>),
    }

    /// The fixed constant pool that generated `Literal` and bound indices point
    /// into. Indices 0..=2 are numbers (usable as bounds); 3 and 4 add a string
    /// and a bool so the typed-singleton distinction is exercised.
    fn const_pool() -> Vec<Obj> {
        vec![
            Obj::Int(0),
            Obj::Int(1),
            Obj::Int(5),
            Obj::Str("a"),
            Obj::Bool(true),
        ]
    }
    const POOL_LEN: usize = 5;

    fn as_num(v: &Obj) -> Option<f64> {
        match v {
            // bool is an int in Python, so it orders numerically.
            Obj::Bool(b) => Some(f64::from(u8::from(*b))),
            Obj::Int(i) => Some(f64::from(*i)),
            Obj::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn val_len(v: &Obj) -> Option<usize> {
        match v {
            Obj::Str(s) => Some(s.chars().count()),
            Obj::List(xs) | Obj::Tuple(xs) | Obj::Set(xs) | Obj::FrozenSet(xs) => Some(xs.len()),
            Obj::Map(m) => Some(m.len()),
            _ => None,
        }
    }

    /// Typed-singleton equality: same type *and* equal, so `Literal[1]` admits
    /// neither `True` nor `1.0`.
    fn typed_eq(constant: &Obj, v: &Obj) -> bool {
        match (constant, v) {
            (Obj::None, Obj::None) | (Obj::Bytes, Obj::Bytes) => true,
            (Obj::Bool(a), Obj::Bool(b)) => a == b,
            (Obj::Int(a), Obj::Int(b)) => a == b,
            (Obj::Float(a), Obj::Float(b)) => a == b,
            (Obj::Str(a), Obj::Str(b)) => a == b,
            _ => false,
        }
    }

    fn bound_holds(constraint: &Constraint, value: &Obj, pool: &[Obj]) -> bool {
        use core::cmp::Ordering;
        let cmp_to = |index: &OperandIx, ok: fn(Ordering) -> bool| {
            match (as_num(value), as_num(&pool[index.get()])) {
                (Some(lhs), Some(rhs)) => lhs.partial_cmp(&rhs).is_some_and(ok),
                _ => false, // a numeric bound on a non-numeric value raises: non-member
            }
        };
        match constraint {
            Constraint::Ge(index) => cmp_to(index, |ord| ord != Ordering::Less),
            Constraint::Gt(index) => cmp_to(index, |ord| ord == Ordering::Greater),
            Constraint::Le(index) => cmp_to(index, |ord| ord != Ordering::Greater),
            Constraint::Lt(index) => cmp_to(index, |ord| ord == Ordering::Less),
            Constraint::MinLen(min) => val_len(value).is_some_and(|len| len >= *min),
            Constraint::MaxLen(max) => val_len(value).is_some_and(|len| len <= *max),
            Constraint::MultipleOf(index) => match (as_num(value), as_num(&pool[index.get()])) {
                (Some(lhs), Some(rhs)) if rhs != 0.0 => lhs % rhs == 0.0,
                _ => false,
            },
            // Not generated for this oracle (opaque user code); never reached.
            Constraint::Predicate(_) | Constraint::Regex(_) => false,
        }
    }

    /// Match a sequence's items against its shape with the oracle's *own* matcher,
    /// sharing no code with the decision procedure under test.
    ///
    /// The denotation, written out: the first `prefix.len()` items must belong to
    /// the prefix schemas positionally, and every item past them to the tail --
    /// with no such item at all when there is no tail. It is short enough to read
    /// against the definition, which is what makes it an oracle rather than a
    /// second implementation of the same walk.
    fn seq_matches(shape: &SeqShape, items: &[Obj], pool: &[Obj]) -> bool {
        let fits = match &shape.tail {
            Some(_) => items.len() >= shape.prefix.len(),
            None => items.len() == shape.prefix.len(),
        };
        fits && items.iter().enumerate().all(|(i, item)| {
            let element = shape.prefix.get(i).or(shape.tail.as_deref());
            element.is_some_and(|schema| member_full(schema, item, pool))
        })
    }

    /// Reference membership over the non-opaque fragment, transcribing each node's
    /// denotation directly.
    fn member_full(schema: &Schema, value: &Obj, pool: &[Obj]) -> bool {
        match schema {
            Schema::Anything | Schema::Dynamic => true,
            Schema::Nothing => false,
            Schema::NoneType => matches!(value, Obj::None),
            Schema::Bool => matches!(value, Obj::Bool(_)),
            Schema::Int => matches!(value, Obj::Bool(_) | Obj::Int(_)), // bool ⊆ int
            Schema::Float => matches!(value, Obj::Float(_)),
            Schema::Str => matches!(value, Obj::Str(_)),
            Schema::Bytes => matches!(value, Obj::Bytes),
            Schema::Literal(index) => typed_eq(&pool[index.get()], value),
            Schema::Set(element) => match value {
                Obj::Set(items) => items.iter().all(|item| member_full(element, item, pool)),
                _ => false,
            },
            Schema::FrozenSet(element) => match value {
                Obj::FrozenSet(items) => items.iter().all(|item| member_full(element, item, pool)),
                _ => false,
            },
            Schema::Seq { container, shape } => match (container, value) {
                (SeqKind::List, Obj::List(items)) | (SeqKind::Tuple, Obj::Tuple(items)) => {
                    seq_matches(shape, items, pool)
                }
                _ => false,
            },
            Schema::KeyedMap { fields, defaults } => match value {
                Obj::Map(entries) => {
                    let fields_ok = fields.iter().all(|field| {
                        match entries.iter().find(|(key, _)| field.name == *key) {
                            Some((_, val)) => member_full(&field.schema, val, pool),
                            None => !field.required,
                        }
                    });
                    let rest_ok = entries.iter().all(|(key, val)| {
                        if fields.iter().any(|field| field.name == *key) {
                            return true;
                        }
                        defaults.iter().any(|clause| {
                            member_full(&clause.key, &Obj::Str(key), pool)
                                && member_full(&clause.value, val, pool)
                        })
                    });
                    fields_ok && rest_ok
                }
                _ => false,
            },
            Schema::Refine { base, constraints } => {
                member_full(base, value, pool)
                    && constraints
                        .iter()
                        .all(|constraint| bound_holds(constraint, value, pool))
            }
            Schema::Union(members) => members
                .iter()
                .any(|member| member_full(member, value, pool)),
            Schema::Intersection(members) => members
                .iter()
                .all(|member| member_full(member, value, pool)),
            Schema::Complement(inner) => !member_full(inner, value, pool),
            // The opaque leaves are excluded from the generator below.
            other => unreachable!("oracle does not model {other:?}"),
        }
    }

    fn sample_values() -> Vec<Obj> {
        vec![
            Obj::None,
            Obj::Bool(true),
            Obj::Bool(false),
            Obj::Int(0),
            Obj::Int(1),
            Obj::Int(2),
            Obj::Int(5),
            Obj::Float(1.0),
            Obj::Float(2.5),
            Obj::Str("a"),
            Obj::Str("b"),
            Obj::Str(""),
            Obj::Bytes,
            Obj::List(vec![]),
            Obj::List(vec![Obj::Int(1)]),
            Obj::List(vec![Obj::Int(1), Obj::Str("a")]),
            Obj::Set(vec![]),
            Obj::Set(vec![Obj::Int(1)]),
            Obj::FrozenSet(vec![]),
            Obj::FrozenSet(vec![Obj::Int(1)]),
            Obj::Tuple(vec![]),
            Obj::Tuple(vec![Obj::Int(1)]),
            Obj::Tuple(vec![Obj::Int(1), Obj::Str("a")]),
            Obj::Map(vec![]),
            Obj::Map(vec![("a", Obj::Int(1))]),
            Obj::Map(vec![("a", Obj::Int(1)), ("b", Obj::Str("a"))]),
            // An entry whose key is not a declared field exercises the open-record
            // `defaults` arm against both a matching and a non-matching value.
            Obj::Map(vec![("c", Obj::Int(1))]),
            Obj::Map(vec![("a", Obj::Int(1)), ("c", Obj::Str("a"))]),
        ]
    }

    /// A bound or length constraint over the pool: comparisons point at the
    /// numeric entries, `MultipleOf` at a nonzero one, lengths at small counts.
    fn constraint_strategy() -> impl Strategy<Value = Constraint> {
        prop_oneof![
            (0usize..3).prop_map(|i| Constraint::Ge(OperandIx::new(i))),
            (0usize..3).prop_map(|i| Constraint::Gt(OperandIx::new(i))),
            (0usize..3).prop_map(|i| Constraint::Le(OperandIx::new(i))),
            (0usize..3).prop_map(|i| Constraint::Lt(OperandIx::new(i))),
            (0usize..4usize).prop_map(Constraint::MinLen),
            (0usize..4usize).prop_map(Constraint::MaxLen),
            (1usize..3).prop_map(|i| Constraint::MultipleOf(OperandIx::new(i))),
        ]
    }

    /// A generator over the non-opaque fragment the value oracle decides: scalars,
    /// pool literals, sets, linear list sequences, refinements, closed records,
    /// and their Boolean combinations.
    /// Keep the first field of each name, as the frontend's uniqueness check
    /// leaves. A record with two fields of one name is an IR the frontend
    /// refuses to build, and the decision procedure reads its field index on
    /// that invariant, so generating one measures the assertion rather than the
    /// rule.
    fn unique_by_name(fields: Vec<Field>) -> Vec<Field> {
        let mut seen: Vec<String> = Vec::new();
        fields
            .into_iter()
            .filter(|field| {
                let fresh = !seen.contains(&field.name);
                if fresh {
                    seen.push(field.name.clone());
                }
                fresh
            })
            .collect()
    }

    fn decidable_schema() -> impl Strategy<Value = Schema> {
        let leaf = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Nothing),
            Just(Schema::Dynamic),
            Just(Schema::NoneType),
            Just(Schema::Bool),
            Just(Schema::Int),
            Just(Schema::Float),
            Just(Schema::Str),
            Just(Schema::Bytes),
            (0usize..POOL_LEN).prop_map(|i| Schema::Literal(ConstIx::new(i))),
        ];
        leaf.prop_recursive(3, 48, 4, |inner| {
            let field =
                (0usize..2, inner.clone(), proptest::bool::ANY).prop_map(|(n, schema, req)| {
                    Field {
                        name: ["a", "b"][n].to_owned(),
                        schema,
                        required: req,
                    }
                });
            // An open record: 0..2 declared fields plus a `Str -> value` default
            // arm, so `member_full`'s `defaults` branch is actually reached.
            let open_record = (
                proptest::collection::vec(field.clone(), 0..2),
                inner.clone(),
            )
                .prop_map(|(fields, value)| Schema::KeyedMap {
                    fields: unique_by_name(fields),
                    defaults: vec![MapClause {
                        key: Schema::Str,
                        value,
                    }],
                });
            prop_oneof![
                inner.clone().prop_map(|s| Schema::Set(Box::new(s))),
                inner.clone().prop_map(|s| Schema::FrozenSet(Box::new(s))),
                inner.clone().prop_map(|s| Schema::Seq {
                    container: SeqKind::List,
                    shape: SeqShape::homogeneous(s),
                }),
                inner.clone().prop_map(|s| Schema::Seq {
                    container: SeqKind::Tuple,
                    shape: SeqShape::homogeneous(s),
                }),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Schema::Seq {
                    container: SeqKind::Tuple,
                    shape: SeqShape::fixed([a, b]),
                }),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Schema::Seq {
                    container: SeqKind::List,
                    shape: SeqShape::fixed([a, b]),
                }),
                (
                    inner.clone(),
                    proptest::collection::vec(constraint_strategy(), 1..3)
                )
                    .prop_map(|(base, constraints)| Schema::Refine {
                        base: Box::new(base),
                        constraints,
                    }),
                // A closed record. The names are deduplicated: a record with two
                // fields of one name is an IR the frontend refuses to build, and
                // the decision procedure reads the field index on the invariant
                // that it does.
                proptest::collection::vec(field, 0..3).prop_map(|fields| Schema::KeyedMap {
                    fields: unique_by_name(fields),
                    defaults: vec![],
                }),
                open_record,
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    /// The oracle's own sequence matcher (independent of `SeqShape::linear`) and
    /// the container arms decide the shapes the generator emits: a homogeneous
    /// `Star` sequence, a fixed `Cat` pair, and the open-record `defaults` branch.
    #[test]
    fn the_value_oracle_matches_sequences_and_open_records_independently() {
        let pool = const_pool();
        // `[int, ...]`: empty and homogeneous lists match; a wrong element does not.
        let int_list = Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::homogeneous(Schema::Int),
        };
        assert!(member_full(&int_list, &Obj::List(vec![]), &pool));
        assert!(member_full(
            &int_list,
            &Obj::List(vec![Obj::Int(1), Obj::Int(5)]),
            &pool
        ));
        assert!(!member_full(
            &int_list,
            &Obj::List(vec![Obj::Str("a")]),
            &pool
        ));
        // A tuple shape is not a list and vice versa.
        let int_pair = Schema::Seq {
            container: SeqKind::Tuple,
            shape: SeqShape::fixed([Schema::Int, Schema::Str]),
        };
        assert!(member_full(
            &int_pair,
            &Obj::Tuple(vec![Obj::Int(1), Obj::Str("a")]),
            &pool
        ));
        assert!(!member_full(
            &int_pair,
            &Obj::Tuple(vec![Obj::Int(1)]),
            &pool
        )); // wrong arity
        assert!(!member_full(
            &int_pair,
            &Obj::List(vec![Obj::Int(1), Obj::Str("a")]),
            &pool
        ));
        // A frozenset is distinct from a set.
        let frozen_int = Schema::FrozenSet(Box::new(Schema::Int));
        assert!(member_full(
            &frozen_int,
            &Obj::FrozenSet(vec![Obj::Int(1)]),
            &pool
        ));
        assert!(!member_full(
            &frozen_int,
            &Obj::Set(vec![Obj::Int(1)]),
            &pool
        ));
        // Open record `{str: int}` with no declared fields: the `defaults` arm
        // accepts a matching extra entry and rejects a mistyped one.
        let open = Schema::KeyedMap {
            fields: vec![],
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        };
        assert!(member_full(
            &open,
            &Obj::Map(vec![("c", Obj::Int(1))]),
            &pool
        ));
        assert!(!member_full(
            &open,
            &Obj::Map(vec![("c", Obj::Str("a"))]),
            &pool
        ));
    }

    proptest! {
        /// Simplification preserves membership over the whole non-opaque fragment,
        /// judged by an independent value-aware oracle. This is the strong form of
        /// the scalar-only check above: it distinguishes literal values and bound
        /// thresholds, so a rewrite that quietly changes which values a literal,
        /// refinement, sequence, set, or record admits is caught.
        #[test]
        fn simplify_preserves_membership_over_values(schema in decidable_schema()) {
            let pool = const_pool();
            let simplified = schema.simplify();
            for value in &sample_values() {
                prop_assert_eq!(
                    member_full(&simplified, value, &pool),
                    member_full(&schema, value, &pool),
                    "simplify changed membership of {:?}", value
                );
            }
        }

        /// A claimed subtype never admits a value its supertype rejects, over the
        /// same value-aware oracle.
        ///
        /// The scalar-and-set property elsewhere in this crate checks the same
        /// statement over a fragment with no sequence, record, refinement or
        /// literal in it, so every structural rule the decision procedure has was
        /// held only by hand-written examples. Soundness is the one property this
        /// library promises without qualification, and it is the direction a
        /// generator can attack: an accept is a claim about every value, and a
        /// counterexample is a single one.
        #[test]
        fn subtyping_is_sound_over_the_structural_fragment(
            a in decidable_schema(),
            b in decidable_schema(),
        ) {
            let pool = const_pool();
            prop_assert!(a.is_subtype_of(&a), "reflexivity");
            if a.is_subtype_of(&b) {
                for value in &sample_values() {
                    prop_assert!(
                        !member_full(&a, value, &pool) || member_full(&b, value, &pool),
                        "{:?} is in the subtype and not in the supertype", value
                    );
                }
            }
            if a.is_empty() {
                for value in &sample_values() {
                    prop_assert!(
                        !member_full(&a, value, &pool),
                        "{:?} is a member of a schema decided empty", value
                    );
                }
            }
        }
    }
}

/// The laws the two index-space remappings obey.
///
/// `Schema::shifted` and `Schema::reindexed` move every pool and definitions
/// index a schema holds. The unit tests above pin single sites by example, which
/// is the direction that catches a site moved *wrongly*; a law holds the whole
/// operation at once, which is the direction that catches a site moved **not at
/// all**. A payload the shift walks past is invisible to an example written for
/// the payloads somebody thought of, and it is the failure the typed index
/// spaces cannot see: the types stop a shift being applied to the wrong space,
/// and say nothing about whether it was applied.
///
/// The generator's job here is site coverage rather than algebraic variety --
/// every node that carries an index, and every structural node that must carry a
/// remap into its children. A variant it omits is a site these laws do not hold.
#[cfg(test)]
mod index_laws {
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
            name: "f".to_owned(),
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
                inner.clone().prop_map(|s| Schema::Set(Box::new(s))),
                inner.clone().prop_map(|s| Schema::FrozenSet(Box::new(s))),
                inner.clone().prop_map(|s| Schema::Complement(Box::new(s))),
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Intersection),
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
                    fields: vec![field(s)],
                }),
                inner.prop_map(|s| Schema::Refine {
                    base: Box::new(s),
                    constraints: vec![
                        Constraint::Ge(OperandIx::new(5)),
                        Constraint::MultipleOf(OperandIx::new(5)),
                        Constraint::Predicate(PredIx::new(6)),
                        // Neither is a pool index; both must survive untouched.
                        Constraint::MinLen(1),
                        Constraint::Regex("x".to_owned()),
                    ],
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
            Schema::Anything
            | Schema::Dynamic
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
            Schema::Set(inner) | Schema::FrozenSet(inner) | Schema::Complement(inner) => {
                indices(inner, pool, defs);
            }
            Schema::Union(members) | Schema::Intersection(members) => {
                for member in members {
                    indices(member, pool, defs);
                }
            }
            Schema::Seq { shape, .. } => {
                for element in shape.elements() {
                    indices(element, pool, defs);
                }
            }
            Schema::KeyedMap { fields, defaults } => {
                for field in fields {
                    indices(&field.schema, pool, defs);
                }
                for clause in defaults {
                    indices(&clause.key, pool, defs);
                    indices(&clause.value, pool, defs);
                }
            }
            Schema::AttrRecord { fields } => {
                for field in fields {
                    indices(&field.schema, pool, defs);
                }
            }
            Schema::Refine { base, constraints } => {
                indices(base, pool, defs);
                for constraint in constraints {
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
}
