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
mod ir;
mod simplify;
mod violation;

pub use decision::{LeafRelations, NoLeafRelations};
pub use ir::{Constraint, Field, PathSegment, Schema, SeqKind, SeqRegex};
pub use violation::Violation;

/// Fresh, process-unique tokens for the transient [`Schema::SelfRef`] marker, so
/// nested `recursive` definitions never resolve each other's self-references.
static NEXT_SELF_TOKEN: AtomicU64 = AtomicU64::new(0);

/// Allocate a fresh self-reference token for a `recursive` definition.
#[must_use]
pub fn fresh_self_token() -> u64 {
    NEXT_SELF_TOKEN.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `node_count` sizes the whole tree, and the binding's schema-size limit is
    /// the only consumer: an undercount admits a schema past the cap. Each arm
    /// carries a distinct total so a wrong operator cannot coincide with a right
    /// answer.
    #[test]
    fn node_count_totals_every_arm() {
        assert_eq!(Schema::Int.node_count(), 1);
        assert_eq!(Schema::Ref(0).node_count(), 1);
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
            Schema::list(SeqRegex::homogeneous(Schema::Complement(Box::new(
                Schema::Int
            ))))
            .node_count(),
            3
        );
        assert_eq!(
            Schema::list(SeqRegex::Cat(vec![
                SeqRegex::Elem(Box::new(Schema::Int)),
                SeqRegex::Elem(Box::new(Schema::Str)),
            ]))
            .node_count(),
            3
        );
        assert_eq!(Schema::list(SeqRegex::Empty).node_count(), 1);
        // A keyed map counts declared fields and both halves of every default.
        assert_eq!(
            Schema::KeyedMap {
                fields: vec![Field {
                    name: "a".into(),
                    schema: Schema::Complement(Box::new(Schema::Int)),
                    required: true,
                }],
                defaults: vec![(Schema::Str, Schema::Bytes)],
            }
            .node_count(),
            5
        );
        assert_eq!(
            Schema::Attrs {
                class_index: 0,
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
        // Star and Elem each add a level under the Seq node.
        assert_eq!(Schema::list(SeqRegex::Empty).depth(), 1);
        assert_eq!(
            Schema::list(SeqRegex::Elem(Box::new(Schema::Int))).depth(),
            2
        );
        assert_eq!(Schema::list(SeqRegex::homogeneous(Schema::Int)).depth(), 3);
        // Cat takes the max over parts, not the sum.
        assert_eq!(
            Schema::list(SeqRegex::Cat(vec![
                SeqRegex::Elem(Box::new(Schema::Int)),
                SeqRegex::Elem(Box::new(Schema::Complement(Box::new(Schema::Str)))),
            ]))
            .depth(),
            4
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
            Schema::Attrs {
                class_index: 0,
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
                Constraint::Ge(1),
                Constraint::Gt(2),
                Constraint::Le(3),
                Constraint::Lt(4),
                Constraint::MultipleOf(5),
                Constraint::MinLen(6),
                Constraint::MaxLen(7),
            ],
        };
        let Schema::Refine { constraints, .. } = refined.shifted(10, 0) else {
            panic!("shifted a Refine into a non-Refine");
        };
        assert_eq!(
            constraints,
            vec![
                Constraint::Ge(11),
                Constraint::Gt(12),
                Constraint::Le(13),
                Constraint::Lt(14),
                Constraint::MultipleOf(15),
                // Length bounds are counts, not pool indices: unmoved.
                Constraint::MinLen(6),
                Constraint::MaxLen(7),
            ]
        );
        // Pooled leaves shift by the pool; definition refs shift by defs.
        assert_eq!(Schema::Literal(1).shifted(10, 3), Schema::Literal(11));
        assert_eq!(Schema::Instance(1).shifted(10, 3), Schema::Instance(11));
        assert_eq!(Schema::Ref(1).shifted(10, 3), Schema::Ref(4));
    }

    /// A recursive body is well-formed only if every self-reference sits under a
    /// structural constructor; the algebraic combinators pass `guarded` through,
    /// so a reference under only a complement or a refinement is UNGUARDED. If
    /// this check misses one, an unguarded fixpoint is admitted and membership
    /// stops being decidable.
    #[test]
    fn occurs_unguarded_sees_through_the_algebraic_combinators() {
        // Bare: unguarded.
        assert!(Schema::Ref(0).occurs_unguarded(0, false));
        assert!(!Schema::Ref(1).occurs_unguarded(0, false));
        // Complement and Refine do NOT guard: the reference stays exposed.
        assert!(Schema::Complement(Box::new(Schema::Ref(0))).occurs_unguarded(0, false));
        assert!(
            Schema::Refine {
                base: Box::new(Schema::Ref(0)),
                constraints: vec![Constraint::MinLen(1)],
            }
            .occurs_unguarded(0, false)
        );
        assert!(Schema::Union(vec![Schema::Int, Schema::Ref(0)]).occurs_unguarded(0, false));
        assert!(Schema::Intersection(vec![Schema::Int, Schema::Ref(0)]).occurs_unguarded(0, false));
        // Nested combinators still pass it through.
        assert!(
            Schema::Complement(Box::new(Schema::Union(vec![Schema::Int, Schema::Ref(0)])))
                .occurs_unguarded(0, false)
        );
        // Structural constructors guard.
        assert!(!Schema::Set(Box::new(Schema::Ref(0))).occurs_unguarded(0, false));
        assert!(!Schema::FrozenSet(Box::new(Schema::Ref(0))).occurs_unguarded(0, false));
        assert!(!Schema::list(SeqRegex::homogeneous(Schema::Ref(0))).occurs_unguarded(0, false));
        // A guarded reference under a combinator is still guarded.
        assert!(
            !Schema::Complement(Box::new(Schema::Set(Box::new(Schema::Ref(0)))))
                .occurs_unguarded(0, false)
        );
    }

    /// The element schema of a homogeneous (`[T, ...]`) sequence node.
    fn homogeneous_elem(schema: &Schema) -> &Schema {
        match schema {
            Schema::Seq { regex, .. } => {
                regex.linear().expect("linear").1.expect("homogeneous tail")
            }
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
            (Schema::Literal(0), "literal", "literal_error"),
            (
                Schema::list(SeqRegex::homogeneous(Schema::Int)),
                "list",
                "list_type",
            ),
            (
                Schema::tuple(SeqRegex::fixed([Schema::Int])),
                "tuple",
                "tuple_type",
            ),
            (Schema::Set(Box::new(Schema::Int)), "set", "set_type"),
            (
                Schema::mapping(Schema::Str, Schema::Int),
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
                    false,
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
        let mapping = Schema::mapping(Schema::Str, Schema::Int);
        let record = Schema::record(Vec::new(), false);
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
            false,
        );
        let schema = Schema::list(SeqRegex::homogeneous(record));
        let opened = schema.with_records_open(true);
        assert!(record_is_open(homogeneous_elem(&opened)));
        // strict flips it back.
        let closed = schema.with_records_open(true).with_records_open(false);
        assert!(!record_is_open(homogeneous_elem(&closed)));
    }

    #[test]
    fn schema_equality_is_structural() {
        assert_eq!(
            Schema::list(SeqRegex::homogeneous(Schema::Int)),
            Schema::list(SeqRegex::homogeneous(Schema::Int))
        );
        assert_ne!(
            Schema::list(SeqRegex::homogeneous(Schema::Int)),
            Schema::list(SeqRegex::homogeneous(Schema::Str))
        );
        assert_ne!(Schema::Literal(0), Schema::Literal(1));
    }

    #[test]
    fn resolve_self_replaces_only_the_matching_token() {
        let body = Schema::list(SeqRegex::homogeneous(Schema::SelfRef(1)));
        let resolved = body.resolve_self(1, 3);
        assert!(matches!(homogeneous_elem(&resolved), Schema::Ref(3)));
        assert!(matches!(
            Schema::SelfRef(2).resolve_self(1, 3),
            Schema::SelfRef(2)
        ));
    }

    #[test]
    fn contractivity_requires_a_structural_guard() {
        assert!(!Schema::list(SeqRegex::homogeneous(Schema::Ref(0))).occurs_unguarded(0, false));
        assert!(Schema::Ref(0).occurs_unguarded(0, false));
        assert!(Schema::Union(vec![Schema::Int, Schema::Ref(0)]).occurs_unguarded(0, false));
        assert!(
            !Schema::list(SeqRegex::homogeneous(Schema::Union(vec![
                Schema::Int,
                Schema::Ref(0)
            ])))
            .occurs_unguarded(0, false)
        );
    }

    #[test]
    fn shifted_remaps_ref_by_the_definition_offset() {
        let shifted = Schema::list(SeqRegex::homogeneous(Schema::Ref(0))).shifted(7, 4);
        assert!(matches!(homogeneous_elem(&shifted), Schema::Ref(4)));
        assert!(matches!(
            Schema::SelfRef(9).shifted(1, 1),
            Schema::SelfRef(9)
        ));
    }

    #[test]
    fn reindexed_maps_pool_indices_through_the_table() {
        let schema = Schema::Union(vec![
            Schema::Literal(0),
            Schema::Instance(1),
            Schema::Refine {
                base: Box::new(Schema::Int),
                constraints: vec![Constraint::Ge(0), Constraint::MinLen(2)],
            },
            Schema::Ref(0),
        ]);
        let reindexed = schema.reindexed(&[10, 11], 5);
        assert_eq!(
            reindexed,
            Schema::Union(vec![
                Schema::Literal(10),  // 0 -> table[0] = 10
                Schema::Instance(11), // 1 -> table[1] = 11
                Schema::Refine {
                    base: Box::new(Schema::Int),
                    // Ge index remaps through the table; MinLen is a length, untouched.
                    constraints: vec![Constraint::Ge(10), Constraint::MinLen(2)],
                },
                Schema::Ref(5), // ref offset by def_offset = 5
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
    fn linear_recognizes_the_frontend_sequence_shapes() {
        let homogeneous = SeqRegex::homogeneous(Schema::Int);
        let (prefix, tail) = homogeneous.linear().expect("homogeneous is linear");
        assert!(prefix.is_empty());
        assert!(matches!(tail, Some(Schema::Int)));

        let fixed = SeqRegex::fixed([Schema::Int, Schema::Str]);
        let (prefix, tail) = fixed.linear().expect("fixed is linear");
        assert_eq!(prefix.len(), 2);
        assert!(tail.is_none());

        let prefix_tail = SeqRegex::prefix_tail([Schema::Str], Schema::Int);
        let (prefix, tail) = prefix_tail.linear().expect("prefix-plus-tail is linear");
        assert_eq!(prefix.len(), 1);
        assert!(matches!(tail, Some(Schema::Int)));

        let (prefix, tail) = SeqRegex::Empty.linear().expect("empty is linear");
        assert!(prefix.is_empty() && tail.is_none());
    }

    #[test]
    fn linear_rejects_the_non_linear_shapes() {
        let elem = || SeqRegex::Elem(Box::new(Schema::Int));
        // Alternation is not a linear sequence.
        assert!(SeqRegex::Or(vec![elem()]).linear().is_none());
        // A repetition of something other than a single element.
        assert!(
            SeqRegex::Star(Box::new(SeqRegex::Cat(vec![])))
                .linear()
                .is_none()
        );
        // A repetition that is not in tail position.
        let star_first = SeqRegex::Cat(vec![SeqRegex::Star(Box::new(elem())), elem()]);
        assert!(star_first.linear().is_none());
        // Alternation nested inside a concatenation.
        let cat_or = SeqRegex::Cat(vec![SeqRegex::Or(vec![SeqRegex::Empty])]);
        assert!(cat_or.linear().is_none());
    }

    #[test]
    fn sequence_transforms_recurse_through_every_regex_arm() {
        // A regex touching Or, Cat, Star, and Elem, with a Ref element and a
        // SelfRef under a repetition, so every arm of the transforms is walked.
        let regex = SeqRegex::Or(vec![
            SeqRegex::Cat(vec![
                SeqRegex::Elem(Box::new(Schema::Ref(0))),
                SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(Schema::SelfRef(7))))),
            ]),
            SeqRegex::Empty,
        ]);
        let seq = Schema::list(regex);

        // The Ref sits under the sequence guard, so it is not unguarded.
        assert!(!seq.occurs_unguarded(0, false));
        // simplify and with_records_open preserve the sequence shape.
        assert!(matches!(seq.simplify(), Schema::Seq { .. }));
        assert!(matches!(seq.with_records_open(true), Schema::Seq { .. }));

        // shifted moves the Ref element by the definitions offset.
        let Schema::Seq {
            regex: SeqRegex::Or(branches),
            ..
        } = seq.shifted(0, 5)
        else {
            panic!("shape preserved")
        };
        let SeqRegex::Cat(parts) = &branches[0] else {
            panic!("Or branch is a Cat")
        };
        let SeqRegex::Elem(head) = &parts[0] else {
            panic!("first part is an element")
        };
        assert!(matches!(**head, Schema::Ref(5)));

        // resolve_self rewrites the SelfRef under the repetition into a Ref.
        let Schema::Seq {
            regex: SeqRegex::Or(branches),
            ..
        } = seq.resolve_self(7, 3)
        else {
            panic!("shape preserved")
        };
        let SeqRegex::Cat(parts) = &branches[0] else {
            panic!("Or branch is a Cat")
        };
        let SeqRegex::Star(inner) = &parts[1] else {
            panic!("second part is a repetition")
        };
        let SeqRegex::Elem(tail) = inner.as_ref() else {
            panic!("repetition wraps an element")
        };
        assert!(matches!(**tail, Schema::Ref(3)));
    }

    #[test]
    fn keyed_map_transforms_recurse_through_fields_and_defaults() {
        let schema = Schema::KeyedMap {
            fields: vec![Field {
                name: "f".to_owned(),
                schema: Schema::Ref(0),
                required: true,
            }],
            defaults: vec![(Schema::Str, Schema::SelfRef(7))],
        };
        // Both the field's Ref and the default's SelfRef sit under the map guard.
        assert!(!schema.occurs_unguarded(0, false));
        // shifted moves the field's Ref by the definitions offset.
        let Schema::KeyedMap { fields, .. } = schema.shifted(0, 5) else {
            panic!("shape preserved")
        };
        assert!(matches!(fields[0].schema, Schema::Ref(5)));
        // resolve_self rewrites the default clause's SelfRef into a Ref.
        let Schema::KeyedMap { defaults, .. } = schema.resolve_self(7, 3) else {
            panic!("shape preserved")
        };
        assert!(matches!(defaults[0].1, Schema::Ref(3)));
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
        // Disjoint basics and disjoint container kinds give an empty intersection.
        assert_eq!(
            Schema::Intersection(vec![Schema::Int, Schema::Str]).simplify(),
            Schema::Nothing
        );
        assert_eq!(
            Schema::Intersection(vec![
                Schema::list(SeqRegex::homogeneous(Schema::Int)),
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
        let list_int = Schema::list(SeqRegex::homogeneous(Schema::Int));
        let tuple_empty = Schema::tuple(SeqRegex::fixed([]));
        assert!(tuple_empty.disjoint(&list_int)); // tuple vs list
        assert!(
            Schema::FrozenSet(Box::new(Schema::Int)).disjoint(&Schema::Set(Box::new(Schema::Int)))
        );
        assert!(Schema::mapping(Schema::Str, Schema::Int).disjoint(&Schema::Int)); // dict vs int
        // Nothing is disjoint from everything.
        assert!(Schema::Nothing.disjoint(&Schema::Int));
        assert!(Schema::Int.disjoint(&Schema::Nothing));
        // Same tag is not disjoint: two list types share the empty list.
        assert!(!list_int.disjoint(&Schema::list(SeqRegex::homogeneous(Schema::Str))));
        assert!(!Schema::Bool.disjoint(&Schema::Int)); // bool is a subtype of int
        assert!(!Schema::Int.disjoint(&Schema::Int));
        // Conservative where the core cannot decide soundly.
        assert!(!Schema::Literal(0).disjoint(&Schema::Int));
        assert!(!Schema::Instance(0).disjoint(&Schema::Int));
        assert!(!Schema::Dynamic.disjoint(&Schema::Int));
        // A refinement is disjoint exactly when its base is.
        let refined = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![Constraint::Ge(0)],
        };
        assert!(refined.disjoint(&Schema::Str));
        assert!(!refined.disjoint(&Schema::Int));
    }
}

#[cfg(test)]
mod laws {
    use super::*;
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
            Just(Schema::Literal(0)),
            Just(Schema::Instance(1)),
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
        assert!(!Schema::Literal(0).is_empty());
        assert!(!Schema::Instance(0).is_empty());
        assert!(!Schema::Set(Box::new(Schema::Int)).is_empty());
        assert!(!Schema::list(SeqRegex::homogeneous(Schema::Int)).is_empty());
        // A scalar mixed with a non-scalar leaf is undecidable here, so it is
        // never claimed empty (an instance could subclass the scalar's type).
        assert!(!Schema::Intersection(vec![Schema::Int, Schema::Instance(0)]).is_empty());
        // The gradual `Any` is never collapsed.
        assert!(!Schema::Intersection(vec![Schema::Dynamic, not(Schema::Dynamic)]).is_empty());
        // Subtyping off the fragment is reflexive only.
        assert!(Schema::Instance(0).is_subtype_of(&Schema::Instance(0)));
        assert!(!Schema::Instance(0).is_subtype_of(&Schema::Instance(1)));
    }

    #[test]
    fn decides_structural_container_emptiness() {
        // A fixed sequence with an impossible element matches no sequence.
        let empty_pair = Schema::tuple(SeqRegex::fixed([Schema::Int, Schema::Nothing]));
        assert!(empty_pair.is_empty());
        // A list or tuple that admits the empty sequence is never empty.
        assert!(!Schema::list(SeqRegex::homogeneous(Schema::Nothing)).is_empty());
        assert!(!Schema::tuple(SeqRegex::fixed([Schema::Int])).is_empty());
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
            list(SeqRegex::homogeneous(Schema::Bool))
                .is_subtype_of(&list(SeqRegex::homogeneous(Schema::Int)))
        );
        assert!(
            !list(SeqRegex::homogeneous(Schema::Int))
                .is_subtype_of(&list(SeqRegex::homogeneous(Schema::Str)))
        );
        // Fixed sequences compare pointwise; a tuple is not a list.
        assert!(
            tuple(SeqRegex::fixed([Schema::Bool, Schema::Str]))
                .is_subtype_of(&tuple(SeqRegex::fixed([Schema::Int, Schema::Str])))
        );
        assert!(
            !tuple(SeqRegex::fixed([Schema::Int]))
                .is_subtype_of(&list(SeqRegex::homogeneous(Schema::Int)))
        );
        // A fixed list is a subtype of a homogeneous list when each element is.
        assert!(
            list(SeqRegex::fixed([Schema::Bool, Schema::Int]))
                .is_subtype_of(&list(SeqRegex::homogeneous(Schema::Int)))
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
            defaults: vec![(k, v)],
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
        // A record and a mapping are not compared — conservative.
        assert!(!narrow.is_subtype_of(&mapping(Schema::Str, Schema::Int)));
    }

    #[test]
    fn decides_sequence_subtyping_with_prefix_tail_and_alternation() {
        // A list `[head, tail*]`: a one-element fixed prefix then a repeated tail.
        let prefix_tail = |head, tail| {
            Schema::list(SeqRegex::Cat(vec![
                SeqRegex::Elem(Box::new(head)),
                SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(tail)))),
            ]))
        };
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
            Schema::list(SeqRegex::fixed([Schema::Bool, Schema::Int]))
                .is_subtype_of(&prefix_tail(Schema::Int, Schema::Int))
        );
        // Alternation distributes: (bool* | int*) ⊆ int*, but int* ⊄ (bool* | str*).
        let alternation = |a, b| {
            Schema::list(SeqRegex::Or(vec![
                SeqRegex::homogeneous(a),
                SeqRegex::homogeneous(b),
            ]))
        };
        assert!(
            alternation(Schema::Bool, Schema::Int)
                .is_subtype_of(&Schema::list(SeqRegex::homogeneous(Schema::Int)))
        );
        assert!(
            !Schema::list(SeqRegex::homogeneous(Schema::Int))
                .is_subtype_of(&alternation(Schema::Bool, Schema::Str))
        );
    }

    #[test]
    fn decides_tuple_prefix_tail_distinctly_from_lists() {
        // The same prefix-plus-tail regex carried by the tuple container. The
        // decision procedure shares the regex with lists, so this pins that the
        // container is honoured throughout subtyping, emptiness, and equivalence.
        let tup = |head, tail| Schema::tuple(SeqRegex::prefix_tail([head], tail));

        // Subtyping is covariant in both the prefix and the repeated tail.
        assert!(tup(Schema::Bool, Schema::Bool).is_subtype_of(&tup(Schema::Int, Schema::Int)));
        assert!(!tup(Schema::Int, Schema::Int).is_subtype_of(&tup(Schema::Int, Schema::Bool)));
        // A fixed-length tuple is a subtype of a prefix-and-tail one it fits.
        assert!(
            Schema::tuple(SeqRegex::fixed([Schema::Bool, Schema::Int]))
                .is_subtype_of(&tup(Schema::Int, Schema::Int))
        );

        // The container is part of the type: a list is never a tuple, even with
        // an identical element regex.
        assert!(
            !Schema::list(SeqRegex::prefix_tail([Schema::Int], Schema::Int))
                .is_subtype_of(&tup(Schema::Int, Schema::Int))
        );
        assert!(!tup(Schema::Int, Schema::Int).is_subtype_of(&Schema::list(
            SeqRegex::prefix_tail([Schema::Int], Schema::Int)
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
            fn compare(&self, a: usize, b: usize) -> Option<Ordering> {
                Some(a.cmp(&b))
            }
        }
        let refine = |constraints: Vec<Constraint>| Schema::Refine {
            base: Box::new(Schema::Int),
            constraints,
        };
        let sub = |a: Vec<Constraint>, b: Vec<Constraint>| {
            refine(a).is_subtype_of_under(&refine(b), &ByIndex, &[])
        };
        assert!(sub(vec![Constraint::Ge(5)], vec![Constraint::Ge(0)]));
        assert!(sub(vec![Constraint::Gt(5)], vec![Constraint::Ge(0)]));
        assert!(sub(vec![Constraint::Le(0)], vec![Constraint::Le(5)]));
        assert!(sub(vec![Constraint::Lt(0)], vec![Constraint::Lt(5)]));
        assert!(sub(
            vec![Constraint::MinLen(5)],
            vec![Constraint::MinLen(2)]
        ));
        assert!(sub(
            vec![Constraint::MaxLen(2)],
            vec![Constraint::MaxLen(5)]
        ));
        // Soundness negatives.
        assert!(!sub(vec![Constraint::Ge(0)], vec![Constraint::Ge(5)]));
        assert!(!sub(vec![Constraint::Le(5)], vec![Constraint::Le(0)]));
        assert!(!sub(vec![Constraint::Ge(5)], vec![Constraint::Gt(5)]));
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
            fn compare(&self, a: usize, b: usize) -> Option<Ordering> {
                Some(a.cmp(&b))
            }
            fn no_int_between(
                &self,
                lo: usize,
                lo_strict: bool,
                hi: usize,
                hi_strict: bool,
            ) -> Option<bool> {
                let least = i64::try_from(lo).unwrap() + i64::from(lo_strict);
                let greatest = i64::try_from(hi).unwrap() - i64::from(hi_strict);
                Some(least > greatest)
            }
        }
        let refine = |base, constraints: Vec<Constraint>| Schema::Refine {
            base: Box::new(base),
            constraints,
        };
        // Gt(0) & Lt(1): the open interval (0, 1) holds no integer, so it is empty.
        assert!(
            refine(Schema::Int, vec![Constraint::Gt(0), Constraint::Lt(1)])
                .is_empty_with(&ByValue, &[])
        );
        // Gt(0) & Lt(2): the integer 1 fits, so it is not empty.
        assert!(
            !refine(Schema::Int, vec![Constraint::Gt(0), Constraint::Lt(2)])
                .is_empty_with(&ByValue, &[])
        );
        // A dense (float) base is not integer-discrete: the discreteness rule must
        // not fire, or it would unsoundly empty a populated interval.
        assert!(
            !refine(Schema::Float, vec![Constraint::Gt(0), Constraint::Lt(1)])
                .is_empty_with(&ByValue, &[])
        );
        // With no value oracle the default `no_int_between` is `None`, so even an
        // integer base stays conservative.
        assert!(!refine(Schema::Int, vec![Constraint::Gt(0), Constraint::Lt(1)]).is_empty());
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
            fn compare(&self, a: usize, b: usize) -> Option<Ordering> {
                Some(a.cmp(&b))
            }
        }
        let list = |element| Schema::list(SeqRegex::homogeneous(element));

        // Bottom-below and top-above on a non-scalar (region_set is None there, so
        // the dedicated arms decide it).
        assert!(Schema::Nothing.is_subtype_of(&list(Schema::Int)));
        assert!(list(Schema::Int).is_subtype_of(&Schema::Anything));
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
                constraints: vec![Constraint::Ge(10), Constraint::Le(0)],
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
        assert!(!refine(vec![Constraint::Ge(5), Constraint::Le(5)]).is_empty_with(&ByIndex, &[]));
        assert!(refine(vec![Constraint::Gt(5), Constraint::Lt(5)]).is_empty_with(&ByIndex, &[]));
        assert!(!refine(vec![Constraint::MinLen(5), Constraint::MaxLen(5)]).is_empty());
        // An intersection's refinement bounds are joined: both sides are needed.
        assert!(
            Schema::Intersection(vec![
                refine(vec![Constraint::Ge(5)]),
                refine(vec![Constraint::Le(0)]),
            ])
            .is_empty_with(&ByIndex, &[])
        );
        assert!(
            !Schema::Intersection(vec![
                refine(vec![Constraint::Ge(0)]),
                refine(vec![Constraint::Le(5)]),
            ])
            .is_empty_with(&ByIndex, &[])
        );

        // Keyed maps: each branch's conjunction is needed -- a depth failure is not
        // rescued by the required-coverage holding.
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        let closed = |fields| Schema::record(fields, false);
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
        assert!(
            !Schema::mapping(Schema::Str, Schema::Int)
                .is_subtype_of(&Schema::mapping(Schema::Bytes, Schema::Int))
        );
        // Mixed record-and-catch-all: required-coverage must hold there too.
        let mixed = |required| Schema::KeyedMap {
            fields: vec![field("x", Schema::Int, required)],
            defaults: vec![(Schema::Str, Schema::Int)],
        };
        assert!(!mixed(false).is_subtype_of(&mixed(true)));
        // A pure mapping is not a subtype of a mixed map that requires a field it
        // lacks: the pure-mapping branch must need both sides field-free.
        assert!(
            !Schema::mapping(Schema::Str, Schema::Int).is_subtype_of(&Schema::KeyedMap {
                fields: vec![field("x", Schema::Int, true)],
                defaults: vec![(Schema::Str, Schema::Int)],
            })
        );
        // A mixed map is not a subtype of one with an extra field whose catch-all
        // would admit an incompatible value: the mixed rule needs matching field
        // names, so its guard needs both an equal count and a name match.
        assert!(
            !Schema::KeyedMap {
                fields: vec![field("x", Schema::Int, false)],
                defaults: vec![(Schema::Str, Schema::Int)],
            }
            .is_subtype_of(&Schema::KeyedMap {
                fields: vec![
                    field("x", Schema::Int, false),
                    field("z", Schema::Bool, false),
                ],
                defaults: vec![(Schema::Str, Schema::Int)],
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
            defaults: vec![(Schema::Str, Schema::Int)],
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
        // Required extra field: undecided, a catch-all does not force presence.
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
        assert!(refine(Schema::Bool, vec![Constraint::Ge(0)]).is_subtype_of(&Schema::Int));
        // More constraints denote a smaller set: a superset of constraints (with
        // the supertype's constraints all present) is a subtype.
        assert!(
            refine(Schema::Int, vec![Constraint::Ge(0), Constraint::Le(1)])
                .is_subtype_of(&refine(Schema::Int, vec![Constraint::Ge(0)]))
        );
        // The looser refinement is not a subtype of the tighter one.
        assert!(
            !refine(Schema::Int, vec![Constraint::Ge(0)]).is_subtype_of(&refine(
                Schema::Int,
                vec![Constraint::Ge(0), Constraint::Le(1)]
            ))
        );
        // The base must still subtype: a refined int is not a str.
        assert!(!refine(Schema::Int, vec![Constraint::Ge(0)]).is_subtype_of(&Schema::Str));
        // An empty base empties the refinement; an inhabited base does not (bound
        // contradictions need value comparison and stay conservative here).
        assert!(refine(Schema::Nothing, vec![Constraint::Ge(0)]).is_empty());
        assert!(!refine(Schema::Int, vec![Constraint::Ge(0), Constraint::Le(0)]).is_empty());
    }

    #[test]
    fn reindexed_remaps_pool_and_definition_indices() {
        // Composing a validator concatenates pools and definitions: `reindexed`
        // remaps each pooled index through the intern map and offsets each `Ref`.
        let schema = Schema::Union(vec![
            Schema::Literal(0),
            Schema::Instance(1),
            Schema::Ref(0),
            Schema::Set(Box::new(Schema::Literal(1))),
        ]);
        // The second pool interned into the first: old 0 -> 5, old 1 -> 6.
        let lit_map = [5, 6];
        let remapped = schema.reindexed(&lit_map, 3);
        assert_eq!(
            remapped,
            Schema::Union(vec![
                Schema::Literal(5),
                Schema::Instance(6),
                Schema::Ref(3),
                Schema::Set(Box::new(Schema::Literal(6))),
            ])
        );

        // `shifted` is the identity-map case: every index moves by a fixed offset.
        let shifted = schema.shifted(5, 3);
        assert_eq!(
            shifted,
            Schema::Union(vec![
                Schema::Literal(5),
                Schema::Instance(6),
                Schema::Ref(3),
                Schema::Set(Box::new(Schema::Literal(6))),
            ])
        );
        // A constraint operand index is remapped too.
        let refined = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![Constraint::Ge(0)],
        };
        assert_eq!(
            refined.reindexed(&lit_map, 0),
            Schema::Refine {
                base: Box::new(Schema::Int),
                constraints: vec![Constraint::Ge(5)],
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
            refine(Schema::Int, vec![Constraint::Ge(0), Constraint::Ge(0)]).simplify(),
            refine(Schema::Int, vec![Constraint::Ge(0)])
        );
        // Constraint order does not matter: both spellings share one normal form.
        assert_eq!(
            refine(Schema::Int, vec![Constraint::Le(1), Constraint::Ge(0)]).simplify(),
            refine(Schema::Int, vec![Constraint::Ge(0), Constraint::Le(1)]).simplify()
        );
        // A refinement of a refinement flattens into one refinement over the base.
        assert_eq!(
            refine(
                refine(Schema::Int, vec![Constraint::Ge(0)]),
                vec![Constraint::Le(1)],
            )
            .simplify(),
            refine(Schema::Int, vec![Constraint::Ge(0), Constraint::Le(1)])
        );
        // The base is simplified before the refinement is rebuilt.
        assert_eq!(
            refine(
                Schema::Union(vec![Schema::Int, Schema::Int]),
                vec![Constraint::Ge(0)],
            )
            .simplify(),
            refine(Schema::Int, vec![Constraint::Ge(0)])
        );
        // Canonicalization is idempotent.
        let once = refine(
            Schema::Int,
            vec![Constraint::Le(1), Constraint::Ge(0), Constraint::Ge(0)],
        )
        .simplify();
        assert_eq!(once.clone(), once.simplify());
    }

    #[test]
    fn every_constructed_sequence_regex_is_linear() {
        // Sequences are built only with these constructors, all linear (a fixed
        // prefix then an optional repeated tail), and the structure-preserving
        // transforms map over elements without changing the regex shape. So every
        // regex that reaches the decision procedure linearizes, and the `Or` and
        // nested-`Star` forms that `regex_subtype` handles defensively are never
        // built outside tests -- its conservative fallback is unreachable from a
        // real schema, and sequence inclusion is decided for every sequence.
        assert!(SeqRegex::homogeneous(Schema::Int).linear().is_some());
        assert!(
            SeqRegex::fixed([Schema::Int, Schema::Str])
                .linear()
                .is_some()
        );
        assert!(SeqRegex::fixed(Vec::<Schema>::new()).linear().is_some());
        assert!(
            SeqRegex::prefix_tail([Schema::Str], Schema::Int)
                .linear()
                .is_some()
        );
        // The element-mapping transform preserves linearity.
        let mapped = SeqRegex::prefix_tail([Schema::Str], Schema::Int).map_elems(&|s| s.clone());
        assert!(mapped.linear().is_some());
    }

    #[test]
    fn decides_multi_clause_mapping_subtyping() {
        let map = |clauses: Vec<(Schema, Schema)>| Schema::KeyedMap {
            fields: Vec::new(),
            defaults: clauses,
        };
        // A mapping is a subtype of one with more clauses that subsume its own.
        assert!(
            map(vec![(Schema::Str, Schema::Int)]).is_subtype_of(&map(vec![
                (Schema::Str, Schema::Int),
                (Schema::Int, Schema::Bool),
            ]))
        );
        // The reverse fails: the extra int-keyed clause is not covered.
        assert!(
            !map(vec![
                (Schema::Str, Schema::Int),
                (Schema::Int, Schema::Bool)
            ])
            .is_subtype_of(&map(vec![(Schema::Str, Schema::Int)]))
        );
        // A clause is subsumed only when both key and value narrow.
        assert!(
            map(vec![(Schema::Str, Schema::Bool)])
                .is_subtype_of(&map(vec![(Schema::Str, Schema::Int)]))
        );
        assert!(
            !map(vec![(Schema::Str, Schema::Int)])
                .is_subtype_of(&map(vec![(Schema::Str, Schema::Bool)]))
        );
        // A closed record is a subtype of an open one that declares its fields.
        let closed = |fields| Schema::record(fields, false);
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        assert!(
            closed(vec![field("x", Schema::Int, true)])
                .is_subtype_of(&Schema::record(vec![field("x", Schema::Int, true)], true))
        );

        // A record mixed with a catch-all narrows field-wise and clause-wise when
        // the field names match; a widening field or value, or differing field
        // names, are not subtypes.
        let mixed = |value_field, value_default| Schema::KeyedMap {
            fields: vec![field("a", value_field, true)],
            defaults: vec![(Schema::Str, value_default)],
        };
        assert!(mixed(Schema::Bool, Schema::Bool).is_subtype_of(&mixed(Schema::Int, Schema::Int)));
        assert!(!mixed(Schema::Int, Schema::Int).is_subtype_of(&mixed(Schema::Int, Schema::Bool)));
        assert!(
            !mixed(Schema::Int, Schema::Bool).is_subtype_of(&mixed(Schema::Bool, Schema::Bool))
        );
        let mixed_b = Schema::KeyedMap {
            fields: vec![field("b", Schema::Int, true)],
            defaults: vec![(Schema::Str, Schema::Int)],
        };
        assert!(!mixed(Schema::Int, Schema::Int).is_subtype_of(&mixed_b));

        // A mixed map with an extra field is a subtype when a supertype catch-all
        // over all string keys covers that field's value.
        let with_extra = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true), field("b", Schema::Str, true)],
            defaults: vec![(Schema::Str, Schema::Bytes)],
        };
        let covering = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true)],
            defaults: vec![(Schema::Str, Schema::Anything)],
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
            defaults: vec![(Schema::Str, Schema::Int)],
        };
        let str_catch_all = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true)],
            defaults: vec![(Schema::Str, Schema::Int)],
        };
        assert!(!extra_uncovered.is_subtype_of(&str_catch_all));
        // The catch-all key must admit the field name: an int-keyed catch-all does
        // not cover a string field name even when its value would.
        let extra_str = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true), field("b", Schema::Str, true)],
            defaults: vec![(Schema::Int, Schema::Int)],
        };
        let int_catch_all = Schema::KeyedMap {
            fields: vec![field("a", Schema::Int, true)],
            defaults: vec![(Schema::Int, Schema::Anything)],
        };
        assert!(!extra_str.is_subtype_of(&int_catch_all));
        // The reverse direction -- the supertype declaring a field the subtype
        // lacks -- stays conservative.
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
            fn compare(&self, a: usize, b: usize) -> Option<Ordering> {
                Some(a.cmp(&b))
            }
        }
        let refine = |constraints| Schema::Refine {
            base: Box::new(Schema::Int),
            constraints,
        };
        // A lower bound above the upper bound is empty.
        assert!(refine(vec![Constraint::Ge(10), Constraint::Le(0)]).is_empty_with(&ByIndex, &[]));
        // Equal bounds with one strict end are empty; both closed is a singleton.
        assert!(refine(vec![Constraint::Ge(5), Constraint::Lt(5)]).is_empty_with(&ByIndex, &[]));
        assert!(!refine(vec![Constraint::Ge(5), Constraint::Le(5)]).is_empty_with(&ByIndex, &[]));
        // A satisfiable range is not empty.
        assert!(!refine(vec![Constraint::Ge(0), Constraint::Le(10)]).is_empty_with(&ByIndex, &[]));
        // A length contradiction needs no value comparison.
        assert!(refine(vec![Constraint::MinLen(5), Constraint::MaxLen(3)]).is_empty());
        // Refinements with contradictory bounds across an intersection are empty.
        let intersection = Schema::Intersection(vec![
            refine(vec![Constraint::Ge(5)]),
            refine(vec![Constraint::Lt(5)]),
        ]);
        assert!(intersection.is_empty_with(&ByIndex, &[]));
        // Without an ordering oracle the numeric bounds stay conservative.
        assert!(!refine(vec![Constraint::Ge(10), Constraint::Le(0)]).is_empty());
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
                field("next", Schema::Ref(0), true),
            ],
            defaults: Vec::new(),
        }];
        assert!(Schema::Ref(0).is_empty_under(&uninhabited));
        // t = None | {next: t} — a base case makes it inhabited.
        let inhabited = [Schema::Union(vec![
            Schema::NoneType,
            Schema::KeyedMap {
                fields: vec![field("next", Schema::Ref(0), true)],
                defaults: Vec::new(),
            },
        ])];
        assert!(!Schema::Ref(0).is_empty_under(&inhabited));
        // t = {next?: t} — an optional self-reference is inhabited by the empty map.
        let optional = [Schema::KeyedMap {
            fields: vec![field("next", Schema::Ref(0), false)],
            defaults: Vec::new(),
        }];
        assert!(!Schema::Ref(0).is_empty_under(&optional));
        // t = [t] — a list of itself is inhabited by the empty list.
        let list_of_self = [Schema::list(SeqRegex::homogeneous(Schema::Ref(0)))];
        assert!(!Schema::Ref(0).is_empty_under(&list_of_self));
        // An unresolved reference stays conservative.
        assert!(!Schema::Ref(9).is_empty_under(&uninhabited));
        // Without the definitions, recursion is not resolved (no-arg is_empty).
        assert!(!Schema::Ref(0).is_empty());
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
        assert!(not(Schema::Literal(0)).is_subtype_of(&not(Schema::Literal(0))));
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
        let identical = [list_of(Schema::Int, 0), list_of(Schema::Int, 1)];
        assert!(Schema::Ref(0).is_equivalent_under(&Schema::Ref(1), &NoLeafRelations, &identical));
        // Depth covariance through the recursion: a bool-valued list is a subtype
        // of an int-valued one (bool ⊆ int), but not the reverse.
        let covary = [list_of(Schema::Bool, 0), list_of(Schema::Int, 1)];
        assert!(Schema::Ref(0).is_subtype_of_under(&Schema::Ref(1), &NoLeafRelations, &covary));
        assert!(!Schema::Ref(1).is_subtype_of_under(&Schema::Ref(0), &NoLeafRelations, &covary));
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
            (0usize..3).prop_map(Constraint::Ge),
            (0usize..3).prop_map(Constraint::Le),
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
            (0usize..3).prop_map(Schema::Literal),
            (0usize..3).prop_map(Schema::Instance),
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
                    regex: SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(s)))),
                }),
                (inner.clone(), inner).prop_map(|(field, default)| Schema::KeyedMap {
                    fields: vec![Field {
                        name: "a".into(),
                        schema: field,
                        required: true,
                    }],
                    defaults: vec![(Schema::Str, default)],
                }),
            ]
        })
    }

    proptest! {
        #[test]
        fn scalar_decision_matches_the_value_oracle(a in scalar_schema(), b in scalar_schema()) {
            let a_empty = SAMPLES.iter().all(|&v| !member(&a, v));
            prop_assert_eq!(a.is_empty(), a_empty);

            let a_sub_b = SAMPLES.iter().all(|&v| !member(&a, v) || member(&b, v));
            let b_sub_a = SAMPLES.iter().all(|&v| !member(&b, v) || member(&a, v));
            prop_assert_eq!(a.is_subtype_of(&b), a_sub_b);
            prop_assert_eq!(a.is_equivalent(&b), a_sub_b && b_sub_a);
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
    /// complemented tree reduces well inside a generous ceiling. A regression to
    /// re-normalising each member per nesting level takes tens of seconds at this
    /// depth and trips the guard.
    #[test]
    fn simplify_stays_linear_on_a_complemented_tower() {
        let schema = complemented_tower(18);
        let start = std::time::Instant::now();
        let reduced = schema.simplify();
        let elapsed = start.elapsed();
        // The duplicate union members collapse, so the reduced form is small;
        // the point is the time it took to get there.
        assert!(matches!(reduced, Schema::Complement(_)));
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "simplify of a depth-18 complemented tower took {elapsed:?}; the \
             bottom-up pass should finish in milliseconds"
        );
    }

    /// The subtyping decision terminates promptly on a deeply nested
    /// intersection-of-unions, where the union and intersection distribution
    /// rules re-explore the schema exponentially in its depth. The work budget
    /// stops the descent and returns the conservative answer instead of running
    /// for minutes; this guards against a regression that removes the bound.
    #[test]
    fn subtyping_terminates_on_a_distributed_tower() {
        let narrow = intersection_of_unions_tower(18, Schema::Int);
        let wide = intersection_of_unions_tower(18, union(Schema::Int, Schema::Float));
        let start = std::time::Instant::now();
        // The verdict on this adversarial shape may be conservative; the property
        // under test is that the decision stops quickly rather than the answer.
        let _ = narrow.is_subtype_of(&wide);
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "is_subtype_of on a depth-18 distributed tower took {elapsed:?}; the \
             work budget should stop it promptly"
        );
    }

    /// Structural depth counts one level per nested constructor, takes the max
    /// over a node's children rather than the sum, and treats a `Ref` back edge
    /// as a leaf so a recursive schema has finite depth. The composition guard
    /// relies on this to bound the native stack every recursive walk descends.
    #[test]
    fn depth_counts_nesting_and_treats_refs_as_leaves() {
        assert_eq!(Schema::Int.depth(), 1);
        assert_eq!(Schema::Ref(0).depth(), 1);
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
        let recursive_list = Schema::list(SeqRegex::homogeneous(Schema::Ref(0)));
        assert!(recursive_list.depth() < 10);
    }

    /// The work budget must not change a verdict a real schema needs, including
    /// under recursion: a recursive list of ints is a subtype of itself and of a
    /// wider recursive list, and the wider one is not a subtype of the narrower.
    #[test]
    fn budgeted_subtyping_decides_recursive_relations() {
        let int_list = Schema::Seq {
            container: SeqKind::List,
            regex: SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(Schema::Ref(0))))),
        };
        let wide_list = Schema::Seq {
            container: SeqKind::List,
            regex: SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(Schema::Ref(1))))),
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
        // `A ⊆ ∅` reaches the (now budgeted) emptiness check.
        assert!(intersection(Schema::Int, Schema::Str).is_subtype_of(&Schema::Nothing));
        assert!(!Schema::Int.is_subtype_of(&Schema::Nothing));
        // Equivalence runs both directions against one budget and still decides.
        assert!(union(Schema::Int, Schema::Str).is_equivalent(&union(Schema::Str, Schema::Int)));
        assert!(!Schema::Int.is_equivalent(&Schema::Str));
        // An uninhabited recursive reference is empty, decided through the
        // subtype-into-bottom crossing under the shared budget.
        let defs = vec![Schema::Ref(0)];
        assert!(Schema::Ref(0).is_subtype_of_under(&Schema::Nothing, &NoLeafRelations, &defs));
    }

    /// Querying a deep schema against bottom routes through the emptiness check,
    /// which shares the subtyping budget, so it stops promptly rather than running
    /// the decision unbounded down a side door.
    #[test]
    fn deep_subtype_into_bottom_terminates() {
        let deep = intersection_of_unions_tower(18, Schema::Int);
        let start = std::time::Instant::now();
        let _ = deep.is_subtype_of(&Schema::Nothing);
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "subtype-into-bottom on a depth-18 tower took {elapsed:?}"
        );
    }

    /// The bottom-up region the emptiness pass folds from a node's children
    /// detects a *collective* cancellation — one where no single member is empty
    /// but their regions cancel — through nesting, exactly as the old whole-subtree
    /// re-walk did. Two disjoint scalar unions cancel; wrapping the result deeper
    /// must not lose that.
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
    /// intersection is decided in time linear in its size. The pre-fix quadratic
    /// re-walk took multiple seconds at this depth (≈ 9 s, growing 4× per
    /// doubling); the bottom-up fold finishes in milliseconds. Run on a large
    /// stack because the schema is intentionally left-nested to this depth.
    #[test]
    fn emptiness_decides_a_deep_intersection_in_linear_time() {
        let worker = std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(|| {
                // ¬Int ∩ ¬Str ∩ … : region-decidable, never empty (the non-scalar
                // region survives), so the walk visits every level — the worst case.
                let mut deep = Schema::Complement(Box::new(Schema::Int));
                for _ in 0..20_000 {
                    deep =
                        Schema::Intersection(vec![deep, Schema::Complement(Box::new(Schema::Str))]);
                }
                let start = std::time::Instant::now();
                let empty = deep.is_empty();
                let elapsed = start.elapsed();
                assert!(!empty);
                assert!(
                    elapsed < std::time::Duration::from_secs(2),
                    "is_empty on a depth-20000 intersection took {elapsed:?}; the \
                     bottom-up region fold should finish in milliseconds, while the \
                     quadratic re-walk takes many seconds"
                );
            })
            .expect("spawn worker thread");
        worker.join().expect("deep emptiness worker panicked");
    }

    /// A structural-attribute schema with an uninhabited required attribute is
    /// empty: no value can carry that attribute, so the dataclass-style schema
    /// denotes nothing. The symmetric keyed-map rule already held; this closes the
    /// asymmetry.
    #[test]
    fn an_uninhabited_required_attribute_empties_the_schema() {
        let empty_field = Schema::Attrs {
            class_index: 0,
            fields: vec![Field {
                name: "x".into(),
                schema: intersection(Schema::Int, Schema::Str),
                required: true,
            }],
        };
        assert!(empty_field.is_empty());
        let live_field = Schema::Attrs {
            class_index: 0,
            fields: vec![Field {
                name: "x".into(),
                schema: Schema::Int,
                required: true,
            }],
        };
        assert!(!live_field.is_empty());
    }

    /// Two attribute schemas over the same class relate by attribute width and
    /// depth: a schema carrying every attribute of the supertype with a narrower
    /// schema is a subtype. Across different classes the relation stays
    /// conservative (the nominal hierarchy is not decided in the core).
    #[test]
    fn attribute_schemas_subtype_by_width_and_depth() {
        let narrow = Schema::Attrs {
            class_index: 0,
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
        let wide = Schema::Attrs {
            class_index: 0,
            fields: vec![Field {
                name: "x".into(),
                schema: Schema::Int, // bool ⊆ int, and x is narrower; y is extra
                required: true,
            }],
        };
        assert!(narrow.is_subtype_of(&wide));
        assert!(!wide.is_subtype_of(&narrow)); // wide lacks y
        let other_class = Schema::Attrs {
            class_index: 1,
            fields: narrow_fields_clone(&narrow),
        };
        // Same fields, different class: conservative (not decided in the core).
        assert!(!narrow.is_subtype_of(&other_class));
    }

    fn narrow_fields_clone(schema: &Schema) -> Vec<Field> {
        match schema {
            Schema::Attrs { fields, .. } => fields.clone(),
            _ => unreachable!(),
        }
    }

    /// A sequence whose repeated tail is empty only under the recursive
    /// definitions matches just its fixed prefix, so it is a subtype of the bare
    /// prefix sequence. The tail's emptiness is decided with the active context,
    /// not the public no-context check, which would miss it.
    #[test]
    fn a_tail_empty_under_defs_reduces_to_the_prefix() {
        // def 0 references only itself: an uninhabited recursive schema.
        let defs = vec![Schema::Ref(0)];
        let with_phantom_tail = Schema::Seq {
            container: SeqKind::List,
            regex: SeqRegex::Cat(vec![
                SeqRegex::Elem(Box::new(Schema::Int)),
                SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(Schema::Ref(0))))),
            ]),
        };
        let just_int = Schema::Seq {
            container: SeqKind::List,
            regex: SeqRegex::Elem(Box::new(Schema::Int)),
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
        let cmp_to = |index: &usize, ok: fn(Ordering) -> bool| {
            match (as_num(value), as_num(&pool[*index])) {
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
            Constraint::MultipleOf(index) => match (as_num(value), as_num(&pool[*index])) {
                (Some(lhs), Some(rhs)) if rhs != 0.0 => lhs % rhs == 0.0,
                _ => false,
            },
            // Not generated for this oracle (opaque user code); never reached.
            Constraint::Predicate(_) | Constraint::Regex(_) => false,
        }
    }

    /// Match a sequence's items against its regex with the oracle's *own* matcher,
    /// sharing no code with the production `SeqRegex::linear`/decision under test.
    /// The generator emits exactly two shapes — `Star(Elem(s))` (a homogeneous
    /// sequence) and `Cat([Elem(a), Elem(b)])` (a fixed pair) — and `simplify`
    /// preserves the shape (it only rewrites the element schemas), so a direct
    /// encoding of those two cases is exact for everything the oracle judges.
    fn seq_matches(regex: &SeqRegex, items: &[Obj], pool: &[Obj]) -> bool {
        match regex {
            // `[s, ...]`: every element is a member of `s`; the empty sequence matches.
            SeqRegex::Star(inner) => match &**inner {
                SeqRegex::Elem(s) => items.iter().all(|item| member_full(s, item, pool)),
                other => unreachable!("oracle does not model regex {other:?}"),
            },
            // `[a, b]`: exactly two elements, matched positionally.
            SeqRegex::Cat(parts) => match parts.as_slice() {
                [SeqRegex::Elem(a), SeqRegex::Elem(b)] => {
                    items.len() == 2
                        && member_full(a, &items[0], pool)
                        && member_full(b, &items[1], pool)
                }
                other => unreachable!("oracle does not model regex {other:?}"),
            },
            other => unreachable!("oracle does not model regex {other:?}"),
        }
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
            Schema::Literal(index) => typed_eq(&pool[*index], value),
            Schema::Set(element) => match value {
                Obj::Set(items) => items.iter().all(|item| member_full(element, item, pool)),
                _ => false,
            },
            Schema::FrozenSet(element) => match value {
                Obj::FrozenSet(items) => items.iter().all(|item| member_full(element, item, pool)),
                _ => false,
            },
            Schema::Seq { container, regex } => match (container, value) {
                (SeqKind::List, Obj::List(items)) | (SeqKind::Tuple, Obj::Tuple(items)) => {
                    seq_matches(regex, items, pool)
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
                        defaults.iter().any(|(key_schema, value_schema)| {
                            member_full(key_schema, &Obj::Str(key), pool)
                                && member_full(value_schema, val, pool)
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
            (0usize..3).prop_map(Constraint::Ge),
            (0usize..3).prop_map(Constraint::Gt),
            (0usize..3).prop_map(Constraint::Le),
            (0usize..3).prop_map(Constraint::Lt),
            (0usize..4usize).prop_map(Constraint::MinLen),
            (0usize..4usize).prop_map(Constraint::MaxLen),
            (1usize..3).prop_map(Constraint::MultipleOf),
        ]
    }

    /// A generator over the non-opaque fragment the value oracle decides: scalars,
    /// pool literals, sets, linear list sequences, refinements, closed records,
    /// and their Boolean combinations.
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
            (0usize..POOL_LEN).prop_map(Schema::Literal),
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
                    fields,
                    defaults: vec![(Schema::Str, value)],
                });
            prop_oneof![
                inner.clone().prop_map(|s| Schema::Set(Box::new(s))),
                inner.clone().prop_map(|s| Schema::FrozenSet(Box::new(s))),
                inner.clone().prop_map(|s| Schema::Seq {
                    container: SeqKind::List,
                    regex: SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(s)))),
                }),
                inner.clone().prop_map(|s| Schema::Seq {
                    container: SeqKind::Tuple,
                    regex: SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(s)))),
                }),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Schema::Seq {
                    container: SeqKind::Tuple,
                    regex: SeqRegex::Cat(vec![
                        SeqRegex::Elem(Box::new(a)),
                        SeqRegex::Elem(Box::new(b)),
                    ]),
                }),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Schema::Seq {
                    container: SeqKind::List,
                    regex: SeqRegex::Cat(vec![
                        SeqRegex::Elem(Box::new(a)),
                        SeqRegex::Elem(Box::new(b)),
                    ]),
                }),
                (
                    inner.clone(),
                    proptest::collection::vec(constraint_strategy(), 1..3)
                )
                    .prop_map(|(base, constraints)| Schema::Refine {
                        base: Box::new(base),
                        constraints,
                    }),
                proptest::collection::vec(field, 0..3).prop_map(|fields| Schema::KeyedMap {
                    fields,
                    defaults: vec![],
                }),
                open_record,
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    /// The oracle's own sequence matcher (independent of `SeqRegex::linear`) and
    /// the container arms decide the shapes the generator emits: a homogeneous
    /// `Star` sequence, a fixed `Cat` pair, and the open-record `defaults` branch.
    #[test]
    fn the_value_oracle_matches_sequences_and_open_records_independently() {
        let pool = const_pool();
        // `[int, ...]`: empty and homogeneous lists match; a wrong element does not.
        let int_list = Schema::Seq {
            container: SeqKind::List,
            regex: SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(Schema::Int)))),
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
            regex: SeqRegex::Cat(vec![
                SeqRegex::Elem(Box::new(Schema::Int)),
                SeqRegex::Elem(Box::new(Schema::Str)),
            ]),
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
            defaults: vec![(Schema::Str, Schema::Int)],
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
    }
}
