#![expect(
    deprecated,
    reason = "the laws of a deprecated reducer are held until it is removed, and a law nobody checks is how a deprecation period ships a regression"
)]

use std::sync::Arc;

use crate::ir::Polarity;

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
        name: name.into(),
        schema: Schema::Int,
        required: true,
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
        Schema::Literal(ConstIx::new(0)),
        Schema::Instance(ClassIx::new(0)),
        Schema::Ref(DefIx::new(0)),
        Schema::SelfRef(0),
        Schema::list(SeqShape::homogeneous(Schema::Int)),
        Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Str])),
        Schema::list(SeqShape::prefix_tail([Schema::Int], Schema::Str)),
        Schema::list(SeqShape::fixed([])),
        Schema::set(Schema::Int),
        Schema::frozen_set(Schema::Int),
        Schema::Complement(Arc::new(Schema::Int)),
        Schema::Union(vec![Schema::Int, Schema::Str].into()),
        Schema::Intersection(vec![Schema::Int, Schema::Str].into()),
        Schema::record(vec![field("a")], Openness::Open),
        Schema::mapping(MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }),
        Schema::AttrRecord {
            fields: vec![field("a")].into(),
        },
        Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![Constraint::MinLen(1)].into(),
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
    assert_eq!(Schema::Complement(Arc::new(Schema::Int)).node_count(), 2);
    assert_eq!(Schema::set(Schema::Str).node_count(), 2);
    assert_eq!(Schema::frozen_set(Schema::Str).node_count(), 2);
    // Union counts every member, not the deepest: three members, not one.
    assert_eq!(
        Schema::Union(vec![Schema::Int, Schema::Str, Schema::Bytes].into()).node_count(),
        4
    );
    assert_eq!(
        Schema::Intersection(vec![Schema::Int, Schema::Complement(Arc::new(Schema::Str))].into())
            .node_count(),
        4
    );
    // A constraint is a node: base + one per constraint.
    assert_eq!(
        Schema::Refine {
            base: Arc::new(Schema::Str),
            constraints: vec![Constraint::MinLen(1), Constraint::MaxLen(9)].into(),
        }
        .node_count(),
        4
    );
    // The regex constructor is not itself a node; its element subtree is.
    assert_eq!(
        Schema::list(SeqShape::homogeneous(Schema::Complement(Arc::new(
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
                    schema: Schema::Complement(Arc::new(Schema::Int)),
                    required: true,
                },
                Field {
                    name: "b".into(),
                    schema: Schema::Str,
                    required: false,
                },
            ]
            .into(),
            defaults: vec![
                MapClause {
                    key: Schema::Str,
                    value: Schema::Bytes,
                },
                MapClause {
                    key: Schema::Int,
                    value: Schema::Complement(Arc::new(Schema::Str))
                },
            ]
            .into(),
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
            ]
            .into(),
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
    assert_eq!(Schema::set(Schema::Int).depth(), 2);
    assert_eq!(Schema::frozen_set(Schema::Int).depth(), 2);
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
            Schema::Complement(Arc::new(Schema::Str))
        ]))
        .depth(),
        3
    );
    assert_eq!(
        Schema::Refine {
            base: Arc::new(Schema::Str),
            constraints: vec![].into()
        }
        .depth(),
        2
    );
    assert_eq!(
        Schema::KeyedMap {
            fields: vec![Field {
                name: "a".into(),
                schema: Schema::Complement(Arc::new(Schema::Int)),
                required: true,
            }]
            .into(),
            defaults: vec![].into(),
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
            }]
            .into(),
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
        base: Arc::new(Schema::Int),
        constraints: vec![
            Constraint::Ge(OperandIx::new(1)),
            Constraint::Gt(OperandIx::new(2)),
            Constraint::Le(OperandIx::new(3)),
            Constraint::Lt(OperandIx::new(4)),
            Constraint::MultipleOf(OperandIx::new(5)),
            Constraint::Predicate(PredIx::new(8)),
            Constraint::MinLen(6),
            Constraint::MaxLen(7),
        ]
        .into(),
    };
    let Schema::Refine { constraints, .. } = refined.shifted(PoolShift::new(10), DefShift::new(0))
    else {
        panic!("shifted a Refine into a non-Refine");
    };
    assert_eq!(
        constraints.to_vec(),
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
        }]
        .into(),
    };
    let Schema::AttrRecord { fields } = record.shifted(PoolShift::new(10), DefShift::new(3)) else {
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
        Schema::Complement(Arc::new(Schema::Ref(DefIx::new(0))))
            .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    assert!(
        Schema::Refine {
            base: Arc::new(Schema::Ref(DefIx::new(0))),
            constraints: vec![Constraint::MinLen(1)].into(),
        }
        .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    assert!(
        Schema::Union(vec![Schema::Int, Schema::Ref(DefIx::new(0))].into())
            .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    assert!(
        Schema::Intersection(vec![Schema::Int, Schema::Ref(DefIx::new(0))].into())
            .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    // Nested combinators still pass it through.
    assert!(
        Schema::Complement(Arc::new(Schema::Union(
            vec![Schema::Int, Schema::Ref(DefIx::new(0))].into()
        )))
        .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    // Structural constructors guard.
    assert!(!Schema::set(Schema::Ref(DefIx::new(0))).occurs_unguarded(DefIx::new(0), Guarded::No));
    assert!(
        !Schema::frozen_set(Schema::Ref(DefIx::new(0)))
            .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    assert!(
        !Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0))))
            .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    // A guarded reference under a combinator is still guarded.
    assert!(
        !Schema::Complement(Arc::new(Schema::set(Schema::Ref(DefIx::new(0)))))
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
        path: vec![PathSegment::Key("name".into()), PathSegment::Index(2)],
        expected: "str".to_owned(),
        value_summary: "5".to_owned(),
    };
    assert_eq!(v.location(), "name[2]");
    assert!(v.to_string().starts_with("at name[2]: expected str"));
}

#[test]
fn labels_and_codes_for_every_variant() {
    let cases = [
        (Schema::ANYTHING, "anything", "anything"),
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
        (Schema::set(Schema::Int), "set", "set_type"),
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
                    name: "k".into(),
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
        path: vec![PathSegment::Key("a".into()), PathSegment::Key("b".into())],
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
            PathSegment::Key("items".into()),
            PathSegment::Index(2),
            PathSegment::Key("id".into()),
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
            name: "k".into(),
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

// THEORY: open-and-close-read-the-region
/// `close` after `open` is **not** `close`, and the projections say so.
///
/// Two closed records, `{"a"?: anything}` and `{}`, are two sets: the first
/// admits a dict carrying `a`, the second admits the empty dict alone. Both
/// open to every dict, because an optional name admitting everything says what
/// the catch-all an opening writes already says -- so `open` is not injective
/// there, and no operator on sets can recover which of the two it was handed.
///
/// The round trip holds wherever the opened term keeps its declared names,
/// which is everywhere but here, so a page promising it outright promises more
/// than the operators give. The value below is the one a law over drawn terms
/// finds, and it is the whole of the exception.
#[test]
fn closing_an_opened_record_is_not_closing_the_record() {
    let optional = |schema| {
        vec![Field {
            name: "a".into(),
            schema,
            required: false,
        }]
    };
    let free = Schema::record(optional(Schema::ANYTHING), Openness::Closed);
    let empty_closed = Schema::record(Vec::new(), Openness::Closed);
    assert_ne!(free, empty_closed);

    // Closing it is the identity: it is closed, and closing moves one region.
    assert_eq!(free.with_records_open(Openness::Closed), free);
    // Closing it *after* opening it is the empty closed record, which is the
    // price of `close` being a function of the set rather than of the term.
    let round_trip = free
        .with_records_open(Openness::Open)
        .with_records_open(Openness::Closed);
    assert_eq!(round_trip, empty_closed);

    // And the same asked of the *sets*, since two terms being unequal is not
    // yet a claim that they admit different values -- which is the whole of
    // what makes this a loss rather than a respelling.
    assert!(!free.is_equivalent(&empty_closed));
    assert!(round_trip.is_equivalent(&empty_closed));
}

// THEORY: no-negative-clause-component
/// Opening reads the labels on the semantic `dom`, which is what makes it a
/// function on sets.
///
/// A field that gives its key exactly what the record already gives every key
/// it does not name adds nothing, so two records differing only in such a
/// field are one record -- and unless it is dropped they open to different
/// ones. `{"a?": nothing}` and `{}` admit the empty dict alone; opened, both
/// must admit every dict.
#[test]
fn opening_drops_a_field_the_record_already_said() {
    let optional = |schema| {
        vec![Field {
            name: "a".into(),
            schema,
            required: false,
        }]
    };
    let empty_closed = Schema::record(Vec::new(), Openness::Closed);
    let redundant = Schema::record(optional(Schema::Nothing), Openness::Closed);
    assert_eq!(
        redundant.with_records_open(Openness::Open),
        empty_closed.with_records_open(Openness::Open)
    );

    // The dual: an open record's catch-all already frees every unnamed key,
    // which is what an optional field admitting everything says, so closing
    // it gives the empty closed record rather than one naming a free key.
    let free = Schema::record(optional(Schema::ANYTHING), Openness::Open);
    assert_eq!(free.with_records_open(Openness::Closed), empty_closed);

    // A *typed* catch-all has not said what a free field says: it frees the
    // keys of one type, and the field frees one key whatever its type. Closing
    // keeps both, and the field is what the guard above is about: the clause
    // claims its own region and says nothing of the key the field names.
    let typed_catch_all = Schema::keyed_map(
        optional(Schema::ANYTHING),
        vec![MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }],
    );
    assert_eq!(
        typed_catch_all.with_records_open(Openness::Closed),
        Schema::keyed_map(
            optional(Schema::ANYTHING),
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        )
    );

    // A field that says something is kept, whether by its type or by being
    // required at all.
    let typed = Schema::record(optional(Schema::Int), Openness::Closed);
    assert_ne!(
        typed.with_records_open(Openness::Open),
        empty_closed.with_records_open(Openness::Open)
    );
    let demanded = Schema::record(
        vec![Field {
            name: "a".into(),
            schema: Schema::Nothing,
            required: true,
        }],
        Openness::Closed,
    );
    assert_ne!(
        demanded.with_records_open(Openness::Open),
        empty_closed.with_records_open(Openness::Open)
    );
}

/// One term has one reading, so the two neighbours close alike.
///
/// `{"a?": anything, ...}` and `dict[anything, anything]` admit exactly the
/// same dicts -- every one -- and the second **is** the term
/// `KeyedMap { fields: [], defaults: [top] }`, which is also what an open
/// record with no field is. One term, and an operator that reads a spelling has
/// to pick a reading for it -- keep a mapping's clauses, drop a record's -- so
/// one of the two neighbours closes to a different set from the other.
///
/// Reading openness as the default of the region no clause claims leaves
/// nothing to pick. `[top]` claims every key, so it *is* the unclaimed region's
/// default whichever way the term was written, and closing sends that region to
/// bottom in both. Which is what makes these operations of the algebra rather
/// than term rewrites: equal sets go to equal sets.
#[test]
fn the_two_readings_of_one_term_close_to_one_set() {
    let free = Schema::record(
        vec![Field {
            name: "a".into(),
            schema: Schema::ANYTHING,
            required: false,
        }],
        Openness::Open,
    );
    let every_dict = Schema::mapping(MapClause::top());
    // One term, two readings.
    assert_eq!(Schema::record(Vec::new(), Openness::Open), every_dict);
    // And one answer, because the reading the operators take is the same one.
    assert_eq!(
        free.with_records_open(Openness::Closed),
        every_dict.with_records_open(Openness::Closed)
    );
    assert_eq!(
        every_dict.with_records_open(Openness::Closed),
        Schema::record(Vec::new(), Openness::Closed),
        "every key refused is the empty closed record"
    );
}

// THEORY: open-and-close-read-the-region
/// Openness is the default of the region no clause claims.
///
/// A clause is a key-type region carrying its own default, so the operators
/// decide the keys the clauses leave over and nothing else. A record claims no
/// region, which is why opening one frees every key; a mapping claims one, so
/// opening it keeps that region and frees the rest.
#[test]
fn opening_a_mapping_frees_the_region_no_clause_claims() {
    let claimed = MapClause {
        key: Schema::Str,
        value: Schema::Int,
    };
    let mapping = Schema::mapping(claimed.clone());

    // Closed already: the `str` region is claimed and every other key-type is
    // refused, which is what the operator would write.
    assert_eq!(mapping.with_records_open(Openness::Closed), mapping);

    let Schema::KeyedMap { fields, defaults } = mapping.with_records_open(Openness::Open) else {
        panic!("a mapping opened into a non-map");
    };
    assert!(fields.is_empty(), "opening declares no name");
    assert_eq!(
        defaults.to_vec(),
        vec![
            claimed,
            MapClause {
                key: Schema::Complement(Arc::new(Schema::Str)),
                value: Schema::ANYTHING,
            },
        ],
        "the claimed region is kept and the rest is freed"
    );

    // And the two are inverse, on a mapping as on a record.
    assert_eq!(
        mapping
            .with_records_open(Openness::Open)
            .with_records_open(Openness::Closed),
        mapping
    );

    let empty_closed = Schema::record(Vec::new(), Openness::Closed);
    assert!(record_is_open(
        &empty_closed.with_records_open(Openness::Open)
    ));
}

// THEORY: no-negative-clause-component
/// What the constructors make canonical, and the one thing they do not.
///
/// ICFP Definition 2.2: a record is a quasi-constant function, and `dom(r)` is
/// `{l | r(l) != z}` for the default `z` -- **semantic**, so a label written
/// with the default as its type is not in the domain. A closed record gives
/// every key it does not name nothing at all, so `{"a?": nothing}` names a key
/// and says what the record already said: one domain, one record.
///
/// The constructors canonicalise the *order* -- the fields and the clauses are
/// sorted, so two terms differing only in how they were written are one term --
/// and they do not canonicalise the domain. That is deviation 3, and this is
/// its extent rather than its cost: `==` is the written form, and the two
/// spellings above are two terms and one set.
///
/// The domain is read where the paper's operators want it. `with_records_open`
/// drops a field the default already gives before it rewrites anything, which
/// is what keeps `open` a function of the set on this pair -- so the gap is in
/// the constructor and not in the operator.
#[test]
fn two_spellings_of_one_keyed_map_are_one_term() {
    let named = |name: &str, schema: Schema, required: bool| Field {
        name: name.into(),
        schema,
        required,
    };

    // The order is canonical, both ways round.
    assert_eq!(
        Schema::record(
            vec![named("a", Schema::Int, true), named("b", Schema::Str, true)],
            Openness::Closed
        ),
        Schema::record(
            vec![named("b", Schema::Str, true), named("a", Schema::Int, true)],
            Openness::Closed
        ),
        "two field lists differing only in order are one term"
    );
    let str_to_int = MapClause {
        key: Schema::Str,
        value: Schema::Int,
    };
    let int_to_str = MapClause {
        key: Schema::Int,
        value: Schema::Str,
    };
    assert_eq!(
        Schema::keyed_map(Vec::new(), vec![str_to_int.clone(), int_to_str.clone()]),
        Schema::keyed_map(Vec::new(), vec![int_to_str, str_to_int]),
        "and two clause lists differing only in order"
    );

    // The domain is not. A key named with what the closed record already gives
    // every key it does not name is outside `dom`, and the two are two terms.
    let empty = Schema::record(Vec::new(), Openness::Closed);
    let redundant = Schema::record(vec![named("a", Schema::Nothing, false)], Openness::Closed);
    assert_ne!(
        empty, redundant,
        "deviation 3: the constructor reads the field list as written"
    );

    // And the operators read it, which is what makes them functions of the set
    // where `==` is not.
    assert_eq!(
        empty.with_records_open(Openness::Open),
        redundant.with_records_open(Openness::Open),
        "the domain is read before the rewrite"
    );
    assert_eq!(
        empty.with_records_open(Openness::Closed),
        redundant.with_records_open(Openness::Closed),
    );
}

// THEORY: open-and-close-read-the-region
/// Opening a record that already claims a region leaves one clause, not two.
///
/// A `TypedDict` builds exactly this: named fields, and `str => anything` for
/// the keys it does not name. Freeing the key-types that clause leaves over
/// gives two clauses carrying one value between them, and two clauses with one
/// value are one clause over the union of their keys -- which is the whole key
/// space, so it is the catch-all a record opened has always had.
///
/// Spelling it as two costs the pair: a clause keyed by a complement is a shape
/// the descriptor's map lowering declines, so the same set written the long way
/// is decided one way and not the other. The merge is what keeps `open` inside
/// the decided fragment.
#[test]
fn opening_a_record_that_claims_a_region_leaves_one_clause() {
    let claimed = MapClause {
        key: Schema::Str,
        value: Schema::ANYTHING,
    };
    let typed_dict = Schema::KeyedMap {
        fields: Arc::from([Field {
            name: "a".into(),
            schema: Schema::Int,
            required: true,
        }]),
        defaults: vec![claimed].into(),
    };

    let Schema::KeyedMap { fields, defaults } = typed_dict.with_records_open(Openness::Open) else {
        panic!("a record opened into a non-map");
    };
    assert_eq!(fields.len(), 1, "the named field is untouched");
    assert_eq!(
        defaults.to_vec(),
        vec![MapClause::top()],
        "one value over every key is one clause"
    );
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
            name: "a".into(),
            schema: Schema::Int,
            required: true,
        }],
        Openness::Closed,
    );
    let pair = Schema::Union(
        vec![
            closed.clone(),
            Schema::Complement(Arc::new(closed.with_records_open(Openness::Open))),
        ]
        .into(),
    );

    assert!(
        matches!(pair, Schema::Union(_)),
        "the two differ before the transform"
    );
    assert_eq!(
        pair.with_records_open(Openness::Open),
        Schema::ANYTHING,
        "and are one schema and its complement after it"
    );
}

// THEORY: open-and-close-read-the-region
#[test]
fn with_records_open_keeps_the_region_a_mapping_claims() {
    // A clause's own region is not the operator's to touch: closing sends the
    // key-types *no* clause claims to bottom, and the paper's own example keeps
    // `String => Bool` through exactly that move. So a mapping closed is the
    // mapping, and a guard that skips it on `!fields.is_empty()` reads "has a
    // name" for "is a record".
    let mapping = Schema::KeyedMap {
        fields: Arc::from([]),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }]
        .into(),
    };
    let Schema::KeyedMap { fields, defaults } = mapping.with_records_open(Openness::Closed) else {
        panic!("a mapping closed into a non-map");
    };
    assert!(fields.is_empty());
    assert_eq!(
        defaults.to_vec(),
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
    assert_eq!(Schema::meet([]), Schema::ANYTHING);
    // A non-empty list arrives in the lattice normal form: idempotence and
    // the identities are settled where the schema is built, and one member
    // is returned unwrapped rather than as a join of one.
    assert_eq!(Schema::union([Schema::Int, Schema::Int]), Schema::Int);
    assert_eq!(Schema::meet([Schema::Int]), Schema::Int);
    assert_eq!(Schema::union([Schema::Int, Schema::Nothing]), Schema::Int);
    assert_eq!(Schema::meet([Schema::Int, Schema::ANYTHING]), Schema::Int);
    // Commutativity: two spellings of one join are one schema.
    assert_eq!(
        Schema::union([Schema::Str, Schema::Int]),
        Schema::union([Schema::Int, Schema::Str])
    );
    // Associativity: a nested join is flattened into the one above it.
    assert_eq!(
        Schema::union([Schema::Int, Schema::union([Schema::Str, Schema::Bytes])]),
        Schema::union([Schema::Int, Schema::Str, Schema::Bytes])
    );
    assert_eq!(
        Schema::Int.complement(),
        Schema::Complement(Arc::new(Schema::Int))
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
                | Schema::Coll { .. }
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

// THEORY: guarded-recursion
#[test]
fn contractivity_requires_a_structural_guard() {
    assert!(
        !Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0))))
            .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    assert!(Schema::Ref(DefIx::new(0)).occurs_unguarded(DefIx::new(0), Guarded::No));
    assert!(
        Schema::Union(vec![Schema::Int, Schema::Ref(DefIx::new(0))].into())
            .occurs_unguarded(DefIx::new(0), Guarded::No)
    );
    assert!(
        !Schema::list(SeqShape::homogeneous(Schema::Union(
            vec![Schema::Int, Schema::Ref(DefIx::new(0))].into()
        )))
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
    let schema = Schema::Union(
        vec![
            Schema::Literal(ConstIx::new(0)),
            Schema::Instance(ClassIx::new(1)),
            Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![Constraint::Ge(OperandIx::new(0)), Constraint::MinLen(2)].into(),
            },
            Schema::Ref(DefIx::new(0)),
        ]
        .into(),
    );
    let reindexed = schema.reindexed(&[10, 11], DefShift::new(5));
    assert_eq!(
        reindexed,
        Schema::Union(
            vec![
                Schema::Literal(ConstIx::new(10)),  // 0 -> table[0] = 10
                Schema::Instance(ClassIx::new(11)), // 1 -> table[1] = 11
                Schema::Refine {
                    base: Arc::new(Schema::Int),
                    // Ge index remaps through the table; MinLen is a length, untouched.
                    constraints: vec![Constraint::Ge(OperandIx::new(10)), Constraint::MinLen(2)]
                        .into(),
                },
                Schema::Ref(DefIx::new(5)), // ref offset by def_offset = 5
            ]
            .into()
        )
    );
}

#[test]
fn refine_delegates_label_and_code_to_its_base() {
    let refined = Schema::Refine {
        base: Arc::new(Schema::Str),
        constraints: vec![Constraint::MinLen(1)].into(),
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
            Schema::set(Schema::SelfRef(9)),
        )),
    ]);
    assert!(buried.has_escaped_self_ref(open_none));
    assert!(!buried.has_escaped_self_ref(open_all));
}

#[test]
fn field_is_cloneable_and_carries_its_flag() {
    let field = Field {
        name: "n".into(),
        schema: Schema::Int,
        required: false,
    };
    let copy = field.clone();
    assert_eq!(&*copy.name, "n");
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
    assert_eq!(fixed.prefix.to_vec(), vec![Schema::Int, Schema::Str]);
    assert!(fixed.tail.is_none());

    let prefixed = SeqShape::prefix_tail([Schema::Str], Schema::Int);
    assert_eq!(prefixed.prefix.to_vec(), vec![Schema::Str]);
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
    assert_eq!(shape.prefix.to_vec(), vec![Schema::Ref(DefIx::new(5))]);

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
            name: "f".into(),
            schema: Schema::Ref(DefIx::new(0)),
            required: true,
        }]
        .into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::SelfRef(7),
        }]
        .into(),
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
    Schema::Complement(Arc::new(s))
}

#[test]
fn simplify_decides_the_complement_laws() {
    // X ∩ ¬X = ⊥ and X ∪ ¬X = ⊤.
    assert_eq!(
        Schema::Intersection(vec![Schema::Int, not(Schema::Int)].into()).simplify(),
        Schema::Nothing
    );
    assert_eq!(
        Schema::Union(vec![Schema::Int, not(Schema::Int)].into()).simplify(),
        Schema::ANYTHING
    );
    // The law is the complementary pair itself, not scalar-region coverage: an
    // opaque member has no region, so `X ∪ ¬X` here is decided only by finding
    // the pair, with the whole universe left unaccounted for by the bitset.
    let opaque = Schema::list(SeqShape::homogeneous(Schema::Int));
    assert_eq!(
        Schema::Union(vec![opaque.clone(), not(opaque)].into()).simplify(),
        Schema::ANYTHING
    );
    // Disjoint basics and disjoint container kinds give an empty intersection.
    assert_eq!(
        Schema::Intersection(vec![Schema::Int, Schema::Str].into()).simplify(),
        Schema::Nothing
    );
    assert_eq!(
        Schema::Intersection(
            vec![
                Schema::list(SeqShape::homogeneous(Schema::Int)),
                Schema::set(Schema::Int),
            ]
            .into()
        )
        .simplify(),
        Schema::Nothing
    );
    // bool ⊆ int, so their intersection is not empty.
    assert_ne!(
        Schema::Intersection(vec![Schema::Bool, Schema::Int].into()).simplify(),
        Schema::Nothing
    );
}

// THEORY: any-is-the-top-spelled
/// Every relation answers the same for both spellings of the top, wherever
/// the top sits in a schema.
///
/// The equality, ordering and hash impls make this hold of any rule keyed on
/// them, which is every rule here; what they cannot stop is a rule that
/// matches the payload directly, and this is what would catch one.
#[test]
fn no_relation_can_tell_the_two_spellings_apart() {
    let around = |top: Schema| {
        [
            top.clone(),
            Schema::Union(vec![top.clone(), Schema::Int].into()),
            Schema::Intersection(vec![top.clone(), Schema::Str].into()),
            Schema::Complement(Arc::new(top.clone())),
            Schema::list(SeqShape::homogeneous(top.clone())),
            Schema::set(top),
        ]
    };
    let others = [
        Schema::ANYTHING,
        Schema::Nothing,
        Schema::Int,
        Schema::Str,
        Schema::list(SeqShape::homogeneous(Schema::Int)),
    ];
    // Equality, ordering and hashing agree, which is what every rule that
    // keys on one of them relies on.
    let hash_of = |schema: &Schema| {
        use core::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        schema.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(Spelling::Any, Spelling::Top);
    assert_eq!(
        Spelling::Any.cmp(&Spelling::Top),
        core::cmp::Ordering::Equal
    );
    assert_eq!(
        Spelling::Any.partial_cmp(&Spelling::Top),
        Some(core::cmp::Ordering::Equal)
    );
    assert_eq!(
        Schema::ANY.cmp(&Schema::ANYTHING),
        core::cmp::Ordering::Equal
    );
    assert_eq!(hash_of(&Schema::ANY), hash_of(&Schema::ANYTHING));

    for (any, top) in around(Schema::ANY)
        .into_iter()
        .zip(around(Schema::ANYTHING))
    {
        assert_eq!(any, top, "the spellings are one schema");
        assert_eq!(any.simplify(), top.simplify());
        assert_eq!(any.is_empty(), top.is_empty(), "{any:?}");
        for other in &others {
            assert_eq!(
                any.is_subtype_of(other),
                top.is_subtype_of(other),
                "{any:?}"
            );
            assert_eq!(
                other.is_subtype_of(&any),
                other.is_subtype_of(&top),
                "{any:?}"
            );
            assert_eq!(any.disjoint(other), top.disjoint(other), "{any:?}");
        }
    }
}

// THEORY: any-is-the-top-spelled
/// `Any` is the top, spelled, so the complement laws hold of it: it is the
/// same node as `anything`, and a rule cannot tell the two apart because the
/// spelling compares equal. What survives is the spelling itself, which
/// `simplify` carries through an identity so `repr` still gives back what
/// the user wrote.
#[test]
fn the_complement_laws_hold_of_the_top_however_it_is_spelled() {
    assert_eq!(Schema::ANY, Schema::ANYTHING);
    for top in [Schema::ANY, Schema::ANYTHING] {
        assert_eq!(
            Schema::Intersection(vec![top.clone(), not(top.clone())].into()).simplify(),
            Schema::Nothing
        );
        assert_eq!(
            Schema::Union(vec![top.clone(), not(top.clone())].into()).simplify(),
            Schema::ANYTHING
        );
        assert_eq!(not(top.clone()).simplify(), Schema::Nothing);
    }
    // Mixed spellings are one set, so the law fires across them too.
    assert_eq!(
        Schema::Intersection(vec![Schema::ANY, not(Schema::ANYTHING)].into()).simplify(),
        Schema::Nothing
    );
    // And the spelling survives an identity, which is what `repr` reads.
    assert!(matches!(
        Schema::Union(vec![Schema::ANY, Schema::Int].into()).simplify(),
        Schema::Anything(Spelling::Any)
    ));
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
    assert!(Schema::frozen_set(Schema::Int).disjoint(&Schema::set(Schema::Int)));
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
    // A refinement is disjoint exactly when its base is.
    let refined = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
    };
    assert!(refined.disjoint(&Schema::Str));
    assert!(!refined.disjoint(&Schema::Int));
}

/// Unfolding reads the body while it has unfoldings and puts the bound where
/// the reference stood once it has none -- the top in a positive position, the
/// bottom in a negative one, and a complement is what flips between them.
#[test]
fn unfolding_reads_the_body_then_stands_in_the_bound_the_position_makes_sound() {
    let defs = vec![Schema::Int];
    let reference = Schema::Ref(DefIx::new(0));

    assert_eq!(reference.unfolded(&defs, 1, Polarity::Widen), Schema::Int);
    assert_eq!(reference.unfolded(&defs, 1, Polarity::Narrow), Schema::Int);
    assert_eq!(
        reference.unfolded(&defs, 0, Polarity::Widen),
        Schema::ANYTHING
    );
    assert_eq!(
        reference.unfolded(&defs, 0, Polarity::Narrow),
        Schema::Nothing
    );

    // Under a complement the polarity flips, so the cut inside is the other
    // bound: `~Ref` in a positive position over-approximates as `~nothing`.
    let negated = Schema::Complement(Arc::new(reference.clone()));
    assert_eq!(
        negated.unfolded(&defs, 0, Polarity::Widen),
        Schema::Complement(Arc::new(Schema::Nothing))
    );
    assert_eq!(
        negated.unfolded(&defs, 0, Polarity::Narrow),
        Schema::Complement(Arc::new(Schema::ANYTHING))
    );

    // A schema with no reference comes back as it stands.
    assert_eq!(Schema::Int.unfolded(&defs, 1, Polarity::Widen), Schema::Int);
}

/// A reference is found under every variant that holds a child, so the caller
/// that skips the rebuild for reference-free schemas skips none that need it.
#[test]
fn a_reference_is_found_under_every_child_holding_variant() {
    let reference = Schema::Ref(DefIx::new(0));
    let field = |schema: Schema| Field {
        name: "a".into(),
        schema,
        required: true,
    };
    let holders = [
        Schema::Union(vec![Schema::Int, reference.clone()].into()),
        Schema::Intersection(vec![Schema::Int, reference.clone()].into()),
        Schema::Complement(Arc::new(reference.clone())),
        Schema::set(reference.clone()),
        Schema::list(SeqShape::fixed([Schema::Int, reference.clone()])),
        Schema::list(SeqShape::prefix_tail([Schema::Int], reference.clone())),
        Schema::record(vec![field(reference.clone())], Openness::Closed),
        Schema::mapping(MapClause {
            key: reference.clone(),
            value: Schema::Int,
        }),
        Schema::mapping(MapClause {
            key: Schema::Str,
            value: reference.clone(),
        }),
        Schema::AttrRecord {
            fields: vec![field(reference.clone())].into(),
        },
        Schema::Refine {
            base: Arc::new(reference.clone()),
            constraints: vec![Constraint::MinLen(1)].into(),
        },
    ];
    for holder in &holders {
        assert!(holder.has_reference(), "{holder:?} holds a reference");
    }
    assert!(Schema::SelfRef(0).has_reference());
    assert!(!Schema::Int.has_reference());
    assert!(!Schema::list(SeqShape::homogeneous(Schema::Int)).has_reference());
}

/// Pruning keeps every definition the schema reaches -- through other
/// definitions included -- and renumbers the survivors in their old order.
#[test]
fn pruning_keeps_what_is_reached_and_renumbers_the_rest_in_order() {
    let reference = |index: usize| Schema::Ref(DefIx::new(index));

    // Everything reached: untouched.
    let (schema, defs) = pruned(reference(0), vec![Schema::Int]);
    assert_eq!(schema, reference(0));
    assert_eq!(defs, vec![Schema::Int]);

    // The middle definition is unreachable; the last one moves up by one and
    // the references to it move with it.
    let (schema, defs) = pruned(
        Schema::Union(vec![reference(0), reference(2)].into()),
        vec![Schema::Int, Schema::Str, Schema::Bytes],
    );
    assert_eq!(
        schema,
        Schema::Union(vec![reference(0), reference(1)].into())
    );
    assert_eq!(defs, vec![Schema::Int, Schema::Bytes]);

    // Reached through a definition's body rather than the schema itself.
    let (schema, defs) = pruned(
        reference(1),
        vec![
            Schema::Str,
            Schema::Complement(Arc::new(reference(2))),
            Schema::Int,
        ],
    );
    assert_eq!(schema, reference(0));
    assert_eq!(
        defs,
        vec![Schema::Complement(Arc::new(reference(1))), Schema::Int]
    );
}

/// Two spellings of one record, and of one map, are one term: fields are
/// ordered by name and clauses by their own order, with a repeat dropped.
#[test]
fn a_record_and_a_map_are_one_term_however_their_parts_are_written() {
    let field = |name: &str, schema: Schema| Field {
        name: name.into(),
        schema,
        required: true,
    };
    assert_eq!(
        Schema::record(
            vec![field("b", Schema::Str), field("a", Schema::Int)],
            Openness::Closed
        ),
        Schema::record(
            vec![field("a", Schema::Int), field("b", Schema::Str)],
            Openness::Closed
        )
    );
    assert_eq!(
        Schema::attr_record(vec![field("b", Schema::Str), field("a", Schema::Int)]),
        Schema::attr_record(vec![field("a", Schema::Int), field("b", Schema::Str)])
    );

    let clause = |key: Schema, value: Schema| MapClause { key, value };
    let one_way = Schema::keyed_map(
        vec![],
        vec![
            clause(Schema::Str, Schema::Int),
            clause(Schema::Int, Schema::Bool),
        ],
    );
    let other_way = Schema::keyed_map(
        vec![],
        vec![
            clause(Schema::Int, Schema::Bool),
            clause(Schema::Str, Schema::Int),
        ],
    );
    let repeated = Schema::keyed_map(
        vec![],
        vec![
            clause(Schema::Int, Schema::Bool),
            clause(Schema::Str, Schema::Int),
            clause(Schema::Int, Schema::Bool),
        ],
    );
    assert_eq!(one_way, other_way);
    assert_eq!(one_way, repeated);
    // And the order is a real one: the two clauses are still two.
    let Schema::KeyedMap { defaults, .. } = &one_way else {
        unreachable!()
    };
    assert_eq!(defaults.len(), 2);
}

/// A record that reopens drops a field its new catch-all already says, and
/// keeps the rest.
///
/// An optional field of the top under an open record is the catch-all written
/// twice: the clause admits the key with any value, and the field says the same
/// thing about that one key. The transform drops it -- and that is the one case
/// where it has to assemble the field list again rather than hand back the one
/// it mapped, which is why the two paths are asked here together.
#[test]
fn reopening_a_record_drops_the_field_its_catch_all_already_says() {
    let field = |name: &str, schema: Schema, required: bool| Field {
        name: name.into(),
        schema,
        required,
    };
    let open = Schema::record(
        vec![
            field("kept", Schema::Int, true),
            field("said", Schema::ANYTHING, false),
        ],
        Openness::Open,
    );
    let Schema::KeyedMap { fields, defaults } = open.with_records_open(Openness::Open) else {
        panic!("a record opens to a record")
    };
    assert_eq!(
        fields.iter().map(|f| &*f.name).collect::<Vec<_>>(),
        ["kept"],
        "the optional top field is what the catch-all says, so it is dropped"
    );
    assert_eq!(defaults.to_vec(), vec![MapClause::top()]);

    // And the field that says something the catch-all does not is kept, which
    // is the path that returns the mapped list rather than rebuilding it.
    let closed = Schema::record(vec![field("kept", Schema::Int, false)], Openness::Closed);
    let Schema::KeyedMap { fields, .. } = closed.with_records_open(Openness::Open) else {
        panic!("a record opens to a record")
    };
    assert_eq!(
        fields.iter().map(|f| &*f.name).collect::<Vec<_>>(),
        ["kept"]
    );
}

/// A mapping keeps the clause it was built with, and only the catch-all clause
/// is the shared one.
///
/// The empty clause list and the open record's `(anything, anything)` are the
/// same lists over and over, so both are allocated once and shared. A clause
/// that is neither is the map's own, and reading it as the catch-all would open
/// a mapping that names a narrower key or value.
#[test]
fn a_map_keeps_a_clause_that_is_not_the_catch_all() {
    let narrow = MapClause {
        key: Schema::Str,
        value: Schema::Int,
    };
    // Through the general constructor, which is where the two shared lists are
    // recognised: a clause that is neither empty nor the catch-all is the map's
    // own, and reading it as the catch-all would open a map that names a
    // narrower key or value.
    let Schema::KeyedMap { defaults, .. } = Schema::keyed_map(Vec::new(), vec![narrow.clone()])
    else {
        panic!("a keyed map is a keyed map")
    };
    assert_eq!(defaults.to_vec(), vec![narrow.clone()]);
    // The catch-all itself still reads as the catch-all, and the empty list as
    // the empty one.
    let Schema::KeyedMap { defaults, .. } = Schema::keyed_map(Vec::new(), vec![MapClause::top()])
    else {
        panic!("a keyed map is a keyed map")
    };
    assert_eq!(defaults.to_vec(), vec![MapClause::top()]);
    let Schema::KeyedMap { defaults, .. } = Schema::keyed_map(Vec::new(), Vec::new()) else {
        panic!("a keyed map is a keyed map")
    };
    assert!(defaults.is_empty());
    // The catch-all beside another clause stays a list of two: it is the *lone*
    // catch-all that is the shared list, not every list that begins with one. A
    // map that lost the second clause would print a set it does not denote,
    // since a clause list is what `repr` reads.
    let Schema::KeyedMap { defaults, .. } =
        Schema::keyed_map(Vec::new(), vec![MapClause::top(), narrow.clone()])
    else {
        panic!("a keyed map is a keyed map")
    };
    assert_eq!(defaults.len(), 2);
}
