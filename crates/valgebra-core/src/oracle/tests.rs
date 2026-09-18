//! What the shared laws decide, and where each declines.
//!
//! The two laws below `oracle.rs` are read by three callers -- the
//! constructors in `ir.rs`, the reducer in `simplify.rs` and the emptiness
//! decision -- so their tests sit with the law rather than with any one of
//! them. A test in a caller's module holds the caller; these hold the law.

use std::sync::Arc;

use super::*;
use crate::descr::lower::Constants;
use crate::ir::{DefIx, Field, MapClause, PredIx, SeqShape};

/// A callback hides behind every container, and the walk that looks for one
/// must enter each.
///
/// `A ∩ ¬A = ∅` is declined for an atom that is not a set, and a predicate
/// nested inside a container makes the whole container one -- so each way a
/// schema holds another is a way the search must descend.
#[test]
fn a_callback_is_found_through_every_container() {
    let predicate = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
    };
    let field = |schema: Schema| Field {
        name: "x".into(),
        schema,
        required: true,
    };
    let wrappers: [Schema; 11] = [
        predicate.clone(),
        Schema::set(predicate.clone()),
        Schema::frozen_set(predicate.clone()),
        Schema::Complement(Arc::new(predicate.clone())),
        Schema::union([predicate.clone(), Schema::Int]),
        Schema::list(SeqShape::fixed([predicate.clone()])),
        Schema::list(SeqShape::homogeneous(predicate.clone())),
        Schema::KeyedMap {
            fields: vec![field(predicate.clone())].into(),
            defaults: Vec::new().into(),
        },
        Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key: Schema::Str,
                value: predicate.clone(),
            }]
            .into(),
        },
        Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key: predicate.clone(),
                value: Schema::Str,
            }]
            .into(),
        },
        Schema::AttrRecord {
            fields: vec![field(predicate.clone())].into(),
        },
    ];
    for wrapper in wrappers {
        assert!(
            !denotes_a_set_within(&wrapper, &NoLeafRelations, &[]),
            "a predicate inside {wrapper:?} is still a predicate"
        );
        let meet = Schema::Intersection(
            vec![
                wrapper.clone(),
                Schema::Complement(Arc::new(wrapper.clone())),
            ]
            .into(),
        );
        assert!(
            !meet.is_empty_under(&[]),
            "so the law must decline {wrapper:?}"
        );
    }

    // Without one, the same shapes are sets and the law still decides.
    let plain = Schema::set(Schema::Int);
    assert!(denotes_a_set_within(&plain, &NoLeafRelations, &[]));
    assert!(
        Schema::Intersection(vec![plain.clone(), Schema::Complement(Arc::new(plain))].into())
            .is_empty_under(&[])
    );
}

/// An oracle that reports every atom it is asked about as a set, standing
/// for the bindings' answer about a pure class.
///
/// Defined here because it is a double for *this* trait: every test that needs
/// a core which can answer about a class needs this one, and a second copy
/// beside the decision tests would be two doubles that could drift apart.
pub(crate) struct Pure;

impl Constants for Pure {}

impl LeafRelations for Pure {
    fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
        None
    }

    fn atom_denotes_a_set(&self, _atom: &Schema) -> Option<bool> {
        Some(true)
    }
}

/// A reference is refused by the fold whatever it stands for. The law asks
/// `A` twice, and a back edge is not a set until it is unfolded -- which the
/// syntactic fold does not do, and which a self-reference has not yet been
/// resolved enough to allow.
#[test]
fn a_reference_is_not_a_set_the_fold_may_cancel() {
    for reference in [Schema::Ref(DefIx::new(0)), Schema::SelfRef(0)] {
        assert!(
            !denotes_a_set_within(&reference, &NoLeafRelations, &[]),
            "{reference:?}"
        );
        // Nested, so a reference reached through a constructor refuses too.
        let nested = Schema::set(reference.clone());
        assert!(!denotes_a_set_within(&nested, &NoLeafRelations, &[]));
        // And the constructors decline the two cancelling laws for it.
        let not = |s: Schema| Schema::Complement(Arc::new(s));
        assert_ne!(
            Schema::union([reference.clone(), not(reference.clone())]),
            Schema::ANYTHING
        );
        assert_ne!(
            Schema::meet([reference.clone(), not(reference)]),
            Schema::Nothing
        );
    }
}

/// A class is referred to the oracle, and a core with none declines it.
///
/// The default answers nothing, which is what makes an unwired core
/// conservative rather than wrong: `Instance ∩ ¬Instance` is not decided
/// empty until the bindings say the class is pure.
#[test]
fn a_class_without_an_oracle_is_not_a_set() {
    let class = Schema::Instance(ClassIx::new(0));
    assert_eq!(NoLeafRelations.atom_denotes_a_set(&class), None);
    assert!(!denotes_a_set_within(&class, &NoLeafRelations, &[]));

    assert!(denotes_a_set_within(&class, &Pure, &[]));
}

/// The default oracle answers no question, which is what makes it sound.
///
/// `NoLeafRelations` is the core's own, and every question it declines is one
/// the rules have to stay conservative about. A default that answered would be
/// a claim about values the core cannot see -- which class holds an instance,
/// how two constants compare, whether one step divides another -- so the
/// declining is the contract rather than an omission.
///
/// Held for each question, because a new one is added by writing a default and
/// a body that answers it, and the body is the half a reviewer reads.
#[test]
fn the_default_oracle_declines_every_question() {
    use crate::ir::{ClassIx, ConstIx};

    let oracle = NoLeafRelations;
    let one = OperandIx::new(0);
    let two = OperandIx::new(1);

    assert_eq!(oracle.leaf_subtype(&Schema::Int, &Schema::Int), None);
    assert_eq!(oracle.compare(one, two), None);
    assert_eq!(oracle.divides(one, two), None);
    assert_eq!(oracle.no_int_between(one, true, two, true), None);
    assert_eq!(oracle.atom_denotes_a_set(&Schema::Int), None);
    assert_eq!(
        oracle.literal_sets_disjoint(&[ConstIx::new(0)], &[ConstIx::new(1)]),
        None
    );
    assert_eq!(
        oracle.literals_disjoint(ConstIx::new(0), ConstIx::new(1)),
        None
    );
    assert_eq!(oracle.literal_kind(ConstIx::new(0)), None);
    assert_eq!(oracle.class_admits_kind(ClassIx::new(0), Kind::Int), None);
    assert_eq!(
        oracle.direct_instance_of_kind(ClassIx::new(0), Kind::Int),
        None
    );
    assert_eq!(oracle.kind_derives_from(Kind::Int, ClassIx::new(0)), None);
}
