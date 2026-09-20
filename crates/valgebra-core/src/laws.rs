#![expect(
    deprecated,
    reason = "the laws of a deprecated reducer are held until it is removed, and a law nobody checks is how a deprecation period ships a regression"
)]

use std::sync::Arc;

use super::*;
use crate::decision::{DECISION_BUDGET, LeafRelations, NoLeafRelations};
use crate::descr::classes::Class;
use crate::descr::lower::{Constants, Operand};
use crate::ir::Polarity;
use crate::kind::Kind;
use crate::verdict::Relation;
use proptest::prelude::*;

/// A small schema generator: atoms combined by union, intersection, and
/// complement. Pool indices are arbitrary but consistent across a value.
fn schema() -> impl Strategy<Value = Schema> {
    let atom = prop_oneof![
        Just(Schema::ANYTHING),
        Just(Schema::Nothing),
        Just(Schema::ANY),
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
            proptest::collection::vec(inner.clone(), 1..4).prop_map(|m| Schema::Union(m.into())),
            proptest::collection::vec(inner.clone(), 1..4)
                .prop_map(|m| Schema::Intersection(m.into())),
            inner.prop_map(|s| Schema::Complement(Arc::new(s))),
        ]
    })
}

/// The Boolean fragment with the shapes a *rule* reads: sequences by arity, and
/// the length-bounded refinements that denote the same sets a different way.
///
/// [`schema`] is the corpus the laws above are written over, and it reaches no
/// rule that looks at a shape. That is where the two deciders can disagree: the
/// arm reducing a refinement to its base answered a refutation it had no grounds
/// for -- a list of at most zero elements is the empty list, and its base is
/// nowhere near it -- and no pair drawn from atoms and connectives would have
/// shown it. The laws are left over the fragment they were written for, because
/// their claim is structural equality after simplification and a sequence is
/// ordered into a normal form differently; the claim here is about answers.
/// A set with no value that no rule proves empty.
///
/// Two sequences of one position each whose positions share no value. Every
/// rule that reads a meet reads its members, and none reads two shapes for the
/// values they hold *together*, so proving this empty takes the descriptor --
/// which is the whole point: a refutation is read against the subject's own
/// emptiness, and the reading that cannot see this one is the reading a law
/// must attack.
///
/// Named rather than waited for. The generators draw every piece of it, and the
/// laws holding the two deciders to one answer went many reports without
/// drawing the combination.
fn without_a_value() -> impl Strategy<Value = Schema> {
    (0usize..2).prop_map(|which| {
        let sides = [Schema::Int, Schema::Str];
        Schema::Intersection(
            (0..sides.len())
                .map(|i| Schema::tuple(SeqShape::fixed([sides[(i + which) % sides.len()].clone()])))
                .collect::<Vec<_>>()
                .into(),
        )
    })
}

/// The containers that carry an element's refutation up to their own, each as
/// a rule that wraps one schema.
///
/// A list and a set take any number of elements, a tuple repeats one, and a
/// record's optional field need not be there -- so each of these holds a value
/// (the empty one) whatever the schema inside it admits. That is what makes
/// them the shapes where a refutation about the part says nothing about the
/// whole.
fn carriers() -> Vec<fn(Schema) -> Schema> {
    vec![
        |s| Schema::list(SeqShape::homogeneous(s)),
        |s| Schema::tuple(SeqShape::homogeneous(s)),
        Schema::set,
        |s| {
            Schema::record(
                vec![Field {
                    name: "k".into(),
                    schema: s,
                    required: false,
                }],
                Openness::Closed,
            )
        },
    ]
}

fn shaped_schema() -> impl Strategy<Value = Schema> {
    let atom = prop_oneof![
        Just(Schema::ANYTHING),
        Just(Schema::Nothing),
        Just(Schema::NoneType),
        Just(Schema::Bool),
        Just(Schema::Int),
        Just(Schema::Str),
        // The two atoms only an oracle can read: a class, and a constant. The
        // corpus reached neither, so every property over it asked its question
        // where nothing could be looked up.
        (0usize..3).prop_map(|i| Schema::Instance(ClassIx::new(i))),
        (0usize..4).prop_map(|i| Schema::Literal(ConstIx::new(i))),
    ];
    // A table of codes: a union of nothing but literals, in the canonical order
    // its constructor leaves it in, which is the shape read as a *set* rather
    // than walked as a list. Built through `Schema::union` and not the variant,
    // so what the rule is asked about is what a frontend builds; the raw unions
    // below stay raw, and a rule that reads an unordered list as a set would
    // answer one of the two wrongly.
    let table = proptest::collection::vec(0usize..4, 1..5).prop_map(|indices| {
        Schema::union(
            indices
                .into_iter()
                .map(|i| Schema::Literal(ConstIx::new(i))),
        )
    });
    let atom = prop_oneof![atom, table];
    let atom = prop_oneof![9 => atom, 1 => without_a_value()];
    atom.prop_recursive(3, 24, 3, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 1..3).prop_map(|m| Schema::Union(m.into())),
            proptest::collection::vec(inner.clone(), 1..3)
                .prop_map(|m| Schema::Intersection(m.into())),
            inner.clone().prop_map(|s| Schema::Complement(Arc::new(s))),
            inner
                .clone()
                .prop_map(|s| Schema::list(SeqShape::homogeneous(s))),
            proptest::collection::vec(inner.clone(), 0..3)
                .prop_map(|e| Schema::list(SeqShape::fixed(e))),
            (inner.clone(), 0usize..3, proptest::bool::ANY).prop_map(|(s, n, upper)| {
                let bound = if upper {
                    Constraint::MaxLen(n)
                } else {
                    Constraint::MinLen(n)
                };
                Schema::refine(Schema::list(SeqShape::homogeneous(s)), vec![bound])
            }),
            // A closed record over the same elements, with each field required
            // or not. A record refutes by a key one side requires and the other
            // does not carry, which is a mismatch like an arity: the same claim
            // as the sequences above, reached by a different rule, and the
            // corpus reaches both since both are believed.
            proptest::collection::vec((inner, proptest::bool::ANY), 0..3).prop_map(|fields| {
                Schema::record(
                    fields
                        .into_iter()
                        .enumerate()
                        .map(|(i, (schema, required))| Field {
                            name: format!("f{i}").into(),
                            schema,
                            required,
                        })
                        .collect(),
                    Openness::Closed,
                )
            }),
        ]
    })
}

/// A small, self-consistent oracle: three classes and three constants.
///
/// Every property in this file that compares the two deciders runs with no
/// oracle, and an oracle is the only place the core learns anything about
/// Python -- where a class becomes a kind, a constant becomes a value, and two
/// literals become disjoint or not. So the one question this pair of properties
/// exists to ask, whether the rules and the set representation ever contradict
/// each other, was asked only where neither of them could look anything up.
///
/// The two halves are consistent *by construction* rather than by agreement
/// between hand-written tables: `leaf_subtype` answers about a pair of
/// instances with `Class::derives_from`, which is the same relation the set
/// representation reads out of [`Constants::class`]. An oracle whose halves
/// disagreed would fail these properties by itself and say nothing about the
/// deciders, which is the trap a throwaway oracle in this tree fell into.
///
/// Class 0 derives from class 1 and neither lays down a layout; class 2 lays
/// one down and confines its instances to the tuple kind, which is the shape a
/// class deriving from a builtin has. Constant 3 repeats constant 0's value,
/// so an index and a value are two things here.
struct CorpusOracle;

impl CorpusOracle {
    fn class_at(index: ClassIx) -> Option<Class> {
        let base = Class::plain(1);
        match index.get() {
            0 => Some(Class::new(0, Class::PLAIN, &[base])),
            1 => Some(base),
            2 => Some(Class::laid_out(2, 9).of_kind(Kind::Tuple)),
            _ => None,
        }
    }
}

impl Constants for CorpusOracle {
    fn constant(&self, index: ConstIx) -> Option<Operand> {
        match index.get() {
            // Index 3 repeats index 0's value. A pool whose every index holds
            // a different value cannot tell a rule that reads a set of
            // *indices* from one that reads a set of values, and telling those
            // apart is what a literal oracle is for.
            0 | 3 => Some(Operand::Integer(0)),
            1 => Some(Operand::Integer(1)),
            2 => Some(Operand::Word(b"a".to_vec(), Kind::Str)),
            _ => None,
        }
    }

    fn operand(&self, index: OperandIx) -> Option<Operand> {
        self.constant(ConstIx::new(index.get()))
    }

    fn class(&self, index: ClassIx) -> Option<Class> {
        Self::class_at(index)
    }
}

impl LeafRelations for CorpusOracle {
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool> {
        match (sub, sup) {
            (Schema::Instance(a), Schema::Instance(b)) => {
                Some(Self::class_at(*a)?.derives_from(&Self::class_at(*b)?))
            }
            _ => None,
        }
    }

    /// Each of the three classes is one the bindings would read: a plain
    /// metaclass, so `isinstance` answers from the hierarchy. An index past
    /// them names no class, and declines.
    fn atom_denotes_a_set(&self, atom: &Schema) -> Option<bool> {
        match atom {
            Schema::Instance(index) => Some(CorpusOracle::class_at(*index).is_some()),
            _ => None,
        }
    }

    fn literal_kind(&self, constant: ConstIx) -> Option<Kind> {
        Some(match self.constant(constant)? {
            Operand::Integer(_) => Kind::Int,
            Operand::Word(_, kind) => kind,
            _ => return None,
        })
    }

    fn literals_disjoint(&self, left: ConstIx, right: ConstIx) -> Option<bool> {
        Some(self.constant(left)? != self.constant(right)?)
    }

    /// The same relation over two sets, which is the form the finite-set rule
    /// asks it in. Answered here rather than left to the default so that rule's
    /// *refutation* is under the properties below: an oracle that declines it
    /// leaves the rule proving and never refuting, and the agreement between
    /// the two deciders would be held over half of what the rule does.
    fn literal_sets_disjoint(&self, left: &[ConstIx], right: &[ConstIx]) -> Option<bool> {
        let read = |set: &[ConstIx]| {
            set.iter()
                .map(|index| self.constant(*index))
                .collect::<Option<Vec<_>>>()
        };
        let (left, right) = (read(left)?, read(right)?);
        Some(!left.iter().any(|value| right.contains(value)))
    }
}

fn union(a: Schema, b: Schema) -> Schema {
    Schema::Union(vec![a, b].into())
}
fn intersection(a: Schema, b: Schema) -> Schema {
    Schema::Intersection(vec![a, b].into())
}
fn not(a: Schema) -> Schema {
    Schema::Complement(Arc::new(a))
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
        Schema::Anything(_) => true,
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
        Just(Schema::ANYTHING),
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
            proptest::collection::vec(inner.clone(), 1..4).prop_map(|m| Schema::Union(m.into())),
            proptest::collection::vec(inner.clone(), 1..4)
                .prop_map(|m| Schema::Intersection(m.into())),
            inner.prop_map(|s| Schema::Complement(Arc::new(s))),
        ]
    })
}

#[test]
fn decides_scalar_emptiness_subtyping_and_equivalence() {
    // Multi-way emptiness the pairwise checks cannot reach.
    assert!(
        Schema::Intersection(vec![Schema::Int, not(Schema::Bool), not(Schema::Int)].into())
            .is_empty()
    );
    assert!(
        Schema::Intersection(
            vec![
                Schema::Union(vec![Schema::Int, Schema::Str].into()),
                not(Schema::Int),
                not(Schema::Str),
            ]
            .into()
        )
        .is_empty()
    );
    assert!(!Schema::Intersection(vec![Schema::Int, not(Schema::Bool)].into()).is_empty());
    // Subtyping, with bool ⊆ int.
    assert!(Schema::Bool.is_subtype_of(&Schema::Int));
    assert!(!Schema::Int.is_subtype_of(&Schema::Bool));
    assert!(!Schema::Float.is_subtype_of(&Schema::Int));
    // Equivalence between structurally different schemas: bool ∪ int = int.
    assert!(Schema::Union(vec![Schema::Bool, Schema::Int].into()).is_equivalent(&Schema::Int));
}

#[test]
fn is_empty_and_subtype_are_sound_off_the_scalar_fragment() {
    // Non-scalar leaves are never decided empty.
    assert!(!Schema::Literal(ConstIx::new(0)).is_empty());
    assert!(!Schema::Instance(ClassIx::new(0)).is_empty());
    assert!(!Schema::set(Schema::Int).is_empty());
    assert!(!Schema::list(SeqShape::homogeneous(Schema::Int)).is_empty());
    // A scalar mixed with a non-scalar leaf is undecidable here, so it is
    // never claimed empty (an instance could subclass the scalar's type).
    assert!(
        !Schema::Intersection(vec![Schema::Int, Schema::Instance(ClassIx::new(0))].into())
            .is_empty()
    );
    // Subtyping off the fragment is reflexive only.
    assert!(Schema::Instance(ClassIx::new(0)).is_subtype_of(&Schema::Instance(ClassIx::new(0))));
    assert!(!Schema::Instance(ClassIx::new(0)).is_subtype_of(&Schema::Instance(ClassIx::new(1))));
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
    assert!(!Schema::set(Schema::Nothing).is_empty());
    assert!(!Schema::frozen_set(Schema::Nothing).is_empty());
    // A keyed map is empty exactly when a required field is impossible.
    let field = |required| Field {
        name: "x".into(),
        schema: Schema::Nothing,
        required,
    };
    assert!(
        Schema::KeyedMap {
            fields: vec![field(true)].into(),
            defaults: Vec::new().into(),
        }
        .is_empty()
    );
    assert!(
        !Schema::KeyedMap {
            fields: vec![field(false)].into(),
            defaults: Vec::new().into(),
        }
        .is_empty()
    );
    // A union is empty only when every member is.
    assert!(Schema::Union(vec![Schema::Nothing, empty_pair.clone()].into()).is_empty());
    assert!(!Schema::Union(vec![Schema::Int, empty_pair].into()).is_empty());
}

#[test]
fn decides_structural_subtyping_between_containers() {
    let set = |s| Schema::set(s);
    let frozenset = |s| Schema::frozen_set(s);
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
        set(Schema::Union(vec![Schema::Bool, Schema::Int].into())).is_equivalent(&set(Schema::Int))
    );
}

#[test]
fn decides_record_and_mapping_subtyping() {
    let field = |name: &str, schema, required| Field {
        name: name.into(),
        schema,
        required,
    };
    let record = |fields| Schema::KeyedMap {
        fields,
        defaults: Vec::new().into(),
    };
    let mapping = |k, v| Schema::KeyedMap {
        fields: Vec::new().into(),
        defaults: vec![MapClause { key: k, value: v }].into(),
    };

    // Width: a closed record with fewer keys is a subtype of one with more.
    let narrow = record(vec![field("x", Schema::Int, true)].into());
    let wide = record(
        vec![
            field("x", Schema::Int, true),
            field("y", Schema::Str, false),
        ]
        .into(),
    );
    assert!(narrow.is_subtype_of(&wide));
    assert!(!wide.is_subtype_of(&narrow)); // wide admits key y; narrow (closed) forbids it
    // Depth: shared field schemas covary (bool ⊆ int).
    assert!(
        record(vec![field("x", Schema::Bool, true)].into())
            .is_subtype_of(&record(vec![field("x", Schema::Int, true)].into()))
    );
    // Required: a field the supertype requires must be required in the subtype.
    let required = record(vec![field("x", Schema::Int, true)].into());
    let optional = record(vec![field("x", Schema::Int, false)].into());
    assert!(required.is_subtype_of(&optional));
    assert!(!optional.is_subtype_of(&required));
    // Mappings covary in key and value.
    assert!(mapping(Schema::Str, Schema::Bool).is_subtype_of(&mapping(Schema::Str, Schema::Int)));
    assert!(!mapping(Schema::Str, Schema::Int).is_subtype_of(&mapping(Schema::Str, Schema::Bool)));
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
    assert!(
        !tup(Schema::Int, Schema::Int).is_subtype_of(&Schema::list(SeqShape::prefix_tail(
            [Schema::Int],
            Schema::Int
        )))
    );

    // Emptiness reasons about position: an uninhabited prefix empties the
    // whole tuple, but an uninhabited *tail* only forbids the repeats, so a
    // single-element tuple matching the prefix still inhabits it.
    assert!(tup(Schema::Nothing, Schema::Int).is_empty());
    assert!(!tup(Schema::Int, Schema::Nothing).is_empty());

    // Equivalence collapses a redundant union in the tail (bool ⊆ int).
    assert!(
        tup(
            Schema::Int,
            Schema::Union(vec![Schema::Bool, Schema::Int].into())
        )
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
    impl crate::descr::lower::Constants for ByIndex {}

    impl LeafRelations for ByIndex {
        fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
            None
        }
        fn compare(&self, a: OperandIx, b: OperandIx) -> Option<Ordering> {
            Some(a.get().cmp(&b.get()))
        }
    }
    let refine = |constraints: Vec<Constraint>| Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: constraints.into(),
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
    impl crate::descr::lower::Constants for ByValue {}

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
        base: Arc::new(base),
        constraints: constraints.into(),
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
    assert!(Schema::Intersection(vec![empty_list.clone(), Schema::ANYTHING].into()).is_empty());
    // Order does not matter: the fold runs over every member.
    assert!(Schema::Intersection(vec![Schema::ANYTHING, empty_list].into()).is_empty());
    // And an intersection of two inhabited members with an opaque region is
    // not reported empty, so the fold is not merely answering true.
    let list_of_int = Schema::list(SeqShape::homogeneous(Schema::Int));
    assert!(!Schema::Intersection(vec![list_of_int, Schema::ANYTHING].into()).is_empty());
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
        Schema::set(Schema::Int),
        Schema::frozen_set(Schema::Int),
        Schema::mapping(MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }),
        Schema::ANYTHING,
        Schema::ANY,
    ] {
        assert!(
            !schema.disjoint(&schema),
            "{schema:?} is disjoint from itself"
        );
    }
    // A refinement takes its base's disjointness, so it is not self-disjoint
    // either -- unless its base is bottom, which is the same one case.
    let refined = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::MinLen(1)].into(),
    };
    assert!(!refined.disjoint(&refined));
}

#[test]
fn decision_arms_are_pinned_independently_of_the_python_suite() {
    // Each assertion fails under a specific mutation of a decision arm, so the
    // core's own unit tests catch a defect without relying on the Python layer.
    use core::cmp::Ordering;
    struct ByIndex;
    impl crate::descr::lower::Constants for ByIndex {}

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
    assert!(list(Schema::Int).is_subtype_of(&Schema::ANYTHING));
    // Below a complement, which is the emptiness reduction rather than a
    // structural arm: a list shares no value with an int, so it lies inside
    // the complement of int. Nothing structural can see this -- there is no
    // shape on the right to recurse into.
    let not = |s| Schema::Complement(Arc::new(s));
    assert!(list(Schema::Int).is_subtype_of(&not(Schema::Int)));
    assert!(map(Schema::Str, Schema::Int).is_subtype_of(&not(Schema::Str)));
    // ...and it stays sound where the two do share values.
    assert!(!list(Schema::Int).is_subtype_of(&not(list(Schema::Bool))));
    assert!(!Schema::Bool.is_subtype_of(&not(Schema::Int)));
    // A meet is below a member of a join, and a conjunct decides a meet's
    // supertype.
    assert!(list(Schema::Bool).is_subtype_of(&Schema::Intersection(
        vec![list(Schema::Int), Schema::ANYTHING].into()
    )));
    assert!(
        Schema::Intersection(vec![list(Schema::Bool), list(Schema::Int)].into())
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
            base: Arc::new(Schema::Int),
            constraints: vec![ge(10), le(0)].into(),
        }
        .is_subtype_of_under(&Schema::Nothing, &ByIndex, &[])
    );

    // Refinement bounds: equal closed bounds are a singleton (not empty), and
    // a strict pair at the same value is empty; a length window that is exactly
    // satisfiable is not empty.
    let refine = |constraints| Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints,
    };
    assert!(!refine(vec![ge(5), le(5)].into()).is_empty_with(&ByIndex, &[]));
    assert!(refine(vec![gt(5), lt(5)].into()).is_empty_with(&ByIndex, &[]));
    assert!(!refine(vec![Constraint::MinLen(5), Constraint::MaxLen(5)].into()).is_empty());
    // An intersection's refinement bounds are joined: both sides are needed.
    assert!(
        Schema::Intersection(vec![refine(vec![ge(5)].into()), refine(vec![le(0)].into()),].into())
            .is_empty_with(&ByIndex, &[])
    );
    assert!(
        !Schema::Intersection(vec![refine(vec![ge(0)].into()), refine(vec![le(5)].into()),].into())
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
        name: name.into(),
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
        fields: vec![field("x", Schema::Int, required)].into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }]
        .into(),
    };
    assert!(!mixed(false).is_subtype_of(&mixed(true)));
    // A pure mapping is not a subtype of a mixed map that requires a field it
    // lacks: the pure-mapping branch must need both sides field-free.
    assert!(
        !map(Schema::Str, Schema::Int).is_subtype_of(&Schema::KeyedMap {
            fields: vec![field("x", Schema::Int, true)].into(),
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int
            }]
            .into(),
        })
    );
    // A mixed map is not a subtype of one with an extra field whose catch-all
    // would admit an incompatible value: the mixed rule needs matching field
    // names, so its guard needs both an equal count and a name match.
    assert!(
        !Schema::KeyedMap {
            fields: vec![field("x", Schema::Int, false)].into(),
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int
            }]
            .into(),
        }
        .is_subtype_of(&Schema::KeyedMap {
            fields: vec![
                field("x", Schema::Int, false),
                field("z", Schema::Bool, false),
            ]
            .into(),
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int
            }]
            .into(),
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
        name: name.into(),
        schema,
        required,
    };
    let with_catch_all = |fields| Schema::KeyedMap {
        fields,
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }]
        .into(),
    };
    let base = || with_catch_all(vec![field("x", Schema::Int, true)].into());
    let plus_y = |schema, required| {
        with_catch_all(vec![field("x", Schema::Int, true), field("y", schema, required)].into())
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
        base: Arc::new(base),
        constraints: constraints.into(),
    };

    // A refinement is a subtype of its base, and of anything its base subtypes.
    assert!(
        refine(Schema::Bool, vec![Constraint::Ge(OperandIx::new(0))]).is_subtype_of(&Schema::Int)
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
        !refine(Schema::Int, vec![Constraint::Ge(OperandIx::new(0))]).is_subtype_of(&Schema::Str)
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
    let schema = Schema::Union(
        vec![
            Schema::Literal(ConstIx::new(0)),
            Schema::Instance(ClassIx::new(1)),
            Schema::Ref(DefIx::new(0)),
            Schema::set(Schema::Literal(ConstIx::new(1))),
        ]
        .into(),
    );
    // The second pool interned into the first: old 0 -> 5, old 1 -> 6. The
    // members come back in canonical order -- a remap that renumbers a member
    // set sorts it again -- so the expectation is written through the
    // constructor rather than as the list the schema above was written as.
    let lit_map = [5, 6];
    let remapped = schema.reindexed(&lit_map, DefShift::new(3));
    let expected = Schema::union([
        Schema::Literal(ConstIx::new(5)),
        Schema::Instance(ClassIx::new(6)),
        Schema::Ref(DefIx::new(3)),
        Schema::set(Schema::Literal(ConstIx::new(6))),
    ]);
    assert_eq!(remapped, expected);

    // `shifted` is the identity-map case: every index moves by a fixed offset.
    let shifted = schema.shifted(PoolShift::new(5), DefShift::new(3));
    assert_eq!(shifted, expected);
    // A constraint operand index is remapped too.
    let refined = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
    };
    assert_eq!(
        refined.reindexed(&lit_map, DefShift::new(0)),
        Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![Constraint::Ge(OperandIx::new(5))].into(),
        }
    );
}

#[test]
fn simplify_canonicalizes_refinement_constraints() {
    let refine = |base, constraints: Vec<Constraint>| Schema::Refine {
        base: Arc::new(base),
        constraints: constraints.into(),
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
            Schema::Union(vec![Schema::Int, Schema::Int].into()),
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
    // Nothing changed, so nothing is rebuilt and the shape answers for itself.
    assert_eq!(shape.mapped_elems(&|_| None), None);

    let complemented = shape
        .mapped_elems(&|s| Some(Schema::Complement(Arc::new(s.clone()))))
        .expect("every element changed");
    assert_eq!(
        complemented.prefix.to_vec(),
        vec![Schema::Complement(Arc::new(Schema::Str))]
    );
    assert_eq!(
        complemented.tail.as_deref(),
        Some(&Schema::Complement(Arc::new(Schema::Int)))
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
        name: name.into(),
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
        fields: vec![field("a", value_field, true)].into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: value_default,
        }]
        .into(),
    };
    assert!(mixed(Schema::Bool, Schema::Bool).is_subtype_of(&mixed(Schema::Int, Schema::Int)));
    assert!(!mixed(Schema::Int, Schema::Int).is_subtype_of(&mixed(Schema::Int, Schema::Bool)));
    assert!(!mixed(Schema::Int, Schema::Bool).is_subtype_of(&mixed(Schema::Bool, Schema::Bool)));
    let mixed_b = Schema::KeyedMap {
        fields: vec![field("b", Schema::Int, true)].into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }]
        .into(),
    };
    assert!(!mixed(Schema::Int, Schema::Int).is_subtype_of(&mixed_b));

    // A mixed map with an extra field is a subtype when a supertype catch-all
    // over all string keys covers that field's value.
    let with_extra = Schema::KeyedMap {
        fields: vec![field("a", Schema::Int, true), field("b", Schema::Str, true)].into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::Bytes,
        }]
        .into(),
    };
    let covering = Schema::KeyedMap {
        fields: vec![field("a", Schema::Int, true)].into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::ANYTHING,
        }]
        .into(),
    };
    assert!(with_extra.is_subtype_of(&covering));
    // The extra field is not covered when the catch-all value is too narrow,
    // even though the catch-all clauses subsume (so only the extra-field
    // coverage decides it -- the "extra" set must be the fields not shared).
    let extra_uncovered = Schema::KeyedMap {
        fields: vec![
            field("a", Schema::Int, true),
            field("b", Schema::Bytes, true),
        ]
        .into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }]
        .into(),
    };
    let str_catch_all = Schema::KeyedMap {
        fields: vec![field("a", Schema::Int, true)].into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }]
        .into(),
    };
    assert!(!extra_uncovered.is_subtype_of(&str_catch_all));
    // The catch-all key must admit the field name: an int-keyed catch-all does
    // not cover a string field name even when its value would.
    let extra_str = Schema::KeyedMap {
        fields: vec![field("a", Schema::Int, true), field("b", Schema::Str, true)].into(),
        defaults: vec![MapClause {
            key: Schema::Int,
            value: Schema::Int,
        }]
        .into(),
    };
    let int_catch_all = Schema::KeyedMap {
        fields: vec![field("a", Schema::Int, true)].into(),
        defaults: vec![MapClause {
            key: Schema::Int,
            value: Schema::ANYTHING,
        }]
        .into(),
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
    impl crate::descr::lower::Constants for ByIndex {}

    impl LeafRelations for ByIndex {
        fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
            None
        }
        fn compare(&self, a: OperandIx, b: OperandIx) -> Option<Ordering> {
            Some(a.get().cmp(&b.get()))
        }
    }
    let refine = |constraints| Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints,
    };
    // A lower bound above the upper bound is empty.
    assert!(
        refine(
            vec![
                Constraint::Ge(OperandIx::new(10)),
                Constraint::Le(OperandIx::new(0))
            ]
            .into()
        )
        .is_empty_with(&ByIndex, &[])
    );
    // Equal bounds with one strict end are empty; both closed is a singleton.
    assert!(
        refine(
            vec![
                Constraint::Ge(OperandIx::new(5)),
                Constraint::Lt(OperandIx::new(5))
            ]
            .into()
        )
        .is_empty_with(&ByIndex, &[])
    );
    assert!(
        !refine(
            vec![
                Constraint::Ge(OperandIx::new(5)),
                Constraint::Le(OperandIx::new(5))
            ]
            .into()
        )
        .is_empty_with(&ByIndex, &[])
    );
    // A satisfiable range is not empty.
    assert!(
        !refine(
            vec![
                Constraint::Ge(OperandIx::new(0)),
                Constraint::Le(OperandIx::new(10))
            ]
            .into()
        )
        .is_empty_with(&ByIndex, &[])
    );
    // A length contradiction needs no value comparison.
    assert!(refine(vec![Constraint::MinLen(5), Constraint::MaxLen(3)].into()).is_empty());
    // Refinements with contradictory bounds across an intersection are empty.
    let intersection = Schema::Intersection(
        vec![
            refine(vec![Constraint::Ge(OperandIx::new(5))].into()),
            refine(vec![Constraint::Lt(OperandIx::new(5))].into()),
        ]
        .into(),
    );
    assert!(intersection.is_empty_with(&ByIndex, &[]));
    // Without an ordering oracle the numeric bounds stay conservative.
    assert!(
        !refine(
            vec![
                Constraint::Ge(OperandIx::new(10)),
                Constraint::Le(OperandIx::new(0))
            ]
            .into()
        )
        .is_empty()
    );
}

// THEORY: two-fixpoints-one-procedure
#[test]
fn detects_uninhabited_recursive_schemas() {
    let field = |name: &str, schema, required| Field {
        name: name.into(),
        schema,
        required,
    };
    // t = {value: int, next: t} — a mandatory self-reference, no base case:
    // no finite value satisfies it.
    let uninhabited = [Schema::KeyedMap {
        fields: vec![
            field("value", Schema::Int, true),
            field("next", Schema::Ref(DefIx::new(0)), true),
        ]
        .into(),
        defaults: Vec::new().into(),
    }];
    assert!(Schema::Ref(DefIx::new(0)).is_empty_under(&uninhabited));
    // t = None | {next: t} — a base case makes it inhabited.
    let inhabited = [Schema::Union(
        vec![
            Schema::NoneType,
            Schema::KeyedMap {
                fields: vec![field("next", Schema::Ref(DefIx::new(0)), true)].into(),
                defaults: Vec::new().into(),
            },
        ]
        .into(),
    )];
    assert!(!Schema::Ref(DefIx::new(0)).is_empty_under(&inhabited));
    // t = {next?: t} — an optional self-reference is inhabited by the empty map.
    let optional = [Schema::KeyedMap {
        fields: vec![field("next", Schema::Ref(DefIx::new(0)), false)].into(),
        defaults: Vec::new().into(),
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
    let not = |s| Schema::Complement(Arc::new(s));
    // ¬A ⊆ ¬B iff B ⊆ A: ¬int ⊆ ¬bool because bool ⊆ int.
    assert!(not(Schema::Int).is_subtype_of(&not(Schema::Bool)));
    assert!(!not(Schema::Bool).is_subtype_of(&not(Schema::Int)));
    // Reflexivity holds for a complement (regression: it failed before this
    // rule existed).
    assert!(not(Schema::Int).is_subtype_of(&not(Schema::Int)));
    assert!(
        not(Schema::Literal(ConstIx::new(0))).is_subtype_of(&not(Schema::Literal(ConstIx::new(0))))
    );
}

// THEORY: recursive-subtyping, two-fixpoints-one-procedure
// THEORY: recursive-subtyping, a-reference-denotes-its-definition
#[test]
fn decides_recursive_subtyping_coinductively() {
    let field = |name: &str, schema, required| Field {
        name: name.into(),
        schema,
        required,
    };
    let list_of = |value, next| {
        Schema::Union(
            vec![
                Schema::NoneType,
                Schema::KeyedMap {
                    fields: vec![
                        field("value", value, true),
                        field("next", Schema::Ref(next), true),
                    ]
                    .into(),
                    defaults: Vec::new().into(),
                },
            ]
            .into(),
        )
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
        Schema::Anything(_) => true,
        Schema::Nothing => false,
        Schema::Coll { element, .. } => match value {
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
        Just(Schema::ANYTHING),
        Just(Schema::Nothing),
        Just(Schema::NoneType),
        Just(Schema::Bool),
        Just(Schema::Int),
        Just(Schema::Float),
        Just(Schema::Str),
        Just(Schema::Bytes),
        scalar_schema().prop_map(Schema::set),
    ];
    leaf.prop_recursive(3, 16, 3, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 1..3).prop_map(|m| Schema::Union(m.into())),
            proptest::collection::vec(inner.clone(), 1..3)
                .prop_map(|m| Schema::Intersection(m.into())),
            inner.prop_map(|s| Schema::Complement(Arc::new(s))),
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
        Just(Schema::ANYTHING),
        Just(Schema::ANY),
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
            proptest::collection::vec(inner.clone(), 1..3).prop_map(|m| Schema::Union(m.into())),
            proptest::collection::vec(inner.clone(), 1..3)
                .prop_map(|m| Schema::Intersection(m.into())),
            inner.clone().prop_map(|s| Schema::Complement(Arc::new(s))),
            inner.clone().prop_map(Schema::set),
            inner.clone().prop_map(Schema::frozen_set),
            (inner.clone(), proptest::collection::vec(constraint(), 0..3)).prop_map(
                |(base, constraints)| Schema::Refine {
                    base: Arc::new(base),
                    constraints: constraints.into(),
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
                }]
                .into(),
                defaults: vec![MapClause {
                    key: Schema::Str,
                    value: default
                }]
                .into(),
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
        Just(Schema::ANYTHING),
    ];
    leaf.prop_recursive(4, 24, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 1..3).prop_map(|m| Schema::Union(m.into())),
            prop::collection::vec(inner.clone(), 1..3).prop_map(|m| Schema::Intersection(m.into())),
            inner.clone().prop_map(|s| Schema::Complement(Arc::new(s))),
            inner.clone().prop_map(|s| Schema::Refine {
                base: Arc::new(s),
                constraints: vec![Constraint::MinLen(1)].into(),
            }),
            inner.clone().prop_map(Schema::set),
            inner
                .clone()
                .prop_map(|s| Schema::list(SeqShape::homogeneous(s))),
            inner.prop_map(|s| Schema::record(
                vec![Field {
                    name: "f".into(),
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
        if Schema::Complement(Arc::new(b.clone())).is_empty() {
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

    // THEORY: lattice-theory, property-testing, each-kind-is-closed
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
        prop_assert!(same(&intersection(a.clone(), Schema::ANYTHING), &a));
        prop_assert!(same(&union(a.clone(), Schema::ANYTHING), &Schema::ANYTHING));
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
        prop_assert!(same(&union(a.clone(), not(a.clone())), &Schema::ANYTHING));
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
        prop_assert_eq!(intersection(a.clone(), Schema::ANYTHING).simplify(), a.clone().simplify());
        prop_assert_eq!(union(a.clone(), Schema::ANYTHING).simplify(), Schema::ANYTHING);
        prop_assert_eq!(intersection(a, Schema::Nothing).simplify(), Schema::Nothing);
    }

    #[test]
    fn double_negation(a in schema()) {
        prop_assert_eq!(not(not(a.clone())).simplify(), a.simplify());
    }

    /// De Morgan, asserted over the values rather than over the two forms.
    ///
    /// The simplifier puts a schema in negation normal form and no further,
    /// and NNF is not canonical -- so `simplify(a) == simplify(b)` is not an
    /// equivalence test, and a law that spells it as one is asserting
    /// *confluence*: that two rewrite sequences reach one form. The two sides
    /// here do take different sequences. The left simplifies a union and then
    /// pushes the complement through it; the right complements each member
    /// first and may then collapse the meet through a rule the left never
    /// reaches. Any sound improvement to the simplifier that fires on one side
    /// only fails such a law, which is what happened to a disjointness rule
    /// that was measured, found sound, and dropped for it.
    ///
    /// What de Morgan actually claims is about the values, and that is what
    /// this asserts: both forms and the unsimplified one admit the same values
    /// over the corpus. The deciders are asked too, in the one direction that
    /// is a defect rather than a decline -- neither may *refute* a pair the law
    /// says is one set.
    #[test]
    fn de_morgan(a in decidable_schema(), b in decidable_schema()) {
        let pool = const_pool();
        // Both forms: the complement of a join is the meet of the complements,
        // and the complement of a meet is the join of the complements.
        let pairs = [
            (
                not(union(a.clone(), b.clone())),
                intersection(not(a.clone()), not(b.clone())),
            ),
            (
                not(intersection(a.clone(), b.clone())),
                union(not(a.clone()), not(b.clone())),
            ),
        ];
        let universe = boundary_values(&[&a, &b]);
        for (left, right) in pairs {
            for value in &universe {
                let held = member_full(&left, value, &pool);
                prop_assert_eq!(
                    held,
                    member_full(&right, value, &pool),
                    "{:?} tells {:?} and {:?} apart",
                    value, left, right
                );
                // And simplifying moves no value of either, which is the
                // property the structural form was standing in for.
                prop_assert_eq!(held, member_full(&left.clone().simplify(), value, &pool));
                prop_assert_eq!(held, member_full(&right.clone().simplify(), value, &pool));
            }
            prop_assert_ne!(
                left.subtype_relation(
                    &right,
                    &NoLeafRelations,
                    &[],
                    &std::cell::Cell::new(DECISION_BUDGET),
                ),
                Relation::Fails,
                "the rules refuted one side of de Morgan against the other"
            );
            prop_assert_ne!(
                right.subtype_relation(
                    &left,
                    &NoLeafRelations,
                    &[],
                    &std::cell::Cell::new(DECISION_BUDGET),
                ),
                Relation::Fails,
                "the rules refuted one side of de Morgan against the other"
            );
        }
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
        prop_assert!(a.is_subtype_of(&Schema::ANYTHING));
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
        return Schema::Complement(Arc::new(Schema::Int));
    }
    let child = complemented_tower(depth - 1);
    Schema::Complement(Arc::new(Schema::Union(vec![child.clone(), child].into())))
}

/// A tower of intersections of unions, the shape whose subtyping decision
/// re-explored shared subtrees before the goal memo. The leaves are sets so
/// the scalar region fast path does not short-circuit the descent.
fn intersection_of_unions_tower(depth: usize, leaf: Schema) -> Schema {
    let mut node = Schema::set(leaf);
    for _ in 0..depth {
        node = Schema::Intersection(
            vec![
                Schema::Union(vec![node.clone(), Schema::set(Schema::Str)].into()),
                Schema::Union(vec![node, Schema::set(Schema::Bytes)].into()),
            ]
            .into(),
        );
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
    // The duplicate union members collapse, so the reduced form is small. What
    // this measures is the work of reaching it, not the size of the result.
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
    assert_eq!(Schema::Complement(Arc::new(Schema::Int)).depth(), 2);
    assert_eq!(union(Schema::Int, Schema::Str).depth(), 2);
    // The max over members, not their sum: one branch is two deep.
    let branchy = union(Schema::Int, Schema::Complement(Arc::new(Schema::Str)));
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
        intersection(cancel.clone(), Schema::ANYTHING),
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
        let mut deep = Schema::Complement(Arc::new(Schema::Int));
        for _ in 0..depth {
            deep =
                Schema::Intersection(vec![deep, Schema::Complement(Arc::new(Schema::Str))].into());
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
        }]
        .into(),
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
        }]
        .into(),
    };
    let other_record = Schema::AttrRecord {
        fields: vec![Field {
            name: "y".into(),
            schema: Schema::Str,
            required: true,
        }]
        .into(),
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
    // Two *different* records are two records; one written twice is one,
    // because a meet is idempotent where the schema is built.
    assert_eq!(
        Schema::meet([class(3), record.clone(), other_record]).object_class(),
        None
    );
    assert_eq!(
        Schema::meet([class(3), record.clone(), record]).object_class(),
        Some(ClassIx::new(3))
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
        ]
        .into(),
    };
    let wide = Schema::AttrRecord {
        fields: vec![Field {
            name: "x".into(),
            schema: Schema::Int, // bool ⊆ int, and x is narrower; y is extra
            required: true,
        }]
        .into(),
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
        }]
        .into(),
    };
    assert!(!maybe_x.is_subtype_of(&wide));
    assert!(
        narrow.is_subtype_of(&Schema::AttrRecord {
            fields: vec![Field {
                name: "x".into(),
                schema: Schema::Int,
                required: false,
            }]
            .into(),
        })
    );
    let object = |class, record: &Schema| Schema::meet([Schema::Instance(class), record.clone()]);
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
    assert_eq!(everything.simplify(), Schema::ANYTHING);
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
#[derive(Clone, Debug, PartialEq)]
enum Obj {
    None,
    Bool(bool),
    /// An integer, as wide as the component that represents one.
    ///
    /// A corpus narrower than the representation cannot reach the ends of it,
    /// and the ends are where the lift a modulus takes stops being exact: the
    /// subtraction it performs is the operation that leaves the range.
    Int(i64),
    Float(f64),
    Str(&'static str),
    /// A byte string, as its length: a length bound over the kind is decided at
    /// a count, and a kind with one value has no count to be decided at.
    Bytes(usize),
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
        #[expect(
            clippy::cast_precision_loss,
            reason = "the model's own comparison, which rounds as Python's float() does"
        )]
        Obj::Int(i) => Some(*i as f64),
        Obj::Float(f) => Some(*f),
        _ => None,
    }
}

fn val_len(v: &Obj) -> Option<usize> {
    match v {
        Obj::Str(s) => Some(s.chars().count()),
        Obj::Bytes(len) => Some(*len),
        Obj::List(xs) | Obj::Tuple(xs) | Obj::Set(xs) | Obj::FrozenSet(xs) => Some(xs.len()),
        Obj::Map(m) => Some(m.len()),
        _ => None,
    }
}

/// Typed-singleton equality: same type *and* equal, so `Literal[1]` admits
/// neither `True` nor `1.0`.
fn typed_eq(constant: &Obj, v: &Obj) -> bool {
    match (constant, v) {
        (Obj::None, Obj::None) => true,
        (Obj::Bytes(a), Obj::Bytes(b)) => a == b,
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
        Schema::Anything(_) => true,
        Schema::Nothing => false,
        Schema::NoneType => matches!(value, Obj::None),
        Schema::Bool => matches!(value, Obj::Bool(_)),
        Schema::Int => matches!(value, Obj::Bool(_) | Obj::Int(_)), // bool ⊆ int
        Schema::Float => matches!(value, Obj::Float(_)),
        Schema::Str => matches!(value, Obj::Str(_)),
        Schema::Bytes => matches!(value, Obj::Bytes(_)),
        Schema::Literal(index) => typed_eq(&pool[index.get()], value),
        Schema::Coll { container, element } => match (container, value) {
            (CollKind::Set, Obj::Set(items)) | (CollKind::FrozenSet, Obj::FrozenSet(items)) => {
                items.iter().all(|item| member_full(element, item, pool))
            }
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
                    match entries.iter().find(|(key, _)| &*field.name == *key) {
                        Some((_, val)) => member_full(&field.schema, val, pool),
                        None => !field.required,
                    }
                });
                let rest_ok = entries.iter().all(|(key, val)| {
                    if fields.iter().any(|field| &*field.name == *key) {
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

/// The values at the edge of every kind, which no schema has to name.
fn kind_edges() -> Vec<Obj> {
    vec![
        Obj::None,
        Obj::Bool(true),
        Obj::Bool(false),
        Obj::Int(0),
        Obj::Int(1),
        Obj::Int(2),
        Obj::Int(5),
        Obj::Int(i64::MIN),
        Obj::Int(i64::MIN + 1),
        Obj::Int(i64::MAX),
        Obj::Float(1.0),
        Obj::Float(2.5),
        Obj::Float(f64::NAN),
        Obj::Float(9_007_199_254_740_992.0),
        Obj::Str("a"),
        Obj::Str("b"),
        Obj::Str(""),
        // The values at the edge of a kind's universe, which is where a
        // representation built by hand stops describing the set it names: the
        // newline a length bound must count, the end of the integer range a
        // residue class is lifted across, the first float that is no integer's
        // equal, and the value outside every order.
        Obj::Str("\n"),
        Obj::Str("a\nb"),
        Obj::Bytes(0),
        Obj::Bytes(1),
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

/// Strings of each small length, so a length bound has a word on both sides.
const SIZED: [&str; 7] = ["", "a", "ab", "abc", "abcd", "abcde", "abcdef"];
/// Field names in the order the generator declares them.
const NAMES: [&str; 6] = ["a", "b", "c", "d", "e", "f"];

/// The corpus's own spelling of a field name, since a value's keys are static.
///
/// A record admits a mapping whose keys are *its* names, so a value built with
/// any other name is a non-member however well it is shaped -- which reads as a
/// universe too thin rather than as the bug it is.
fn static_name(name: &str) -> Option<&'static str> {
    NAMES.into_iter().find(|known| *known == name)
}
/// How far past a schema's own width the derived universe reaches.
const EDGE_SPAN: usize = 1;

/// The universe a property is judged over: every kind's edge, and every value
/// the schemas drawn put a boundary at.
///
/// A corpus written by hand holds the values its author thought of, and a
/// generator draws schemas whose boundaries that author never saw: a length
/// bound at three decides on sequences of two, three and four, and a universe
/// holding none of them reports the relation sound because nothing in it can
/// say otherwise. So each schema contributes its own edges -- the operand of
/// every bound with its two neighbours, a container of each kind at each side
/// of every length, the constant of every literal, and a mapping at each side
/// of every record's width -- and the property is judged over the union of
/// those with the fixed kind edges.
fn boundary_values(schemas: &[&Schema]) -> Vec<Obj> {
    let mut pool = kind_edges();
    // Twice over the same pool. The first pass is what each schema contributes
    // on its own; the second builds the composites again with every schema's
    // contribution already in the pool, so a node is probed with values the
    // other schema put there. One pass cannot: the pair is read left to right,
    // and the left schema's nodes are built before the right has contributed.
    let mut sources: Vec<Vec<Obj>> = Vec::new();
    for _ in 0..2 {
        sources.clear();
        for schema in schemas {
            sources.push(edges_of(schema, &mut pool));
        }
    }
    sources.push(crossed_sequences(schemas, &mut pool));
    // The fixed edges first, then the sources round-robin.
    //
    // Round-robin rather than concatenated, and the reason is the cap below: a
    // universe read off a concatenation spends its budget on whichever source
    // came first, and the cross pass -- which is last, and which is the only
    // source holding one schema's head beside another's tail -- filled the
    // budget on its own. A pair whose subject contributed nothing that
    // survived then read as "no value refutes it" while the universe held 768
    // values of the other schema.
    let mut values = kind_edges();
    values.extend(interleaved(&sources));
    // The cap is read after the duplicates leave, so a node that repeats a
    // value cannot push another node's own values past the end.
    let mut values = deduplicated(values);
    values.truncate(CAP_UNIVERSE);
    values
}

/// The values in order, each kept the first time it is seen.
///
/// Read before a cap, because a cap spent on a value the list already holds
/// buys nothing and costs the value behind it. The universe's cap has always
/// been read this way; a *node's* cap needs it for the same reason and at a
/// shape where it decides the answer. `candidates` round-robins three sources
/// that overlap by construction -- the pool the inherited values are strided
/// from already holds the node's own -- so a variadic tuple over the top spent
/// four of its first twenty on one repeated single-element tuple, and the
/// tuple holding a `bool` fell past the quarter-cap. That is the value which
/// separates the shape from a tuple over the complement of `bool`, and a
/// refutation between the two then had no witness in the universe.
///
/// Compared rather than rendered to a key. A key per value is what the
/// universe could afford at one call; this is read once per node, and a node
/// builds a member from every candidate of its element, so spelling a tree of
/// values at each of them costs four times the laws' running time -- measured,
/// against one-and-a-third for the comparison. Two NaNs compare unequal and
/// both stay, which is the only repeat a rendered key would have dropped and
/// this keeps.
fn deduplicated(values: Vec<Obj>) -> Vec<Obj> {
    let mut kept: Vec<Obj> = Vec::with_capacity(values.len());
    for value in values {
        if !kept.contains(&value) {
            kept.push(value);
        }
    }
    kept
}

/// Values of one sequence whose positions come from another's.
///
/// The construction a per-schema universe cannot reach, and a refutation about
/// a sequence needs it. A pair of sequences is separated at a position: a value
/// of `a` whose position *i* lies outside `b`'s element there. Building each
/// schema's values on its own gives `a`'s positions filled from `a` and `b`'s
/// from `b`, and the witness is one schema's head beside the other's tail -- a
/// combination neither builds.
///
/// `list[not None, X] <= list[not tuple[Any, Float], Y]` is the shape that
/// found this. It is refuted, because a two-element tuple ending in a float is
/// not `None` and is not outside `tuple[Any, Float]`. The universe held
/// `[that tuple, None]` from the right schema and `[a small tuple, an int]`
/// from the left, and never `[that tuple, an int]`, which is the value under
/// the refutation.
///
/// Bounded the way everything here is: each ordered pair, each position of the
/// one with a fixed width, each of a capped set of candidates.
fn crossed_sequences(schemas: &[&Schema], out: &mut Vec<Obj>) -> Vec<Obj> {
    let shapes: Vec<(SeqKind, &SeqShape)> = schemas
        .iter()
        .filter_map(|schema| match schema {
            Schema::Seq { container, shape } => Some((*container, shape)),
            _ => None,
        })
        .collect();
    // The schema at one position of a shape: the prefix where it reaches, and
    // the repeating tail past it. A tailed shape has a schema at every width,
    // which is what lets a fixed one be crossed against it.
    let at = |shape: &SeqShape, position: usize| {
        shape
            .prefix
            .get(position)
            .or(shape.tail.as_deref())
            .cloned()
    };
    let mut made = Vec::new();
    for (container, mine) in &shapes {
        if mine.tail.is_some() {
            continue; // no fixed width to fill
        }
        let Some(base): Option<Vec<Obj>> = mine
            .prefix
            .iter()
            .map(|element| a_member(element, MEMBER_FUEL))
            .collect()
        else {
            continue; // a position with no member of its own builds no value
        };
        for (_, theirs) in &shapes {
            for position in 0..base.len() {
                let Some(element) = at(theirs, position) else {
                    continue;
                };
                for value in candidates(&element, out).into_iter().take(CAP_PER_NODE / 4) {
                    let mut items = base.clone();
                    items[position] = value;
                    made.push(hold(*container, items));
                }
            }
        }
    }
    // And the widths a *length bound* separates, filled from the other schema's
    // elements. A bound is told from the shape it narrows by a value of that
    // shape at a width the bound refuses, and the elements have to come from
    // the schema the value must be a member of: `list[set[Any]]` below
    // `list[Any]` of at most two is refuted by a three-element list of sets,
    // and neither the bound -- which builds its widths out of a fixed item --
    // nor the subject -- which reaches two past its own prefix -- spells one.
    for (container, len) in bounded_widths(schemas) {
        for (_, theirs) in &shapes {
            for element in theirs
                .elements()
                .flat_map(|element| candidates(element, out))
                .take(CAP_PER_NODE / 4)
            {
                for width in len.saturating_sub(EDGE_SPAN)..=(len + EDGE_SPAN) {
                    made.push(hold(container, vec![element.clone(); width]));
                }
            }
        }
    }
    made.truncate(CAP_PER_NODE * 4);
    out.extend(made.iter().cloned());
    made
}

/// Each length bound a sequence in `schemas` carries, with its container.
///
/// Read through the refinement, because a bound is a node above the shape it
/// narrows and the shape is what a value of that width has to be built as.
fn bounded_widths(schemas: &[&Schema]) -> Vec<(SeqKind, usize)> {
    let mut found = Vec::new();
    for schema in schemas {
        let Schema::Refine { base, constraints } = schema else {
            continue;
        };
        let Schema::Seq { container, .. } = base.as_ref() else {
            continue;
        };
        for constraint in constraints.iter() {
            if let Constraint::MinLen(len) | Constraint::MaxLen(len) = constraint {
                found.push((*container, *len));
            }
        }
    }
    found
}

// THEORY: each-kind-is-closed
/// The universe separates a pair at a position *inside* a shape.
///
/// What this universe is for, asked at the depth its caps govern rather than
/// at the top. `list[tuple[*Any], list[float]]` is refuted below the same list
/// whose tuple runs over a complement, and it is refuted at the first
/// position: the witness is a list whose head is a tuple carrying a value the
/// complement excludes, and every part of that is a value some node of the
/// pair already names.
///
/// The universe held none of them, and for two reasons that are one reason. A
/// scalar sat behind twelve sized containers in the one-per-kind probes, and a
/// repeated single-element tuple took four of the first twenty places in a
/// candidate list, so the quarter-cap a position reads through was spent
/// before either reached the tuple. A cap spent on what it did not need to buy
/// is what this asserts against; the two values are how it showed.
#[test]
fn the_universe_separates_a_pair_inside_a_shape() {
    let pool = const_pool();
    let defs = fixpoint_defs();
    let variadic = |element: Schema| Schema::Seq {
        container: SeqKind::Tuple,
        shape: SeqShape::homogeneous(element),
    };
    let floats = Schema::Seq {
        container: SeqKind::List,
        shape: SeqShape::homogeneous(Schema::Float),
    };
    let a = Schema::Seq {
        container: SeqKind::List,
        shape: SeqShape::fixed([variadic(Schema::ANYTHING), floats]),
    };
    for excluded in [Schema::NoneType, Schema::Bool, Schema::Int, Schema::Str] {
        let b = Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::fixed([
                variadic(Schema::Complement(Arc::new(excluded.clone()))),
                Schema::Ref(DefIx::new(1)),
            ]),
        };
        // The premise the law carries: a row whose pair is not refuted asserts
        // nothing about the universe.
        assert_eq!(
            a.subtype_relation_under(&b, &NoLeafRelations, &defs),
            Relation::Fails,
            "the pair excluding {excluded:?} is not refuted"
        );
        let subject = unfold_for_oracle(&a, &defs, ORACLE_UNFOLDS);
        let other = unfold_for_oracle(&b, &defs, ORACLE_UNFOLDS);
        assert!(
            boundary_values(&[&subject, &other]).iter().any(|value| {
                member_full(&subject, value, &pool) && !member_full(&other, value, &pool)
            }),
            "no value of the universe refutes the pair excluding {excluded:?}"
        );
    }
}

/// A union contributes a value of *each* branch, not sixty-four of the first.
///
/// The pair: `list[float] | {"b": anything, str => nothing}` against the
/// fixpoint `μt. int | list[¬t]`. The refutation is the record branch -- the
/// map `{"b": 1}` is a member of the union and is neither an `int` nor a list,
/// so it is outside the fixpoint -- and the sequence branch has no witness at
/// all, because every list of floats the universe spells is a value of the
/// fixpoint too.
///
/// The universe held sixty-four values of the union and not one map. A node's
/// cap was read off the branches **concatenated**, so the sequence branch spent
/// it before the record branch contributed anything, and the record's own
/// edges -- which lead with exactly the witness -- never reached the universe.
/// The record spent part of its own cap twice over the same map, which is the
/// second half of the same cause.
#[test]
fn a_union_contributes_a_value_of_every_branch() {
    let pool = const_pool();
    let defs = fixpoint_defs();
    let a = union(
        Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::homogeneous(Schema::Float),
        },
        Schema::keyed_map(
            vec![Field {
                name: "b".into(),
                schema: Schema::ANYTHING,
                required: true,
            }],
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Nothing,
            }],
        ),
    );
    let b = Schema::Ref(DefIx::new(1));
    // The premise the law carries: a pair that is not refuted asserts nothing.
    assert_eq!(
        a.subtype_relation_under(&b, &NoLeafRelations, &defs),
        Relation::Fails,
        "the pair is not refuted"
    );
    let subject = unfold_for_oracle(&a, &defs, ORACLE_UNFOLDS);
    let other = unfold_for_oracle(&b, &defs, ORACLE_UNFOLDS);
    assert!(
        boundary_values(&[&subject, &other]).iter().any(|value| {
            member_full(&subject, value, &pool) && !member_full(&other, value, &pool)
        }),
        "no value of the universe refutes the pair"
    );
}

/// A length bound is separated by a value of the shape it narrows.
///
/// The pair: `list[set[Any]]` against `list[Any]` of at most two. It is refuted
/// by a **three-element list of sets** -- a member of the subject, one element
/// too long for the bound -- and nothing else separates them, because every
/// shorter list of sets is a value of both.
///
/// Neither side built one. The bound makes its widths out of a fixed item, so
/// the universe held three-element lists of *integers*, which are not members
/// of the subject. The subject reaches two elements past its own prefix, which
/// is what tells a tailed shape from a fixed one and is one short of what tells
/// it from a bound of two. The value has to be crossed: the width from the
/// bound, the element from the schema it must be a member of.
#[test]
fn a_length_bound_is_separated_by_the_shape_it_narrows() {
    let pool = const_pool();
    let defs = fixpoint_defs();
    let a = Schema::Seq {
        container: SeqKind::List,
        shape: SeqShape::homogeneous(Schema::Coll {
            container: CollKind::Set,
            element: Arc::new(Schema::ANYTHING),
        }),
    };
    let b = Schema::Refine {
        base: Arc::new(Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::homogeneous(Schema::ANYTHING),
        }),
        constraints: vec![Constraint::MaxLen(2)].into(),
    };
    assert_eq!(
        a.subtype_relation_under(&b, &NoLeafRelations, &defs),
        Relation::Fails,
        "the pair is not refuted"
    );
    let subject = unfold_for_oracle(&a, &defs, ORACLE_UNFOLDS);
    let other = unfold_for_oracle(&b, &defs, ORACLE_UNFOLDS);
    assert!(
        boundary_values(&[&subject, &other]).iter().any(|value| {
            member_full(&subject, value, &pool) && !member_full(&other, value, &pool)
        }),
        "no value of the universe refutes the pair"
    );
}

/// Every value of `len` items, in each container a length is read from.
fn containers_of(len: usize, out: &mut Vec<Obj>) {
    let items = vec![Obj::Int(1); len];
    let entries = NAMES
        .iter()
        .take(len)
        .map(|name| (*name, Obj::Int(1)))
        .collect();
    out.push(Obj::Str(SIZED[len.min(SIZED.len() - 1)]));
    out.push(Obj::List(items.clone()));
    out.push(Obj::Tuple(items.clone()));
    out.push(Obj::Set(items.clone()));
    out.push(Obj::FrozenSet(items));
    out.push(Obj::Map(entries));
}

/// The numbers on both sides of a bound's operand, and the floats beside them.
fn numbers_around(index: OperandIx, out: &mut Vec<Obj>) {
    let Some(operand) = as_num(&const_pool()[index.get()]) else {
        return;
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "every pool operand is a small integer or a bool"
    )]
    let whole = operand as i64;
    for step in -1..=1 {
        out.push(Obj::Int(whole.saturating_add(step)));
    }
    out.push(Obj::Float(operand));
    out.push(Obj::Float(operand + 0.5));
    out.push(Obj::Float(operand * 2.0));
}

/// The length a constraint asks a container for, where it asks for one.
fn length_wanted(constraint: &Constraint) -> Option<usize> {
    match constraint {
        Constraint::MinLen(len) | Constraint::MaxLen(len) => Some(*len),
        _ => None,
    }
}

/// Containers of `len` members of the schema's element, in every shape a length
/// is read from.
///
/// A set holds its members once, so its `len` members are *distinct*: padding
/// one with a repeat would claim a value the interpreter reads as shorter. Where
/// the element cannot supply that many, there is no such set and none is built.
fn sized_like(schema: &Schema, len: usize, fuel: usize) -> Vec<Obj> {
    let element = match schema {
        Schema::Seq { shape, .. } => shape.elements().next().unwrap_or(&Schema::ANYTHING),
        Schema::Coll { element, .. } => element,
        _ => &Schema::ANYTHING,
    };
    let pool = const_pool();
    let mut distinct: Vec<Obj> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    // A series per scalar kind, because a set of three `bytes` needs three byte
    // strings and a fixed corpus carries two: the kinds with more values than a
    // bound names have to supply them on demand.
    let series = (0..len.min(SIZED.len())).flat_map(|index| {
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_wrap,
            reason = "the index is a small count, exact in both carriers"
        )]
        let (whole, float) = (index as i64, index as f64);
        [
            Obj::Int(whole),
            Obj::Float(float),
            Obj::Str(SIZED[index]),
            Obj::Bytes(index),
        ]
    });
    for candidate in a_member(element, fuel.saturating_sub(1))
        .into_iter()
        .chain(series)
        .chain(kind_edges())
    {
        let key = format!("{candidate:?}");
        if !seen.contains(&key) && member_full(element, &candidate, &pool) {
            seen.push(key);
            distinct.push(candidate);
        }
        if distinct.len() == len {
            break;
        }
    }
    let Some(first) = distinct.first().cloned() else {
        return Vec::new(); // an element with no member builds no container
    };
    // A sequence repeats, so one member fills every position. A set does not,
    // so it is built only where the element supplied that many values that
    // differ -- which is the same reading the emptiness decision takes.
    let repeated = vec![first.clone(); len];
    let mut built = vec![
        Obj::Str(SIZED[len.min(SIZED.len() - 1)]),
        Obj::Bytes(len),
        Obj::List(repeated.clone()),
        Obj::Tuple(repeated),
        Obj::Map(
            NAMES
                .iter()
                .take(len)
                .map(|name| (*name, first.clone()))
                .collect(),
        ),
    ];
    if distinct.len() == len {
        built.push(Obj::Set(distinct.clone()));
        built.push(Obj::FrozenSet(distinct));
    }
    built
}

/// How deep a constructed member is built before the attempt is given up.
const MEMBER_FUEL: usize = 4;

/// A value of the schema, built from the schema's own denotation.
///
/// The universe's other half. The kind edges and the derived boundaries are
/// values a *reader* would think of; a refutation, though, asserts that a value
/// of the subject lies outside the other schema, and the subject may be a shape
/// no fixed corpus spells -- a list of a set of a record. So each subterm is
/// asked for one member of itself, and what it answers joins the universe.
///
/// `None` is honest rather than convenient: an empty schema has no member, and
/// a leaf the oracle does not model (a class, a reference, an attribute record)
/// is one this cannot build. A law reads the answer as "no witness here", never
/// as "no witness exists".
fn a_member(schema: &Schema, fuel: usize) -> Option<Obj> {
    if fuel == 0 {
        return None;
    }
    let pool = const_pool();
    match schema {
        // The top is answered by any value at all, and an integer is the
        // shortest one to read in a failure message.
        Schema::Anything(_) | Schema::Int => Some(Obj::Int(1)),
        Schema::NoneType => Some(Obj::None),
        Schema::Bool => Some(Obj::Bool(true)),
        Schema::Float => Some(Obj::Float(1.0)),
        Schema::Str => Some(Obj::Str("a")),
        Schema::Bytes => Some(Obj::Bytes(1)),
        Schema::Literal(index) => Some(pool[index.get()].clone()),
        Schema::Seq { container, shape } => {
            let mut items = shape
                .prefix
                .iter()
                .map(|element| a_member(element, fuel - 1))
                .collect::<Option<Vec<Obj>>>()?;
            if let Some(tail) = shape.tail.as_deref() {
                // One element past the prefix where the tail admits one, so a
                // length bound over the shape has a value on the long side.
                items.extend(a_member(tail, fuel - 1));
            }
            Some(match container {
                SeqKind::List => Obj::List(items),
                SeqKind::Tuple => Obj::Tuple(items),
            })
        }
        Schema::Coll {
            container, element, ..
        } => {
            let items = a_member(element, fuel - 1).map_or_else(Vec::new, |item| vec![item]);
            Some(match container {
                CollKind::Set => Obj::Set(items),
                CollKind::FrozenSet => Obj::FrozenSet(items),
            })
        }
        Schema::KeyedMap { fields, .. } => Some(Obj::Map(
            fields
                .iter()
                .filter(|field| field.required)
                .map(|field| {
                    Some((
                        static_name(&field.name)?,
                        a_member(&field.schema, fuel - 1)?,
                    ))
                })
                .collect::<Option<Vec<(&'static str, Obj)>>>()?,
        )),
        Schema::Union(members) => members.iter().find_map(|member| a_member(member, fuel - 1)),
        // A meet and a complement are asked of the candidates rather than
        // constructed: the oracle decides membership, so a value that answers
        // is a member and one that does not is dropped.
        Schema::Intersection(members) => members
            .iter()
            .filter_map(|member| a_member(member, fuel - 1))
            .chain(kind_edges())
            .find(|value| member_full(schema, value, &pool)),
        // A refinement is asked of candidates rather than constructed: the
        // base's own member, that member grown to each length a constraint
        // names, and the kind edges. A length bound is the constraint a fixed
        // corpus cannot answer -- a set of two is no kind's edge -- so it is
        // the one the candidates are built for.
        Schema::Refine { base, constraints } => {
            let mut candidates: Vec<Obj> = a_member(base, fuel - 1).into_iter().collect();
            for length in constraints.iter().filter_map(length_wanted) {
                candidates.extend(sized_like(base, length, fuel));
            }
            candidates.extend(kind_edges());
            candidates
                .into_iter()
                .find(|value| member_full(schema, value, &pool))
        }
        Schema::Complement(_) => kind_edges()
            .into_iter()
            .find(|value| member_full(schema, value, &pool)),
        // The empty set has no member, and a leaf this does not model is one
        // no member can be built for. A law reads either as "no witness here".
        Schema::Nothing
        | Schema::Ref(_)
        | Schema::SelfRef(_)
        | Schema::Instance(_)
        | Schema::AttrRecord { .. } => None,
    }
}

/// The values every container arm is probed with, whatever its element says.
///
/// One per kind at each small width, so a container of a schema and a container
/// of its complement are told apart by an element that belongs to one of them.
///
/// Three kinds were missing outright, and a position excluding tuples is
/// separated only by a tuple: a refutation about such a position read as "no
/// value refutes it" while the value existed. The widths matter one step in,
/// because an empty tuple is outside `tuple[Any, Any]` as surely as an integer
/// is.
fn probes() -> Vec<Obj> {
    // Round-robin over the two groups, for the reason [`interleaved`] gives: a
    // cap read over a concatenation spends itself on whichever came first.
    //
    // The sized containers ran ahead of the scalars, on the argument that a
    // scalar is separated by a value `kind_edges` already holds and a
    // two-element tuple is not. That holds of a position at the top and fails
    // of one inside a shape: `kind_edges` carrying `None` does not put
    // `(None,)` in the universe, and a variadic tuple over the top is
    // separated from one over the complement of `None` by exactly that value.
    // Both groups are wanted at every depth, so the cap cuts them equally.
    let mut sized = Vec::new();
    for width in 0..=2 {
        let items = vec![Obj::Int(1); width];
        sized.push(Obj::Tuple(items.clone()));
        sized.push(Obj::List(items.clone()));
        sized.push(Obj::Set(items.clone()));
        sized.push(Obj::FrozenSet(items));
    }
    let scalars = vec![
        Obj::None,
        Obj::Bool(true),
        Obj::Int(1),
        Obj::Float(1.0),
        Obj::Str("a"),
        Obj::Bytes(0),
        Obj::Bytes(1),
        Obj::Map(vec![]),
    ];
    interleaved(&[sized, scalars])
}

/// How many values one node contributes, so a nested shape stays bounded.
const CAP_PER_NODE: usize = 64;
/// How many values the whole universe holds.
const CAP_UNIVERSE: usize = 768;

fn hold(container: SeqKind, items: Vec<Obj>) -> Obj {
    match container {
        SeqKind::List => Obj::List(items),
        SeqKind::Tuple => Obj::Tuple(items),
    }
}

fn gather(container: CollKind, items: Vec<Obj>) -> Obj {
    match container {
        CollKind::Set => Obj::Set(items),
        CollKind::FrozenSet => Obj::FrozenSet(items),
    }
}

/// The values one schema puts a boundary at: its own, and its subterms' lifted
/// into it.
///
/// Composed bottom-up, which is the half a flat corpus cannot have. A list of
/// anything and a list of everything-but-`None` are separated by `[None]` and
/// by nothing else, so a container arm builds itself around each value its
/// element schemas contributed -- and around a fixed probe per kind, so the
/// separation survives an element the oracle builds no member of. Every node is
/// capped: a shape three deep would otherwise multiply its levels together.
fn edges_of(schema: &Schema, out: &mut Vec<Obj>) -> Vec<Obj> {
    let mut mine: Vec<Obj> = a_member(schema, MEMBER_FUEL).into_iter().collect();
    match schema {
        Schema::Literal(index) => mine.push(const_pool()[index.get()].clone()),
        Schema::Refine { base, constraints } => {
            mine.extend(edges_of(base, out));
            for constraint in constraints.iter() {
                match constraint {
                    Constraint::Ge(index)
                    | Constraint::Gt(index)
                    | Constraint::Le(index)
                    | Constraint::Lt(index)
                    | Constraint::MultipleOf(index) => numbers_around(*index, &mut mine),
                    Constraint::MinLen(len) | Constraint::MaxLen(len) => {
                        for width in len.saturating_sub(EDGE_SPAN)..=(len + EDGE_SPAN) {
                            containers_of(width, &mut mine);
                        }
                    }
                    Constraint::Predicate(_) | Constraint::Regex(_) => {}
                }
            }
        }
        Schema::Seq { container, shape } => {
            mine.extend(sequence_edges(*container, shape, out));
        }
        Schema::Coll {
            container, element, ..
        } => {
            let elements = candidates(element, out);
            mine.push(gather(*container, vec![]));
            mine.extend(
                elements
                    .into_iter()
                    .map(|item| gather(*container, vec![item])),
            );
        }
        Schema::KeyedMap { fields, defaults } => {
            mine.extend(record_edges(fields, defaults, out));
        }
        Schema::Union(members) | Schema::Intersection(members) => {
            // Round-robin rather than concatenated, for the reason
            // `boundary_values` reads its own sources that way: a cap taken off
            // a concatenation is spent on whichever branch came first. A union
            // of a sequence beside a record contributed sixty-four lists and no
            // map, and the map was the only value that told the union from a
            // fixpoint over lists -- so the refutation had no witness and the
            // branch that held one had not been asked.
            let branches: Vec<Vec<Obj>> =
                members.iter().map(|member| edges_of(member, out)).collect();
            mine.extend(interleaved(&branches));
        }
        // A complement is separated from its inner set by the inner set's own
        // values: one of them is in exactly one of the two.
        Schema::Complement(inner) => mine.extend(edges_of(inner, out)),
        _ => {}
    }
    mine.truncate(CAP_PER_NODE);
    out.extend(mine.iter().cloned());
    mine
}

/// Round-robin over the groups, so a cap cuts each of them equally.
///
/// A record's second field and a sequence's second position are where a pair is
/// separated as often as the first, and a cap read over a concatenation spends
/// itself on whichever came first.
fn interleaved(groups: &[Vec<Obj>]) -> Vec<Obj> {
    let longest = groups.iter().map(Vec::len).max().unwrap_or(0);
    (0..longest)
        .flat_map(|index| {
            groups
                .iter()
                .filter_map(move |group| group.get(index).cloned())
        })
        .collect()
}

/// The values a position is probed with: the schema's own, then one per kind.
///
/// The schema's own come first because they are what separates it from the
/// schema beside it -- a set where the other admits `None` -- and a cap that cut
/// them in favour of the fixed probes would leave the pair undecided by the
/// universe. The probes follow for the schema whose own values are thin: the top
/// builds one member and is separated from a complement by any of the others.
fn candidates(schema: &Schema, out: &mut Vec<Obj>) -> Vec<Obj> {
    let before = out.len();
    let mut mine = deduplicated(edges_of(schema, out));
    // A quarter of a node's budget, so the two sources that follow are inside
    // every cap a caller of this applies: a position probed with the schema's
    // own values alone is one no `bool` reaches where the schema is an `int`.
    mine.truncate(CAP_PER_NODE / 4);
    // What the rest of the comparison contributed, most recent first.
    //
    // This is the half a fixed probe list cannot supply. A position is
    // separated from the *other* schema's position by a value of the shape
    // that one names, and no list written here holds it: `tuple[Any, Float]`
    // is refuted by a two-element tuple whose second item is a float, and the
    // next pair asks for a different shape again. The values are already built
    // -- the other schema contributed them when its own edges were read -- so
    // a position is let see them rather than the list guessing wider.
    let wanted = CAP_PER_NODE / 4;
    // Strided rather than taken from either end: the pool is ordered by the
    // node that contributed, so its tail is one schema's outermost values and
    // its head is the fixed kind edges. What separates a position sits in the
    // middle, where another schema's *inner* nodes put theirs.
    let step = (before / wanted).max(1);
    let inherited: Vec<Obj> = out[..before]
        .iter()
        .rev()
        .step_by(step)
        .take(wanted)
        .cloned()
        .collect();
    // Round-robin rather than concatenated, for the reason `interleaved` gives:
    // a cap read over a concatenation spends itself on whichever came first,
    // and the three sources separate a pair about equally often. Deduplicated
    // because the three overlap -- the pool the inherited values are strided
    // from already holds this node's own -- and a caller caps what comes back.
    deduplicated(interleaved(&[mine, probes(), inherited]))
}

/// The sequences one shape puts a boundary at: its own member with each
/// position probed in turn, and a container at each side of its width.
///
/// Probing a position inside the shape's own member is what a flat corpus
/// cannot do. `[Any, X]` against `[Any, int]` is separated only by a two-element
/// list whose *second* element is an `X` outside `int`, and a corpus of
/// single-element containers holds no such value whatever it holds at depth one.
fn sequence_edges(container: SeqKind, shape: &SeqShape, out: &mut Vec<Obj>) -> Vec<Obj> {
    let mut mine = Vec::new();
    let base: Option<Vec<Obj>> = shape
        .prefix
        .iter()
        .map(|element| a_member(element, MEMBER_FUEL))
        .collect();
    let mut elements: Vec<Obj> = shape
        .elements()
        .flat_map(|element| candidates(element, out))
        .collect();
    elements.truncate(CAP_PER_NODE);
    let mut groups: Vec<Vec<Obj>> = Vec::new();
    for (position, element) in shape.prefix.iter().enumerate() {
        let Some(base) = base.clone() else {
            break; // a position with no member of its own builds no sequence
        };
        groups.push(
            candidates(element, out)
                .into_iter()
                .map(|value| {
                    let mut items = base.clone();
                    items[position] = value;
                    hold(container, items)
                })
                .collect(),
        );
    }
    mine.extend(interleaved(&groups));
    if let (Some(base), Some(tail)) = (base.clone(), shape.tail.as_deref()) {
        // The prefix, and one and two elements past it. That is where a tail
        // and a fixed width are told apart, and where a length bound over the
        // shape is decided.
        //
        // Two, not one. A tailed shape is separated from a *fixed* one of
        // width n by a value of width n whose elements are the tail's, and one
        // element past the prefix reaches n = 1 only: `list[float, ...]`
        // against the complement of `list[Any, Any]` is refuted by a
        // two-element list of floats, and the universe held every one-element
        // list and no two-element one.
        mine.push(hold(container, base.clone()));
        for value in candidates(tail, out) {
            let mut items = base.clone();
            items.push(value.clone());
            mine.push(hold(container, items.clone()));
            items.push(value);
            mine.push(hold(container, items));
        }
    }
    mine.extend(
        elements
            .into_iter()
            .map(|element| hold(container, vec![element])),
    );
    let width = shape.prefix.len();
    for len in width.saturating_sub(EDGE_SPAN)..=(width + EDGE_SPAN) {
        containers_of(len, &mut mine);
    }
    mine
}

/// The mappings one record puts a boundary at: the three widths its own clauses
/// separate, and each field and default clause probed under a key it governs.
fn record_edges(fields: &Fields, defaults: &Clauses, out: &mut Vec<Obj>) -> Vec<Obj> {
    // Every probe sits in a mapping that carries the record's required fields,
    // because a mapping missing one is refused before the probed entry is read.
    let base: Vec<(&'static str, Obj)> = fields
        .iter()
        .filter(|field| field.required)
        .filter_map(|field| {
            Some((
                static_name(&field.name)?,
                a_member(&field.schema, MEMBER_FUEL)?,
            ))
        })
        .collect();
    let with = |key: &'static str, value: Obj| {
        let mut entries: Vec<(&'static str, Obj)> = base
            .iter()
            .filter(|(name, _)| *name != key)
            .cloned()
            .collect();
        entries.push((key, value));
        Obj::Map(entries)
    };
    let mut mine = Vec::new();
    let mut groups: Vec<Vec<Obj>> = Vec::new();
    for field in fields.iter() {
        let Some(name) = static_name(&field.name) else {
            continue;
        };
        groups.push(
            candidates(&field.schema, out)
                .into_iter()
                .map(|value| with(name, value))
                .collect(),
        );
    }
    // A default clause governs the keys no field declares, so its probes go
    // under names this record does not declare: under a declared one the field
    // decides and the clause is never read. More than one name, because the key
    // that separates this record from another is one *neither* declares -- a
    // record of no fields would otherwise probe under the name the other one
    // declares, and the value it builds would be admitted by both.
    let undeclared: Vec<&'static str> = NAMES
        .into_iter()
        .chain(["z"])
        .filter(|name| !fields.iter().any(|field| &*field.name == *name))
        .take(2)
        .collect();
    for clause in defaults.iter() {
        for key in &undeclared {
            groups.push(
                candidates(&clause.value, out)
                    .into_iter()
                    .map(|value| with(key, value))
                    .collect(),
            );
        }
    }
    mine.extend(interleaved(&groups));
    // The three widths a record's own clauses separate: exactly the declared
    // names, one short of them, and one past them.
    let declared: Vec<(&'static str, Obj)> = fields
        .iter()
        .filter_map(|field| Some((static_name(&field.name)?, Obj::Int(1))))
        .collect();
    let mut short = declared.clone();
    short.pop();
    let mut wide = declared.clone();
    wide.push(("z", Obj::Int(1)));
    mine.push(Obj::Map(declared));
    mine.push(Obj::Map(short));
    mine.push(Obj::Map(wide));
    mine
}

/// The definitions the soundness laws draw against.
///
/// Two fixpoints: one with a word branch, whose unfolding has to keep the two
/// word kinds apart, and one whose reference sits **under a complement**, which
/// is where the polarity of an unfolding decides whether a difference is sound.
fn fixpoint_defs() -> Vec<Schema> {
    let list_of = |element: Schema| Schema::Seq {
        container: SeqKind::List,
        shape: SeqShape::homogeneous(element),
    };
    vec![
        union(Schema::Str, list_of(Schema::Ref(DefIx::new(0)))),
        union(Schema::Int, list_of(not(Schema::Ref(DefIx::new(1))))),
    ]
}

/// How far a reference is unfolded before the value oracle reads it.
///
/// Every value of the universe is shallower than this, so a membership answer
/// is exact whatever sits at the cut: the walk reaches the cut only where the
/// value is deeper than the unfolding, and no value here is.
const ORACLE_UNFOLDS: usize = 6;

/// A schema with its references resolved, for the oracle to read.
///
/// The decision procedure is asked about the schema *with* its definitions;
/// the oracle is a transcription of the denotation and has no rule for a
/// cycle, so it reads an unfolding deep enough that the corpus cannot tell the
/// two apart.
fn unfold_for_oracle(schema: &Schema, defs: &[Schema], fuel: usize) -> Schema {
    match schema {
        Schema::Ref(index) => match defs.get(index.get()) {
            Some(body) if fuel > 0 => unfold_for_oracle(body, defs, fuel - 1),
            // Past the fuel, and an unresolved reference: no value of the
            // corpus reaches either, so the empty set is the exact answer.
            _ => Schema::Nothing,
        },
        Schema::Seq { container, shape } => Schema::Seq {
            container: *container,
            shape: SeqShape {
                prefix: shape
                    .prefix
                    .iter()
                    .map(|element| unfold_for_oracle(element, defs, fuel))
                    .collect(),
                tail: shape
                    .tail
                    .as_deref()
                    .map(|tail| Arc::new(unfold_for_oracle(tail, defs, fuel))),
            },
        },
        Schema::Coll { container, element } => Schema::Coll {
            container: *container,
            element: Arc::new(unfold_for_oracle(element, defs, fuel)),
        },
        Schema::KeyedMap { fields, defaults } => Schema::KeyedMap {
            fields: fields
                .iter()
                .map(|field| Field {
                    name: Arc::clone(&field.name),
                    schema: unfold_for_oracle(&field.schema, defs, fuel),
                    required: field.required,
                })
                .collect(),
            defaults: defaults
                .iter()
                .map(|clause| MapClause {
                    key: unfold_for_oracle(&clause.key, defs, fuel),
                    value: unfold_for_oracle(&clause.value, defs, fuel),
                })
                .collect(),
        },
        Schema::Refine { base, constraints } => Schema::Refine {
            base: Arc::new(unfold_for_oracle(base, defs, fuel)),
            constraints: constraints.clone(),
        },
        Schema::Union(members) => Schema::Union(
            members
                .iter()
                .map(|member| unfold_for_oracle(member, defs, fuel))
                .collect(),
        ),
        Schema::Intersection(members) => Schema::Intersection(
            members
                .iter()
                .map(|member| unfold_for_oracle(member, defs, fuel))
                .collect(),
        ),
        Schema::Complement(inner) => {
            Schema::Complement(Arc::new(unfold_for_oracle(inner, defs, fuel)))
        }
        other => other.clone(),
    }
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
    let mut seen: Vec<std::sync::Arc<str>> = Vec::new();
    fields
        .into_iter()
        .filter(|field| {
            let fresh = !seen.contains(&field.name);
            if fresh {
                seen.push(std::sync::Arc::clone(&field.name));
            }
            fresh
        })
        .collect()
}

/// Whether no `complement` stands anywhere in the term.
///
/// The two laws that read the direction draw from `positive_schema`, so this is
/// their detector rather than their filter: it says the fragment is the one
/// they are about, and it fails if that stops being true.
fn under_no_complement(schema: &Schema) -> bool {
    !matches!(schema, Schema::Complement(_)) && schema.children().all(under_no_complement)
}

/// Whether a drawn term may carry a `complement`.
///
/// The projections rewrite the record they reach, and a negation between that
/// record and the top of the term reverses which way the whole schema moves.
/// Two laws are about the positive fragment and **draw** it rather than
/// filtering for it: `prop_assume!` spends a case on every rejected draw and
/// proptest stops a test that rejects more than a thousand, so at the nightly's
/// case count a filter fails the lane instead of stating the law.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Negation {
    Allowed,
    Absent,
}

/// Every decidable shape, a negated one among them.
fn decidable_schema() -> impl Strategy<Value = Schema> {
    schema_fragment(Negation::Allowed, Producer::Any)
}

/// The same fragment with no `complement` anywhere in it.
fn positive_schema() -> impl Strategy<Value = Schema> {
    schema_fragment(Negation::Absent, Producer::Any)
}

/// The fragment a caller can write: every refinement it draws is one the
/// frontend builds.
///
/// [`decidable_schema`] draws a bound over any base, which the fuzz target
/// reaches and the decision must be sound over, and which the frontend
/// refuses at build. A law about what the pages promise a caller -- an
/// equivalence decided, a split placed -- is stated over this fragment, since
/// a set the frontend refuses to name is not one the pages promise to decide.
fn buildable_schema() -> impl Strategy<Value = Schema> {
    schema_fragment(Negation::Allowed, Producer::Frontend)
}

/// Who a drawn refinement is held to.
#[derive(Clone, Copy)]
enum Producer {
    /// Any producer of the IR: a bound over any base, as the fuzz target draws.
    Any,
    /// The frontend: a bound only over a base it accepts the bound on.
    Frontend,
}

/// Whether the frontend builds `constraint` over `base`.
///
/// The rule `build/refine.rs` applies at build, read for the bases this
/// generator draws: an order bound wants a base whose values compare with the
/// pool's operands, which are numbers, so a numeric kind; a divisor the same;
/// a length wants a sized kind. A base that is a union of such kinds is read
/// as its members, since the frontend reads it that way. Held in step with
/// the frontend by reading rather than by a test, which is the limit of a
/// generator that lives in the core.
fn frontend_builds(base: &Schema, constraint: &Constraint) -> bool {
    fn numeric(base: &Schema) -> bool {
        match base {
            Schema::Int | Schema::Float | Schema::Bool => true,
            Schema::Union(members) => members.iter().all(numeric),
            _ => false,
        }
    }
    fn sized(base: &Schema) -> bool {
        match base {
            Schema::Str
            | Schema::Bytes
            | Schema::Seq { .. }
            | Schema::Coll { .. }
            | Schema::KeyedMap { .. } => true,
            Schema::Union(members) => members.iter().all(sized),
            _ => false,
        }
    }
    match constraint {
        Constraint::Ge(_)
        | Constraint::Gt(_)
        | Constraint::Le(_)
        | Constraint::Lt(_)
        | Constraint::MultipleOf(_) => numeric(base),
        Constraint::MinLen(_) | Constraint::MaxLen(_) => sized(base),
        Constraint::Predicate(_) | Constraint::Regex(_) => false,
    }
}

/// `base` narrowed by the constraints the producer builds over it.
///
/// Under the frontend a constraint it refuses over this base is dropped, and
/// a refinement left with none is its base: the constructors fold that
/// wrapper away and the frontend never builds it.
fn refined(base: Schema, constraints: Vec<Constraint>, producer: Producer) -> Schema {
    let constraints: Vec<Constraint> = match producer {
        Producer::Any => constraints,
        Producer::Frontend => constraints
            .into_iter()
            .filter(|constraint| frontend_builds(&base, constraint))
            .collect(),
    };
    if constraints.is_empty() {
        base
    } else {
        Schema::Refine {
            base: Arc::new(base),
            constraints: constraints.into(),
        }
    }
}

fn schema_fragment(negation: Negation, producer: Producer) -> impl Strategy<Value = Schema> {
    // `Copy`, and moved rather than borrowed: `prop_recursive` takes a closure
    // that outlives this frame.
    let leaf = prop_oneof![
        Just(Schema::ANYTHING),
        Just(Schema::Nothing),
        Just(Schema::ANY),
        Just(Schema::NoneType),
        Just(Schema::Bool),
        Just(Schema::Int),
        Just(Schema::Float),
        Just(Schema::Str),
        Just(Schema::Bytes),
        (0usize..POOL_LEN).prop_map(|i| Schema::Literal(ConstIx::new(i))),
    ];
    leaf.prop_recursive(3, 48, 4, move |inner| {
        let field =
            (0usize..2, inner.clone(), proptest::bool::ANY).prop_map(|(n, schema, req)| Field {
                name: ["a", "b"][n].into(),
                schema,
                required: req,
            });
        // An open record: 0..2 declared fields plus a `Str -> value` default
        // arm, so `member_full`'s `defaults` branch is actually reached.
        let open_record = (
            proptest::collection::vec(field.clone(), 0..2),
            inner.clone(),
        )
            .prop_map(|(fields, value)| Schema::KeyedMap {
                fields: unique_by_name(fields).into(),
                defaults: vec![MapClause {
                    key: Schema::Str,
                    value,
                }]
                .into(),
            });
        // Two clauses claiming one key: a literal-keyed clause over the `"a"`
        // constant beside the `Str` clause, which the walk reads as the
        // disjunction of the two and the oracle transcribes the same way.
        let two_clauses =
            (inner.clone(), inner.clone()).prop_map(|(named, rest)| Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![
                    MapClause {
                        key: Schema::Literal(ConstIx::new(3)),
                        value: named,
                    },
                    MapClause {
                        key: Schema::Str,
                        value: rest,
                    },
                ]
                .into(),
            });
        let positive = prop_oneof![
            inner.clone().prop_map(Schema::set),
            inner.clone().prop_map(Schema::frozen_set),
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
                .prop_map(move |(base, constraints)| refined(
                    base,
                    constraints,
                    producer
                )),
            // A closed record. The names are deduplicated: a record with two
            // fields of one name is an IR the frontend refuses to build, and
            // the decision procedure reads the field index on the invariant
            // that it does.
            proptest::collection::vec(field, 0..3).prop_map(|fields| Schema::KeyedMap {
                fields: unique_by_name(fields).into(),
                defaults: vec![].into(),
            }),
            open_record,
            two_clauses,
            proptest::collection::vec(inner.clone(), 1..3).prop_map(|m| Schema::Union(m.into())),
            proptest::collection::vec(inner.clone(), 1..3)
                .prop_map(|m| Schema::Intersection(m.into())),
        ];
        match negation {
            Negation::Absent => positive.boxed(),
            // One arm in thirteen, which is the weight it carries where the
            // thirteen are spelled out as alternatives of one another.
            Negation::Allowed => prop_oneof![
                12 => positive,
                1 => inner.prop_map(|s| Schema::Complement(Arc::new(s))),
            ]
            .boxed(),
        }
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
    let frozen_int = Schema::frozen_set(Schema::Int);
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
        fields: vec![].into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }]
        .into(),
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
        for value in &boundary_values(&[&schema]) {
            prop_assert_eq!(
                member_full(&simplified, value, &pool),
                member_full(&schema, value, &pool),
                "simplify changed membership of {:?}", value
            );
        }
    }

    // THEORY: subtyping-is-inclusion
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
        let universe = boundary_values(&[&a, &b]);
        prop_assert!(a.is_subtype_of(&a), "reflexivity");
        if a.is_subtype_of(&b) {
            for value in &universe {
                prop_assert!(
                    !member_full(&a, value, &pool) || member_full(&b, value, &pool),
                    "{:?} is in the subtype and not in the supertype", value
                );
            }
        }
        if a.is_empty() {
            for value in &universe {
                prop_assert!(
                    !member_full(&a, value, &pool),
                    "{:?} is a member of a schema decided empty", value
                );
            }
        }
    }

    // THEORY: open-and-close-read-the-region
    /// Equal sets close to equal sets, whatever the term is written out of.
    ///
    /// This is what puts `close` in the algebra rather than beside it: an
    /// operator on *sets* must send two spellings of one set to one set, and
    /// openness is the default of the key-type region no clause claims -- a
    /// region, which is a set of keys rather than a way of writing one.
    ///
    /// `open` has no such law and is not missing one. `test_projection_laws.py`
    /// carries the pair that separates them (`{"a?": int}` against
    /// `{} | {"a": int}`, which open into two sets), so the silence here is
    /// deliberate.
    ///
    /// The respelling is **absorption**, `a | (a & b)`, and the choice is the
    /// whole worth of the law: `a | (a & a)` reads like a respelling and is
    /// not one, because the constructors fold the meet to `a` and the union to
    /// `a` and the law then compares a term with itself. A law that cannot
    /// fail passes for the same reason a true one does.
    #[test]
    fn closing_is_a_function_of_the_set_however_it_is_spelled(
        a in decidable_schema(),
        b in decidable_schema(),
    ) {
        let pool = const_pool();
        // Asked of the drawn term *and* of its opened form. Closing is the
        // identity on every term this generator draws directly -- a record it
        // builds is closed already, or carries a typed clause that is not the
        // one an openness moves -- so a law asked only of those compares two
        // terms neither of which the operator touched.
        let opened = a.with_records_open(Openness::Open);
        for spelling in [&a, &opened] {
            let respelled =
                Schema::union([spelling.clone(), Schema::meet([spelling.clone(), b.clone()])]);
            let universe = boundary_values(&[spelling, &b, &respelled]);
            for value in &universe {
                prop_assert_eq!(
                    member_full(&respelled, value, &pool),
                    member_full(spelling, value, &pool),
                    "absorption is a different set at {:?}", value
                );
            }

            let closed = spelling.with_records_open(Openness::Closed);
            let respelled_closed = respelled.with_records_open(Openness::Closed);
            for value in &universe {
                prop_assert_eq!(
                    member_full(&closed, value, &pool),
                    member_full(&respelled_closed, value, &pool),
                    "two spellings of one set closed to two sets, parting at {:?}", value
                );
            }

            // And the pair the draw gives directly, which reaches spellings no
            // rewrite of one term produces: two terms decided equal must close
            // to one set as well.
            //
            // Decided, not sampled. Agreement over a finite spread of values is
            // not equality of sets, and this law read it as one: it drew
            // `union({str => anything})` against `{str => anything}`, found no
            // value in hand to part their *openings*, and then held their
            // closings to each other -- which is a claim about two different
            // sets. The relation is a proof, and soundness is held next door.
            if spelling.is_subtype_of(&b) && b.is_subtype_of(spelling) {
                let other = b.with_records_open(Openness::Closed);
                for value in &universe {
                    prop_assert_eq!(
                        member_full(&closed, value, &pool),
                        member_full(&other, value, &pool),
                        "two equal sets closed to two sets, parting at {:?}", value
                    );
                }
            }
        }
    }

    // THEORY: open-and-close-read-the-region
    /// `open` only widens, `close` only narrows, and closing an opened term is
    /// **at most** closing the term.
    ///
    /// At most, and not equal, which is the shape this law read as until a
    /// generator drew the term that parts them. `{"a"?: anything}` and `{}` are
    /// two sets and both open to every dict, so `open` is not injective and
    /// nothing `close` does can recover which of the two it was handed. The
    /// equality holds wherever the opened term keeps its declared names; a name
    /// a full catch-all makes redundant is dropped, because two spellings of
    /// one set must close to one set, and
    /// `an_optional_field_a_catch_all_already_says_is_dropped` carries the
    /// value that shows it.
    ///
    /// Drawn from the fragment with no complement in it, and the law below is
    /// the other half: under a negation the two operators swap, so "open admits
    /// more" is a claim about the record the transform rewrites and about the
    /// whole schema only where nothing negates it in between.
    #[test]
    fn opening_widens_closing_narrows_and_the_round_trip_is_at_most_closing(
        schema in positive_schema(),
    ) {
        // The detector: a fragment that started carrying a negation would make
        // both directions below read the other way round, and read green.
        prop_assert!(under_no_complement(&schema));
        let pool = const_pool();
        let opened = schema.with_records_open(Openness::Open);
        let closed = schema.with_records_open(Openness::Closed);
        let round_trip = opened.with_records_open(Openness::Closed);
        for value in &boundary_values(&[&schema, &opened, &closed]) {
            prop_assert!(
                !member_full(&schema, value, &pool) || member_full(&opened, value, &pool),
                "opening dropped {:?}", value
            );
            prop_assert!(
                !member_full(&closed, value, &pool) || member_full(&schema, value, &pool),
                "closing admitted {:?}, which the term does not", value
            );
            prop_assert!(
                !member_full(&round_trip, value, &pool) || member_full(&closed, value, &pool),
                "closing an opened term admitted {:?}, which closing it does not",
                value
            );
        }
    }

    // THEORY: open-and-close-read-the-region
    /// Under a negation the two operators swap, and the swap is exact.
    ///
    /// `open` rewrites the record it reaches whatever stands above it, so a
    /// complement over a widened set is a narrowed one. A caller reads the
    /// name and expects the schema to admit more; inside a `complement` it
    /// admits less, and nothing about the call site says so. The law is here
    /// so the direction is a property of the tree rather than a paragraph.
    #[test]
    fn opening_under_a_complement_narrows_and_closing_widens(
        schema in positive_schema(),
    ) {
        // As above: the negation this law is about is the one it writes.
        prop_assert!(under_no_complement(&schema));
        let pool = const_pool();
        let negated = schema.clone().complement();
        let opened = negated.with_records_open(Openness::Open);
        let closed = negated.with_records_open(Openness::Closed);
        for value in &boundary_values(&[&schema, &negated]) {
            prop_assert!(
                !member_full(&opened, value, &pool) || member_full(&negated, value, &pool),
                "opening under a complement admitted {:?}", value
            );
            prop_assert!(
                !member_full(&negated, value, &pool) || member_full(&closed, value, &pool),
                "closing under a complement dropped {:?}", value
            );
        }
    }

    // THEORY: recursive-subtyping, subtyping-is-inclusion
    /// A proof over a fixpoint has no witness against it either.
    ///
    /// The property above draws from a fragment with no reference in it, so
    /// every recursive pair the procedure accepts was held by hand-written
    /// examples. That is the fragment where an accept is *hardest* to earn:
    /// the rules assume the goal, unfold, and discharge it coinductively, so a
    /// proof rests on an assumption rather than on a finished derivation. An
    /// assumption discharged wrongly is an accept over a set that does not
    /// contain the other, which is unsoundness a caller reads as `subset`.
    ///
    /// The bodies are drawn rather than fixed, and the definitions table
    /// carries a **negative** occurrence -- a reference under a complement --
    /// because that is the shape where a fixpoint's polarity decides which cut
    /// an unfolding takes, and the shape a positive-only table never reaches.
    ///
    /// The oracle reads the schema unfolded past every value's depth, so a
    /// membership answer is exact whatever sits at the cut.
    #[test]
    fn a_proof_over_a_fixpoint_has_no_witness_against_it(
        a in recursive_schema(),
        b in recursive_schema(),
    ) {
        let pool = const_pool();
        let defs = fixpoint_defs();
        prop_assert!(
            a.is_subtype_of_under(&a, &NoLeafRelations, &defs),
            "reflexivity over {a:?}"
        );
        if a.is_subtype_of_under(&b, &NoLeafRelations, &defs) {
            let subject = unfold_for_oracle(&a, &defs, ORACLE_UNFOLDS);
            let other = unfold_for_oracle(&b, &defs, ORACLE_UNFOLDS);
            for value in &boundary_values(&[&subject, &other]) {
                prop_assert!(
                    !member_full(&subject, value, &pool)
                        || member_full(&other, value, &pool),
                    "{value:?} is in {a:?} and not in {b:?}, which is accepted"
                );
            }
        }
    }
}

/// The structural fragment with a reference into [`fixpoint_defs`] among its
/// leaves, so a drawn schema reaches a fixpoint with a word branch and one
/// whose reference sits under a complement.
fn recursive_schema() -> impl Strategy<Value = Schema> {
    prop_oneof![
        8 => decidable_schema(),
        1 => Just(Schema::Ref(DefIx::new(0))),
        1 => Just(Schema::Ref(DefIx::new(1))),
        2 => decidable_schema().prop_map(|schema| union(schema, Schema::Ref(DefIx::new(0)))),
        2 => decidable_schema().prop_map(|schema| Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape::fixed([schema, Schema::Ref(DefIx::new(1))]),
        }),
    ]
}

proptest! {
    // THEORY: subtyping-is-inclusion
    /// A refutation stands on a value, and the universe names it.
    ///
    /// `Fails` is the strongest answer either decider gives: it asserts that a
    /// value of the subject lies outside the other schema, and a caller reads
    /// it as `"not_subset"`. Every other property here checks a *proof* -- that
    /// an accepted inclusion admits no counterexample -- and a wrong refutation
    /// passes all of them, because nothing asks the refutation for its witness.
    ///
    /// The universe is the schemas' own: each subterm contributes a member of
    /// itself and each bound its two neighbours, so a shape no fixed corpus
    /// spells still has a value here. A failure is either a refutation with
    /// nothing under it or a universe too thin to hold the witness, and the
    /// message carries the pair so the two are told apart by reading it.
    #[test]
    fn a_refutation_is_a_value(a in recursive_schema(), b in recursive_schema()) {
        let pool = const_pool();
        let defs = fixpoint_defs();
        if a.subtype_relation_under(&b, &NoLeafRelations, &defs) == Relation::Fails {
            let subject = unfold_for_oracle(&a, &defs, ORACLE_UNFOLDS);
            let other = unfold_for_oracle(&b, &defs, ORACLE_UNFOLDS);
            prop_assert!(
                boundary_values(&[&subject, &other]).iter().any(|value| {
                    member_full(&subject, value, &pool) && !member_full(&other, value, &pool)
                }),
                "{:?} <= {:?} is refuted and no value of the universe refutes it",
                a, b
            );
        }
    }

    // THEORY: each-kind-is-closed
    /// An emptiness is refused by every value of the universe.
    ///
    /// The companion claim, on the same universe: `is_empty` asserts that *no*
    /// value belongs, which a single member refutes. Held over the recursive
    /// fragment too, where a fixpoint unfolded with the wrong polarity is the
    /// way an inhabited schema reads empty.
    #[test]
    fn an_emptiness_is_refused_by_every_boundary_value(a in recursive_schema()) {
        let pool = const_pool();
        let defs = fixpoint_defs();
        if a.is_empty_under(&defs) {
            let schema = unfold_for_oracle(&a, &defs, ORACLE_UNFOLDS);
            for value in &boundary_values(&[&schema]) {
                prop_assert!(
                    !member_full(&schema, value, &pool),
                    "{:?} is a member of {:?}, which is decided empty",
                    value, a
                );
            }
        }
    }
}

proptest! {
    /// A schema read as inhabited without a descent has a value, and the
    /// corpus holds it.
    ///
    /// Every refutation is read against the subject's own emptiness, at every
    /// level of one, so the shapes whose inhabitance is their own form answer
    /// without the descent the general fold takes: a scalar atom, a container
    /// that admits an empty one, a union with such a member. That reading is a
    /// *claim about values*, and the claim is checked against values rather
    /// than against the other reading -- the fold is conservative where this is
    /// exact (an empty closed record holds the empty mapping and the fold
    /// declines to say so), so agreement between the two would be the weaker
    /// property and would fail for the wrong reason.
    ///
    /// Over the generator the value corpus can decide, since the claim is
    /// checked by finding a value: an `Instance` is a class the corpus does not
    /// model, and a property that cannot look for a witness cannot make this
    /// claim.
    #[test]
    fn a_shallow_reading_of_a_value_names_one(a in decidable_schema()) {
        if a.holds_a_value_shallowly() {
            let pool = const_pool();
            prop_assert!(
                boundary_values(&[&a])
                    .iter()
                    .any(|value| member_full(&a, value, &pool)),
                "no value of the corpus is in a schema read as inhabited: {:?}", a
            );
            prop_assert_ne!(
                a.verdict(),
                Verdict::Empty,
                "the fold proves empty a schema read as inhabited: {:?}", a
            );
        }
    }

    /// A refutation about a part with no value is not one, whatever the part
    /// is compared against.
    ///
    /// The laws below draw their two schemas independently, so the shape this
    /// one is about -- a part with no value, inside a container, against the
    /// same container over something that part is not below -- is a
    /// coincidence they wait for, and waited many reports for. It is named
    /// here instead: whatever the other schema is, the subject denotes the
    /// *empty* container and nothing else, the supertype holds that container
    /// too, and the inclusion holds. A rule may decline it; no rule may refute
    /// it.
    #[test]
    fn a_refutation_about_a_part_with_no_value_is_not_one(
        part in without_a_value(),
        other in shaped_schema(),
        carrier in 0usize..4,
    ) {
        let wrap = carriers();
        let Some(wrap) = wrap.get(carrier) else {
            return Err(TestCaseError::reject("the carrier table is shorter than the draw"));
        };
        let (sub, sup) = (wrap(part.clone()), wrap(other.clone()));
        let budget = std::cell::Cell::new(DECISION_BUDGET);
        prop_assert_ne!(
            sub.subtype_relation(&sup, &CorpusOracle, &[], &budget),
            Relation::Fails,
            "a container of a part with no value is the empty container, which {:?} holds: {:?}",
            other,
            part
        );
    }

    /// A refuted relation is a claim, and the sets are the second opinion.
    ///
    /// The three-valued answer distinguishes a rule that *refutes* the
    /// inclusion from one that declines to decide it, which is only worth
    /// anything if the refutations are true. The descriptor decides the same
    /// question a different way, so where it says the inclusion holds, no rule
    /// may say it fails: one of the two would be wrong, and the rules are the
    /// half that just gained a way to be.
    #[test]
    fn a_refuted_inclusion_is_not_one_the_sets_decide(a in schema(), b in schema()) {
        let budget = std::cell::Cell::new(DECISION_BUDGET);
        if a.subtype_relation(&b, &NoLeafRelations, &[], &budget) == Relation::Fails {
            prop_assert!(
                a.descriptor_contained_in(&b, &NoLeafRelations, &[]) != Relation::Holds,
                "the rules refuted {a:?} <= {b:?} and the sets decide it holds"
            );
        }
    }

    // THEORY: the-descriptor
    /// Two spellings of one difference never give two different answers.
    ///
    /// `a ∧ ¬(b ∨ c)` and `a ∧ ¬b ∧ ¬c` are one set, by De Morgan. Every law
    /// above that touches the pair reads them against *values*, where the two
    /// are indistinguishable by construction; this reads the verdict, which is
    /// what a caller asking whether the difference is empty receives.
    ///
    /// **Equality of the two verdicts is not the law, and the reason is worth
    /// stating.** `Unknown` is a refusal, the two spellings reach the width
    /// bound through different intermediates, and the constructors fold one of
    /// them further -- `a ∧ ¬a ∧ ¬c` is `nothing` where `a ∧ ¬(a ∨ c)` is a
    /// term the sets have to decide. So one spelling deciding where the other
    /// declines is the bound and the folds, not a disagreement. What no
    /// spelling may do is *contradict* another: one proving the difference
    /// empty while another proves it inhabited would make at least one of them
    /// a wrong answer, and both are answers a caller acts on.
    ///
    /// The shapes where one spelling is the harder one by construction are rows
    /// of `tests/test_completeness_ledger.py` rather than draws, because a
    /// decided relation is a claim about a named pair.
    #[test]
    fn the_verdict_is_stable_under_de_morgan(
        a in decidable_schema(),
        b in decidable_schema(),
        c in decidable_schema(),
    ) {
        let joined = Schema::meet([
            a.clone(),
            Schema::union([b.clone(), c.clone()]).complement(),
        ]);
        let spelled = Schema::meet([a, b.complement(), c.complement()]);
        prop_assert!(
            !matches!(
                (joined.verdict(), spelled.verdict()),
                (Verdict::Empty, Verdict::Inhabited) | (Verdict::Inhabited, Verdict::Empty)
            ),
            "one set, two spellings, two answers: {:?} says {:?} and {:?} says {:?}",
            joined,
            joined.verdict(),
            spelled,
            spelled.verdict()
        );
    }

    /// Inclusion is transitive wherever the rules decide it.
    ///
    /// Inclusion is a preorder in the model -- it is set containment -- so two
    /// proofs must never meet a refutation. That is a property of *three*
    /// schemas, and a law drawing two cannot state it: the folds that carry a
    /// refutation up are exactly where a wrong one would enter, and each of
    /// them is antisymmetric and agrees with equality while still admitting a
    /// cycle. The order over the integer sets was one such fold, and a sort of
    /// guards found the cycle a pair could not.
    ///
    /// Only the decided corners are asserted. A decline says nothing, so a
    /// pair the rules leave unknown constrains nothing here.
    #[test]
    fn inclusion_is_transitive_where_the_rules_decide_it(
        a in shaped_schema(),
        b in shaped_schema(),
        c in shaped_schema(),
    ) {
        let oracle = CorpusOracle;
        let relation = |x: &Schema, y: &Schema| {
            let budget = std::cell::Cell::new(DECISION_BUDGET);
            x.subtype_relation(y, &oracle, &[], &budget)
        };
        if relation(&a, &b) == Relation::Holds && relation(&b, &c) == Relation::Holds {
            prop_assert_ne!(
                relation(&a, &c),
                Relation::Fails,
                "{:?} <= {:?} <= {:?} and the pair at the ends is refuted",
                a, b, c
            );
        }
        // And the same of the two proofs an equivalence is made of.
        if a.is_equivalent_under(&b, &oracle, &[]) && b.is_equivalent_under(&c, &oracle, &[]) {
            prop_assert_ne!(
                relation(&a, &c),
                Relation::Fails,
                "{:?} and {:?} are each equivalent to {:?} and the pair is refuted",
                a, c, b
            );
        }
    }

    // THEORY: semantic-subtyping
    /// The same claim where the two deciders can look something up.
    ///
    /// An oracle is the only place the core learns about Python -- a class's
    /// derivation and kind, a constant's value -- and both deciders read it.
    /// Two readings of one oracle that contradict each other is the defect this
    /// pair of properties exists for, and it is unreachable while neither
    /// decider can look anything up.
    #[test]
    fn the_two_deciders_agree_under_an_oracle(a in shaped_schema(), b in shaped_schema()) {
        let oracle = CorpusOracle;
        let budget = std::cell::Cell::new(DECISION_BUDGET);
        let rules = a.subtype_relation(&b, &oracle, &[], &budget);
        let sets = a.descriptor_contained_in(&b, &oracle, &[]);
        if rules == Relation::Fails {
            prop_assert!(
                sets != Relation::Holds,
                "the rules refuted {a:?} <= {b:?} and the sets decide it holds"
            );
        }
        if sets == Relation::Fails {
            prop_assert!(
                rules != Relation::Holds,
                "the sets refuted {a:?} <= {b:?} and the rules prove it holds"
            );
        }
    }

    /// The claim the other decider makes, held the same way.
    ///
    /// The descriptor refutes an inclusion by proving the difference holds a
    /// value, and that value is in the subject and outside the other schema.
    /// No rule may prove the inclusion over it: as above, one of the two would
    /// be wrong, and this is the direction that opened when the descriptor
    /// started answering in three values rather than two.
    #[test]
    fn an_inclusion_the_sets_refute_is_not_one_the_rules_prove(
        a in shaped_schema(),
        b in shaped_schema(),
    ) {
        if a.descriptor_contained_in(&b, &NoLeafRelations, &[]) == Relation::Fails {
            let budget = std::cell::Cell::new(DECISION_BUDGET);
            prop_assert!(
                a.subtype_relation(&b, &NoLeafRelations, &[], &budget) != Relation::Holds,
                "the sets refuted {a:?} <= {b:?} and the rules prove it holds"
            );
        }
    }

    /// And the other direction of the same claim: a proof stays a proof.
    ///
    /// Every relation the rules prove must still be one the public relation
    /// reports, so the three values cannot have quietly narrowed what is
    /// decided.
    #[test]
    fn a_proven_inclusion_is_the_relation_the_boundary_reports(
        a in schema(),
        b in schema(),
    ) {
        let budget = std::cell::Cell::new(DECISION_BUDGET);
        if a.subtype_relation(&b, &NoLeafRelations, &[], &budget) == Relation::Holds {
            prop_assert!(a.is_subtype_of(&b));
        }
    }
}

/// Every way of assigning each branch to one of `arity` positions, as a
/// digit string: the `k^|N|` enumeration JACM Lemma 6.5 states over the
/// negative atoms of a clause, which at arity two is its `2^|N|` subsets.
fn assignments(count: usize, arity: usize) -> impl Iterator<Item = Vec<usize>> {
    (0..arity.pow(u32::try_from(count).unwrap_or(0))).map(move |mut code| {
        (0..count)
            .map(|_| {
                let digit = code % arity;
                code /= arity;
                digit
            })
            .collect()
    })
}

/// A meet of the subject's component at `position` with the negation of that
/// component of every branch the assignment sends there: the clause the lemma
/// reads for emptiness on one side of the product.
fn side_clause(
    component: &Schema,
    branches: &[Vec<Schema>],
    assignment: &[usize],
    position: usize,
) -> Schema {
    let mut members = vec![component.clone()];
    for (branch, &sent_to) in branches.iter().zip(assignment) {
        if sent_to == position {
            members.push(not(branch[position].clone()));
        }
    }
    intersection_of(members)
}

/// A meet written as one node, so the emptiness asked of it is the scalar
/// fragment's exact reading and not a fold over pairs.
fn intersection_of(members: Vec<Schema>) -> Schema {
    Schema::Intersection(members.into())
}

/// JACM Lemma 6.5 for a product, as the paper states it for pairs:
///
/// ```text
/// t1 × t2 <= ⋁_{i∈N} (s1ᵢ × s2ᵢ)
///   iff  ∀ N' ⊆ N.  t1 ∧ ⋀_{i∈N'} ¬s1ᵢ = ∅   or   t2 ∧ ⋀_{i∈N∖N'} ¬s2ᵢ = ∅
/// ```
///
/// and applied at any fixed arity, as Castagna & Duboc state the tuple rule:
/// every assignment of the branches to positions has some position whose
/// clause is empty. Each clause's emptiness is asked of the scalar fragment,
/// where it is exact, so the value this returns is the relation itself and
/// not an approximation of it.
fn lemma_6_5(components: &[Schema], branches: &[Vec<Schema>]) -> bool {
    assignments(branches.len(), components.len()).all(|assignment| {
        (0..components.len()).any(|position| {
            side_clause(&components[position], branches, &assignment, position).is_empty()
        })
    })
}

/// A branch of the same arity as the subject, over the scalar fragment.
fn scalar_pair() -> impl Strategy<Value = Vec<Schema>> {
    (scalar_schema(), scalar_schema()).prop_map(|(a, b)| vec![a, b])
}

/// Branches that cover `subject` only taken together: one per member of the
/// union at `position`, with the other components as they are.
///
/// A random union of products rarely covers a random product, so a property
/// drawn from both alone asks the rule about refutations and declines and
/// seldom about the split it exists for. These are the splits: the lemma
/// holds of every one, and a rule that stopped narrowing declines them.
fn covering_split(subject: &[Schema], position: usize) -> Vec<Vec<Schema>> {
    let members: Vec<Schema> = match &subject[position] {
        Schema::Union(members) => members.iter().cloned().collect(),
        other => vec![other.clone()],
    };
    members
        .into_iter()
        .map(|member| {
            let mut branch = subject.to_vec();
            branch[position] = member;
            branch
        })
        .collect()
}

/// A product whose components are unions of scalar atoms, so a split has
/// branches to be made of.
fn union_product(arity: usize) -> impl Strategy<Value = Vec<Schema>> {
    proptest::collection::vec(
        proptest::collection::vec(scalar_schema(), 1..4).prop_map(union_of),
        arity,
    )
}

/// A union written as one node over the members, which is what a split reads.
fn union_of(members: Vec<Schema>) -> Schema {
    Schema::Union(members.into())
}

/// The same at three positions, which is the fixed component count the rule
/// generalises to and the pair cannot show.
fn scalar_triple() -> impl Strategy<Value = Vec<Schema>> {
    (scalar_schema(), scalar_schema(), scalar_schema()).prop_map(|(a, b, c)| vec![a, b, c])
}

/// Two definitions drawn from the structural fragment, each with a reference
/// under a container so the body is guarded, and each reaching the other.
///
/// The fixed pair in [`fixpoint_defs`] is two shapes; the fixpoint laws are
/// about every recursive definition a caller can write, so the bodies are
/// drawn: a leaf beside a list, a tuple or an optional field over a
/// reference, with the reference optionally under a complement, which is the
/// shape the polarity cut is for. The leaf is drawn from the buildable
/// fragment -- what a caller writes -- since the equivalence below is a
/// promise the pages make to a caller, and a body holding a refinement the
/// frontend refuses is one no caller can observe the promise on.
fn drawn_defs() -> impl Strategy<Value = Vec<Schema>> {
    let guarded = |index: usize| {
        (buildable_schema(), 0usize..4).prop_map(move |(leaf, shape)| {
            let reference = Schema::Ref(DefIx::new(index));
            let guard = match shape {
                0 => Schema::list(SeqShape::homogeneous(reference)),
                1 => Schema::tuple(SeqShape::fixed([Schema::Int, reference])),
                2 => Schema::record(
                    vec![Field {
                        name: "n".into(),
                        schema: reference,
                        required: false,
                    }],
                    Openness::Closed,
                ),
                _ => Schema::list(SeqShape::homogeneous(not(reference))),
            };
            union(leaf, guard)
        })
    };
    (guarded(0), guarded(1)).prop_map(|(a, b)| vec![a, b])
}

proptest! {
    #![proptest_config(ProptestConfig {
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    // THEORY: a-cut-reference-proves
    /// The polarity cut widens a difference and never narrows it.
    ///
    /// A reference read positively is cut to the top and one read under a
    /// complement to the bottom, so the schema a lowering reads on the
    /// subject's side contains the real one and the schema it reads on the
    /// other side is contained by it: `self⁺ ⊇ self` and `other⁻ ⊆ other`,
    /// which together give `self⁺ ∧ ¬other⁻ ⊇ self ∧ ¬other`. That containment
    /// is what makes an empty cut difference a proof of the inclusion, and it
    /// is held here on values rather than on the one pair that showed it: the
    /// oracle reads the real set six unfoldings deep, which is deeper than any
    /// value of the universe, and the cut set one unfolding deep, as the
    /// descriptor does.
    #[test]
    fn a_cut_reference_widens_the_subject_and_narrows_the_other(
        a in recursive_schema(),
        b in recursive_schema(),
    ) {
        let pool = const_pool();
        let defs = fixpoint_defs();
        let real_a = unfold_for_oracle(&a, &defs, ORACLE_UNFOLDS);
        let real_b = unfold_for_oracle(&b, &defs, ORACLE_UNFOLDS);
        let wide = a.unfolded(&defs, crate::descr::lower::UNFOLDS, Polarity::Widen);
        let narrow = b.unfolded(&defs, crate::descr::lower::UNFOLDS, Polarity::Narrow);
        prop_assert!(!wide.has_reference() && !narrow.has_reference(), "the cut leaves a reference");
        for value in &boundary_values(&[&real_a, &real_b]) {
            if member_full(&real_a, value, &pool) {
                prop_assert!(
                    member_full(&wide, value, &pool),
                    "{value:?} is in {a:?} and outside its widening"
                );
            }
            if member_full(&narrow, value, &pool) {
                prop_assert!(
                    member_full(&real_b, value, &pool),
                    "{value:?} is in the narrowing of {b:?} and outside it"
                );
            }
        }
    }

    // THEORY: a-sequence-splits-across-a-union
    /// The product rule decides exactly what JACM Lemma 6.5 says it decides.
    ///
    /// `Φ` is the backtrack-free form of the lemma's `2^|N|` enumeration, and
    /// the two are held to one answer over drawn products: a fixed pair
    /// against a union of fixed pairs, every component a Boolean combination
    /// of scalar atoms, where emptiness is exact and so the lemma's right-hand
    /// side is the relation itself. Asked of the rules alone, since the public
    /// relation asks the descriptor after them and would decide the pair for
    /// the other reason.
    #[test]
    fn the_product_rule_is_lemma_6_5(
        subject in scalar_pair(),
        branches in proptest::collection::vec(scalar_pair(), 1..4),
    ) {
        let pair = |components: &[Schema]| Schema::tuple(SeqShape::fixed(components.to_vec()));
        let tuple = pair(&subject);
        let split = Schema::union(branches.iter().map(|branch| pair(branch)));
        let budget = std::cell::Cell::new(DECISION_BUDGET);
        let rules = tuple.subtype_relation(&split, &NoLeafRelations, &[], &budget);
        let lemma = lemma_6_5(&subject, &branches);
        prop_assert_eq!(
            rules.holds(),
            lemma,
            "the rules answer {:?} and the lemma {} for {:?} <= {:?}",
            rules, lemma, tuple, split
        );
    }

    // THEORY: a-sequence-splits-across-a-union
    /// The same at three positions, where the lemma's subsets are the `3^|N|`
    /// assignments of the branches and the rule's narrowing runs per position.
    #[test]
    fn the_product_rule_is_lemma_6_5_at_three_positions(
        subject in scalar_triple(),
        branches in proptest::collection::vec(scalar_triple(), 1..4),
    ) {
        let triple = |components: &[Schema]| Schema::tuple(SeqShape::fixed(components.to_vec()));
        let tuple = triple(&subject);
        let split = Schema::union(branches.iter().map(|branch| triple(branch)));
        let budget = std::cell::Cell::new(DECISION_BUDGET);
        let rules = tuple.subtype_relation(&split, &NoLeafRelations, &[], &budget);
        let lemma = lemma_6_5(&subject, &branches);
        prop_assert_eq!(rules.holds(), lemma, "{:?} <= {:?}", tuple, split);
    }

    // THEORY: a-sequence-splits-across-a-union
    /// The rule decides every covering split, at two and three positions.
    ///
    /// A subject whose component at some position is a union is covered by
    /// the branches that take one member each, and by no branch alone: the
    /// lemma says the inclusion holds, and the rule must find it through
    /// the narrowing. Held apart from the random rows above because those
    /// draw a split by accident, and a property that catches a dropped
    /// narrowing by accident is one that misses it by accident too.
    #[test]
    fn the_product_rule_decides_every_covering_split(
        arity in 2usize..4,
        subject in union_product(3),
        position in 0usize..3,
    ) {
        let subject: Vec<Schema> = subject.into_iter().take(arity).collect();
        let position = position % arity;
        let branches = covering_split(&subject, position);
        let product = |components: &[Schema]| Schema::tuple(SeqShape::fixed(components.to_vec()));
        let tuple = product(&subject);
        let split = Schema::union(branches.iter().map(|branch| product(branch)));
        prop_assert!(lemma_6_5(&subject, &branches), "the lemma refutes a covering split");
        let budget = std::cell::Cell::new(DECISION_BUDGET);
        let rules = tuple.subtype_relation(&split, &NoLeafRelations, &[], &budget);
        prop_assert!(rules.holds(), "the rules answer {:?} for {:?} <= {:?}", rules, tuple, split);
    }

    // THEORY: two-fixpoints-one-procedure
    /// The fixpoint laws over definitions the test draws rather than lists.
    ///
    /// The proof direction of both fixpoints at once: an inclusion proved
    /// between two schemas over drawn definitions admits no counterexample
    /// the oracle finds, and an emptiness proved of one is refused by every
    /// boundary value. The definitions are the variable here; the schemas
    /// are the recursive fragment already drawn over the fixed pair.
    #[test]
    fn the_fixpoint_laws_hold_over_drawn_definitions(
        a in recursive_schema(),
        b in recursive_schema(),
        defs in drawn_defs(),
    ) {
        let pool = const_pool();
        let subject = unfold_for_oracle(&a, &defs, ORACLE_UNFOLDS);
        let other = unfold_for_oracle(&b, &defs, ORACLE_UNFOLDS);
        let values = boundary_values(&[&subject, &other]);
        if a.is_subtype_of_under(&b, &NoLeafRelations, &defs) {
            for value in &values {
                prop_assert!(
                    !member_full(&subject, value, &pool) || member_full(&other, value, &pool),
                    "{value:?} is in {a:?} and not in {b:?} under {defs:?}"
                );
            }
        }
        if a.is_empty_under(&defs) {
            for value in &values {
                prop_assert!(
                    !member_full(&subject, value, &pool),
                    "{value:?} is a member of {a:?}, decided empty under {defs:?}"
                );
            }
        }
    }

    // THEORY: a-reference-denotes-its-definition
    /// A reference and the definition it names are one set, over drawn
    /// definitions, by the deciders and by the values.
    ///
    /// The equirecursive reading: `Ref(i)` denotes what `defs[i]` denotes,
    /// so the two are equivalent in both directions -- which is the
    /// coinductive rule deciding a pair whose one side is the other's
    /// unfolding -- and the oracle, reading both past every value's depth,
    /// admits the same values through either.
    #[test]
    fn a_reference_and_its_definition_are_one_set_over_drawn_definitions(
        defs in drawn_defs(),
        which in 0usize..2,
    ) {
        let pool = const_pool();
        let reference = Schema::Ref(DefIx::new(which));
        let body = defs[which].clone();
        // The corpus oracle, which reads the literals: an oracle that declines
        // every literal question leaves a body holding one undecided, which is
        // a decline of the oracle and not of the reading.
        //
        // A reference and its body are one set in both directions: the
        // reference below its body by the reference arm's unfolding, and the
        // body below the reference member by member -- a plain member through
        // the same arm, a meet member through the meet rule where a member of
        // it is below, and through the unfolding where none is, since the
        // meet is a branch of the definition. Neither direction is ever
        // refuted, of the whole or of any member.
        let forward = reference.subtype_relation_under(&body, &CorpusOracle, &defs);
        let backward = body.subtype_relation_under(&reference, &CorpusOracle, &defs);
        prop_assert!(forward.holds(), "{reference:?} is not decided below its body");
        prop_assert!(
            backward.holds(),
            "the body is not decided below {reference:?} under {defs:?}: {backward:?}"
        );
        let members: Vec<Schema> = match &body {
            Schema::Union(members) => members.iter().cloned().collect(),
            other => vec![other.clone()],
        };
        for member in &members {
            let relation = member.subtype_relation_under(&reference, &CorpusOracle, &defs);
            prop_assert!(relation != Relation::Fails, "{member:?} is refuted below {reference:?}");
        }
        let through_reference = unfold_for_oracle(&reference, &defs, ORACLE_UNFOLDS + 1);
        let through_body = unfold_for_oracle(&body, &defs, ORACLE_UNFOLDS);
        for value in &boundary_values(&[&through_reference, &through_body]) {
            prop_assert_eq!(
                member_full(&through_reference, value, &pool),
                member_full(&through_body, value, &pool),
                "{:?} separates the reference from its body", value
            );
        }
    }
}
