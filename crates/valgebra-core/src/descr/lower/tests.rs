use std::sync::Arc;

use super::{BUDGET, Bounds, Constants, DEPTH, Operand, lower, lower_within};
use crate::descr::classes::Class;
use crate::descr::{Descr, Value};
use crate::ir::{
    ClassIx, ConstIx, Constraint, Field, MapClause, Openness, OperandIx, Schema, SeqKind, SeqShape,
};
use crate::kind::Kind;
use crate::verdict::Verdict;

/// A pool that answers from a list, which is what the bindings do from the
/// validator's object table.
struct Pool(Vec<Operand>);

impl Constants for Pool {
    fn operand(&self, index: OperandIx) -> Option<Operand> {
        self.0.get(index.get()).cloned()
    }

    fn constant(&self, index: ConstIx) -> Option<Operand> {
        self.0.get(index.get()).cloned()
    }

    /// A class reaches the core through the bindings, and these tests have
    /// none: a pooled `Instance` reads as one this pool cannot see.
    fn class(&self, index: ClassIx) -> Option<Class> {
        match self.0.get(index.get()) {
            Some(Operand::Instance(class)) => Some(class.clone()),
            _ => None,
        }
    }
}

fn empty_pool() -> Pool {
    Pool(Vec::new())
}

/// The kinds and the two ends map straight across.
#[test]
fn the_leaves_lower_to_the_sets_they_denote() {
    let pool = empty_pool();
    let leaf = |schema| lower(&schema, &pool).expect("a leaf lowers");

    assert_eq!(leaf(Schema::ANYTHING), Descr::anything());
    assert_eq!(leaf(Schema::Nothing), Descr::nothing());
    assert_eq!(leaf(Schema::Float), Descr::of_kind(Kind::Float));
    assert_eq!(leaf(Schema::Str), Descr::of_kind(Kind::Str));
}

/// `bool` subclasses `int`, so the schema that denotes every integer
/// denotes both kinds.
///
/// The one leaf that is not one kind. Keeping `Bool` its own kind is what
/// makes the two components independent, and the price is that `int` says
/// so here rather than being read off the name.
#[test]
fn the_integers_include_the_booleans() {
    let pool = empty_pool();
    let ints = lower(&Schema::Int, &pool).expect("int lowers");

    assert!(ints.admits(Value::integer(1)));
    assert!(ints.admits(Value::boolean(true)));
    assert!(!ints.admits(Value::float(1.0)));
    assert!(
        !lower(&Schema::Bool, &pool)
            .expect("bool lowers")
            .admits(Value::integer(1))
    );
}

/// The three operations lower to the three operations, which is the whole
/// point of the map.
#[test]
fn the_operations_lower_to_the_operations() {
    let pool = empty_pool();
    let joined = lower(
        &Schema::Union(vec![Schema::Str, Schema::Float].into()),
        &pool,
    )
    .expect("a small union");
    let expected = Descr::of_kind(Kind::Str)
        .union(&Descr::of_kind(Kind::Float))
        .expect("a small union");
    assert_eq!(joined, expected);

    let barred =
        lower(&Schema::Complement(Arc::new(Schema::Str)), &pool).expect("a small complement");
    assert_eq!(barred, Descr::of_kind(Kind::Str).complement());
}

/// A refinement is a *meet of sets*, so the order the constraints are
/// written in carries no meaning and an impossible pair decides.
#[test]
fn a_bound_pair_that_cannot_hold_lowers_to_the_empty_set() {
    let pool = Pool(vec![Operand::Integer(0)]);
    let refined = |constraints: Vec<Constraint>| {
        lower(
            &Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: constraints.into(),
            },
            &pool,
        )
        .expect("a small refinement")
    };

    let ge_then_lt = refined(vec![
        Constraint::Ge(OperandIx::new(0)),
        Constraint::Lt(OperandIx::new(0)),
    ]);
    let lt_then_ge = refined(vec![
        Constraint::Lt(OperandIx::new(0)),
        Constraint::Ge(OperandIx::new(0)),
    ]);
    assert_eq!(ge_then_lt.emptiness(), Verdict::Empty);
    assert_eq!(ge_then_lt, lt_then_ge, "the order says nothing");

    let non_negative = refined(vec![Constraint::Ge(OperandIx::new(0))]);
    assert!(non_negative.admits(Value::integer(0)));
    assert!(!non_negative.admits(Value::integer(-1)));
}

/// The relations a decision procedure cannot reach, decided on the sets.
///
/// Each is a *value* fact the term layer has no rule for: a step divides
/// another, a literal misses a kind, a kind is exhausted by its literals, a
/// set is its own hole plus what fills it. Lowering them puts each in the
/// kind set it denotes, where the answer is `a ∧ ¬b = ∅` and nothing else.
#[test]
fn the_value_relations_decide_on_the_sets() {
    let pool = Pool(vec![
        Operand::Integer(4),
        Operand::Integer(2),
        Operand::Integer(1),
        Operand::Boolean(true),
        Operand::Boolean(false),
    ]);
    let set = |schema| lower(&schema, &pool).expect("a small schema");
    let step = |at| {
        set(Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![Constraint::MultipleOf(OperandIx::new(at))].into(),
        })
    };
    let literal = |at| set(Schema::Literal(ConstIx::new(at)));
    let within = |a: &Descr, b: &Descr| {
        a.intersect(&b.complement())
            .expect("two small sets")
            .emptiness()
    };

    // A step is a subset of the steps it is a multiple of, and of no other.
    assert_eq!(within(&step(0), &step(1)), Verdict::Empty);
    assert_eq!(within(&step(1), &step(0)), Verdict::Inhabited);

    // `bool` is its own kind, so an integer literal is never a boolean.
    let no_bool = set(Schema::Complement(Arc::new(Schema::Bool)));
    assert_eq!(within(&literal(2), &no_bool), Verdict::Empty);

    // A kind with finitely many values is exhausted by naming them all.
    let both = set(Schema::Union(
        vec![
            Schema::Literal(ConstIx::new(3)),
            Schema::Literal(ConstIx::new(4)),
        ]
        .into(),
    ));
    assert_eq!(within(&set(Schema::Bool), &both), Verdict::Empty);
    assert_eq!(within(&both, &set(Schema::Bool)), Verdict::Empty);

    // A set is the hole punched in it, put back: `a = (a ∧ ¬v) ∨ v` for a
    // value `a` holds.
    let split = set(Schema::Union(
        vec![
            Schema::meet(vec![
                Schema::Int,
                Schema::Complement(Arc::new(Schema::Literal(ConstIx::new(2)))),
            ]),
            Schema::Literal(ConstIx::new(2)),
        ]
        .into(),
    ));
    assert_eq!(within(&set(Schema::Int), &split), Verdict::Empty);
    assert_eq!(within(&split, &set(Schema::Int)), Verdict::Empty);
}

/// A bound orders the booleans as well as the integers.
///
/// `bool` is a kind of its own here, and `int` denotes both -- so a bound
/// over `int` that spoke only for the integer component would give a
/// *smaller* set than the schema denotes, and a smaller set has a larger
/// complement. `True` is `1` to every comparison Python makes, so a bound
/// admitting `1` admits it.
#[test]
fn a_bound_over_the_integers_orders_the_booleans_too() {
    let pool = Pool(vec![Operand::Integer(1)]);
    let bounded = |constraint| {
        lower(
            &Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![constraint].into(),
            },
            &pool,
        )
        .expect("a small refinement")
    };

    let at_least_one = bounded(Constraint::Ge(OperandIx::new(0)));
    assert!(at_least_one.admits(Value::boolean(true)));
    assert!(!at_least_one.admits(Value::boolean(false)));
    assert!(at_least_one.admits(Value::integer(1)));

    let below_one = bounded(Constraint::Lt(OperandIx::new(0)));
    assert!(below_one.admits(Value::boolean(false)));
    assert!(!below_one.admits(Value::boolean(true)));

    // A bound over a **float** base lands in the float component instead,
    // with the operand read as a number: `Annotated[float, Ge(0)]` carries
    // the integer zero and orders the floats all the same. Narrowing a float
    // base to a set of *integers* is what would be unsound, and that is not
    // what happens.
    // The pool's only operand is the integer 1, so this is "float >= 1".
    let floats_at_least_one = lower(
        &Schema::Refine {
            base: Arc::new(Schema::Float),
            constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
        },
        &pool,
    )
    .expect("a bound over floats lowers");
    assert!(floats_at_least_one.admits(Value::float(1.0)));
    assert!(floats_at_least_one.admits(Value::float(1.5)));
    assert!(!floats_at_least_one.admits(Value::float(0.5)));
    assert!(!floats_at_least_one.admits(Value::float(-1.0)));
    // `nan` is outside every interval, which is the comparison Python makes.
    assert!(!floats_at_least_one.admits(Value::float(f64::NAN)));
    // And it is floats and nothing else: the integer 1 is not in it.
    assert!(!floats_at_least_one.admits(Value::integer(1)));

    // A base that is neither whole numbers nor floats alone still refuses,
    // for the reason both sides refuse: narrowing it to one component gives
    // a smaller set than the schema denotes, and a smaller set has a larger
    // complement.
    assert!(
        lower(
            &Schema::Refine {
                base: Arc::new(Schema::ANYTHING),
                constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
            },
            &pool,
        )
        .is_none()
    );
}

/// A bound over floats reads a float operand, and reads both directions.
///
/// The test above writes the bound as `Annotated[float, Ge(1)]`, whose
/// operand is the *integer* one; this writes the one a caller reaches for
/// more often, `Ge(1.5)`, whose fractional part is the whole difference --
/// read as an integer it would round, and `1.25` would land inside a set
/// that excludes it. The upper direction is a separate arm from the lower,
/// and a reading that carried only one of them refuses the other, leaving
/// the relation undecided where the schema is perfectly ordinary.
#[test]
fn a_float_bound_is_read_as_a_float_in_both_directions() {
    let pool = Pool(vec![Operand::Float(1.5)]);
    let bounded = |constraint| {
        lower(
            &Schema::Refine {
                base: Arc::new(Schema::Float),
                constraints: vec![constraint].into(),
            },
            &pool,
        )
        .expect("a bound over floats lowers")
    };

    let at_least = bounded(Constraint::Ge(OperandIx::new(0)));
    assert!(at_least.admits(Value::float(1.5)));
    assert!(!at_least.admits(Value::float(1.25)), "1.5 is not 1");
    let above = bounded(Constraint::Gt(OperandIx::new(0)));
    assert!(above.admits(Value::float(1.75)) && !above.admits(Value::float(1.5)));

    let at_most = bounded(Constraint::Le(OperandIx::new(0)));
    assert!(at_most.admits(Value::float(1.5)) && at_most.admits(Value::float(0.0)));
    assert!(!at_most.admits(Value::float(1.75)));
    // `nan` is outside every interval, on this side as on the other.
    assert!(!at_most.admits(Value::float(f64::NAN)));
    let below = bounded(Constraint::Lt(OperandIx::new(0)));
    assert!(below.admits(Value::float(1.25)) && !below.admits(Value::float(1.5)));
}

/// A step is the constraint no union of intervals can spell, and it meets
/// the bounds rather than being checked beside them.
#[test]
fn a_step_meets_the_bounds() {
    let pool = Pool(vec![Operand::Integer(2), Operand::Integer(1)]);
    let evens = lower(
        &Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![
                Constraint::MultipleOf(OperandIx::new(0)),
                Constraint::Ge(OperandIx::new(1)),
            ]
            .into(),
        },
        &pool,
    )
    .expect("a small refinement");

    assert!(evens.admits(Value::integer(2)) && evens.admits(Value::integer(4)));
    assert!(!evens.admits(Value::integer(0)), "the bound excludes it");
    assert!(!evens.admits(Value::integer(3)), "the step excludes it");
}

/// A sequence lowers through the one constructor its three spellings share.
#[test]
fn a_sequence_lowers_to_its_shape() {
    const ONE: &[Value] = &[Value::integer(1)];
    const TWO: &[Value] = &[Value::integer(1), Value::integer(1)];

    let pool = empty_pool();
    let of_ints = lower(
        &Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape {
                prefix: Vec::new().into(),
                tail: Some(Arc::new(Schema::Int)),
            },
        },
        &pool,
    )
    .expect("a small sequence");

    assert!(of_ints.admits(Value::sequence(ONE, Kind::List)));
    assert!(of_ints.admits(Value::sequence(TWO, Kind::List)));
    assert!(
        !of_ints.admits(Value::sequence(ONE, Kind::Tuple)),
        "a tuple is not a list"
    );
}

/// A set lowers to the powerset of what it holds, hashability included.
#[test]
fn a_set_lowers_to_a_powerset() {
    const NOTHING: &[Value] = &[];

    let pool = empty_pool();
    let of_lists = lower(
        &Schema::set(Schema::Seq {
            container: SeqKind::List,
            shape: SeqShape {
                prefix: Vec::new().into(),
                tail: Some(Arc::new(Schema::Int)),
            },
        }),
        &pool,
    )
    .expect("a small set");

    // The members are the element schema as written: a `list` subclass that
    // defines `__hash__` is a list and a legal member, so the set of lists is
    // not the set of nothing.
    assert!(of_lists.admits(Value::sequence(NOTHING, Kind::Set)));
    assert_ne!(
        of_lists,
        Descr::set(&Descr::nothing(), Kind::Set).expect("a set kind")
    );
}

/// The schemas both the procedure and the descriptor understand, as a
/// corpus to compare them over.
fn leaves() -> Vec<Schema> {
    let seq = |tail| Schema::Seq {
        container: SeqKind::List,
        shape: SeqShape {
            prefix: Vec::new().into(),
            tail: Some(Arc::new(tail)),
        },
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
        seq(Schema::Int),
        seq(Schema::Str),
        Schema::set(Schema::Int),
    ]
}

/// The leaves, their complements, and every pair joined and met.
///
/// Quadratic in the leaves, so the *containment* check below takes the
/// leaves alone: it is quadratic again over whatever it is given, and a
/// descriptor meet builds automata.
fn corpus() -> Vec<Schema> {
    let leaves = leaves();
    let mut corpus = leaves.clone();
    for left in &leaves {
        for right in &leaves {
            corpus.push(Schema::Union(vec![left.clone(), right.clone()].into()));
            corpus.push(Schema::Intersection(
                vec![left.clone(), right.clone()].into(),
            ));
        }
        corpus.push(Schema::Complement(Arc::new(left.clone())));
    }
    corpus
}

// THEORY: a-cut-reference-proves
/// The descriptor and the procedure agree about emptiness wherever both
/// decide it.
///
/// The check the whole milestone is for: two representations of one meaning,
/// asked the same question. Neither is taken as the oracle -- each has
/// answers the other lacks, so the claim is only that they never *contradict*
/// each other, and a proof from one is never met by the opposite proof from
/// the other.
#[test]
fn the_descriptor_and_the_procedure_never_contradict_each_other() {
    let pool = empty_pool();
    for schema in corpus() {
        let Some(descr) = lower(&schema, &pool) else {
            continue;
        };
        if descr.emptiness() == Verdict::Empty {
            assert!(
                schema.is_empty(),
                "the descriptor proved {schema:?} empty and the procedure did not"
            );
        }
        if schema.is_empty() {
            assert_ne!(
                descr.emptiness(),
                Verdict::Inhabited,
                "the procedure proved {schema:?} empty and the descriptor denied it"
            );
        }
    }
}

/// And about containment, which is the relation emptiness is asked for.
#[test]
fn the_two_agree_about_containment_where_both_decide() {
    let pool = empty_pool();
    let corpus: Vec<Schema> = leaves()
        .iter()
        .flat_map(|leaf| [leaf.clone(), Schema::Complement(Arc::new(leaf.clone()))])
        .collect();
    for left in &corpus {
        for right in &corpus {
            let (Some(a), Some(b)) = (lower(left, &pool), lower(right, &pool)) else {
                continue;
            };
            let Some(difference) = a.intersect(&b.complement()) else {
                continue;
            };
            if difference.emptiness() == Verdict::Empty {
                assert!(
                    left.is_subtype_of(right),
                    "the descriptor made {left:?} a subtype of {right:?} and the \
                     procedure did not"
                );
            }
        }
    }
}

/// A literal lowers to the singleton its pooled value names, under the kind
/// that reads that value.
///
/// The typing spec keeps `Literal[1]`, `Literal[True]` and `Literal["1"]`
/// apart, and so does this: each lands in its own kind's component, so no
/// two of them meet.
#[test]
fn a_literal_lowers_to_the_singleton_its_kind_reads() {
    let pool = Pool(vec![
        Operand::Integer(1),
        Operand::Boolean(true),
        Operand::Word(b"a".to_vec(), Kind::Str),
        Operand::NoneType,
        Operand::Float(1.5),
    ]);
    let literal =
        |slot| lower(&Schema::Literal(ConstIx::new(slot)), &pool).expect("a pooled constant");

    assert!(literal(0).admits(Value::integer(1)) && !literal(0).admits(Value::integer(2)));
    assert!(literal(1).admits(Value::boolean(true)) && !literal(1).admits(Value::boolean(false)));
    assert!(literal(2).admits(Value::word(b"a", Kind::Str)));
    assert!(literal(3).admits(Value::of_kind(Kind::NoneType)));
    assert!(literal(4).admits(Value::float(1.5)));

    // The three the spec keeps apart stay apart, because each is a
    // different kind's component.
    for (left, right) in [(0, 1), (0, 2), (1, 2)] {
        assert_eq!(
            literal(left)
                .intersect(&literal(right))
                .expect("two singletons")
                .emptiness(),
            Verdict::Empty,
            "{left} and {right}"
        );
    }
}

/// A class the pool can see lowers to the set of its instances, carrying the
/// order it was given: the subclass is a subtype, and the two ends are not.
#[test]
fn a_class_the_pool_knows_lowers_to_its_instances() {
    let animal = Class::new(0, Class::PLAIN, &[]);
    let dog = Class::new(1, Class::PLAIN, std::slice::from_ref(&animal));
    let pool = Pool(vec![
        Operand::Instance(animal),
        Operand::Instance(dog),
        Operand::Instance(Class::new(2, Class::PLAIN, &[])),
    ]);
    let instances = |at| lower(&Schema::Instance(ClassIx::new(at)), &pool).expect("a class");
    let (animals, dogs, others) = (instances(0), instances(1), instances(2));

    // `a ≤ b` is `a ∧ ¬b = ∅`, which is the whole of the subtyping test.
    let within = |a: &Descr, b: &Descr| {
        a.intersect(&b.complement())
            .expect("two classes")
            .emptiness()
    };
    assert_eq!(within(&dogs, &animals), Verdict::Empty);
    // The converse is not merely undecided: an animal that is not a dog is
    // a value the descriptor can name.
    assert_eq!(within(&animals, &dogs), Verdict::Inhabited);
    // Two unrelated classes share `object`, so nothing here proves them
    // disjoint -- only that neither contains the other.
    assert_eq!(
        animals.intersect(&others).expect("two classes").emptiness(),
        Verdict::Unknown
    );
    // A class is a set of objects and nothing else: no scalar is one.
    assert!(!animals.admits(Value::integer(0)));
}

/// A literal naming a class instance refuses: a class is a set of objects,
/// and the literal pins *which* instance, which the descriptor cannot say.
#[test]
fn a_literal_naming_an_instance_refuses() {
    let pool = Pool(vec![Operand::Instance(Class::laid_out(1, 1))]);
    assert!(lower(&Schema::Literal(ConstIx::new(0)), &pool).is_none());
}

/// A length bound and a pattern are sets of *words*, and they lower under
/// whichever word kind the base admits.
#[test]
fn the_word_constraints_lower_to_languages() {
    let pool = empty_pool();
    let refined = |base, constraints| {
        lower(
            &Schema::Refine {
                base: Arc::new(base),
                constraints,
            },
            &pool,
        )
    };

    let non_empty =
        refined(Schema::Str, vec![Constraint::MinLen(1)].into()).expect("a length bound");
    assert!(non_empty.admits(Value::word(b"a", Kind::Str)));
    assert!(!non_empty.admits(Value::word(b"", Kind::Str)));

    let short = refined(Schema::Str, vec![Constraint::MaxLen(1)].into()).expect("a length bound");
    assert!(
        short.admits(Value::word(b"", Kind::Str)) && short.admits(Value::word(b"a", Kind::Str))
    );
    assert!(!short.admits(Value::word(b"ab", Kind::Str)));

    let matching =
        refined(Schema::Str, vec![Constraint::Regex("a+".to_owned())].into()).expect("a pattern");
    assert!(matching.admits(Value::word(b"a", Kind::Str)));
    assert!(!matching.admits(Value::word(b"b", Kind::Str)));

    // Two of them meet rather than being checked one after the other, which
    // is what makes an impossible pair decide.
    let impossible = refined(
        Schema::Str,
        vec![Constraint::MinLen(2), Constraint::MaxLen(1)].into(),
    )
    .expect("two length bounds");
    assert_eq!(impossible.emptiness(), Verdict::Empty);
}

/// A word constraint on a base that is *more* than words refuses too.
///
/// A length is not a word's alone -- a list, a set and a dict all have one --
/// so a bound over `anything` constrains values the word component cannot
/// speak about. Lowering it as if it only spoke about words would give a
/// smaller set than the schema denotes, and a smaller set has a *larger*
/// complement: `set[anything] <= ~Annotated[anything, MinLen(0)]` would be
/// proved, with the empty set standing against it.
#[test]
fn a_length_bound_over_more_than_words_refuses() {
    let pool = empty_pool();
    assert!(
        lower(
            &Schema::Refine {
                base: Arc::new(Schema::ANYTHING),
                constraints: vec![Constraint::MinLen(0)].into(),
            },
            &pool,
        )
        .is_none()
    );

    // Narrowed to the words first, the same bound lowers.
    assert!(
        lower(
            &Schema::Refine {
                base: Arc::new(Schema::Str),
                constraints: vec![Constraint::MinLen(0)].into(),
            },
            &pool,
        )
        .is_some()
    );
}

/// A word constraint on a base with no words refuses.
///
/// The bound has no component to land in: an integer has no length, so
/// there is no language to meet the base with. Refusing says that; lowering
/// to the empty set would claim the schema *denotes* nothing, which is a
/// stronger statement than this map is entitled to make.
#[test]
fn a_length_bound_on_a_base_with_no_words_refuses() {
    let pool = empty_pool();
    assert!(
        lower(
            &Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![Constraint::MinLen(1)].into(),
            },
            &pool,
        )
        .is_none()
    );
}

/// A schema past a bound refuses rather than spending without end.
///
/// Lowering builds at every node, so a wide schema is a lot of work and a
/// deep one is that work raised to a power. There are two bounds because
/// there are two quantities: [`BUDGET`] counts the nodes and [`DEPTH`] the
/// nesting, and each is asserted in both directions -- a bound that only
/// ever refuses would pass half of this.
///
/// Refusing is safe, because the caller decides the old way.
#[test]
fn a_schema_past_a_bound_refuses() {
    let pool = empty_pool();
    let wide = |members: usize| {
        Schema::Union(
            core::iter::repeat_n(Schema::Str, members)
                .collect::<Vec<_>>()
                .into(),
        )
    };
    assert!(
        lower(&wide(BUDGET as usize), &pool).is_none(),
        "too many nodes"
    );
    assert!(lower(&wide(BUDGET as usize / 2), &pool).is_some());

    let nested =
        |levels: u32| (0..levels).fold(Schema::Str, |inner, _| Schema::Complement(Arc::new(inner)));
    assert!(lower(&nested(DEPTH + 1), &pool).is_none(), "too deep");
    assert!(lower(&nested(DEPTH - 1), &pool).is_some());
}

/// A build that would cost too much refuses, and the same schema lowers
/// under an allowance that covers it.
///
/// The refusal is about the *work*, not about the schema: nothing here is a
/// form the descriptor cannot hold. Which is why it is asserted in both
/// directions -- an allowance that only ever refuses would pass half of it.
#[test]
fn a_build_past_its_allowance_refuses() {
    let pool = empty_pool();
    let meet = Schema::Intersection(
        vec![
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Schema::list(SeqShape::homogeneous(Schema::Str)),
        ]
        .into(),
    );

    assert!(
        lower_within(
            Bounds {
                work: 0,
                ..Bounds::DEFAULT
            },
            &meet,
            &pool
        )
        .is_none(),
        "nothing to spend"
    );
    assert!(lower_within(Bounds::DEFAULT, &meet, &pool).is_some());
    // A leaf takes no product, so it costs nothing and lowers on an empty
    // allowance: the budget bounds what multiplies, not what is read. `int`
    // is not one of those -- it is the union of two kinds -- which is why
    // this says `str`.
    assert!(
        lower_within(
            Bounds {
                work: 0,
                ..Bounds::DEFAULT
            },
            &Schema::Str,
            &pool
        )
        .is_some()
    );
    assert!(
        lower_within(
            Bounds {
                work: 0,
                ..Bounds::DEFAULT
            },
            &Schema::Int,
            &pool
        )
        .is_none(),
        "a union"
    );
}

/// The allowance a lowering is given is its own: an expensive one does not
/// leave the next one poorer.
#[test]
fn a_build_does_not_spend_the_next_one_s_allowance() {
    let pool = empty_pool();
    let deep = (0..6).fold(
        Schema::record(
            vec![Field {
                name: "leaf".into(),
                schema: Schema::Int,
                required: true,
            }],
            Openness::Closed,
        ),
        |inner, _| {
            Schema::record(
                vec![Field {
                    name: "child".into(),
                    schema: Schema::list(SeqShape::homogeneous(inner)),
                    required: true,
                }],
                Openness::Closed,
            )
        },
    );

    for _ in 0..3 {
        let _ = lower(&deep, &pool);
        assert!(
            lower(&Schema::set(Schema::Int), &pool).is_some(),
            "the allowance came back"
        );
    }
}

/// The map is partial, and it refuses rather than approximating.
///
/// Each of these has no component to land in, or none that would mean what
/// the schema does: a dict has no map component, an attribute record beside
/// a builtin kind wants a descriptor that is a union of lines, and a
/// reference is a cycle.
#[test]
fn the_forms_with_nowhere_to_land_refuse() {
    let pool = empty_pool();
    for schema in [
        // A class needs the object pool to say what it derives from, and the
        // core has none.
        Schema::Instance(crate::ir::ClassIx::new(0)),
        Schema::Ref(crate::ir::DefIx::new(0)),
        // A key schema that is neither a kind nor a constant covers part of
        // a part, and the default is a function on the parts.
        Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key: Schema::Refine {
                    base: Arc::new(Schema::Str),
                    constraints: vec![Constraint::MinLen(1)].into(),
                },
                value: Schema::Int,
            }]
            .into(),
        },
    ] {
        assert!(lower(&schema, &pool).is_none(), "{schema:?}");
    }
    // A refusal inside a form refuses the whole form rather than dropping
    // the part it could not read.
    assert!(
        lower(
            &Schema::Union(vec![Schema::Str, Schema::Ref(crate::ir::DefIx::new(0))].into()),
            &pool
        )
        .is_none()
    );
    // `Any` is the top, spelled, so it lowers to the top rather than
    // refusing: the spelling is not a set and the descriptor holds sets.
    assert_eq!(lower(&Schema::ANY, &pool), Some(Descr::anything()));
}

/// An attribute record lowers without the pool, because a field's name and
/// its type are the whole of it.
///
/// It is the half of an object schema the core can read: the class beside it
/// needs the pool, and the two meet once the pool is in reach. The record
/// narrows a value of *any* kind, so what lowers here is not scoped to the
/// values that have no kind.
#[test]
fn an_attribute_record_lowers_to_the_values_carrying_it() {
    const CARRIED: &[(&str, Value)] = &[("a", Value::integer(1))];
    let pool = empty_pool();
    let field = |name: &str, schema, required| crate::ir::Field {
        name: name.into(),
        schema,
        required,
    };
    let record = Schema::AttrRecord {
        fields: vec![field("a", Schema::Int, true)].into(),
    };
    let lowered = lower(&record, &pool).expect("an attribute record lowers");
    assert!(lowered.admits(Value::object(CARRIED)));
    // And it narrows a value that also has a kind, which is the whole point
    // of holding the record on a line rather than beside the kinds.
    assert!(lowered.admits(Value::integer(7).carrying(CARRIED)));
    assert!(!lowered.admits(Value::integer(7)));
    // A field whose type admits nothing empties the record.
    let empty = Schema::AttrRecord {
        fields: vec![field("a", Schema::Nothing, true)].into(),
    };
    assert!(lower(&empty, &pool).expect("it lowers").is_empty());
}

/// A map lowers to the atom its keys spell, and the three rows the report
/// lists as undecided fall out of the semantic `dom`.
///
/// A label whose type its part's default already gives it says nothing, and
/// the atom drops it -- so a key that must be absent from a closed record
/// leaves the empty map, whichever of the three ways it was written.
#[test]
fn the_maps_that_name_nothing_are_the_empty_map() {
    let pool = empty_pool();
    let closed =
        |fields: Vec<crate::ir::Field>, defaults: Vec<crate::ir::MapClause>| Schema::KeyedMap {
            fields: fields.into(),
            defaults: defaults.into(),
        };
    let field = |name: &str, schema, required| crate::ir::Field {
        name: name.into(),
        schema,
        required,
    };
    let empty = lower(&closed(Vec::new(), Vec::new()), &pool).expect("`{}` lowers");

    // `{"a?": nothing}`: the key may be absent and holds nothing, which is
    // what the closed default already says, so the label is absorbed.
    let optional_nothing = closed(vec![field("a", Schema::Nothing, false)], Vec::new());
    assert_eq!(lower(&optional_nothing, &pool), Some(empty.clone()));

    // `dict[str, nothing]`: every `str` key maps into nothing, so there are
    // none, and no other part was opened.
    let no_str_values = closed(
        Vec::new(),
        vec![MapClause {
            key: Schema::Str,
            value: Schema::Nothing,
        }],
    );
    assert_eq!(lower(&no_str_values, &pool), Some(empty.clone()));

    // `dict[nothing, int]`: no key at all is governed, so none is admitted.
    let no_keys = closed(
        Vec::new(),
        vec![MapClause {
            key: Schema::Nothing,
            value: Schema::Int,
        }],
    );
    assert_eq!(lower(&no_keys, &pool), Some(empty.clone()));

    // And the empty map is not the empty *set*: it holds one dict.
    assert!(!empty.is_empty());
    assert!(empty.admits(Value::dict(&[])));
}

/// A required field is required, and an open record admits the keys it does
/// not name.
#[test]
fn a_record_lowers_closed_and_a_catch_all_opens_it() {
    const A_IS_INT: &[(Value, Value)] = &[(Value::word(b"a", Kind::Str), Value::integer(1))];
    const A_AND_B: &[(Value, Value)] = &[
        (Value::word(b"a", Kind::Str), Value::integer(1)),
        (Value::word(b"b", Kind::Str), Value::integer(2)),
    ];
    let pool = empty_pool();
    let field = crate::ir::Field {
        name: "a".into(),
        schema: Schema::Int,
        required: true,
    };
    let shut = lower(
        &Schema::KeyedMap {
            fields: vec![field.clone()].into(),
            defaults: Vec::new().into(),
        },
        &pool,
    )
    .expect("a closed record lowers");
    assert!(shut.admits(Value::dict(A_IS_INT)));
    assert!(!shut.admits(Value::dict(&[])), "the field is required");
    assert!(!shut.admits(Value::dict(A_AND_B)), "and nothing else is");

    let open = lower(
        &Schema::KeyedMap {
            fields: vec![field].into(),
            defaults: vec![MapClause::top()].into(),
        },
        &pool,
    )
    .expect("an open record lowers");
    assert!(open.admits(Value::dict(A_IS_INT)));
    assert!(open.admits(Value::dict(A_AND_B)), "a catch-all opens it");
    assert!(
        !open.admits(Value::dict(&[])),
        "the field is still required"
    );
}

/// Each key kind is its own part, and a clause opens the one its key names.
///
/// The default is a function on the partition, so `dict[int, V]` says what an
/// integer key maps to and leaves a string key forbidden -- the map was
/// closed, and only the part the clause named was opened.
#[test]
fn a_clause_opens_the_part_its_key_names() {
    /// One key of each part, beside the schema that names that part.
    const KEYS: [(Kind, &[(Value, Value)]); 6] = [
        (
            Kind::NoneType,
            &[(Value::of_kind(Kind::NoneType), Value::integer(1))],
        ),
        (Kind::Bool, &[(Value::boolean(true), Value::integer(1))]),
        (Kind::Int, &[(Value::integer(7), Value::integer(1))]),
        (Kind::Float, &[(Value::float(1.5), Value::integer(1))]),
        (
            Kind::Str,
            &[(Value::word(b"a", Kind::Str), Value::integer(1))],
        ),
        (
            Kind::Bytes,
            &[(Value::word(b"a", Kind::Bytes), Value::integer(1))],
        ),
    ];
    let atom = |kind| match kind {
        Kind::NoneType => Schema::NoneType,
        Kind::Bool => Schema::Bool,
        Kind::Int => Schema::Int,
        Kind::Float => Schema::Float,
        Kind::Bytes => Schema::Bytes,
        _ => Schema::Str,
    };
    let pool = empty_pool();
    for (kind, _) in KEYS {
        let opened = lower(
            &Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![MapClause {
                    key: atom(kind),
                    value: Schema::Int,
                }]
                .into(),
            },
            &pool,
        )
        .expect("a mapping lowers");
        for (other, entry) in KEYS {
            // A `bool` key is an `int` key: the walk reads `{True: 1}` as a
            // dict whose key is an integer, so an `int`-keyed clause covers it.
            let covered = other == kind || (kind == Kind::Int && other == Kind::Bool);
            assert_eq!(
                opened.admits(Value::dict(entry)),
                covered,
                "a {kind:?}-keyed map against a {other:?} key"
            );
        }
    }
}

/// A union of key schemas opens each part it names, and a `Literal` names a
/// key rather than a part.
#[test]
fn a_union_opens_each_part_and_a_literal_names_one_key() {
    const A: &[(Value, Value)] = &[(Value::word(b"a", Kind::Str), Value::integer(1))];
    const B: &[(Value, Value)] = &[(Value::word(b"b", Kind::Str), Value::integer(1))];
    const ONE: &[(Value, Value)] = &[(Value::integer(1), Value::integer(1))];
    let pool = empty_pool();
    let either = lower(
        &Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key: Schema::Union(vec![Schema::Str, Schema::Int].into()),
                value: Schema::Int,
            }]
            .into(),
        },
        &pool,
    )
    .expect("a union of key kinds lowers");
    assert!(either.admits(Value::dict(A)));
    assert!(either.admits(Value::dict(ONE)));

    // A literal key is a label: the key it names is governed, and every
    // other key of that part is not.
    let named = Pool(vec![Operand::Word(b"a".to_vec(), Kind::Str)]);
    let one_key = lower(
        &Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key: Schema::Literal(ConstIx::new(0)),
                value: Schema::Int,
            }]
            .into(),
        },
        &named,
    )
    .expect("a literal key lowers");
    assert!(one_key.admits(Value::dict(A)));
    assert!(
        one_key.admits(Value::dict(&[])),
        "a clause does not require it"
    );
    assert!(!one_key.admits(Value::dict(B)), "and names no other key");
}

/// A literal key of any constant kind is a label, not only a `str` one.
///
/// The descriptor names the key by the constant it is, so reading a *value's*
/// key has to answer with the same constant -- and a key it cannot name is
/// read through its part's default instead, which a closed map shuts. Each
/// kind of constant is its own arm on both sides, and a missing one turns a
/// named key into an anonymous one that the map then rejects.
#[test]
fn a_literal_key_of_every_constant_kind_names_its_key() {
    const NONE_KEY: &[(Value, Value)] = &[(Value::of_kind(Kind::NoneType), Value::integer(1))];
    const TRUE_KEY: &[(Value, Value)] = &[(Value::boolean(true), Value::integer(1))];
    const FALSE_KEY: &[(Value, Value)] = &[(Value::boolean(false), Value::integer(1))];
    const ONE_KEY: &[(Value, Value)] = &[(Value::integer(1), Value::integer(1))];
    const TWO_KEY: &[(Value, Value)] = &[(Value::integer(2), Value::integer(1))];
    const RAW_KEY: &[(Value, Value)] = &[(Value::word(b"a", Kind::Bytes), Value::integer(1))];

    /// One constant, a dict whose key it names, and one of the same part it
    /// does not.
    type Case = (
        Operand,
        &'static [(Value, Value)],
        &'static [(Value, Value)],
    );

    let cases: [Case; 4] = [
        (Operand::NoneType, NONE_KEY, ONE_KEY),
        (Operand::Boolean(true), TRUE_KEY, FALSE_KEY),
        (Operand::Integer(1), ONE_KEY, TWO_KEY),
        (Operand::Word(b"a".to_vec(), Kind::Bytes), RAW_KEY, ONE_KEY),
    ];
    for (constant, named, other) in cases {
        let pool = Pool(vec![constant.clone()]);
        let map = lower(
            &Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![MapClause {
                    key: Schema::Literal(ConstIx::new(0)),
                    value: Schema::Int,
                }]
                .into(),
            },
            &pool,
        )
        .expect("a literal key lowers");
        assert!(map.admits(Value::dict(named)), "{constant:?} names its key");
        assert!(
            !map.admits(Value::dict(other)),
            "{constant:?} names no other"
        );
    }
}

/// An operand the pool cannot read refuses the constraint that names it.
#[test]
fn an_operand_the_pool_cannot_read_refuses() {
    let pool = Pool(vec![Operand::Float(0.5)]);
    assert!(
        lower(
            &Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
            },
            &pool,
        )
        .is_none(),
        "a float bound is not an integer set"
    );
}

/// A length bound lands in the kinds the base has and in no other: the bound
/// over a word admits no sequence, and the bound over a sequence no word.
#[test]
fn a_length_bound_stays_within_the_kinds_of_its_base() {
    const ONE: &[Value] = &[Value::integer(1)];
    let pool = empty_pool();
    let over_words = lower(
        &Schema::Refine {
            base: Arc::new(Schema::Str),
            constraints: vec![Constraint::MinLen(1)].into(),
        },
        &pool,
    )
    .expect("a bound over words lowers");
    assert!(over_words.admits(Value::word(b"a", Kind::Str)));
    assert!(!over_words.admits(Value::sequence(ONE, Kind::List)));

    let over_lists = lower(
        &Schema::Refine {
            base: Arc::new(Schema::list(SeqShape::homogeneous(Schema::ANYTHING))),
            constraints: vec![Constraint::MinLen(1)].into(),
        },
        &pool,
    )
    .expect("a bound over sequences lowers");
    assert!(over_lists.admits(Value::sequence(ONE, Kind::List)));
    assert!(!over_lists.admits(Value::word(b"a", Kind::Str)));
    assert!(!over_lists.admits(Value::sequence(ONE, Kind::Tuple)));
}

/// An integer bound no float equals is read against the neighbour on the
/// bound's own side, and which side that is decides which neighbour.
///
/// Past 2^53 the floats are two apart, so an odd integer there lies between two
/// of them and a bound written over it cuts at one or the other. A lower bound
/// cuts at the float *above*, an upper bound at the float *below*; reading the
/// nearest one instead moves the boundary by one representable step, and
/// reading the wrong side of it turns a lower bound into an upper one.
#[test]
fn a_float_bound_over_an_inexact_integer_cuts_at_the_right_neighbour() {
    // Two integers with no float of their own, one either side of the rounding:
    // the nearest float is *below* the first and *above* the second, so between
    // them they reach both directions the neighbour is picked in. Which side the
    // bound is on decides which neighbour, and the rounding decides nothing.
    const ROUNDS_DOWN: (i64, f64, f64) = (
        9_007_199_254_740_993,
        9_007_199_254_740_992.0,
        9_007_199_254_740_994.0,
    );
    const ROUNDS_UP: (i64, f64, f64) = (
        9_007_199_254_740_995,
        9_007_199_254_740_994.0,
        9_007_199_254_740_996.0,
    );

    for (between, below, above) in [ROUNDS_DOWN, ROUNDS_UP] {
        let pool = Pool(vec![Operand::Integer(between)]);
        let refined = |constraints: Vec<Constraint>| {
            lower(
                &Schema::Refine {
                    base: Arc::new(Schema::Float),
                    constraints: constraints.into(),
                },
                &pool,
            )
            .expect("a small refinement")
        };

        for lower_bound in [Constraint::Ge, Constraint::Gt] {
            let set = refined(vec![lower_bound(OperandIx::new(0))]);
            assert!(
                set.admits(Value::float(above)),
                "a lower bound over {between} admits the float above it"
            );
            assert!(
                !set.admits(Value::float(below)),
                "a lower bound over {between} refuses the float below it"
            );
        }
        for upper_bound in [Constraint::Le, Constraint::Lt] {
            let set = refined(vec![upper_bound(OperandIx::new(0))]);
            assert!(
                set.admits(Value::float(below)),
                "an upper bound over {between} admits the float below it"
            );
            assert!(
                !set.admits(Value::float(above)),
                "an upper bound over {between} refuses the float above it"
            );
        }

        // No float is on both sides, so the two together admit none -- which is
        // the answer only where each cut at its own neighbour.
        assert!(
            refined(vec![
                Constraint::Ge(OperandIx::new(0)),
                Constraint::Le(OperandIx::new(0)),
            ])
            .is_empty(),
            "no float lies between two adjacent ones ({between})"
        );
    }
}

/// A clause over a key kind with finitely many keys is exhausted by naming them.
///
/// `bool` has two keys and `None` has one, so a constraint that excludes every
/// key of such a part leaves its default governing nothing. Reading the default
/// as a witness anyway reports a map inhabited that holds no dict -- which is a
/// refutation standing on a value nobody has, since the difference below is the
/// dicts a `bool`-keyed map holds and a map keyed by both booleans does not.
#[test]
fn a_clause_over_a_finite_key_part_is_exhausted_by_its_keys() {
    let pool = Pool(vec![Operand::Boolean(true), Operand::Boolean(false)]);
    let keyed = |key: Schema| {
        lower(
            &Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![MapClause {
                    key,
                    value: Schema::Int,
                }]
                .into(),
            },
            &pool,
        )
        .expect("a small mapping")
    };

    let literal = |slot| Schema::Literal(ConstIx::new(slot));
    let both = Schema::Union(vec![literal(0), literal(1)].into());

    let apart = |a: &Descr, b: &Descr| a.intersect(&b.complement()).expect("a small difference");

    // Both booleans named: the kind has no key left, so neither side holds a
    // dict the other misses.
    assert!(
        apart(&keyed(Schema::Bool), &keyed(both.clone())).is_empty(),
        "every bool key is one of the two the names list"
    );
    assert!(
        apart(&keyed(both), &keyed(Schema::Bool)).is_empty(),
        "and every named key is a bool"
    );

    // One of the two named: the other is a key the clause still governs, so the
    // difference holds a dict. Reading the part as exhausted by *either* name
    // reports it empty.
    assert!(
        !apart(&keyed(Schema::Bool), &keyed(literal(0))).is_empty(),
        "one boolean named leaves the other"
    );
    assert!(
        !apart(&keyed(Schema::Bool), &keyed(literal(1))).is_empty(),
        "whichever of the two it is"
    );

    // `None` is the other finite part, and it has one key rather than two.
    let none_pool = Pool(vec![Operand::NoneType]);
    let none_keyed = |key: Schema| {
        lower(
            &Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![MapClause {
                    key,
                    value: Schema::Int,
                }]
                .into(),
            },
            &none_pool,
        )
        .expect("a small mapping")
    };
    assert!(
        apart(&none_keyed(Schema::NoneType), &none_keyed(literal(0))).is_empty(),
        "the one key of the None part is exhausted by naming it"
    );
    assert!(
        !apart(&none_keyed(Schema::NoneType), &none_keyed(Schema::Nothing)).is_empty(),
        "and naming nothing leaves it"
    );
}
