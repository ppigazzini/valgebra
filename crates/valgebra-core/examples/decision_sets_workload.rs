//! One corpus of relations asked of the two deciders: what a rule's answer
//! costs, and what building the sets costs for the same answer.
//!
//! The pages say the rules are asked first because building a set
//! representation costs far more than a rule that already answers, and that
//! ratio is the whole argument for the order. A ratio of two workloads over
//! different pairs measures the pairs as much as the deciders, so this one asks
//! nine pairs of both: `rules` through the public relation, which the rules
//! answer on every pair here without reaching the sets, and `sets` through the
//! reading `Schema::subtype_relation_under` takes where they decline -- each
//! side lowered, the supertype complemented, the two met under the same work
//! allowance, and the difference asked for a value.
//!
//! The two halves fold the same three-valued answer per pair, so their
//! checksums are one fold at two iteration counts: a pair one decider answered
//! and the other declined, or answered the other way, moves one and not the
//! other. The iteration counts differ by a thousand because the costs do, and
//! each half is held well above the process's startup, which is about half a
//! million instructions.
//!
//! The pairs are the common ones: a scalar against a union of kinds, two
//! kinds, a list against a wider one, two tuples of one arity, a record
//! against a wider record, a literal table against a wider one, two lower
//! bounds, a set against a frozen set, and an optional against its base. Keep
//! the corpus and both iteration counts fixed; changing either moves both
//! records.
//!
//! A third reading, `lengths`, asks the sets a corpus of its own: a list of
//! integers bounded in length against a looser bound and a tighter one. A
//! length bound is an automaton with a state per element, met with the
//! element's shape, so this is the shape that reads what a bound costs per
//! element; no pair of the nine has one. Its bounds are short enough that
//! every pair is decided, so a change that lets a longer bound lower moves
//! the count and not the answers.

use std::sync::Arc;

use valgebra_core::descr::budget;
use valgebra_core::descr::classes::Class;
use valgebra_core::descr::lower::{Constants, Operand, WORK, lower};
use valgebra_core::{
    ClassIx, ConstIx, Constraint, Field, LeafRelations, Openness, OperandIx, Relation, Schema,
    SeqShape, Verdict,
};

/// Iterations of the corpus through the rules: a rule's answer is a few
/// hundred instructions, so this many put the half far above the startup.
const RULES_ITERATIONS: usize = 20_000;

/// Iterations through the sets, a thousand times fewer, which leaves the two
/// halves within a factor of two of each other in total.
const SETS_ITERATIONS: usize = 20;

/// Iterations of the length bounds through the sets. A bound builds a state per
/// element and meets it with the element's shape, so two passes are already
/// far above the startup.
const LENGTHS_ITERATIONS: usize = 2;

/// A pool that reads every operand and constant as its own index, and orders
/// two bounds by it, so both deciders can read what a literal and a bound name
/// without an interpreter.
struct Indexed;

impl Constants for Indexed {
    fn operand(&self, index: OperandIx) -> Option<Operand> {
        i64::try_from(index.get()).ok().map(Operand::Integer)
    }

    fn constant(&self, index: ConstIx) -> Option<Operand> {
        i64::try_from(index.get()).ok().map(Operand::Integer)
    }

    fn class(&self, _index: ClassIx) -> Option<Class> {
        None
    }
}

impl LeafRelations for Indexed {
    fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
        None
    }

    fn compare(&self, left: OperandIx, right: OperandIx) -> Option<core::cmp::Ordering> {
        Some(left.get().cmp(&right.get()))
    }
}

/// The relation as the sets give it: the reading the decision takes where the
/// rules decline, for a pair with no reference to unfold and no literal key.
fn by_the_sets(sub: &Schema, sup: &Schema) -> Relation {
    let Some(mine) = lower(sub, &Indexed) else {
        return Relation::Unknown;
    };
    if mine.emptiness() == Verdict::Empty {
        return Relation::Holds;
    }
    let Some(theirs) = lower(sup, &Indexed) else {
        return Relation::Unknown;
    };
    budget::under(WORK, || {
        mine.intersect(&theirs.complement())
            .map(|difference| difference.emptiness())
    })
    .map_or(Relation::Unknown, |emptiness| match emptiness {
        Verdict::Empty => Relation::Holds,
        Verdict::Inhabited => Relation::Fails,
        Verdict::Unknown => Relation::Unknown,
    })
}

fn required(name: &str, schema: Schema) -> Field {
    Field {
        name: name.into(),
        schema,
        required: true,
    }
}

/// The nine pairs, each a subject and the schema it is asked against.
fn corpus() -> Vec<(Schema, Schema)> {
    let list = |element| Schema::list(SeqShape::homogeneous(element));
    let literals = |n: usize| Schema::union((0..n).map(|i| Schema::Literal(ConstIx::new(i))));
    let at_least = |n: usize| Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Ge(OperandIx::new(n))].into(),
    };
    let optional = |schema| Schema::union([schema, Schema::NoneType]);
    vec![
        (Schema::Bool, Schema::union([Schema::Int, Schema::Str])),
        (Schema::Int, Schema::Str),
        (
            list(Schema::Int),
            list(Schema::union([Schema::Int, Schema::Str])),
        ),
        (
            Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Str])),
            Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Int])),
        ),
        (
            Schema::record(vec![required("a", Schema::Int)], Openness::Closed),
            Schema::record(vec![required("a", optional(Schema::Int))], Openness::Closed),
        ),
        (literals(8), literals(9)),
        (at_least(5), at_least(0)),
        (Schema::set(Schema::Int), Schema::frozen_set(Schema::Int)),
        (optional(Schema::Int), Schema::Int),
    ]
}

/// The four length bounds, each a subject and the bound it is asked against:
/// a longer lower bound inside a shorter one, the converse, a shorter upper
/// bound inside a longer one, and a bound on both sides inside a lower one.
fn lengths() -> Vec<(Schema, Schema)> {
    let bounded = |constraints| {
        Schema::refine(
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            constraints,
        )
    };
    vec![
        (
            bounded(vec![Constraint::MinLen(32)]),
            bounded(vec![Constraint::MinLen(16)]),
        ),
        (
            bounded(vec![Constraint::MinLen(16)]),
            bounded(vec![Constraint::MinLen(32)]),
        ),
        (
            bounded(vec![Constraint::MaxLen(16)]),
            bounded(vec![Constraint::MaxLen(32)]),
        ),
        (
            bounded(vec![Constraint::MinLen(8), Constraint::MaxLen(16)]),
            bounded(vec![Constraint::MinLen(4)]),
        ),
    ]
}

fn main() {
    // A reading is named rather than defaulted: a run measuring the wrong
    // half would print a checksum of the wrong size and nothing else.
    let reading = std::env::args().nth(1);
    let (iterations, decide, pairs): (usize, fn(&Schema, &Schema) -> Relation, _) =
        match reading.as_deref() {
            Some("rules") => (
                RULES_ITERATIONS,
                |sub: &Schema, sup: &Schema| sub.subtype_relation_under(sup, &Indexed, &[]),
                corpus(),
            ),
            Some("sets") => (SETS_ITERATIONS, by_the_sets, corpus()),
            Some("lengths") => (LENGTHS_ITERATIONS, by_the_sets, lengths()),
            _ => {
                eprintln!("usage: decision_sets_workload rules|sets|lengths");
                std::process::exit(2);
            }
        };
    // Fold the three-valued answer, so a pair moving between decided and
    // declined, or between the two answers, moves the checksum.
    let weigh = |relation: Relation| match relation {
        Relation::Holds => 3,
        Relation::Fails => 2,
        Relation::Unknown => 1,
    };
    let mut checksum: usize = 0;
    for _ in 0..iterations {
        for (sub, sup) in &pairs {
            checksum += weigh(decide(std::hint::black_box(sub), std::hint::black_box(sup)));
        }
    }
    println!("{checksum}");
}
