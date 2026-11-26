//! One corpus of meets asked two ways: `is_empty`, and the relation below
//! `nothing`, which is the same question by the other procedure.
//!
//! The emptiness of a set is its relation below the empty set, and the two
//! procedures are held to one answer by the laws. Their cost has no such
//! holder: a reading one of them asks and the other does not can make the same
//! question dear by one route and cheap by the other, and a workload asking only
//! one route never sees it. So `empty` asks `Schema::is_empty_with` of each meet
//! and `below` asks `Schema::subtype_relation_under` below `Schema::Nothing`,
//! and the two counts are one corpus read by each procedure.
//!
//! The meets are the shapes a translation from JSON Schema writes for a cell of
//! a model checker: an integer kind, as `int | float & MultipleOf(1)` beside the
//! complement of `bool`, met with a range and the complements of `n` ranges, at
//! 8 and 16 complements and not empty; the same at 16 with two ranges covering
//! the range, so empty; the 32-complement cell `decision_sets_workload` asks of
//! the sets alone; a meet of two complements whose rest a rule decides; a list
//! beside the complements of `list[int]` and of the JSON values, empty only
//! below the fixpoint; and a cell of sixty complements whose every bound is
//! spelled per kind, past the nodes one lowering reads, not empty. Keep the
//! corpus fixed; changing it moves both records.

use valgebra_core::descr::classes::Class;
use valgebra_core::descr::lower::{Constants, Operand};
use valgebra_core::{
    ClassIx, ConstIx, Constraint, DefIx, LeafRelations, MapClause, OperandIx, Relation, Schema,
    SeqShape,
};

/// A pool that reads every operand as its own index, and orders two bounds by
/// it, so both procedures read what a bound names without an interpreter.
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

fn bounds(low: usize, high: usize) -> Vec<Constraint> {
    vec![
        Constraint::Ge(OperandIx::new(low)),
        Constraint::Le(OperandIx::new(high)),
    ]
}

/// The integers as a translation from JSON Schema writes them: an `int`, or a
/// `float` that is a whole number.
fn integral() -> Schema {
    Schema::union([
        Schema::Int,
        Schema::refine(
            Schema::Float,
            vec![Constraint::MultipleOf(OperandIx::new(1))],
        ),
    ])
}

/// `{"type": "integer", "minimum": low, "maximum": high}`: a bounded `float` or
/// `int`, met with the integers and the complement of `bool`.
fn number(low: usize, high: usize) -> Schema {
    Schema::meet([
        Schema::union([
            Schema::refine(Schema::Float, bounds(low, high)),
            Schema::refine(Schema::Int, bounds(low, high)),
        ]),
        integral(),
        Schema::Bool.complement(),
    ])
}

/// The integer kind met with `[500, 1000]` and the complements of `n` ranges,
/// each ending below 1000, so 1000 is a value of the meet; `covered` swaps two
/// of them for ranges covering `[450, 1001]`, which empties it.
fn cell(n: usize, covered: bool) -> Schema {
    let kind = Schema::meet([integral(), Schema::Bool.complement()]);
    let mut members = vec![kind, number(500, 1000)];
    for i in 0..n {
        let (low, high) = match (covered, i) {
            (true, 0) => (450, 800),
            (true, 1) => (780, 1001),
            _ => {
                let low = (37 * i) % 400;
                (low, low + 200 + (53 * i) % 399)
            }
        };
        members.push(number(low, high).complement());
    }
    Schema::meet(members)
}

/// A range with thirty-two taken out of it, `decision_sets_workload`'s `steps`
/// cell, not empty.
fn steps_cell() -> Schema {
    let range = |low: usize, high: usize| {
        Schema::union([
            Schema::refine(Schema::Int, bounds(low, high)),
            Schema::refine(
                Schema::Float,
                std::iter::once(Constraint::MultipleOf(OperandIx::new(1)))
                    .chain(bounds(low, high))
                    .collect(),
            ),
        ])
    };
    Schema::meet(
        std::iter::once(range(0, 40 * 32))
            .chain((0..32).map(|i| range(40 * i, 40 * i + 20).complement())),
    )
}

/// A bound over a number spelled per kind, as a translation from JSON Schema
/// writes one when it reads `int` and `float` apart.
fn per_kind(bound: Constraint) -> Schema {
    Schema::union([
        Schema::refine(Schema::Int, vec![bound.clone()]),
        Schema::refine(Schema::Float, vec![bound]),
    ])
}

/// The integers between `lows`' greatest and `highs`' least, every bound a
/// member of its own, beside the complement of `bool`.
fn integers_within(lows: &[usize], highs: &[usize]) -> Vec<Schema> {
    std::iter::once(integral())
        .chain(
            lows.iter()
                .map(|&low| per_kind(Constraint::Ge(OperandIx::new(low)))),
        )
        .chain(
            highs
                .iter()
                .map(|&high| per_kind(Constraint::Le(OperandIx::new(high)))),
        )
        .chain(std::iter::once(Schema::Bool.complement()))
        .collect()
}

/// `[500, 1000]` written as seven bounds a side, met with the complements of `n`
/// cells written the same way, each ending below 1000, so 1000 is a value: a
/// model checker's cell over an integer argument, at sixty complements past the
/// nodes one lowering reads.
fn per_kind_cell(n: usize) -> Schema {
    let mut members = integers_within(
        &[500, 497, 494, 491, 488, 485, 482],
        &[1000, 1003, 1006, 1009, 1012, 1015, 1018],
    );
    for i in 0..n {
        let low = (37 * i) % 400;
        let high = low + 200 + (53 * i) % 399;
        members.push(Schema::meet(integers_within(&[low], &[high])).complement());
    }
    Schema::meet(members)
}

/// The JSON values, `μj. None | bool | int | float | str | list[j] | dict[str, j]`.
fn json() -> (Schema, Vec<Schema>) {
    let at = Schema::Ref(DefIx::new(0));
    let body = Schema::union([
        Schema::NoneType,
        Schema::Bool,
        Schema::Int,
        Schema::Float,
        Schema::Str,
        Schema::list(SeqShape::homogeneous(at.clone())),
        Schema::mapping(MapClause {
            key: Schema::Str,
            value: at.clone(),
        }),
    ]);
    (at, vec![body])
}

/// Each meet, with the definitions it reads.
fn corpus() -> Vec<(Schema, Vec<Schema>)> {
    let (json, defs) = json();
    let two = Schema::meet([
        Schema::Int,
        Schema::Bool.complement(),
        Schema::refine(Schema::Int, bounds(3, 3)).complement(),
    ]);
    let below_the_fixpoint = Schema::meet([
        Schema::list(SeqShape::homogeneous(json.clone())),
        Schema::list(SeqShape::homogeneous(Schema::Int)).complement(),
        json.complement(),
    ]);
    vec![
        (cell(8, false), Vec::new()),
        (cell(16, false), Vec::new()),
        (cell(16, true), Vec::new()),
        (steps_cell(), Vec::new()),
        (two, Vec::new()),
        (below_the_fixpoint, defs),
        (per_kind_cell(60), Vec::new()),
    ]
}

fn main() {
    // A reading is named rather than defaulted: a run measuring the wrong
    // procedure would print a checksum of the wrong value and nothing else.
    let reading = std::env::args().nth(1);
    let below = match reading.as_deref() {
        Some("empty") => false,
        Some("below") => true,
        _ => {
            eprintln!("usage: decision_emptiness_workload empty|below");
            std::process::exit(2);
        }
    };
    // Fold each answer into its place, so a meet moving between decided and
    // declined, or between the two answers, moves the checksum.
    let mut checksum: usize = 0;
    for (meet, defs) in &corpus() {
        let answer = if below {
            match std::hint::black_box(meet).subtype_relation_under(
                &Schema::Nothing,
                &Indexed,
                defs,
            ) {
                Relation::Holds => 3,
                Relation::Fails => 2,
                Relation::Unknown => 1,
            }
        } else if std::hint::black_box(meet).is_empty_with(&Indexed, defs) {
            3
        } else {
            2
        };
        checksum = checksum * 4 + answer;
    }
    println!("{checksum}");
}
