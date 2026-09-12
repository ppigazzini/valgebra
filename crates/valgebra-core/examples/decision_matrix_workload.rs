//! The relation shapes the structural readings decide, which no other workload
//! carries.
//!
//! The three sibling decision workloads measure a wide record, a union, a
//! literal table, a complement of a scalar and a nesting. None of them holds a
//! refinement, a sequence read as a star, a complement against a class, a
//! dataclass against a union, or a reference on the supertype's side -- so a
//! week of readings that cut the relation matrix's slow set from 4.14 ms to
//! 0.68 moved every one of those four workloads by 0.00%. A gate that cannot
//! see a change cannot hold the next one.
//!
//! **This workload is a floor, not evidence.** It was written after the
//! readings it exercises, so its first recorded count is a baseline the *next*
//! change is held to and says nothing about the ones already landed -- a
//! workload chosen to make a delta match a win is not a measurement. The number
//! to compare against is `--against <rev>`, which builds both sides.
//!
//! The corpus is the matrix's own slow set, with the pairs named for what makes
//! each expensive rather than for the answer. Keep it and `ITERATIONS` fixed;
//! changing either moves the budget and requires re-recording it.
//!
//! The checksum folds the **three-valued** answer rather than the public
//! `bool`. Folded as a bool it is zero here -- every pair below is refuted or
//! declined -- and a sum of zero stays zero however badly the procedure breaks,
//! which is a guard that guards nothing.
//!
//! Both halves of that were measured rather than assumed, and the first attempt
//! got both wrong. Written against `NoLeafRelations` -- which declines every
//! class question -- `outside_every_kind` and the complement's kind search
//! could never fire, so the workload carried their cost and none of their
//! saving: deleting the union distribution read **0.15% faster**, and the
//! checksum did not move. With the oracle below answering as the bindings do
//! for a plain class, the same deletion moves the checksum from 68,000 to
//! 60,000 and the recorded count is 226M rather than 645M.
//!
//! So on this corpus these readings buy completeness and not only speed: four
//! pairs per iteration fall from refuted to undecided when one is removed,
//! because the descriptor cannot reach them either. A cost-only change is what
//! the instruction count is for, and the checksum is the guard that the count
//! was taken of the same answers.

use std::sync::Arc;

use valgebra_core::{
    ClassIx, Constraint, DefIx, Field, Kind, LeafRelations, OperandIx, Relation, Schema, SeqShape,
};

/// Two classes that lay down no builtin layout, answered as the bindings answer
/// a plain Python class.
///
/// `NoLeafRelations` declines every one of these, and declining is what the
/// class readings here are *for*: with it in place `outside_every_kind` and the
/// complement's kind search can never fire, so the workload would carry their
/// cost and none of their saving. Measured that way it read 0.15% **faster**
/// with the union distribution deleted, which is the shape of a workload that
/// exercises nothing.
struct PlainClasses;

impl valgebra_core::descr::lower::Constants for PlainClasses {}

impl LeafRelations for PlainClasses {
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool> {
        match (sub, sup) {
            (Schema::Instance(a), Schema::Instance(b)) => Some(a == b),
            _ => None,
        }
    }

    /// A class laying down no layout confines its instances to no kind, and a
    /// subclass of it may derive from a builtin, so this declines.
    fn class_admits_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        None
    }

    /// A *direct* instance of such a class is a plain object: no kind the
    /// partition names.
    fn direct_instance_of_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        Some(false)
    }

    /// And no builtin derives from one.
    fn kind_derives_from(&self, _kind: Kind, _class: ClassIx) -> Option<bool> {
        Some(false)
    }

    fn atom_denotes_a_set(&self, atom: &Schema) -> Option<bool> {
        Some(matches!(atom, Schema::Instance(_)))
    }
}

/// Iterations per relation, on the siblings' scale: large enough that process
/// startup is a rounding error against the work measured.
const ITERATIONS: usize = 2_000;

/// `list[T]`, the shape read as `T*` where two sequences meet.
fn list_of(element: Schema) -> Schema {
    Schema::list(SeqShape::homogeneous(element))
}

/// A refinement carrying one constraint: the node whose pair had no reading of
/// its own until the constraint rule learned to hand a pair on.
fn refined(base: Schema, constraint: Constraint) -> Schema {
    Schema::Refine {
        base: Arc::new(base),
        constraints: vec![constraint].into(),
    }
}

/// A class met with the attributes its instances carry -- what a dataclass
/// lowers to, and the meet whose witness is a *direct* instance.
fn dataclass(class: ClassIx, width: usize) -> Schema {
    let fields = (0..width)
        .map(|i| Field {
            name: format!("f{i}").into(),
            schema: Schema::Int,
            required: true,
        })
        .collect::<Vec<_>>();
    Schema::meet([
        Schema::Instance(class),
        Schema::AttrRecord {
            fields: fields.into(),
        },
    ])
}

/// A recursive record, reached through the reference that names it.
fn tree_defs() -> Vec<Schema> {
    vec![Schema::KeyedMap {
        fields: vec![
            Field {
                name: "value".into(),
                schema: Schema::Int,
                required: true,
            },
            Field {
                name: "left".into(),
                schema: Schema::Ref(DefIx::new(0)),
                required: false,
            },
        ]
        .into(),
        defaults: Vec::new().into(),
    }]
}

fn main() {
    let long_list = refined(list_of(Schema::Int), Constraint::MinLen(2));
    let positive = refined(Schema::Int, Constraint::Ge(OperandIx::new(0)));
    let long_text = refined(Schema::Str, Constraint::MinLen(1));
    let longer_list = refined(list_of(Schema::Int), Constraint::MinLen(3));
    let bare_list = list_of(Schema::Int);
    let strings = list_of(Schema::Str);
    let nested = list_of(list_of(Schema::Int));
    let point = dataclass(ClassIx::new(0), 2);
    let kinds = Schema::union([Schema::Int, Schema::Str]);
    let optional = Schema::union([Schema::Int, Schema::NoneType]);
    let not_int = Schema::Complement(Arc::new(Schema::Int));
    let a_class = Schema::Instance(ClassIx::new(1));
    let defs = tree_defs();
    let tree = Schema::Ref(DefIx::new(0));

    // Fold the three-valued verdict so nothing is optimized away, and so a
    // pair moving between decided and declined is visible.
    let weigh = |relation: Relation| match relation {
        Relation::Holds => 3,
        Relation::Fails => 2,
        Relation::Unknown => 1,
    };
    let ask =
        |sub: &Schema, sup: &Schema| weigh(sub.subtype_relation_under(sup, &PlainClasses, &[]));
    let ask_under =
        |sub: &Schema, sup: &Schema| weigh(sub.subtype_relation_under(sup, &PlainClasses, &defs));
    let mut checksum: usize = 0;
    for _ in 0..ITERATIONS {
        // A refinement on both sides, whose bases are of two kinds: the pair
        // the constraint rule leaves unproven and hands on. And one the rule
        // does prove, so the arm before the hand-on is measured too.
        checksum += ask(&long_list, &positive);
        checksum += ask(&positive, &long_list);
        checksum += ask(&longer_list, &long_list);
        // Two sequences whose elements share no value, separated by the bound
        // that rules the empty sequence out.
        checksum += ask(&long_list, &strings);
        checksum += ask(&long_list, &nested);
        // A bare sequence against a bound it cannot meet, a word refinement
        // against a list one, and a refinement below its own base.
        checksum += ask(&bare_list, &long_list);
        checksum += ask(&long_text, &long_list);
        checksum += ask(&long_list, &bare_list);
        // A meet of a class and its attributes against a union of kinds, against
        // one kind, and against the class it is built on.
        checksum += ask(&point, &kinds);
        checksum += ask(&point, &optional);
        checksum += ask(&point, &Schema::Int);
        checksum += ask(&point, &Schema::Instance(ClassIx::new(0)));
        // A complement against a class, which names its witness among the kinds
        // its inner schema is not.
        checksum += ask(&not_int, &a_class);
        checksum += ask(&not_int, &point);
        // A reference on the supertype's side, unfolded once for its kind.
        checksum += ask_under(&point, &tree);
        checksum += ask_under(&bare_list, &tree);
    }
    // Printing forces the checksum to be observed.
    println!("checksum={checksum}");
}
