//! The per-validator precompute: record-field lookups, literal-union decision
//! tables, and compiled string patterns, built once and reused across calls.

use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt, PyString};
use regex_automata::util::syntax;
use regex_automata::{MatchKind, meta};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::sync::{Arc, LazyLock, Mutex, Weak};
use valgebra_core::{Constraint, DefIx, MapClause, OperandIx, Schema, SeqKind};

use super::memo::{self, MemoPlan, for_each_child};
use crate::input::Value;
use crate::oracle::is_stdlib_class;

pub(crate) struct RecordPlan {
    pub(crate) by_name: FxHashMap<Arc<str>, usize>,
    pub(crate) required: usize,
    /// The interned key of each declared field, in field order.
    ///
    /// The same thing [`AttrsPlan`] holds for `getattr`, for the other lookup a
    /// record does. The explain walk asks the dict for each declared key by
    /// name, and a Rust `&str` handed to `get_item` is decoded into a fresh
    /// `PyString` and hashed before the lookup can start -- per field, per
    /// call. An interned key carries its hash with it, so the lookup is the
    /// probe alone. The deciding walk asks by these keys too wherever they
    /// settle the record (`keyed_map_asks_for_its_keys`), and scans the dict
    /// once, resolving each key it finds through `by_name`, where a clause reads
    /// an undeclared key together with its value or a probe raised.
    ///
    /// **Interned rather than merely built, and that is most of what the key
    /// is for.** A dict probe compares the key it is given with the key it
    /// holds by *pointer* first and only then by hash and contents, so two
    /// interned keys settle in one comparison and two equal strings that are
    /// not the same object walk their bytes. Every dict Python writes as a
    /// literal, every `**kwargs`, and every `__dict__` carries interned keys,
    /// so interning this side is what lets the probe take its fast path:
    /// measured on the fifty-field record walk, the two sides interned read
    /// **29.1% cheaper** than neither, and interning one side alone buys about
    /// a percent. Interning costs a hash and a probe of the interpreter's own
    /// table, once per field per validator.
    pub(crate) keys: Vec<Py<PyString>>,
    /// The fields the deciding walk reads before the declared order: each one
    /// decided by its own value and declared after one that is not, in
    /// declared order.
    ///
    /// Fields are stored sorted by name, so a tag such as `type` beside a
    /// recursive `left` is read after the child, and a union of records told
    /// apart by the tag walked the whole child in every branch it refused:
    /// branches to the power of the depth. Read first, a tag refuses a wrong
    /// branch in one comparison whatever it is called. Empty for every record
    /// whose declared order already reads them first, which is most of them,
    /// and then the deciding walk is the declared loop alone.
    pub(crate) early: Box<[usize]>,
}

/// Whether a field is decided by its own value, without a walk below it: a
/// scalar kind, a literal, or a union of them.
pub(crate) fn decided_by_its_value(schema: &Schema) -> bool {
    match schema {
        Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Str
        | Schema::Bytes
        | Schema::Literal(_) => true,
        Schema::Union(members) => members.iter().all(decided_by_its_value),
        _ => false,
    }
}

/// The fields a record's deciding walk reads first ([`RecordPlan::early`]).
fn early_fields(fields: &[valgebra_core::Field]) -> Box<[usize]> {
    let decided = |at: &usize| {
        fields
            .get(*at)
            .is_some_and(|f| decided_by_its_value(&f.schema))
    };
    (0..fields.len())
        .skip_while(decided)
        .filter(decided)
        .collect()
}

/// The interned attribute names of one [`Schema::AttrRecord`] node, in field
/// order.
///
/// `getattr` takes a name, and passing a Rust `&str` builds a fresh `PyString`
/// for every attribute of every value checked. The names are a property of the
/// schema, so they are built once here and the walk hands the interpreter the
/// same objects each time.
pub(crate) struct AttrsPlan {
    pub(crate) names: Vec<Py<PyString>>,
}

/// The attribute index for a whole validator, keyed and rebuilt like
/// [`RecordIndex`].
pub(crate) type AttrsIndex = FxHashMap<usize, AttrsPlan>;

/// The record index for a whole validator: each record's `fields`-buffer address
/// mapped to its [`RecordPlan`]. The buffer address is stable for the life of the
/// (immutable) schema, and the index is rebuilt per validator from its own
/// schema, so an entry always refers to the same live node.
pub(crate) type RecordIndex = FxHashMap<usize, RecordPlan>;

/// A precomputed decision table for one union whose members are all literals: the
/// `int`-typed literal values that fit a machine integer, and the `str`-typed
/// literal values. An exact `int`/`str` value's membership is then a single set
/// lookup instead of a scan of every branch.
///
/// And the addresses of the pooled constants behind those values, so the
/// constant itself -- an interned string, a cached small integer, which is what
/// a program spelling the literal in its own source holds -- is found by its
/// address, without decoding or hashing its text. The pool holds each constant
/// for as long as the validator lives, so no other object can have its address
/// meanwhile, and an exact `int` or `str` is equal to itself: an address found
/// here is a value the set beside it holds.
pub(crate) struct UnionPlan {
    ints: FxHashSet<i64>,
    strs: FxHashSet<Box<str>>,
    pooled: FxHashSet<usize>,
}

impl UnionPlan {
    /// Decide membership of `value` if the value's exact type is one this plan
    /// covers, else `None` to defer to the linear scan. Only an exact `int`
    /// matches `int` literals and only an exact `str` matches `str` literals
    /// (Python's cross-type equality is excluded by the same-type literal rule),
    /// so the set lookup is authoritative for those two types. A boolean, float,
    /// `None`, bytes, big integer, subclass instance, or JSON value returns `None`
    /// and is scanned linearly, matching `literal_matches` exactly.
    pub(crate) fn decide(&self, value: &Value<'_, '_>) -> Option<bool> {
        let Value::Py(v) = value else { return None };
        if v.is_exact_instance_of::<PyInt>() {
            if self.pooled.contains(&(v.as_ptr() as usize)) {
                return Some(true);
            }
            // A big integer (outside i64) cannot equal any i64-valued literal, so
            // deferring to the scan handles it against any big-integer literal.
            let i = v.extract::<i64>().ok()?;
            return Some(self.ints.contains(&i));
        }
        if v.is_exact_instance_of::<PyString>() {
            if self.pooled.contains(&(v.as_ptr() as usize)) {
                return Some(true);
            }
            let s = v.cast::<PyString>().ok()?.to_str().ok()?;
            return Some(self.strs.contains(s));
        }
        None
    }
}

/// The union index for a whole validator: each all-literal union's members-buffer
/// address mapped to its [`UnionPlan`]. Keyed and rebuilt like [`RecordIndex`].
pub(crate) type UnionIndex = FxHashMap<usize, UnionPlan>;

/// A union of records that each require one key, pinned to `int` or `str`
/// constants no two records share: the one record a dict can be a member of is
/// the one its value at that key names, so the deciding walk reads the key once
/// and walks that record, where the scan reads it once a record. The design
/// reference is `pydantic-core`'s tagged union, a lookup and then one branch.
///
/// Every record's constants are in the tables, so a tag of an exact `int` or
/// `str` that no table holds is no record's. Any other tag is the scan's.
pub(crate) struct TaggedPlan {
    /// The key every record requires, interned.
    pub(crate) key: Py<PyString>,
    ints: FxHashMap<i64, usize>,
    strs: FxHashMap<Box<str>, usize>,
}

/// What a tag says of a tagged union's members.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tagged {
    /// The member at this position, the one record whose constants hold it.
    Branch(usize),
    /// No record's constants hold it, so no member admits the dict.
    Nothing,
}

impl Tagged {
    fn of(position: Option<&usize>) -> Tagged {
        position.map_or(Tagged::Nothing, |position| Tagged::Branch(*position))
    }
}

impl TaggedPlan {
    /// What `tag` says of the union's members, or `None` where it is no exact
    /// `int` or `str`, which the scan reads.
    pub(crate) fn branch(&self, tag: &Bound<'_, PyAny>) -> Option<Tagged> {
        if tag.is_exact_instance_of::<PyInt>() {
            // A big integer equals no constant the table holds, and a record
            // pinning one has no plan, so it is the scan's.
            return Some(Tagged::of(self.ints.get(&tag.extract::<i64>().ok()?)));
        }
        if tag.is_exact_instance_of::<PyString>() {
            let text = tag.cast::<PyString>().ok()?.to_str().ok()?;
            return Some(Tagged::of(self.strs.get(text)));
        }
        None
    }
}

/// Each tagged union's members-buffer address mapped to its [`TaggedPlan`].
pub(crate) type TaggedIndex = FxHashMap<usize, TaggedPlan>;

/// An exact builtin type a reading settles a value of, as its bit: a value of
/// one of these types is what a leaf ([`LeafPlan`]) decides by its type, and
/// for an `int`, a `bool` or a `float` by its number, and a list or a dict is
/// what a tree of scalars ([`TreePlan`]) reads on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Exact {
    None = 1,
    Bool = 1 << 1,
    Int = 1 << 2,
    Float = 1 << 3,
    Str = 1 << 4,
    Bytes = 1 << 5,
    List = 1 << 6,
    Dict = 1 << 7,
}

impl Exact {
    /// Every exact type, in the order of their bits.
    pub(crate) const ALL: [Exact; 8] = [
        Exact::None,
        Exact::Bool,
        Exact::Int,
        Exact::Float,
        Exact::Str,
        Exact::Bytes,
        Exact::List,
        Exact::Dict,
    ];

    /// The place of this type among [`Exact::ALL`].
    fn place(self) -> usize {
        (self as u8).trailing_zeros() as usize
    }
}

/// A schema decided by a value's exact builtin type and, for an `int`, a
/// `bool` or a `float`, by the number it holds: a union, meet or complement of
/// scalar kinds, refinements of `int` by order bounds whose operands are exact
/// `int`s a machine word holds, and refinements of `float` by order bounds and
/// steps whose operands a float holds. JSON Schema's `integer`, `(int |
/// Annotated[float, MultipleOf(1)]) & ~bool`, is one, and its `number` another.
///
/// Every node of one reads a value's type before anything in it, and for a
/// value of an exact builtin type none runs Python or reads past the number: a
/// complement's inner walk decides, a refinement's comparison and step are the
/// two numbers' own. So where the levels to its deepest type test are free, a
/// leaf answers as the walk does.
///
/// A reference to a tree of scalars ([`TreePlan`]) is a leaf too: a value of a
/// scalar type is its leaves' answer, and a list or a dict, which the tree reads
/// on, is the walk's to read ([`Leaf::Walk`]). The JSON values met with a
/// keyword's guard, `(~NUMBER | Annotated[int, Ge(0)] | ...) & J`, are one.
#[derive(Clone)]
pub(crate) enum Leaf {
    /// A scalar kind, by the exact type it admits every value of; `int`
    /// admits a `bool` too, which subclasses it.
    Kind(Exact),
    /// A refinement of `int` by order bounds, as the inclusive range they
    /// admit; a `bool` is read as its number, as the comparison reads it.
    Ints(i64, i64),
    /// A refinement of `float`: every test holds of the number.
    Floats(Box<[FloatTest]>),
    /// A complement.
    Not(Box<Leaf>),
    /// A meet: the top, `anything`, where it has no member.
    All(Box<[Leaf]>),
    /// A union: the bottom, `nothing`, where it has no member.
    Any(Box<[Leaf]>),
    /// A reference to a tree of scalars: its leaves over a scalar, and the walk
    /// over a container it reads on.
    Tree(Box<TreeAtom>),
    /// What a tree answers over a container it reads on: the walk's answer,
    /// which no leaf settles. A plan reads it for a list or a dict alone, and no
    /// number is read for either.
    Walk,
}

/// A reference to a tree of scalars, as a leaf reads it.
#[derive(Clone)]
pub(crate) struct TreeAtom {
    /// The tree's members that are leaves, as their union.
    scalars: Leaf,
    /// The containers the tree reads on, their [`Exact`] bits.
    containers: u8,
}

/// One constraint of a refinement of `float`, its operand read as a float.
#[derive(Clone, Copy)]
pub(crate) enum FloatTest {
    /// An order bound.
    Order(CompareOp, f64),
    /// A step, `number % step == 0`: for two floats, the remainder `fmod`
    /// leaves, which is `nan` where Python's raises, for a step of zero.
    Step(f64),
}

impl FloatTest {
    fn holds(self, number: f64) -> bool {
        match self {
            FloatTest::Order(op, operand) => ordered(number, op, operand),
            FloatTest::Step(step) => number % step == 0.0,
        }
    }
}

/// The number a value of an exact numeric type holds, which a refinement reads.
#[derive(Clone, Copy)]
pub(crate) enum Number {
    Int(i64),
    Float(f64),
    /// A value of no numeric type.
    None,
}

impl Leaf {
    /// The leaf `schema` is, its operands read out of `pool` and its references
    /// to a tree of scalars out of `trees`, or `None` where some node of it is
    /// no leaf.
    fn of(
        schema: &Schema,
        pool: &[Py<PyAny>],
        py: Python<'_>,
        trees: &[Option<TreePlan>],
    ) -> Option<Leaf> {
        let operand = |index: OperandIx| Some(pool.get(index.get())?.bind(py));
        Some(match schema {
            Schema::Anything(_) => Leaf::All(Box::default()),
            Schema::Nothing => Leaf::Any(Box::default()),
            Schema::NoneType => Leaf::Kind(Exact::None),
            Schema::Bool => Leaf::Kind(Exact::Bool),
            Schema::Int => Leaf::Kind(Exact::Int),
            Schema::Float => Leaf::Kind(Exact::Float),
            Schema::Str => Leaf::Kind(Exact::Str),
            Schema::Bytes => Leaf::Kind(Exact::Bytes),
            Schema::Refine { base, constraints } => match **base {
                Schema::Int => {
                    let word = |index: OperandIx| {
                        let operand = operand(index)?;
                        if !operand.is_exact_instance_of::<PyInt>() {
                            return None;
                        }
                        operand.extract::<i64>().ok()
                    };
                    let (mut least, mut most) = (i64::MIN, i64::MAX);
                    for constraint in constraints.iter() {
                        match constraint {
                            Constraint::Ge(i) => least = least.max(word(*i)?),
                            Constraint::Gt(i) => least = least.max(word(*i)?.checked_add(1)?),
                            Constraint::Le(i) => most = most.min(word(*i)?),
                            Constraint::Lt(i) => most = most.min(word(*i)?.checked_sub(1)?),
                            _ => return None,
                        }
                    }
                    Leaf::Ints(least, most)
                }
                Schema::Float => Leaf::Floats(
                    constraints
                        .iter()
                        .map(|constraint| {
                            let (index, test): (_, fn(f64) -> FloatTest) = match constraint {
                                Constraint::Ge(i) => (i, |f| FloatTest::Order(CompareOp::Ge, f)),
                                Constraint::Gt(i) => (i, |f| FloatTest::Order(CompareOp::Gt, f)),
                                Constraint::Le(i) => (i, |f| FloatTest::Order(CompareOp::Le, f)),
                                Constraint::Lt(i) => (i, |f| FloatTest::Order(CompareOp::Lt, f)),
                                Constraint::MultipleOf(i) => (i, FloatTest::Step),
                                _ => return None,
                            };
                            Some(test(float_operand(operand(*index)?)?))
                        })
                        .collect::<Option<_>>()?,
                ),
                _ => return None,
            },
            Schema::Union(members) => Leaf::Any(Leaf::each(members, pool, py, trees)?),
            Schema::Intersection(members) => Leaf::All(Leaf::each(members, pool, py, trees)?),
            Schema::Complement(inner) => Leaf::Not(Box::new(Leaf::of(inner, pool, py, trees)?)),
            Schema::Ref(id) => {
                let tree = trees.get(id.get())?.as_ref()?;
                Leaf::Tree(Box::new(TreeAtom {
                    scalars: tree.scalars.clone(),
                    containers: tree.containers,
                }))
            }
            _ => return None,
        })
    }

    /// Each of `members` as a leaf, or `None` where one is none.
    fn each(
        members: &[Schema],
        pool: &[Py<PyAny>],
        py: Python<'_>,
        trees: &[Option<TreePlan>],
    ) -> Option<Box<[Leaf]>> {
        members
            .iter()
            .map(|m| Leaf::of(m, pool, py, trees))
            .collect()
    }

    /// The levels the walk opens from this node's own to its deepest type
    /// test: a refinement asks its base a level below itself, and a union asks
    /// a scalar branch where the branch's level is free.
    fn levels(&self) -> usize {
        match self {
            Leaf::Kind(_) => 1,
            Leaf::Ints(..) | Leaf::Floats(_) => 2,
            Leaf::Not(inner) => 1 + inner.levels(),
            Leaf::All(parts) | Leaf::Any(parts) => {
                1 + parts.iter().map(Leaf::levels).max().unwrap_or_default()
            }
            // The reference's own level, and the tree's union below it.
            Leaf::Tree(atom) => 1 + atom.scalars.levels(),
            Leaf::Walk => 0,
        }
    }

    /// Whether a reference to a tree is part of this leaf: a walk of it enters a
    /// definition, which the trail bounds and an armed memo keeps.
    fn reaches_a_tree(&self) -> bool {
        match self {
            Leaf::Tree(_) => true,
            Leaf::Not(inner) => inner.reaches_a_tree(),
            Leaf::All(parts) | Leaf::Any(parts) => parts.iter().any(Leaf::reaches_a_tree),
            Leaf::Kind(_) | Leaf::Ints(..) | Leaf::Floats(_) | Leaf::Walk => false,
        }
    }

    /// This leaf over the values of the exact type `exact` alone: each kind
    /// folded to the top or the bottom, which a meet and a union absorb, so
    /// what is left is the test of the number, or a constant where that type
    /// decides it.
    fn specialized(&self, exact: Exact) -> Leaf {
        let constant = |admitted: bool| {
            if admitted {
                Leaf::All(Box::default())
            } else {
                Leaf::Any(Box::default())
            }
        };
        match self {
            Leaf::Kind(kind) => constant(admits(*kind, exact)),
            Leaf::Ints(least, most) if admits(Exact::Int, exact) => Leaf::Ints(*least, *most),
            Leaf::Floats(tests) if exact == Exact::Float => Leaf::Floats(tests.clone()),
            Leaf::Ints(..) | Leaf::Floats(_) => constant(false),
            Leaf::Not(inner) => match inner.specialized(exact) {
                inner if inner.constant().is_some() => constant(inner.constant() == Some(false)),
                inner => Leaf::Not(Box::new(inner)),
            },
            Leaf::All(parts) => Leaf::joined(parts, exact, false),
            Leaf::Any(parts) => Leaf::joined(parts, exact, true),
            Leaf::Tree(atom) if atom.containers & exact as u8 != 0 => Leaf::Walk,
            Leaf::Tree(atom) => atom.scalars.specialized(exact),
            Leaf::Walk => Leaf::Walk,
        }
    }

    /// The parts of a meet (`deciding` false) or a union (`deciding` true)
    /// specialized and joined: a part that answers `deciding` decides the
    /// whole, one that answers the other is dropped, and one part left is the
    /// whole.
    fn joined(parts: &[Leaf], exact: Exact, deciding: bool) -> Leaf {
        let mut left = Vec::new();
        for part in parts {
            let part = part.specialized(exact);
            match part.constant() {
                Some(answer) if answer == deciding => return part,
                Some(_) => {}
                None => left.push(part),
            }
        }
        match (left.len(), deciding) {
            (1, _) => left.pop().unwrap_or_else(|| Leaf::Any(Box::default())),
            (_, false) => Leaf::All(left.into()),
            (_, true) => Leaf::Any(left.into()),
        }
    }

    /// What this answers whatever the number: a meet of no part admits, and a
    /// union of none refuses.
    fn constant(&self) -> Option<bool> {
        match self {
            Leaf::All(parts) if parts.is_empty() => Some(true),
            Leaf::Any(parts) if parts.is_empty() => Some(false),
            _ => None,
        }
    }

    /// What this answers for a value of the exact type `exact` holding
    /// `number`.
    fn holds(&self, exact: Exact, number: Number) -> bool {
        match self {
            Leaf::Kind(kind) => admits(*kind, exact),
            Leaf::Ints(least, most) => {
                matches!(number, Number::Int(number) if *least <= number && number <= *most)
            }
            Leaf::Floats(tests) => {
                matches!(number, Number::Float(number) if tests.iter().all(|test| test.holds(number)))
            }
            Leaf::Not(inner) => !inner.holds(exact, number),
            Leaf::All(parts) => parts.iter().all(|part| part.holds(exact, number)),
            Leaf::Any(parts) => parts.iter().any(|part| part.holds(exact, number)),
            // Specialized away for every scalar type, and a container's number
            // is never read ([`Leaf::Walk`]).
            Leaf::Tree(_) | Leaf::Walk => false,
        }
    }
}

/// Whether the scalar kind `kind` admits every value of the exact type
/// `exact`: its own, and for `int` a `bool` as well, which subclasses it.
fn admits(kind: Exact, exact: Exact) -> bool {
    kind == exact || (kind == Exact::Int && exact == Exact::Bool)
}

/// A [`Leaf`] with what it answers for each exact type it decides without the
/// number, read once a validator, and the levels the walk opens to its deepest
/// type test.
pub(crate) struct LeafPlan {
    /// The leaf over the values of each exact type ([`Leaf::specialized`]).
    specialized: [Leaf; 8],
    /// What the leaf answers for every value of each exact type, where it
    /// answers without the number.
    decided: [Option<bool>; 8],
    /// The levels below the level the leaf is asked at, its own included,
    /// that the walk opens to reach its deepest type test.
    pub(crate) levels: usize,
    /// The leaf over `float`s as the interval it admits, where it is order
    /// bounds alone ([`Interval::of`]).
    floats: Option<Interval>,
    /// Whether the leaf holds a reference to a tree of scalars, whose walk
    /// pushes a pair on the trail and is kept where a memo is armed: a reader
    /// reads it where neither can bear on the answer.
    pub(crate) trees: bool,
}

impl LeafPlan {
    fn new(leaf: &Leaf) -> LeafPlan {
        let specialized = Exact::ALL.map(|exact| leaf.specialized(exact));
        let floats = specialized.get(Exact::Float.place()).and_then(Interval::of);
        LeafPlan {
            floats,
            decided: specialized.each_ref().map(Leaf::constant),
            specialized,
            levels: leaf.levels(),
            trees: leaf.reaches_a_tree(),
        }
    }

    /// The plan of `schema`, where it is a leaf.
    fn of(
        schema: &Schema,
        pool: &[Py<PyAny>],
        py: Python<'_>,
        trees: &[Option<TreePlan>],
    ) -> Option<LeafPlan> {
        Leaf::of(schema, pool, py, trees)
            .as_ref()
            .map(LeafPlan::new)
    }

    /// What the leaf answers for every value of the exact type `exact`, or
    /// `None` where the answer is the number's ([`LeafPlan::holds`]).
    #[inline]
    pub(crate) fn decided(&self, exact: Exact) -> Option<bool> {
        self.decided.get(exact.place()).copied().flatten()
    }

    /// The interval a `float` is admitted in, where the leaf over `float`s is
    /// order bounds alone.
    #[inline]
    pub(crate) fn floats(&self) -> Option<Interval> {
        self.floats
    }

    /// What the leaf answers for a value of the exact type `exact` holding
    /// `number`.
    pub(crate) fn holds(&self, exact: Exact, number: Number) -> bool {
        self.specialized
            .get(exact.place())
            .is_some_and(|leaf| leaf.holds(exact, number))
    }
}

/// A definition whose body is a union of which every member is a leaf
/// ([`LeafPlan`]) refusing every list and dict, a list of the definition, or a
/// mapping of `str` keys to it, and one member at least a container: a tree of
/// scalars. The values `json.loads` returns are one, `recursive(lambda j: None
/// | bool | int | str | list[j] | dict[str, j] | Annotated[float,
/// Ge(-inf)])`, and `recursive(lambda t: int | list[t])` is another.
///
/// Every member reads a value's type before anything in it, and only the two
/// containers read further, each of them the definition again for every value
/// it holds. So a value of an exact builtin type is the leaves' answer, and a
/// list or a dict the same question asked of what it holds (`tree_member` in
/// `check/walk/tree.rs`).
/// A leaf over `float`s that is order bounds alone, as the interval they
/// admit: each end a bound or an infinity, closed or open.
///
/// The bounds are tested one by one at every number otherwise, through the
/// leaf's tree; as an interval they are two comparisons. A `nan` is in no
/// interval, as it holds no order bound, and an infinity lies where its bounds
/// put it, so the interval admits exactly the numbers the bounds do. A leaf
/// with no bound, or with a step, a complement or a union in it, is no
/// interval and is read by its tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Interval {
    least: f64,
    least_open: bool,
    most: f64,
    most_open: bool,
}

impl Interval {
    /// The interval of every number but `nan`.
    const NUMBERS: Interval = Interval {
        least: f64::NEG_INFINITY,
        least_open: false,
        most: f64::INFINITY,
        most_open: false,
    };

    /// The interval `leaf` admits, or `None` where it is not order bounds alone.
    fn of(leaf: &Leaf) -> Option<Interval> {
        match leaf {
            Leaf::Floats(tests) if !tests.is_empty() => tests
                .iter()
                .try_fold(Interval::NUMBERS, |range, test| range.bounded(*test)),
            Leaf::All(parts) if !parts.is_empty() => {
                parts.iter().try_fold(Interval::NUMBERS, |range, part| {
                    Some(range.meet(Interval::of(part)?))
                })
            }
            _ => None,
        }
    }

    /// This interval narrowed by one test, or `None` where it is no order bound.
    fn bounded(self, test: FloatTest) -> Option<Interval> {
        let FloatTest::Order(op, operand) = test else {
            return None;
        };
        if operand.is_nan() {
            return None;
        }
        let end = |open| Interval {
            least: operand,
            least_open: open,
            ..Interval::NUMBERS
        };
        let top = |open| Interval {
            most: operand,
            most_open: open,
            ..Interval::NUMBERS
        };
        Some(self.meet(match op {
            CompareOp::Ge => end(false),
            CompareOp::Gt => end(true),
            CompareOp::Le => top(false),
            CompareOp::Lt => top(true),
            _ => return None,
        }))
    }

    /// The numbers both intervals admit. No end is `nan`, and two ends at one
    /// number, `-0.0` and `0.0` among them, are one end, open where either is.
    fn meet(self, other: Interval) -> Interval {
        use std::cmp::Ordering::{Equal, Greater, Less};
        let (least, least_open) = match other.least.partial_cmp(&self.least) {
            Some(Greater) => (other.least, other.least_open),
            Some(Equal) => (self.least, self.least_open || other.least_open),
            _ => (self.least, self.least_open),
        };
        let (most, most_open) = match other.most.partial_cmp(&self.most) {
            Some(Less) => (other.most, other.most_open),
            Some(Equal) => (self.most, self.most_open || other.most_open),
            _ => (self.most, self.most_open),
        };
        Interval {
            least,
            least_open,
            most,
            most_open,
        }
    }

    /// Whether `number` lies in the interval.
    #[inline]
    pub(crate) fn holds(self, number: f64) -> bool {
        (if self.least_open {
            number > self.least
        } else {
            number >= self.least
        }) && (if self.most_open {
            number < self.most
        } else {
            number <= self.most
        })
    }
}

pub(crate) struct TreePlan {
    /// The members that are leaves, as their union.
    pub(crate) leaf: LeafPlan,
    /// That union as a leaf, which a leaf referring to the tree reads.
    scalars: Leaf,
    /// The containers among the members, their [`Exact`] bits.
    containers: u8,
}

impl TreePlan {
    /// The plan for the definition at `def` whose body is `body`, or `None`
    /// where it is no tree of scalars.
    fn of(def: usize, body: &Schema, pool: &[Py<PyAny>], py: Python<'_>) -> Option<TreePlan> {
        let Schema::Union(members) = body else {
            return None;
        };
        let itself =
            |schema: Option<&Schema>| matches!(schema, Some(Schema::Ref(id)) if id.get() == def);
        let mut containers = 0;
        let mut leaves = Vec::new();
        for member in members.iter() {
            match member {
                Schema::Seq {
                    container: SeqKind::List,
                    shape,
                } if shape.prefix.is_empty() && itself(shape.tail.as_deref()) => {
                    containers |= Exact::List as u8;
                }
                Schema::KeyedMap { fields, defaults }
                    if fields.is_empty()
                        && matches!(&defaults[..], [clause]
                            if matches!(clause.key, Schema::Str) && itself(Some(&clause.value))) =>
                {
                    containers |= Exact::Dict as u8;
                }
                // A tree's member refers to no other tree: one that did would
                // read a container on, which the tree reads itself.
                _ => leaves.push(Leaf::of(member, pool, py, &[])?),
            }
        }
        let scalars = Leaf::Any(leaves.into());
        let leaf = LeafPlan::new(&scalars);
        let containers_alone = [Exact::List, Exact::Dict]
            .into_iter()
            .all(|container| leaf.decided(container) == Some(false));
        (containers != 0 && containers_alone).then_some(TreePlan {
            leaf,
            scalars,
            containers,
        })
    }

    /// Whether a member reads a value of the container type `container` on.
    #[inline]
    pub(crate) fn reads(&self, container: Exact) -> bool {
        self.containers & container as u8 != 0
    }
}

/// A compiled pattern: the engine `regex` builds a `Regex` on, built as it
/// builds one ([`compile_pattern`]), so it matches the same language and can
/// say what memory it keeps.
pub(crate) type Regex = meta::Regex;

/// Each `Regex(...)` constraint's pattern-buffer address mapped to its compiled,
/// anchored regex, built once per validator so a string-pattern refinement
/// matches natively without recompiling on every call.
///
/// Keyed by address rather than by the pattern text, as the record and union
/// plans are keyed: the buffer is stable for the life of the (immutable) schema
/// and the index is rebuilt per validator from its own schema, so an entry
/// always refers to the same live constraint. The text is a key that must be
/// hashed in full for every value a pattern refinement checks; the address is
/// one integer. Two occurrences of the same text share one compiled program,
/// and so do two validators holding it ([`program`]).
pub(crate) type RegexIndex = FxHashMap<usize, Arc<Regex>>;

/// Anchor a user pattern so the whole string must match (`re.fullmatch`
/// semantics): `\A` and `\z` are absolute string boundaries, and the
/// non-capturing group keeps the user's alternation from escaping them.
///
/// In extended mode a `#` runs to the end of the line, so a pattern that ends
/// inside a comment swallows the wrapper's own `)\z` and the anchored form
/// fails to parse while the pattern alone is fine. A newline ends the comment,
/// and in extended mode a newline is whitespace that denotes nothing -- so the
/// second wrapper is the same language as the first wherever the first parses,
/// and is reached only where it does not. Outside extended mode nothing is
/// line-terminated, so a pattern the first wrapper rejects the second rejects
/// too, and the caller gets the first wrapper's message either way.
///
/// The pattern is parsed on its own first. One that does not parse alone can
/// parse once wrapped -- `a)|(b` closes the wrapper's group and opens one the
/// wrapper closes -- and its alternation then escapes both anchors, so the
/// wrappers would compile a language the caller never wrote.
///
/// Built with the configuration `regex::RegexBuilder` hands the engine -- the
/// leftmost-first match, the UTF-8 syntax, and the lazy DFA's cache at the 2 MiB
/// the engine defaults to -- and its errors read as `regex`'s, so the language
/// and the refusal are the ones `regex` gives.
pub(crate) fn compile_pattern(pattern: &str) -> Result<Regex, regex::Error> {
    regex_syntax::Parser::new()
        .parse(pattern)
        .map_err(|err| regex::Error::Syntax(err.to_string()))?;
    let build = |anchored: &str| {
        meta::Builder::new()
            .configure(
                meta::Config::new()
                    .nfa_size_limit(Some(PATTERN_SIZE_LIMIT))
                    .match_kind(MatchKind::LeftmostFirst)
                    .utf8_empty(true),
            )
            .syntax(syntax::Config::new().utf8(true))
            .build(anchored)
            .map_err(|err| match (err.size_limit(), err.syntax_error()) {
                (Some(limit), _) => regex::Error::CompiledTooBig(limit),
                (None, Some(syntax)) => regex::Error::Syntax(syntax.to_string()),
                (None, None) => regex::Error::Syntax(err.to_string()),
            })
    };
    build(&format!(r"\A(?:{pattern})\z"))
        .or_else(|anchored| build(&format!("\\A(?:{pattern}\n)\\z")).map_err(|_| anchored))
}

/// The programs a live validator holds, by their text, held weakly
/// ([`program`]).
static PROGRAMS: LazyLock<Mutex<Programs>> = LazyLock::new(Mutex::default);

/// The table [`PROGRAMS`] guards: a program for each text some validator holds,
/// and the length at which the entries of programs gone are dropped.
#[derive(Default)]
struct Programs {
    by_text: FxHashMap<Box<str>, Weak<Regex>>,
    prune_at: usize,
}

impl Programs {
    /// Hold `compiled` as the program of `pattern`, first dropping the entries
    /// of programs gone where the table has reached the length it last doubled
    /// to: the entries left then set the next length, twice theirs.
    fn hold(&mut self, pattern: &str, compiled: &Arc<Regex>) {
        if self.by_text.len() >= self.prune_at {
            self.by_text.retain(|_, held| held.strong_count() > 0);
            self.prune_at = 2 * self.by_text.len().max(1);
        }
        self.by_text
            .insert(pattern.into(), Arc::downgrade(compiled));
    }
}

/// The program of `pattern`, shared with every validator that holds it while
/// one does, or compiled ([`compile_pattern`]) where none does.
///
/// A validator built from another -- a meet, a union, a complement, the same
/// annotation built again -- holds the other's patterns, and compiled for it
/// each would keep a program of its own: twenty over `^\w{1,200}$` kept 240
/// MB, and a few hundred passed the memory there was, where a failed
/// allocation aborts the interpreter. Held weakly, a program lives as long as
/// a validator holds it and no longer, so validators holding one text hold one
/// program, and a text no validator holds any more is compiled again. The
/// table keeps no program alive, and drops the entries of programs gone each
/// time it has doubled, so its upkeep is constant a program.
///
/// Compiled outside the lock: a counted Unicode class takes hundreds of
/// milliseconds, and two threads compiling one text at once each hold a
/// correct program. A poisoned lock shares nothing and compiles.
pub(crate) fn program(pattern: &str) -> Result<Arc<Regex>, regex::Error> {
    let held = PROGRAMS
        .lock()
        .ok()
        .and_then(|programs| programs.by_text.get(pattern).and_then(Weak::upgrade));
    if let Some(held) = held {
        return Ok(held);
    }
    let compiled = Arc::new(compile_pattern(pattern)?);
    if let Ok(mut programs) = PROGRAMS.lock() {
        programs.hold(pattern, &compiled);
    }
    Ok(compiled)
}

/// A validator's patterns, each text compiled once, within [`PATTERNS_HELD`]
/// together.
///
/// A validator keeps the program of every pattern it holds for as long as it
/// lives, so the bound on what one pattern may keep is a bound on one of them
/// only: a hundred patterns of 16 MiB each held a gigabyte, and the allocation
/// past the memory there is aborts the interpreter rather than raising. Read
/// together, the memory each compiled pattern keeps is charged to one
/// allowance, and the pattern that would pass it is refused where the schema is
/// built, with the patterns compiled before it. A text met twice is one
/// program, charged once. A program another validator holds is shared
/// ([`program`]) and charged all the same, so what a schema is refused for does
/// not depend on what else is alive.
///
/// What it returns is the table the walk's index takes, so a validator's
/// patterns compile once, when it is built.
pub(crate) fn compile_patterns(
    schema: &Schema,
    definitions: &[Schema],
) -> Result<RegexIndex, String> {
    compile_patterns_within(PATTERNS_HELD, schema, definitions)
}

/// [`compile_patterns`] within `allowance` bytes, the figure a test can make
/// small enough to reach with patterns that compile in a millisecond.
pub(crate) fn compile_patterns_within(
    allowance: usize,
    schema: &Schema,
    definitions: &[Schema],
) -> Result<RegexIndex, String> {
    let mut compiled = RegexIndex::default();
    let mut by_text: FxHashMap<&str, Arc<Regex>> = FxHashMap::default();
    let mut left = allowance;
    for tree in std::iter::once(schema).chain(definitions) {
        for pattern in tree.patterns() {
            let regex = if let Some(regex) = by_text.get(pattern) {
                regex.clone()
            } else {
                let regex = program(pattern)
                    .map_err(|err| format!("invalid regular expression {pattern:?}: {err}"))?;
                left = left.checked_sub(regex.memory_usage()).ok_or_else(|| {
                    format!(
                        "the patterns of one schema compile to at most {} MiB together, \
                         and {pattern:?} takes them past it: a validator keeps every \
                         pattern's program while it lives",
                        allowance >> 20
                    )
                })?;
                by_text.insert(pattern, Arc::clone(&regex));
                regex
            };
            compiled.insert(pattern.as_ptr() as usize, regex);
        }
    }
    Ok(compiled)
}

thread_local! {
    /// The patterns a walk met outside any index, each compiled under its text,
    /// a refusal among them, for the query that walked it.
    static UNINDEXED: RefCell<FxHashMap<Box<str>, Option<Regex>>> = RefCell::default();
}

/// Whether `text` is in the language of `pattern`, a pattern no index holds.
///
/// Only the relation oracle walks a schema no validator built an index for,
/// and it walks every constant of a table through one supertype: compiled per
/// value, a table against the alternation of its own words compiles the
/// pattern once a constant, quadratic in the table. So a pattern compiles once
/// a query, under its text -- a schema the query builds holds a copy of the
/// text at an address of its own, which no address key would find again -- and
/// [`forget_unindexed`] drops what the query compiled when its oracle goes.
///
/// Out of line and cold: a validator's own walk never reaches it, and inlined
/// into the pattern check every refined value reads, the table and the
/// thread-local around it lay out the walk's hot path anew.
#[cold]
#[inline(never)]
pub(crate) fn matches_unindexed(pattern: &str, text: &str) -> bool {
    UNINDEXED.with_borrow_mut(|compiled| {
        if let Some(regex) = compiled.get(pattern) {
            return regex.as_ref().is_some_and(|regex| regex.is_match(text));
        }
        let regex = compile_pattern(pattern).ok();
        let matched = regex.as_ref().is_some_and(|regex| regex.is_match(text));
        compiled.insert(pattern.into(), regex);
        matched
    })
}

/// Drop the patterns [`matches_unindexed`] compiled on this thread: the query
/// that walked them is over, and nothing outside one misses an index.
pub(crate) fn forget_unindexed() {
    UNINDEXED.take();
}

/// How many patterns [`matches_unindexed`] holds on this thread.
#[cfg(test)]
pub(crate) fn unindexed_held() -> usize {
    UNINDEXED.with_borrow(FxHashMap::len)
}

/// The program one pattern may compile to: 32 MiB, three times the `regex`
/// crate's default program size.
///
/// The default refuses a counted Unicode class written as an ordinary bound.
/// The class holds some 700 ranges and a count copies it once a repetition, so
/// `\w{1,255}` keeps 14 MiB and `\w{1,255}@\w{1,255}` 28, each built in 150 to
/// 400 ms; `re` takes both. Past this, `\w{1,1000}` is refused as before. The
/// ASCII class of the same counts, `(?-u)\w{1,255}`, keeps 40 KiB.
pub(crate) const PATTERN_SIZE_LIMIT: usize = 32 << 20;

/// The memory a validator's patterns may keep together: 256 MiB, eight programs
/// at [`PATTERN_SIZE_LIMIT`] ([`compile_patterns`]).
///
/// A validator keeps every pattern's program while it lives, and the `regex`
/// crate bounds one program, leaving a collection to its caller; pydantic-core
/// keeps each pattern under the crate's bound and none together. Unbounded,
/// a hundred and twenty `\w{1,255}`, 14.3 MB a program, held 1.7 GB, and the
/// allocation past the memory aborted the interpreter. Held to the one
/// program's bound, a record of sixteen string fields each `^[\w.-]{1,n}$`,
/// 2.2 to 3 MB a program, was refused at its twelfth field, a schema every
/// release built. At this figure that record keeps 43 MB and builds, and the
/// `\w{1,255}` union is refused at its nineteenth pattern, built and walked
/// within an address space of 1 GiB.
pub(crate) const PATTERNS_HELD: usize = 256 << 20;

/// The per-validator precompute: record-field lookups, literal-union decision
/// tables, and compiled string patterns, all built once from the finished schema
/// and reused across calls.
#[derive(Default)]
pub(crate) struct ValidatorIndex {
    /// Per-record declared-field lookups, keyed by the address of each record's
    /// `fields` buffer. The keyed-map fast path reads it instead of rebuilding
    /// the name map on every call; a node absent from it falls back to building
    /// the map.
    pub(crate) records: RecordIndex,
    /// Per-attribute-schema interned names. The attribute walk hands these to
    /// `getattr` instead of building a fresh `PyString` per attribute per value.
    pub(crate) attrs: AttrsIndex,
    /// Per-union value sets for unions whose members are all literals, keyed by
    /// the address of the union's members buffer. It answers only for an exact
    /// int or str value: the deciding walk takes its answer either way instead
    /// of scanning every branch, and the explaining walk takes only its yes, for
    /// the elements of a list of the union, and walks every other element. Any
    /// other case (a non-literal union, a value of another type, a JSON value)
    /// falls back to the linear scan.
    pub(crate) unions: UnionIndex,
    /// Per-union dispatch for unions of records pinned by one key to constants
    /// no two share ([`TaggedPlan`]), keyed as `unions` is. The deciding walk
    /// reads it and the explaining walk scans, so a report is the scan's.
    pub(crate) tagged: TaggedIndex,
    /// Compiled string patterns, keyed by the address of the pattern's own
    /// text in the schema; the refinement walk reads it for a `Regex(...)`
    /// constraint instead of recompiling.
    pub(crate) regexes: RegexIndex,
    /// Which definitions a walk may remember at a value, and which nodes arm
    /// the memo ([`MemoPlan`]). Empty for a schema with no definitions.
    pub(crate) memo: MemoPlan,
    /// What each order bound's operand is as a float, at its place in the pool,
    /// where one holds it exactly ([`float_operand`]). A slot past the table
    /// holds none, and a schema with no order bound has no table, so no answer
    /// depends on it: the rich comparison reads what it does not hold.
    float_operands: Vec<Option<f64>>,
    /// Whether each order bound's operand is the standard library's `Decimal`,
    /// at its place in the pool, which a float is compared with as the
    /// `Decimal` it equals. Beside [`Self::float_operands`] rather than one
    /// table of both, so the float reading reads the floats' table alone; the
    /// measurement is at `compared` in `walk/scalar.rs`.
    decimal_operands: Vec<bool>,
    /// The readings of leaves and of trees of scalars ([`Readings`]), none
    /// where the validator holds neither, so the reference's arm asks one word
    /// of a validator with no tree. One word, which the table's alignment
    /// leaves free: a field of two moves every frame that holds a validator.
    /// Not built on `PyPy`, whose walk asks its own stack probe at every level
    /// a reading would pass.
    readings: Option<Box<Readings>>,
}

/// The leaves the validator reads by a value's type, and its trees of scalars.
#[derive(Default)]
struct Readings {
    /// Each definition's [`TreePlan`] at its place in the definitions table,
    /// and empty where no definition is a tree.
    trees: Vec<Option<TreePlan>>,
    /// The [`LeafPlan`] of each list's element that is a union, a meet or a
    /// complement of leaves, keyed by the element's address.
    leaves: FxHashMap<usize, LeafPlan>,
}

impl ValidatorIndex {
    /// Whether the index holds a reading, of a tree of scalars or of a leaf:
    /// one word, which the reference's arm asks before it asks for a tree. A
    /// validator holding leaves and no tree pays a call that declines at each
    /// reference, where asking for the trees themselves would be a load more in
    /// that arm, which every walk of a nested list reads three instructions a
    /// level dearer on the profiled wheel.
    #[inline]
    pub(crate) fn has_readings(&self) -> bool {
        self.readings.is_some()
    }

    /// The plan of the definition `def`, where it is read as a tree of scalars.
    #[inline]
    pub(crate) fn tree(&self, def: DefIx) -> Option<&TreePlan> {
        self.readings.as_ref()?.trees.get(def.get())?.as_ref()
    }

    /// The plan of a list's `element`, where it is a union, a meet or a
    /// complement of leaves.
    #[inline]
    pub(crate) fn leaf(&self, element: &Schema) -> Option<&LeafPlan> {
        self.readings
            .as_ref()?
            .leaves
            .get(&(std::ptr::from_ref(element) as usize))
    }

    /// Drop every reading of leaves and trees, so the walk reads each node
    /// generally: what the tests hold the readings to.
    #[cfg(all(test, feature = "interpreter-tests"))]
    pub(crate) fn forget_readings(&mut self) {
        self.readings = None;
    }

    /// The float the order bound's operand at `slot` is exactly, where the index
    /// holds one.
    #[inline]
    pub(crate) fn float_operand(&self, slot: usize) -> Option<f64> {
        self.float_operands.get(slot).copied().flatten()
    }

    /// Whether the order bound's operand at `slot` is the standard library's
    /// `Decimal`, which a float is compared with as the `Decimal` it equals;
    /// `None` past the table, as every slot of the empty index a transient
    /// probe walks with is, where the caller asks the operand.
    #[inline]
    pub(crate) fn decimal_operand(&self, slot: usize) -> Option<bool> {
        self.decimal_operands.get(slot).copied()
    }

    /// Hold what the operand of the order bound `constraint` is, read out of the
    /// pool: the float it is exactly, or the standard library's `Decimal`. Any
    /// other constraint holds none.
    fn hold_operand(&mut self, constraint: &Constraint, pool: &[Py<PyAny>], py: Python<'_>) {
        let (Constraint::Ge(index)
        | Constraint::Gt(index)
        | Constraint::Le(index)
        | Constraint::Lt(index)) = constraint
        else {
            return;
        };
        let slot = index.get();
        let Some(operand) = pool.get(slot).map(|operand| operand.bind(py)) else {
            return;
        };
        let float = float_operand(operand);
        let decimal = float.is_none()
            && is_stdlib_class(&operand.get_type(), "decimal", "Decimal").unwrap_or(false);
        if self.float_operands.len() <= slot {
            self.float_operands.resize(slot + 1, None);
            self.decimal_operands.resize(slot + 1, false);
        }
        if let Some(held) = self.float_operands.get_mut(slot) {
            *held = float;
        }
        if let Some(held) = self.decimal_operands.get_mut(slot) {
            *held = decimal;
        }
    }

    /// Every Python object this index owns, for the validator's `__traverse__`.
    ///
    /// The interned attribute names and the interned record keys: the other
    /// plans hold Rust values -- integer and string sets, compiled patterns --
    /// copied out of the pool rather than referenced. A `str` joins no cycle,
    /// so nothing here can be the edge a collector needs; a traversal that
    /// skipped an owned reference would still be wrong on its own terms, which
    /// is why the keys join this the moment the record plan starts holding
    /// them.
    pub(crate) fn interned_names(&self) -> impl Iterator<Item = &Py<PyString>> {
        self.attrs
            .values()
            .flat_map(|plan| plan.names.iter())
            .chain(self.records.values().flat_map(|plan| plan.keys.iter()))
            .chain(self.tagged.values().map(|plan| &plan.key))
    }
}

/// Build the index for a finished schema plus its recursion definitions. `pool`
/// is the validator's constants pool, needed to read each literal's value while
/// building union plans. A record with no declared fields, and a union with a
/// non-literal member, are skipped; the walk falls back to its general path for
/// anything not indexed, which answers the same membership. A report is the
/// exception: a record with no plan is read by its entries rather than its
/// keys, and which branch of a union of records is explained follows that
/// reading, so the traversal descends into every node a record can sit under.
pub(crate) fn build_index(
    py: Python<'_>,
    schema: &Schema,
    defs: &[Schema],
    pool: &[Py<PyAny>],
) -> ValidatorIndex {
    build_index_from(py, schema, defs, pool, &RegexIndex::default())
}

/// [`build_index`] taking the patterns a validator compiled when it was built
/// ([`compile_patterns`]), so none compiles twice; a pattern the table does
/// not hold is compiled here.
pub(crate) fn build_index_from(
    py: Python<'_>,
    schema: &Schema,
    defs: &[Schema],
    pool: &[Py<PyAny>],
    compiled: &RegexIndex,
) -> ValidatorIndex {
    let mut index = ValidatorIndex::default();
    index.regexes.clone_from(compiled);
    collect(py, schema, pool, &mut index);
    for def in defs {
        collect(py, def, pool, &mut index);
    }
    index.memo = memo::plan(py, schema, defs, pool);
    if !cfg!(PyPy) {
        index.readings = readings(schema, defs, pool, py);
    }
    index
}

/// The readings of the leaves and the trees of scalars `schema` and `defs`
/// hold, or `None` where they hold neither. Out of line: the index is built
/// where a validator is first used, and inlined there the plans grow the frame
/// of every caller that builds one.
#[inline(never)]
fn readings(
    schema: &Schema,
    defs: &[Schema],
    pool: &[Py<PyAny>],
    py: Python<'_>,
) -> Option<Box<Readings>> {
    let mut readings = Readings::default();
    let trees: Vec<_> = defs
        .iter()
        .enumerate()
        .map(|(def, body)| TreePlan::of(def, body, pool, py))
        .collect();
    for node in std::iter::once(schema).chain(defs) {
        hold_leaves(node, pool, py, &trees, &mut readings.leaves);
    }
    if trees.iter().any(Option::is_some) {
        readings.trees = trees;
    }
    (!readings.trees.is_empty() || !readings.leaves.is_empty()).then(|| Box::new(readings))
}

/// Hold the plan of each list's element below `schema` that is a union, a
/// meet or a complement of leaves, by the element's address. A union of
/// scalar kinds has a reader of its own, and holds none: a validator holding
/// no plan keeps the list's tail call to that reader.
fn hold_leaves(
    schema: &Schema,
    pool: &[Py<PyAny>],
    py: Python<'_>,
    trees: &[Option<TreePlan>],
    leaves: &mut FxHashMap<usize, LeafPlan>,
) {
    // A list's element, or the value a mapping of `str` keys holds.
    let held = match schema {
        Schema::Seq {
            container: SeqKind::List,
            shape,
        } if shape.prefix.is_empty() => shape.tail.as_deref(),
        Schema::KeyedMap { fields, defaults } if fields.is_empty() => match &defaults[..] {
            [
                MapClause {
                    key: Schema::Str,
                    value,
                },
            ] => Some(value),
            _ => None,
        },
        _ => None,
    };
    if let Some(element @ (Schema::Union(_) | Schema::Intersection(_) | Schema::Complement(_))) =
        held
        && !matches!(element, Schema::Union(members) if members.iter().all(|m| matches!(m, Schema::NoneType | Schema::Bool | Schema::Int | Schema::Float | Schema::Str | Schema::Bytes)))
        && let Some(plan) = LeafPlan::of(element, pool, py, trees)
    {
        leaves.insert(std::ptr::from_ref(element) as usize, plan);
    }
    for_each_child(schema, &mut |child| {
        hold_leaves(child, pool, py, trees, leaves);
    });
}

fn collect(py: Python<'_>, schema: &Schema, pool: &[Py<PyAny>], index: &mut ValidatorIndex) {
    match schema {
        Schema::KeyedMap { fields, defaults } => {
            if !fields.is_empty() {
                index
                    .records
                    .entry(fields.as_ptr() as usize)
                    .or_insert_with(|| RecordPlan {
                        by_name: fields
                            .iter()
                            .enumerate()
                            .map(|(i, f)| (Arc::clone(&f.name), i))
                            .collect(),
                        required: fields.iter().filter(|f| f.required).count(),
                        keys: fields
                            .iter()
                            .map(|f| PyString::intern(py, &f.name).unbind())
                            .collect(),
                        early: early_fields(fields),
                    });
            }
            for f in fields.iter() {
                collect(py, &f.schema, pool, index);
            }
            for clause in defaults.iter() {
                collect(py, &clause.key, pool, index);
                collect(py, &clause.value, pool, index);
            }
        }
        Schema::Union(members) => {
            if let Some(plan) = literal_union_plan(py, members, pool) {
                index
                    .unions
                    .entry(members.as_ptr() as usize)
                    .or_insert(plan);
            } else if let Some(plan) = tagged_union_plan(py, members, pool) {
                index
                    .tagged
                    .entry(members.as_ptr() as usize)
                    .or_insert(plan);
            }
            for member in members.iter() {
                collect(py, member, pool, index);
            }
        }
        Schema::Intersection(members) => {
            for member in members.iter() {
                collect(py, member, pool, index);
            }
        }
        Schema::Coll { element: inner, .. } | Schema::Complement(inner) => {
            collect(py, inner, pool, index);
        }
        Schema::Refine { base, constraints } => {
            collect(py, base, pool, index);
            for constraint in constraints.iter() {
                index.hold_operand(constraint, pool, py);
                if let Constraint::Regex(pattern) = constraint
                    && let Entry::Vacant(slot) = index.regexes.entry(pattern.as_ptr() as usize)
                    && let Ok(compiled) = program(pattern)
                {
                    slot.insert(compiled);
                }
            }
        }
        Schema::AttrRecord { fields } => {
            if !fields.is_empty() {
                index
                    .attrs
                    .entry(fields.as_ptr() as usize)
                    .or_insert_with(|| AttrsPlan {
                        names: fields
                            .iter()
                            .map(|f| PyString::intern(py, &f.name).unbind())
                            .collect(),
                    });
            }
            for f in fields.iter() {
                collect(py, &f.schema, pool, index);
            }
        }
        Schema::Seq { shape, .. } => {
            for element in shape.prefix.iter().chain(shape.tail.as_deref()) {
                collect(py, element, pool, index);
            }
        }
        _ => {}
    }
}

/// The float `operand` is exactly: an exact `float`, or an exact `int` a float
/// holds without rounding, and `None` for anything else.
///
/// An order bound over an exact `float` compares two numbers, whatever the other
/// is, and asks nothing else of either, so where the operand is one of these the
/// comparison is the two floats' own, `nan` refused by all four bounds as Python
/// refuses it. An `int` up to `2^53` is read as itself, and past it two integers
/// round to one float, which would read a value between them as equal to the
/// bound. A subclass may answer a comparison with code of its own, and a `bool`
/// is left to the comparison for no gain: neither is exact.
pub(crate) fn float_operand(operand: &Bound<'_, PyAny>) -> Option<f64> {
    if let Ok(float) = operand.cast_exact::<PyFloat>() {
        return Some(float.value());
    }
    if !operand.is_exact_instance_of::<PyInt>() {
        return None;
    }
    let whole = operand.extract::<i64>().ok()?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "an integer of at most 2^53 is a float exactly"
    )]
    (whole.unsigned_abs() <= 1 << 53).then_some(whole as f64)
}

/// What the order bound `op` makes of two floats, `number` against `operand`:
/// the comparison an exact `float` and an operand [`float_operand`] holds make
/// in Python, `nan` refused by every bound.
#[inline]
pub(crate) fn ordered(number: f64, op: CompareOp, operand: f64) -> bool {
    match op {
        CompareOp::Ge => number >= operand,
        CompareOp::Gt => number > operand,
        CompareOp::Le => number <= operand,
        _ => number < operand,
    }
}

/// Build a [`UnionPlan`] when every union member is a literal, bucketing the
/// `int` and `str` literal values; returns `None` (so the union stays a linear
/// scan) when any member is not a literal. A big-integer or other-typed literal
/// is not bucketed — values of those types are scanned linearly.
fn literal_union_plan(py: Python<'_>, members: &[Schema], pool: &[Py<PyAny>]) -> Option<UnionPlan> {
    let mut ints = FxHashSet::default();
    let mut strs = FxHashSet::default();
    let mut pooled = FxHashSet::default();
    for member in members {
        let Schema::Literal(idx) = member else {
            return None;
        };
        // Bounds-check rather than index directly: a corrupt pool index abandons
        // the precompute (no fast path) instead of panicking across the FFI
        // boundary, matching the defensive `.get` posture used elsewhere.
        let constant = pool.get(idx.get()).map(|obj| obj.bind(py))?;
        if constant.is_exact_instance_of::<PyInt>() {
            if let Ok(i) = constant.extract::<i64>() {
                // Sized on its first member for the whole table, so a wide
                // table is one allocation rather than a rehash per doubling.
                if ints.is_empty() {
                    ints.reserve(members.len());
                }
                ints.insert(i);
                pooled.insert(constant.as_ptr() as usize);
            }
        } else if constant.is_exact_instance_of::<PyString>()
            && let Some(s) = constant
                .cast::<PyString>()
                .ok()
                .and_then(|s| s.to_str().ok())
        {
            if strs.is_empty() {
                strs.reserve(members.len());
            }
            strs.insert(s.into());
            pooled.insert(constant.as_ptr() as usize);
        }
    }
    Some(UnionPlan { ints, strs, pooled })
}

/// The fewest records a tagged union holds before its plan is built.
///
/// Below it, reading the tag and then the record it names costs more than
/// reading the records in turn, each of which reads the tag first and refuses
/// on it: the record the plan names reads the tag again. Measured on the
/// release build, averaged over the tags a value can carry, two records read in
/// turn take 0.115 us and dispatched 0.128 us, three 0.128 and 0.125, four
/// 0.144 and 0.125, and a thousand 48 and 0.14.
pub(crate) const TAGGED_MIN_RECORDS: usize = 3;

/// Build a [`TaggedPlan`] where at least [`TAGGED_MIN_RECORDS`] members are
/// records requiring a key the first record requires, each pinning it to exact
/// `int` or `str` constants and no two sharing one: the first such key of the
/// first record. `None` where no key is, and the union stays a scan.
fn tagged_union_plan(py: Python<'_>, members: &[Schema], pool: &[Py<PyAny>]) -> Option<TaggedPlan> {
    let Some(Schema::KeyedMap { fields: first, .. }) = members.first() else {
        return None;
    };
    if members.len() < TAGGED_MIN_RECORDS {
        return None;
    }
    first
        .iter()
        .filter(|candidate| candidate.required)
        .find_map(|candidate| {
            let mut ints = FxHashMap::default();
            let mut strs = FxHashMap::default();
            for (position, member) in members.iter().enumerate() {
                let Schema::KeyedMap { fields, .. } = member else {
                    return None;
                };
                let field = fields
                    .iter()
                    .find(|field| field.required && field.name == candidate.name)?;
                for constant in pinned_constants(&field.schema)? {
                    let constant = pool.get(constant)?.bind(py);
                    let other = if constant.is_exact_instance_of::<PyInt>() {
                        ints.insert(constant.extract::<i64>().ok()?, position)
                    } else if constant.is_exact_instance_of::<PyString>() {
                        let text = constant.cast::<PyString>().ok()?.to_str().ok()?;
                        strs.insert(text.into(), position)
                    } else {
                        return None;
                    };
                    if other.is_some_and(|other| other != position) {
                        return None;
                    }
                }
            }
            Some(TaggedPlan {
                key: PyString::intern(py, &candidate.name).unbind(),
                ints,
                strs,
            })
        })
}

/// The pool positions of the constants a field is pinned to: one literal, or
/// a union of nothing but literals.
fn pinned_constants(schema: &Schema) -> Option<Vec<usize>> {
    match schema {
        Schema::Literal(index) => Some(vec![index.get()]),
        Schema::Union(members) => members
            .iter()
            .map(|member| match member {
                Schema::Literal(index) => Some(index.get()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
