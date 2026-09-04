//! Lowering a schema to a descriptor.
//!
//! A schema is a *term*: a tree the walk reads and the printer renders. A
//! descriptor is a *set*. Lowering is the map between them, and the two live
//! side by side rather than one replacing the other -- the walk still decides
//! membership, and the descriptor is what the algebra reasons in.
//!
//! **The map is partial, and the type says so.** A form the descriptor cannot
//! yet hold lowers to `None` rather than to something close: a descriptor that
//! stood for a schema it does not denote would be complemented into one that is
//! wrong the other way, and a refusal is the only sound answer to give.
//!
//! The core holds no Python objects, so the constants a schema names are indices
//! into a pool the bindings keep. [`Constants`] is the way in: the caller reads
//! the pool and answers what a comparison operand or a literal carries.

use super::classes::Class;
use super::{Descr, integers::IntSet};
use crate::decision::Kind;
use crate::ir::{ConstIx, Constraint, OperandIx, Schema, SeqKind, SeqShape};
use std::cell::Cell;

/// A pooled value, as far as a descriptor can read one.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// One of the two booleans.
    Boolean(bool),
    /// An integer, which is never a boolean: `bool` is its own kind here.
    Integer(i64),
    /// A float, `nan` included.
    Float(f64),
    /// A word, under the kind that reads it -- a `str` as its UTF-8 bytes.
    Word(Vec<u8>, Kind),
    /// `None`.
    NoneType,
    /// An instance of a pure class.
    Instance(Class),
}

/// What a lowering needs from the pool the core does not hold.
///
/// The core stays free of Python objects, so a schema names its constants by
/// index. Answering `None` is the honest reading of an operand this cannot see
/// -- an impure class, a callable, an object with a custom `__eq__` -- and it
/// refuses the lowering rather than guessing.
pub trait Constants {
    /// What a comparison or `MultipleOf` operand carries.
    fn operand(&self, index: OperandIx) -> Option<Operand>;

    /// What a `Literal` names.
    fn constant(&self, index: ConstIx) -> Option<Operand>;
}

/// A pool that knows nothing, for a caller that has none.
///
/// The core's own relations have no object table -- the constants live in the
/// bindings -- so a schema naming one refuses to lower here rather than being
/// guessed at. What still lowers is everything whose meaning is structural: the
/// kinds, the sequences, the sets, the three operations, and the length and
/// pattern constraints, which carry their operand inline.
pub struct NoConstants;

impl Constants for NoConstants {
    fn operand(&self, _index: OperandIx) -> Option<Operand> {
        None
    }

    fn constant(&self, _index: ConstIx) -> Option<Operand> {
        None
    }
}

/// The nodes this will lower before refusing.
///
/// A schema is a tree the caller writes, and lowering both recurses over it and
/// *builds* at every node -- a sequence node determinises an automaton, a set
/// node takes a powerset. Without a bound a deep enough schema exhausts the
/// stack and a wide one exhausts the clock, which is why the procedure beside
/// this one carries a work budget too.
///
/// A refusal past the bound is the same refusal as for a form with nowhere to
/// land, and it reaches the same place: the caller decides the old way. It is a
/// bound on the *lowering*, not on the algebra -- the components have their own,
/// and those are limits of what they can represent rather than of what they will
/// spend.
///
/// Sized by what can be *built* rather than by what can be parsed. Complementing
/// a descriptor complements the guards inside it, so a chain of complements
/// deepens the descriptor as well as the schema, and the operations recurse
/// through that nesting: past roughly a hundred the stack goes rather than the
/// clock. An annotation anyone writes is orders of magnitude inside this.
pub const BUDGET: u32 = 64;

/// The set a schema denotes, or `None` where the descriptor cannot yet hold it.
///
/// # Errors
///
/// Refuses rather than approximating. The forms it refuses are the ones with no
/// component to land in -- a dict, an attribute record beside a builtin kind, a
/// recursive reference, the gradual `Any` -- and the ones whose operand the pool
/// could not read.
pub fn lower(schema: &Schema, pool: &dyn Constants) -> Option<Descr> {
    descend(schema, pool, &Cell::new(BUDGET))
}

/// [`lower`] with the nodes left to spend, which every node spends one of.
fn descend(schema: &Schema, pool: &dyn Constants, budget: &Cell<u32>) -> Option<Descr> {
    budget.set(budget.get().checked_sub(1)?);
    match schema {
        Schema::Anything => Some(Descr::anything()),
        Schema::Nothing => Some(Descr::nothing()),
        Schema::NoneType => Some(Descr::of_kind(Kind::NoneType)),
        Schema::Bool => Some(Descr::of_kind(Kind::Bool)),
        // `bool` subclasses `int`, so every boolean is an integer. The two are
        // separate *kinds* here, which is what makes their components
        // independent -- so the schema that denotes both spells both.
        Schema::Int => Descr::of_kind(Kind::Int).union(&Descr::of_kind(Kind::Bool)),
        Schema::Float => Some(Descr::of_kind(Kind::Float)),
        Schema::Str => Some(Descr::of_kind(Kind::Str)),
        Schema::Bytes => Some(Descr::of_kind(Kind::Bytes)),
        Schema::Literal(index) => singleton(&pool.constant(*index)?),
        Schema::Seq { container, shape } => sequence(*container, shape, pool, budget),
        Schema::Set(elements) => Descr::set(&descend(elements, pool, budget)?, Kind::Set),
        Schema::FrozenSet(elements) => {
            Descr::set(&descend(elements, pool, budget)?, Kind::FrozenSet)
        }
        Schema::Union(members) => members.iter().try_fold(Descr::nothing(), |whole, member| {
            whole.union(&descend(member, pool, budget)?)
        }),
        Schema::Intersection(members) => {
            members.iter().try_fold(Descr::anything(), |whole, member| {
                whole.intersect(&descend(member, pool, budget)?)
            })
        }
        Schema::Complement(inner) => Some(descend(inner, pool, budget)?.complement()),
        Schema::Refine { base, constraints } => refine(base, constraints, pool, budget),
        // No component to land in, or none that would mean what the schema does.
        // `Dynamic` is the gradual type and not the top, a dict has no map
        // component yet, an attribute record beside a builtin kind wants a
        // descriptor that is a union of lines, and a reference is a cycle a
        // finite descriptor has no room for.
        Schema::Dynamic
        | Schema::KeyedMap { .. }
        | Schema::Instance(_)
        | Schema::Attrs { .. }
        | Schema::Ref(_)
        | Schema::SelfRef(_) => None,
    }
}

/// The set holding one pooled value and nothing else.
fn singleton(constant: &Operand) -> Option<Descr> {
    match constant {
        Operand::Boolean(value) => Some(Descr::boolean(*value)),
        Operand::Integer(value) => Some(Descr::integer(*value)),
        Operand::Float(value) => Some(Descr::float(*value)),
        Operand::Word(word, kind) => Descr::word(word, *kind),
        Operand::NoneType => Some(Descr::of_kind(Kind::NoneType)),
        // A class is a set of objects, not one value; a literal naming an
        // instance also pins *which* instance, which the descriptor cannot say.
        Operand::Instance(_) => None,
    }
}

/// The sequences a shape spells, under the kind that reads them.
fn sequence(
    container: SeqKind,
    shape: &SeqShape,
    pool: &dyn Constants,
    budget: &Cell<u32>,
) -> Option<Descr> {
    let kind = match container {
        SeqKind::List => Kind::List,
        SeqKind::Tuple => Kind::Tuple,
    };
    let prefix: Option<Vec<Descr>> = shape
        .prefix
        .iter()
        .map(|element| descend(element, pool, budget))
        .collect();
    let tail = match &shape.tail {
        Some(element) => Some(descend(element, pool, budget)?),
        None => None,
    };
    Descr::sequence(&prefix?, tail.as_ref(), kind)
}

/// The base narrowed by every constraint, which is a meet.
///
/// Each constraint is a *set* here rather than a test, so narrowing is
/// intersection and the order the constraints are written in carries no meaning
/// -- which is what makes `Ge(0) ∧ Lt(0)` decide rather than run.
fn refine(
    base: &Schema,
    constraints: &[Constraint],
    pool: &dyn Constants,
    budget: &Cell<u32>,
) -> Option<Descr> {
    let mut narrowed = descend(base, pool, budget)?;
    for constraint in constraints {
        narrowed = narrowed.intersect(&constrained(constraint, &narrowed, pool)?)?;
    }
    Some(narrowed)
}

/// The set one constraint denotes, read under the kinds the base admits.
///
/// A bound is a set of *numbers* and a length bound a set of *words*, so which
/// component a constraint lands in depends on what it is narrowing. A constraint
/// the descriptor cannot read -- a predicate, a bound on a float, a length bound
/// on a sequence -- refuses.
fn constrained(constraint: &Constraint, base: &Descr, pool: &dyn Constants) -> Option<Descr> {
    let integers = |set: IntSet| {
        let mut descr = Descr::nothing();
        descr.integers(set);
        Some(descr)
    };
    match constraint {
        Constraint::Ge(index) | Constraint::Gt(index) => {
            let Operand::Integer(bound) = pool.operand(*index)? else {
                return None;
            };
            let lo = if matches!(constraint, Constraint::Gt(_)) {
                bound.checked_add(1)?
            } else {
                bound
            };
            integers(IntSet::between(Some(lo), None))
        }
        Constraint::Le(index) | Constraint::Lt(index) => {
            let Operand::Integer(bound) = pool.operand(*index)? else {
                return None;
            };
            let hi = if matches!(constraint, Constraint::Lt(_)) {
                bound.checked_sub(1)?
            } else {
                bound
            };
            integers(IntSet::between(None, Some(hi)))
        }
        Constraint::MultipleOf(index) => {
            let Operand::Integer(step) = pool.operand(*index)? else {
                return None;
            };
            integers(IntSet::multiple_of(step)?)
        }
        Constraint::MinLen(least) => words(&format!(".{{{least},}}"), base),
        Constraint::MaxLen(most) => words(&format!(".{{0,{most}}}"), base),
        Constraint::Regex(pattern) => words(pattern, base),
        // A callback is a leaf the core cannot read, which is what makes it
        // opaque to the procedure beside this one too.
        Constraint::Predicate(_) => None,
    }
}

/// The words a pattern matches, under whichever word kind the base admits.
///
/// A length bound counts what the kind's alphabet counts -- code points for a
/// `str`, bytes for `bytes` -- which is why the pattern is read under the kind
/// rather than compiled once.
///
/// **Refuses unless the base is words and nothing else.** A length is not a
/// word's alone: a list, a tuple, a set and a dict all have one, and a bound on
/// them is a constraint the word component cannot express. Lowering the bound as
/// if it only spoke about words would give a *smaller* set than the schema
/// denotes -- and a smaller set has a larger complement, which is a subtype
/// proof that no value supports.
fn words(pattern: &str, base: &Descr) -> Option<Descr> {
    let mut alphabets = Descr::nothing();
    for kind in [Kind::Str, Kind::Bytes] {
        alphabets = alphabets.union(&Descr::of_kind(kind))?;
    }
    if !base.intersect(&alphabets.complement())?.is_empty() {
        return None;
    }
    let mut whole = Descr::nothing();
    for kind in [Kind::Str, Kind::Bytes] {
        if Descr::of_kind(kind).intersect(base)?.is_empty() {
            continue;
        }
        whole = whole.union(&Descr::pattern(pattern, kind)?)?;
    }
    (!whole.is_empty()).then_some(whole)
}

#[cfg(test)]
mod tests {
    use super::{BUDGET, Constants, Operand, lower};
    use crate::decision::{Kind, Verdict};
    use crate::descr::classes::Class;
    use crate::descr::{Descr, Value};
    use crate::ir::{ConstIx, Constraint, OperandIx, Schema, SeqKind, SeqShape};

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
    }

    fn empty_pool() -> Pool {
        Pool(Vec::new())
    }

    /// The kinds and the two ends map straight across.
    #[test]
    fn the_leaves_lower_to_the_sets_they_denote() {
        let pool = empty_pool();
        let leaf = |schema| lower(&schema, &pool).expect("a leaf lowers");

        assert_eq!(leaf(Schema::Anything), Descr::anything());
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
        let joined =
            lower(&Schema::Union(vec![Schema::Str, Schema::Float]), &pool).expect("a small union");
        let expected = Descr::of_kind(Kind::Str)
            .union(&Descr::of_kind(Kind::Float))
            .expect("a small union");
        assert_eq!(joined, expected);

        let barred =
            lower(&Schema::Complement(Box::new(Schema::Str)), &pool).expect("a small complement");
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
                    base: Box::new(Schema::Int),
                    constraints,
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

    /// A step is the constraint no union of intervals can spell, and it meets
    /// the bounds rather than being checked beside them.
    #[test]
    fn a_step_meets_the_bounds() {
        let pool = Pool(vec![Operand::Integer(2), Operand::Integer(1)]);
        let evens = lower(
            &Schema::Refine {
                base: Box::new(Schema::Int),
                constraints: vec![
                    Constraint::MultipleOf(OperandIx::new(0)),
                    Constraint::Ge(OperandIx::new(1)),
                ],
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
                    prefix: Vec::new(),
                    tail: Some(Box::new(Schema::Int)),
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
            &Schema::Set(Box::new(Schema::Seq {
                container: SeqKind::List,
                shape: SeqShape {
                    prefix: Vec::new(),
                    tail: Some(Box::new(Schema::Int)),
                },
            })),
            &pool,
        )
        .expect("a small set");

        // A list is unhashable, so the only member left is none at all.
        assert!(of_lists.admits(Value::sequence(NOTHING, Kind::Set)));
        assert_eq!(
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
                prefix: Vec::new(),
                tail: Some(Box::new(tail)),
            },
        };
        vec![
            Schema::Anything,
            Schema::Nothing,
            Schema::NoneType,
            Schema::Bool,
            Schema::Int,
            Schema::Float,
            Schema::Str,
            Schema::Bytes,
            seq(Schema::Int),
            seq(Schema::Str),
            Schema::Set(Box::new(Schema::Int)),
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
                corpus.push(Schema::Union(vec![left.clone(), right.clone()]));
                corpus.push(Schema::Intersection(vec![left.clone(), right.clone()]));
            }
            corpus.push(Schema::Complement(Box::new(left.clone())));
        }
        corpus
    }

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
            .flat_map(|leaf| [leaf.clone(), Schema::Complement(Box::new(leaf.clone()))])
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
        assert!(
            literal(1).admits(Value::boolean(true)) && !literal(1).admits(Value::boolean(false))
        );
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

    /// A literal naming a class instance refuses: a class is a set of objects,
    /// and the literal pins *which* instance, which the descriptor cannot say.
    #[test]
    fn a_literal_naming_an_instance_refuses() {
        let pool = Pool(vec![Operand::Instance(Class::root(1))]);
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
                    base: Box::new(base),
                    constraints,
                },
                &pool,
            )
        };

        let non_empty = refined(Schema::Str, vec![Constraint::MinLen(1)]).expect("a length bound");
        assert!(non_empty.admits(Value::word(b"a", Kind::Str)));
        assert!(!non_empty.admits(Value::word(b"", Kind::Str)));

        let short = refined(Schema::Str, vec![Constraint::MaxLen(1)]).expect("a length bound");
        assert!(
            short.admits(Value::word(b"", Kind::Str)) && short.admits(Value::word(b"a", Kind::Str))
        );
        assert!(!short.admits(Value::word(b"ab", Kind::Str)));

        let matching =
            refined(Schema::Str, vec![Constraint::Regex("a+".to_owned())]).expect("a pattern");
        assert!(matching.admits(Value::word(b"a", Kind::Str)));
        assert!(!matching.admits(Value::word(b"b", Kind::Str)));

        // Two of them meet rather than being checked one after the other, which
        // is what makes an impossible pair decide.
        let impossible = refined(
            Schema::Str,
            vec![Constraint::MinLen(2), Constraint::MaxLen(1)],
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
                    base: Box::new(Schema::Anything),
                    constraints: vec![Constraint::MinLen(0)],
                },
                &pool,
            )
            .is_none()
        );

        // Narrowed to the words first, the same bound lowers.
        assert!(
            lower(
                &Schema::Refine {
                    base: Box::new(Schema::Str),
                    constraints: vec![Constraint::MinLen(0)],
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
                    base: Box::new(Schema::Int),
                    constraints: vec![Constraint::MinLen(1)],
                },
                &pool,
            )
            .is_none()
        );
    }

    /// A schema past the budget refuses rather than spending without end.
    ///
    /// Lowering builds at every node, so a deep schema is both a deep recursion
    /// and a lot of work. The bound is what keeps a caller that asks about an
    /// adversarial schema from paying for it -- and refusing is safe, because
    /// the caller decides the old way.
    #[test]
    fn a_schema_past_the_budget_refuses() {
        let pool = empty_pool();
        let mut deep = Schema::Int;
        for _ in 0..BUDGET {
            deep = Schema::Complement(Box::new(deep));
        }
        assert!(lower(&deep, &pool).is_none());

        // Just inside it still lowers, so the bound is a bound and not a wall.
        let mut shallow = Schema::Int;
        for _ in 0..(BUDGET / 2) {
            shallow = Schema::Complement(Box::new(shallow));
        }
        assert!(lower(&shallow, &pool).is_some());
    }

    /// The map is partial, and it refuses rather than approximating.
    ///
    /// Each of these has no component to land in, or none that would mean what
    /// the schema does: the gradual type is not the top, a dict has no map
    /// component, an attribute record beside a builtin kind wants a descriptor
    /// that is a union of lines, and a reference is a cycle.
    #[test]
    fn the_forms_with_nowhere_to_land_refuse() {
        let pool = empty_pool();
        for schema in [
            Schema::Dynamic,
            Schema::KeyedMap {
                fields: Vec::new(),
                defaults: Vec::new(),
            },
            Schema::Ref(crate::ir::DefIx::new(0)),
        ] {
            assert!(lower(&schema, &pool).is_none(), "{schema:?}");
        }
        // A refusal inside a form refuses the whole form rather than dropping
        // the part it could not read.
        assert!(lower(&Schema::Union(vec![Schema::Str, Schema::Dynamic]), &pool).is_none());
    }

    /// An operand the pool cannot read refuses the constraint that names it.
    #[test]
    fn an_operand_the_pool_cannot_read_refuses() {
        let pool = Pool(vec![Operand::Float(0.5)]);
        assert!(
            lower(
                &Schema::Refine {
                    base: Box::new(Schema::Int),
                    constraints: vec![Constraint::Ge(OperandIx::new(0))],
                },
                &pool,
            )
            .is_none(),
            "a float bound is not an integer set"
        );
    }
}
