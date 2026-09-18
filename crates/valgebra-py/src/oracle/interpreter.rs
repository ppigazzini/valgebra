//! The oracle's own corpus: every question `LeafRelations` asks, driven here.
//!
//! This file exists because the mutation sweep said it had to. The answers in
//! this module are exercised all day by the decision suite in `tests/`, and
//! that suite is pytest, which `cargo mutants` cannot observe: swept on the
//! interpreter the lane pins, `oracle.rs` gave **62 survivors of 109**, because
//! the only Rust rows reaching it constructed the pool and read it back.
//!
//! So the rows below hand the oracle two pool slots, or a class and a kind, and
//! read the `Option<bool>` it answers. That is deliberately *not* what the
//! pytest suite does: there a schema is compiled and `is_subtype_of` is asked,
//! and what the oracle returned is visible only through the decision the core
//! made from it. A row that did the same in Rust would prove the decision, not
//! the answer, and would leave the same mutants alive.
//!
//! Three answers everywhere, and the third is the point. `None` is "this oracle
//! cannot read that", which the core folds into a conservative verdict; `false`
//! is a refutation the core is entitled to act on. Confusing the two is the
//! unsound direction, so every row that expects `None` says so rather than
//! reading it as a negative.

use std::ffi::CString;

use pyo3::types::{PyDict, PyList};

use super::*;
use crate::validator::Validator;

/// A validator holding `literals` as its pool, for an oracle to read.
fn pooled(literals: Vec<Py<PyAny>>) -> Validator {
    Validator::new(Schema::ANYTHING, literals, Vec::new())
}

/// Evaluate `source` in a namespace with `types` and `enum` available.
fn built<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyAny> {
    let namespace = PyDict::new(py);
    for module in ["types", "enum", "typing"] {
        namespace
            .set_item(module, py.import(module).expect("the module imports"))
            .expect("a namespace holds it");
    }
    py.eval(
        &CString::new(source).expect("no interior nul"),
        Some(&namespace),
        None,
    )
    .expect("the expression evaluates")
}

/// Run `read` against an oracle over `literals`.
fn asking<T>(
    py: Python<'_>,
    literals: Vec<Py<PyAny>>,
    read: impl FnOnce(&PoolRelations<'_, '_>) -> T,
) -> T {
    let held = pooled(literals);
    let oracle = PoolRelations::new(py, &held.literals, &held.definitions);
    read(&oracle)
}

#[test]
fn a_literal_is_a_subtype_of_the_kind_and_the_class_that_hold_it() {
    // `leaf_subtype` is the widest question the trait asks: is this leaf's set
    // inside that leaf's. A literal against a scalar kind is read from the
    // value's own type, and against a class from `isinstance`.
    Python::attach(|py| {
        let seven = 7i64.into_pyobject(py).unwrap().into_any().unbind();
        let text = "ab".into_pyobject(py).unwrap().into_any().unbind();
        let integer = py.get_type::<PyInt>().into_any().unbind();
        let string = py.get_type::<PyString>().into_any().unbind();
        asking(py, vec![seven, text, integer, string], |oracle| {
            let literal = |slot| Schema::Literal(ConstIx::new(slot));
            let instance = Schema::Instance(ClassIx::new(2));
            assert_eq!(oracle.leaf_subtype(&literal(0), &Schema::Int), Some(true));
            assert_eq!(oracle.leaf_subtype(&literal(0), &Schema::Str), Some(false));
            assert_eq!(oracle.leaf_subtype(&literal(1), &Schema::Str), Some(true));
            // An `int` literal is an instance of `int`, and a `str` is not.
            assert_eq!(oracle.leaf_subtype(&literal(0), &instance), Some(true));
            assert_eq!(oracle.leaf_subtype(&literal(1), &instance), Some(false));
            // A class against a class is subclassing, which is the other arm.
            let text_class = Schema::Instance(ClassIx::new(3));
            assert_eq!(oracle.leaf_subtype(&instance, &instance), Some(true));
            assert_eq!(oracle.leaf_subtype(&instance, &text_class), Some(false));
            // And a bare *kind* on the left is declined rather than answered.
            // A kind is the core's own partition; what the core asks the
            // bindings is about the values it cannot see, and `Int` is not one
            // of those. Reading the decline as a refutation here would make
            // every such pair refute.
            assert_eq!(oracle.leaf_subtype(&Schema::Int, &instance), None);
        });
    });
}

#[test]
fn a_literal_reports_the_kind_its_value_has() {
    // `literal_kind` places a constant in the partition. A `bool` is its own
    // kind rather than an integer, which is the distinction every rule that
    // reads a literal union rests on.
    Python::attach(|py| {
        let values = vec![
            py.None(),
            true.into_pyobject(py)
                .unwrap()
                .to_owned()
                .into_any()
                .unbind(),
            7i64.into_pyobject(py).unwrap().into_any().unbind(),
            "ab".into_pyobject(py).unwrap().into_any().unbind(),
            // A class is not a scalar: the oracle declines rather than guessing.
            py.get_type::<PyInt>().into_any().unbind(),
        ];
        asking(py, values, |oracle| {
            let kind = |slot| oracle.literal_kind(ConstIx::new(slot));
            assert_eq!(kind(0), Some(Kind::NoneType));
            assert_eq!(kind(1), Some(Kind::Bool));
            assert_eq!(kind(2), Some(Kind::Int));
            assert_eq!(kind(3), Some(Kind::Str));
            assert_eq!(kind(4), None, "a class has no scalar kind to report");
        });
    });
}

#[test]
fn two_constants_are_disjoint_unless_they_are_the_same_value() {
    // `literals_disjoint`: do these two singletons share a value. Two `1`s do;
    // `1` and `True` do not, although `1 == True` -- the pool reads a constant
    // by its exact type, and a rule that believed `==` would make `Literal[1]`
    // admit `True`.
    Python::attach(|py| {
        let values = vec![
            1i64.into_pyobject(py).unwrap().into_any().unbind(),
            1i64.into_pyobject(py).unwrap().into_any().unbind(),
            2i64.into_pyobject(py).unwrap().into_any().unbind(),
            true.into_pyobject(py)
                .unwrap()
                .to_owned()
                .into_any()
                .unbind(),
        ];
        asking(py, values, |oracle| {
            let pair = |a, b| oracle.literals_disjoint(ConstIx::new(a), ConstIx::new(b));
            assert_eq!(pair(0, 1), Some(false), "the same value, twice");
            assert_eq!(pair(0, 2), Some(true), "1 and 2 share nothing");
            assert_eq!(pair(0, 3), Some(true), "1 is not True");
        });
    });
}

#[test]
fn two_sets_of_constants_are_disjoint_when_no_member_is_shared() {
    // `literal_sets_disjoint` is the same relation over two *sets*, which is
    // the form the core asks when it compares a union with a union: member by
    // member is quadratic in this call, and an implementor that can hash its
    // constants answers in one pass. Both edges and the empty case.
    Python::attach(|py| {
        let values: Vec<_> = (1..=4)
            .map(|n| i64::from(n).into_pyobject(py).unwrap().into_any().unbind())
            .collect();
        asking(py, values, |oracle| {
            let ix = |slots: &[usize]| -> Vec<ConstIx> {
                slots.iter().map(|slot| ConstIx::new(*slot)).collect()
            };
            let sets = |a: &[usize], b: &[usize]| oracle.literal_sets_disjoint(&ix(a), &ix(b));
            assert_eq!(
                sets(&[0, 1], &[2, 3]),
                Some(true),
                "one-two against three-four"
            );
            assert_eq!(sets(&[0, 1], &[1, 2]), Some(false), "2 is in both");
            assert_eq!(sets(&[0], &[0]), Some(false), "one shared member");
            // A set with nothing in it shares nothing with anything, itself
            // included: the empty set is disjoint from every set.
            assert_eq!(sets(&[], &[0, 1]), Some(true));
            assert_eq!(sets(&[], &[]), Some(true));
        });
    });
}

#[test]
fn two_bounds_are_ordered_where_python_orders_them() {
    // `compare` orders two refinement operands. Python orders numbers with
    // numbers and text with text and raises across the two, and a bound whose
    // operand is in another group than its base compares nothing -- so the
    // oracle declines rather than inventing an order.
    Python::attach(|py| {
        let values = vec![
            1i64.into_pyobject(py).unwrap().into_any().unbind(),
            2i64.into_pyobject(py).unwrap().into_any().unbind(),
            1i64.into_pyobject(py).unwrap().into_any().unbind(),
            "ab".into_pyobject(py).unwrap().into_any().unbind(),
        ];
        asking(py, values, |oracle| {
            let cmp = |a, b| oracle.compare(OperandIx::new(a), OperandIx::new(b));
            assert_eq!(cmp(0, 1), Some(core::cmp::Ordering::Less));
            assert_eq!(cmp(1, 0), Some(core::cmp::Ordering::Greater));
            assert_eq!(cmp(0, 2), Some(core::cmp::Ordering::Equal));
            assert_eq!(cmp(0, 3), None, "an int and a str order nothing");
        });
    });
}

#[test]
fn a_class_admits_the_kind_its_layout_lays_down() {
    // `class_admits_kind`: can a value of this kind be an instance of that
    // class. Read from the layout, because a class deriving from a builtin
    // holds that builtin's kind and no other, and a subclass cannot escape it.
    //
    // The declines are the half that matters. A class laying down no layout is
    // declined rather than refused: its own instances are plain objects, but
    // `isinstance` reads the subtree beneath it and a subclass may derive from
    // a builtin as well.
    Python::attach(|py| {
        let module = "types.SimpleNamespace(\
             plain=type('Plain', (), {}),\
             from_str=type('FromStr', (str,), {}),\
         )";
        let built = built(py, module);
        let plain = built.getattr("plain").unwrap().unbind();
        let from_str = built.getattr("from_str").unwrap().unbind();
        let integer = py.get_type::<PyInt>().into_any().unbind();
        asking(py, vec![integer, from_str, plain], |oracle| {
            let admits = |slot, kind| oracle.class_admits_kind(ClassIx::new(slot), kind);
            assert_eq!(admits(0, Kind::Int), Some(true), "int holds ints");
            assert_eq!(admits(0, Kind::Str), Some(false), "and holds no str");
            assert_eq!(admits(1, Kind::Str), Some(true), "a str subclass is str");
            assert_eq!(admits(1, Kind::Int), Some(false));
            // The decline: a plain class lays down no layout, and a subclass of
            // it may still derive from a builtin.
            assert_eq!(admits(2, Kind::Str), None, "a plain class is not read");
        });
    });
}

#[test]
fn a_direct_instance_is_read_from_the_class_itself() {
    // `direct_instance_of_kind` asks about instances of *this* class rather
    // than of the subtree below it, so a plain class is answerable where
    // `class_admits_kind` declines: its own instances are plain objects.
    Python::attach(|py| {
        let built = built(
            py,
            "types.SimpleNamespace(plain=type('Plain', (), {}), from_str=type('FromStr', (str,), {}))",
        );
        let plain = built.getattr("plain").unwrap().unbind();
        let from_str = built.getattr("from_str").unwrap().unbind();
        asking(py, vec![plain, from_str], |oracle| {
            let direct = |slot, kind| oracle.direct_instance_of_kind(ClassIx::new(slot), kind);
            assert_eq!(
                direct(0, Kind::Str),
                Some(false),
                "a plain object is no str"
            );
            assert_eq!(direct(1, Kind::Str), Some(true));
            assert_eq!(direct(1, Kind::Int), Some(false));
        });
    });
}

#[test]
fn a_kind_derives_from_a_class_when_its_values_are_instances() {
    // `kind_derives_from` is the other direction: is every value of this kind
    // an instance of that class. `bool` derives from `int` because the typing
    // spec says a boolean is an integer, which is the relation this answer
    // exists to carry.
    Python::attach(|py| {
        let integer = py.get_type::<PyInt>().into_any().unbind();
        let text = py.get_type::<PyString>().into_any().unbind();
        let object = built(py, "type('O', (), {}).__mro__[1]").unbind();
        asking(py, vec![integer, text, object], |oracle| {
            let derives = |kind, slot| oracle.kind_derives_from(kind, ClassIx::new(slot));
            assert_eq!(derives(Kind::Int, 0), Some(true));
            assert_eq!(derives(Kind::Bool, 0), Some(true), "a bool is an int");
            assert_eq!(derives(Kind::Str, 0), Some(false));
            assert_eq!(derives(Kind::Str, 1), Some(true));
            assert_eq!(derives(Kind::Int, 2), Some(true), "everything is an object");
        });
    });
}

#[test]
fn an_atom_denotes_a_set_unless_the_class_decides_for_itself() {
    // `atom_denotes_a_set` asks whether an atom stands for a fixed set of
    // values at all. A class whose metaclass overrides `__instancecheck__`
    // answers `isinstance` with user code, so what it holds is not a property
    // of a value's type and no rule may reason about it.
    Python::attach(|py| {
        let source = "types.SimpleNamespace(\
             plain=type('Plain', (), {}),\
             lying=type('Meta', (type,), {'__instancecheck__': lambda self, other: True})('Lying', (), {}),\
         )";
        let built = built(py, source);
        let plain = built.getattr("plain").unwrap().unbind();
        let lying = built.getattr("lying").unwrap().unbind();
        asking(py, vec![plain, lying], |oracle| {
            let atom = |slot| oracle.atom_denotes_a_set(&Schema::Instance(ClassIx::new(slot)));
            assert_eq!(
                atom(0),
                Some(true),
                "an ordinary class denotes its instances"
            );
            assert_eq!(
                atom(1),
                Some(false),
                "a class that answers for itself does not"
            );
            // A scalar kind is not an atom this reads. The question is about
            // a *class* -- whether `isinstance` against it is a property of a
            // value's type -- and a kind is the partition the core already
            // owns, so the oracle declines rather than agreeing.
            assert_eq!(oracle.atom_denotes_a_set(&Schema::Int), None);
        });
    });
}

#[test]
fn the_integers_between_two_bounds_are_counted_where_both_are_integers() {
    // `no_int_between` decides whether a pair of bounds admits an integer, and
    // it is the answer a contradiction between two bounds rests on. Its one
    // call site reads `== Some(true)`, so a decline and a refutation collapse
    // there -- which is why the `Some(false)` direction is the one a mutation
    // of it must not take, and why the rows below pin all three.
    Python::attach(|py| {
        let values = vec![
            1i64.into_pyobject(py).unwrap().into_any().unbind(),
            2i64.into_pyobject(py).unwrap().into_any().unbind(),
            5i64.into_pyobject(py).unwrap().into_any().unbind(),
            "ab".into_pyobject(py).unwrap().into_any().unbind(),
        ];
        asking(py, values, |oracle| {
            // Strict at both ends: the open interval between the two bounds.
            let open =
                |a, b| oracle.no_int_between(OperandIx::new(a), true, OperandIx::new(b), true);
            // Nothing strictly between 1 and 2.
            assert_eq!(open(0, 1), Some(true));
            // 2, 3 and 4 sit strictly between 1 and 5.
            assert_eq!(open(0, 2), Some(false));
            // A bound that is not an integer is not a bound this can count to.
            assert_eq!(open(0, 3), None);
            // Closed at both ends the same pair holds 1 and 2 themselves, so
            // the answer flips: the strictness is part of the question rather
            // than a detail of how it is asked.
            assert_eq!(
                oracle.no_int_between(OperandIx::new(0), false, OperandIx::new(1), false),
                Some(false)
            );
        });
    });
}

#[test]
fn one_step_divides_another_by_the_operator_the_walk_uses() {
    // `divides` settles an inclusion between two `MultipleOf` refinements
    // between the two steps, so its answer is about the operands and not about
    // any representation. Three answers, and each has a call site: `Some(true)`
    // proves the inclusion, `Some(false)` leaves it unproven, and `None` is a
    // pair `%` cannot be asked of at all.
    Python::attach(|py| {
        let values = vec![
            2500i64.into_pyobject(py).unwrap().into_any().unbind(),
            5000i64.into_pyobject(py).unwrap().into_any().unbind(),
            7i64.into_pyobject(py).unwrap().into_any().unbind(),
            0.125f64.into_pyobject(py).unwrap().into_any().unbind(),
            0.25f64.into_pyobject(py).unwrap().into_any().unbind(),
            "ab".into_pyobject(py).unwrap().into_any().unbind(),
        ];
        asking(py, values, |oracle| {
            let divides = |s, m| oracle.divides(OperandIx::new(s), OperandIx::new(m));
            // Every multiple of 5,000 is a multiple of 2,500, and the size of
            // neither step enters into it.
            assert_eq!(divides(0, 1), Some(true));
            // And not the other way: 2,500 is a multiple of itself and not of
            // 5,000, which is what makes the answer a claim rather than a
            // symmetry.
            assert_eq!(divides(1, 0), Some(false));
            assert_eq!(divides(2, 0), Some(false));
            // A step divides itself, which is the reflexive row every
            // entailment rests on.
            assert_eq!(divides(0, 0), Some(true));
            // The operator is the operands', so a float step answers by float
            // division rather than by a rule about integers.
            assert_eq!(divides(3, 4), Some(true));
            assert_eq!(divides(4, 3), Some(false));
            // A pair `%` cannot be asked of declines rather than guessing: a
            // string and a number raise, and the inclusion stays unproven.
            assert_eq!(divides(0, 5), None);
            assert_eq!(divides(5, 0), None);
            // An index the pool does not hold is not a question at all.
            assert_eq!(oracle.divides(OperandIx::new(0), OperandIx::new(99)), None);
        });
    });
}

#[test]
fn an_enumeration_lists_its_members_up_to_the_bound() {
    // The enum reading is what turns a class into the union of the values it
    // lists, and `MAX_ENUM_MEMBERS` is where it stops: past the bound the
    // relation would become one membership question per member, so the oracle
    // declines and the class stays an instance check.
    Python::attach(|py| {
        let small = built(py, "enum.Enum('Small', {'A': 1, 'B': 2})").unbind();
        let wide = built(py, "enum.Enum('Wide', {f'M{n}': n for n in range(600)})").unbind();
        let plain = built(py, "type('Plain', (), {})").unbind();
        let listed =
            |held: &Py<PyAny>| PoolRelations::enum_members(held.bind(py)).map(|got| got.len());
        assert_eq!(listed(&small), Some(2), "two members, listed");
        assert_eq!(
            listed(&wide),
            None,
            "past MAX_ENUM_MEMBERS the class stays an instance check"
        );
        assert_eq!(listed(&plain), None, "a plain class lists nothing");
    });
}

#[test]
fn a_pool_slot_out_of_range_declines_rather_than_answering() {
    // Every reader here indexes the pool, and a slot the pool does not hold is
    // a schema and a pool that were not built together. The oracle declines,
    // which the core folds conservatively; answering would be a claim built on
    // a value nobody supplied.
    Python::attach(|py| {
        asking(
            py,
            vec![1i64.into_pyobject(py).unwrap().into_any().unbind()],
            |oracle| {
                assert_eq!(oracle.literal_kind(ConstIx::new(9)), None);
                assert_eq!(
                    oracle.literals_disjoint(ConstIx::new(0), ConstIx::new(9)),
                    None
                );
                assert_eq!(oracle.compare(OperandIx::new(9), OperandIx::new(0)), None);
                assert_eq!(oracle.class_admits_kind(ClassIx::new(9), Kind::Int), None);
            },
        );
    });
}

#[test]
fn a_list_is_not_a_scalar_and_is_read_as_nothing_the_pool_can_order() {
    // The pool holds four kinds of object and a constant is only one of them.
    // A container in a constant slot is not an operand, not a kind, and not
    // ordered -- three declines from one value, which is the shape a schema
    // built against the wrong pool would produce.
    Python::attach(|py| {
        let listed = PyList::new(py, [1i64]).unwrap().into_any().unbind();
        let seven = 7i64.into_pyobject(py).unwrap().into_any().unbind();
        asking(py, vec![listed, seven], |oracle| {
            assert_eq!(oracle.literal_kind(ConstIx::new(0)), None);
            assert_eq!(oracle.constant(ConstIx::new(0)), None);
            assert_eq!(
                oracle.compare(OperandIx::new(0), OperandIx::new(1)),
                None,
                "a list orders against nothing"
            );
        });
    });
}

#[test]
fn a_type_whose_equality_is_its_own_is_not_trusted_to_decide_disjointness() {
    // Two constants are disjoint when they are different values, and "different"
    // is `==`. That is the value's own `==`, so a type that defines one can call
    // two distinct objects equal -- or two equal ones distinct -- and a rule
    // built on its answer would decide a set relation from a user's method.
    //
    // So the question is asked only where the equality is one this oracle can
    // trust: a builtin scalar, whose `==` is Python's, or a type comparing by
    // identity, where two objects are two values by definition. Anything else
    // is declined, and the decline is the whole of the guard -- reading it as
    // "not disjoint" would merge two literals the caller wrote apart.
    Python::attach(|py| {
        let source = "types.SimpleNamespace(\
             always=type('Always', (), {'__eq__': lambda self, other: True, '__hash__': lambda self: 0}),\
             never=type('Never', (), {'__eq__': lambda self, other: False, '__hash__': lambda self: 0}),\
         )";
        let kinds = built(py, source);
        let always = kinds.getattr("always").unwrap();
        let never = kinds.getattr("never").unwrap();
        let values = vec![
            always.call0().unwrap().unbind(),
            always.call0().unwrap().unbind(),
            never.call0().unwrap().unbind(),
            never.call0().unwrap().unbind(),
            1i64.into_pyobject(py).unwrap().into_any().unbind(),
        ];
        asking(py, values, |oracle| {
            let pair = |a, b| oracle.literals_disjoint(ConstIx::new(a), ConstIx::new(b));
            // Two objects of a type that says everything is equal: declined,
            // not answered "not disjoint".
            assert_eq!(pair(0, 1), None, "a type that answers == for itself");
            // And one that says nothing is equal, including itself: declined
            // too, or two constants that are one value would read as two.
            assert_eq!(pair(2, 3), None);
            // A builtin beside it is still answered: the guard is about the
            // type of the values compared, not about the pool holding one.
            assert_eq!(pair(4, 4), Some(false), "1 against itself");

            // The set form applies the same guard per member, so a set holding
            // one untrusted value declines whatever else it holds.
            let ix = |slots: &[usize]| -> Vec<ConstIx> {
                slots.iter().map(|slot| ConstIx::new(*slot)).collect()
            };
            assert_eq!(oracle.literal_sets_disjoint(&ix(&[0]), &ix(&[4])), None);
            assert_eq!(oracle.literal_sets_disjoint(&ix(&[4]), &ix(&[0])), None);
            // Both sides trusted, so the set question is answered.
            assert_eq!(
                oracle.literal_sets_disjoint(&ix(&[4]), &ix(&[4])),
                Some(false)
            );

            // A set holding *both* a shared member and an untrusted one is
            // answered rather than declined, and which side is hashed decides
            // that. The smaller side is hashed and the larger probed, and the
            // probe stops at the first shared member it finds -- so a shared
            // member found before an untrusted one is reached is a proof of
            // non-disjointness, which stands whatever else the set holds.
            //
            // Hashing the larger side instead walks the untrusted value first
            // and declines. Both answers are sound -- a decline is always
            // sound -- so this is precision rather than correctness, and it is
            // pinned because it is the one thing the choice of side decides.
            assert_eq!(
                oracle.literal_sets_disjoint(&ix(&[4, 0]), &ix(&[4])),
                Some(false),
                "a shared member found first is an answer, not a decline"
            );
        });
    });
}

#[test]
fn the_descriptor_reads_an_operand_through_the_same_pool_as_a_constant() {
    // `Constants` has two readers and they answer the same question of two
    // index types: a refinement bound and a literal are both pooled objects,
    // and the descriptor reaches them through this trait rather than touching
    // Python. A reader that declined would leave the lowering with no value.
    Python::attach(|py| {
        let values = vec![
            7i64.into_pyobject(py).unwrap().into_any().unbind(),
            "ab".into_pyobject(py).unwrap().into_any().unbind(),
        ];
        asking(py, values, |oracle| {
            assert_eq!(oracle.operand(OperandIx::new(0)), Some(Operand::Integer(7)));
            assert_eq!(
                oracle.operand(OperandIx::new(1)),
                Some(Operand::Word(b"ab".to_vec(), Kind::Str))
            );
            assert_eq!(oracle.operand(OperandIx::new(9)), None, "past the pool");
            // The two readers agree, which is what makes one pool serve both.
            assert_eq!(
                oracle.operand(OperandIx::new(0)),
                oracle.constant(ConstIx::new(0))
            );
        });
    });
}

#[test]
fn two_builtins_lay_down_layouts_that_conflict() {
    // The layout tag is what says two classes share no instance: a class
    // deriving from `int` and one deriving from `str` cannot both describe one
    // value, because Python refuses a class body laying down two layouts.
    // `Class::PLAIN` is the tag for laying none down, and it conflicts with
    // nothing -- so a builtin whose tag collapsed into it would stop refuting
    // the pairs it exists to refute.
    //
    // `int` is the first row of the table the tags are read from, which makes
    // it the one an off-by-one collapses into `PLAIN`. The assertions are the
    // relation rather than the number, because the number is the oracle's own
    // business and the relation is what the core reads.
    Python::attach(|py| {
        let integer = py.get_type::<PyInt>().into_any().unbind();
        let text = py.get_type::<PyString>().into_any().unbind();
        let from_int = built(py, "type('FromInt', (int,), {})").unbind();
        let plain = built(py, "type('Plain', (), {})").unbind();
        asking(py, vec![integer, text, from_int, plain], |oracle| {
            let class = |slot| oracle.class(ClassIx::new(slot)).expect("the class is read");
            let (integer, text) = (class(0), class(1));
            let (from_int, plain) = (class(2), class(3));
            assert!(
                integer.disjoint_from(&text),
                "an int and a str share no value"
            );
            // A subclass derives from its base, so the two are not disjoint
            // however their layouts compare.
            assert!(!integer.disjoint_from(&from_int));
            assert!(from_int.derives_from(&integer));
            // And a plain class conflicts with nothing: it lays down no layout,
            // and a subclass of it may derive from a builtin as well.
            assert!(!plain.disjoint_from(&integer));
            assert!(!integer.disjoint_from(&plain));
        });
    });
}

#[test]
fn a_class_relation_needs_both_sides_to_denote_a_set() {
    // `leaf_subtype` reads the class order for a pair of `Instance` atoms, and
    // the order answers for a class only where `isinstance` is a property of a
    // value's type. A metaclass that computes either check breaks that on its
    // own side alone: a hooked *superclass* holds whatever its code says, and a
    // hooked *subject* is a set no snapshot of the order describes. So the
    // decline is either side, not both -- reading it as both lets one hooked
    // class through whichever position it takes.
    Python::attach(|py| {
        let source = "types.SimpleNamespace(\
             plain=type('Plain', (), {}),\
             also=type('Also', (), {}),\
             instance_hook=type('MetaI', (type,), {'__instancecheck__': lambda self, other: True})('HookedI', (), {}),\
             subclass_hook=type('MetaS', (type,), {'__subclasscheck__': lambda self, other: True})('HookedS', (), {}),\
         )";
        let built = built(py, source);
        let slots: Vec<Py<PyAny>> = ["plain", "also", "instance_hook", "subclass_hook"]
            .iter()
            .map(|name| built.getattr(*name).unwrap().unbind())
            .collect();
        asking(py, slots, |oracle| {
            let relate = |sub: usize, sup: usize| {
                oracle.leaf_subtype(
                    &Schema::Instance(ClassIx::new(sub)),
                    &Schema::Instance(ClassIx::new(sup)),
                )
            };
            assert_eq!(
                relate(0, 1),
                Some(false),
                "two ordinary classes are read from the order"
            );
            assert_eq!(relate(0, 0), Some(true), "a class is below itself");
            for hooked in [2, 3] {
                assert_eq!(
                    relate(0, hooked),
                    None,
                    "a hooked supertype is a set the order does not describe"
                );
                assert_eq!(relate(hooked, 0), None, "and so is a hooked subject");
            }
        });
    });
}
