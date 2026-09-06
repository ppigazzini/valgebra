use super::*;
use valgebra_core::{DefIx, Field, MapClause};

/// A validator whose root is a bare back edge, so every schema it declares
/// sits in the definitions table. This is the shape a rewrite reaching only
/// the root leaves untouched, and it is what `recursive` builds.
fn recursive_record() -> Validator {
    Validator::new(
        Schema::Ref(DefIx::new(0)),
        Vec::new(),
        vec![Schema::record(
            vec![Field {
                name: "a".to_owned(),
                schema: Schema::Int,
                required: true,
            }],
            Openness::Closed,
        )],
    )
}

/// An oracle over a pool of objects, for the reader the descriptor uses.
fn pooled_over(literals: Vec<Py<PyAny>>) -> Validator {
    Validator::new(Schema::ANYTHING, literals, Vec::new())
}

#[test]
fn the_pool_reads_a_scalar_by_its_exact_type() {
    Python::attach(|py| {
        let literals = [
            py.None(),
            true.into_pyobject(py)
                .unwrap()
                .to_owned()
                .into_any()
                .unbind(),
            7i64.into_pyobject(py).unwrap().into_any().unbind(),
            0.5f64.into_pyobject(py).unwrap().into_any().unbind(),
            "ab".into_pyobject(py).unwrap().into_any().unbind(),
            PyBytes::new(py, b"ab").into_any().unbind(),
        ];
        let held = pooled_over(literals.into_iter().collect());
        let oracle = PoolRelations {
            py,
            literals: &held.literals,
            definitions: &held.definitions,
            classes: RefCell::default(),
        };
        let read: Vec<_> = (0..6)
            .map(|slot| oracle.constant(ConstIx::new(slot)))
            .collect();
        assert_eq!(
            read,
            [
                Some(Operand::NoneType),
                // `True` is a `bool`, not the integer 1: the two are distinct
                // operands although they compare equal.
                Some(Operand::Boolean(true)),
                Some(Operand::Integer(7)),
                Some(Operand::Float(0.5)),
                Some(Operand::Word(b"ab".to_vec(), Kind::Str)),
                Some(Operand::Word(b"ab".to_vec(), Kind::Bytes)),
            ]
        );
    });
}

#[test]
fn the_pool_declines_a_value_it_cannot_kind() {
    Python::attach(|py| {
        let scope = PyDict::new(py);
        py.run(c"class Odd(int): pass\nodd = Odd(1)", None, Some(&scope))
            .expect("an int subclass");
        let held = pooled_over(vec![scope.get_item("odd").unwrap().unwrap().unbind()]);
        let oracle = PoolRelations {
            py,
            literals: &held.literals,
            definitions: &held.definitions,
            classes: RefCell::default(),
        };
        // An `int` subclass carries its own `__eq__`, so its equality is not
        // the one the descriptor's integer sets are built on.
        assert_eq!(oracle.constant(ConstIx::new(0)), None);
    });
}

#[test]
fn a_class_carries_the_bases_its_mro_lists() {
    Python::attach(|py| {
        let scope = PyDict::new(py);
        py.run(
            c"class A: pass\nclass B(A): pass\nclass C: pass",
            None,
            Some(&scope),
        )
        .expect("three plain classes");
        let named = |name: &str| scope.get_item(name).unwrap().unwrap().unbind();
        let held = pooled_over(vec![named("A"), named("B"), named("C")]);
        let oracle = PoolRelations {
            py,
            literals: &held.literals,
            definitions: &held.definitions,
            classes: RefCell::default(),
        };
        let a = oracle.class(ClassIx::new(0)).expect("A denotes a set");
        let b = oracle.class(ClassIx::new(1)).expect("B denotes a set");
        let c = oracle.class(ClassIx::new(2)).expect("C denotes a set");
        assert!(b.derives_from(&a), "B lists A in its `__mro__`");
        assert!(!a.derives_from(&b));
        // Two unrelated pure classes still share `object`, and neither is
        // built on a builtin, so nothing here proves them disjoint.
        assert!(!a.disjoint_from(&c));
    });
}

#[test]
fn a_class_built_on_a_builtin_is_disjoint_from_one_built_on_another() {
    Python::attach(|py| {
        let scope = PyDict::new(py);
        py.run(
            c"class S(str): pass\nclass L(list): pass",
            None,
            Some(&scope),
        )
        .expect("two classes on conflicting layouts");
        let named = |name: &str| scope.get_item(name).unwrap().unwrap().unbind();
        let held = pooled_over(vec![named("S"), named("L")]);
        let oracle = PoolRelations {
            py,
            literals: &held.literals,
            definitions: &held.definitions,
            classes: RefCell::default(),
        };
        let s = oracle.class(ClassIx::new(0)).expect("S denotes a set");
        let l = oracle.class(ClassIx::new(1)).expect("L denotes a set");
        // Python refuses `class Both(S, L)`, so no value is an instance of
        // both and the two carry no common instance for the core to hold.
        assert!(s.disjoint_from(&l));
    });
}

#[test]
fn a_class_that_answers_isinstance_itself_refuses_the_lowering() {
    Python::attach(|py| {
        let scope = PyDict::new(py);
        py.run(
            c"import abc\nclass Hooked(abc.ABC): pass",
            None,
            Some(&scope),
        )
        .expect("an abstract base class");
        let held = pooled_over(vec![scope.get_item("Hooked").unwrap().unwrap().unbind()]);
        let oracle = PoolRelations {
            py,
            literals: &held.literals,
            definitions: &held.definitions,
            classes: RefCell::default(),
        };
        // `ABCMeta.register` can add a subclass after the schema is built, so
        // the class names no fixed set and the lowering declines it.
        assert_eq!(oracle.class(ClassIx::new(0)), None);
    });
}

#[test]
fn map_schemas_rewrites_the_definitions_table() {
    Python::attach(|py| {
        let mapped = recursive_record()
            .map_schemas(py, |schema| schema.with_records_open(Openness::Open))
            .expect("opening one record stays within the construction bounds");
        // The root is a leaf, so the rewrite has to land in the definitions
        // table for the record to have been opened at all.
        assert_eq!(mapped.schema, Schema::Ref(DefIx::new(0)));
        let Schema::KeyedMap { defaults, .. } = &mapped.definitions[0] else {
            panic!("the definition is a record");
        };
        assert_eq!(
            defaults.as_slice(),
            [MapClause::top()],
            "the record in the definitions table gained its catch-all clause"
        );
    });
}

#[test]
fn map_schemas_measures_what_the_rewrite_produced() {
    Python::attach(|py| {
        // Routing through `checked` rather than `new` is what rejects a
        // rewrite that grows the schema past the node bound.
        let wide = Validator::new(
            Schema::Union(vec![Schema::Int; MAX_SCHEMA_NODES / 2]),
            Vec::new(),
            Vec::new(),
        );
        assert!(wide.map_schemas(py, Schema::clone).is_ok());
        let doubled = wide.map_schemas(py, |schema| {
            Schema::Union(vec![schema.clone(), schema.clone()])
        });
        assert!(
            doubled.is_err(),
            "a rewrite that doubles the node count is past the bound"
        );
    });
}
