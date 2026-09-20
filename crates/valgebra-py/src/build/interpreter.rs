use std::sync::Arc;

use super::classes::*;
use super::refine::*;
use super::*;
use crate::render::render;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::ffi::CString;
use valgebra_core::{
    Carries, Constraint, MapClause, SeqShape, carries_division, carries_length, carries_pattern,
    carries_through,
};

/// The namespace a row's expression is evaluated in.
///
/// `at` holds the marker doubles, named for the vocabulary they stand in
/// for so a row reads as the line a caller would write.
/// The release a corpus row needs, as `3.n`.
///
/// A corpus reads *live* objects, so a row naming `typing.Required` or a star
/// inside a subscript is a row about an interpreter that has them: below the
/// release that added one, the name is an `AttributeError` and the syntax is a
/// `SyntaxError`, and the row would be asserting about the release rather than
/// about this crate. The release rides on the row, which is what lets a reader
/// see which lane drives it -- and every lane above it does.
///
/// One spelling, because `tests/test_version_gates.py` holds each release to
/// the lanes and reads this one: a comparison written out here would carry a
/// release nothing holds. The Python suite writes the same condition as
/// `skipif(sys.version_info < (3, n))`, and the ledger reads both.
#[derive(Clone, Copy)]
struct Since(u8);

impl Since {
    /// Whether the interpreter this links is at or above the release.
    fn met(self, py: Python<'_>) -> bool {
        py.version_info() >= (3, self.0)
    }
}

fn namespace(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let namespace = PyDict::new(py);
    for module in ["typing", "dataclasses", "enum", "re", "types"] {
        namespace.set_item(module, py.import(module)?)?;
    }
    py.run(
        &CString::new(
            "import types\n\
             class at:\n\
             \x20   Ge = staticmethod(lambda n: types.SimpleNamespace(ge=n))\n\
             \x20   Le = staticmethod(lambda n: types.SimpleNamespace(le=n))\n\
             \x20   MinLen = staticmethod(\n\
             \x20       lambda n: types.SimpleNamespace(min_length=n)\n\
             \x20   )\n\
             class slotted:\n\
             \x20   class Ge:\n\
             \x20       __slots__ = ('ge',)\n\
             \x20       def __init__(self, n): self.ge = n\n\
             \x20   class MinLen:\n\
             \x20       __slots__ = ('min_length',)\n\
             \x20       def __init__(self, n): self.min_length = n\n\
             class guarded:\n\
             \x20   __slots__ = ('ge',)\n\
             \x20   def __init__(self, n): self.ge = n\n\
             \x20   def __getattr__(self, name): raise AttributeError(name)\n\
             class hooked:\n\
             \x20   def __getattr__(self, name):\n\
             \x20       if name == 'ge': return 0\n\
             \x20       raise AttributeError(name)\n\
             class Timezone:\n\
             \x20   pass\n\
             Timezone.__module__ = 'annotated_types'\n\
             class grouped:\n\
             \x20   __is_annotated_types_grouped_metadata__ = True\n\
             \x20   def __init__(self, *items): self.items = items\n\
             \x20   def __iter__(self): return iter(self.items)\n\
             class endless:\n\
             \x20   __is_annotated_types_grouped_metadata__ = True\n\
             \x20   def __iter__(self): return iter([self])\n\
             def nest(depth):\n\
             \x20   marker = at.Ge(0)\n\
             \x20   for _ in range(depth): marker = grouped(marker)\n\
             \x20   return marker\n",
        )?,
        Some(&namespace),
        None,
    )?;
    Ok(namespace)
}

/// Build a schema from an annotation expression and render it back.
fn built(py: Python<'_>, expression: &str) -> PyResult<String> {
    let namespace = namespace(py)?;
    let annotation = py.eval(&CString::new(expression)?, Some(&namespace), None)?;
    let mut pool = Pool::default();
    let mut defs = Vec::new();
    let schema = build_schema(&annotation, &mut pool, &mut defs)?;
    let active = RefCell::new(FxHashMap::default());
    Ok(render(py, &schema, pool.items(), &defs, &active, 0))
}

/// Every spelling the frontend dispatches on, and the schema it must reach.
///
/// One row per arm rather than per feature: a mutant that folds two arms
/// together is killed by the row of either, and a mutant that drops an arm
/// is killed by its own.
#[test]
fn each_spelling_builds_its_own_schema() {
    Python::attach(|py| {
        for (expression, wanted) in [
            // The scalars and the two bounds, which are the leaves every
            // other row is built out of.
            ("int", "int"),
            // A literal whose constant *can* be asked the constraint narrows
            // exactly as its kind does, so the refusals above are about the
            // kind rather than about the form.
            (
                "typing.Annotated[typing.Literal['ab'], at.MinLen(1)]",
                "Annotated[Literal['ab'], MinLen(1)]",
            ),
            (
                "typing.Annotated[typing.Literal[4], at.Ge(0)]",
                "Annotated[Literal[4], Ge(0)]",
            ),
            // A marker standing for the constraints it yields, which is the
            // protocol `annotated_types` documents and the shape `Interval` and
            // `Len` are written in. Read by attribute alone it is metadata this
            // frontend does not recognise, which leaves the base admitting
            // everything the marker excludes.
            (
                "typing.Annotated[int, grouped(at.Ge(0), at.Le(10))]",
                "Annotated[int, Ge(0), Le(10)]",
            ),
            // A group of groups bottoms out, and the depth it is followed to is
            // counted rather than assumed.
            (
                "typing.Annotated[int, grouped(grouped(at.Ge(0)))]",
                "Annotated[int, Ge(0)]",
            ),
            ("typing.Annotated[int, nest(7)]", "Annotated[int, Ge(0)]"),
            ("bool", "bool"),
            ("float", "float"),
            ("str", "str"),
            ("bytes", "bytes"),
            ("None", "None"),
            ("type(None)", "None"),
            ("object", "anything"),
            ("typing.Any", "Any"),
            // A bare container class is its kind, which is what the typing
            // spec assigns an unparameterised generic.
            ("list", "list[anything]"),
            ("set", "set[anything]"),
            ("frozenset", "frozenset[anything]"),
            // The parameterised forms, one per container arm.
            ("list[int]", "list[int]"),
            ("set[str]", "set[str]"),
            ("frozenset[bytes]", "frozenset[bytes]"),
            ("dict[str, int]", "dict[str, int]"),
            // A tuple is four arms: fixed, homogeneous, prefix-tail, and
            // the unpacked spellings of the last two.
            ("tuple[int, str]", "tuple[int, str]"),
            ("tuple[int, ...]", "tuple[int, ...]"),
            ("tuple[()]", "tuple[()]"),
            // The list-literal spellings, which are this library's own and
            // the only place a prefix and a repeated tail are written
            // without `Unpack`.
            ("[int]", "list[int]"),
            ("[int, str]", "[int, str]"),
            ("[int, ...]", "list[int]"),
            ("[str, int, ...]", "[str, int, ...]"),
            ("[]", "[]"),
            // The connectives, in both spellings where there are two.
            ("int | str", "int | str"),
            ("typing.Union[int, str]", "int | str"),
            ("typing.Optional[int]", "None | int"),
            // A union inside a generic argument, in both spellings: the
            // argument is read by its own path, and each origin is one of
            // the two a union can have.
            ("list[int | str]", "list[int | str]"),
            (
                "dict[str, typing.Union[int, None]]",
                "dict[str, None | int]",
            ),
            // A literal is a typed singleton; several are a union of them.
            ("typing.Literal[1]", "Literal[1]"),
            ("typing.Literal['a', 'b']", "Literal['a'] | Literal['b']"),
            // A callable is a class here: what a runtime check can ask of a
            // value is whether it is one, not what it accepts or returns.
            ("typing.Callable[[int], int]", "Callable"),
            // Metadata the frontend does not recognise is ignored, as the
            // typing spec says to -- unless it is a constraint from the
            // vocabulary, which is the refusal below.
            ("typing.Annotated[int, 'a note']", "int"),
            (
                "typing.Annotated[int, types.SimpleNamespace(multiple_of=3)]",
                "Annotated[int, MultipleOf(3)]",
            ),
            // A refinement carries its markers on the base it narrows, and
            // a nested one folds onto that base rather than nesting.
            ("typing.Annotated[int, at.Ge(0)]", "Annotated[int, Ge(0)]"),
            // The shapes a marker keeps its names in, which decide where the
            // frontend reads them from. `SimpleNamespace` above keeps them in
            // an instance dictionary; the vocabulary this stands in for ships
            // `slots` dataclasses, which keep them on the *type* as
            // descriptors, and a marker may answer through `__getattr__` for a
            // name no dictionary of either holds. For three releases this
            // corpus asked only the first, so the reading of the others was
            // held by pytest alone -- which the mutation sweep cannot see.
            (
                "typing.Annotated[int, slotted.Ge(0)]",
                "Annotated[int, Ge(0)]",
            ),
            (
                "typing.Annotated[str, slotted.MinLen(1)]",
                "Annotated[str, MinLen(1)]",
            ),
            ("typing.Annotated[int, hooked()]", "Annotated[int, Ge(0)]"),
            // Both at once: the name is on the type *and* the type has a hook,
            // which is the one marker whose mask is exactly the bit for that
            // name beside the hook's. A reading that compared those two the
            // wrong way would answer "nothing carried" for precisely this
            // shape and for no other, so the row is the shape rather than an
            // example.
            ("typing.Annotated[int, guarded(0)]", "Annotated[int, Ge(0)]"),
            (
                "typing.Annotated[str, at.MinLen(1)]",
                "Annotated[str, MinLen(1)]",
            ),
            (
                "typing.Annotated[typing.Annotated[int, at.Ge(0)], at.Le(9)]",
                "Annotated[int, Ge(0), Le(9)]",
            ),
        ] {
            let got = built(py, expression).unwrap_or_else(|error| {
                panic!("{expression} did not build: {error}");
            });
            assert_eq!(got, wanted, "{expression}");
        }
    });
}

/// The spellings a later release adds, each held where the release has it.
///
/// Apart from the table above because a row is read by whoever runs it, and
/// these are read by fewer lanes: the release each needs rides on the row and
/// [`Since`] says what that means. Together they are one table -- a spelling
/// belongs to the frontend's dispatch whether or not the floor can write it.
#[test]
fn each_spelling_a_release_adds_builds_where_that_release_has_it() {
    Python::attach(|py| {
        for (since, expression, wanted) in [
            (Since(11), "typing.Never", "nothing"),
            // The two unpacked tuple spellings: a star inside a subscript is
            // a syntax error before 3.11, so the row cannot even be written
            // for that release to read.
            (
                Since(11),
                "tuple[str, *tuple[int, ...]]",
                "tuple[str, int, ...]",
            ),
            (
                Since(11),
                "tuple[str, *tuple[int, bool]]",
                "tuple[str, int, bool]",
            ),
        ] {
            if !since.met(py) {
                continue;
            }
            let got = built(py, expression).unwrap_or_else(|error| {
                panic!("{expression} did not build: {error}");
            });
            assert_eq!(got, wanted, "{expression}");
        }
    });
}

/// The forms the frontend refuses, and the message each refusal carries.
///
/// A refusal is a decision about the algebra -- a construct that names no
/// set does not silently become one -- so the message is asserted, not only
/// the failure: a mutant that swaps two refusals leaves both refusing.
#[test]
fn each_refusal_says_what_it_refuses() {
    Python::attach(|py| {
        let refuses = |expression: &str, wanted: &str| {
            let error = match built(py, expression) {
                Err(error) => error.to_string(),
                Ok(schema) => panic!("{expression} built {schema} instead of refusing"),
            };
            assert!(
                error.contains(wanted),
                "{expression} refused with {error}, which does not name {wanted}"
            );
        };
        for (expression, wanted) in [
            ("typing.TypeVar('T')", "TypeVar"),
            ("list['Account']", "get_type_hints"),
            ("typing.Annotated[int, at.MinLen(1)]", "length"),
            ("[..., int]", "only as the last element"),
            ("typing.Annotated[int, Timezone()]", "does not check"),
            // A grouping that never bottoms out, and one nested past the bound:
            // following either to the end is a stack this library does not have.
            // A constraint put to a literal is put to the values of the kind
            // its constant belongs to. The bare-kind rows are refused already,
            // and a literal is a value *of* a kind, so the two spellings are
            // the same question: one row per kind a constant can have, because
            // each is a separate reading of the constant.
            (
                "typing.Annotated[typing.Literal[1], at.MinLen(1)]",
                "have no length",
            ),
            (
                "typing.Annotated[typing.Literal[True], at.MinLen(1)]",
                "have no length",
            ),
            (
                "typing.Annotated[typing.Literal[1.5], at.MinLen(1)]",
                "have no length",
            ),
            (
                "typing.Annotated[typing.Literal['a'], at.Ge(0)]",
                "have no order",
            ),
            (
                "typing.Annotated[typing.Literal[b'a'], at.Ge(0)]",
                "have no order",
            ),
            // A union of them is the same question asked of each member, which
            // is the fold the rewrite has to survive.
            (
                "typing.Annotated[typing.Literal[1, 2], at.MinLen(1)]",
                "have no length",
            ),
            // And a refinement of a refinement, which is the other fold: the
            // inner base is where the literal sits.
            (
                "typing.Annotated[typing.Annotated[typing.Literal[1], at.Ge(0)], at.MinLen(1)]",
                "have no length",
            ),
            // A name and the same name with a trailing `?` are one field
            // written twice, which asks the record to hold two disjoint types
            // under one key and to have it both required and absent.
            ("{'a': int, 'a?': str}", "declared twice"),
            ("typing.Annotated[int, nest(9)]", "nested too deeply"),
            ("typing.Annotated[int, endless()]", "nested too deeply"),
        ] {
            refuses(expression, wanted);
        }
        // The star inside a subscript is a syntax error before 3.11, so below
        // it the refusal a row reads is the *parser's* rather than this one's.
        // The release rides on the row, as it does in the table above.
        for (since, expression, wanted) in [(
            Since(11),
            "tuple[*list[int]]",
            "only a tuple can be unpacked",
        )] {
            if since.met(py) {
                refuses(expression, wanted);
            }
        }
    });
}

/// An alias that names itself is the fixpoint it writes, and one that does
/// not is the schema its value builds.
///
/// The `type` statement is 3.12 syntax, so the source is run rather than
/// written here, and the case is skipped where the interpreter this links
/// cannot parse it -- which is a skip rather than a silent pass because the
/// assertion below says which it was.
#[test]
fn a_self_naming_alias_ties_its_own_fixpoint() {
    Python::attach(|py| {
        if !Since(12).met(py) {
            return;
        }
        let namespace = PyDict::new(py);
        py.run(
            &CString::new(
                "type Json = int | list[Json]\n\
                 type Plain = int | str\n\
                 type Bad = int | Bad\n",
            )
            .expect("a source with no interior nul"),
            Some(&namespace),
            None,
        )
        .expect("the aliases define");
        let build = |name: &str| {
            let alias = namespace
                .get_item(name)
                .expect("the namespace answers")
                .expect("the alias is in it");
            let mut pool = Pool::default();
            let mut defs = Vec::new();
            build_schema(&alias, &mut pool, &mut defs).map(|schema| (schema, defs))
        };
        // The knot is tied: the alias becomes a definition and the body
        // names it.
        let (schema, defs) = build("Json").expect("a recursive alias builds");
        assert!(
            matches!(schema, Schema::Ref(_)),
            "{schema:?} is no fixpoint"
        );
        assert_eq!(defs.len(), 1, "one alias, one definition");
        // No knot, no definition: an ordinary alias is what its value builds.
        let (schema, defs) = build("Plain").expect("a plain alias builds");
        assert!(matches!(schema, Schema::Union(_)), "{schema:?}");
        assert!(defs.is_empty(), "nothing to define");
        // A self-reference outside a constructor denotes no set, and is
        // refused where it is written.
        let refusal = match build("Bad") {
            Err(refusal) => refusal.to_string(),
            Ok((schema, _)) => panic!("`type Bad = int | Bad` built {schema:?}"),
        };
        assert!(
            refusal.contains("not contractive"),
            "the refusal does not say why: {refusal}"
        );
    });
}

/// What a base's values can be asked, per constraint and per base.
///
/// The four `carries_*` predicates decide whether a marker narrows a base or
/// empties it, and the difference is a refusal at construction rather than a
/// schema that admits nothing. Asked directly: each is a table over the node
/// set, and a row through an annotation would exercise one cell of it.
#[test]
fn each_base_answers_the_constraints_its_values_can() {
    Python::attach(|py| {
        let seq = Schema::list(SeqShape::homogeneous(Schema::Int));
        let map = Schema::keyed_map(Vec::new(), vec![MapClause::top()]);
        // length: the shaped kinds have one, the scalars do not, and a class
        // does not say.
        for (base, answer) in [
            (Schema::Str, Carries::Yes),
            (Schema::Bytes, Carries::Yes),
            (seq.clone(), Carries::Yes),
            (Schema::set(Schema::Int), Carries::Yes),
            (Schema::frozen_set(Schema::Int), Carries::Yes),
            (map.clone(), Carries::Yes),
            (Schema::Int, Carries::No),
            (Schema::Bool, Carries::No),
            (Schema::Float, Carries::No),
            (Schema::NoneType, Carries::No),
            (Schema::ANY, Carries::Maybe),
        ] {
            assert!(carries_length(&base) == answer, "length of {base:?}");
        }
        // text for a pattern: `str` alone, and `bytes` explicitly not --
        // a pattern here is matched against text.
        for (base, answer) in [
            (Schema::Str, Carries::Yes),
            (Schema::Bytes, Carries::No),
            (Schema::Int, Carries::No),
            (seq.clone(), Carries::No),
            (map.clone(), Carries::No),
            (Schema::ANY, Carries::Maybe),
        ] {
            assert!(carries_pattern(&base) == answer, "pattern on {base:?}");
        }
        // a divisor: the numbers, `bool` among them since it subclasses int.
        for (base, answer) in [
            (Schema::Int, Carries::Yes),
            (Schema::Bool, Carries::Yes),
            (Schema::Float, Carries::Yes),
            (Schema::Str, Carries::No),
            (Schema::Bytes, Carries::No),
            (seq.clone(), Carries::No),
            (map.clone(), Carries::No),
            (Schema::ANY, Carries::Maybe),
        ] {
            assert!(carries_division(&base) == answer, "divisor on {base:?}");
        }
        // order: the operand's group has to be the base's, because Python
        // raises across the groups rather than ordering them. This is the one
        // check that reads *two* values, so the table is a product: every
        // orderable kind against an operand of its own group and against one of
        // another, and the two kinds that order against nothing.
        let five = 5i64.into_pyobject(py).expect("an int");
        let word = PyString::new(py, "a");
        let raw = PyBytes::new(py, b"a");
        let listed = PyList::new(py, [1i64]).expect("a list");
        let fixed = PyTuple::new(py, [1i64]).expect("a tuple");
        let members = PySet::new(py, [1i64]).expect("a set");
        let frozen = PyFrozenSet::new(py, [1i64]).expect("a frozen set");
        let tuple_seq = Schema::tuple(SeqShape::homogeneous(Schema::Int));
        let set = Schema::set(Schema::Int);
        let frozen_set = Schema::frozen_set(Schema::Int);
        for (base, operand, answer) in [
            // The numbers order with any number, `bool` among them.
            (Schema::Int, five.as_any(), Carries::Yes),
            (Schema::Float, five.as_any(), Carries::Yes),
            (Schema::Bool, five.as_any(), Carries::Yes),
            (Schema::Str, five.as_any(), Carries::No),
            (Schema::Bytes, five.as_any(), Carries::No),
            // Text with text, bytes with bytes, and neither with the other.
            (Schema::Str, word.as_any(), Carries::Yes),
            (Schema::Int, word.as_any(), Carries::No),
            (Schema::Bytes, raw.as_any(), Carries::Yes),
            (Schema::Str, raw.as_any(), Carries::No),
            (Schema::Bytes, word.as_any(), Carries::No),
            // A sequence orders against a sequence of its own container: a list
            // and a tuple are two kinds to Python's comparison as to this one.
            (seq.clone(), listed.as_any(), Carries::Yes),
            (seq.clone(), fixed.as_any(), Carries::No),
            (seq.clone(), five.as_any(), Carries::No),
            (tuple_seq.clone(), fixed.as_any(), Carries::Yes),
            (tuple_seq.clone(), listed.as_any(), Carries::No),
            // The two set kinds share one order, which is inclusion, so either
            // spelling of the operand answers for either base.
            (set.clone(), members.as_any(), Carries::Yes),
            (set.clone(), frozen.as_any(), Carries::Yes),
            (frozen_set.clone(), members.as_any(), Carries::Yes),
            (set.clone(), five.as_any(), Carries::No),
            // And the two that order against nothing at all: a dict has no
            // comparison, and `None` compares with no value including itself.
            (map.clone(), five.as_any(), Carries::No),
            (map.clone(), word.as_any(), Carries::No),
            (Schema::NoneType, five.as_any(), Carries::No),
            // A class says nothing: it may define the comparison itself.
            (Schema::ANY, five.as_any(), Carries::Maybe),
        ] {
            assert!(
                carries_order(&base, operand) == answer,
                "order of {base:?} against {operand}"
            );
        }
    });
}

/// A compound base answers for its parts, and the fold is a union's.
///
/// A member that can answer makes the constraint a narrowing of the union
/// rather than an emptying of it, which is why the fold is `or` and not
/// `and`. An intersection and a complement narrow a set this check does not
/// compute, so they stand aside; a refinement answers with its own base.
#[test]
fn a_compound_base_answers_for_its_parts() {
    let refined = Schema::Refine {
        base: Arc::new(Schema::Str),
        constraints: vec![Constraint::MinLen(1)].into(),
    };
    for (base, answer) in [
        (Schema::union([Schema::Str, Schema::Int]), Carries::Yes),
        (Schema::union([Schema::Int, Schema::Float]), Carries::No),
        (refined.clone(), Carries::Yes),
        (Schema::meet([Schema::Str, Schema::Int]), Carries::Maybe),
        (Schema::Str.complement(), Carries::Maybe),
    ] {
        assert!(carries_length(&base) == answer, "length of {base:?}");
    }
    // A plain base is not compound, and the fold says so by declining --
    // which is what sends the caller to the table above.
    assert!(carries_through(&Schema::Str, &carries_length).is_none());
}

/// A compiled pattern's flags are part of the pattern, and each is written
/// into it or refused by name.
///
/// Dropping one silently widens the set the marker was written for, so the
/// two directions are asserted together: the flags this engine spells arrive
/// inline, and the three it does not are refusals rather than approximations.
#[test]
fn a_patterns_flags_are_written_into_it_or_refused() {
    Python::attach(|py| {
        let re = py.import("re").expect("re imports");
        let compiled = |flags: &str| {
            let namespace = PyDict::new(py);
            namespace.set_item("re", &re).expect("a namespace holds it");
            py.eval(
                &CString::new(format!("re.compile('a', {flags})")).expect("no nul"),
                Some(&namespace),
                None,
            )
            .expect("the pattern compiles")
        };
        for (flags, inline) in [
            ("re.I", "(?i)"),
            ("re.M", "(?m)"),
            ("re.S", "(?s)"),
            ("re.X", "(?x)"),
            // Two flags are one group, in the order the engine spells them.
            ("re.I | re.M", "(?im)"),
        ] {
            let marker = compiled(flags);
            let probes = Probes::of(&marker).expect("the marker reads");
            let pattern = with_inline_flags(&marker, &probes, "a".to_owned())
                .unwrap_or_else(|error| panic!("{flags} refused: {error}"));
            assert!(
                pattern.contains(inline),
                "{flags} gave {pattern}, which does not carry {inline}"
            );
        }
        // The three refused flags are asked of a marker carrying the bit
        // rather than of a compiled pattern: `re` will not compile `re.L`
        // against a `str` at all, and the frontend reads `.flags` off
        // whatever carries it.
        let namespace = PyDict::new(py);
        namespace
            .set_item("types", py.import("types").expect("types imports"))
            .expect("a namespace holds it");
        let marked = |bit: u32| {
            py.eval(
                &CString::new(format!("types.SimpleNamespace(flags={bit})")).expect("no nul"),
                Some(&namespace),
                None,
            )
            .expect("the marker builds")
        };
        for (bit, name) in [(256u32, "re.ASCII"), (4, "re.LOCALE"), (128, "re.DEBUG")] {
            let carrier = marked(bit);
            let probes = Probes::of(&carrier).expect("the marker reads");
            let refusal = match with_inline_flags(&carrier, &probes, "a".to_owned()) {
                Err(refusal) => refusal.to_string(),
                Ok(pattern) => panic!("{name} was written into {pattern}"),
            };
            assert!(
                refusal.contains(name),
                "{name} refused without naming itself: {refusal}"
            );
        }
        // A marker with no flags at all is the pattern it carries.
        let bare = PyString::new(py, "a");
        assert_eq!(
            with_inline_flags(
                &bare,
                &Probes::of(&bare).expect("a str reads"),
                "a".to_owned()
            )
            .expect("no flags to read"),
            "a"
        );
    });
}

/// A constant pools by value where the type is exact, and by identity
/// everywhere else.
///
/// The two float cases are the ones a reader has to be told: a `nan` is
/// equal to nothing, so pooling two by value would make one constant of two
/// values that share no membership; and `-0.0` and `0.0` are equal, so their
/// keys must agree or one literal would build two nodes.
#[test]
fn a_constant_pools_by_value_only_where_equality_is_pythons() {
    Python::attach(|py| {
        let mut pool = Pool::default();
        let eval = |source: &str| {
            py.eval(&CString::new(source).expect("no nul"), None, None)
                .expect("the expression evaluates")
        };
        // Equal values, two objects: one slot.
        let first = pool.intern_const(&eval("1000000"));
        let again = pool.intern_const(&eval("10 ** 6"));
        assert_eq!(first, again, "two spellings of one int");
        // The signed zeros are one constant, because they are one value.
        let zero = pool.intern_const(&eval("0.0"));
        let minus = pool.intern_const(&eval("-0.0"));
        assert_eq!(zero, minus, "0.0 == -0.0");
        // Two nans are two slots: nothing is equal to a nan, so nothing is
        // pooled with one.
        let nan = pool.intern_const(&eval("float('nan')"));
        let other = pool.intern_const(&eval("float('nan')"));
        assert_ne!(nan, other, "a nan is not equal to a nan");
        // A `bool` is not the `int` it equals, because a literal is typed.
        let one = pool.intern_const(&eval("1"));
        let true_ = pool.intern_const(&eval("True"));
        assert_ne!(one, true_, "Literal[1] is not Literal[True]");
    });
}

/// A seeded pool carries the constants it was given, and yields them back.
#[test]
fn a_seeded_pool_holds_what_it_was_seeded_with() {
    Python::attach(|py| {
        let held = vec![PyString::new(py, "x").into_any().unbind()];
        let mut pool = Pool::seeded(py, held);
        assert_eq!(pool.items().len(), 1, "the seed is in the pool");
        // A constant equal to the seed joins its slot rather than taking a
        // new one, which is what seeding is for when two validators merge.
        assert_eq!(
            pool.intern_const(&PyString::new(py, "x").into_any()),
            ConstIx::new(0),
            "an equal constant joins the seeded slot"
        );
        assert_eq!(pool.into_items().len(), 1);
    });
}

/// A key schema that narrows its keys is refused, at any depth a connective
/// can hide the narrowing.
#[test]
fn a_narrowing_key_is_found_under_every_connective() {
    let narrowing = Schema::Refine {
        base: Arc::new(Schema::Str),
        constraints: vec![Constraint::MinLen(1)].into(),
    };
    for schema in [
        narrowing.clone(),
        Schema::union([Schema::Int, narrowing.clone()]),
        Schema::meet([Schema::Str, narrowing.clone()]),
        narrowing.clone().complement(),
        Schema::union([Schema::Int, narrowing.clone().complement()]),
    ] {
        assert!(narrows_its_keys(&schema), "{schema:?} narrows its keys");
    }
    // A refinement carrying no constraint narrows nothing, and neither does
    // a plain key type: both are keys a map may be written with.
    for schema in [
        Schema::Str,
        Schema::union([Schema::Str, Schema::Int]),
        Schema::Refine {
            base: Arc::new(Schema::Str),
            constraints: Vec::new().into(),
        },
    ] {
        assert!(!narrows_its_keys(&schema), "{schema:?} keys as it is");
    }
}

/// The frontend descends one level past the construction bound and no
/// further, so the schema *at* the bound builds and the one past it is
/// refused by name.
#[test]
fn the_build_descends_one_level_past_the_construction_bound() {
    Python::attach(|py| {
        let nested = |depth: usize| {
            let mut annotation = "int".to_owned();
            for _ in 0..depth {
                annotation = format!("list[{annotation}]");
            }
            annotation
        };
        // The boundary itself, from both sides: a chain reaching the bound
        // builds, and one level more is refused by name. Asserting the pair
        // is what pins the *number* -- either side alone holds only that
        // the guard exists somewhere.
        // Written from the *published* bound rather than from the
        // frontend's own, so the two are held apart: a change to either
        // constant alone moves this boundary and fails here.
        let deepest = crate::validator::MAX_SCHEMA_DEPTH;
        built(py, &nested(deepest)).expect("a chain at the bound builds");
        let refusal = match built(py, &nested(deepest + 1)) {
            Err(refusal) => refusal.to_string(),
            Ok(schema) => panic!("one level past the bound built {schema}"),
        };
        assert!(
            refusal.contains("too deep"),
            "the refusal does not name the depth: {refusal}"
        );
        // And the counter comes back down: a refusal leaves the depth where
        // it found it, so the next build starts from nought rather than
        // from wherever the last one stopped.
        built(py, &nested(deepest)).expect("the guard unwound");
    });
}

/// A class whose fields are declared builds the record beside the class,
/// and one whose are not is the class alone.
///
/// Asserted on the schema rather than on its render, because the render
/// prints a class by name either way: a dataclass read as a bare
/// `isinstance` and one read as a record of fields print the same string
/// and denote different sets.
#[test]
fn a_declared_field_becomes_an_attribute_beside_the_class() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the corpus namespace builds");
        py.run(
            &CString::new(
                "import dataclasses, typing\n\
                 @dataclasses.dataclass\n\
                 class Point:\n\
                 \x20   x: int\n\
                 class Pair(typing.NamedTuple):\n\
                 \x20   left: int\n\
                 \x20   right: str\n\
                 class Bare(tuple):\n\
                 \x20   pass\n\
                 @dataclasses.dataclass\n\
                 class Boxed(tuple):\n\
                 \x20   x: int\n\
                 @typing.runtime_checkable\n\
                 class Sized(typing.Protocol):\n\
                 \x20   def __len__(self) -> int: ...\n\
                 class Quiet(typing.Protocol):\n\
                 \x20   def __len__(self) -> int: ...\n",
            )
            .expect("a source with no interior nul"),
            Some(&namespace),
            None,
        )
        .expect("the corpus classes define");
        let build = |name: &str| {
            let annotation = namespace
                .get_item(name)
                .expect("the namespace answers")
                .expect("the class is in it");
            let mut pool = Pool::default();
            let mut defs = Vec::new();
            build_schema(&annotation, &mut pool, &mut defs)
        };
        let fields = |name: &str| match build(name) {
            Ok(Schema::Intersection(members)) => members
                .iter()
                .find_map(|member| match member {
                    Schema::AttrRecord { fields } => Some(
                        fields
                            .iter()
                            .map(|field| field.name.to_string())
                            .collect::<Vec<_>>(),
                    ),
                    _ => None,
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        // The positions a class lays out, where it lays any out.
        let positions = |name: &str| match build(name) {
            Ok(Schema::Intersection(members)) => members
                .iter()
                .find_map(|member| match member {
                    Schema::Seq { shape, .. } => Some(shape.prefix.to_vec()),
                    _ => None,
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        // A dataclass carries its fields as attributes. A named tuple lays the
        // same fields out as *positions* -- its instances are tuples, and the
        // two readings would describe the same values twice -- so it carries
        // them there and not beside them.
        assert_eq!(fields("Point"), vec!["x".to_owned()]);
        assert!(
            fields("Pair").is_empty(),
            "a named tuple's fields are its positions"
        );
        assert_eq!(positions("Pair"), vec![Schema::Int, Schema::Str]);
        assert!(
            positions("Point").is_empty(),
            "a dataclass lays out no positions"
        );
        assert!(fields("Bare").is_empty(), "a tuple subclass declares none");
        // Deriving from `tuple` is not laying out a tuple. A dataclass that
        // does carries fields the class never puts at a position -- its
        // instances are tuples of whatever they were built from -- so it keeps
        // its attribute record and lays out nothing. The two conditions that
        // separate it from a named tuple are both needed, and each alone admits
        // this class.
        assert_eq!(fields("Boxed"), vec!["x".to_owned()]);
        assert!(
            positions("Boxed").is_empty(),
            "deriving from tuple is not laying one out"
        );
        assert!(
            matches!(build("Bare"), Ok(Schema::Instance(_))),
            "a class with no declared field is the isinstance atom"
        );
        // A protocol is an isinstance check, and only where the class says
        // the check is allowed.
        assert!(matches!(build("Sized"), Ok(Schema::Instance(_))));
        let refusal = match build("Quiet") {
            Err(refusal) => refusal.to_string(),
            Ok(schema) => panic!("a plain Protocol built {schema:?}"),
        };
        assert!(
            refusal.contains("runtime_checkable"),
            "the refusal does not say what is missing: {refusal}"
        );
    });
}

/// A `TypedDict` says which keys it admits beyond the ones it declares, and
/// the two ways it says so are read.
#[test]
fn a_typed_dict_says_which_other_keys_it_admits() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the corpus namespace builds");
        py.run(
            &CString::new(
                "import typing\n\
                 class Open(typing.TypedDict):\n\
                 \x20   name: str\n\
                 class Closed(typing.TypedDict):\n\
                 \x20   name: str\n\
                 Closed.__closed__ = True\n\
                 class Extra(typing.TypedDict):\n\
                 \x20   name: str\n\
                 Extra.__extra_items__ = int\n",
            )
            .expect("a source with no interior nul"),
            Some(&namespace),
            None,
        )
        .expect("the corpus classes define");
        for (name, wanted) in [
            // Open is the spec's default, and open over the string keys:
            // a `TypedDict` relates to `Mapping[str, object]` and nothing
            // wider.
            ("Open", "{'name': str, str: anything}"),
            // `closed=True` (PEP 728) leaves the declared keys alone.
            ("Closed", "{'name': str}"),
            // `extra_items` types the rest rather than admitting anything.
            ("Extra", "{'name': str, str: int}"),
        ] {
            let annotation = namespace
                .get_item(name)
                .expect("the namespace answers")
                .expect("the class is in it");
            let mut pool = Pool::default();
            let mut defs = Vec::new();
            let schema = build_schema(&annotation, &mut pool, &mut defs)
                .unwrap_or_else(|error| panic!("{name} did not build: {error}"));
            let active = RefCell::new(FxHashMap::default());
            assert_eq!(
                render(py, &schema, pool.items(), &defs, &active, 0),
                wanted,
                "{name}"
            );
        }
    });
}

/// A class is read through its declared fields, and what it declares is not
/// every annotation on it.
#[test]
fn a_class_is_read_through_what_it_declares() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the corpus namespace builds");
        // `NotRequired` reaches `typing` in 3.11, so the key it marks is
        // declared where the interpreter has the marker to declare it with --
        // the class is written the way a caller writes it, rather than built
        // and then patched. Below 3.11 the record is the required half alone,
        // which is the record that release gives a caller for this class.
        let optional = Since(11).met(py);
        let note = if optional {
            "\x20   note: typing.NotRequired[int]\n"
        } else {
            ""
        };
        py.run(
            &CString::new(format!(
                "import dataclasses, typing\n\
                 @dataclasses.dataclass\n\
                 class Point:\n\
                 \x20   x: int\n\
                 \x20   tag: typing.ClassVar[str] = 'p'\n\
                 class Row(typing.TypedDict):\n\
                 \x20   name: str\n\
                 {note}"
            ))
            .expect("a source with no interior nul"),
            Some(&namespace),
            None,
        )
        .expect("the corpus classes define");
        for (name, wanted) in [
            // The `ClassVar` annotates the class rather than an instance, so
            // it is not a field of the record the instances hold.
            ("Point", "Point"),
            // A `TypedDict` is a record, open as the typing spec defines
            // one, and `NotRequired` marks the key rather than its type.
            (
                "Row",
                if optional {
                    "{'name': str, 'note?': int, str: anything}"
                } else {
                    "{'name': str, str: anything}"
                },
            ),
        ] {
            let annotation = namespace
                .get_item(name)
                .expect("the namespace answers")
                .expect("the class is in it");
            let mut pool = Pool::default();
            let mut defs = Vec::new();
            let schema = build_schema(&annotation, &mut pool, &mut defs)
                .unwrap_or_else(|error| panic!("{name} did not build: {error}"));
            let active = RefCell::new(FxHashMap::default());
            assert_eq!(
                render(py, &schema, pool.items(), &defs, &active, 0),
                wanted,
                "{name}"
            );
        }
    });
}
/// A qualifier states required-ness only from the outside of the hint.
///
/// `qualified_required` walks a hint outward-in, unwrapping the qualifiers that
/// carry no answer of their own -- `ReadOnly[NotRequired[T]]` is legal -- and
/// stopping at anything else. The stop is what this pins. A form that is *not*
/// a field qualifier ends the search whatever it wraps, so `list[Required[int]]`
/// says nothing about its key: the `Required` inside it qualifies the list's
/// element position, where required-ness has no meaning, and reading it as the
/// field's would make a key required because of the shape of its value.
///
/// Written against the walk rather than through a `TypedDict`, because the
/// distinction is one step of the walk and a class would reach it only if the
/// spec allowed the spelling. The unqualified rows are the other direction: a
/// hint that never reaches a qualifier answers `None` by running out, not by
/// stopping early, and a search that stopped at the wrong sign would swap them.
#[test]
fn a_qualifier_states_required_ness_only_from_the_outside() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the namespace builds");
        let answer = |expression: &str| {
            let hint = py
                .eval(
                    &CString::new(expression).expect("no interior nul"),
                    Some(&namespace),
                    None,
                )
                .unwrap_or_else(|error| panic!("{expression} does not evaluate: {error}"));
            qualified_required(&hint).expect("the walk answers")
        };
        for (expression, wanted) in [
            // Not stated: nothing here is a field qualifier.
            ("int", None),
            ("list[int]", None),
            ("typing.Annotated[int, 1]", None),
        ] {
            assert_eq!(answer(expression), wanted, "{expression}");
        }
        // Every other row names a marker `typing` gained in 3.11, so each
        // carries the release it needs: below it the name is an
        // `AttributeError` and the row would be about the interpreter rather
        // than about the walk. The rows themselves are what every lane above
        // the floor drives.
        for (since, expression, wanted) in [
            // Stated, and read.
            (Since(11), "typing.Required[int]", Some(true)),
            (Since(11), "typing.NotRequired[int]", Some(false)),
            // Stated behind a qualifier that carries no answer of its own.
            (
                Since(11),
                "typing.Required[typing.Annotated[int, 1]]",
                Some(true),
            ),
            // The one the walk must not read through: `list` is not a field
            // qualifier, so the search ends at it and never sees the
            // `Required` it holds.
            (Since(11), "list[typing.Required[int]]", None),
            (Since(11), "dict[str, typing.NotRequired[int]]", None),
        ] {
            if since.met(py) {
                assert_eq!(answer(expression), wanted, "{expression}");
            }
        }
    });
}
