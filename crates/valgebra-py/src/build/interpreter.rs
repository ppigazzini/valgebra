use std::sync::Arc;

use super::classes::*;
use super::refine::*;
use super::*;
use crate::render::render;
use pyo3::exceptions::PyKeyboardInterrupt;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::ffi::CString;
use valgebra_core::{
    Carries, Constraint, MapClause, SeqShape, carries_division, carries_length, carries_pattern,
    carries_through,
};

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

/// The namespace a row's expression is evaluated in.
///
/// `at` holds the marker doubles, named for the vocabulary they stand in
/// for so a row reads as the line a caller would write. Each double carries
/// the vocabulary's module, because a constraint is read off that vocabulary
/// and no other; the embedded interpreter starts on the base prefix, where
/// `annotated_types` may not be installed.
fn namespace(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let namespace = PyDict::new(py);
    for module in ["typing", "dataclasses", "enum", "re", "types", "warnings"] {
        namespace.set_item(module, py.import(module)?)?;
    }
    py.run(
        &CString::new(
            "import types\n\
             class namespace(types.SimpleNamespace):\n\
             \x20   pass\n\
             class at:\n\
             \x20   Ge = staticmethod(lambda n: namespace(ge=n))\n\
             \x20   Le = staticmethod(lambda n: namespace(le=n))\n\
             \x20   MinLen = staticmethod(lambda n: namespace(min_length=n))\n\
             \x20   MultipleOf = staticmethod(lambda n: namespace(multiple_of=n))\n\
             \x20   Predicate = staticmethod(lambda f: namespace(func=f))\n\
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
             class Unit:\n\
             \x20   def __init__(self, unit): self.unit = unit\n\
             for vocabulary in (namespace, slotted.Ge, slotted.MinLen, guarded, hooked, Timezone, Unit):\n\
             \x20   vocabulary.__module__ = 'annotated_types'\n\
             class deprecated:\n\
             \x20   def __init__(self, message): self.message = message\n\
             \x20   def __call__(self, value): raise TypeError(value)\n\
             class py_deprecated(deprecated):\n\
             \x20   pass\n\
             deprecated.__module__ = 'warnings'\n\
             py_deprecated.__module__ = '_py_warnings'\n\
             class Kilograms:\n\
             \x20   symbol = 'kg'\n\
             class grouped:\n\
             \x20   __is_annotated_types_grouped_metadata__ = True\n\
             \x20   def __init__(self, *items): self.items = items\n\
             \x20   def __iter__(self): return iter(self.items)\n\
             class endless:\n\
             \x20   __is_annotated_types_grouped_metadata__ = True\n\
             \x20   def __iter__(self): return iter([self])\n\
             class Floor(namespace):\n\
             \x20   pass\n\
             class Tagged(Timezone):\n\
             \x20   pass\n\
             Floor.__module__ = Tagged.__module__ = 'elsewhere'\n\
             class carrying(grouped):\n\
             \x20   ge = 5\n\
             carrying.__module__ = 'annotated_types'\n\
             class flagged_off(namespace):\n\
             \x20   __is_annotated_types_grouped_metadata__ = False\n\
             class hooked_group:\n\
             \x20   __slots__ = ('items',)\n\
             \x20   __is_annotated_types_grouped_metadata__ = True\n\
             \x20   def __init__(self, *items): self.items = items\n\
             \x20   def __iter__(self): return iter(self.items)\n\
             \x20   def __getattr__(self, name): raise AttributeError(name)\n\
             class flagged_here:\n\
             \x20   def __init__(self, *items):\n\
             \x20       self.items = items\n\
             \x20       self.__is_annotated_types_grouped_metadata__ = True\n\
             \x20   def __iter__(self): return iter(self.items)\n\
             def nest(depth):\n\
             \x20   marker = at.Ge(0)\n\
             \x20   for _ in range(depth): marker = grouped(marker)\n\
             \x20   return marker\n\
             class loud:\n\
             \x20   def index(raised):\n\
             \x20       class Index:\n\
             \x20           def __index__(self): raise raised\n\
             \x20       return Index()\n\
             \x20   def repr(raised):\n\
             \x20       class Repr:\n\
             \x20           def __repr__(self): raise raised\n\
             \x20       return Repr()\n\
             \x20   def kind(raised):\n\
             \x20       class Meta(type):\n\
             \x20           def __repr__(cls): raise raised\n\
             \x20       return Meta('Kind', (), {})\n",
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
    render(py, &schema, pool.items(), &defs, &active, 0)
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
            // And a class is such metadata, where it carries no name a
            // constraint is read through: a unit, a tag, an enumeration.
            ("typing.Annotated[float, Kilograms]", "float"),
            (
                "typing.Annotated[int, at.MultipleOf(3)]",
                "Annotated[int, MultipleOf(3)]",
            ),
            // A refinement carries its markers on the base it narrows, and
            // a nested one folds onto that base rather than nesting.
            ("typing.Annotated[int, at.Ge(0)]", "Annotated[int, Ge(0)]"),
            // The shapes a marker keeps its names in, which decide where the
            // frontend reads them from. `at` above keeps them in an instance
            // dictionary; the vocabulary this stands in for ships
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
            // Both at once: the name is on the type *and* the type has a hook.
            // Its mask carries the vocabulary's bits beside the name's and the
            // hook's, so the mask that is exactly those two is a grouped
            // marker's, `hooked_group` in the vocabulary test.
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
/// An object carrying `__metadata__` and no `__origin__` is not an `Annotated`
/// form. It is read as every other object is, the literal of itself, rather
/// than failing on the name it lacks; with both names it is the refinement.
#[test]
fn metadata_without_an_origin_is_not_annotated() {
    Python::attach(|py| {
        let alone =
            built(py, "types.SimpleNamespace(__metadata__=(at.Ge(0),))").expect("an object builds");
        assert!(alone.starts_with("Literal["), "{alone}");
        let both = built(
            py,
            "types.SimpleNamespace(__metadata__=(at.Ge(0),), __origin__=int)",
        )
        .expect("a form builds");
        assert_eq!(both, "Annotated[int, Ge(0)]");
    });
}

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

/// A constraint is read off the vocabulary's markers and no other's: a name is
/// no protocol, and another library's `ge`, `pattern` or `func` means what that
/// library means by it.
#[test]
fn a_constraint_is_read_off_its_vocabulary_alone() {
    Python::attach(|py| {
        for (expression, wanted) in [
            ("typing.Annotated[int, types.SimpleNamespace(ge=0)]", "int"),
            (
                "typing.Annotated[str, types.SimpleNamespace(pattern='a')]",
                "str",
            ),
            (
                "typing.Annotated[int, types.SimpleNamespace(func=bool)]",
                "int",
            ),
            // A class deriving from the vocabulary is read as the class it
            // derives from, and one carrying nothing the frontend reads is
            // ignored rather than refused, instance or class: only the
            // vocabulary's own classes were written to narrow a schema.
            (
                "typing.Annotated[int, Floor(ge=3)]",
                "Annotated[int, Ge(3)]",
            ),
            ("typing.Annotated[int, Tagged()]", "int"),
            ("typing.Annotated[int, Tagged]", "int"),
            // The vocabulary's predicate carries its callable, and a pattern
            // is read off a compiled one.
            (
                "typing.Annotated[int, at.Predicate(bool)]",
                "Annotated[int, Predicate(...)]",
            ),
            (
                "typing.Annotated[str, re.compile('a')]",
                "Annotated[str, Regex('a')]",
            ),
            // A group is unpacked before anything is read off it: what it
            // carries as attributes is its members' to say, and read off the
            // group as well it is said twice, or differently.
            (
                "typing.Annotated[int, carrying(at.Ge(0))]",
                "Annotated[int, Ge(0)]",
            ),
            // The flag says it, true or false, wherever the marker keeps it.
            (
                "typing.Annotated[int, flagged_off(ge=2)]",
                "Annotated[int, Ge(2)]",
            ),
            (
                "typing.Annotated[int, flagged_here(at.Ge(0))]",
                "Annotated[int, Ge(0)]",
            ),
            // A flag on the type behind a hook, and nothing else the type
            // carries: the one marker whose mask is exactly the probe's bit
            // beside the hook's, now that a marker of the vocabulary carries
            // its class's bits beside them too.
            (
                "typing.Annotated[int, hooked_group(at.Ge(0))]",
                "Annotated[int, Ge(0)]",
            ),
        ] {
            let got = built(py, expression).unwrap_or_else(|error| {
                panic!("{expression} did not build: {error}");
            });
            assert_eq!(got, wanted, "{expression}");
        }
    });
}

/// A typing form in `Annotated` metadata is not a predicate: every one is
/// callable, and calling one builds a value rather than answering whether the
/// value belongs. An `Annotated` alias carrying a marker this frontend reads is
/// refused, since it was written to narrow; any other typing form is ignored.
#[test]
fn a_typing_form_in_metadata_is_not_a_predicate() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the namespace builds");
        py.run(
            c"class TxForm:\n\
              \x20   def __call__(self, value): return value\n\
              TxForm.__module__ = 'typing_extensions'\n\
              class Call:\n\
              \x20   def __call__(self, value): return value\n\
              class DocInfo:\n\
              \x20   pass\n\
              DocInfo.__module__ = 'annotated_types'\n",
            Some(&namespace),
            None,
        )
        .expect("the stand-ins compile");
        let int = py.get_type::<pyo3::types::PyInt>().into_any();
        let validator = Bound::new(py, crate::complement(&int).expect("the complement builds"))
            .expect("it binds");
        namespace.set_item("v", validator).expect("bound");
        let built = |expression: &str| -> PyResult<String> {
            let annotation = py.eval(&CString::new(expression)?, Some(&namespace), None)?;
            let mut pool = Pool::default();
            let mut defs = Vec::new();
            let schema = build_schema(&annotation, &mut pool, &mut defs)?;
            render(py, &schema, pool.items(), &defs, &RefCell::default(), 0)
        };
        for (expression, wanted) in [
            // A `types.GenericAlias`, a form whose class is `typing`'s or
            // `typing_extensions`', and a union, which is not callable at all.
            ("typing.Annotated[str, list[int]]", "str"),
            ("typing.Annotated[int, int | str]", "int"),
            ("typing.Annotated[str, typing.List[int]]", "str"),
            ("typing.Annotated[int, typing.NewType('U', int)]", "int"),
            ("typing.Annotated[int, TxForm()]", "int"),
            // An alias carrying only what this frontend ignores: a string, a
            // class from elsewhere, the documentation marker, a typing form.
            ("typing.Annotated[int, typing.Annotated[int, 'doc']]", "int"),
            (
                "typing.Annotated[int, typing.Annotated[int, Tagged]]",
                "int",
            ),
            (
                "typing.Annotated[int, typing.Annotated[int, DocInfo()]]",
                "int",
            ),
            (
                "typing.Annotated[int, typing.Annotated[int, list[int]]]",
                "int",
            ),
            // Every other callable is still a predicate.
            (
                "typing.Annotated[int, abs]",
                "Annotated[int, Predicate(...)]",
            ),
            (
                "typing.Annotated[int, Call()]",
                "Annotated[int, Predicate(...)]",
            ),
        ] {
            let got = built(expression).unwrap_or_else(|error| {
                panic!("{expression} did not build: {error}");
            });
            assert_eq!(got, wanted, "{expression}");
        }
        // An alias carrying a validator, a marker of the vocabulary, one
        // derived from it, a pattern, a marker class, or a predicate.
        for expression in [
            "typing.Annotated[int, typing.Annotated[int, v]]",
            "typing.Annotated[int, typing.Annotated[int, at.Ge(1)]]",
            "typing.Annotated[int, typing.Annotated[int, Floor(ge=3)]]",
            "typing.Annotated[str, typing.Annotated[str, re.compile('a')]]",
            "typing.Annotated[int, typing.Annotated[int, Timezone]]",
            "typing.Annotated[int, typing.Annotated[int, abs]]",
        ] {
            let error = match built(expression) {
                Err(error) => error.to_string(),
                Ok(schema) => panic!("{expression} built {schema} instead of refusing"),
            };
            assert!(
                error.contains("is an Annotated alias") && error.contains("annotates nothing"),
                "{expression}: {error}"
            );
        }
    });
}

/// Metadata that excludes no value is ignored wherever it is written: the
/// standard library's `deprecated`, whose class `warnings` defines from 3.13 and
/// `_py_warnings` from 3.14 and whose call raises for a value, and the
/// vocabulary's `Unit`, which names what a number is measured in. Read as
/// narrowing, the first would be a predicate refusing every value and the second
/// a constraint refused for what it would admit; and a group yielding `Unit`
/// beside a bound is the bound. A vocabulary marker that does narrow is still
/// refused where this frontend does not check it.
#[test]
fn metadata_that_excludes_no_value_is_ignored() {
    Python::attach(|py| {
        let reads = |expression: &str, wanted: &str| {
            let got = built(py, expression).unwrap_or_else(|error| {
                panic!("{expression} did not build: {error}");
            });
            assert_eq!(got, wanted, "{expression}");
        };
        let bound = built(py, "typing.Annotated[int, at.Ge(0)]").expect("a bound builds");
        for (expression, wanted) in [
            ("typing.Annotated[int, deprecated('x')]", "int"),
            ("typing.Annotated[int, py_deprecated('x')]", "int"),
            (
                "typing.Annotated[int, typing.Annotated[int, deprecated('x')]]",
                "int",
            ),
            ("typing.Annotated[float, Unit('m')]", "float"),
            ("typing.Annotated[float, Unit]", "float"),
            (
                "typing.Annotated[float, typing.Annotated[float, Unit('m')]]",
                "float",
            ),
            (
                "typing.Annotated[int, grouped(Unit('m'), at.Ge(0))]",
                bound.as_str(),
            ),
            ("typing.Annotated[int, Unit('m'), at.Ge(0)]", bound.as_str()),
        ] {
            reads(expression, wanted);
        }
        if Since(13).met(py) {
            reads("typing.Annotated[int, warnings.deprecated('x')]", "int");
        }
        let refusal = built(py, "typing.Annotated[int, Timezone()]")
            .expect_err("a marker that narrows is refused");
        assert!(refusal.to_string().contains("does not check"), "{refusal}");
    });
}

/// A callable that is no class is refused wherever a schema is read -- the top,
/// an element, a field, a type argument, a union member -- and a `Literal`
/// names it as the one object it is. The callables the dispatch refuses for
/// what they are keep that refusal inside a `Literal` too: a special form, a
/// class factory, and from 3.13 `Annotated` written bare.
#[test]
fn a_callable_is_no_schema_and_a_literal_names_it() {
    Python::attach(|py| {
        let refused = |expression: &str, wanted: &str| {
            let error = match built(py, expression) {
                Err(error) => error.to_string(),
                Ok(schema) => panic!("{expression} built {schema} instead of refusing"),
            };
            assert!(error.contains(wanted), "{expression} refused with {error}");
        };
        for expression in [
            "abs",
            "[abs]",
            "{'a': abs}",
            "list[abs]",
            "typing.Union[int, abs]",
        ] {
            refused(expression, "a callable is not a schema");
        }
        assert_eq!(
            built(py, "typing.Literal[abs]").expect("a callable constant builds"),
            "Literal[<built-in function abs>]"
        );
        refused("typing.Literal[typing.ClassVar]", "typing construct");
        refused(
            "typing.Literal[typing.TypedDict]",
            "the base a class is declared from",
        );
        if Since(13).met(py) {
            refused("typing.Literal[typing.Annotated]", "annotates nothing");
        }
    });
}

/// A typing form is refused where a constant or a type argument is read, and
/// so is a form that names no set at all: each was built quietly as something
/// else.
#[test]
fn a_typing_form_is_refused_where_a_constant_or_a_type_argument_is_read() {
    Python::attach(|py| {
        let refuses = |expression: &str, wanted: &str| {
            let error = match built(py, expression) {
                Err(error) => error.to_string(),
                Ok(schema) => panic!("{expression} built {schema} instead of refusing"),
            };
            assert!(error.contains(wanted), "{expression} refused with {error}");
        };
        for (expression, wanted) in [
            ("typing.Annotated", "annotates nothing"),
            ("dataclasses.InitVar[int]", "constructor parameter"),
            // A supertype is a type argument, where a string is a reference.
            ("typing.NewType('N', 'int')", "forward reference"),
            // A typing form is no constant, whichever reading would build it.
            ("typing.Literal[list[int]]", "rather than a constant"),
            ("typing.Literal[typing.Any]", "rather than a constant"),
            ("typing.Literal[typing.NoReturn]", "rather than a constant"),
            (
                "typing.Literal[typing.NewType('U', int)]",
                "rather than a constant",
            ),
        ] {
            refuses(expression, wanted);
        }
        for (since, expression, wanted) in [
            (
                Since(11),
                "typing.Literal[typing.Never]",
                "rather than a constant",
            ),
            (
                Since(12),
                "typing.Literal[typing.TypeAliasType('A', int)]",
                "rather than a constant",
            ),
            // An alias's value is a type argument, as a supertype is.
            (
                Since(12),
                "typing.TypeAliasType('A', 'int')",
                "forward reference",
            ),
        ] {
            if since.met(py) {
                refuses(expression, wanted);
            }
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
            ("typing.TypedDict", "the base a class is declared from"),
            // The fallback names every subscripted form the dispatch reads.
            (
                "typing.Mapping[str, int]",
                "list, set, frozenset, dict, tuple",
            ),
            ("typing.NamedTuple", "the base a class is declared from"),
            ("typing.Protocol", "the base a protocol is declared from"),
            (
                "types.new_class('Box', (typing.Protocol[typing.TypeVar('T')],))",
                "generic Protocol",
            ),
            (
                "types.new_class('Shared', (typing.Protocol,), exec_body=lambda ns: \
                 ns.update(__annotations__={'count': typing.ClassVar[int]}))",
                "declared ClassVar or Final",
            ),
            ("list['Account']", "get_type_hints"),
            ("typing.Annotated[int, at.MinLen(1)]", "length"),
            ("[..., int]", "only as the last element"),
            ("typing.Annotated[int, Timezone()]", "does not check"),
            // A marker written without its parentheses: the class of one the
            // frontend reads, and the class of one it refuses. Both would widen
            // the schema to its base if they were ignored.
            ("typing.Annotated[int, slotted.Ge]", "write Ge(...)"),
            ("typing.Annotated[int, Timezone]", "write Timezone(...)"),
            ("typing.Annotated[str, re.Pattern]", "write Pattern(...)"),
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
        // it the refusal the row reads is the *parser's* rather than this one's.
        if Since(11).met(py) {
            refuses("tuple[*list[int]]", "only a tuple can be unpacked");
        }
    });
}

/// Building asks a length bound for its `__index__`, and a refusal asks what it
/// refuses for its `__repr__`. An ordinary exception from either is the refusal
/// the bound or the object earns; a fatal signal is the interpreter unwinding,
/// and comes back out of the build in its place.
#[test]
fn a_fatal_signal_while_a_refusal_is_written_propagates() {
    Python::attach(|py| {
        for (expression, fatal) in [
            (
                "typing.Annotated[list, at.MinLen(loud.index(KeyboardInterrupt))]",
                true,
            ),
            (
                "typing.Annotated[list, at.MinLen(loud.index(ValueError))]",
                false,
            ),
            (
                "typing.Annotated[list, at.MinLen(loud.repr(KeyboardInterrupt))]",
                true,
            ),
            (
                "typing.Annotated[list, at.MinLen(loud.repr(ValueError))]",
                false,
            ),
            ("typing.Literal[loud.kind(KeyboardInterrupt)]", true),
            ("typing.Literal[loud.kind(ValueError)]", false),
        ] {
            let error = match built(py, expression) {
                Err(error) => error,
                Ok(schema) => panic!("{expression} built {schema} instead of refusing"),
            };
            assert_eq!(
                error.is_instance_of::<PyKeyboardInterrupt>(py),
                fatal,
                "{expression} raised {error}"
            );
            if !fatal && expression.contains("repr") {
                assert!(error.to_string().contains("<unrepresentable>"), "{error}");
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

/// Define the aliases `source` writes and read each expression as a schema.
///
/// The `type` statement is 3.12 syntax, so the source is run rather than
/// written here, and the caller skips where the interpreter cannot parse it.
fn alias_reader<'py>(
    py: Python<'py>,
    source: &str,
) -> impl Fn(&str) -> PyResult<(Schema, Vec<Schema>)> + 'py {
    let namespace = PyDict::new(py);
    py.run(
        &CString::new(source).expect("a source with no interior nul"),
        Some(&namespace),
        None,
    )
    .expect("the aliases define");
    move |expression: &str| {
        let form = py.eval(&CString::new(expression)?, Some(&namespace), None)?;
        let mut pool = Pool::default();
        let mut defs = Vec::new();
        build_schema(&form, &mut pool, &mut defs).map(|schema| (schema, defs))
    }
}

/// A generic alias applied to its arguments is its body with them substituted.
///
/// Through the body's own parameter order, so a body naming its parameters in
/// another order than the alias declares them is not transposed; a body naming
/// a generic class bare is left the class it names; and the count the runtime
/// never checks is checked here, each refusal naming the alias.
#[test]
fn a_generic_alias_is_its_body_with_the_arguments_substituted() {
    Python::attach(|py| {
        if !Since(12).met(py) {
            return;
        }
        let read = alias_reader(
            py,
            "class Box[T]:\n\
             \x20   pass\n\
             type Pair[T] = tuple[T, T]\n\
             type Swap[T, U] = dict[U, T]\n\
             type Two[T, U] = dict[T, U]\n\
             type Boxes[T] = list[Box]\n\
             type Spread[*Ts] = tuple[*Ts]\n",
        );
        let schema = |expression: &str| read(expression).expect(expression).0;
        let refusal = |expression: &str| match read(expression) {
            Err(refusal) => refusal.to_string(),
            Ok((schema, _)) => panic!("{expression} built {schema:?}"),
        };
        assert_eq!(schema("Pair[int]"), schema("tuple[int, int]"));
        assert_eq!(schema("Swap[str, int]"), schema("dict[int, str]"));
        assert_eq!(schema("list[Pair[str]]"), schema("list[tuple[str, str]]"));
        // `Box` names a parameter of its own, which no argument here stands for.
        assert_eq!(schema("Boxes[int]"), schema("list[Box]"));
        assert!(
            refusal("Pair[int, str]")
                .ends_with("Pair takes 1 type argument, and Pair[int, str] gives it 2"),
            "{}",
            refusal("Pair[int, str]")
        );
        assert!(
            refusal("Two[int]").ends_with(
                "Two takes 2 type arguments, and Two[int] gives it 1: U has no default \
                 to stand in"
            ),
            "{}",
            refusal("Two[int]")
        );
        assert!(
            refusal("Pair").ends_with(
                "Pair is a generic alias, and its parameter T has no default: write \
                 Pair[...] with the type it stands for"
            ),
            "{}",
            refusal("Pair")
        );
        let spread = refusal("Spread[int]");
        assert!(spread.contains("Spread declares Ts"), "{spread}");
    });
}

/// A recursive generic alias ties one fixpoint per argument list it meets.
///
/// `Tree[int]` is a fresh object at each read, and the body it substitutes to
/// names an equal one, which is the back edge; an alias alternating its
/// arguments meets the first list again one unfolding on, and one applied to
/// an argument naming none of its parameters settles on that list. One whose
/// argument nests its own parameter meets a new list at every unfolding and is
/// refused before it is built.
#[test]
fn a_recursive_generic_alias_is_tied_by_its_arguments() {
    Python::attach(|py| {
        if !Since(12).met(py) {
            return;
        }
        let read = alias_reader(
            py,
            "type Tree[T] = T | list[Tree[T]]\n\
             type Swapping[T, U] = None | dict[T, Swapping[U, T]]\n\
             type Settles[T] = T | list[Settles[int]]\n\
             type Nest[T] = T | list[Nest[list[T]]]\n",
        );
        let (tree, defs) = read("Tree[int]").expect("a regular recursive alias builds");
        assert!(matches!(tree, Schema::Ref(_)), "{tree:?} is no fixpoint");
        assert_eq!(defs.len(), 1, "one argument list, one definition");
        assert_eq!(read("Tree[int]").expect("again").0, tree);
        let (swapping, defs) = read("Swapping[int, str]").expect("alternating builds");
        assert!(matches!(swapping, Schema::Ref(_)), "{swapping:?}");
        assert_eq!(defs.len(), 1, "{defs:?}");
        // An argument naming no parameter of the alias settles on one list a
        // step on, and is tied there.
        let (settles, defs) = read("Settles[str]").expect("a concrete argument builds");
        assert!(matches!(settles, Schema::Union(_)), "{settles:?}");
        assert_eq!(defs.len(), 1, "{defs:?}");
        let refusal = match read("Nest[int]") {
            Err(refusal) => refusal.to_string(),
            Ok((schema, _)) => panic!("Nest[int] built {schema:?}"),
        };
        assert!(
            refusal.contains("Nest applies itself to Nest[list[T]]"),
            "{refusal}"
        );
    });
}

/// A parameter missing an argument takes its default, which may name an
/// earlier parameter, and an alias whose every parameter has one reads bare.
#[test]
fn a_default_stands_in_for_a_missing_argument() {
    Python::attach(|py| {
        // `typing_extensions` gives a type variable a default on 3.12, where
        // the `type` statement has no syntax for one.
        if !Since(12).met(py) || py.import("typing_extensions").is_err() {
            return;
        }
        let read = alias_reader(
            py,
            "import typing, typing_extensions\n\
             T = typing_extensions.TypeVar('T', default=int)\n\
             U = typing_extensions.TypeVar('U', default=T)\n\
             Same = typing.TypeAliasType('Same', dict[T, U], type_params=(T, U))\n",
        );
        let schema = |expression: &str| read(expression).expect(expression).0;
        assert_eq!(schema("Same"), schema("dict[int, int]"));
        assert_eq!(schema("Same[str]"), schema("dict[str, str]"));
        assert_eq!(schema("Same[str, bytes]"), schema("dict[str, bytes]"));
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
                carries_order(&base, operand).expect("the group reads") == answer,
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

/// An exact builtin scalar is read as its constant before the dispatch, and
/// only an exact one: a subclass -- an `IntEnum` member, a `str` subclass -- a
/// container and `None` are left to the arms that read them, and a validator
/// is composed rather than read as a constant.
#[test]
fn an_exact_builtin_scalar_is_its_own_constant() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"import enum\n\
              class Color(enum.IntEnum):\n\
              \x20   RED = 1\n\
              class Text(str):\n\
              \x20   pass\n\
              SCALARS = [7, 2 ** 70, 'x', True, 1.5, float('nan'), b'y']\n\
              OTHERS = [None, Color.RED, Text('x'), [1], (1,), {'a': 1}, int]\n",
            c"scalars.py",
            c"scalars",
        )
        .expect("the module compiles");
        let listed = |name: &str| {
            module
                .getattr(name)
                .and_then(|list| list.try_iter()?.collect::<PyResult<Vec<_>>>())
                .expect("a list")
        };
        let mut pool = Pool::default();
        for scalar in listed("SCALARS") {
            let read = builtin_constant(&scalar, &mut pool);
            assert!(
                matches!(read, Some(Schema::Literal(_))),
                "{scalar} is its own constant, not {read:?}"
            );
        }
        for other in listed("OTHERS") {
            assert!(
                builtin_constant(&other, &mut pool).is_none(),
                "{other} takes the walk"
            );
        }
    });
}

/// A pool asked to make room for constants has it before the first arrives.
#[test]
fn a_pool_makes_room_for_what_it_is_told_is_coming() {
    let mut pool = Pool::default();
    pool.reserve(100);
    assert!(pool.items.capacity() >= 100);
    assert!(pool.index.capacity() >= 100);
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

/// A pool seeded by a validator's shared keys pools every constant into the
/// slot a pool seeded by rebuilding them does: one key the seed holds in two
/// slots, the later winning it; constants equal to a seeded one; constants new
/// to both; and every kind a key reads -- `str`, `bytes`, `int`, `bool`,
/// `float` with its signed zero, and a class by its address. A validator whose
/// keys a relation has read is pooled through them into the same slots as one
/// read fresh.
#[test]
fn a_pool_seeded_by_shared_keys_pools_as_one_seeded_by_rebuilding() {
    Python::attach(|py| {
        let eval = |source: &str| {
            py.eval(&CString::new(source).expect("no nul"), None, None)
                .expect("the expression evaluates")
                .unbind()
        };
        let seed: Vec<Py<PyAny>> = ["'x'", "b'y'", "1", "True", "0.0", "int", "'x'"]
            .iter()
            .map(|source| eval(source))
            .collect();
        let asked: Vec<Py<PyAny>> = [
            "'x'", "b'y'", "1", "True", "-0.0", "int", "'z'", "2", "False", "b'y'", "'z'",
        ]
        .iter()
        .map(|source| eval(source))
        .collect();
        let copy = |items: &[Py<PyAny>]| items.iter().map(|o| o.clone_ref(py)).collect::<Vec<_>>();
        let slots = |mut pool: Pool| {
            let found: Vec<usize> = asked.iter().map(|obj| pool.intern(obj.bind(py))).collect();
            (found, pool.items().len())
        };
        let rebuilt = slots(Pool::seeded(py, copy(&seed)));
        let shared = slots(Pool::seeded_by(
            copy(&seed),
            Arc::new(PoolKeys::of(py, &seed)),
        ));
        assert_eq!(
            shared, rebuilt,
            "the shared keys pool as the rebuilt ones do"
        );
        assert_eq!(
            rebuilt.0.first(),
            Some(&6),
            "the later of two slots wins the key"
        );

        let schema = Schema::union((0..seed.len()).map(|at| Schema::Literal(ConstIx::new(at))));
        let validator =
            Bound::new(py, Validator::new(schema, copy(&seed), Vec::new())).expect("a validator");
        let pooled = || {
            let mut pool = Pool::seeded(py, copy(&asked));
            let schema = build_schema(validator.as_any(), &mut pool, &mut Vec::new())
                .expect("a validator builds");
            let items: Vec<usize> = pool.items().iter().map(|o| o.as_ptr() as usize).collect();
            (schema, items)
        };
        let fresh = pooled();
        assert!(validator.get().keys.get(py).is_none(), "nothing read yet");
        validator.get().pool_keys(py);
        assert!(
            validator.get().keys.get(py).is_some(),
            "read once, and kept"
        );
        assert_eq!(pooled(), fresh, "the kept keys pool as the fresh ones do");
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

/// A bound of exactly `int`, `float` or `bool` is placed in `numbers.Number`'s
/// register without asking the ABC, whose question is a Python function called
/// once a bound; a bound of any other type is asked. The count is of this
/// thread's questions, so a test running beside it asks nothing it reads.
#[test]
fn a_builtin_number_bound_is_placed_without_asking_the_abc() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"import abc, decimal, numbers, threading\n\
              asked = []\n\
              armed = [None]\n\
              _original = abc.ABCMeta.__instancecheck__\n\
              def _counting(cls, instance):\n\
              \x20   if cls is numbers.Number and armed[0] == threading.get_ident():\n\
              \x20       asked.append(type(instance).__name__)\n\
              \x20   return _original(cls, instance)\n\
              abc.ABCMeta.__instancecheck__ = _counting\n\
              class Ge:\n\
              \x20   def __init__(self, ge):\n\
              \x20       self.ge = ge\n\
              Ge.__module__ = 'annotated_types'\n\
              def arm():\n\
              \x20   asked.clear()\n\
              \x20   armed[0] = threading.get_ident()\n\
              def disarm():\n\
              \x20   armed[0] = None\n\
              \x20   return list(asked)\n\
              BOUNDS = [0, 1.5, True, decimal.Decimal(2)]\n",
            c"numbers_asked.py",
            c"numbers_asked",
        )
        .expect("the module compiles");
        let annotated = py
            .import("typing")
            .and_then(|typing| typing.getattr("Annotated"))
            .expect("typing has Annotated");
        let marker = module.getattr("Ge").expect("the marker is defined");
        let bounds = module.getattr("BOUNDS").expect("the bounds are defined");
        for (bound, wanted) in bounds.try_iter().expect("a list").zip([
            vec![],
            vec![],
            vec![],
            vec!["Decimal".to_owned()],
        ]) {
            let bound = bound.expect("a bound");
            let spelling = annotated
                .get_item((
                    py.get_type::<PyInt>(),
                    marker.call1((&bound,)).expect("a marker"),
                ))
                .expect("the annotation builds");
            module
                .getattr("arm")
                .and_then(|arm| arm.call0())
                .expect("armed");
            let built = build_schema(&spelling, &mut Pool::default(), &mut Vec::new());
            let asked: Vec<String> = module
                .getattr("disarm")
                .and_then(|disarm| disarm.call0())
                .and_then(|asked| asked.extract())
                .expect("disarmed");
            built.expect("an ordered bound builds");
            assert_eq!(asked, wanted, "the ABC asked of {bound}");
        }
    });
}

/// A dataclass declares the attributes `dataclasses.fields` returns, in its
/// order, however its fields were declared: a class variable and an init-only
/// parameter are no attribute of an instance, a field kept out of `__init__` is
/// one, and a subclass declares its bases' fields first. A table that is not
/// exactly a `dict` is left to the call, and reads the same.
#[test]
fn a_dataclass_declares_what_fields_returns() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"import dataclasses\n\
              from typing import ClassVar\n\
              @dataclasses.dataclass\n\
              class Plain:\n\
              \x20   a: int\n\
              \x20   b: str = ''\n\
              @dataclasses.dataclass\n\
              class Mixed:\n\
              \x20   a: int\n\
              \x20   shared: ClassVar[int] = 0\n\
              \x20   seed: dataclasses.InitVar[int] = 0\n\
              \x20   later: int = dataclasses.field(init=False, default=1)\n\
              @dataclasses.dataclass\n\
              class Derived(Mixed):\n\
              \x20   c: bytes = b''\n\
              @dataclasses.dataclass(frozen=True, slots=True)\n\
              class Slotted:\n\
              \x20   x: float\n\
              class Table(dict):\n\
              \x20   pass\n\
              @dataclasses.dataclass\n\
              class Rebound:\n\
              \x20   a: int\n\
              \x20   seed: dataclasses.InitVar[int] = 0\n\
              Rebound.__dataclass_fields__ = Table(Rebound.__dataclass_fields__)\n\
              def names(cls):\n\
              \x20   return [f.name for f in dataclasses.fields(cls)]\n",
            c"declared.py",
            c"declared",
        )
        .expect("the module compiles");
        for (class, wanted) in [
            ("Plain", vec!["a", "b"]),
            ("Mixed", vec!["a", "later"]),
            ("Derived", vec!["a", "later", "c"]),
            ("Slotted", vec!["x"]),
            ("Rebound", vec!["a"]),
        ] {
            let ty = module.getattr(class).expect("the class is defined");
            let ty = ty.cast::<PyType>().expect("a class");
            let declared: Vec<String> = declared_fields(ty)
                .expect("a dataclass declares its fields")
                .iter()
                .map(|name| name.extract().expect("a name is text"))
                .collect();
            let called: Vec<String> = module
                .getattr("names")
                .and_then(|names| names.call1((ty,)))
                .and_then(|names| names.extract())
                .expect("the call answers");
            assert_eq!(declared, wanted, "{class}");
            assert_eq!(declared, called, "{class} reads as the call does");
        }
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
                 \x20   x: int\n",
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
    });
}

/// The protocols the two rows below read, defined in the corpus namespace.
///
/// `Sized` and `Quiet` differ in the decorator alone, `Inherits` takes the
/// decorator's mark from a base, and `Members` declares one member of each
/// kind [`ProtocolMember`] tells apart, over a protocol base.
fn protocols(py: Python<'_>) -> Bound<'_, PyDict> {
    let namespace = namespace(py).expect("the corpus namespace builds");
    py.run(
        &CString::new(
            "import typing\n\
             @typing.runtime_checkable\n\
             class Sized(typing.Protocol):\n\
             \x20   def __len__(self) -> int: ...\n\
             class Quiet(typing.Protocol):\n\
             \x20   def __len__(self) -> int: ...\n\
             class Inherits(Sized, typing.Protocol):\n\
             \x20   def __bool__(self) -> bool: ...\n\
             @typing.runtime_checkable\n\
             class Members(Sized, typing.Protocol):\n\
             \x20   data: int\n\
             \x20   defaulted: str = 'a'\n\
             \x20   value = 0\n\
             \x20   @property\n\
             \x20   def computed(self) -> bytes: ...\n\
             \x20   @property\n\
             \x20   def unhinted(self): ...\n\
             \x20   def run(self) -> None: ...\n",
        )
        .expect("a source with no interior nul"),
        Some(&namespace),
        None,
    )
    .expect("the corpus protocols define");
    namespace
}

/// A protocol is the record of the members it declares, decorated or not.
///
/// The decorator is what lets `isinstance` answer, and nothing here asks
/// `isinstance`, so the same declaration is the same record with it, without
/// it, and with its mark inherited from a base.
#[test]
fn a_protocol_is_the_record_of_its_members_whatever_its_decorator() {
    Python::attach(|py| {
        let namespace = protocols(py);
        let rendered = |name: &str| {
            let annotation = namespace
                .get_item(name)
                .expect("the namespace answers")
                .expect("the class is in it");
            let mut pool = Pool::default();
            let mut defs = Vec::new();
            let schema = build_schema(&annotation, &mut pool, &mut defs)
                .unwrap_or_else(|refusal| panic!("{name} was refused: {refusal}"));
            let active = RefCell::new(FxHashMap::default());
            render(py, &schema, pool.items(), &defs, &active, 0).expect("the record renders")
        };
        assert_eq!(rendered("Sized"), "object(__len__=Callable)");
        assert_eq!(rendered("Quiet"), rendered("Sized"));
        assert_eq!(
            rendered("Inherits"),
            "object(__bool__=Callable, __len__=Callable)"
        );
        assert_eq!(
            rendered("Members"),
            "object(__len__=Callable, computed=bytes, data=int, defaulted=str, \
             run=Callable, unhinted=anything, value=anything)"
        );
    });
}

/// Each member is classified as `typing` lists and calls it.
///
/// From 3.12 `typing` caches a protocol's members as `__protocol_attrs__`, and
/// `@runtime_checkable` caches the ones whose class attribute is not callable
/// as `__non_callable_proto_members__`. The classifier reads the first and
/// derives its own split, so the two caches are the second opinion: the names
/// are the same, and the methods are exactly the members `typing` calls
/// callable. A member that is annotated and holds a callable on the class would
/// part the two, and none of these does.
#[test]
fn each_protocol_member_is_classified_as_typing_lists_it() {
    Python::attach(|py| {
        let cached = Since(12).met(py);
        if !cached {
            return;
        }
        let namespace = protocols(py);
        for name in ["Sized", "Inherits", "Members"] {
            let class = namespace
                .get_item(name)
                .expect("the namespace answers")
                .expect("the class is in it")
                .cast_into::<PyType>()
                .expect("a protocol is a class");
            let members = protocol_members(&class).expect("the members classify");
            let names: Vec<String> = members.iter().map(|(name, _)| name.to_string()).collect();
            let mut listed: Vec<String> = class
                .getattr("__protocol_attrs__")
                .expect("3.12 caches the members")
                .try_iter()
                .expect("a set iterates")
                .map(|name| name.expect("a name").to_string())
                .collect();
            listed.sort();
            assert_eq!(names, listed, "{name}'s members");
            let not_callable: Vec<String> = match class
                .getattr_opt("__non_callable_proto_members__")
                .expect("an optional attribute")
            {
                Some(cached) => cached
                    .try_iter()
                    .expect("a set iterates")
                    .map(|name| name.expect("a name").to_string())
                    .collect(),
                None => continue,
            };
            for (member, kind) in &members {
                let callable = !not_callable.contains(&member.to_string());
                assert_eq!(
                    matches!(kind, ProtocolMember::Method),
                    callable,
                    "{name}.{member} is classified against typing's callable split"
                );
            }
        }
    });
}

/// A compiled validator in `Annotated` metadata narrows by its set: the schema is
/// the meet of the refined base and the validator's set, written on its own,
/// beside a bound, or yielded by a grouped marker.
#[test]
fn a_validator_in_the_metadata_is_met_with_the_base() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the corpus namespace builds");
        let text = py
            .get_type::<Validator>()
            .call1((py.get_type::<pyo3::types::PyString>(),))
            .expect("Validator(str) builds");
        namespace
            .set_item("text", text)
            .expect("the namespace takes the validator");
        for (expression, wanted) in [
            ("typing.Annotated[int, text]", "intersection(int, str)"),
            (
                "typing.Annotated[int, at.Ge(0), text]",
                "intersection(str, Annotated[int, Ge(0)])",
            ),
            (
                "typing.Annotated[int, grouped(text)]",
                "intersection(int, str)",
            ),
        ] {
            let annotation = py
                .eval(
                    &CString::new(expression).expect("an expression with no interior nul"),
                    Some(&namespace),
                    None,
                )
                .unwrap_or_else(|error| panic!("{expression} did not evaluate: {error}"));
            let mut pool = Pool::default();
            let mut defs = Vec::new();
            let schema = build_schema(&annotation, &mut pool, &mut defs)
                .unwrap_or_else(|error| panic!("{expression} did not build: {error}"));
            let active = RefCell::new(FxHashMap::default());
            assert_eq!(
                render(py, &schema, pool.items(), &defs, &active, 0).expect("it renders"),
                wanted,
                "{expression}"
            );
        }
        // A validator is no constant: in a `Literal` it is refused, where the
        // schema reading would have built the set it checks.
        let literal = py
            .eval(c"typing.Literal[text]", Some(&namespace), None)
            .expect("the literal evaluates");
        assert!(
            build_schema(&literal, &mut Pool::default(), &mut Vec::new())
                .is_err_and(|err| err.to_string().contains("rather than a constant")),
            "a validator in a Literal is refused"
        );
    });
}

/// `Validator[int]` is an annotation for a static checker, and a schema is refused
/// it by name: the subscript is a `types.GenericAlias` whose origin is the class,
/// which the dispatch reads as a parametrized form it does not know.
#[test]
fn a_subscripted_validator_is_an_annotation_and_not_a_schema() {
    Python::attach(|py| {
        let class = py.get_type::<Validator>();
        let alias = class
            .get_item(py.get_type::<pyo3::types::PyInt>())
            .expect("the class subscripts");
        assert!(
            alias
                .getattr("__origin__")
                .expect("the alias has an origin")
                .is(&class),
            "the subscript is an alias of the class itself"
        );
        let mut pool = Pool::default();
        let mut defs = Vec::new();
        let refusal = match build_schema(&alias, &mut pool, &mut defs) {
            Err(refusal) => refusal.to_string(),
            Ok(schema) => panic!("Validator[int] built {schema:?}"),
        };
        assert!(
            refusal.contains("is the annotation a static checker reads"),
            "the refusal does not say what the alias is: {refusal}"
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
                render(py, &schema, pool.items(), &defs, &active, 0).expect("it renders"),
                wanted,
                "{name}"
            );
        }
    });
}

/// "No `extra_items` given" is written with the sentinel of the implementation
/// that built the class, and read against that one: `typing` and
/// `typing_extensions` carry two different objects before 3.15. Where the
/// implementation has a sentinel, `None` is a type the author gave; where it
/// has none, `None` is how it said nothing.
#[test]
fn a_typed_dict_is_read_against_the_sentinel_its_own_implementation_wrote() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the corpus namespace builds");
        py.run(
            &CString::new(
                "import sys, types\n\
                 with_sentinel = types.ModuleType('corpus_with_no_extra_items')\n\
                 with_sentinel.NoExtraItems = object()\n\
                 before_sentinel = types.ModuleType('corpus_before_no_extra_items')\n\
                 for module in (with_sentinel, before_sentinel):\n\
                 \x20   sys.modules[module.__name__] = module\n\
                 class Meta(type):\n\
                 \x20   pass\n\
                 Meta.__module__ = with_sentinel.__name__\n\
                 class OlderMeta(type):\n\
                 \x20   pass\n\
                 OlderMeta.__module__ = before_sentinel.__name__\n\
                 def record(meta, extra):\n\
                 \x20   class Record(metaclass=meta):\n\
                 \x20       __required_keys__ = frozenset({'name'})\n\
                 \x20       __extra_items__ = extra\n\
                 \x20       name: str\n\
                 \x20   return Record\n\
                 Unsaid = record(Meta, with_sentinel.NoExtraItems)\n\
                 NoneValued = record(Meta, None)\n\
                 OlderUnsaid = record(OlderMeta, None)\n\
                 OlderTyped = record(OlderMeta, int)\n",
            )
            .expect("a source with no interior nul"),
            Some(&namespace),
            None,
        )
        .expect("the corpus classes define");
        for (name, wanted) in [
            // The sentinel its own implementation wrote: the open default.
            ("Unsaid", "{'name': str, str: anything}"),
            // `extra_items=None` where a sentinel exists: extra values are None.
            ("NoneValued", "{'name': str, str: None}"),
            // An implementation older than the sentinel said nothing with None.
            ("OlderUnsaid", "{'name': str, str: anything}"),
            ("OlderTyped", "{'name': str, str: int}"),
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
                render(py, &schema, pool.items(), &defs, &active, 0).expect("it renders"),
                wanted,
                "{name}"
            );
        }
    });
}

/// A `TypedDict` that says nothing about the keys it does not name says what
/// its nearest base says, depth first through `__orig_bases__`; one that says
/// `closed=False` itself is open whatever its base says.
#[test]
fn a_typed_dict_inherits_the_tail_its_base_states() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the corpus namespace builds");
        py.run(
            c"import sys, types\n\
              module = types.ModuleType('corpus_inherited_tail')\n\
              module.NoExtraItems = object()\n\
              sys.modules[module.__name__] = module\n\
              class Meta(type):\n\
              \x20   pass\n\
              Meta.__module__ = module.__name__\n\
              def record(name, bases, closed, extra, keys):\n\
              \x20   cls = Meta(name, (), {\n\
              \x20       '__required_keys__': frozenset(keys),\n\
              \x20       '__closed__': closed,\n\
              \x20       '__extra_items__': extra,\n\
              \x20       '__annotations__': {key: int for key in sorted(keys)},\n\
              \x20   })\n\
              \x20   cls.__orig_bases__ = bases\n\
              \x20   return cls\n\
              unsaid = module.NoExtraItems\n\
              Shut = record('Shut', (), True, unsaid, {'a'})\n\
              Typed = record('Typed', (), None, str, {'a'})\n\
              Child = record('Child', (Shut,), None, unsaid, {'a', 'b'})\n\
              Grandchild = record('Grandchild', (Child,), None, unsaid, {'a', 'b'})\n\
              TypedChild = record('TypedChild', (object, Typed), None, unsaid, {'a', 'b'})\n\
              Reopened = record('Reopened', (Shut,), False, unsaid, {'a'})\n",
            Some(&namespace),
            None,
        )
        .expect("the classes build");
        for (name, wanted) in [
            ("Child", "{'a': int, 'b': int}"),
            ("Grandchild", "{'a': int, 'b': int}"),
            ("TypedChild", "{'a': int, 'b': int, str: str}"),
            ("Reopened", "{'a': int, str: anything}"),
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
                render(py, &schema, pool.items(), &defs, &active, 0).expect("it renders"),
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
                render(py, &schema, pool.items(), &defs, &active, 0).expect("it renders"),
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
            // `Annotated` carries no answer of its own either, and what it
            // annotates may be the qualifier.
            (
                Since(11),
                "typing.Annotated[typing.NotRequired[int], 1]",
                Some(false),
            ),
            (
                Since(11),
                "typing.Annotated[typing.Required[int], 1]",
                Some(true),
            ),
            (Since(11), "dict[str, typing.NotRequired[int]]", None),
        ] {
            if since.met(py) {
                assert_eq!(answer(expression), wanted, "{expression}");
            }
        }
    });
}

/// A form `typing_extensions` spells with an object of its own reads as its
/// `typing` spelling does: `Any` the top, `Never` the bottom, an alias its
/// aliased type, a qualifier the type it qualifies, `Unpack` a tuple's tail,
/// and `Self` a construct to refuse.
///
/// The module is the installed one where there is one. The corpora run on an
/// interpreter's own standard library, so where there is none a stand-in
/// carries the spellings the frontend asks for, under the module's name, for as
/// long as the row builds.
#[test]
fn a_typing_extensions_form_reads_as_its_typing_spelling() {
    Python::attach(|py| {
        let namespace = namespace(py).expect("the corpus namespace builds");
        py.run(
            c"import sys, types\n\
              try:\n\
              \x20   import typing_extensions as extensions\n\
              \x20   stand_in = False\n\
              except ImportError:\n\
              \x20   extensions = types.ModuleType('typing_extensions')\n\
              \x20   class _SpecialForm:\n\
              \x20       pass\n\
              \x20   class Any:\n\
              \x20       pass\n\
              \x20   class TypeAliasType:\n\
              \x20       def __init__(self, name, value):\n\
              \x20           self.__name__ = name\n\
              \x20           self.__value__ = value\n\
              \x20           self.__type_params__ = ()\n\
              \x20   for name in ('Never', 'Self', 'Required', 'NotRequired', 'ReadOnly', 'Unpack'):\n\
              \x20       setattr(extensions, name, _SpecialForm())\n\
              \x20   extensions._SpecialForm = _SpecialForm\n\
              \x20   extensions.Any = Any\n\
              \x20   extensions.TypeAliasType = TypeAliasType\n\
              \x20   sys.modules['typing_extensions'] = extensions\n\
              \x20   stand_in = True\n\
              def qualified(form, *args):\n\
              \x20   return typing._GenericAlias(form, args)\n",
            Some(&namespace),
            None,
        )
        .expect("the module or its stand-in is in place");
        let schema_of = |expression: &str| -> PyResult<Schema> {
            let form = py.eval(&CString::new(expression)?, Some(&namespace), None)?;
            build_schema(&form, &mut Pool::default(), &mut Vec::new())
        };
        let read = |expression: &str| schema_of(expression).expect(expression);
        assert_eq!(read("extensions.Any"), Schema::ANY);
        assert_eq!(read("extensions.Never"), Schema::Nothing);
        let ints = Schema::list(SeqShape::homogeneous(Schema::Int));
        assert_eq!(read("extensions.TypeAliasType('Ints', list[int])"), ints);
        for qualifier in ["Required", "NotRequired", "ReadOnly"] {
            assert_eq!(
                read(&format!("qualified(extensions.{qualifier}, int)")),
                Schema::Int,
                "{qualifier}"
            );
        }
        // And the two that say whether the key is required say it, alone and
        // behind the qualifier that says nothing.
        for (expression, wanted) in [
            ("qualified(extensions.Required, int)", Some(true)),
            ("qualified(extensions.NotRequired, int)", Some(false)),
            (
                "qualified(extensions.ReadOnly, qualified(extensions.NotRequired, int))",
                Some(false),
            ),
        ] {
            let hint = py
                .eval(
                    &CString::new(expression).expect("no interior nul"),
                    Some(&namespace),
                    None,
                )
                .expect(expression);
            assert_eq!(
                qualified_required(&hint).expect(expression),
                wanted,
                "{expression}"
            );
        }
        assert_eq!(
            read("tuple[int, qualified(extensions.Unpack, tuple[str, ...])]"),
            Schema::tuple(SeqShape::prefix_tail([Schema::Int], Schema::Str)),
        );
        // An element whose origin is a form, or a class, is no `Unpack`: it is
        // read as an element, and only the tail is spliced.
        assert_eq!(
            read(
                "tuple[typing.Literal[1], list[int], qualified(extensions.Unpack, tuple[str, ...])]"
            ),
            Schema::tuple(SeqShape::prefix_tail(
                [read("typing.Literal[1]"), ints.clone()],
                Schema::Str
            )),
        );
        assert!(
            schema_of("extensions.Self")
                .is_err_and(|err| err.is_instance_of::<PyNotImplementedError>(py)),
            "Self is a construct, not a value"
        );
        // The top and the bottom are forms in a `Literal`, as in `typing`.
        for form in ["Any", "Never"] {
            let refused = schema_of(&format!("typing.Literal[extensions.{form}]"));
            assert!(
                refused.is_err_and(|err| err.to_string().contains("a constant")),
                "{form}"
            );
        }
        py.run(
            c"if stand_in:\n    del sys.modules['typing_extensions']\n",
            Some(&namespace),
            None,
        )
        .expect("the stand-in is gone");
    });
}

/// An optional attribute reads what `getattr_opt` reads, on every kind of
/// object a build asks one of: a class that has the name, one that lacks it,
/// one holding `None` there, a class whose metaclass answers every name, one
/// whose metaclass refuses every name with `AttributeError` and one whose
/// metaclass raises something else, and an object that is no class.
///
/// On 3.12 a class is asked through the builtin `getattr` with a sentinel, so
/// what the two readings share is the whole of the claim: the same object,
/// the same absence, the same error.
#[test]
fn an_optional_attribute_reads_what_getattr_opt_reads() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"class Has:\n\
              \x20   name = 1\n\
              class Lacks:\n\
              \x20   pass\n\
              class HoldsNone:\n\
              \x20   name = None\n\
              class Answers(type):\n\
              \x20   def __getattr__(cls, name):\n\
              \x20       return name\n\
              class AnswersAll(metaclass=Answers):\n\
              \x20   pass\n\
              class Refuses(type):\n\
              \x20   def __getattr__(cls, name):\n\
              \x20       raise AttributeError(name)\n\
              class RefusesAll(metaclass=Refuses):\n\
              \x20   pass\n\
              class Raises(type):\n\
              \x20   def __getattr__(cls, name):\n\
              \x20       raise ValueError(name)\n\
              class RaisesAll(metaclass=Raises):\n\
              \x20   pass\n\
              instance = Has()\n",
            c"optional.py",
            c"optional",
        )
        .expect("the module compiles");
        let name = intern!(py, "name");
        for object in [
            "Has",
            "Lacks",
            "HoldsNone",
            "AnswersAll",
            "RefusesAll",
            "RaisesAll",
            "instance",
        ] {
            let held = module.getattr(object).expect("the object");
            match (optional_attribute(&held, name), held.getattr_opt(name)) {
                (Ok(ours), Ok(theirs)) => assert_eq!(
                    ours.as_ref().map(Bound::as_ptr),
                    theirs.as_ref().map(Bound::as_ptr),
                    "{object} reads one attribute"
                ),
                (Err(ours), Err(theirs)) => assert!(
                    ours.get_type(py).is(theirs.get_type(py)),
                    "{object} raises one error"
                ),
                (ours, theirs) => panic!("{object} reads {ours:?} against {theirs:?}"),
            }
        }
    });
}
