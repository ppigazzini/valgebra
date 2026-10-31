//! What a class declares: dispatch step 5, and the record the class's own
//! attributes describe.
//!
//! A `TypedDict` says so by carrying `__required_keys__`, an enum by subclassing
//! `Enum`, a dataclass by the `__dataclass_fields__` that
//! `dataclasses.is_dataclass` reads, a `Protocol` by
//! `_is_protocol`; every other class names its instances and is an
//! `isinstance` atom. The section "What a class declares" in
//! `docs/dev/03-frontend.md` is this module.

use pyo3::PyTypeInfo;
use pyo3::exceptions::{PyBaseException, PyException, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyNone, PySet, PyString, PyTuple,
    PyType,
};
use valgebra_core::{Attribute, Field, MapClause, Schema, SeqShape};

use super::generics::{MAX_QUALIFIER_DEPTH, is_field_qualifier};
use super::{
    ClassScope, MAX_BUILD_DEPTH, Pool, build_schema, forms, is_extension, loaded_modules,
    not_implemented,
};
use crate::errors::{summarize, unless_fatal};

/// What [`declared_fields`] reads a dataclass through, held apart from
/// [`Forms`](super::Forms) and resolved on the first dataclass a build reads.
///
/// Not in the cache beside the other forms, because that cache is built the
/// first time anything is compiled and `dataclasses` is a module most programs
/// never import: it pulls `inspect`, `copy`, `functools` and their own imports
/// in with it, and the objects they leave behind are *tracked* -- so every
/// later collection walks them. Measured on the shape that compiles a
/// fifty-field record of plain types, which reaches no dataclass: importing it
/// with the rest reads **6.45% dearer**, all of it after the import, in the
/// generational walks a build's own allocations trigger. Reached only after
/// [`is_dataclass`] has answered yes, so the module is already imported by the
/// time this cell is filled, and a program that compiles no dataclass never
/// imports it.
static DATACLASS_READING: PyOnceLock<DataclassReading> = PyOnceLock::new();

/// `dataclasses.fields`, and the marker it keeps a field by.
struct DataclassReading {
    /// `dataclasses.fields`, the call [`fields_as_declared`] stands in for.
    fields: Py<PyAny>,
    /// `dataclasses._FIELD`, the `_field_type` of a field an instance keeps;
    /// `None` where the module has no such name, and the call answers alone.
    kept: Option<Py<PyAny>>,
}

/// Whether `ty` is a dataclass, read as `dataclasses.is_dataclass` reads it.
///
/// That function's body, on every interpreter the matrix runs (3.10 to 3.15
/// and `PyPy`), is `hasattr(cls, "__dataclass_fields__")` once its argument is
/// a class, which `ty` is: 3.10's further test for a `GenericAlias` cannot pass
/// on an object that is a `type`. So the attribute is asked here and the
/// function is not called. That spares a Python frame on every class node past
/// the enum arm, and it leaves `dataclasses` unimported by a schema that names
/// classes and no dataclass, for the reason [`DATACLASS_READING`] gives.
/// `a_class_is_a_dataclass_where_is_dataclass_says_so` in `build/interpreter.rs`
/// holds the reading to the call.
pub(super) fn is_dataclass(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    ty.hasattr(intern!(ty.py(), "__dataclass_fields__"))
}

/// Whether a class declares fields the frontend reads beside the class: a
/// dataclass, or a named tuple.
pub(crate) fn declares_fields(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    Ok(is_dataclass(ty)?
        || (ty.is_subclass_of::<PyTuple>()? && ty.hasattr(intern!(ty.py(), "_fields"))?))
}

/// Build the class alone: its instances, whatever it declares.
///
/// [`build_type_object`] with the declaration left unread. A dataclass and a
/// named tuple are their `isinstance` atom here, where that step meets the
/// atom with the fields they declare; every other class is the node that step
/// builds. A `TypedDict` has no instances and a `Protocol`'s set is the record
/// of its members, so neither has a class alone to give, and `isinstance`
/// refuses both too.
pub(crate) fn build_instances(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = ty.py();
    if ty.hasattr(intern!(py, "__required_keys__"))? {
        return Err(PyTypeError::new_err(format!(
            "{} is a TypedDict, whose values are dicts rather than instances of \
             it; Validator(...) reads it as the record it declares",
            summarize(ty.as_any())?
        )));
    }
    if is_truthy_attr(ty, intern!(py, "_is_protocol"))? {
        return Err(PyTypeError::new_err(format!(
            "{} is a Protocol, whose set is the values carrying its members \
             rather than instances of it; Validator(...) reads it as that record",
            summarize(ty.as_any())?
        )));
    }
    if declares_fields(ty)? {
        return Ok(Schema::Instance(lits.intern_class(ty.as_any())));
    }
    build_type_object(ty, lits, defs)
}

/// Build the schema for a Python type object (a builtin, `TypedDict`, `Enum`,
/// dataclass, `NamedTuple`, `Protocol`, or `object`).
pub(super) fn build_type_object(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = ty.py();
    if ty.is(py.get_type::<PyBool>()) {
        return Ok(Schema::Bool);
    }
    if ty.is(py.get_type::<PyInt>()) {
        return Ok(Schema::Int);
    }
    if ty.is(py.get_type::<PyFloat>()) {
        return Ok(Schema::Float);
    }
    if ty.is(py.get_type::<PyString>()) {
        return Ok(Schema::Str);
    }
    if ty.is(py.get_type::<PyBytes>()) {
        return Ok(Schema::Bytes);
    }
    if ty.is(py.None().bind(py).get_type()) {
        return Ok(Schema::NoneType);
    }
    // A bare container class is its kind: `list` admits every list, which is the
    // set `list[object]` names and the set the typing spec assigns an
    // unparameterized generic. Read as an `isinstance` atom it would be a
    // different sort of thing from the sequence node beside it -- in no kind at
    // all -- and neither spelling would be decided below the other. See "What a
    // bare builtin class denotes" in `docs/dev/01-schema-ir.md`.
    if ty.is(py.get_type::<PyList>()) {
        return Ok(Schema::list(SeqShape::homogeneous(Schema::ANYTHING)));
    }
    if ty.is(py.get_type::<PyTuple>()) {
        return Ok(Schema::tuple(SeqShape::homogeneous(Schema::ANYTHING)));
    }
    if ty.is(py.get_type::<PySet>()) {
        return Ok(Schema::set(Schema::ANYTHING));
    }
    if ty.is(py.get_type::<PyFrozenSet>()) {
        return Ok(Schema::frozen_set(Schema::ANYTHING));
    }
    if ty.is(py.get_type::<PyDict>()) {
        return Ok(Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause::top()].into(),
        });
    }
    let forms = forms(py)?;
    if ty.is(forms.object.bind(py)) {
        return Ok(Schema::ANYTHING);
    }
    // `typing.Generic` is the base a generic class derives from, and no type:
    // the annotation grammar has no production for it, and read as a class it
    // would be the instances of every generic class, which no annotation means.
    if ty.is(forms.generic.bind(py)) {
        return Err(not_implemented(format!(
            "{} is the base a generic class is declared from, not a type: pass \
             the generic class itself, applied to the arguments its fields need",
            summarize(ty.as_any())?
        )));
    }
    // A bare typing special form is a class on some Pythons (notably `Union`).
    // It is not a value, so reject it rather than building an instance check that
    // accepts nothing; the special forms that are not types are rejected on the
    // value fallthrough in build_schema.
    for form in [&forms.union, &forms.optional] {
        if ty.is(form.bind(py)) {
            return Err(not_implemented(format!(
                "{} is a typing special form, not a value; write a concrete type \
                 (for a union, X | Y or Union[X, Y])",
                summarize(ty.as_any())?
            )));
        }
    }
    // TypedDict: a closed record whose required keys come from the class.
    if ty.hasattr(intern!(ty.py(), "__required_keys__"))? {
        return build_typed_dict(ty, None, lits, defs);
    }
    // Enum: an instance of the enumeration class (any of its members).
    if ty.is_subclass(forms.enum_class.bind(py))? {
        return Ok(Schema::Instance(lits.intern_class(ty.as_any())));
    }
    // dataclass / NamedTuple: isinstance plus a deep check of each field.
    if declares_fields(ty)? {
        return build_object(ty, None, lits, defs);
    }
    // Protocol: the record of the members it declares, read off the value as
    // every attribute is. `@runtime_checkable` is not read: it is what lets
    // `isinstance` answer, and nothing here asks `isinstance`.
    if is_truthy_attr(ty, intern!(py, "_is_protocol"))? {
        return build_protocol(ty, lits, defs);
    }
    // `Any` is a class from 3.11, and `typing_extensions.Any` a class of its
    // own on 3.10. Read as one, it is an instance check nothing passes.
    // `read_annotation` answers `typing.Any` before it reaches this, which
    // `instance_of` does not pass through: asked only as `typing_extensions`'
    // copy, which is `typing`'s from 3.11, `instance_of(typing.Any)` would be
    // the top where that module is loaded and refuse every value where not.
    if ty.is(forms.any.bind(py)) {
        return Ok(Schema::ANY);
    }
    if is_extension(ty.as_any(), |held| &held.any)? {
        return Ok(Schema::ANY);
    }
    // Any other class names its instances: a bare class is an isinstance check.
    // This covers the remaining builtins (complex, bytearray, memoryview, range,
    // the `collections.abc` ABCs including Callable, ...) and arbitrary user
    // classes uniformly.
    Ok(Schema::Instance(lits.intern_class(ty.as_any())))
}

/// What a protocol declares one of its members as.
///
/// The typing spec's member list is every name the class body and its protocol
/// bases define or annotate; this is which of those a member is, read from the
/// annotation and the class attribute behind the name. Each kind is read off a
/// value by `getattr`, as every record field is, and what differs is the set
/// the field admits.
pub(super) enum ProtocolMember<'py> {
    /// An annotated attribute, and its hint: the value's attribute is a member
    /// of the hint's set.
    Data(Bound<'py, PyAny>),
    /// A name the class defines with a callable and nothing annotates -- a
    /// method, a special method, any callable the body holds. The value's
    /// attribute is callable; what it accepts is not readable, as for
    /// `Callable[...]`.
    Method,
    /// A property or a cached property, and its getter's return hint where it
    /// has one: the value's attribute is what the getter returns.
    Property(Option<Bound<'py, PyAny>>),
    /// A name the class defines with a value that is neither callable nor a
    /// property, and nothing annotates: the value carries the attribute,
    /// holding anything.
    Value,
}

/// Each member a protocol declares, in name order, with what it is declared as.
///
/// The names are the ones `typing` lists. Every protocol carries them as
/// `__protocol_attrs__` from 3.12, and a `typing_extensions` protocol on every
/// release; elsewhere `typing._get_protocol_attrs` derives them, which is the
/// derivation that attribute caches -- the keys of each base's namespace and
/// annotations along the `__mro__`, less the names Python and `typing` write
/// there themselves. The cache is read first because the derivation of an
/// older `typing` does not know the names a newer `typing_extensions` writes,
/// and on 3.10 lists `__protocol_attrs__` itself as a member.
///
/// A member is a property where the first class on the `__mro__` defining the
/// name holds one or a `functools.cached_property`, data where an annotation names it, a method where the class
/// attribute is callable, and a value otherwise -- callable as `typing` asks
/// it, `callable(getattr(cls, name))`. A member annotated `ClassVar` or `Final`
/// is refused: the qualifier says where the value lives, and reading it as the
/// type it wraps is a reading no proposal has argued for yet.
pub(super) fn protocol_members<'py>(
    ty: &Bound<'py, PyType>,
) -> PyResult<Vec<(Bound<'py, PyString>, ProtocolMember<'py>)>> {
    let py = ty.py();
    let forms = forms(py)?;
    let names = match ty.getattr_opt(intern!(py, "__protocol_attrs__"))? {
        Some(cached) => cached,
        None => forms
            .get_protocol_attrs
            .as_ref()
            .ok_or_else(|| {
                not_implemented(
                    "this release's typing lists no protocol members, so a Protocol \
                     cannot be read as a schema",
                )
            })?
            .bind(py)
            .call1((ty,))?,
    };
    let mut names = names
        .try_iter()?
        .map(|name| Ok(name?.cast_into::<PyString>()?))
        .collect::<PyResult<Vec<_>>>()?;
    names.sort_by_cached_key(ToString::to_string);
    let hints = resolve_type_hints(ty)?;
    let hints = hints.cast::<PyDict>()?;
    let namespaces = ty
        .getattr(intern!(py, "__mro__"))?
        .try_iter()?
        .map(|base| base?.getattr(intern!(py, "__dict__")))
        .collect::<PyResult<Vec<_>>>()?;
    let mut members = Vec::with_capacity(names.len());
    for name in names {
        let member = if let Some(defined) = defined_on(&namespaces, &name)?
            && let Some(getter) = getter_of(&defined)?
        {
            ProtocolMember::Property(getter_return(&getter)?)
        } else if let Some(hint) = hints.get_item(&name)? {
            if is_qualified_member(&hint)? {
                return Err(not_implemented(format!(
                    "member {} of {} is declared ClassVar or Final, which says where \
                     the value lives rather than what it is: declare it with its type",
                    summarize(name.as_any())?,
                    summarize(ty.as_any())?
                )));
            }
            ProtocolMember::Data(hint)
        } else if ty.getattr(&name)?.is_callable() {
            ProtocolMember::Method
        } else {
            ProtocolMember::Value
        };
        members.push((name, member));
    }
    Ok(members)
}

/// What the first of `namespaces` -- each class's own on a `__mro__`, in its
/// order -- to define `name` holds there: the object an attribute lookup on an
/// instance finds before it asks the instance.
///
/// The namespaces are read once per protocol rather than once per member. Each
/// is the class's `__dict__`, a view of the dictionary rather than a copy of
/// it, so a member is still looked up in what the class holds when it is read.
fn defined_on<'py>(
    namespaces: &[Bound<'py, PyAny>],
    name: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    for namespace in namespaces {
        if namespace.contains(name)? {
            return Ok(Some(namespace.get_item(name)?));
        }
    }
    Ok(None)
}

/// The function a member's value is computed by, where the class defines it
/// with a `property` -- its getter, `None` where it has none -- or a
/// `functools.cached_property`, whose function an instance's first read calls
/// and keeps. `None` overall for any other class attribute.
///
/// Read as a value, a cached property would be `anything`, dropping the type
/// its function returns, which a checker reads as a property's.
///
/// The attribute's type is asked, rather than the attribute by `isinstance`,
/// which reads its `__class__` once the type answers no: an attribute lookup
/// per method a protocol declares and per class asked, and asked that way the
/// second class costs the fifty-member protocol build 0.86%.
fn getter_of<'py>(defined: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = defined.py();
    let forms = forms(py)?;
    let class = defined.get_type();
    if class.is_subclass(forms.property.bind(py))? {
        return Ok(Some(defined.getattr(intern!(py, "fget"))?));
    }
    if class.is_subclass(forms.cached_property.bind(py))? {
        return Ok(Some(defined.getattr(intern!(py, "func"))?));
    }
    Ok(None)
}

/// The return hint a member's getter carries, resolved as `get_type_hints`
/// resolves it, or `None` for no getter or no return hint.
fn getter_return<'py>(getter: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = getter.py();
    if getter.is_none() {
        return Ok(None);
    }
    let options = PyDict::new(py);
    options.set_item(intern!(py, "include_extras"), true)?;
    let hints = forms(py)?
        .get_type_hints
        .bind(py)
        .call((getter,), Some(&options))?;
    hints
        .get_item(intern!(py, "return"))
        .map(Some)
        .or_else(|err| {
            if err.is_instance_of::<pyo3::exceptions::PyKeyError>(py) {
                Ok(None)
            } else {
                Err(err)
            }
        })
}

/// Whether a member's hint is `ClassVar` or `Final`, bare or parametrized,
/// and under `Annotated`, which wraps a qualifier without changing what it
/// says.
///
/// A class `type` made is neither, and `typing.get_origin` answers one with
/// `None`, or with `Generic` for `Generic` itself, so the answer is read off it
/// without the call: a Python function, asked of nearly every member.
fn is_qualified_member(hint: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = hint.py();
    if hint.get_type().is(py.get_type::<PyType>()) {
        return Ok(false);
    }
    let forms = forms(py)?;
    let hint = if hint.get_type().is(forms.annotated_alias.bind(py)) {
        &hint.getattr(intern!(py, "__origin__"))?
    } else {
        hint
    };
    let origin = forms.get_origin.bind(py).call1((hint,))?;
    Ok([&forms.class_var, &forms.final_qualifier]
        .iter()
        .any(|qualifier| hint.is(qualifier.bind(py)) || origin.is(qualifier.bind(py))))
}

/// Build the record a protocol denotes: every value carrying each member the
/// protocol declares, as [`ProtocolMember`] reads it.
///
/// `typing.Protocol` itself is the base a protocol is declared from, and names
/// no set; a generic protocol names a family of them, one per argument, and is
/// refused until a reading of its parameters is argued for.
pub(super) fn build_protocol(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = ty.py();
    let forms = forms(py)?;
    let _open = ClassScope::enter(ty, None)?;
    if ty.is(forms.protocol.bind(py)) || is_extension(ty.as_any(), |held| &held.protocol)? {
        return Err(not_implemented(format!(
            "{} is the base a protocol is declared from, not a type: pass the \
             protocol class itself",
            summarize(ty.as_any())?
        )));
    }
    if ty
        .getattr_opt(intern!(py, "__parameters__"))?
        .is_some_and(|parameters| parameters.len().is_ok_and(|count| count > 0))
    {
        return Err(not_implemented(format!(
            "{} is a generic Protocol, which names one set per type argument: \
             declare its members with concrete types",
            summarize(ty.as_any())?
        )));
    }
    let mut fields = Vec::new();
    for (name, member) in protocol_members(ty)? {
        let schema = match member {
            ProtocolMember::Data(hint) | ProtocolMember::Property(Some(hint)) => {
                build_schema(&hint, lits, defs)?
            }
            ProtocolMember::Method => build_schema(forms.callable.bind(py), lits, defs)?,
            ProtocolMember::Property(None) | ProtocolMember::Value => Schema::ANYTHING,
        };
        fields.push(Attribute {
            name: field_name(&name)?.into(),
            schema,
        });
    }
    Ok(Schema::attr_record(fields))
}

/// True if `obj.<name>` exists and is truthy; false on absence or an ordinary
/// error, and a fatal signal raised.
///
/// The name arrives interned and the attribute is asked for *optionally*, which
/// is what absence costs here: every one of these names is absent on an
/// ordinary class, and a bare `getattr` answers that by building an
/// `AttributeError`, raising it and dropping it -- per class node, for an answer
/// that is `false`.
pub(super) fn is_truthy_attr(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<bool> {
    let py = obj.py();
    match unless_fatal(optional_attribute(obj, name), py, None)? {
        Some(value) => unless_fatal(value.is_truthy(), py, false),
        None => Ok(false),
    }
}

/// `obj.name`, or `None` where `obj` has no attribute of that name: the answer
/// `getattr_opt` gives, at what a class's miss costs on the interpreter.
///
/// Below 3.13 `PyO3` reads a miss as an `AttributeError` raised and cleared,
/// and a class formats the error's message on the way -- some 3,300
/// instructions a name, asked per field of a `TypedDict`, per argument of a
/// tuple and per class of a build, nearly always of a name the class lacks.
/// On 3.12 the builtin `getattr` with a default asks a class through
/// `_PyObject_LookupAttr`, which answers its miss without building the error,
/// so there a class is asked through it, with a sentinel no attribute holds
/// standing for absence. An error other than `AttributeError` propagates on
/// either road. 3.10 and 3.11 build the error on both, and from 3.13 `PyO3`
/// asks through the lookup that does not, so neither takes the detour; nor
/// does an object that is not a class, which this lookup reads as `PyO3`
/// does.
#[cfg(all(Py_3_12, not(Py_3_13)))]
pub(super) fn optional_attribute<'py>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    /// The builtin `getattr`, and an object nothing else holds.
    static LOOKUP: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();
    if obj.is_instance_of::<PyType>() {
        let py = obj.py();
        let (getattr, absent) = LOOKUP.get_or_try_init(py, || {
            let builtins = py.import(intern!(py, "builtins"))?;
            Ok::<_, PyErr>((
                builtins.getattr(intern!(py, "getattr"))?.unbind(),
                builtins.getattr(intern!(py, "object"))?.call0()?.unbind(),
            ))
        })?;
        let found = getattr.bind(py).call1((obj, name, absent))?;
        Ok(if found.is(absent) { None } else { Some(found) })
    } else {
        obj.getattr_opt(name)
    }
}

/// `obj.name`, or `None` where `obj` has no attribute of that name: `PyO3`'s
/// `getattr_opt`, which asks without an exception from 3.13 and builds one
/// either way on 3.10 and 3.11.
#[cfg(not(all(Py_3_12, not(Py_3_13))))]
pub(super) fn optional_attribute<'py>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    obj.getattr_opt(name)
}

/// Resolve a class's type hints with `Annotated` metadata preserved.
///
/// `include_extras=True` keeps `Annotated[...]` field types intact so a field's
/// refinement markers reach [`build_refine`](super::refine::build_refine);
/// without it `get_type_hints` strips the metadata and the field's constraints
/// are silently lost.
///
/// **The annotations as written are the answer wherever evaluation would hand
/// each one back unchanged**, and [`annotations_as_written`] reads them without
/// the call. `get_type_hints` exists to evaluate forward references, and on a
/// class with none it still copies every base's namespace and walks every
/// annotation in Python: 60 to 70% of compiling a dataclass or a `TypedDict`.
/// Anything the reading is not certain of goes to `get_type_hints`, so an
/// answer or an error from this function is always the one it gives.
pub(super) fn resolve_type_hints<'py>(ty: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
    // A failure inside the reading is a decline and never an answer: the
    // exception a caller sees is the one `get_type_hints` raises.
    if let Ok(Some(hints)) = annotations_as_written(ty) {
        return Ok(hints.into_any());
    }
    let py = ty.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "include_extras"), true)?;
    forms(py)?
        .get_type_hints
        .bind(py)
        .call((ty,), Some(&kwargs))
}

/// A class's resolved hints with the type variables its generic bases bind
/// in place: the arguments it is applied to where `applied` gives them, the
/// arguments each base it derives from was applied to, and a parameter's PEP
/// 696 default where nothing else binds it.
///
/// `class Child(Base[int])` declares `x: T` through `Base`, and `typing` hands
/// the hint back as `T`: the base's arguments are the one place `T` is said to
/// be `int`. They are read off what each class was written deriving from
/// ([`Scopes::of`]), and each field's hint in the scope of the class that
/// declares it ([`declarers`]). Every class the walk reaches is generic or
/// derives from one, so a class with no `Generic` on its `__mro__` binds
/// nothing and pays one scan of the order for it.
pub(super) fn bound_hints<'py>(
    ty: &Bound<'py, PyType>,
    applied: Option<&Bound<'py, PyTuple>>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = ty.py();
    let hints = resolve_type_hints(ty)?.cast_into::<PyDict>()?;
    let generic = forms(py)?.generic.bind(py);
    if applied.is_none() && !ty.mro().iter_borrowed().any(|base| base.is(generic)) {
        return Ok(hints);
    }
    let scopes = Scopes::of(ty, applied)?;
    let declarers = declarers(ty, &hints, &scopes)?;
    let bound = PyDict::new(py);
    for (name, hint) in hints.iter() {
        let scope = match declarers.get_item(&name)? {
            Some(class) => scopes.of_class(&class),
            None => &scopes.read,
        };
        let hint = super::substitute(&hint, &scope.parameters, &scope.arguments)?;
        // A parameter of the class's own that nothing bound is the class
        // written bare, and a field declared with it names no one type.
        let read = &scopes.read;
        for (parameter, argument) in read.parameters.iter().zip(&read.arguments) {
            if !argument.is(&parameter) {
                continue;
            }
            let names = hint.is(&parameter)
                || hint
                    .getattr_opt(intern!(py, "__parameters__"))?
                    .and_then(|named| named.cast_into::<PyTuple>().ok())
                    .is_some_and(|named| named.iter().any(|variable| variable.is(&parameter)));
            if names {
                let (class, variable) = (
                    super::declared_name(ty.as_any())?,
                    super::declared_name(&parameter)?,
                );
                return Err(not_implemented(format!(
                    "{class} is generic in {variable}, and its field {} is declared \
                     with it: write {class}[...] with the type {variable} stands for",
                    summarize(&name)?
                )));
            }
        }
        bound.set_item(&name, hint)?;
    }
    Ok(bound)
}

/// Refuse binding `class`'s parameters where one of them is no `TypeVar`.
///
/// A `TypeVarTuple` stands for a list of types and a `ParamSpec` for a
/// signature, and an application binds one to a slice of its arguments, which
/// the schema language has no form for: a field `tuple[*Ts]` of `Row[int, str]`
/// is `tuple[int, str]`, and read one argument a parameter it held `(1,)` and
/// refused `(1, "a")`. Refused as an alias declaring one is, where the class is
/// applied, takes a default, or is a base written applied.
fn one_type_each(class: &Bound<'_, PyType>, parameters: &Bound<'_, PyTuple>) -> PyResult<()> {
    let py = class.py();
    let type_var = forms(py)?.type_var.bind(py);
    for parameter in parameters.iter() {
        if !parameter.get_type().is(type_var) {
            return Err(not_implemented(format!(
                "{} declares {}, which stands for a list of types or a signature \
                 rather than for one type, so no argument here substitutes it: write \
                 the class over TypeVars alone",
                super::declared_name(class.as_any())?,
                super::declared_name(&parameter)?
            )));
        }
    }
    Ok(())
}

/// Refuse a `TypedDict` qualifier given as one of `class`'s type arguments.
///
/// `Required`, `NotRequired` and `ReadOnly` qualify a `TypedDict` item where it
/// is declared, and the typing spec admits them nowhere else (PEP 655, PEP
/// 705): `Holds[NotRequired[int]]` names no type, and the runtime keeps the key
/// it reaches required. Substituted into the field, the qualifier read as the
/// item's own, turning a required key optional, or vanished from a dataclass's
/// field. Read through `Annotated`, which a qualifier may sit under.
fn no_qualifier_among(class: &Bound<'_, PyType>, arguments: &Bound<'_, PyTuple>) -> PyResult<()> {
    let py = class.py();
    let forms = forms(py)?;
    for argument in arguments.iter() {
        let mut held = argument.clone();
        // Bounded as a literal's field is read, so the loop ends whatever a
        // form's `__origin__` claims.
        for _ in 0..MAX_QUALIFIER_DEPTH {
            if !held.get_type().is(forms.annotated_alias.bind(py)) {
                break;
            }
            held = held.getattr(intern!(py, "__origin__"))?;
        }
        let origin = forms.get_origin.bind(py).call1((&held,))?;
        if is_field_qualifier(&origin)? {
            return Err(not_implemented(format!(
                "{} as a type argument of {} is a TypedDict qualifier, which \
                 qualifies an item where it is declared and nowhere else: declare \
                 the field Required, NotRequired or ReadOnly in the class, and pass \
                 the type alone",
                summarize(&argument)?,
                super::declared_name(class.as_any())?
            )));
        }
    }
    Ok(())
}

/// The PEP 696 default `parameter` takes in one application of its class: where
/// nothing is `written` for it, and where what is written is the default itself
/// -- `CPython` hands an application left short back with the default filled in,
/// as the very object the parameter holds, unsubstituted (`Pairs[int]` is
/// `Pairs[int, list[T]]`). A default may name the parameters before it, which
/// the application binds, so the caller substitutes them: `Pairs[int]`'s `U`
/// is `list[int]`. `None` where an argument of the caller's own is written, or
/// where the parameter has no default.
fn defaulted<'py>(
    parameter: &Bound<'py, PyAny>,
    written: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(default) = super::default_of(parameter)? else {
        return Ok(None);
    };
    Ok(written
        .is_none_or(|written| written.is(&default))
        .then_some(default))
}

/// What the parameters of a generic class and of each class it derives from
/// stand for in the class read: the class's own, bound to the arguments it is
/// applied to or their defaults, and each base's, bound to the arguments it was
/// written applied to, read through the class written deriving from it.
///
/// **A type variable is its class's own.** `class IntBox(Box[int],
/// Generic[T])` reuses the `T` that `Box` declares its field with, and the two
/// are two parameters to `typing` -- `Box[int]` closes `Box`'s, and
/// `IntBox`'s is another, the same object in another scope -- as mypy maps a
/// base's fields through the arguments the derived class gives it. So each
/// field is substituted in the scope of the class that declares it, never
/// through one table keyed by the variable, which reads `IntBox[str]`'s `T` as
/// `int` and `str` at once. One class reached through two bases under two
/// arguments is the conflict that is left: a field it declares names two
/// types, and is refused.
struct Scopes<'py> {
    /// The class read.
    read: Scope<'py>,
    /// Each class it derives from, once, in the order the walk meets them.
    bases: Vec<Scope<'py>>,
}

/// One class's parameters and what each stands for in the class read, and the
/// classes it was written deriving from.
struct Scope<'py> {
    class: Bound<'py, PyType>,
    parameters: Bound<'py, PyTuple>,
    arguments: Vec<Bound<'py, PyAny>>,
    written: Vec<Bound<'py, PyType>>,
}

impl<'py> Scopes<'py> {
    /// The scopes `ty`, applied to `applied` or written bare, reaches through
    /// every class it was written deriving from. A parameter nothing binds
    /// stands for itself.
    fn of(ty: &Bound<'py, PyType>, applied: Option<&Bound<'py, PyTuple>>) -> PyResult<Self> {
        let parameters = parameters_of(ty)?;
        let mut arguments = Vec::with_capacity(parameters.len());
        if let Some(applied) = applied {
            no_qualifier_among(ty, applied)?;
        }
        for (at, parameter) in parameters.iter().enumerate() {
            let written = applied.and_then(|args| args.get_item(at).ok());
            let argument = match defaulted(&parameter, written.as_ref())? {
                Some(default) => super::substitute(&default, &parameters, &arguments)?,
                None => written.unwrap_or_else(|| parameter.clone()),
            };
            if !argument.is(&parameter) {
                one_type_each(ty, &parameters)?;
            }
            arguments.push(argument);
        }
        let mut scopes = Self {
            read: Scope {
                class: ty.clone(),
                parameters,
                arguments,
                written: Vec::new(),
            },
            bases: Vec::new(),
        };
        let mut pending = vec![ty.clone()];
        while let Some(class) = pending.pop() {
            let scope = scopes.of_class(class.as_any());
            let mut reached = Vec::new();
            for (base, written) in written_bases(&class)? {
                let own = parameters_of(&base)?;
                let arguments = match written {
                    Some(written) => {
                        one_type_each(&base, &own)?;
                        no_qualifier_among(&base, &written)?;
                        let mut arguments = Vec::with_capacity(written.len());
                        for (at, argument) in written.iter().enumerate() {
                            let parameter = own.get_item(at).ok();
                            let default = match &parameter {
                                Some(parameter) => defaulted(parameter, Some(&argument))?,
                                None => None,
                            };
                            arguments.push(match default {
                                Some(default) => super::substitute(&default, &own, &arguments)?,
                                None => super::substitute(
                                    &argument,
                                    &scope.parameters,
                                    &scope.arguments,
                                )?,
                            });
                        }
                        arguments
                    }
                    None => own.iter().collect(),
                };
                reached.push((base, arguments));
            }
            let written = reached.iter().map(|(base, _)| base.clone()).collect();
            for (base, arguments) in reached {
                if scopes.reach(ty, &base, arguments)? {
                    pending.push(base);
                }
            }
            if let Some(scope) = scopes.find_mut(&class) {
                scope.written = written;
            }
        }
        Ok(scopes)
    }

    /// The scope of `class`, or the class read's where the walk did not reach
    /// it.
    fn of_class(&self, class: &Bound<'py, PyAny>) -> &Scope<'py> {
        self.find(class).unwrap_or(&self.read)
    }

    /// The scope of `class`, where the walk reached it.
    fn find(&self, class: &Bound<'py, PyAny>) -> Option<&Scope<'py>> {
        core::iter::once(&self.read)
            .chain(&self.bases)
            .find(|scope| scope.class.is(class))
    }

    /// [`Self::find`], to be written to.
    fn find_mut(&mut self, class: &Bound<'py, PyAny>) -> Option<&mut Scope<'py>> {
        core::iter::once(&mut self.read)
            .chain(&mut self.bases)
            .find(|scope| scope.class.is(class))
    }

    /// Hold `class` under `arguments`, and say whether it is new. Reached
    /// again under other arguments, it is refused: a field it declares names
    /// two types.
    fn reach(
        &mut self,
        ty: &Bound<'py, PyType>,
        class: &Bound<'py, PyType>,
        arguments: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<bool> {
        if let Some(seen) = self.bases.iter().find(|scope| scope.class.is(class)) {
            let held = seen.parameters.iter().zip(&seen.arguments).zip(&arguments);
            for ((parameter, held), argument) in held {
                if !held.eq(argument)? {
                    return Err(not_implemented(format!(
                        "{ty} binds {parameter} of {class} to {held} through one base \
                         and to {argument} through another, so a field {class} \
                         declares with it names two types: derive from {class} once",
                        ty = super::declared_name(ty.as_any())?,
                        parameter = super::declared_name(&parameter)?,
                        class = super::declared_name(class.as_any())?,
                        held = summarize(held)?,
                        argument = summarize(argument)?,
                    )));
                }
            }
            return Ok(false);
        }
        let parameters = parameters_of(class)?;
        // A base applied to as many arguments as it has parameters binds them;
        // any other leaves them standing for themselves.
        let arguments = if arguments.len() == parameters.len() {
            arguments
        } else {
            parameters.iter().collect()
        };
        self.bases.push(Scope {
            class: class.clone(),
            parameters,
            arguments,
            written: Vec::new(),
        });
        Ok(true)
    }
}

/// A class's `__parameters__`, or none where it has no such attribute.
fn parameters_of<'py>(class: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyTuple>> {
    match class.getattr_opt(intern!(class.py(), "__parameters__"))? {
        Some(found) => Ok(found.cast_into::<PyTuple>()?),
        None => Ok(PyTuple::empty(class.py())),
    }
}

/// A class a class was written deriving from, with the arguments it was
/// written applied to, or `None` where it was written bare.
type Written<'py> = (Bound<'py, PyType>, Option<Bound<'py, PyTuple>>);

/// The classes `class` was written deriving from: its `__orig_bases__`, or its
/// `__bases__` where it has none, since a `TypedDict`'s bases are not on its
/// `__mro__`.
fn written_bases<'py>(class: &Bound<'py, PyType>) -> PyResult<Vec<Written<'py>>> {
    let py = class.py();
    let generic = forms(py)?.generic.bind(py);
    let written = class
        .getattr(intern!(py, "__dict__"))?
        .call_method1(intern!(py, "get"), (intern!(py, "__orig_bases__"),))?;
    let bases = if written.is_none() {
        class.getattr(intern!(py, "__bases__"))?
    } else {
        written
    };
    let mut found = Vec::new();
    for base in bases.try_iter()? {
        let base = base?;
        if let Ok(derived) = base.cast::<PyType>() {
            found.push((derived.clone(), None));
            continue;
        }
        // An applied generic, `Base[int]`. The function `TypedDict` and
        // `NamedTuple` are written as carries no origin, and `Generic[T]`
        // declares the class's own parameters.
        let Some(origin) = base.getattr_opt(intern!(py, "__origin__"))? else {
            continue;
        };
        let Ok(origin) = origin.cast_into::<PyType>() else {
            continue;
        };
        if origin.is(generic) {
            continue;
        }
        let arguments = base
            .getattr(intern!(py, "__args__"))?
            .cast_into::<PyTuple>()?;
        found.push((origin, Some(arguments)));
    }
    Ok(found)
}

/// The class that declares each field of `ty`, by the field's name.
///
/// The first class on its `__mro__` whose own body annotates the field, which
/// is the class `get_type_hints` takes the hint from; the order is read until
/// every field is placed. A `TypedDict` is the exception: its bases are not on
/// its `__mro__`, and its annotations are its bases' merged with its body's,
/// so a field is a base's where the base holds it under the hint the class
/// read holds, down the bases each was written deriving from (`scopes`). Each
/// class's annotations are read once.
fn declarers<'py>(
    ty: &Bound<'py, PyType>,
    hints: &Bound<'py, PyDict>,
    scopes: &Scopes<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = ty.py();
    let found = PyDict::new(py);
    if !ty.hasattr(intern!(py, "__required_keys__"))? {
        for class in ty.mro().iter() {
            if found.len() >= hints.len() {
                break;
            }
            let Some(own) = own_body(&class)? else {
                continue;
            };
            for name in own.try_iter()? {
                let name = name?;
                if !found.contains(&name)? {
                    found.set_item(name, &class)?;
                }
            }
        }
        return Ok(found);
    }
    let mut read: Vec<(&Bound<'py, PyType>, Bound<'py, PyDict>)> = Vec::new();
    for (name, hint) in hints.iter() {
        let mut at = &scopes.read;
        'down: loop {
            for base in &at.written {
                let held = if let Some((_, hints)) = read.iter().find(|(class, _)| class.is(base)) {
                    hints.get_item(&name)?
                } else {
                    let hints = resolve_type_hints(base)?.cast_into::<PyDict>()?;
                    let held = hints.get_item(&name)?;
                    read.push((base, hints));
                    held
                };
                if let Some(held) = held
                    && held.eq(&hint)?
                    && let Some(scope) = scopes.find(base)
                {
                    at = scope;
                    continue 'down;
                }
            }
            break;
        }
        found.set_item(name, &at.class)?;
    }
    Ok(found)
}

/// What `class` annotates in its own body, read as [`annotations_as_written`]
/// reads a class's own annotations, or `None` where it annotates nothing.
fn own_body<'py>(class: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if annotates_nothing(class) {
        return Ok(None);
    }
    let py = class.py();
    #[cfg(Py_3_14)]
    let own = own_annotations(class, &evaluation(py)?.own_annotations)?;
    #[cfg(not(Py_3_14))]
    let own = class.getattr(intern!(py, "__dict__"))?.call_method1(
        intern!(py, "get"),
        (intern!(py, "__annotations__"), PyDict::new(py)),
    )?;
    // A class with no annotations of its own, read as `get_type_hints` reads it.
    #[cfg(not(Py_3_14))]
    if own.is_instance(evaluation(py)?.getset_descriptor.bind(py))? {
        return Ok(None);
    }
    Ok(Some(own))
}

/// The objects [`annotations_as_written`] reads a class and its annotations
/// through, resolved once per interpreter.
struct Evaluation {
    /// `(typing._GenericAlias, types.GenericAlias, types.UnionType)`: the
    /// classes whose `__args__` `typing._eval_type` descends into. Every other
    /// object it returns as it was given.
    aliases: Py<PyTuple>,
    /// `types.GenericAlias`, the one alias `_eval_type` rebuilds whatever its
    /// arguments are.
    generic_alias: Py<PyAny>,
    /// `types.GetSetDescriptorType`, which `get_type_hints` reads as a class
    /// with no annotations of its own, below 3.14, where a class's namespace
    /// holds them.
    #[cfg(not(Py_3_14))]
    getset_descriptor: Py<PyAny>,
    /// How `get_type_hints` reads one class's own annotations from 3.14.
    #[cfg(Py_3_14)]
    own_annotations: OwnAnnotations,
}

/// The two readings [`own_annotations`] takes a class's annotations through:
/// `type`'s own `__annotations__` descriptor, which `annotationlib` asks
/// first, and `annotationlib.get_annotations`, which asks it.
#[cfg(Py_3_14)]
struct OwnAnnotations {
    /// `type.__dict__["__annotations__"].__get__`, which
    /// `annotationlib._BASE_GET_ANNOTATIONS` names.
    descriptor: Py<PyAny>,
    /// `annotationlib.get_annotations`.
    call: Py<PyAny>,
}

static EVALUATION: PyOnceLock<Evaluation> = PyOnceLock::new();

fn evaluation(py: Python<'_>) -> PyResult<&'static Evaluation> {
    EVALUATION.get_or_try_init(py, || {
        let types = py.import("types")?;
        let generic_alias = types.getattr("GenericAlias")?;
        let aliases = PyTuple::new(
            py,
            [
                py.import("typing")?.getattr("_GenericAlias")?,
                generic_alias.clone(),
                types.getattr("UnionType")?,
            ],
        )?;
        Ok(Evaluation {
            aliases: aliases.unbind(),
            generic_alias: generic_alias.unbind(),
            #[cfg(not(Py_3_14))]
            getset_descriptor: types.getattr("GetSetDescriptorType")?.unbind(),
            #[cfg(Py_3_14)]
            own_annotations: OwnAnnotations {
                descriptor: py
                    .get_type::<PyType>()
                    .getattr("__dict__")?
                    .get_item("__annotations__")?
                    .getattr("__get__")?
                    .unbind(),
                call: py
                    .import("annotationlib")?
                    .getattr("get_annotations")?
                    .unbind(),
            },
        })
    })
}

/// How deep an annotation is walked before the walk declines. `_eval_type`
/// recurses without a bound of its own and meets the interpreter's recursion
/// limit; a walk on the Rust stack stops well before its own.
const MAX_ANNOTATION_DEPTH: usize = 64;

/// A class's type hints read as `typing.get_type_hints(ty, include_extras=True)`
/// returns them, or `None` where that call would change a value.
///
/// The reading is the call's own, step for step. Each base in reversed
/// `__mro__` contributes its own annotations -- from its namespace's
/// `__annotations__` below 3.14, as `annotationlib.get_annotations` reads them
/// from 3.14 (`own_annotations`) -- a later name replacing an earlier one in
/// place, and `None` read as `type(None)`. What the call adds is `_eval_type`
/// over each value, and that returns the value unchanged unless [`evaluates`]
/// says otherwise. A class marked `__no_type_check__` is left to the call,
/// which answers `{}` for it. A base [`annotates_nothing`] names contributes
/// nothing on any release, and is not asked.
///
/// **Equal to the call's answer, and the objects themselves.** Two values
/// differ only where `_eval_type` rebuilds a builtin alias, and every such
/// alias is declined. `annotations_as_written_are_the_hints_get_type_hints_returns`
/// in `build/tests.rs` holds the reading to the call, value for value and in
/// order, and `tests/test_classes.py` holds the compiled validators to it on
/// every interpreter the matrix runs.
pub(super) fn annotations_as_written<'py>(
    ty: &Bound<'py, PyType>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let py = ty.py();
    let names = evaluation(py)?;
    if let Some(flag) = optional_attribute(ty, intern!(py, "__no_type_check__"))?
        && flag.is_truthy()?
    {
        return Ok(None);
    }
    let mut hints = PyDict::new(py);
    let mro = ty.getattr(intern!(py, "__mro__"))?;
    for base in mro.cast::<PyTuple>()?.iter().rev() {
        if annotates_nothing(&base) {
            continue;
        }
        #[cfg(Py_3_14)]
        let own = own_annotations(&base, &names.own_annotations)?;
        // `base.__dict__.get('__annotations__', {})`, the call's own spelling,
        // so a namespace a metaclass supplies is read the way the call reads it.
        #[cfg(not(Py_3_14))]
        let own = base.getattr(intern!(py, "__dict__"))?.call_method1(
            intern!(py, "get"),
            (intern!(py, "__annotations__"), PyDict::new(py)),
        )?;
        #[cfg(not(Py_3_14))]
        if own.is_instance(names.getset_descriptor.bind(py))? {
            continue;
        }
        // `get_type_hints` reads any mapping through `.items()`; a dict is the
        // one this reading takes, and anything else is left to the call.
        let Ok(own) = own.cast::<PyDict>() else {
            return Ok(None);
        };
        let mut holds_none = false;
        for (_, value) in own.iter() {
            if value.is_instance_of::<PyString>() || evaluates(&value, names, 0)? {
                return Ok(None);
            }
            holds_none |= value.is_none();
        }
        // The first base that says anything is copied whole, which is one
        // table copy against an insertion per name; a later one merges in, a
        // name it repeats keeping its place.
        if hints.is_empty() {
            hints = own.copy()?;
        } else {
            hints.update(own.as_mapping())?;
        }
        if holds_none {
            for (name, value) in own.iter() {
                if value.is_none() {
                    hints.set_item(name, py.get_type::<PyNone>())?;
                }
            }
        }
    }
    Ok(Some(hints))
}

/// `annotationlib.get_annotations(base)`, which is how `get_type_hints` reads
/// a class's own annotations from 3.14, without the call where the call's
/// first question answers.
///
/// For a class, the call asks `type`'s own `__annotations__` descriptor and,
/// handed a `dict`, returns a copy of it, and that much is asked here: the
/// descriptor is C, where the call is two Python functions -- a `match` on its
/// format, an `isinstance`, a `try` -- run for every class a build compiles.
/// Anything else the descriptor answers -- `None`, a `dict` subclass, an error
/// -- is the call's to answer, and the call asks the descriptor again and
/// answers it: a class whose annotations raise is read twice before the error
/// leaves, which every reading that declines does anyway, `get_type_hints`
/// asking again behind it.
///
/// Compiled from 3.14, as the struct it reads is: the extension is built for
/// one interpreter, and below 3.14 there is no call to stand in for, since
/// `get_type_hints` reads a class's namespace there.
#[cfg(Py_3_14)]
fn own_annotations<'py>(
    base: &Bound<'py, PyAny>,
    reading: &OwnAnnotations,
) -> PyResult<Bound<'py, PyAny>> {
    let py = base.py();
    if let Ok(found) = reading.descriptor.bind(py).call1((base,))
        && let Ok(found) = found.cast_exact::<PyDict>()
    {
        return Ok(found.copy()?.into_any());
    }
    reading.call.bind(py).call1((base,))
}

/// Whether `base` is a builtin class whose own annotations are none on every
/// release: `object`, and the builtins a class's `__mro__` most often holds
/// beside it -- `tuple` under a `NamedTuple`, `dict` under a `TypedDict`.
///
/// Each is a static type, which refuses an assignment to any name, and holds
/// neither `__annotations__` nor `__annotate__` in its namespace, so what
/// either reading takes from it is the empty table. Read anyway, it is a call
/// into `annotationlib.get_annotations` from 3.14, where a static type's
/// `__annotations__` and `__annotate__` answer by raising -- an
/// `AttributeError` with its message formatted, then dropped -- for every class
/// a build compiles; and a namespace lookup below 3.14.
pub(super) fn annotates_nothing(base: &Bound<'_, PyAny>) -> bool {
    const STATIC: [fn(Python<'_>) -> Bound<'_, PyType>; 12] = [
        PyAny::type_object,
        PyTuple::type_object,
        PyDict::type_object,
        PyList::type_object,
        PyInt::type_object,
        PyFloat::type_object,
        PyString::type_object,
        PyBytes::type_object,
        PySet::type_object,
        PyFrozenSet::type_object,
        PyBaseException::type_object,
        PyException::type_object,
    ];
    STATIC.iter().any(|class| base.is(class(base.py())))
}

/// Whether `typing._eval_type` would hand back anything other than `value`./// Whether `typing._eval_type` would hand back anything other than `value`.
///
/// It resolves a forward reference, and descends only into the `__args__` of
/// the three alias classes. A builtin `types.GenericAlias` it rebuilds as
/// `origin[args]` from 3.11: a string argument becomes a forward reference, an
/// unpacked alias becomes `Unpack[...]`, and a `collections.abc.Callable`'s
/// argument list is re-nested. Each of those is a change, so each is `true`. A
/// string under any other alias -- a `Literal`'s value -- is returned as it is.
fn evaluates(value: &Bound<'_, PyAny>, names: &Evaluation, depth: usize) -> PyResult<bool> {
    let py = value.py();
    if depth > MAX_ANNOTATION_DEPTH {
        return Ok(true);
    }
    // A class is the common annotation, and `isinstance` answers `false` for
    // it only after looking up its `__class__`. For an object whose type is
    // exactly `type` that lookup returns `type` whatever the class defines --
    // the metatype's own descriptor wins -- and `type` derives from none of
    // the classes asked below, so every one of them answers `false`.
    if value.get_type().is(py.get_type::<PyType>()) {
        return Ok(false);
    }
    if let Some(forward_ref) = &forms(py)?.forward_ref
        && value.is_instance(forward_ref.bind(py))?
    {
        return Ok(true);
    }
    if !value.is_instance(names.aliases.bind(py))? {
        return Ok(false);
    }
    let args = value.getattr(intern!(py, "__args__"))?;
    let args = args.cast::<PyTuple>()?;
    if value.is_instance(names.generic_alias.bind(py))?
        && (is_truthy_attr(value, intern!(py, "__unpacked__"))?
            || value
                .getattr(intern!(py, "__origin__"))?
                .is(forms(py)?.callable.bind(py))
            || args.iter().any(|arg| arg.is_instance_of::<PyString>()))
    {
        return Ok(true);
    }
    for arg in args.iter() {
        if evaluates(&arg, names, depth + 1)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Read a record field name as Rust text, refusing a key that is not valid
/// Unicode (one carrying a lone surrogate). A field name is stored as UTF-8 and
/// matched against dict keys as UTF-8, so a surrogate key cannot round-trip;
/// refusing it at build time turns silent corruption — a lossy replacement that
/// makes the field unmatchable — into an explicit error.
///
/// The name is handed back borrowed from the Python string rather than copied
/// into one of its own. Every caller turns it into the shared name a field
/// holds, and that conversion copies the text anyway, so owning it here would
/// buy an allocation per field and free it one line later.
pub(super) fn field_name<'a>(name: &'a Bound<'_, PyString>) -> PyResult<&'a str> {
    name.to_str().map_err(|_| {
        PyValueError::new_err(
            "a record key must be valid Unicode; a field name cannot contain a lone surrogate",
        )
    })
}

/// Build the record a `TypedDict` denotes: its keys, and what it says about the
/// ones it does not name.
///
/// **Open unless the class says otherwise**, which is the set the typing spec
/// assigns it -- "By default, `TypedDict`s are open". `Validator(TD)` reads an
/// annotation whose meaning is fixed elsewhere, and reading it as a narrower set
/// is a deviation a caller has no way to see, because the class carries no mark
/// of it. The dict-literal form `{"a": int}` stays closed: it is this library's
/// own spelling, and a schema written as a *shape* means that shape
/// (`docs/dev/01-schema-ir.md`, "What a `TypedDict` denotes").
///
/// `ReadOnly` needs no arm here. It constrains writers and a value has none, and
/// `get_type_hints` has already resolved it away.
pub(super) fn build_typed_dict<'py>(
    ty: &Bound<'py, PyType>,
    applied: Option<&Bound<'py, PyTuple>>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let _open = ClassScope::enter(ty, applied)?;
    let hints = bound_hints(ty, applied)?;
    let required = ty.getattr(intern!(ty.py(), "__required_keys__"))?;
    let mut fields = Vec::with_capacity(hints.len());
    for (name, hint) in hints.iter() {
        let (stated, read) = qualified_required(&hint)?;
        fields.push(Field {
            name: field_name(&name.str()?)?.into(),
            schema: build_schema(&read, lits, defs)?,
            // The qualifier on the *resolved* hint wins over the class's key
            // sets. CPython fills `__required_keys__` when the class is created,
            // from the annotations as written -- and under `from __future__
            // import annotations` those are strings, so `NotRequired[...]` is
            // invisible to it, and read there every optional key would compile
            // required.
            // `get_type_hints` resolves the string and keeps the qualifier, so
            // reading it there is reading what the author wrote.
            required: match stated {
                Some(stated) => stated,
                None => required.contains(&name)?,
            },
        });
    }
    Ok(Schema::keyed_map(fields, unnamed_keys(ty, lits, defs)?))
}

/// Whether a resolved hint states its own required-ness, and which.
///
/// `Required[T]` and `NotRequired[T]` say it; every other form, `ReadOnly[T]`
/// included, says nothing and leaves the class's key sets to answer. A qualifier
/// may wrap another -- `ReadOnly[NotRequired[T]]` is legal -- and `Annotated`
/// may wrap either, so the search goes through the ones that carry no answer
/// rather than stopping at the first.
///
/// It reads on below the first that answers, because the typing spec makes two
/// spellings of one item an error and they do not name the same set:
/// `Required[NotRequired[T]]` is a key the runtime's key sets read as required
/// and pydantic as optional, so it is refused, in either order and with an
/// `Annotated` between. A qualifier nested in itself, `Required[Required[T]]`,
/// is an error that names one set, and is read as the one qualifier.
///
/// **The second half is the hint below the qualifiers at its top**, which is
/// what the field's type is built from: the builder reads a qualifier as the
/// type it wraps, and reaches one only after asking the hint every form before
/// it, so handing it the type this walk has already read saves that and pays
/// for the walk's look below the first qualifier. Nothing under an `Annotated`
/// is taken off, since the alias carries the field's metadata, and a qualifier
/// holding other than one argument is left for the builder to refuse.
pub(super) fn qualified_required<'py>(
    hint: &Bound<'py, PyAny>,
) -> PyResult<(Option<bool>, Bound<'py, PyAny>)> {
    let py = hint.py();
    let forms = forms(py)?;
    let mut current = hint.clone();
    let mut stated = None;
    let mut below = hint.clone();
    let mut at_the_top = true;
    for _ in 0..MAX_BUILD_DEPTH {
        // Asked rather than tried: a field that carries no qualifier is the
        // common one, and an attribute that is absent answers by *raising* --
        // an exception built, thrown and dropped per field. `getattr_opt`
        // reads the same absence without one.
        let Some(origin) = optional_attribute(&current, intern!(py, "__origin__"))? else {
            return Ok((stated, below));
        };
        // An `Annotated` alias's origin is the type it annotates, which may be
        // the qualifier: `Annotated[NotRequired[int], Ge(0)]` says the key is
        // optional one level down. Stopping here read every annotated key from
        // the class's key sets, which under `from __future__ import
        // annotations` CPython fills from strings and gets wrong both ways. The
        // alias is known by its class: asked by attribute, `__metadata__` is a
        // raise in `typing`'s `__getattr__` on every other alias a field is
        // written with. A class it annotates carries no qualifier.
        if current.get_type().is(forms.annotated_alias.bind(py)) {
            if origin.is_instance_of::<PyType>() {
                return Ok((stated, below));
            }
            current = origin;
            at_the_top = false;
            continue;
        }
        let mut says = None;
        for (marker, answer) in [(&forms.required, true), (&forms.not_required, false)] {
            if let Some(marker) = marker
                && origin.is(marker.bind(py))
            {
                says = Some(answer);
            }
        }
        // `typing_extensions`' spellings, which are forms: the origin of an
        // `Annotated` or a generic field is a class, and asks nothing.
        if says.is_none() && !origin.is_instance_of::<PyType>() {
            if is_extension(&origin, |held| &held.required)? {
                says = Some(true);
            } else if is_extension(&origin, |held| &held.not_required)? {
                says = Some(false);
            }
        }
        if let Some(again) = says {
            if *stated.get_or_insert(again) != again {
                return Err(not_implemented(format!(
                    "{} says the key is both required and not required, which \
                     the typing spec makes an error and which names no one \
                     record: write one of Required and NotRequired",
                    summarize(hint)?
                )));
            }
        } else if !is_field_qualifier(&origin)? {
            return Ok((stated, below));
        }
        let Some(args) = current.getattr_opt(intern!(py, "__args__"))? else {
            return Ok((stated, below));
        };
        let Ok(inner) = args.get_item(0) else {
            return Ok((stated, below));
        };
        if at_the_top && args.len()? == 1 {
            below = inner.clone();
        }
        // A class carries no qualifier, and it is what a qualifier most often
        // wraps: answered by its type, without asking it for an origin.
        if inner.is_instance_of::<PyType>() {
            return Ok((stated, below));
        }
        current = inner;
    }
    Ok((stated, below))
}

/// Whether `__extra_items__` says its author gave no `extra_items`.
///
/// PEP 728 fills the attribute either way: with the type the author wrote, or
/// with a `NoExtraItems` sentinel to say there was none. Each implementation
/// writes its own -- `typing` from 3.15, `typing_extensions` on every release,
/// two objects until 3.15 -- so the sentinel is the one the module defining the
/// class's metaclass carries, which is the implementation that wrote it. The
/// sentinel is not a type, and read as one it turns the open default into a
/// record admitting exactly the sentinel.
///
/// An implementation with no sentinel predates it and wrote `None` for "none
/// given". With one, `None` is a type the author gave: extra values are `None`.
fn gave_no_extra_items(ty: &Bound<'_, PyType>, extra: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(match no_extra_items(ty)? {
        Some(sentinel) => extra.is(&sentinel),
        None => extra.is_none(),
    })
}

/// The `NoExtraItems` sentinel of the implementation that built `ty`, if it has
/// one.
///
/// Looked up in `sys.modules` rather than imported: the class exists, so its
/// metaclass's module is loaded, and a lookup runs no module's code. The table
/// is the one [`loaded_modules`] holds, since every `TypedDict` from 3.15 says
/// what it gives its extra keys and asks this once per class compiled.
fn no_extra_items<'py>(ty: &Bound<'py, PyType>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = ty.py();
    let module = ty.get_type().getattr(intern!(py, "__module__"))?;
    match loaded_modules(py)?.get_item(module)? {
        Some(module) => module.getattr_opt(intern!(py, "NoExtraItems")),
        None => Ok(None),
    }
}

/// What a `TypedDict` says about the keys it does not name.
///
/// `closed=True` shuts them and `extra_items=T` gives them a type -- PEP 728,
/// which the typing spec carries. Both are read off the class, where the
/// implementation that built it puts them, and off its bases where the class
/// says neither: the spec's open default holds "except when inheriting from
/// another `TypedDict` that is not open". A `TypedDict` built without PEP 728
/// cannot have said either, and the spec's default is what is left.
pub(super) fn unnamed_keys(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Vec<MapClause>> {
    Ok(match inherited_tail(ty)? {
        Tail::Closed => Vec::new(),
        // A `TypedDict`'s keys are strings, so the type it gives the extra ones
        // governs the string keys and leaves no other kind admitted.
        Tail::Typed(extra) => vec![MapClause {
            key: Schema::Str,
            value: build_schema(&extra, lits, defs)?,
        }],
        // A `TypedDict`'s keys are strings -- the spec relates one to
        // `Mapping[str, object]` and to nothing wider -- so being open is being
        // open to further *string* keys, not to keys of every kind.
        Tail::Open => vec![MapClause {
            key: Schema::Str,
            value: Schema::ANYTHING,
        }],
    })
}

/// What a `TypedDict` says about the keys it does not name.
enum Tail<'py> {
    /// `closed=True`: no other key.
    Closed,
    /// `extra_items=T`: every other string key maps into `T`.
    Typed(Bound<'py, PyAny>),
    /// Open over the string keys, which the spec reads as `ReadOnly[object]`.
    Open,
}

/// What `ty` says about its unnamed keys, or `None` where it says nothing.
///
/// The runtime writes both attributes on every class, with the class's own
/// keywords: `__closed__` is `None` and `__extra_items__` the `NoExtraItems`
/// sentinel where the class gave neither, whatever its bases gave.
fn stated_tail<'py>(ty: &Bound<'py, PyType>) -> PyResult<Option<Tail<'py>>> {
    let py = ty.py();
    if let Some(flag) = optional_attribute(ty, intern!(py, "__closed__"))?
        && !flag.is_none()
    {
        return Ok(Some(if flag.is_truthy()? {
            Tail::Closed
        } else {
            Tail::Open
        }));
    }
    if let Some(extra) = optional_attribute(ty, intern!(py, "__extra_items__"))?
        && !gave_no_extra_items(ty, &extra)?
    {
        return Ok(Some(Tail::Typed(extra)));
    }
    Ok(None)
}

/// What `ty` has for its unnamed keys: its own, or the first a `TypedDict` base
/// states, depth first in the order the bases are written; open where none does.
///
/// The bases are `__orig_bases__`, since a `TypedDict`'s `__bases__` is `dict`
/// alone. A base written generic, `Base[int]`, is read through its origin. The
/// walk is bounded like the build, so a hierarchy past the bound reads open.
fn inherited_tail<'py>(ty: &Bound<'py, PyType>) -> PyResult<Tail<'py>> {
    let py = ty.py();
    let mut pending = vec![ty.clone()];
    for _ in 0..MAX_BUILD_DEPTH {
        let Some(class) = pending.pop() else {
            break;
        };
        if let Some(tail) = stated_tail(&class)? {
            return Ok(tail);
        }
        let Some(bases) = class.getattr_opt(intern!(py, "__orig_bases__"))? else {
            continue;
        };
        let mut typed = Vec::new();
        for base in bases.try_iter()? {
            let base = base?;
            let base = match base.getattr_opt(intern!(py, "__origin__"))? {
                Some(origin) => origin,
                None => base,
            };
            if let Ok(base) = base.cast_into::<PyType>()
                && base.hasattr(intern!(py, "__required_keys__"))?
            {
                typed.push(base);
            }
        }
        pending.extend(typed.into_iter().rev());
    }
    Ok(Tail::Open)
}

/// The attribute names a class declares, in declaration order.
///
/// Not every annotation on a dataclass names an attribute of its instances:
/// `ClassVar` annotates the class, and `InitVar` names a constructor parameter
/// the instance does not keep. Reading the hints alone asks an instance for
/// attributes it cannot have, which is a schema no instance of the class
/// satisfies — the empty set wearing the class's name. Each kind of class keeps
/// its own list of what it declares, so that list is what is read: a dataclass's
/// `fields()` and a named tuple's `_fields`. A field declared `init=False` is on
/// the instance and stays.
pub(super) fn declared_fields<'py>(ty: &Bound<'py, PyType>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = ty.py();
    // Both questions go through the held handles. Spelled as a module import
    // and two `call_method1`s, this asked `sys.modules` for `dataclasses` and
    // decoded three names per class node -- and asked `is_dataclass` a second
    // time, the caller having just had its answer.
    if is_dataclass(ty)? {
        let reading = DATACLASS_READING.get_or_try_init(py, || {
            let module = py.import("dataclasses")?;
            Ok::<_, PyErr>(DataclassReading {
                fields: module.getattr("fields")?.unbind(),
                kept: module.getattr_opt("_FIELD")?.map(Bound::unbind),
            })
        })?;
        let fields = match fields_as_declared(ty, reading)? {
            Some(fields) => fields,
            None => reading
                .fields
                .bind(py)
                .call1((ty,))?
                .try_iter()?
                .collect::<PyResult<_>>()?,
        };
        return fields
            .iter()
            .map(|field| field.getattr(intern!(py, "name")))
            .collect();
    }
    ty.getattr(intern!(py, "_fields"))?.try_iter()?.collect()
}

/// A dataclass's fields as `dataclasses.fields(ty)` returns them, or `None`
/// where that call should answer.
///
/// The call is `tuple(f for f in ty.__dataclass_fields__.values() if
/// f._field_type is _FIELD)` on every interpreter the matrix runs, 3.10 to 3.15
/// and `PyPy`, and running that generator in Python is 15% of compiling a
/// fifty-field dataclass. The same steps are taken here, in its order -- the
/// table's values, each asked its `_field_type` and kept by identity with the
/// marker -- and where one does not read as the call's own, the call answers: a
/// table that is not exactly a `dict`, whose `values` could be anyone's, and a
/// module with no marker to compare against.
/// `a_dataclass_declares_what_fields_returns` in `build/interpreter.rs` holds
/// the reading to the call.
fn fields_as_declared<'py>(
    ty: &Bound<'py, PyType>,
    reading: &DataclassReading,
) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    let py = ty.py();
    let Some(kept) = &reading.kept else {
        return Ok(None);
    };
    let Some(table) = ty.getattr_opt(intern!(py, "__dataclass_fields__"))? else {
        return Ok(None);
    };
    let Ok(table) = table.cast_exact::<PyDict>() else {
        return Ok(None);
    };
    // A snapshot of the values, taken once per class: a live iteration panics if
    // the table changes size under it, and a field is asked for an attribute.
    let values = table.values();
    let mut fields = Vec::with_capacity(values.len());
    for field in values.iter() {
        if field.getattr(intern!(py, "_field_type"))?.is(kept.bind(py)) {
            fields.push(field);
        }
    }
    Ok(Some(fields))
}

/// Build the schema of a class with declared attributes: the meet of its
/// `isinstance` atom and a record of its fields, whose types come from the
/// resolved hints. Every attribute an instance declares is required, because an
/// instance carries it.
///
/// A class with no annotated field is the atom alone. The meet is what the form
/// means, and spelling it out is what lets each half relate on its own; the
/// surface does not change, because `render` reads the pair back as the class
/// name and the walk stops at the failing conjunct, which is what
/// `docs/dev/01-schema-ir.md` records under "What a class with attributes is, on
/// the surface".
pub(super) fn build_object<'py>(
    ty: &Bound<'py, PyType>,
    applied: Option<&Bound<'py, PyTuple>>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let _open = ClassScope::enter(ty, applied)?;
    let hints = bound_hints(ty, applied)?;
    let hints = &hints;
    let instance = Schema::Instance(lits.intern_class(ty.as_any()));
    // A named tuple's positions are its attributes, so the record of them is
    // not built beside the positions: it would say the same of the same values
    // (see `named_tuple_positions`), and a record built and dropped is still
    // counted by the node guard, which holds every node a build has read.
    if let Some(positions) = named_tuple_positions(ty, hints, lits, defs)? {
        return Ok(Schema::meet([instance, positions]));
    }
    let declared = declared_fields(ty)?;
    let mut fields = Vec::with_capacity(declared.len());
    for name in declared {
        // A name the hints do not carry is unannotated -- a `collections`
        // namedtuple's fields are the case -- so there is no type to check and
        // the class's own isinstance test is the whole of it.
        let Some(hint) = hints.get_item(&name)? else {
            continue;
        };
        // A class is never a `Final`, and most fields are one.
        let hint = if hint.is_instance_of::<PyType>() {
            hint
        } else {
            field_hint(hint, &name)?
        };
        fields.push(Attribute {
            name: field_name(&name.str()?)?.into(),
            schema: build_schema(&hint, lits, defs)?,
        });
    }
    if fields.is_empty() {
        return Ok(instance);
    }
    Ok(Schema::meet([instance, Schema::attr_record(fields)]))
}

/// The hint a field's value is read against: the hint, or the type a `Final`
/// wraps.
///
/// The typing spec makes `x: Final[int]` in a dataclass body a field `x` that
/// holds an `int` and is not assigned to after `__init__`. `Final` says the
/// attribute is not rebound, which is not a question about the value it holds
/// -- the reading `ReadOnly` already has on a `TypedDict` key. A bare `Final`
/// names no type, since a checker infers one from the default, and is left to
/// the dispatch, which refuses it. The alias's own `__origin__` is read rather
/// than calling `typing.get_origin`, so a field pays no Python call for it.
///
/// `Annotated` wraps an annotation expression without changing what it means,
/// in the spec's words, so `Annotated[Final[int], m]` is a final field holding
/// `Annotated[int, m]`, and `dataclasses` makes it a field too: it is read as
/// that. `Annotated[ClassVar[int], m]` is a class variable to the spec and a
/// field to `dataclasses`, which reads only an outermost `ClassVar`, so the
/// two disagree about whether there is a field at all, and the class is
/// refused naming the spelling each reading has.
///
/// A class is never a `Final`, and the caller answers one without calling
/// this: looking a missing `__origin__` up on `int` walks its whole `__mro__`,
/// and a fifty-field dataclass would pay 7% of its build for it, and the call
/// itself, made for every field, another half percent.
fn field_hint<'py>(
    hint: Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = hint.py();
    let forms = forms(py)?;
    let Some(origin) = hint.getattr_opt(intern!(py, "__origin__"))? else {
        return Ok(hint);
    };
    if origin.is(forms.final_qualifier.bind(py)) {
        return hint.getattr(intern!(py, "__args__"))?.get_item(0);
    }
    if !hint.get_type().is(forms.annotated_alias.bind(py)) {
        return Ok(hint);
    }
    // An `Annotated` over a class qualifies nothing, and a class asked for an
    // `__origin__` it lacks walks its `__mro__`.
    if origin.is_instance_of::<PyType>() {
        return Ok(hint);
    }
    let Some(qualifier) = origin.getattr_opt(intern!(py, "__origin__"))? else {
        return Ok(hint);
    };
    if qualifier.is(forms.final_qualifier.bind(py)) {
        let mut parts = vec![origin.getattr(intern!(py, "__args__"))?.get_item(0)?];
        for item in hint.getattr(intern!(py, "__metadata__"))?.try_iter()? {
            parts.push(item?);
        }
        return forms.annotated.bind(py).get_item(PyTuple::new(py, parts)?);
    }
    if qualifier.is(forms.class_var.bind(py)) {
        return Err(not_implemented(format!(
            "field {} is annotated {}, a class variable to the typing spec and \
             a field to dataclasses, which reads only an outermost ClassVar: \
             write ClassVar[Annotated[T, ...]] for a class variable, or \
             Annotated[T, ...] for a field",
            summarize(name)?,
            summarize(&hint)?
        )));
    }
    Ok(hint)
}

/// The tuple shape a named tuple's fields lay out, or `None` for a class that
/// lays out none.
///
/// A named tuple's positions *are* its attributes: an instance is a tuple of
/// exactly as many elements as the class declares, the i-th holding the i-th
/// field. The class says so and the frontend reads it, so the schema says it
/// too -- otherwise the set the schema denotes is wider than the set the class
/// has, admitting an instance whose attributes are integers and whose positions
/// are anything, which no instance is.
///
/// It is the shape *instead of* an attribute record rather than beside one,
/// because the two would say the same thing about the same values: a wrong
/// field would be reported twice, and every passing value would be read twice.
/// The shape is the more precise of the pair -- it carries the arity as well as
/// the types -- and every rule that reads a shape reads it, so a relation
/// between a named tuple and the tuple it lays out is decided structurally
/// rather than declined. What the attribute record carried and this does not is
/// the field's *name* in a failure's path, which becomes its position.
///
/// A field the annotations do not carry -- a `collections.namedtuple` has none
/// -- takes `anything` at its position, which is the arity without the types.
pub(super) fn named_tuple_positions(
    ty: &Bound<'_, PyType>,
    hints: &Bound<'_, PyDict>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Option<Schema>> {
    if !(ty.is_subclass_of::<PyTuple>()? && ty.hasattr(intern!(ty.py(), "_fields"))?) {
        return Ok(None);
    }
    let mut positions = Vec::new();
    for name in declared_fields(ty)? {
        positions.push(match hints.get_item(&name)? {
            Some(hint) => build_schema(&hint, lits, defs)?,
            None => Schema::ANYTHING,
        });
    }
    Ok(Some(Schema::tuple(SeqShape::fixed(positions))))
}
