//! A refutation from a value the subject's own spelling builds.
//!
//! `t ⊆ s` fails exactly where a value of `t` lies outside `s` [FCB08, §4.1],
//! so one such value refutes the pair whatever either decider can read of the
//! two sets. Where both deciders and the finite reading declined, the relation
//! builds a few small values from the subject's spelling -- each constant it
//! names, a value of each scalar kind it names, a sequence, a set or a dict of
//! what its items and fields build, a reference read a few levels down -- and
//! walks each through both sides. A value the subject admits and the supertype
//! refuses, each walk finishing without a stop, refutes the pair.
//!
//! Beside them it builds values aimed outside the supertype, which is the
//! difference `t ∧ ¬s` the refutation lives in: the negation of the supertype
//! is pushed one constructor in, as not-rewriting pushes it through each
//! operator of a schema [ABC+22, §5.2] -- outside `list[X]` are the other
//! kinds and a list holding an item outside `X`, outside a record a dict that
//! lacks a required key, holds a field outside its type or a key the record
//! closes out, outside a bound a number beside it, outside a complement its
//! operand's members -- so a guess is built to
//! fall outside the supertype rather than hoped to.
//!
//! The values are guesses, not claims. A guess outside the subject is refused
//! by the subject's own walk and decides nothing, and one inside the supertype
//! by the supertype's, so the builder may guess loosely: a meet guesses what
//! each member guesses, and the values outside a union what each member
//! misses. A pair no guess refutes stays undecided, since nothing here proves
//! an inclusion. The walks read both sides as a proof would only where no atom
//! of either runs code a second call may answer differently
//! ([`walked_alike`](crate::finite::walked_alike)), as the finite reading's do.

use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyList, PySet, PyString, PyTuple,
};
use valgebra_core::{CollKind, Constraint, Field, MapClause, Schema, SeqKind, SeqShape};

/// How many levels of containers a guess is built through: a sequence, a set
/// and a record each take one, and a reference none, since every definition is
/// contractive and so names itself only from inside a container.
///
/// **This bound is cost.** Each level multiplies the guesses a container
/// builds from the level below by [`WITNESS_FANOUT`] at most; three hold a list
/// of a list of a scalar, the shape a refutation past one unfolding of a
/// recursive supertype most often needs.
pub(crate) const WITNESS_DEPTH: usize = 3;

/// The guesses a node of the spelling hands the node above it, but for a union
/// or a meet, which hands up its members' guesses to [`WITNESS_VALUES`].
///
/// **This bound is cost**, as [`WITNESS_DEPTH`] is: a container builds its
/// guesses from its items', so each level multiplies by this at most. A union
/// hands up the first guess of each member before the second of any, so a wide
/// union keeps one guess of each kind it names before it keeps two of one.
pub(crate) const WITNESS_FANOUT: usize = 8;

/// The guesses a relation walks through both sides.
///
/// **This bound is cost.** Each guess is built as a Python object and walked
/// twice, where the deciders spend no step; it bounds a relation the readings
/// before it left undecided to a hundred and twenty-eight short walks.
pub(crate) const WITNESS_VALUES: usize = 64;

/// The fields or positions a record or a fixed sequence may hold for a guess to
/// be built of it. **This bound is cost**: a guess of `n` fields is `n`
/// objects built before either walk reads one.
const WITNESS_PARTS: usize = 64;

/// At most [`WITNESS_VALUES`] small values, each a guess at a member of
/// `subject` outside `supertype`: the subject's own guesses ([`guesses`])
/// beside the values aimed outside the supertype ([`misses`]), the first of
/// each before the second of either.
pub(crate) fn guesses_against<'py>(
    py: Python<'py>,
    subject: &Schema,
    supertype: &Schema,
    literals: &[Py<PyAny>],
    definitions: &[Schema],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut all = interleaved(&[
        guesses(py, subject, literals, definitions)?,
        misses(py, supertype, literals, definitions)?,
    ]);
    all.truncate(WITNESS_VALUES);
    Ok(all)
}

/// At most [`WITNESS_VALUES`] small values built from `schema`'s spelling,
/// each a guess at a member of it.
pub(crate) fn guesses<'py>(
    py: Python<'py>,
    schema: &Schema,
    literals: &[Py<PyAny>],
    definitions: &[Schema],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let reading = Reading {
        py,
        literals,
        definitions,
    };
    let mut all = reading.guessed(schema, WITNESS_DEPTH)?;
    all.truncate(WITNESS_VALUES);
    Ok(all)
}

/// At most [`WITNESS_VALUES`] small values built from `schema`'s spelling,
/// each a guess at a value outside it [ABC+22, §5.2].
pub(crate) fn misses<'py>(
    py: Python<'py>,
    schema: &Schema,
    literals: &[Py<PyAny>],
    definitions: &[Schema],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let reading = Reading {
        py,
        literals,
        definitions,
    };
    reading.outside(schema, WITNESS_DEPTH)
}

/// What a guess is built from: the interpreter, the constants pool and the
/// definitions a reference names.
struct Reading<'py, 'a> {
    py: Python<'py>,
    literals: &'a [Py<PyAny>],
    definitions: &'a [Schema],
}

impl<'py> Reading<'py, '_> {
    /// At most [`WITNESS_FANOUT`] guesses at a member of `schema`, built
    /// through at most `depth` more levels of containers and references, a
    /// complement guessed as what its operand misses [ABC+22, §5.2].
    fn guessed(&self, schema: &Schema, depth: usize) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let py = self.py;
        let mut guesses = match schema {
            Schema::Nothing
            | Schema::Instance(_)
            | Schema::AttrRecord { .. }
            | Schema::SelfRef(_) => Vec::new(),
            Schema::NoneType => vec![py.None().into_bound(py)],
            Schema::Bool => vec![
                PyBool::new(py, false).to_owned().into_any(),
                PyBool::new(py, true).to_owned().into_any(),
            ],
            Schema::Int => vec![int(py, 0)?, int(py, 1)?],
            Schema::Float => vec![PyFloat::new(py, 0.5).into_any()],
            Schema::Str => vec![
                PyString::new(py, "").into_any(),
                PyString::new(py, "a").into_any(),
            ],
            Schema::Bytes => vec![PyBytes::new(py, b"").into_any()],
            Schema::Literal(index) => self
                .literals
                .get(index.get())
                .map(|constant| constant.bind(py).clone())
                .into_iter()
                .collect(),
            // The top holds values of kinds the spelling does not name, so the
            // guess is one value of each kind; the subject's walk keeps the
            // ones it holds.
            Schema::Anything(_) => one_of_each_kind(py)?,
            // A complement holds what its operand misses [ABC+22, §5.2].
            Schema::Complement(inner) => return self.outside(inner, depth),
            // A union or a meet hands up what its members do, each already
            // bounded, so a branch past the first few keeps its guesses: a
            // list beside six scalar kinds hands up a list of a list.
            Schema::Union(members) | Schema::Intersection(members) => {
                let mut each = Vec::with_capacity(members.len());
                for member in members.iter() {
                    each.push(self.guessed(member, depth)?);
                }
                let mut all = interleaved(&each);
                all.truncate(WITNESS_VALUES);
                return Ok(all);
            }
            // A refinement and a reference hand up what their base and their
            // definition do, as the node above would have read them.
            Schema::Refine { base, .. } => return self.guessed(base, depth),
            // A reference spends no level: every definition is contractive, so
            // a container stands between two references and spends one.
            Schema::Ref(at) => {
                return match self.definitions.get(at.get()) {
                    Some(body) => self.guessed(body, depth),
                    None => Ok(Vec::new()),
                };
            }
            Schema::Seq { container, shape } => match depth.checked_sub(1) {
                Some(below) => self.sequences(*container, shape, below)?,
                None => Vec::new(),
            },
            Schema::Coll { container, element } => match depth.checked_sub(1) {
                Some(below) => self.collections(*container, element, below)?,
                None => Vec::new(),
            },
            Schema::KeyedMap { fields, defaults } => match depth.checked_sub(1) {
                Some(below) => self.records(fields, defaults, below)?,
                None => Vec::new(),
            },
        };
        guesses.truncate(WITNESS_FANOUT);
        Ok(guesses)
    }

    /// At most [`WITNESS_VALUES`] guesses at a value outside `schema`: the
    /// values of its own kinds it refuses ([`Self::missed`]), then one value of
    /// each kind, which a schema of one kind refuses but for its own, as a
    /// typed operator's negation holds the other types [ABC+22, §5.2]. Outside
    /// the top there is nothing, and outside a complement what its operand
    /// holds.
    fn outside(&self, schema: &Schema, depth: usize) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mut all = match schema {
            Schema::Anything(_) => return Ok(Vec::new()),
            Schema::Complement(inner) => return self.guessed(inner, depth),
            _ => self.missed(schema, depth)?,
        };
        all.extend(one_of_each_kind(self.py)?);
        all.truncate(WITNESS_VALUES);
        Ok(all)
    }

    /// Guesses at values of `schema`'s own kinds that it refuses, with the
    /// negation pushed one constructor in [ABC+22, §5.2]. A complement misses
    /// what its operand holds, a union what each member misses, and a meet what any member does; a refinement what its
    /// base misses, the base's own guesses, which a constraint may refuse, and
    /// a number beside each of its order bounds; a sequence, a set or a record
    /// a container of its kind with one part outside its schema, a record one
    /// lacking a required key or holding a key it closes out.
    fn missed(&self, schema: &Schema, depth: usize) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let py = self.py;
        let mut misses = match schema {
            // Outside a complement are its operand's members, its negation's
            // negation [ABC+22, §5.2], which the operand's guesses build.
            Schema::Complement(inner) => return self.guessed(inner, depth),
            Schema::Nothing
            | Schema::Anything(_)
            | Schema::Instance(_)
            | Schema::AttrRecord { .. }
            | Schema::SelfRef(_)
            | Schema::NoneType
            | Schema::Bool
            | Schema::Int
            | Schema::Float
            | Schema::Str
            | Schema::Bytes => Vec::new(),
            // A constant misses the value beside it: the next integer, or the
            // string one character longer.
            Schema::Literal(index) => match self.literals.get(index.get()) {
                Some(constant) => beside_the_constant(constant.bind(py))?,
                None => Vec::new(),
            },
            Schema::Union(members) | Schema::Intersection(members) => {
                let mut each = Vec::with_capacity(members.len());
                for member in members.iter() {
                    each.push(self.missed(member, depth)?);
                }
                let mut all = interleaved(&each);
                all.truncate(WITNESS_VALUES);
                return Ok(all);
            }
            Schema::Refine { base, constraints } => {
                let mut beside = Vec::new();
                for constraint in constraints.iter() {
                    beside.extend(self.beside_the_bound(constraint)?);
                }
                let mut all = interleaved(&[
                    beside,
                    self.guessed(base, depth)?,
                    self.missed(base, depth)?,
                ]);
                all.truncate(WITNESS_FANOUT);
                return Ok(all);
            }
            Schema::Ref(at) => {
                return match self.definitions.get(at.get()) {
                    Some(body) => self.missed(body, depth),
                    None => Ok(Vec::new()),
                };
            }
            Schema::Seq { container, shape } => match depth.checked_sub(1) {
                Some(below) => self.sequences_missed(*container, shape, below)?,
                None => Vec::new(),
            },
            Schema::Coll { container, element } => match depth.checked_sub(1) {
                Some(below) => self.collections_missed(*container, element, below)?,
                None => Vec::new(),
            },
            Schema::KeyedMap { fields, defaults } => match depth.checked_sub(1) {
                Some(below) => self.records_missed(fields, defaults, below)?,
                None => Vec::new(),
            },
        };
        misses.truncate(WITNESS_FANOUT);
        Ok(misses)
    }

    /// The numbers beside an order bound whose operand is an exact `int` or
    /// `float`: one below and the bound itself for a lower bound, the bound and
    /// one above for an upper one, as integers where the operand is one.
    fn beside_the_bound(&self, constraint: &Constraint) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let (index, below) = match constraint {
            Constraint::Ge(index) | Constraint::Gt(index) => (index, true),
            Constraint::Le(index) | Constraint::Lt(index) => (index, false),
            _ => return Ok(Vec::new()),
        };
        let py = self.py;
        let Some(operand) = self
            .literals
            .get(index.get())
            .map(|operand| operand.bind(py))
        else {
            return Ok(Vec::new());
        };
        let step = if below { -1 } else { 1 };
        if operand.is_exact_instance_of::<pyo3::types::PyInt>() {
            let Ok(bound) = operand.extract::<i64>() else {
                return Ok(Vec::new());
            };
            return Ok(match bound.checked_add(step) {
                Some(next) => vec![int(py, next)?, int(py, bound)?],
                None => vec![int(py, bound)?],
            });
        }
        if operand.is_exact_instance_of::<PyFloat>() {
            let bound = operand.extract::<f64>()?;
            let next = if below { bound - 0.5 } else { bound + 0.5 };
            return Ok(vec![
                PyFloat::new(py, next).into_any(),
                PyFloat::new(py, bound).into_any(),
            ]);
        }
        Ok(Vec::new())
    }

    /// Lists or tuples of `shape` with one part outside it [ABC+22, §5.2]. Over
    /// one item schema each holds a single item outside it; over fixed
    /// positions each holds the first guess of every position but one, which
    /// is outside its schema, or is one position short; and where no item may
    /// follow -- no tail, or a tail of `nothing` -- one holds one item too
    /// many, `None` after the positions' first guesses.
    fn sequences_missed(
        &self,
        container: SeqKind,
        shape: &SeqShape,
        depth: usize,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        // A tail of `nothing` admits no item past the prefix, as no tail does.
        let tail = shape
            .tail
            .as_deref()
            .filter(|item| !matches!(item, Schema::Nothing));
        let mut rows: Vec<Vec<Bound<'py, PyAny>>> = Vec::new();
        if shape.prefix.is_empty() {
            match tail {
                Some(item) => rows.extend(
                    self.outside(item, depth)?
                        .into_iter()
                        .map(|miss| vec![miss]),
                ),
                None => rows.push(vec![self.py.None().into_bound(self.py)]),
            }
        } else {
            if shape.prefix.len() > WITNESS_PARTS {
                return Ok(Vec::new());
            }
            let mut first = Vec::with_capacity(shape.prefix.len());
            for item in shape.prefix.iter() {
                match self.guessed(item, depth)?.into_iter().next() {
                    Some(guess) => first.push(guess),
                    None => return Ok(Vec::new()),
                }
            }
            for (at, item) in shape.prefix.iter().enumerate() {
                if let Some(miss) = self.outside(item, depth)?.into_iter().next() {
                    let mut row = first.clone();
                    if let Some(place) = row.get_mut(at) {
                        *place = miss;
                    }
                    rows.push(row);
                }
            }
            let mut shorter = first.clone();
            shorter.pop();
            rows.push(shorter);
            if tail.is_none() {
                let mut longer = first;
                longer.push(self.py.None().into_bound(self.py));
                rows.push(longer);
            }
        }
        rows.truncate(WITNESS_FANOUT);
        let mut built = Vec::with_capacity(rows.len());
        for row in rows {
            built.push(match container {
                SeqKind::List => PyList::new(self.py, row)?.into_any(),
                SeqKind::Tuple => PyTuple::new(self.py, row)?.into_any(),
            });
        }
        Ok(built)
    }

    /// Sets or frozensets holding one element outside `element`, of those that
    /// hash, as outside an array's items is an array holding one item outside
    /// them [ABC+22, §5.2].
    fn collections_missed(
        &self,
        container: CollKind,
        element: &Schema,
        depth: usize,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let py = self.py;
        let mut built = Vec::new();
        for miss in self.outside(element, depth)? {
            if miss.hash().is_err() {
                continue;
            }
            built.push(match container {
                CollKind::Set => PySet::new(py, [miss])?.into_any(),
                CollKind::FrozenSet => PyFrozenSet::new(py, [miss])?.into_any(),
            });
        }
        Ok(built)
    }

    /// Dicts a record refuses [ABC+22, §5.2]. Each holds the record's required
    /// fields under their first guesses, with one field left out or under a
    /// value outside its schema, or with a key no field names beside them,
    /// under `None` where the record closes such keys out and under a value
    /// outside a clause's where one governs it. A required field with no guess
    /// leaves no dict to build.
    fn records_missed(
        &self,
        fields: &[Field],
        defaults: &[MapClause],
        depth: usize,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        if fields.len() > WITNESS_PARTS {
            return Ok(Vec::new());
        }
        let py = self.py;
        let base = PyDict::new(py);
        for field in fields.iter().filter(|field| field.required) {
            let Some(guess) = self.guessed(&field.schema, depth)?.into_iter().next() else {
                return Ok(Vec::new());
            };
            base.set_item(PyString::new(py, &field.name), guess)?;
        }
        let mut built = Vec::new();
        for field in fields {
            if field.required {
                let without = base.copy()?;
                without.del_item(PyString::new(py, &field.name))?;
                built.push(without);
            }
            if let Some(miss) = self.outside(&field.schema, depth)?.into_iter().next() {
                let wrong = base.copy()?;
                wrong.set_item(PyString::new(py, &field.name), miss)?;
                built.push(wrong);
            }
        }
        // One of `fields.len() + 1` names is one no field takes.
        let undeclared = (0..=fields.len())
            .map(|at| format!("_{at}"))
            .find(|name| fields.iter().all(|field| &*field.name != name.as_str()));
        if let Some(name) = undeclared {
            let value = match defaults.first() {
                None => Some(py.None().into_bound(py)),
                Some(clause) => self.outside(&clause.value, depth)?.into_iter().next(),
            };
            if let Some(value) = value {
                let wider = base.copy()?;
                wider.set_item(PyString::new(py, &name), value)?;
                built.push(wider);
            }
        }
        built.truncate(WITNESS_FANOUT);
        Ok(built.into_iter().map(Bound::into_any).collect())
    }

    /// Guesses at a list or a tuple of `shape`: for one item schema, two of its
    /// guesses together, each guess alone and no item; for fixed positions, the
    /// `j`-th guess of every position for each `j`, with a tail item or none.
    fn sequences(
        &self,
        container: SeqKind,
        shape: &SeqShape,
        depth: usize,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let tail = match shape.tail.as_deref() {
            Some(item) => self.guessed(item, depth)?,
            None => Vec::new(),
        };
        let mut rows: Vec<Vec<Bound<'py, PyAny>>> = Vec::new();
        if shape.prefix.is_empty() {
            if let [first, second, ..] = tail.as_slice() {
                rows.push(vec![first.clone(), second.clone()]);
            }
            rows.extend(tail.iter().map(|guess| vec![guess.clone()]));
            rows.push(Vec::new());
        } else {
            if shape.prefix.len() > WITNESS_PARTS {
                return Ok(Vec::new());
            }
            let mut positions = Vec::with_capacity(shape.prefix.len());
            for item in shape.prefix.iter() {
                let guesses = self.guessed(item, depth)?;
                if guesses.is_empty() {
                    return Ok(Vec::new());
                }
                positions.push(guesses);
            }
            let widest = positions.iter().map(Vec::len).max().unwrap_or(0);
            for at in 0..widest {
                rows.push(nth_of_each(&positions, at));
            }
            if let (Some(first), Some(extra)) = (rows.first().cloned(), tail.first()) {
                let mut longer = first;
                longer.push(extra.clone());
                rows.push(longer);
            }
        }
        rows.truncate(WITNESS_FANOUT);
        let mut built = Vec::with_capacity(rows.len());
        for row in rows {
            built.push(match container {
                SeqKind::List => PyList::new(self.py, row)?.into_any(),
                SeqKind::Tuple => PyTuple::new(self.py, row)?.into_any(),
            });
        }
        Ok(built)
    }

    /// Guesses at a set or a frozenset: none of the element's guesses, and each
    /// that hashes alone. A guess that does not hash is no member of either.
    fn collections(
        &self,
        container: CollKind,
        element: &Schema,
        depth: usize,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let py = self.py;
        let mut rows: Vec<Vec<Bound<'py, PyAny>>> = vec![Vec::new()];
        rows.extend(
            self.guessed(element, depth)?
                .into_iter()
                .filter(|guess| guess.hash().is_ok())
                .map(|guess| vec![guess]),
        );
        let mut built = Vec::with_capacity(rows.len());
        for row in rows {
            built.push(match container {
                CollKind::Set => PySet::new(py, row)?.into_any(),
                CollKind::FrozenSet => PyFrozenSet::new(py, row)?.into_any(),
            });
        }
        Ok(built)
    }

    /// Guesses at a dict of a record's fields: for each `j`, the `j`-th guess
    /// of every required field, then of every field; and the first of these
    /// with one key a clause names beside them, under one of its value's
    /// guesses. A required field with no guess leaves no dict to build.
    fn records(
        &self,
        fields: &[Field],
        defaults: &[MapClause],
        depth: usize,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        if fields.len() > WITNESS_PARTS {
            return Ok(Vec::new());
        }
        let mut each = Vec::with_capacity(fields.len());
        for field in fields {
            let guesses = self.guessed(&field.schema, depth)?;
            if guesses.is_empty() && field.required {
                return Ok(Vec::new());
            }
            each.push(guesses);
        }
        let widest = each.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut built = Vec::new();
        for at in 0..widest {
            for every in [false, true] {
                let dict = PyDict::new(self.py);
                for (field, guesses) in fields.iter().zip(&each) {
                    if !(field.required || every) {
                        continue;
                    }
                    let Some(guess) = guesses.get(at % guesses.len().max(1)) else {
                        continue;
                    };
                    dict.set_item(PyString::new(self.py, &field.name), guess)?;
                }
                built.push(dict);
            }
        }
        if let (Some(first), Some(clause)) = (built.first(), defaults.first()) {
            let keys = self.guessed(&clause.key, depth)?;
            let values = self.guessed(&clause.value, depth)?;
            if let (Some(key), Some(value)) = (keys.first(), values.first()) {
                let wider = first.copy()?;
                if wider.set_item(key, value).is_ok() {
                    built.insert(1, wider);
                }
            }
        }
        built.truncate(WITNESS_FANOUT);
        Ok(built.into_iter().map(Bound::into_any).collect())
    }
}

/// One value of each kind the guesses of a complement may need: `None`, a
/// `bool`, an `int`, a `float`, a `str`, a list, a dict and a tuple.
fn one_of_each_kind(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyAny>>> {
    Ok(vec![
        py.None().into_bound(py),
        PyBool::new(py, false).to_owned().into_any(),
        int(py, 0)?,
        PyFloat::new(py, 0.5).into_any(),
        PyString::new(py, "").into_any(),
        PyList::empty(py).into_any(),
        PyDict::new(py).into_any(),
        PyTuple::empty(py).into_any(),
    ])
}

/// The value beside a constant of a plain kind: the next integer, or the
/// string or bytes one character longer. A constant of another kind has none.
fn beside_the_constant<'py>(constant: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = constant.py();
    if constant.is_exact_instance_of::<pyo3::types::PyInt>() {
        return Ok(
            match constant
                .extract::<i64>()
                .ok()
                .and_then(|n| n.checked_add(1))
            {
                Some(next) => vec![int(py, next)?],
                None => Vec::new(),
            },
        );
    }
    if constant.is_exact_instance_of::<PyString>() {
        return Ok(vec![constant.add("a")?]);
    }
    if constant.is_exact_instance_of::<PyBytes>() {
        return Ok(vec![constant.add(PyBytes::new(py, b"a"))?]);
    }
    Ok(Vec::new())
}

/// An `int` as the Python object it is.
fn int(py: Python<'_>, value: i64) -> PyResult<Bound<'_, PyAny>> {
    Ok(value.into_pyobject(py)?.into_any())
}

/// The first guess of each list, then the second of each, and so on.
fn interleaved<'py>(each: &[Vec<Bound<'py, PyAny>>]) -> Vec<Bound<'py, PyAny>> {
    let widest = each.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = Vec::new();
    for at in 0..widest {
        out.extend(each.iter().filter_map(|guesses| guesses.get(at).cloned()));
    }
    out
}

/// The `at`-th guess of every position, each position cycling through its own.
fn nth_of_each<'py>(positions: &[Vec<Bound<'py, PyAny>>], at: usize) -> Vec<Bound<'py, PyAny>> {
    positions
        .iter()
        .filter_map(|guesses| guesses.get(at % guesses.len().max(1)).cloned())
        .collect()
}

#[cfg(all(test, feature = "interpreter-tests"))]
mod tests;
