//! valgebra schema intermediate representation.
//!
//! A schema denotes a set of Python values; validation is membership. This
//! crate is pure Rust: it defines the IR, the denotation of every node, and the
//! structured [`Violation`] produced when membership fails. Inspecting a Python
//! object requires `PyO3`, so the validator walk itself lives in the bindings
//! crate; this crate is the stable, language-agnostic core.

use std::fmt::Write as _;

/// A single step in the location of a value inside a composite structure.
///
/// Scalar schemas never produce a path; structural schemas (records, sequences,
/// tuples, sets, mappings) push a segment per level as they descend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    /// A mapping or record key.
    Key(String),
    /// A sequence, tuple, or set position.
    Index(usize),
}

/// The schema intermediate representation.
///
/// Each variant documents its denotation: the set of Python values it accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Schema {
    /// Top. Denotes every Python value; membership always holds.
    Anything,
    /// The gradual dynamic type. At runtime it admits every value like the top,
    /// but it is a distinct atom: the simplifier must not rewrite it by the
    /// lattice laws, so `Any` and [`Schema::Anything`] are kept separate.
    Any,
    /// Bottom. Denotes the empty set; membership never holds.
    Nothing,
    /// Denotes the singleton set `{None}`.
    NoneType,
    /// Denotes `{True, False}`, exactly the `bool` instances.
    ///
    /// Because `bool` is a subclass of `int`, this set is a subset of
    /// [`Schema::Int`]: `Bool` is a subtype of `Int`.
    Bool,
    /// Denotes every `int` instance: `isinstance(x, int)`.
    ///
    /// In Python `bool` is a subclass of `int`, so `True` and `False` are
    /// integers and are members of this set. No value is carved out: subtyping
    /// is subset inclusion, so [`Schema::Bool`] is a subtype of `Int` rather
    /// than disjoint from it.
    Int,
    /// Denotes every `float` instance: `isinstance(x, float)`.
    ///
    /// `int` does not subclass `float`, so `Int` and `Float` are disjoint and
    /// an integer is not a member.
    Float,
    /// Denotes the `str` instances.
    Str,
    /// Denotes the `bytes` instances.
    Bytes,
    /// Denotes the typed singleton `{c}` for a fixed constant `c`: a value is a
    /// member iff it has the *same type* as `c` and is equal to it. Same-type is
    /// what makes this a singleton — Python's `==` conflates across types
    /// (`1 == True == 1.0`), so equality alone would make `Literal[1]` also
    /// admit `True` and `1.0`. Requiring `type(x) is type(c)` keeps the typing
    /// spec's distinction between `Literal[1]`, `Literal[True]`, and
    /// `Literal[1.0]`.
    ///
    /// The constant itself is not stored here — the core stays free of Python
    /// objects. The payload is an index into a constants pool held alongside the
    /// compiled validator. The same-type test is applied in the bindings, where
    /// the Python value is in hand.
    Literal(usize),
    /// Denotes lists whose every element belongs to the inner schema.
    Sequence(Box<Schema>),
    /// Denotes tuples matched positionally at exactly this length.
    Tuple(Vec<Schema>),
    /// Denotes tuples of any length whose every element belongs to the inner
    /// schema (the homogeneous `tuple[T, ...]` form).
    VariadicTuple(Box<Schema>),
    /// Denotes sets whose every element belongs to the inner schema.
    Set(Box<Schema>),
    /// Denotes frozensets whose every element belongs to the inner schema.
    FrozenSet(Box<Schema>),
    /// Denotes dicts whose keys all match `key` and values all match `value`.
    Mapping {
        /// Schema every key must satisfy.
        key: Box<Schema>,
        /// Schema every value must satisfy.
        value: Box<Schema>,
    },
    /// Denotes dicts with named fields, closed (strict): a required field's key
    /// must be present with a matching value; an optional field's value is
    /// checked only when its key is present; no key outside the declared field
    /// names is admitted. The empty record denotes only the empty dict.
    Record {
        /// The declared fields, in order.
        fields: Vec<Field>,
    },
    /// Denotes the union of the member sets: a value is a member iff it belongs
    /// to at least one member schema.
    Union(Vec<Schema>),
    /// Denotes instances of a class, by `isinstance`. The class is held in the
    /// validator's object pool; the payload is its index.
    Instance(usize),
    /// Denotes instances of a class whose attributes satisfy the given fields.
    ///
    /// `isinstance` against the pooled class at `class_index` must hold, and
    /// every field's attribute must be present and match. This is the deep
    /// check for dataclasses and named tuples.
    Object {
        /// Index of the class in the validator's object pool.
        class_index: usize,
        /// Per-attribute field schemas; all required.
        fields: Vec<Field>,
    },
    /// Denotes the subset of the base set satisfying every constraint
    /// (`{ x in [[base]] | all constraints hold }`). The base is checked first.
    Refine {
        /// The base schema; a value must belong to it before constraints apply.
        base: Box<Schema>,
        /// Constraints that further narrow the base set, checked in order.
        constraints: Vec<Constraint>,
    },
}

/// A constraint narrowing a [`Schema::Refine`] base set.
///
/// Comparison and predicate operands live in the validator's object pool; the
/// payload is an index. Length bounds carry the length directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constraint {
    /// `value >= pool[i]`.
    Ge(usize),
    /// `value > pool[i]`.
    Gt(usize),
    /// `value <= pool[i]`.
    Le(usize),
    /// `value < pool[i]`.
    Lt(usize),
    /// `len(value) >= n`.
    MinLen(usize),
    /// `len(value) <= n`.
    MaxLen(usize),
    /// `pool[i](value)` is truthy. The documented Python-callback slow path.
    Predicate(usize),
}

/// A named field of a [`Schema::Record`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// The key name.
    pub name: String,
    /// Schema the field's value must satisfy.
    pub schema: Schema,
    /// Whether the key must be present.
    pub required: bool,
}

impl Schema {
    /// A short, stable label naming the expected set, shown in violations.
    #[must_use]
    pub fn expected(&self) -> &'static str {
        match self {
            Schema::Anything => "anything",
            Schema::Any => "any",
            Schema::Nothing => "nothing",
            Schema::NoneType => "None",
            Schema::Bool => "bool",
            Schema::Int => "int",
            Schema::Float => "float",
            Schema::Str => "str",
            Schema::Bytes => "bytes",
            // The py layer renders the concrete constant; this is a fallback.
            Schema::Literal(_) => "literal",
            Schema::Sequence(_) => "list",
            Schema::Tuple(_) | Schema::VariadicTuple(_) => "tuple",
            Schema::Set(_) => "set",
            Schema::FrozenSet(_) => "frozenset",
            Schema::Mapping { .. } | Schema::Record { .. } => "dict",
            Schema::Union(_) => "union",
            // The py layer renders the concrete class name; these are fallbacks.
            Schema::Instance(_) => "instance",
            Schema::Object { .. } => "object",
            // A refinement's type is its base; constraints report their own.
            Schema::Refine { base, .. } => base.expected(),
        }
    }

    /// The stable, machine-readable code emitted when membership fails.
    #[must_use]
    pub fn error_code(&self) -> &'static str {
        match self {
            // Anything and Any never fail; the codes are for completeness.
            Schema::Anything => "anything",
            Schema::Any => "any",
            Schema::Nothing => "no_match",
            Schema::NoneType => "none_type",
            Schema::Bool => "bool_type",
            Schema::Int => "int_type",
            Schema::Float => "float_type",
            Schema::Str => "string_type",
            Schema::Bytes => "bytes_type",
            Schema::Literal(_) => "literal_value",
            Schema::Sequence(_) => "list_type",
            Schema::Tuple(_) | Schema::VariadicTuple(_) => "tuple_type",
            Schema::Set(_) => "set_type",
            Schema::FrozenSet(_) => "frozenset_type",
            Schema::Mapping { .. } | Schema::Record { .. } => "dict_type",
            Schema::Union(_) => "union_error",
            Schema::Instance(_) | Schema::Object { .. } => "instance_type",
            Schema::Refine { base, .. } => base.error_code(),
        }
    }
}

/// A validation failure: a value did not belong to a schema's set.
#[derive(Debug, Clone)]
pub struct Violation {
    /// Stable, machine-readable code.
    pub code: &'static str,
    /// Location of the offending value from the validation root; empty at root.
    pub path: Vec<PathSegment>,
    /// Short label of the expected set (e.g. `int`).
    pub expected: String,
    /// Short repr-style summary of the offending value.
    pub value_summary: String,
}

impl Violation {
    /// Render the path as a location string (`name[2].id`); empty at the root.
    #[must_use]
    pub fn location(&self) -> String {
        let mut out = String::new();
        for segment in &self.path {
            match segment {
                PathSegment::Key(key) => {
                    if !out.is_empty() {
                        out.push('.');
                    }
                    out.push_str(key);
                }
                PathSegment::Index(index) => {
                    let _ = write!(out, "[{index}]");
                }
            }
        }
        out
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let location = self.location();
        if location.is_empty() {
            write!(
                f,
                "expected {}, got {} [{}]",
                self.expected, self.value_summary, self.code
            )
        } else {
            write!(
                f,
                "at {}: expected {}, got {} [{}]",
                location, self.expected, self.value_summary, self.code
            )
        }
    }
}

impl std::error::Error for Violation {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn violation_renders_root_message() {
        let v = Violation {
            code: "int_type",
            path: Vec::new(),
            expected: "int".to_owned(),
            value_summary: "'x'".to_owned(),
        };
        assert_eq!(v.location(), "");
        assert_eq!(v.to_string(), "expected int, got 'x' [int_type]");
    }

    #[test]
    fn violation_renders_nested_location() {
        let v = Violation {
            code: "string_type",
            path: vec![PathSegment::Key("name".to_owned()), PathSegment::Index(2)],
            expected: "str".to_owned(),
            value_summary: "5".to_owned(),
        };
        assert_eq!(v.location(), "name[2]");
        assert!(v.to_string().starts_with("at name[2]: expected str"));
    }

    #[test]
    fn labels_and_codes_for_every_variant() {
        let cases = [
            (Schema::Anything, "anything", "anything"),
            (Schema::Nothing, "nothing", "no_match"),
            (Schema::NoneType, "None", "none_type"),
            (Schema::Bool, "bool", "bool_type"),
            (Schema::Int, "int", "int_type"),
            (Schema::Float, "float", "float_type"),
            (Schema::Str, "str", "string_type"),
            (Schema::Bytes, "bytes", "bytes_type"),
            (Schema::Literal(0), "literal", "literal_value"),
            (Schema::Sequence(Box::new(Schema::Int)), "list", "list_type"),
            (Schema::Tuple(vec![Schema::Int]), "tuple", "tuple_type"),
            (Schema::Set(Box::new(Schema::Int)), "set", "set_type"),
            (
                Schema::Mapping {
                    key: Box::new(Schema::Str),
                    value: Box::new(Schema::Int),
                },
                "dict",
                "dict_type",
            ),
            (
                Schema::Record {
                    fields: vec![Field {
                        name: "k".to_owned(),
                        schema: Schema::Int,
                        required: true,
                    }],
                },
                "dict",
                "dict_type",
            ),
        ];
        for (schema, expected, code) in cases {
            assert_eq!(schema.expected(), expected, "expected for {schema:?}");
            assert_eq!(schema.error_code(), code, "code for {schema:?}");
        }
    }

    #[test]
    fn location_renders_keys_indices_and_their_mix() {
        let key_only = Violation {
            code: "x",
            path: vec![
                PathSegment::Key("a".to_owned()),
                PathSegment::Key("b".to_owned()),
            ],
            expected: String::new(),
            value_summary: String::new(),
        };
        assert_eq!(key_only.location(), "a.b");

        let index_only = Violation {
            code: "x",
            path: vec![PathSegment::Index(0), PathSegment::Index(3)],
            expected: String::new(),
            value_summary: String::new(),
        };
        assert_eq!(index_only.location(), "[0][3]");

        let mixed = Violation {
            code: "x",
            path: vec![
                PathSegment::Key("items".to_owned()),
                PathSegment::Index(2),
                PathSegment::Key("id".to_owned()),
            ],
            expected: "int".to_owned(),
            value_summary: "'x'".to_owned(),
        };
        assert_eq!(mixed.location(), "items[2].id");
        assert_eq!(
            mixed.to_string(),
            "at items[2].id: expected int, got 'x' [x]"
        );
    }

    #[test]
    fn mapping_and_record_share_the_dict_label() {
        let mapping = Schema::Mapping {
            key: Box::new(Schema::Str),
            value: Box::new(Schema::Int),
        };
        let record = Schema::Record { fields: Vec::new() };
        assert_eq!(mapping.expected(), record.expected());
        assert_eq!(mapping.error_code(), record.error_code());
    }

    #[test]
    fn schema_equality_is_structural() {
        assert_eq!(
            Schema::Sequence(Box::new(Schema::Int)),
            Schema::Sequence(Box::new(Schema::Int))
        );
        assert_ne!(
            Schema::Sequence(Box::new(Schema::Int)),
            Schema::Sequence(Box::new(Schema::Str))
        );
        assert_ne!(Schema::Literal(0), Schema::Literal(1));
    }

    #[test]
    fn refine_delegates_label_and_code_to_its_base() {
        let refined = Schema::Refine {
            base: Box::new(Schema::Str),
            constraints: vec![Constraint::MinLen(1)],
        };
        assert_eq!(refined.expected(), "str");
        assert_eq!(refined.error_code(), "string_type");
    }

    #[test]
    fn field_is_cloneable_and_carries_its_flag() {
        let field = Field {
            name: "n".to_owned(),
            schema: Schema::Int,
            required: false,
        };
        let copy = field.clone();
        assert_eq!(copy.name, "n");
        assert!(!copy.required);
        assert_eq!(copy.schema, Schema::Int);
    }
}
