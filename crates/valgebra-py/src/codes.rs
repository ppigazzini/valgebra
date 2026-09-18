//! The failure codes this crate writes, named once.
//!
//! A code is the part of a failure a caller writes code against: `exc.code ==
//! "missing_key"` is a branch in somebody's error handler, so the set of them
//! is a published vocabulary rather than an implementation detail. Half of it
//! is the core's -- a leaf mismatch reports the node's own code, and
//! [`Schema::error_code`] is the arm per node -- and the half here is what the
//! *walk* reports: a bound that failed, a key that is missing, a walk that ran
//! out of levels.
//!
//! Written as literals at the sites that report them, as they were, the set
//! could only be recovered by scanning files for strings that look like codes.
//! That reading misses a code written in a file nobody thought to scan, and
//! invents a cell for any other snake-case string in one that was -- it had
//! `not_subset`, which is an answer [`relation_to`](crate::validator) gives and
//! no failure anybody reports. A caller's vocabulary is not a thing to recover
//! by pattern, so it is declared, and `tests/test_code_table.py` holds every
//! name here to a site that writes it and every site to a name here.

use valgebra_core::Schema;

/// A failure code, as `ValidationError.code` reports one.
///
/// A newtype rather than a `&'static str` so that the constants below are the
/// only way to write one: a helper taking a `Code` cannot be handed a string
/// that looks like a code, and the compiler is what says so.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Code(&'static str);

impl Code {
    /// The code a caller compares against.
    pub(crate) const fn as_str(self) -> &'static str {
        self.0
    }

    /// The code the *core* writes for this node.
    ///
    /// The one crossing between the two tables. A leaf mismatch is the node's
    /// own failure -- an `int` that is a `str`, a literal that is another --
    /// and the node is what names it, so this reads the core's arm rather than
    /// restating twenty-one of them here.
    pub(crate) fn of_schema(schema: &Schema) -> Code {
        Code(schema.error_code())
    }
}

/// A value that is not a `dict` where a mapping is expected.
pub(crate) const DICT_TYPE: Code = Code("dict_type");

/// A key the record does not declare, on a record that forbids extras.
pub(crate) const EXTRA_FORBIDDEN: Code = Code("extra_forbidden");

/// A value that is not a `frozenset`.
pub(crate) const FROZEN_SET_TYPE: Code = Code("frozen_set_type");

/// A value below an exclusive lower bound.
pub(crate) const GREATER_THAN: Code = Code("greater_than");

/// A value below an inclusive lower bound.
pub(crate) const GREATER_THAN_EQUAL: Code = Code("greater_than_equal");

/// A value that is not an instance of the class the schema names.
pub(crate) const INSTANCE_TYPE: Code = Code("instance_type");

/// A document the JSON parser refused before any schema saw it.
pub(crate) const JSON_INVALID: Code = Code("json_invalid");

/// A value above an exclusive upper bound.
pub(crate) const LESS_THAN: Code = Code("less_than");

/// A value above an inclusive upper bound.
pub(crate) const LESS_THAN_EQUAL: Code = Code("less_than_equal");

/// A list whose length the shape does not admit.
pub(crate) const LIST_LENGTH: Code = Code("list_length");

/// A value that is not a `list`.
pub(crate) const LIST_TYPE: Code = Code("list_type");

/// A value that is not one of the literals the schema lists.
pub(crate) const LITERAL_ERROR: Code = Code("literal_error");

/// An attribute the schema requires that the object does not carry.
pub(crate) const MISSING_ATTRIBUTE: Code = Code("missing_attribute");

/// A key the schema requires that the mapping does not carry.
pub(crate) const MISSING_KEY: Code = Code("missing_key");

/// A number that is not a multiple of the step the schema names.
pub(crate) const MULTIPLE_OF: Code = Code("multiple_of");

/// A container that changed while the walk was reading it.
pub(crate) const MUTATED_DURING_VALIDATION: Code = Code("mutated_during_validation");

/// A predicate that raised rather than answering.
pub(crate) const PREDICATE_ERROR: Code = Code("predicate_error");

/// A predicate that answered `False`.
pub(crate) const PREDICATE_FAILED: Code = Code("predicate_failed");

/// A value nested deeper than the walk's bound.
pub(crate) const RECURSION_LIMIT: Code = Code("recursion_limit");

/// A value that contains itself, found by identity rather than by equality.
pub(crate) const RECURSION_LOOP: Code = Code("recursion_loop");

/// A value that is not a `set`.
pub(crate) const SET_TYPE: Code = Code("set_type");

/// A string the pattern does not match.
pub(crate) const STRING_PATTERN_MISMATCH: Code = Code("string_pattern_mismatch");

/// A value longer than the schema admits.
pub(crate) const TOO_LONG: Code = Code("too_long");

/// A value shorter than the schema admits.
pub(crate) const TOO_SHORT: Code = Code("too_short");

/// A tuple whose length the shape does not admit.
pub(crate) const TUPLE_LENGTH: Code = Code("tuple_length");

/// A value that is not a `tuple`.
pub(crate) const TUPLE_TYPE: Code = Code("tuple_type");

/// A value a complement excludes: it matched the schema that was negated.
pub(crate) const UNEXPECTED_MATCH: Code = Code("unexpected_match");

/// A value outside every branch of a union.
pub(crate) const UNION_ERROR: Code = Code("union_error");

/// A reference whose definition the walk cannot resolve.
pub(crate) const UNRESOLVED_RECURSION: Code = Code("unresolved_recursion");

/// The exception's own code, carried where no single violation owns the
/// failure.
pub(crate) const VALIDATION_ERROR: Code = Code("validation_error");
