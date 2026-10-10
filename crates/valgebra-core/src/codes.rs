//! The codes a node reports for itself that the binding's walk reports too,
//! spelled once.
//!
//! A leaf mismatch reports the node's own code, and [`Schema::error_code`] is
//! the arm per node. Ten of those codes are also written where the binding's
//! walk builds a node's failure itself, mostly holding the node's parts rather
//! than the node -- a sequence's walk its container and shape, a set's walk its
//! reading, a union its members, a complement its inner schema -- so the
//! binding names them in its own table too, and both tables read the string
//! here. A code is what a caller compares against, and two spellings of one
//! would be two codes for one failure.
//!
//! [`Schema::error_code`]: crate::Schema::error_code

/// A value that is not a `dict` where a mapping is expected.
pub const DICT_TYPE: &str = "dict_type";

/// A value that is not a `frozenset`.
pub const FROZEN_SET_TYPE: &str = "frozen_set_type";

/// A value that is not an instance of the class the schema names.
pub const INSTANCE_TYPE: &str = "instance_type";

/// A value that is not a `list`.
pub const LIST_TYPE: &str = "list_type";

/// A value that is not one of the literals the schema lists.
pub const LITERAL_ERROR: &str = "literal_error";

/// A value that is not a `set`.
pub const SET_TYPE: &str = "set_type";

/// A value that is not a `tuple`.
pub const TUPLE_TYPE: &str = "tuple_type";

/// A value a complement excludes: it matched the schema that was negated.
pub const UNEXPECTED_MATCH: &str = "unexpected_match";

/// A value outside every branch of a union.
pub const UNION_ERROR: &str = "union_error";

/// A reference whose definition the walk cannot resolve.
pub const UNRESOLVED_RECURSION: &str = "unresolved_recursion";
