//! Where the two regex engines read one pattern as two languages, and the
//! forms refused for it.
//!
//! The engine here is Rust's, and a pattern Python spells and this one does not
//! raises at build: the parse fails and the caller reads the parse error. That
//! is the loud direction, and it needs no help.
//!
//! The quiet direction is a pattern **both** engines accept and read
//! differently. Inside a character class, `--`, `&&` and `~~` combine classes
//! here and are literal characters to Python, and a nested `[` opens a class
//! here and is a literal `[` there. `[\w--\d]` is the non-digit word characters
//! under this engine and the word characters plus `-` under `re`, and nothing
//! about compiling it says so.
//!
//! `re` gives three different answers to the four forms, and that is the
//! argument for refusing all of them rather than trusting it to object:
//!
//! - `[\w--\d]` **raises**, so a ported pattern fails there at once;
//! - `[\w&&\d]` and `[\w~~\d]` **warn** that `re` may one day read them as
//!   operators, which is a reservation rather than a verdict;
//! - `[a[bc]]` says **nothing at all**, which is the case this is most for.
//!
//! A pattern read with another engine's meaning is a wrong answer a caller
//! cannot see, so each is refused by the name *this* engine would have read it
//! under -- the fact a reader porting the pattern needs, and for two of the
//! three a sentence `re` has already shown them.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Refuse a pattern whose character classes carry an operator `re` reserves.
///
/// The scan is a character walk rather than a parse: it needs only to know
/// whether it stands inside a class, which `\` escapes and `[`/`]` delimit, and
/// a pattern whose classes do not balance is refused by the compile that
/// follows this. A form found outside a class is nothing -- `a--b` is three
/// literals to both engines -- so the position is what decides, as it does in
/// `re`.
pub(crate) fn reject_reserved_class_syntax(pattern: &str) -> PyResult<()> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut cursor = chars.iter().copied().enumerate().peekable();
    let mut inside = false;
    // Where the open class's first member sits. A doubled character *there* is
    // two literal characters to both engines, so the position is what tells an
    // operator from a member -- `[--/]` is the range from `-` to `/` either way.
    let mut first = 0;
    // The cursor advances by taking from the iterator and never by arithmetic,
    // so the scan terminates by construction: an arm that looks ahead either
    // takes what it saw or leaves it for the next turn.
    while let Some((at, here)) = cursor.next() {
        let next = cursor.peek().copied();
        match here {
            // An escape carries its character with it, whichever side of a
            // class it is on: `[\[&&x]` holds an operator and no nested set.
            '\\' => {
                cursor.next();
            }
            '[' if !inside => {
                inside = true;
                // A `^` negates the class and is not a member of it, so the
                // first member is what follows.
                let opening = if next.map(|(_, symbol)| symbol) == Some('^') {
                    cursor.next();
                    cursor.peek().copied()
                } else {
                    next
                };
                first = opening.map_or(at, |(where_, _)| where_);
                // A `]` in the first member's place is that member rather than
                // the class's close, and `re` reads it the same way. Being a
                // member, it *takes* the position where a doubled character is
                // literal: `[]&&x]` carries the operator at the next one, which
                // is where `re` warns about it.
                if opening.map(|(_, symbol)| symbol) == Some(']') {
                    cursor.next();
                }
            }
            // A `]` outside a class is a literal and closes nothing, so this
            // arm takes no guard: the flag is already down, and putting it down
            // again is the same statement.
            ']' => inside = false,
            // `[:name:]` is a POSIX class, which this engine reads and `re`
            // does not -- a divergence `docs/05-refinements.md` names, with the
            // example that shows how the two read it. It is skipped whole, so
            // an operator after it is still read as one.
            '[' if next.map(|(_, symbol)| symbol) == Some(':') => {
                cursor.next();
                let mut previous = ':';
                for (_, symbol) in cursor.by_ref() {
                    if previous == ':' && symbol == ']' {
                        break;
                    }
                    previous = symbol;
                }
            }
            // Every other `[` inside a class opens a nested set, which is a
            // union here and four literal characters there.
            '[' => return Err(reserved("nested set", pattern, at)),
            '-' | '&' | '~'
                if inside && next.map(|(_, symbol)| symbol) == Some(here) && at > first =>
            {
                let what = match here {
                    '-' => "set difference",
                    '&' => "set intersection",
                    _ => "set symmetric difference",
                };
                return Err(reserved(what, pattern, at));
            }
            _ => {}
        }
    }
    Ok(())
}

/// The refusal, in the words `re` warns with, plus what to write instead.
fn reserved(what: &str, pattern: &str, index: usize) -> PyErr {
    PyValueError::new_err(format!(
        "possible {what} at position {index} in {pattern:?}: this is a \
         class-set operator here and a literal character to `re`, so the two \
         read the pattern as different sets. Escape the characters to mean \
         them literally, or write the classes out."
    ))
}

#[cfg(test)]
mod tests;
