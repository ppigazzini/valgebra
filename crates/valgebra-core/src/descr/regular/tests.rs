use super::{Alphabet, RegularSet};
use proptest::prelude::*;

/// The words a law is checked over.
///
/// Every word of up to two letters over `{a, b, c}`, plus a three-letter one
/// and a non-ASCII one. Small, but every generated pattern below is written
/// over that alphabet, so a language that differs from another differs on a
/// word this short -- two regular languages agreeing on all words up to the
/// product of their state counts are equal, and these automata have a
/// handful of states each.
fn universe() -> Vec<&'static [u8]> {
    vec![
        b"",
        b"a",
        b"b",
        b"c",
        b"aa",
        b"ab",
        b"ac",
        b"ba",
        b"bb",
        b"bc",
        b"ca",
        b"cb",
        b"cc",
        b"aba",
        b"abc",
        "é".as_bytes(),
    ]
}

/// Whether two sets agree about every word in the universe.
///
/// A *weaker* question than equality, and deliberately: two regular
/// languages agree exactly when they agree on every word shorter than the
/// product of their state counts, which for the automata here is longer
/// than any universe can enumerate. So this is used in the direction it can
/// support -- two equal sets must agree about every word -- and equality
/// itself is checked against the emptiness decision instead.
fn agree_on_words(a: &RegularSet, b: &RegularSet) -> bool {
    universe().into_iter().all(|w| a.holds(w) == b.holds(w))
}

/// Whether two sets hold the same words, decided by the algebra.
///
/// The semantic-subtyping reduction in both directions: `A = B` when
/// `A ∧ ¬B` and `B ∧ ¬A` are both empty. This reaches the whole language
/// where enumeration cannot, and it reads the answer off the reachability
/// walk rather than off the canonical form -- so comparing it with `==`
/// below is two independent pieces of this module agreeing, not one of them
/// agreeing with itself.
fn same_language(a: &RegularSet, b: &RegularSet) -> bool {
    let inside = |x: &RegularSet, y: &RegularSet| {
        x.intersect(&y.complement())
            .is_some_and(|met| met.is_empty())
    };
    inside(a, b) && inside(b, a)
}

/// A pattern's language, or the empty set where the bound refuses it.
fn language(pattern: &str) -> RegularSet {
    RegularSet::pattern(pattern, Alphabet::Text).expect("a small pattern builds")
}

/// The walk's reading of a pattern: the `regex` matcher, over the whole text.
fn matcher(pattern: &str) -> regex::Regex {
    regex::Regex::new(&format!(r"\A(?:{pattern})\z")).expect("a drawn pattern compiles")
}

/// Every word over `{a, b}` of up to four letters, and one outside the alphabet.
fn short_words() -> Vec<String> {
    let mut words = vec![String::new(), "c".to_owned()];
    let mut layer = vec![String::new()];
    for _ in 0..4 {
        layer = layer
            .iter()
            .flat_map(|word| [format!("{word}a"), format!("{word}b")])
            .collect();
        words.extend(layer.iter().cloned());
    }
    words
}

/// Patterns over `{a, b}` in the shapes a search ranks: an alternation, whose
/// earlier branch may be a prefix of a later one or empty, and a repetition,
/// greedy or lazy.
fn ranked_pattern() -> impl Strategy<Value = String> {
    let leaf = prop_oneof![
        Just(String::new()),
        Just("a".to_owned()),
        Just("b".to_owned()),
        Just("[ab]".to_owned()),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(x, y)| format!("{x}{y}")),
            (inner.clone(), inner.clone()).prop_map(|(x, y)| format!("(?:{x}|{y})")),
            (
                inner,
                prop::sample::select(vec!["*", "+", "?", "*?", "+?", "??", "{1,2}", "{1,2}?"]),
            )
                .prop_map(|(x, quantifier)| format!("(?:{x}){quantifier}")),
        ]
    })
}

/// Languages over the three-letter alphabet the universe covers.
fn regular_set() -> impl Strategy<Value = RegularSet> {
    let leaf = prop_oneof![
        Just(RegularSet::empty()),
        Just(RegularSet::all(Alphabet::Bytes)),
        Just(language("a")),
        Just(language("b")),
        Just(language("ab?")),
        Just(language("[ab]+")),
        Just(language("a*")),
        Just(RegularSet::word(b"ab")),
        Just(RegularSet::at_least(1, Alphabet::Text).expect("a small bound")),
        Just(RegularSet::at_most(1, Alphabet::Text).expect("a small bound")),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a
                .union(&b)
                .unwrap_or_else(|| RegularSet::all(Alphabet::Bytes))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(RegularSet::empty)),
            inner.prop_map(|a| a.complement()),
        ]
    })
}

proptest! {
    // A quarter of the default, for the reason the descriptor module
    // gives: every operation here is an automaton product, and shrinking a
    // failure over thousands of draws outruns a mutation sweep's patience.
    // A fraction rather than a count, so a deeper run reaches here too.
    #![proptest_config(ProptestConfig {
        cases: ProptestConfig::default().cases / 4,
        // A bounded shrink, so a broken invariant cannot turn a caught
        // mutation into a run that outlasts a sweep.
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    // THEORY: each-kind-is-closed
    /// The Boolean algebra, checked by equality of the canonical forms.
    ///
    /// Equality *is* language equality here -- that is what the minimal,
    /// canonically-numbered table earns, and the property below holds it to
    /// the emptiness decision. So a law is checked at full strength rather
    /// than over whatever words a universe can list, which for a regular
    /// language is never enough.
    #[test]
    fn the_lattice_laws_hold_of_the_languages(
        a in regular_set(),
        b in regular_set(),
        c in regular_set(),
    ) {
        prop_assert_eq!(a.union(&b), b.union(&a));
        prop_assert_eq!(a.intersect(&b), b.intersect(&a));
        prop_assert_eq!(
            a.union(&b).and_then(|ab| ab.union(&c)),
            b.union(&c).and_then(|bc| a.union(&bc))
        );
        prop_assert_eq!(
            a.intersect(&b).and_then(|ab| ab.intersect(&c)),
            b.intersect(&c).and_then(|bc| a.intersect(&bc))
        );
        if let Some(met) = a.intersect(&b) {
            prop_assert_eq!(a.union(&met), Some(a.clone()));
        }
        if let Some(joined) = a.union(&b) {
            prop_assert_eq!(a.intersect(&joined), Some(a.clone()));
        }
        if let (Some(left), Some(right)) = (
            b.union(&c).and_then(|bc| a.intersect(&bc)),
            a.intersect(&b).and_then(|ab| {
                a.intersect(&c).and_then(|ac| ab.union(&ac))
            }),
        ) {
            prop_assert_eq!(left, right);
        }
    }

    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_languages(a in regular_set(), b in regular_set()) {
        prop_assert!(
            a.intersect(&a.complement())
                .is_some_and(|met| met.is_empty())
        );
        prop_assert_eq!(a.union(&a.complement()), Some(RegularSet::all(Alphabet::Bytes)));
        prop_assert_eq!(&a.complement().complement(), &a);
        prop_assert_eq!(
            a.union(&b).map(|set| set.complement()),
            a.complement().intersect(&b.complement())
        );
        prop_assert_eq!(
            a.intersect(&b).map(|set| set.complement()),
            a.complement().union(&b.complement())
        );
    }

    /// Equality of the canonical forms is equality of the languages.
    ///
    /// The claim the whole representation rests on, held against the
    /// *emptiness* decision -- a reachability walk, which shares no code
    /// with the minimisation and renumbering that make the form canonical.
    /// Two spellings of one language have one table, and two languages that
    /// differ have two.
    #[test]
    fn being_equal_is_holding_the_same_language(a in regular_set(), b in regular_set()) {
        prop_assert_eq!(same_language(&a, &b), a == b);
    }

    /// Two equal sets agree about every word, which is the direction
    /// enumeration can support.
    #[test]
    fn equal_sets_agree_about_every_word(a in regular_set(), b in regular_set()) {
        if a == b {
            prop_assert!(agree_on_words(&a, &b));
        }
    }

    /// An empty verdict is contradicted by no word.
    #[test]
    fn an_empty_language_holds_no_word(a in regular_set()) {
        if a.is_empty() {
            prop_assert!(universe().into_iter().all(|w| !a.holds(w)));
        }
    }

    /// A pattern's language is the words the walk's matcher admits whole.
    ///
    /// The laws above check the automata against each other, which holds as
    /// well of a table that drops words as of one that keeps them. This holds
    /// the table one pattern lowers to against the matcher, on the patterns
    /// where a search's preference among matches could leave words out.
    #[test]
    fn a_pattern_holds_the_words_its_matcher_admits(pattern in ranked_pattern()) {
        let set = language(&pattern);
        let whole = matcher(&pattern);
        for word in short_words() {
            prop_assert_eq!(
                set.holds(word.as_bytes()),
                whole.is_match(&word),
                "{} against {:?}",
                pattern,
                word
            );
        }
    }
}

/// The words a search would not report first are in the language all the same:
/// a later branch an earlier one is a prefix of, an empty first branch, and
/// what a lazy repetition matches past its shortest.
#[test]
fn a_pattern_holds_every_word_it_matches_whole() {
    for (pattern, word) in [
        ("a|ab", "ab"),
        ("http|https", "https"),
        (r"\d+|\d+\.\d+", "1.5"),
        ("|a", "a"),
        ("a*?", "aaa"),
        ("a+?", "aa"),
        (".*?", "text"),
    ] {
        assert!(
            language(pattern).holds(word.as_bytes()),
            "{pattern} holds {word:?}"
        );
    }
    // And the inclusion the dropped words would refute is refuted.
    assert!(
        !language("a|ab")
            .intersect(&RegularSet::word(b"a").complement())
            .is_some_and(|set| set.is_empty())
    );
}

/// The two relations this component exists to decide, both declined by the
/// structural procedure because it relates patterns only when they are
/// written identically.
#[test]
fn one_pattern_is_decided_inside_another() {
    let narrow = language("a");
    let wide = language("ab?");
    // `a` is below `ab?`: the meet with the complement is empty, which is
    // the semantic-subtyping reduction applied to a language.
    assert!(
        narrow
            .intersect(&wide.complement())
            .is_some_and(|set| set.is_empty())
    );
    // And not the other way: `ab` is in the wider one alone.
    assert!(
        !wide
            .intersect(&narrow.complement())
            .is_some_and(|set| set.is_empty())
    );
    assert!(wide.holds(b"ab") && !narrow.holds(b"ab"));

    // A pattern whose language is one word is that word's literal.
    assert_eq!(language("a"), RegularSet::word(b"a"));
    assert_eq!(language("abc"), RegularSet::word(b"abc"));
    // Which is what decides `Regex("a") <= Literal["a"]`, in both
    // directions: the two are one set.
    assert!(
        language("a")
            .intersect(&RegularSet::word(b"a").complement())
            .is_some_and(|set| set.is_empty())
    );
}

/// Two spellings of one language are one set, which is the canonicity claim
/// stated on the cases a reader would doubt.
#[test]
fn two_spellings_of_one_language_are_one_set() {
    assert_eq!(language("a|a"), language("a"));
    assert_eq!(language("(a)"), language("a"));
    assert_eq!(language("a{1}"), language("a"));
    assert_eq!(language("[aa]"), language("a"));
    assert_eq!(language("a|b"), language("[ab]"));
    assert_eq!(language("a*a*"), language("a*"));
    assert_eq!(language("(a|b)*"), language("[ab]*"));
    // The order of the branches and the laziness of a repetition rank a
    // search's matches and change no word the pattern matches whole.
    assert_eq!(language("a|ab"), language("ab|a"));
    assert_eq!(language("a*?"), language("a*"));
    // And two that are *not* one language stay apart.
    assert_ne!(language("a"), language("b"));
    assert_ne!(language("a*"), language("a+"));
}

/// A pattern matches the whole word, which is what a `Regex` constraint
/// means: the walk matches against the text, not a substring of it.
#[test]
fn a_pattern_is_anchored_at_both_ends() {
    let a = language("a");
    assert!(a.holds(b"a"));
    assert!(!a.holds(b"ab"));
    assert!(!a.holds(b"ba"));
    assert!(!a.holds(b""));
    // An alternation is anchored as a whole rather than per branch, which is
    // why the pattern is wrapped before it is built.
    let either = language("a|bb");
    assert!(either.holds(b"a") && either.holds(b"bb"));
    assert!(!either.holds(b"abb") && !either.holds(b"ab"));
}

/// A pattern that parses only once wrapped is refused: `a)|(b` closes the
/// wrapper's group and opens one the wrapper closes, and the language built
/// would be one nobody wrote.
#[test]
fn a_pattern_that_parses_only_wrapped_is_refused() {
    for pattern in ["a)|(b", ")|(", "a)(b"] {
        assert!(
            RegularSet::pattern(pattern, Alphabet::Text).is_none(),
            "{pattern} built"
        );
    }
}

/// A length bound counts symbols, and which symbol depends on the alphabet:
/// a code point for text, a byte for bytes.
#[test]
fn a_length_bound_counts_the_alphabet_symbols() {
    let two_text = RegularSet::at_most(2, Alphabet::Text).expect("a small bound");
    assert!(two_text.holds(b"ab"));
    assert!(!two_text.holds(b"abc"));
    // Two code points, four bytes: the text alphabet counts the first.
    assert!(two_text.holds("éé".as_bytes()));
    assert!(!two_text.holds("ééé".as_bytes()));

    let two_bytes = RegularSet::at_most(2, Alphabet::Bytes).expect("a small bound");
    assert!(two_bytes.holds(b"ab"));
    // One code point is two bytes, so the byte alphabet counts two.
    assert!(two_bytes.holds("é".as_bytes()));
    assert!(!two_bytes.holds("éé".as_bytes()));

    // A minimum, and the two bounds meeting at an exact length.
    let least = RegularSet::at_least(2, Alphabet::Text).expect("a small bound");
    assert!(!least.holds(b"a") && least.holds(b"ab") && least.holds(b"abc"));
    let exactly = least.intersect(&two_text).expect("a small meet");
    assert!(exactly.holds(b"ab") && !exactly.holds(b"a") && !exactly.holds(b"abc"));
    // A minimum above a maximum admits nothing, which is a bound
    // conjunction decided by the language rather than by comparing bounds.
    let impossible = RegularSet::at_least(3, Alphabet::Text)
        .expect("a small bound")
        .intersect(&two_text)
        .expect("a small meet");
    assert!(impossible.is_empty());
}

/// A pattern whose determinisation is exponential refuses at the builder,
/// not after it.
///
/// `(a|b)*a(a|b){k}` doubles its DFA states for every `k`: the automaton
/// must remember the last `k` letters to know whether an `a` sat `k` back.
/// `MAX_STATES` is checked in `from_automaton`, which runs after
/// `regex-automata` has built the whole dense table, so before the size
/// limits this family allocated until it either answered slowly or aborted
/// the process -- 668 MB at `k = 20`, and a failed four-gigabyte allocation
/// at 25. The limits move the refusal to where the memory would be spent.
///
/// The small `k` still answers, so the bound is not a blanket refusal of
/// the shape.
#[test]
fn a_pattern_whose_determinisation_explodes_refuses_at_the_builder() {
    let family = |k: u32| format!("(a|b)*a(a|b){{{k}}}");
    assert!(RegularSet::pattern(&family(4), Alphabet::Text).is_some());
    for k in [24, 40, 64] {
        assert!(
            RegularSet::pattern(&family(k), Alphabet::Text).is_none(),
            "k = {k} must refuse rather than allocate"
        );
    }
    // A long pattern that stays small determinised is unaffected: the limit
    // is on the table, not on the source.
    assert!(RegularSet::pattern(&"a".repeat(2000), Alphabet::Text).is_some());
}

/// A pattern that does not build, and one whose automaton is too large, are
/// refused rather than answered.
#[test]
fn an_unbuildable_or_oversized_pattern_is_refused() {
    assert!(RegularSet::pattern("(", Alphabet::Text).is_none());
    assert!(RegularSet::pattern("a{2,1}", Alphabet::Text).is_none());
    // A byte pattern is not a text pattern: a non-UTF-8 byte class is
    // refused in text mode and built in byte mode.
    assert!(RegularSet::pattern(r"(?-u:\xFF)", Alphabet::Text).is_none());
    assert!(RegularSet::pattern(r"(?-u:\xFF)", Alphabet::Bytes).is_some());
    // A length bound past the state bound is refused rather than truncated.
    assert!(RegularSet::at_least(super::MAX_STATES + 1, Alphabet::Text).is_none());
}

/// The empty and universal languages, and the empty *word*, which is a
/// different thing from either.
#[test]
fn the_empty_language_and_the_empty_word_are_different_sets() {
    assert!(RegularSet::empty().is_empty());
    assert!(!RegularSet::empty().holds(b""));
    assert!(!RegularSet::all(Alphabet::Bytes).is_empty());
    assert!(
        RegularSet::all(Alphabet::Bytes).holds(b"")
            && RegularSet::all(Alphabet::Bytes).holds(b"anything")
    );

    let just_empty_word = RegularSet::word(b"");
    assert!(!just_empty_word.is_empty());
    assert!(just_empty_word.holds(b""));
    assert!(!just_empty_word.holds(b"a"));
    assert_eq!(just_empty_word, language(""));

    assert_eq!(
        RegularSet::empty().complement(),
        RegularSet::all(Alphabet::Bytes)
    );
    assert_eq!(
        RegularSet::all(Alphabet::Bytes).complement(),
        RegularSet::empty()
    );
}

/// The `str` kind's universe is the words a `str` can hold, and no others.
///
/// Every byte string is a `bytes`; a `str` is one that encodes a sequence of
/// code points, **surrogates included** -- `"\u{d800}"` is one character long
/// and is a member of `str`, and no codec writes it. So the language is WTF-8,
/// and the difference from every byte string is the whole of what a `str`
/// complement must not contain. The acceptor is held to the encoding's own
/// edges: the four sequence lengths, the surrogate block, and each way a byte
/// string fails to be one.
#[test]
fn the_text_universe_is_the_words_a_str_holds() {
    let text = RegularSet::all(Alphabet::Text);
    for word in [
        "".as_bytes(),
        "a".as_bytes(),
        "\u{e9}".as_bytes(),
        "\u{20ac}".as_bytes(),
        "\u{1f600}".as_bytes(),
        "a\u{e9}\u{1f600}".as_bytes(),
        "\u{10ffff}".as_bytes(),
        // The surrogate block, which a `str` carries and UTF-8 does not spell:
        // `U+D800`, `U+DFFF`, and one among ordinary characters.
        b"\xed\xa0\x80".as_slice(),
        b"\xed\xbf\xbf",
        b"a\xed\xa0\x80b",
    ] {
        assert!(text.holds(word), "a str's bytes: {word:?}");
    }
    for word in [
        b"\xff".as_slice(),
        b"\x80",
        b"\xc0\x80",
        b"\xc1\xbf",
        b"\xe0\x80\x80",
        b"\xf0\x80\x80\x80",
        b"\xf4\x90\x80\x80",
        b"\xf5\x80\x80\x80",
        b"\xc2",
        b"a\xc2",
        // A surrogate lead byte with no continuation is still a broken word.
        b"\xed\xa0",
    ] {
        assert!(!text.holds(word), "not a str's bytes: {word:?}");
    }
    assert!(RegularSet::all(Alphabet::Bytes).holds(b"\xff"));
}

/// A pattern's language holds no surrogate, and the kind's universe does.
///
/// The two are what the relation between `str` and a catch-all pattern rests
/// on: a `Regex` matches the *text* of a string, a lone surrogate has none, and
/// the difference between the kind and the pattern is exactly those words. A
/// universe that stopped where UTF-8 does made that difference empty.
#[test]
fn a_pattern_does_not_reach_the_words_only_the_kind_holds() {
    let universe = RegularSet::all(Alphabet::Text);
    let every_text = RegularSet::pattern("(?s:.)*", Alphabet::Text).expect("a catch-all builds");
    assert!(every_text.holds("a\u{1f600}".as_bytes()));
    assert!(!every_text.holds(b"\xed\xa0\x80"));
    let difference = universe
        .intersect(&every_text.complement())
        .expect("the difference builds");
    assert!(!difference.is_empty(), "the words no pattern reaches");
    assert!(difference.holds(b"\xed\xa0\x80"));
    assert!(!difference.holds(b"a"));
}

/// A length bound counts a surrogate as the one character it is.
#[test]
fn a_length_bound_counts_the_characters_a_str_holds() {
    let one_or_more = RegularSet::at_least(1, Alphabet::Text).expect("a bound builds");
    let at_most_one = RegularSet::at_most(1, Alphabet::Text).expect("a bound builds");
    assert!(one_or_more.holds(b"\xed\xa0\x80"), "one character");
    assert!(at_most_one.holds(b"\xed\xa0\x80"));
    assert!(!at_most_one.holds(b"\xed\xa0\x80\xed\xa0\x80"), "two");
    assert!(!one_or_more.holds(b""));
    // And the ordinary characters are counted as they were.
    assert!(at_most_one.holds("\u{1f600}".as_bytes()));
    assert!(!at_most_one.holds("ab".as_bytes()));
    assert!(
        one_or_more.holds("\n".as_bytes()),
        "a newline is a character"
    );
}

/// A complement cut to the text universe holds no word outside the encoding.
///
/// [`RegularSet::complement`] is the flip over every byte string, because a
/// `RegularSet` carries no alphabet. The cut belongs to the caller that knows
/// the kind, which for `str` is `descr/lines.rs`; this is the composition it
/// performs.
#[test]
fn a_text_complement_cut_to_its_kind_holds_only_str_words() {
    let letter = RegularSet::pattern("a", Alphabet::Text).expect("a pattern");
    let universe = RegularSet::all(Alphabet::Text);
    let outside = universe
        .intersect(&letter.complement())
        .expect("a product of two small tables");

    assert!(outside.holds("b".as_bytes()));
    assert!(!outside.holds("a".as_bytes()));
    assert!(!outside.holds(b"\xff"), "an invalid word is in no str set");
    assert!(
        letter.complement().holds(b"\xff"),
        "the uncut flip is the one over every byte string"
    );
    assert_eq!(
        letter.union(&outside),
        Some(universe),
        "the two halves are the kind"
    );
}

/// Complementing a minimal table leaves a minimal table.
///
/// The property `RegularSet::complement` relies on to skip a minimisation pass:
/// a word distinguishes two states in `L` exactly when it distinguishes them in
/// `¬L`, and the transitions are untouched, so both the partition and the
/// canonical numbering survive the flip.
#[test]
fn a_complement_is_already_minimal() {
    for pattern in ["a", "a*", "(a|b)*c", "[0-9]{2,4}", "", "a(bc)*d?"] {
        for alphabet in [Alphabet::Text, Alphabet::Bytes] {
            let Some(set) = RegularSet::pattern(pattern, alphabet) else {
                continue;
            };
            let once = set.complement();
            assert_eq!(
                once,
                once.complement().complement(),
                "{pattern:?} under {alphabet:?}"
            );
            assert_eq!(set, once.complement(), "the flip is an involution");
        }
    }
    let universe = RegularSet::all(Alphabet::Text);
    // The flip of the text universe is the words that are *not* UTF-8, which is
    // a language over bytes and not an empty one; cut back to the kind it is.
    assert!(!universe.complement().is_empty());
    assert_eq!(
        universe.intersect(&universe.complement()),
        Some(RegularSet::empty())
    );
    assert_eq!(universe.complement().complement(), universe);

    let bytes = RegularSet::all(Alphabet::Bytes);
    assert!(bytes.complement().is_empty());
    assert_eq!(bytes.complement().complement(), bytes);
}

/// A table whose lookups land outside it stops a debug build where it is
/// minimised, rather than reading the missing state as state zero.
///
/// The passes answer a miss with zero because a well-formed table has none,
/// and that is a different language wherever it does happen; the assertion is
/// what makes the answer about the table rather than about a lookup.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a table is minimised as it was built")]
fn a_table_with_a_transition_off_it_stops_a_debug_build() {
    let malformed = super::Dfa {
        classes: vec![0; 256],
        class_count: 1,
        transitions: vec![5],
        accepting: vec![true],
    };
    let _ = malformed.minimal();
}

/// Each clause of a well-formed table refuses a table that breaks it alone, at
/// the edge of the clause: a transition to the state one past the last, a class
/// equal to the class count, a row short by one, a byte with no class, and no
/// state at all. The table they are made from is well formed, with its last
/// state and its last class in use.
#[test]
fn each_defect_of_a_table_is_one_well_formed_refuses() {
    let good = || super::Dfa {
        classes: (0..256).map(|byte| u16::from(byte % 2 == 1)).collect(),
        class_count: 2,
        transitions: vec![1, 0, 1, 1],
        accepting: vec![false, true],
    };
    assert!(good().well_formed());
    let mut off_the_table = good();
    off_the_table.transitions[1] = 2;
    let mut class_past_the_count = good();
    class_past_the_count.classes[3] = 2;
    let mut short_row = good();
    short_row.transitions.pop();
    let mut byte_without_a_class = good();
    byte_without_a_class.classes.pop();
    let no_state = super::Dfa {
        classes: vec![0; 256],
        class_count: 1,
        transitions: Vec::new(),
        accepting: Vec::new(),
    };
    let no_class = super::Dfa {
        class_count: 0,
        ..good()
    };
    for (defect, table) in [
        ("a transition off the table", off_the_table),
        ("a class past the count", class_past_the_count),
        ("a short row", short_row),
        ("a byte without a class", byte_without_a_class),
        ("no state", no_state),
        ("no class", no_class),
    ] {
        assert!(!table.well_formed(), "{defect}");
    }
}
