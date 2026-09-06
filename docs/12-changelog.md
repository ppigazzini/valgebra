---
description: Every released version, what it changed, and what to do on upgrade.
---

## Versioning, and what counts as a break

The version follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
and while the line is `0.x` the **minor** number is the breaking one: `0.1.0` to
`0.2.0` may break, `0.1.0` to `0.1.1` may not.

What a break means here is specific, because a validator has two surfaces and
only one of them is the API:

- **A break.** A value that validated and no longer does, or a value that did
  not and now does. An error `code` or a violation `path` that changes for a
  failure that was already reported. A name removed or renamed, a parameter that
  stops accepting what it accepted, an exception type that changes.
- **Not a break.** A relation that answered `False` ("not proven") and now
  answers `True`. Every relation is sound in both lines, and the conservative
  answer is documented as "no, or not proven"
  ([decidability](15-decidability.md)) -- so code that treats `False` as a proof
  of the negative was reading a guarantee that was never given. Widenings are
  listed in the notes below, and they are the commonest entry.
- **Not a break, and worth reading anyway.** A `repr` that changes, an
  `expected` string reworded, a message improved. They are described in the
  notes because tests pin them, and pinning one is a choice to be told.

## Deprecation

A name on its way out keeps working for **one minor release** and warns:
calling it raises a `DeprecationWarning` naming what replaces it, and the notes
below carry an entry saying when it goes. It is then removed in the next minor
release.

Nothing is removed without that window, and nothing that warns is left warning
indefinitely -- a warning that never resolves is a cost with no end, and a
removal with no warning is a break a reader had no way to see coming.

<!-- The repository's CHANGELOG.md, served verbatim. One list, rendered in two
     places: a copy here would be a second list nothing holds to the first. -->

--8<-- "CHANGELOG.md"
