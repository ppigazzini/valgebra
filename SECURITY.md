# Security policy

valgebra is **pre-alpha**. Pre-release `0.0.x` wheels are published to PyPI, but
the code has not been independently audited. Treat it accordingly: do not place
it on a trust boundary in production until it has a stable release and external
review. This policy describes what a security issue is and how to report one.

## Supported versions

On the `0.0.x` line only the latest published release receives fixes, and a fix
supersedes the release it patches rather than being backported to an earlier
one.

## What is a security issue

valgebra's load-bearing security property is **soundness of acceptance**: if
`is_valid`, `in` or `is_valid_json` (or a non-raising `validate`,
`validate_json`, `ensure` or `load`) reports a value as valid, that value genuinely belongs to the schema's set. Downstream code trusts
that contract, so a value that is **wrongly accepted** is a vulnerability.

A relation's proof carries the same contract one step removed. The
[decidability page](docs/15-decidability.md) documents a positive answer as a
guarantee, so a caller that skips checking a value of `A` against `B` because
`A` is proved a subtype of `B` accepts that value on the proof's word.

Please report, privately:

- **Unsound acceptance.** A value that is *not* a member of a schema's set is
  accepted — `is_valid` returns `True`, or `validate`, `ensure` or `load` does not
  raise, for a value
  the schema's denotation excludes.
- **An unsound proof.** `is_subtype_of`, `is_equivalent` or `is_empty` returns
  `True`, or `relation_to` answers `"subset"`, where the schemas' denotations
  say otherwise.
- **A crash or unbounded resource use** on an input that is *within* the
  documented [resource limits](docs/10-limits.md): a native stack overflow, an abort,
  a hang, or memory growth unbounded by the value's size.
- **Memory unsafety.** The crates forbid `unsafe`, so any memory-safety defect is
  in scope.

## What is not a security issue

- **Conservatism of the decision procedure.** `is_subtype_of`, `is_equivalent`,
  and `is_empty` are deliberately conservative: a `False` (or "not empty"), and
  `relation_to`'s `"undecided"`, may be a relation valgebra cannot prove. This is
  documented in the [decidability boundary](docs/15-decidability.md) and is not a
  vulnerability. A wrong `"not_subset"` is a defect to report as an issue: a
  refutation sends a caller to check a value, never past the check.
- **Rejecting malformed or hostile input.** A value meeting a documented limit —
  an over-deep nesting, a self-referential value — is *meant* to be rejected;
  that the guard fires is correct behavior.
- **Resource cost driven by the schema you wrote.** The limits bound work driven
  by the untrusted *value*; the size of a schema you author (the width of a
  union, the number of fields) is yours to choose.

## Reporting a vulnerability

Do **not** open a public issue for a suspected vulnerability.

- Preferred: GitHub's private vulnerability reporting — the **Report a
  vulnerability** button under the repository's *Security* tab — which opens a
  private advisory with the maintainer.
- Alternatively, email the maintainer at <pasquale.pigazzini@gmail.com> with
  "valgebra security" in the subject.

Please include a minimal reproduction: the schema, the value, what verdict you
got, and what verdict the denotation requires.

## Process

This is a solo, pre-release project, so timelines are best-effort:

- Acknowledgement within **7 days**.
- An initial assessment (in scope, severity) within **30 days**.
- Coordinated disclosure: a fix and an advisory published together, crediting the
  reporter unless anonymity is requested. A patched `0.0.x` release ships to PyPI
  alongside the advisory.
