# Security policy

valgebra is **pre-alpha** and not yet published to PyPI. It has not been
independently audited. Treat it accordingly: do not place it on a trust boundary
in production until it has a stable release and external review. This policy
describes what a security issue is and how to report one.

## Supported versions

Until a stable `1.0`, only the latest commit on `main` is supported. There are no
backports.

## What is a security issue

valgebra's load-bearing security property is **soundness of acceptance**: if
`is_valid` (or a non-raising `validate`/`validate_json`) reports a value as
valid, that value genuinely belongs to the schema's set. Downstream code trusts
that contract, so a value that is **wrongly accepted** is a vulnerability.

Please report, privately:

- **Unsound acceptance.** A value that is *not* a member of a schema's set is
  accepted — `is_valid` returns `True`, or `validate` does not raise, for a value
  the schema's denotation excludes.
- **A crash or unbounded resource use** on an input that is *within* the
  documented [resource limits](docs/limits.md): a native stack overflow, an abort,
  a hang, or memory growth unbounded by the value's size.
- **Memory unsafety.** The crates forbid `unsafe`, so any memory-safety defect is
  in scope.

## What is not a security issue

- **Conservatism of the decision procedure.** `is_subtype_of`, `is_equivalent`,
  and `is_empty` are deliberately conservative: a `False` (or "not empty") may be
  a relation valgebra cannot prove. This is documented in the
  [decidability boundary](docs/decidability.md) and is not a vulnerability.
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
  reporter unless anonymity is requested. Because there is no published release
  yet, a fix lands on `main`; once releases exist, a patched version ships with
  the advisory.
