# valgebra developer documentation

What the codebase does, for a contributor or an agent reading it cold. It
describes the tree **as it is**, not as it is intended to become; where a
capability is unbuilt, its page says so and says what the absence costs.

This set is not published. The site under [../](../README.md) is the user guide —
what valgebra does, for someone using it. These pages are what it is made of.

Read in order. Each page owns a zone of the source and is the live claim about
it.

| Page | Owns |
|---|---|
| [00-architecture.md](00-architecture.md) | the two crates, the module layers, the dependency direction |
| [01-schema-ir.md](01-schema-ir.md) | `crates/valgebra-core/src/ir.rs` — the node set and what each node denotes |
| [02-decision.md](02-decision.md) | `crates/valgebra-core/src/decision.rs` and `simplify.rs` — what is decided and what stays conservative |
| [03-frontend.md](03-frontend.md) | `crates/valgebra-py/src/build.rs` — typing annotations into the IR |
| [04-walk.md](04-walk.md) | `crates/valgebra-py/src/check/` — the membership walk, both input paths, three modes |
| [05-errors.md](05-errors.md) | the violation model, the error codes, the annotation render |
| [06-type-design.md](06-type-design.md) | the value domain: what each type denotes, why it has that shape, what it does not promise |
| [07-tooling-ci.md](07-tooling-ci.md) | the gates, the lanes, the exit codes, and each instrument's blind spots |
| [08-testing.md](08-testing.md) | the test layers, the two mutation sweeps, the eleven ledgers, the completeness probe |
| [09-releasing.md](09-releasing.md) | `.github/workflows/release.yml` and the version surfaces — the order a release is cut in |
| [10-theory.md](10-theory.md) | the papers the design rests on, and which code each touches |
| [11-references.md](11-references.md) | the reference implementations and the toolchain baseline |
| [12-writing.md](12-writing.md) | how to write a page here, and a comment in the source |
| [13-glossary.md](13-glossary.md) | the words this set uses without stopping to define them |

## Docs are part of the change, not after it

Each page is a live claim about code someone is about to touch. Change a zone,
fix its page **in the same commit**: a doc is wrong from the moment the code
lands, and nobody knows which claim broke better than the person who broke it.

```bash
python scripts/docs_lint.py
```

catches a dead link, a named path that does not exist, a reference into the
untracked working area, and a budget number copied into prose. It **cannot** tell
you a sentence has become false. That part is yours, and
[12-writing.md](12-writing.md) names the three classes it cannot reach.

## Two documentation surfaces

- **This set, plus [../../README.md](../../README.md),
  [../../CONTRIBUTING.md](../../CONTRIBUTING.md) and
  [../../AGENTS.md](../../AGENTS.md), ships.** A clone carries it.
- **A second surface is untracked**, so a clone does not. It holds the
  engineering contract, the operator prompt, the milestone backlog, and
  user-requested analyses.

Do not converge them. A shipped page must not carry campaign history, and an
untracked note must not be the only place a shipped fact lives. No shipped file
may name the untracked area's location either — `scripts/docs_lint.py` sweeps
every tracked file for it, because the reference dangles for every reader but its
author.
