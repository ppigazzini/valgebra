"""Name the commit a change is measured against, and print it.

The bench gate builds this commit beside `HEAD`, and the diff-scoped mutation
sweeps take `git diff <base>...HEAD` to choose their files, so all three need
the same answer to one question: where does this change start?

The event says: a pull request's base, or a push's `before`. A force-push
breaks the second. The `before` it names is the tip the push replaced, which no
branch reaches any more, so a checkout does not carry it -- and the fallback to
the default branch's tip is, for a push to the default branch, `HEAD` itself.
Measured from there, an amended commit changes nothing: the sweep reads `core
files: none` and passes, and a survivor the first push reported is gone from
the second.

So the base is read in this order:

1. `BASE_SHA`, when the checkout has it;
2. `BASE_SHA` fetched by its id, where the host still serves a commit no branch
   reaches;
3. the tip of `DEFAULT_BRANCH`.

and then taken to its merge base with `HEAD`, which is where the pushed commits
start: a rewritten history's fork point, not the tip it replaced. A base that
is still `HEAD` gives way to `HEAD`'s parent, so the last commit is measured
rather than nothing.

Usage (the two variables come from the workflow's event):
    BASE_SHA=... DEFAULT_BRANCH=main python scripts/change_base.py

Exit 0 with the commit on stdout; 2 when no base can be named.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

EXIT_OK = 0
EXIT_CANNOT_RUN = 2


def _git(*args: str) -> str | None:
    """Run git, giving its output, or None where it fails."""
    done = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() if done.returncode == 0 else None


def _commit(revision: str) -> str | None:
    return _git("rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}")


def _event_base(base: str) -> str | None:
    """Find the event's base in the checkout or, failing that, fetch it by id."""
    if not base:
        return None
    if (found := _commit(base)) is not None:
        return found
    # A commit no branch reaches is not in the checkout. A host that keeps it
    # serves it by id; one that does not refuses the fetch, and that is an
    # answer rather than an error.
    if _git("fetch", "--quiet", "--no-tags", "origin", base) is None:
        return None
    return _commit(base)


def _branch_tip(branch: str) -> str | None:
    if not branch or _git("fetch", "--quiet", "--no-tags", "origin", branch) is None:
        return None
    return _commit("FETCH_HEAD")


def main(argv: list[str]) -> int:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    ).parse_args(argv)
    head = _commit("HEAD")
    if head is None:
        print("change_base: HEAD names no commit", file=sys.stderr)
        return EXIT_CANNOT_RUN
    base = _event_base(os.environ.get("BASE_SHA", ""))
    if base is None:
        print("change_base: the event's base is not reachable", file=sys.stderr)
        base = _branch_tip(os.environ.get("DEFAULT_BRANCH", ""))
    if base is None:
        print("change_base: no base to measure from", file=sys.stderr)
        return EXIT_CANNOT_RUN
    base = _git("merge-base", base, head) or base
    if base == head:
        parent = _commit("HEAD~1")
        if parent is None:
            print("change_base: the base is HEAD, which has no parent", file=sys.stderr)
            return EXIT_CANNOT_RUN
        print("change_base: the base is HEAD; using its parent", file=sys.stderr)
        base = parent
    print(base)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
