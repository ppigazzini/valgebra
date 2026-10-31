"""Name the commit a change is measured against, and print it.

The bench gate builds this commit beside `HEAD`, and the diff-scoped mutation
sweeps take `git diff <base> HEAD` to choose their files, so all three need
the same answer to one question: what does this change change?

**A push is measured from the tip it replaced**, as that tip stands. The
difference between the two trees is what the push changes, whatever history
carries it: a line amended a hundred commits deep is one line, and the sweeps
take the one function holding it. Measured from the fork point instead, a
rewrite sweeps every function its rewritten commits touch, each of which their
first push swept already.

**A pull request is measured from where its branch left the base.** Its event
names the default branch's tip, which may have moved on since the branch left
it, and the difference from a moved tip holds that branch's commits too.

The base is read in this order:

1. `BASE_SHA`, when the checkout has it;
2. `BASE_SHA` fetched by its id, where the host still serves a commit no branch
   reaches -- a force-push's `before`;
3. the tip of `DEFAULT_BRANCH`, taken to its merge base with `HEAD`.

A pull request's base (`EVENT` is `pull_request`) is taken to its merge base
with `HEAD` as the fallback is. A base that is still `HEAD` gives way to
`HEAD`'s parent, so the last commit is measured rather than nothing: a force-push
to the default branch whose `before` the host no longer serves falls back to
the commit under test.

Usage (the three variables come from the workflow's event):
    BASE_SHA=... DEFAULT_BRANCH=main EVENT=push python scripts/change_base.py

Exit 0 with the commit on stdout; 2 when no base can be named.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from _lib import EXIT_CANNOT_RUN, EXIT_OK, arguments, git_output

#: The checkout this is asked about: the one it is started in, which is the
#: lane's checkout and, under test, a history built for the question.
CHECKOUT = Path()


def _git(*args: str) -> str | None:
    """Ask git about the checkout this runs in."""
    return git_output(*args, cwd=CHECKOUT)


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
    arguments(__doc__).parse_args(argv)
    head = _commit("HEAD")
    if head is None:
        print("change_base: HEAD names no commit", file=sys.stderr)
        return EXIT_CANNOT_RUN
    # A push's base is the tip it replaced, as it stands; a branch tip --
    # a pull request's base, or the fallback -- is where `HEAD` left it.
    forked = os.environ.get("EVENT", "") == "pull_request"
    base = _event_base(os.environ.get("BASE_SHA", ""))
    if base is None:
        print("change_base: the event's base is not reachable", file=sys.stderr)
        base = _branch_tip(os.environ.get("DEFAULT_BRANCH", ""))
        forked = True
    if base is None:
        print("change_base: no base to measure from", file=sys.stderr)
        return EXIT_CANNOT_RUN
    if forked:
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
