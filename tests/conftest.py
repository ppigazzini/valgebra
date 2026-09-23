"""Shared pytest configuration: hypothesis example budgets.

Three property-test profiles select the example budget by the
``HYPOTHESIS_PROFILE`` environment variable, defaulting to ``dev``:

- ``dev`` (default, local): the fast budget for an edit-test loop.
- ``ci`` (pull requests): a wider budget that still finishes a merge gate, drawn
  derandomised so a red merge gate is the same red on a re-run.
- ``nightly`` (scheduled): the deep budget that hunts the long tail, drawn at
  random with a database, so each night asks examples the last one did not and
  a failure is kept for the replay.

Each profile says ``derandomize`` and ``database`` itself rather than inheriting
them. Hypothesis reads ``CI`` from the environment and makes its own ``ci``
profile the default a registration inherits from, which is derandomised with no
database: a ``nightly`` that named neither replayed the same examples every
night on a runner and hunted nothing.

The decision and completeness suites read these so the hole-hunting fuzzers run
shallow locally and deep on a schedule without per-test settings.

No per-example wall-clock deadline is set. A global deadline is a flaky guard on a
loaded shared runner, and too coarse to catch a superlinear regression that still
finishes one example under the ceiling. Algorithmic regressions are caught instead
by the deterministic cachegrind instruction-count gate (``scripts/perf_gate.py``,
core and binding), and the decision procedures are bounded by their own work
budget so they cannot run unbounded; a genuine hang is caught by the job timeout.
"""

import os

from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase

settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
    database=None,
)
settings.register_profile(
    "nightly",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=False,
    database=DirectoryBasedExampleDatabase(".hypothesis/examples"),
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
