# projection_adaptive_buffer

Smoke test for the adaptive sort buffer feature (spec

Validates that `graph_project` completes successfully and produces
byte-identical output when `ExternalRecordSort` and `ExternalEdgeSort`
use the new adaptive default (sentinel `0` at construction time,
resolving to `max(256 MB, MemAvailable × 3/4)` at runtime).

The test does NOT measure wall-clock time — CI variance would make
that flaky. It only asserts:
1. `graph_project` completes without exception on a small graph.
2. The projection YIELD result matches the pre-adaptive baseline
   (so correctness of the sort is unaffected by the buffer size).

Dataset: 3 `Person` nodes, 3 `KNOWS` edges — small enough to always
fit even in the 256 MB floor, so the test exercises the default
path deterministically.

Related:
- `tests/gql/test_suites/projection_comprehensive/` — broader
  coverage that also exercises the adaptive path transparently
  after Task 10 of this plan.
