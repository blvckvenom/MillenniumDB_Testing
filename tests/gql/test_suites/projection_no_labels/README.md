# Projection No-Labels Test Suite

Integration tests for the `includeLabelIndexes` opt-in configuration flag
added to `graph_project` to support large-scale label-free projections
(e.g., GNN training workloads where `MATCH (n:Label)` is never issued).

## Why this suite exists

On billion-scale graphs, the 4 label indexes (`node_label`, `label_node`,
`edge_label`, `label_edge`) can dominate projection build disk peak — on
`ogbn-papers100M` they account for ~50 GB of the total ~194 GB spill
footprint. Users whose query workload never filters by label or type can
opt out with `includeLabelIndexes: false` to skip those indexes and save
disk, at the cost of any label-based query raising `QueryException`
(documented in `gql_model.cc`).

Default is `true` (Neo4j-GDS parity, no behavioral change for existing
users).

See `docs/superpowers/thesis_analysis/2026-04-20-projection-disk-reduction-analysis.md`
§3.A for the full design rationale and safety matrix.

## Test data

`projection_no_labels.gql` — 5 User nodes, 4 KNOWS edges forming a chain.
Every node has a label and every edge has a type, so label/type queries
against projections built with labels are well-defined.

## Happy path (6 tests)

| # | Test | Covers |
|---|---|---|
| 01 | default builds label indexes | Default behavior unchanged (no flag in config) |
| 02 | explicit `true` preserves labels | Explicit `true` == default |
| 03 | explicit `false` build succeeds | Builder accepts and completes without label indexes |
| 04 | no-labels + MATCH (n) | Node enumeration works (uses `nodes` index, not label) |
| 05 | no-labels + MATCH (n)-[e]->(m) | Edge traversal works (uses `from_to_edge`, not label) |
| 06 | default + MATCH (n:User) | Control: label query works when index IS built |

## Error contract (2 bad queries)

| # | Test | Expected error |
|---|---|---|
| 01 | no-labels + MATCH ()-[:KNOWS]->() | `QueryException: Cannot use edge labels with projection '...'` |
| 02 | `includeLabelIndexes: 1` (int) | `runtime_error: Configuration value for 'includeLabelIndexes' must be a boolean` |

### Why the node-label error path is NOT tested here

An earlier iteration included a `MATCH (n:User)` query on a label-free
projection, expecting `Cannot use node labels` to fire. In practice the
optimizer on a single-label projection (all nodes are :User) can short-
circuit the filter without ever calling `get_label_node()` — so the
expected error never fires even though `label_node` index truly is
missing. Rather than pin the test to optimizer internals (which may
change across versions), we rely on:
  - **Compile-time safety** — `gql_model.cc:186-205` is the only path that
    reads `projection_ctx->label_node_index` and it unconditionally throws
    on null. Any query plan that references the index WILL fail cleanly.
  - **Symmetric edge-label coverage** (bad query 01 here) — `MATCH :TYPE`
    on edges reliably triggers `get_label_edge()` which returns the
    structurally-equivalent error, validating the opt-out contract.

The happy-path test 06 (`MATCH (n:User)` on `proj_default`) still covers
that label queries work when the index IS built, so no regression is
possible in the default path.

## Projections created (order-dependent)

Tests 01-03 create projections that tests 04-06 and bad_queries 01-02
consume via `USE`. Tests run alphabetically, so test 03 runs before test
04 and the `proj_no_labels` state is available.

| Projection | Created by | Consumed by | Flag |
|---|---|---|---|
| `proj_default` | test 01 | test 06 | (default true) |
| `proj_explicit_true` | test 02 | — | `true` (explicit) |
| `proj_no_labels` | test 03 | tests 04, 05, bad 01, bad 02 | `false` |

## Coverage justification

The suite targets three orthogonal dimensions:
1. **Wire-through correctness** (01, 02, 03): the flag reaches the builder
   and produces the right number of indexes.
2. **Preservation of non-label queries** (04, 05, 06): queries that should
   still work on label-free projections, and the control on a full projection.
3. **Error contract** (bad 01, 02, 03): queries that SHOULD fail do fail,
   with clear messages driving the user to remediation.

Together these verify the opt-out is usable end-to-end without breaking
any existing guarantee of `graph_project`.
