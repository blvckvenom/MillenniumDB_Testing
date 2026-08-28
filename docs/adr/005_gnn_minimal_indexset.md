# ADR-005: GNN_MINIMAL Index Set for Projections

**Date:** 2026-04-25
**Status:** Accepted
**Supersedes:** None
**Context:** Projections always materialize 10 B+Tree indexes to support arbitrary GQL query patterns, but GNN workloads use only 5 of them. The cost of the unused 5 at papers100M scale is 57% of total projection storage (~106 GB out of 187 GB) and proportionally ~57% of build time.

---

## Context

### The measurement driving this decision

An empirical analysis of projection overhead (commit `62bf02be` and earlier, benchmark
script at `scripts/analyze_leaf_compression.py`) found that MillenniumDB builds 10
topology-related B+Tree indexes per projection:

| Index | Record<N> | Bytes/record | Needed for GNN sampling? |
|---|---|---|:---:|
| `nodes` | Record<1> | 8 | ✅ |
| `node_label` | Record<2> | 16 | ✅ |
| `label_node` | Record<2> | 16 | ✅ |
| `from_to_edge` | Record<3> | 24 | ✅ |
| `to_from_edge` | Record<3> | 24 | ✅ |
| `edge_from_to` | Record<3> | 24 | ❌ |
| `edge_direction` | Record<2> | 16 | ❌ |
| `edge_n1_n2` | Record<3> | 24 | ❌ |
| `edge_label` | Record<2> | 16 | ❌ |
| `label_edge` | Record<2> | 16 | ❌ |

GNN training uses only the 5 marked "✅" — topology traversal via `from_to_edge` +
`to_from_edge` plus node-label bookkeeping. The other 5 support GQL queries
(`MATCH (a)-[r:TYPE]->(b) WHERE id(r) = X`, etc.) that `gnn_offline_sample` and
`gnn_train` never invoke.

### Why this matters at scale

On papers100M (1.6 B edges):
- ALL mode: 187 GB, ~225 min (Run 7 measurement).
- GNN_MINIMAL expected: ~81 GB, ~97 min (projected 57% reduction).

On 30 GB RAM / 937 GB disk hardware (celebi), the 106 GB savings is the difference
between having disk headroom for GNN features (~56 GB) and training artifacts vs
running the projection build to within 50 GB of disk-full.

### Why existing knobs don't solve this

The projection builder has three pre-existing switches:
- `MDB_PROJECTION_SORTER=radix` (Spec #1) — changes sort algorithm, not index count.
- `MDB_PROJECTION_SERIAL_SCAN=1` (Spec #2) — changes scan pipeline, not index count.
- `MDB_SORT_BUFFER_MB=<mb>` — buffer size, not index count.

None control which of the 10 indexes get built. The feature described here is
orthogonal to all three and composes cleanly with them.

---

## Decision

Add a user-facing parameter `indexSet` to the `graph_project` GQL procedure config
map, with three preset values that correspond to specific subsets of
`ProjectionIndex` bits:

- `'ALL'` (default): 10 topology indexes materialized — exact current behavior.
- `'GNN_MINIMAL'`: 5 indexes materialized — sufficient for `gnn_offline_sample`,
  `gnn_materialize_batches`, `gnn_build_feature_store`, `gnn_train`,
  `gnn_predict`, and `EmbeddingWriter`'s on-the-fly k-hop.
- `'READONLY_TRAVERSAL'`: 7 indexes — `GNN_MINIMAL` + `edge_label` + `label_edge` for
  label-filtered GQL traversal without edge-id lookups.

The preset is persisted in the projection catalog (v1.4) and consulted at
both build time (gate `build_one_index()` calls) and query time (raise
`QueryException` with remediation hint when a dropped index is accessed).

Property indexes (`node_key_value`, `key_value_node`, `edge_key_value`,
`key_value_edge`) remain conditional on property configuration in the
`nodeProjection` / `relationshipProjection` maps — they are NOT affected by
`indexSet`. This preserves GNN feature-ingestion flow when the projection is
created with both `includeFeatures: 'X'` and `indexSet: 'GNN_MINIMAL'`.

---

## Alternatives considered

### A1 — Env variable per process

`MDB_PROJECTION_INDEX_SET=GNN_MINIMAL` analogous to `MDB_PROJECTION_SORTER`.

**Rejected:** env vars are process-level. Multiple projections built in the same
server process may legitimately want different index sets (one for analytics, one
for GNN). Config-map-per-call expresses this naturally; an env var would need
careful push/pop semantics to avoid contaminating sibling projections.

### A2 — Bitmask from string list

Let users specify arbitrary subsets like `indexSet: ['FROM_TO_EDGE', 'TO_FROM_EDGE']`.

**Rejected:** exposes internal enum to GQL layer, combinatorial test matrix
explodes (2^10 possible subsets), easy to produce non-functional configurations
(e.g., `FROM_TO_EDGE` without `NODES`). Presets enforce coherent combinations.

### A3 — Lazy/deferred index materialization

Build index on first query access.

**Rejected:** conflicts with the projection-is-immutable invariant (Spec #2 I6).
Lazy indexes require runtime mutation of the projection directory, concurrency
control, and partial-state rollback on crash — all of which break the invariant
that gave Specs #1 and #2 their simplicity.

### A4 — Make the default GNN_MINIMAL

Flip default from `ALL` to `GNN_MINIMAL`, saving disk for 95% of users.

**Rejected:** silent behavior change. Existing test suites and user workflows
depending on full-index projections would break without clear migration path.
Default `ALL` preserves backwards compatibility; opt-in migration for users who
actually benefit from the smaller footprint.

### A5 — Arbitrary per-index boolean config

`{buildFromToEdge: true, buildEdgeLabel: false, ...}` explicit map.

**Rejected:** 10+ new config keys, user-error-prone, hard to document coherently.
Presets are cleaner and cover 99% of real usage patterns.

---

## Consequences

### Positive

1. **Disk savings on GNN workloads.** Measured -61% on ogbn-products (Gate A
   benchmark). Extrapolated -57% on papers100M.
2. **Wall-clock savings** proportional to disk savings. Measured -57% on
   ogbn-products. Dominant effect is elision of 3 of 6 edge-index sort passes.
3. **Composable with prior Specs.** Works under {classic, radix} × {serial,
   parallel} scan combinations; 12-mode golden compare validates byte-identical
   behavior for indexes common across modes.
4. **Clear migration path.** Catalog v1.4 reads v1.3 projections as `ALL`;
   existing projections work unchanged.
5. **Unified query error semantics.** Missing-index access raises a formatted
   `QueryException` naming the preset, the missing index, and the minimum
   preset that would fix the query.

### Negative

1. **Per-projection preset choice moves complexity to the user.** GNN users
   must know to pass `indexSet: 'GNN_MINIMAL'`; absence defaults to ALL which
   still works but is wasteful. Mitigated by wiki documentation and benchmark
   table showing concrete savings.
2. **Preset is fixed at creation.** Rebuilding is required to switch presets —
   no "promote READONLY to ALL in place". Mitigated by making `drop_projection`
   + re-create cheap; projection builds are fast enough that rebuild is
   acceptable.
3. **New surface area for tests.** 347 GQL integration tests now run under
   `ALL` mode implicitly; ~60 tests legitimately requiring dropped indexes are
   annotated `skip-in-gnn-minimal` (T3.12). Classic path and serial path both
   require the guard (T3.7 + T3.8).
4. **Query planner must be taught the preset.** The `gql_model.cc` layer gained
   13 new null-reader checks (T3.9) that throw clear errors instead of
   segfaulting. Some queries surface errors as "missing edge_from_to" when
   the user might expect "missing edge_label" — the error message names the
   actual missing index, not the one the user was thinking of.

### Neutral

1. **Property indexes unchanged.** `node_key_value` etc. stay gated only by
   the property config. GNN feature ingestion still works with
   `indexSet: 'GNN_MINIMAL' + includeFeatures: 'X'`.
2. **Memory usage unchanged.** Peak RSS on ogbn-products was 1935 MB with
   GNN_MINIMAL vs 1999 MB with ALL — within noise. Sort scratch dominates,
   not index count (Spec #1 RADIX bounded at 2 GB worker pool).

---

## Implementation commits

- `2ac17a56`: IndexSet enum + preset mask helper.
- `62bf02be`: review fixes — QueryException instead of std::invalid_argument,
  assert-on-unknown for safer default, compile-time drift guard.
- `2593b3fd`: thread IndexSet through `graph_project` → `NativeProjectionBuilder`.
- `2a416fe2`: catalog v1.4 persists IndexSet with v3→v4 read migration.
- `96d52bc5`: gate `build_one_index()` in both SERIAL and CLASSIC paths; also
  gate `open_all_bplustree_readers_()` to prevent O_CREAT producing 0-byte
  .leaf files for elided indexes.
- `d0f478b0`: GQL layer raises friendly `QueryException` on missing-index
  access with remediation hint.
- `3ce99cd1`: `bench_indexset.sh` — Gate A measurement harness.

---

## Validation evidence

- Full GQL regression: 347/347 passes (verified in T3.9 commit `d0f478b0`).
- Unit tests: IndexSet 14 + projection_missing_index 15 + projection_storage
  v1.4 5 = 34 new tests, all pass.
- Integration shell scripts:
  - `scripts/test_projection_indexset_build.sh`: 86 checks covering 3 presets ×
    2 scan paths + byte-identical comparison.
  - `scripts/test_projection_missing_index_query.sh`: 8 E2E scenarios for error
    messages across presets.
- Benchmark report: `docs/research/2026-04-25-indexset-bench.md` confirms
  Gate A criteria met on ogbn-products.

---

## Task scope — Node and Link Property Prediction

`GNN_MINIMAL` is sized specifically for **Node Property Prediction (NPP)** and
**Link Property Prediction (LPP)** workloads — the two GNN task families that
MillenniumDB's current procedure surface is designed for. This section
documents which indexes each task requires, so future readers understand why
the 5-index subset is sufficient and when a larger preset is the right call.

### Node Property Prediction (NPP) — the default thesis workload

**Examples:** Cora, ogbn-arxiv, ogbn-products, papers100M (classifying papers
by subject, users by activity, molecules by property).

**Pipeline stages and index requirements:**

| Stage | Operation | Index used |
|---|---|---|
| `gnn_offline_sample` | k-hop seed sampling, node-centric | `from_to_edge` (NATURAL/UNDIRECTED) + `to_from_edge` (REVERSE/UNDIRECTED) |
| `gnn_materialize_batches` | persist sampled subgraphs to disk | same as above |
| `gnn_build_feature_store` | classify node frequencies + build L1-L4 caches | `nodes`, `node_label`, plus property indexes (not controlled by IndexSet) |
| `gnn_train` | forward/backward with `labelProperty` + `splitProperty` | all of the above; labels are stored in `labels.bin` (per-node int64, indexed by RowMapping — not a B+Tree query) |
| `EmbeddingWriter` (Phase 6) | on-the-fly k-hop for non-seed nodes + tensor property write-back | `from_to_edge` + `to_from_edge` + property index for the write target |

NPP never asks "given edge X, what are its endpoints?" — always "given node
v, what are its neighbors?". The 5-index `GNN_MINIMAL` subset exactly
matches the node-centric access pattern.

### Link Property Prediction (LPP) — natural extension

**Examples:** ogbl-citation (predict citation links), ogbl-collab,
recommender systems over single-type graphs.

**Pipeline sketch (not yet implemented in MDB, but would reuse the existing
sampler):**

```
for (src, dst) in positive_edges:
    neg_dst = random_node()
    h_src = k_hop_sample(src)           # node-centric via from_to_edge
    h_dst = k_hop_sample(dst)           # node-centric
    h_neg = k_hop_sample(neg_dst)       # node-centric
    pos_score = scoring_fn(h_src, h_dst)
    neg_score = scoring_fn(h_src, h_neg)
    loss = margin_loss(pos_score, neg_score)
```

The outer iteration over `positive_edges` is a **linear scan** of
`from_to_edge.leaf`, not an edge-by-id lookup. `edge_from_to` remains
unnecessary even when the training loop iterates over edges, because
the record already contains `(from, to, edge_id)` — you get the endpoints
directly from the iteration.

**Two LPP subcases:**

1. **Homogeneous edges** (one relationship type, e.g. "CITES" in a
   citation graph): `GNN_MINIMAL` is sufficient. The 5-index subset
   supports both endpoint-sampling and positive-edge iteration.

2. **Heterogeneous edges** (multiple types, e.g. "DRUG_TARGET" and
   "PROTEIN_PROTEIN" in ogbl-biokg): `READONLY_TRAVERSAL` is required.
   The scoring function typically dispatches per edge type, so
   `MATCH ()-[r:DRUG_TARGET]->()` filtering at training time needs
   `edge_label` + `label_edge` — which `GNN_MINIMAL` drops but
   `READONLY_TRAVERSAL` keeps.

**Property access during LPP:** if edges carry properties (weight,
timestamp, confidence score) that the scoring function reads, those are
covered by the property indexes (`edge_key_value`), which are orthogonal
to IndexSet and gate on `relationshipProperties` in the projection
config. A GNN_MINIMAL or READONLY_TRAVERSAL projection with
`relationshipProperties: ['weight']` materializes `edge_key_value`
regardless of the topology IndexSet — so LPP with weighted edges is
fully supported.

### Preset-to-task reference

| Task | Recommended preset | Commit with example config |
|---|---|---|
| Node classification (GraphSAGE, GIN node-level) | `GNN_MINIMAL` | existing `gnn_train` procedure |
| Node regression | `GNN_MINIMAL` | same pipeline with regression loss |
| Link prediction (homogeneous) | `GNN_MINIMAL` | natural extension; sampler reusable |
| Link prediction (heterogeneous / typed) | `READONLY_TRAVERSAL` | requires per-type `edge_label` filter |
| Knowledge-graph completion (TransE / RotatE style) | `READONLY_TRAVERSAL` | needs `edge_label` for relation embeddings |
| GQL exploratory / analytics | `ALL` (default) | preserves full SQL-like query surface |

### What is explicitly out of scope

This ADR does NOT cover Graph Property Prediction (graph classification /
regression, e.g. molecule datasets like ogbg-molhiv). The projection
model is single-graph; GPP datasets are multi-graph and require a
different storage abstraction (one record per graph, with per-graph
adjacency + features). If MDB later adds GPP support, it will need a
new storage primitive — not a new IndexSet preset.

---

## Forward-looking

After Spec #3, the natural next step is Spec #4 (`TopologySnapshot` — optional
mmap-backed CSR sibling file). Spec #3's `GNN_MINIMAL` preset reduces the number
of indexes, but each kept index is still a B+Tree with O(log N) lookup — this is
the bottleneck Spec #4 addresses. The two are composable: `GNN_MINIMAL` shrinks
disk, `TopologySnapshot` accelerates sampling; stacked, papers100M projection
moves from "187 GB + 3K seeds/sec" to "81 GB + 200K seeds/sec" while the B+Tree
API remains intact for GQL queries.
