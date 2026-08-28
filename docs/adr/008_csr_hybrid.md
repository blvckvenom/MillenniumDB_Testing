# ADR-008: CSR-in-B+Tree Hybrid for Edge Indexes under GNN Workloads

**Date:** 2026-04-24
**Status:** Accepted — Gate D PASS-WITH-CAVEATS (size + scan + correctness pass; sampling partial, pending Spec #8-B).
**Supersedes (conditionally):** ADR-006 (Spec #4-B `TopologySnapshot`) when `graphStorage: 'CSR_HYBRID'` is set — the sidecar is silently skipped since the B+Tree leaves already carry the CSR layout.
**Context:** Spec #4-B (`TopologySnapshot` sidecar) solved the GNN sampling latency problem by emitting a pair of mmap-backed CSR files next to the B+Tree, at the cost of roughly 34 % extra disk per projection. Spec #5 (`DELTA_VARINT`) compressed the B+Tree leaves themselves to ~20 % of the bitset baseline but left the structure untouched. At papers100M scale the sidecar duplication still costs ~7 GB that we already paid for inside the B+Tree. Spec #8 unifies both by making the B+Tree leaf pages **be** the CSR: the edge indexes store per-page CSR slices, eliminating the sidecar round-trip and cutting disk footprint a further 80 % over the v2 baseline.

---

## Context

**What Spec #4-B left on the table.** ADR-006's `topology_fwd.csr` +
`topology_rev.csr` sidecars deliver O(1) neighbour access but
materialise a second adjacency copy on disk: ≈ 37 % duplication on
cora_gnn (150 KB vs 55 KB), ≈ 34 % on ogbn-arxiv (12.7 MB vs 4.4 MB),
≈ 32 % projected on papers100M (~22 GB vs ~7 GB). The duplication is
structural: every edge appears once in the B+Tree leaf and once in
`COL_IDX[]`. The sidecar exists only because the B+Tree encoding does
not expose random access to a node's neighbour list.

**What Spec #5 left on the table.** ADR-007 switched leaf encoding
from bitset (v1) to zigzag-delta LEB128 (v2), dropping `from_to_edge`
on ogbn-arxiv from 28 MB → 5.5 MB (0.195×). But the record layout
remained `Record<3> = (src, dst, edge_id)`, so the src column is
still materialised once per edge — ~1.5 B/record after compression.

**The observation.** On any edge index sorted by `(src, dst, edge_id)`,
a single 4 KB leaf holds records sharing at most a handful of distinct
src values (typically 1-3 on ogbn-arxiv, 1-10 on papers100M hubs).
Collapsing repeated src bytes into an offset table + dst-stream drops
the per-page cost from ~1.5 B/edge (Spec #5) to ~0.5 B/edge (Spec #8),
with O(1) random access for a given src instead of v2's O(log k).

---

## Decision

Introduce a **page-level v3 leaf format** (`LeafFormat::CSR_HYBRID`),
opt-in per projection via the new GQL config key `graphStorage`. The
v3 page layout:

- 16-byte header (`format_version=3`, `record_width=3`,
  `num_src_nodes`, `num_edges`, `next_leaf`, reserved).
- Offset table (`num_src_nodes + 1` uint32 entries).
- Src table (full LEB128 src-ids).
- CSR entries: per src, sorted dst list using Spec #5's DELTA_VARINT
  codec (zigzag-delta + LEB128).
- Zero-pad to 4 KB.

`graphStorage` is persisted **per projection** as a uint8 in catalog
v1.6. **Scope = edge indexes only** (`FROM_TO_EDGE`, `TO_FROM_EDGE`);
all other indexes continue to use v1/v2 per `leafFormat`. The reader
(`BptIter`) dispatches 3-way at page-open on the header magic byte:
`01 → BPTLeafV1`, `02 → BPTLeafV2`, `03 → BPTLeafCSR`. The writer
(`BPTLeafCSRWriter`) is wired into both classic and radix bulk-load
pipelines. When `graphStorage: 'CSR_HYBRID'` is set,
`buildTopologySnapshot` is silently ignored.

---

## Alternatives considered

**L1 — Fixed-width uint32 dst column.** Rejected: couples Spec #8 to
unfinished Spec #6 bit-packing; DELTA_VARINT already delivers
~0.5 B/edge, undercutting 4 B fixed.

**L2 — Per-page Huffman over dst stream.** Rejected: +300 LOC codec,
regresses fuzz surface, 4-6 % marginal gain on synthetic arxiv.

**L3 — Row-major with src implicit on row index.** Rejected: requires
fixed-degree per src; papers100M median degree ~7, fixed padding would
waste ~50 %.

**H1 — Overflow pages for hubs.** Deferred to Spec #8-B: arxiv max
degree 13 161 fits single page post-DELTA_VARINT; papers100M hubs
(max 3.8 M) do need it.

**H2 — Split hub into consecutive leaves (chosen).** Reuses existing
`next_leaf` chain; no new pointer type; O(d/k) page reads for a
degree-d hub.

**H3 — Separate page-pool keyed by hub src.** Rejected: splits the
single `.leaf` file into two physicals, breaks every operational tool.

**R1 — Binary search over offset table (chosen).** O(log k) per src
lookup; k typically 1-3, amortises to ~2 comparisons.

**R2 — Linear scan over src table.** Retained only as debug-build
cross-check under `MDB_BPT_LEAF_CSR_LINEAR_VALIDATE=1`.

**C1 — Per-index catalog byte (as Spec #5 did).** Rejected: CSR_HYBRID
meaningful only on edge indexes; per-index would record NO-OP values.

**C2 — Per-projection catalog byte (chosen).** Minimal catalog churn
(1 byte), unambiguous semantics, dispatch table precomputed at open.

---

## Consequences

### Positive

1. **Storage reduction on edge indexes:** -81.5 % on ogbn-arxiv
   (35 MB → 6.5 MB, 0.185×), -63.9 % on cora_gnn (150 KB → 54 KB).
   Composes with DELTA_VARINT on non-edge indexes.
2. **Supersedes Spec #4-B sidecar:** silently skips the ~34 % extra
   disk cost and removes the SHA-256 staleness round-trip at open.
3. **Scan parity (post T8.12b):** 1.046× vs bitset baseline on
   ogbn-arxiv after the sequential cursor cache in `BPTLeafCSR`
   (commit `b9ca276f`, 343× speedup vs pre-fix O(n²)). Below
   Gate D's 1.20× threshold.
4. **GNN sampling on cora validated** — end-to-end
   `gnn_offline_sample` byte-identical to the v2 reference (T8.10).
5. **Composes with Specs #1-#5:** any {classic, radix} × {ALL,
   GNN_MINIMAL, READONLY_TRAVERSAL} × {BITSET, DELTA_VARINT}
   combination produces v3 on edge indexes when CSR_HYBRID is set.
6. **Opt-in zero regression:** all 347 GQL + 181 MQL + 809 SPARQL
   integration tests green across all 13 implementation commits with
   `graphStorage` defaulting to `'BTREE'`.
7. **Fail-safe decode:** 500 K random + 10 K boundary + 1 K
   tamper-flip fuzz iterations under seed `0xC5B8_1234_5678_9ABC`,
   zero mismatches, 100 % tamper detection.

### Negative

1. **`edge_id` not persisted in v3.** General GQL `count(e)` returns
   inflated counts or synthesised ids. Tracked for Spec #8-B via an
   optional `EDGE_IDS[]` stream gated by a header flag.
2. **ogbn-arxiv CSR sampling hits `get_dst_at` decode_tuple_ failure
   at position 3974.** cora_gnn passes; arxiv-scale regression
   tracked as pending Spec #8-B fix. Bench script skips this combo.
3. **v3 pages immutable.** Switching `graphStorage` requires
   `drop_projection` + recreate. Mitigated by cheap builds (~3 s on
   ogbn-arxiv).
4. **Adds one virtual dispatch per page open** (~2-4 ns indirect
   call, dominated by the disk read it follows).
5. **Hub pages may span multiple leaves under H2.** papers100M hubs
   (max degree 3.8 M) occupy ~940 leaves each; Spec #8-B's H1 fixes.

### Neutral

1. **Catalog +1 byte per projection** (the `graphStorage` field in
   v1.6); pre-Spec-#8 catalogs read implicitly as `'BTREE'`.
2. **16-byte header per page** — identical to v2.
3. **24 B mutable reader cache per `BPTLeafCSR` instance** (cursor
   triple: last-src + last-offset + last-dst-count).
4. **Build time unchanged** (~0.06 s cora_gnn, ~3.1 s ogbn-arxiv) —
   sort dominates.

---

## Implementation commits

Chronological on `feature-GNN`:

- `cae73fe7` — T8.1 + T8.2: Spec #8 design + plan.
- `e68c6581` — T8.3: `LeafFormat::CSR_HYBRID` enum + v3 header struct.
- `106abfd0` — T8.4: `BPTLeafCSR` reader (offset-table lookup +
  DELTA_VARINT dst decode).
- `b94c4ca2` — T8.5: `BPTLeafCSRWriter` bulk-load (src grouping,
  offset table, hub-split per H2).
- `a2711fd9` — T8.6: `BptIter` 3-way dispatch on LeafFormat magic.
- `f23e6284` — T8.7: catalog v1.6 per-projection byte; v1.5 → v1.6
  read migration (implicit BTREE).
- `1a68c587` — T8.8: GQL parser + `project_procedure.cc` accept
  `graphStorage`; thread through `NativeProjectionBuilder`.
- `652174bd` — T8.9: integration + sidecar-skip rule +
  `TopologyAccessor` v3 short-circuit.
- `5633c5f1` — T8.10: 4-mode golden compare
  (`scripts/test_projection_csr_hybrid.sh`).
- `0babc891` — T8.11: 500 K-iteration fuzz
  (`bpt_leaf_csr_fuzz_test`).
- `7e56cc01` — T8.12: bench script + initial run (revealed O(n²)).
- `b9ca276f` — T8.12b: sequential cursor cache in
  `BPTLeafCSR::search_index` (343× scan speedup).
- T8.13 + T8.14 land concurrently with this ADR.

---

## Validation evidence

- **Gate D pass-with-caveats:** 7 / 8 criteria green; open caveat
  arxiv CSR sampling (Spec #8-B).
- **T8.11 fuzz:** 500 K random + 1 K smoke (seed +1) + 10 K boundary
  (N ∈ {1, 2, 3}) + 1 K tamper-flip — zero mismatches, 100 %
  detection.
- **T8.10 golden compare:** 4-mode matrix (BTREE / CSR_HYBRID ×
  {classic, radix}) on cora_gnn — all byte-identical.
- **T8.13 bench:** 0.185× size baseline on ogbn-arxiv; 1.046× scan
  (pre-fix was 343×); cora GNN sampling byte-identical to sidecar.
- **Regression suites:** 347/347 GQL + 181/181 MQL + 809/809 SPARQL +
  375 GNN unit + 73 E2E + 25 gnn_training green across all 13
  commits.
- **Unit tests added:** `bpt_leaf_csr_format_test`,
  `bpt_leaf_csr_reader_test`, `bpt_leaf_csr_writer_test`,
  `bpt_leaf_csr_fuzz_test`, extended `bpt_iter_dispatch_test`,
  `projection_catalog_v6_test`,
  `projection_graph_storage_config_test`,
  `graph_storage_integration_test`.

---

## Related ADRs

- ADR-004 (`004_radix_partition_sort.md`): RADIX sort backend.
- ADR-005 (`005_gnn_minimal_indexset.md`): IndexSet preset.
- ADR-006 (`006_topology_snapshot.md`): TopologySnapshot sidecar —
  conditionally superseded by Spec #8 under opt-in.
- ADR-007 (`007_delta_varint_leaf.md`): DELTA_VARINT leaf — composes
  with CSR_HYBRID via the dst-stream codec.

Together, ADR-004 + ADR-005 + ADR-006 + ADR-007 + ADR-008 form the
**GQL projection compression stack** for the thesis.

---

## Follow-ups (Spec #8-B and beyond)

- **`edge_id` stream persistence** — add optional `EDGE_IDS[num_edges]`
  section to the v3 page gated by a header flag. Restores exact
  `count(e)` semantics.
- **arxiv CSR sampling `get_dst_at` decode_tuple_ fix** — investigate
  the position-3974 failure; likely a hub-split boundary interaction
  with the sequential-cursor cache + DELTA_VARINT resync.
- **Custom `TopologyAccessor` fast-path** — direct `BPTLeafCSR::slice(src)`
  bypassing `BptIter` for an additional 2-5× sampling speedup.
- **H1 root-overflow pages** — for papers100M hubs (max degree
  3.8 M) where H2's in-chain split costs ~940 leaves per hub.
- **Spec #9 HNSW auto-integration** — index Phase 6 queryable
  embeddings into HNSW at `gnn_train` finalisation (separate roadmap).

**End of ADR.**
