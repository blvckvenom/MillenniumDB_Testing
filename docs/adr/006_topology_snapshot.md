# ADR-006: TopologySnapshot — mmap-backed CSR Sidecar for Projection Sampling

**Date:** 2026-04-25
**Status:** Accepted (pending Gate B sign-off)
**Supersedes:** None
**Context:** After Spec #3 (`GNN_MINIMAL` IndexSet) a papers100M projection shrinks from 187 GB to ~81 GB, but neighbor lookup remains `O(log N)` per edge via B+Tree. At 1.6 B edges with fanout [15, 10] this gives ~1 K seeds/sec — ~30 h/epoch on papers100M. GNN training needs ~30 K seeds/sec (30 min/epoch) to be practical within a thesis calendar.

---

## Context

### The measurement driving this decision

Empirical sampling throughput (benito_pc, fanout [15, 10], UNIFORM strategy, 10 K seeds):

| Dataset | Edges | seeds/sec | Time for 1 epoch on full graph |
|---|---:|---:|---|
| cora_gnn | 5 K | ~150 K | instant |
| ogbn-arxiv | 1.2 M | ~8 K | ~21 s for 169 K seeds |
| ogbn-products | 62 M | ~2-4 K | ~12-24 min for 2.5 M seeds |
| papers100M (projected) | 1.6 B | ~0.5-1 K | ~30 h for 111 M seeds |

Every seeds/sec data point is bottlenecked on B+Tree `get_range()` calls. Each call walks the tree in `O(log M)` and returns an iterator; extracting `k` neighbors costs another `k` reads. The logarithmic factor compounds per-neighbor, and at papers100M scale the working set for a `from_to_edge` B+Tree (~37 GB under `GNN_MINIMAL`) exceeds physical RAM, forcing page-cache churn that doesn't amortize across seeds.

### Why Spec #3 alone is insufficient

Spec #3 reduces the number of materialized indexes from 10 to 5. It does NOT change the access-complexity of the surviving indexes. A 57 % disk reduction buys nothing in sampling throughput.

### Why in-RAM CSR (Neo4j GDS style) is wrong for MillenniumDB

Neo4j's `HugeAdjacencyList` keeps the full topology in heap. On papers100M the uint64-backed adjacency is ~24 GB of row_ptr + col_idx. Within a 30 GB RAM budget that leaves ~6 GB for everything else (sort scratch, OS, features). Incompatible with MillenniumDB's disk-first philosophy and with the target hardware class.

---

## Decision

Introduce an **optional mmap-backed CSR sidecar** file pair alongside the B+Tree projection output:

- `topology_fwd.csr` — outgoing adjacency (source-indexed).
- `topology_rev.csr` — incoming adjacency (target-indexed).

Built at `graph_project` time when the new config key `buildTopologySnapshot: true` is set (default `false`). Read at sampling time by `TopologyAccessor::Impl` via a runtime branch: if the sidecar is present + validates, use the mmap fast-path; else fall back to the existing B+Tree path. The sidecar's staleness is guarded by a SHA-256 of the source `.leaf` file stored in the CSR header; on mismatch the reader logs and falls back silently.

Fixed binary layout for v1 of the sidecar (64-byte header + ROW_PTR[N+1] + COL_IDX[M] + optional EDGE_IDS[M], all uint64 little-endian; see spec §5.1). Two independent files (not one interleaved) to enable per-direction fallback when only one sidecar is stale.

---

## Alternatives considered

### A — In-RAM heap CSR (Neo4j GDS `HugeAdjacencyList`)

Load full topology into heap at query open-time.

**Rejected:** requires full topology in RAM. Papers100M topology is ~24 GB; leaves ~6 GB for everything else on a 30 GB workstation. Violates MDB's disk-first differentiator and blocks deployment on celebi-class hardware.

### B — mmap sidecar file (this decision)

OS pages in on demand; random-access sampling hits working-set ~2 GB for fanout [15, 10].

**Accepted:** bounds RAM to kernel-managed LRU + mmap pages; fallback path preserves correctness when sidecar is missing/stale; compatible with projection-immutability invariant (Spec #2 I6).

### C — Tiered (hot CSR in RAM, cold CSR on mmap)

Load the top-degree nodes' adjacency into RAM; page-in the long tail.

**Rejected:** added complexity for marginal gain. The OS page cache already does LRU paging on the mmap'd file. A tiered layer would duplicate state and add cache-coherency bugs.

### D — Lazy per-batch materialization (DiskGNN style)

On each mini-batch, materialize only that batch's adjacency from B+Tree into a scratch CSR.

**Rejected:** the B+Tree walk cost is exactly the bottleneck we're trying to eliminate. Rebuilding from B+Tree per batch inherits `O(log N)`.

### E — Single interleaved CSR file (fwd + rev in one file with section offsets)

Merge fwd + rev into one sidecar to halve file-open syscalls.

**Rejected:** one-file design prevents independent fallback. If `rev` becomes stale (e.g., the user rebuilt only `to_from_edge.leaf` externally), we'd lose both directions. Two-file design is operationally more robust at zero measurable perf cost.

### F — Embed CSR inside B+Tree leaf pages (thesis-novel Spec #8)

Repurpose leaf pages to hold CSR-compressed adjacency lists instead of Record<3> triples.

**Deferred to Spec #8.** Spec #4-B is the cheap-first accelerator; Spec #8 is the thesis-novel architectural change. They are composable: Spec #4-B's sidecar can coexist with Spec #8's in-leaf CSR as independent layers.

---

## Consequences

### Positive

1. **100-1000× sampling speedup on ogbn-products and papers100M** (projected; measured at Gate B). Small-graph datasets (cora) already hit page-cache resident B+Tree so speedup is modest — acceptable, since those aren't the thesis target.
2. **Composes with Spec #3.** The sidecar builder gates on `project_index_mask_for(index_set)` for both FROM_TO_EDGE + TO_FROM_EDGE; works under all three presets (`ALL`, `GNN_MINIMAL`, `READONLY_TRAVERSAL`).
3. **Composes with Specs #1 and #2.** The sidecar is a post-step after the B+Tree is finalized; orthogonal to sort backend + scan strategy.
4. **No catalog version bump.** Sidecar presence is detected by filesystem probe; projections pre-dating Spec #4-B simply never have a `.csr` file and fall back to B+Tree — zero migration cost.
5. **Opt-in by default.** Users paying only for GQL querying don't incur the ~7% disk overhead or the ~10% build-time overhead.
6. **Fallback-first architecture** guarantees correctness under any failure mode (stale hash, truncated file, mmap failure, disk-full during build). Log warning, use B+Tree. Sampling output is bit-identical across both paths under fixed RNG seed — enforced by the T4.11 determinism test.
7. **SHA-256 staleness detection** is strong: forgery-resistant (unlike mtime), self-verifying, consistent with `model_checkpoint.cc`'s existing hash pattern.

### Negative

1. **Extra disk cost.** Projected ~7% on papers100M (6-8 GB of sidecars on an 81 GB projection), ~34% on ogbn-products (fixed 64-byte header + degree=0 row_ptr entries have higher relative cost at small N).
2. **Extra build-time cost.** Projected ~10% wall-clock on papers100M (one O(M) scan per direction for degrees, plus one O(M) scan for edge streaming, plus SHA-256 of source `.leaf` ~40-70 s for 37 GB files). Empirical measurement at Gate B.
3. **Open-time SHA-256 cost.** On papers100M, opening a sidecar for the first time costs ~40-70 s for the SHA-256 pass. Amortized over a training session (hours) this is acceptable; cached per-`TopologyAccessor` lifetime so follow-up operations are free.
4. **Writer invariant fragility.** Spec #2-style immutability guarantees that source `.leaf` files don't change after projection build; but external tools (benchmarks, manual edits) could silently invalidate the sidecar. Handled by SHA-256 staleness check; user-facing log warns when fallback engages.
5. **New error surface at build time.** Disk-full mid-sidecar-write must cleanup the `.tmp` without corrupting the already-finalized B+Tree. Handled by the writer's atomic-rename pattern (`.tmp` → fsync → rename → fsync-dir) and the builder's wrap of writer invocation in try/catch.

### Neutral

1. **Byte-identical sampling across paths.** The sidecar stores neighbors in the same scan order as the B+Tree (Record<3> keys sorted by `(src, dst, edge_id)`). The reader's slice preserves that order; the sampler's RNG-seeded selection chooses identical k neighbors regardless of which backend served the access.
2. **Works with both uint64 and future uint32 ObjectIds (Spec #6).** The sidecar's `id_width` byte in the header reserves exactly this extension. When Spec #6 lands, v1 sidecars continue to mean `id_width=8`; v2 sidecars can mean `id_width=4`.
3. **Does NOT interact with property indexes** (`node_key_value` etc.). Those are orthogonal to topology; GNN feature flow is unaffected.

---

## Implementation commits

- `33183def` (T4.3): CSR file format + header + parse/serialize helpers.
- `581a0f89` (T4.4): `TopologySnapshotWriter` with streaming SHA-256 + atomic rename.
- `57ab132c` (T4.4 hardening): always-on invariant checks in `append_edge`; 64 KiB SHA-256 buffer; file-size guard; doc comment fix.
- `d9e326bd` (T4.5): mmap-backed `TopologySnapshotReader` with fallback-first `open()` and stubbed SHA-256 verification.
- `520838b0` (T4.6): builder integration via new constructor flag + per-direction post-finalize writer invocation. Adds `detail::build_topology_snapshots_for_test()` hook for hermetic testing.
- `165ea3c1` (T4.7): `TopologyAccessor::Impl` fast-path with B+Tree fallback; per-direction readers mmap'd at construction; determinism invariant verified via fixed-seed sample_neighbors test. Includes link-order fix in CMakeLists (re-list `millenniumdb` after `mdb_gnn_core` for single-pass static linker symbol resolution).
- `1b81c0f9` (T4.13): wiki section "Topology snapshot" + `buildTopologySnapshot` row in global config table in `docs/MillenniumDB.wiki/GQL-Projections.md` (the only committed user-facing doc; design/plan/ADR are local-only per repo convention).
- `98909547` (T4.8): GQL config parser adds `buildTopologySnapshot` key (bool, default false, non-bool rejected via `QueryException`) and `topologySnapshotBytes` YIELD field (sum of on-disk sidecar sizes).
- `a3b10ea3` (T4.9): `gnn_build_topology_snapshot(projection)` post-hoc procedure with per-direction IndexSet gating, idempotent overwrite, YIELDs fwdBytes/revBytes/durationMillis. Registered under `#ifdef ENABLE_GNN`.
- `7350ad44` (T4.10): real SHA-256 verification in `TopologySnapshotReader::open()` (64 KiB streaming via OpenSSL EVP). Mismatch / missing-source / I/O-error collapse to `has_data()==false` with a one-line warning. Pinned stub test replaced by producer/consumer match invariant + mutated-source + missing-source fallback tests.
- `bc14e6bb` (T4.11): multi-layer `sample_khop_neighbors` determinism matrix (8 scenarios covering NATURAL / REVERSE / UNDIRECTED × {[5,5], [3,2], [2,2,1]}) with set-equality + global-pair remap (uses `unordered_set` iteration order-invariant comparison). Includes mixed-path scenario (fwd=csr, rev=bpt) and OOB-seed robustness test.
- `87dc68c9` (T4.12): `scripts/bench_topology_snapshot.sh` measures projection build overhead + sampling throughput (seeds/sec) across cora_gnn / ogbn-arxiv / ogbn-products × {on, off}; papers100M hard-refused. Smoke-tested on cora_gnn.
- `47a1bcae` (T4.12 follow-up fix): strip `ObjectId::VALUE_MASK` before using row-0 of a B+Tree Record<3> as a degrees[] / row_ptr subscript, in both builder + test helper + TopologyAccessor fast-path guards. Bug was masked (pun intended) by synthetic gtest fixtures using untagged uint64 ids; surfaced during the bench smoke-test on real cora_gnn data where node ObjectIds carry a non-zero type tag. Documented limitation: the sidecar assumes dense `[0, N)` ObjectId values after masking — true for single-label projections (thesis targets cora_gnn, ogbn-arxiv, ogbn-products, papers100M) but not for multi-label / sparse-id projections, which transparently fall back to the B+Tree path.
- `47584b1c` (post-Gate-B docs): wiki section "Matriz de tareas GNN soportadas" documents which `indexSet` preset maps to each GNN task family — NPP uses `GNN_MINIMAL`, LPP-homogeneous uses `GNN_MINIMAL`, LPP-heterogeneous uses `READONLY_TRAVERSAL`. Mirror rationale in ADR-005 "Task scope — Node and Link Property Prediction" section.
- `c63dacb3` (T4.17): `bench_topology_micro` — isolated C++ micro-benchmark that drives `TopologyAccessor::sample_khop_neighbors` directly, bypassing `gnn_offline_sample`'s PackedBatchStore + Torch tensor overhead. Revealed that the pipeline-level ~1× speedup observed in T4.12 is Amdahl's-Law-bound (topology lookup is ~4% of the procedure's wall clock); at the API layer the CSR fast-path runs 1.48-1.59× faster across cora/arxiv/products on benito_pc. The 50-500× design projection remains plausible but requires the cache-miss regime only reachable on papers100M × celebi.
- `7769da7e` (T4.18): integrate sidecar generation into main scan via the observer/piggy-back pattern. Two independent wins combined: (a) mmap-based re-reader (`topology_snapshot_from_leaf.{h,cc}`) replaces BPT-iterator walks with sequential page-cache reads, and (b) 1 MiB per-section coalescing buffers inside `TopologySnapshotWriter::append_edge` collapse the ~4.6M per-edge `pwrite` syscalls on ogbn-arxiv from 2222 ms to 14 ms (**154× speedup in the writer hot loop**). Byte-identical `.csr` output vs the post-hoc path is enforced by new golden-compare gtest `IntegratedPathProducesByteIdenticalOutputVsPostHoc`, and zero-impact on projections without the flag is enforced by `FlagOffEmitsNoSidecars`. Result: cora_gnn build overhead +48% → +9.1%; ogbn-arxiv +154% → +5.9% — Gate B's ≤15% criterion now passes on both measured datasets. Projected ogbn-products similarly in range.
- (T4.9 pending): `gnn_build_topology_snapshot` post-hoc procedure.
- (T4.10 pending): SHA-256 staleness verification (replaces T4.5 stub).

---

## Validation evidence

To be populated as T4.11-T4.15 land:

- Unit tests (T4.3 + T4.4 + T4.5): 38 tests, all green as of commit `d9e326bd`.
- Integration tests (T4.6 + T4.11): determinism golden-compare (fixed RNG seed → bit-identical SampledSubgraph across CSR and B+Tree paths).
- Benchmark (T4.12 + T4.15): `scripts/bench_topology_snapshot.sh` on cora_gnn + ogbn-arxiv + ogbn-products. Expected ≥50× sampling throughput on ogbn-products per Gate B criteria (master plan §9).

---

## Forward-looking

After Spec #4-B, the natural follow-ups are Spec #5 (delta+varint leaf encoding) and Spec #6 (bit-packed uint32 Record<N>) which shrink the B+Tree and the sidecar in parallel. Spec #8 (thesis-novel CSR-in-B+Tree-leaves hybrid) is the longer-horizon goal that merges the B+Tree and the sidecar into a single format. Spec #4-B's sidecar and Spec #8's in-leaf CSR are compatible — the sidecar can be a static aggregate of the per-page CSR slices if needed — but Spec #8's in-leaf CSR is the preferred long-term path because it unifies the storage (one format = fewer staleness vectors, simpler operational discipline).

**End of ADR.**
