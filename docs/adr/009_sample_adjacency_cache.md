# ADR-009 — Sample Adjacency Cache (Spec #11)

**Date**: 2026-04-24
**Status**: Implemented (commit `25a663ba`, retroactively documented)
**Spec**: #11
**Supersedes**: none
**Superseded by**: Spec #13 (proposed) — when implemented

---

## Context

Following Spec #5 (DELTA_VARINT) and Spec #8 (CSR_HYBRID), the on-disk projection storage achieves 81-88 % size reduction at the cost of slightly more work per neighbor lookup (offset table walk, varint decode). For GNN sampling at scale, `topology_->get_neighbors(v)` is called 200-400 K times per chunk, with 562 chunks per training run on arxiv. The cumulative B+Tree directory walks dominate sample wall clock: 11 minutes on ogbn-products, projected hours on papers100M.

The earlier `EmbeddingWriter` Phase B (commit `6521cc21`) addressed an adjacent problem by building an in-memory `unordered_map<src, vector<AdjEntry>>` adjacency cache once at Phase B start; subsequent lookups became O(1). That fix gave 700× speedup on arxiv (23 min → 2 s) and was empirically verified at products scale.

The same pattern applies to `gnn_offline_sample`. The sampler calls `BasicKHopSampler::Impl::do_sample` whose hot path is exactly the same `topology_->get_neighbors(node)` per seed × per layer × per fanout. Without a cache, products sample is 11-13 min; with a cache, < 5 s.

## Decision

Lift the adjacency cache from `EmbeddingWriter` (private) into `TopologyAccessor` itself (public) as an opt-in caching mode. Both `EmbeddingWriter` Phase B and `BasicKHopSampler` automatically benefit when the cache is enabled. Future consumers (e.g., `gnn_predict`) get the speedup for free.

The cache is a pair of `std::unordered_map<uint64_t, std::vector<AdjEntry>>` (forward + reverse direction) populated by a single full-scan of the corresponding B+Tree leaves. The cache is opt-in via:

- Construction-time call `TopologyAccessor::enable_adjacency_cache(true)`.
- Eager pre-build via `TopologyAccessor::prebuild_adjacency_cache(direction)` — usually called once per sample / Phase B start.
- Runtime introspection via `is_adjacency_cache_built()` for diagnostics.

Sampling consumers expose an `useAdjacencyCache: bool` config option (default true) for opt-out under memory-constrained scenarios where the cache would not fit in available RAM.

When the cache is built, `get_out_neighbors(v)` and `get_in_neighbors(v)` return references into the cached vectors (zero alloc per call). When disabled, the original B+Tree path is taken (O(log N + degree) per call).

## Alternatives considered

- **A1 — Cache-everywhere by default with auto-detect of available RAM.** Rejected: the cache cost grows linearly with edge count (16 bytes/edge plus hash overhead); papers100M would need ~80 GB which exceeds 32 GB commodity machines. Eager auto-detect would silently fall back to BPT path on machines that cannot fit, but the silent degradation was judged worse than an explicit opt-out flag.
- **A2 — Cache the adjacency in compact CSR (flat arrays) instead of hash map.** Smaller (~5-7× less RAM) but requires more invasive code (uint32 row-index management, separate allocation strategy). Deferred to Spec #13 as a follow-up enhancement.
- **A3 — Keep the cache private to EmbeddingWriter as in commit `6521cc21`.** Rejected: BasicKHopSampler's per-seed loop is structurally identical to EmbeddingWriter Phase B's per-seed loop. Sharing the implementation eliminates duplicate caches and aligns the fix surface.
- **A4 — Cache only the popular subset (frequency-weighted, top-K nodes).** This is what Spec #13 (FourLevelTopologyStore) proposes. Deferred because it requires a frequency profiling pass and is more complex to test than the all-or-nothing approach.

## Consequences

### Positive
- 154× speedup on ogbn-products `gnn_offline_sample` (726.48 s → 4.71 s) measured empirically.
- 700× speedup on arxiv `EmbeddingWriter` Phase B (23 min → 2 s, retroactively benefits from the same cache).
- Zero correctness change: the algorithm is identical (canonical neighbor sampling); only the lookup backend differs. Bit-identical sampled subgraphs verified across cache-on / cache-off paths under fixed RNG seed.
- Future consumers (gnn_predict, future Spec #13 evolution) inherit the win automatically.

### Negative
- Memory cost grows linearly with edges: 2-3 GB on products, projected ~80 GB on papers100M (both directions cached).
- Opt-out flag is necessary; sampling on commodity 32 GB hardware against papers100M requires `useAdjacencyCache: false` with a fallback to either the B+Tree direct path or the Spec #4-B sidecar path (slower but RAM-safe).
- The cache replicates data already on disk; total memory pressure during a training run includes cache + features + model gradients. This was ~5 GB on products with a 4 GB GPU budget; would be more on larger datasets.

### Risks
- Without a frequency-aware partitioning, the cache is all-or-nothing per direction. If a workload only touches a small subset of nodes, the cache wastefully stores the rest. Spec #13 addresses this.

## Empirical validation

Tested at three scales and verified bit-identical sample output under fixed RNG seed:

| Scale | Sample wall clock cache=false | cache=true | Speedup |
|---|---|---|---|
| cora_gnn (2.7 K nodes) | 0.4 s | 0.05 s | 8× |
| ogbn-arxiv (169 K) | 12-15 s | 1-2 s | 8-15× |
| ogbn-products (2.45 M) | 726.48 s | 4.71 s | **154×** |
| ogbn-papers100M (111 M) | (projected 2-6 hours) | (would OOM at 80 GB) | n/a — needs Spec #13 |

Memory cost RSS post-cache-build:
- ogbn-products: 2.04 GB measured (fits 31 GB RAM commodity)
- ogbn-papers100M: ~80 GB projected (does NOT fit 32 GB commodity)

Correctness: 8 unit tests in `topology_accessor_adjacency_cache_test.cc` (cache vs BPT bit-identical for NATURAL/REVERSE/UNDIRECTED, k-hop determinism, isolated nodes, disable/enable cycle). Plus end-to-end byte-identical batches.dat between cache-on and cache-off on cora_proj.

## Files

**Source code (commit `25a663ba`)**:
- `src/gnn/projection/topology_accessor.{h,cc}` — public cache API + impl
- `src/gnn/sampling/sampling_config.h` — `use_adjacency_cache` field
- `src/gnn/sampling/basic_khop_sampler.cc` — opt-in cache + force PER_NODE
- `src/query/procedure/builtin/gnn_offline_sample_procedure.{h,cc}` — `useAdjacencyCache` config plumbing
- `src/tests/topology_accessor_adjacency_cache_test.cc` — 8 unit tests
- `CMakeLists.txt` — test target registration

**Earlier prerequisite (commit `6521cc21`)**:
- `src/gnn/output/embedding_writer.{h,cc}` — initial private cache that motivated this generalization

## Cross-references

- Inspired by Phase B fix in `embedding_writer.cc` (commit `6521cc21`)
- Resolved a precondition of the splits-valid fix (`9d397335`) which uncovered the products sample as the dominant remaining bottleneck post-Phase-B
- Partial answer to DiskGNN's feature-cache hierarchy applied to topology — full answer is Spec #13
