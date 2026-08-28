# ADR-010 — Four-Level Topology Store (Spec #13)

**Date**: 2026-04-25
**Status**: PROPOSED (not implemented)
**Spec**: #13
**Supersedes**: none — augments Spec #11 + Spec #4-B
**Estimated scope**: 2-3 weeks engineering + 1 week tests + 1 week benchmarks

---

## Context

Spec #11 (commit `25a663ba`) implemented a flat in-memory adjacency cache for `gnn_offline_sample` that achieved 154× speedup on ogbn-products. Spec #4-B (commit `33183def` and following) implemented an mmap-backed CSR sidecar that achieves O(1) neighbor lookup with kernel-managed memory residency.

Both work well at their target scales:

- **Spec #11 RAM hash cache** is ~2 GB on products, fits commodity 32 GB RAM. But on papers100M (1.6 B edges) it projects to ~80 GB — exceeds 32 GB commodity. Not viable.
- **Spec #4-B mmap sidecar** is bounded by kernel page cache (~5-15 GB resident on demand). Cabe en commodity but its O(1) cost includes kernel page-fault latency for cold pages, ~30-100 μs vs ~0.05 μs for RAM hash. On papers100M sample, this works out to 30 min - 2 hours.

Neither path optimally uses the full available RAM. Spec #11 is wasteful on small graphs (ALL adjacency cached even if 5% of nodes are sampled) and impossible on large. Spec #4-B is conservative on RAM and pays page-fault tax.

DiskGNN's contribution to **features** (not topology) is exactly this kind of hybrid: a four-level cache hierarchy (GPU + CPU + disk-cache + packed chunks) where each tier is sized to its respective resource budget and content is partitioned by frequency.

The Spec #13 proposal applies DiskGNN's **four-level pattern** to **topology / adjacency** — a domain DiskGNN's paper explicitly does not address (they delegate sampling to DGL with full-graph in CPU RAM).

## Decision

Implement a `FourLevelTopologyStore` that partitions adjacency between four tiers based on frequency profiling. Replaces Spec #11's monolithic RAM hash cache when active. Composes with Spec #4-B sidecar files as the disk tier.

The four tiers:

1. **L1 — RAM hot hash map** (`unordered_map<uint64_t, vector<AdjEntry>>`): top-K most-popular nodes' adjacency, full hash-map lookup O(1) + ~5-20 ns latency.
2. **L2 — RAM compact CSR** (`vector<uint32_t> row_ptr_, col_idx_, edge_ids_`): warm nodes' adjacency in flat arrays. O(1) array indexing + ~50-200 ns latency. ~5× more memory-dense than L1 due to no hash overhead.
3. **L3 — mmap-backed CSR** (Spec #4-B sidecar files): cold nodes' adjacency on disk. Kernel manages residency. O(1) virtual-address access + page-fault latency ~30-100 μs cold or ~5-20 μs hot.
4. **L4 — B+Tree direct** (existing default fallback): coldest tail. O(log N + degree) directory walk + leaf scan, ~30-100 μs per call.

Cache budgets are determined at startup from available RAM, with sane defaults proportional to total system memory.

## Alternatives considered

### A1 — Two-tier only (RAM hash + mmap sidecar)

Simpler than four tiers. Skip L2 compact CSR and L4 B+Tree. Cache decision is binary: hot in RAM hash, cold in mmap sidecar.

Pros: less code (~half the implementation effort); easier to reason about.
Cons: RAM hash overhead (~30% above raw entry size) means less coverage per GB; misses the opportunity to use compact CSR layout that would fit ~5× more nodes per GB.

Rejected: the modest implementation cost of L2 compact CSR is worth the ~5× memory efficiency for the warm tier. With 24 GB RAM available, two-tier covers ~50 M edges; four-tier covers ~250 M edges in the same budget.

### A2 — Cache the bipartite graph from sampling (DiskGNN's L3 disk cache approach for features)

Apply MinHash reorder to nodes' "set of mini-batches that need them", then store cold nodes in disk-cache reordered layout. This is exactly what DiskGNN does for features, just applied to adjacency.

Pros: provably optimal locality post-reorder.
Cons: requires the offline sample to already exist, which means Spec #13 cannot be used to accelerate the FIRST sample. Bootstrapping problem: you need samples to compute popularity, but our goal is to speed up sampling.

Rejected. Frequency proxy via degree (for first run) + sample-derived frequency (for subsequent runs) is a good-enough approximation without the bootstrapping issue.

### A3 — Defer to OS page cache (just enable Spec #4-B sidecar without hot tier)

This is what current "Camino D" is. Works at papers100M-on-30-GB-RAM, but pays full page-fault tax on every random access.

Pros: zero new code; just a config flag.
Cons: 30-100 μs per cold lookup is 1000× slower than RAM hash (0.05 μs). Bound by SSD random-read latency, not by application logic.

Spec #13 reduces this by holding hot subset in RAM where it's most needed. Camino D is the **fallback**; Spec #13 is the **optimization**.

### A4 — GPU-resident L1

DiskGNN does this for features: GPU memory holds the most popular features, accessed at HBM bandwidth.

For topology: sampling is CPU-side (BasicKHopSampler is C++/CPU code). Putting adjacency on GPU doesn't help because the sampler doesn't read from GPU. Would require porting the sampling algorithm to CUDA — a much larger Spec #14+ scope.

Rejected for v1. Future work if/when sampling moves to GPU.

## Consequences

### Positive

- **Bounded RAM, configurable**: user specifies `cache_budget_mb` (or auto-detect from /proc/meminfo). Cache fills L1 + L2 to that budget. Remaining nodes fall to L3 (mmap) or L4 (BPT direct).
- **Scales to papers100M on 30 GB commodity**: with 24 GB RAM, can hold ~30M nodes' adjacency in L1+L2 (top frequency-sorted). Remaining 81M served by L3 mmap. Mixed access pattern → ~80% hot tier hits → ~10-50 μs avg per get_neighbors.
- **Compatible with Spec #11 opt-out**: when `useFourLevelTopologyStore: false`, defaults to existing Spec #11 / #4-B / B+Tree fallback chain.
- **Faithful extension of DiskGNN pattern**: same four-level hierarchy, same MinHash reorder, but applied to adjacency (DiskGNN does features only). Paper-grade contribution.
- **Reuses existing infrastructure**: MinHashReorderer (Spec #5), TopologySnapshotReader (Spec #4-B), TopologyAccessor (Spec #11). Spec #13 adds a coordinator + L1 + L2 layers + frequency profiling.

### Negative

- **Implementation cost**: ~2-3 weeks plus ~2 weeks tests/benchmarks/docs. Comparable to Spec #5 or Spec #8 in scope.
- **Build-time overhead**: a frequency-profiling pass (or read of cached node_counts.bin from prior sampling). For papers100M this is ~5-15 min one-time cost.
- **Disk overhead**: mmap'd L3 sidecar is ~27 GB on papers100M (per Spec #4-B precedent). If the user has tight disk, this is a constraint.
- **Determinism subtlety**: cross-tier moves over time could change cache hit patterns. Need to ensure that within a single training run, the tiers are frozen post-build.

### Risks

- **L1 vs L2 boundary tuning**: how to decide which K nodes go to L1 hash vs which to L2 compact? Frequency-sorted top-K with K bounded by configured L1 budget. Simple but may not be optimal. May need to revisit if benchmarks show suboptimality.
- **Cold-page-fault bursts**: if sampling pattern doesn't match frequency profile, cold pages thrash. Mitigation: MinHash reorder of L3 to cluster similar nodes, same as DiskGNN's disk cache.

## Empirical projections (to be validated by Gate E benchmark)

### papers100M on 30 GB commodity RAM

Target wall clock for `gnn_offline_sample`:

```
L1 RAM hash    ~  5 GB → covers ~5% of nodes (highest-frequency hubs) → ~70% of accesses
L2 compact CSR ~ 14 GB → covers ~15% of nodes (warm) → ~25% of accesses
L3 mmap sidecar  ~27 GB disk → 80% of nodes (cold) → ~5% of accesses
L4 BPT direct  → not used unless catastrophic memory pressure

Estimated avg latency:
  0.70 × 0.05 μs (L1) +
  0.25 × 0.20 μs (L2) +
  0.05 × 50 μs    (L3) =
  0.035 + 0.05 + 2.5 = ~2.6 μs avg

For 370M get_neighbors per training:
  370M × 2.6 μs = ~16 minutes total sample step

Compared with Camino D (sidecar only): ~30-120 min projected
Compared with Spec #11 (RAM only): impossible (OOM)
```

If validated, Spec #13 closes the papers100M sample gap to a manageable wall clock without requiring 80 GB hardware.

### Memory utilization

```
Available RAM: 24 GB
L1 budget:      5 GB
L2 budget:     14 GB
Total cache:   19 GB
Headroom:       5 GB (training workspace + features)
```

User can configure budgets via:

```gql
useFourLevelTopologyStore: true,
l1CacheMb: 5120,        -- 5 GB hash map
l2CacheMb: 14336,       -- 14 GB compact CSR
useL3MmapSidecar: true, -- requires buildTopologySnapshot at projection time
```

Auto-detect mode (default): read available RAM at startup, allocate 70% to cache split as 25% L1 / 75% L2 of cache budget.

## Files (planned)

```
src/gnn/projection/four_level_topology_store.{h,cc}    [NEW, ~600 LOC]
src/gnn/projection/topology_frequency_profiler.{h,cc}  [NEW, ~200 LOC]
src/gnn/projection/topology_accessor.{h,cc}            [MODIFIED]
src/query/procedure/builtin/gnn_offline_sample_procedure.{h,cc}  [MODIFIED]
src/tests/four_level_topology_store_test.cc            [NEW, ~400 LOC]
src/tests/topology_frequency_profiler_test.cc         [NEW, ~150 LOC]
scripts/bench_four_level_topology.sh                  [Gate E harness]
```

## Cross-references

- ADR-006 — Spec #4-B TopologySnapshot (provides L3 implementation)
- ADR-009 — Spec #11 sample adjacency cache (provides L1 implementation pattern)
- ADR-007 — Spec #5 DELTA_VARINT (precedent for how-to-reorganize-storage compose with Spec #8)
- DiskGNN paper §4 — four-level feature store (the analog architecture for features)
- `docs/research/2026-04-25-diskgnn-source-verified.md` — verified DiskGNN architecture, what we replicate vs extend

## Validation gates (Gate E proposed)

The implementation must pass:

1. **Correctness**: bit-identical sampled subgraphs vs Spec #11 RAM-only path under fixed RNG seed (testable on cora_gnn).
2. **Memory bounded**: peak RSS ≤ configured `cache_budget_mb` + workspace overhead.
3. **Speedup**: products sample with FourLevelTopologyStore ≤ 2× the Spec #11 wall-clock (i.e., not significantly slower despite the more complex tiering).
4. **papers100M validation on celebi**: sample wall clock within projected ~10-30 min.
5. **No regression**: existing Spec #11 and Spec #4-B paths still work when this Store is disabled.
6. **Determinism**: same RNG seed + same dataset → bit-identical samples across multiple runs.

## Out of scope for v1

- GPU-resident L1 (would require porting BasicKHopSampler to CUDA).
- Async I/O overlap of L3 cold reads with L1/L2 lookups (DiskGNN paper §5.3 pipeline equivalent).
- Online cache eviction / promotion based on observed access pattern (current design is build-time partition + frozen).

These are natural Spec #14+ candidates after Spec #13 lands and benchmarks reveal next bottlenecks.
