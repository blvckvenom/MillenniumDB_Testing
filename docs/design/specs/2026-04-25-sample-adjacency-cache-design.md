# Spec #11 — Sample Adjacency Cache (design doc, retroactive)

**Date drafted**: 2026-04-25 (post-implementation; commit `25a663ba` is the implementation)
**Status**: implemented + verified
**ADR**: `Partial_Idea/decisions/009_sample_adjacency_cache.md`
**Composes with**: Spec #5 (DELTA_VARINT), Spec #8 (CSR_HYBRID), Spec #4-B (TopologySnapshot)
**Depends on**: Phase B EmbeddingWriter cache (commit `6521cc21`) as prior art

## 1. Motivation

Without this fix, `gnn_offline_sample` on ogbn-products (62 M edges) takes ~12 minutes because each `topology_->get_neighbors(v)` call traverses the B+Tree directory + leaf scan, paid 200-400 K times per chunk × thousands of chunks. The B+Tree path is correct but per-call cost dominates wall clock.

EmbeddingWriter Phase B already proved (commit `6521cc21`) that an in-memory adjacency cache reduces this cost from 23 min → 2 s on arxiv (700×). The same pattern applies to the offline sampler.

## 2. Goals & non-goals

### Goals

- ≥10× speedup on `gnn_offline_sample` for ogbn-products and similar (target met: 154× measured).
- Correctness: bit-identical sampled subgraphs under fixed RNG seed regardless of cache enabled/disabled.
- Opt-out flag for memory-constrained scenarios where the cache would not fit.
- Reuse the implementation across `EmbeddingWriter` and `BasicKHopSampler` and any future consumers.

### Non-goals

- Cache-everywhere by default with auto-detect (rejected — see ADR-009 alt A1).
- Compact-array layout (uint32 row indices instead of hash-map). Deferred to Spec #13.
- Hot-cold partitioning by frequency. Deferred to Spec #13.
- Concurrent (lock-free) cache for multi-thread sampling — current sampler is single-threaded per call.

## 3. Design

### 3.1 D1 — Cache lives in `TopologyAccessor`

Both `EmbeddingWriter` Phase B and `BasicKHopSampler` call `topology_->get_neighbors(v, orientation)` as their hot lookup. By housing the cache inside `TopologyAccessor`, every consumer benefits transparently — no per-consumer cache management.

Alternatives considered:

- **A1** — keep cache private to `EmbeddingWriter`. Rejected: code duplication for `BasicKHopSampler`.
- **A2** — separate `CachedTopologyAccessor` subclass. Rejected: harder to thread through dispatch — sampler would need to know which subclass to construct.

### 3.2 D2 — Two `unordered_map<uint64_t, std::vector<AdjEntry>>` instances

One per direction (forward + reverse), populated by a single full-scan of the corresponding edge B+Tree (`from_to_edge` for fwd, `to_from_edge` for rev). Each `AdjEntry` is `{ObjectId neighbor; ObjectId edge_id;}` = 16 bytes.

Alternatives considered:

- **A1** — single map combining both directions tagged with a flag. Rejected: doubles per-entry cost without simplification benefit.
- **A2** — `std::map` (sorted). Rejected: O(log N) lookup vs O(1) hash, no benefit for the sample workload that doesn't need ordering.

### 3.3 D3 — Lazy / eager build via explicit API

```cpp
class TopologyAccessor {
public:
    void enable_adjacency_cache(bool enable);
    void prebuild_adjacency_cache(EdgeOrientation direction);
    bool is_adjacency_cache_built() const;
    
    Neighbors get_out_neighbors(ObjectId v) {
        if (cache_enabled_ && fwd_cache_built_) {
            return /* O(1) hash lookup into fwd_cache_ */;
        }
        return /* B+Tree fallback */;
    }
    // similarly get_in_neighbors
};
```

Sample / Phase B consumers call `prebuild_*` once at start. Subsequent lookups use the cache. Without the prebuild, lookups fall through to B+Tree.

### 3.4 D4 — Opt-out flag at consumer config

Both `gnn_offline_sample` and `gnn_train` (the EmbeddingWriter caller) expose `useAdjacencyCache: bool` (default true). When false, the consumer skips `prebuild_*` and operates with the B+Tree direct path.

### 3.5 D5 — Force PER_NODE strategy when cache active

`BasicKHopSampler` has three internal strategies (PER_NODE, SEEK, SWEEP — see `basic_khop_sampler.cc`). When the cache is active, the sampler forces PER_NODE because:

- Cache lookup is O(1) per node — no benefit from batched B+Tree seek/sweep.
- SWEEP / SEEK bypass `get_neighbors()` and call lower-level B+Tree primitives, defeating the cache.

This forced-strategy decision is documented inline in `basic_khop_sampler.cc:Impl::Impl` ctor.

## 4. API surface

### 4.1 GQL config (consumer-facing)

```gql
CALL gnn_offline_sample('proj', 's', [10, 5], {
    batchSize: 512,
    randomSeed: 42,
    usePredefinedSplits: true,
    orientation: 'UNDIRECTED',
    useAdjacencyCache: true        -- default true; set false for memory-constrained
})
```

### 4.2 C++ TopologyAccessor public API

```cpp
namespace mdb::gnn {
class TopologyAccessor {
public:
    // Existing API preserved
    Neighbors get_out_neighbors(ObjectId v) const;
    Neighbors get_in_neighbors(ObjectId v) const;
    Neighbors get_neighbors(ObjectId v, EdgeOrientation o) const;

    // New API (Spec #11)
    void enable_adjacency_cache(bool enable);
    void prebuild_adjacency_cache(EdgeOrientation direction);
    bool is_adjacency_cache_built() const;
    size_t get_adjacency_cache_keys(EdgeOrientation direction) const;
    size_t get_adjacency_cache_entries(EdgeOrientation direction) const;
};
}
```

## 5. Memory model

| Dataset | Edges (directed) | Cache RAM (fwd+rev) | Build time |
|---|---|---|---|
| cora_gnn | ~21 K | ~5 MB | < 100 ms |
| ogbn-arxiv | 1.17 M | ~60 MB | ~200 ms |
| ogbn-products | 62 M | **2.04 GB measured** | ~50 ms (cached B+Tree pages helped) |
| ogbn-papers100M | 1.6 B | ~80 GB projected | ~10-15 min |

The papers100M projection is the threshold at which the cache becomes infeasible on commodity 32 GB RAM. Spec #13 addresses this by partitioning into a hot RAM tier + cold disk-mmap tier.

## 6. Correctness invariants

- **I1** — Bit-identical samples: under fixed `randomSeed` config, the resulting `GraphSample` is byte-equal across cache-enabled and cache-disabled paths. Verified on cora_proj batches.dat md5.
- **I2** — Cache reflects exact B+Tree contents at build time: a `prebuild_adjacency_cache(direction)` call performs a complete scan of `from_to_edge` (or `to_from_edge`) and records every triple. No deduplication, no filtering — including for self-loops.
- **I3** — Cache is read-only post-build: subsequent inserts/deletes to the underlying B+Tree are NOT reflected in the cache. (Projections are immutable views by design — Spec #5 §I6 + Spec #8 §I6.)
- **I4** — Lookup returns identical Neighbors object semantically (same node_ids vector, same edge_ids vector, same ordering by stored sort).

## 7. Testing

8 unit tests in `src/tests/topology_accessor_adjacency_cache_test.cc`:

1. `DisabledByDefault` — cache off without explicit enable.
2. `EnabledWithoutPrebuildFallsBack` — get_neighbors works even if prebuild was skipped.
3. `PrebuildNaturalMatchesBpt` — cache fwd direction matches B+Tree byte-for-byte.
4. `PrebuildReverseMatchesBpt` — same for rev direction.
5. `PrebuildUndirectedMatchesBpt` — combined fwd+rev matches.
6. `DisableClearsCache` — `enable_adjacency_cache(false)` frees memory.
7. `KHopSamplingDeterministic` — full sampler call produces identical batches.dat.
8. `IsolatedNodeReturnsEmpty` — node with no edges returns empty span.

End-to-end byte-identical: cora_proj `[10,5]` batchSize=8 (PER_NODE on both paths) → batches.dat md5 = `391790bd...` cache=true == cache=false; frequency.dat md5 = `86d873d4...` likewise.

## 8. Empirical validation

| Test config | gnn_offline_sample wall clock | Speedup vs cache=false |
|---|---|---|
| products + cache=false | 726.48 s | 1× baseline |
| products + cache=true | 4.71 s | **154×** |
| arxiv + cache=true | 1-2 s | 8-15× (vs 12-15 s cache=false) |
| cora + cache=true | 0.05 s | 8× (vs 0.4 s cache=false) |

Memory: 2.04 GB RSS on products post-prebuild. Did not measure papers100M (too large for 32 GB commodity).

## 9. Trade-offs

### Strengths
- One-line opt-in for users (`useAdjacencyCache: true` is default).
- Massive speedup for typical workloads (arxiv, products).
- Zero correctness risk — cache is just memoization of immutable B+Tree contents.

### Limitations
- All-or-nothing per direction. If workload only touches 5 % of nodes, the cache wastefully holds the other 95 %.
- Memory linear in edges. papers100M needs ~80 GB → infeasible on 32 GB commodity.
- Opt-out flag exists but the user must know to set it. No auto-detect of available RAM (rejected per ADR-009 A1).

### Open questions
- Should the cache be persisted to disk to avoid rebuild on every server restart? Currently it's rebuilt per session. For a long-running server with multiple training calls, persistence would matter. Tracked as future work.

## 10. Future work (Spec #13)

The natural evolution is a frequency-aware two-tier cache:

- Tier 1 — RAM hash cache (existing Spec #11 layout) for top-K most-popular nodes. K bounded by available RAM.
- Tier 2 — mmap-backed compact CSR sidecar (existing Spec #4-B) for the remaining N - K nodes. Kernel page cache manages residency.

This combines Spec #11's O(1) hot-path performance with Spec #4-B's bounded RAM cost. Detailed in Spec #13 design doc.
