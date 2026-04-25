# Spec #13 — Four-Level Topology Store (design doc)

**Status**: PROPOSED (not implemented). 2-3 weeks engineering scope.
**ADR**: `Partial_Idea/decisions/010_four_level_topology_store.md`
**Composes with**: Spec #4-B (TopologySnapshot), Spec #5 (DELTA_VARINT), Spec #8 (CSR_HYBRID), Spec #11 (sample adjacency cache)
**Targets**: papers100M-scale sampling on commodity 32 GB RAM hardware

## 1. Motivation

For a complete picture see `Partial_Idea/decisions/010_four_level_topology_store.md`. Briefly:

- Spec #11 RAM hash cache: 154× speedup on products, but ~80 GB RAM at papers100M → infeasible on commodity.
- Spec #4-B mmap sidecar (Camino D): cabe en commodity RAM but pays page-fault tax 30-100 μs/lookup → 30 min - 2 hs sample on papers100M.
- DiskGNN's four-level pattern is for **features** only; their sampler delegates to DGL Python and requires full graph in CPU RAM.

Spec #13 closes the papers100M-on-30-GB-RAM gap by partitioning adjacency across four tiers sized to available RAM:

| Tier | Storage | Latency | Memory cost (per node) |
|---|---|---|---|
| L1 | RAM `unordered_map<src, vec<AdjEntry>>` | ~5-20 ns | ~32 + 24 + (16 × degree) bytes |
| L2 | RAM compact CSR `vector<uint64> row_ptr_, vector<uint32> col_idx_, vector<uint32> edge_ids_` | ~50-200 ns | 8 + (8 × degree) bytes |
| L3 | mmap-backed CSR sidecar (Spec #4-B) | ~5-100 μs | 0 (disk-resident, kernel page cache) |
| L4 | B+Tree direct (existing) | ~30-100 μs | 0 (disk-resident, BufferManager page cache) |

L2 is ~5× more memory-dense than L1 because it eliminates hash map overhead and uses uint32 row indices (papers100M < 2^32 nodes).

## 2. Design

### 2.1 D1 — Frequency profiler

A `TopologyFrequencyProfiler` class scans either:

- **Cold start (no prior samples)**: degree-based proxy. Top-K nodes by degree become L1 hot, next-J by degree become L2 warm, rest L3.
- **Warm start (previous offline sample exists)**: read `node_counts.bin` from a prior `gnn_offline_sample` run. Frequency-sort and partition.

Output: a vector `tier_assignment[N]` with each node tagged L1/L2/L3.

### 2.2 D2 — Tier sizing

Auto-detect available RAM at startup via `/proc/meminfo` (reuses `src/misc/available_ram.h`). Allocate:

- 70% of available RAM → cache budget (leave 30% for OS, training workspace, features).
- Of cache budget: 25% to L1, 75% to L2.

User can override via config:

```gql
{
    useFourLevelTopologyStore: true,
    l1CacheMb: 5120,
    l2CacheMb: 14336,
    useL3MmapSidecar: true       -- requires buildTopologySnapshot:true at projection
}
```

K (L1 size) and J (L2 size) are computed from budgets + average degree.

### 2.3 D3 — Build phase

```
1. Open all required B+Tree readers (from_to_edge, to_from_edge if undirected).
2. Compute frequency via degree pass OR read prior node_counts.bin.
3. Sort nodes by frequency descending.
4. Determine K, J such that:
   sum_{i=0..K} degree(i) × 16 ≤ l1_budget
   sum_{i=K+1..K+J} degree(i) × 8 ≤ l2_budget
5. For each tier, populate the corresponding data structures:
   L1: insert into hash map
   L2: build flat row_ptr_ + col_idx_ + edge_ids_ arrays
   L3: ensure topology_snapshot file exists (if not, build it like Spec #4-B)
6. Build `tier_lookup_[N]` array mapping node_id → tier (uint8_t).
```

Build time on papers100M: ~5-15 min (dominated by degree scan + sort).

### 2.4 D4 — Runtime lookup

```cpp
Neighbors get_out_neighbors(ObjectId v) const {
    uint64_t row_idx = row_mapping_.lookup(v);
    uint8_t tier = tier_lookup_[row_idx];
    
    switch (tier) {
        case 1: {  // L1 hash
            auto it = l1_fwd_.find(v.id);
            return Neighbors{it->second};
        }
        case 2: {  // L2 compact CSR
            uint32_t l2_idx = l2_index_lookup_[row_idx];
            uint64_t start = l2_row_ptr_fwd_[l2_idx];
            uint64_t end   = l2_row_ptr_fwd_[l2_idx + 1];
            return Neighbors{
                l2_col_idx_fwd_.data() + start,
                end - start,
                l2_edge_ids_fwd_.data() + start
            };
        }
        case 3:    // L3 mmap sidecar
            return fwd_csr_sidecar_.neighbors(row_idx);
        default:   // L4 fallback (rare)
            return bpt_get_range(v, NATURAL);
    }
}
```

### 2.5 D5 — Reorder L3 mmap with MinHash

Apply Spec #5 MinHashReorderer (SEGMENTED Algorithm 1) to L3 nodes. This is the same reorder DiskGNN does for its disk-cache features. Goal: cluster nodes whose 1-hop neighborhoods overlap (across the seed set) into adjacent disk pages → cold-tier hits cascade.

The reorder requires either:

- A prior `gnn_offline_sample` to derive the bipartite graph (sample ↔ accessed-nodes), or
- A degree-degree similarity proxy if no prior samples exist.

### 2.6 D6 — L1 / L2 layout details

**L1 hash entry**:
```cpp
struct L1Entry {
    std::vector<AdjEntry> neighbors;  // 24 bytes header + 16 × degree
};
// hash overhead: ~32 bytes/key
```

**L2 compact arrays**:
```cpp
std::vector<uint64_t> l2_row_ptr_fwd_;     // (K+J+1) × 8 bytes
std::vector<uint32_t> l2_col_idx_fwd_;     // total_l2_edges × 4 bytes
std::vector<uint32_t> l2_edge_ids_fwd_;    // total_l2_edges × 4 bytes (optional)
```

uint32 row_index works because papers100M < 4.3B nodes. Future-proof check: assert `num_nodes < UINT32_MAX` at construction.

### 2.7 D7 — Threading + safety

The cache is **read-only post-build**. No concurrent insertion. Multiple sampler threads can call `get_neighbors` concurrently (returns are read-only spans).

Tier transitions are static post-build. No runtime promotion / demotion.

### 2.8 D8 — Composes with `useAdjacencyCache:false`

When user passes `useAdjacencyCache: false`:

- Spec #11 hash cache is NOT built (saves L1 RAM).
- Spec #13 four-tier is NOT built either (would defeat the opt-out).
- Falls back to Spec #4-B mmap sidecar if available, else B+Tree direct.

The two flags are coupled: either you opt into caching (Spec #11 simple OR Spec #13 four-tier) or you opt out entirely.

User config:

```gql
useFourLevelTopologyStore: true   -- (implies useAdjacencyCache: true; conflict if both false/inconsistent)
useAdjacencyCache: true            -- legacy Spec #11 (default true; superseded by useFourLevelTopologyStore when set)
useAdjacencyCache: false           -- skip both caches, use sidecar/BPT direct
```

## 3. API

### 3.1 GQL config

```gql
CALL gnn_offline_sample('proj', 's', [10, 5], {
    batchSize: 512,
    randomSeed: 42,
    usePredefinedSplits: true,
    orientation: 'UNDIRECTED',
    useFourLevelTopologyStore: true,  -- new
    l1CacheMb: 5120,                   -- new
    l2CacheMb: 14336,                  -- new
    useL3MmapSidecar: true             -- new (requires sidecar files exist)
})
```

### 3.2 C++ class

```cpp
namespace mdb::gnn {

class FourLevelTopologyStore {
public:
    struct Config {
        size_t l1_budget_mb = 0;       // 0 = auto-detect
        size_t l2_budget_mb = 0;       // 0 = auto-detect
        bool use_l3_mmap_sidecar = false;
        EdgeOrientation orientation = EdgeOrientation::UNDIRECTED;
    };

    FourLevelTopologyStore(BPlusTree<3>& fwd, BPlusTree<3>& rev,
                            const RowMapping& row_mapping,
                            std::filesystem::path projection_dir,
                            Config config);

    void build();        // populates all four tiers
    bool is_built() const;

    Neighbors get_out_neighbors(ObjectId v) const;
    Neighbors get_in_neighbors(ObjectId v) const;

    // Diagnostics
    size_t l1_node_count() const;
    size_t l2_node_count() const;
    size_t l3_node_count() const;
    size_t l4_node_count() const;
    size_t total_ram_used() const;
};

}
```

### 3.3 Integration with `TopologyAccessor`

`TopologyAccessor` gets a new optional `FourLevelTopologyStore` member. When set, `get_out_neighbors` dispatches to it:

```cpp
Neighbors TopologyAccessor::get_out_neighbors(ObjectId v) {
    if (four_level_store_) return four_level_store_->get_out_neighbors(v);
    if (cache_built_)      return /* Spec #11 hash */;
    if (fwd_csr_.has_data()) return /* Spec #4-B sidecar */;
    return /* BPT direct */;
}
```

Backwards compat: existing consumers that don't opt into Spec #13 see no change.

## 4. Empirical projections

See ADR-010 for full empirical projection. Brief summary for papers100M on 30 GB RAM:

```
L1 RAM hash    5 GB → covers ~5% nodes (top hubs)   → 70% of accesses
L2 compact CSR 14 GB → covers ~15% nodes (warm)     → 25% of accesses  
L3 mmap sidecar 27 GB on disk, ~5-15 GB resident   → 5% of accesses

Avg latency:
  0.7 × 0.05 + 0.25 × 0.2 + 0.05 × 50 = ~2.6 μs

Total for 370M sample lookups:
  370M × 2.6 μs ≈ ~16 minutes
  
vs Camino D (sidecar only): ~30-120 min
vs Spec #11 (RAM only): impossible (80 GB OOM)
```

## 5. Test plan

### Unit tests (`src/tests/four_level_topology_store_test.cc`)

1. `EmptyProjection_HandlesGracefully` — zero-edge graph builds without crash.
2. `L1OnlyConfig_AllNodesHot` — small graph fits entirely in L1.
3. `L1L2Split_FrequencySorted` — ensure top-K nodes go to L1, next-J to L2 by frequency.
4. `L1L2L3Split_FullPipeline` — bigger graph spills to all three RAM-or-mmap tiers.
5. `MatchesBpt_NaturalOrientation` — for each node, get_out_neighbors returns identical span vs B+Tree.
6. `MatchesBpt_ReverseOrientation` — same for in_neighbors.
7. `MatchesBpt_UndirectedOrientation` — same for combined.
8. `IsolatedNode_ReturnsEmpty` — zero-degree nodes handled.
9. `DegreeProxy_NoNodeCounts` — frequency profiler falls back to degree when prior samples absent.
10. `WarmStart_UsesNodeCounts` — frequency profiler uses cached node_counts.bin when available.
11. `RamBudgetRespected` — peak RSS ≤ config budgets + small overhead.
12. `RebuildIdempotent` — calling build() twice does not corrupt.

### Integration tests

- `tests/gql/test_suites/four_level_topology/run_test.sh` — full pipeline with cora + arxiv.
- Byte-identical samples on cora_gnn vs Spec #11 RAM cache (same RNG seed).

### Bench (`scripts/bench_four_level_topology.sh`)

Sweep across:
- Datasets: cora, arxiv, products
- Configs: sample with Spec #11 (cache=true), Spec #13 (this store), Camino D (sidecar only, cache=false), BPT direct
- Report: sample wall clock + RAM RSS + L1/L2/L3/L4 hit ratios

Goal: validate Spec #13 papers100M projection of ~16 min sample wall clock.

## 6. Migration path

For existing projections:

- No catalog version bump — Spec #13 is a runtime cache, not a storage layout change.
- `topology_snapshot` files are reused as L3 if they exist.
- A first run of `gnn_offline_sample` with `useFourLevelTopologyStore: true` triggers the build phase. Cache is in-memory and rebuilt per-session (no on-disk persistence in v1).

## 7. Future work

- **Persist L1 + L2 to disk** for fast restart (avoid rebuild on server restart).
- **GPU-resident L0 tier** when sampling moves to CUDA kernels.
- **Online cache promotion / demotion** based on observed access frequency.
- **Async I/O for L3 cold reads** overlapped with L1/L2 lookups.
- **Cross-projection cache sharing** when multiple projections exist on the same DB.
