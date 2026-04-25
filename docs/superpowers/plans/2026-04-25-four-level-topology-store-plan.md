# Spec #13 — Four-Level Topology Store Implementation Plan

**Date**: 2026-04-25
**Design**: `docs/superpowers/specs/2026-04-25-four-level-topology-store-design.md`
**ADR**: `Partial_Idea/decisions/010_four_level_topology_store.md`
**Estimated total**: 4-5 weeks (engineering + tests + bench + docs)

## Task breakdown

### Phase 1 — Frequency profiler (Week 1, ~3 days)

**T13.1** — `TopologyFrequencyProfiler` skeleton + interface

- Create `src/gnn/projection/topology_frequency_profiler.{h,cc}`
- Class API:
  ```cpp
  class TopologyFrequencyProfiler {
    explicit TopologyFrequencyProfiler(const TopologyAccessor& topo,
                                        std::filesystem::path projection_dir);
    void compute(EdgeOrientation direction);
    const std::vector<uint64_t>& frequency() const;
    bool warm_start_used() const;  // true if loaded from prior node_counts.bin
  };
  ```
- Two profile sources:
  1. `compute_from_node_counts()` — reads prior `node_counts.bin` from a previous `gnn_offline_sample` run (DiskGNN's pattern).
  2. `compute_from_degrees()` — falls back to degree as a frequency proxy when no prior counts exist.
- Unit tests: `topology_frequency_profiler_test.cc` with 4 tests covering cold start, warm start, mixed orientation, papers100M-scale sanity check (synthetic).

**T13.2** — `node_counts.bin` standardization

- Audit existing `gnn_offline_sample` to ensure it persists `node_counts.bin` consistently.
- If not, add the persistence (small fix).
- Document the file format: 8-byte magic `NODECNT0` + uint64 num_nodes + uint64[num_nodes] counts.

**T13.3** — Tier sizing logic

- Function `compute_tier_assignment(frequency, l1_budget_bytes, l2_budget_bytes, avg_degree) -> vector<uint8_t>`.
- Greedy: sort by frequency descending, assign to L1 until budget exhausted, then L2, then L3.
- Edge case: when total cache budget exceeds graph size, all nodes go to L1.

### Phase 2 — L1 + L2 cache structures (Week 1-2, ~5 days)

**T13.4** — L1 hash cache (reuse Spec #11 logic)

- Lift the existing `unordered_map<uint64_t, vector<AdjEntry>>` from Spec #11 `TopologyAccessor` into a separate `L1HashCache` class.
- Add: only insert nodes assigned to L1 tier (per `tier_assignment[]`).
- Tests: cache parity with Spec #11 RAM cache for L1-only workloads.

**T13.5** — L2 compact CSR (new layout)

- Class `L2CompactCsr` with:
  ```cpp
  std::vector<uint64_t> row_ptr_;     // (num_l2_nodes + 1)
  std::vector<uint32_t> col_idx_;
  std::vector<uint32_t> edge_ids_;
  std::unordered_map<uint64_t, uint32_t> node_to_l2_idx_;  // node_id → l2 row index
  ```
- Build: walk B+Tree for L2-tier nodes, append to flat arrays.
- Lookup: `Neighbors get(ObjectId v)` → array indexing.
- uint32 row index assertion: verify `total_l2_edges < UINT32_MAX` at build time.
- Tests: 6 tests covering build correctness, lookup parity vs B+Tree, isolated nodes, large degree.

**T13.6** — Tier dispatch in `FourLevelTopologyStore::get_out_neighbors`

- Switch on `tier_lookup_[row_idx]`:
  - L1 → `l1_cache_.get(v)`
  - L2 → `l2_compact_.get(v)`
  - L3 → `l3_sidecar_.get(row_idx)` (if available)
  - L4 → BPT direct fallback
- Tests: 8 tests covering all 4 tier paths + edge cases.

### Phase 3 — Coordinator + integration (Week 2, ~3 days)

**T13.7** — `FourLevelTopologyStore` coordinator

- Create `src/gnn/projection/four_level_topology_store.{h,cc}`.
- Constructor takes `BPlusTree<3>& fwd, rev`, `RowMapping`, `projection_dir`, `Config`.
- `build()` orchestrates: profiler → tier sizing → L1 build → L2 build → L3 link → L4 fallback config.
- Diagnostics: `l1_node_count()`, `l2_node_count()`, `l3_node_count()`, `l4_node_count()`, `total_ram_used()`.

**T13.8** — `TopologyAccessor` integration

- Add `std::unique_ptr<FourLevelTopologyStore> four_level_store_` member.
- Add `enable_four_level_store(Config)` method.
- Modify `get_out_neighbors / get_in_neighbors` to dispatch to four-level when set.
- Backwards compat: when `four_level_store_` is null, existing Spec #11 / #4-B / BPT logic intact.

**T13.9** — `gnn_offline_sample` config plumbing

- Add `useFourLevelTopologyStore: bool` (default false; opt-in initially, can flip default later after Gate E).
- Add `l1CacheMb: int`, `l2CacheMb: int` (0 = auto-detect via available_ram).
- Add `useL3MmapSidecar: bool`.
- Validation: if `useFourLevelTopologyStore: true` AND `useAdjacencyCache: false` → error (incompatible).

### Phase 4 — Build optimization (Week 2-3, ~3 days)

**T13.10** — Fast L2 build via Spec #4-B sidecar reuse

- If a Spec #4-B `topology_fwd.csr` sidecar exists, parse it directly to populate L1 + L2 instead of re-walking B+Tree.
- ~5× faster build for arxiv/products.

**T13.11** — Frequency-aware L3 reorder

- Apply `MinHashReorderer::SEGMENTED` (Spec #5) to L3-tier nodes.
- Goal: cluster cold nodes by sample-set similarity. Hot pages cascade.
- Skip on cold start (no `node_counts.bin`); only run on warm start.

### Phase 5 — Testing (Week 3, ~5 days)

**T13.12** — Unit tests (12 tests per design §5)

**T13.13** — Integration tests

- `tests/gql/test_suites/four_level_topology/run_test.sh`
- Cora + arxiv full pipeline with `useFourLevelTopologyStore: true`.
- Byte-identical samples vs Spec #11 cache under fixed RNG seed.

**T13.14** — Fuzz testing

- Random graph topology + random tier budgets → verify no crash, correct lookups.
- Reuse existing fuzz infrastructure from `bpt_leaf_csr_fuzz_test.cc` patterns.

### Phase 6 — Benchmark (Week 4, ~3 days)

**T13.15** — `scripts/bench_four_level_topology.sh`

- Sweep: cora, arxiv, products × {Spec #11, Spec #13, Camino D, BPT direct}.
- Report: sample wall clock, RAM RSS, L1/L2/L3/L4 hit ratios.

**T13.16** — papers100M validation

- Document the projected 16-min wall clock vs measured.
- If implementation matches projection: thesis-defensible at scale.
- If 2-3× slower: investigate L3 cold-fault dominance, possibly enable L3 reorder more aggressively.

### Phase 7 — Gate E sign-off (Week 4-5, ~2 days)

**T13.17** — Gate E report

- `docs/research/2026-04-25-gate-e-report.md`.
- 5 criteria from design §5:
  1. Bit-identical samples vs Spec #11 ✓
  2. Memory bounded by config ✓
  3. Speed ≤ 2× Spec #11 wall-clock at products scale ✓
  4. papers100M sample within projected window ✓
  5. No regression on existing tests ✓
  6. Determinism across runs ✓

**T13.18** — Documentation update

- Update CLAUDE.md with Spec #13 section + config keys.
- Update `docs/research/2026-04-25-e2e-validation-report.md` with FourLevelTopologyStore numbers.
- Update `docs/research/2026-04-24-defense-qa.md` with Spec #13 framing for thesis defense.

## Risk register

1. **uint32 row_index overflow at >2^32 nodes**: assertion at build, fail loudly.
2. **L3 cold-fault thrashing if frequency proxy is bad**: mitigated by MinHash reorder + degree-based fallback for cold start.
3. **Build time exceeds budget**: optimize via Spec #4-B sidecar reuse (T13.10).
4. **Memory accounting drift**: rigorous RSS validation in T13.12.

## Success metrics

- ✅ papers100M `gnn_offline_sample` completes in ≤30 min on 30 GB commodity RAM hardware.
- ✅ products sample remains ≤ 5 s (no perf regression vs Spec #11).
- ✅ All existing 125+ ctest pass.
- ✅ New 22 tests added (12 unit + 10 integration/fuzz).
- ✅ Gate E report signed.

## Dependencies

- Spec #4-B TopologySnapshot (already implemented)
- Spec #5 MinHashReorderer (already implemented; reused for L3 reorder)
- Spec #11 sample adjacency cache (already implemented; lifted into L1)
- `available_ram.h` (already exists for sort buffer sizing; reused for tier auto-detect)
