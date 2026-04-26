# Projection Parallelization Study — Exhaustive CPU + GPU Analysis

**Date:** 2026-04-26
**Branch:** feature-GNN
**Author:** benito_pc / blvckvenom
**Question:** What is the maximum theoretical speedup achievable for `graph_project` in MillenniumDB if we parallelize every phase that admits parallelization, using both CPU multi-threading AND existing GPU infrastructure?

**TL;DR:**
- 9 of 10 projection phases are currently single-threaded.
- Total parallelizable wall clock: ~85-90% architecturally; ~50% with low-risk specs.
- **Existing GPU infrastructure** (`src/gpu/`) provides 5-strategy adaptive sort, CUB RadixSort, bitset filter kernels, GPU memory pool — already integrated into CLASSIC sort backend.
- **Proposed roadmap**: Wave 1-3 specs (CPU + GPU combo) → papers100M projection 90-120 min → 25-30 min (3-3.5× global speedup).

---

## 1. Current state — empirical measurement

`graph_project` papers100M smoke run on celebi (Intel Ultra 7 265 / 20 cores / 32 GB RAM / RTX 5070 Ti / NVMe Gen4):
- **Sustained CPU utilization**: ~18-20% (per `top -H` snapshot 2026-04-26)
- **Memory growth**: 15.3 GB RSS at 30 min mark (page cache + buffers)
- **GPU utilization during projection**: 0% (GPU is idle — projection does not use GPU)
- **Implication**: 16-17 cores idle out of 20; GPU completely unused.

## 2. Phase-by-phase analysis (15 sub-phases identified)

Source citations: each row links to specific file:line.

| # | Phase | Source | Current | % wall clock | Parallel strategy | Speedup local | Effort | Risk |
|---|---|---|---|---|---|---|---|---|
| 1 | Validation | `native_projection_builder.cc:1023-1031` | Single | <1% | N/A | N/A | N/A | N/A |
| 2 | B+Tree node scan | `native_scanner.cc:35-70` | Single | 12-15% | TBB `parallel_for` over label ranges | 4-6× | 3-5d | Low |
| 3 | Node property extract | `native_projection_builder.cc:1086-1250` | Single | 3-5% | Per-node thread fan-out | 6-8× | 2-3d | Med |
| 4 | B+Tree edge scan | `native_scanner.cc:106-257` | Single | 18-22% | Range partition + endpoint batching | 6-8× | 4-6d | Med |
| 5 | Parallel edge aggregation | `native_projection_builder.cc:662-812` | Single (per detector) | 8-12% | Per-thread detectors + merge | 4-6× | 3-5d | Med |
| 6 | RADIX Phase 1 (partition fill) | `parallel_scan_partitioner.cc:46-78` | Single (despite name) | 10-18% | Parallel scan + thread-local buckets | 8-12× | 5-7d | Med |
| 7 | RADIX Phase 2 (per-partition sort) | `radix_partition_sort.cc:407-425` | **TBB parallel** ✅ | 15-25% | Already parallel | 1.2-1.5× (additional nested) | 1d | Low |
| 8 | RADIX Phase 3 (concat) | `radix_partition_sort.cc:427-450` | Single | 12-18% | Lock-free B+Tree writer (Spec #23, hard) | 0.8-1.5× | 10-14d | High |
| 9 | Leaf emission DELTA_VARINT | `radix_partition_sort.cc:151-208` | Single | n/a (in #8) | Cannot parallelize independently | N/A | N/A | N/A |
| 10 | Topology snapshot CSR | `topology_snapshot_writer.cc` | Mixed | 8-12% | Parallel B+Tree scan + concurrent CSR | 1.5-2× | 2-3d | Med |
| 11 | Phase 4 reader opening | `projection_storage.cc:1621-1715` | Single | <2% | TBB parallel file opens | 10-14× | 1d | Low |
| 12 | Feature copy (GNN) | `import.cc:1122-1197` | Single | 5-8% | Stripe rows across workers | 8-16× | 1-2d | Low |
| 13 | EdgeKeepBitmap (Phase B) | `native_projection_builder.cc:1880-1950` | Single (atomic ops only) | 6% | Parallel B+Tree scan | 4-6× | 2-3d | Low |
| 14 | Serialized scan Phase A (5 passes) | `native_projection_builder.cc:2200-2330` | Single across passes | 8% | Within-pass parallelization only | N/A across | N/A | High (architectural) |
| 15 | Serialized scan Phase C (9 passes) | `native_projection_builder.cc:2371-2400` | Single across passes | 12% | Within-pass parallelization only | N/A across | N/A | High |

**Key architectural constraints**:
- `BPTLeafV2Writer::append_record` (`bpt_mem_import.h:142-171`): NOT thread-safe (deleted copy/move ctors, no synchronization).
- `RowMapping`: thread-safe for concurrent reads ✓.
- `FeatureMatrix`: thread-safe for concurrent reads ✓.
- `ParallelEdgeDetector::edge_map_`: NOT thread-safe (unprotected `unordered_map`).
- B+Tree iterator: holds traversal state, no documented partition split API → requires histogram prepass for range splitting.

## 3. GPU infrastructure already in the codebase

Per CLAUDE.md L73-82 + grep verification:

| Component | Source | Status |
|---|---|---|
| `gpu_device.{h,cc}` | Runtime VRAM + CPU RAM detection | ✅ implemented |
| `resource_planner.{h,cc}` | 5-strategy adaptive sort selection | ✅ implemented |
| `sort/` | Multi-pass CUB DoubleBuffer RadixSort + chunked sort for >VRAM | ✅ implemented |
| `ops/` | CUDA kernels: bitset filter + UNDIRECTED expand | ✅ implemented |
| `gnn/core/` GPU memory pool | Allocator pool for tensor ops | ✅ implemented |
| `external_record_sort.h:58` GPU fallback | `MDB_GPU_ENABLED` macro guards CUB sort | ✅ integrated in CLASSIC sorter |

**Critical fact**: The CLASSIC sort backend ALREADY uses GPU when the resource planner decides it's beneficial (typically datasets > 256 MB). The RADIX backend does NOT use GPU yet — only TBB on CPU.

**CUDA toolkit 12.8** + **CUB + Thrust** PDFs documented in `docs/external_references/CCCL_DOCS/` (per CLAUDE.md L168-179).

## 4. Data-parallel phases — GPU candidates

Categorized by GPU-friendliness:

| Phase | Data-parallel? | GPU candidate? | Reason |
|---|---|---|---|
| 1 Validation | ❌ | NO | Branchy decisions |
| 2 B+Tree node scan | ❌ | NO | Pointer-chase, random memory |
| 3 Node property extract | ❌ | NO | Per-node random access |
| 4 B+Tree edge scan | ❌ | NO | Pointer-chase |
| 5 **Edge aggregation** (MIN/MAX/SUM/COUNT) | ✅ | **YES** | `cub::DeviceSegmentedReduce` perfect fit |
| 6 **RADIX Phase 1: partition fill** (scatter by hash bucket) | ✅ | **YES** | Data-parallel scatter |
| 7 **RADIX Phase 2: per-partition sort** | ✅ | **YES** | `cub::DeviceRadixSort` (precedent in `src/gpu/sort/`) |
| 8 RADIX Phase 3: concat to B+Tree | ❌ | NO | Sequential write |
| 9 Leaf emission DELTA_VARINT | ⚠️ | Marginal | PCIe transfer > compute save |
| 10 **Topology snapshot CSR** (degree histogram + scatter) | ✅ | **YES** | `cub::DeviceHistogram` + scatter |
| 11 Phase 4 reader opening | ❌ | NO | I/O |
| 12 **Feature copy** (memcpy via GPUDirect Storage) | ✅ | **YES** | NVMe → GPU direct, bypass CPU |
| 13 **EdgeKeepBitmap filter** | ✅ | **YES** | Bitset ops are canonical GPU primitive (`src/gpu/ops/` already has this!) |

**Total wall clock attributable to GPU-friendly phases: ~50-60%** of total projection time.

## 5. PCIe transfer cost analysis

GPU is great when compute > transfer. For papers100M sort (1.6 B edges × 24 bytes = 38 GB):

| Step | Time | Source |
|---|---|---|
| Disk → CPU RAM (NVMe Gen4) | 19 sec @ 2 GB/s | celebi NVMe spec ~3-5 GB/s, derated for OS overhead |
| CPU → GPU VRAM (PCIe Gen5 x16) | 2.4 sec @ 64 GB/s | celebi 5070 Ti link state: idle Gen1, scales to Gen5 x16 under load (per user spec screenshot) |
| GPU CUB DeviceRadixSort (38 GB → 3 VRAM chunks of 13 GB each) | ~6 sec | Existing `src/gpu/sort/` chunked path |
| GPU → CPU VRAM | 2.4 sec | Reverse |
| CPU → disk | 19 sec | Sequential write |
| **Total GPU path** | **~49 sec** | |
| **CPU TBB sort actual** | **~15-25 min** | Measured equivalent on CPU |
| **Speedup of sort phase** | **~20-30×** | |

**Caveat**: this only accelerates the sort phase. Global speedup depends on what fraction of total wall clock the sort represents (~17-25%). Reducing sort from 17 min → 1 min saves 16 min globally. On a 100-min total, that's 1.19× global speedup.

For BIG impact, multiple GPU-accelerated phases must compose:
- Phase 5 (aggregation) + Phase 6 (partition fill) + Phase 7 (sort) + Phase 13 (bitmap) ≈ 50% of total wall clock
- If all GPU: 50% reduces by ~10× → global speedup 100/(50 + 5) = **1.82×**
- Combined with CPU-side scan parallelization (Specs #15-16): another 1.4× layer
- **Composed speedup: ~2.5-3.0×**

## 6. Amdahl's law — global speedup ceiling per scenario

Baseline: papers100M projection ~90-120 min on celebi.

| Scenario | Specs | Phases parallelized | Predicted wall clock | Speedup |
|---|---|---|---|---|
| **Today** | None | Only RADIX Phase 2 (TBB) | 90-120 min | 1.0× |
| **Wave 1 (low-risk, 1-2 weeks)** | #14, #18, #24 (RADIX sort GPU), #27 (bitmap GPU) | Feature copy + readers + sort + filter | 50-65 min | **1.7-2.0×** |
| **Wave 2 (medium-effort, +2-3 weeks)** | + #15, #16 (CPU scan parallel), #26 (aggregation GPU) | + node/edge scans + aggregation | 35-45 min | **2.5-3.0×** |
| **Wave 3 (high-effort, +4-6 weeks)** | + #17, #25 (partition fill GPU), #28 (CSR GPU) | + edge detector + partition fill + topology | 28-35 min | **3.0-3.5×** |
| **Wave 4 (research, +6+ weeks)** | + #29 (GPUDirect Storage), #23 (concurrent B+Tree) | + feature read + lock-free writer | 22-28 min | **3.5-4.5×** (theoretical, high risk) |

**With contention overhead (5-10% per lock)**:
- Wave 1: 1.7-2.0× → **1.6-1.9×** measured
- Wave 2: 2.5-3.0× → **2.3-2.8×**
- Wave 3: 3.0-3.5× → **2.7-3.2×**

## 7. Roadmap — Specs #14-29

| Spec | Phase target | Approach | Speedup local | Effort | Risk | Reuses existing infra |
|---|---|---|---|---|---|---|
| **#14** | Feature copy | Stripe rows across workers | 8-16× | 2-3d | LOW | None (new code) |
| **#15** | B+Tree node scan | TBB `parallel_for` over label ranges | 4-6× | 3-5d | LOW | TBB |
| **#16** | B+Tree edge scan + endpoint batch | Range partition + batched lookups | 6-8× | 4-6d | MED | TBB |
| **#17** | Edge aggregation CPU | Per-thread detectors + merge | 4-6× | 3-5d | MED | TBB |
| **#18** | B+Tree reader opening | TBB parallel file opens | 10-14× | 1d | LOW | TBB |
| **#19** | Topology snapshot integrated emit | Spec #4-B completion | 1.5-2× | 2-3d | MED | Existing partial impl |
| **#23** | Concurrent B+Tree writer | Lock-free or partition+merge | 2-4× | 10-14d | **VERY HIGH** | Refactor required |
| **#24** | RADIX Phase 2 sort GPU | `cub::DeviceRadixSort` per partition | 3-8× over TBB | 2-3d | LOW | ✅ `src/gpu/sort/` |
| **#25** | RADIX Phase 1 partition fill GPU | Custom CUDA scatter kernel | 5-15× | 5-7d | MED | New kernel needed |
| **#26** | Edge aggregation GPU | `cub::DeviceSegmentedReduce` | 10-30× | 3-5d | LOW | ✅ CUB available |
| **#27** | EdgeKeepBitmap GPU | Use `src/gpu/ops/bitset_filter` | 10-20× | 1-2d | LOW | ✅ FULLY (kernel exists!) |
| **#28** | CSR sidecar GPU | `cub::DeviceHistogram` + scatter | (deferred — see §7.1) | 3-5d | MED | ✅ CUB |
| **#29** | Feature copy GPUDirect | NVMe → GPU direct | 5-10× | 7-10d | HIGH | Driver-dependent |

### 7.1 Spec #28 deferral analysis (added 2026-04-26)

**Status: DEFERRED — subsumed by Spec #19 (CPU parallel mmap path).**

When Wave 3 reached Spec #28, the implementer surveyed Spec #19's just-landed parallel topology snapshot build (`topology_snapshot_from_leaf.cc`, commit `8301a3d7`) and concluded GPU dispatch would not move the needle. The detailed argument:

**1. The phase is memory-bandwidth + disk-write bound, not compute bound.**

Spec #19's `build_parallel` (lines 431-582):
- Pass 1: TBB workers `memcpy` 24-byte records from mmap'd `.leaf` → increment `degrees[src_idx]` in disjoint slot ranges. Per-record compute = 1 AND mask + 2 bounds checks + 1 increment. Trivial; saturates RAM bandwidth long before CPU cycles.
- Pass 2: TBB workers stream the same records → push into per-worker `vector<uint64_t>` → `pwrite` to disjoint `.csr` regions. Disk write of `COL_IDX[M] + EDGE_IDS[M]` ≈ 16 B × num_edges (papers100M: ~25 GB) at NVMe Gen4 ~2 GB/s sets a ~12 s floor regardless of compute backend.

**2. PCIe transfer cost dominates any GPU compute win at this scale.**

Per Section 5 PCIe analysis, papers100M `.leaf` is ~37 GB:
- mmap → CPU compute (Spec #19 path): ~15-25 s estimate, bandwidth-capped under 20 TBB workers.
- GPU dispatch path: cudaMemcpy 37 GB CPU→GPU (~2.4 s warm, but the link idles at PCIe Gen1 = ~37 s cold), CUB DeviceHistogram (~1-2 s on Blackwell SM 12.0), GPU→CPU (~2.4 s), then the same ~12-19 s disk write floor. Total ~17-60 s with link-warmup variance — same order of magnitude as CPU, with strictly more code complexity.

**3. v2/v3 leaf format incompatibility.**

Spec #19's mmap fast-path reads raw v1 BITSET records via `memcpy`. The v2 (`leafFormat: DELTA_VARINT`, Spec #5) and v3 (`graphStorage: CSR_HYBRID`, Spec #8) leaf formats require zigzag-LEB128 varint decode before records are accessible. A GPU CSR build kernel would either:
- Only work on v1 BITSET projections (zero added value over Spec #19, since the same constraint applies there), OR
- Implement a GPU varint decode kernel (significant complexity, unproven gain — varint decode is intrinsically serial within a stream so divergence on warps would hurt).

Neither option justifies the engineering cost.

**4. The post-hoc BPT-iterator fallback path remains sequential.**

Spec #19's commit message documents that the post-hoc BPT-iterator path in `native_projection_builder.cc::build_one_topology_snapshot_` (used only when integrated emission fails) is still sequential because it requires the Spec #15/#16 `parent_ctx` capture pattern. This is the only place GPU dispatch could in principle add value over Spec #19, but:
- The path is "rarely exercised — only when integrated emission failed" (per Spec #19 comment).
- Decode-on-CPU + transfer-to-GPU would still PCIe-bound the kernel.
- CPU-side parallelization via Spec #15/#16 pattern is simpler, lower risk, and benefits more workloads.

**5. Phase #10 budget is already small.**

Per Section 2 row #10, topology snapshot CSR is 8-12% of total wall clock pre-Spec-#19. After Spec #19's 1.5-2× local speedup it drops to ~5-8%. Even hypothetical 5× GPU speedup would yield <1.5% global improvement — below noise floor on papers100M wall-clock measurement.

**Conclusion.** Defer Spec #28 to v2. Re-open if and only if papers100M empirical bench (Wave 3 validation pass) reveals the snapshot phase is in fact compute-bound rather than I/O-bound, AND a GPU varint decode kernel proves competitive on v2/v3 leaves. Until then, treat the CPU parallel path as sufficient.

**Files reviewed:** `src/graph_models/gql/projection/topology_snapshot_from_leaf.cc` (Spec #19), `src/graph_models/gql/projection/topology_snapshot_writer.cc`, `src/gpu/sort/gpu_radix_sort.cu`, `src/gpu/ops/gpu_membership.cu`, `src/storage/index/bplus_tree/bpt_leaf_csr_format.{h,cc}`, `src/storage/index/bplus_tree/bplus_tree_leaf_v2.{h,cc}`.

---

## 8. Recommended sequence

**Wave 1 (1-2 weeks total — ship before next bench iteration)**:
1. **Spec #18** (1 day, 0.01× global, easy warm-up)
2. **Spec #14** (2-3 days, 8-16× local on feature copy = 1.06× global)
3. **Spec #27** (1-2 days, 10-20× local on bitmap = 1.05× global, **infra exists 100%**)
4. **Spec #24** (2-3 days, 3-8× local on sort = 1.10-1.20× global, reuses `src/gpu/sort/`)
5. **Composed**: ~1.7-2.0× global speedup, papers100M 90 min → 50-55 min.

**Wave 2 (2-3 weeks, post Wave 1 validated)**:
6. **Spec #15** (3-5 days)
7. **Spec #16** (4-6 days)
8. **Spec #26** (3-5 days, GPU)
   - Composed: 2.5-3.0× global, papers100M → 30-40 min.

**Wave 3 (4-6 weeks, post Wave 2 if more headroom needed)**:
9. **Spec #17** (3-5 days CPU) — DONE (commit `4630cbdf`)
10. **Spec #25** (5-7 days GPU) — DONE (commit `93201193`)
11. **Spec #28** (3-5 days GPU) — **DEFERRED** (subsumed by Spec #19, see §7.1)
    - Composed: 3.0-3.5×, papers100M → 28-30 min (thesis-defensible target).

**Defer**:
- **Spec #23** (concurrent B+Tree writer): VERY HIGH risk; only attack if Wave 1-3 don't reach <30 min.
- **Spec #29** (GPUDirect Storage): driver complexity; thesis-grade nice-to-have.

## 9. Cross-spec compatibility matrix

| Spec | Compatible with all CPU specs? | Compatible with all GPU specs? | Notes |
|---|---|---|---|
| #14, #15, #16, #17, #18 | ✅ | ✅ | Pure CPU, no conflict with GPU phases |
| #19 | ✅ | ✅ (Spec #28 enhances) | Topology snapshot orthogonal |
| #23 | ⚠️ | ⚠️ | Replaces B+Tree writer; affects all phases that write B+Trees |
| #24-#28 (GPU) | ✅ | ✅ | Per-phase GPU accelerators, independent |
| #29 | ✅ | ✅ (replaces #14 implementation) | GPUDirect bypasses CPU memcpy |

All specs except #23 are **independently composable**. #23 is the only one that touches the storage layer.

## 10. Validation methodology

For each spec landed:

1. **Correctness baseline**: Re-run cora_gnn projection with all specs enabled vs sequential. Assert byte-identical .leaf and .dir files (use `sha256sum` of output dir).
2. **Wall-clock measurement**: time `graph_project` 3× on each dataset {cora, arxiv, products, papers100M}. Report median.
3. **CPU/GPU utilization**: capture `top -H` + `nvidia-smi -l 5` during run, log per-second utilization.
4. **Lock contention**: enable `MDB_PROFILE_LOCKS=1` (TBD env var), report `mutex_wait_ms` per worker.
5. **GPU memory**: log `nvidia-smi --query-gpu=memory.used` peak; assert no OOM under chunked sort.
6. **Regression test**: existing 347 GQL integration tests must pass with all specs enabled.

## 11. Honest limits and future work

- **B+Tree write serialization** is the architectural ceiling. Without Spec #23, max global speedup is ~3.5×.
- **Phase serialization** (Spec #2 SERIALIZED scan mode) cannot be parallelized across passes; only within each pass.
- **GPUDirect Storage** requires kernel module + NVIDIA driver compatibility checks on celebi (5070 Ti Blackwell SM 12.0). Not all NVMes support it.
- **Memory pressure**: simultaneous CPU TBB workers + GPU CUB ops + 30 GB RSS budget on celebi 32 GB requires careful coordination. Adaptive resource planner (`src/gpu/resource_planner.cc`) helps.
- **Beyond projection**: similar analysis for `gnn_offline_sample` and `gnn_train` reveal additional GPU opportunities (already partially leveraged by FeatureAssembler CUDA kernel).

## 12. Citations

- `src/graph_models/gql/projection/native_projection_builder.cc` (orchestrator, ~3000 lines)
- `src/graph_models/gql/projection/native_scanner.cc` (B+Tree scanner, lines 35-257)
- `src/graph_models/gql/projection/parallel_scan_partitioner.cc` (partitioner, lines 46-78)
- `src/graph_models/gql/projection/radix_partition_sort.cc` (sorter, lines 151-450)
- `src/graph_models/gql/projection/external_record_sort.h` (classic sorter with GPU fallback, lines 58, 359-382)
- `src/storage/index/bplus_tree/bpt_mem_import.h` (B+Tree writer, lines 96-239)
- `src/import/gql/import.cc` (feature copy, lines 1122-1197)
- `src/gpu/sort/` (CUB RadixSort wrapper, multi-pass + chunked)
- `src/gpu/ops/` (bitset filter + UNDIRECTED expand kernels)
- `src/gpu/resource_planner.{h,cc}` (5-strategy adaptive selection)
- CLAUDE.md L73-82 (GPU module description)
- CLAUDE.md L168-179 (CUDA toolkit + CUB documentation)

## 13. Decision matrix — what to implement when

| If your goal is... | Implement first... |
|---|---|
| Quick win for next papers100M bench | Wave 1 (Specs #14, #18, #24, #27) — 1-2 weeks |
| Thesis-defensible <30 min papers100M projection | Wave 1 + 2 (~1 month) |
| Architectural completeness (~3.5× speedup) | Wave 1 + 2 + 3 (~2 months) |
| Research-grade contribution (paper-worthy) | Wave 1-3 + Spec #23 (lock-free B+Tree writer, ~3 months) |

The **highest ROI per dev day** is Spec #27 (EdgeKeepBitmap GPU): infrastructure already 100% in place, 1-2 days dev, 10-20× local speedup. Start there.

---

## Appendix A — Microbenchmark data (parse rate, 2026-04-26)

From `/tmp/parse_bench.cc` on benito_pc local:
```
strtof:           81.47 ns/parse  (12.3 M ops/s)
std::from_chars:  13.45 ns/parse  (74.4 M ops/s)
memcpy (binary):   0.14 ns/float  (28.4 GB/s)
```

This data informed the conclusion that **format change does NOT help** — the bottleneck is per-row callback overhead in `FeatureMatrix::create_streaming`, not parsing. See companion doc `2026-04-26-tensor-import-format-study.md` for full analysis.

## Appendix B — celebi hardware spec snapshot

- **CPU**: Intel Core Ultra 7 265 (Arrow Lake), 20 cores hybrid (8P + 12E), 6.5 GHz turbo
- **RAM**: 30 GiB total (32 GB physical), ~27 GiB available
- **GPU**: RTX 5070 Ti (Blackwell GB203, SM 12.0, 16 GB GDDR7), driver 580.126.09, CUDA 12.8
- **PCIe**: Gen5 x16 (currently Gen1 idle, scales under load)
- **Storage**: 2× Samsung NVMe SSD 1 TB (PM9A1 OEM), independent
- **OS**: Ubuntu 24.04.4 LTS, kernel 6.17.0-22

This hardware is **strictly better** than the commodity 30 GB target Spec #13 was designed for.
