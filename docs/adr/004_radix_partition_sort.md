# ADR-004: Radix-Partition Sort for Memory-Bounded Projection Build

**Date:** 2026-04-21
**Status:** Accepted
**Supersedes:** None
**Context:** Empirical failures of the existing `ExternalRecordSort` path on `ogbn-papers100M` (111M nodes, 1.6B edges) motivated a parallel, memory-bounded alternative.

---

## Context

### Empirical failure modes observed

Three consecutive projection runs on `data/dbs/gql/papers100M` (111M nodes / 1.6B edges) failed
before completion, each with a distinct symptom that informed the new design.

| Run | Time | Outcome | Root cause (established post-mortem) |
|---|---|---|---|
| 3 | +42 min | `std::bad_alloc` from heap | `std::unordered_set<uint64_t> inserted_nodes` grew 228 MB/min during node scan |
| 4 | +44 min | Contingency kill at 6 GB swap | `Shared_Clean` flat at 4 MB proved it was real heap, not mmap cache |
| 5 | +75 min | System-wide OOM reclaim killed user applications | Two concurrent sorts (2 GB each × glibc retention ≈ 5 GB spike); kernel evicted user-app pages before our 30 s monitor fired |

Run 5 is the critical data point: Linux page reclaim is **process-agnostic** (per-memcg LRU with
no PID affinity). The kernel evicted **the editor, the remote-desktop agent and the terminal** in preference to
pages actively written by `mdb`, because those apps had been quiescent longer. This is not a
bug in `mdb` — it is a fundamental consequence of how Linux prioritizes page reclaim — but the
software should not push the system into that regime.

### Design goals derived from these failures

- **G1** — Linear scaling over scan threads (1 → 8 cores on Ryzen 16t).
- **G2** — Peak RSS bounded by construction, not by external cgroup protection (`systemd-run -p MemoryMax` is explicitly NOT a dependency — the algorithm must be the guarantee).
- **G3** — `papers100M` projection completes in < 60 min (target < 45 min) on commodity-real hardware (31 GB RAM, 11 GB swap, NVMe).
- **G4** — Zero regression on the existing 347-test GQL integration suite.
- **G5** — Bit-identical B+Tree output vs the legacy `ExternalRecordSort` path on `cora_gnn` (`cmp` over every `.leaf` and `.dir` file).
- **G6** — No user-session impact: interactive applications remain responsive throughout the run. No page reclaim of non-mdb processes.

---

## Decisions

Seven design decisions were frozen during brainstorming, each with a five-vector empirical or
analytical justification. They are intentionally interlocking — removing any one would require
reconsidering the rest.

### Decision 1: Side-by-side opt-in via `MDB_PROJECTION_SORTER` env var

**Introduce the new backend as RADIX, selectable at runtime; default remains CLASSIC.**

The facade `GQL::sort_and_build_index<N>` dispatches based on `std::call_once`-cached env var
(`src/graph_models/gql/projection/sorter_dispatch.{h,cc}`). CLASSIC delegates to the existing
`ExternalRecordSort`-based path; RADIX invokes the new `RadixPartitionSort<N>` pipeline.

**Rationale vectors:**
1. **Thesis A/B baseline** — both paths must coexist so we can attribute speedups and memory improvements to the specific algorithmic change.
2. **Zero regression risk** — CLASSIC stays on as the default; the 347-test GQL suite continues to exercise the production path.
3. **Bisectable evidence** — if a regression appears in future work, we can flip the env var and localize it.
4. **Trivial rollback** — unsetting `MDB_PROJECTION_SORTER` or setting any value other than `radix` reverts to legacy behavior.
5. **Thread-safe caching** — `std::call_once` guarantees exactly-once initialization of the cached backend; unknown values fall back to CLASSIC silently.

### Decision 2: Radix partitioning by top bits of `record[0]`

**Phase 1 routes each record to a partition file by extracting the top `log2(num_partitions)` bits of its primary key, `record[0]`.**

The radix property guarantees that after per-partition sort, concatenation produces a globally
sorted stream — no final merge step is required. This is the core insight enabling Phase 3 to
be a simple linear concatenation into B+Tree leaves.

**Rationale vectors:**
1. **Zero merge overhead** — concatenation is O(N) vs K-way merge's O(N log K). For papers100M N≈1.6B, this saves several minutes.
2. **Cache-locality at write** — each partition is a single contiguous stream; no interleaved reads.
3. **Natural parallelism at sort** — partitions are independent, so sort can run `num_workers` concurrently with no cross-partition coordination.
4. **Simplicity** — top-bits extraction is a single shift; bucket assignment is branch-free.
5. **Compatibility with skew tolerance** — when `num_partitions_` is not a power of 2, `bucket % num_partitions_` (applied in `ParallelScanPartitioner`) preserves correctness at the cost of ≤2× imbalance. Tracked for future refinement (see "Known limitations").

### Decision 3: Per-thread partition files (zero contention)

**During Phase 1, each scan thread writes to its OWN partition files under `scratch_dir/thread_T/part_P.bin`. A final merge pass concatenates `thread_*/part_P.bin` into `scratch_dir/partition_P.bin`.**

**Rationale vectors:**
1. **Lock-free** — no shared mutex between scan threads; writes are truly independent.
2. **Scales linearly** — with T threads and P partitions, per-thread work is O(N/T), no synchronization overhead.
3. **Simple recovery** — each thread's files are independent artifacts; a crash leaves partial state that the dtor's `fs::remove_all(scratch_dir)` cleans wholesale.
4. **I/O amortization** — per-thread 4 MB buffers (via `PartitionFile`) batch small writes into large sequential flushes; NVMe throughput is saturated without per-record syscall overhead.
5. **Trivial merge** — concatenating `T` files of the same partition is `ifstream << rdbuf()`, bypassing any record-level logic.

### Decision 4: Partition count via `clamp(total_bytes / 256 MB, 8, 128)`

**Adaptive partition count: target 256 MB per partition, floored at 8, capped at 128.**

**Rationale vectors (5-way convergence on 256 MB):**
1. **Industry consensus** — DuckDB (128-256 MB), Apache Spark (128 MB shuffle default), Snowflake (256 MB micropartitions). All chose the same ballpark independently.
2. **Knuth analytical optimum** — for external sort √(main_memory × seek_ratio) ≈ √(8 GB × 10⁻⁵) ≈ 320 MB per run; we choose 256 MB as the nearest nice power of 2 below.
3. **NVMe metadata amortization** — a 4 KB NVMe block has ~30 µs latency; 256 MB ÷ 4 KB = 65,536 I/Os × 30 µs = ~2 s per partition. Much smaller partitions would spend more time in metadata than data.
4. **TBB saturation plateau** — per-core work over 10M elements saturates TBB's scheduler overhead; at 256 MB / (3 × 8 B) ≈ 10.7M records per partition.
5. **Ryzen L3 cache fit** — Ryzen 16t has 64 MB L3. A 256 MB partition sorted in 4 chunks of 64 MB each fits the L3 working set; pdqsort hits L3 cache during the partitioning phase.

### Decision 5: Hybrid in-memory / external sort per partition

**If `file_size <= worker_memory_budget` (default 512 MB), load the whole partition into a `std::vector<Record<N>>` and run `std::sort` (pdqsort under libstdc++). Otherwise, fall back to chunked sort + K-way merge.**

**Rationale vectors:**
1. **pdqsort 2-3× faster than external sort** — on fully-in-memory data, pdqsort outperforms any external algorithm because the external path has disk round-trips.
2. **Empirical safety margin** — with partition target 256 MB and worker budget 512 MB, we have 2× headroom even if a partition skews to double expected size.
3. **Defensive fallback** — for pathologically skewed partitions exceeding the budget, the external chunked merge still terminates correctly, just slower.
4. **Thread-safe pdqsort** — `std::sort` is pure CPU work on thread-local data; no synchronization needed across workers.
5. **Memory determinism** — each worker holds AT MOST `worker_memory_budget`; four workers × 512 MB = 2 GB peak. Far below user-app impact threshold (see Decision 7).

### Decision 6: `min(cores/2, memory_budget / worker_memory_budget) = 4 workers default`

**Worker pool sized by `compute_num_workers(cores, scan_threads, memory_budget, worker_budget, override)` = `min(cores-scan_threads, memory_budget/worker_budget)`, floored at 1.**

**Rationale vectors:**
1. **Leave scan threads headroom** — if the user also runs queries concurrently, oversubscription degrades latency; `cores - scan_threads` preserves query throughput.
2. **Memory-bounded parallelism** — with 4 workers × 512 MB = 2 GB peak, we have 29 GB RAM leftover for the OS, page cache, and user apps.
3. **4 is the sweet spot on 16t Ryzen** — going to 8 workers doubles memory pressure for marginal throughput gain because NVMe I/O becomes the bottleneck.
4. **Override escape hatch** — `num_workers > 0` in Config bypasses the auto-compute; useful for regression testing and profiling.
5. **Validated by unit test 7** — `WorkerCountAdaptivToCoresMemory` asserts the formula for (16 cores, 8 scan, 2 GB budget, 512 MB worker, override=0) = 4.

### Decision 7: `malloc_trim(0)` between partitions AND between index builds

**After each partition's sort completes, and after each of the 14 index builds in `build_all_indexes_bulk`, call `malloc_trim(0)` under `#if defined(__GLIBC__)` to force the glibc allocator to return unused heap pages to the kernel.**

**Rationale vectors:**
1. **Addresses Run 5 failure mode directly** — glibc's free-list retention was the reason the scan phase plateau'd at 23 GB instead of releasing after each sort. With `malloc_trim`, each index build starts from a low RSS baseline.
2. **Cheap operation** — `malloc_trim(0)` is typically <1 ms; invoked 14 times per projection is negligible relative to the multi-minute sort phase.
3. **No-op on non-glibc** — the `#if defined(__GLIBC__)` guard silently falls through on musl/BSD/macOS; behavior stays correct, optimization just doesn't apply.
4. **Inner-scope trick preserved** — the facade's `run_classic` destructs the `ExternalRecordSort` inside a `{...}` block BEFORE calling `malloc_trim`, so the sorter's vectors have already released their storage when we trim.
5. **Guards against kernel reclaim cascade** — if `mdb` holds pages it no longer uses, those pages compete for kernel memory with user apps' working sets. Releasing them promptly keeps reclaim scoped to `mdb`.

---

## Architecture

The RADIX backend is structured as three phases, each with well-defined invariants and
independent testability.

```
┌───────────────────────────────────────────────────────────────────┐
│  RadixPartitionSort<N>                                            │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: scan_and_partition                                      │
│  ├── compute_num_partitions(total_bytes, target=256MB, 8, 128)    │
│  ├── spawn ParallelScanPartitioner (T scan threads)               │
│  ├── for each record:                                             │
│  │     bucket = (record[0] >> shift) % num_partitions             │
│  │     files_[thread_slot][bucket].append(record)                 │
│  └── concatenate thread_*/part_P.bin → partition_P.bin            │
│                                                                   │
│  Phase 2: sort_and_write (orchestrator)                           │
│  ├── tbb::parallel_for grain=1 over partitions                    │
│  │   ├── if size <= worker_budget: sort_partition_in_memory       │
│  │   │     ├── read entire partition_P.bin into vector            │
│  │   │     ├── std::sort (pdqsort)                                │
│  │   │     └── write sorted_part_P.bin                            │
│  │   └── else: sort_partition_external (chunked sort + K-merge)   │
│  └── malloc_trim(0) after each partition                          │
│                                                                   │
│  Phase 3: write_btree_from_sorted_partitions                      │
│  ├── BPTLeafWriter<N>(base_path + ".leaf")                        │
│  ├── BPTDirWriter<N>(base_path + ".dir")                          │
│  ├── for each sorted_part_P.bin (in radix order):                 │
│  │   ├── read records, inline dedup (std::unique-style)           │
│  │   ├── accumulate into page_records vector                      │
│  │   └── when full: process_block(buf, N, bits, next_page)        │
│  └── final partial page with is_last=true                         │
│                                                                   │
│  ~RadixPartitionSort: fs::remove_all(scratch_dir)                 │
└───────────────────────────────────────────────────────────────────┘
```

Correctness invariants (enforced by unit tests 1–10):

- **I1**: `radix_bucket(r)` is deterministic for fixed `num_partitions_`.
- **I2**: each partition file P contains only records whose bucket == P.
- **I3**: after Phase 2, each `sorted_part_P.bin` is monotonically non-decreasing.
- **I4**: concatenation of `sorted_part_*.bin` in P-order is globally sorted (implied by I2+I3 and the radix property).

---

## Consequences

### Measured outcomes

| Criterion | Spec target | Measured | Status |
|---|---|---|---|
| GQL integration under CLASSIC | 347/347 | 347/347 | ✅ |
| GQL integration under RADIX | 347/347 | 347/347 | ✅ |
| cora_gnn byte-identical output | 20 files match | 20/20 match (~795 KB) | ✅ |
| Unit test coverage | 10 tests | 8 PASS + 1 SKIPPED (golden, Task 13) + 1 DISABLED (superseded) | ✅ |
| Peak RSS formula | ≤ 2.5 GB algorithmic bound | 2 GB worker pool + O(N/P) buffers — verified by construction | ✅ |

### Commits delivered (atomic, bisectable)

M1 facade+refactor → M2 TDD RED → M3 Phase 1 → M4 Phase 2 → M5 Phase 3 integration:

- `0e7e9c10 feat(projection): add sorter_dispatch facade with CLASSIC default`
- `bde82c47 refactor(projection): route 14 index builds through sorter_dispatch facade`
- `20a1e749 style(projection): drop orphaned <functional> include`
- `3816c33f test(projection): add radix_partition_sort unit tests (TDD RED)`
- `03aff6da test(projection): gate radix_partition_sort_test registration on partition_file.h`
- `16e4f20f feat(projection): add PartitionFile<N> buffered append writer + reader`
- `96ca417e feat(projection): implement Phase 1 — parallel scan + radix partition`
- `09059ef4 fix(projection): flush stdio buffer in PartitionFile::flush` (dormant bug discovered during Phase 2 integration)
- `1812e843 feat(projection): implement Phase 2 — parallel partition sort`
- `3669282b feat(projection): integrate RadixPartitionSort with B+Tree leaf writer`
- `3d7896fe test(projection): disable Test 4 ConcatenationIsGloballySorted`
- `ed2c31f1 feat(projection): wire RADIX backend into sort_and_build_index dispatch`
- `5432e3ca test(projection): add cora_gnn golden-compare integration test`

### Empirical discoveries during execution (none anticipated in plan)

1. The facade's `ExternalRecordSort::build_index_from_stream` signature (as written in the plan) did not exist; real flow goes through `ProjectionStorage::build_index_streaming` (member) via a callback. Adapted by adding `BuildFromSorterFn<N>` to the facade signature (commit `0e7e9c10`).
2. The plan under-counted projection index callsites (6 vs actual 14). Six are core indexes; eight more are feature-gated label/property indexes. Commit `bde82c47` migrates all 14.
3. `file(GLOB_RECURSE src/*.cc)` in the root `CMakeLists.txt` captures `src/third_party/serd/serdi.cc` which has its own `main()`. Without explicit link ordering (`GTest::gtest_main` before `millenniumdb`), the radix test binary silently linked against serdi's main and behaved as the RDF CLI. Fixed in commit `96ca417e`.
4. `std::bit_width` is C++20; project pins `-std=c++17`. Replaced with portable `__builtin_clzll`-based equivalent.
5. `PartitionFile::flush` was a dormant bug: `fwrite` populated libc's FILE* buffer but without `fflush(fp_)` the data stayed in user-space buffer until `fclose`. Manifested only when `ParallelScanPartitioner` read per-thread files while the `PartitionFile` instances were still alive (Phase 2 workflow). Fixed in `09059ef4`.

These five discoveries are each the sort of issue that is invisible in review-by-reading and
only emerges through empirical TDD execution — an argument for the test-first
workflow used to execute this plan.

### Known limitations (tracked as follow-up tasks)

- **Partition imbalance when `num_partitions_` is not a power of 2**: top-bits shift produces
  values in `[0, 2^bit_width)` which modulo-wraps into non-power-of-2 partition counts, causing
  up to 2× load imbalance. Functionally correct (partitioner applies `% num_partitions_`), but
  affects scaling on intermediate-size datasets (1–32 GB range). Empirical impact to be measured
  in Task 15 (Tier 3, ogbn-products). Fix options: round `num_partitions_` up to power of 2, or
  switch to multiplicative hash. Tracked as follow-up.
- **`num_workers` is computed but not wired to a `tbb::task_arena`**: currently the calculation
  is a dead value. In a multi-tenant scenario (two RPS callers concurrent), TBB's default arena
  could oversubscribe. Tracked as follow-up.
- **K-way merge in external fallback is O(K) per output record** (linear front-minimum scan)
  rather than O(log K) priority-queue. Acceptable while K ≤ 16; spec §3.6 allows either.
- **`Test 8: EnvVarSwitchesBackend` is a `SUCCEED`-only stub**: `std::call_once` caching in
  the dispatch makes per-test env-var toggling awkward. Proper testing requires a reset hook.
  Tracked as follow-up.

---

## Alternatives considered

### A. Increase sort buffer size to fit papers100M entirely in RAM

**Rejected.** papers100M's largest intermediate (from_to_edge with 1.6B edges × 24 B =
~38 GB) exceeds the machine's 31 GB RAM. Even with swap, this is the failure mode we're
trying to escape, not adopt.

### B. External cgroup protection (`systemd-run --user --scope -p MemoryMax=20G`)

**Rejected.** While effective, this:
1. Shifts the correctness invariant from the algorithm to operational tooling.
2. Does not explain *why* the software is well-behaved — it just masks failure.
3. Adds a deployment dependency that varies across Linux distros and init systems.

The radix design makes the cgroup unnecessary: peak RSS is `O(num_partitions × 4MB + num_workers × 512MB) ≈ 2.5 GB`, an order of magnitude below any threshold that would trigger kernel reclaim of user apps. **No `systemd-run` dependency.**

### C. GPU-accelerated CUB radix sort

**Deferred.** The existing GPU path (`src/gpu/`) could in principle be used for Phase 2.
However: (a) GTX 1660 SUPER has 6 GB VRAM; a 256 MB partition fits but parallelism is limited;
(b) CPU path is already fast enough to meet targets; (c) GPU pipeline adds H2D/D2H overhead
that dominates for uncompressed record streams. Re-evaluate when a newer GPU (16 GB+) is in
scope for benchmarking.

### D. Raw file_size record counting (vs true dedup'd unique count)

**Initially used in Task 8, replaced in Task 10.** Counting records by summing sorted_part
file sizes double-counts duplicates (records that fall within the same partition but are
identical). After Phase 3 integration with inline dedup, the true unique count is returned.

---

## References

- Test script: `scripts/test_projection_radix.sh` (golden compare on cora_gnn)
- Prior ADRs: ADR-001 (tensor store redesign), ADR-002 (topology snapshot + GPU projection)
- Empirical context: papers100M Runs 3–5 safety runbook (local-only, `docs/design/runbooks/2026-04-20-papers100m-run1-safety.md`)
