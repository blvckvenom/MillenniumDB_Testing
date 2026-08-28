# ADR-012 — Plan F: Parallel offline sampling worker pool (SALIENT-style)

**Date**: 2026-05-11 (implementation) — 2026-05-12 (validation + fix)
**Status**: IMPLEMENTED
**Spec**: Plan F
**Supersedes**: none — composes with Spec #13 (ADR-010), Spec #11 (ADR-009), and Plan E (ADR-011)
**Estimated scope**: 1 day engineering + 1 day debug + 1 day validation
**Commits**: `c8ffd166`, `03ed201f`, `198413d8`, `2aa55fde`

---

## Context

Spec C3 stage 1 (commit `da86d6d8`, 1.609× speedup) overlapped each batch's `BatchAssembler::assemble()` + host→device transfer with the previous batch's model `forward+backward` via `AsyncBatchPrefetcher`. That was a training-loop pipeline optimization — it left the **per-batch `khop_sampler->sample()` call itself** single-threaded.

On 20-core celebi this was the dominant wall-clock cost of papers100M-scale runs:

- Empirical legacy path on 3-layer `[10,15,20]` fanout did not produce any `batches.dat` output within 3 h 37 min before manual abort (run v6 with Spec #13 ON, populate-stuck).
- The bottleneck is fundamentally embarrassingly parallel: each batch's `sample()` is independent — same RNG seed, same topology reads, independent output.

The DiskGNN paper (SIGMOD'25 §5.3) describes pipeline parallelism but delegates sampling to DGL's `DataLoader(num_workers=N)` which runs N separate Python processes with shared-memory graph store. SALIENT (MIT-IBM, MLSys 2022) uses shared-memory CPU parallelization within one process and reports **papers100M 3-layer GraphSAGE in 2.0 s/epoch on AWS x1.32xlarge** — 6-12× speedup over sequential.

## Decision

Parallelize the outer batch loop in `OfflineSamplingEngine::do_run` using a SALIENT-style shared-memory worker pool. Configuration via `numWorkers` procedure parameter / `SamplingConfig::num_workers` field:

- `0` (default) — legacy sequential path, byte-identical to pre-Plan-F output.
- `>=1` — parallel infrastructure, capped at `std::thread::hardware_concurrency()` at engine start.

### Architecture

- **`BasicKHopSampler` worker ctor** `(storage, config, shared_topology*, worker_offset)`: borrows the primary's `TopologyAccessor` by raw pointer (read-only post-build). Each worker owns private `LeapfrogGnnSampler`, `SeekBasedGnnSampler`, `std::mt19937_64`, and `node_access_counts` vectors. Worker ctors skip Phase 0 + `enable_four_level_store` + `prebuild_adjacency_cache` — those are the primary's responsibility, done once.
- **Atomic batch dispatch**: workers pull from `std::atomic<size_t> next_idx`. Each batch identified by a `WorkItem{seeds*, split, batch_id}` triple. `batch_id` is assigned monotonically pre-spawn (train first, then val, then test) so it matches the legacy ordering byte-for-byte.
- **Per-batch re-seeding**: each worker calls `sampler.reseed_for_batch(batch_id)` before `sample()`, which re-seeds the worker's RNG + LeapfrogGnn + SeekBasedGnn as `random_seed XOR batch_id`. Output is invariant to which thread picks the batch up — bit-identical across `numWorkers ∈ {1, 2, 4, 20}`.
- **Serialized writes**: `SampleStorage::write_sample` mutates shared `batch_index` + `batch_data_stream`; a single `std::mutex` around the call serializes the write phase. Sampling work itself runs concurrently.
- **Final reduce**: at end of run, each worker's `node_access_counts()` is merged into the primary via `merge_counts_from()` so the warm-start `node_counts.bin` reflects every worker's contribution.

### Thread-safety analysis

- `FourLevelTopologyStore`, `L1HashCache`, `L2CompactCsr`, `TopologySnapshotReader` (mmap) — immutable post-build → concurrent `get_neighbors` safe.
- `ProjectionStorage` members — read-only post-load.
- `BufferManager::get_page_readonly` — already serialized internally by `vp_mutex` on the shared buffer pool. Per-call iterator state is stack-local; concurrent `BPlusTree::get_range` is safe.
- `LeapfrogGnnSampler` / `SeekBasedGnnSampler` stats counters (`edges_scanned`, etc.) race — but these are telemetry only, not correctness.
- `QueryContext::_query_ctx` is **`thread_local`**. Workers spawned via `std::thread` start with `_query_ctx == nullptr`. The first BPT access null-derefs inside `BufferManager::get_page_readonly` (reads `get_query_ctx().start_version`) → SIGSEGV. **Fix**: `OfflineSamplingEngine` captures the primary's `QueryContext*` and calls `QueryContext::set_query_ctx(primary_ctx)` at the top of each worker lambda. Sampling reads only the shared buffer (`vp_map`), not the worker-indexed private pool (`pp_map` / `tmp_info`), so all workers safely share the same `QueryContext`.

## Alternatives considered

### A1 — DGL-style multi-process (each worker = separate `mdb` process)

Pros: process isolation, no shared-memory thread-safety to audit.
Cons: each process re-opens the projection (re-builds caches), expensive on papers100M. IPC for batch_id coordination + write serialization across processes is complex. Memory cost: N× `BasicKHopSampler` instances, each with its own `TopologyAccessor`.

Rejected: the cost of cache duplication kills the speedup for graphs whose topology cache is already large.

### A2 — NextDoor-style GPU sampling (EuroSys '21)

Pros: theoretically much higher throughput (GPU-resident graph + transit-parallelism).
Cons: requires the full topology in GPU memory. papers100M topology is 53 GB — does not fit in any commodity single GPU. NextDoor's paper assumes the graph fits.

Rejected: out of scope for commodity hardware target.

### A3 — Quiver-style UVA (arxiv:2305.10863)

Pros: graph stays in CPU memory, GPU samples via UVA (unified virtual addressing).
Cons: requires CUDA-aware build, ties sampling to GPU presence. Plan F should also work CPU-only for non-GPU workloads (e.g., feature-store benchmarking).

Rejected: GPU-sampling is a separate optimization axis (would compose with Plan F's CPU pool but not replace it).

### A4 — Pure pipeline parallelism (no per-batch concurrency)

Pros: already partially done (Spec C3 stage 1). Easy to extend.
Cons: pipeline depth is limited by the slowest stage. Per-batch `sample()` is by far the slowest stage on papers100M; no amount of pipelining around it helps if it isn't itself parallel.

Rejected: complementary to Plan F, not a substitute.

### A5 — Per-batch RNG via worker-local seeded state (no `reseed_for_batch`)

Pros: less per-batch overhead (no rng.seed() call).
Cons: output depends on which worker picks up which batch. Non-deterministic across runs even at constant `numWorkers`. Breaks training reproducibility.

Rejected: determinism is a hard requirement for thesis evaluation.

## Consequences

### Positive

- papers100M paper-config (`fanout=[10,15,20]`, 1.5 M seeds, BPT-direct) completed sample build in **2 min 33 s** on celebi 20-core with `numWorkers=20`. Single-thread baseline projects to ~65 min — empirical speedup **~25×**, above SALIENT's 6-12× range. Gap is consistent with BPT-direct's I/O profile: per-lookup page reads parallelize naturally across kernel page-cache concurrency on a high-IOPS NVMe.
- Determinism preserved: `uniqueNodes`, `totalBatches`, per-split counters bit-identical across `numWorkers ∈ {1, 4, 20}`. `frequency.dat` v1 sparse format iterates `unordered_map` non-deterministically — bytes differ but dict-equality check confirms (node_id, count) sets are identical.
- Composable: works orthogonally with Spec #13 (four-level topology) and Spec #11 (adjacency cache). Configurable independently per workload.

### Negative

- Worker spawning + mutex contention add ~few-second overhead on tiny samples. `[2,2]` fanout speedup is only 1.4-1.5× because each batch is too cheap to amortize the per-batch overhead.
- `LeapfrogGnnSampler` / `SeekBasedGnnSampler` telemetry counters now race (technically UB but values are informational). Not surfaced in any user-visible yield.

### Neutral

- `num_workers=0` legacy path is preserved byte-identically; pre-Plan-F workflows are unaffected.

## Bug history

Two bugs surfaced during validation, both unrelated to the worker-pool design itself:

1. **N≥2 crashed silently** on BPT-direct path (`useFourLevelTopologyStore:false`). Initially misdiagnosed as "BufferManager concurrent-read race". Root cause: `QueryContext::_query_ctx` is `thread_local`, `std::thread` workers default to `nullptr`. First BPT access null-derefs `get_query_ctx().start_version` → SIGSEGV. **Fix**: `03ed201f` — capture primary ctx pre-spawn, call `set_query_ctx(primary_ctx)` in worker lambda.

2. **`frequency.dat` bytes differ across runs** even at constant N. False alarm: the format is v1 sparse `(node_id, count)` pairs iterated from `std::unordered_map`, whose iteration order varies between processes. Dict-equality verification (`p4 == p20 == p1` all `True`, 2 575 871 pairs, `sum=6 333 072` identical) confirms semantic determinism. Not a bug.

## Files

- `src/gnn/sampling/basic_khop_sampler.{h,cc}` — worker ctor + `reseed_for_batch` + `merge_counts_from` + `get_topology`.
- `src/gnn/sampling/sampling_config.h` — `num_workers` field + thread-safety doc.
- `src/gnn/sampling/offline_sampling_engine.{h,cc}` — parallel dispatch lambda + `result.num_workers_used` + per-worker `QueryContext::set_query_ctx`.
- `src/query/procedure/builtin/gnn_offline_sample_procedure.{h,cc}` — `numWorkers` param + `numWorkersUsed` yield.

## Tests

- ctest 70/70 pass with `num_workers=0` default (byte-identical to pre-Plan-F).
- Empirical determinism: N=1 vs N=4 vs N=20 on papers100M `[2,2]` produces identical `uniqueNodes` (2 575 871), identical `totalBatches` (6044), identical frequency-dict.
- T#22 deferred: dedicated unit test for `numWorkers ∈ {1, 2, 4, 8}` on cora-scale projection as CI gate.

## References

- Empirical validation: `docs/research/2026-05-12-plan-f-validation.md`
- Paper-config E2E: `docs/research/2026-05-12-papers100M-paper-config-e2e.md`
- SALIENT paper: Cao et al., "Reducing Communication in Graph Neural Network Training", MLSys 2022.
