# ADR-011 — Plan E: Phase 0 auto-profile to unblock Spec #13 cold-start

**Date**: 2026-05-11
**Status**: IMPLEMENTED
**Spec**: Plan E
**Supersedes**: none — augments Spec #13 (ADR-010)
**Estimated scope**: 1 day engineering + 1 day tests + iterative debug
**Commits**: `6e6776fc`, `42c6970b`, `a05f7687`, `f71b3bf0`, `c426e1b3`, `8c0ca27d`

---

## Context

Spec #13 (`FourLevelTopologyStore`) requires `<projection_dir>/node_counts.bin` to enable the L3 MinHash reorder permutation. The file is normally produced as a side-effect of `gnn_offline_sample` **at the end** of a sample build — it accumulates real per-node access counts during the sampling run and persists them on success.

This creates a chicken-and-egg dependency: the very first sample on a fresh projection cold-starts without the reorder, falling back to random mmap access over the Spec #4-B `topology_*.csr` sidecar.

On graphs whose sidecar exceeds available RAM, the cold path thrashes the kernel page cache and the sample never completes. Empirical evidence on papers100M (`topology_*.csr` = 53 GB on celebi's 30 GB host): the first `gnn_offline_sample` call ran 24 h+ without producing any `batches.dat` output before being aborted.

The DiskGNN paper (SIGMOD'25) does not hit this because they apply their four-level cache hierarchy to **features**, not topology, and assume the full topology is pinned in CPU RAM (`g.pin_memory_()` in their sampling.py). MillenniumDB's Spec #13 extends the four-level pattern to topology and inherits the cold-start gap.

## Decision

Insert a cheap random-walk profile pass between the sampler constructor and the four-level store's `build()`:

1. **`TopologyWalkProfiler::profile()`** issues N random walks of L steps each over the Spec #4-B reverse sidecar via the reader's O(1) `neighbors()` slice. Seeds are drawn from a **Vose alias-method table weighted by in-degree** — uniform seed selection on real citation graphs lands ~99% of walks on leaves (papers nobody cited), producing all-zero counts.

2. **`node_counts_io::persist()`** writes the resulting per-node counts to `<projection_dir>/node_counts.bin` using the existing `NODECNT0` format already consumed by `TopologyFrequencyProfiler::compute_from_node_counts_`.

3. **`BasicKHopSampler::Impl::run_phase0_auto_profile_if_needed_()`** runs steps 1+2 before `topology->enable_four_level_store(tcfg)`. The four-level store then activates its warm-start path on its very first build, performing the MinHash reorder of L3 with usable counts.

Default: 100 000 walks × 5 steps = 500 000 lookups vs ~4.8 billion for a full 3-layer `[10,15,20]` sample (~10 000× less work, ~10-30 s wall-clock on papers100M). Opt-out via `autoProfileOnColdStart: false` config flag + procedure parameter.

## Alternatives considered

### A1 — Two-pass sampling (run a smaller sample first, then full)

Pros: bit-accurate counts (real sampling pattern, not random walks).
Cons: pays the full sample I/O cost twice, even for the "small" warm-up pass. On papers100M scale, even a 100k-batch warm-up takes ~5-10 min before the main sample can start.

Rejected: random walks are ~10000× cheaper than even a small sample and the qualitative ranking (hot vs warm vs cold) is what tier assignment + MinHash bucketing actually consume. Per-node exactness is not required.

### A2 — Profile from BPT direct (no Spec #4-B sidecar dependency)

Pros: works on projections without `buildTopologySnapshot:true`.
Cons: BPT direct access is ~30-100 μs/lookup vs ~5 μs through mmap sidecar — 6-20× slower. For 500k lookups: ~50 s vs ~5 s. Also: profiling triggers the same page-cache thrashing we are trying to avoid.

Rejected: Spec #4-B sidecar reader is O(1) and bounded by kernel page-cache residency; the profile pass naturally tours just the hot hubs so it ends up resident anyway.

### A3 — Skip profiling, accept the cold-start hit

Pros: no new code.
Cons: papers100M cold-start does not complete on commodity 32 GB hardware — the entire Spec #13 path is locked out. The thesis evaluation cannot run paper-config replications without manual intervention.

Rejected: this is the blocking issue Plan E was opened to solve.

### A4 — Pre-compute `node_counts.bin` offline as a separate `gnn_warm_start` procedure

Pros: clean separation; user can decide when to spend the 5-30 s.
Cons: adds a new procedure to the user-facing API surface. Most users will not know to call it. Forgetting it produces silent slowness (cold-start path) rather than a clear failure mode.

Rejected: putting the profile inline in the sampler ctor is invisible to users who don't care, and the cost (~30 s) is small enough to amortize on the first build.

## Consequences

### Positive

- papers100M cold-start now completes Phase 0 in ~27 s + L1+L2 populate ~5-20 min depending on budget. Previously: 24 h+ with no progress.
- `node_counts.bin` written by Phase 0 is immediately consumed by the same build's warm-start path — no separate "first run primes the cache" step.
- Cold-start failures on graphs with sidecar > RAM are no longer silent (they previously presented as a curl timeout with zero log output).

### Negative

- Adds a ~5-30 s overhead to every fresh sample build (when `node_counts.bin` is absent). On warm-start (file present), Phase 0 is a no-op.
- Two new dependencies in `BasicKHopSampler::Impl`: `TopologyWalkProfiler` + `node_counts_io`. Both are owned by `mdb::gnn::` and contribute ~250 LOC.
- The walk profiler's degree-weighted seed selection (Vose alias) is itself non-trivial code that needed three iterations to stabilize on real data (see `Empirical iterations` below).

### Neutral

- The random walks' resulting counts are not bit-identical to a real sampling pass but the qualitative ranking is preserved within ±15% per quartile (validated by unit tests).
- Phase 0 is single-threaded and serial — but it is cheap enough (~30 s) that parallelizing it is not on the roadmap.

## Empirical iterations

Plan E went through six runs on papers100M before stabilizing. Each bug surfaced live and got a separate fix commit:

| Run / Commit | Symptom | Root cause | Fix |
|---|---|---|---|
| v1 / `6e6776fc` | 100k walks → 100k restarts (100%) | Uniform seed dist lands 99% on leaves (papers nobody cited) | Degree-weighted Vose alias (`42c6970b`) |
| v2 / `42c6970b` | 100k walks → 100k restarts STILL | Vose `leftover-large` cleanup leaves `alias_[l]=0` pointing at isolated node 0 | Build alias domain only over `degree > 0` nodes (`a05f7687`) |
| v3 / `a05f7687` | 100k walks → 100k restarts STILL | Sidecar `COL_IDX` stores `dst` as raw `ObjectId.id` (top 8 bits = type tag 0xD4); defensive `next >= n` flagged correctly but treated as corruption | Mask `next = neighbors[i] & 0x00FFFFFFFFFFFFFFULL` (`f71b3bf0`) |
| v4-debug | Diagnostic only (num_walks=10 + verbose logs) | (instrumentation, no production change) | (reverted before commit) |
| v5 / `f71b3bf0` | Phase 0 OK (380k lookups, 60k restarts) but L3 set empty (every node in L1+L2) | `compute_tier_assignment` averages frequency (~0.003/node from sparse Phase 0 counts) → per-node cost collapses to fixed-overhead only → 111M nodes "fit" in L1+L2 nominally but populate takes 30+ min | `avg_degree = l3_rev_->num_edges() / num_nodes()` from sidecar header (`c426e1b3`) |
| v6 / `c426e1b3` | Phase 0 OK, warm-start activated MinHash reorder over 90M L3-tier nodes via 16 frequency bins. Sample build then hit the **Spec #13 populate single-thread bottleneck** documented separately. | (separate issue — not Plan E) | (out of scope; Plan F unblocks the orthogonal sampling axis) |

Final commit: `8c0ca27d` (documentation consolidation).

## Files

- `src/gnn/projection/topology_walk_profiler.{h,cc}` — Vose alias + walk loop.
- `src/gnn/sampling/node_counts_io.{h,cc}` — atomic writer for `node_counts.bin`.
- `src/gnn/sampling/basic_khop_sampler.{h,cc}` — Phase 0 hook.
- `src/gnn/sampling/sampling_config.h` — 3 new flags.
- `src/gnn/sampling/offline_sampling_engine.{h,cc}` — telemetry plumbing.
- `src/query/procedure/builtin/gnn_offline_sample_procedure.{h,cc}` — yields + parse.

## Tests

`src/tests/topology_walk_profiler_test.cc` — 9 unit tests:

- Empty reader → empty result.
- Deterministic seeds (same input → same output).
- Different seeds → different output.
- Isolated-node skip via alias weights.
- Lookup bound (sum ≤ num_walks × walk_length).
- Defaults when 0 passed.
- `MostlyIsolatedGraphLandsOnHub` — 1-hub-19-leaves fixture, regression guard for v2 leftover-large bug.

## References

- Empirical validation: `docs/research/2026-05-11-plan-e-validation.md`
