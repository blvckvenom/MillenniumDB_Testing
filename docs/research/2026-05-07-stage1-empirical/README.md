# Stage 1 (async batch prefetcher) — empirical validation

**Date**: 2026-05-07 / 2026-05-08
**Hardware**: celebi (Intel Core Ultra 7 265, 20 cores, 30 GB RAM, RTX 5070 Ti 16 GB, NVMe Gen4)
**Dataset**: papers100M (111M nodes, 3.3B edges, 128-dim features)
**Sample**: `papers100M_caminoD_sample`
**Projection**: `papers100M_e2e_opt`

This directory persists the raw outputs of the bench scripts that justified
the Spec C3 stage 1 default flip (`useAsyncPrefetcher: true` since commit
`072adfbb`). The bench scripts themselves live outside the repo, in
`~/Desktop/spec13_papers100m_e2e/post_pop_os/`. The CSV / summary outputs
were copied here so the empirical record is auditable post-`git push`.

## Index

| Subdirectory | Origin (Desktop) | Bench script | Purpose |
|---|---|---|---|
| [`stage0_baseline/`](./stage0_baseline/) | `05_stage0_20260507_135634/` | `05_stage0_baseline.sh` | Establish per-stage timing breakdown to decide whether Stage 1 is worth implementing |
| [`stage1_ab/`](./stage1_ab/) | `06_stage1_20260507_151218/` | `06_stage1_validation.sh` | A/B comparison with vs without `AsyncBatchPrefetcher` |

## Stage 0 baseline (Spec C3 stage 0)

`git HEAD: 264b5cf3` — commit that landed per-stage timing instrumentation.

Verbatim summary:

```
ranEpochs:         5
bestValAccuracy:   0.5912905
trainSeconds:      153.57578
  assembleSeconds: 100.84427
  forwardSeconds:  2.9997833
  backwardSeconds: 5.980981

Cache:
  l1HitRatio:      1.0
  l2HitRatio:      0.0

RATIO assemble/train: 0.6566 (65.7%)

DECISION:          STAGE 1 RECOMMENDED — strong gain expected
Expected gain:     1.3-1.7× wall-clock
```

The 65.7% ratio drove the decision to implement Stage 1 (async prefetcher).

## Stage 1 A/B comparison

`git HEAD: c688e8d3` — commit that fixed the prefetch-before-next deadlock.

Verbatim summary:

```
                   BASELINE     PREFETCHER   delta
                   ----------   ----------   -----
trainSeconds:      154.51225    96.014015    1.609×
assembleSeconds:   101.72431    45.63546     55.1% reduction
forwardSeconds:    2.965442     2.5625277
backwardSeconds:   5.8469844    5.062232

Speedup (train wall-clock): 1.609×
Decision:                    STAGE 1 CONFIRMED — proceed to Stage 2 / Stage 3
```

**Accuracy parity**: `bestValAccuracy` 0.5942 (BASELINE) vs 0.5940 (PREFETCHER)
— within stochastic noise, validates the prefetcher does not change training
semantics.

## Per-batch cost model (post-fact analysis)

Across 1.6M batches over 5 epochs, the prefetched run reduces total
assemble time from 101.7 s to 45.6 s (-56 s). With compute (forward +
backward) totalling 8.5 s, this only makes sense if the prefetcher hides
~56 s of CPU+kernel work behind compute that itself only spans 8.5 s.
The math:

- Worker thread (read_sample + edge_index + assemble_kernel): ~35 μs/batch
- Main thread `.to(device)` (edge/labels/mask) + fwd + bwd: ~33 μs/batch
- Pipelined throughput ≈ max(35, 33) = 35 μs/batch vs sequential 68 μs/batch
- Inner-loop speedup 1.94×; full-train 1.609× (validation phase still serial)

## Comparison with paper

DiskGNN SIGMOD'25 Table 6 reports 1.71-2.44× sequential→pipelined speedup
for 4 different (dataset, memory) configurations. Our 1.609× sits below
the paper's range because:

1. Validation phase is not overlapped (paper §5.3 covers full pipeline);
2. `assemble_kernel` and `model.forward` share the default CUDA stream
   (Stage 3 of our C3 plan splits them; modules 1-5 already landed,
   empirical validation pending in `07_stage3_validation.sh`).

## Reproducing

The exact bench scripts live in `~/Desktop/spec13_papers100m_e2e/post_pop_os/`
and are NOT committed (they hard-code Desktop paths). To reproduce on
another machine: copy the `*.sh` scripts into your local `Desktop/...` dir
and adjust `cd ~/MillenniumDB_Testing` if needed.

The build commit referenced (`c688e8d3`) requires:
- HEAD ≥ `c688e8d3` (deadlock fix; otherwise the prefetcher hangs when
  `train_batches > prefetch_queue_size`)
- `data/dbs/gql/papers100M/gnn_features/node_features_store.meta` present
- Sample directory `papers100M_caminoD_sample` populated

Then: `bash ~/Desktop/spec13_papers100m_e2e/post_pop_os/06_stage1_validation.sh`.
