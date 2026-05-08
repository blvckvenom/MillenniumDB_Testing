# Stage 3 (CUDA streams) — empirical validation

**Date**: 2026-05-08
**Hardware**: celebi (Intel Core Ultra 7 265, 20 cores, 30 GB RAM, RTX 5070 Ti 16 GB, NVMe Gen4)
**Dataset**: papers100M (111M nodes, 3.3B edges, 128-dim features)
**Sample**: `papers100M_caminoD_sample`
**git HEAD**: `3cb4be3e` (Stage 3 module 5 wired into TrainingLoop)
**Bench script**: `~/Desktop/spec13_papers100m_e2e/post_pop_os/07_stage3_validation.sh`

## TL;DR

**Stage 3 (separate CUDA streams for `assemble_kernel` and `model.forward+backward`) delivers only 1.014× over Stage 1 alone on this hardware.** Default for `useCudaStreams` stays **false**. Stage 1 remains the dominant gain (1.477× over sequential baseline).

## Three-way comparison (5 epochs, SAGE 2-layer hidden=256 dropout=0.3 lr=0.001 seed=42)

|                   | A — SEQUENTIAL | B — STAGE 1 | C — STAGE 1+3 |
|-------------------|----------------|-------------|---------------|
| trainSeconds      | 143.88         | 97.39       | 96.05         |
| assembleSeconds   |  92.59         | 46.80       | 45.69         |
| forwardSeconds    |   2.97         |  1.74       |  1.89         |
| backwardSeconds   |   5.83         |  5.17       |  4.96         |
| bestValAccuracy   |   0.6009       |  0.6009     |  0.5907       |

### Speedups

| Comparison | Ratio | Notes |
|---|---|---|
| **A → B (Stage 1)** | **1.477×** | async prefetcher, single CUDA stream |
| **B → C (Stage 3 add)** | **1.014×** | dual streams add only 1.4% wall-clock |
| **A → C (full pipeline)** | **1.498×** | all of C3's wall-clock benefit comes from B |

### Accuracy parity

| Comparison | Δ |
|---|---|
| \|A_val - B_val\| | 0.000072 (within stochastic noise) |
| \|B_val - C_val\| | 0.010210 (slightly above 0.01 threshold; one-shot run) |

The 0.01 accuracy delta between B and C is at the edge of the threshold I picked for "safe defaults". Across multiple seeds it should average out, but on a single deterministic seed=42 run, the C path landed slightly worse. Not enough signal to reject Stage 3 on accuracy grounds, but reinforces the speed argument: there is no compelling reason to flip the default to true on this hardware.

## Why Stage 3 was neutral

DiskGNN paper §5.3 reports 1.71-2.44× speedup from a 4-stage pipeline (feature loading + feature assembling + graph loading + model training, all overlapped via producer-consumer queues + separate CUDA streams for the GPU stages). My implementation of Stage 1 (single async prefetcher overlapping CPU+UVA assembly with model compute) already extracts most of that gain on this hardware: 1.477×.

The remaining margin in the paper comes from **separating GPU work across streams**, but our measurements show this gives only 1.4% extra. Likely reasons:

1. **`assemble_kernel` is small**: it dispatches one CUDA block per output entry (5000-30000 blocks per batch on papers100M-sized fanouts) and each block is a cooperative copy of `feature_dim` floats. The kernel runs in microseconds; even sharing the GPU with `model.forward` doesn't create much contention.
2. **PyTorch implicit syncs**: `loss.item<double>()` and `mini.label_mask.any().item<bool>()` are issued every batch and force a host-blocking synchronization. These eat into the cross-stream concurrency window.
3. **GPU saturation**: RTX 5070 Ti has 70 SMs. With 256 threads/block and ~5000 blocks for assemble, the GPU is heavily loaded already; adding a second stream on top doesn't find empty SMs.
4. **PCIe-bound `.to(device)`**: small int64 tensors (edge_indices, labels, mask) are transferred per batch even with the prefetcher. PCIe is a single resource — splitting these across streams doesn't increase bandwidth.

A larger model (3 layers, hidden=512+, GAT with attention heads) where `forward+backward` takes longer would likely benefit more from Stage 3 because the overlap window grows.

## Decision

- **Keep `useCudaStreams: false` as the procedure default**. Set to true only as a research opt-in.
- **Stage 1 (`useAsyncPrefetcher: true`) remains the primary speedup** at 1.477× on this run (1.609× on the previous Stage 1 A/B run from 2026-05-07; 8% variance is within noise across 5-epoch micro-benchmarks).
- The 14 commits in the C3 series stay — Stage 3 infrastructure is committed but inactive by default. If we later target larger datasets or larger models on a beefier GPU, flipping the flag is a one-line change.

## Files

| File | Origin (Desktop) | Notes |
|---|---|---|
| [`a_sequential.csv`](./a_sequential.csv) | `07_stage3_20260508_084722/` | A — useAsyncPrefetcher=false, useCudaStreams=false |
| [`b_stage1.csv`](./b_stage1.csv) | same | B — useAsyncPrefetcher=true, useCudaStreams=false |
| [`c_dualstream.csv`](./c_dualstream.csv) | same | C — useAsyncPrefetcher=true, useCudaStreams=true |
| [`summary.txt`](./summary.txt) | same | bench script's auto-generated comparison |

## Reproducing

```bash
cd ~/MillenniumDB_Testing
cmake --build build/Release -j $(nproc)            # need HEAD ≥ 3cb4be3e
./build/Release/bin/mdb server data/dbs/gql/papers100M --port 29950 \
    --threads 16 --browser false --timeout 86400

# In another terminal:
bash ~/Desktop/spec13_papers100m_e2e/post_pop_os/07_stage3_validation.sh
```

The bench takes ~13 minutes (3 × ~5 min runs). All six artifacts (3 CSVs + summary + server.log + this README's source) end up in `~/Desktop/.../07_stage3_<TS>/`.
