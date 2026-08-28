# ADR-002: TopologySnapshot, GPU-Accelerated Projection, and LibTorch Removal

**Date:** 2026-03-24
**Status:** Accepted
**Supersedes:** None
**Context:** DiskGNN comparative analysis revealed sampling performance gap and optimization opportunities

---

## Context

Analysis of DiskGNN (Liu et al., SIGMOD 2025) showed that MillenniumDB's B+Tree-based sampling
operates at O(log E) per neighbor lookup, while DGL's CSC format provides O(1). For graphs at
papers100M scale (1.62B edges), this translates to sampling times of 5-15 minutes vs seconds.

Additionally, graph_project takes 30-60 minutes for large graphs because the 64 MB
StreamingRecordBuffer forces external sort for any graph with >2.66M edges.

## Decisions

### Decision 1: TopologySnapshot for O(1) Sampling

**Build a temporary CSC array in RAM from the projection's B+Tree during offline_sample().**

- Read projection's `from_to_edge` leaf pages in a single sequential scan
- Extract node VALUES via `oid.id & VALUE_MASK` (VALUES are dense 0..N-1, verified in import.cc:139)
- Store as uint32 indices + uint32 or uint64 indptr (adaptive based on edge count)
- Use Fisher-Yates partial selection (same as DGL) instead of Reservoir Sampling
- Free snapshot after sampling completes
- Projection remains persisted on disk (advisor requirement)

**Memory:** ~6.6 GB for papers100M (50% less than DiskGNN's 13.2 GB)
**Build time:** ~13s on NVMe (single sequential scan)

### Decision 2: graph_project GPU Acceleration (Adaptive Scheduler)

**Runtime detection of GPU VRAM + CPU RAM to choose optimal sort strategy.**

Strategies:
- GPU_FULL: sort data fits in VRAM -> CUB DeviceRadixSort (~2s)
- GPU_CHUNKED: sort in VRAM-sized chunks -> chunked RadixSort + CPU merge (~8s)
- CPU_PARALLEL: data fits in RAM -> std::sort(par_unseq) with TBB (~15s)
- EXTERNAL_SORT: fallback -> current disk spill behavior (minutes)

Key technique: pack sort keys as `(from_val << 32 | to_val)` into uint64 for RadixSort.
Works because node VALUES < 2^32 for all known datasets.

Libraries: CUB + Thrust from CUDA Toolkit (header-only, no extra dependencies).

**Immediate optimization (zero code changes):** increase buffer size to keep sort in RAM.
This alone reduces graph_project from 30-60 min to ~2.5 min.

### Decision 3: LibTorch Removal (Post-Pipeline)

**After Phase 4-6 are complete and validated with LibTorch, replace with native CUDA Toolkit.**

Replacement mapping:
- torch::Tensor -> raw float* + metadata struct
- torch::mm() -> cublasSgemm()
- torch::sparse::mm() -> cusparseSpMM()
- Autograd -> manual backward pass per GNN model (~200 LOC each)
- torch::optim::Adam -> custom Adam kernel (~30 LOC)
- CPU fallback -> Eigen (header-only)

Required CUDA Toolkit libraries: cuBLAS, cuSPARSE, cuRAND, CUB, Thrust.

**Sequence:** LibTorch first (functional) -> validate -> nativize (separate task)

## Consequences

### Positive
- Sampling performance matches DGL/DiskGNN at 50% less RAM
- graph_project can be 50-150x faster with buffer + GPU optimizations
- LibTorch removal eliminates 4 GB dependency, simplifies build
- System adapts to available hardware (GPU/CPU/disk) automatically

### Negative
- TopologySnapshot requires 6-17 GB RAM temporarily during sampling
- GPU acceleration requires CUDA Toolkit (already needed for GNN)
- Manual backward passes limit GNN model flexibility (fixed set of models)
- LibTorch removal is significant refactoring effort

### Risks
- uint32 indptr overflows for graphs >4.29B edges (mitigated by adaptive uint64 fallback)
- GPU sort chunking adds complexity for small-VRAM GPUs
- Manual backward passes are error-prone (mitigated by comparison with LibTorch output)

## References

- DiskGNN analysis: `docs/research/2026-03-24-diskgnn-analysis-and-sampling-optimization.md`
- DiskGNN paper: `docs/DiskGNN-Bridging-IO-Efficiency-and-Model-Accuracy-for-Out-of-Core-GNN-Training/`
- DGL source inspection: `/home/benito_pc/Escritorio/dgl_inspect/`
- ObjectId density verification: `src/import/gql/import.cc:139`
- Existing parallel sort: `src/graph_models/gql/projection/projection_storage.cc:740`
