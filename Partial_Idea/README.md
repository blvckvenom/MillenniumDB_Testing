# GNN Training System for MillenniumDB

> Native Graph Neural Network training infrastructure using LibTorch as the sole deep learning dependency.

**Status:** ACTIVE DEVELOPMENT - Phases 0-5 Complete (GraphSAGE MEAN), GPU Scheduler Complete, Phase 6 Pending
**Target:** MillenniumDB GQL Model
**Deep Learning Backend:** LibTorch (PyTorch C++ Frontend) — planned native CUDA migration post-pipeline
**Last Updated:** 2026-03-26

---

## Current Progress

| Phase | Component | Status | Notes |
|-------|-----------|--------|-------|
| Sprint 0 | Foundations | ✅ **COMPLETE** | All blockers resolved |
| Phase 0 | LibTorch Foundation | ✅ **COMPLETE** | ~2,700 lines implemented |
| Phase 1 | Projection Integration | ✅ **COMPLETE** | EdgeOrientation, NodeIterator added |
| Phase 2 | Offline Sampling | ✅ **COMPLETE** | 3 sampler variants implemented |
| Phase 3 | Feature Store | ✅ **COMPLETE** (2026-03-19) | Steps 0-3 all complete |
| Phase 4 | Training Pipeline | ✅ **COMPLETE** | LabelStore, SplitStore, BatchAssembler, TrainingLoop, gnn_train, NpyWriter |
| Phase 5 | GNN Models | ✅ **COMPLETE** (GraphSAGE MEAN) | GCN, GAT, GIN remain pending |
| Phase 6 | Output & Storage | ⬜ **PENDING** | Requires Phase 5 |

### Post-Pipeline Tasks (after Phase 6)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| TopologySnapshot | O(1) CSC sampling from projection B+Trees | ⬜ **PLANNED** | ADR-002; 50% less RAM than DGL |
| graph_project optimization | Adaptive GPU/CPU sort scheduler | ✅ **COMPLETE** (2026-03-26) | 20 commits, src/gpu/ module, 30 tests |
| GPU scheduler benchmarks | Work/Span, Amdahl, Speedup measurements | ✅ **COMPLETE** (2026-03-26) | Scatter-back is bottleneck (74%), TBB competitive |
| Adaptive RAM buffer | Dynamic ExternalRecordSort buffer from /proc/meminfo | ⬜ **PLANNED** | Highest-impact optimization: 256 MB → ~15 GB |
| Planner TBB preference | Prefer CPU_PARALLEL over GPU for large in-memory data | ⬜ **PLANNED** | TBB beats GPU pipeline at N > 5M |
| Scatter-back elimination | Sort full records on GPU, not just indices | ⬜ **PLANNED** | Removes 74% of GPU pipeline overhead |
| Streaming aggregation gate | Enable SUM/MIN/MAX at scale | ⬜ **PLANNED** | Separate from GPU scheduler |
| LibTorch removal | Native CUDA (cuBLAS + cuSPARSE + cuRAND) | ⬜ **PLANNED** | ADR-002; eliminate 4 GB dependency |

See [ADR-002](decisions/002_topology_snapshot_and_gpu_projection.md) for full rationale.

### Completed Implementations

**Core Layer (`src/gnn/core/`):**
- `cuda_context.{h,cc}` - CUDA device management, streams, memory queries
- `tensor_utils.{h,cc}` - ObjectId ↔ Tensor conversion, FeatureMatrix
- `memory_pool.{h,cc}` - GPU arena allocation, batch memory estimation
- `sparse_ops.{h,cc}` - SpMM, scatter/gather, edge_softmax (PyG replacement)

**Storage Layer (`src/gnn/storage/`):**
- `feature_matrix.{h,cc}` - ✅ Immutable [N,D] matrix with mmap read access (Step 0)
- `feature_matrix_header.h` - ✅ 64-byte on-disk format ("GNNF") (Step 0)
- `row_mapping.{h,cc}` - ✅ ObjectId[N] mmap array (Step 0)
- `gnn_dtype.h` - GNN-specific data types
- `gnn_tensor_converter.{h,cc}` - LibTorch tensor conversion
- `file_gnn_tensor_store.{h,cc}` - DEPRECATED (superseded by FeatureMatrix)
- `inmemory_gnn_tensor_store.{h,cc}` - DEPRECATED (superseded by FeatureMatrix)
- `gnn_tensor_store.h` - DEPRECATED (abstract interface, no longer used)

> ✅ **Redesign Step 0 complete (2026-03-11):** FeatureMatrix replaces key-value tensor store.
> ✅ **Step 1 complete:** PackedBatchStore (L4 per-batch packed files).
> ✅ **Step 2a complete:** MinHashReorderer (L3 disk cache reordering, 2 strategies).
> ✅ **Step 2b complete (2026-03-18):** BatchMaterializer + `gnn_materialize_batches` procedure.
> ✅ **Step 3 complete (2026-03-19):** GPU Cache (L1) + CPU Cache (L2) + DirectIO + CUDA Assembler + FourLevelStore coordinator.
> See [ADR-001](decisions/001_tensor_store_redesign.md).

**Training Pipeline (`src/gnn/training/`):** ✅ COMPLETE
- `label_store.{h,cc}` - mmap reader for `labels.bin` (node classification labels)
- `split_store.{h,cc}` - mmap reader for `splits.bin` (train/val/test masks)
- `batch_assembler.{h,cc}` - unifies features + topology + labels into `MiniBatch`
- `training_loop.{h,cc}` - epoch iteration, early stopping, validation, checkpoints
- `npy_writer.{h,cc}` - exports learned embeddings as NumPy `.npy` for external validation
- `gnn_train` GQL procedure — full orchestration of the training pipeline

> ✅ **graph_project GNN extension complete:** `includeFeatures`, `labelProperty`, `splitProperty` config fields; projection now produces `gnn_meta.bin`, `labels.bin`, `splits.bin`. `usePredefinedSplits` option added to `gnn_offline_sample`. `featureDim` and `numClasses` added as YIELD fields.

**GNN Models (`src/gnn/models/`):** ✅ GraphSAGE MEAN COMPLETE
- `graphsage.{h,cc}` - GraphSAGE MEAN aggregator using `scatter_sum` from `sparse_ops.h`
  - Configurable: `hidden_dim`, `num_layers`, `dropout`, L2 normalize
  - Forward pass iterates outside-in for mini-batch GraphSAGE
  - `get_embeddings()` for hidden representation export
- GCN, GAT, GIN: pending (Phase 5 continuation)

**Projection Layer (`src/gnn/projection/`):**
- `feature_accessor.{h,cc}` - Batch feature extraction with caching
- `topology_accessor.{h,cc}` - Neighbor traversal, EdgeOrientation, NodeIterator
- `gnn_projection_adapter.{h,cc}` - High-level GNN mini-batch API

### Key Design: Dual Tensor Architecture

> **MDB Tensors vs GNN Feature Storage**
>
> - **MDB `TensorManager`**: Source of truth — stores tensors per-node, queryable via GQL
> - **GNN `FeatureMatrix` (redesigned)**: Derived working copy — flat [N,D] matrix for bulk I/O
>
> `FeatureAccessor` uses `TensorManager` to read imported features.
> `FeatureMatrixMaterializer` bridges TensorManager → FeatureMatrix for training.
> `EmbeddingWriter` (Phase 6) writes learned embeddings back to TensorManager.
>
> See [ADR-001](decisions/001_tensor_store_redesign.md) for full rationale.

---

## Overview

This implementation brings native GNN training capabilities to MillenniumDB, following the DiskGNN architecture (SIGMOD 2025) for out-of-core training on graphs that don't fit in memory.

### Key Design Principles

1. **LibTorch Only** - No PyTorch Geometric, DGL, or other GNN frameworks
2. **Native Integration** - Uses MDB's existing storage primitives (B+Trees, BufferManager, ObjectId)
3. **Offline Sampling** - Pre-compute all mini-batches for optimal I/O patterns
4. **Hierarchical Caching** - Four-level feature store (GPU → CPU → Disk → Packed)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GNN TRAINING SYSTEM ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         USER INTERFACE (GQL)                                │ │
│  │                                                                              │ │
│  │  CALL gnn.offline_sample(...)  →  CALL gnn.train(...)  →  CALL gnn.infer() │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                      PHASE 0: LIBTORCH FOUNDATION                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │ CUDA Context │  │ Tensor Utils │  │ Memory Pool  │  │ Sparse Ops   │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│  ┌─────────────────────────────────────┼─────────────────────────────────────┐  │
│  │           PHASE 1: PROJECTION       │                                      │  │
│  │  ┌──────────────────────────────────▼─────────────────────────────────┐   │  │
│  │  │ ProjectionStorage Adapter │ Feature Accessor │ Topology Accessor   │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                         │
│  ┌─────────────────────────────────────┼─────────────────────────────────────┐  │
│  │        PHASE 2: OFFLINE SAMPLING    ▼                                      │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐             │  │
│  │  │   Seed     │ │  K-Hop     │ │ Comp Graph │ │ Frequency  │             │  │
│  │  │  Selector  │→│  Sampler   │→│  Builder   │→│  Analyzer  │             │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └─────┬──────┘             │  │
│  │                                                      │                     │  │
│  │                              ┌────────────────────────▼────────────────┐   │  │
│  │                              │    Mini-batch Materializer (B+Trees)   │   │  │
│  │                              └─────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                         │
│  ┌─────────────────────────────────────┼─────────────────────────────────────┐  │
│  │     PHASE 3: FOUR-LEVEL STORE       ▼                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │  │
│  │  │  L1: GPU Cache    (LibTorch CUDA Tensor, ~2GB)                       │ │  │
│  │  │  L2: CPU Cache    (MDB BufferManager extension, ~32GB)               │ │  │
│  │  │  L3: Disk Cache   (MinHash-reordered features)                       │ │  │
│  │  │  L4: Packed Chunks (Per-batch packed features)                       │ │  │
│  │  └──────────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                         │
│  ┌─────────────────────────────────────┼─────────────────────────────────────┐  │
│  │      PHASE 4: TRAINING PIPELINE     ▼                                      │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐               │  │
│  │  │ Feature  │──▶│ Feature  │──▶│  Graph   │──▶│  Model   │               │  │
│  │  │ Loader   │   │ Assembler│   │  Loader  │   │ Trainer  │               │  │
│  │  │   (T1)   │   │   (T2)   │   │   (T3)   │   │   (T4)   │               │  │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘               │  │
│  │       ▲              │              │              │                       │  │
│  │       │         [Queues for Pipelining]           ▼                       │  │
│  │       └────────────────────────────────── Loss, Gradients                 │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                         │
│  ┌─────────────────────────────────────┼─────────────────────────────────────┐  │
│  │       PHASE 5: GNN MODELS           ▼                                      │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐             │  │
│  │  │ GraphSAGE  │ │    GCN     │ │    GAT     │ │    GIN     │             │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘             │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                         │
│  ┌─────────────────────────────────────┼─────────────────────────────────────┐  │
│  │      PHASE 6: OUTPUT & STORAGE      ▼                                      │  │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐                 │  │
│  │  │   Embedding    │ │    Model       │ │   Feedback     │                 │  │
│  │  │    Writer      │ │  Checkpoint    │ │     Loop       │                 │  │
│  │  └────────────────┘ └────────────────┘ └────────────────┘                 │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Dependencies

```
Phase 0 (LibTorch Foundation)
    │
    ├──► Phase 1 (Projection Integration)
    │        │
    │        └──► Phase 2 (Offline Sampling Engine)
    │                  │
    │                  └──► Phase 3 (Four-Level Feature Store)
    │                            │
    │                            └──► Phase 4 (Training Pipeline)
    │                                      │
    └──────────────────────────────────────┴──► Phase 5 (GNN Models)
                                                      │
                                                      └──► Phase 6 (Output & Storage)
```

---

## Directory Structure

```
Implementation/
├── README.md                           # This file
├── SPRINT_00_FOUNDATIONS.md            # Blockers analysis (resolved)
├── GQL_PROCEDURE_INTEGRATION.md        # Cross-cutting: How to add GNN procedures
├── cross_cutting/
│   └── EMBEDDING_MANAGEMENT.md         # HNSW + external embedding import
│
├── decisions/
│   ├── README.md                       # Decision log and ADR template
│   └── 001_tensor_store_redesign.md    # ADR-001: Key-Value → FeatureMatrix (2026-03-09)
│
├── phase_00_libtorch_foundation/
│   ├── README.md                       # Phase overview
│   ├── 00_GNN_TENSOR_STORE.md          # Original design (SUPERSEDED)
│   ├── 00_GNN_TENSOR_STORE_REDESIGN.md # ⚠️ NEW: FeatureMatrix + PackedBatchStore design
│   ├── 01_CMAKE_INTEGRATION.md         # CMake configuration
│   ├── 02_CUDA_CONTEXT.md              # GPU device management
│   ├── 03_TENSOR_UTILS.md              # ObjectId ↔ Tensor conversion
│   ├── 04_MEMORY_POOL.md               # GPU memory management
│   └── 05_SPARSE_OPS.md                # Sparse matrix operations
│
├── phase_01_projection_integration/
│   ├── README.md
│   ├── 01_PROJECTION_ADAPTER.md
│   ├── 02_FEATURE_ACCESSOR.md
│   └── 03_TOPOLOGY_ACCESSOR.md
│
├── phase_02_offline_sampling/
│   ├── README.md
│   ├── 01_SEED_SELECTOR.md
│   ├── 02_KHOP_SAMPLER.md
│   ├── 03_COMPUTATIONAL_GRAPH.md
│   ├── 04_FREQUENCY_ANALYZER.md
│   ├── 05_MINIBATCH_MATERIALIZER.md
│   └── 06_LEAPFROG_INTEGRATION.md      # Leapfrog Triejoin for neighbor traversal
│
├── phase_03_feature_store/
│   ├── README.md
│   ├── 01_GPU_CACHE.md
│   ├── 02_CPU_CACHE.md
│   ├── 03_MINHASH_REORDERER.md
│   ├── 04_DISK_CACHE.md
│   └── 05_PACKED_CHUNKS.md
│
├── phase_04_training_pipeline/
│   ├── README.md
│   ├── 01_BLOCKING_QUEUE.md
│   ├── 02_FEATURE_LOADER.md
│   ├── 03_FEATURE_ASSEMBLER.md
│   ├── 04_GRAPH_LOADER.md
│   ├── 05_MODEL_TRAINER.md
│   └── 06_PIPELINE_ORCHESTRATOR.md
│
├── phase_05_gnn_models/
│   ├── README.md
│   ├── 01_BASE_LAYER.md
│   ├── 02_GRAPHSAGE.md
│   ├── 03_GCN.md
│   ├── 04_GAT.md
│   └── 05_GIN.md
│
└── phase_06_output_storage/
    ├── README.md
    ├── 01_EMBEDDING_WRITER.md
    ├── 02_MODEL_CHECKPOINT.md
    └── 03_FEEDBACK_LOOP.md
```

---

## Cross-Cutting Concerns

These documents address concerns that span multiple phases:

### GQL Procedure Integration

**Document:** `GQL_PROCEDURE_INTEGRATION.md`

Describes how to add new CALL statements for GNN functionality:
- `CALL gnn.offline_sample(...)` - Pre-compute mini-batches
- `CALL gnn.train(...)` - Train GNN model
- `CALL gnn.infer(...)` - Run inference
- `CALL gnn.write_embeddings(...)` - Store results

Key files:
- ANTLR Grammar: `src/query/parser/grammar/gql/GQLParser.g4`
- Visitor: `src/query/parser/grammar/gql/query_visitor.cc`
- Procedure Base: `src/query/procedure/procedure.h`
- Registration: `src/graph_models/gql/gql_model.cc`

### Embedding Management & HNSW (Cross-Cutting)

**Document:** `cross_cutting/EMBEDDING_MANAGEMENT.md`

Implemented after Phase 2, HNSW provides ANN (Approximate Nearest Neighbor) search
on embeddings — whether imported externally (LaBSE, BERT) or generated by GNN training (Phase 5-6).

- `CALL gnn.hnsw.create(...)` - Build HNSW index on embeddings
- `CALL gnn.hnsw.find_similar(...)` - Similarity search
- `CALL gnn.hnsw.list()` / `drop()` / `info()` - Index management
- `CALL gnn.sample_list()` / `sample_info()` / `sample_drop()` - Sample management

Key files: `src/storage/index/hnsw/` (15 files)

### Leapfrog Triejoin Optimization

**Document:** `phase_02_offline_sampling/06_LEAPFROG_INTEGRATION.md`

Leverages MillenniumDB's worst-case optimal join algorithm for efficient neighbor sampling:
- Sorted batch sampling using `seek()` operations
- Multi-type edge intersection
- Bidirectional traversal merge

Key files:
- B+Tree Iterator: `src/storage/index/leapfrog/leapfrog_bpt_iter.{h,cc}`
- Join Algorithm: `src/query/executor/binding_iter/leapfrog_join.{h,cc}`

**Important Finding:** GQL edge plans don't currently use Leapfrog (`get_leapfrog_iter()` returns nullptr). This is an optimization opportunity for k-hop sampling.

---

## Implementation Timeline

| Sprint/Phase | Component | Estimated Effort | Dependencies | Status |
|--------------|-----------|------------------|--------------|--------|
| Sprint 0 | Foundations | 3-4 weeks | None | ✅ **Complete** |
| Phase 0 | LibTorch Foundation | 2-3 weeks | Sprint 0 | ✅ **Complete** |
| Phase 1 | Projection Integration | 1-2 weeks | Phase 0 | ✅ **Complete** |
| Phase 2 | Offline Sampling Engine | 3-4 weeks | Phase 1 | ✅ **Complete** |
| Phase 3 | Four-Level Feature Store | 3-4 weeks | Phase 2 | ✅ **Complete** (2026-03-19) |
| Phase 4 | Training Pipeline | 2-3 weeks | Phase 3 | ✅ **Complete** |
| Phase 5 | GNN Models | 2-3 weeks | Phase 4 | ✅ **Complete** (GraphSAGE MEAN; GCN/GAT/GIN pending) |
| Phase 6 | Output & Storage | 1-2 weeks | Phase 5 | ⬜ Pending |

**Remaining Effort: ~1-2 weeks** (Phase 6)

---

## Testing Strategy

Each phase includes:
1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Cross-component testing
3. **Benchmark Tests** - Performance validation

Test datasets:
- **Small**: Cora (2.7K nodes) - rapid iteration
- **Medium**: ogbn-arxiv (170K nodes) - functional validation
- **Large**: ogbn-papers100M (111M nodes) - scale testing

---

## References

- DiskGNN (SIGMOD 2025): Offline sampling, MinHash reordering
- LibTorch Documentation: https://pytorch.org/cppdocs/
- MillenniumDB Architecture: See `CLAUDE.md` and `docs/`
