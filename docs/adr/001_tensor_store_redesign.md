# ADR-001: GNN Tensor Store Redesign — Key-Value → FeatureMatrix

> **Architectural Decision Record (ADR)**
> **Date:** 2026-03-09
> **Status:** IMPLEMENTED (Step 0 complete, 2026-03-11)
> **Supersedes:** `phase_00_libtorch_foundation/00_GNN_TENSOR_STORE.md` (original spec)
> **Design Document:** `phase_00_libtorch_foundation/00_GNN_TENSOR_STORE_REDESIGN.md`
> **Implementation:** `src/gnn/storage/feature_matrix.{h,cc}`, `row_mapping.{h,cc}`, `feature_matrix_header.h`
> **Tests:** `src/gnn/tests/test_feature_matrix.cc` (52 GTest tests)
> **Commits:** 14 commits on `feature-matrix-pipeline` branch, merged to `feature-GNN` (2026-03-12)

---

## Context

The GNN module needs persistent tensor storage for training on datasets that don't fit
in memory, following the DiskGNN architecture (SIGMOD 2025). The original design spec
(`00_GNN_TENSOR_STORE.md`, Jan 2026) proposed a generic key-value store:

```cpp
// Original design: key-value abstraction
uint64_t id = store.store(data_ptr, shape, dtype);    // store by key
TensorInfo info = store.load(id);                      // load by key
store.remove(id);                                       // delete by key
store.compact();                                        // reclaim space
```

This was implemented as `FileGnnTensorStore` (~730 LOC + ~225 LOC header) and
`InMemoryGnnTensorStore` (~118 LOC + ~96 LOC header). HNSW index creation was used
as initial validation of the tensor store.

## Problem

After a detailed critique of the implementation and analysis of DiskGNN's I/O patterns,
we identified a **design mismatch**: the key-value abstraction doesn't align with how
the training pipeline actually accesses features.

### Findings from the Critique (11 total)

**CRITICAL (2):**
- **Dangling pointer in GnnTensorView**: `load()` returns raw pointer that becomes invalid
  after releasing the internal lock. Readers can read freed/moved memory.
- **Crash-safety window in compact()**: Between the rename of old files and `save_index()`,
  there is no valid `index.bin` on disk. A crash leaves the store unrecoverable.

**HIGH (3):**
- `fsync()` called per individual `store()` call — devastating for batch writes of 100K+ tensors
- Zero tests exist (CMakeLists.txt references `test_gnn_tensor_store.cc` which was never created)
- `count()` is O(n) instead of O(1) — iterates all entries

**MEDIUM (3):**
- `InMemoryGnnTensorStore` allows move construction, which is UB with `std::shared_mutex`
- No buffer size validation in `load()`
- Backup loop in `compact()` breaks on first gap in shard numbering

**LOW (3):**
- Missing FLOAT16/BFLOAT16 dtypes (needed for Phase 5 mixed precision)
- Generation counter in view is unused by most callers
- Shard directory iteration order is implementation-defined

### The Core Mismatch

The key-value store was designed for:
```
store("node_features", blob)  →  load("node_features")  →  random blob access
```

But DiskGNN needs:
```
FeatureMatrix [N, D]  →  sequential scan  →  sorted row extraction  →  reordering
```

| Feature Built | Feature Needed |
|---|---|
| String-keyed blobs | Row-indexed matrix [N, D] |
| Mutable (store/remove/compact) | Immutable after creation |
| fsync per write | Single sequential write + one fsync |
| Random mmap access | Sequential scan with MADV_SEQUENTIAL |
| Sharding by space | Sharding by row range |
| Soft-delete + compaction | No deletion needed |

**Summary**: The design anticipated correctly WHAT was needed (persistent tensor storage,
sharding, mmap) but didn't optimize for HOW it would be used (sequential scan, batch
extraction, reordering).

## Decision

Replace `FileGnnTensorStore` / `InMemoryGnnTensorStore` with three purpose-built components:

### 1. `FeatureMatrix` — Immutable flat [N, D] matrix on disk

- Row-major, contiguous, memory-mapped (read-only after creation)
- Supports: `create()`, `create_streaming()`, `open()`, `row()`, `scan()`, `extract_rows()`, `create_reordered()`
- **No locks needed** — immutability eliminates all concurrency issues
- Uses `madvise()` hints: `MADV_SEQUENTIAL` for scan, `MADV_WILLNEED` for batch extraction, `MADV_RANDOM` for individual lookups
- File format: 64-byte header (magic "GNNF", version, N, D, dtype) + raw row-major data

### 2. `PackedBatchStore` — L4 per-batch packed feature files

- Separate `Writer` (preprocessing, single-threaded) and `Reader` (training, multi-threaded)
- Each batch file is a small (~1-5 MB) contiguous block → one sequential read
- Writer buffers writes, does single `fsync()` at `finalize()`
- Reader opens per-call file descriptor → inherently thread-safe

### 3. `FeatureMatrixMaterializer` — Bridge from MDB TensorManager

- Reads per-node tensors from `TensorManager` via projection
- Writes contiguous `FeatureMatrix` using `create_streaming()`
- Single sequential write pass (TensorManager reads are random but unavoidable)

### Dual Tensor Architecture (confirmed)

Both `TensorManager` and `FeatureMatrix` are needed — they serve different purposes:

| | MDB TensorManager | GNN FeatureMatrix |
|---|---|---|
| **Purpose** | Source of truth, per-node | Derived working copy, bulk I/O |
| **Access** | Random (by ObjectId) | Sequential scan, batch extraction |
| **Queryable** | Yes (`RETURN n.embedding`) | No (internal to training) |
| **Lifecycle** | Permanent | Reconstructible from TensorManager |
| **Analogy** | Row store | Column store |

## Consequences

**Positive:**
- Eliminates all 11 critique findings (dangling pointers, crash-safety, fsync-per-write, etc.)
- Directly supports DiskGNN's I/O patterns (scan, extract, reorder)
- Simpler code (~300 LOC estimated vs. ~730 LOC current)
- No locks needed for reads (immutable mmap)

**Negative:**
- Requires rewrite of two callsites (`import/gql/import.cc`, `gnn_hnsw_create_procedure.cc`)
- Immutability means re-creating the full matrix for any update (acceptable for training workflow)
- InMemoryGnnTensorStore loses its purpose (testing can use FeatureMatrix with tmpfs)

**Neutral:**
- `GnnDtype` stays unchanged (may add FLOAT16/BFLOAT16 later)
- `GnnTensorConverter` adapts to new interface
- Namespace stays `mdb::gnn`

## Open Questions

1. **Shard size**: 1 GB? 4 GB? Configurable? (64-bit Linux can mmap ~128 TB, may not need sharding)
2. **Row ID mapping**: Dense [0..N-1] with separate ObjectId→row file, or sparse?
3. **Compression**: LZ4 for L4 packed chunks? (DiskGNN mentions optional compression)
4. **FLOAT16/BFLOAT16**: Uniform dtype per matrix, or mixed-dtype columns?

---

*This ADR captures the analysis and decision from the 2026-03-09 design session.
The full design proposal is in `00_GNN_TENSOR_STORE_REDESIGN.md`.*
