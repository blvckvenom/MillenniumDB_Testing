# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# First: Attribution
Never claim authorship of the code or mention that you wrote it.

## Overview

MillenniumDB is a graph-oriented database management system (DBMS) developed by the Millennium Institute for Foundational Research on Data (IMFD). It supports multiple graph models and query languages:

- **RDF Model**: SPARQL 1.1 support (see wiki: SPARQL-Implementation-Status.md)
- **Quad Model (QM)**: Property graphs with single edge labels and directed edges, using a custom Cypher-like query language (MQL)
- **GQL Model**: Property graphs supporting the GQL standard with undirected edges and multiple edge labels (early implementation, still missing functionality)

The project is in active development and not production-ready. Each graph model has its own query language - once you import data in one model, you must use that model's query language.

## Build Commands

```bash
# Release build (recommended)
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release && cmake --build build/Release -j $(nproc)

# Debug build (with sanitizers)
cmake -B build/Debug -D CMAKE_BUILD_TYPE=Debug && cmake --build build/Debug -j $(nproc)

# Verify build
build/Release/bin/mdb help
```

For detailed setup, dependencies, and Boost installation: see `docs/MillenniumDB.wiki/Setup.md`

## Testing

```bash
./scripts/run-tests           # All tests
./scripts/run-tests sparql    # SPARQL integration tests
./scripts/run-tests mql       # MQL integration tests
./scripts/run-tests gql       # GQL integration tests
./scripts/run-tests unit      # Unit tests via ctest
```

The test script automatically sets up Python venv, builds Debug, and runs integration tests.

## Database Paths

**GQL Data:**
- **Import source (example data)**: `data/example/gql/` - Sample GQL datasets for importing
- **Database destination**: `data/dbs/gql/` - Location where imported GQL databases are stored

## Code Architecture

### High-Level Directory Structure

- **src/bin/**: Entry point (`mdb.cc`) with command-line argument parsing
- **src/cli/**: Interactive CLI implementation
- **src/graph_models/**: Core graph model implementations
  - `common/`: Shared code across models
  - `rdf_model/`: RDF/SPARQL implementation
  - `quad_model/`: Quad Model/MQL implementation
  - `gql/`: GQL implementation
  - `object_id.h`: Central ObjectId type (64-bit with 8-bit type prefix)
- **src/storage/**: Persistent storage layer
  - `index/`: B+tree implementations and index structures
  - `page/`: Page-based storage management
  - `catalog/`: Database metadata and model identification
  - `dictionary/`: External string storage
  - `tuple_collection/`: Intermediate result storage
- **src/query/**: Query processing pipeline
  - `parser/`: ANTLR4-based parsers (separate for SPARQL, MQL, GQL)
  - `executor/`: Iterator-based query execution (`binding_iter/`)
  - `optimizer/`: Query optimization
  - `rewriter/`: Query rewriting passes
- **src/gpu/**: GPU-accelerated operations module (library: mdb_gpu)
  - `gpu_device.h/cc`: Runtime GPU VRAM + CPU RAM detection
  - `resource_planner.h/cc`: 5-strategy adaptive sort selection
  - `sort/`: Multi-pass CUB RadixSort (DoubleBuffer), CPU fallback, chunked sort
  - `ops/`: CUDA kernels (bitset filter, UNDIRECTED expand)
  - Zero MillenniumDB dependencies; optional CUDA via `MDB_GPU_ENABLED`
- **src/gnn/**: GNN training pipeline (library: mdb_gnn_core)
  - `core/`: CUDA context, sparse ops (scatter_sum/mean/max), memory pool, FeatureAssembler (CUDA kernel + fallback)
  - `storage/`: FeatureMatrix [N,D] mmap, RowMapping, PackedBatchStore, FourLevelStore
  - `sampling/`: Offline k-hop sampling, SeedSelector, MinHash reorderer
  - `projection/`: GnnProjectionAdapter, FeatureAccessor, TopologyAccessor, GnnMeta
  - `training/`: LabelStore, SplitStore, BatchAssembler, TrainingLoop, NpyWriter
  - `models/`: GraphSAGE MEAN (torch::nn::Module)
  - Build: `ENABLE_GNN=OFF` by default, requires LibTorch
- **src/import/**: Data importers for each model and format
- **src/network/**: HTTP server and client protocol handling
- **tests/**: Integration test suites (sparql/, mql/, gql/)



### Key Architectural Concepts

**ObjectId System (src/graph_models/object_id.h):**
All database values are represented as 64-bit ObjectIds:
- 8 bits: Type information (4-bit generic type, 2-bit subtype, 2-bit modifier)
- 56 bits: Value payload

Type encoding supports storage modes (inline, external, tmp) and type hierarchy (NULL, nodes, IRIs, strings, numerics, datetime, booleans, edges, paths, tensors, GQL-specific types).

**Model Isolation:**
Each graph model (RDF, Quad, GQL) is completely separate with different import formats, query parsers (ANTLR grammars), execution strategies, and indexing (e.g., RDF uses SPO/POS/OSP permutations).

**Storage Layer:**
Custom B+tree implementations, page-based storage with buffer management, string dictionary for external storage, model-specific catalog.

**Query Execution:**
Iterator-based volcano model with binding iterators in `src/query/executor/binding_iter/`. Supports worst-case optimal joins and regular path query evaluation.

**Tensor Compatibility:**
Tensors for GNN must not be related to existing tensor implementation. See `docs/MillenniumDB.wiki/Working-with-tensors.md`.

**GNN Training Pipeline:**
End-to-end GNN training within MillenniumDB. Flow: `graph_project` (with `includeFeatures`, `labelProperty`, `splitProperty`) → `gnn_offline_sample` (default orientation: UNDIRECTED, with optional `usePredefinedSplits`) → `gnn_materialize_batches` → `gnn_build_feature_store` → `gnn_train` (with optional `writeProperty` for embedding write-back). The `gnn_train` procedure requires a FourLevelStore (no fallback), creates a GraphSAGE model, trains via BatchAssembler + TrainingLoop, reports cache stats (l1/l2 hit ratios, l3/l4 reads), and exports model (.pt) + embeddings (.npy). With `writeProperty` set, EmbeddingWriter persists all node embeddings (including non-seeds via on-the-fly k-hop inference) as queryable tensor properties in the projection. Features loaded via FourLevelStore cache hierarchy (L1 GPU + L2 CPU pinned + L3 disk + L4 packed). FeatureAssembler dispatches to a CUDA kernel when gpu_features is on CUDA, otherwise uses LibTorch index_copy_ fallback. (Updated 2026-04-13)

**Phase 6 — Queryable Embeddings:**
After `gnn_train(..., writeProperty: 'embedding')`, embeddings are queryable via GQL: `USE proj MATCH (n) RETURN n.embedding` shows the tensor values, `cosineDistance(a.embedding, b.embedding)` computes similarity. Requires tensor type support in GQL (6 MASK_TENSOR types in GQL_OID) and the `cosineDistance` built-in function (justified by ISO/IEC 39075 §4.12.2 as implementation extension).

**Phase 6 Sub-2 — Model Checkpoints (Completed 2026-04-17):**
`gnn_train` saves atomic two-file checkpoints (`.pt` + `.ckptmeta`) to `<proj_dir>/gnn_output/<outputDir>/checkpoints/`. Both `best_model` (overwritten on each strict val-accuracy improvement) and `final_model` (written once at end) are produced; either can be disabled via `saveOnBestVal: false` / `saveFinal: false`. Pass `resumeFrom: 'best_model'` (relative, resolved against `<outputDir>/checkpoints/`) or an absolute path to `gnn_train` to continue training with preserved Adam optimizer state (`m`/`v` momenta), patience counter, best-val tracker, and full loss history. Three new YIELDs: `bestCheckpointPath`, `finalCheckpointPath`, `resumedFromEpoch`.

Four new GQL procedures: `gnn_predict(sample, feature, ckptName [, opts])` runs inference from a saved checkpoint (optionally writing embeddings back via `writeProperty`); `gnn_list_checkpoints(projection [, outputDir [, name]])` enumerates checkpoints sorted by creation time; `gnn_checkpoint_exists(projection, outputDir, name)` returns a boolean; `gnn_checkpoint_delete(projection, outputDir, name)` removes both files idempotently. Validation at load time enforces architecture dims (input/hidden/classes/layers), projection_name match, and SHA-256 hash of `gnn_meta.bin` — rejecting cross-projection or stale-data resumes with clear errors.

Implementation: `src/gnn/output/model_checkpoint.{h,cc}` (stateless utility: save_full/save_weights/load_*/validate_compat/list/exists/delete) + `src/gnn/output/auto_checkpointer.{h,cc}` (stateful policy observer wired into `TrainingLoop::Config::on_epoch_end`). Checkpoints use `GNNCKPT\0` magic, atomic write via `.tmp` → fsync → rename → fsync-dir sequence. See `docs/superpowers/specs/2026-04-16-model-checkpoint-design.md` for full design rationale. Validated by unit tests (23 ModelCheckpoint + 8 AutoCheckpointer + 8 TrainingLoop resume), E2E Step 10 (12 checks), and invariant tests in gnn_training suite (bit-identical predict reproducibility + resume parity within 0.03 testAcc delta).

## Development Notes

- **Language:** C++17 with `-std=c++17` required
- **Compiler flags:** `-march=native` for CPU optimizations
- **Debug builds:** Include AddressSanitizer and UndefinedBehaviorSanitizer
- **Release builds:** `-O3` optimization with optional IPO
- **Profile builds:** Require `PROFILE=ON` flag and gperftools/tcmalloc

## Commit Style

One commit per logical fix — never bulk commits grouping unrelated changes. Use `<type>(<scope>): <summary>` format with a detailed body (2-5 lines) explaining *what* and *why*. Add files explicitly by name (never `git add -A`). Types: fix, feat, refactor, chore, docs, test. Scopes: gnn, gql, hnsw, build, etc. Never mention agents, AI tools, review classifications (C1/H2/M3/L4), or verification footers (e.g. "Verified: build clean, unit 9/9...") in commit messages.

## ANTLR4 Parser Regeneration

The query parsers (GQL, MQL, SPARQL) are generated from `.g4` grammars using ANTLR4. When modifying grammar files, the autogenerated parser code must be regenerated.

**Prerequisites:**
- Java (openjdk-11+ works, tested with openjdk-21)
- zsh (`sudo apt install zsh`) — required by the generate scripts
- ANTLR4 jar at `/usr/local/lib/antlr-4.13.1-complete.jar` — **must match runtime version 4.13.1**

**Install ANTLR4 jar:**
```bash
sudo curl -o /usr/local/lib/antlr-4.13.1-complete.jar https://www.antlr.org/download/antlr-4.13.1-complete.jar
```

**Regenerate GQL parser:**
```bash
cd src/query/parser/grammar/gql/
./generate.sh
```

Each grammar model has its own `generate.sh`:
- `src/query/parser/grammar/gql/generate.sh`
- `src/query/parser/grammar/mql/generate.sh`
- `src/query/parser/grammar/sparql/generate.sh`

Generated files go to the `autogenerated/` subdirectory. The scripts rename `.cpp` → `.cc` to match project conventions.

## Documentation References

- **Setup guide**: `docs/MillenniumDB.wiki/Setup.md`
- **Usage examples**: `docs/MillenniumDB.wiki/Creating-and-running-a-database.md`
- **Model documentation**: `docs/MillenniumDB.wiki/Database-models.md`, `Quad-Model.md`, `MQL.md`, `GQL*.md`
- **SPARQL status**: `docs/MillenniumDB.wiki/SPARQL-Implementation-Status.md`
- **Example data**: `data/example/{rdf,qm,gql}/`

## Roadmap

The GNN pipeline roadmap is documented in `Partial_Idea/`:
- `Partial_Idea/README.md` - Phase overview and status
- `Partial_Idea/decisions/` - Architecture Decision Records (ADRs)
- `Partial_Idea/phase_00_libtorch_foundation/` through `phase_06_output_storage/` - Per-phase details

## External Reference Documentation

**ISO GQL Standard (ISO/IEC 39075:2024):**
- Full PDF: `docs/external_references/ISO_IEC_39075_extracted/ISO IEC 39075-2024.pdf`
- Section Index: `docs/external_references/ISO_IEC_39075_extracted/INDEX.md`

**Neo4j Graph Data Science:**
- Full Manual: `docs/external_references/NEO4J_USER_MANUAL_DOC/neo4j_graph_data_science_manual_.md`

**NVIDIA CUDA Toolkit Documentation (68 PDFs, v13.2):**
- Full collection: `docs/external_references/NVIDIA_CUDA_DOCS/` (104 MB)
- Includes all CUDA libraries, architecture guides, tools, compiler docs

**NVIDIA CCCL — CUDA Core Compute Libraries (CUB + Thrust + libcudacxx):**
- PDFs with API reference: `docs/external_references/GNN_ESSENTIAL_DOCS/{CUB,Thrust,libcudacxx,cudax}.pdf`
- HTML offline (with search): `docs/external_references/CCCL_DOCS/` (open `index.html`)
- CUB: Device-wide primitives (DeviceRadixSort, DeviceReduce, DeviceScan, DeviceSelect), Block/Warp collectives
- Thrust: High-level parallel algorithms (sort_by_key, exclusive_scan, reduce_by_key, transform, gather/scatter)
- libcudacxx: CUDA C++ Standard Library (atomics, barriers, memory model, synchronization, math, mdspan)
- Thrust uses CUB internally as GPU backend

**GNN-Essential CUDA Documentation (22 PDFs, curated):**
- Location: `docs/external_references/GNN_ESSENTIAL_DOCS/` (63 MB)
- Core: cuda-programming-guide, CUDA Runtime/Driver API, Math API, Best Practices
- GNN pillars: cuBLAS (matmul), cuSPARSE (SpMM), cuRAND (sampling), cuSOLVER
- CCCL: CUB (757p), Thrust (714p), libcudacxx (769p), cudax (112p)
- Profiling: Nsight Compute + Profiling Guide, Nsight Systems
- I/O: GPUDirect RDMA, GPUDirect Storage API + Best Practices

## GQL Native Projection

**Implementation:** `src/graph_models/gql/projection/` (18 files)
**Procedure:** `src/query/procedure/builtin/project_procedure.h/cc`
**Documentation:** `docs/native_projection_review/` (7 documents) and `docs/MillenniumDB.wiki/GQL-Projections.md`

Key capabilities:
- `CALL graph_project(name, nodeProjection, relProjection [, config])` — creates disk-based subgraph projections
- Supports STRING, LIST, MAP (Neo4j GDS) syntax for node/relationship projection
- Orientation: NATURAL, REVERSE, UNDIRECTED (per-type overrides)
- Aggregation: SINGLE, COUNT, SUM, MIN, MAX (per-type overrides)
- Property configuration: renaming, defaults, per-property aggregation
- Query via `USE projection_name` (no GRAPH keyword)
- Tests: `tests/gql/test_suites/projection_native/`, `projection_properties/`, `projection_comprehensive/`, `projection_advanced/`, `projection_adaptive_buffer/`
- Sort buffer sizing: adaptive at runtime (`max(256 MB, MemAvailable * 3/4)`) via `src/misc/available_ram.h`, overridable with env var `MDB_SORT_BUFFER_MB=<integer_MB>`. See `docs/MillenniumDB.wiki/GQL-Projections.md` "Memory tuning" section. 23 unit tests in `src/tests/available_ram-test.cc`.

### Sort backend selector — `MDB_PROJECTION_SORTER` (added 2026-04-21, ADR 004)

Two backends are now available for the projection B+Tree index build phase, selected at runtime via the `MDB_PROJECTION_SORTER` environment variable:

- `MDB_PROJECTION_SORTER=classic` (default) — legacy `ExternalRecordSort` pipeline via `sorter_dispatch::run_classic`. Identical behavior to pre-2026-04-21 code.
- `MDB_PROJECTION_SORTER=radix` — new `RadixPartitionSort<N>` pipeline (Phase 1 parallel scan + per-thread partition files → Phase 2 parallel per-partition sort with `malloc_trim(0)` between partitions → Phase 3 concatenation into BPTLeafWriter/BPTDirWriter).

Key properties of the RADIX backend:

- **Peak RSS bounded by construction:** `O(num_partitions × 4 MB + num_workers × 512 MB) ≈ 2.5 GB`, independent of dataset size. Designed to eliminate the Run 5 failure mode on `papers100M` without requiring external cgroup protection (`systemd-run -p MemoryMax=...`).
- **Opt-in, zero regression:** 347/347 GQL integration tests pass under both backends. 20/20 B+Tree `.leaf` / `.dir` files byte-identical between backends on `cora_gnn` (validated by `scripts/test_projection_radix.sh`).
- **Adaptive parallelism:** partition count `clamp(total_bytes / 256 MB, 8, 128)`; worker pool `min(cores − scan_threads, memory_budget / 512 MB)` default 4.
- **Files:** `src/graph_models/gql/projection/{sorter_dispatch,partition_file,parallel_scan_partitioner,radix_partition_sort}.{h,cc}`, `src/tests/radix_partition_sort_test.cc`.
- **Tests:** 10 unit tests (`RadixPartitionSortTests` / `SorterDispatch` in ctest), plus golden-compare integration test on `cora_gnn` via `scripts/test_projection_radix.sh`.
- **Design record:** ADR 004 (`Partial_Idea/decisions/004_radix_partition_sort.md`), spec `docs/superpowers/specs/2026-04-21-radix-partition-sort-design.md`, plan `docs/superpowers/plans/2026-04-21-radix-partition-sort-plan.md`.

Related env vars: `MDB_SORT_BUFFER_MB` (CLASSIC backend's external-sort buffer) still applies; `MDB_PROJECTION_SPILL_DIR` for spill file location.

## Claude Code Configuration

This project includes Claude Code configuration:

- **MCP Servers**: `.mcp.json` - C++ semantic analysis via cclsp (C/C++ Language Server Protocol)
- **Hooks**: `.claude/hooks/` - Automatic code validation
- **Skills**: `.claude/skills/` - Specialized tools (architecture-researcher, mdb-patterns, run-plan)

See `.claude/` directory for full configuration details.
