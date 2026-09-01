# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# First: Attribution
Never claim authorship of the code or mention that you wrote it.

# Source of truth (read this before citing any number)

There is a **published paper** and it is the canonical source for every headline figure. This file
and `docs/research/` are NOT.

1. **The paper**: `~/papers/2026-amw-millenniumai/sample-1col.tex` (outside this repo), plus its
   figure/table generators (`make_results_figure.py`, `make_table.py`).
2. **`docs/research/2026-07-01-papers100m-repro/REPRODUCE.md`** — the canonical procedure.
3. Notes in `docs/research/` dated **on or after 2026-07-14** (the campaign that backs the paper).
4. **Everything in `docs/research/` dated before 2026-07-14 is working material, not citable.**
   That is 97 of its 103 entries.

Not all notes are the same kind of document, and they were being cited interchangeably. Some are
measurements; some are root-cause analyses with the failure actually reproduced; some are cost
analyses with no run behind them; and some are **design-space explorations run by agent panels**
whose own header says `DESIGN ONLY — nothing implemented` (e.g. `GRAPH_PROJECT_REDESIGN.md`). The
last kind is not evidence of any sort.

**Canonical papers100M figures (from the paper):** preparation 95.7 min = projection **44.5** +
sampling 13.3 + feature store 37.9 (the block bake is NOT included). *Corrected 2026-08-17: this
line said 44.6, which sums to 95.8 and contradicts the 95.7 in the same sentence. The paper's figure
generators (`make_results_figure.py:180`, `make_cost_figure.py:83`) and `REPRODUCE.md:81,235` all
use 44.5.* training 41.2 min for 50
epochs = 48.9 s/epoch mean over 19 runs; test@bestVal 0.6579 ± 0.0009; bestVal 0.6907 ± 0.0011;
peak 96 % of the 16 GB GPU and 26.7 GiB of host RAM; feature cache 9.2 GB.

**Hardware note:** the machine has **two NVMe drives**, not one — `/` with 188 GB free and
`/mnt/data_disk` with 357 GB free, 545 GB total. It also has 103 GiB of active swap that no
measurement has ever controlled for.

## Branches, and where each thing lives (since 2026-09-01)

Two roles, not three.

| Branch | Role | Carries |
|---|---|---|
| `main` | published, clean | upstream layout, full `src/gnn`, `docs/{adr,design,MillenniumDB.wiki}`, 35 scripts |
| `feature-GNN` | development, and the only real history | `CLAUDE.md`, `Partial_Idea/`, `docs/superpowers/`, research notes, the campaign scripts, and the readability refactor |
| `refactor/readability` | redundant | the same commit as `feature-GNN`; kept only until the move is confirmed |

The 2026-08-28 version of this table claimed three distinct roles. It was wrong:
`refactor/readability` was never a sibling of `feature-GNN`, it was a strict
descendant, so one contained the other entirely. On 2026-09-01 `feature-GNN` was
fast-forwarded onto it — 18 commits gained, none lost, no merge commit, because a
fast-forward only moves the label. `feature-GNN` is the branch to develop on; its
name is what every research note, ADR and `REPRODUCE.md` cites.

**Fixes born on `main` do not reach `feature-GNN` by themselves.** The onboarding
repairs of 2026-08-31 were made directly on `main`, so `feature-GNN` still carries
their defects until they are cherry-picked across. Merging `main` into a
development branch is NOT the way to do it: `main` was built by squash-merging and
then deleting, and those deletions would take `CLAUDE.md`, `Partial_Idea/` and
`docs/superpowers/` with them.

**A second worktree holds `main` at `~/mdb_main_clean`.** Do NOT `git checkout main` inside this
repository: 758 files that `main` does not track would be deleted from disk, including this file.

`main` must never receive: `CLAUDE.md`, `Partial_Idea/` under that name, `docs/superpowers/`,
`docs/bibliography/`, the `docs/research` notes (except `REPRODUCE.md`), or the campaign scripts.
Their useful content already travelled renamed: decision records are `docs/adr/`, design documents
are `docs/design/`.

## The verification contract for any refactor

Three numbers. If one moves, behaviour changed:

```
46          switches registered through Ablation:: in the source   (static layer)
37          switches declared during one cora run                  (runtime layer)
0.8648649   scripts/cora_fastloop.sh, exact and deterministic
```

The runtime layer is what catches the damage that splitting a function can do silently: a
`static const bool x = Ablation::flag(...)` moved onto a less frequent path stops being declared
and nothing errors. Capture it by running the four procedures against a cora server and diffing the
`[ABLATION]` lines.

`scripts/arxiv_e2e.sh` is NOT deterministic (it samples with 8 workers): testAcc ranges 0.6867 to
0.6884 on the same binary. Do not treat its accuracy as a contract; its `useAddrTablesEffective`
and `l1HitRatio` assertions are exact and do count.

**Comments must not point at documents.** A readability pass removed every reference from code to a
`.md` file, because most pointed at documents that do not ship with the repository. If a claim needs
a source, cite published literature with its exact section, and read it first; if the reasoning is
ours, write it in the comment instead of citing anything. `docs/bibliography/README.md` records
every source that was verified, and the code never cites that file.

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
End-to-end GNN training within MillenniumDB. Flow: `graph_project` (with `includeFeatures`, `labelProperty`, `splitProperty`) → `gnn_offline_sample` (default orientation: UNDIRECTED, with optional `usePredefinedSplits`; Plan F parallel via `numWorkers`; **since 2026-07-06 the fanouts list is read in DGL/GraphBolt order — the LAST element samples the hop adjacent to the seeds — so configs compare string-for-string with DGL-based systems; `fanoutsNearestFirst:true` restores the legacy hop-order reading, and pre-flip docs/memories/scripts use the legacy notation**) → `gnn_materialize_batches` → `gnn_build_feature_store` → `gnn_train` (with optional `writeProperty` for embedding write-back). Inference / checkpoint ops: `gnn_predict`, `gnn_list_checkpoints`, `gnn_checkpoint_exists`, `gnn_checkpoint_delete`. The `gnn_train` procedure requires the **Spec D Four-Level Feature Store** (L1 GPU + L2 CPU pinned + L3 disk + L4 packed; no fallback). The **Spec #13 Four-Level Topology Store** is independent and optional — disable via `useFourLevelTopologyStore:false` to skip its populate cost (Plan F parallel sampling works on the BPT-direct path too). `gnn_train` creates a GraphSAGE model, trains via BatchAssembler + TrainingLoop, reports cache stats (l1/l2 hit ratios, l3/l4 reads), and exports model (.pt) + embeddings (.npy). With `writeProperty` set, EmbeddingWriter persists all node embeddings (including non-seeds via on-the-fly k-hop inference) as queryable tensor properties in the projection. FeatureAssembler dispatches to a CUDA kernel when gpu_features is on CUDA, otherwise uses LibTorch index_copy_ fallback. (Updated 2026-05-12)

**Reproducible papers100M procedure (exact, pinned params):** the consolidated end-to-end procedure — all 4 stages (`graph_project` → `gnn_offline_sample` → `gnn_build_feature_store` → `gnn_train`) with exact commands, expected outputs, and caveats — is in `docs/research/2026-07-01-papers100m-repro/REPRODUCE.md`. Fanout notation below is the pre-2026-07-06 hop order (the doc's commands predate the DGL-order default; re-running them verbatim on a current build requires `fanoutsNearestFirst:true`, or reverse the lists). Canonical accuracy hop-order `[10,15,20]` (= `[20,15,10]` in today's DGL-order notation): **test 0.6527 / val 0.6836** (50 epochs). **Current BEST: the hop-order `[20,15,10]` variant = today's `[10,15,20]`, the same string DiskGNN uses = test 0.6564 / val 0.6894**, cutting the gap to the DiskGNN paper (0.6591) to 0.27 pt; use it when comparing against DGL-convention papers. Note: the samples are `orientation:UNDIRECTED` (NOT REVERSE — the "rev" name means reverse-edges-added = undirected); the older `e2e5ep` sample is obsolete. Accuracy is invariant to the L1/L2 cache sizing (tiering only). `gnn_materialize_batches` in the flow above is deprecated (the build is self-sufficient from the sample). Sampling engine facts (measured 2026-07-01, study in `docs/research/2026-07-01-sampling-rng-study/RESULTS.md`): papers100M defaults to the **GPU_VRAM_COPY symmetric** backend (13.8 GB slice pinned, 1.41× the forced-CPU path) and the GPU sample is **bit-reproducible run-to-run** (pure function of seed + `topology_sym.csr` + splits); the warm-start `node_counts.bin` drift caveat applies to the CPU/four-level path only. MDB GPU sampling RNG = Philox 4x32-10 (counter-based), MDB CPU = mt19937_64, DGL GraphBolt = PCG32 — all exactly uniform (χ²-validated); cross-framework bit-parity is impossible by construction and statistically unnecessary. The backend planner's RAM gate charges the baked symmetric slice its true locked cost (ROW_PTR + staging window ≈ 1.16 GB, not the full 13.8 GB CSR — the slice's COL_IDX is served tiled from reclaimable page cache), so back-to-back samples within one server session both stay on the GPU backend (fixed 2026-07-03, `DirCsrDims::tiled_mmap`; wall-clock for a `[20,15,10]` sample ≈ 11.2 min, write-bound).

**Phase 6 — Queryable Embeddings:**
After `gnn_train(..., writeProperty: 'embedding')`, embeddings are queryable via GQL: `USE proj MATCH (n) RETURN n.embedding` shows the tensor values, `cosineDistance(a.embedding, b.embedding)` computes similarity. Requires tensor type support in GQL (6 MASK_TENSOR types in GQL_OID) and the `cosineDistance` built-in function (justified by ISO/IEC 39075 §4.12.2 as implementation extension).

**Phase 6 Sub-2 — Model Checkpoints (Completed 2026-04-17):**
`gnn_train` saves atomic two-file checkpoints (`.pt` + `.ckptmeta`) to `<proj_dir>/gnn_output/<outputDir>/checkpoints/`. Both `best_model` (overwritten on each strict val-accuracy improvement) and `final_model` (written once at end) are produced; either can be disabled via `saveOnBestVal: false` / `saveFinal: false`. Pass `resumeFrom: 'best_model'` (relative, resolved against `<outputDir>/checkpoints/`) or an absolute path to `gnn_train` to continue training with preserved Adam optimizer state (`m`/`v` momenta), patience counter, best-val tracker, and full loss history. Three new YIELDs: `bestCheckpointPath`, `finalCheckpointPath`, `resumedFromEpoch`.

Four new GQL procedures: `gnn_predict(sample, feature, ckptName [, opts])` runs inference from a saved checkpoint (optionally writing embeddings back via `writeProperty`); `gnn_list_checkpoints(projection [, outputDir [, name]])` enumerates checkpoints sorted by creation time; `gnn_checkpoint_exists(projection, outputDir, name)` returns a boolean; `gnn_checkpoint_delete(projection, outputDir, name)` removes both files idempotently. Validation at load time enforces architecture dims (input/hidden/classes/layers), projection_name match, and SHA-256 hash of `gnn_meta.bin` — rejecting cross-projection or stale-data resumes with clear errors.

Implementation: `src/gnn/output/model_checkpoint.{h,cc}` (stateless utility: save_full/save_weights/load_*/validate_compat/list/exists/delete) + `src/gnn/output/auto_checkpointer.{h,cc}` (stateful policy observer wired into `TrainingLoop::Config::on_epoch_end`). Checkpoints use `GNNCKPT\0` magic, atomic write via `.tmp` → fsync → rename → fsync-dir sequence. See `docs/superpowers/specs/2026-04-16-model-checkpoint-design.md` for full design rationale. Validated by unit tests (23 ModelCheckpoint + 8 AutoCheckpointer + 8 TrainingLoop resume), E2E Step 10 (12 checks), and invariant tests in gnn_training suite (bit-identical predict reproducibility + resume parity within 0.03 testAcc delta).

### GNN dataset access-skew profile — scope guidance for disk-cache work (DiskGNN SIGMOD'25 §5.1, §7)

DiskGNN's segmented-disk-cache + heuristic-search machinery is **dataset-shape-dependent**. The access-frequency skew of the seed-node neighborhood (i.e., what fraction of total feature accesses go to the top-K% most popular nodes) determines whether pure packed-feature chunks fit in any reasonable disk budget or whether a disk cache layer must be activated to deduplicate features across mini-batches.

| Dataset (paper abbr.)  | Top-1% access | 1-5% | 5-10% | >10% | Pure-pack blowup vs feature size | Disk cache needed? |
|---|---|---|---|---|---|---|
| Papers100M (PS)        | 43.2% | 36.5% | 11.2% | 9.1%  | ≤1× (fits) | No |
| MAG240M    (MG)        | 56.4% | 32.3% | 7.3%  | 4.0%  | <1× (fits) | No |
| Friendster (FS)        | 14.1% | 25.5% | 18.2% | 42.1% | 5.39×      | Yes |
| IGB260M    (IG)        | 22.5% | 30.0% | 20.8% | 26.7% | 10.19×     | Yes |

The disk_cache layer in DiskGNN is **opt-in** (activated only when `disk_size` constraint is exceeded by pure packing). Their heuristic for choosing segment size `s` (with `m=1` fixed) runs in seconds vs. brute-force in minutes-to-hours: at IG 7× blowup, 4.61 s vs. 3261 s = **707× speedup** at equal/better I/O amplification (paper §7 Table 7). Triggered only when `disk_size` constraint is set AND the dataset's natural blowup exceeds it.

**Implication for our roadmap (2026-05-07):** Spec C2 (heuristic search for `s` under `disk_size`) and the budget-driven activation logic in Spec D would only pay off if we target FS/IG-class datasets (low access skew). For PS/MG-class datasets where the top-1% nodes already dominate accesses, pure packing already fits in ≤1× feature size and our existing `Strategy::SEGMENTED` MinHash via composite hash (the GLOBAL `_reordered.fmat` consolidates DiskGNN's per-segment caches into a single file via `(segment_id<<32) | hash` ordering) is functionally sufficient. Priority for further work should therefore be **pipeline overlap (paper §5.3)** — the 4-stage producer-consumer overlap of feature loading, feature assembling, graph loading, and model training — which benefits all dataset shapes uniformly.

The Spec D telemetry (`gnn_build_feature_store` yields `slimMb`, `reorderedMb`, `gpuCacheMb`, `cpuCacheMb`, `totalDiskMb`, `overBudget`) lets us empirically verify whether the actual blowup of a given (sample, feature_dim) configuration matches the paper's prediction before deciding whether to invest in C2.

## Performance Optimization History

This section consolidates empirical validation results from completed performance work. Each subsection is a frozen record — see commit messages and `docs/research/` for raw data.

### Feature-store build folded into offline sampling + host-adaptive caches (delivered 2026-06-29)

Part of a 9-phase plan to dissolve `gnn_build_feature_store` into the surrounding procedures (full plan §9.2 of `docs/research/2026-06-28-fmat-investigation/PLANO_FEATURES_CODISENADO_V2.md`). All flags default OFF (additive, zero regression); gated bit-identical on cora (`scripts/cora_fastloop.sh` = 0.8648649 deterministic) + ctest 99/99. **12 commits, HEAD `d74d2ee7`.** The F0–F1 dissolution bullets are below; the assemble-optimization continuation (where the real epoch wins are) follows them.

- **`buildFeatureStore: true`** on `gnn_offline_sample` runs the four-level feature-store build as a tail step right after the sample is written, so a single call replaces the separate `gnn_build_feature_store` in the common flow. A shared parser (`src/query/procedure/builtin/gnn_feature_store_config.h`, `build_feature_store_config`) turns the options map into the build config; the standalone verb was refactored to use the same parser (single source of truth) and remains available. **Important correction to the documented pipeline flow:** `FourLevelStore::build` is self-sufficient from the sample — it does **not** require `gnn_materialize_batches` (which stays deprecated).
- **`skip_edges` in the L3 reorder re-read**: the co-access graph build only needs `all_unique_nodes`, so the reorder pass now skips the per-layer edge arrays — on papers100M ~87.5 GB → ~16 GB of re-read, bit-identical (the permutation is unchanged). Edges are restored for the later address-table/block builds.
- **`noCacheBin: true`**: build defers the L1 (GPU) and L2 (CPU) cache `.bin` (drops a `<feature>_cache.deferred` marker instead); the `FourLevelStore` runtime constructor lazy-builds them on first train, **sized for the train host's VRAM/RAM** (`detect_resources`), eliminating the build-host ≠ train-host cache-size mismatch and keeping the `.bin` out of the build artifacts. Built once, reused thereafter.
- **Packed-full feature store deprecated and removed** (`packFullFeatures` hard-refused at all entry points; infeasible at ~18× the feature matrix).
- Tool: `src/gnn/tools/ratio_probe.cc` measures the active-set ratio `|A_{L-1}|/|A_L|` from a sample (gates the deepest-layer aggregation memoization, a later phase) — measured 0.0893 on papers100M `[10,15,20]`.

**Assemble-optimization continuation (2026-06-30, same branch) — where the real epoch wins are:**
- **Self-contained blocks are mandatory for fast training (the −58% win).** Build/bake the offline computation-graph blocks (`bakeBlocks:true`); a sample WITHOUT them trains in `online` mode ~2× slower (the per-batch `sample_read + active + edge` index rebuild ≈ 64% of the assemble is otherwise redone every epoch). Verify `feature-load mode: self-contained` in the train log. Measured: papers100M `rev_10_15_20` epoch 98→41 s after baking. `bakeBlocks:true` on an already-built store bakes them via the fingerprint fast-path (do NOT add `force:true`, which deletes them).
- **GPU feature-cache sizing + `expandable_segments` (−15-17%, the second lever).** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is now the DEFAULT (set in `src/bin/mdb.cc` `main()`, overwrite=0) — it reclaims fragmented VRAM so a large GPU cache fits without OOM. Building the store with a bigger `l1CacheMb` (papers100M/16GB ≈ 5500-6144) raises the L1 hit ratio 71.5→91.4% (PCIe −74%, disk −60%). Rebuilding caches at a new size needs `force:true` + `force_reorder:false`+`force_packed_slim:false` (reuses the 56 GB reorder; slim regenerates).
- **VRAM calibration + auto-apply (measured cache sizing).** `gnn_train` measures peak VRAM and yields `peakVramMb` / `recommendedL1CacheMb` (= total_vram − measured non-cache reserve − margin), persisted to `<feature>_vram_calib.txt` (format centralized in `src/gnn/storage/vram_calib_io.h`). `gnn_build_feature_store(..., {autoCache:true})` reads that file and sizes the L1 budget to the recommendation, closing the self-calibrating loop (an explicit `gpu_budget_mb` still wins; a missing calib is a no-op). The safety margin is 1280 MB: a measured e2e showed the earlier 512 MB left the auto-sized L1 at the VRAM ceiling (~5.4 GB recommendation → 2 transient CUDA OOMs in the cold-start epoch that fall back to a per-batch path zero-filling those batches; they self-heal to real-sample-read by epoch 2). 1280 MB covers the cold-start spike + run-to-run measurement noise → recommendation ~4.9 GB, OOM-free. papers100M/16GB e2e: L1 sized to 5426 MiB (11.11M nodes), steady epoch 41.5s at 90% L1 hit. **Rebuild recipe to resize L1: `{force:true, force_reorder:false, bakeBlocks:true}` — let `force_packed_slim`/`force_caches`/`force_meta` default to true (they regenerate the tier-dependent caches/slim/addr_tables/blocks; only the layout-independent MinHash reorder is safely reused). `force_packed_slim:false` would leave a stale slim → inconsistent store.**
- **Opt-in assemble kernels (default OFF, bit-identical, all modest after the two levers above):** `MDB_FMAT_PAGE_ALIGN` (F2 — page-aligned `.fmat` v2, cuFile-ready, `data_offset` 64→4096, backward-compatible), `MDB_GNN_FUSED_ASSEMBLER` (F3.a — read L2 straight from the pinned cache via UVA, −5%), `MDB_GNN_DEVICE_SCATTER` (B — bulk H2D + HBM scatter instead of per-row UVA, −2%).
- **A deepest-layer fused gather+mean kernel ("K2") was investigated and REJECTED for epoch speedup (~0 post-cache-sizing):** the async prefetcher hides the forward, and the kernel still must read 100% of the receptive field (measured). The offline aggregation memo that WOULD cut the row count needs a ~92 GB artifact (keyed per (batch,node), no dedup) — justified by data but deferred. Tools `src/gnn/tools/deepest_source_probe.cc` (read-fraction probe) + the VRAM calibration above support these findings.

### Offline computation-graph blocks + self-contained blocks (delivered 2026-06-08)

"Finish Offline Sampling" Piece A. The per-batch computation graph (active-set sizes + int32 edge_index) is baked OFFLINE during `gnn_build_feature_store(bakeBlocks:true)` into `<sample_dir>/blocks/block_NNNNNN.blk` (keyed by `compute_batch_content_hash`), and consumed in `BatchAssembler` to skip `build_active_indices`/`build_edge_indices` at train. **Self-contained blocks** (format v2: header carries `store_fp` + `num_unique_nodes` + `num_seeds` + `split` + a seed-id tail) let `gnn_train` read ONLY the compact block and SKIP deserializing `batches.dat` entirely — a minimal placeholder `GraphSample` (size N + seeds) drives the v2 addr-table feature gather (which uses only `.size()`+`batch_id`, never node contents), graph+seeds come from the block, with a `last_used_addr_tables()` safety net + graceful fallback. Files: `src/gnn/storage/block_format.h`, `block_store.{h,cc}` (BlockWriter/BlockReader + `is_fresh`/`open_self_contained` + `block_filename`), `src/gnn/training/graph_block_builder.{h,cc}` (extracted build_active/build_edge), `four_level_store.cc` (bake in `build_addr_tables_`), `batch_assembler.{h,cc}` (consume + `apply_block_mode_`/`set_block_mode_override`), procedure flags `bakeBlocks`/`blocksMb`/`blocksBuiltOk` + per-call `noBlocks`/`noSelfContained` (env `MDB_GNN_NO_BLOCKS`/`MDB_GNN_NO_SELF_CONTAINED`).

**Empirical (papers100M `e2e5ep` [10,15,20], 1179 train batches, celebi, clean same-session):** N=1 sequential epoch **498.5 → 252.7 s (−49%)** (self-contained per-stage: `sample_read=0, active=0, edge=0`, `assembler_kernel=143.3`); N=8 production epoch **~127 → ~59 s (−54%)**. MDB N=8 self-contained (59 s) vs the then-current DiskGNN celebi reference (123.5 s/ep, single-thread). **⚠️ SUPERSEDED 2026-07-10: the real DiskGNN celebi average is 61.66 s/ep over 50 epochs at a 5 GB cache (`~/logs_diskgnn/06_train50.log`), and the cache sweep gives 81.3 / 61.6 / 41.4 s at 5 / 10 / 15 GB — at 15 GB DiskGNN BEATS us. The defensible claim is 1.27× at matched cache budget, never 2× or 3×.** — best-achievable-on-celebi each (DiskGNN's batched_packing OOMs at N>1 on 30 GB). HONEST: per-batch (sequential) MDB is still **~2× slower** than DiskGNN (252.7 vs 123.5) — the win is parallelism + the producer-CPU reduction, NOT per-batch efficiency. Bit-identical (cora N=1 dropout=0 = **0.8574939** self-contained == online) + **ctest 83/83**. Gotchas: `bakeBlocks` on an already-built store needs `force_caches:false,force_reorder:false,force_packed_slim:false,force_meta:false` to hit the phase5-only fast path; `is_fresh` skips v1 blocks by `sample_fp` (rm `blocks/` to re-bake v2). Full writeup: `docs/research/2026-06-07-offline-blocks/RESULT.md`.

**Packed-full feature store (EXECUTED 2026-06-08 → NEGATIVE for papers100M; kept as opt-in).** Attacked the dominant per-batch stage (`assembler_kernel` ≈110–143 s feature load+assemble) with a DiskGNN-`LoadFeats_Direct`-style per-batch contiguous `[N_b,D]` pack (`packed_full.dat`+`.idx`) read in ONE O_DIRECT pass = `mini.features` (no 4-tier scatter, no addr_table), composing with the self-contained block. Implemented end-to-end (7 commits `adf6dc3e`..`48b235ca`) and **bit-identical** (cora N=1 dropout=0 = **0.8574939** packed-full ON==OFF; ctest **84/84**). **BUT infeasible for papers100M `[10,15,20]`: it needs ~1.03 TB = 18× the 57 GB feature matrix** — MEASURED exactly from the 1512 self-contained block headers (Σ `num_unique_nodes` = 2.01 B node-instances) — because it stores every batch's full receptive field with **NO cross-batch dedup** (each ~111 M node recurs ~18× across batches). It would also READ ~1 TB/epoch vs the 4-tier's measured 50.9 GB ⇒ **slower too**. The disk guard correctly refused the build. **Root insight (verified in `~/DiskGNN` `examples/feat_packing.py` + `src/load.cc` + paper): DiskGNN does NOT naive-full-pack — it splits cold nodes by popularity (>1-batch → a shared, dedup'd, MinHash-reordered disk cache; ==1-batch → per-batch chunk) + segments to a disk budget; the 4 KB `_unique2(in_idx/num_feat_per_page)` page-merge cuts read amplification, not duplication. That hybrid IS our 4-tier (L3 `reordered.fmat` = disk cache, L4 `packed_slim` = packed chunk). `packed_full` was the naive "pure packing" the paper rejects.** Also clarified "why our N=1 sequential epoch is slower than DiskGNN's 123.5 s": not apples-to-apples — DiskGNN's "single-thread" baseline runs the train loop single-threaded but its feature load is internally parallel (4 io_uring threads + OpenMP memcpy); our N=1 (`prefetchNumWorkers:1`) is one producer thread. At matched parallelism N=8 = 111.4 s/ep vs the then-current DiskGNN reference of 123.5. **⚠️ SUPERSEDED 2026-07-10 — see the correction above: DiskGNN's real celebi average is 61.66 s/ep, so this comparison no longer holds.** **Conclusion: the 4-tier (dedup) is the right design for deep-fanout large graphs.** `packFullFeatures` build + auto-detect consume + `noPackedFull` override are kept opt-in (pay off only for shallow fanout / small receptive fields). v2 restored via `scripts/papers100m_restore_v2.sh`. Spec `docs/superpowers/specs/2026-06-08-feature-gather-design.md`, plan `docs/superpowers/plans/2026-06-08-feature-gather.md`, full writeup `docs/research/2026-06-08-packed-full/RESULT.md`. Piece B (addr_table v3 assemble arrays) remains deferred (skip-assemble probe ~0 payoff).

### GNN training pipeline overlap — Spec C3 stage 1 (delivered 2026-05-07)

`TrainingLoop` overlaps each batch's `BatchAssembler::assemble()` + `host→device` transfer with the previous batch's model `forward+backward` via `AsyncBatchPrefetcher` — a single-worker, bounded-queue producer-consumer fronting the assembler. Default queue size 2 (matches DiskGNN SIGMOD'25 §6 "the sizes of all shared queues are set to 2"). Procedure parameter `useAsyncPrefetcher: bool` defaults **true** since 2026-05-07 (`gnn_train` opts).

**Empirical validation** (papers100M_caminoD_sample, 5 epochs SAGE 2-layer hidden=256 dropout=0.3 lr=0.001 random_seed=42 on celebi RTX 5070 Ti + 20-core CPU + 30 GB RAM):

| Metric           | Sequential (no prefetcher) | With prefetcher (queue=2) | delta            |
|------------------|----------------------------|---------------------------|------------------|
| trainSeconds     | 154.51                     | 96.01                     | **1.609× speedup** |
| assembleSeconds  | 101.72                     | 45.64                     | -55.1%           |
| forwardSeconds   | 2.97                       | 2.56                      | within noise     |
| backwardSeconds  | 5.85                       | 5.06                      | within noise     |
| bestValAccuracy  | 0.5942                     | 0.5940                    | match (±0.0002)  |
| l1HitRatio       | 1.0                        | 1.0                       | ✓                |

**Per-batch cost model** (1.6M batches across 5 epochs):
- Worker thread (read_sample + edge_index_build + assemble_kernel): ~35 μs/batch
- Main thread `.to(device)` for edge/labels/mask + compute: ~33 μs/batch
- Pipeline throughput ≈ max(35, 33) = 35 μs/batch vs sequential 68 μs/batch
- Inner-loop speedup 1.94× collapses to 1.609× total because the validation phase remains sequential.

**Comparison with DiskGNN SIGMOD'25 Table 6** (sequential vs pipelined): paper reports 1.71-2.44× across FS/IG configurations. Our 1.609× sits below their range because (i) validation is not overlapped, and (ii) the `assemble_kernel` and `model.forward` share the default CUDA stream (Stage 3 of the C3 plan would split them via `cudaStream_t` + `cudaEventRecord` / `cudaStreamWaitEvent`, deferred due to LibTorch C++ stream-API complexity).

**Files:** `src/gnn/training/async_batch_prefetcher.{h,cc}` (the prefetcher class, 8 unit tests in `test_async_batch_prefetcher.cc`), `TrainingLoop::Config::use_async_prefetcher` + `prefetch_queue_size` (the toggle), `TrainingLoop::Result::{assemble,forward,backward}_seconds` (the per-stage timings yielded as `assembleSeconds`, `forwardSeconds`, `backwardSeconds`).

**Commits**: `264b5cf3` (Stage 0 timing instrumentation), `994dabc7` (Stage 1.A `AsyncBatchPrefetcher` class), `da86d6d8` (Stage 1.B `TrainingLoop` integration), `c688e8d3` (deadlock fix when `train_batches > queue_size` — the original Stage 1.B prefetched the lookahead BEFORE calling `next()`, which blocked on backpressure once the queue was primed full).

**A/B validation harness**: `~/Desktop/spec13_papers100m_e2e/post_pop_os/06_stage1_validation.sh` — sets `useAsyncPrefetcher` to false and true on the same sample with identical hyperparameters and reports the speedup ratio + per-stage breakdown. Raw outputs persisted in `docs/research/2026-05-07-stage1-empirical/`.

### Spec C3 stage 3 — CUDA streams (validated NEUTRAL on celebi, 2026-05-08)

Stage 3 splits `assemble_kernel` (worker stream) and `model.forward + backward` (train stream) onto separate CUDA streams via `c10::cuda::CUDAStream` + `at::cuda::CUDAEvent`, following DiskGNN paper §5.3. All five sub-modules landed (commits `98c6302e`, `74b159ba`, `51ed0523`, `980c311a`, `3cb4be3e`) with 14 unit tests covering: stream pool acquisition, event lifecycle, cross-stream sync via `event.block`, model forward/backward stream consistency, prefetcher event recording, and end-to-end training-loop completion. Procedure parameter `useCudaStreams: bool` defaults **false**.

**Empirical A/B/C** (papers100M_caminoD_sample, 5 epochs SAGE 2-layer, git HEAD `3cb4be3e`):

| Run | trainSeconds | A→B | B→C | A→C |
|---|---|---|---|---|
| A — sequential (no prefetcher, no streams) | 143.88 |  |  |  |
| B — Stage 1 only                            |  97.39 | **1.477×** |  | |
| C — Stage 1+3 (dual stream)                 |  96.05 |  | **1.014×** | **1.498×** |

**Stage 3 dual-stream gives only 1.4% over Stage 1**. Reasons: (i) `assemble_kernel` is small (microseconds, ~5000 blocks/batch) so GPU contention isn't the bottleneck; (ii) PyTorch `loss.item()` / `label_mask.any().item()` introduce per-batch host-blocking syncs that consume the cross-stream concurrency window; (iii) RTX 5070 Ti's 70 SMs are already heavily utilized by either kernel alone, leaving no idle SMs for parallel work; (iv) `.to(device)` for small int64 tensors is PCIe-bandwidth-bound, not stream-parallelizable. Larger models (3 layers, hidden=512+, GAT) on bigger GPUs would likely benefit more — flipping the default is a one-line change when a workload that justifies it appears.

Raw outputs in `docs/research/2026-05-08-stage3-empirical/`. Bench script: `~/Desktop/spec13_papers100m_e2e/post_pop_os/07_stage3_validation.sh`.

### Plan E Phase 0 — auto-profile to unblock Spec #13 cold-start (added 2026-05-11)

Spec #13's `FourLevelTopologyStore` requires `<projection_dir>/node_counts.bin` to enable the L3 MinHash reorder. The file is normally produced as a side-effect of `gnn_offline_sample` *at the end* of a sample build (accumulating real access counts during sampling), which creates a chicken-and-egg dependency: the very first sample on a projection cold-starts without the reorder, falling back to random mmap access over the Spec #4-B `topology_*.csr` sidecar. On graphs whose sidecar exceeds available RAM (papers100M `topology_*.csr` ≈ 53 GB on celebi's 30 GB host) the cold path thrashes the page cache and the sample never completes — observed empirically as a 24 h+ `curl` timeout with zero output produced.

Plan E inserts a cheap random-walk profile pass between the sampler constructor and the four-level store's `build()`:

1. **`TopologyWalkProfiler::profile()`** issues N random walks of L steps each over the Spec #4-B reverse sidecar via the reader's O(1) `neighbors()` slice. Seeds are drawn from a **Vose alias-method table weighted by in-degree** — uniform seed selection on real citation graphs lands ~99% of walks on leaves (papers nobody cited) and produces unusable counts; degree-weighting matches the access pattern a real k-hop sampler would generate. Default: 100k walks × 5 steps = 500k lookups vs ~4.8 B for a full 3-layer `[10,15,20]` sample (~10 000× less work, ~10-30 s wall-clock on papers100M).
2. **`node_counts_io::persist()`** writes the resulting per-node counts to `<projection_dir>/node_counts.bin` using the existing `NODECNT0` format already consumed by `TopologyFrequencyProfiler::compute_from_node_counts_`.
3. **`BasicKHopSampler::Impl::run_phase0_auto_profile_if_needed_()`** runs steps 1+2 before `topology->enable_four_level_store(tcfg)`. The four-level store then activates its warm-start path on its very first build, performing the MinHash reorder of L3 with usable counts.

Opt-out: `autoProfileOnColdStart: false` config flag + procedure parameter. No-op when the sidecar is absent, `node_counts.bin` already exists, or the projection dir isn't resolvable. Telemetry exposed via new procedure yields: `phase0Triggered`, `phase0Succeeded`, `phase0WalksDone`, `phase0LookupsDone`, `phase0Millis`.

**Files**: `src/gnn/projection/topology_walk_profiler.{h,cc}` (Vose alias + walk loop), `src/gnn/sampling/node_counts_io.{h,cc}` (atomic writer for `node_counts.bin`), `src/gnn/sampling/basic_khop_sampler.{h,cc}` (Phase 0 integration), `src/gnn/sampling/sampling_config.h` (3 new flags), `src/gnn/sampling/offline_sampling_engine.{h,cc}` (telemetry plumbing), `src/query/procedure/builtin/gnn_offline_sample_procedure.{h,cc}` (yields + parse).
**Tests**: `src/tests/topology_walk_profiler_test.cc` (9 unit tests covering empty reader, deterministic seeds, isolated-node skip via alias weights, lookup bounds, defaults, and the papers100M regression fixture of 1-hub + many leaves).

**Commit cluster** (each fixed a distinct papers100M bug surfaced by live runs):

| SHA | What it fixes |
|---|---|
| `6e6776fc` | Plan E core implementation: `TopologyWalkProfiler` + `node_counts_io` + integration in `BasicKHopSampler::Impl` + procedure yields + 8 unit tests. 14 files, 921 insertions. |
| `42c6970b` | Degree-weighted seed selection (Vose alias). Uniform seed selection over [0,N) landed ~99% of walks on isolated leaves on papers100M, producing all-zero counts useless to the warm-start path. |
| `a05f7687` | Eligible-only alias domain. Even with degree-weighted alias, leftover-large cleanup left `alias_[i]=0` pointing at an isolated node 0; the fix builds the alias only over nodes with `degree>0` (mapping draws back through an `eligible_nodes` index). Added a 1-hub-19-leaves regression test. |
| `f71b3bf0` | Strip `ObjectId` type tag when following sidecar neighbors. The sidecar stores `dst` as the raw `ObjectId.id` (top 8 bits = type tag); my defensive `if (next >= n)` was correctly flagging all results as out-of-range but wrongly treating them as corruption. Mask with `0x00FFFFFFFFFFFFFFULL` first. |
| `c426e1b3` | Tier-assignment `avg_degree` must reflect real graph degree, not access-count frequency. `compute_tier_assignment` uses `avg_degree` to estimate bytes-per-node; deriving it from sparse Phase-0 counts (mean 0.003 on papers100M) collapsed per-node cost to fixed-overhead-only and packed the entire 111M-node graph into L1+L2 with L3 empty. Fix: read `num_edges/num_nodes` from the Spec #4-B sidecar reader when available. |

**Operational note**: on celebi (30 GB RAM) the default L1/L2 budgets (5 GB + 15 GB) still fit the full papers100M topology in RAM after the `c426e1b3` fix, so L3 only exercises when the user passes explicit `l1CacheMb` + `l2CacheMb` smaller than the graph fits in. To replicate the DiskGNN paper's 10%-cache configuration: pass `l1CacheMb: 1024, l2CacheMb: 2048` — that puts the top 21M nodes in RAM caches and 90M in the L3 mmap tier where the MinHash reorder permutation is actually consulted.

### Plan F — parallel offline sampling worker pool (added 2026-05-11)

Spec C3 stage 1 (commit `da86d6d8`, 1.609× speedup) overlapped one batch's `assemble + transfer` with the previous batch's model `forward+backward`, but the per-batch `khop_sampler->sample()` call itself remained single-threaded. On 20-core celebi this was the dominant wall-clock cost of papers100M-scale runs — empirically the legacy path on a 3-layer `[10,15,20]` fanout did not produce any `batches.dat` output within 3 h 37 min before manual abort.

Plan F parallelizes the outer batch loop in `OfflineSamplingEngine::do_run` using a SALIENT-style shared-memory worker pool. Configuration: `numWorkers` procedure parameter / `SamplingConfig::num_workers` field. 0 (default) = legacy sequential path, byte-identical to pre-Plan-F output. >=1 = parallel infrastructure, capped at `std::thread::hardware_concurrency()` at engine start.

Architecture:

- **`BasicKHopSampler` worker ctor** `(storage, config, shared_topology*, worker_offset)` — borrows the primary's `TopologyAccessor` (`FourLevelTopologyStore` + adjacency caches are read-only post-build) and owns private `LeapfrogGnnSampler`, `SeekBasedGnnSampler`, `std::mt19937_64`, and `node_access_counts` vectors. Worker ctors skip Phase 0 + `enable_four_level_store` + `prebuild_adjacency_cache` — those are the primary's responsibility, done once.
- **Atomic batch dispatch** — workers pull from `std::atomic<size_t> next_idx`, each batch identified by a `WorkItem{seeds*, split, batch_id}` triple. `batch_id` is assigned monotonically pre-spawn (train first, then val, then test) so it matches the legacy ordering byte-for-byte.
- **Per-batch re-seeding** — each worker calls `sampler.reseed_for_batch(batch_id)` before `sample()`, which re-seeds the worker's RNG + LeapfrogGnn + SeekBasedGnn as `random_seed XOR batch_id`. Output is therefore invariant to which thread picks the batch up — bit-identical across `numWorkers ∈ {1, 2, 4, 20}`.
- **Serialized writes** — `SampleStorage::write_sample` is `BPlusTree`-adjacent and not thread-safe; a single `std::mutex` around the call serializes the write phase. Sampling work itself runs concurrently.
- **Final reduce** — at end of run, each worker's `node_access_counts()` is merged into the primary via `merge_counts_from()` so the warm-start `node_counts.bin` reflects every worker's contribution.

Thread-safety — the four-level path is read-only post-build (FourLevelTopologyStore + L1/L2/L3 caches), and the legacy BPT-direct path serializes shared-buffer reads through `BufferManager::vp_mutex`, so both are safe for concurrent workers. The only Plan F-specific requirement is that each worker thread carries a valid `QueryContext::_query_ctx` pointer — `_query_ctx` is `thread_local`, and `std::thread`-spawned workers default it to `nullptr`, so the first BPT access null-derefs inside `BufferManager::get_page_readonly` and SIGSEGVs the server. `OfflineSamplingEngine` handles this by capturing the primary's `QueryContext*` and calling `QueryContext::set_query_ctx(primary_ctx)` at the top of each worker lambda before any sampling. Sampling reads only the shared buffer (`vp_map`) and never touches the worker-indexed private pool (`pp_map`, `tmp_info`), so workers safely share a single `QueryContext`. Validation 2026-05-12 on papers100M fanout `[2,2]` BPT-direct path:

| `numWorkers` | wall-clock | batches | uniqueNodes | freq dict |
|---|---|---|---|---|
| 1            | 62.4 s     | 6044    | 2 575 871   | reference |
| 4            | 37.7 s     | 6044    | 2 575 871   | identical |
| 20           | 39.6 s     | 6044    | 2 575 871   | identical |

`frequency.dat` v1 sparse format iterates `node_frequencies: std::unordered_map<uint64_t, uint64_t>` non-deterministically — raw bytes differ between runs (even at constant N) due to hash bucket ordering, but the dict-equality check confirms the (node_id, count) set is identical across N. `uniqueNodes`, `totalBatches`, and per-split counters match bit-for-bit. Determinism is therefore preserved at the semantic level (every batch is reseeded as `random_seed XOR batch_id`, so each batch's output depends only on `(seed, batch_id)`). The fix that made this work landed in commit `03ed201f`.

> ⚠️ **SCOPE CORRECTION (2026-06-19, measured):** the "each batch's output depends only on `(seed, batch_id)`" claim holds ONLY within a FIXED four-level tier layout — i.e. cora (single-tier) or papers100M with `node_counts.bin` frozen. With `useFourLevelTopologyStore:true` AND the warm-start `node_counts.bin` feedback (the DEFAULT on papers100M), the frequency-based tier assignment DRIFTS run-to-run (each sample rewrites `node_counts.bin` from its own access counts), which moves nodes between tiers (L1 hash / L2 CSR / L3 mmap / L4 BPT) that emit a node's neighbors in DIFFERENT orders, so the fixed-seed sampler draws different neighbors. MEASURED: two warm regens of the canonical papers100M config gave DIFFERENT `uniqueNodes` (48,724,978 vs 48,726,532) and `sampleContentFp` — the SAMPLE is NOT bit-reproducible run-to-run. Cold-start (no `node_counts.bin`) IS reproducible (fp `46f7d670644f36d2` twice). The Plan F numWorkers-invariance above was validated WITHIN a single fixed state / on cora, NOT across the warm-start drift. Reproducible run-to-run on papers100M requires `MDB_GNN_FREEZE_NODE_COUNTS=1` (validated bit-reproducible) or a seed-deterministic reorder. Full root-cause: `docs/research/2026-06-18-feature-store-study/FINAL_REPORT.md`.

Yields exposed via `gnn_offline_sample`: `numWorkersUsed` (effective pool size after the `hardware_concurrency()` cap, with `0` resolved to `1` for the legacy path).

**Files**: `src/gnn/sampling/basic_khop_sampler.{h,cc}` (worker ctor + `reseed_for_batch` + `merge_counts_from` + `get_topology`), `src/gnn/sampling/sampling_config.h` (`num_workers` + thread-safety doc), `src/gnn/sampling/offline_sampling_engine.{h,cc}` (parallel dispatch lambda + result.num_workers_used + per-worker `QueryContext::set_query_ctx`), `src/query/procedure/builtin/gnn_offline_sample_procedure.{h,cc}` (`numWorkers` param + `numWorkersUsed` yield).

**Commits**: `c8ffd166` feat(gnn): Plan F core. `03ed201f` fix(gnn): propagate QueryContext to workers (the SIGSEGV root cause). `198413d8` docs(gnn): Plan F BPT-direct is thread-safe after QueryContext fix.

**Paper-config validation on papers100M (2026-05-12)**:

```
fanout=[10,15,20], batchSize=1024, randomSeed=42, orientation=REVERSE,
usePredefinedSplits=true, useFourLevelTopologyStore=false,
useAdjacencyCache=false, numWorkers=20
```

Result: `totalBatches=1512, trainBatches=1179, valBatches=123, testBatches=210, uniqueNodes=9,157,028, computeMillis=152,658, numWorkersUsed=20`.

Wall-clock: **2 min 33 s** for the entire sample build vs the previous single-thread runs (legacy path or v6 Spec #13 populate-stuck path) that did not produce a `batches.dat` within 3 h 37 min before manual abort. Extrapolating from the small-fanout measurements (`numWorkers=1` `[2,2]` ≈ 62 s for 6044 batches with ~6 node receptive field vs paper-config `[10,15,20]` ~3000 max per seed and 1512 batches), the single-thread paper-config baseline projects to ~65 min — so the empirical speedup is ≈ **25×**, above the SALIENT MLSys'22 range. The gap is consistent with BPT-direct's I/O profile: per-lookup page reads parallelize naturally across kernel page-cache concurrency on a high-IOPS NVMe (celebi Gen4 NVMe), and Plan F unlocks them without paying the Spec #13 populate cost.

**Future work**:
- T#22 (deferred): cora-scale bit-identical regression test for `numWorkers={1, 2, 4, 8}` as a CI gate.
- Spec #13 populate parallelization — `populate_direction_via_sidecar_` is single-threaded today; with l1/l2 budgets tuned for papers100M scale it runs 15-30+ min on celebi. Independent of Plan F; once landed it unblocks the four-level path for papers100M-scale runs (where Plan F's 6-12× speedup will compound with the four-level cache-hit ratio).
- Speedup on small fanouts (`[2,2]`) is modest (1.4-1.5×) because each batch is too cheap to amortize the per-batch `assemble + sort + edge-index build`; the paper's 3-layer `[10,15,20]` fanout has ~30× more work per batch and should see speedups closer to the SALIENT 6-12× range.

### Fix #21 — pipeline overlap in create_reordered + L4 packed_slim (added 2026-05-14)

Opt-in via env var `MDB_GNN_PIPELINE_OVERLAP=1`. Replaces the per-chunk serial `read → pack → pwrite` loop with a single-producer/single-consumer `ChunkPipeline<T>` (capacity 2): the producer thread packs the next chunk while the consumer pwrites the previous one. Applied to three hot loops:

1. **`create_reordered` chunked path** — the default L3 reorder path (Fix #13). When pipeline ON, downgrades from multi-worker (`MDB_GNN_REORDER_WORKERS` default 2) to a single producer + single consumer; the kernel-side pwrite contention against a single NVMe outweighs the multi-worker concurrency.
2. **`create_reordered_external_sort_` Pass 2** — the bucket-merge phase of Fix #15 ext_sort. Producer reads buckets sequentially; consumer pwrites the assembled chunks.
3. **L4 `packed_slim` worker loop** — each of the N outer Plan F workers (`MDB_GNN_L4_WORKERS` default 4) spawns ONE inner writer thread; total threads = N×2 + main = 9 with defaults.

All three sites use the RAII `WriterGuard` / try-catch-rethrow pattern: when an exception propagates out of the consumer, the producer is signalled via `pipe.set_error()` and joined before the lambda returns, preventing the `std::thread` destruction → `std::terminate()` failure mode.

**Files:** `src/gnn/common/pipeline_overlap.h` (header-only template, ~100 LOC), `src/gnn/storage/feature_matrix.cc` (both create_reordered paths), `src/gnn/storage/four_level_store.cc` (L4 worker_fn).
**Tests:** 8 unit tests in `pipeline_overlap_test` (round-trip, backpressure, exception propagation, close semantics, single-slot, move-only payload, post-close pop, concurrent close+pop). Full ctest gate (72/72 PASS) verified in three configs: default off, pipeline on, pipeline on + ext_sort + mmap.

**Empirical (papers100M_paper_und, A/B/C/D bench on celebi 2026-05-14, run dir `~/Desktop/spec13_papers100m_e2e/post_pop_os/27_pipeline_overlap/`):**

| Run | L3 reorder | L4 packed_slim | Total wall-clock | Δ vs A |
|---|---|---|---|---|
| A baseline (chunked, no pipeline, `MDB_GNN_NO_FADVISE=1`) | 478s @ 113 MB/s | 248s | 1087.84s = **18:08** | — |
| B pipeline only (chunked + pipeline, `MDB_GNN_NO_FADVISE=1`) | 799s @ 67.9 MB/s | 284s | 1445.11s = **24:05** | +32.9% REGRESSION |
| C pipeline + fadvise (chunked + pipeline + fadvise) | 871s @ 62.2 MB/s | 267s | 1491.25s = **24:51** | +37.1% REGRESSION |
| D ext_sort + pipeline + fadvise | Pass1=263s + Pass2=158s = 421s | **210s** | 979.99s = **16:20** | **−9.9% WIN** |

**Findings:**
- Pipeline overlap on the **chunked path** is a **net regression** on celebi (Intel Core Ultra 7 265, 20 cores, 30 GB RAM, Gen4 NVMe). The chunked path's default `MDB_GNN_REORDER_WORKERS=2` runs 2 concurrent pwriters and saturates the NVMe write bandwidth (113 MB/s logical); the pipeline downgrade to one producer + one consumer halves throughput.
- Pipeline overlap on the **external_sort Pass 2** is **marginally slower** than the legacy sequential Pass 2 (158s vs run 25's 146s = +8%), but the ext_sort path as a whole still beats the chunked default because Pass 1 + Pass 2 ≈ 421s vs chunked's 478s.
- Fix #22 fadvise hints on the L4 packed_slim writes are the dominant win: D's L4 drops to 210s vs A's 248s (−15%) and vs run 25's 308s (−32%). The hints prevent the late-phase L4 throughput collapse documented in run 26.
- The acceptance gate (≥10% improvement vs A) was missed by D (9.9%) by a hair, but D delivers the page-cache-pressure resilience the bench was designed to measure.

**Recommendations:**
- `MDB_GNN_PIPELINE_OVERLAP=1` stays opt-in default OFF — DO NOT enable alone, regression on chunked path.
- For best wall-clock on celebi-class hardware: `MDB_GNN_REORDER_STRATEGY=external_sort MDB_GNN_PIPELINE_OVERLAP=1` together (config D).
- Fix #22 fadvise/madvise (default ON, disable via `MDB_GNN_NO_FADVISE=1`) is the primary deliverable of this work; it lands the L4 throughput win in any config.

> **UPDATE 2026-06-01 — OPTIMIZED PATHS ARE NOW THE IN-CODE DEFAULT (not opt-in).** The L3 reorder default flipped from `chunked` to **`external_sort`** (`feature_matrix.cc` `create_reordered`), and the L4 packed_slim default flipped to **Lever B partitioned** (`four_level_store.cc`, with a `RLIMIT_NOFILE` raise + non-partitioned fallback so the per-batch-fd requirement is safe under any inherited `ulimit -n`). A plain `gnn_build_feature_store` now takes the fast paths with NO env flags. Override to restore the old behaviour: `MDB_GNN_REORDER_STRATEGY=chunked` (use when temp disk for ext_sort bucket files is tight) and `MDB_GNN_SLIM_PARTITIONED=0`. Pipeline-overlap remains opt-in OFF (the A/B/C/D data shows it is neutral-to-negative on ext_sort and a regression on chunked). RATIONALE: on hosts where the source FM exceeds RAM, the legacy `chunked` reorder collapses to a random-gather page-cache thrash (~27 MB/s, ~34 min on papers100M) and the legacy packed_slim gathers cold rows randomly (~0.3 batches/s, ~84 min) — the ext_sort + Lever B sequential-scan paths avoid both. Separately, the reordered-feature `.idx` sidecar gained an IDX_VERSION-2 permutation fingerprint (`row_mapping.{h,cc}`) so a stale index from a prior permutation can no longer be silently adopted (it had been corrupting the L4 feature rows and capping papers100M val_acc at ~0.16). All gated by 79/79 ctest + cora E2E (testAcc ~0.88, O_DIRECT bit-identical).

### Fix #22 — page-cache relief via fadvise/madvise DONTNEED (added 2026-05-14)

Always-on (disable via env `MDB_GNN_NO_FADVISE=1` for ablation studies). Issues `posix_fadvise(POSIX_FADV_DONTNEED)` on regions of `batches.dat`, `reordered.fmat`, and per-chunk output once consumed, and `madvise(MADV_DONTNEED)` on equivalent mmap regions. Helper supports the Linux `len == 0` = "from offset to EOF" idiom in `fadvise_dontneed`.

Motivation is empirical: run 26 on 30 GB RAM showed the late-phase L4 throughput collapsing from 5.5 → 3.6 batches/s as the combined 56 GB reordered.fmat + 87 GB batches.dat + 8 GB caches working set saturated the kernel's clean-page budget, triggering swap thrash. With explicit DONTNEED hints the working set tracks active phases instead of accumulating.

Hint sites:
- `create_reordered_external_sort_` Pass 1: fadvise each bucket file after flush; madvise source FM mmap after Pass 1 completes.
- `create_reordered_external_sort_` Pass 2: fadvise consumed bucket file before close; fadvise output region after pwrite.
- `create_reordered` chunked path: fadvise output region after each chunk pwrite.
- L4 `packed_slim` worker: fadvise each `.bin` file after write_all.
- `sample_storage::read_sample_impl`: fadvise/madvise batches.dat range after deserialize (both ifstream and mmap branches).
- End of L4: `features.release_cache()` + `reordered_holder->release_cache()` so downstream callers (e.g., `gnn_train` running back-to-back) start with a clean page-cache budget.

**Files:** `src/gnn/common/page_cache_hint.h` (header-only, `fadvise_dontneed` + `madvise_dontneed`), `src/gnn/storage/feature_matrix.{h,cc}` (new public `release_cache()`), `src/gnn/storage/four_level_store.cc`, `src/gnn/sampling/sample_storage.cc`.
**Tests:** 4 unit tests in `page_cache_hint_test` (valid fd, invalid fd, valid mmap, env-disable short-circuit). Full ctest gate (72/72 PASS) verified.

**Empirical (papers100M_paper_und, A/B/C/D bench on celebi 2026-05-14):**

- **L4 throughput win is the headline result.** In the D config (ext_sort + pipeline + fadvise), L4 packed_slim took **210s** vs the no-fadvise A baseline's 248s (**−15%**) and vs run 25's no-fadvise / no-pipeline 308s (**−32%**). The fadvise hints prevent the late-phase L4 throughput collapse observed in run 26 (where mmap'd batches.dat + chunked output competed for the 30 GB page-cache budget and dropped throughput from 5.5 → 3.6 batches/s).
- **Combined with pipeline-on-ext_sort**, this delivers the **D = 9.9% wall-clock win over baseline A** (979.99s = 16:20 vs 1087.84s = 18:08) — see Fix #21 table for the full A/B/C/D breakdown.
- **fadvise hints are net-neutral on the chunked path's hot loop** — the C config (chunked + pipeline + fadvise) is 46s slower than B (chunked + pipeline, no fadvise), but the L4 phase still benefits (267s vs 284s). The chunked-path slowdown comes from the pipeline downgrade, not from fadvise overhead.
- **Bench run dir:** `~/Desktop/spec13_papers100m_e2e/post_pop_os/27_pipeline_overlap/`.

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

### Index set selection — `indexSet` config parameter (added 2026-04-23, ADR 005)

Projections can now opt into a reduced set of B+Tree indexes appropriate for GNN workloads. Config parameter on `graph_project`:

- `indexSet: 'ALL'` (default) — materializes all 10 topology indexes. Preserves pre-2026-04-23 behavior exactly.
- `indexSet: 'GNN_MINIMAL'` — materializes only 5 indexes (`nodes`, `node_label`, `label_node`, `from_to_edge`, `to_from_edge`). Sufficient for `gnn_offline_sample`, `gnn_materialize_batches`, `gnn_build_feature_store`, `gnn_train`, `gnn_predict`, and `EmbeddingWriter` on-the-fly k-hop.
- `indexSet: 'READONLY_TRAVERSAL'` — materializes 7 indexes (`GNN_MINIMAL` + `edge_label` + `label_edge`). Supports label-filtered GQL traversal but not edge-id lookups.

Example:

```gql
CALL graph_project('gnn_proj', 'Paper', 'CITES', {
    orientation: 'NATURAL',
    indexSet: 'GNN_MINIMAL',
    includeFeatures: 'node_features'
}) YIELD graphName, nodeCount, relCount RETURN *
```

Persisted in catalog v1.4 (projections from pre-Spec-#3 era read as `ALL` automatically, full backwards compat). Querying a dropped index raises `QueryException` with remediation hint naming the minimum preset that includes it.

Empirical measurements on ogbn-products (62 M edges, 2.5 M nodes):
- **GNN_MINIMAL:** 2.95 GB (-61% disk), 213 s (-57% wall-clock) vs ALL 7.24 GB / 493 s.
- **READONLY_TRAVERSAL:** 4.86 GB (-36%), 281 s (-43%).

Property indexes (`node_key_value`, etc.) are NOT controlled by `indexSet`; they remain conditional on the property configuration in `nodeProjection`/`relationshipProjection`, so GNN feature ingestion works with any preset.

**Files:** `src/graph_models/gql/projection/index_set.{h,cc}`, gated `build_one_index()` calls in `native_projection_builder.cc` + `projection_storage.cc`, catalog v1.4 in `projection_catalog.{h,cc}`, runtime checks in `gql_model.cc`.
**Tests:** 14 unit (IndexSet) + 15 unit (projection_missing_index) + 5 catalog v1.4 roundtrip + 86 shell-script checks (`scripts/test_projection_indexset_build.sh`) + 8 E2E missing-index query scenarios (`scripts/test_projection_missing_index_query.sh`).
**Benchmark:** `scripts/bench_indexset.sh` produces CSV with wall-clock/disk/RSS per dataset × preset.
**Design record:** ADR 005 (`Partial_Idea/decisions/005_gnn_minimal_indexset.md`), spec `docs/superpowers/specs/2026-04-25-gnn-minimal-indexset-design.md`, plan `docs/superpowers/plans/2026-04-25-gnn-minimal-indexset-plan.md`, master plan `docs/superpowers/plans/2026-04-23-projection-compression-stack-plan.md` §6.

### Topology snapshot — `buildTopologySnapshot` config parameter (added 2026-04-25, ADR 006)

Projections can now emit a pair of optional mmap-backed CSR sidecar files that accelerate GNN neighbor lookup from `O(log N)` (B+Tree) to `O(1)` (slice). Opt-in via `graph_project` config:

- `buildTopologySnapshot: true` — at finalize, emit `topology_fwd.csr` + `topology_rev.csr` next to the B+Tree files. Composes with any `indexSet` preset: each direction is skipped when its edge index (`FROM_TO_EDGE` / `TO_FROM_EDGE`) is absent from the active mask.
- `buildTopologySnapshot: false` (default) — preserves pre-2026-04-25 behavior exactly.

Example:

```gql
CALL graph_project('gnn_proj', 'Paper', 'CITES', {
    orientation: 'NATURAL',
    indexSet: 'GNN_MINIMAL',
    buildTopologySnapshot: true,
    includeFeatures: 'node_features'
}) YIELD graphName, nodeCount, relCount, topologySnapshotBytes RETURN *
```

The reader side (`TopologyAccessor::Impl`) detects sidecar presence at construction, validates magic/version/size/ROW_PTR invariants + SHA-256 of the source `.leaf` file, and falls back silently to the B+Tree path on any absence/failure/staleness. GNN sampling output is bit-identical across both paths under a fixed RNG seed (determinism test T4.11). Staleness detection is `SHA-256` of the source `.leaf` cached per-`TopologyAccessor` lifetime — typically 40-70 s one-time open cost on papers100M scale, free thereafter.

Post-hoc build for projections created without the flag: `CALL gnn_build_topology_snapshot('proj_name')` (procedure T4.9).

**Expected measurements on ogbn-products** (62 M edges, fanout [15, 10] UNIFORM, to be verified by Gate B benchmark):
- Sampling throughput: ≥ 50× baseline (target per master plan §9).
- Disk overhead: ~34% of projection size (fixed header amortization improves at scale — projected ~7-8% on papers100M).
- Build-time overhead: ~10% wall-clock (one O(M) degree scan + one O(M) streaming pass per direction + SHA-256 of source `.leaf`).

On-disk sidecar layout (per direction): 64-byte header (magic `TOPOCSR1`, version=1, id_width=8, flags, num_nodes, num_edges, source_sha256[32]) + `ROW_PTR[N+1]` + `COL_IDX[M]` + optional `EDGE_IDS[M]` (all uint64 little-endian). No catalog version bump — sidecar presence is detected by filesystem probe so pre-Spec-#4-B projections are fully forward-compat.

**Files:** `src/graph_models/gql/projection/topology_snapshot.h` (format header), `topology_snapshot_writer.{h,cc}` (streaming SHA-256 + atomic rename), `topology_snapshot_reader.{h,cc}` (mmap-backed `TopologySnapshotReader`), integration in `native_projection_builder.{h,cc}` (constructor 18th param + `build_topology_snapshots_()`), fast-path in `src/gnn/projection/topology_accessor.cc`.
**Tests:** 8 format + 12 writer + 17 reader + 3 builder-integration + sampling-determinism tests in `src/tests/`; shell integration scripts TBD in T4.12.
**Benchmark:** `scripts/bench_topology_snapshot.sh` (T4.12) produces CSV with sampling throughput + disk/build overhead per dataset.
**Design record:** ADR 006 (`Partial_Idea/decisions/006_topology_snapshot.md`), spec `docs/superpowers/specs/2026-04-25-topology-snapshot-design.md`, plan `docs/superpowers/plans/2026-04-25-topology-snapshot-plan.md`, master plan §8-9.

### Leaf encoding — `leafFormat` config parameter (added 2026-04-24, ADR 007)

Projections can now encode B+Tree leaf pages with a delta + LEB128-varint format that exploits the sort order of records, instead of the v1 redundant-byte-bitset that assumes shared prefix bytes. Opt-in per projection via the new `graph_project` config key `leafFormat`; persisted **per index** as a uint8 in catalog v1.5 (added 2026-04-24).

- `leafFormat: 'BITSET'` (default) — v1 leaf format. Byte-identical to pre-2026-04 output. Catalog v1.4 and earlier projections read implicitly as BITSET.
- `leafFormat: 'DELTA_VARINT'` — v2 leaf format: 16-byte header (`format_version=2`, `record_width`, `value_count`, `next_leaf`) + record 0 as N full LEB128 varints + records 1..k-1 as N zigzag-delta LEB128 varints. Each v2 page is bounds-checked, fail-safe via `BPTLeafV2DecodeException`, and decoded once-per-open into a stack-local `k×N` uint64 buffer.

Example:

```gql
CALL graph_project('paper_gnn', 'Paper', 'CITES', {
    orientation: 'NATURAL',
    leafFormat: 'DELTA_VARINT',
    indexSet: 'GNN_MINIMAL',
    buildTopologySnapshot: true,
    includeFeatures: 'node_features'
}) YIELD graphName, nodeCount, relCount RETURN *
```

Empirical measurements (with `indexSet: 'GNN_MINIMAL'`):
- **cora_gnn:** 368 KB → 72 KB (-80% disk). Read throughput 31.1 ms → 31.6 ms = 1.016× (Gate C ≤ 1.20×).
- **ogbn-arxiv:** 60.10 MB → 12.91 MB (-79%). Per-index: `from_to_edge` 28 MB → 5.5 MB (0.195×); `to_from_edge` 28 MB → 7.2 MB (0.256×, asymmetric due to sort-order entropy). Read throughput 437.6 ms → 435.7 ms = 0.996× (slightly faster than v1).
- **ogbn-products:** TBD — see bench report (T5.15).

Build time is unchanged in both directions (~0.06 s cora_gnn; ~3.1 s ogbn-arxiv); sort dominates, leaf serialisation is < 5%. v2 pages are immutable — switching `leafFormat` requires `drop_projection` + recreate. The T5.13b cursor-cache fix (commit `a94c06cf`) made `BPTLeafV2::search_index` sequentially O(k) instead of O(k²) per-iteration, restoring read parity.

**Files:** `src/storage/index/bplus_tree/bpt_leaf_format.{h,cc}`, `varint.{h,cc}`, `bplus_tree_leaf_v2.{h,cc}`, `bpt_mem_import.h` (`BPTLeafV2Writer` bulk-load), `src/graph_models/gql/projection/projection_catalog.{h,cc}` (catalog v1.5 per-index byte), `src/query/procedure/builtin/project_procedure.cc` (parse `leafFormat`).
**Tests:** unit suites `bpt_leaf_v2_format_test`, `varint_test`, `bpt_leaf_v2_writer_test`, `bpt_leaf_v2_reader_test`, `bpt_iter_dispatch_test`, `projection_catalog_v5_test`, `projection_leaffmt_config_test`; fuzz `bpt_leaf_v2_fuzz_test` (500 K random + 1 K smoke + 10 K boundary roundtrips, 100% tamper-flip detection); integration `scripts/test_projection_leaffmt.sh` (6-mode golden compare).
**Benchmark:** `scripts/bench_leaffmt.sh` (Gate C harness — size + read throughput per dataset × format).
**Design record:** ADR 007 (`Partial_Idea/decisions/007_delta_varint_leaf.md`), spec `docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md`, plan `docs/superpowers/plans/2026-04-25-delta-varint-leaf-plan.md`, master plan `docs/superpowers/plans/2026-04-23-projection-compression-stack-plan.md`.

### Graph storage — `graphStorage` config parameter (added 2026-04-24, ADR 008)

Spec #8 makes the B+Tree leaves of edge indexes **be** the CSR (Compressed Sparse Row) layout, unifying the Spec #4-B topology sidecar with the Spec #5 compressed leaf. Opt-in per projection via the `graph_project` config key `graphStorage`; scope is edge indexes only (`FROM_TO_EDGE`, `TO_FROM_EDGE`); persisted as a uint8 in catalog v1.6.

- `graphStorage: 'BTREE'` (default) — legacy per-index B+Tree; leaf encoding follows `leafFormat` (BITSET or DELTA_VARINT). Catalog v1.5 and earlier projections read implicitly as BTREE.
- `graphStorage: 'CSR_HYBRID'` — edge indexes emit v3 leaves: 16-byte header + offset table + src table + DELTA_VARINT-encoded dst stream. Node/property indexes preserve their `leafFormat`. When active, `buildTopologySnapshot` is silently ignored (the v3 leaves already provide O(1) neighbour access).

Example (full compression stack):

```gql
CALL graph_project('gnn_proj', 'Paper', 'CITES', {
    orientation: 'NATURAL',
    indexSet: 'GNN_MINIMAL',
    leafFormat: 'DELTA_VARINT',
    graphStorage: 'CSR_HYBRID'
}) YIELD graphName, nodeCount, relCount RETURN *
```

Empirical measurements (with `indexSet: 'GNN_MINIMAL'`):
- **cora_gnn:** edge B+Trees 150 KB → 54 KB (-63.9%).
- **ogbn-arxiv:** edge B+Trees 35 MB → 6.5 MB (-81.5%, ratio 0.185). Scan throughput 1.046× baseline (Gate D ≤ 1.20×) after T8.12b sequential cursor cache (commit `b9ca276f`, 343× fix). GNN sampling byte-identical on cora_gnn.

Build time is unchanged; sort dominates. v3 pages are immutable — switching `graphStorage` requires `drop_projection` + recreate. Pre-Spec-#8 catalogs (v1.5 and earlier) read as BTREE with full backwards compat.

**Known limitations (Spec #8-B follow-up — updated 2026-04-25):**
1. ~~`edge_id` is NOT persisted in the v3 layout — `count(e)` queries over CSR_HYBRID projections return inflated counts.~~ **RESOLVED via commits `bf534c8f` (non-hub) + `d2c2f5aa` (hub chain-head + continuation).** All edge_ids now persist parallel to dst stream. arxiv `count(e)` matches BTREE (1,166,243). Trade-off: CSR_HYBRID edge index disk size on arxiv 6.5 MB → 18.22 MB after persisting eids (+180%). The CSR architectural win shifts from "size reduction" to "O(1) neighbor access for sampling"; size is now equivalent-to-slightly-larger than BTREE+DELTA_VARINT, not 81.5% smaller.
2. ~~`gnn_offline_sample` on arxiv-scale CSR_HYBRID hits `decode_tuple_` failure at hub position 3974~~ **RESOLVED via T8-B.1 (commit `1428c66c`, 2026-04-24).** Three composed bugs (reader inflated total_tuples_ for hubs + ContinuationTag ctor missing + writer leaf-chain break) fixed cohesively. Arxiv CSR_HYBRID sampling now 3148 s/s (was 0, crashed). Validated E2E with n.embedding queryable across 169343 nodes.
3. Property indexes (`node_key_value`, `key_value_node`, `edge_key_value`, `key_value_edge`) always open as BITSET regardless of `leafFormat` config — this is intentional (hybrid mutability model). T8-B.2 fix (commit `49664262`, 2026-04-24) aligns reader dispatch with what writer actually persists. Topology indexes remain immutable (v2/v3); property indexes mutable (v1) for EmbeddingWriter writeback. See design rationale in `docs/research/2026-04-24-defense-qa.md` §2.1.
4. ~~EmbeddingWriter non-seed inference has O(chunk_idx) growth per chunk.~~ **RESOLVED via commits `896b3897` + `6521cc21` (2026-04-24).** Three-layer root cause (TopologyAccessor dedup collapsing under CSR zero edge_ids + BPT iter cross-page bug + cascade amplification). Fix: in-memory undirected adjacency cache built once per Phase B start. Arxiv 23 min → 1-2 s = 700×, products Phase B (39k non-seeds) <1 s.
5. **TensorManager external-tensor reads past 1 GiB static buffer were corrupted** — commit `4d4ba179` (2026-04-24) fixes two defects in the dynamic-buffer branch: `get_tensor` missed `ptr += num_bytes_read` after LEB128 size prefix (shifted every float by 2 bytes); `bytes_eq` had `size_t` underflow during dedup (silent duplicate writes). Triggered only when `tensors.dat > 1 GiB` (e.g., products-scale embedding writeback). Arxiv (347 MB tensors.dat) was unaffected; products (1.19 GB) was visibly broken pre-fix. Reading is now bit-correct at any scale.
6. **GNN split property mapping**: the writer in `try_extract_gnn_property` (commit `9d397335`) accepts `train`, `val`, `valid`, `validation`, `test` as canonical split tokens. Pre-fix only `val` was recognized for the validation bucket; `valid` (the OGB convention) silently fell to UNLABELED (255). On products this wiped 39,323 val nodes → bestValAccuracy=0. Post-fix: bestValAccuracy reaches 0.7274-0.8911 within published GraphSAGE-MEAN range.

### Sample adjacency cache — `useAdjacencyCache` (Spec #11, added 2026-04-24, ADR 009)

`gnn_offline_sample` exposes an opt-in adjacency cache via the new `useAdjacencyCache: bool` config option (default true). When enabled, `TopologyAccessor::prebuild_adjacency_cache` performs a single full-scan of `from_to_edge` + `to_from_edge` B+Trees and populates an in-memory `unordered_map<src, vector<AdjEntry>>`. Subsequent `get_neighbors(v)` calls become O(1) hash lookups instead of O(log N) B+Tree directory walks.

Empirical wins: products `gnn_offline_sample` 726.48s → **4.71s (154×)**, arxiv 12-15s → 1-2s. Memory cost: 16 bytes per directed edge × 2 directions + hash overhead = ~2 GB on products, ~80 GB projected on papers100M.

`useAdjacencyCache: false` falls back to the B+Tree direct path or to a Spec #4-B mmap'd sidecar if `buildTopologySnapshot:true` was used at projection build time. The sidecar path is recommended for memory-constrained scenarios (e.g., papers100M on commodity 32 GB RAM).

**Files:** `src/gnn/projection/topology_accessor.{h,cc}` (cache impl), `src/gnn/sampling/sampling_config.h`, `src/gnn/sampling/basic_khop_sampler.cc` (forces PER_NODE strategy when cache active), `src/query/procedure/builtin/gnn_offline_sample_procedure.{h,cc}` (`useAdjacencyCache` parsing).
**Tests:** 8 unit (`topology_accessor_adjacency_cache_test.cc`) covering NATURAL/REVERSE/UNDIRECTED parity, k-hop determinism, isolated nodes, disable/enable cycle. End-to-end byte-identical batches.dat verified on cora_proj.
**Design record:** ADR 009 (`Partial_Idea/decisions/009_sample_adjacency_cache.md`), spec `docs/superpowers/specs/2026-04-25-sample-adjacency-cache-design.md`.
**Future work:** Spec #13 — frequency-aware two-tier topology cache (RAM hot subset + mmap'd cold tier) for papers100M-scale where the full hash cache exceeds available RAM.

### Four-Level Topology Store (Spec #13, default true since 2026-04-25, ADR 010)

`gnn_offline_sample` automatically uses the Four-Level Topology Store from Spec #13 since 2026-04-25 (commits `30548d4f` + `043507bf` + this default flip). On any `gnn_offline_sample` invocation without explicit cache flags, the system:

1. Reads `/proc/meminfo` for available RAM, allocates 70% to cache (25% L1, 75% L2).
2. Profiles per-node degree (cold start) or reads prior `node_counts.bin` (warm start, deferred to a future phase).
3. Greedy frequency-sort tier assignment: top-K hubs → L1 RAM hash; next-J warm → L2 RAM compact CSR (uint32, ~5× denser than L1); rest → L3 mmap sidecar (Spec #4-B, if `buildTopologySnapshot:true` at projection time) or L4 BPT direct fallback.
4. Streaming populate (single BPT scan, src-monotone, distribute inline by tier — peak transient memory `O(max_degree × 16 B)` ≈ 2 MB even on papers100M).
5. Runtime dispatch: each `get_out_neighbors(v)` enrutes to L1 (~10-20 ns) / L2 (~50-200 ns) / L3 (~5-100 μs mmap) / L4 (~30-100 μs BPT) based on `tier_lookup_[row_idx]`.

User config keys (all optional — defaults are sensible auto-detect):

```gql
CALL gnn_offline_sample('proj', 's', [5, 10], {
    useFourLevelTopologyStore: true,   -- default true since 2026-04-25
    useAdjacencyCache: true,            -- default true; false disables both Spec #11 and Spec #13
    l1CacheMb: 0,                       -- 0 = auto-detect via /proc/meminfo
    l2CacheMb: 0,                       -- 0 = auto-detect
    useL3MmapSidecar: true              -- default true; graceful fallback to L4 if sidecar absent
})
```

D8 validation: `useFourLevelTopologyStore:true && useAdjacencyCache:false` throws (incompatible). Both true is canonical (Spec #13 supersedes Spec #11). Both false skips all caching (legacy fallback to sidecar/BPT direct).

**Why this is the new default**: Spec #13 is strictly better than Spec #11 — matches it for small graphs (everything fits in L1, equivalent to the monolithic hash) and avoids OOM on graphs larger than available RAM (which Spec #11 cannot do). Power-law access distribution means ~5% of nodes (hubs) receive ~70% of sample lookups; keeping those in L1 RAM gives ~3,000-15,000× speedup on the dominant path while letting the cold tail live on disk.

**Files:** `src/gnn/projection/four_level_topology_store.{h,cc}` (build orchestrator + dispatcher + Neighbors::for_each_dst helpers), `topology_frequency_profiler.{h,cc}` (Phase 1), `l1_hash_cache.{h,cc}` (Phase 2), `l2_compact_csr.{h,cc}` (Phase 2), `adj_entry.h` (shared type), `edge_orientation.h` (lightweight enum hoist), `topology_accessor.{h,cc}` (`enable_four_level_store` + early-return dispatch), `sampling_config.h` (4 flags + D8 validate), `gnn_offline_sample_procedure.cc` (parse + propagate).
**Tests:** 4 profiler + 6 L1 + 7 L2 + 13 dispatcher + 5 integration + 8 Spec #11 backwards-compat + 3 tag-dispatch + 6 warm-start = **52 unit tests**. End-to-end byte-identical sample output vs Spec #11 verified via `TopologyAccessorFourLevel.EnableDisable_Roundtrip` BPT-oracle test. cold→warm roundtrip empirically validated on cora_gnn (`node_counts.bin` written + read).
**Bench:** `scripts/bench_four_level_topology.sh` sweeps `{cora_gnn, ogbn-arxiv}` × `{spec13, spec11, caminoD, bpt}`. papers100M validation procedure documented in `docs/research/2026-04-25-spec13-papers100m-procedure.md` (celebi-side action).
**Design record:** ADR 010 (`Partial_Idea/decisions/010_four_level_topology_store.md`), design `docs/superpowers/specs/2026-04-25-four-level-topology-store-design.md`, plan `docs/superpowers/plans/2026-04-25-four-level-topology-store-plan.md`, **Gate E report `docs/research/2026-04-25-gate-e-report.md`**.
**Phase status (2026-04-25):** All 7 phases LANDED. Phase 0 (docs `586deadb`), Phase 1 (`5a1ed29b` + `d2ed5a63`), Phase 2 (`7aae3155` + `b2087925`), Phase 3 (`30548d4f` + `043507bf` + `3a028d1b`), Phase 4 (`a16e81e7` + `ce67ba69`), Phase 5 (`15727edc`), Phase 6 (`9b96df0b` + `2bfad825`), Phase 7 Gate E template (this commit). **Gate E verdict: PASS-PENDING-PAPERS100M** — 5 of 6 acceptance criteria empirically validated; criterion #4 (papers100M sample window ≤ 30 min) pending celebi-side measurement.

**Files:** `src/storage/index/bplus_tree/bpt_leaf_csr_format.{h,cc}` (enum + v3 header), `bplus_tree_leaf_csr.{h,cc}` (reader with sequential cursor cache), `bpt_mem_import.h` (`BPTLeafCSRWriter` bulk-load), `src/graph_models/gql/projection/projection_catalog.{h,cc}` (catalog v1.6 per-projection byte), `src/query/procedure/builtin/project_procedure.cc` (parse `graphStorage`), `src/gnn/projection/topology_accessor.cc` (v3 fast-path).
**Tests:** unit suites `bpt_leaf_csr_format_test`, `bpt_leaf_csr_reader_test`, `bpt_leaf_csr_writer_test`, `bpt_iter_dispatch_test` (extended 3-way), `projection_catalog_v6_test`, `projection_graph_storage_config_test`, `graph_storage_integration_test`; fuzz `bpt_leaf_csr_fuzz_test` (500 K random + 10 K boundary + 1 K tamper-flip under seed `0xC5B8_1234_5678_9ABC`); integration `scripts/test_projection_csr_hybrid.sh` (4-mode golden compare + sidecar supersedence check).
**Benchmark:** `scripts/bench_csr_hybrid.sh` (Gate D harness — size + scan throughput + GNN sampling per dataset × mode).
**Design record:** ADR 008 (`Partial_Idea/decisions/008_csr_hybrid.md`), spec `docs/superpowers/specs/2026-04-25-csr-hybrid-design.md`, plan `docs/superpowers/plans/2026-04-25-csr-hybrid-plan.md`, master plan `docs/superpowers/plans/2026-04-23-projection-compression-stack-plan.md`.

## Claude Code Configuration

This project includes Claude Code configuration:

- **MCP Servers**: `.mcp.json` - C++ semantic analysis via cclsp (C/C++ Language Server Protocol)
- **Hooks**: `.claude/hooks/` - Automatic code validation
- **Skills**: `.claude/skills/` - Specialized tools (architecture-researcher, mdb-patterns, run-plan)

See `.claude/` directory for full configuration details.
