# CSR-in-B+Tree Hybrid Leaves (Spec #8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task with fresh implementer + reviewer subagents per task.

**Goal:** Introduce an opt-in `graphStorage: 'CSR_HYBRID'` mode for GQL projection edge indexes (`from_to_edge`, `to_from_edge`) that replaces the B+Tree's record-oriented leaf format with a CSR-oriented leaf format — grouping consecutive edges by source node, storing `src` implicitly via an in-page offset table, and delta+varint encoding col_idx lists within each entry. This supersedes Spec #4-B's `topology_*.csr` sidecar files (the B+Tree leaves themselves become the CSR). Target: edge-index disk ≤ 0.80× Spec #5 baseline on ogbn-products; sampling throughput matches or exceeds Spec #4-B sidecar; preserves full B+Tree API semantics and composes with Specs #1, #2, #3, #5.

**Architecture:** User-facing opt-in via GQL `graph_project` config key (`graphStorage`). Persisted in projection catalog v1.6 as one `uint8` per projection. At write time the sorter feeds a new `BPTLeafCSRWriter<N>` for edge indexes (grouping by src, emitting v3 pages with a 16-byte header + uint16 offset table + per-entry varint-encoded col_idx lists, handling high-degree hubs via page chaining with continuation markers). At read time, `BPlusTree<N>::open_leaf_page` dispatches 3-way (v1 BITSET / v2 DELTA_VARINT / v3 CSR_HYBRID); v3 pages support `out_neighbors(src)` in O(log srcs_per_page) via offset-table binary search + O(degree) col_idx decode. `TopologyAccessor::Impl` detects CSR_HYBRID at construction and skips Spec #4-B sidecar readers entirely. The sidecar build step is no-op'd under CSR_HYBRID.

**Tech Stack:** C++17, existing LEB128 + zigzag codec (`src/storage/index/bplus_tree/varint.{h,cc}` from Spec #5), MillenniumDB's 4 KB `Page` buffer manager, GoogleTest, catalog v1.6 binary serialization. No new third-party dependencies.

**Spec reference:** `docs/superpowers/specs/2026-04-25-csr-hybrid-design.md` (§§ 1-10)

**Master plan reference:** `docs/superpowers/plans/2026-04-23-projection-compression-stack-plan.md` §8, §16

**Prerequisites met (verified):**
- Spec #3 (`GNN_MINIMAL`) passed Gate A — catalog v1.4 in place, per-index enumeration is solved.
- Spec #4-B (`buildTopologySnapshot`) passed Gate B — sidecar path exists; Spec #8 supersedes it under opt-in.
- Spec #5 (`leafFormat: 'DELTA_VARINT'`) passed Gate C — catalog v1.5, 3-format dispatch scaffolding, varint codec utility, `BPTLeafBase<N>` virtual interface all present.
- `BPTLeafV2<N>` reader-ctor pattern (ReadTag + header validation) provides a copy-able template for `BPTLeafCSR<N>`'s reader constructor.

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `src/storage/index/bplus_tree/bplus_tree_leaf_csr.h` | `BPTLeafCSR<N>` declaration inheriting `BPTLeafBase<N>` |
| `src/storage/index/bplus_tree/bplus_tree_leaf_csr.cc` | Reader impl: offset-table binary search, varint decode, chain follow |
| `src/graph_models/gql/projection/bpt_leaf_csr_writer.h` | `BPTLeafCSRWriter<N>` declaration |
| `src/graph_models/gql/projection/bpt_leaf_csr_writer.cc` | Writer impl: group-by-src, size estimation, overflow, hub chain |
| `src/tests/bpt_leaf_csr_format_test.cc` | 8+ header + flags + static_assert tests |
| `src/tests/bpt_leaf_csr_writer_test.cc` | 18+ writer behaviour tests incl. worked example golden |
| `src/tests/bpt_leaf_csr_reader_test.cc` | 20+ reader + bounds-check + chain tests |
| `src/tests/bpt_iter_csr_dispatch_test.cc` | 10+ tests for 3-way leaf-format dispatch |
| `src/tests/projection_catalog_v1_6_test.cc` | 6+ catalog roundtrip + migration tests |
| `src/tests/bpt_leaf_csr_fuzz_test.cc` | 1M-iteration randomised roundtrip + hub subset + tamper injection |
| `scripts/test_projection_csrhybrid.sh` | 4-mode golden compare on cora_gnn |
| `scripts/bench_csr_hybrid.sh` | 3 datasets × 4 configurations benchmark |
| `docs/research/2026-04-25-csr-hybrid-bench.md` | Benchmark report |
| `docs/research/2026-04-25-gate-d-report.md` | Gate D sign-off |
| `Partial_Idea/decisions/008_csr_hybrid_leaves.md` | ADR-008 (local-only) |

### Files to modify

| Path | Change |
|---|---|
| `src/storage/index/bplus_tree/bpt_leaf_format.h` | Add `LeafFormat::CSR_HYBRID = 3`; new `BPTLeafCSRHeader` struct + `static_assert(sizeof == 16)`; `CSRHybridFlags` constants namespace |
| `src/storage/index/bplus_tree/bpt_leaf_format.cc` | Add CSR_HYBRID to `leaf_format_to_string` / `parse_leaf_format` |
| `src/storage/index/bplus_tree/bplus_tree.h` | Extend `open_leaf_page` dispatch from 2-way to 3-way; add `leaf_format_ == CSR_HYBRID` branch in `BptIter` |
| `src/storage/index/bplus_tree/bplus_tree.cc` | 3-way dispatch; chain-traversal-aware `BptIter::next()` for v3 leaves |
| `src/graph_models/gql/projection/projection_config.h` | Add `GraphStorageMode` enum + `graph_storage` field on `ProjectionBuilderConfig` |
| `src/graph_models/gql/projection/projection_catalog.{h,cc}` | Bump catalog to v1.6 adding per-projection `graph_storage: uint8`; read path defaults v1.5 bytes to BTREE; write path always emits v1.6 |
| `src/graph_models/gql/projection/sorter_dispatch.cc` | When graph_storage=CSR_HYBRID AND index is from_to_edge/to_from_edge, route sorted records to `BPTLeafCSRWriter` instead of `BPTLeafV1Writer`/`BPTLeafV2Writer` |
| `src/graph_models/gql/projection/native_projection_builder.cc` | Skip `build_topology_snapshots_()` when graph_storage=CSR_HYBRID; emit warning when both graphStorage=CSR_HYBRID AND buildTopologySnapshot=true are set |
| `src/query/procedure/builtin/project_procedure.cc` | Parse `graphStorage` config key (STRING) |
| `src/gnn/projection/topology_accessor.cc` | Detect CSR_HYBRID at ctor; make `fwd_csr_` / `rev_csr_` inert (don't attempt to open sidecars); optional T8.9 fast-path shortcut |
| `CMakeLists.txt` | Register 5 new test executables + ctest entries |
| `docs/MillenniumDB.wiki/GQL-Projections.md` | New "Graph storage — `graphStorage` config parameter" section |
| `CLAUDE.md` | New subsection under "GQL Native Projection" documenting `graphStorage` + Spec #4-B supersedence rule |

---

## Task Overview (executable in order; each task is a subagent-driven-development cycle)

| # | Task ID | Subject | Milestone | Depends |
|---|---|---|---|---|
| 1 | T8.1 | Design document (this session) | M0 (done) | — |
| 2 | T8.2 | Implementation plan (this document) | M0 (done) | T8.1 |
| 3 | T8.3 | `LeafFormat::CSR_HYBRID` enum + v3 header struct + `static_assert` + flags namespace | M1 | T8.2 |
| 4 | T8.4 | `BPTLeafCSR<N>` read path — header validate, offset-table binary search, col_idx iteration (no chain support yet) | M2 | T8.3 |
| 5 | T8.5 | `BPTLeafCSRWriter<N>` bulk-load — group-by-src, offset table, varint col_idx, hub overflow into chain | M2 | T8.4 |
| 6 | T8.6 | `BptIter<N>` / `open_leaf_page` 3-way dispatch on catalog+byte | M3 | T8.5 |
| 7 | T8.7 | Catalog v1.6 with per-projection `graph_storage` byte + v1.5 migration | M3 | T8.6 |
| 8 | T8.8 | Plumb `graphStorage` config through builder + GQL parser + sorter_dispatch | M3 | T8.7 |
| 9 | T8.9 | Disable sidecar build under CSR_HYBRID; `TopologyAccessor` detects CSR_HYBRID and skips sidecar readers | M4 | T8.8 |
| 10 | T8.10 | 4-mode golden compare script (BTREE×BITSET, BTREE×DELTA_VARINT, CSR_HYBRID×BITSET, CSR_HYBRID×DELTA_VARINT) | M4 | T8.9 |
| 11 | T8.11 | 1M-iteration fuzz harness for CSR format incl. hub subset + tamper | M5 | T8.5 |
| 12 | T8.12 | Benchmark script `bench_csr_hybrid.sh` — 3 datasets × 4 configs | M5 | T8.10 |
| 13 | T8.13 | Benchmark report + Gate D report | M6 | T8.12 |
| 14 | T8.14 | Wiki + ADR-008 + CLAUDE.md | M6 | T8.13 |
| — | Gate D | Verification ceremony | M7 | All above |

**Milestones (gate-able commit points):**

- M0: Planning complete (design + plan docs).
- M1: Format scaffolding ready in isolation (header, enum, flags).
- M2: Leaf v3 writer + reader unit-tested against golden bytes (no-chain first, then hub chain).
- M3: Catalog + dispatch + GQL surface live; projections can be built with `graphStorage: 'CSR_HYBRID'`.
- M4: Spec #4-B supersedence wired; golden compare green (4 mode combinations semantically identical; no sidecar under CSR_HYBRID).
- M5: Fuzz + benchmark complete.
- M6: Documentation complete.
- M7: Gate D passed → optionally Spec #7 (Zstd wrap) or stack declared complete.

---

## Task Detail

Each task is TDD-compatible. Subagent flow per master plan §23.1: implementer (writes tests first where applicable, then code, then self-reviews) → spec-compliance reviewer → code-quality reviewer → fix loop until both approve → mark completed.

### Task T8.3 — `LeafFormat::CSR_HYBRID` enum + v3 header struct

**Acceptance criteria:**
- `src/storage/index/bplus_tree/bpt_leaf_format.h` adds:
  - `LeafFormat::CSR_HYBRID = 3` to the existing enum.
  - `struct BPTLeafCSRHeader` matching design §3.4 byte layout.
  - `static_assert(sizeof(BPTLeafCSRHeader) == 16)`.
  - `namespace CSRHybridFlags { inline constexpr uint8_t kIsContinuation = 0x01; inline constexpr uint8_t kHasEdgeIds = 0x02; }`.
- `bpt_leaf_format.cc` extends `leaf_format_to_string` / `parse_leaf_format`:
  - `leaf_format_to_string(CSR_HYBRID)` returns `"CSR_HYBRID"`.
  - `parse_leaf_format("CSR_HYBRID")` returns `CSR_HYBRID`; unknown strings raise `std::invalid_argument`.
- 8+ unit tests in `src/tests/bpt_leaf_csr_format_test.cc`:
  - `HeaderSizeIs16Bytes` (runtime confirmation of static_assert).
  - `HeaderRoundtrip_ChainHead` (flags bit 0 = 0).
  - `HeaderRoundtrip_Continuation` (flags bit 0 = 1, min_src_id_low repurposed as chain_head_page_id).
  - `FlagsBitAccessors_IsContinuation_HasEdgeIds`.
  - `ParseString_CSR_HYBRID_Returns3`.
  - `ParseString_CaseSensitive`.
  - `ParseString_Unknown_Raises`.
  - `FormatVersionByteIsThree`.

**Signals of done:** 8/8 tests pass; header struct compiles with no padding on x86_64; ASan/UBSan Debug build clean.

**Commit:** `feat(bpt): add LeafFormat::CSR_HYBRID enum and v3 leaf header struct`

---

### Task T8.4 — `BPTLeafCSR<N>` read path

**Acceptance criteria:**
- `src/storage/index/bplus_tree/bplus_tree_leaf_csr.{h,cc}` define `BPTLeafCSR<N>` inheriting `BPTLeafBase<N>`, with:
  - `ReadTag` disambiguator and reader-mode constructor taking `const char*` backing buffer.
  - Page-open validation per design §5.5 (byte-0 = 3, byte-1 = N, flags, reserved-zero, value_count bounds, offset_table monotonicity and bounds).
  - `NeighborRange out_neighbors(uint64_t src_id)` — O(log value_count) binary search via offset_table + O(degree) col_idx decode. Returns an iterator yielding decoded `(dst, edge_id)` pairs; when the looked-up entry's flags indicate chain continuation (via the header-level chain-head marker), the iterator follows `next_leaf` to gather continuation chunks. First iteration of T8.4 implements the **no-chain** path; chain-follow is stubbed and covered fully by T8.5's writer-integration + T8.11's fuzz.
  - `BPTLeafBase<N>` virtual contract:
    - `get_record(pos)` — linear walk over offset_table + per-entry col_idx decode; pos is a logical index across all records on the page.
    - `search_index(record)` — offset-table binary search on `record.get_key()` (src field), then linear within entry.
    - `set_record`, `update_record` — analogous.
    - `check_range`, `check`, `print`.
    - `insert`, `delete_record`, `update_to_next_leaf` — raise `std::logic_error("CSR_HYBRID leaves are immutable; rebuild projection")` per I6.
  - `BPTLeafCSRDecodeException` extending `std::runtime_error`.
- 20+ tests in `src/tests/bpt_leaf_csr_reader_test.cc` per design §7.1:
  - Basic parse (valid / invalid version / invalid record_width / bad flags / non-zero reserved).
  - `OutNeighbors_*` (single-src-single-dst, single-src-multi-dst, multi-src, src not present → empty range, binary search bounds).
  - `GetRecord_*` (linear advance across entries, across multi-source page).
  - `SearchIndex_*` (first / last / missing src).
  - `BoundsCheckedOffsetTable_Tampered_Raises`.
  - `BoundsCheckedVarint_Tampered_Raises`.
  - `EmptyPage_ValueCount0_Valid`.
  - `HasEdgeIds_DecodesParallel`.
  - `NoEdgeIds_EdgeIdEmptySpan`.
  - `Insert_RaisesImmutability`.
  - `ConcurrentReaders_NoInterference`.

**Signals of done:** 20/20 tests pass; ASan Debug build clean on randomized inputs; worked example from design §3.4 decodes bit-identically to the input triples.

**Commit:** `feat(bpt): implement BPTLeafCSR read path with offset-table binary search`

---

### Task T8.5 — `BPTLeafCSRWriter<N>` bulk-load

**Acceptance criteria:**
- `src/graph_models/gql/projection/bpt_leaf_csr_writer.{h,cc}`:
  - Constructor: `(name, has_edge_ids, BufferManager&)` — opens leaf file, registers with buffer manager analogously to existing BPT leaf writers.
  - `append(const Record<N>& rec)` — pre-sorted records; writer groups by `rec[0]` (src); when src transitions, finalizes the previous entry and stages it in the current page.
  - Size estimation per design §5.3 before appending each entry; page flush + new page if overflow.
  - Hub detection: when a single src's estimated entry size exceeds `4080 - 2` bytes (won't fit even in an empty fresh page), spill into a page chain.
  - Chain spill: write chain head with `(src_id, total_degree, chain_pages)` in header plus chunk_0; subsequent chunks in continuation pages with `flags & kIsContinuation = 1` and `chain_head_page_id` set.
  - `finalize()` — flush last page, commit page count, close buffer manager handle.
  - `bytes_written() const` and `page_count() const` for YIELD stats.
- Golden test: the worked example from design §3.4 (3 sparse sources with has_edge_ids) — when fed through the writer, the emitted page bytes must be byte-identical to the computed layout.
- 18+ tests in `src/tests/bpt_leaf_csr_writer_test.cc` per design §7.1 (listed in full there).

**Signals of done:** 18/18 tests pass; writer produces v3 pages that the T8.4 reader decodes bit-identically; hub-chain test produces N-page chains (N computed from degree × avg-varint-bytes) with continuation headers correctly set.

**Commit:** `feat(projection): implement BPTLeafCSRWriter with hub-chain overflow`

---

### Task T8.6 — `BptIter<N>` / 3-way page-open dispatch

**Acceptance criteria:**
- `src/storage/index/bplus_tree/bplus_tree.h`: `BplusTree<N>::open_leaf_page(Page&, LeafFormat)` extended to dispatch on `LeafFormat::CSR_HYBRID` → construct `BPTLeafCSR<N>` via ReadTag ctor.
- `bplus_tree.cc`: under `leaf_format_ == CSR_HYBRID`, `BptIter<N>::next()` iterates all `(src, dst, edge_id)` triples on the current page (via `BPTLeafCSR::get_record(pos)`) then advances to `next_leaf`. For chain-continuation pages encountered during full-range scan, the iter treats them as transparent — returning their fragment's records in sequence under the chain head's src.
- Cross-check: when catalog says CSR_HYBRID but page byte 0 ≠ 3, raise `BPTLeafCSRDecodeException("catalog/page format mismatch")`. Mirrors Spec #5 §3.6.
- 10+ tests in `src/tests/bpt_iter_csr_dispatch_test.cc`:
  - `Dispatch_CSR_HYBRID_UsesBPTLeafCSR`.
  - `Dispatch_Mismatch_PageByte3_CatalogSays1_Raises` / `_CatalogSays2_Raises`.
  - `BptIter_Range_WorksUnderCSR`.
  - `BptIter_Range_ResultSetMatchesAcrossFormats` (record-level equality BITSET vs DELTA_VARINT vs CSR_HYBRID).
  - `BptIter_PastLastPage_ReturnsNullptr`.
  - `BptIter_SingleRecord_CSR`.
  - `BptIter_EmptyPage_CSR`.
  - `BptIter_Destruct_ClearsUniquePtr` (no leak under ASan).
  - `BptIter_HubChain_IteratesAllEdges`.
  - `BptIter_RangeFilter_TopologyScope_Correct`.

**Signals of done:** Dispatch correct across all three formats; ASan-clean destruction; cross-format range queries return identical record sets; chain traversal yields the right total record count.

**Commit:** `feat(bpt): dispatch leaf readers 3-way on catalog leaf_format with page-byte cross-check`

---

### Task T8.7 — Catalog v1.6 with per-projection `graph_storage`

**Acceptance criteria:**
- `src/graph_models/gql/projection/projection_catalog.{h,cc}`:
  - Bump catalog version 1.5 → 1.6. Magic: `"MDBGQL\0\x06"`.
  - After the v1.5 body (per-index `leaf_format` array from Spec #5), append one `uint8 graph_storage` byte (values: 1 = BTREE, 2 = CSR_HYBRID; 0 and 3+ reserved).
  - Read path:
    - v1.3 / v1.4 / v1.5: default graph_storage = BTREE.
    - v1.6: parse the byte; reject values outside {1, 2}.
  - Write path: always emit v1.6.
- 6+ tests in `src/tests/projection_catalog_v1_6_test.cc`:
  - `CatalogV1_6_Roundtrip_BTree`.
  - `CatalogV1_6_Roundtrip_CSRHybrid`.
  - `CatalogV1_5_ReadAsV1_6_DefaultsBTree`.
  - `CatalogV1_6_TruncatedGraphStorageByte_Rejected`.
  - `CatalogV1_6_InvalidGraphStorageValue_Rejected`.
  - `CatalogV1_6_ExhaustiveFieldCoverage` — all fields set to non-default, roundtrip.

**Signals of done:** Existing v1.5 projections open as BTREE by default; new projections emit v1.6 and report correct graph_storage byte on reopen.

**Commit:** `feat(projection): add catalog v1.6 with per-projection graphStorage byte`

---

### Task T8.8 — Plumb `graphStorage` config through builder + GQL parser + sorter_dispatch

**Acceptance criteria:**
- `src/graph_models/gql/projection/projection_config.h`:
  - `enum class GraphStorageMode : uint8_t { BTREE = 1, CSR_HYBRID = 2 }`.
  - Field `GraphStorageMode graph_storage = GraphStorageMode::BTREE` on `ProjectionBuilderConfig`.
- `src/query/procedure/builtin/project_procedure.cc`:
  - Parse config key `graphStorage` (STRING). Accepts `"BTREE"` / `"CSR_HYBRID"` (case-sensitive). Unknown → `QueryException`. Non-string type → `QueryException`. Default (key absent) = BTREE.
- `src/graph_models/gql/projection/sorter_dispatch.cc`:
  - When `graph_storage == CSR_HYBRID` AND the index being built is `from_to_edge` or `to_from_edge`, route the sorted `Record<3>` stream to `BPTLeafCSRWriter<3>` instead of the usual `BPTLeafV1Writer<3>` / `BPTLeafV2Writer<3>`.
  - Other indexes continue to dispatch on `leaf_format` alone, regardless of `graph_storage`.
- 6+ tests in `src/tests/projection_graphstorage_config_test.cc`:
  - `Config_Absent_DefaultsBTree`.
  - `Config_CSRHybrid_Accepted`.
  - `Config_Unknown_Rejected`.
  - `Config_NonString_Rejected`.
  - `Config_ThreadedThroughToBuilder`.
  - `SorterDispatch_CSRHybrid_Uses_BPTLeafCSRWriter_ForEdgeIndexes` (verified by mock or grep-level check on build output).

**Signals of done:** `CALL graph_project('x', ..., {graphStorage: 'CSR_HYBRID'}) YIELD *` produces a projection whose edge `.leaf` files have byte 0 = 3 and whose catalog records `graph_storage = 2`.

**Commit:** `feat(gql): parse graphStorage config and route edge indexes through CSR writer`

---

### Task T8.9 — Supersede Spec #4-B under CSR_HYBRID

**Acceptance criteria:**
- `src/graph_models/gql/projection/native_projection_builder.cc`:
  - In `build_topology_snapshots_()` (or equivalent hook): early-return if `config_.graph_storage == CSR_HYBRID`.
  - When both `config_.graph_storage == CSR_HYBRID` AND `config_.build_topology_snapshot == true` are set, emit a warning message to the client (e.g. via the logger used for other build-phase warnings) before early-returning. Warning text: `"buildTopologySnapshot is redundant under graphStorage=CSR_HYBRID; sidecar build skipped"`.
  - `topologySnapshotBytes` YIELD = 0 under CSR_HYBRID.
- `src/gnn/projection/topology_accessor.cc`:
  - In `Impl` constructor, detect CSR_HYBRID via `storage_.get_catalog().graph_storage == CSR_HYBRID`.
  - When CSR_HYBRID, do NOT construct `fwd_csr_` / `rev_csr_` `TopologySnapshotReader` instances; leave the members default-initialized (inert, `has_data()` returns false).
  - `try_csr_out_neighbors` / `try_csr_in_neighbors` return `std::nullopt` immediately under CSR_HYBRID; `get_out_neighbors` / `get_in_neighbors` fall through to the B+Tree path, which now serves O(1) slices from v3 leaves.
  - (Optional, deferred-to-T8.9b if time permits) Direct-shortcut fast path: when CSR_HYBRID is detected at ctor, construct a `BPlusTree<3>` handle and cache it; `out_neighbors(v)` calls the B+Tree directory to locate the leaf page, then invokes `BPTLeafCSR::out_neighbors(v)` directly on the pinned page — bypassing `BptIter` overhead for single-src lookups.
- 4+ tests in `src/tests/topology_accessor_csr_hybrid_test.cc`:
  - `CsrHybrid_NoSidecarReadersConstructed` (count `TopologySnapshotReader::open` calls).
  - `CsrHybrid_OutNeighbors_ReturnsCorrectSet` (decode via B+Tree path, verify set equals the raw edge list).
  - `CsrHybrid_InNeighbors_ReturnsCorrectSet`.
  - `CsrHybrid_Sampling_MatchesSidecarSampling` (with fixed RNG seed, sample produced under CSR_HYBRID == sample under BTREE + sidecar for the same dataset).

**Signals of done:** Running `CALL graph_project(... {graphStorage: 'CSR_HYBRID', buildTopologySnapshot: true})` produces a client warning, zero `.csr` files on disk, and a working GNN sampling pipeline.

**Commit:** `feat(projection): skip sidecar and route TopologyAccessor to CSR leaves under CSR_HYBRID`

---

### Task T8.10 — 4-mode golden compare script

**Acceptance criteria:**
- New script `scripts/test_projection_csrhybrid.sh`.
- Builds cora_gnn projection **4 times** covering the matrix `{BTREE, CSR_HYBRID} × {BITSET, DELTA_VARINT}`.
- Assertions:
  - **Record-level equality** for every edge index: `mdb_leaf_dump` decodes each `from_to_edge.leaf` under its appropriate format into a text `(src, dst, edge_id)` list; all 4 lists must be identical after sorting (they should already be sorted if readers are correct).
  - **Non-edge index byte-identity**: `nodes.leaf`, `node_label.leaf`, `label_node.leaf` identical within same leafFormat regardless of graphStorage.
  - **No sidecar under CSR_HYBRID**: runs 3 and 4 produce zero `.csr` files in the projection directory.
  - **Sidecar under BTREE + buildTopologySnapshot=true**: a 5th run with `{graphStorage: 'BTREE', buildTopologySnapshot: true, leafFormat: 'DELTA_VARINT'}` produces `topology_fwd.csr` and `topology_rev.csr`, and its sampled neighbors match run 4's sampled neighbors under a fixed RNG seed (Spec #4-B supersedence determinism).
  - **USE proj + basic scan**: all 5 projections open under `USE <proj>` and `MATCH (n)-[e]->(m) RETURN count(e)` returns the expected count.
- Exit code 0 on full pass, non-zero with diagnostic diff on any failure.

**Signals of done:** Script reproducible on benito_pc in ≤ 5 minutes; CI-friendly.

**Commit:** `test(projection): add 4-mode golden compare for graphStorage and leafFormat`

---

### Task T8.11 — 1M-iteration fuzz harness

**Acceptance criteria:**
- `src/tests/bpt_leaf_csr_fuzz_test.cc` — a GTest binary that:
  - Runs 1,000,000 iterations of randomised encode → decode roundtrip.
  - Each iteration generates `(num_srcs, per-src degree distribution, has_edge_ids)`, builds a v3 page via the writer, reads back via the reader, asserts record-for-record bit equality.
  - Mixture per design §7.2: uniform degree, clustered degree, sparse-only, dense-only.
  - Seed: hardcoded `0xC5B8_1234_5678_9ABC`; a smoke seed `+1` for a second fast run.
- Hub-chain subset (10,000 iterations):
  - Generate one src with degree in `[5000, 30000]`, forcing chain formation.
  - Verify chain length, chain-head metadata, continuation cursor consistency.
- Tamper-injection sub-test (≥ 1,000 iterations):
  - Flip one random bit in the encoded bytes.
  - Assert `BPTLeafCSRDecodeException` is raised OR the decoded records differ from the input (not silent wrong results).
- Runtime budget ≤ 90 seconds on benito_pc.
- On failure, emit `(N, seed, iteration, num_srcs, degrees, bit_flipped)` for reproducibility.

**Signals of done:** 1M iterations complete under both seeds with zero mismatches; tamper sub-test detects 100% of bit flips.

**Commit:** `test(bpt): add 1M-iteration fuzz test for CSR hybrid leaf encoding`

---

### Task T8.12 — Benchmark script `bench_csr_hybrid.sh`

**Acceptance criteria:**
- `scripts/bench_csr_hybrid.sh`.
- For each `(dataset, graphStorage, leafFormat)` in `{cora_gnn, ogbn-arxiv, ogbn-products} × {BTREE, CSR_HYBRID} × {BITSET, DELTA_VARINT}` = 12 runs:
  1. Build topology-only projection with `indexSet: 'GNN_MINIMAL'`.
  2. When `graphStorage == BTREE` AND the config is the "with sidecar" comparator, also set `buildTopologySnapshot: true`; record `topologySnapshotBytes` separately.
  3. Measure build wall-clock + peak RSS.
  4. Measure edge-index `.leaf` bytes (sum of `from_to_edge.leaf + to_from_edge.leaf`) + total projection bytes.
  5. Measure full-range scan wall-clock (`MATCH (n)-[e]->(m) RETURN count(e)`, second invocation timed).
  6. Measure 10 K-seed k-hop sampling wall-clock with fanout `[15, 10]` and fixed RNG seed, using `gnn_offline_sample` or an equivalent probe.
  7. Drop the projection.
- papers100M explicitly skipped (script aborts with clear log if requested).
- Output: CSV to `/tmp/bench_csrhybrid_<epoch>.csv` + markdown summary to stdout.
- Configurable via `DATASETS`, `MODES` env vars.
- Budget ≤ 60 minutes total on benito_pc.

**Signals of done:** Produces complete 12-row CSV; numbers show `CSR_HYBRID × DELTA_VARINT` edge-index bytes ≤ 0.80 × `BTREE × DELTA_VARINT` on ogbn-products; sampling throughput under CSR_HYBRID matches or exceeds `BTREE + sidecar`.

**Commit:** `bench(projection): add bench_csr_hybrid.sh for CSR hybrid size and sampling throughput`

---

### Task T8.13 — Benchmark report + Gate D report

**Acceptance criteria:**
- `docs/research/2026-04-25-csr-hybrid-bench.md`:
  - Methodology (datasets, configurations, measurement tools — mirrors Spec #5's `2026-04-25-leaffmt-bench.md` structure).
  - Full measurement table from T8.12 CSV converted to markdown.
  - Per-dataset size ratio (CSR_HYBRID×DELTA_VARINT / BTREE×DELTA_VARINT).
  - Sampling throughput comparison including vs. Spec #4-B sidecar mode.
  - Comparison vs design §7.4 expected; if outside band, analysis of why.
  - Verdict: accept Spec #8 for Gate D, recommend further action, or re-open design.
- `docs/research/2026-04-25-gate-d-report.md`:
  - Section-by-section checklist against design §10 (the 8-category success criteria).
  - Regression evidence (test counts, suite names).
  - Empirical evidence (CSV summary).
  - Fuzz evidence (iteration count, seeds, hub subset, tamper stats).
  - Spec #4-B supersedence verification (no sidecar files, warning emitted, TopologyAccessor readers inert).
  - Documentation delivery checklist.

**Signals of done:** Both reports committed (local-only per repo convention); Gate D report is the sign-off artefact.

**Commit:** `docs(projection): add CSR hybrid benchmark and Gate D report`

---

### Task T8.14 — Wiki + ADR-008 + CLAUDE.md

**Acceptance criteria:**
- `docs/MillenniumDB.wiki/GQL-Projections.md` gets a new "Graph storage — `graphStorage` config parameter" section:
  - Mirrors existing "Sort backend selector", "Index set selection", "Topology snapshot", "Leaf encoding" sections in structure.
  - Documents: enum values (`BTREE` / `CSR_HYBRID`), default, expected size reduction, sampling throughput improvement, composition with `leafFormat` and `indexSet`, supersedence of `buildTopologySnapshot`, how to migrate.
  - Worked example: a `graph_project` call using `graphStorage: 'CSR_HYBRID'` + `leafFormat: 'DELTA_VARINT'` + `indexSet: 'GNN_MINIMAL'`.
- `Partial_Idea/decisions/008_csr_hybrid_leaves.md` — ADR-008 (local-only):
  - Context (Spec #4-B duplication, Spec #5 per-edge src cost).
  - Decision (CSR-in-leaf for edge indexes; opt-in; supersedes Spec #4-B sidecar under opt-in).
  - Alternatives considered (covers the 12 design decisions D1–D12 summarized).
  - Consequences (positive / negative / neutral).
  - Implementation commits list.
  - Validation evidence.
  - ~220 lines.
- `CLAUDE.md` gets a new subsection "Graph storage — `graphStorage` config parameter (added 2026-04-25, ADR 008)":
  - Mirrors existing "Sort backend selector", "Index set selection", "Topology snapshot", "Leaf encoding" subsections.
  - Lists the 2 preset values, default, empirical size/time numbers from T8.13, Spec #4-B supersedence rule, file references.

**Signals of done:** All three docs committed; ADR-008 local-only per `.gitignore`; CLAUDE.md and wiki changes are tracked.

**Commit:** `docs(projection): document graphStorage in wiki, ADR-008, and CLAUDE.md`

---

## Gate D Protocol

Per master plan §13 + §16:

```
Section 1: Regression verification
  □ ./scripts/run-tests (all 347 GQL + 181 MQL + 809 SPARQL green)
  □ ctest passes (new CSR format / writer / reader / dispatch / catalog v1.6 / fuzz tests)
  □ 375 GNN unit tests green
  □ 73 E2E checks green
  □ 25 gnn_training tests green
  □ scripts/test_projection_radix.sh green (Spec #1 regression)
  □ scripts/test_projection_indexset.sh green (Spec #3 regression)
  □ scripts/bench_topology_snapshot.sh smoke green (Spec #4-B regression)
  □ scripts/test_projection_leaffmt.sh green (Spec #5 regression)
  □ Release + Debug builds clean, zero new warnings

Section 2: Correctness (fuzz + golden compare)
  □ scripts/test_projection_csrhybrid.sh green on all 4 mode combinations + supersedence check
  □ bpt_leaf_csr_fuzz_test: 1,000,000+ iterations under seed 0xC5B8_1234_5678_9ABC, zero mismatches
  □ bpt_leaf_csr_fuzz_test: smoke seed +1, zero mismatches
  □ Hub-chain subset: 10,000+ iterations, zero mismatches
  □ Tamper-injection sub-test: ≥ 1,000 bit flips, 100% detection

Section 3: Empirical validation
  □ bench_csr_hybrid.sh CSV committed
  □ ogbn-products CSR_HYBRID × DELTA_VARINT edge-index bytes ≤ 0.80 × BTREE × DELTA_VARINT baseline
  □ ogbn-products sampling throughput under CSR_HYBRID ≥ Spec #4-B sidecar throughput (±10 %)
  □ Full-range scan wall-clock under CSR_HYBRID ≤ 1.20 × DELTA_VARINT baseline
  □ Build wall-clock degradation ≤ 15 %
  □ Total projection disk under CSR_HYBRID ≤ 0.85 × (Spec #5 + sidecar) baseline on ogbn-products

Section 4: Spec #4-B supersedence
  □ No topology_*.csr files in a projection built under CSR_HYBRID (filesystem scan post-build)
  □ When graphStorage=CSR_HYBRID AND buildTopologySnapshot=true, client receives a warning and topologySnapshotBytes=0
  □ TopologyAccessor under CSR_HYBRID does not construct sidecar readers (counter-based assertion in test)
  □ Sampling output under CSR_HYBRID bit-identical to sampling under Spec #4-B sidecar mode with fixed RNG seed

Section 5: Composition
  □ All 48 combinations (indexSet × sorter × scan × leafFormat × graphStorage) build on cora_gnn
  □ GNN training pipeline E2E under CSR_HYBRID + DELTA_VARINT + GNN_MINIMAL: testAccuracy ≥ 0.77

Section 6: Documentation delivery
  □ Design doc (local)
  □ Plan doc (local)
  □ ADR-008 (local)
  □ GQL-Projections.md "Graph storage" section (tracked)
  □ CLAUDE.md subsection (tracked)
  □ Benchmark report (local) + raw CSV attached
  □ Gate D report (local)

Section 7: Sign-off
  □ All T8.* tasks completed in TaskList
  □ Master plan updated with actual vs projected metrics
  □ Final reviewer approval
```

On Gate D pass, the thesis-relevant phase of the stack is complete. Spec #7 (per-page Zstd) may follow as an additive step wrapping v3 leaves transparently; alternatively, the stack is declared complete at Spec #8 and focus shifts to papers100M × celebi validation (master plan §18).

---

## Risk Mitigations (Implementation Phase)

Per design §9. Key ones for implementation phase:

1. **Writer size estimation drift (R1).** The `est_entry_bytes` formula in §5.3 uses an `AVG_DELTA_VARINT_BYTES ≈ 1.5` constant. If real workloads diverge from this, pages are either under-packed (wasted padding) or over-packed (re-emission). T8.5 must include a "estimation error distribution" sub-test that tracks estimated-vs-actual over 10 K random pages; if the p99 error exceeds ±5%, adjust the constant or add a reserved-slack margin.

2. **Hub chain corruption (R2).** The reader must assert `sum(chunk_count across chain) == chain head's total_degree` before returning to the caller. If this assertion is skipped, silent truncation can happen when `next_leaf` is corrupt mid-chain. T8.4's `ChainCorrupted_Detected_Raises` test is mandatory.

3. **Dispatch misfire (R7).** Catalog is the primary source of truth; page byte is cross-check only. Same rule as Spec #5 §3.6. T8.6's `Dispatch_Mismatch_*` tests enforce this. Add a code comment at the dispatch site.

4. **Sidecar/CSR duplication (R5).** T8.9's integration test `CsrHybrid_NoSidecarReadersConstructed` uses a counter (or mock) to verify that `TopologySnapshotReader::open` is NOT called when CSR_HYBRID is active. This is cheaper than a full filesystem scan and catches regressions at ctor time.

5. **Catalog v1.5 → v1.6 migration (R9).** T8.7 tests every shape: BTREE-only, CSR_HYBRID-only, mixed-per-projection (multiple projections with different modes in the same catalog), v1.5 read as v1.6 (default BTREE), truncated v1.6 byte rejected. No short-cuts.

6. **Immutability violations (I6).** Any caller that reaches `BPTLeafCSR::insert` or `::delete_record` indicates a bug upstream (the projection build code shouldn't emit these calls). The `std::logic_error` raise provides a clear diagnostic.

7. **Subagent pause mitigation (learned from Specs #3 + #4-B + #5):** Every implementer dispatch includes: *"When complete, commit and exit. Do NOT wait for wakeup or pause mid-task."*

8. **Commit hook adherence (learned from Specs #3, #4-B, #5):** No phrasing like "critical", "thesis-critical", count footers ("18/18 tests pass"), AI attribution, "CRITICAL". Describe WHAT changed + WHY, not verification status.

9. **BptIter range-scan regression (R4):** T8.12 includes full-range scan wall-clock. If regression exceeds 20%, first check the sequential-cursor cache pattern (Spec #5 T5.13b) has been applied to `BPTLeafCSR::get_record(pos)` — repeated `pos, pos+1, pos+2, ...` calls must cache the running cursor state across entries.

10. **Hub chain tail conditions (R3):** When a hub chain's last continuation page has `chunk_count < average`, the writer must still emit the page (not elide it). Test: `HubChain_LastPartialChunk_Correct`.

---

## Commit Policy

Per CLAUDE.md:
- One commit per logical fix (usually one per task).
- `<type>(<scope>): <summary>` header + 2-5 line body explaining WHY.
- No AI attribution.
- No count footers, no "CRITICAL", no review-classification labels.
- Add files explicitly by name (never `git add -A`).

Expected commits per task:

- T8.3: `feat(bpt): add LeafFormat::CSR_HYBRID enum and v3 leaf header struct`
- T8.4: `feat(bpt): implement BPTLeafCSR read path with offset-table binary search`
- T8.5: `feat(projection): implement BPTLeafCSRWriter with hub-chain overflow`
- T8.6: `feat(bpt): dispatch leaf readers 3-way on catalog leaf_format with page-byte cross-check`
- T8.7: `feat(projection): add catalog v1.6 with per-projection graphStorage byte`
- T8.8: `feat(gql): parse graphStorage config and route edge indexes through CSR writer`
- T8.9: `feat(projection): skip sidecar and route TopologyAccessor to CSR leaves under CSR_HYBRID`
- T8.10: `test(projection): add 4-mode golden compare for graphStorage and leafFormat`
- T8.11: `test(bpt): add 1M-iteration fuzz test for CSR hybrid leaf encoding`
- T8.12: `bench(projection): add bench_csr_hybrid.sh for CSR hybrid size and sampling throughput`
- T8.13: `docs(projection): add CSR hybrid benchmark and Gate D report`
- T8.14: `docs(projection): document graphStorage in wiki, ADR-008, and CLAUDE.md`
- Gate D: `docs(projection): Gate D sign-off report for Spec 8`

---

**End of Plan.**

On commit/creation of this document and the paired design doc, T8.1 and T8.2 are complete. T8.3 becomes the next executable task (first subagent dispatch).
