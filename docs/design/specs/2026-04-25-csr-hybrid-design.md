# Spec #8 — CSR-in-B+Tree Hybrid Leaves for Projection Edge Indexes

**Date**: 2026-04-25
**Author**: feature-GNN branch (thesis project)
**Status**: Design (awaiting approval → implementation)
**Supersedes**: Spec #4-B (`topology_fwd.csr` / `topology_rev.csr` sidecars) — architecturally absorbed when `graphStorage: 'CSR_HYBRID'` is active; the sidecar build step is skipped and the TopologyAccessor fast-path reads the in-leaf CSR directly.
**Relates to**:
- `2026-04-21-radix-partition-sort-design.md` — Spec #1 (build-time RADIX sort; unchanged)
- `2026-04-21-serialized-scan-pipeline-design.md` — Spec #2 (build-time SERIAL scan; unchanged)
- `2026-04-25-gnn-minimal-indexset-design.md` — Spec #3 (indexSet preset; composed)
- `2026-04-25-topology-snapshot-design.md` — Spec #4-B (sidecar; superseded when CSR_HYBRID active)
- `2026-04-25-delta-varint-leaf-design.md` — Spec #5 (v2 varint encoding; composed inside each CSR entry's col_idx list)
- Master plan: `docs/design/plans/2026-04-23-projection-compression-stack-plan.md` §8, §16
- ADR-006: `Partial_Idea/decisions/006_topology_snapshot.md` (predecessor)
- ADR-007: `Partial_Idea/decisions/007_delta_varint_leaf.md` (composed)
- Prior empirical: `docs/research/2026-04-23-csr-prototype-results.md`, `docs/research/2026-04-25-gate-b-report.md`, `docs/research/2026-04-25-leaffmt-bench.md`

---

## 1. Problem Statement

### 1.1 What Spec #4-B solved and what it left on the table

Spec #4-B introduced optional mmap-backed CSR sidecars (`topology_fwd.csr` and `topology_rev.csr`) that sit next to the B+Tree files of a projection and accelerate GNN sampling from `O(log N + k)` per `out_neighbors(v)` call to `O(1 + k)`. Gate B (docs/research/2026-04-25-gate-b-report.md §2) confirmed the algorithmic speedup as 1.48–1.59× on locally-measurable datasets (cache-resident regime) with an expected 50–500× in the cache-miss regime that only papers100M × celebi exercises. That is enough to make papers100M GNN training operationally feasible — the original thesis objective.

What Spec #4-B left on the table is **duplication and maintenance cost**:

1. **On-disk duplication.** For a projection that enables `buildTopologySnapshot: true`, the `from_to_edge.leaf` B+Tree already stores every edge as a `(src, dst, edge_id)` `Record<3>` in sorted order; the sidecar then stores the same `(src, dst)` pairs again as `(ROW_PTR, COL_IDX)` plus optionally `(EDGE_IDS)` a third time. Gate B measured +65% disk overhead on ogbn-products (7.24 GB projection → +~1 GB sidecar with `has_edge_ids=true`), projected to +7–8% on papers100M.
2. **Staleness coupling.** The sidecar validates against a SHA-256 of the source `.leaf` stored in its header (design §3.3). On a papers100M scale (37 GB `.leaf` under GNN_MINIMAL), the first open pays 40–70 s of hash time; staleness detection catches only the case where the `.leaf` changed out-of-band (impossible under the Spec #2 immutability invariant I6, but defense-in-depth demands it). Good correctness properties, measurable open-time cost.
3. **Two-path TopologyAccessor.** `TopologyAccessor::Impl` (src/gnn/projection/topology_accessor.cc:20-97) holds two `TopologySnapshotReader` members plus the B+Tree path, with a `has_data()`-based runtime branch in every `out_neighbors`/`in_neighbors` call. Correct and well-tested (T4.7 + T4.11 = 18 unit tests), but the double accounting is precisely what a unified storage layer should eliminate.
4. **Build-step coupling.** `native_projection_builder.cc`'s `build_one_topology_snapshot_()` runs as a second pass over the just-built `.leaf` file. T4.18 (commit `7769da7e`) folded that pass into the main scan, bringing overhead down to +1.4% on products — but the code path remains a separate concern.

### 1.2 What Spec #5 did and did not address

Spec #5 introduced the v2 (`DELTA_VARINT`) leaf encoding. Gate C measured compression ratios of 0.196–0.253 on full projections (cora_gnn / ogbn-arxiv / ogbn-products) under GNN_MINIMAL. That is the single largest per-byte compression win in the stack so far.

But Spec #5 is still a **record-oriented** encoding: each `(src, dst, edge_id)` triple is encoded individually (record 0 as full varints, records 1..k-1 as zigzag-delta varints). The `src` field is re-encoded per-edge, even though in a sorted edge stream the same `src` appears as many times as its out-degree. On papers100M with average out-degree 14.4, that is ~14 re-encodings of `src` per distinct source node — each absorbing ~1–3 varint bytes. The delta step mitigates this (consecutive edges from the same `src` have `delta_src = 0`, encoded as 1 byte), but even 1 byte × 1.6 B edges = 1.6 GB of storage devoted to "same src as before."

Spec #5 also preserves `O(log N)` per-neighbor lookup on the read path: directory walk → leaf page → linear-scan through the varint stream for the target record. Spec #4-B's sidecar is what shortens the lookup, not Spec #5.

### 1.3 The thesis-novel opportunity

CSR (Compressed Sparse Row) has two properties that the B+Tree-of-triples format lacks:

1. **Source identity implicit in position.** In a CSR structure `(ROW_PTR[N+1], COL_IDX[M])`, `src = i` is known simply from `ROW_PTR[i] ≤ offset < ROW_PTR[i+1]`. No per-edge `src` storage at all.
2. **O(1) neighbor lookup.** `COL_IDX[ROW_PTR[i] .. ROW_PTR[i+1])` is a contiguous slice. No tree walk, no scan, no key comparison — just arithmetic.

Spec #4-B's sidecar exploits both properties but stores CSR as a **separate file** paralleling the B+Tree. The thesis-novel step is to **merge the two** — replace the B+Tree leaf pages (which today hold `Record<3>` triples, or under Spec #5 hold varint-encoded triples) with leaf pages that hold CSR-formatted adjacency segments directly.

Concretely: the B+Tree directory continues to route a query `find(src = v)` to one specific leaf page. Instead of that leaf holding triples, it holds one or more source nodes' full adjacency lists `(src_id, degree, col_idx_0, col_idx_1, ..., col_idx_{deg-1})` with an in-page offset table providing O(1) lookup of any src's starting offset within the page. The col_idx list itself is delta + varint encoded (composing with Spec #5).

This is the structure Spec #8 builds. It yields three compounding wins:

- **Disk:** No per-edge `src` field (eliminated by position). No sidecar duplication (the leaves ARE the CSR). Col_idx still varint-compressed per Spec #5.
- **Read:** O(1) neighbor lookup within a leaf (same as sidecar) without a separate file's mmap / fd / staleness check / SHA-256.
- **Simplicity:** One storage artifact per edge index (the B+Tree), one staleness gate (the page buffer manager), one TopologyAccessor fast-path (the leaf format dispatch). Spec #4-B's two-reader-member class becomes a one-leaf-format-dispatch class.

### 1.4 Why this is Spec #8 and not Spec #4-C

Spec #4-B ships independently of Spec #8 because Spec #4-B was the "cheap-win first" path: zero B+Tree changes, sidecar as a new file, preserves all existing invariants with a new optional flag. Spec #8 is "the real restructure": the leaf format fundamentally changes for the two edge indexes, a new leaf reader dispatch branch is added, and the sidecar is superseded. Gate B reports validate that the sidecar works; Gate D here reports that the CSR-in-leaf path matches or exceeds the sidecar's behavior without the duplication.

### 1.5 Why papers100M makes this necessary, not optional

At papers100M scale (master plan §1.2, 187 GB baseline → stacked to ~18 GB with Specs #3+#4-B+#5+#6+#7+#8):

- **Without Spec #8**, Spec #4-B's sidecar adds ~7–8% overhead (~6–7 GB on top of a Spec-#5-compressed projection). On a 937 GB celebi disk that's fine, but every GB taken by sidecar duplication is a GB unavailable for a second projection or a larger feature matrix.
- **Without Spec #8**, the Spec #5 varint-encoded leaves still devote ~1.6 GB to "redundant `src` field across consecutive-edge-same-source" storage even after delta=0 encoding.
- **With Spec #8**, both overheads vanish. The leaf IS the CSR, the CSR's row_ptr is encoded implicitly by an in-page offset table whose size is `O(sources_per_page × 2 bytes)` — negligible at scale.

Master plan §1.2 projects Spec #8's impact as `-60%` projection size and `+10×` intra-projection sampling. Spec #8 is the terminal compression step; subsequent Spec #9 (hypothetical BVGraph) would slot on top.

### 1.6 Why existing knobs are insufficient

- `MDB_PROJECTION_SORTER=radix` (Spec #1) — changes sort algorithm, not storage format.
- `MDB_PROJECTION_SERIAL_SCAN=1` (Spec #2) — changes scan memory profile, not storage format.
- `indexSet: 'GNN_MINIMAL'` (Spec #3) — drops indexes, doesn't change how surviving ones store data.
- `buildTopologySnapshot: true` (Spec #4-B) — adds a sidecar; per §1.1 this is what Spec #8 absorbs.
- `leafFormat: 'DELTA_VARINT'` (Spec #5) — per-record compression; per §1.2 this is what Spec #8 composes with (col_idx uses varint inside).

Spec #8 is the first knob that attacks the **storage shape** of edge indexes — unifying what Spec #4-B treated as sidecar-vs-primary and collapsing redundancies that Spec #5 could only partially touch.

---

## 2. Goals and Non-Goals

### 2.1 Goals

| G | Goal | Measurable criterion |
|---|---|---|
| G1 | Further reduce edge-index disk size vs Spec #5 baseline by eliminating the per-edge `src` field | ogbn-products `from_to_edge.leaf` + `to_from_edge.leaf` combined ≤ 0.70 × Spec-#5 `DELTA_VARINT` baseline |
| G2 | Deliver O(1) `out_neighbors(v)` without a separate sidecar file | TopologyAccessor uses only the B+Tree leaf buffer; no `topology_fwd.csr` / `topology_rev.csr` files present when `graphStorage: 'CSR_HYBRID'` is active |
| G3 | Opt-in per-projection with zero default behaviour change | `graphStorage: 'BTREE'` (default) byte-identical to pre-Spec-#8 output; `BTREE` and `CSR_HYBRID` are user-selectable |
| G4 | Preserve B+Tree API semantics (BptIter, search_index, get_range) for edge indexes | All 347 GQL + 181 MQL + 809 SPARQL tests pass under both storage modes |
| G5 | Records round-trip byte-identical across formats when decoded | Golden compare `BITSET` vs `DELTA_VARINT` vs `CSR_HYBRID` yields identical `(src, dst, edge_id)` sets on every edge index (different bytes, same records) |
| G6 | Bounded read-throughput cost on non-sampling queries | Full-range scan of `from_to_edge` wall-clock ≤ 1.20 × `DELTA_VARINT` baseline on ogbn-products |
| G7 | Sampling throughput matches or exceeds Spec #4-B | Seeds/sec on ogbn-products via TopologyAccessor ≥ Spec-#4-B sidecar throughput (Gate D target: ≥ 50× the B+Tree-only path, same as Spec #4-B's Gate B projection) |
| G8 | Compose cleanly with Specs #1, #2, #3, #5 | 48+ configuration combinations build successfully on cora_gnn |
| G9 | Graceful degradation on high-degree hubs | No edge is lost regardless of source node's degree; hub adjacency spans multiple pages when needed |
| G10 | Fail-safe on corruption | Byte-level tampering of an in-page offset table or a varint payload raises a typed decode exception (inheriting Spec #5's `BPTLeafV2DecodeException`-style), not silent wrong results |

### 2.2 Non-goals

- **N1 — Not a change to non-edge indexes.** `nodes`, `node_label`, `label_node`, `edge_label`, `label_edge` remain classic record-style B+Trees. Their leaves follow whatever `leafFormat` is configured (`BITSET` or `DELTA_VARINT`). Only `from_to_edge` and `to_from_edge` get the CSR_HYBRID treatment when enabled.
- **N2 — Not a change to the B+Tree directory.** `.dir` files are unchanged. The directory continues to route queries to the right leaf page based on min-key-per-page; only the leaf payload format changes.
- **N3 — Not a change to the on-disk Page abstraction.** Every leaf is still exactly 4 KB, still accessed through the buffer manager. Multi-page overflow for high-degree hubs is handled via existing leaf chaining.
- **N4 — Not supporting in-place mutation of v3 leaves.** CSR_HYBRID leaves are immutable post-build, inheriting Spec #5's I6 invariant. Updates require projection rebuild.
- **N5 — Not a main-storage change.** Main IMFD/GQL base-storage is untouched. Spec #8 applies exclusively to projections created via `graph_project`.
- **N6 — Not automatic mode selection.** User picks per projection; no runtime heuristic that chooses `BTREE` vs `CSR_HYBRID` based on edge count or degree distribution. A future Spec #8-B could add that.
- **N7 — Not a replacement for Spec #5.** CSR_HYBRID's col_idx lists are delta+varint-encoded, reusing Spec #5's `BPT::varint_encode` / `BPT::zigzag_encode_i64`. The two compose; they are not alternatives.
- **N8 — Not GPU-resident.** Leaves remain mmap'd CPU memory. GPU-side CSR materialization via `torch.sparse_csr_tensor` is a future work item (possible Spec #8-C).

---

## 3. Design Decisions (All Justified)

### 3.1 D1 — Leaf page layout: multiple source nodes per page with in-page offset table (L-B)

**Decision:** Each 4 KB leaf page holds the full adjacency lists of **one or more** consecutive source nodes in sorted `src` order. A small in-page offset table (one `uint16` per src) maps src-ordinal-within-page to its entry's byte offset. An entry consists of `(src_id_varint, degree_varint, col_idx_0_varint, col_idx_1_varint, ..., col_idx_{deg-1}_varint)` where col_idx_0 is a full varint and subsequent col_idx are zigzag(delta) varints against the previous col_idx.

Hub nodes whose adjacency cannot fit in a single page are handled by D2 (page chain with continuation).

**Why L-B and not L-A (one-src-per-page):**

Papers100M has degree statistics `min=0, avg=14.4, median≈8, max≈25K` (from OGB documentation). One-src-per-page would waste enormous space on sparse nodes: a node with degree 8 would occupy 4 KB of disk for ~30 bytes of actual content. At 111 M nodes, that's a theoretical 453 GB just for the sparse-node slack — more than the original unstacked projection. Even with 95th-percentile degree ~100, a single edge's col_idx varint payload is ~3 bytes, so a full page is ≤ 1% utilized for most nodes.

L-B packs sparse nodes densely. For papers100M with ~4 sources per page on average (`total_pages ≈ 1.6 B edges / 300 edges-per-page-avg`), the offset table overhead is `4 sources × 2 bytes = 8 bytes/page` — negligible next to the 4080-byte payload budget.

**Why L-B and not L-C (hybrid micro-pages):**

L-C's two-tier structure (small-sources packed multi-per-page + hub-sources splitting into multiple micro-pages within a 4 KB page) adds substantial complexity: the leaf header must encode where micro-page boundaries fall, the reader must dispatch between "this page holds multi-sources" and "this page holds one source's fragment." The book-keeping is a source of bugs without clear size wins over L-B + D2 (which handles hubs via whole-page chaining). L-B is the conventional CSR-page layout used in GraphBolt and cuGraph.

**Alternatives considered:**

- **A1 (L-A) — One source per page.** Rejected per above (wastes space on sparse nodes).
- **A2 (L-C) — Hybrid micro-pages.** Rejected — complexity without measurable benefit over L-B + D2.
- **A3 — Source-blocks of fixed size (e.g. 64 sources per page).** Rejected: sparse/hub variance means 64 sources could fit comfortably or not at all; adaptive sizing via offset table is simpler.

### 3.2 D2 — High-degree hubs via page chain with continuation (H-A)

**Decision:** When a source node's adjacency exceeds what can fit in a single 4 KB page (after subtracting the 16-byte header and the 2-byte offset-table slot), the source spans **N contiguous leaf pages**. The first page holds `(src_id, total_degree, chain_pages)` in its header plus `chunk_0` of the col_idx list. Subsequent pages are marked `is_continuation = true` in their flags byte and hold `(chunk_k)` with a back-pointer to the chain head's page id.

The B+Tree directory continues to key on min-src-id per page; chain pages 1..N-1 carry the same min-src-id as chain page 0, so they route correctly. The `BPTLeafCSR::out_neighbors(src)` reader, on finding a chain-head header, follows `next_leaf` (the standard leaf chain pointer) through the N-1 continuation pages, concatenating their chunks into the result span.

**Why H-A and not H-B (inline index-to-external):**

H-B stores `(src_id, external_ref)` for hubs, with the full adjacency in a separate overflow area. That introduces a second storage region with its own file, its own fsync ordering, its own staleness gate — precisely the kind of duplication Spec #8 is designed to eliminate. It would also break the "one storage artifact per edge index" simplification claim of G2.

**Why H-A and not H-C (hierarchical CSR):**

H-C puts a mini-B+Tree inside each super-leaf, making the hub's adjacency `O(log k)` rather than `O(1)` to index. This loses the very property (O(1) neighbor access) that motivates the CSR restructure. Chain traversal in H-A is `O(chain_pages)` where chain_pages ≈ `ceil(degree_bytes / 4080)` is typically 1–3 pages even for 25K-degree hubs at 1-byte-avg-per-col_idx.

**Chain-length estimate for papers100M:**

Max degree ~25,000. At typical delta-varint encoding of 1–2 bytes per col_idx after delta encoding, one hub's adjacency is 25K × 1.5 B ≈ 37 KB ≈ 10 pages. 99.9th-percentile degree is ~1,000, fitting in 1–2 pages. Chain traversal is bounded and infrequent.

**Alternatives considered:**

- **A1 (H-B) — Inline index-to-external.** Rejected per above (second storage region).
- **A2 (H-C) — Hierarchical CSR.** Rejected per above (loses O(1) claim).
- **A3 — Refuse to build CSR_HYBRID if any source has degree > threshold.** Rejected: makes the mode unusable for real thesis datasets.

### 3.3 D3 — In-page offset table (R-A)

**Decision:** After the fixed 16-byte header, each v3 leaf contains a `uint16_t` array of length `value_count` (the number of src entries in this page). Entry `i`'s byte offset within the page is `offset_table[i]`. Offsets are relative to the start of the page, so they fit in 2 bytes (page size 4096 < 65536).

The reader's lookup flow:
1. Binary search the src_ids in the page (log₂ value_count comparisons, each reading a varint at `offset_table[mid]`).
2. On match, slice `col_idx_list` starting at `offset_table[matched] + varint(src_id) + varint(degree)` and running `degree` entries.

**Why R-A and not R-B (linear scan of varint entries):**

Spec #5 already established that `≤ 1.20 × read throughput` is the acceptance criterion (design §2.1 G5). Linear scan over a page of `k` src entries would cost `O(k)` varint decodes per lookup — for a page with 10+ sources (typical), that's 10+ varint decodes when we only need 1. Over the 2.2 B neighbor lookups per epoch on papers100M, the `k`-factor dominates. Binary search with offset-indexed random access is the standard move.

**Why uint16_t and not uint32_t:**

Page size 4096 bytes; `uint16_t` can index any byte. Saves 2 bytes per entry. For an average 10-source page, that's 20 bytes / 4080 bytes = 0.5% page-budget saved — small, but compounds over billions of pages.

**Why not a log₂-indexed skip list (every 8th entry indexed):**

Same reasoning as Spec #5 §3.4: the offset table is `value_count × 2 bytes`, typically 20–40 bytes per page — much smaller than 1/8-index's 5 bytes per group (not enough savings). Full index gives O(log k) on the src and simplifies the reader.

**Alternatives considered:**

- **A1 (R-B) — Linear scan.** Rejected per G6.
- **A2 — Offset table at end of page (pre-header order reversed).** Rejected: writer needs to know value_count at flush time anyway to size the table; end-of-page placement complicates overflow checks during append.

### 3.4 D4 — On-disk format per leaf page (v3)

Concrete layout for a multi-source page (non-continuation):

```
Offset  Size                 Field               Notes
──────────────────────────────────────────────────────────────────────────
0       1                    format_version      uint8 = 3 (CSR_HYBRID)
1       1                    record_width        uint8 = N, for defense-in-depth
                                                 (always 3 for edge indexes; cross-check)
2       1                    flags               bit 0 = is_continuation  (0 here)
                                                 bit 1 = has_edge_ids     (1 iff col_idx has parallel edge_id list)
                                                 bits 2..7 reserved, must be 0
3       1                    reserved            uint8 = 0
4       4                    value_count         uint32 LE, number of src entries in this page
8       4                    next_leaf           uint32 LE, page id of next leaf (0 = last)
12      4                    min_src_id_low      uint32 LE, low 32 bits of min_src_id
                                                 (the directory routes on this; stored
                                                  redundantly as a fsck cross-check)
─── end of 16-byte header ───
16      2 × value_count      offset_table        uint16 LE per entry; offset_table[i] is the
                                                 byte offset in [0, 4096) of entry i's start
                                                 (relative to page start)
?       variable             entries             entry i at offset_table[i]:
                                                   varint(src_id_i)
                                                   varint(degree_i)
                                                   [ varint(col_idx_0) — full varint ]
                                                   [ varint(delta_col_idx_1) — zigzag ]
                                                   ...
                                                   [ varint(delta_col_idx_{degree_i - 1}) ]
                                                   [ IF flags bit 1:
                                                     varint(edge_id_0 — full varint)
                                                     varint(delta_edge_id_1 — zigzag)
                                                     ...
                                                     varint(delta_edge_id_{degree_i - 1}) ]
...     padding              zero-filled to PAGE_SIZE = 4096
```

For a continuation page (flags bit 0 = 1):

```
Offset  Size                 Field               Notes
──────────────────────────────────────────────────────────────────────────
0       1                    format_version      uint8 = 3
1       1                    record_width        uint8 = N
2       1                    flags               bit 0 = is_continuation (1 here)
3       1                    reserved            uint8 = 0
4       4                    chunk_count         uint32 LE, number of col_idx entries in this chunk
8       4                    next_leaf           uint32 LE
12      4                    chain_head_page_id  uint32 LE, page id of the chain head
─── end of 16-byte header ───
16      variable             chunk               continuation of the hub's col_idx list:
                                                   varint(first_col_idx_in_chunk — zigzag
                                                   against the last entry of previous chunk)
                                                   varint(delta), varint(delta), ...
                                                 (the running-cursor continues across the
                                                  whole chain; chunk boundaries are invisible
                                                  to the decoded sequence)
                                                 [ IF chain head's flags bit 1 set:
                                                   parallel edge_id chunk follows ]
...     padding              zero-filled
```

**Header invariants for static_assert:**

```cpp
struct BPTLeafCSRHeader {
    uint8_t  format_version;   // == 3
    uint8_t  record_width;     // == 3 for edge indexes
    uint8_t  flags;            // see above
    uint8_t  reserved;         // == 0
    uint32_t value_count;      // or chunk_count for continuation pages
    uint32_t next_leaf;        // 0 = last
    uint32_t min_src_id_low;   // or chain_head_page_id for continuation pages
};
static_assert(sizeof(BPTLeafCSRHeader) == 16, "v3 header must be 16 bytes");
```

16-byte header matches the Spec #5 v2 header size, making the page dispatch site uniform: `switch (byte_0)` routes to v1/v2/v3 readers that each parse their own 16-byte header.

**Why encode `min_src_id_low` in only 32 bits:**

Full ObjectId is 64 bits. Low 32 bits are the row_index after Spec #4-B's ObjectId-mask-stripping convention (commit `47a1bcae`). For the defensive fsck cross-check this is sufficient; the B+Tree directory stores full 64-bit keys for routing. If an install ever uses full 64-bit src_ids the high 32 bits live only in the varint payload of entry 0.

**Worked example — sparse 3-source page, Record<3> (src, dst, edge_id), flags=has_edge_ids:**

Suppose the page covers sources `src = 1000, 1001, 1002` with adjacencies:

```
src = 1000, degree = 2, dsts = [5000, 5003],        edge_ids = [7000, 7002]
src = 1001, degree = 3, dsts = [100, 5001, 5005],   edge_ids = [8000, 8001, 8005]
src = 1002, degree = 1, dsts = [9000],              edge_ids = [9999]
```

Encoding:

- **Header (16 B):** `03 03 02 00 | 03 00 00 00 | 00 00 00 00 | E8 03 00 00`
  - format_version=3, record_width=3, flags=0x02 (has_edge_ids, not continuation), reserved=0
  - value_count=3
  - next_leaf=0
  - min_src_id_low=1000
- **Offset table (2 × 3 = 6 B):** offsets of entries 0, 1, 2 — computed after encoding the entries, written into bytes [16..22).
- **Entry 0 (src=1000, degree=2):**
  - varint(1000) = `0xE8 0x07` (2 bytes)
  - varint(2)    = `0x02`        (1 byte)
  - varint(5000) = `0x88 0x27`   (2 bytes) — first col_idx full
  - zigzag(5003 - 5000) = 6 → varint(6) = `0x06` (1 byte)
  - varint(7000) = `0xD8 0x36`   (2 bytes) — first edge_id full
  - zigzag(7002 - 7000) = 4 → varint(4) = `0x04` (1 byte)
  - Total: 9 bytes
- **Entry 1 (src=1001, degree=3):**
  - varint(1001) = `0xE9 0x07` (2 bytes)  — note: varint-encoded absolutely, not zigzag-delta, because each src entry is self-contained (D5 decision)
  - varint(3)    = `0x03` (1 byte)
  - varint(100)  = `0x64` (1 byte)      — first col_idx full
  - zigzag(5001 - 100) = 9802 → varint(9802) = `0x8A 0x4C` (2 bytes)
  - zigzag(5005 - 5001) = 8 → varint(8) = `0x08` (1 byte)
  - varint(8000) = `0xC0 0x3E` (2 bytes) — first edge_id full
  - zigzag(8001 - 8000) = 2 → varint(2) = `0x02` (1 byte)
  - zigzag(8005 - 8001) = 8 → varint(8) = `0x08` (1 byte)
  - Total: 11 bytes
- **Entry 2 (src=1002, degree=1):**
  - varint(1002) = `0xEA 0x07` (2 bytes)
  - varint(1)    = `0x01` (1 byte)
  - varint(9000) = `0xA8 0x46` (2 bytes)
  - varint(9999) = `0x8F 0x4E` (2 bytes)
  - Total: 7 bytes

**Page total:** 16 (header) + 6 (offset table) + 9 (entry 0) + 11 (entry 1) + 7 (entry 2) = **49 bytes** for 6 edges.

**Per-edge cost:** 49 / 6 = **8.17 B/edge (without edge_ids: ~5.5 B/edge)**. Compare against Spec #5 on ogbn-arxiv: 11.5 MB / 1.166 M edges = ~10.4 B/edge for `from_to_edge.leaf` (including partial page padding and headers). The worked example deliberately has low degree; realistic multi-source pages pack tighter.

### 3.5 D5 — Composition with Spec #5 varint

**Decision:** The col_idx list within each entry uses Spec #5's `BPT::varint_encode` + `BPT::zigzag_encode_i64` codec. First col_idx in an entry is a full varint (unsigned LEB128). Subsequent col_idx are zigzag(delta) against the previous col_idx **within the same entry**.

The delta-chain does NOT cross entry boundaries. A new entry (new src) resets the running cursor: entry 1's first col_idx is a full varint, not a delta against entry 0's last col_idx. This keeps entries self-contained so a reader can jump to any offset in the offset table and decode from there without replaying previous entries.

**Edge_id list (when flags bit 1 is set):** same scheme, parallel to col_idx. First edge_id full; subsequent edge_ids zigzag-delta within the same entry.

**Continuation chunks (D2 hub chain):** the running cursor **does** cross chunk boundaries within a single chain, because a chain-page cluster represents one logical adjacency list. Continuation pages don't reset the cursor.

**Worked example — running delta:**

Within entry src=1000, col_idx=[5000, 5003, 5004, 5100]:

```
varint(5000)           = 0x88 0x27     (2 bytes, full)
zigzag-varint(5003-5000)   = zigzag(3)=6,  varint(6)=0x06     (1 byte)
zigzag-varint(5004-5003)   = zigzag(1)=2,  varint(2)=0x02     (1 byte)
zigzag-varint(5100-5004)   = zigzag(96)=192, varint(192)=0xC0 0x01 (2 bytes)
```

Total for degree-4 col_idx: 6 bytes. Raw uint64×4 = 32 bytes. Ratio 0.19.

**Rationale for within-entry delta only:**

Cross-entry delta would couple entries (entry 1 depends on entry 0's last col_idx). This breaks the offset table's "random access" property — T5.9's analogue for Spec #8. Within-entry only is the standard CSR compression pattern (cf. WebGraph's intra-list deltas).

### 3.6 D6 — Scope: only FROM_TO_EDGE and TO_FROM_EDGE (S-A)

**Decision:** `graphStorage: 'CSR_HYBRID'` applies **only** to the two edge indexes `from_to_edge` and `to_from_edge`. Other indexes (`nodes`, `node_label`, `label_node`, and when GNN_MINIMAL is relaxed, `edge_label`, `label_edge`, etc.) retain their current `leafFormat`-dispatched record layouts (BITSET or DELTA_VARINT).

**Why S-A and not S-B:**

The thesis-critical path is GNN sampling, which reads `from_to_edge` and (for UNDIRECTED) `to_from_edge`. These are the indexes whose (`src`, `dst`, `edge_id`) record structure has a clean CSR interpretation (`src` becomes implicit). Other indexes have different record structures — `node_label` holds `(node, label)` pairs with no obvious CSR-style grouping; `edge_label` holds `(edge, label)`. Applying CSR_HYBRID to them would require either a different CSR semantics ("group by node, list labels") or falling back to BITSET/DELTA_VARINT — the latter is what Spec #8 actually does.

Expanding scope to S-B (all src-keyed indexes) would triple the test surface and the number of paths a corruption fuzzer must cover, for zero thesis benefit. If post-thesis profiling shows `label_node`-style indexes becoming a bottleneck, a future Spec can add CSR-group-by-key for them.

**Alternatives considered:**

- **A1 (S-B) — All indexes whose records form `(key, value)` tuples sortable by key.** Rejected per above.
- **A2 — Add CSR_HYBRID to `edge_label` opportunistically.** Rejected — edge_label's value distribution is very different (labels are low-cardinality; grouping by edge gives per-group size 1 in the unlabelled case). Not a CSR-shaped workload.

### 3.7 D7 — Catalog: per-projection `graphStorage` byte (C-A)

**Decision:** `graphStorage` is a **single byte per projection**, recorded in the catalog, applying uniformly to both `from_to_edge` and `to_from_edge` when they are materialized. Values: `BTREE` (default, backwards compatible) or `CSR_HYBRID`.

Catalog bumps from v1.5 (Spec #5's leaf_format-per-index) to v1.6 with one additional byte per projection between the v1.5 body and any future fields.

The byte is named `graphStorage` (not `edgeStorage`) to preserve future latitude for applying the same dispatch to other storage shapes (a future "quotient-graph" or "hyperedge" store would plug in here).

**Why C-A and not C-B:**

C-B (per-index opt-in, extending Spec #5's leaf_format-per-index pattern) would permit `from_to_edge = CSR_HYBRID` with `to_from_edge = DELTA_VARINT` on the same projection. No coherent workload wants that: UNDIRECTED sampling uses both indexes symmetrically; directed workloads use one but don't care about the other's format. The configuration-space explosion buys nothing observable.

C-A also simplifies the CLI: one flag, one mental model for the user.

**Alternatives considered:**

- **A1 (C-B) — Per-index opt-in.** Rejected per above.
- **A2 — Env variable `MDB_PROJECTION_GRAPH_STORAGE=CSR_HYBRID`.** Rejected per Spec #3's reasoning (env vars are process-level, less expressive than per-projection persistent config).

### 3.8 D8 — Relationship to Spec #4-B (supersedence rules)

**Decision:** When `graphStorage: 'CSR_HYBRID'` is active:

1. **No `topology_fwd.csr` / `topology_rev.csr` sidecar is built.** The `build_topology_snapshots_()` helper in `native_projection_builder.cc` is a no-op under CSR_HYBRID.
2. **If the user explicitly passes `buildTopologySnapshot: true` AND `graphStorage: 'CSR_HYBRID'`**, the builder emits a one-line warning to the client ("buildTopologySnapshot is redundant under graphStorage=CSR_HYBRID; sidecar build skipped") and silently skips the sidecar. No error. The client code path continues normally.
3. **TopologyAccessor::Impl dispatches on leaf format, not sidecar presence.** When the open projection's catalog says `graphStorage: 'CSR_HYBRID'`, the Impl skips the `TopologySnapshotReader::open()` calls entirely (both `fwd_csr_` and `rev_csr_` are inert). `out_neighbors(v)` goes straight to the B+Tree path, which now returns O(1) slices from v3 leaves.
4. **When `graphStorage: 'BTREE'` (default) AND `buildTopologySnapshot: true`**, behavior is exactly Spec #4-B — sidecar is built and used. Spec #8 is a pure addition in this mode.

**Supersedence matrix (from the composability perspective):**

| `graphStorage` | `buildTopologySnapshot` | What happens |
|---|---|---|
| BTREE (default) | false (default) | Pre-Spec-#8 behavior: B+Tree leaves, no sidecar, O(log N) sampling |
| BTREE | true | Spec #4-B behavior: B+Tree leaves + sidecar, O(1) sampling via sidecar |
| CSR_HYBRID | false | Spec #8 behavior: CSR-in-leaf, no sidecar, O(1) sampling via leaves |
| CSR_HYBRID | true | Spec #8 behavior + warning about redundant flag; sidecar NOT built |

**Why supersede rather than coexist:**

If both sidecar and CSR-in-leaf were built, the duplication problem (§1.1) returns. The user also has to choose which TopologyAccessor should use — adds another config knob for no correctness benefit. The warning on conflict makes the supersedence audit-friendly.

**Alternatives considered:**

- **A1 — Error on conflict.** Rejected: friction for users who set both flags by habit (e.g., copy-paste from Spec-#4-B examples) when upgrading to Spec #8. Warning + skip is gentler.
- **A2 — Build sidecar anyway.** Rejected: duplication defeats the architectural claim.

### 3.9 D9 — Build path integration via a third writer

**Decision:** Introduce `BPTLeafCSRWriter<N>` as a third writer alongside `BPTLeafV1Writer` (BITSET) and the implicit v2 writer inside `BPTLeafV2::append_record` (DELTA_VARINT). The writer takes a **sorted stream of `(src, dst, edge_id)` triples** — the same stream that today feeds `BPTLeafV1`/`V2` writers — and produces v3 CSR-hybrid leaf pages.

**Writer algorithm:**

```text
open current_page with empty offset_table, payload cursor = 16 + 2 * max_srcs_per_page (reserve)
current_src = none
current_src_start_offset = ?
current_entry_bytes = 0

for each (src, dst, edge_id) in sorted input:
    if src != current_src:
        if current_src is not none:
            finalize entry: write varint(src_id), varint(degree) at start of entry,
                then varint(col_idx list) then varint(edge_id list)
            append to offset_table
        if would_overflow(page, estimate_new_entry_size(src)):
            # Single-src entry is too large for remaining space?
            if current page has existing entries:
                flush current page (patch offset table, zero-pad)
                open new page
                go to "new src" logic
            else:
                # Start a hub chain: first page holds chain head, subsequent pages are continuations
                spill_into_chain(src, ...)
                continue
        current_src = src
        current_entry_degree = 0
        current_entry_col_idx_buf.reset()
        current_entry_edge_id_buf.reset()

    # Append dst and edge_id to the running entry
    if current_entry_degree == 0:
        current_entry_col_idx_buf.append(varint(dst))
        if has_edge_ids: current_entry_edge_id_buf.append(varint(edge_id))
    else:
        delta_dst = dst - previous_dst   # signed
        current_entry_col_idx_buf.append(varint(zigzag(delta_dst)))
        if has_edge_ids: ...
    current_entry_degree += 1

at EOF:
    finalize last entry + flush last page
```

**Why sorted input invariant:**

The input stream IS the output of the sorter (Spec #1 RADIX or classic). By contract it is in `(src, dst, edge_id)` lex order. The writer can therefore detect src transitions trivially and never needs to re-buffer.

**Integration with `sorter_dispatch.cc`:**

`sorter_dispatch.cc`'s `run_classic` and `run_radix` both emit a sorted iterator. Today they feed a `BPTLeafWriter` chosen by the `leafFormat` branch. Under Spec #8 the dispatch extends: if `graphStorage == CSR_HYBRID` AND the index is `from_to_edge` or `to_from_edge`, the sorted iterator feeds `BPTLeafCSRWriter<3>` instead. Other indexes continue to dispatch on `leafFormat` alone.

**Directory writer interaction (unchanged):**

`BPTDirWriter` receives one `(min_key, page_id)` pair per leaf page. Under CSR_HYBRID, `min_key` for a page is `(min_src_id_in_page, 0, 0)` for `Record<3>`. For continuation pages, the min_key is the chain head's src_id (same as the chain head's entry, since continuation pages hold fragments of a single src's adjacency). This preserves the directory's routing invariant: a query `find(src=v)` locates the chain head page; the reader then traverses the chain.

**Alternatives considered:**

- **A1 — Fold CSR logic into `BPTLeafV2` via a mode flag.** Rejected: the writer algorithm is fundamentally different (group-by-src vs append-record). Separation keeps each class focused.
- **A2 — Emit an intermediate CSR text dump, then load it.** Rejected: two-phase build doubles wall clock.

### 3.10 D10 — Read path integration

**Decision:** Extend Spec #5's leaf-format dispatch from 2-way to 3-way.

Under Spec #5, `BPlusTree<N>::open_leaf_page(Page&, LeafFormat)` dispatches on the configured `LeafFormat` to construct either `BPTLeafV1<N>` or `BPTLeafV2<N>`. Under Spec #8 the dispatch extends to `BPTLeafCSR<N>`:

```cpp
static std::unique_ptr<BPTLeafBase<N>> open_leaf_page(Page& page, LeafFormat fmt) {
    switch (fmt) {
        case LeafFormat::BITSET:       return std::make_unique<BPTLeafV1<N>>(page);
        case LeafFormat::DELTA_VARINT: return std::make_unique<BPTLeafV2<N>>(page.bytes(), BPTLeafV2<N>::ReadTag{});
        case LeafFormat::CSR_HYBRID:   return std::make_unique<BPTLeafCSR<N>>(page.bytes(), BPTLeafCSR<N>::ReadTag{});
    }
}
```

`BPTLeafCSR<N>` extends the `BPTLeafBase<N>` contract with two new CSR-specific methods:

```cpp
// O(1) within this page via in-page offset table binary search.
// Returns an iterator yielding (src=implicit, dst, edge_id) triples for v's adjacency.
// If v spans into continuation pages, the iterator internally follows next_leaf.
NeighborRange out_neighbors(uint64_t src) const;

// Full-range scan. Yields all (src, dst, edge_id) triples in the page in order.
// For BptIter range query compatibility.
std::unique_ptr<RecordIter<N>> scan_all() const;
```

The `BPTLeafBase<N>` `get_record(pos)` and `search_index(record)` methods are implemented on top of these primitives:

- `get_record(pos)` — linear walk through offset_table entries, accumulating `(src, dst, edge_id)` triples by iterating each entry's col_idx list. `pos` is a logical index over the total record count on the page. For `BptIter` support: this method is called during range scans, and its cost is amortized `O(1)` per record (the sequential-access pattern means the running cursor advances to the right entry without restarting).
- `search_index(record)` — binary search on src (via offset_table), then linear scan within the matched entry's col_idx list. `O(log value_count + log degree)` in the good case.

**TopologyAccessor fast-path under CSR_HYBRID:**

Today's `TopologyAccessor::Impl::out_neighbors(ObjectId node_id)` dispatches through `fwd_csr_.has_data()` to either the sidecar path or the B+Tree path. Under Spec #8 the sidecar readers are inert (see D8), so execution always flows through the B+Tree path — but now that path's `BPlusTree<3>::get_range(from=(v,0,0), to=(v,MAX,MAX))` is served by a `BPTLeafCSR` page that delivers O(1) neighbor slices via `out_neighbors(v)`. The same topology_accessor.cc code works under both modes; the performance characteristics adjust automatically based on the leaf format.

**Optimization (T8.9): direct `out_neighbors` shortcut.** For the sampling hot path, `TopologyAccessor::Impl` can detect CSR_HYBRID at construction (by reading the catalog's `graphStorage`) and, when active, bypass `BptIter` entirely in favor of a direct `BPTLeafCSR::out_neighbors(v)` call on the matched leaf page. This avoids the `BptIter` range-iterator overhead for single-src lookups. The B+Tree directory walk is the same either way.

**Alternatives considered:**

- **A1 — Keep BptIter as the single read path.** Simpler but loses ~2× in tight inner loops per T4.17's observation. Acceptable default; T8.9 is optional optimization.
- **A2 — Expose CSR_HYBRID-specific APIs at the projection storage layer and have TopologyAccessor call them directly.** Cleaner separation but more intrusive. Deferred.

### 3.11 D11 — Correctness invariants

The following invariants must hold for Spec #8 to pass Gate D:

1. **I1 — Record equivalence:** For any projection built under `graphStorage: 'CSR_HYBRID'`, the set of `(src, dst, edge_id)` triples obtainable by decoding all pages equals the set obtainable from the same data under `graphStorage: 'BTREE' + leafFormat: 'BITSET'` and under `graphStorage: 'BTREE' + leafFormat: 'DELTA_VARINT'`. Byte representations differ; record content is identical.
2. **I2 — Sort order preservation:** When iterated in leaf-page order, the `(src, dst, edge_id)` triples appear in the same lexicographic order they would under the classic formats.
3. **I3 — Directory routing:** The B+Tree directory's `find(record)` returns the correct leaf page for any record lookup under all three leaf formats.
4. **I4 — No new GQL surface types:** The user sees one new config key (`graphStorage`). No new YIELD fields unless explicitly specified for benchmark-oriented observability. No new query syntax.
5. **I5 — Sampling determinism:** Under a fixed RNG seed, `sample_khop_neighbors(seeds, fanouts)` produces bit-identical `SampledSubgraph` across all three storage modes (inheriting Spec #4-B's T4.11 determinism test matrix).
6. **I6 — Immutability:** v3 leaves are read-only post-build. Any attempt to `insert` or `delete_record` through the `BPTLeafBase<N>` contract on a v3 page raises `std::logic_error("CSR_HYBRID leaves are immutable; rebuild projection")`. Matches Spec #2 I6 and Spec #5 §2.2 non-goal.
7. **I7 — Chain integrity:** When a hub's adjacency spans pages, all chain-continuation pages are reachable via `next_leaf` from the chain head. Corruption of a mid-chain `next_leaf` pointer is detected at read time (reader assertion: chain length × chunk_count reaches exactly `total_degree`).
8. **I8 — Offset table well-formedness:** Every offset in `offset_table[0..value_count)` points to a valid entry-start byte within `[16 + 2*value_count, 4096)`. Monotonicity: `offset_table[i] < offset_table[i+1]` for all i < value_count - 1. Writer enforces; reader cross-checks at page-open and raises on violation.
9. **I9 — No silent wrong results:** Malformed varints, truncated entries, or offset-table bounds violations raise `BPTLeafCSRDecodeException` (extending Spec #5's typed-exception policy), never a silent wrong record.
10. **I10 — Empty projections:** A projection with zero edges produces zero `.leaf` pages for `from_to_edge`/`to_from_edge`. The `BPlusTree` open path handles this gracefully (inherits BITSET/DELTA_VARINT behavior).

### 3.12 D12 — Projected size and performance

**Disk size estimate for papers100M under GNN_MINIMAL + Specs #3+#5+#8:**

- **Current stacked (#3+#5):** from master plan §1.2 → ~66 GB projection (pre-sidecar). Edge indexes dominate: ~55 GB combined.
- **Without per-edge `src` storage:** Under Spec #5 the `src` field (uint64 varint-encoded, delta=0 most of the time) still costs ~1 B/edge × 3.2 B edges (fwd + rev) = ~3.2 GB that Spec #8 eliminates.
- **Without sidecar:** Spec #4-B's sidecar would add ~7 GB on top of Spec #5. Spec #8 eliminates that entirely.
- **In-page offset table overhead:** ~2 B × 4 srcs/page × 400 M pages = 3.2 GB **new cost**. Net: eliminates 3.2 GB (sidecar col_idx + edge_ids) + 3.2 GB (redundant src) but pays back 3.2 GB (offset table). **Net savings: ~3 GB (~5%) on edge indexes + ~7 GB (eliminated sidecar).**
- **Total projected:** ~55 - 3 = ~52 GB edge indexes; projection total ~52 + 11 (non-edge indexes) = **~63 GB** (vs ~66 GB for #3+#5 alone, vs ~73 GB for #3+#5+#4-B).

**Read throughput estimate:**

- **Full-range scan:** Spec #5's v2 already amortizes at O(1) per record. CSR_HYBRID adds one extra indirection (offset table + entry header) per src transition — bounded by `(value_count × per-src-overhead) / total_records_per_page ≈ 3 × 5B / 300 records = 0.05 B overhead per record`. Projected degradation: ≤ 10% (bounded within G6 ≤ 1.20× criterion).
- **Point lookup (sampling hot path):** O(log value_count + degree) per page-local lookup, amortized ~O(1) for typical 3–10 srcs/page. Compare Spec #4-B sidecar's O(1) slice: within a factor of 2–3×. On papers100M where Spec #4-B projects 50–500× over B+Tree-only, CSR_HYBRID is expected in the same 50–500× regime — possibly slightly slower than the bare mmap slice (in-page overhead), offset by eliminated SHA-256 open-time check (40–70 s one-off cost on papers100M that is paid over training lifetime, but Spec #8 eliminates even that).

**How T8.13 bench verifies:**

`scripts/bench_csr_hybrid.sh` (T8.13) builds 3 datasets × 4 configurations (`graphStorage × leafFormat`):

| `graphStorage` | `leafFormat` | Semantic |
|---|---|---|
| BTREE | BITSET | Pre-Spec-#5 baseline |
| BTREE | DELTA_VARINT | Spec #5 baseline (Gate C's) |
| CSR_HYBRID | BITSET | CSR-in-leaf with uncompressed col_idx (sanity check; not a recommended mode) |
| CSR_HYBRID | DELTA_VARINT | Spec #8 target mode (col_idx varint-compressed within each entry) |

For each config: edge-index bytes, full-range scan wall-clock, 10 K-seed sampling throughput. Target: the `CSR_HYBRID × DELTA_VARINT` row matches or beats `BTREE × DELTA_VARINT` on size (by ≥ 20%) and sampling throughput (≥ equal — matching Spec #4-B's sidecar), while beating `BTREE × DELTA_VARINT + sidecar` on total disk (sidecar overhead eliminated).

---

## 4. API Surface

### 4.1 GQL procedure signature (unchanged + new optional key)

```
CALL graph_project(
    name: STRING,
    nodeProjection: ANY,
    relProjection: ANY,
    config: MAP    // adds optional graphStorage
)
YIELD graphName, nodeCount, relCount, projectMillis, topologySnapshotBytes, ...
```

New config key:

- `graphStorage: STRING` — one of `'BTREE'` (default, same as pre-Spec-#8) or `'CSR_HYBRID'`.

No mandatory new YIELD columns. Optional observability (if benchmark clients need it):

- `edgeIndexFormat: STRING` — the resolved storage mode actually used (e.g., `'CSR_HYBRID'`). Useful for scripts to confirm the config was honored.

When `graphStorage: 'CSR_HYBRID'` AND `buildTopologySnapshot: true`, the builder emits a one-line warning (logged to client via ServerResponse warning frame); `topologySnapshotBytes = 0` in YIELD.

### 4.2 C++ types

```cpp
// src/storage/index/bplus_tree/bpt_leaf_format.h — extend existing enum

namespace BPT {

enum class LeafFormat : uint8_t {
    BITSET       = 1,   // v1 — existing redundant-bitset encoding (pre-Spec-#5)
    DELTA_VARINT = 2,   // v2 — delta + LEB128 varint encoding (Spec #5)
    CSR_HYBRID   = 3,   // v3 — CSR-in-B+Tree leaves (Spec #8)
};

struct BPTLeafCSRHeader {
    uint8_t  format_version;      // == 3
    uint8_t  record_width;        // == 3 for edge indexes
    uint8_t  flags;               // bit 0 = is_continuation, bit 1 = has_edge_ids
    uint8_t  reserved;            // == 0
    uint32_t value_count;         // number of src entries (chain head) or chunk_count (continuation)
    uint32_t next_leaf;           // page id of next leaf (0 = last)
    uint32_t min_src_id_low;      // low 32 bits of min src_id (chain head) or chain_head_page_id (continuation)
};
static_assert(sizeof(BPTLeafCSRHeader) == 16);

namespace CSRHybridFlags {
    inline constexpr uint8_t kIsContinuation = 0x01;
    inline constexpr uint8_t kHasEdgeIds     = 0x02;
}

}  // namespace BPT
```

```cpp
// src/storage/index/bplus_tree/bplus_tree_leaf_csr.h (new)

template <std::size_t N>
class BPTLeafCSR : public BPTLeafBase<N> {
public:
    struct ReadTag {};

    // Writer-mode constructor (used by BPTLeafCSRWriter<N>).
    explicit BPTLeafCSR(char* page_bytes, uint32_t next_leaf = 0) noexcept;

    // Reader-mode constructor.
    BPTLeafCSR(const char* page_bytes, ReadTag);

    ~BPTLeafCSR() override = default;

    // --- BPTLeafBase<N> contract ---
    uint32_t      get_value_count() const override;   // total edge-record count (sum of degrees)
    bool          has_next() const override;
    Record<N>     get_record(uint_fast32_t pos) const override;
    void          set_record(uint_fast32_t pos, Record<N>& out) const override;
    void          update_record(uint_fast32_t pos, Record<N>& out) const override;
    uint_fast32_t search_index(const Record<N>& record) const noexcept override;
    bool          check_range(const Record<N>& r) const override;

    std::unique_ptr<BPlusTreeSplit<N>> insert(const Record<N>&, bool& error) override;
    bool delete_record(const Record<N>&) override;
    void update_to_next_leaf() override;
    bool check(std::ostream& os) const override;
    void print(std::ostream& os) const override;

    // --- CSR-specific primitive (D10) ---
    struct NeighborRange { ... };  // iterator over (dst, edge_id) pairs for one src
    NeighborRange out_neighbors(uint64_t src_id) const;

private:
    const char*              read_page_bytes_ = nullptr;
    BPT::BPTLeafCSRHeader    read_header_{};
    const uint16_t*          offset_table_ = nullptr;
    // Writer state (analogous to BPTLeafV2):
    char*                    page_bytes_ = nullptr;
    std::vector<uint8_t>     scratch_;
    std::vector<uint16_t>    writer_offset_table_;
};

class BPTLeafCSRDecodeException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};
```

```cpp
// src/graph_models/gql/projection/bpt_leaf_csr_writer.h (new)

template <std::size_t N>
class BPTLeafCSRWriter {
public:
    BPTLeafCSRWriter(const std::string& name,
                     bool has_edge_ids,
                     BufferManager& buffer_mgr);

    // Append one pre-sorted record. Groups by src internally.
    void append(const Record<N>& rec);

    // Finalize: flush the final page (and any pending hub-chain continuations).
    void finalize();

    // Report statistics for YIELD.
    uint64_t bytes_written() const noexcept;
    uint64_t page_count()    const noexcept;

private:
    void flush_current_page_();
    void start_new_src_(uint64_t src);
    void spill_hub_chain_(uint64_t src, const std::vector<uint64_t>& col_idx_buf,
                          const std::vector<uint64_t>& edge_id_buf);
    // ... internal state ...
};
```

### 4.3 Projection config plumbing

```cpp
// src/graph_models/gql/projection/projection_config.h

enum class GraphStorageMode : uint8_t {
    BTREE       = 1,   // classic B+Tree of Record<3> triples (default)
    CSR_HYBRID  = 2,   // CSR-in-leaf for edge indexes (Spec #8)
};

struct ProjectionBuilderConfig {
    // ... Spec #3 index_set, Spec #4-B build_topology_snapshot,
    //     Spec #5 leaf_format fields ...
    GraphStorageMode graph_storage = GraphStorageMode::BTREE;
};
```

### 4.4 Observable user behaviour

**Creating a CSR_HYBRID projection:**

```cypher
CALL graph_project('gnn_proj', 'Paper', 'CITES', {
    orientation: 'NATURAL',
    indexSet: 'GNN_MINIMAL',
    graphStorage: 'CSR_HYBRID',
    leafFormat: 'DELTA_VARINT',      -- composes: col_idx varint-compressed within entries
    includeFeatures: 'node_features'
}) YIELD graphName, nodeCount, relCount, projectMillis, edgeIndexFormat
RETURN *
```

Result: `from_to_edge.leaf` and `to_from_edge.leaf` use v3 (CSR_HYBRID) layout. The non-edge `.leaf` files (`nodes.leaf`, `node_label.leaf`, `label_node.leaf`) use v2 (DELTA_VARINT) — per the composed `leafFormat` setting. No `topology_fwd.csr` / `topology_rev.csr` files are produced. `edgeIndexFormat = 'CSR_HYBRID'` in YIELD.

**Querying:**

```cypher
USE gnn_proj
MATCH (a)-[r]->(b) RETURN a.id, b.id LIMIT 10
```

Reader sniffs byte 0 of each loaded leaf page. Edge-index pages dispatch to `BPTLeafCSR<3>`; non-edge pages dispatch to `BPTLeafV2<N>` per Spec #5. Returned records are the same as under any other mode combination.

**GNN sampling:**

```cypher
CALL gnn_offline_sample('gnn_proj', 'seeds', [15, 10]) YIELD ...
```

`TopologyAccessor::Impl` dispatches to the B+Tree path (sidecar readers inert under CSR_HYBRID). Per-seed neighbor lookup serves O(1) from `BPTLeafCSR::out_neighbors(src)`, matching Spec #4-B sidecar's performance without the separate mmap file.

**Mixed-mode siblings:**

A user can build `gnn_proj_btree` and `gnn_proj_csr` from the same base data, compare the `from_to_edge.leaf` files directly (different bytes), and confirm semantic equivalence via the T8.10 golden-compare script that decodes both and diffs records.

---

## 5. On-Disk Format (detailed)

### 5.1 Full byte-level layout — multi-source chain-head page

Expanded from §3.4. Total budget: 4096 bytes.

```
[bytes 0..16)      — fixed 16-byte BPTLeafCSRHeader (see §3.4)
[bytes 16..16+2k)  — offset_table: uint16_t[value_count]
                     offset_table[i] = byte offset in [0, 4096) of entry i's first byte
                     (entries always start ≥ 16 + 2k, so low bits of each entry don't collide
                      with header bits)
[bytes 16+2k..P)   — entries, laid out contiguously in src-ascending order:
                     entry i starts at offset_table[i]; contents:
                       varint(src_id_i)        — full unsigned LEB128
                       varint(degree_i)        — full unsigned LEB128
                       varint(col_idx_0)       — full
                       zigzag+varint(col_idx_j - col_idx_{j-1}) for j in [1, degree)
                       [IF flags & kHasEdgeIds:
                         varint(edge_id_0)     — full
                         zigzag+varint(edge_id_j - edge_id_{j-1}) for j in [1, degree) ]
[bytes P..4096)    — zero-padded
```

### 5.2 Full byte-level layout — continuation page

```
[bytes 0..16)      — fixed 16-byte BPTLeafCSRHeader
                     flags & kIsContinuation = 1
                     value_count field repurposed as chunk_count
                     min_src_id_low field repurposed as chain_head_page_id
[bytes 16..P)      — chunk: col_idx continuation
                     First entry: zigzag+varint(first_col_idx - last_col_idx_of_previous_chunk)
                     Subsequent entries: zigzag+varint delta within chunk
                     Total chunk_count entries
                     [IF chain head's flags & kHasEdgeIds:
                       parallel edge_id chunk immediately after col_idx chunk,
                       same length ]
[bytes P..4096)    — zero-padded
```

### 5.3 Writer size estimation (for overflow planning)

Before appending a new src's entry to the current page, the writer estimates the entry's size as:

```
est_entry_bytes(src_id, degree, first_col_idx, has_edge_ids):
    = varint_size(src_id) + varint_size(degree)
    + varint_size(first_col_idx)
    + degree * AVG_DELTA_VARINT_BYTES   # AVG ≈ 1.5 from empirical measurement
    + (has_edge_ids ? varint_size(first_edge_id) + degree * AVG_DELTA_VARINT_BYTES : 0)
```

If `current_page_bytes + 2 (new offset table slot) + est_entry_bytes > 4080`, the writer flushes the current page (zero-padding, header finalize) and opens a new page. For hub detection: if `est_entry_bytes > 4080 - 16 - 2` (won't fit in a fresh empty page), the writer enters hub-chain mode.

Encoding is opportunistic — if the actual encoding turns out shorter than estimated, the trailing padding grows. If longer (rare, with a conservative AVG), a second pass may be needed; writer validates post-encoding and re-emits if needed. An alternative is to reserve slack (AVG × 1.3) upfront; T8.5 benchmarks decide.

### 5.4 Padding and page boundaries

- Every leaf is exactly 4096 bytes.
- Entries are self-contained within one page (chain head) or within one chunk (continuation page); **no entry straddles a chain boundary**.
- Col_idx deltas reset at page boundaries for chain heads (each new src's col_idx_0 is a full varint) and carry-through for chain continuations (the running cursor survives the page boundary).
- After the last byte of content, bytes up to 4096 are zero-filled. Reader must not attempt to decode past `offset_table[value_count - 1]` + that entry's length.

### 5.5 Validation on read

Per-page page-open validation:

1. Byte 0 is `3`. Catalog says CSR_HYBRID; otherwise raise `BPTLeafCSRDecodeException`.
2. Byte 1 is `N` (expected record_width; always 3 for edge indexes).
3. Byte 2 flags — only bits 0 and 1 may be set; bits 2..7 must be zero.
4. Byte 3 (reserved) must be zero.
5. If `flags & kIsContinuation == 0` (chain head):
   - `value_count ∈ [0, max_srcs_per_page]` where `max_srcs_per_page = (4096 - 16) / (2 + MIN_ENTRY_BYTES)`. `MIN_ENTRY_BYTES = 3` (1-byte varint for src + 1-byte varint for degree=0 + 1-byte placeholder).
   - `offset_table[0] ≥ 16 + 2 * value_count` (first entry starts after the offset table).
   - `offset_table[i] < offset_table[i+1]` for all i < value_count - 1 (monotonic).
   - `offset_table[value_count - 1] < 4096` (last entry starts within the page).
6. If `flags & kIsContinuation == 1` (continuation):
   - `value_count` (repurposed as chunk_count) is ≥ 1.
   - `min_src_id_low` (repurposed as chain_head_page_id) is non-zero.
7. Lazy per-entry decode: varint bounds-checked against page end; throws on overflow.

---

## 6. Migration Path

### 6.1 Backwards compatibility

- **Existing v1.5 catalogs** (Spec #5) read under v1.6 code: no `graphStorage` byte present → default `BTREE`. Zero behavior change.
- **Existing `.leaf` files**: reader sniffs byte 0. BITSET pages byte 0 ∈ [0, 150] (value_count LSB); v2 pages byte 0 = 2; v3 pages byte 0 = 3. The dispatch uses catalog as primary source of truth (mirroring Spec #5 §3.6 rule): page byte is cross-check.
- **Pre-Spec-#8 binaries** reading v1.6 catalogs: v1.6-only fields (i.e. `graphStorage` byte) are ignored by v1.5 readers? No — v1.5 readers fail closed with "catalog version 1.6 not supported" (same policy as Spec #3's v1.3→v1.4 and Spec #5's v1.4→v1.5 gates). User must upgrade binary or rebuild projection under `graphStorage: 'BTREE'`.

### 6.2 Forward compatibility

- v1.5-only binaries refuse v1.6 catalogs with clear error message.
- v1.5-only binaries encountering a v3 leaf page (byte 0 = 3): would dispatch to v2 reader (catalog said DELTA_VARINT), which would fail its byte-1 cross-check (record_width ≠ N after misinterpretation) and raise `BPTLeafV2DecodeException`. Clean failure, not silent wrong results.

### 6.3 Composability matrix (Spec #5 + Spec #8)

| `graphStorage` | `leafFormat` | `from_to_edge.leaf` / `to_from_edge.leaf` | other `.leaf` |
|---|---|---|---|
| BTREE | BITSET | v1 bitset | v1 bitset |
| BTREE | DELTA_VARINT | v2 varint records | v2 varint records |
| CSR_HYBRID | BITSET | v3 CSR with raw-uint64 col_idx (unusual; diagnostic mode) | v1 bitset |
| CSR_HYBRID | DELTA_VARINT | v3 CSR with varint col_idx (recommended mode) | v2 varint records |

`CSR_HYBRID × BITSET` exists for golden-compare purposes (verifies CSR structure is correct independently of varint codec). Not recommended for production — loses Spec #5's compression on the col_idx list.

### 6.4 Downgrade path

User who regrets CSR_HYBRID rebuilds with `graphStorage: 'BTREE'`. On-disk bytes differ; query results identical.

### 6.5 Mixed per-index handling

Per §3.7 D7, `graphStorage` is projection-scoped. A future extension (Spec #8-B) could make it per-index. For now, both edge indexes always share the same `graphStorage`.

---

## 7. Test Plan

### 7.1 Unit tests (≥ 60 tests across 5 files)

**`src/tests/bpt_leaf_csr_format_test.cc`** (T8.3) — ~8 tests:
- `HeaderSizeIs16Bytes` — static_assert as test.
- `FormatVersionByteIsThree`.
- `FlagsBitsAccessors` — is_continuation, has_edge_ids.
- `HeaderRoundtrip_ChainHead`.
- `HeaderRoundtrip_Continuation`.
- `BadFormatVersion_Rejected`.
- `BadRecordWidth_Rejected`.
- `ReservedBitsNonZero_Rejected`.

**`src/tests/bpt_leaf_csr_writer_test.cc`** (T8.5) — ~18 tests:
- `EmptyInput_NoPagesEmitted`.
- `SingleSource_SingleDst_ChainHead`.
- `SingleSource_ManyDsts_SinglePage`.
- `SingleSource_HubExceedsPage_ChainsSpawned`.
- `ThreeSparseSources_SinglePage`.
- `ManySparseSources_MultiPage`.
- `OffsetTableMonotonicity`.
- `OffsetTableBoundsWithinPage`.
- `DeltaVarintColIdx_WithinEntry_ResetsAcrossEntries`.
- `HasEdgeIds_FlagPreserved`.
- `NoEdgeIds_ColIdxOnly`.
- `WorkedExample_ThreeSparseSources_ExactBytes` — §3.4 worked example as golden test.
- `ChainContinuation_DeltaCursorCarriesThrough`.
- `ChainContinuation_ChainHeadPageIdSet`.
- `MultiplePagesAdvance_NextLeafPointer`.
- `LastPage_NextLeafZero`.
- `PaddingZeroed`.
- `WriterInvokesOverflowEstimationBeforeAppend`.

**`src/tests/bpt_leaf_csr_reader_test.cc`** (T8.4) — ~20 tests:
- `ParseHeader_ValidPage`.
- `ParseHeader_InvalidVersion_Raises`.
- `OutNeighbors_SingleSrc_ReturnsCorrectDsts`.
- `OutNeighbors_BinarySearch_FindsExactMatch`.
- `OutNeighbors_SrcNotPresent_EmptyRange`.
- `OutNeighbors_HubChain_FollowsContinuations`.
- `GetRecord_LinearAdvance_MatchesInput`.
- `GetRecord_MultiSource_SpansEntries`.
- `SearchIndex_FirstSrc_Pos0`.
- `SearchIndex_LastSrc_ValidPos`.
- `BoundsCheckedOffsetTable_Tampered_Raises`.
- `BoundsCheckedVarint_Tampered_Raises`.
- `EmptyPage_ValueCount0_Valid`.
- `MaxSrcsPerPage_ValueCountUpperBound_Valid`.
- `ContinuationPage_StandaloneParse_Fails` — continuation page parsed without chain context must be recognizable but distinguishable.
- `IsolatedNode_Degree0_EntryOnePage`.
- `HasEdgeIds_DecodesParallel`.
- `NoEdgeIds_EdgeIdEmptySpan`.
- `FullPageRoundTrip_MatchesWriterOutput`.
- `ConcurrentReaders_NoInterference`.

**`src/tests/bpt_iter_csr_dispatch_test.cc`** (T8.6) — ~10 tests:
- `Dispatch_CSR_HYBRID_UsesBPTLeafCSR`.
- `Dispatch_Mismatch_PageByte3_CatalogSays1_Raises`.
- `BptIter_Range_WorksUnderCSR`.
- `BptIter_Range_ResultSetMatchesAcrossFormats` — BITSET records == DELTA_VARINT records == CSR_HYBRID records when decoded.
- `BptIter_PastLastPage_ReturnsNullptr`.
- `BptIter_SingleRecord_CSR`.
- `BptIter_EmptyPage_CSR`.
- `BptIter_Destruct_ClearsUniquePtr` — no leak under ASan.
- `BptIter_HubChain_IteratesAllEdges`.
- `BptIter_RangeFilter_TopologyScope_Correct`.

**`src/tests/projection_catalog_v1_6_test.cc`** (T8.7) — ~6 tests:
- `CatalogV1_6_Roundtrip_BTree`.
- `CatalogV1_6_Roundtrip_CSRHybrid`.
- `CatalogV1_5_ReadAsV1_6_DefaultsBTree`.
- `CatalogV1_6_TruncatedGraphStorageByte_Rejected`.
- `CatalogV1_6_InvalidGraphStorageValue_Rejected`.
- `CatalogV1_6_ExhaustiveFieldCoverage` — all fields set to non-default, roundtrip.

### 7.2 Fuzz testing (T8.11)

`src/tests/bpt_leaf_csr_fuzz_test.cc` — deterministic-seed harness:

For each of 1,000,000 iterations:
1. Generate a random `num_srcs ∈ [1, max_srcs_per_page]`.
2. For each src, generate a random `degree ∈ [0, max_deg_considering_remaining_page_budget]`.
3. Generate random sorted `col_idx[]` (and optionally `edge_id[]`) per src.
4. Feed the stream through `BPTLeafCSRWriter`.
5. Read back via `BPTLeafCSR<3>`.
6. Assert bit-identical to input.

Hub-chain subset (10 K iterations):
- Generate one src with degree in `[5000, 30000]`, forcing chain formation.
- Verify chain length, chain-head metadata, continuation cursor consistency.

Tamper-injection sub-test (1 K iterations):
- Flip one random bit in the encoded bytes.
- Assert either `BPTLeafCSRDecodeException` is raised OR the decoded records differ from the input (not silent wrong results).

Runtime budget: ≤ 90 s on benito_pc.

Seed: hardcoded `0xC5B8_1234_5678_9ABC`; smoke run at `seed + 1`.

### 7.3 Golden compare (T8.10)

`scripts/test_projection_csrhybrid.sh` — builds cora_gnn projection **4 ways**:

| Run | `graphStorage` | `leafFormat` |
|---|---|---|
| 1 | BTREE | BITSET |
| 2 | BTREE | DELTA_VARINT |
| 3 | CSR_HYBRID | BITSET |
| 4 | CSR_HYBRID | DELTA_VARINT |

Assertions:

- **Semantic equality:** `mdb_leaf_dump` (the helper binary from Spec #5 T5.12) reads each edge-index `.leaf` file under its appropriate format and produces a text `(src, dst, edge_id)` list. All 4 lists must be identical.
- **Non-edge index byte-identity:** `nodes.leaf`, `node_label.leaf`, `label_node.leaf` must be byte-identical within same `leafFormat` regardless of `graphStorage`.
- **Edge-index byte differences logged but not asserted:** `from_to_edge.leaf` differs between all 4 runs (different formats by construction).
- **No sidecar under CSR_HYBRID:** runs 3 and 4 must produce zero `.csr` files.
- **Sidecar under BTREE + buildTopologySnapshot=true:** a fifth run with `{graphStorage: 'BTREE', buildTopologySnapshot: true}` verifies the sidecar path still works.
- **USE proj + basic scan:** all 4 projections open under `USE <proj>` and `MATCH (n)-[e]->(m) RETURN count(e)` returns the expected edge count.

### 7.4 Benchmark (T8.12)

`scripts/bench_csr_hybrid.sh` — 3 datasets × 4 configurations = 12 runs.

Measurements per run:
- Edge-index `.leaf` bytes (sum of from_to + to_from).
- Projection build wall-clock.
- Peak RSS during build.
- Full-range scan wall-clock.
- 10 K-seed k-hop sampling wall-clock, fanout [15, 10].
- `topologySnapshotBytes` YIELD (expected 0 under CSR_HYBRID).

Comparison axes:
- `CSR_HYBRID × DELTA_VARINT` vs `BTREE × DELTA_VARINT` (Spec #8 vs Spec #5 alone): disk + sampling.
- `CSR_HYBRID × DELTA_VARINT` vs `BTREE × DELTA_VARINT + buildTopologySnapshot=true` (Spec #8 vs Spec #5 + Spec #4-B): total disk including sidecar + sampling.

Expected numbers on ogbn-products (fanout [15, 10]):

| Config | Edge-idx bytes | Sampling seeds/sec | Total disk |
|---|---|---|---|
| BTREE × BITSET | 2.78 GB (from Spec #5 Gate C) | ~3 K (B+Tree only) | 2.86 GB |
| BTREE × DELTA_VARINT | 730 MB | ~3 K (B+Tree only) | 742 MB |
| BTREE × DELTA_VARINT + sidecar | 730 MB + ~1 GB sidecar | ~150-500 K | 1.74 GB |
| CSR_HYBRID × DELTA_VARINT | ~580 MB (target -20%) | ~150-500 K | ~590 MB |

Output: CSV + markdown → `docs/research/2026-04-25-csr-hybrid-bench.md`.

### 7.5 Regression testing

- 347 GQL tests under each of 4 mode combinations. All green.
- 181 MQL + 809 SPARQL unchanged.
- 375 GNN unit + 73 E2E + 25 gnn_training must pass under `CSR_HYBRID × DELTA_VARINT`.
- `scripts/test_projection_radix.sh` (Spec #1 golden compare) — verify CSR_HYBRID does not break the RADIX/CLASSIC sort backend equivalence.
- `scripts/test_projection_indexset.sh` (Spec #3 regression) — CSR_HYBRID under ALL / GNN_MINIMAL / READONLY_TRAVERSAL presets.
- `scripts/test_projection_leaffmt.sh` (Spec #5 regression) — 6-mode compare extended to include CSR_HYBRID runs.

### 7.6 E2E

End-to-end GNN pipeline on cora_gnn with `graphStorage: 'CSR_HYBRID' + leafFormat: 'DELTA_VARINT'`:
1. `graph_project` with Spec #3 GNN_MINIMAL + CSR_HYBRID + DELTA_VARINT.
2. `gnn_offline_sample(seeds=42, fanouts=[15, 10])` → `gnn_materialize_batches` → `gnn_build_feature_store` → `gnn_train`.
3. Assert `testAccuracy ≥ 0.77` (vs 0.79 baseline).
4. Assert `gnn_offline_sample` output bit-identical to the same call under BTREE × DELTA_VARINT (sampling determinism, T8.10).
5. Assert no `topology_*.csr` files on disk.

---

## 8. Implementation Notes (Forward-Looking, Non-Binding)

### 8.1 File-level change summary

| File | Change | LOC est. |
|---|---|---:|
| `src/storage/index/bplus_tree/bpt_leaf_format.h` | Add `LeafFormat::CSR_HYBRID`, `BPTLeafCSRHeader`, flags namespace | +60 |
| `src/storage/index/bplus_tree/bpt_leaf_format.cc` | CSR_HYBRID serialize/deserialize helpers, string parsing | +50 |
| `src/storage/index/bplus_tree/bplus_tree_leaf_csr.h` | New: `BPTLeafCSR<N>` declaration + `NeighborRange` | +200 |
| `src/storage/index/bplus_tree/bplus_tree_leaf_csr.cc` | Reader impl: offset-table binary search, varint decode, chain follow | +550 |
| `src/storage/index/bplus_tree/bplus_tree.h` | Extend `open_leaf_page` dispatch to 3-way | +30 |
| `src/storage/index/bplus_tree/bplus_tree.cc` | 3-way dispatch, chain-traversal aware `BptIter` | +120 |
| `src/graph_models/gql/projection/bpt_leaf_csr_writer.h` | New writer interface | +100 |
| `src/graph_models/gql/projection/bpt_leaf_csr_writer.cc` | Writer impl: group-by-src, overflow, hub chain | +500 |
| `src/graph_models/gql/projection/sorter_dispatch.cc` | Dispatch edge-index writer on `graph_storage` | +40 |
| `src/graph_models/gql/projection/projection_config.h` | Add `GraphStorageMode` enum + builder field | +30 |
| `src/graph_models/gql/projection/projection_catalog.{h,cc}` | Catalog v1.6 bump + per-projection graphStorage byte | +120 |
| `src/graph_models/gql/projection/native_projection_builder.cc` | Skip sidecar under CSR_HYBRID; warning emission | +60 |
| `src/query/procedure/builtin/project_procedure.cc` | Parse `graphStorage` key | +30 |
| `src/gnn/projection/topology_accessor.cc` | Detect CSR_HYBRID at construction; optional T8.9 fast-path | +60 |
| `src/tests/bpt_leaf_csr_format_test.cc` | 8+ tests | +150 |
| `src/tests/bpt_leaf_csr_writer_test.cc` | 18+ tests | +400 |
| `src/tests/bpt_leaf_csr_reader_test.cc` | 20+ tests | +400 |
| `src/tests/bpt_iter_csr_dispatch_test.cc` | 10+ tests | +250 |
| `src/tests/projection_catalog_v1_6_test.cc` | 6+ tests | +120 |
| `src/tests/bpt_leaf_csr_fuzz_test.cc` | 1M-iteration fuzz + hub subset + tamper | +300 |
| `scripts/test_projection_csrhybrid.sh` | 4-mode golden compare | +350 |
| `scripts/bench_csr_hybrid.sh` | Benchmark harness | +400 |
| `docs/MillenniumDB.wiki/GQL-Projections.md` | New "Graph storage — `graphStorage` config" section | +130 |
| `Partial_Idea/decisions/008_csr_hybrid_leaves.md` | ADR-008 (local-only) | +200 |
| `docs/research/2026-04-25-csr-hybrid-bench.md` | Benchmark report | n/a |
| `docs/research/2026-04-25-gate-d-report.md` | Gate D sign-off | n/a |
| `docs/MillenniumDB.wiki/` | New subsection | +35 |
| `CMakeLists.txt` | Add 5 test executables | +25 |

Total: ~4700 LOC + docs.

### 8.2 Dependencies

- **Varint codec:** already implemented in `src/storage/index/bplus_tree/varint.{h,cc}` (Spec #5 T5.4 + T5.5). Reused unchanged.
- **SHA-256:** not needed for Spec #8 (no sidecar file to stale-check). Removed from Spec #8's dependency surface.
- **CUDA / LibTorch:** none. Spec #8 is pure CPU disk-format work.
- **Catalog v1.5 → v1.6:** extends existing catalog format; serializer bump is mechanical.

### 8.3 Concurrency

- v3 leaves are read-only post-build. Multiple concurrent `TopologyAccessor` instances may share the same pinned page via the buffer manager.
- Writer runs single-threaded per index (no per-page contention; one writer consumes the sorted stream and emits pages sequentially).
- Chain traversal on read side: the reader pins one page at a time via `BufferManager::get_page`; pinning a continuation page happens only when the `out_neighbors` iterator crosses the chain boundary. No lock held across pin acquisitions.

### 8.4 Code anchors (existing)

- `src/storage/index/bplus_tree/bplus_tree.h:104` — `open_leaf_page(Page&, LeafFormat)` — the dispatch site extended to 3-way.
- `src/storage/index/bplus_tree/bplus_tree_leaf_base.h:27` — `BPTLeafBase<N>` virtual interface — BPTLeafCSR inherits.
- `src/storage/index/bplus_tree/bplus_tree_leaf_v2.h:40` — `BPTLeafV2<N>` reference implementation for the reader-constructor pattern (ReadTag, header validation).
- `src/graph_models/gql/projection/sorter_dispatch.cc` — writer dispatch site (today: BITSET vs DELTA_VARINT; extended to include CSR_HYBRID for edge indexes).
- `src/graph_models/gql/projection/native_projection_builder.cc` — sidecar-build hook (to be no-op'd under CSR_HYBRID).
- `src/gnn/projection/topology_accessor.cc:33-47` — `TopologyAccessor::Impl` ctor where sidecar readers are opened; under CSR_HYBRID they become inert.

---

## 9. Risk Register

| # | Risk | Impact | Probability | Mitigation |
|---|---|---|---|---|
| R1 | Writer size estimation (§5.3) is wrong by > 5%, causing frequent page re-emission or overruns | High | Medium | T8.5 includes estimation-vs-actual delta asserts on 10 K random pages; if AVG_DELTA_VARINT_BYTES needs tuning, T8.12 bench provides data. |
| R2 | Hub chain corruption (mid-chain page lost) returns silently-truncated adjacency | High | Low | Reader asserts sum(chunk_count) == total_degree at chain head before returning to caller. Unit test `ChainCorrupted_Detected_Raises`. |
| R3 | In-page offset table overflow: a very-dense page has `value_count > 2040` srcs (2040 × 2 B = 4080 B offset table alone) | Medium | Very Low | Upper bound checked at write: each src contributes ≥ 3 B (varint(src) + varint(0) + padding) + 2 B offset slot, so value_count ≤ 815 per page in the absolute extreme. Writer refuses to exceed 800 srcs/page. |
| R4 | `BptIter`-based range queries over CSR_HYBRID leaves are substantially slower than over DELTA_VARINT (regression > 20%) | Medium | Medium | T8.12 bench gates. Likely remediation: cache cursor across adjacent get_record calls (inherits Spec #5 T5.13b pattern). |
| R5 | Sidecar-skip logic in the builder misfires, producing both v3 leaves AND a sidecar (duplication returns) | High | Low | Integration test `CsrHybridNoSidecar` scans projection dir after build, asserts absence of `topology_*.csr`. |
| R6 | TopologyAccessor fails to detect CSR_HYBRID at construction, leaving sidecar readers attempting to open non-existent files (log noise) | Low | Low | Explicit catalog check before opening readers; sidecar readers are simply not constructed. |
| R7 | Page byte-0 collision: a DELTA_VARINT page's first payload byte coincidentally = 3 | Medium | Low-Med | Dispatch uses catalog as primary source; page byte is cross-check only. Mirrors Spec #5 §3.6 resolution. |
| R8 | Writer-side record_width mismatch: v3 written with N=2 but reader expects N=3 | Medium | Low | Writer only emits CSR_HYBRID for Record<3> edge indexes (S-A scope); other N values are a build-time error. |
| R9 | Catalog v1.6 migration loses `graph_storage` byte on roundtrip | High | Low | T8.7 roundtrip + truncation rejection tests. |
| R10 | Sampling throughput under CSR_HYBRID falls short of Spec #4-B sidecar's numbers | Medium | Low-Med | T8.12 bench gates; T8.9 direct shortcut falls back if general BptIter path is too slow. |

---

## 10. Success Criteria (Gate D sign-off)

Per master plan §13 + §16, Gate D requires:

1. **Regression:**
   - All 347 GQL + 181 MQL + 809 SPARQL tests green under each (graphStorage × leafFormat × indexSet × sorter × scan-mode) combination exercised by `scripts/test_projection_csrhybrid.sh` and `scripts/run-tests`.
   - All 375 GNN unit tests + 73 E2E checks + 25 gnn_training tests green.
   - All ≥ 60 new unit tests green.
   - Release + Debug builds clean, zero new warnings under strict `-Wall -Wextra -Wunused-parameter`.

2. **Correctness (fuzz):**
   - `bpt_leaf_csr_fuzz_test.cc` completes 1,000,000+ random roundtrips with zero mismatches under seed `0xC5B8_1234_5678_9ABC`.
   - Smoke-run seed `+1` also zero mismatches.
   - Hub-chain subset (10 K iterations) zero mismatches.
   - Tamper-injection sub-test: ≥ 1 K bit flips, 100% detection (`BPTLeafCSRDecodeException` OR records-diff-input).

3. **Golden compare (all three formats produce identical records):**
   - `scripts/test_projection_csrhybrid.sh` green across all 4 mode combinations on cora_gnn.
   - Record-level equality: BITSET records == DELTA_VARINT records == CSR_HYBRID records for every edge index.
   - Non-edge indexes byte-identical within same leafFormat regardless of graphStorage.

4. **Empirical (size):**
   - Edge-index bytes under `CSR_HYBRID × DELTA_VARINT` ≤ 0.80 × the Spec-#5 `BTREE × DELTA_VARINT` baseline on ogbn-products.
   - Total projection disk (including any sidecar) under Spec #8 ≤ 0.85 × the Spec #4-B + Spec #5 combined baseline (Spec #4-B's sidecar eliminated).

5. **Empirical (sampling throughput):**
   - 10 K-seed throughput on ogbn-products under CSR_HYBRID ≥ the throughput under Spec #4-B sidecar (±10% tolerance).
   - Full-range scan wall-clock under CSR_HYBRID ≤ 1.20 × Spec #5 DELTA_VARINT baseline.

6. **Supersedence of Spec #4-B:**
   - When `graphStorage: 'CSR_HYBRID'` is set, no `topology_fwd.csr` / `topology_rev.csr` files exist after build (verified by T8.10 script's post-build filesystem scan).
   - When both `graphStorage: 'CSR_HYBRID'` AND `buildTopologySnapshot: true` are set, the server emits a warning message; `topologySnapshotBytes = 0` in YIELD; no sidecar files.
   - TopologyAccessor under CSR_HYBRID does not attempt to open sidecar readers (verified via strace or by a counter in the Impl).
   - GNN sampling output under CSR_HYBRID is bit-identical to sampling under Spec #4-B sidecar mode with the same RNG seed.

7. **Composition:**
   - All combinations {ALL, GNN_MINIMAL, READONLY_TRAVERSAL} × {classic, radix} × {serial, parallel} × {BITSET, DELTA_VARINT} × {BTREE, CSR_HYBRID} build successfully on cora_gnn. That is 3 × 2 × 2 × 2 × 2 = **48 configurations**. All green.
   - GNN training E2E under CSR_HYBRID: testAccuracy ≥ 0.77 on cora_gnn.

8. **Documentation:**
   - This design doc committed (local).
   - Plan doc committed (local).
   - ADR-008 committed (local).
   - `GQL-Projections.md` "Graph storage" section committed (tracked).
   - `docs/MillenniumDB.wiki/` subsection committed (tracked).
   - Benchmark report `docs/research/2026-04-25-csr-hybrid-bench.md` committed (local).
   - Gate D report `docs/research/2026-04-25-gate-d-report.md` committed (local).

Only after all 8 sections pass does the master plan proceed to Spec #7 (per-page Zstd, which wraps CSR_HYBRID leaves transparently) or directly to Gate F if the stack is declared complete at Spec #8.

---

**End of design.**
