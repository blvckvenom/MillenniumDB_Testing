# ADR-007: Delta + LEB128 Varint Leaf Encoding for GQL Projection B+Trees

**Date:** 2026-04-24
**Status:** Accepted — Gate C passed.
**Supersedes:** None
**Context:** After Spec #3 (`GNN_MINIMAL`) and Spec #4-B (`TopologySnapshot`), the projection B+Tree leaves remain encoded with the v1 redundant-byte-bitset format. On real GNN datasets the bitset's prefix-correlation assumption breaks down: `from_to_edge` on ogbn-arxiv averages ~7.3 B/edge — within 9% of raw 8 B/edge uint64. At papers100M scale (Spec #1 Run 7 baseline 187 GB, Spec #3 reduces to ~81 GB) the leaf bytes still dominate. A leaf-level encoder that exploits sortedness rather than prefix-correlation can reclaim ~80% of the remaining footprint.

---

## Context

### The measurement driving this decision

Pre-Spec-#5 leaf size on `from_to_edge` (Record\<3\>):

| Dataset | Edges | Leaf bytes (BITSET) | Bytes/edge | vs raw 24 B |
|---------|------:|--------------------:|-----------:|:-----------:|
| cora_gnn | 10 K | 368 KB | 36.8 | 153% |
| ogbn-arxiv | 1.2 M | 28 MB | 23.3 | 97% |
| papers100M (projected) | 1.6 B | ~37 GB | 23.0 | 96% |

The bitset format (1-bit-per-byte mark + redundant-byte stripe) was
designed for record types whose high bytes share a common prefix (RDF
triples with shared graph IRI). On graph projections, where Record\<3\>
holds three independent uint64 ids, the bitset rarely flags more than
0-2 bytes per record — yielding compression ratios that approach 1.0
(no compression).

### Root cause: bitset is the wrong primitive for sorted graph data

A B+Tree leaf is **always sorted by key**. Sortedness implies that
adjacent records' fields differ by small numerical deltas, not by
shared prefix bytes. The right primitive is **delta encoding** + a
variable-length integer codec — the same combination used by Lucene,
Parquet, and every column-store on the planet.

### Why Spec #3 + Spec #4-B alone are insufficient

Spec #3 cuts the **number** of indexes (10 → 5 for GNN_MINIMAL). Spec
#4-B accelerates **sampling** (B+Tree O(log N) → CSR O(1)). Neither
shrinks the **per-edge byte cost** of the surviving B+Trees. At
papers100M Spec #3 alone leaves us at ~37 GB just for `from_to_edge`
out of ~81 GB total projection. Spec #5 is the orthogonal compression
axis that unlocks the remaining headroom.

### Why existing knobs don't solve this

The projection builder has four pre-existing surface knobs:

- `MDB_PROJECTION_SORTER=radix` (Spec #1) — sort algorithm, not encoding.
- `MDB_PROJECTION_SERIAL_SCAN=1` (Spec #2) — scan pipeline, not encoding.
- `indexSet: 'GNN_MINIMAL'` (Spec #3) — index set, not encoding.
- `buildTopologySnapshot: true` (Spec #4-B) — sampling sidecar, not B+Tree encoding.

None modify the on-disk byte layout of B+Tree leaves. Spec #5 is
strictly orthogonal and composes with all four.

---

## Decision

Introduce a **page-level v2 leaf format** (`DELTA_VARINT`), opt-in per
projection via the new GQL config key `leafFormat`. The v2 page layout
is:

- 16-byte header (`format_version=2`, `record_width`, `value_count`,
  `next_leaf`, reserved fields).
- Record 0: N full LEB128 varints (unsigned, 1-10 bytes each).
- Records 1..k-1: N **zigzag(delta-vs-previous)** as LEB128 varints.

The `leaf_format` is persisted **per index** as a uint8 in the catalog
v1.5 metadata (added 2026-04-24). The reader (`BptIter`) dispatches on
the catalog byte at open time. The writer is a sibling
(`BPTLeafV2Writer`) wired into the existing bulk-load path so both
the classic external-sort and radix-partition pipelines produce v2
output identically.

In-page lookup is a **linear scan** with a sequential cursor cache
(T5.13b fix in commit `a94c06cf`): `search_index(k)` decodes from
`cached_position` instead of from record 0 when the requested index
follows the previous request, restoring O(k) total decode cost across
a B+Tree iteration. Without this cache, naive linear scan is O(k²)
and lookup throughput regresses 5-10×.

---

## Alternatives considered

### A1 — Single-format catalog-wide migration (no opt-in)

Make v2 the default for all newly built projections; require a one-shot
migrator for pre-existing v1 projections.

**Rejected:** breaks the projection-immutability invariant (Spec #2
I6) by requiring a rewrite path. Forces all callers (test fixtures,
external benchmarks, regression suites) to migrate simultaneously.
Per-index opt-in via catalog byte is operationally less risky and
preserves byte-identical output for existing call sites.

### A2 — Separate `.leaf2` files alongside `.leaf` (file-level format flag)

Ship v2 as a sibling file format readable by a separate `.leaf2`-aware
reader; coexist with v1 `.leaf` files for backwards compat.

**Rejected:** doubles the file count per projection (10 → 20 in `ALL`
preset). Forces every operational tool (backups, file-counters, disk
usage scripts) to learn the dual-extension convention. Per-index
catalog byte is invisible at the filesystem level — operations stay
the same, only the **content** of `.leaf` differs.

### A3 — Prefix-varint instead of LEB128

Use a length-prefixed variant where the leading byte's high bits encode
the total byte count (Google's `prefixvarint` / Apache Parquet's
`UNSIGNED` encoding). Slightly faster decode (1 branch instead of N).

**Rejected:** loses the **stream-decodable** property. LEB128's
continuation-bit-per-byte protocol means a corrupted byte affects only
its own integer — decode resynchronises on the next continuation-bit
boundary. Prefix-varint corruption cascades to the next N integers.
For a fail-safe leaf format used in long-lived projections, the
self-synchronising property of LEB128 is worth the marginal decode cost.

### A4 — Group-varint (4 lengths in 1 byte, then 4 payloads)

Decode 4 varints as a unit; SIMD-friendly. Used by Google's Snappy and
some BigQuery internal formats.

**Rejected:** complex decoder (8 cases per group based on the length
nibble), low return on a workload where decode is not the bottleneck
(T5.13 measurements show decode is 4% of GQL query wall-clock, sort
and B+Tree traversal dominate). Marginal SIMD speedup not justified
against the +200 LOC complexity and increased fuzz surface.

### A5 — Delta-from-record-0 block base (instead of delta-from-previous)

Encode all records 1..k-1 as deltas against record 0 only, not against
their immediate predecessor. Permits parallel block-decode on multi-core.

**Rejected:** worse compression on the page interior. Since records
are sorted, `r[k] - r[0]` is monotonically growing — the deltas at
the page tail can be 1-2 bytes larger than `r[k] - r[k-1]`. Empirical
measurements on synthetic ogbn-arxiv distributions show ~15% higher
total size with delta-from-base. Not worth the parallelism gain on a
4 KB page (decode is < 100 µs serial).

### A6 — Offset index for in-page binary search

Embed a 2-byte offset table at page tail mapping record index → byte
offset, enabling O(log k) in-page lookup instead of linear decode.

**Rejected as premature optimisation, but kept as a future option.**
Initial T5.4 / T5.7 implementation went linear. The performance risk
materialised in T5.11b's first benchmark (read throughput regressed
5-10×) but the root cause was not the linear scan — it was a missing
sequential-access cache. T5.13b's cursor-cache fix (commit `a94c06cf`)
restored O(k) total decode cost. Spec #5-B may revisit the offset table
if heuristic encoding selection demands true O(log k) random access.

---

## Consequences

### Positive

1. **Size reduction on real GNN datasets:** -80% on cora_gnn
   (368 KB → 72 KB), -79% on ogbn-arxiv (60.10 MB → 12.91 MB). Per-index
   breakdown on ogbn-arxiv shows `from_to_edge` 0.195× (28 MB → 5.5 MB)
   and `to_from_edge` 0.256× (28 MB → 7.2 MB).
2. **Pareto-dominant on read throughput:** measured 0.996-1.016× vs
   BITSET on cora_gnn + ogbn-arxiv after the T5.13b cursor-cache fix.
   Gate C threshold was ≤ 1.20×; the result is effectively a tie,
   meaning v2 is strictly better than v1 on the thesis workload.
3. **Composes cleanly with Specs #1-#4:** any combination of
   {classic, radix} sort × {serial, parallel} scan ×
   {ALL, GNN_MINIMAL, READONLY_TRAVERSAL} indexSet ×
   {topology snapshot on/off} produces v2 output when `leafFormat:
   'DELTA_VARINT'` is set. T5.12 golden-compare validates 6 mode
   combinations are byte-identical to their respective baselines.
4. **Opt-in zero regression:** all 347 GQL + 181 MQL + 809 SPARQL
   integration tests remained green across all 16 implementation
   commits with `leafFormat` defaulting to `'BITSET'`.
5. **Per-index granularity in catalog:** sets up Spec #5-B (heuristic
   per-index format selection based on degree distribution) without
   further catalog-version churn.
6. **Fail-safe decode:** bounds-checked LEB128 with typed
   `BPTLeafV2DecodeException` (offset-precise). T5.14 fuzz harness
   validates 100% tamper-flip detection across 500 K random + 1 K
   smoke + 10 K boundary roundtrips.

### Negative

1. **Adds ~8 B per `BPlusTree<N>` instance in RAM** (vtable pointer +
   `leaf_format_` member). Negligible on the small number of B+Tree
   instances per projection (≤ 14 in `ALL`).
2. **Virtual dispatch per page open.** `BPTLeafBase<N>` is the new
   abstract base; v1 and v2 are subclasses. The dispatch cost is
   one indirect call per page-open (~2-4 ns on modern x86), dominated
   by the disk read it follows.
3. **v2 pages are immutable.** Switching a projection's `leafFormat`
   requires `drop_projection` + recreate. Mitigated by projection
   builds being relatively cheap (~3 s for ogbn-arxiv).
4. **Adds shared storage-layer footprint** (`BPlusTree<N>` constructor
   gains a `LeafFormat` parameter; `BPTLeafBase<N>` is a new virtual
   abstract base) that all callers transitively inherit. Default
   behaviour is preserved for every pre-Spec caller via the BITSET
   default and the new `using BPTLeaf<N> = BPTLeafV1<N>` alias —
   external callers see no API change.

### Neutral

1. **Build time is unchanged:** measured ~0.06 s on cora_gnn and
   ~3.1 s on ogbn-arxiv for both formats (sort dominates, leaf
   serialisation is < 5% of wall-clock).
2. **Cross-projection portability:** v2 leaf bytes are identical
   under both classic and radix sort backends. Projections are
   bit-portable across hardware (LEB128 is little-endian-agnostic
   at the integer level; the 16-byte header explicitly mandates LE).
3. **Disk cache behaviour:** v2 pages decode the full leaf into a
   stack-local k×N uint64 buffer once per leaf-open (T5.7), then
   serve all in-page queries from that buffer. Page-cache footprint
   is identical to v1 (still 4 KB per leaf); only the on-disk byte
   distribution changes.

---

## Implementation commits

Chronological by commit on `feature-GNN`:

- `d69d4e1c` — T5.3: `LeafFormat` enum + v2 header struct + helpers.
- `a1fa392c` — T5.4: LEB128 varint codec + `BPTLeafV2DecodeException`.
- `102768b0` — T5.5: zigzag encode/decode helpers + roundtrip tests.
- `42ff7ca5` — T5.6: extract `BPTLeafBase<N>` virtual base; rename
  existing leaf to `BPTLeafV1<N>`; alias `BPTLeaf<N> = BPTLeafV1<N>`.
- `b7d4b8ae` — T5.7: `BPTLeafV2` write path (header + record 0 full
  varints + records 1..k-1 zigzag-delta varints).
- `e4430dfa` — T5.8: `BPTLeafV2` read path with linear `search_index`.
- `7d4ffbd5` — T5.9: `BptIter` dispatches on `LeafFormat` from catalog
  (default BITSET).
- `f7ca44e9` — T5.10: catalog v1.5 with per-index `leaf_format` byte;
  v1.4 → v1.5 read migration (treat as BITSET).
- `65e8c1dd` — T5.11: GQL parser accepts `leafFormat` config key;
  threads `LeafFormat` through `NativeProjectionBuilder`.
- `e6856c49` — T5.11b: integrate `BPTLeafV2Writer` into the bulk-load
  path; `BptIter` v2 read dispatch wired end-to-end.
- `9a1fd264` — T5.12: 6-mode golden compare script
  (`scripts/test_projection_leaffmt.sh`).
- `fdf5eba1` — T5.14: 1 M-iteration fuzz harness
  (`bpt_leaf_v2_fuzz_test`).
- `d2d7e4d0` — UB fix: signed delta subtraction must cast both operands
  to `int64_t` before subtracting (caught by UBSan in T5.14).
- `41140dad` — T5.13: bench harness `scripts/bench_leaffmt.sh`.
- `a94c06cf` — T5.13b: sequential cursor cache in `BPTLeafV2::search_index`
  (O(k²) → O(k) total decode cost across B+Tree iteration). Restored
  read throughput from 5-10× regression to 1.0× parity vs BITSET.
- (T5.15 + T5.16 land after this ADR — see `feature-GNN` head.)

---

## Validation evidence

- **T5.12 golden compare:** 22/22 semantic equality assertions + 24/24
  byte-identical pairs across the 6-mode matrix
  (BITSET/DELTA_VARINT × {classic, radix} × {serial, parallel}) on
  `cora_gnn`.
- **T5.14 fuzz:** 500 K random + 1 K smoke + 10 K boundary roundtrips
  across N ∈ {1, 2, 3}; 100% tamper-flip detection on a single-byte
  payload mutation.
- **T5.13 + T5.13b bench:** size -80% (cora_gnn), -79% (ogbn-arxiv);
  read throughput 0.996-1.016× vs BITSET (Gate C threshold ≤ 1.20×).
- **Regression suites:** 347/347 GQL + 181/181 MQL + 809/809 SPARQL
  integration tests remained green across all 16 implementation
  commits.
- **Unit tests added:** `bpt_leaf_v2_format_test`, `varint_test`,
  `bpt_leaf_v2_writer_test`, `bpt_leaf_v2_reader_test`,
  `bpt_iter_dispatch_test`, `projection_catalog_v5_test`,
  `projection_leaffmt_config_test`, `bpt_leaf_v2_fuzz_test`.

---

## Related ADRs

- ADR-004 (`004_radix_partition_sort.md`): RADIX sort backend.
- ADR-005 (`005_gnn_minimal_indexset.md`): IndexSet preset.
- ADR-006 (`006_topology_snapshot.md`): TopologySnapshot CSR sidecar.

Together, ADR-004 + ADR-005 + ADR-006 + ADR-007 form the **GQL
projection compression stack** for the thesis: parallel sort,
fewer indexes, faster sampling, smaller leaves. They compose
without precedence — any subset is enabled via independent config keys.

---

## Follow-ups

- **Spec #6** (bit-packed uint32 Record\<N\>): when projection-local
  ObjectIds fit in 32 bits (always true at papers100M scale after
  RowMapping densification), pack them as uint32 instead of uint64,
  halving the raw record size and amplifying the v2 compression ratio.
- **Spec #7** (per-page Zstd): apply a generic block-level compressor
  on top of v2 for cold-archival projections; trades CPU for ~50%
  additional space at the cost of decode latency.
- **Spec #8** (CSR-in-B+Tree-leaves hybrid, **thesis-novel**):
  repurpose v2 leaves to store per-page CSR adjacency slices instead
  of Record\<3\> triples. Unifies the B+Tree path with the
  TopologySnapshot sidecar at the page level, eliminating the
  staleness-vector that ADR-006 manages via SHA-256.

**End of ADR.**
