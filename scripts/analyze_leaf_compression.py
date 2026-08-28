#!/usr/bin/env python3
"""
Empirical compression analyzer for MillenniumDB B+Tree leaf files.

Parses .leaf files per the format documented in
src/storage/index/bplus_tree/bplus_tree_leaf.{h,cc}:

  Per 4 KB page:
    [value_count: uint32 LE, 4 B]
    [next_leaf:  uint32 LE, 4 B]
    [bitset:     N bytes]           # N = record_width_in_u64; marks byte positions common across ALL records
    [redundant:  popcount(bitset) bytes]  # common byte values stored once
    [records:    value_count × (8*N - popcount(bitset)) bytes]  # only variant bytes per record
    [padding:    remainder of 4096]

Reconstructs full Record<N> values per page, then measures what each of the
following compression techniques WOULD achieve on that same page:

 (1) Current (as-is):        the measured file size.
 (2) Raw uncompressed:       8*N bytes/record (for comparison baseline).
 (3) Prefix-K:               find longest leading byte-sequence shared by all
                             records in the page; store prefix once + suffixes.
 (4) Varint per field:       encode each uint64 field as varint (1-9 bytes).
 (5) Delta+varint:           for sorted leaves, store first record full + each
                             subsequent as field-wise (cur - prev) in varint.
 (6) Packed-CSR analog:      for 3-field records (from, to, edge), treat as
                             adjacency list: store 1 varint per distinct `from`
                             + count + delta-varints for (to, edge) pairs.

Results aggregated across all pages of a file, then across all files.

Usage:
    python3 scripts/analyze_leaf_compression.py <leaf_file> [<leaf_file> ...]
    python3 scripts/analyze_leaf_compression.py --dataset cora_gnn
    python3 scripts/analyze_leaf_compression.py --all
"""

import os
import sys
import struct
import glob
from pathlib import Path
from collections import defaultdict

PAGE_SIZE = 4096
HEADER_SIZE = 8  # value_count(4) + next_leaf(4)
REPO_ROOT = Path(__file__).resolve().parent.parent
DBS_DIR = REPO_ROOT / "data" / "dbs" / "gql"


def popcount_bitset(bitset_bytes: bytes) -> int:
    return sum(bin(b).count("1") for b in bitset_bytes)


def varint_len(value: int) -> int:
    """Length in bytes of varint encoding of a non-negative integer (LEB128)."""
    if value == 0:
        return 1
    n = 0
    while value:
        n += 1
        value >>= 7
    return n


def parse_leaf_page(page: bytes, record_width_u64: int) -> list:
    """Reconstruct full records from one leaf page. Returns list of tuples of uint64."""
    if len(page) < PAGE_SIZE:
        return []
    value_count, next_leaf = struct.unpack_from("<II", page, 0)
    if value_count == 0:
        return []

    n_bytes_per_record = 8 * record_width_u64
    bitset_n_bytes = record_width_u64  # N bytes = 8*N bits = one per byte-position
    bitset = page[HEADER_SIZE : HEADER_SIZE + bitset_n_bytes]
    redundant_count = popcount_bitset(bitset)
    redundant = page[HEADER_SIZE + bitset_n_bytes : HEADER_SIZE + bitset_n_bytes + redundant_count]

    variant_bytes_per_record = n_bytes_per_record - redundant_count
    records_start = HEADER_SIZE + bitset_n_bytes + redundant_count

    # Build mask: for each byte position (0..n_bytes_per_record-1), is it redundant?
    # Bit i of bitset (little-endian within each byte) -> byte position i
    is_redundant = []
    for byte_idx in range(n_bytes_per_record):
        byte_of_bitset = bitset[byte_idx // 8]
        bit_within = byte_idx % 8
        is_redundant.append(bool(byte_of_bitset & (1 << bit_within)))

    # Reconstruct records
    records = []
    for rec_idx in range(value_count):
        full = bytearray(n_bytes_per_record)
        red_pos = 0
        var_pos = 0
        rec_offset = records_start + rec_idx * variant_bytes_per_record
        for byte_idx in range(n_bytes_per_record):
            if is_redundant[byte_idx]:
                full[byte_idx] = redundant[red_pos]
                red_pos += 1
            else:
                full[byte_idx] = page[rec_offset + var_pos]
                var_pos += 1

        # Parse full record as N × uint64 LE
        fields = struct.unpack_from(f"<{record_width_u64}Q", bytes(full), 0)
        records.append(fields)
    return records


def measure_prefix_k(records: list, record_width_u64: int) -> int:
    """Bytes needed to store records with byte-level longest-common-prefix compression per page."""
    if not records:
        return 0
    n_bytes = 8 * record_width_u64
    # Convert all records to their byte representation
    def to_bytes(rec):
        return b"".join(struct.pack("<Q", f) for f in rec)
    rec_bytes = [to_bytes(r) for r in records]

    # Find longest common prefix
    prefix_len = 0
    first = rec_bytes[0]
    for i in range(n_bytes):
        byte = first[i]
        if all(r[i] == byte for r in rec_bytes):
            prefix_len = i + 1
        else:
            break

    suffix_len = n_bytes - prefix_len
    # Storage: prefix stored once + per-record suffix
    # Also 1B for prefix_len + 4B for record count
    return 5 + prefix_len + len(rec_bytes) * suffix_len


def measure_varint(records: list) -> int:
    """Bytes needed to store each field of each record as varint."""
    total = 0
    for rec in records:
        for field in rec:
            total += varint_len(field)
    return total


def measure_delta_varint(records: list) -> int:
    """Bytes: first record full varint + each subsequent as (field-wise delta) varint.
    Leaf is sorted so deltas should often be small."""
    if not records:
        return 0
    total = sum(varint_len(f) for f in records[0])
    for i in range(1, len(records)):
        for j, field in enumerate(records[i]):
            prev = records[i - 1][j]
            # Signed delta (could be negative if field is not monotonic)
            delta = field - prev
            # Zigzag-encode signed integer
            zz = (delta << 1) ^ (delta >> 63)  # Python signed shift works for negatives
            if zz < 0:
                zz = (delta << 1) ^ -(delta < 0)
                # Simpler zigzag via abs + sign bit: use standard zigzag
                zz = (abs(delta) << 1) | (1 if delta < 0 else 0)
            total += varint_len(zz)
    return total


def measure_packed_csr(records: list, record_width_u64: int) -> int:
    """For 3-field records (from, to, edge): group by `from`, emit:
       [from_varint] [count_varint] [per-record: (to_delta, edge_delta) varints].
       Returns total bytes."""
    if record_width_u64 != 3 or not records:
        return measure_delta_varint(records)  # fallback

    total = 0
    i = 0
    n = len(records)
    while i < n:
        f = records[i][0]
        # Count group
        j = i
        while j < n and records[j][0] == f:
            j += 1
        group = records[i:j]
        total += varint_len(f)
        total += varint_len(len(group))
        prev_to, prev_edge = 0, 0
        for rec in group:
            _, to_, edge_ = rec
            dto = to_ - prev_to
            dedge = edge_ - prev_edge
            # Zigzag for possible negative
            zzto = (abs(dto) << 1) | (1 if dto < 0 else 0)
            zzedge = (abs(dedge) << 1) | (1 if dedge < 0 else 0)
            total += varint_len(zzto) + varint_len(zzedge)
            prev_to = to_
            prev_edge = edge_
        i = j
    return total


def analyze_file(path: Path, record_width_u64: int):
    data = path.read_bytes()
    file_size = len(data)
    if file_size == 0 or file_size % PAGE_SIZE != 0:
        return None

    n_pages = file_size // PAGE_SIZE
    total_records = 0
    current_bytes = 0  # bytes actually stored in .leaf (= file_size)
    raw_bytes = 0       # uncompressed: records × 8*N
    prefix_bytes = 0    # prefix-K compression per page
    varint_bytes = 0
    delta_bytes = 0
    csr_bytes = 0
    page_overhead_bytes = 0  # headers + bitsets + redundant

    empty_pages = 0

    for pg_idx in range(n_pages):
        page = data[pg_idx * PAGE_SIZE : (pg_idx + 1) * PAGE_SIZE]
        records = parse_leaf_page(page, record_width_u64)
        if not records:
            empty_pages += 1
            continue
        total_records += len(records)
        raw_bytes += len(records) * 8 * record_width_u64
        prefix_bytes += measure_prefix_k(records, record_width_u64)
        varint_bytes += measure_varint(records)
        delta_bytes += measure_delta_varint(records)
        csr_bytes += measure_packed_csr(records, record_width_u64)
        # Page overhead = HEADER + bitset + redundant
        value_count, _ = struct.unpack_from("<II", page, 0)
        bitset = page[HEADER_SIZE : HEADER_SIZE + record_width_u64]
        page_overhead_bytes += HEADER_SIZE + record_width_u64 + popcount_bitset(bitset)

    if total_records == 0:
        return None

    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "record_width_u64": record_width_u64,
        "file_size": file_size,
        "n_pages_used": n_pages - empty_pages,
        "empty_pages": empty_pages,
        "total_records": total_records,
        "page_overhead_bytes": page_overhead_bytes,
        "current_bytes": file_size,
        "raw_bytes": raw_bytes,
        "prefix_bytes": prefix_bytes,
        "varint_bytes": varint_bytes,
        "delta_bytes": delta_bytes,
        "csr_bytes": csr_bytes,
    }


def record_width_for(index_name: str) -> int:
    """Return N (record width in uint64 units) for a given index name."""
    rec3 = {
        "from_to_edge", "to_from_edge", "edge_from_to", "edge_n1_n2",
        "node_key_value", "key_value_node", "edge_key_value", "key_value_edge",
    }
    rec2 = {
        "edge_direction", "edge_label", "label_edge",
        "node_label", "label_node",
        "equal_d_edge", "equal_u_edge",
    }
    rec1 = {"nodes"}
    stem = index_name.split(".")[0]
    if stem in rec3:
        return 3
    if stem in rec2:
        return 2
    if stem in rec1:
        return 1
    return 3  # conservative default


def print_result(r: dict):
    size_mb = r["file_size"] / (1024 * 1024)
    raw_mb = r["raw_bytes"] / (1024 * 1024)
    bpr_current = r["current_bytes"] / max(r["total_records"], 1)
    bpr_raw = r["raw_bytes"] / max(r["total_records"], 1)
    bpr_prefix = r["prefix_bytes"] / max(r["total_records"], 1)
    bpr_varint = r["varint_bytes"] / max(r["total_records"], 1)
    bpr_delta = r["delta_bytes"] / max(r["total_records"], 1)
    bpr_csr = r["csr_bytes"] / max(r["total_records"], 1)

    def pct(saved, ref):
        if ref == 0:
            return "n/a"
        return f"{100 * (1 - saved / ref):+.1f}%"

    print(f"\n=== {r['file']} (Record<{r['record_width_u64']}>, {r['total_records']:,} records, {size_mb:.2f} MB) ===")
    print(f"  {'Technique':<20s} {'Total bytes':>15s} {'B/record':>10s} {'vs current':>12s}")
    print(f"  {'-' * 60}")
    print(f"  {'current file':<20s} {r['current_bytes']:>15,} {bpr_current:>10.2f}  {'(baseline)':>12s}")
    print(f"  {'raw (8*N*records)':<20s} {r['raw_bytes']:>15,} {bpr_raw:>10.2f} {pct(r['raw_bytes'], r['current_bytes']):>13s}")
    print(f"  {'prefix-K per-page':<20s} {r['prefix_bytes']:>15,} {bpr_prefix:>10.2f} {pct(r['prefix_bytes'], r['current_bytes']):>13s}")
    print(f"  {'varint fields':<20s} {r['varint_bytes']:>15,} {bpr_varint:>10.2f} {pct(r['varint_bytes'], r['current_bytes']):>13s}")
    print(f"  {'delta+varint':<20s} {r['delta_bytes']:>15,} {bpr_delta:>10.2f} {pct(r['delta_bytes'], r['current_bytes']):>13s}")
    print(f"  {'packed-CSR analog':<20s} {r['csr_bytes']:>15,} {bpr_csr:>10.2f} {pct(r['csr_bytes'], r['current_bytes']):>13s}")


def main():
    args = sys.argv[1:]
    paths = []

    if not args:
        print(__doc__)
        print("\nRun with --all or specify leaf files.")
        return 1

    if "--all" in args:
        for dataset in ["cora_gnn", "cora", "ogbn-arxiv"]:
            d = DBS_DIR / dataset
            if d.is_dir():
                for leaf in sorted(d.glob("*.leaf")):
                    if leaf.stat().st_size > PAGE_SIZE:  # skip empty single-page
                        paths.append(leaf)
    elif "--dataset" in args:
        idx = args.index("--dataset")
        dataset = args[idx + 1]
        d = DBS_DIR / dataset
        for leaf in sorted(d.glob("*.leaf")):
            if leaf.stat().st_size > PAGE_SIZE:
                paths.append(leaf)
    else:
        paths = [Path(a) for a in args if not a.startswith("--")]

    if not paths:
        print("No leaf files to analyze.")
        return 1

    all_results = []
    for p in paths:
        stem = p.name.replace(".leaf", "")
        N = record_width_for(stem)
        r = analyze_file(p, N)
        if r is None:
            print(f"  (skipping {p.name}: empty or non-page-aligned)")
            continue
        print_result(r)
        all_results.append(r)

    # Aggregate totals
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("AGGREGATE across analyzed files")
        print("=" * 70)
        agg = {
            "current": sum(r["current_bytes"] for r in all_results),
            "raw": sum(r["raw_bytes"] for r in all_results),
            "prefix": sum(r["prefix_bytes"] for r in all_results),
            "varint": sum(r["varint_bytes"] for r in all_results),
            "delta": sum(r["delta_bytes"] for r in all_results),
            "csr": sum(r["csr_bytes"] for r in all_results),
        }
        tot_records = sum(r["total_records"] for r in all_results)

        def mb(n):
            return n / (1024 * 1024)

        def pct(saved):
            return f"{100 * (1 - saved / agg['current']):+.1f}%"

        print(f"  {'Technique':<22s} {'Total MB':>12s} {'B/record':>10s} {'vs current':>12s}")
        print(f"  {'-' * 62}")
        print(f"  {'current':<22s} {mb(agg['current']):>12.2f} {agg['current']/tot_records:>10.2f}  (baseline)")
        print(f"  {'raw (uncompressed)':<22s} {mb(agg['raw']):>12.2f} {agg['raw']/tot_records:>10.2f}  {pct(agg['raw'])}")
        print(f"  {'prefix-K':<22s} {mb(agg['prefix']):>12.2f} {agg['prefix']/tot_records:>10.2f}  {pct(agg['prefix'])}")
        print(f"  {'varint':<22s} {mb(agg['varint']):>12.2f} {agg['varint']/tot_records:>10.2f}  {pct(agg['varint'])}")
        print(f"  {'delta+varint':<22s} {mb(agg['delta']):>12.2f} {agg['delta']/tot_records:>10.2f}  {pct(agg['delta'])}")
        print(f"  {'packed-CSR':<22s} {mb(agg['csr']):>12.2f} {agg['csr']/tot_records:>10.2f}  {pct(agg['csr'])}")
        print(f"\n  Total records analyzed: {tot_records:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
