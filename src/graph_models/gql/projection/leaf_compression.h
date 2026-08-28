#pragma once

// ===========================================================================
// Redundant-byte leaf compression for BTREE+BITSET (v1) projection leaves.
//
// Re-enables the leaf format's pre-existing redundant-byte compression that
// ProjectionStorage's bulk/streaming index builders had been bypassing by
// always passing an all-zero bitset to BPTLeafWriter::process_block. A set
// bit b (0..N*8-1) marks a byte position that holds the SAME value in every
// record on the page; that byte is then stored ONCE (in the redundant
// section) instead of once per record.
//
// The reader (BPTLeafV1 ctor + set_record, src/storage/index/bplus_tree/
// bplus_tree_leaf.{h,cc}) ALREADY decompresses this layout. This header only
// computes the per-page bitset and lays the page bytes out EXACTLY as the
// reader reconstructs them:
//
//   on-disk page (after the 8-byte value_count/next_leaf header):
//     [ N bytes ]                        redundant bitset (LE bytes)
//     [ redundant_count bytes ]          the shared byte for each set bit,
//                                         in ascending byte-position order
//     [ count * (N*8 - redundant_count) ] per-record non-redundant bytes,
//                                         each record's non-set positions in
//                                         ascending byte-position order
//
// This matches BPTLeafV1::set_record's decode loop byte-for-byte:
//   - bitset_ptr  = page + 8
//   - redundant_bytes = page + 8 + N
//   - records     = page + 8 + N + redundant_count
//   - record r byte i = redundant_bitset[i] ? redundant_bytes[redundant_pos++]
//                                            : records[r*(8N-rc) + unique_pos++]
//
// Scope: BPTLeafWriter (v1 / BITSET) only. The CSR_HYBRID (BPTLeafCSRWriter)
// and DELTA_VARINT (BPTLeafV2Writer) paths are SEPARATE and untouched.
// ===========================================================================

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "misc/ablation_registry.h"
#include "storage/index/record.h"

namespace GQL {

// Read MDB_PROJECTION_NO_LEAF_COMPRESSION once. When set to a truthy value
// ({"1","true","yes"}) the builders restore the legacy all-zero bitset path,
// providing an escape hatch if a problem surfaces.
inline bool leaf_compression_disabled() {
    static const bool disabled = [] {
        // The five truthy spellings are the ones this switch has always
        // honoured, and they stay the whole of it: "on" or "y" read as "leave
        // compression enabled" before and still do. choice() only adds that
        // such a value is now reported instead of being indistinguishable from
        // an unset variable. The static keeps the lock out of the writers.
        const std::string s = Ablation::choice(
            "MDB_PROJECTION_NO_LEAF_COMPRESSION", "0",
            {"0", "1", "true", "yes", "TRUE", "YES"});
        return s == "1" || s == "true" || s == "yes" || s == "TRUE" || s == "YES";
    }();
    return disabled;
}

// Compute the redundant-byte bitset for a page of `count` records (each
// Record<N> = N uint64 = N*8 bytes). Bit b is set iff byte position b holds
// the SAME value across every record on the page.
//
// Byte access is direct (no std::bitset::to_ulong, which caps at 64 bits) so
// this is correct for any N. For count == 0 the bitset is all-zero (no page is
// ever written for an empty record set in the callers). For count == 1 every
// byte is trivially constant, so every bit is set — this is the maximally-
// compressed degenerate page and the reader handles it (redundant_count==N*8,
// records section is empty, set_record reads every byte from redundant_bytes).
template <std::size_t N>
std::bitset<N * 8> compute_redundant_bitset(const Record<N>* page_records, uint32_t count) {
    std::bitset<N * 8> bitset;
    if (count == 0) {
        return bitset; // all zero
    }

    const unsigned char* first = reinterpret_cast<const unsigned char*>(&page_records[0]);

    // Start assuming every byte position is constant, then clear positions
    // that differ in any subsequent record.
    bitset.set();

    for (uint32_t r = 1; r < count; ++r) {
        const unsigned char* rec = reinterpret_cast<const unsigned char*>(&page_records[r]);
        for (std::size_t b = 0; b < N * 8; ++b) {
            if (bitset[b] && rec[b] != first[b]) {
                bitset.set(b, false);
            }
        }
    }
    return bitset;
}

// Greedily determine how many of the `available` sorted records (starting at
// `page_records[0]`) fit into ONE compressed leaf page, and return that count
// together with the redundant-byte bitset for exactly those records.
//
// The on-disk page cost for `n` records under bitset `bs` is
//   2*sizeof(uint32_t) [value_count+next_leaf header]
//     + N               [bitset bytes]
//     + bs.count()      [shared redundant bytes]
//     + n*(N*8 - bs.count())  [per-record non-redundant bytes]
// which must be <= Page::SIZE (4096). This mirrors
// BPTLeafV1::get_page_size(bs, n) exactly.
//
// Because adding a record can only CLEAR bitset bits (reduce redundancy), the
// per-record cost is monotonically non-decreasing as the page grows, so a
// simple incremental scan (extend the page one record at a time, recomputing
// the running bitset, stop when the next record would overflow) finds a valid
// packing. We always keep at least one record per page (a single record always
// fits: 8 + N + N*8 = 8 + 9*N <= 4096 for N <= 3).
//
// `max_cap` bounds the result so callers can keep page sizes within any
// pre-sized scratch buffer (pass the uncompressed max_records_per_leaf).
//
// page_bytes is the Page::SIZE budget (4096); header_bytes is the 8-byte
// value_count+next_leaf prefix. Returns {records_in_page, page_bitset}.
template <std::size_t N>
struct CompressedPagePlan {
    uint32_t           records_in_page;
    std::bitset<N * 8> bitset;
};

template <std::size_t N>
CompressedPagePlan<N> plan_compressed_page(const Record<N>* page_records,
                                           std::size_t available,
                                           std::size_t max_cap,
                                           std::size_t page_budget = 4096,
                                           std::size_t header_bytes = 2 * sizeof(uint32_t)) {
    constexpr std::size_t REC_BITS = N * 8;

    CompressedPagePlan<N> plan;
    if (available == 0 || max_cap == 0) {
        plan.records_in_page = 0;
        return plan;
    }

    const auto fits = [&](const std::bitset<REC_BITS>& bs, std::size_t n) {
        const std::size_t rc = bs.count();
        return header_bytes + N + rc + n * (REC_BITS - rc) <= page_budget;
    };

    // Running bitset over records [0..n-1]. Start with record 0 (all bits set
    // — a single record is trivially all-redundant), then extend.
    std::bitset<REC_BITS> bs;
    bs.set();
    const unsigned char* first = reinterpret_cast<const unsigned char*>(&page_records[0]);

    std::size_t n = 1; // record 0 always taken (guaranteed to fit)
    const std::size_t limit = std::min(available, max_cap);

    while (n < limit) {
        // Tentatively fold record n into the bitset.
        std::bitset<REC_BITS> next_bs = bs;
        const unsigned char* rec = reinterpret_cast<const unsigned char*>(&page_records[n]);
        for (std::size_t b = 0; b < REC_BITS; ++b) {
            if (next_bs[b] && rec[b] != first[b]) {
                next_bs.set(b, false);
            }
        }
        if (!fits(next_bs, n + 1)) {
            break;
        }
        bs = next_bs;
        ++n;
    }

    plan.records_in_page = static_cast<uint32_t>(n);
    plan.bitset = bs;
    return plan;
}

// Pack a page of `count` records into `out` using `bitset`, producing the
// EXACT byte layout BPTLeafWriter::process_block copies and BPTLeafV1 reads.
//
// `out` must have room for N + redundant_count + count*(N*8 - redundant_count)
// bytes. Returns the number of bytes written (the same expression
// process_block computes from its size + bitset arguments).
template <std::size_t N>
std::size_t pack_compressed_page(const Record<N>* page_records,
                                 uint32_t count,
                                 const std::bitset<N * 8>& bitset,
                                 char* out) {
    constexpr std::size_t REC_BYTES = sizeof(uint64_t) * N; // = N*8
    const std::size_t redundant_count = bitset.count();

    std::size_t pos = 0;

    // 1) bitset: N bytes, little-endian byte order (byte k holds bits
    //    [k*8 .. k*8+7], LSB first) — matches the reader's ctor loop that does
    //    `(bitset_ptr[i] >> bit) & 1` for byte i, bit 0..7.
    for (std::size_t i = 0; i < N; ++i) {
        unsigned char byte_val = 0;
        for (int bit = 0; bit < 8; ++bit) {
            if (bitset[i * 8 + bit]) {
                byte_val |= static_cast<unsigned char>(1u << bit);
            }
        }
        out[pos++] = static_cast<char>(byte_val);
    }

    // 2) redundant bytes: for each SET bit position (ascending), the single
    //    shared byte value (taken from record 0 — guaranteed constant across
    //    the page by construction of the bitset).
    if (redundant_count > 0) {
        const unsigned char* first = reinterpret_cast<const unsigned char*>(&page_records[0]);
        for (std::size_t b = 0; b < REC_BYTES; ++b) {
            if (bitset[b]) {
                out[pos++] = static_cast<char>(first[b]);
            }
        }
    }

    // 3) per-record non-redundant bytes: for each record, the bytes at
    //    NON-set positions (ascending), (N*8 - redundant_count) bytes each.
    for (uint32_t r = 0; r < count; ++r) {
        const unsigned char* rec = reinterpret_cast<const unsigned char*>(&page_records[r]);
        for (std::size_t b = 0; b < REC_BYTES; ++b) {
            if (!bitset[b]) {
                out[pos++] = static_cast<char>(rec[b]);
            }
        }
    }

    return pos;
}

} // namespace GQL
