// Standalone dumper for projection B+Tree .leaf files (Spec #5 T5.12).
//
// Reads a projection .leaf file page-by-page and prints each decoded record
// to stdout as N space-separated decimal uint64 fields, one record per line.
//
// Auto-detects the leaf page format from byte 0:
//   0x02  -> v2 (Spec #5 DELTA_VARINT); decoded via BPTLeafV2<N>(page, ReadTag{})
//   else  -> v1 (pre-Spec-#5 BITSET);    decoded inline with a minimal reader
//
// The output is deterministic across formats and record-widths, which makes
// it suitable as the canonical "record sequence" that
// scripts/test_projection_leaffmt.sh diffs across BITSET vs DELTA_VARINT
// runs of the same projection index.
//
// Usage:  mdb_leaf_dump <path-to-.leaf> <record-width-N>
// Exit:   0 on success, 1 on usage/decode error.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"
#include "storage/page/page.h"

namespace {

constexpr size_t PAGE_SIZE = Page::SIZE;  // 4096

// Inline V1 (BITSET) reader: extracts records from one raw 4 KB page buffer,
// mirroring the reconstruction logic in BPTLeafV1<N>::set_record() without
// requiring a BufferManager-backed Page handle. Layout (pre-Spec-#5):
//   bytes 0..3   : value_count   (uint32 LE)
//   bytes 4..7   : next_leaf     (uint32 LE)
//   bytes 8..8+N : bitset[N] — bit pos_bitset=i*8+bit marks byte i
//                  of the N*8-byte record as "redundant" (constant across
//                  all records on this page, stored once)
//   next redundant_count bytes   : the redundant byte values in order
//   remaining bytes              : records[value_count], each sized
//                                  (N*8 - redundant_count) bytes holding
//                                  only the non-redundant byte positions
//
// For N in {1,2,3} the Record<N> is sizeof(uint64_t)*N bytes on-disk.
//
// `out` receives one line per record, formatted "v0 v1 ... v{N-1}\n".
template <std::size_t N>
void dump_v1_page(const unsigned char* p, std::ostream& out)
{
    uint32_t value_count = 0;
    std::memcpy(&value_count, p, sizeof(uint32_t));

    const unsigned char* bitset_bytes = p + 2 * sizeof(uint32_t);

    // Reconstruct the redundant-byte positions and count.
    bool     is_redundant[N * sizeof(uint64_t)] = {};
    uint32_t redundant_count = 0;
    for (size_t i = 0; i < N; ++i) {
        for (int bit = 0; bit < 8; ++bit) {
            const size_t pos = i * 8 + bit;
            const bool   b   = ((bitset_bytes[i] >> bit) & 1u) != 0;
            is_redundant[pos] = b;
            if (b) ++redundant_count;
        }
    }

    const unsigned char* redundant_bytes = bitset_bytes + N;
    const unsigned char* records =
        redundant_bytes + redundant_count;

    const size_t rec_bytes = N * sizeof(uint64_t) - redundant_count;

    for (uint32_t r = 0; r < value_count; ++r) {
        unsigned char  buf[N * sizeof(uint64_t)];
        const unsigned char* current = records + r * rec_bytes;
        size_t redundant_pos = 0;
        size_t unique_pos    = 0;
        for (size_t i = 0; i < N * sizeof(uint64_t); ++i) {
            if (is_redundant[i]) {
                buf[i] = redundant_bytes[redundant_pos++];
            } else {
                buf[i] = current[unique_pos++];
            }
        }
        uint64_t fields[N];
        std::memcpy(fields, buf, sizeof(fields));
        for (size_t j = 0; j < N; ++j) {
            if (j) out << ' ';
            out << fields[j];
        }
        out << '\n';
    }
}


template <std::size_t N>
void dump_v2_page(const char* p, std::ostream& out)
{
    // The ReadTag ctor validates the 16-byte header and throws on corruption.
    BPTLeafV2<N> leaf(p, typename BPTLeafV2<N>::ReadTag{});
    const uint32_t vc = leaf.get_value_count();
    for (uint32_t i = 0; i < vc; ++i) {
        Record<N> r = leaf.get_record(i);
        for (size_t j = 0; j < N; ++j) {
            if (j) out << ' ';
            out << r[j];
        }
        out << '\n';
    }
}


template <std::size_t N>
int dump_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "mdb_leaf_dump: cannot open " << path << '\n';
        return 1;
    }

    std::vector<char> buf(PAGE_SIZE);
    size_t page_idx = 0;
    while (in.read(buf.data(), PAGE_SIZE)) {
        const unsigned char format_byte =
            static_cast<unsigned char>(buf[0]);
        try {
            if (format_byte == 0x02) {
                dump_v2_page<N>(buf.data(), std::cout);
            } else {
                dump_v1_page<N>(
                    reinterpret_cast<const unsigned char*>(buf.data()),
                    std::cout);
            }
        } catch (const std::exception& e) {
            std::cerr << "mdb_leaf_dump: decode error on page "
                      << page_idx << " of " << path << ": "
                      << e.what() << '\n';
            return 1;
        }
        ++page_idx;
    }
    if (in.gcount() != 0) {
        std::cerr << "mdb_leaf_dump: file " << path
                  << " has a trailing " << in.gcount()
                  << "-byte partial page (expected multiple of "
                  << PAGE_SIZE << ")\n";
        return 1;
    }
    return 0;
}

}  // namespace


int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path-to-.leaf> <record-width-N>\n"
                  << "  record-width-N: 1, 2, or 3\n";
        return 1;
    }
    const std::string path = argv[1];
    const int         n    = std::atoi(argv[2]);

    switch (n) {
        case 1: return dump_file<1>(path);
        case 2: return dump_file<2>(path);
        case 3: return dump_file<3>(path);
        default:
            std::cerr << "mdb_leaf_dump: unsupported record width "
                      << n << " (must be 1, 2, or 3)\n";
            return 1;
    }
}
