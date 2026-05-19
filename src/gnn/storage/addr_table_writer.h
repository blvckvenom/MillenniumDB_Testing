// src/gnn/storage/addr_table_writer.h
#pragma once

#include "gnn/storage/addr_table.h"
#include "graph_models/object_id.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace mdb::gnn {

/// In-memory buffers backing a single batch's AddrTable.
/// Filled by AddrTableWriter::build, serialized by AddrTableWriter::write_atomic.
struct AddrTableBuffers {
    AddrTableHeader        header{};
    std::vector<uint32_t>  l1_positions;
    std::vector<uint32_t>  l1_indices;
    std::vector<uint32_t>  l2_positions;
    std::vector<uint32_t>  l2_indices;
    std::vector<uint32_t>  l3_positions;
    std::vector<uint64_t>  l3_row_idxs;
    std::vector<uint32_t>  l4_positions;
    std::vector<uint32_t>  l4_indices;
    std::vector<uint32_t>  zero_positions;

    /// Total bytes the serialized file will occupy (header + all 9 arrays).
    size_t total_bytes() const { return header.expected_file_size(); }
};

class AddrTableWriter {
public:
    using RmapFind = std::function<std::optional<uint64_t>(ObjectId)>;

    /// Classify each node in unique_nodes into one of {L1, L2, L4, L3, zero}
    /// in that lookup order; fill `out` in place.
    /// Templated on cache types so tests can pass a mock.
    template <typename CacheA, typename CacheB>
    static void build(
        const std::vector<ObjectId>& unique_nodes,
        const CacheA*                gpu_cache,
        const CacheB*                cpu_cache,
        const std::unordered_map<uint64_t, uint32_t>& slim_oid_to_idx,
        const RmapFind&              rmap_find,
        uint64_t                     meta_sha256_head,
        AddrTableBuffers&            out);

    /// Atomically write `buf` to `path`: temp file (.tmp suffix) +
    /// fsync + rename. Throws std::runtime_error on any I/O error;
    /// never leaves a partial file at `path`.
    static void write_atomic(const std::filesystem::path& path,
                              const AddrTableBuffers& buf);
};

// ---------------------------------------------------------------------------
// Template definition — must be in the header so any cache type (including
// mocks in tests) can instantiate it without explicit instantiation.
// ---------------------------------------------------------------------------

template <typename CacheA, typename CacheB>
void AddrTableWriter::build(
    const std::vector<ObjectId>& unique_nodes,
    const CacheA*                gpu_cache,
    const CacheB*                cpu_cache,
    const std::unordered_map<uint64_t, uint32_t>& slim_oid_to_idx,
    const RmapFind&              rmap_find,
    uint64_t                     meta_sha256_head,
    AddrTableBuffers&            out)
{
    out = AddrTableBuffers{};
    out.l1_positions.reserve(unique_nodes.size() / 4);
    out.l1_indices.reserve(unique_nodes.size() / 4);
    out.l2_positions.reserve(unique_nodes.size() / 8);
    out.l2_indices.reserve(unique_nodes.size() / 8);
    out.l4_positions.reserve(unique_nodes.size() / 4);
    out.l4_indices.reserve(unique_nodes.size() / 4);
    out.l3_positions.reserve(unique_nodes.size() / 8);
    out.l3_row_idxs.reserve(unique_nodes.size() / 8);

    for (uint32_t i = 0; i < static_cast<uint32_t>(unique_nodes.size()); ++i) {
        const ObjectId& oid = unique_nodes[i];

        // L1 — GPU cache (fastest)
        if (gpu_cache) {
            auto idx = gpu_cache->find_index(oid);
            if (idx.has_value()) {
                out.l1_positions.push_back(i);
                out.l1_indices.push_back(*idx);
                continue;
            }
        }
        // L2 — CPU pinned cache
        if (cpu_cache) {
            auto idx = cpu_cache->find_index(oid);
            if (idx.has_value()) {
                out.l2_positions.push_back(i);
                out.l2_indices.push_back(*idx);
                continue;
            }
        }
        // L4 — packed_slim (disk, random-access by slot index)
        {
            auto slim_it = slim_oid_to_idx.find(oid.id);
            if (slim_it != slim_oid_to_idx.end()) {
                out.l4_positions.push_back(i);
                out.l4_indices.push_back(slim_it->second);
                continue;
            }
        }
        // L3 — reordered FeatureMatrix (mmap, row index via rmap)
        if (rmap_find) {
            auto row = rmap_find(oid);
            if (row.has_value()) {
                out.l3_positions.push_back(i);
                out.l3_row_idxs.push_back(*row);
                continue;
            }
        }
        // zero — node not found in any tier; position left as zeros
        out.zero_positions.push_back(i);
    }

    out.header = AddrTableHeader::make(
        static_cast<uint32_t>(out.l1_positions.size()),
        static_cast<uint32_t>(out.l2_positions.size()),
        static_cast<uint32_t>(out.l3_positions.size()),
        static_cast<uint32_t>(out.l4_positions.size()),
        static_cast<uint32_t>(out.zero_positions.size()),
        meta_sha256_head);
}

} // namespace mdb::gnn
