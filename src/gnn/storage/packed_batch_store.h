#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/gnn_dtype.h"
#include "graph_models/object_id.h"

namespace mdb::gnn {

/**
 * @brief On-disk header for packed batch files (.bin). Always 32 bytes, at offset 0.
 *
 * File layout: [Header: 32 bytes][Data: num_nodes * feature_dim * dtype_size bytes]
 * Data is row-major, contiguous, no padding between rows.
 */
struct PackedBatchHeader {
    static constexpr uint32_t MAGIC       = 0x474E4E42; // "GNNB"
    static constexpr uint32_t VERSION     = 1;
    static constexpr uint32_t MAX_VERSION = 2;
    static constexpr size_t   SIZE        = 32;

    uint32_t magic;
    uint32_t version;
    uint64_t num_nodes;
    uint64_t feature_dim;
    uint8_t  dtype;
    uint8_t  reserved[7];

    static PackedBatchHeader make(uint64_t nodes, uint64_t dim, GnnDtype dt) {
        if (static_cast<uint8_t>(dt) > static_cast<uint8_t>(GnnDtype::MAX_VALUE)) {
            throw std::invalid_argument("PackedBatchHeader::make: invalid dtype");
        }
        PackedBatchHeader h{};
        std::memset(&h, 0, sizeof(h));
        h.magic       = MAGIC;
        h.version     = VERSION;
        h.num_nodes   = nodes;
        h.feature_dim = dim;
        h.dtype       = static_cast<uint8_t>(dt);
        return h;
    }

    /// Create a v2 header (includes ObjectId table after header).
    static PackedBatchHeader make_v2(uint64_t nodes, uint64_t dim, GnnDtype dt) {
        auto h = make(nodes, dim, dt);
        h.version = 2;
        return h;
    }

    /// Unlike FeatureMatrixHeader, num_nodes == 0 IS valid (empty batch).
    bool is_valid() const {
        return magic == MAGIC && version >= 1 && version <= MAX_VERSION
            && feature_dim > 0
            && dtype <= static_cast<uint8_t>(GnnDtype::MAX_VALUE);
    }

    /// Whether this header has an ObjectId table (v2+).
    bool has_oid_table() const { return version >= 2; }

    /// Offset from file start to feature data.
    /// v1: SIZE (32 bytes, immediately after header)
    /// v2: SIZE + num_nodes * 8 (after header + OID table)
    size_t data_offset() const {
        return SIZE + (has_oid_table() ? num_nodes * sizeof(uint64_t) : 0);
    }

    GnnDtype get_dtype() const {
        return static_cast<GnnDtype>(dtype);
    }

    /// Returns data size in bytes. Throws std::overflow_error on overflow.
    size_t data_bytes() const {
        size_t dim_bytes = feature_dim * dtype_size(get_dtype());
        if (feature_dim > 0 && dim_bytes / feature_dim != dtype_size(get_dtype())) {
            throw std::overflow_error("PackedBatchHeader::data_bytes: dimension overflow");
        }
        size_t total = num_nodes * dim_bytes;
        if (num_nodes > 0 && total / num_nodes != dim_bytes) {
            throw std::overflow_error("PackedBatchHeader::data_bytes: total size overflow");
        }
        return total;
    }
};

static_assert(sizeof(PackedBatchHeader) == 32, "PackedBatchHeader must be exactly 32 bytes");
static_assert(std::is_standard_layout_v<PackedBatchHeader>,
              "PackedBatchHeader must be standard layout for direct I/O");
static_assert(std::is_trivially_copyable_v<PackedBatchHeader>,
              "PackedBatchHeader must be trivially copyable for direct I/O");

/**
 * @brief Writes per-batch packed feature files.
 *
 * Each write_batch() creates one self-contained file with header + data.
 * NOT thread-safe. Call from a single thread.
 */
class PackedBatchWriter {
public:
    /// Throws std::invalid_argument if feature_dim == 0.
    PackedBatchWriter(const std::filesystem::path& dir,
                      uint64_t feature_dim,
                      GnnDtype dtype);

    // Non-copyable (batches_written_ aliasing would be dangerous)
    PackedBatchWriter(const PackedBatchWriter&) = delete;
    PackedBatchWriter& operator=(const PackedBatchWriter&) = delete;
    PackedBatchWriter(PackedBatchWriter&&) = default;
    PackedBatchWriter& operator=(PackedBatchWriter&&) = default;

    /// batch_id must equal batches_written() (sequential from 0).
    void write_batch(uint64_t batch_id, const void* data, uint64_t num_nodes);

    uint64_t feature_dim() const { return feature_dim_; }
    GnnDtype dtype() const { return dtype_; }
    uint64_t batches_written() const { return batches_written_; }
    const std::filesystem::path& dir() const { return dir_; }

private:
    std::filesystem::path dir_;
    uint64_t feature_dim_;
    GnnDtype dtype_;
    uint64_t batches_written_ = 0;

    std::filesystem::path batch_path(uint64_t batch_id) const;
};

/**
 * @brief Reads per-batch packed feature files.
 *
 * Thread-safe: each read_batch() opens its own fd.
 */
class PackedBatchReader {
public:
    /// Throws std::runtime_error if dir does not exist.
    PackedBatchReader(const std::filesystem::path& dir,
                      uint64_t num_batches,
                      uint64_t feature_dim,
                      GnnDtype dtype);

    /// Thread-safe. Validates header against expected feature_dim/dtype.
    uint64_t read_batch(uint64_t batch_id, void* out, size_t out_capacity) const;

    PackedBatchHeader read_header(uint64_t batch_id) const;

    uint64_t num_batches() const { return num_batches_; }
    uint64_t feature_dim() const { return feature_dim_; }
    GnnDtype dtype() const { return dtype_; }
    const std::filesystem::path& dir() const { return dir_; }

private:
    std::filesystem::path dir_;
    uint64_t num_batches_;
    uint64_t feature_dim_;
    GnnDtype dtype_;

    std::filesystem::path batch_path(uint64_t batch_id) const;
};

/// Primary API: callback-based (streaming, constant memory).
void generate_packed_batches(
    const FeatureMatrix& features,
    uint64_t num_batches,
    std::function<std::vector<uint64_t>(uint64_t batch_id)> batch_provider,
    const std::filesystem::path& output_dir
);

/// Convenience: materialized assignments (for tests, small datasets).
void generate_packed_batches(
    const FeatureMatrix& features,
    const std::vector<std::vector<uint64_t>>& batch_assignments,
    const std::filesystem::path& output_dir
);

/// Spec B1 (DiskGNN-style inverted loop): partitioned packer.
///
/// Inverts the per-batch outer loop into a feature-partition-oriented
/// sequential scan. Each region of the FeatureMatrix is read exactly once,
/// then per-(partition, batch) contributions are appended to their batch
/// files via pwrite. All batch FDs are kept open across Phase 2 (relies on
/// ulimit -n; 1024+ recommended).
///
/// Two output formats, selected by whether `oid_provider` is supplied:
///   - v1 (oid_provider empty): header + features section; rows grouped by
///     feature partition (ascending fmat row), keeping row_provider order
///     within each group. PRECONDITION for positional consumers: this equals
///     row_provider order — and the output is bit-identical to
///     generate_packed_batches — only when row_provider returns rows sorted
///     ascending by fmat row (true for the four_level_store caller).
///   - v2 (oid_provider non-empty): header + OID table + features section;
///     rows in **partition iteration order** (NOT row_provider order).
///     `oid_table[i] ↔ data[i]` consistency is preserved. Functionally
///     equivalent to v1 but byte layout differs.
///
/// Memory profile: peak ~ row_bytes * partition_rows + 16B/ref inverted
/// index + 8B/ref OID array (v2 only). Partition size of 256 MB on
/// papers100M-class datasets keeps peak RSS under 2 GB.
///
/// partition_bytes: target bytes per partition (rounded down to whole rows).
///
/// DiskGNN-adoption Plan 1 (optional consolidated cold-feature file): when
/// `consolidated_path` is non-empty, the SAME single .fmat scan ALSO writes one
/// consolidated file — a ConsolidatedSlimHeader (consolidated_slim.h) followed
/// by every batch's data section concatenated, batch b at a 4096-aligned offset.
/// The per-batch data payload is byte-identical to the per-batch .bin data
/// section (same partition-iteration order), so addr_table.l4_indices index it
/// identically. Per-batch (offset, length) are returned in out_consolidated_*
/// for the addr_table v2 header. The per-batch .bin files are written unchanged
/// (the consolidated file is an additive artifact, not a replacement). Disabled
/// (empty path) => behaviour byte-identical to before. `consolidated_perm_fp` /
/// `consolidated_meta_sha` are stamped into the header for stale-rejection.
void generate_packed_batches_partitioned(
    const FeatureMatrix& features,
    uint64_t num_batches,
    std::function<std::vector<uint64_t>(uint64_t batch_id)> row_provider,
    const std::filesystem::path& output_dir,
    size_t partition_bytes = 256ULL * 1024 * 1024,
    std::function<std::vector<ObjectId>(uint64_t batch_id)> oid_provider = {},
    const std::filesystem::path& consolidated_path = {},
    uint64_t consolidated_perm_fp = 0,
    uint64_t consolidated_meta_sha = 0,
    std::vector<uint64_t>* out_consolidated_offsets = nullptr,
    std::vector<uint64_t>* out_consolidated_lengths = nullptr
);

} // namespace mdb::gnn
