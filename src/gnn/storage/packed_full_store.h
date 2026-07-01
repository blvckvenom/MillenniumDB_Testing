// src/gnn/storage/packed_full_store.h
#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>
#include "gnn/storage/packed_full_format.h"
namespace mdb::gnn {

// Writer: append per-batch [num_nodes, D] payloads to packed_full.dat (offset
// aligned to PackedFullHeader::ALIGNMENT) and emit packed_full.idx atomically.
// write_batch may be called in ANY batch order (entries indexed by batch_id);
// finalize() writes the .idx. NOT thread-safe across write_batch calls - the
// caller serializes (or reserves offsets) for parallel packing.
class PackedFullWriter {
public:
    PackedFullWriter(const std::filesystem::path& dir, uint64_t store_fp,
                     uint32_t feature_dim, uint32_t dtype, uint64_t row_bytes);
    ~PackedFullWriter();
    // payload = num_nodes * row_bytes contiguous bytes (D features per node, dtype).
    void write_batch(uint64_t batch_id, const void* payload, uint64_t num_nodes);
    void finalize();  // write packed_full.idx atomically + fsync; close .dat
private:
    std::filesystem::path dir_;
    PackedFullHeader hdr_;
    int dat_fd_ = -1;
    uint64_t cur_offset_ = 0;
    std::vector<PackedFullEntry> entries_;  // indexed by batch_id
    bool finalized_ = false;
};

class PackedFullReader {
public:
    // nullopt if missing / bad magic-version / store_fp mismatch (stale).
    static std::optional<PackedFullReader> open(const std::filesystem::path& dir,
                                                uint64_t expected_store_fp);
    uint64_t num_batches() const { return hdr_.num_batches; }
    uint32_t feature_dim() const { return hdr_.feature_dim; }
    uint32_t dtype()       const { return hdr_.dtype; }
    uint64_t row_bytes()   const { return hdr_.row_bytes; }
    const PackedFullHeader& header() const { return hdr_; }
    PackedFullEntry entry(uint64_t batch_id) const { return entries_.at(batch_id); }
    std::filesystem::path dat_path() const { return dir_ / "packed_full.dat"; }
    // Test-only: plain pread of a batch payload into `out` (num_nodes*row_bytes).
    void read_payload_for_test(uint64_t batch_id, void* out) const;
private:
    std::filesystem::path dir_;
    PackedFullHeader hdr_{};
    std::vector<PackedFullEntry> entries_;
};
} // namespace mdb::gnn
