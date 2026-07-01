#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace mdb::gnn {

/**
 * @brief Immutable mmap-backed array mapping row indices to train/val/test split assignments.
 *
 * File layout (splits.bin):
 *
 * Offset  Size  Field
 *  0       8    magic: "GNNS\0\0\0\0"
 *  8       4    version: uint32 (1)
 * 12       4    reserved: uint32 (0)
 * 16       8    num_nodes: uint64
 * 24      N×1   splits: uint8[N]  (0=TRAIN, 1=VAL, 2=TEST, 255=UNLABELED)
 *
 * Header is 24 bytes. Data starts at offset 24.
 *
 * Thread-safe for concurrent reads after construction.
 *
 * Usage:
 *   SplitStore::write("splits.bin", splits);
 *   auto ss = SplitStore::open("splits.bin");
 *   SplitStore::Split s = ss.get(row_index);
 *   auto mask = ss.gather_mask(row_indices, SplitStore::TRAIN);
 */
class SplitStore {
public:
    enum Split : uint8_t {
        TRAIN     = 0,
        VAL       = 1,
        TEST      = 2,
        UNLABELED = 255
    };

    static SplitStore open(const std::filesystem::path& path);
    static void       write(const std::filesystem::path& path,
                            const std::vector<uint8_t>& splits);

    /// Parses a split name string.
    /// "train" -> TRAIN, "val"/"validation" -> VAL, "test" -> TEST,
    /// anything else -> UNLABELED
    static Split parse_split_string(const std::string& s);

    uint64_t num_nodes() const { return num_nodes_; }

    /// Returns the split assignment at row_index. Throws std::out_of_range if index >= num_nodes.
    Split get(uint64_t row_index) const;

    /// Returns a 1-D bool tensor of shape [B].
    /// Element i is true when splits[row_indices[i]] == target.
    torch::Tensor gather_mask(const std::vector<uint64_t>& row_indices, Split target) const;

    // Move-only (owns mmap region)
    SplitStore(SplitStore&&) noexcept;
    SplitStore& operator=(SplitStore&&) noexcept;
    SplitStore(const SplitStore&) = delete;
    SplitStore& operator=(const SplitStore&) = delete;
    ~SplitStore();

private:
    SplitStore() = default;

    static constexpr size_t   HEADER_SIZE = 24;
    static constexpr uint8_t  MAGIC[8]    = {'G','N','N','S','\0','\0','\0','\0'};
    static constexpr uint32_t VERSION     = 1;

    void*    mmap_ptr_  = nullptr;
    size_t   mmap_size_ = 0;
    uint64_t num_nodes_ = 0;

    const uint8_t* data_ptr() const;
};

} // namespace mdb::gnn
