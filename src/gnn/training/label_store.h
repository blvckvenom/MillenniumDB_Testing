#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace mdb::gnn {

/**
 * @brief Immutable mmap-backed array mapping row indices to classification labels.
 *
 * File layout (labels.bin):
 *
 * Offset  Size  Field
 *  0       8    magic: "GNNL\0\0\0\0"
 *  8       4    version: uint32 (1)
 * 12       4    reserved: uint32 (0)
 * 16       8    num_nodes: uint64
 * 24       8    num_classes: uint64
 * 32      N×8   labels: int64[N]  (-1 = unlabeled)
 *
 * Header is 32 bytes. Data starts at offset 32.
 *
 * Thread-safe for concurrent reads after construction.
 *
 * Usage:
 *   LabelStore::write("labels.bin", labels, num_classes);
 *   auto ls = LabelStore::open("labels.bin");
 *   int64_t label = ls.get(row_index);
 *   auto tensor   = ls.gather(row_indices);
 */
class LabelStore {
public:
    static LabelStore open(const std::filesystem::path& path);
    static void       write(const std::filesystem::path& path,
                            const std::vector<int64_t>& labels,
                            uint64_t num_classes);

    uint64_t num_nodes()   const { return num_nodes_;   }
    uint64_t num_classes() const { return num_classes_; }

    /// Returns the label at row_index. Throws std::out_of_range if index >= num_nodes.
    int64_t get(uint64_t row_index) const;

    /// Gathers labels for a batch of row indices.
    /// Returns a 1-D int64 tensor of shape [B], with -1 for unlabeled nodes.
    torch::Tensor gather(const std::vector<uint64_t>& row_indices) const;

    // Move-only (owns mmap region)
    LabelStore(LabelStore&&) noexcept;
    LabelStore& operator=(LabelStore&&) noexcept;
    LabelStore(const LabelStore&) = delete;
    LabelStore& operator=(const LabelStore&) = delete;
    ~LabelStore();

private:
    LabelStore() = default;

    static constexpr size_t   HEADER_SIZE = 32;
    static constexpr uint8_t  MAGIC[8]    = {'G','N','N','L','\0','\0','\0','\0'};
    static constexpr uint32_t VERSION     = 1;

    void*    mmap_ptr_   = nullptr;
    size_t   mmap_size_  = 0;
    uint64_t num_nodes_  = 0;
    uint64_t num_classes_= 0;

    const int64_t* data_ptr() const;
};

} // namespace mdb::gnn
