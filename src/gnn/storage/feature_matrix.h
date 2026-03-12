#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

#include "gnn/storage/feature_matrix_header.h"
#include "gnn/storage/gnn_dtype.h"

namespace mdb::gnn {

/**
 * @brief Immutable flat [N, D] matrix stored on disk with memory-mapped read access.
 *
 * Thread-safe for concurrent reads (read-only mmap after construction).
 * No locks needed. File is immutable after create()/create_streaming() returns.
 *
 * Usage:
 *   auto fm = FeatureMatrix::create("features.fmat", N, D, FLOAT32, data_ptr);
 *   auto fm = FeatureMatrix::open("features.fmat");
 *   const float* row0 = fm.row_as<float>(0);
 *   fm.scan([](uint64_t i, const void* data) { ... });
 */
class FeatureMatrix {
public:
    // --- Construction (one-time writes) ---

    static FeatureMatrix create(
        const std::filesystem::path& path,
        uint64_t num_rows,
        uint64_t num_cols,
        GnnDtype dtype,
        const void* data
    );

    using RowWriter = std::function<void(uint64_t row_id, void* dest, uint64_t row_bytes)>;
    static FeatureMatrix create_streaming(
        const std::filesystem::path& path,
        uint64_t num_rows,
        uint64_t num_cols,
        GnnDtype dtype,
        RowWriter writer
    );

    static FeatureMatrix open(const std::filesystem::path& path);

    // --- Move only (owns mmap) ---
    FeatureMatrix(FeatureMatrix&& other) noexcept;
    FeatureMatrix& operator=(FeatureMatrix&& other) noexcept;
    FeatureMatrix(const FeatureMatrix&) = delete;
    FeatureMatrix& operator=(const FeatureMatrix&) = delete;
    ~FeatureMatrix();

    // --- Row access (O(1), thread-safe) ---

    const void* row(uint64_t row_id) const;

    template<typename T>
    const T* row_as(uint64_t row_id) const {
        assert(sizeof(T) == dtype_size(header_.get_dtype())
               && "row_as<T>() type size mismatch with stored dtype");
        return static_cast<const T*>(row(row_id));
    }

    // --- Sequential scan (MADV_SEQUENTIAL) ---

    using RowCallback = std::function<void(uint64_t row_id, const void* data)>;
    void scan(RowCallback callback) const;

    // --- Batch extraction (sorted I/O + MADV_WILLNEED) ---
    // Output order matches INPUT order of row_ids.
    void extract_rows(const std::vector<uint64_t>& row_ids, void* out) const;

    // --- Reordering (for MinHash L3) ---
    // permutation[i] = source row that goes to position i in output.
    static FeatureMatrix create_reordered(
        const FeatureMatrix& source,
        const std::vector<uint64_t>& permutation,
        const std::filesystem::path& output_path
    );

    // --- Metadata ---
    uint64_t    num_rows()    const { return header_.num_rows; }
    uint64_t    num_cols()    const { return header_.num_cols; }
    GnnDtype    dtype()       const { return header_.get_dtype(); }
    size_t      row_bytes()   const { return header_.row_bytes(); }
    size_t      total_bytes() const { return header_.data_bytes(); }
    const std::filesystem::path& path()    const { return path_; }

private:
    FeatureMatrix() = default;

    FeatureMatrixHeader header_{};
    std::filesystem::path path_;
    void*    mmap_ptr_  = nullptr;  // points to start of mmap (includes header)
    size_t   mmap_size_ = 0;        // total mmap size (header + data)

    const void* data_ptr() const;   // pointer to start of data (after header)
};

} // namespace mdb::gnn
