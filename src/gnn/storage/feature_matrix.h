#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

    /**
     * Parallel variant of create_streaming. Pre-allocates the output via
     * ftruncate, then dispatches num_workers std::threads each handling a
     * disjoint contiguous row range via pwrite() at the appropriate file
     * offset. The supplied RowWriter MUST be thread-safe — it is called
     * concurrently from multiple threads on different (row_id, dest) pairs.
     *
     * num_workers == 0 falls back to the sequential single-thread path.
     *
     * On any worker exception the partial file is removed and the first
     * captured exception is rethrown after all workers join.
     */
    static FeatureMatrix create_parallel(
        const std::filesystem::path& path,
        uint64_t num_rows,
        uint64_t num_cols,
        GnnDtype dtype,
        RowWriter writer,
        unsigned num_workers
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
    // STEP 8: `fingerprint` (default 0) is embedded in header.reserved[0..7] so
    // the feature store can detect a reordered.fmat built for a different sample
    // and recompute it instead of opening it with the wrong shape.
    static FeatureMatrix create_reordered(
        const FeatureMatrix& source,
        const std::vector<uint64_t>& permutation,
        const std::filesystem::path& output_path,
        uint64_t fingerprint = 0
    );

    // --- Metadata ---
    uint64_t    num_rows()    const { return header_.num_rows; }
    uint64_t    num_cols()    const { return header_.num_cols; }
    GnnDtype    dtype()       const { return header_.get_dtype(); }
    size_t      row_bytes()   const { return header_.row_bytes(); }
    size_t      total_bytes() const { return header_.data_bytes(); }
    const std::filesystem::path& path()    const { return path_; }

    // STEP 8: content fingerprint embedded in header.reserved[0..7] at
    // create_reordered() time. 0 = absent (legacy files, or non-reordered FMs).
    uint64_t fingerprint() const {
        uint64_t f;
        std::memcpy(&f, header_.reserved, sizeof(f));
        return f;
    }

    /**
     * @brief Hint the kernel that the mapped region can be evicted from
     *        the page cache. Useful after a single-pass scan where the
     *        caller won't read these pages again soon.
     *
     * Fix #22: papers100M's 56 GB source + 56 GB reordered + 8 GB caches
     * exceed the 30 GB host RAM. At the end of gnn_build_feature_store,
     * downstream callers (e.g. gnn_train running immediately after)
     * benefit from a clean page-cache budget instead of inheriting 100+ GB
     * of stale pages competing for eviction.
     */
    void release_cache() const;

private:
    // Fix #15 helper — needs access to private mmap members for source
    // mmap and to construct the result via the same path as create_reordered.
    static FeatureMatrix create_reordered_external_sort_(
        const FeatureMatrix& source,
        const std::vector<uint64_t>& permutation,
        const std::filesystem::path& output_path,
        uint64_t fingerprint = 0);

    FeatureMatrix() = default;

    FeatureMatrixHeader header_{};
    std::filesystem::path path_;
    void*    mmap_ptr_  = nullptr;  // points to start of mmap (includes header)
    size_t   mmap_size_ = 0;        // total mmap size (header + data)

    const void* data_ptr() const;   // pointer to start of data (after header)
};

} // namespace mdb::gnn
