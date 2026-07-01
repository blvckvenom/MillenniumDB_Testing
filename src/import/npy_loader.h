#pragma once

/**
 * @file npy_loader.h
 * @brief Utility for loading NumPy .npy files containing node embeddings.
 *
 * Uses the libnpy header-only library for parsing NPY format.
 * Supports float32 and float64 arrays in C-order or Fortran-order.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Import {

/**
 * @brief Metadata extracted from NPY file header.
 */
struct NpyMetadata {
    std::vector<uint64_t> shape;  // e.g., {169343, 128} for node embeddings
    bool is_float64;              // true = float64 (double), false = float32 (float)
    bool fortran_order;           // true = column-major, false = row-major (C-order)
};

/**
 * @brief RAII handle for a memory-mapped .npy file.
 *
 * Owns an mmap region that covers the full .npy file. The `data` pointer
 * indexes past the header into the raw array bytes, suitable for C-order
 * row reads at `data + row_id * row_bytes`. Destructor calls munmap().
 *
 * Moves transfer ownership; copies are disabled.
 */
class NpyMemmap {
public:
    NpyMemmap() = default;
    ~NpyMemmap();

    NpyMemmap(NpyMemmap&& other) noexcept;
    NpyMemmap& operator=(NpyMemmap&& other) noexcept;

    NpyMemmap(const NpyMemmap&) = delete;
    NpyMemmap& operator=(const NpyMemmap&) = delete;

    const void*  data()       const { return data_; }
    std::size_t  data_bytes() const { return data_bytes_; }
    const NpyMetadata& metadata() const { return metadata_; }
    bool valid() const { return data_ != nullptr; }

private:
    friend class NpyLoader;

    void*       mmap_ptr_   = nullptr;
    std::size_t mmap_size_  = 0;
    const void* data_       = nullptr;  // points into mmap_ptr_ past the header
    std::size_t data_bytes_ = 0;
    NpyMetadata metadata_{};
};

/**
 * @brief Loader for NumPy .npy files.
 *
 * Provides static methods for loading embedding matrices from NPY files.
 * The typical use case is loading node feature matrices where:
 * - shape[0] = number of nodes
 * - shape[1] = embedding dimension
 */
class NpyLoader {
public:
    /**
     * @brief Load NPY file as float32 array.
     *
     * @param path Path to .npy file
     * @param metadata_out Output parameter for array metadata
     * @return Flattened array data (row-major order), empty on error
     */
    static std::vector<float> load_float32(
        const std::string& path,
        NpyMetadata& metadata_out
    );

    /**
     * @brief Load NPY file as float64 array.
     *
     * @param path Path to .npy file
     * @param metadata_out Output parameter for array metadata
     * @return Flattened array data (row-major order), empty on error
     */
    static std::vector<double> load_float64(
        const std::string& path,
        NpyMetadata& metadata_out
    );

    /**
     * @brief Validate NPY file without loading data.
     *
     * Checks:
     * - File exists and is readable
     * - Valid NPY format (magic bytes, header)
     * - Supported dtype (float32 or float64)
     * - 2D array shape
     *
     * @param path Path to .npy file
     * @param error_out Output parameter for error message
     * @return true if valid, false otherwise
     */
    static bool validate(const std::string& path, std::string& error_out);

    /**
     * @brief Read dtype from NPY header without loading data.
     *
     * Returns the itemsize: 4 for float32, 8 for float64, 0 on error.
     * This allows choosing the correct load function without trial-and-error.
     *
     * @param path Path to .npy file
     * @return Itemsize (4 or 8), or 0 on error
     */
    static int get_dtype_itemsize(const std::string& path);

    /**
     * @brief Get file size in bytes.
     *
     * @param path Path to file
     * @return File size, or 0 if file doesn't exist
     */
    static size_t get_file_size(const std::string& path);

    /**
     * @brief Memory-map a .npy file for streaming access without loading to RAM.
     *
     * Use this for large tensor files (e.g., papers100M features = 53 GiB)
     * where allocating a std::vector of the full array would OOM on commodity
     * hardware. The returned handle lets callers read individual rows via
     * `mm.data() + row_id * row_bytes` without materializing the full array.
     *
     * Fortran-order files are rejected: transposing requires a full read,
     * which defeats the purpose of mmap-streaming. Convert to C-order first.
     *
     * @param path Path to .npy file
     * @param error_out Error message on failure (empty on success)
     * @return NpyMemmap; check `.valid()` — invalid when error_out is set
     */
    static NpyMemmap load_memmapped(const std::string& path, std::string& error_out);
};

} // namespace Import
