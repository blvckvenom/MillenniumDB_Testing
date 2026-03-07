#pragma once

/**
 * @file npy_loader.h
 * @brief Utility for loading NumPy .npy files containing node embeddings.
 *
 * Uses the libnpy header-only library for parsing NPY format.
 * Supports float32 and float64 arrays in C-order or Fortran-order.
 */

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
};

} // namespace Import
