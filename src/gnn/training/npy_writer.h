#pragma once

/**
 * @file npy_writer.h
 * @brief Utility for exporting LibTorch tensors as NumPy .npy files.
 *
 * Supports 1D and 2D tensors in float32 and int64 dtypes.
 * Writes NumPy v1.0 format with 64-byte-aligned headers (C-order, little-endian).
 *
 * Intended use: export training outputs (embeddings, predictions, labels)
 * so Python scripts can verify GNN training results externally.
 */

#include <filesystem>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace mdb::gnn {

/**
 * @brief Exports LibTorch tensors to NumPy .npy files (v1.0 format).
 *
 * Both methods:
 *  - Move the tensor to CPU and ensure contiguous memory layout.
 *  - Create parent directories if they do not exist.
 *  - Throw std::runtime_error on dtype mismatch or I/O failure.
 *
 * NumPy v1.0 layout:
 *   6B magic ("\x93NUMPY") + 2B version (1.0) + 2B header_len (uint16 LE)
 *   + header (ASCII dict, space-padded, \n-terminated, 64B-aligned total)
 *   + raw data (C-order)
 *
 * Usage:
 * @code
 *   auto embeddings = torch::randn({100, 64});
 *   NpyWriter::write_float32("embeddings.npy", embeddings);
 *
 *   auto preds = torch::zeros({100}, torch::kInt64);
 *   NpyWriter::write_int64("predictions.npy", preds);
 * @endcode
 */
class NpyWriter {
public:
    /**
     * @brief Write a float32 tensor to a .npy file.
     *
     * @param path   Destination file path (parent dirs are created automatically).
     * @param tensor Tensor with dtype torch::kFloat32 (1D or 2D).
     * @throws std::runtime_error if dtype is not float32 or on I/O failure.
     */
    static void write_float32(const std::filesystem::path& path,
                               const torch::Tensor& tensor);

    /**
     * @brief Write an int64 tensor to a .npy file.
     *
     * @param path   Destination file path (parent dirs are created automatically).
     * @param tensor Tensor with dtype torch::kInt64 (1D or 2D).
     * @throws std::runtime_error if dtype is not int64 or on I/O failure.
     */
    static void write_int64(const std::filesystem::path& path,
                             const torch::Tensor& tensor);

private:
    /**
     * @brief Core write implementation.
     *
     * @param path   Destination file path.
     * @param tensor Tensor already on CPU and contiguous.
     * @param descr  NumPy dtype descriptor string (e.g. "<f4", "<i8").
     */
    static void write_impl(const std::filesystem::path& path,
                           const torch::Tensor& tensor,
                           const std::string& descr);
};

} // namespace mdb::gnn
