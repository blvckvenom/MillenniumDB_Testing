#include "gnn/training/npy_writer.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace mdb::gnn {

// ============================================================================
// Internal helper
// ============================================================================

void NpyWriter::write_impl(const fs::path& path,
                            const torch::Tensor& tensor,
                            const std::string& descr)
{
    // Ensure the tensor is on CPU and has contiguous memory layout.
    auto t = tensor.to(torch::kCPU).contiguous();

    // -----------------------------------------------------------------------
    // Build the shape string.
    //   1D [N]     → "(N,)"         (trailing comma required by NumPy spec)
    //   2D [N, D]  → "(N, D)"
    // -----------------------------------------------------------------------
    std::string shape = "(";
    for (int64_t i = 0; i < t.dim(); i++) {
        shape += std::to_string(t.size(i));
        if (i < t.dim() - 1) {
            shape += ", ";
        } else if (t.dim() == 1) {
            shape += ",";   // trailing comma for 1-D arrays
        }
    }
    shape += ")";

    // -----------------------------------------------------------------------
    // Build the header dict string (no trailing \n yet).
    //   {'descr': '<f4', 'fortran_order': False, 'shape': (N, D), }
    // -----------------------------------------------------------------------
    std::string header_dict =
        "{'descr': '" + descr +
        "', 'fortran_order': False, 'shape': " + shape + ", }";

    // -----------------------------------------------------------------------
    // Compute padding so the full preamble is a multiple of 64 bytes.
    //
    //   preamble = 6 (magic) + 2 (version) + 2 (HEADER_LEN field) = 10 bytes
    //   unpadded = preamble + header_dict.size() + 1 (\n)
    //   padded   = next multiple of 64 >= unpadded
    //   padding  = padded - unpadded     (extra space chars before the \n)
    // -----------------------------------------------------------------------
    constexpr size_t preamble = 10;
    size_t unpadded = preamble + header_dict.size() + 1;   // +1 for '\n'
    size_t padded   = ((unpadded + 63) / 64) * 64;
    size_t padding  = padded - unpadded;

    // Full header: dict + spaces + '\n'
    std::string header = header_dict + std::string(padding, ' ') + "\n";

    // HEADER_LEN is the number of bytes in the header field only (not preamble).
    auto header_len = static_cast<uint16_t>(header.size());

    // -----------------------------------------------------------------------
    // Create parent directories and open the file.
    // -----------------------------------------------------------------------
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("NpyWriter: cannot open file for writing: " +
                                 path.string());
    }

    // -----------------------------------------------------------------------
    // Write: magic + version + header_len + header + raw data
    // -----------------------------------------------------------------------
    f.write("\x93NUMPY", 6);
    f.put(static_cast<char>(1));    // major version
    f.put(static_cast<char>(0));    // minor version

    // header_len: 2 bytes, little-endian
    f.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));

    f.write(header.data(), static_cast<std::streamsize>(header.size()));

    f.write(reinterpret_cast<const char*>(t.data_ptr()),
            static_cast<std::streamsize>(t.nbytes()));

    if (!f) {
        throw std::runtime_error("NpyWriter: I/O error while writing: " +
                                 path.string());
    }
}

// ============================================================================
// Public API
// ============================================================================

void NpyWriter::write_float32(const fs::path& path, const torch::Tensor& tensor)
{
    if (tensor.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "NpyWriter::write_float32: tensor dtype must be float32, got " +
            std::string(torch::toString(tensor.dtype().toScalarType())));
    }
    write_impl(path, tensor, "<f4");
}

void NpyWriter::write_int64(const fs::path& path, const torch::Tensor& tensor)
{
    if (tensor.dtype() != torch::kInt64) {
        throw std::runtime_error(
            "NpyWriter::write_int64: tensor dtype must be int64, got " +
            std::string(torch::toString(tensor.dtype().toScalarType())));
    }
    write_impl(path, tensor, "<i8");
}

} // namespace mdb::gnn
