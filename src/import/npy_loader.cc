#include "npy_loader.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <npy.hpp>

namespace Import {

// --- NpyMemmap RAII ---

NpyMemmap::~NpyMemmap() {
    if (mmap_ptr_ != nullptr && mmap_size_ > 0) {
        ::munmap(mmap_ptr_, mmap_size_);
    }
}

NpyMemmap::NpyMemmap(NpyMemmap&& other) noexcept
    : mmap_ptr_(other.mmap_ptr_),
      mmap_size_(other.mmap_size_),
      data_(other.data_),
      data_bytes_(other.data_bytes_),
      metadata_(std::move(other.metadata_)) {
    other.mmap_ptr_   = nullptr;
    other.mmap_size_  = 0;
    other.data_       = nullptr;
    other.data_bytes_ = 0;
}

NpyMemmap& NpyMemmap::operator=(NpyMemmap&& other) noexcept {
    if (this != &other) {
        if (mmap_ptr_ != nullptr && mmap_size_ > 0) {
            ::munmap(mmap_ptr_, mmap_size_);
        }
        mmap_ptr_   = other.mmap_ptr_;
        mmap_size_  = other.mmap_size_;
        data_       = other.data_;
        data_bytes_ = other.data_bytes_;
        metadata_   = std::move(other.metadata_);
        other.mmap_ptr_   = nullptr;
        other.mmap_size_  = 0;
        other.data_       = nullptr;
        other.data_bytes_ = 0;
    }
    return *this;
}

std::vector<float> NpyLoader::load_float32(
    const std::string& path,
    NpyMetadata& metadata_out
) {
    std::vector<unsigned long> shape;
    bool fortran_order = false;
    std::vector<float> data;

    try {
        npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    } catch (const std::exception& e) {
        std::cerr << "NpyLoader::load_float32 error for '" << path << "': " << e.what() << "\n";
        return {};
    }

    // Convert shape to uint64_t
    metadata_out.shape.clear();
    metadata_out.shape.reserve(shape.size());
    for (auto dim : shape) {
        metadata_out.shape.push_back(static_cast<uint64_t>(dim));
    }
    metadata_out.is_float64 = false;
    metadata_out.fortran_order = fortran_order;

    return data;
}

std::vector<double> NpyLoader::load_float64(
    const std::string& path,
    NpyMetadata& metadata_out
) {
    std::vector<unsigned long> shape;
    bool fortran_order = false;
    std::vector<double> data;

    try {
        npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    } catch (const std::exception& e) {
        std::cerr << "NpyLoader::load_float64 error for '" << path << "': " << e.what() << "\n";
        return {};
    }

    // Convert shape to uint64_t
    metadata_out.shape.clear();
    metadata_out.shape.reserve(shape.size());
    for (auto dim : shape) {
        metadata_out.shape.push_back(static_cast<uint64_t>(dim));
    }
    metadata_out.is_float64 = true;
    metadata_out.fortran_order = fortran_order;

    return data;
}

int NpyLoader::get_dtype_itemsize(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return 0;
    }
    try {
        std::string header_s = npy::read_header(file);
        npy::header_t header = npy::parse_header(header_s);
        if (header.dtype.kind == 'f' && (header.dtype.itemsize == 4 || header.dtype.itemsize == 8)) {
            return header.dtype.itemsize;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NpyLoader::get_dtype_itemsize error for '" << path << "': " << e.what() << "\n";
        return 0;
    }
}

bool NpyLoader::validate(const std::string& path, std::string& error_out) {
    // Check file exists
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error_out = "Cannot open file: " + path;
        return false;
    }

    try {
        // Read and parse header using libnpy
        std::string header_s = npy::read_header(file);
        npy::header_t header = npy::parse_header(header_s);

        // Check dtype - must be float32 or float64
        if (header.dtype.kind != 'f') {
            error_out = "Unsupported dtype kind '" + std::string(1, header.dtype.kind) +
                       "'. Expected 'f' (float). Use float32 or float64.";
            return false;
        }

        if (header.dtype.itemsize != 4 && header.dtype.itemsize != 8) {
            error_out = "Unsupported itemsize " + std::to_string(header.dtype.itemsize) +
                       ". Expected 4 (float32) or 8 (float64).";
            return false;
        }

        // Check byte order - accept little-endian or native only
        if (header.dtype.byteorder == '>') {
            error_out = "Big-endian byte order not supported. Convert to little-endian.";
            return false;
        }

        // Check dimensionality - must be 2D for embedding matrices
        if (header.shape.size() != 2) {
            error_out = "Expected 2D array (num_nodes, embedding_dim), got " +
                       std::to_string(header.shape.size()) + "D";
            return false;
        }

        // Check for reasonable dimensions
        if (header.shape[0] == 0 || header.shape[1] == 0) {
            error_out = "Array has zero dimension: shape=(" +
                       std::to_string(header.shape[0]) + ", " +
                       std::to_string(header.shape[1]) + ")";
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        error_out = "Invalid NPY format: " + std::string(e.what());
        return false;
    }
}

size_t NpyLoader::get_file_size(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return 0;
    }
    return static_cast<size_t>(file.tellg());
}

NpyMemmap NpyLoader::load_memmapped(const std::string& path, std::string& error_out) {
    error_out.clear();
    NpyMemmap out;

    // Parse header via ifstream so we learn shape + dtype + data offset.
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error_out = "Cannot open file: " + path;
        return out;
    }

    npy::header_t header;
    try {
        std::string header_s = npy::read_header(file);
        header = npy::parse_header(header_s);
    } catch (const std::exception& e) {
        error_out = std::string("Invalid NPY header: ") + e.what();
        return out;
    }

    if (header.dtype.kind != 'f' || (header.dtype.itemsize != 4 && header.dtype.itemsize != 8)) {
        error_out = "Unsupported dtype (expected float32/float64)";
        return out;
    }
    if (header.dtype.byteorder == '>') {
        error_out = "Big-endian byte order not supported";
        return out;
    }
    if (header.fortran_order) {
        error_out = "Fortran-order .npy not supported for memmap streaming "
                    "(transposing would defeat streaming). Convert to C-order first.";
        return out;
    }

    const std::streamoff data_offset = file.tellg();
    if (data_offset < 0) {
        error_out = "Failed to determine data offset after header parse";
        return out;
    }
    file.close();

    // Compute expected data size from shape + itemsize, detect truncation.
    uint64_t num_elems = 1;
    for (auto dim : header.shape) {
        if (dim == 0) {
            error_out = "Array has zero dimension";
            return out;
        }
        if (num_elems > UINT64_MAX / dim) {
            error_out = "Array size overflow";
            return out;
        }
        num_elems *= dim;
    }
    const size_t expected_data_bytes = static_cast<size_t>(num_elems) * header.dtype.itemsize;

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error_out = std::string("open() failed: ") + std::strerror(errno);
        return out;
    }
    struct stat st{};
    if (::fstat(fd, &st) < 0) {
        const int e = errno;
        ::close(fd);
        error_out = std::string("fstat() failed: ") + std::strerror(e);
        return out;
    }
    const size_t file_size = static_cast<size_t>(st.st_size);
    if (static_cast<size_t>(data_offset) + expected_data_bytes > file_size) {
        ::close(fd);
        error_out = "NPY file truncated: header claims more data than file holds";
        return out;
    }

    void* mmap_ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // mmap holds its own reference; fd can be closed.
    if (mmap_ptr == MAP_FAILED) {
        error_out = std::string("mmap() failed: ") + std::strerror(errno);
        return out;
    }
    ::madvise(mmap_ptr, file_size, MADV_SEQUENTIAL);

    out.mmap_ptr_   = mmap_ptr;
    out.mmap_size_  = file_size;
    out.data_       = static_cast<const char*>(mmap_ptr) + data_offset;
    out.data_bytes_ = expected_data_bytes;
    out.metadata_.shape.clear();
    out.metadata_.shape.reserve(header.shape.size());
    for (auto dim : header.shape) {
        out.metadata_.shape.push_back(static_cast<uint64_t>(dim));
    }
    out.metadata_.is_float64    = (header.dtype.itemsize == 8);
    out.metadata_.fortran_order = false;
    return out;
}

} // namespace Import
