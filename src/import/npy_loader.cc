#include "npy_loader.h"

#include <fstream>
#include <iostream>
#include <npy.hpp>

namespace Import {

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

} // namespace Import
