#pragma once

/**
 * @file gnn_dtype.h
 * @brief Data types for GNN tensor storage.
 *
 * This file defines the data types supported by the GNN tensor storage system.
 * These types are COMPLETELY INDEPENDENT from MillenniumDB's existing tensor
 * infrastructure (TensorManager, TensorsHash, etc.).
 *
 * IMPORTANT: The GNN system must maintain its own tensor storage to avoid
 * coupling with MDB's internal tensor implementation.
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mdb::gnn {

/**
 * @brief Data types supported by GNN tensor storage.
 *
 * This enum is independent from any MDB tensor types and is specific
 * to the GNN subsystem. The values are chosen to be compact (uint8_t)
 * for efficient storage in binary formats.
 */
enum class GnnDtype : uint8_t {
    FLOAT32 = 0,   ///< 32-bit IEEE 754 floating point (most common for GNN)
    FLOAT64 = 1,   ///< 64-bit IEEE 754 floating point (high precision)
    INT32   = 2,   ///< 32-bit signed integer
    INT64   = 3,   ///< 64-bit signed integer
    UINT8   = 4,   ///< 8-bit unsigned integer (labels, masks)
    BOOL    = 5,   ///< Boolean (stored as 1 byte)

    // Sentinel: update this when adding new dtypes above.
    // Used by FeatureMatrixHeader::is_valid() for range validation.
    MAX_VALUE = BOOL,
};

/**
 * @brief Returns the size in bytes for a given GnnDtype.
 *
 * @param dtype The data type to query
 * @return Size in bytes per element
 * @throws std::invalid_argument if dtype is unknown
 */
inline size_t dtype_size(GnnDtype dtype) {
    switch (dtype) {
        case GnnDtype::FLOAT32: return 4;
        case GnnDtype::FLOAT64: return 8;
        case GnnDtype::INT32:   return 4;
        case GnnDtype::INT64:   return 8;
        case GnnDtype::UINT8:   return 1;
        case GnnDtype::BOOL:    return 1;
    }
    throw std::invalid_argument("Unknown GnnDtype value: " + std::to_string(static_cast<int>(dtype)));
}

/**
 * @brief Returns a human-readable name for a GnnDtype.
 *
 * @param dtype The data type to describe
 * @return String name of the type
 */
inline std::string dtype_name(GnnDtype dtype) {
    switch (dtype) {
        case GnnDtype::FLOAT32: return "float32";
        case GnnDtype::FLOAT64: return "float64";
        case GnnDtype::INT32:   return "int32";
        case GnnDtype::INT64:   return "int64";
        case GnnDtype::UINT8:   return "uint8";
        case GnnDtype::BOOL:    return "bool";
    }
    throw std::invalid_argument("Unknown GnnDtype value: " + std::to_string(static_cast<int>(dtype)));
}

/**
 * @brief Parse a dtype from its string name.
 *
 * @param name String name (e.g., "float32", "int64")
 * @return Corresponding GnnDtype
 * @throws std::invalid_argument if name is not recognized
 */
inline GnnDtype dtype_from_name(const std::string& name) {
    if (name == "float32" || name == "f32") return GnnDtype::FLOAT32;
    if (name == "float64" || name == "f64") return GnnDtype::FLOAT64;
    if (name == "int32" || name == "i32")   return GnnDtype::INT32;
    if (name == "int64" || name == "i64")   return GnnDtype::INT64;
    if (name == "uint8" || name == "u8")    return GnnDtype::UINT8;
    if (name == "bool")                     return GnnDtype::BOOL;
    throw std::invalid_argument("Unknown dtype name: " + name);
}

/**
 * @brief Check if a dtype is a floating-point type.
 */
inline bool is_floating_point(GnnDtype dtype) {
    return dtype == GnnDtype::FLOAT32 || dtype == GnnDtype::FLOAT64;
}

/**
 * @brief Check if a dtype is an integer type.
 */
inline bool is_integer(GnnDtype dtype) {
    return dtype == GnnDtype::INT32 || dtype == GnnDtype::INT64 ||
           dtype == GnnDtype::UINT8;
}

} // namespace mdb::gnn
