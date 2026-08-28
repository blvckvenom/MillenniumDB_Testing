#pragma once

#include <stdexcept>
#include <string>

namespace mdb::gnn {

/// Validate that a name is safe for use in filesystem paths.
/// Rejects empty names, ".", "..", names with path separators, null bytes, and control characters.
inline void validate_safe_name(const std::string& name, const std::string& param_name) {
    if (name.empty()) {
        throw std::runtime_error(param_name + " cannot be empty");
    }
    if (name == "." || name == "..") {
        throw std::runtime_error(param_name + " cannot be '.' or '..': '" + name + "'");
    }
    for (char c : name) {
        if (c == '/' || c == '\\' || c == '\0') {
            throw std::runtime_error(
                param_name + " contains invalid character: '" + name + "'");
        }
        if (static_cast<unsigned char>(c) < 0x20) {
            throw std::runtime_error(
                param_name + " contains control character (byte " +
                std::to_string(static_cast<unsigned char>(c)) + "): '" + name + "'");
        }
    }
}

} // namespace mdb::gnn
