#pragma once

#include <string>
#include <vector>

#include "system/file_manager.h"

namespace GQL::Procedures {

/// Get the database folder path with trailing slash stripped.
/// Used by all GNN procedures that access the filesystem.
inline std::string get_db_folder() {
    std::string db_folder = file_manager.get_file_path("");
    if (!db_folder.empty() && db_folder.back() == '/') {
        db_folder.pop_back();
    }
    return db_folder;
}

/// Format a "not found, available: [list]" error message.
inline std::string format_not_found_error(
    const std::string& item_type,
    const std::string& name,
    const std::vector<std::string>& available,
    const std::string& create_hint = ""
) {
    std::string msg = item_type + " '" + name + "' not found.\n\n";
    if (available.empty()) {
        msg += "No " + item_type + "s exist.";
        if (!create_hint.empty()) {
            msg += " Create one first with:\n  " + create_hint;
        }
    } else {
        msg += "Available " + item_type + "s: [";
        for (size_t i = 0; i < available.size(); i++) {
            if (i > 0) msg += ", ";
            msg += "'" + available[i] + "'";
        }
        msg += "]";
    }
    return msg;
}

} // namespace GQL::Procedures
