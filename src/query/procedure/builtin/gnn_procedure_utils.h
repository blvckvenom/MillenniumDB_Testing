#pragma once

#include <string>

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

} // namespace GQL::Procedures
