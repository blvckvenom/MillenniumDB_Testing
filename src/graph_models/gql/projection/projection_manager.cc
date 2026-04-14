#include "projection_manager.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>

#include "projection_catalog.h"

namespace fs = std::filesystem;

namespace GQL {

ProjectionManager& ProjectionManager::get_instance() {
    static ProjectionManager instance;
    return instance;
}

void ProjectionManager::init(const std::string& db_folder_) {
    std::lock_guard<std::mutex> lock(mutex);

    db_folder = db_folder_;
    projections_root = db_folder + "/projections";

    // Create projections directory if it doesn't exist
    if (!fs::exists(projections_root)) {
        fs::create_directories(projections_root);
    }

    // Scan existing projections
    scan_projections();
}

void ProjectionManager::scan_projections() {
    projection_dirs.clear();

    if (!fs::exists(projections_root)) {
        return;
    }

    for (const auto& entry : fs::directory_iterator(projections_root)) {
        if (entry.is_directory()) {
            std::string proj_name = entry.path().filename().string();
            std::string proj_dir = entry.path().string();

            // Verify it has a catalog file
            if (fs::exists(proj_dir + "/catalog.dat")) {
                projection_dirs[proj_name] = proj_dir;
            }
        }
    }
}

std::string ProjectionManager::create_projection(const std::string& projection_name) {
    std::lock_guard<std::mutex> lock(mutex);

    if (projection_dirs.find(projection_name) != projection_dirs.end()) {
        throw std::runtime_error("Projection '" + projection_name + "' already exists");
    }

    std::string proj_dir = projections_root + "/" + projection_name;

    // Create projection directory (create_directories returns false if already exists, which is fine)
    try {
        fs::create_directories(proj_dir);
    } catch (const fs::filesystem_error& e) {
        throw std::runtime_error("Failed to create projection directory: " + proj_dir + " - " + e.what());
    }

    // Verify the directory exists
    if (!fs::exists(proj_dir) || !fs::is_directory(proj_dir)) {
        throw std::runtime_error("Projection directory does not exist after creation: " + proj_dir);
    }

    projection_dirs[projection_name] = proj_dir;

    return proj_dir;
}

bool ProjectionManager::projection_exists(const std::string& projection_name) const {
    std::lock_guard<std::mutex> lock(mutex);
    return projection_dirs.find(projection_name) != projection_dirs.end();
}

std::string ProjectionManager::get_projection_dir(const std::string& projection_name) const {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = projection_dirs.find(projection_name);
    if (it == projection_dirs.end()) {
        throw std::runtime_error("Projection '" + projection_name + "' not found");
    }

    return it->second;
}

std::vector<ProjectionInfo> ProjectionManager::list_projections() const {
    std::lock_guard<std::mutex> lock(mutex);

    std::vector<ProjectionInfo> result;

    for (const auto& [name, dir] : projection_dirs) {
        try {
            ProjectionCatalog catalog(dir);

            ProjectionInfo info;
            info.name = catalog.projection_name;
            info.node_count = catalog.node_count;
            info.edge_count = catalog.edge_count;
            info.creation_timestamp = catalog.creation_timestamp;
            info.directory = dir;

            result.push_back(info);
        } catch (const std::exception& e) {
            // Skip projections with corrupted catalogs
            continue;
        }
    }

    return result;
}

std::unique_ptr<ProjectionCatalog> ProjectionManager::get_projection_catalog(
    const std::string& projection_name) const
{
    std::string proj_dir = get_projection_dir(projection_name);
    return std::make_unique<ProjectionCatalog>(proj_dir);
}

bool ProjectionManager::drop_projection(const std::string& projection_name) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = projection_dirs.find(projection_name);
    if (it == projection_dirs.end()) {
        return false;
    }

    std::string proj_dir = it->second;

    try {
        // Remove the directory and all its contents
        fs::remove_all(proj_dir);

        projection_dirs.erase(it);
        return true;
    } catch (const fs::filesystem_error& e) {
        throw std::runtime_error("Failed to delete projection: " + std::string(e.what()));
    }
}

} // namespace GQL
