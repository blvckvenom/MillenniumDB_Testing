#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GQL {

class ProjectionCatalog;
class ProjectionStorage;

struct ProjectionInfo {
    std::string name;
    uint64_t node_count;
    uint64_t edge_count;
    uint64_t creation_timestamp;
    std::string directory;
};

class ProjectionManager {
public:
    static ProjectionManager& get_instance();

    // Initialize projection manager for a database
    void init(const std::string& db_folder);

    // Create a new projection (returns projection directory path)
    std::string create_projection(const std::string& projection_name);

    // Check if projection exists
    bool projection_exists(const std::string& projection_name) const;

    // Get projection directory
    std::string get_projection_dir(const std::string& projection_name) const;

    // List all projections
    std::vector<ProjectionInfo> list_projections() const;

    // Get projection catalog
    std::unique_ptr<ProjectionCatalog> get_projection_catalog(const std::string& projection_name) const;

    // Delete a projection
    bool drop_projection(const std::string& projection_name);

    // Get database folder
    const std::string& get_db_folder() const { return db_folder; }

    // Refresh projection cache by rescanning projection directory
    // Useful after creating new projections to make them immediately visible
    void scan_projections();

private:
    ProjectionManager() = default;
    ~ProjectionManager() = default;

    ProjectionManager(const ProjectionManager&) = delete;
    ProjectionManager& operator=(const ProjectionManager&) = delete;
    void remove_directory(const std::string& path);

    std::string db_folder;
    std::string projections_root;

    mutable std::mutex mutex;
    std::unordered_map<std::string, std::string> projection_dirs; // name -> directory path
};

} // namespace GQL
