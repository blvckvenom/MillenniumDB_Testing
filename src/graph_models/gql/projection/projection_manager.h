#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GQL {

class ProjectionCatalog;
class ProjectionStorage;

/**
 * @brief Summary information about a graph projection.
 *
 * Lightweight struct returned by ProjectionManager::list_projections()
 * containing essential metadata without loading full projection.
 */
struct ProjectionInfo {
    std::string name;              ///< Projection name (user-provided)
    uint64_t node_count;           ///< Number of nodes in projection
    uint64_t edge_count;           ///< Number of edges in projection
    uint64_t creation_timestamp;   ///< Unix timestamp when created
    std::string directory;         ///< Full path to projection directory
};

/**
 * @brief Thread-safe singleton manager for graph projections.
 *
 * Provides centralized lifecycle management for all projections in a database.
 * Handles creation, discovery, access, and deletion of projection directories.
 *
 * ## Directory Structure
 *
 * ```
 * <db_folder>/
 * └── projections/
 *     ├── my_projection/
 *     │   ├── catalog.dat
 *     │   └── *.btree files
 *     └── another_projection/
 *         └── ...
 * ```
 *
 * ## Thread Safety
 *
 * All public methods are thread-safe via internal mutex. Multiple queries
 * can safely access different projections concurrently.
 *
 * ## Usage
 *
 * ```cpp
 * auto& manager = ProjectionManager::get_instance();
 * manager.init("/path/to/db");
 *
 * // Create new projection
 * std::string proj_dir = manager.create_projection("my_proj");
 *
 * // List existing projections
 * for (const auto& info : manager.list_projections()) {
 *     std::cout << info.name << ": " << info.node_count << " nodes\n";
 * }
 * ```
 *
 * @see ProjectionStorage for actual index management
 * @see ProjectionCatalog for metadata persistence
 */
class ProjectionManager {
public:
    /**
     * @brief Gets singleton instance.
     * @return Reference to global ProjectionManager
     */
    static ProjectionManager& get_instance();

    /**
     * @brief Initializes manager for a database.
     *
     * Creates projections directory if needed and scans for existing projections.
     *
     * @param db_folder Root database directory
     */
    void init(const std::string& db_folder);

    /**
     * @brief Creates a new projection directory.
     * @param projection_name Unique name for the projection
     * @return Full path to created projection directory
     * @throws std::runtime_error if projection already exists
     */
    std::string create_projection(const std::string& projection_name);

    /**
     * @brief Checks if a projection exists.
     * @param projection_name Name to check
     * @return true if projection exists and is registered
     */
    bool projection_exists(const std::string& projection_name) const;

    /**
     * @brief Gets directory path for a projection.
     * @param projection_name Projection name
     * @return Full path to projection directory
     * @throws std::runtime_error if projection doesn't exist
     */
    std::string get_projection_dir(const std::string& projection_name) const;

    /**
     * @brief Lists all available projections.
     * @return Vector of ProjectionInfo for each discovered projection
     */
    std::vector<ProjectionInfo> list_projections() const;

    /**
     * @brief Loads catalog for a projection.
     * @param projection_name Projection to load
     * @return Unique pointer to loaded ProjectionCatalog
     */
    std::unique_ptr<ProjectionCatalog> get_projection_catalog(const std::string& projection_name) const;

    /**
     * @brief Deletes a projection and all its data.
     * @param projection_name Projection to delete
     * @return true if successfully deleted, false if not found
     */
    bool drop_projection(const std::string& projection_name);

    /// @brief Returns database root folder path
    const std::string& get_db_folder() const { return db_folder; }

    /**
     * @brief Rescans projection directory to refresh cache.
     *
     * Call after external modifications or to discover newly created
     * projections that aren't yet in the cache.
     *
     * Thread-safe: Acquires mutex internally.
     */
    void scan_projections();

private:
    ProjectionManager() = default;

    /**
     * @brief Internal implementation of scan_projections (no locking).
     *
     * Called by init() (which already holds lock) and scan_projections().
     * Must only be called while holding the mutex.
     */
    void scan_projections_internal();
    ~ProjectionManager() = default;

    /// @brief Deleted copy constructor (singleton)
    ProjectionManager(const ProjectionManager&) = delete;
    /// @brief Deleted assignment operator (singleton)
    ProjectionManager& operator=(const ProjectionManager&) = delete;

    /// @brief Recursively removes a directory and its contents
    void remove_directory(const std::string& path);

    std::string db_folder;         ///< Database root directory
    std::string projections_root;  ///< Path to projections/ subdirectory

    mutable std::mutex mutex;      ///< Thread synchronization mutex
    std::unordered_map<std::string, std::string> projection_dirs;  ///< name → directory path cache
};

} // namespace GQL
