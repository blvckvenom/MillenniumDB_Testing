#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"

namespace GQL {

// Forward declaration (NativeScanner implemented by another agent)
class NativeScanner;

/**
 * @brief Orchestrates native graph projection creation.
 *
 * Coordinates scanning of label_node/label_edge B+Trees and batch writing
 * to disk-based ProjectionStorage. Achieves O(n+m) complexity by avoiding
 * pattern matching queries.
 *
 * Architecture:
 *   1. NativeScanner scans source B+Trees
 *   2. Batch nodes/edges in memory (BATCH_SIZE = 1000)
 *   3. Flush batches to ProjectionStorage
 *   4. Track statistics and progress
 *
 * @see ARCHITECTURE_DESIGN.md Section 3.2 for complete design
 */
class NativeProjectionBuilder {
public:
    static constexpr size_t BATCH_SIZE = 1000;

    struct Statistics {
        uint64_t node_count = 0;
        uint64_t relationship_count = 0;
        std::chrono::milliseconds duration_ms{0};
    };

    /**
     * @brief Constructs builder for new projection.
     *
     * @param projection_name Name for the new projection
     * @param db_folder Database root folder
     * @param node_properties Optional list of node property keys to project
     * @param edge_properties Optional list of edge property keys to project
     */
    NativeProjectionBuilder(
        const std::string& projection_name,
        const std::string& db_folder,
        const std::vector<std::string>& node_properties = {},
        const std::vector<std::string>& edge_properties = {}
    );

    ~NativeProjectionBuilder();

    /**
     * @brief Scans nodes with specified labels.
     *
     * @param labels Vector of label names (e.g., ["User", "Post"])
     * @throws std::runtime_error if label doesn't exist
     */
    void scan_nodes_by_labels(const std::vector<std::string>& labels);

    /**
     * @brief Scans edges with specified relationship types.
     *
     * @param types Vector of type names (e.g., ["KNOWS", "LIKES"])
     * @throws std::runtime_error if type doesn't exist
     */
    void scan_edges_by_types(const std::vector<std::string>& types);

    /**
     * @brief Finalizes projection and flushes to disk.
     *
     * @return Statistics for the created projection
     * @throws std::runtime_error on I/O errors
     */
    Statistics finalize();

    /**
     * @brief Gets current statistics (for progress tracking).
     */
    const Statistics& get_statistics() const { return stats; }

private:
    std::string projection_name;
    std::string db_folder;
    std::unique_ptr<ProjectionStorage> storage;
    std::unique_ptr<NativeScanner> scanner;
    Statistics stats;

    // Property configuration
    std::vector<std::string> node_property_keys;
    std::vector<std::string> edge_property_keys;

    std::chrono::steady_clock::time_point start_time;

    // Batch buffers
    std::vector<ProjectedNode> node_batch;
    std::vector<ProjectedEdge> edge_batch;

    void flush_nodes();
    void flush_edges();
    void validate_label_exists(const std::string& label);
    void validate_type_exists(const std::string& type);

    // Property extraction helpers
    void extract_node_properties(ObjectId node_id);
    void extract_edge_properties(ObjectId edge_id);
};

} // namespace GQL
