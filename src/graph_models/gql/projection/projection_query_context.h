#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace GQL {

/**
 * @brief Query execution context for graph projections.
 *
 * Manages projection storage lifetime and provides cached index pointers
 * for efficient query execution. Created when a query uses `USE GRAPH`
 * to target a projection instead of the main database.
 *
 * ## Lifetime Management
 *
 * The context owns the ProjectionStorage and keeps it alive for the
 * duration of query execution. When the context is destroyed (typically
 * at query end), all B+tree resources are released.
 *
 * ## Index Access Pattern
 *
 * Raw pointers to B+tree indexes are cached at construction time for
 * fast access during query execution. This avoids repeated virtual
 * calls through the storage interface.
 *
 * ```cpp
 * // In query executor:
 * auto ctx = std::make_unique<ProjectionQueryContext>("my_proj");
 * if (ctx->is_valid()) {
 *     auto cursor = ctx->from_to_edge_index->get_cursor();
 *     // ... execute query using projection indexes
 * }
 * ```
 *
 * ## Optional Indexes
 *
 * Optional index pointers (labels, properties) may be nullptr if the
 * projection was created without those features. Query operators must
 * check for nullptr before accessing optional indexes.
 *
 * @see ProjectionStorage for index definitions
 * @see GQLModel for how projections integrate with main model
 */
class ProjectionQueryContext {
public:
    std::string projection_name;                   ///< Name of active projection
    std::unique_ptr<ProjectionStorage> storage;    ///< Owned storage instance

    /// @name Required Index Pointers
    /// @brief Always non-null after successful construction.
    /// @{
    BPlusTree<1>* nodes_index = nullptr;           ///< Node existence check
    BPlusTree<3>* from_to_edge_index = nullptr;    ///< Outgoing edge traversal
    BPlusTree<3>* to_from_edge_index = nullptr;    ///< Incoming edge traversal
    BPlusTree<2>* edge_direction_index = nullptr;  ///< Edge direction lookup
    BPlusTree<3>* edge_from_to_index = nullptr;    ///< Edge-first lookup (directed)
    BPlusTree<3>* edge_n1_n2_index = nullptr;      ///< Edge-first lookup (undirected)
    /// @}

    /// @name Optional Label Index Pointers
    /// @brief nullptr if projection created without INCLUDE LABELS.
    /// @{
    BPlusTree<2>* node_label_index = nullptr;   ///< Node → labels
    BPlusTree<2>* label_node_index = nullptr;   ///< Label → nodes
    BPlusTree<2>* edge_label_index = nullptr;   ///< Edge → labels
    BPlusTree<2>* label_edge_index = nullptr;   ///< Label → edges
    /// @}

    /// @name Optional Property Index Pointers
    /// @brief nullptr if projection created without INCLUDE PROPERTIES.
    /// @{
    BPlusTree<3>* node_key_value_index = nullptr;  ///< Node property lookup
    BPlusTree<3>* key_value_node_index = nullptr;  ///< Property → nodes
    BPlusTree<3>* edge_key_value_index = nullptr;  ///< Edge property lookup
    BPlusTree<3>* key_value_edge_index = nullptr;  ///< Property → edges
    /// @}

    /// @name Property Key Mappings
    /// @brief Projection-specific key name → ID mappings for synthetic properties.
    /// @{
    std::unordered_map<std::string, uint64_t> node_keys2id;  ///< Node key name → index
    std::unordered_map<std::string, uint64_t> edge_keys2id;  ///< Edge key name → index
    /// @}

    /**
     * @brief Constructs context and opens projection for reading.
     *
     * Loads projection from disk and caches all index pointers.
     * After construction, call is_valid() to verify required indexes loaded.
     *
     * @param proj_name Name of projection to open
     * @throws std::runtime_error if projection doesn't exist
     */
    explicit ProjectionQueryContext(const std::string& proj_name)
        : projection_name(proj_name)
    {
        auto& manager = ProjectionManager::get_instance();
        std::string proj_dir = manager.get_projection_dir(proj_name);
        std::string db_folder = manager.get_db_folder();

        storage = std::make_unique<ProjectionStorage>(proj_dir, db_folder);
        storage->open();  // Open existing projection

        // Cache required index pointers for fast access during query execution
        nodes_index = storage->get_nodes_index();
        from_to_edge_index = storage->get_from_to_edge_index();
        to_from_edge_index = storage->get_to_from_edge_index();
        edge_direction_index = storage->get_edge_direction_index();
        edge_from_to_index = storage->get_edge_from_to_index();
        edge_n1_n2_index = storage->get_edge_n1_n2_index();

        // Cache optional label indexes (may be nullptr if projection doesn't include labels)
        node_label_index = storage->get_node_label_index();
        label_node_index = storage->get_label_node_index();
        edge_label_index = storage->get_edge_label_index();
        label_edge_index = storage->get_label_edge_index();

        // Cache optional property indexes (may be nullptr if projection doesn't include properties)
        node_key_value_index = storage->get_node_key_value_index();
        key_value_node_index = storage->get_key_value_node_index();
        edge_key_value_index = storage->get_edge_key_value_index();
        key_value_edge_index = storage->get_key_value_edge_index();

        // Load projection-specific key mappings from catalog (for synthetic properties like _count)
        ProjectionCatalog catalog(proj_dir);
        node_keys2id = catalog.node_keys2id;
        edge_keys2id = catalog.edge_keys2id;
    }

    ~ProjectionQueryContext() = default;

    /**
     * @brief Checks if context loaded successfully.
     *
     * Verifies all required indexes are non-null. Should be called
     * after construction before using any index pointers.
     *
     * @return true if all required indexes are available
     */
    bool is_valid() const {
        return nodes_index != nullptr &&
               from_to_edge_index != nullptr &&
               to_from_edge_index != nullptr &&
               edge_direction_index != nullptr;
    }
};

} // namespace GQL
