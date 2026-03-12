#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "storage/catalog/catalog.h"

#ifdef ENABLE_GNN
#include "storage/index/hnsw/hnsw_index_manager.h"
#endif

/**
 * @brief Persistent metadata catalog for GQL databases.
 *
 * Stores and persists database statistics and name-to-ID mappings for the GQL model.
 * Inherits from Catalog base class for disk persistence.
 *
 * ## Stored Metadata
 *
 * ### Statistics
 * - Node and edge counts (directed and undirected separately)
 * - Label and property key counts
 * - Self-loop counts (equal_*_edges)
 * - Per-label and per-key occurrence counts
 *
 * ### Name Mappings
 * Bidirectional mappings for efficient lookup:
 * - Label names ↔ Label IDs (for both nodes and edges)
 * - Property key names ↔ Key IDs (for both nodes and edges)
 *
 * ## Persistence
 *
 * The catalog is saved to `<db_folder>/catalog.dat` and loaded on database open.
 * Use save() to persist changes after modifications.
 *
 * @see GQLModel for the storage model that uses this catalog
 * @see Catalog for base class persistence interface
 */
class GQLCatalog : public Catalog {
public:
    static constexpr uint8_t MODEL_ID = 2;         ///< GQL model identifier (distinguishes from RDF=0, QM=1)
    static constexpr uint8_t MAJOR_VERSION = 1;    ///< Catalog format major version
    static constexpr uint8_t MINOR_VERSION = 3;    ///< Catalog format minor version (3 = feature names + HNSW metadata)

    /**
     * @brief Constructs catalog from file.
     * @param filename Path to catalog file (creates if doesn't exist)
     */
    GQLCatalog(const std::string& filename);

    ~GQLCatalog();

    /**
     * @brief Prints catalog statistics to output stream.
     * @param os Output stream for formatted statistics
     */
    void print(std::ostream& os);

    /**
     * @brief Persists catalog to disk.
     * @note Call after any modifications to ensure durability
     */
    void save();

    /**
     * @brief Converts ID→count map to vector of string representations.
     * @param map Map of ID to occurrence count
     * @return Vector of formatted strings for each entry
     */
    std::vector<std::string> convert_map_to_vec(boost::unordered_flat_map<uint64_t, uint64_t> map);

    /// @name Graph Statistics
    /// @{
    uint64_t nodes_count = 0;                    ///< Total number of nodes
    uint64_t directed_edges_count = 0;           ///< Number of directed edges
    uint64_t undirected_edges_count = 0;         ///< Number of undirected edges
    uint64_t node_labels_count = 0;              ///< Number of distinct node labels
    uint64_t edge_labels_count = 0;              ///< Number of distinct edge labels
    uint64_t node_properties_count = 0;          ///< Number of distinct node property keys
    uint64_t edge_properties_count = 0;          ///< Number of distinct edge property keys
    uint64_t equal_directed_edges_count = 0;     ///< Directed self-loops (from == to)
    uint64_t equal_undirected_edges_count = 0;   ///< Undirected self-loops
    /// @}

    /// @name Per-Label Statistics
    /// @{
    boost::unordered_flat_map<uint64_t, uint64_t> node_label2total_count;  ///< label_id → node count with that label
    boost::unordered_flat_map<uint64_t, uint64_t> edge_label2total_count;  ///< label_id → edge count with that label
    /// @}

    /// @name Per-Property Statistics
    /// @{
    boost::unordered_flat_map<uint64_t, uint64_t> node_key2total_count;    ///< key_id → node count with that property
    boost::unordered_flat_map<uint64_t, uint64_t> edge_key2total_count;    ///< key_id → edge count with that property
    /// @}

    /// @name Node Label Mappings
    /// @{
    std::vector<std::string> node_labels_str;                              ///< Index → label name
    boost::unordered_flat_map<std::string, uint64_t> node_labels2id;       ///< Label name → index
    /// @}

    /// @name Edge Label Mappings
    /// @{
    std::vector<std::string> edge_labels_str;                              ///< Index → label name
    boost::unordered_flat_map<std::string, uint64_t> edge_labels2id;       ///< Label name → index
    /// @}

    /// @name Node Property Key Mappings
    /// @{
    std::vector<std::string> node_keys_str;                                ///< Index → key name
    boost::unordered_flat_map<std::string, uint64_t> node_keys2id;         ///< Key name → index
    /// @}

    /// @name Edge Property Key Mappings
    /// @{
    std::vector<std::string> edge_keys_str;                                ///< Index → key name
    boost::unordered_flat_map<std::string, uint64_t> edge_keys2id;         ///< Key name → index
    /// @}

    /// @name GNN Feature Registry
    /// @{
    /// Names of FeatureMatrix files stored under <db_folder>/gnn_features/.
    /// Each name (e.g. "node_features") maps to files:
    ///   <db_folder>/gnn_features/<name>.fmat  (FeatureMatrix)
    ///   <db_folder>/gnn_features/<name>.rmap  (RowMapping)
    std::vector<std::string> gnn_feature_names;

    /// Register a GNN feature name with validation and duplicate prevention.
    /// @param name Feature name (must be a safe filesystem name)
    /// @return true if newly registered, false if already present
    /// @throws std::runtime_error if name fails validation
    bool register_gnn_feature(const std::string& name);

    /// Convenience: true if any GNN feature matrices are registered.
    bool has_gnn_features() const { return !gnn_feature_names.empty(); }
    /// @}

#ifdef ENABLE_GNN
    /// @name HNSW Index Management for GNN Embeddings
    /// @{
    HNSW::HNSWIndexManager hnsw_index_manager;  ///< Manages HNSW indexes over GNN embeddings
    /// @}
#endif
};
