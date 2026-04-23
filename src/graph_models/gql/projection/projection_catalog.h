#pragma once

#include <cstdint>
#include <fstream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace GQL {

// Forward-declared to avoid pulling in native_projection_builder.h (and its
// transitive dependencies) through every catalog consumer. The full enum is
// defined in graph_models/gql/projection/index_set.h; projection_catalog.cc
// includes it directly. The underlying type must match the real definition.
enum class IndexSet : uint8_t;

/**
 * @brief Persistent metadata catalog for graph projections.
 *
 * Stores and persists projection configuration, statistics, and timing
 * information. Each projection has its own catalog file that tracks:
 *
 * ## Stored Metadata
 *
 * - **Identity**: Name, creation timestamp, original query
 * - **Statistics**: Node/edge counts, label counts
 * - **Feature Flags**: Which optional indexes were created
 * - **Property Lists**: Which specific properties were included
 * - **Timing**: How long projection took to build
 *
 * ## File Format
 *
 * Binary format with magic number `{0x10, 0x0D, 0xEC, 0xAD, 0xE5, 0xDB}`
 * for file type identification and version tracking for backward compatibility.
 *
 * ## Versioning
 *
 * | Version | Changes |
 * |---------|---------|
 * | 1.0 | Initial format |
 * | 1.1 | Added optional labels/properties support |
 * | 1.2 | Added property key mappings (index-based, DEPRECATED) |
 * | 1.3 | Fixed key mappings to persist actual IDs (not indices) |
 * | 1.4 | Added IndexSet preset byte (Spec #3 T3.6) |
 *
 * @see ProjectionStorage for the actual index storage
 * @see ProjectionManager for projection lifecycle management
 */
class ProjectionCatalog {
public:
    /// @name Format Constants
    /// @{
    static constexpr uint8_t MAJOR_VERSION = 1;    ///< Catalog format major version
    static constexpr uint8_t MINOR_VERSION = 4;    ///< Minor version (1.4 adds IndexSet preset byte)
    static constexpr uint8_t magic_number[] = {0x10, 0x0D, 0xEC, 0xAD, 0xE5, 0xDB};  ///< File type identifier
    static constexpr uint8_t MODEL_ID = 255;       ///< Special ID distinguishing from GQL/RDF catalogs
    /// @}

    /**
     * @brief Constructs catalog for a projection directory.
     * @param projection_dir Full path to projection folder
     */
    ProjectionCatalog(const std::string& projection_dir);

    ~ProjectionCatalog();

    /**
     * @brief Prints human-readable catalog summary.
     * @param os Output stream for formatted output
     */
    void print(std::ostream& os) const;

    /// @brief Persists catalog to disk (projection_dir/catalog.dat)
    void save();

    /// @brief Loads catalog from disk
    void load();

    /**
     * @brief Registers a node property key in the projection.
     * @param key_name Property name (e.g., "_count")
     * @param key_id The numeric key ID (without MASK)
     */
    void add_node_key(const std::string& key_name, uint64_t key_id);

    /**
     * @brief Registers an edge property key in the projection.
     * @param key_name Property name (e.g., "_count")
     * @param key_id The numeric key ID (without MASK)
     */
    void add_edge_key(const std::string& key_name, uint64_t key_id);

    /**
     * @brief Looks up a node property key by name.
     * @param key_name Property name to look up
     * @return Key ID if found, 0 if not found
     */
    uint64_t get_node_key_id(const std::string& key_name) const;

    /**
     * @brief Looks up an edge property key by name.
     * @param key_name Property name to look up
     * @return Key ID if found, 0 if not found
     */
    uint64_t get_edge_key_id(const std::string& key_name) const;

    /// @name Projection Identity
    /// @{
    std::string projection_name;         ///< User-provided projection name
    uint64_t creation_timestamp = 0;     ///< Unix timestamp of creation
    /// @}

    /// @name Graph Statistics
    /// @{
    uint64_t node_count = 0;             ///< Total nodes in projection
    uint64_t edge_count = 0;             ///< Total edge entries
    uint64_t directed_edge_count = 0;    ///< Count of directed edges
    uint64_t undirected_edge_count = 0;  ///< Count of undirected edges
    /// @}

    /// @name Feature Flags
    /// @brief Indicate which optional indexes exist in this projection.
    /// @{
    bool includes_node_labels = false;      ///< Node label indexes created
    bool includes_edge_labels = false;      ///< Edge label indexes created
    bool includes_node_properties = false;  ///< Node property indexes created
    bool includes_edge_properties = false;  ///< Edge property indexes created
    /// @}

    /// @name Legacy Flags (Deprecated)
    /// @brief Maintained for backward compatibility with older projections.
    /// @{
    bool has_node_properties = false;       ///< @deprecated Use includes_node_properties
    bool has_edge_properties = false;       ///< @deprecated Use includes_edge_properties
    bool undirected_relationships = false;  ///< @deprecated Check undirected_edge_count > 0
    /// @}

    /// @name Property Metadata
    /// @brief Lists which specific properties were included (empty = all).
    /// @{
    std::vector<std::string> included_node_properties;  ///< Node property names (empty = all)
    std::vector<std::string> included_edge_properties;  ///< Edge property names (empty = all)
    /// @}

    /// @name Legacy Property Names (Deprecated)
    /// @{
    std::vector<std::string> node_property_names;  ///< @deprecated Use included_node_properties
    std::vector<std::string> edge_property_names;  ///< @deprecated Use included_edge_properties
    /// @}

    /// @name Label Statistics
    /// @{
    uint64_t distinct_node_labels = 0;  ///< Number of unique node labels
    uint64_t distinct_edge_labels = 0;  ///< Number of unique edge labels
    /// @}

    /// @name Index Materialization (v1.4+)
    /// @brief Preset chosen at build time controlling which B+Tree indexes
    /// were materialized. Consumed by the query layer (T3.9) to raise a
    /// descriptive error when a query requires an index that wasn't built.
    /// For v1.3 and earlier catalogs, the reader defaults this to IndexSet::ALL
    /// (the historical behavior before Spec #3).
    /// @{
    IndexSet index_set = static_cast<IndexSet>(0);  ///< IndexSet::ALL (fwd-declared)
    /// @}

    /// @name Debug Information
    /// @{
    std::string original_query;    ///< GQL query that created this projection
    uint64_t projection_millis = 0; ///< Time to build projection (milliseconds)
    /// @}

    /// @name Property Key Mappings (v1.2+)
    /// @brief Projection-specific key name ↔ ID mappings.
    ///
    /// These are essential for synthetic properties like `_count` that only
    /// exist in projections, not in the main catalog. When a query accesses
    /// `e._count` on a projection, these mappings translate the name to the
    /// correct key_id for B+Tree lookups.
    /// @{
    std::vector<std::string> node_keys_str;                                ///< Index → key name
    std::unordered_map<std::string, uint64_t> node_keys2id;                ///< Key name → index
    std::vector<std::string> edge_keys_str;                                ///< Index → key name
    std::unordered_map<std::string, uint64_t> edge_keys2id;                ///< Key name → index
    /// @}

private:
    std::string catalog_path;  ///< Full path to catalog.dat file

    /// @name Binary I/O Helpers
    /// @{
    uint8_t read_uint8(std::fstream& file);                        ///< Read single byte
    uint32_t read_uint32(std::fstream& file);                      ///< Read 32-bit unsigned
    uint64_t read_uint64(std::fstream& file);                      ///< Read 64-bit unsigned
    std::string read_string(std::fstream& file);                   ///< Read length-prefixed string
    std::vector<std::string> read_strvec(std::fstream& file);      ///< Read string vector

    void write_uint8(std::fstream& file, uint8_t value);           ///< Write single byte
    void write_uint32(std::fstream& file, uint32_t value);         ///< Write 32-bit unsigned
    void write_uint64(std::fstream& file, uint64_t value);         ///< Write 64-bit unsigned
    void write_string(std::fstream& file, const std::string& str); ///< Write length-prefixed string
    void write_strvec(std::fstream& file, const std::vector<std::string>& vec);  ///< Write string vector
    /// @}
};

} // namespace GQL
