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
 * | 1.4 | Added IndexSet preset byte (selects which B+Tree indexes to materialize) |
 * | 1.5 | Added per-index leaf_format byte array (delta + LEB128-varint leaf encoding) |
 * | 1.6 | Added per-projection graphStorage byte (CSR-hybrid graph storage: edge-index B+Tree leaves store the CSR layout inline) |
 *
 * ## v1.5 additions
 *
 * After the v1.4 IndexSet preset byte, v1.5 appends a length-prefixed byte
 * array naming the on-disk leaf encoding (BPT::LeafFormat) of each
 * materialized index. The array has one entry per materialized index, in
 * the canonical ProjectionIndex single-bit enum order (NODES, NODE_LABEL,
 * LABEL_NODE, NODE_KEY_VALUE, KEY_VALUE_NODE, FROM_TO_EDGE, TO_FROM_EDGE,
 * EDGE_DIRECTION, EDGE_FROM_TO, EDGE_N1_N2, EDGE_LABEL, LABEL_EDGE,
 * EDGE_KEY_VALUE, KEY_VALUE_EDGE). Values: 1 = BITSET (legacy redundant
 * bitset encoding), 2 = DELTA_VARINT (delta + LEB128-varint v2 encoding).
 * Catalogs with MINOR < 5 are read by populating leaf_formats with
 * all-BITSET (1) for every materialized index, preserving the behavior of
 * projections built before delta + LEB128-varint leaf encoding was introduced.
 *
 * ## v1.6 additions
 *
 * After the v1.5 leaf_formats section, v1.6 appends a single per-projection
 * graph_storage byte selecting the on-disk topology representation. Values:
 * 1 = BTREE (classic per-index B+Tree leaves using leaf_format), 2 =
 * CSR_HYBRID (edge-index B+Tree leaves store the CSR layout inline so that
 * neighbor lookup is O(1) without a separate sidecar file; non-edge indexes
 * continue to use leaf_format normally). Catalogs with MINOR < 6 are read
 * by defaulting graph_storage to 1 (BTREE), preserving the behavior of
 * projections built before CSR-hybrid storage was introduced.
 *
 * @see ProjectionStorage for the actual index storage
 * @see ProjectionManager for projection lifecycle management
 */
class ProjectionCatalog {
public:
    /// @name Format Constants
    /// @{
    static constexpr uint8_t MAJOR_VERSION = 1;    ///< Catalog format major version
    /// Minor version (1.6 adds per-projection graphStorage byte for CSR-hybrid graph storage)
    static constexpr uint8_t MINOR_VERSION = 6;
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
    /// were materialized. Consumed by the query layer to raise a
    /// descriptive error when a query requires an index that wasn't built.
    /// For v1.3 and earlier catalogs, the reader defaults this to IndexSet::ALL
    /// (the historical behavior before the IndexSet preset was introduced).
    /// @{
    IndexSet index_set = static_cast<IndexSet>(0);  ///< IndexSet::ALL (fwd-declared)
    /// @}

    /// @name Per-Index Leaf Format (v1.5+)
    /// @brief Per-index LeafFormat byte (delta + LEB128-varint leaf encoding,
    /// v1.5+). Size == number of materialized indexes (those whose bit is set
    /// in index_set). Order follows the canonical ProjectionIndex enum (NODES,
    /// NODE_LABEL, LABEL_NODE, NODE_KEY_VALUE, KEY_VALUE_NODE, FROM_TO_EDGE,
    /// TO_FROM_EDGE, EDGE_DIRECTION, EDGE_FROM_TO, EDGE_N1_N2, EDGE_LABEL,
    /// LABEL_EDGE, EDGE_KEY_VALUE, KEY_VALUE_EDGE). Values: 1 = BITSET
    /// (legacy), 2 = DELTA_VARINT (delta + LEB128-varint v2 encoding). v1.4
    /// and earlier catalogs read under v1.5 code default every entry to
    /// BITSET (1). The on-disk representation is uint8_t per slot; conversion
    /// to BPT::LeafFormat happens in consumers that wire this field into
    /// BPlusTree<N>::ctor.
    /// @{
    std::vector<uint8_t> leaf_formats;
    /// @}

    /// @brief Accessor mirroring get-style helpers elsewhere in the catalog.
    /// Returns the persisted per-index leaf_format byte array (const).
    const std::vector<uint8_t>& get_leaf_formats() const { return leaf_formats; }

    /// @name Graph Storage Mode (v1.6+)
    /// @{
    /// Per-projection graph-storage mode byte (CSR-hybrid graph storage,
    /// v1.6+). Values:
    ///   1 = BTREE       — classic B+Tree leaves with per-index leaf_format
    ///                      (default, preserves behavior of projections built
    ///                      before CSR-hybrid storage was introduced).
    ///   2 = CSR_HYBRID  — edge indexes (FROM_TO_EDGE, TO_FROM_EDGE) emit
    ///                      B+Tree leaves that store the CSR layout inline,
    ///                      enabling O(1) neighbor access without a separate
    ///                      sidecar file; non-edge indexes use leaf_format
    ///                      normally.
    ///
    /// Pre-v1.6 catalogs read under v1.6 code default to BTREE (1),
    /// preserving behavior.
    uint8_t graph_storage = 1;  // default = BTREE
    /// @}

    /// @brief Accessor returning the persisted per-projection graphStorage
    /// byte. Never throws; returns the in-memory value populated by load()
    /// (or the default 1 = BTREE if no catalog has been loaded yet).
    uint8_t get_graph_storage() const noexcept { return graph_storage; }

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
