#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph_models/gql/projection/bloom_filter.h"
#include "graph_models/gql/projection/streaming_record_buffer.h"
#include "graph_models/object_id.h"
#include "storage/index/record.h"

template<std::size_t N>
class BPlusTree;

namespace GQL {

// Forward declaration for streaming build
template<std::size_t N>
class ExternalRecordSort;

class ProjectionCatalog;

/**
 * @brief Data structure representing an edge in a graph projection.
 *
 * Stores edge connectivity and optional properties for projected edges.
 * Supports both directed and undirected edges per ISO GQL semantics.
 *
 * @see ProjectionStorage::add_edge for insertion
 */
struct ProjectedEdge {
    ObjectId from_node;   ///< Source node ObjectId
    ObjectId to_node;     ///< Target node ObjectId
    ObjectId edge_id;     ///< Unique edge identifier
    bool is_directed;     ///< true for directed (→), false for undirected (~)

    /// @brief Optional edge properties (key name → value ObjectId)
    std::unordered_map<std::string, ObjectId> properties;
};

/**
 * @brief Data structure representing a node in a graph projection.
 *
 * Stores node identity and optional properties for projected nodes.
 *
 * @see ProjectionStorage::add_node for insertion
 */
struct ProjectedNode {
    ObjectId node_id;     ///< Unique node identifier

    /// @brief Optional node properties (key name → value ObjectId)
    std::unordered_map<std::string, ObjectId> properties;
};

/**
 * @brief Persistent B+Tree-based storage for graph projections.
 *
 * Manages disk-based indexes for graph projections, enabling efficient
 * traversal and querying of projected subgraphs. Supports both required
 * and optional indexes based on projection configuration.
 *
 * ## Index Architecture
 *
 * ### Required Indexes (always present)
 * | Index | Keys | Purpose |
 * |-------|------|---------|
 * | nodes_index | (node_id) | Fast node existence check |
 * | from_to_edge | (from, to, edge_id) | Outgoing edge traversal |
 * | to_from_edge | (to, from, edge_id) | Incoming edge traversal |
 * | edge_direction | (edge_id, is_directed) | Direction lookup |
 * | edge_from_to | (edge_id, from, to) | Edge-first lookup for directed edges |
 * | edge_n1_n2 | (edge_id, n1, n2) | Edge-first lookup for undirected edges |
 *
 * ### Optional Label Indexes (when include_*_labels = true)
 * | Index | Keys | Purpose |
 * |-------|------|---------|
 * | node_label | (node_id, label_id) | Labels for node |
 * | label_node | (label_id, node_id) | Nodes with label |
 * | edge_label | (edge_id, label_id) | Labels for edge |
 * | label_edge | (label_id, edge_id) | Edges with label |
 *
 * ### Optional Property Indexes (when include_*_properties = true)
 * | Index | Keys | Purpose |
 * |-------|------|---------|
 * | node_key_value | (node_id, key_id, value) | Node properties |
 * | key_value_node | (key_id, value, node_id) | Property lookup |
 * | edge_key_value | (edge_id, key_id, value) | Edge properties |
 * | key_value_edge | (key_id, value, edge_id) | Property lookup |
 *
 * ## Performance Considerations
 *
 * - Uses batched writes (1000 items) for B+tree efficiency
 * - Pre-allocates capacity for 10000 items
 * - Sorted batch insertion for optimal B+tree performance
 * - Duplicate detection via hash sets before disk write
 *
 * @see ProjectionCatalog for metadata management
 * @see NativeProjectionBuilder for projection creation
 */
class ProjectionStorage {
public:
    /// @name Configuration Constants
    /// @{
    static constexpr size_t BATCH_SIZE = 250;          ///< Flush threshold for batched writes (reduced from 1000 to lower buffer pool pressure)
    static constexpr size_t INITIAL_CAPACITY = 10000;  ///< Pre-allocation size for buffers
    /// @}

    /**
     * @brief Configuration flags for optional projection indexes.
     *
     * Controls which optional indexes are created during projection build.
     * Omitting unused indexes saves disk space and build time.
     */
    struct Features {
        bool include_node_labels = false;      ///< Create node label indexes
        bool include_edge_labels = false;      ///< Create edge label indexes
        bool include_node_properties = false;  ///< Create node property indexes
        bool include_edge_properties = false;  ///< Create edge property indexes
    };

    /// @name Constructors
    /// @{

    /**
     * @brief Opens existing projection storage (read-only).
     * @param projection_dir Full path to projection (e.g., "db/projections/my_proj")
     * @param db_folder Database root folder (e.g., "db")
     */
    ProjectionStorage(const std::string& projection_dir, const std::string& db_folder);

    /**
     * @brief Creates new projection storage with catalog entry.
     * @param projection_dir Full path to projection directory
     * @param db_folder Database root folder
     * @param projection_name Human-readable projection name
     */
    ProjectionStorage(const std::string& projection_dir, const std::string& db_folder, const std::string& projection_name);

    /**
     * @brief Creates new projection with specified feature flags.
     * @param projection_dir Full path to projection directory
     * @param db_folder Database root folder
     * @param projection_name Human-readable projection name
     * @param features Configuration for optional indexes
     */
    ProjectionStorage(const std::string& projection_dir, const std::string& db_folder, const std::string& projection_name, const Features& features);

    ~ProjectionStorage();
    /// @}

    /// @name Lifecycle Methods
    /// @{

    /**
     * @brief Initializes new projection storage (write mode).
     *
     * Creates B+tree index files for all required indexes and
     * optional indexes based on feature flags. Must be called
     * before any add_* operations.
     *
     * @note For new projections only. Use open() for existing.
     */
    void init();

    /**
     * @brief Opens existing projection storage (read mode).
     *
     * Loads B+tree indexes from disk. Feature flags are read
     * from the projection catalog to determine which indexes exist.
     *
     * @note For existing projections only. Use init() for new.
     */
    void open();
    /// @}

    /// @name Write Methods (Buffered)
    /// @{

    /**
     * @brief Adds a node to the projection.
     *
     * Buffers the node for batch insertion. Automatically flushes
     * when buffer reaches BATCH_SIZE. Duplicate nodes are silently ignored.
     *
     * @param node Node to add with optional properties
     */
    void add_node(const ProjectedNode& node);

    /**
     * @brief Adds an edge to the projection.
     *
     * Buffers the edge for batch insertion. Handles both directed and
     * undirected edges. For UNDIRECTED orientation, the caller should
     * add both (A,B) and (B,A) entries with the same edge_id.
     *
     * @param edge Edge to add with connectivity and optional properties
     * @param skip_bloom_check If true, bypasses Bloom filter duplicate check.
     *        Use when edges are already guaranteed unique (e.g., from streaming
     *        aggregation which deduplicates via sorting). Default: false.
     */
    void add_edge(const ProjectedEdge& edge, bool skip_bloom_check = false);

    /**
     * @brief Adds a label to a node (requires include_node_labels).
     * @param node_id Target node ObjectId
     * @param label_id Label ObjectId to associate
     */
    void add_node_label(ObjectId node_id, ObjectId label_id);

    /**
     * @brief Adds a label to an edge (requires include_edge_labels).
     * @param edge_id Target edge ObjectId
     * @param label_id Label ObjectId to associate
     */
    void add_edge_label(ObjectId edge_id, ObjectId label_id);

    /**
     * @brief Adds a property to a node (requires include_node_properties).
     * @param node_id Target node ObjectId
     * @param key_id Property key ObjectId
     * @param value_id Property value ObjectId
     */
    void add_node_property(ObjectId node_id, ObjectId key_id, ObjectId value_id);

    /**
     * @brief Adds a property to an edge (requires include_edge_properties).
     * @param edge_id Target edge ObjectId
     * @param key_id Property key ObjectId
     * @param value_id Property value ObjectId
     */
    void add_edge_property(ObjectId edge_id, ObjectId key_id, ObjectId value_id);

    /**
     * @brief Registers a node property key mapping for this projection.
     *
     * Used for projection-specific keys like `_count` that don't exist in the
     * main catalog. The mapping is saved in the projection catalog.
     *
     * @param key_name Property name (e.g., "_count")
     * @param key_id The numeric key ID (without ObjectId::MASK_NODE_KEY)
     */
    void register_node_key(const std::string& key_name, uint64_t key_id);

    /**
     * @brief Registers an edge property key mapping for this projection.
     *
     * Used for projection-specific keys like `_count` that don't exist in the
     * main catalog. The mapping is saved in the projection catalog.
     *
     * @param key_name Property name (e.g., "_count")
     * @param key_id The numeric key ID (without ObjectId::MASK_EDGE_KEY)
     */
    void register_edge_key(const std::string& key_name, uint64_t key_id);
    /// @}

    /// @name Query Methods
    /// @{

    /**
     * @brief Checks if a node exists in the projection.
     * @param node_id Node ObjectId to check
     * @return true if node is in projection, false otherwise
     */
    bool has_node(ObjectId node_id) const;

    /**
     * @brief Checks if an edge exists between two nodes.
     * @param from Source node ObjectId
     * @param to Target node ObjectId
     * @return true if edge exists in projection
     */
    bool has_edge(ObjectId from, ObjectId to) const;

    /**
     * @brief Get a node property value by key.
     *
     * Searches the node_key_value_index for the property value.
     * Requires include_node_properties to be true during creation.
     *
     * @param node_id Node ObjectId
     * @param key_id Property key ObjectId
     * @return Property value ObjectId if found, nullopt otherwise
     */
    std::optional<ObjectId> get_node_property(ObjectId node_id, ObjectId key_id) const;

    /**
     * @brief Get an edge property value by key.
     *
     * Searches the edge_key_value_index for the property value.
     * Requires include_edge_properties to be true during creation.
     *
     * @param edge_id Edge ObjectId
     * @param key_id Property key ObjectId
     * @return Property value ObjectId if found, nullopt otherwise
     */
    std::optional<ObjectId> get_edge_property(ObjectId edge_id, ObjectId key_id) const;

    /// @brief Returns total number of nodes in projection
    uint64_t get_node_count() const { return node_count; }

    /// @brief Returns total number of edges (directed + undirected entries)
    uint64_t get_edge_count() const { return edge_count; }

    /// @brief Returns count of directed edge entries
    uint64_t get_directed_edge_count() const { return directed_edge_count; }

    /// @brief Returns count of undirected edge entries
    uint64_t get_undirected_edge_count() const { return undirected_edge_count; }
    /// @}

    /**
     * @brief Flushes all buffered data to disk.
     *
     * Forces immediate write of all batched nodes and edges to
     * their B+tree indexes. Called automatically on destruction.
     *
     * @note Must be called before reading from recently-written projection.
     */
    void flush();

    /// @name Inspection Methods
    /// @{

    /**
     * @brief Retrieves all node IDs in the projection.
     * @return Vector of all node ObjectIds
     * @note For debugging/testing. May be slow for large projections.
     */
    std::vector<ObjectId> get_all_node_ids() const;

    /**
     * @brief Retrieves complete edge information for all edges.
     * @return Vector of tuples (from_node, to_node, edge_id, is_directed)
     * @note For debugging/testing. May be slow for large projections.
     */
    std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>> get_all_edges_info() const;
    /// @}

    /// @name Required Index Accessors
    /// @brief Always available - used by ProjectionQueryContext for query execution.
    /// @{
    BPlusTree<1>* get_nodes_index() { return nodes_index.get(); }                ///< Node existence index
    BPlusTree<3>* get_from_to_edge_index() { return from_to_edge_index.get(); }  ///< Outgoing edge traversal
    BPlusTree<3>* get_to_from_edge_index() { return to_from_edge_index.get(); }  ///< Incoming edge traversal
    BPlusTree<2>* get_edge_direction_index() { return edge_direction_index.get(); }  ///< Edge direction lookup
    BPlusTree<3>* get_edge_from_to_index() { return edge_from_to_index.get(); }  ///< Edge-first directed lookup
    BPlusTree<3>* get_edge_n1_n2_index() { return edge_n1_n2_index.get(); }      ///< Edge-first undirected lookup
    /// @}

    /// @name Optional Label Index Accessors
    /// @brief Returns nullptr if include_*_labels was false during creation.
    /// @{
    BPlusTree<2>* get_node_label_index() { return node_label_index.get(); }  ///< Node → labels (may be null)
    BPlusTree<2>* get_label_node_index() { return label_node_index.get(); }  ///< Label → nodes (may be null)
    BPlusTree<2>* get_edge_label_index() { return edge_label_index.get(); }  ///< Edge → labels (may be null)
    BPlusTree<2>* get_label_edge_index() { return label_edge_index.get(); }  ///< Label → edges (may be null)
    /// @}

    /// @name Optional Property Index Accessors
    /// @brief Returns nullptr if include_*_properties was false during creation.
    /// @{
    BPlusTree<3>* get_node_key_value_index() { return node_key_value_index.get(); }  ///< Node properties (may be null)
    BPlusTree<3>* get_key_value_node_index() { return key_value_node_index.get(); }  ///< Property → nodes (may be null)
    BPlusTree<3>* get_edge_key_value_index() { return edge_key_value_index.get(); }  ///< Edge properties (may be null)
    BPlusTree<3>* get_key_value_edge_index() { return key_value_edge_index.get(); }  ///< Property → edges (may be null)
    /// @}

    /// @name Property Key Mapping Accessors
    /// @brief Access projection-specific property key mappings.
    /// @{

    /**
     * @brief Get node property key mappings for this projection.
     * @return Map of property name to key ID (without MASK_NODE_KEY)
     */
    const std::unordered_map<std::string, uint64_t>& get_node_keys() const {
        return node_keys2id_;
    }

    /**
     * @brief Get edge property key mappings for this projection.
     * @return Map of property name to key ID (without MASK_EDGE_KEY)
     */
    const std::unordered_map<std::string, uint64_t>& get_edge_keys() const {
        return edge_keys2id_;
    }

    /**
     * @brief Look up a node property key ID by name.
     * @param key_name Property name
     * @return Key ID if found, nullopt otherwise
     */
    std::optional<uint64_t> get_node_key_id(const std::string& key_name) const {
        auto it = node_keys2id_.find(key_name);
        if (it != node_keys2id_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * @brief Look up an edge property key ID by name.
     * @param key_name Property name
     * @return Key ID if found, nullopt otherwise
     */
    std::optional<uint64_t> get_edge_key_id(const std::string& key_name) const {
        auto it = edge_keys2id_.find(key_name);
        if (it != edge_keys2id_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    /// @}

    /// @name Bloom Filter Management
    /// @{

    /**
     * @brief Resizes the Bloom filter for expected edge count.
     *
     * Call this BEFORE adding edges if you have an estimate of the total
     * edge count. Improves accuracy by reducing false positive rate.
     *
     * When the filter is undersized for the actual edge count, the false
     * positive rate increases dramatically (e.g., 10M default vs 61M actual
     * edges causes ~92% FPR, rejecting 44% of legitimate edges).
     *
     * @param expected_edges Estimated number of edges to be added
     * @param fpr False positive rate (default: 0.01 = 1%)
     */
    void resize_bloom_filter(size_t expected_edges, double fpr = BLOOM_FILTER_FPR);
    /// @}

    /// @name Const Index Accessors
    /// @brief Read-only versions for const-correct access.
    /// @{
    const BPlusTree<1>* get_nodes_index() const { return nodes_index.get(); }
    const BPlusTree<3>* get_from_to_edge_index() const { return from_to_edge_index.get(); }
    const BPlusTree<3>* get_to_from_edge_index() const { return to_from_edge_index.get(); }
    const BPlusTree<2>* get_edge_direction_index() const { return edge_direction_index.get(); }
    const BPlusTree<3>* get_edge_from_to_index() const { return edge_from_to_index.get(); }
    const BPlusTree<3>* get_edge_n1_n2_index() const { return edge_n1_n2_index.get(); }

    const BPlusTree<2>* get_node_label_index() const { return node_label_index.get(); }
    const BPlusTree<2>* get_label_node_index() const { return label_node_index.get(); }
    const BPlusTree<2>* get_edge_label_index() const { return edge_label_index.get(); }
    const BPlusTree<2>* get_label_edge_index() const { return label_edge_index.get(); }
    const BPlusTree<3>* get_node_key_value_index() const { return node_key_value_index.get(); }
    const BPlusTree<3>* get_key_value_node_index() const { return key_value_node_index.get(); }
    const BPlusTree<3>* get_edge_key_value_index() const { return edge_key_value_index.get(); }
    const BPlusTree<3>* get_key_value_edge_index() const { return key_value_edge_index.get(); }
    /// @}

private:
    /// @name Internal Batch Operations
    /// @{
    void flush_node_batch();  ///< Writes buffered nodes to streaming buffer
    void flush_edge_batch();  ///< Writes buffered edges to streaming buffers
    void save_catalog();      ///< Persists projection metadata to catalog file
    void initialize_streaming_buffers();  ///< Creates streaming buffers with temp file paths

    /**
     * @brief Check if an edge already exists in the B+tree indexes.
     *
     * Used for duplicate detection after batch flush. Checks the from_to_edge_index
     * for an exact match of (from, to, edge_id).
     *
     * @param from Source node ObjectId
     * @param to Target node ObjectId
     * @param edge_id Edge ObjectId
     * @return true if edge exists in B+tree, false otherwise
     *
     * Performance: O(log n) B+tree lookup
     */
    bool edge_exists_in_btree(ObjectId from, ObjectId to, ObjectId edge_id) const;
    /// @}

    /// @name Path Configuration
    /// @{
    std::string projection_dir;   ///< Full path (e.g., "test_db/projections/my_proj")
    std::string rel_dir;          ///< Relative path from db_folder
    std::string projection_name;  ///< Human-readable projection name
    /// @}

    Features features;  ///< Configuration flags for optional indexes

    /// @name Required B+Tree Indexes
    /// @brief Always created - essential for graph traversal.
    /// @{
    std::unique_ptr<BPlusTree<1>> nodes_index;           ///< {node_id} - existence check
    std::unique_ptr<BPlusTree<3>> from_to_edge_index;    ///< {from, to, edge_id} - outgoing
    std::unique_ptr<BPlusTree<3>> to_from_edge_index;    ///< {to, from, edge_id} - incoming
    std::unique_ptr<BPlusTree<2>> edge_direction_index;  ///< {edge_id, is_directed} - type lookup
    std::unique_ptr<BPlusTree<3>> edge_from_to_index;    ///< {edge_id, from, to} - edge-first lookup (for directed)
    std::unique_ptr<BPlusTree<3>> edge_n1_n2_index;      ///< {edge_id, n1, n2} - edge-first lookup (for undirected)
    /// @}

    /// @name Optional Label B+Tree Indexes
    /// @brief Created only when include_*_labels = true.
    /// @{
    std::unique_ptr<BPlusTree<2>> node_label_index;  ///< {node_id, label_id} - node's labels
    std::unique_ptr<BPlusTree<2>> label_node_index;  ///< {label_id, node_id} - label's nodes
    std::unique_ptr<BPlusTree<2>> edge_label_index;  ///< {edge_id, label_id} - edge's labels
    std::unique_ptr<BPlusTree<2>> label_edge_index;  ///< {label_id, edge_id} - label's edges
    /// @}

    /// @name Optional Property B+Tree Indexes
    /// @brief Created only when include_*_properties = true.
    /// @{
    std::unique_ptr<BPlusTree<3>> node_key_value_index;  ///< {node_id, key_id, value} - node props
    std::unique_ptr<BPlusTree<3>> key_value_node_index;  ///< {key_id, value, node_id} - prop lookup
    std::unique_ptr<BPlusTree<3>> edge_key_value_index;  ///< {edge_id, key_id, value} - edge props
    std::unique_ptr<BPlusTree<3>> key_value_edge_index;  ///< {key_id, value, edge_id} - prop lookup
    /// @}

    /// @name Legacy Indexes (Deprecated)
    /// @brief Maintained for backward compatibility with older projections.
    /// @{
    std::unique_ptr<BPlusTree<3>> node_properties_index;  ///< @deprecated Use node_key_value_index
    std::unique_ptr<BPlusTree<4>> edge_properties_index;  ///< @deprecated Use edge_key_value_index
    /// @}

    /// @name Statistics
    /// @{
    uint64_t node_count = 0;            ///< Total nodes in projection
    uint64_t edge_count = 0;            ///< Total edge entries (including both directions for undirected)
    uint64_t directed_edge_count = 0;   ///< Count of directed edge entries
    uint64_t undirected_edge_count = 0; ///< Count of undirected edge entries
    /// @}

    /// @name Property Key Mappings
    /// @brief Projection-specific key name ↔ ID mappings for synthetic properties.
    /// @{
    std::vector<std::string> node_keys_str_;       ///< Index → node key name
    std::unordered_map<std::string, uint64_t> node_keys2id_;  ///< Node key name → index
    std::vector<std::string> edge_keys_str_;       ///< Index → edge key name
    std::unordered_map<std::string, uint64_t> edge_keys2id_;  ///< Edge key name → index
    /// @}

    /// @name Duplicate Detection
    /// @{
    std::unordered_set<uint64_t> inserted_nodes;  ///< Tracks inserted node IDs

    /**
     * @brief Bloom filter for memory-efficient edge deduplication.
     *
     * Replaces the previous hash set approach to reduce memory usage from
     * O(n) to O(1) with configurable false positive rate (default 1%).
     *
     * False positives are acceptable because:
     * 1. They only cause a small number of legitimate edges to be skipped
     * 2. Final correctness is guaranteed by std::unique() during bulk index build
     *
     * Memory savings: ~4 GB for 123M edges (vs 4.25 GB for hash set)
     *
     * @see BloomFilter for implementation details
     * @see build_index_bulk() for final deduplication
     */
    std::unique_ptr<BloomFilter> edge_bloom_filter_;

    /**
     * @brief Expected number of edges for Bloom filter sizing.
     *
     * Set during construction or dynamically resized if exceeded.
     * Default: 10M edges (~125 MB memory at 1% FPR)
     */
    static constexpr size_t DEFAULT_EXPECTED_EDGES = 10000000;

    /**
     * @brief False positive rate for edge Bloom filter.
     *
     * Lower = more memory, higher = more false skips (but still correct due to std::unique).
     * 1% is a good balance: ~10 bits per element.
     */
    static constexpr double BLOOM_FILTER_FPR = 0.01;
    /// @}

    /// @name Write Buffers
    /// @{
    std::vector<ProjectedNode> node_batch;  ///< Buffered nodes pending B+tree insertion
    std::vector<ProjectedEdge> edge_batch;  ///< Buffered edges pending B+tree insertion
    /// @}

    /// @name Streaming Record Buffers
    /// @brief Memory-bounded streaming buffers for bulk import.
    ///
    /// Records are accumulated with configurable memory threshold, automatically
    /// spilling to disk when exceeded. This enables building projections of
    /// arbitrary size (500M+ edges) with bounded memory (~512 MB total).
    ///
    /// Memory savings: 12-14 GB (vectors) → ~512 MB (streaming buffers)
    ///
    /// @see StreamingRecordBuffer for implementation details
    /// @{

    /// @brief Memory threshold per streaming buffer (64 MB default)
    static constexpr size_t STREAMING_BUFFER_THRESHOLD = 64 * 1024 * 1024;

    // Node index records (streaming)
    std::unique_ptr<StreamingRecordBuffer<1>> node_records_buffer_;

    // Edge connectivity records (streaming)
    std::unique_ptr<StreamingRecordBuffer<3>> from_to_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<3>> to_from_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<2>> direction_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<3>> edge_from_to_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<3>> edge_n1_n2_records_buffer_;

    // Label index records (optional, streaming)
    std::unique_ptr<StreamingRecordBuffer<2>> node_label_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<2>> label_node_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<2>> edge_label_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<2>> label_edge_records_buffer_;

    // Property index records (optional, streaming)
    std::unique_ptr<StreamingRecordBuffer<3>> node_key_value_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<3>> key_value_node_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<3>> edge_key_value_records_buffer_;
    std::unique_ptr<StreamingRecordBuffer<3>> key_value_edge_records_buffer_;
    /// @}

    /// @name Bulk Import Build Methods
    /// @{

    /**
     * @brief Build all B+tree indexes using bulk import.
     *
     * Called during finalize() to construct all indexes sequentially
     * using BPTLeafWriter/BPTDirWriter, completely bypassing the
     * buffer pool. Each index is built one at a time to minimize
     * memory pressure.
     */
    void build_all_indexes_bulk();

    /**
     * @brief Build a single B+tree index using bulk import.
     *
     * Sorts the records, writes leaf pages directly to disk using
     * BPTLeafWriter, and builds directory structure using BPTDirWriter.
     * No buffer pool involvement - scales to arbitrary dataset sizes.
     *
     * @tparam N Number of keys in the record (1, 2, 3, or 4)
     * @param records Vector of records to write (will be sorted in place)
     * @param base_path Full path prefix for index files (without .leaf/.dir extension)
     */
    template<std::size_t N>
    size_t build_index_bulk(std::vector<Record<N>>& records, const std::string& base_path);

    /**
     * @brief Build a single B+tree index using streaming external sort.
     *
     * Uses ExternalRecordSort to stream sorted records with bounded memory,
     * writing directly to B+tree leaf pages with inline deduplication.
     * This is the memory-efficient alternative to build_index_bulk for
     * large datasets that don't fit in RAM.
     *
     * ## Memory Model
     *
     * - Sort phase: O(buffer_size) - typically 256 MB
     * - Build phase: O(max_records_per_leaf) - one page of records
     * - Total: O(buffer_size) regardless of dataset size
     *
     * @tparam N Number of keys in the record (1, 2, or 3)
     * @param sorter ExternalRecordSort instance with runs to merge
     * @param base_path Full path prefix for index files (without .leaf/.dir extension)
     * @return Number of unique records written to the index
     */
    template<std::size_t N>
    size_t build_index_streaming(ExternalRecordSort<N>& sorter, const std::string& base_path);
    /// @}
};

} // namespace GQL
