#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_models/gql/projection/bloom_filter.h"
#include "graph_models/gql/projection/edge_keep_bitmap.h"
#include "graph_models/gql/projection/streaming_record_buffer.h"
#include "graph_models/object_id.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/record.h"

template<std::size_t N>
class BPlusTree;

namespace GQL {

// Forward declaration for streaming build
template<std::size_t N>
class ExternalRecordSort;

class ProjectionCatalog;

// Forward declaration; full definition in native_projection_builder.h.
// Included via .cc to avoid circular include with native_projection_builder
// (which depends on ProjectionStorage).
enum class ProjectionIndex : uint32_t;

// Forward-declared; full definition in graph_models/gql/projection/index_set.h.
// Set on ProjectionStorage before finalize, then persisted via save_catalog().
enum class IndexSet : uint8_t;

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

    /**
     * @brief Hard cap on the node presence bitmap (see `node_bitmap_`).
     *
     * WHY A CAP EXISTS. The bitmap costs one bit per id in [min_id, max_id],
     * so it is sized by the id RANGE, not by the node count. Density is a
     * property of the data, never an invariant of this class: a node filter
     * selecting a sparse subset, or a projection whose nodes carry more than
     * one ObjectId type tag, makes that range arbitrarily large while the node
     * count stays small. Without a cap the allocation is unbounded.
     *
     * WHY 64 MiB, AND WHY A CONSTANT. 64 MiB covers a contiguous range of
     * 536,870,912 ids; papers100M spans 111,059,955, i.e. 20.7% of the
     * ceiling, and no graph in this project comes near it. A constant makes
     * the decision a pure function of the DATA, so two runs of the same A/B
     * protocol cannot silently take different paths — which is exactly what a
     * MemAvailable-derived budget would allow.
     *
     * WHY NOT BIGGER. The one failure mode of this optimization with a
     * catastrophic penalty is the bitmap being paged out: a scattered probe
     * against a swapped page costs ~66 us instead of ~1.6 ns, and only 0.44%
     * of the 3.231e9 probes need to hit one to erase the whole saving. A
     * ceiling this small cannot be the allocation that swaps a host, and it is
     * 7.6% of the 847 MB `collected_nodes_` already holds for the same node
     * set and never releases.
     *
     * A range that does not fit is NOT an error: the bitmap is simply not
     * built and has_node() runs exactly the code it ran before.
     */
    static constexpr size_t NODE_BITMAP_MAX_BYTES = 64ULL * 1024 * 1024;
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

    /// @brief Returns the projection directory path (e.g., "test_db/projections/my_proj")
    const std::string& get_projection_dir() const { return projection_dir; }

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

    /**
     * @brief Drains internal node/edge batches into the streaming record buffers.
     *
     * Called by NativeProjectionBuilder::finalize_serialized_() before each
     * build_one_index() pass to ensure all records emitted by scan callbacks
     * have been flushed from the internal BATCH_SIZE=250 storage batches into
     * the StreamingRecordBuffers that build_one_index() reads from.
     *
     * Without this, up to (BATCH_SIZE-1)=249 node records and a similar number
     * of edge records can remain in the internal batches when build_one_index()
     * runs, causing those records to be omitted from the serialized B+Tree build
     * and later re-processed by the fallback flush() call — overwriting the
     * correctly-built index with a partial dataset.
     */
    void drain_pending_batches();

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

    /// Create empty node property B+Tree indexes if they don't exist.
    /// Called by EmbeddingWriter when the projection was built without
    /// property indexes (STRING syntax in graph_project).
    void ensure_node_property_indexes();
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

    /// @name Serialized Scan Pipeline (one-index-at-a-time edge scan)
    /// @{

    /**
     * @brief Build a SINGLE projection index (for serialized scan pipeline).
     *
     * Used by NativeProjectionBuilder::finalize_serialized_() between scan
     * passes. Does NOT create BPlusTree reader instances — the caller
     * opens them via build_all_indexes_bulk's Phase 4 or its equivalent.
     *
     * @param which Exactly one single-bit ProjectionIndex value.
     * @throws std::invalid_argument on a multi-bit ProjectionIndex value
     *         (NONE, ALL_NODE, ALL_EDGE, ALL, or any unknown combination).
     */
    void build_one_index(ProjectionIndex which);

    /**
     * @brief Reset (remove + recreate empty) the sort scratch directory.
     *
     * Called by NativeProjectionBuilder::finalize_serialized_() between
     * serialized scan passes so peak scratch disk stays bounded to
     * O(max single index) instead of O(sum all indexes).
     */
    void reset_sort_scratch_();

    /**
     * @brief Begin a serialized edge scan pass for a single index.
     *
     * Sets serial_write_mask_ so that flush_edge_batch() only populates
     * the streaming buffer(s) belonging to `which`. Also clears the edge
     * bloom filter so the fresh per-pass scan emits all edges regardless
     * of what prior passes added to the filter.
     *
     * Must be called before scan_edges_impl_serialized_() for the pass
     * and paired with end_serial_edge_pass_() after build_one_index().
     * Spill files left by any previous pass's non-target buffers are
     * cleared here, bounding peak disk to O(max single pass).
     *
     * @param which Single-bit ProjectionIndex identifying the target index.
     */
    void begin_serial_edge_pass_(ProjectionIndex which);

    /**
     * @brief End a serialized edge scan pass.
     *
     * Clears serial_write_mask_ (reverts flush_edge_batch() to ALL-buffer
     * mode) ready for the next pass or for any post-serialized operation
     * that expects classic write semantics.
     */
    void end_serial_edge_pass_();

    /**
     * @brief Open all B+Tree index readers after the index files have been built.
     *
     * Called by both build_all_indexes_bulk() (CLASSIC path) and
     * NativeProjectionBuilder::finalize_serialized_() (SERIALIZED path).
     *
     * Under CLASSIC, build_all_indexes_bulk() builds all 14 indexes and then
     * calls this method to open the resulting .leaf/.dir files.
     *
     * Under SERIALIZED, finalize_serialized_() builds each index piecemeal
     * (one per scan pass) and calls this method after the last pass, so the
     * projection is ready for use immediately after finalize() returns.
     *
     * Also removes the sort_tmp scratch directory as a final cleanup step.
     *
     * @note Safe to call on already-open readers — reassignment of the
     *       unique_ptr members destroys any existing BPlusTree instance
     *       (closing its file handles) before the new one is constructed.
     */
    void open_all_bplustree_readers_();
    /// @}

    /// @name Node-scan Lifecycle
    /// @{

    /**
     * @brief Finalize the scan-phase node set.
     *
     * MUST be called after the node scan phase completes and before any
     * `has_node()` call during edge scan. Sorts `collected_nodes_` and
     * removes duplicates in place, releasing excess capacity.
     *
     * Idempotent: a second call returns immediately.
     *
     * Complexity: O(N log N) for the sort (one-time, not per-insert).
     * After this call `has_node()` runs in O(log N) instead of
     * O(N) linear scan.
     */
    void finalize_node_scan();

    /**
     * @brief Read-only access to the sorted, dedup'd in-memory node set.
     *
     * After `finalize_node_scan()`, the returned vector is sorted by
     * value and contains each ObjectId.id at most once. Used by
     * `EdgeKeepBitmapGpuBatcher` to upload a single sorted array to the
     * GPU for parallel binary-search membership testing, mirroring exactly
     * the data structure that `has_node()` walks on the CPU path.
     *
     * Pre-`finalize_node_scan()` the vector is unsorted with possible
     * duplicates — callers needing the sorted invariant must call
     * finalize_node_scan() first.
     */
    const std::vector<uint64_t>& collected_nodes() const noexcept {
        return collected_nodes_;
    }

    /**
     * @brief Bytes held by the node presence bitmap; 0 when it was not built.
     *
     * Exists so the BUDGET DECISION is assertable from outside the class.
     * Without it, "built a bitmap" and "fell back to binary search" are
     * indistinguishable, because both answer every query identically by
     * construction. That indistinguishability is exactly how a bitmap that is
     * never adopted would ship as a silent no-op: correct, tested, and worth
     * nothing. It is also what the edge-scan profile line reads to report
     * which structure served the run being measured.
     */
    std::size_t node_bitmap_bytes() const noexcept {
        return node_bitmap_ ? node_bitmap_->bytes_allocated() : 0;
    }

    /// Why the bitmap was or was not adopted. Stable, greppable strings:
    /// "adopted", "span-exceeds-budget", "empty-node-set", "disabled-by-env",
    /// "invalidated-by-late-add", "not-finalized". Printed in the
    /// [NODE_BITMAP] line and asserted in tests.
    const char* node_bitmap_status() const noexcept { return node_bitmap_status_; }
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

    /// @name Catalog metadata (populated by the builder before finalize)
    /// @brief Free-form lists surfaced via `mdb inspect-projection`. Set by the
    /// caller because ProjectionStorage doesn't know which property names were
    /// "requested" (vs merely encountered).
    /// @{
    std::vector<std::string> requested_node_properties;
    std::vector<std::string> requested_edge_properties;
    /// @brief IndexSet preset picked at build time. Persisted in catalog v1.4
    /// so the reader knows which B+Tree indexes were materialized. Default is
    /// IndexSet::ALL (ordinal 0); set by NativeProjectionBuilder before
    /// finalize. Stored here rather than in Features because Features only
    /// covers the four optional label/property index pairs, while IndexSet
    /// also gates required indexes like NODES or FROM_TO_EDGE.
    /// @}
    /// @{
    IndexSet requested_index_set = static_cast<IndexSet>(0);
    /// @}

    /// @brief Returns the IndexSet preset this projection was built under.
    ///
    /// Populated at build time by NativeProjectionBuilder and restored at
    /// read time from the catalog via ProjectionStorage::open(). Consumed
    /// by the query-layer error diagnostic to report the active preset
    /// when a missing index is accessed.
    IndexSet get_index_set() const { return requested_index_set; }

    /// @brief Leaf-encoding preset (BITSET or DELTA_VARINT) for every B+Tree
    /// index this projection owns. Set by NativeProjectionBuilder from the GQL
    /// `leafFormat` config key before flush()/open(). Consumed by:
    ///   (1) every `std::make_unique<BPlusTree<N>>(...)` call in open() and
    ///       open_all_bplustree_readers_() (passes it as the second ctor
    ///       argument so BptIter dispatches on the right leaf layout).
    ///   (2) save_catalog() to populate ProjectionCatalog::leaf_formats with
    ///       one byte per materialized index (catalog v1.5). Default BITSET
    ///       preserves byte-identical behavior for projections built without
    ///       delta + LEB128-varint leaf compression.
    BPT::LeafFormat requested_leaf_format = BPT::LeafFormat::BITSET;

    /// @brief Returns the leaf-format preset this projection was built under.
    BPT::LeafFormat get_leaf_format() const noexcept { return requested_leaf_format; }

    /// @brief Per-projection graph-storage mode (BTREE or CSR_HYBRID). Set by
    /// NativeProjectionBuilder from the GQL `graphStorage` config key
    /// before flush(), restored on open() from ProjectionCatalog::
    /// graph_storage (v1.6 byte). Consumed by:
    ///   (1) save_catalog() which forwards it into catalog.graph_storage
    ///       for v1.6 persistence.
    ///   (2) The edge-index dispatch in sorter_dispatch.cc, which selects
    ///       BPTLeafCSRWriter under CSR_HYBRID (where edge-index B+Tree leaves
    ///       store the CSR layout directly for O(1) neighbor access). Under
    ///       BTREE the dispatch path emits standard B+Tree leaves regardless,
    ///       so CSR_HYBRID projections currently round-trip the catalog byte
    ///       only.
    /// Default BTREE preserves byte-identical behavior for projections that
    /// do not use the CSR-hybrid graph storage mode.
    BPT::GraphStorage requested_graph_storage = BPT::GraphStorage::BTREE;

    /// @brief Returns the graph-storage mode this projection was built under.
    BPT::GraphStorage get_graph_storage() const noexcept {
        return requested_graph_storage;
    }

    /// @name Integrated topology CSR sidecar emission
    /// @{

    /**
     * @brief Enable inline CSR sidecar emission during edge index builds.
     *
     * When set to true before `flush()` / `build_one_index(FROM_TO_EDGE|
     * TO_FROM_EDGE)` runs, each of the two edge index builders emits the
     * matching `topology_{fwd,rev}.csr` right after the `.leaf` lands on
     * disk, reading back via mmap over the freshly-written file. This
     * replaces the post-hoc 3-pass-per-direction builder previously
     * invoked from `NativeProjectionBuilder::build_topology_snapshots_()`.
     *
     * Default `false` preserves byte-identical behavior for every caller
     * that does not request topology snapshots.
     */
    void set_build_topology_snapshot(bool enable) {
        build_topology_snapshot_ = enable;
    }
    bool get_build_topology_snapshot() const noexcept {
        return build_topology_snapshot_;
    }

    /**
     * @brief Returns true if `topology_fwd.csr` was successfully emitted
     *        by the integrated path during this storage's build.
     *
     * Used by NativeProjectionBuilder's logging layer to decide whether
     * the legacy post-hoc builder needs to run as a fallback.
     */
    bool fwd_topology_snapshot_built() const noexcept {
        return fwd_topology_snapshot_built_;
    }
    bool rev_topology_snapshot_built() const noexcept {
        return rev_topology_snapshot_built_;
    }
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

    /**
     * @brief Sorted-vector tracker for inserted node IDs during scan.
     *
     * Replaces an earlier std::unordered_set<uint64_t>, whose per-entry
     * overhead (~48 B on libstdc++ — node header, bucket pointer, and
     * padding around the 8-byte key) was dominating scan-phase RSS on
     * 100M+ node graphs. A sorted vector stores only the key itself.
     *
     * Protocol:
     *   - `add_node` appends unconditionally (no per-call dedup check).
     *   - `finalize_node_scan` sorts + `std::unique`s and flips
     *     `collected_nodes_sorted_` to true.
     *   - `has_node` does `std::binary_search` when sorted, and a
     *     linear-scan fallback otherwise (defensive — the builder is
     *     required to call `finalize_node_scan` before edge scan).
     *
     * Memory: ~8 B per unique node (vs ~48 B for the hash set).
     */
    std::vector<uint64_t> collected_nodes_;
    bool                  collected_nodes_sorted_ = false;

    /**
     * @brief Dense presence bitmap over [node_bitmap_min_, +span], keyed by
     *        `raw_object_id - node_bitmap_min_`.
     *
     * WHY. has_node() is called twice per candidate edge
     * (native_projection_builder.cc:799-800) and is the single most expensive
     * thing the projection does: 943.4 s on papers100M, 29.01% of the entire
     * build, 292.0 ns per probe (measured 2026-08-20). The cost is not
     * computation, it is memory latency: binary search over the 847 MB sorted
     * vector above walks ~27 steps across lines that are nowhere near each
     * other, and 847 MB cannot live in a 30 MiB L3. One bit per id answers the
     * same question from ONE cache line: 1.56 ns measured at real scale with
     * the real access pattern, a 174.5x ratio, in a harness whose
     * binary-search arm reproduced the in-situ 292.0 ns to within 6.5%. Huge
     * pages changed nothing (1.59 ns), which is the proof that the win is
     * cache residency and not address-translation reach.
     *
     * WHY IT FITS. GQL node ids are minted as a dense counter OR'd with a
     * constant type tag, so a whole-graph projection collects a contiguous
     * range: papers100M spans exactly its 111,059,956 ids, i.e. 100% dense,
     * for 13.24 MiB — 1.6% of what the vector above already costs.
     *
     * WHY THE TAG IS NOT MASKED OFF. Indexing is by RAW id minus the minimum,
     * so a foreign type tag shifts the index by a multiple of 2^56 and lands
     * far past any span the budget admits, where EdgeKeepBitmap::is_kept()'s
     * own bound check rejects it. Masking would cost an extra operation on
     * every probe AND would make two ids with different tags collide.
     *
     * NULL means "not built": either it did not fit, the env switch disabled
     * it, the node set is empty, or a late add_node() dropped it. In every one
     * of those cases has_node() runs the pre-existing binary search.
     */
    std::unique_ptr<EdgeKeepBitmap> node_bitmap_;
    uint64_t                        node_bitmap_min_ = 0;
    const char*                     node_bitmap_status_ = "not-finalized";

    /// Builds `node_bitmap_` from the finalized `collected_nodes_`, or leaves
    /// it null and records why. Called only from finalize_node_scan().
    void build_node_bitmap_();

    /// One line per projection build recording the adopt/refuse decision.
    void log_node_bitmap_decision_(std::size_t num_ids,
                                   uint64_t min_id,
                                   uint64_t max_id,
                                   uint64_t span) const;

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

    /// @name Integrated topology CSR sidecar state
    /// @{
    /// Opt-in flag set by NativeProjectionBuilder via
    /// set_build_topology_snapshot() before finalize. When true, the two
    /// edge index builders (build_from_to_edge_index_ / build_to_from_edge_index_)
    /// invoke GQL::Projection::build_topology_snapshot_from_leaf() right
    /// after the `.leaf` is written to disk, emitting the matching mmap-backed
    /// CSR sidecar file (topology_fwd.csr or topology_rev.csr) over the fresh
    /// file. Default false preserves the original behavior for every existing
    /// caller that does not request topology snapshots.
    bool build_topology_snapshot_ = false;

    /// Per-direction "already emitted" flags, set by the integrated path.
    /// Consulted by NativeProjectionBuilder::build_topology_snapshots_()
    /// so the legacy post-hoc walker is skipped whenever the integrated
    /// path has already produced the sidecar. In SERIALIZED mode each
    /// edge index is built exactly once, so these are a single-transition
    /// false -> true per direction.
    bool fwd_topology_snapshot_built_ = false;
    bool rev_topology_snapshot_built_ = false;
    /// @}

    /// @name Serialized-mode edge-write mask
    /// @{
    /**
     * @brief Bitmask controlling which edge streaming buffers flush_edge_batch()
     * populates during the serialized one-index-at-a-time scan pipeline.
     *
     * 0 (default) = CLASSIC mode: write to all applicable buffers.
     * Non-zero    = SERIAL mode: only write to buffers whose ProjectionIndex
     *               bit is set in the mask. Used by
     *               ProjectionStorage::begin_serial_edge_pass_() to bound peak
     *               scratch disk to O(max single pass) on large datasets
     *               (see projection_storage.cc for the papers100M motivation).
     *
     * Also gates the edge bloom filter: when the mask is active the bloom check
     * is skipped so each fresh per-pass scan emits all edges independently.
     */
    uint32_t serial_write_mask_ = 0;
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

    // Per-index build methods (extracted from build_all_indexes_bulk).
    // Each builds exactly one B+Tree .leaf/.dir file from its backing
    // StreamingRecordBuffer via GQL::sort_and_build_index.
    // Called by build_all_indexes_bulk() in CLASSIC mode and by
    // build_one_index(ProjectionIndex) in SERIALIZED mode.

    // Builds the nodes B+Tree. Side effect: updates the class-scope
    // node_count member (consumed by catalog finalization and by any
    // build_one_index(NODES) caller from the SERIALIZED dispatcher).
    // This is the only per-index method that mutates non-local state.
    void build_nodes_index_();
    void build_node_label_index_();
    void build_label_node_index_();
    void build_node_key_value_index_();
    void build_key_value_node_index_();
    void build_from_to_edge_index_();
    void build_to_from_edge_index_();
    void build_edge_direction_index_();
    void build_edge_from_to_index_();
    void build_edge_n1_n2_index_();
    void build_edge_label_index_();
    void build_label_edge_index_();
    void build_edge_key_value_index_();
    void build_key_value_edge_index_();

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
