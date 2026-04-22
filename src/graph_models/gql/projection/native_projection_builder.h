#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph_models/gql/projection/edge_aggregation_record.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/procedure/builtin/project_procedure.h"  // For Orientation enum

#ifdef ENABLE_GNN
#include "gnn/storage/row_mapping.h"
#endif

namespace GQL {

/**
 * @brief Bitmask enum identifying which projection B+Tree index is being
 *        populated during a scan pass. Used by the serialized scan pipeline
 *        (Spec #2) to gate record emissions in scan callbacks.
 *
 * The 14 indexes correspond to the 6 core + 8 feature-gated B+Trees built
 * by ProjectionStorage::build_all_indexes_bulk. In CLASSIC mode the
 * default ALL mask preserves exact current behavior (all buffers emit).
 * In SERIALIZED mode, each scan pass uses a single-bit mask.
 */
enum class ProjectionIndex : uint32_t {
    NONE            = 0,
    NODES           = 1u << 0,
    NODE_LABEL      = 1u << 1,
    LABEL_NODE      = 1u << 2,
    NODE_KEY_VALUE  = 1u << 3,
    KEY_VALUE_NODE  = 1u << 4,
    FROM_TO_EDGE    = 1u << 5,
    TO_FROM_EDGE    = 1u << 6,
    EDGE_DIRECTION  = 1u << 7,
    EDGE_FROM_TO    = 1u << 8,
    EDGE_N1_N2      = 1u << 9,
    EDGE_LABEL      = 1u << 10,
    LABEL_EDGE      = 1u << 11,
    EDGE_KEY_VALUE  = 1u << 12,
    KEY_VALUE_EDGE  = 1u << 13,
    ALL_NODE        = (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4),
    ALL_EDGE        = (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9) |
                      (1u << 10) | (1u << 11) | (1u << 12) | (1u << 13),
    ALL             = 0x3FFFu,
};

constexpr ProjectionIndex operator|(ProjectionIndex a, ProjectionIndex b) {
    return static_cast<ProjectionIndex>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr ProjectionIndex operator&(ProjectionIndex a, ProjectionIndex b) {
    return static_cast<ProjectionIndex>(
        static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr bool has_flag(ProjectionIndex mask, ProjectionIndex bit) {
    return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(bit)) != 0;
}

// Compile-time drift guard: ensures the numerically-computed ALL_NODE /
// ALL_EDGE / ALL presets stay consistent with the single-bit enumerators.
// If a future committer adds a 15th bit (e.g., EDGE_N2_N1 = 1u << 14)
// without updating ALL_EDGE and ALL, these static_asserts fire during
// compilation, not at runtime.
static_assert(
    ProjectionIndex::ALL == (ProjectionIndex::ALL_NODE | ProjectionIndex::ALL_EDGE),
    "ProjectionIndex::ALL must equal ALL_NODE | ALL_EDGE");
static_assert(
    static_cast<uint32_t>(ProjectionIndex::ALL) == 0x3FFFu,
    "ProjectionIndex::ALL must equal 0x3FFF (14 single-bit projection indexes)");

/**
 * @brief Conditional per-phase timing for graph_project pipeline.
 *
 * Activated by MDB_BENCHMARK=1 environment variable. Zero overhead when disabled
 * (all timing calls short-circuit on `!enabled`). Prints to stderr with
 * [BENCHMARK] prefix for easy grep.
 */
struct ProjectionTimers {
    double node_scan_ms     = 0;
    double edge_scan_ms     = 0;
    double property_ms      = 0;
    double sort_ms          = 0;
    double btree_write_ms   = 0;
    double aggregation_ms   = 0;
    double metadata_ms      = 0;
    double total_ms         = 0;
    bool enabled            = false;

    void print(const std::string& proj_name, uint64_t edge_count) const {
        if (!enabled) return;
        auto pct = [&](double v) -> double { return total_ms > 0 ? v / total_ms * 100 : 0; };
        fprintf(stderr, "[BENCHMARK] graph_project '%s' — %llu edges\n",
                proj_name.c_str(), (unsigned long long)edge_count);
        fprintf(stderr, "[BENCHMARK]   node_scan:     %8.1f ms  (%4.1f%%)\n", node_scan_ms, pct(node_scan_ms));
        fprintf(stderr, "[BENCHMARK]   edge_scan:     %8.1f ms  (%4.1f%%)\n", edge_scan_ms, pct(edge_scan_ms));
        fprintf(stderr, "[BENCHMARK]   property:      %8.1f ms  (%4.1f%%)\n", property_ms, pct(property_ms));
        fprintf(stderr, "[BENCHMARK]   sort:          %8.1f ms  (%4.1f%%)\n", sort_ms, pct(sort_ms));
        fprintf(stderr, "[BENCHMARK]   btree_write:   %8.1f ms  (%4.1f%%)\n", btree_write_ms, pct(btree_write_ms));
        fprintf(stderr, "[BENCHMARK]   aggregation:   %8.1f ms  (%4.1f%%)\n", aggregation_ms, pct(aggregation_ms));
        fprintf(stderr, "[BENCHMARK]   metadata:      %8.1f ms  (%4.1f%%)\n", metadata_ms, pct(metadata_ms));
        fprintf(stderr, "[BENCHMARK]   total:         %8.1f ms\n", total_ms);
        fprintf(stderr, "[BENCHMARK]   sort_fraction: %.3f\n", total_ms > 0 ? sort_ms / total_ms : 0);
    }
};

// Use Orientation, Aggregation, and PropertyConfig from Procedures namespace
using Procedures::Orientation;
using Procedures::Aggregation;
using Procedures::PropertyConfig;

// Forward declaration (NativeScanner implemented by another agent)
class NativeScanner;

// Forward declaration for the serialized-scan pipeline (Spec #2).
// EdgeFilter lives in edge_filter.h; forward-declaring here keeps
// the header free of that include so only the .cc compilation units that
// actually touch the filter pay the cost.
// EdgeFilter holds two EdgeKeepBitmap instances (directed / undirected) and
// routes set_kept / is_kept calls by the ObjectId's top-byte type tag,
// keying each bitmap by the 56-bit counter portion (ObjectId::VALUE_MASK),
// not the raw edge_id. This avoids the ~2e9 GB resize that would occur if
// the full tagged id were used as an index.
class EdgeFilter;

/**
 * @brief Hash key for detecting parallel edges (multigraph support).
 *
 * Composite key (from_node, to_node, type_id) excludes edge_id to enable
 * parallel edge detection. Multiple edges between same node pair with same
 * type will hash to same key.
 *
 * Memory: 24 bytes (3 × uint64_t)
 */
struct ParallelEdgeKey {
    uint64_t from_node;  ///< Source node ID
    uint64_t to_node;    ///< Target node ID
    uint64_t type_id;    ///< Relationship type ID

    bool operator==(const ParallelEdgeKey& other) const {
        return from_node == other.from_node &&
               to_node == other.to_node &&
               type_id == other.type_id;
    }
};

/**
 * @brief Hash functor for ParallelEdgeKey.
 *
 * Uses XOR with bit shifts for hash combination (standard pattern).
 * Collision rate: <0.001% for typical graphs.
 */
struct ParallelEdgeKeyHash {
    std::size_t operator()(const ParallelEdgeKey& k) const {
        return std::hash<uint64_t>{}(k.from_node) ^
               (std::hash<uint64_t>{}(k.to_node) << 1) ^
               (std::hash<uint64_t>{}(k.type_id) << 2);
    }
};

/**
 * @brief Tracks aggregation state for parallel edges.
 *
 * Maintains aggregation state for edges with same (from, to, type) triple.
 * Supports 5 strategies: SINGLE, MIN, MAX, SUM, COUNT.
 *
 * Memory: ~64 bytes per instance
 */
class EdgeAggregator {
public:
    EdgeAggregator(Aggregation strategy)
        : strategy_(strategy)
        , count_(0)
        , sum_value_(0.0)
        , min_value_(std::numeric_limits<double>::max())
        , max_value_(std::numeric_limits<double>::lowest())
        , has_value_(false)
        , first_edge_id_(0)
        , representative_edge_id_(0)
    {}

    /**
     * @brief Processes an edge for aggregation.
     *
     * @param edge_id The edge identifier
     * @param property_value Optional property value (for MIN/MAX/SUM)
     * @return true if edge should be kept, false if aggregated away
     */
    bool process_edge(ObjectId edge_id, std::optional<double> property_value);

    /**
     * @brief Gets the count of parallel edges seen.
     */
    uint64_t get_count() const { return count_; }

    /**
     * @brief Returns true if at least one non-NULL property value was seen.
     */
    bool has_value() const { return has_value_; }

    /**
     * @brief Gets the first edge ID (the one actually stored in the projection batch).
     */
    ObjectId get_first_edge() const { return first_edge_id_; }

    /**
     * @brief Gets the representative edge ID (for MIN/MAX, may differ from first).
     */
    ObjectId get_representative_edge() const { return representative_edge_id_; }

    /**
     * @brief Gets the aggregated property value (for SUM/MIN/MAX/COUNT).
     */
    double get_aggregated_value() const;

private:
    Aggregation strategy_;
    uint64_t count_;
    double sum_value_;
    double min_value_;
    double max_value_;
    bool has_value_;
    ObjectId first_edge_id_;
    ObjectId representative_edge_id_;
};

/**
 * @brief Detects and aggregates parallel edges during projection creation.
 *
 * Hash-based streaming aggregation with RAII memory management.
 * Cleared after each batch (1000 edges) to maintain constant memory.
 *
 * Memory overhead: 132 KB constant (independent of graph size)
 * Performance: O(1) average per edge (hash table lookup)
 */
class ParallelEdgeDetector {
public:
    explicit ParallelEdgeDetector(Aggregation strategy)
        : strategy_(strategy)
    {}

    /**
     * @brief Processes an edge and detects if it's a parallel edge.
     *
     * @param from_node Source node ID
     * @param to_node Target node ID
     * @param type_id Relationship type ID
     * @param edge_id Edge identifier
     * @param property_value Optional property value for aggregation
     * @return true if this is the first occurrence (add to batch), false if duplicate
     */
    bool process_edge(
        uint64_t from_node,
        uint64_t to_node,
        uint64_t type_id,
        ObjectId edge_id,
        std::optional<double> property_value = std::nullopt
    );

    /**
     * @brief Clears the hash table (called after batch flush).
     *
     * Frees ~132 KB memory per batch to maintain constant overhead.
     */
    void clear() {
        edge_map_.clear();
    }

    /**
     * @brief Gets statistics for debugging.
     */
    size_t get_map_size() const { return edge_map_.size(); }

    /**
     * @brief Gets aggregated values for edges that need property updates (SUM/COUNT).
     *
     * Returns a map of representative edge_id -> aggregated value for edges that
     * need their properties updated with aggregated values.
     *
     * @return Map of first_edge_id -> aggregated value (for SUM/COUNT/MIN/MAX modes)
     */
    std::unordered_map<uint64_t, double> get_aggregated_property_values() const {
        std::unordered_map<uint64_t, double> result;

        // SINGLE doesn't produce aggregated values
        if (strategy_ == Aggregation::SINGLE) {
            return result;
        }

        for (const auto& [key, aggregator] : edge_map_) {
            // COUNT always has a value; MIN/MAX/SUM only if a non-NULL property was seen
            if (strategy_ != Aggregation::COUNT && !aggregator.has_value()) {
                continue;  // Skip: all property values were NULL, sentinel would overflow int64_t
            }
            ObjectId first_edge = aggregator.get_first_edge();
            double agg_value = aggregator.get_aggregated_value();
            result[first_edge.id] = agg_value;
        }

        return result;
    }

private:
    Aggregation strategy_;
    std::unordered_map<ParallelEdgeKey, EdgeAggregator, ParallelEdgeKeyHash> edge_map_;
};

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
    /**
     * @brief Scan-pipeline selection for the projection build (Spec #2).
     *
     * CLASSIC (default)   — legacy pipeline: every scan pass emits to every
     *                       buffer simultaneously. Identical behavior to
     *                       pre-Spec-#2 code.
     * SERIALIZED          — new pipeline: one scan pass per target B+Tree,
     *                       each pass emits only to a single-index mask.
     *
     * Selected at process start via the MDB_PROJECTION_SERIAL_SCAN env var
     * ("1" / "true" / "yes" => SERIALIZED, anything else => CLASSIC) and
     * cached for the process lifetime via get_scan_mode().
     */
    enum class ScanMode { CLASSIC, SERIALIZED };

    static constexpr size_t BATCH_SIZE = 1000;

    /// @brief Synthetic key ID for COUNT aggregation's _count property.
    /// Real catalog key IDs start at 0 and grow. Value 1000 is safely above any
    /// realistic edge property count while avoiding vector over-allocation in catalog.
    static constexpr uint64_t COUNT_KEY_SYNTHETIC_ID = 1000;

    /// @brief Starting ID for synthetic keys allocated when a property is renamed
    /// (e.g. {score: {property: 'rating'}}). Kept above COUNT_KEY_SYNTHETIC_ID to
    /// avoid collisions; allocation is monotonic per-builder.
    static constexpr uint64_t RENAME_KEY_SYNTHETIC_START = 2000;

    /// @brief Threshold for switching to external sort-aggregate (1M edges)
    /// Below this threshold, in-memory hash-based aggregation is used.
    /// Above this threshold, external sort-aggregate provides bounded memory.
    static constexpr size_t STREAMING_AGGREGATION_THRESHOLD = 1000000;

    struct Statistics {
        uint64_t node_count = 0;
        uint64_t relationship_count = 0;
        std::chrono::milliseconds duration_ms{0};
        uint32_t feature_dim  = 0;   ///< GNN feature dimension (0 when GNN disabled)
        uint64_t num_classes  = 0;   ///< GNN number of unique label classes (0 when GNN disabled)
    };

    /**
     * @brief Constructs builder for new projection.
     *
     * @param projection_name Name for the new projection
     * @param db_folder Database root folder
     * @param node_properties Optional list of node property keys to project
     * @param edge_properties Optional list of edge property keys to project
     * @param orientation Edge directionality (NATURAL/REVERSE/UNDIRECTED), defaults to NATURAL
     * @param aggregation Parallel edge aggregation strategy (SINGLE/MIN/MAX/SUM/COUNT), defaults to SINGLE
     * @param aggregation_property Property to use for MIN/MAX/SUM aggregation (empty for COUNT/SINGLE)
     * @param type_orientations Per-type orientation overrides (type_name -> Orientation)
     * @param type_aggregations Per-type aggregation overrides (type_name -> Aggregation)
     * @param type_agg_properties Per-type aggregation property overrides (type_name -> property_key)
     * @param node_property_configs Per-property configuration for nodes (projected_name -> PropertyConfig)
     * @param edge_property_configs Per-property configuration for edges (projected_name -> PropertyConfig)
     * @param include_features GNN feature matrix name (empty = disabled)
     * @param label_property GNN classification label property (empty = disabled)
     * @param split_property GNN train/val/test split property (empty = disabled)
     * @param include_label_indexes If true (default, matches Neo4j GDS), builds
     *        node_label + label_node + edge_label + label_edge B+Tree indexes
     *        so `MATCH (n:Label)` queries on the projection use O(log n)
     *        lookups. Set to false to skip those 4 indexes when the workload
     *        doesn't query by label (e.g. pure GNN training), saving
     *        significant disk (~50 GB peak on papers100M) and build time.
     *        Queries that require labels on a projection built with
     *        include_label_indexes=false will throw QueryException with a
     *        message suggesting re-creation.
     *        See docs/superpowers/thesis_analysis/2026-04-20-projection-disk-reduction-analysis.md §3.A.
     */
    NativeProjectionBuilder(
        const std::string& projection_name,
        const std::string& db_folder,
        const std::vector<std::string>& node_properties = {},
        const std::vector<std::string>& edge_properties = {},
        Orientation orientation = Orientation::NATURAL,
        Aggregation aggregation = Aggregation::SINGLE,
        const std::string& aggregation_property = "",
        const std::unordered_map<std::string, Orientation>& type_orientations = {},
        const std::unordered_map<std::string, Aggregation>& type_aggregations = {},
        const std::unordered_map<std::string, std::string>& type_agg_properties = {},
        const std::unordered_map<std::string, PropertyConfig>& node_property_configs = {},
        const std::unordered_map<std::string, PropertyConfig>& edge_property_configs = {},
        const std::string& include_features = "",
        const std::string& label_property = "",
        const std::string& split_property = "",
        bool include_label_indexes = true
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
    /**
     * @brief Reads MDB_PROJECTION_SERIAL_SCAN env var once per process.
     *
     * Thread-safe via C++11 magic statics. Values "1"/"true"/"yes" enable
     * SERIALIZED; anything else (including unset) falls back to CLASSIC.
     * Parallel to MDB_PROJECTION_SORTER from Spec #1 — same env-var-opt-in
     * discipline, same process-lifetime caching.
     */
    static ScanMode get_scan_mode();

    // Classic single-pass scan implementations (current behavior, extracted
    // verbatim from scan_nodes_by_labels / scan_edges_by_types). Called
    // from the public API when ScanMode == CLASSIC (default).
    void scan_nodes_impl_classic_(const std::vector<std::string>& labels);
    void scan_edges_impl_classic_(const std::vector<std::string>& types);

    // Serialized multi-pass scan implementations (Spec #2). Each call
    // emits records ONLY to buffers matching target_mask. Called in a
    // loop by finalize_serialized_ with single-bit masks.
    // Implementations land in Tasks 7 (nodes) and 9 (edges).
    void scan_nodes_impl_serialized_(const std::vector<std::string>& labels,
                                     ProjectionIndex target_mask);
    void scan_edges_impl_serialized_(const std::vector<std::string>& types,
                                     ProjectionIndex target_mask,
                                     const EdgeFilter* filter);

    /**
     * @brief Serialized mode Phase B: precompute the edge-keep filter.
     *
     * Runs a single full scan over all edges of the given @p types,
     * evaluates the has_node() filter on both endpoints, and records the
     * outcome as a bit per edge counter in a fresh EdgeFilter. No record
     * emissions happen here — the filter is the sole side-effect.
     *
     * EdgeFilter holds two EdgeKeepBitmap instances (directed / undirected)
     * keyed by the 56-bit counter portion of the ObjectId (lower bits,
     * obtained via ObjectId::VALUE_MASK), NOT the raw tagged edge_id.id.
     * GQL edge ObjectIds carry an 8-bit type prefix (MASK_DIRECTED_EDGE =
     * 0xE0.. or MASK_UNDIRECTED_EDGE = 0xE4..) that would make the raw id
     * value ~1.6e19, causing std::bad_alloc on the first resize. The counter
     * portion is a dense index starting from 0 per orientation, keeping peak
     * RSS at ~200 MB for papers100M (1.6B directed edges).
     *
     * Consumed read-only by Phase C's 9 edge-index passes
     * (scan_edges_impl_serialized_ with different target masks), so the
     * has_node() work is paid once instead of 9×. ParallelEdgeDetector
     * is NOT run here — it lives in scan_edges_impl_serialized_ (first
     * FROM_TO_EDGE pass) with per-batch clear() to keep memory bounded.
     * Called exactly once by finalize_serialized_ (Task 10) after
     * Phase A has populated ProjectionStorage::collected_nodes_.
     *
     * Also resizes ProjectionStorage's Bloom filter based on the total
     * estimated edge count, matching scan_edges_impl_classic_'s
     * contract so downstream has_edge() probes keep their <1% false
     * positive target on large graphs.
     *
     * Memory: ~1 bit per edge counter per orientation.
     * Papers100M (1.6B directed CITES edges): ~200 MB directed bitmap,
     * ~0 MB undirected bitmap.
     *
     * Spec: §4 Phase B, §6 invariant I2 (filter is write-once).
     *
     * @param types Relationship type names to scan (same set accepted
     *        by scan_edges_by_types / scan_edges_impl_classic_).
     * @return Owning pointer to the finalized filter (Phase C consumes
     *         it via raw pointer).
     * @throws std::runtime_error if any type is absent from the catalog.
     */
    std::unique_ptr<EdgeFilter> precompute_edge_filter_(
        const std::vector<std::string>& types);

    /**
     * @brief Spec #2 orchestrator: executes Phase A + B + C when ScanMode is
     *        SERIALIZED. Called by finalize() (wired in Task 11) after the
     *        public scan_*_by_* methods have captured inputs into stored_labels_
     *        / stored_types_.
     *
     * Falls back to the classic single-pass path when has_non_single_
     * aggregation_() returns true (spec §3 D8) because aggregation state
     * would be too large to persist across 9 edge-index passes.
     */
    void finalize_serialized_();

    /**
     * @brief Compute the ordered list of ProjectionIndex single-bit values to
     *        iterate during Phase A + Phase C, based on features flags.
     *
     * Node phase (NODES + optional label/property indexes) is emitted
     * first, followed by edge phase (core 5 + optional label/property).
     *
     * IMPORTANT: under ENABLE_GNN, pushes NODE_KEY_VALUE + KEY_VALUE_NODE
     * whenever gnn_row_mapping_ is non-null, even if no node properties are
     * configured. This preserves classic's GNN label/split extraction
     * side-effect via try_extract_gnn_property (see
     * scan_nodes_impl_serialized_'s TODO(task10-gnn)).
     */
    std::vector<ProjectionIndex> enabled_indexes_() const;

    /**
     * @brief Returns true if any type in stored_types_ uses an aggregation
     *        mode other than SINGLE.
     *
     * Serialized mode falls back to classic in that case (spec §3 D8).
     * Aggregation state (COUNT/SUM/MIN/MAX maps) is too large to persist
     * across 9 edge-index passes.
     */
    bool has_non_single_aggregation_() const;

    // Input state captured from the public scan_* methods for later
    // replay in finalize_serialized_ (Task 10). scan_inputs_captured_
    // prevents double-capture.
    std::vector<std::string> stored_labels_;
    std::vector<std::string> stored_types_;
    bool scan_inputs_captured_ = false;

    std::string projection_name;
    std::string db_folder;
    std::unique_ptr<ProjectionStorage> storage;
    std::unique_ptr<NativeScanner> scanner;
    Statistics stats;
    bool finalized_ = false;

    // Property configuration (simple property lists)
    std::vector<std::string> node_property_keys;
    std::vector<std::string> edge_property_keys;

    // Per-property configuration (Phase 3: renaming, defaults, per-property aggregation)
    std::unordered_map<std::string, PropertyConfig> node_prop_configs;
    std::unordered_map<std::string, PropertyConfig> edge_prop_configs;

    // Orientation configuration
    Orientation orientation;

    // Aggregation configuration (global defaults)
    Aggregation aggregation;
    std::string aggregation_property_key;  // Property to use for MIN/MAX/SUM aggregation

    // GNN extension fields (stored for Task 11 extraction logic)
    std::string include_features_;
    std::string label_property_;
    std::string split_property_;

    // Opt-in flag (default true) controlling whether label indexes are built.
    // Kept as a builder member so the decision propagates to features
    // initialization in ctor body without re-reading the constructor arg.
    bool include_label_indexes_ = true;

    // Per-type configuration overrides (Neo4j GDS per-type config)
    std::unordered_map<std::string, Orientation> per_type_orientations;
    std::unordered_map<std::string, Aggregation> per_type_aggregations;
    std::unordered_map<std::string, std::string> per_type_agg_properties;

    // Inverse catalog maps: masked key ID -> property name (O(1) lookup)
    // Built once in constructor, replaces O(K) linear scans in extract_*_properties()
    std::unordered_map<uint64_t, std::string> node_key_id_to_name;
    std::unordered_map<uint64_t, std::string> edge_key_id_to_name;

    // Synthetic key IDs allocated for renamed/defaulted properties that have no
    // counterpart in the main catalog. Keyed by projected property name.
    std::unordered_map<std::string, uint64_t> synthetic_node_key_ids;
    std::unordered_map<std::string, uint64_t> synthetic_edge_key_ids;
    uint64_t next_synthetic_key_id = RENAME_KEY_SYNTHETIC_START;

    // Allocates (or returns) a synthetic key id for a projected property name
    // and registers it with the projection storage. Used when renaming produces
    // a key absent from the main catalog, or when a default value must materialize
    // under a projected name that doesn't exist on any source node.
    uint64_t ensure_projected_node_key(const std::string& projected_name);
    uint64_t ensure_projected_edge_key(const std::string& projected_name);

    std::chrono::steady_clock::time_point start_time;

    // Benchmark instrumentation (activated by MDB_BENCHMARK=1 env var)
    ProjectionTimers benchmark_timers_;

    // Per-type lookup methods (returns global default if type not in map)
    Orientation get_orientation_for_type(const std::string& type_name) const;
    Aggregation get_aggregation_for_type(const std::string& type_name) const;
    std::string get_aggregation_property_for_type(const std::string& type_name) const;

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

    /**
     * @brief Extracts edge properties while excluding a specific property.
     *
     * Used during SUM/COUNT aggregation to prevent storing the original
     * property value before the aggregated value is computed. Without this,
     * the B+tree would contain both values and queries would return the
     * original instead of the aggregated value.
     *
     * @param edge_id The edge to extract properties from
     * @param exclude_property Property name to skip (the aggregation property)
     */
    void extract_edge_properties_excluding(ObjectId edge_id, const std::string& exclude_property);

    /**
     * @brief Scans edges using external sort-aggregate for memory-efficient COUNT/SUM.
     *
     * This method is used when the estimated edge count exceeds
     * STREAMING_AGGREGATION_THRESHOLD. It uses O(B) memory instead of O(N)
     * by streaming edges to disk, sorting externally, and aggregating in
     * a single pass.
     *
     * ## Algorithm
     * 1. Collection: Stream edges to EdgeAggregationBuffer (64 MB threshold)
     * 2. Sort: External K-way merge-sort by (from, to, type)
     * 3. Aggregate: Single pass with O(1) memory per group
     *
     * @param types Vector of relationship type names
     * @param type_id_map Map from type name to ObjectId for fast lookup
     */
    void scan_edges_with_streaming_aggregation(
        const std::vector<std::string>& types,
        const std::unordered_map<std::string, ObjectId>& type_id_map
    );

    /**
     * @brief Extracts a single property value from an edge for aggregation.
     *
     * Looks up the specified property on the edge and converts it to a double
     * for use in MIN/MAX/SUM aggregation.
     *
     * @param edge_id The edge to extract property from
     * @param property_key The property name to extract
     * @return std::optional<double> containing the property value, or std::nullopt if:
     *         - Property doesn't exist on this edge
     *         - Property value is not numeric
     */
    std::optional<double> get_edge_property_value_for_aggregation(
        ObjectId edge_id,
        const std::string& property_key
    );

    // GNN label/split extraction helper (called from both code paths in extract_node_properties)
    void try_extract_gnn_property(ObjectId node_id, const std::string& key_name, ObjectId value_id);

#ifdef ENABLE_GNN
    // GNN data extraction state (only present when GNN module is enabled)
    std::unique_ptr<mdb::gnn::RowMapping> gnn_row_mapping_;
    std::vector<int64_t>  labels_buffer_;
    std::vector<uint8_t>  splits_buffer_;
    std::unordered_set<int64_t> unique_classes_;
    uint32_t feature_dim_ = 0;
#endif
};

namespace detail {
    /**
     * @brief Test-only helper that re-runs the MDB_PROJECTION_SERIAL_SCAN
     *        parser against an explicit env-var value, bypassing the
     *        process-lifetime cache in NativeProjectionBuilder::get_scan_mode().
     *
     * The production path uses a C++11 magic static that is set once per
     * process on first call, so per-test env-var variations cannot be
     * exercised through it. This helper shares the exact same parse logic
     * and returns the identical enum, making it safe to use for coverage
     * of truthy / unknown / null inputs.
     *
     * @param env_val nullable C-string as returned by std::getenv.
     */
    NativeProjectionBuilder::ScanMode init_scan_mode_for_test(const char* env_val);
}

} // namespace GQL
