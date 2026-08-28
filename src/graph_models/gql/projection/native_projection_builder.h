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
#include "storage/index/bplus_tree/bpt_leaf_format.h"   // For BPT::LeafFormat (delta + LEB128-varint leaf encoding)

namespace GQL { enum class IndexSet : uint8_t; }  // fwd: defined in index_set.h

#ifdef ENABLE_GNN
#include "gnn/storage/row_mapping.h"
#endif

namespace GQL {

/**
 * @brief Bitmask enum identifying which projection B+Tree index is being
 *        populated during a scan pass. Used by the serialized scan pipeline
 *        to gate record emissions in scan callbacks (one pass per target
 *        B+Tree, each pass emits only to a single-index mask).
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
 *
 * Three properties this struct is required to hold, each of which it previously
 * did not:
 *
 * 1. The clock is `Clock`, aliased to steady_clock, and every call site uses
 *    that alias rather than naming a clock itself. libstdc++ aliases
 *    high_resolution_clock to system_clock, whose `is_steady` is false and which
 *    steps under NTP correction; a projection of a large graph runs for tens of
 *    minutes, which is ample time for that to happen mid-measurement.
 *
 * 2. `total_ms` spans the builder's whole lifetime, from construction to the end
 *    of finalize(), so it CONTAINS every phase below it. It shares its anchor
 *    with `Statistics::duration_ms` but not its endpoint: duration_ms closes
 *    before the finalize tail (GNN sidecars, topology snapshot, catalog
 *    refresh), so total_ms is that duration plus the tail and can never be the
 *    smaller of the two.
 *
 * 3. Whatever the phase timers do not account for is printed as `residual`
 *    rather than left invisible. The columns therefore sum to 100% by
 *    construction, and an unmeasured phase surfaces as a large residual instead
 *    of silently pushing the other columns past 100%.
 *
 * Phases named here are only those actually measured. A phase that is not timed
 * is absent rather than printed as 0.0 ms, because a zero reads as "this costs
 * nothing" when the truth is "nobody looked".
 */
struct ProjectionTimers {
    using Clock = std::chrono::steady_clock;

    double node_scan_ms  = 0;
    double edge_scan_ms  = 0;
    /// Sort and index build combined. These two are NOT separated: the code
    /// times one interval that covers both. Splitting them requires timing
    /// inside the sorter, which the interval here cannot see.
    double sort_btree_ms = 0;
    double metadata_ms   = 0;
    double total_ms      = 0;
    bool enabled         = false;

    double accounted_ms() const {
        return node_scan_ms + edge_scan_ms + sort_btree_ms + metadata_ms;
    }

    /// Time inside the total that no phase timer claimed. Must never be
    /// negative: a negative residual means two timers covered the same
    /// interval, which would inflate every percentage.
    double residual_ms() const { return total_ms - accounted_ms(); }

    void print(const std::string& proj_name, uint64_t edge_count) const {
        if (!enabled) return;
        auto pct = [&](double v) -> double { return total_ms > 0 ? v / total_ms * 100 : 0; };
        fprintf(stderr, "[BENCHMARK] graph_project '%s' — %llu edges\n",
                proj_name.c_str(), (unsigned long long)edge_count);
        fprintf(stderr, "[BENCHMARK]   node_scan:     %8.1f ms  (%4.1f%%)\n", node_scan_ms, pct(node_scan_ms));
        fprintf(stderr, "[BENCHMARK]   edge_scan:     %8.1f ms  (%4.1f%%)\n", edge_scan_ms, pct(edge_scan_ms));
        fprintf(stderr, "[BENCHMARK]   sort+btree:    %8.1f ms  (%4.1f%%)\n", sort_btree_ms, pct(sort_btree_ms));
        fprintf(stderr, "[BENCHMARK]   metadata:      %8.1f ms  (%4.1f%%)\n", metadata_ms, pct(metadata_ms));
        fprintf(stderr, "[BENCHMARK]   residual:      %8.1f ms  (%4.1f%%)\n", residual_ms(), pct(residual_ms()));
        fprintf(stderr, "[BENCHMARK]   total:         %8.1f ms  (100.0%%)\n", total_ms);
        if (residual_ms() < 0) {
            fprintf(stderr,
                    "[BENCHMARK]   WARNING: negative residual — phase timers overlap, "
                    "percentages above are not trustworthy\n");
        }
    }
};

// Use Orientation, Aggregation, and PropertyConfig from Procedures namespace
using Procedures::Orientation;
using Procedures::Aggregation;
using Procedures::PropertyConfig;

// Forward declaration (NativeScanner implemented by another agent)
class NativeScanner;

// Forward declaration for the serialized-scan pipeline (one pass per
// target B+Tree index, gated by a single-bit ProjectionIndex mask).
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
 * Under SINGLE mode the map is cleared after each batch (BATCH_SIZE = 1000
 * edges) to maintain constant memory; MIN/MAX/SUM/COUNT keep their state
 * for the full per-type scan and clear once per type.
 *
 * SINGLE-mode caveat: the per-batch clear() drops the seen-edge set (which
 * IS the detector's SINGLE state), so duplicate detection is best-effort
 * within a BATCH_SIZE-edge window of the scan order. Two parallel edges
 * more than BATCH_SIZE apart are BOTH stored without the QueryException.
 * Deliberate memory trade-off: an uncleared detector holds every kept edge
 * (~138 bytes each — 25 GB RSS on papers100M). An exact check would have
 * to run where records arrive sorted by (from, to), i.e. during the
 * from_to_edge index build in ProjectionStorage.
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
                continue;  // Skip: all property values were NULL — the min/max/sum sentinels are not real aggregates
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
 * @brief Packs an aggregated parallel-edge value into a persistable ObjectId.
 *
 * COUNT aggregates are integral by construction and pack as integers.
 * SUM/MIN/MAX aggregate double-typed property values and must pack as
 * doubles (via the persistent string_manager encoding, since projections
 * live on disk across sessions): truncating them to int64 would silently
 * corrupt fractional aggregates and change the property's type. Not
 * meaningful for SINGLE, which produces no aggregated values.
 */
ObjectId pack_aggregated_property_value(Aggregation aggregation, double agg_value);

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
 */
class NativeProjectionBuilder {
public:
    /**
     * @brief Scan-pipeline selection for the projection build.
     *
     * CLASSIC (default)   — legacy pipeline: every scan pass emits to every
     *                       buffer simultaneously. Identical behavior to the
     *                       original single-pass code.
     * SERIALIZED          — new pipeline: one scan pass per target B+Tree,
     *                       each pass emits only to a single-index mask.
     * PARALLEL            — builder-level parallel edge scan: has_node() +
     *                       endpoint orientation run inside TBB workers; the
     *                       order-sensitive tail (ParallelEdgeDetector window,
     *                       edge_batch push, add_edge_label) replays on the
     *                       main thread in ascending-partition / key order so
     *                       the sorter input is byte-identical to CLASSIC.
     *                       Only engaged when every type is SINGLE-aggregation
     *                       with no aggregation property and the projection has
     *                       no edge-property config; otherwise falls back to
     *                       the CLASSIC edge scan. Node scan is unaffected.
     *
     * CLASSIC / SERIALIZED are selected at process start via the
     * MDB_PROJECTION_SERIAL_SCAN env var ("1" / "true" / "yes" =>
     * SERIALIZED). PARALLEL is selected via the SEPARATE
     * MDB_PROJECTION_PARALLEL_SCAN env var ("1" / "true" / "yes"). When both
     * are set, SERIALIZED wins (it is the more invasive pipeline). The result
     * is cached for the process lifetime via get_scan_mode().
     */
    enum class ScanMode { CLASSIC, SERIALIZED, PARALLEL };

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
     * @param index_set User-selected preset controlling which B+Tree indexes
     *        will be materialized at build time. Defaults to IndexSet::ALL
     *        which preserves prior behavior. The value is propagated to
     *        ProjectionStorage and consulted by the build phases, which skip
     *        any topology index absent from the preset's mask; property
     *        indexes remain governed solely by the property configuration.
     * @param build_topology_snapshot When true, emit `topology_fwd.csr`
     *        and/or `topology_rev.csr` mmap-backed CSR sidecar files
     *        alongside the projection's B+Tree files after the normal
     *        `finalize_serialized_()` / flush step completes. These sidecars
     *        enable O(1) neighbor slicing for GNN sampling (versus O(log N)
     *        B+Tree lookups). Each direction is emitted only when the
     *        corresponding edge index (FROM_TO_EDGE / TO_FROM_EDGE) is
     *        present in the active IndexSet mask; otherwise the direction is
     *        skipped with a single warning line. Errors during sidecar
     *        generation are non-fatal: the projection remains valid and the
     *        user can retry via the post-hoc `gnn_build_topology_snapshot`
     *        procedure. Default is false. The GQL surface (config key +
     *        YIELD field) is wired in the project_procedure; this constructor
     *        parameter is the builder-level hook the test suite drives
     *        directly.
     * @param leaf_format Selects the on-disk B+Tree leaf encoding for every
     *        index materialized by this projection.
     *        BPT::LeafFormat::BITSET (default) preserves the prior
     *        byte-identical behavior. BPT::LeafFormat::DELTA_VARINT opts
     *        into the delta + LEB128-varint v2 leaf layout, which exploits
     *        sort-order to reduce leaf disk size ~80% on typical graphs.
     *        The value is threaded to ProjectionStorage (for per-index
     *        BPlusTree reader construction) and persisted per materialized
     *        index in catalog v1.5.
     * @param graph_storage Selects the per-projection graph-storage mode.
     *        BTREE (default) preserves prior behavior byte-for-byte.
     *        CSR_HYBRID opts the edge-index B+Tree leaves into the CSR
     *        layout directly (the leaf IS the CSR, enabling O(1) neighbor
     *        access from the B+Tree itself); however, this constructor
     *        parameter only plumbs the config through the builder and
     *        catalog — the build pipeline itself still emits BTREE leaves
     *        regardless of this value. Wiring BPTLeafCSRWriter into
     *        sorter_dispatch.cc completes the feature.
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
        bool include_label_indexes = true,
        IndexSet index_set = static_cast<IndexSet>(0),  // IndexSet::ALL (fwd-declared)
        bool build_topology_snapshot = false,
        BPT::LeafFormat leaf_format = BPT::LeafFormat::BITSET,
        BPT::GraphStorage graph_storage = BPT::GraphStorage::BTREE
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

    /**
     * @brief Returns the IndexSet preset stored on the builder.
     *
     * Reflects the value wired in by graph_project's `indexSet` config key
     * (or IndexSet::ALL when the key is absent / the legacy positional
     * constructor is used). Exposed for testability of the parsing logic;
     * the build phases and catalog serialization consume the same stored
     * value. Defined out-of-line in the .cc so this header can keep
     * IndexSet as a forward declaration.
     */
    IndexSet get_index_set() const noexcept;

    /**
     * @brief Returns whether the builder will emit mmap-backed CSR topology
     *        sidecar files (topology_fwd.csr / topology_rev.csr) after
     *        finalize.
     *
     * Mirrors the `build_topology_snapshot` constructor argument. A true
     * value is advisory: the builder still skips the emission for any
     * direction whose edge index is not in the active IndexSet mask.
     */
    bool get_build_topology_snapshot() const noexcept { return build_topology_snapshot_; }

    /**
     * @brief Returns the BPT::LeafFormat stored on the builder.
     *
     * Reflects the value wired in by graph_project's `leafFormat` config key
     * (BITSET = legacy shared-prefix bitset; DELTA_VARINT = delta + LEB128
     * varint v2 layout; or BITSET when the key is absent / the legacy
     * positional constructor is used). Exposed for testability; the actual
     * propagation to ProjectionStorage and per-index BPlusTree readers
     * happens inside the ctor body.
     */
    BPT::LeafFormat get_leaf_format() const noexcept { return leaf_format_; }

    /**
     * @brief Returns the BPT::GraphStorage stored on the builder.
     *
     * Reflects the value wired in by graph_project's `graphStorage` config
     * key (BTREE = legacy B+Tree leaves; CSR_HYBRID = the edge-index B+Tree
     * leaves store the CSR layout directly enabling O(1) neighbor access;
     * or BTREE when the key is absent / the legacy positional constructor is
     * used). Exposed for testability; the actual propagation to
     * ProjectionStorage (and thence to the v1.6 catalog byte) happens inside
     * the ctor body.
     */
    BPT::GraphStorage get_graph_storage() const noexcept {
        return graph_storage_;
    }

private:
    /**
     * @brief Reads MDB_PROJECTION_SERIAL_SCAN env var once per process.
     *
     * Thread-safe via C++11 magic statics. Values "1"/"true"/"yes" enable
     * SERIALIZED; anything else (including unset) falls back to CLASSIC.
     * Follows the same env-var-opt-in discipline and process-lifetime
     * caching pattern used by the MDB_PROJECTION_SORTER selector.
     */
    static ScanMode get_scan_mode();

    // Classic single-pass scan implementations (current behavior, extracted
    // verbatim from scan_nodes_by_labels / scan_edges_by_types). Called
    // from the public API when ScanMode == CLASSIC (default).
    void scan_nodes_impl_classic_(const std::vector<std::string>& labels);
    void scan_edges_impl_classic_(const std::vector<std::string>& types);

    /**
     * @brief Builder-level PARALLEL edge scan (ScanMode::PARALLEL).
     *
     * Runs the per-edge has_node() membership filter + endpoint orientation
     * INSIDE TBB workers (one per id sub-range), accumulating kept
     * ProjectedEdges into a thread-local per-partition vector. Then, on the
     * main thread, replays the kept edges in ASCENDING partition / key order
     * through the EXACT sequential tail the CLASSIC path runs
     * (ParallelEdgeDetector::process_edge windowed dedup, edge_batch push,
     * add_edge_label, BATCH_SIZE flush + detector->clear()). This reproduces
     * the CLASSIC global scan order, hence byte-identical sorter input.
     *
     * Precondition: all_single_no_aggprop_(types) is true. The caller
     * (scan_edges_by_types) gates on it and otherwise dispatches to
     * scan_edges_impl_classic_. SINGLE-only with no aggregation property and
     * no edge-property config keeps the parallelized part a pure per-edge
     * function and avoids the non-thread-safe property/aggregation tails.
     */
    void scan_edges_impl_parallel_(const std::vector<std::string>& types);

    /**
     * @brief Predicate gating ScanMode::PARALLEL eligibility.
     *
     * True iff every type in @p types resolves to Aggregation::SINGLE with an
     * empty aggregation property, AND the projection has no edge-property
     * configuration (edge_property_keys + edge_prop_configs both empty). Under
     * those conditions the per-edge work parallelized in
     * scan_edges_impl_parallel_ (has_node + orientation) is a pure function of
     * each edge and the order-sensitive tail is limited to the windowed
     * SINGLE-duplicate detector + edge_batch + add_edge_label, all of which
     * are replayed single-threaded in the ordered merge.
     */
    bool all_single_no_aggprop_(const std::vector<std::string>& types) const;

    // Serialized multi-pass scan implementations. Each call emits records
    // ONLY to buffers matching target_mask. Called in a loop by
    // finalize_serialized_ with single-bit masks: one scan pass per
    // target B+Tree index.
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
     * Called exactly once by finalize_serialized_ after
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
     * Invariant: the filter is written exactly once here (Phase B) and
     *            consumed read-only by all subsequent Phase C edge-index passes.
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
     * @brief Serialized-mode orchestrator: executes Phase A (node scan) +
     *        Phase B (edge filter precomputation) + Phase C (per-index edge
     *        scans) when ScanMode is SERIALIZED. Called by finalize() after
     *        the public scan_*_by_* methods have captured inputs into
     *        stored_labels_ / stored_types_.
     *
     * Falls back to the classic single-pass path when
     * has_non_single_aggregation_() returns true because aggregation state
     * (COUNT/SUM/MIN/MAX maps) would be too large to persist across 9
     * separate edge-index passes.
     */
    void finalize_serialized_();

    /**
     * @brief Emit mmap-backed CSR topology sidecar files for the projection
     *        when requested.
     *
     * No-op when `build_topology_snapshot_` is false. Otherwise, for each
     * of the two directions (FORWARD / REVERSE), the corresponding edge
     * index bit in the active IndexSet mask is probed; when the bit is set
     * and the projection's B+Tree is open, a `topology_fwd.csr` or
     * `topology_rev.csr` file is generated via TopologySnapshotWriter.
     * These sidecar files enable O(1) neighbor slicing for GNN sampling
     * instead of O(log N) B+Tree directory walks.
     *
     * Called at the tail of `finalize()` after `storage->flush()` so the
     * B+Tree `.leaf` / `.dir` files are complete and fsync'd on disk
     * before the sidecar writer hashes them.
     *
     * Non-fatal: any exception raised while building a sidecar is caught
     * and logged; the projection remains valid and the user can retry
     * through the post-hoc `gnn_build_topology_snapshot` procedure.
     */
    void build_topology_snapshots_();

    /**
     * @brief Single-direction helper used by `build_topology_snapshots_()`.
     *
     * Scans the supplied B+Tree twice: first to build the per-source
     * degree histogram, then to stream edges in src-monotonic order into
     * a freshly constructed TopologySnapshotWriter that emits the mmap-backed
     * CSR sidecar file. Caller is responsible for gating on the active
     * IndexSet mask — this helper trusts its arguments.
     */
    void build_one_topology_snapshot_(int direction /* 0=FORWARD, 1=REVERSE */,
                                      void* edge_bpt_opaque);

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
     * side-effect via try_extract_gnn_property (see the matching
     * comment in scan_nodes_impl_serialized_).
     */
    std::vector<ProjectionIndex> enabled_indexes_() const;

    /**
     * @brief Returns true if any type in stored_types_ uses an aggregation
     *        mode other than SINGLE.
     *
     * Serialized mode falls back to classic (with a warning) in that case:
     * the aggregation state (COUNT/SUM/MIN/MAX maps) grows with the number
     * of distinct (from, to, type) groups — tens of GB on billion-edge
     * graphs — and holding it across 9 edge-index passes would defeat the
     * bounded-memory point of the serialized pipeline.
     */
    bool has_non_single_aggregation_() const;

    // Input state captured from the public scan_* methods for later
    // replay in finalize_serialized_. scan_inputs_captured_
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

    // GNN extension fields consumed by try_extract_gnn_property
    std::string include_features_;
    std::string label_property_;
    std::string split_property_;

    // Opt-in flag (default true) controlling whether label indexes are built.
    // Kept as a builder member so the decision propagates to features
    // initialization in ctor body without re-reading the constructor arg.
    bool include_label_indexes_ = true;

    // User-selected index preset controlling which B+Tree indexes are
    // materialized. Stored here so catalog serialization and build-phase
    // gating can consume it without re-plumbing the constructor. Default
    // is ALL; the actual default-initialization lives in the ctor to keep
    // the forward-declared enum viable as a class member.
    IndexSet index_set_;

    // Opt-in flag controlling mmap-backed CSR topology sidecar emission.
    // When true, `finalize()` emits `topology_fwd.csr` / `topology_rev.csr`
    // (gated per-direction by the active IndexSet mask) after the normal
    // B+Tree build completes, enabling O(1) neighbor slicing for GNN
    // sampling. Default false preserves existing behavior for all callers
    // that do not request sidecars.
    bool build_topology_snapshot_ = false;

    // On-disk B+Tree leaf encoding preset (BITSET = legacy shared-prefix
    // bitset; DELTA_VARINT = delta + LEB128-varint v2 layout ~80% smaller).
    // Threaded through ProjectionStorage before build so each materialized
    // B+Tree reader is constructed with the matching BPT::LeafFormat, and
    // persisted per-index in catalog v1.5 (one byte per materialized index).
    // Default BITSET preserves byte-identical behavior for all prior callers.
    BPT::LeafFormat leaf_format_ = BPT::LeafFormat::BITSET;

    // Per-projection graph-storage mode for edge indexes. Threaded through
    // ProjectionStorage so save_catalog() can populate the v1.6
    // graph_storage byte. BTREE (default) preserves prior byte-for-byte
    // behavior. CSR_HYBRID opts the edge-index B+Tree leaves into the CSR
    // layout (the leaf IS the CSR, enabling O(1) neighbor access directly
    // from the B+Tree); wiring BPTLeafCSRWriter into the edge-index
    // sorter-dispatch path completes the feature.
    BPT::GraphStorage graph_storage_ = BPT::GraphStorage::BTREE;

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

    /**
     * @brief Test-only hook that runs the exact same mmap-backed CSR sidecar
     *        emission logic that `NativeProjectionBuilder::finalize()` invokes,
     *        but against a supplied ProjectionStorage instead of a live builder.
     *
     * This is the cleanest hook the builder path exposes for gtest coverage
     * without requiring the full graph-load + catalog machinery. The
     * production builder method (`build_topology_snapshots_`) is a thin
     * gate over the same routine (it consults the IndexSet mask and calls
     * per-direction builds); the underlying work — scan B+Tree twice,
     * stream into TopologySnapshotWriter, finalize — is shared byte-for-byte.
     *
     * @param storage Open projection storage with `from_to_edge_index` /
     *        `to_from_edge_index` already populated via
     *        `build_all_indexes_bulk()` and mapped by
     *        `open_all_bplustree_readers_()`.
     * @param build_forward If true, emit `topology_fwd.csr`.
     * @param build_reverse If true, emit `topology_rev.csr`.
     *
     * Throws `std::runtime_error` on I/O failure (different policy than
     * the production method, which swallows errors). Tests want the
     * exception so a regression is caught at assert time instead of
     * drifting into stderr.
     */
    void build_topology_snapshots_for_test(
        ProjectionStorage& storage,
        bool build_forward,
        bool build_reverse);
}

} // namespace GQL
