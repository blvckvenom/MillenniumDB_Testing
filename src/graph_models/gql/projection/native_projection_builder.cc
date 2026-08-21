#include "native_projection_builder.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/edge_filter.h"
#include "graph_models/gql/projection/edge_keep_bitmap_gpu.h"
#include "graph_models/gql/projection/external_edge_sort.h"
#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_scanner.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/streaming_aggregator.h"
#include "graph_models/gql/projection/topology_snapshot_writer.h"
#include "query/exceptions.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "system/string_manager.h"

#ifdef ENABLE_GNN
#include "gnn/projection/gnn_meta.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_object_id.h"
#endif

using namespace GQL;

// ============================================================================
// ScanMode selector: MDB_PROJECTION_SERIAL_SCAN env-var parsing.
//
// The production path (get_scan_mode) caches the result for the process
// lifetime via a C++11 magic static so downstream callers pay zero overhead
// after the first invocation.
//
// The detail::init_scan_mode_for_test helper exposes the same parse rules
// without the cache so unit tests can cover truthy / unknown / null inputs
// deterministically.
// ============================================================================
namespace {
bool truthy_env_(const char* env) {
    if (env == nullptr) return false;
    std::string v(env);
    return v == "1" || v == "true" || v == "yes";
}

NativeProjectionBuilder::ScanMode init_scan_mode() {
    // SERIALIZED is the more invasive pipeline and wins when both env vars
    // are set. PARALLEL gates on its OWN separate env var so it never
    // collides with the scanner's MDB_PROJECTION_PARALLEL_EDGE_SCAN knob.
    if (truthy_env_(std::getenv("MDB_PROJECTION_SERIAL_SCAN"))) {
        return NativeProjectionBuilder::ScanMode::SERIALIZED;
    }
    if (truthy_env_(std::getenv("MDB_PROJECTION_PARALLEL_SCAN"))) {
        return NativeProjectionBuilder::ScanMode::PARALLEL;
    }
    return NativeProjectionBuilder::ScanMode::CLASSIC;
}
} // namespace

NativeProjectionBuilder::ScanMode NativeProjectionBuilder::get_scan_mode() {
    static const ScanMode cached = init_scan_mode();
    return cached;
}

namespace GQL {
namespace detail {
NativeProjectionBuilder::ScanMode init_scan_mode_for_test(const char* env_val) {
    if (env_val == nullptr) return NativeProjectionBuilder::ScanMode::CLASSIC;
    std::string v(env_val);
    if (v == "1" || v == "true" || v == "yes") {
        return NativeProjectionBuilder::ScanMode::SERIALIZED;
    }
    return NativeProjectionBuilder::ScanMode::CLASSIC;
}
} // namespace detail
} // namespace GQL

uint64_t NativeProjectionBuilder::ensure_projected_node_key(const std::string& projected_name)
{
    // Prefer reusing a main-catalog id so the projection stays compatible with
    // any code that reads through the shared key namespace.
    auto cat_it = gql_model.catalog.node_keys2id.find(projected_name);
    if (cat_it != gql_model.catalog.node_keys2id.end()) {
        storage->register_node_key(projected_name, cat_it->second);
        return cat_it->second;
    }

    auto it = synthetic_node_key_ids.find(projected_name);
    if (it != synthetic_node_key_ids.end()) {
        return it->second;
    }

    uint64_t new_id = next_synthetic_key_id++;
    synthetic_node_key_ids[projected_name] = new_id;
    storage->register_node_key(projected_name, new_id);
    return new_id;
}

uint64_t NativeProjectionBuilder::ensure_projected_edge_key(const std::string& projected_name)
{
    auto cat_it = gql_model.catalog.edge_keys2id.find(projected_name);
    if (cat_it != gql_model.catalog.edge_keys2id.end()) {
        storage->register_edge_key(projected_name, cat_it->second);
        return cat_it->second;
    }

    auto it = synthetic_edge_key_ids.find(projected_name);
    if (it != synthetic_edge_key_ids.end()) {
        return it->second;
    }

    uint64_t new_id = next_synthetic_key_id++;
    synthetic_edge_key_ids[projected_name] = new_id;
    storage->register_edge_key(projected_name, new_id);
    return new_id;
}

// Persistently packs a double into the string_manager and returns an ObjectId
// tagged MASK_DOUBLE_EXTERN. Avoids the tmp_manager fallback in the generic
// pack_double helper, which is transient and produces invalid reads once the
// session ends (the projection lives on disk across restarts).
static ObjectId pack_double_persistent(double value)
{
    auto bytes = reinterpret_cast<const char*>(&value);
    uint64_t bytes_id = string_manager.get_or_create(bytes, sizeof(double));
    return ObjectId(ObjectId::MASK_DOUBLE_EXTERN | bytes_id);
}

ObjectId GQL::pack_aggregated_property_value(Aggregation aggregation, double agg_value)
{
    // COUNT is integral by construction. SUM/MIN/MAX aggregate double-typed
    // property values, so the result must persist as a double: truncating to
    // int64 corrupts fractional aggregates (SUM of 0.5 + 0.5 must be 1.0,
    // MIN of 2.7 must stay 2.7) and changes the property's type from double
    // to int even when the value happens to be whole.
    if (aggregation == Aggregation::COUNT) {
        return Common::Conversions::pack_int(static_cast<int64_t>(agg_value));
    }
    return pack_double_persistent(agg_value);
}

NativeProjectionBuilder::NativeProjectionBuilder(
    const std::string& projection_name_,
    const std::string& db_folder_,
    const std::vector<std::string>& node_properties,
    const std::vector<std::string>& edge_properties,
    Orientation orientation_,
    Aggregation aggregation_,
    const std::string& aggregation_property,
    const std::unordered_map<std::string, Orientation>& type_orientations,
    const std::unordered_map<std::string, Aggregation>& type_aggregations,
    const std::unordered_map<std::string, std::string>& type_agg_properties,
    const std::unordered_map<std::string, PropertyConfig>& node_property_configs,
    const std::unordered_map<std::string, PropertyConfig>& edge_property_configs,
    const std::string& include_features,
    const std::string& label_property,
    const std::string& split_property,
    bool include_label_indexes,
    IndexSet index_set,
    bool build_topology_snapshot,
    BPT::LeafFormat leaf_format,
    BPT::GraphStorage graph_storage
)
    : projection_name(projection_name_)
    , db_folder(db_folder_)
    , node_property_keys(node_properties)
    , edge_property_keys(edge_properties)
    , node_prop_configs(node_property_configs)
    , edge_prop_configs(edge_property_configs)
    , orientation(orientation_)
    , aggregation(aggregation_)
    , aggregation_property_key(aggregation_property)
    , include_features_(include_features)
    , label_property_(label_property)
    , split_property_(split_property)
    , include_label_indexes_(include_label_indexes)
    , index_set_(index_set)
    , build_topology_snapshot_(build_topology_snapshot)
    , leaf_format_(leaf_format)
    , graph_storage_(graph_storage)
    , per_type_orientations(type_orientations)
    , per_type_aggregations(type_aggregations)
    , per_type_agg_properties(type_agg_properties)
    , start_time(std::chrono::steady_clock::now())
{
    // Create projection directory
    auto& manager = ProjectionManager::get_instance();
    std::string proj_dir = manager.create_projection(projection_name);

    // Configure features based on property lists
    ProjectionStorage::Features features;
    features.include_node_properties = !node_property_keys.empty()
                                     || !node_prop_configs.empty();
    // Enable edge properties if explicitly requested OR if COUNT aggregation will create _count
    bool has_count_aggregation = (aggregation == Aggregation::COUNT);
    for (const auto& [type, agg] : type_aggregations) {
        if (agg == Aggregation::COUNT) {
            has_count_aggregation = true;
            break;
        }
    }
    features.include_edge_properties = !edge_property_keys.empty()
                                     || !edge_prop_configs.empty()
                                     || has_count_aggregation;
    // Default (Neo4j-GDS parity) is to always build node+edge label indexes.
    // The opt-out (`includeLabelIndexes: false` in graph_project config) skips
    // those 4 indexes when the workload has no `MATCH (n:Label)` queries
    // — e.g. pure GNN training on the projection. See analysis doc §3.A.
    features.include_node_labels = include_label_indexes_;
    features.include_edge_labels = include_label_indexes_;

    // Initialize projection storage with features
    storage = std::make_unique<ProjectionStorage>(proj_dir, db_folder, projection_name, features);
    storage->init();

    // Surface the user-requested property name lists so `mdb inspect-projection`
    // can report them. Build the union of the simple `relationshipProperties` /
    // `nodeProperties` list and any per-property config names (renames + defaults),
    // deduplicated, so renames like {score: {property: 'rating'}} still appear.
    {
        std::vector<std::string> req_nodes = node_property_keys;
        for (const auto& [name, _cfg] : node_property_configs) {
            if (std::find(req_nodes.begin(), req_nodes.end(), name) == req_nodes.end()) {
                req_nodes.push_back(name);
            }
        }
        std::vector<std::string> req_edges = edge_property_keys;
        for (const auto& [name, _cfg] : edge_property_configs) {
            if (std::find(req_edges.begin(), req_edges.end(), name) == req_edges.end()) {
                req_edges.push_back(name);
            }
        }
        storage->requested_node_properties = std::move(req_nodes);
        storage->requested_edge_properties = std::move(req_edges);
    }

    // Surface the IndexSet preset so ProjectionStorage can persist it in the
    // v1.4 catalog. This is the serialization-only half; the query layer
    // consumes the persisted value to raise descriptive errors when a query
    // needs an index that wasn't materialized (e.g., EDGE_LABEL under
    // GNN_MINIMAL).
    storage->requested_index_set = index_set_;

    // Push the leaf-format preset (BITSET or DELTA_VARINT) down so every
    // BPlusTree reader constructed by ProjectionStorage uses the matching
    // encoding, and save_catalog() can populate the v1.5 `leaf_formats`
    // byte array (one byte per materialized index). Default BITSET is
    // byte-identical to behavior for every caller that doesn't set the key.
    storage->requested_leaf_format = leaf_format_;

    // Push the graph-storage mode (BTREE or CSR_HYBRID) down so
    // save_catalog() persists the v1.6 graph_storage byte. The value is
    // plumbed here; the build pipeline consumes requested_graph_storage
    // from ProjectionStorage to dispatch the edge indexes through
    // BPTLeafCSRWriter under CSR_HYBRID (where the edge-index B+Tree
    // leaves ARE the CSR layout, providing O(1) neighbor access).
    storage->requested_graph_storage = graph_storage_;

    // Push the topology-snapshot opt-in down into storage so the two
    // edge-index builders (build_from_to_edge_index_ and
    // build_to_from_edge_index_) emit mmap-backed CSR sidecar files
    // (topology_fwd.csr / topology_rev.csr) inline during the B+Tree
    // build, instead of requiring a separate 3-pass-per-direction
    // post-hoc walker. The post-hoc walker is retained as a safety
    // fallback: build_topology_snapshots_() skips any direction whose
    // integrated emit already succeeded, and only runs the legacy
    // BPT-iterator path for the others (should not occur on a successful
    // build).
    storage->set_build_topology_snapshot(build_topology_snapshot_);

    // Initialize native scanner with main graph indexes
    // Include edge_from_to and edge_n1_n2 for O(log n) edge endpoint lookup
    scanner = std::make_unique<NativeScanner>(
        &gql_model.get_label_node(),
        &gql_model.get_label_edge(),
        &gql_model.get_from_to_edge(),    // Directed edges
        &gql_model.get_edge_from_to(),    // Directed edges (fast lookup)
        &gql_model.get_n1_n2_edge(),      // Undirected edges
        &gql_model.get_edge_n1_n2()       // Undirected edges (fast lookup)
    );

    // Reserve batch buffer capacity for efficiency
    node_batch.reserve(BATCH_SIZE);
    edge_batch.reserve(BATCH_SIZE);

    // Build inverse catalog maps once for O(1) property key name lookup.
    // Replaces per-property-per-node/edge O(K) linear scan through catalog maps.
    for (const auto& [name, id] : gql_model.catalog.node_keys2id) {
        node_key_id_to_name[id | ObjectId::MASK_NODE_KEY] = name;
    }
    for (const auto& [name, id] : gql_model.catalog.edge_keys2id) {
        edge_key_id_to_name[id | ObjectId::MASK_EDGE_KEY] = name;
    }

    // Enable benchmark timers if MDB_BENCHMARK env var is set
    benchmark_timers_.enabled = (std::getenv("MDB_BENCHMARK") != nullptr);

#ifdef ENABLE_GNN
    // Load RowMapping and FeatureMatrix metadata for GNN label/split extraction.
    // The RowMapping provides the ObjectId -> row_index mapping needed to index
    // labels and splits by the same row order as the feature matrix.
    if (!include_features_.empty()) {
        namespace fs = std::filesystem;
        auto rmap_path = fs::path(db_folder) / "gnn_features" / (include_features_ + ".rmap");
        auto fmat_path = fs::path(db_folder) / "gnn_features" / (include_features_ + ".fmat");

        if (fs::exists(rmap_path)) {
            gnn_row_mapping_ = std::make_unique<mdb::gnn::RowMapping>(
                mdb::gnn::RowMapping::open(rmap_path));
        }
        if (fs::exists(fmat_path)) {
            auto fm = mdb::gnn::FeatureMatrix::open(fmat_path);
            feature_dim_ = static_cast<uint32_t>(fm.num_cols());
        }

        if (gnn_row_mapping_) {
            if (!label_property_.empty()) {
                labels_buffer_.resize(gnn_row_mapping_->size(), -1);  // -1 = unlabeled
            }
            if (!split_property_.empty()) {
                splits_buffer_.resize(gnn_row_mapping_->size(), 255);  // 255 = UNLABELED
            }
        }
    }
#endif
}

IndexSet NativeProjectionBuilder::get_index_set() const noexcept {
    return index_set_;
}

// ============================================================================
// EdgeAggregator implementation
// ============================================================================

bool EdgeAggregator::process_edge(ObjectId edge_id, std::optional<double> property_value) {
    count_++;

    // First edge - always keep it (added to batch)
    if (count_ == 1) {
        first_edge_id_ = edge_id;
        representative_edge_id_ = edge_id;

        if (property_value.has_value()) {
            has_value_ = true;
            sum_value_ = property_value.value();
            min_value_ = property_value.value();
            max_value_ = property_value.value();
        }

        return true;  // Keep first edge
    }

    // Parallel edge detected - apply aggregation strategy
    switch (strategy_) {
        case Aggregation::SINGLE:
            // SINGLE is strict by design: multiple edges between the same
            // (from, to, type) triple are ambiguous. Instead of picking one
            // silently (which loses data), we fail and ask the caller how
            // they want to collapse parallels.
            //
            // Caveat: detection is windowed, not exact. The detector is
            // cleared every BATCH_SIZE edges to bound memory, so parallels
            // more than BATCH_SIZE apart in scan order escape detection and
            // are both stored (see the ParallelEdgeDetector class doc).
            //
            // Pick the strategy that matches your intent:
            //
            //   COUNT  — keep one edge with synthetic `_count` property
            //   SUM    — keep one edge with `aggregationProperty` summed
            //   MIN    — keep the edge with the smallest property value
            //   MAX    — keep the edge with the largest property value
            //
            // Common fixes:
            //   CALL graph_project('g', nodes, edges, {aggregation: 'COUNT'})
            //   CALL graph_project('g', nodes, edges,
            //       {aggregation: 'SUM', aggregationProperty: 'amount',
            //        relationshipProperties: ['amount']})
            //
            // For mixed types use the per-type MAP form:
            //   CALL graph_project('g', nodes,
            //       {TYPE_A: {aggregation: 'COUNT'},
            //        TYPE_B: {aggregation: 'SUM', aggregationProperty: 'x'}})
            throw QueryException(
                "Parallel edges detected but aggregation is SINGLE (the default).\n"
                "SINGLE refuses to drop data silently — choose how to collapse parallels:\n"
                "  aggregation: 'COUNT'  -> emit one edge with synthetic `_count`\n"
                "  aggregation: 'SUM'    -> sum `aggregationProperty` across parallels\n"
                "  aggregation: 'MIN'    -> keep edge with smallest property value\n"
                "  aggregation: 'MAX'    -> keep edge with largest property value\n"
                "\n"
                "Example:\n"
                "  CALL graph_project('g', nodes, edges, {aggregation: 'COUNT'})\n"
                "\n"
                "For wildcard ('*') you almost always want COUNT because the\n"
                "projection will touch every edge type, including ones with parallels.\n"
                "For per-type control use the MAP form on relationshipProjection."
            );

        case Aggregation::MIN:
            if (property_value.has_value()) {
                has_value_ = true;
                if (property_value.value() < min_value_) {
                    min_value_ = property_value.value();
                    representative_edge_id_ = edge_id;
                }
            }
            return false;

        case Aggregation::MAX:
            if (property_value.has_value()) {
                has_value_ = true;
                if (property_value.value() > max_value_) {
                    max_value_ = property_value.value();
                    representative_edge_id_ = edge_id;
                }
            }
            return false;

        case Aggregation::SUM:
            if (property_value.has_value()) {
                has_value_ = true;
                sum_value_ += property_value.value();
            }
            return false;

        case Aggregation::COUNT:
            // Just count, no property needed
            return false;  // Aggregate away

        default:
            throw std::runtime_error("Unknown aggregation strategy");
    }
}

double EdgeAggregator::get_aggregated_value() const {
    switch (strategy_) {
        case Aggregation::MIN:
            return min_value_;
        case Aggregation::MAX:
            return max_value_;
        case Aggregation::SUM:
            return sum_value_;
        case Aggregation::COUNT:
            return static_cast<double>(count_);
        case Aggregation::SINGLE:
            return 0.0;  // Not used for SINGLE
        default:
            return 0.0;
    }
}

// ============================================================================
// ParallelEdgeDetector implementation
// ============================================================================

bool ParallelEdgeDetector::process_edge(
    uint64_t from_node,
    uint64_t to_node,
    uint64_t type_id,
    ObjectId edge_id,
    std::optional<double> property_value
) {
    // Create hash key
    ParallelEdgeKey key{from_node, to_node, type_id};

    // Check if edge already exists in map
    auto it = edge_map_.find(key);

    if (it == edge_map_.end()) {
        // First occurrence - create new aggregator
        EdgeAggregator aggregator(strategy_);
        aggregator.process_edge(edge_id, property_value);
        edge_map_.emplace(key, std::move(aggregator));
        return true;  // Add to batch (first occurrence)
    } else {
        // Parallel edge - aggregate
        return it->second.process_edge(edge_id, property_value);
    }
}

// ============================================================================
// NativeProjectionBuilder implementation
// ============================================================================

NativeProjectionBuilder::~NativeProjectionBuilder() {
    // Clean up orphaned projection if finalize() was never called (e.g., exception)
    if (!finalized_) {
        try {
            auto& manager = ProjectionManager::get_instance();
            if (manager.projection_exists(projection_name)) {
                manager.drop_projection(projection_name);
            }
        } catch (...) {
            // Destructor must not throw — silently ignore cleanup failures
        }
    }
}

// ============================================================================
// Per-type configuration lookup methods
// ============================================================================

Orientation NativeProjectionBuilder::get_orientation_for_type(const std::string& type_name) const {
    auto it = per_type_orientations.find(type_name);
    if (it != per_type_orientations.end()) {
        return it->second;
    }
    return orientation;  // Fall back to global default
}

Aggregation NativeProjectionBuilder::get_aggregation_for_type(const std::string& type_name) const {
    auto it = per_type_aggregations.find(type_name);
    if (it != per_type_aggregations.end()) {
        return it->second;
    }
    return aggregation;  // Fall back to global default
}

std::string NativeProjectionBuilder::get_aggregation_property_for_type(const std::string& type_name) const {
    auto it = per_type_agg_properties.find(type_name);
    if (it != per_type_agg_properties.end()) {
        return it->second;
    }
    return aggregation_property_key;  // Fall back to global default
}

void NativeProjectionBuilder::scan_nodes_by_labels(const std::vector<std::string>& labels) {
    if (get_scan_mode() == ScanMode::SERIALIZED) {
        // Defer actual scan: store inputs for finalize_serialized_ to replay
        // via the 14-pass (one-index-at-a-time) serialized pipeline.
        stored_labels_ = labels;
        scan_inputs_captured_ = true;
        return;
    }
    scan_nodes_impl_classic_(labels);
}

void NativeProjectionBuilder::scan_edges_by_types(const std::vector<std::string>& types) {
    if (get_scan_mode() == ScanMode::SERIALIZED) {
        // Defer actual scan: store inputs for finalize_serialized_ to replay
        // via the 14-pass (one-index-at-a-time) serialized pipeline.
        stored_types_ = types;
        scan_inputs_captured_ = true;
        return;
    }
    if (get_scan_mode() == ScanMode::PARALLEL && all_single_no_aggprop_(types)) {
        // Builder-level parallel edge scan: has_node + orientation in TBB
        // workers, order-sensitive tail single-threaded in ascending merge.
        // Only safe for SINGLE-no-aggprop, no-edge-property projections;
        // anything else falls through to the byte-identical classic path.
        scan_edges_impl_parallel_(types);
        return;
    }
    scan_edges_impl_classic_(types);
}

bool NativeProjectionBuilder::all_single_no_aggprop_(
    const std::vector<std::string>& types) const
{
    // The parallel path replays only the windowed SINGLE detector + edge_batch
    // + add_edge_label in its single-threaded merge. Any edge-property config
    // would pull in the non-thread-safe property-write tail
    // (extract_edge_properties / register_edge_key), so require it empty.
    if (!edge_property_keys.empty() || !edge_prop_configs.empty()) {
        return false;
    }
    for (const auto& type : types) {
        if (get_aggregation_for_type(type) != Aggregation::SINGLE) {
            return false;
        }
        if (!get_aggregation_property_for_type(type).empty()) {
            return false;
        }
    }
    return true;
}

void NativeProjectionBuilder::scan_nodes_impl_classic_(const std::vector<std::string>& labels) {
    auto bench_t0 = benchmark_timers_.enabled ? ProjectionTimers::Clock::now()
                                               : ProjectionTimers::Clock::time_point{};

    for (const auto& label : labels) {
        validate_label_exists(label);

        // Convert label string to ObjectId via catalog lookup
        auto it = gql_model.catalog.node_labels2id.find(label);
        if (it == gql_model.catalog.node_labels2id.end()) {
            throw std::runtime_error(
                "Label '" + label + "' not found in catalog"
            );
        }
        ObjectId label_id(it->second | ObjectId::MASK_NODE_LABEL);

        // Scan all nodes with this label
        scanner->scan_label_node(label_id, [this, label_id](ObjectId node_id) {
            // Add to batch
            ProjectedNode node;
            node.node_id = node_id;
            node_batch.push_back(node);

            // Extract properties if configured, or if GNN needs label/split data
            bool need_properties = !node_property_keys.empty() || !node_prop_configs.empty();
#ifdef ENABLE_GNN
            need_properties = need_properties || (gnn_row_mapping_ != nullptr);
#endif
            if (need_properties) {
                extract_node_properties(node_id);
            }

            // Store node label (automatic - always included like Neo4j GDS)
            storage->add_node_label(node_id, label_id);

            // Auto-flush when batch is full
            if (node_batch.size() >= BATCH_SIZE) {
                flush_nodes();
            }
        });
    }

    // Flush any remaining nodes to storage's collection vectors
    if (!node_batch.empty()) {
        flush_nodes();
    }

    // Finalise the scan-phase node tracker: sorts `collected_nodes_` and
    // collapses duplicates (see ProjectionStorage::finalize_node_scan). This
    // transitions has_node() from a linear-scan fallback to O(log N) binary
    // search, which is the contract scan_edges_by_types relies on for
    // filtering edge endpoints. Called here (rather than at the caller) so
    // every invocation path — procedure, tests, embedding writer — gets the
    // invariant for free.
    storage->finalize_node_scan();

    if (benchmark_timers_.enabled) {
        benchmark_timers_.node_scan_ms += std::chrono::duration<double, std::milli>(
            ProjectionTimers::Clock::now() - bench_t0).count();
    }
}

void NativeProjectionBuilder::scan_edges_impl_classic_(const std::vector<std::string>& types) {
    auto bench_t0 = benchmark_timers_.enabled ? ProjectionTimers::Clock::now()
                                               : ProjectionTimers::Clock::time_point{};

    // STREAMING AGGREGATION DECISION:
    // For COUNT/SUM aggregation on large graphs, use external sort-aggregate
    // to achieve O(B) memory instead of O(N) for the hash-based approach.

    // Step 1: Build type_id_map and check if streaming aggregation is needed
    std::unordered_map<std::string, ObjectId> type_id_map;
    bool all_types_count = true;  // Only use streaming if ALL types have COUNT
    uint64_t estimated_edge_count = 0;

    for (const auto& type : types) {
        validate_type_exists(type);

        auto it = gql_model.catalog.edge_labels2id.find(type);
        if (it == gql_model.catalog.edge_labels2id.end()) {
            throw std::runtime_error("Type '" + type + "' not found in catalog");
        }
        ObjectId type_id(it->second | ObjectId::MASK_EDGE_LABEL);
        type_id_map[type] = type_id;

        // Check aggregation mode - streaming only supports COUNT correctly.
        // SUM/MIN/MAX use the hash-based path which handles property exclusion
        // and value storage properly for all modes.
        Aggregation type_aggregation = get_aggregation_for_type(type);
        if (type_aggregation != Aggregation::COUNT) {
            all_types_count = false;
        }

        // Estimate edge count for this type (quick scan of label_edge index)
        estimated_edge_count += scanner->count_edges_by_type(type_id);
    }

    // Resize Bloom filter based on estimated edge count (prevents oversaturation)
    // Without this, the default 10M edge filter becomes saturated for large graphs,
    // causing up to 92% false positive rate and rejecting 44% of legitimate edges.
    if (estimated_edge_count > 0) {
        storage->resize_bloom_filter(estimated_edge_count);
    }

    // Step 2: Decide whether to use streaming aggregation
    // Only use streaming if ALL types have COUNT/SUM aggregation.
    // If any type has SINGLE/MIN/MAX, we must use hash-based approach to process those correctly.
    // (Previously, mixed aggregation types would silently ignore SINGLE/MIN/MAX types)
    if (all_types_count && estimated_edge_count > STREAMING_AGGREGATION_THRESHOLD) {
        std::cout << "[Builder] Using streaming aggregation for " << estimated_edge_count
                  << " edges (threshold: " << STREAMING_AGGREGATION_THRESHOLD << ")" << std::endl;
        scan_edges_with_streaming_aggregation(types, type_id_map);
        return;
    }

    // Step 3: Use original hash-based approach for smaller datasets
    // STREAMING AGGREGATION: Process each type's aggregates immediately after scanning.
    // This reduces peak memory from O(total_edges) to O(edges_per_type).
    // Memory savings: ~4 GB for large graphs (vs holding all detectors until end).

    // ---- Edge-scan profiling (env MDB_PROJECTION_EDGE_SCAN_PROFILE=1).
    //
    // Isolates the SERIAL consumer cost from the PARALLEL B+Tree walk. The
    // scanner (scan_label_edge_with_endpoints) walks + resolves endpoints in
    // TBB workers, but invokes THIS callback single-threaded on the consumer
    // thread, so consumer_ns_acc measures the funnel wall-time and
    // detector_ns_acc the windowed-dedup share of it. If consumer_ns_acc
    // approaches the edge-scan wall, the scan is consumer-bound (a serial-tail
    // lever pays off); if it is a small fraction, the parallel walk dominates
    // and a serial-tail lever cannot help. Zero overhead when the flag is off
    // (a single predictable branch per edge).
    const bool edge_scan_profile =
        truthy_env_(std::getenv("MDB_PROJECTION_EDGE_SCAN_PROFILE"));
    // ---- Serial-tail lever (env MDB_PROJECTION_SKIP_WINDOWED_DEDUP=1).
    //
    // For SINGLE aggregation with no aggregation property, the windowed
    // ParallelEdgeDetector is a REDUNDANT early-drop: the authoritative dedup
    // is the std::unique() during bulk index build (projection_storage.cc:981),
    // a pure function of the edge multiset. Skipping process_edge here removes
    // the largest serial-consumer cost while keeping the final sorted leaves
    // identical (std::unique collapses any true duplicate the window would have
    // dropped). The only semantic change is that a true duplicate no longer
    // raises the SINGLE QueryException — it is silently deduped like COUNT-of-1,
    // which is acceptable for GNN-equivalence projections (papers100M/cora_gnn
    // have no duplicate (from,to,type) edges, so behavior is unchanged and the
    // cora bit-identical gate holds). Default OFF; only engaged for the A/B.
    const bool edge_scan_skip_dedup =
        truthy_env_(std::getenv("MDB_PROJECTION_SKIP_WINDOWED_DEDUP"));
    uint64_t consumer_ns_acc = 0;
    uint64_t detector_ns_acc = 0;
    uint64_t has_node_ns_acc = 0;
    uint64_t consumer_calls = 0;
    struct EdgeScanProfileGuard {
        uint64_t& acc;
        bool on;
        std::chrono::steady_clock::time_point t;
        EdgeScanProfileGuard(uint64_t& a, bool o) : acc(a), on(o) {
            if (on) t = std::chrono::steady_clock::now();
        }
        ~EdgeScanProfileGuard() {
            if (on) {
                acc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - t)
                           .count();
            }
        }
    };
    const auto edge_scan_wall_t0 = std::chrono::steady_clock::now();

    for (const auto& type : types) {
        // Get per-type configuration (falls back to global defaults)
        Orientation type_orientation = get_orientation_for_type(type);
        Aggregation type_aggregation = get_aggregation_for_type(type);
        std::string type_agg_property = get_aggregation_property_for_type(type);

        // Serial-tail lever: only SINGLE-no-aggprop can rely on sort-time
        // std::unique for dedup; COUNT/SUM/MIN/MAX need the detector's
        // accumulation, so never skip for those.
        const bool skip_windowed_dedup = edge_scan_skip_dedup &&
            (type_aggregation == Aggregation::SINGLE) && type_agg_property.empty();
        if (skip_windowed_dedup) {
            std::cout << "[Builder] SINGLE-no-aggprop '" << type
                      << "': skipping windowed detector "
                         "(authoritative dedup at sort-time std::unique)" << std::endl;
        }

        // Get type_id from pre-built map
        ObjectId type_id = type_id_map[type];

        // Create detector for this type with type-specific aggregation
        auto detector = std::make_unique<ParallelEdgeDetector>(type_aggregation);

        // Scan all edges with this type (OPTIMIZED: get endpoints in single pass)
        scanner->scan_label_edge_with_endpoints(
            type_id,
            [this, &detector, type_id, type_orientation,
             type_aggregation, &type_agg_property, skip_windowed_dedup,
             edge_scan_profile, &consumer_ns_acc, &detector_ns_acc,
             &has_node_ns_acc, &consumer_calls]
            (ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
            EdgeScanProfileGuard _consumer_guard(consumer_ns_acc, edge_scan_profile);
            if (edge_scan_profile) ++consumer_calls;
            // Filter: only include if both endpoints are in projection
            std::chrono::steady_clock::time_point _hn_t0;
            if (edge_scan_profile) _hn_t0 = std::chrono::steady_clock::now();
            bool has_from = storage->has_node(from_node);
            bool has_to = storage->has_node(to_node);
            if (edge_scan_profile) {
                has_node_ns_acc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - _hn_t0)
                                       .count();
            }

            if (!has_from || !has_to) {
                return; // Skip edge - endpoints not in projection
            }

            // Extract property value for aggregation (if needed) - use type-specific property
            std::optional<double> property_value = std::nullopt;
            if (!type_agg_property.empty()) {
                property_value = get_edge_property_value_for_aggregation(edge_id, type_agg_property);
            }

            // Parallel edge detection: Check if this edge is a duplicate
            // detector.process_edge() returns true for first occurrence, false for duplicates
            // For SINGLE mode: throws exception on duplicate
            // For MIN/MAX/SUM: aggregates based on property_value
            // For COUNT: just counts (property_value ignored)
            std::chrono::steady_clock::time_point _det_t0;
            if (edge_scan_profile) _det_t0 = std::chrono::steady_clock::now();
            bool is_first_occurrence;
            if (skip_windowed_dedup) {
                // Lever ON: rely on the bulk-build std::unique() for dedup.
                is_first_occurrence = true;
            } else {
                is_first_occurrence = detector->process_edge(
                    from_node.id,
                    to_node.id,
                    type_id.id,
                    edge_id,
                    property_value
                );
            }
            if (edge_scan_profile) {
                detector_ns_acc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - _det_t0)
                                       .count();
            }

            if (!is_first_occurrence) {
                return;  // Skip duplicate edge (aggregated)
            }

            // Add to batch based on TYPE-SPECIFIC orientation
            switch (type_orientation) {
                case Orientation::NATURAL: {
                    // Single edge: from → to (as specified)
                    ProjectedEdge edge;
                    edge.from_node = from_node;
                    edge.to_node = to_node;
                    edge.edge_id = edge_id;
                    uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
                    edge.is_directed = (edge_type != ObjectId::MASK_UNDIRECTED_EDGE);
                    edge_batch.push_back(edge);
                    break;
                }
                case Orientation::REVERSE: {
                    // Single edge: to → from (reversed)
                    ProjectedEdge edge;
                    edge.from_node = to_node;  // Swap endpoints
                    edge.to_node = from_node;  // Swap endpoints
                    edge.edge_id = edge_id;
                    uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
                    edge.is_directed = (edge_type != ObjectId::MASK_UNDIRECTED_EDGE);
                    edge_batch.push_back(edge);
                    break;
                }
                case Orientation::UNDIRECTED: {
                    // MEMORY OPTIMIZATION: Canonical storage for undirected edges
                    // Store each edge ONCE with canonical ordering (smaller node ID first).
                    // Bidirectional traversal still works via from_to_edge + to_from_edge indexes.
                    //
                    // ISO 39075 §3.4.13: "An undirected edge expresses a relationship
                    // that is necessarily symmetric" - canonical storage preserves this.
                    ProjectedEdge edge;
                    if (from_node.id <= to_node.id) {
                        edge.from_node = from_node;
                        edge.to_node = to_node;
                    } else {
                        edge.from_node = to_node;
                        edge.to_node = from_node;
                    }
                    edge.edge_id = edge_id;
                    edge.is_directed = false;
                    edge_batch.push_back(edge);
                    break;
                }
            }

            // Extract properties ONCE per logical edge (not duplicated for UNDIRECTED)
            // For aggregation modes: exclude the aggregation property — it will be stored
            // later with the aggregated value (SUM/MIN/MAX) or as _count (COUNT)
            if (!edge_property_keys.empty()) {
                if (type_aggregation != Aggregation::SINGLE && !type_agg_property.empty()) {
                    extract_edge_properties_excluding(edge_id, type_agg_property);
                } else {
                    extract_edge_properties(edge_id);
                }
            }

            // Store edge label (automatic - always included like Neo4j GDS)
            storage->add_edge_label(edge_id, type_id);

            // Auto-flush when batch is full (SINGLE only).
            // MIN/MAX/SUM/COUNT must NOT flush+clear here because their aggregated
            // values are read after the full type scan at get_aggregated_property_values().
            // Clearing the detector mid-scan would lose accumulated min/max/sum/count state.
            //
            // KNOWN LIMITATION: the clear() also drops SINGLE's seen-edge set, so
            // duplicate detection is best-effort within a BATCH_SIZE-edge window of
            // the scan order — two parallel edges more than BATCH_SIZE apart are
            // BOTH stored without the QueryException. Deliberate memory trade-off
            // (an uncleared detector holds every kept edge; 25 GB RSS on
            // papers100M). See the ParallelEdgeDetector class doc.
            if (type_aggregation == Aggregation::SINGLE && edge_batch.size() >= BATCH_SIZE) {
                flush_edges();
                detector->clear();
            }
        });

        // Store aggregated property values for all non-SINGLE modes after type scan.
        // This releases memory right away instead of holding all detectors until the end.
        if (type_aggregation != Aggregation::SINGLE) {
            // Get map of edge_id -> aggregated_value
            auto aggregated_values = detector->get_aggregated_property_values();

            // Convert aggregation property key to ObjectId for storage
            ObjectId property_key_id(0);
            std::string property_key_name = type_agg_property;

            // For COUNT mode, use synthetic "_count" property (same as streaming path)
            if (type_aggregation == Aggregation::COUNT) {
                property_key_name = "_count";
                property_key_id = ObjectId(COUNT_KEY_SYNTHETIC_ID | ObjectId::MASK_EDGE_KEY);
                storage->register_edge_key("_count", COUNT_KEY_SYNTHETIC_ID);
            } else {
                // Look up property key in catalog (SUM/MIN/MAX use existing properties)
                auto key_it = gql_model.catalog.edge_keys2id.find(property_key_name);
                if (key_it != gql_model.catalog.edge_keys2id.end()) {
                    property_key_id = ObjectId(key_it->second | ObjectId::MASK_EDGE_KEY);
                } else {
                    std::cerr << "[Builder] Warning: Property key '"
                              << property_key_name
                              << "' not found in catalog. Aggregated values "
                                 "may not be stored correctly." << std::endl;
                }
            }

            // Store aggregated value as property on each representative edge.
            // COUNT packs as int; SUM/MIN/MAX pack as persistent doubles so
            // fractional aggregates survive (see pack_aggregated_property_value).
            for (const auto& [edge_id_raw, agg_value] : aggregated_values) {
                ObjectId edge_id(edge_id_raw);
                ObjectId value_oid = pack_aggregated_property_value(type_aggregation, agg_value);

                if (property_key_id.id != 0) {
                    storage->add_edge_property(edge_id, property_key_id, value_oid);
                }
            }
        }

        // Clear detector immediately after processing this type (release memory)
        detector->clear();
    }

    // Flush any remaining edges
    if (!edge_batch.empty()) {
        flush_edges();
    }

    if (edge_scan_profile) {
        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - edge_scan_wall_t0).count();
        const double consumer_ms = static_cast<double>(consumer_ns_acc) / 1e6;
        const double detector_ms = static_cast<double>(detector_ns_acc) / 1e6;
        const double has_node_ms = static_cast<double>(has_node_ns_acc) / 1e6;
        // emit = the order-independent record-emission tail (orientation +
        // add_edge_label + flush_edges -> storage->add_edge to the streaming
        // buffers). Derived: consumer minus the two measured regions. Includes
        // a little per-edge timer overhead, so it is a slight over-estimate.
        const double emit_ms = consumer_ms - detector_ms - has_node_ms;
        const double consumer_pct = wall_ms > 0 ? 100.0 * consumer_ms / wall_ms : 0.0;
        const double detector_pct =
            consumer_ms > 0 ? 100.0 * detector_ms / consumer_ms : 0.0;
        const double has_node_pct =
            consumer_ms > 0 ? 100.0 * has_node_ms / consumer_ms : 0.0;
        const double emit_pct = consumer_ms > 0 ? 100.0 * emit_ms / consumer_ms : 0.0;
        std::cout << "[EDGE_SCAN_PROFILE]"
                  << " edges_to_consumer=" << consumer_calls
                  << " edge_scan_wall_ms=" << static_cast<uint64_t>(wall_ms)
                  << " serial_consumer_ms=" << static_cast<uint64_t>(consumer_ms)
                  << " (" << consumer_pct << "% of wall)"
                  << " has_node_ms=" << static_cast<uint64_t>(has_node_ms)
                  << " (" << has_node_pct << "% of consumer)"
                  << " windowed_detector_ms=" << static_cast<uint64_t>(detector_ms)
                  << " (" << detector_pct << "% of consumer)"
                  << " emit_ms=" << static_cast<uint64_t>(emit_ms)
                  << " (" << emit_pct << "% of consumer)"
                  // Which structure actually served this run. Without it, a
                  // budget written against max_id instead of (max-min) would
                  // reject on every real graph and be indistinguishable from
                  // success: correct in everything, worth nothing.
                  << " has_node_mode="
                  << (storage->node_bitmap_bytes() > 0 ? "bitmap" : "binary")
                  << " has_node_bitmap_bytes=" << storage->node_bitmap_bytes()
                  << std::endl;
        std::cout << "[EDGE_SCAN_PROFILE] interpretation:"
                  << " consumer>=~80% of wall => consumer-bound (serial-tail lever helps);"
                  << " consumer<<wall => parallel-walk-bound (serial-tail lever cannot help)"
                  << std::endl;
    }

    if (benchmark_timers_.enabled) {
        benchmark_timers_.edge_scan_ms += std::chrono::duration<double, std::milli>(
            ProjectionTimers::Clock::now() - bench_t0).count();
    }
}

void NativeProjectionBuilder::scan_edges_impl_parallel_(const std::vector<std::string>& types) {
    auto bench_t0 = benchmark_timers_.enabled ? ProjectionTimers::Clock::now()
                                               : ProjectionTimers::Clock::time_point{};

    std::cout << "[Builder] Using PARALLEL edge scan (has_node + orientation "
                 "in TBB workers; mutex-guarded parallel emit, detector dropped)" << std::endl;

    // Prelude mirrors scan_edges_impl_classic_'s type validation + Bloom
    // filter sizing. (No streaming-aggregation branch: PARALLEL is gated on
    // all-SINGLE via all_single_no_aggprop_, so the COUNT streaming path is
    // never reachable here.)
    std::unordered_map<std::string, ObjectId> type_id_map;
    uint64_t estimated_edge_count = 0;
    for (const auto& type : types) {
        validate_type_exists(type);
        auto it = gql_model.catalog.edge_labels2id.find(type);
        if (it == gql_model.catalog.edge_labels2id.end()) {
            throw std::runtime_error("Type '" + type + "' not found in catalog");
        }
        ObjectId type_id(it->second | ObjectId::MASK_EDGE_LABEL);
        type_id_map[type] = type_id;
        estimated_edge_count += scanner->count_edges_by_type(type_id);
    }
    if (estimated_edge_count > 0) {
        storage->resize_bloom_filter(estimated_edge_count);
    }

    // ---- Parallel emit (bounded-memory, OOM-free). FACTUAL basis: the
    // papers100M edge-scan profile (2026-06-16, MDB_PROJECTION_EDGE_SCAN_PROFILE)
    // shows the serial consumer is 98% of the edge-scan wall, and within it
    // has_node is ~72%, record-emission ~22%, the windowed detector ~6%. So the
    // win is to run has_node + orientation IN the TBB workers (parallel; the
    // has_node read is a const lock-free binary search over the finalized
    // collected_nodes_) and serialize only the emit under one mutex.
    //
    // The windowed ParallelEdgeDetector is DROPPED here: this path is gated on
    // all_single_no_aggprop_(), and for SINGLE-no-aggprop the authoritative dedup
    // is the sort-time std::unique() in build_one_index — a pure function of the
    // edge MULTISET. So emit order is irrelevant (workers race for the mutex in
    // nondeterministic order, but the final sorted+deduped leaves are identical
    // to the classic path), and NO ordered merge is needed — hence NO
    // per-partition buffering, hence the 64 GB-on-papers100M OOM is gone (memory
    // is bounded by edge_batch / BATCH_SIZE). The only behavior dropped vs the
    // classic detector is the SINGLE duplicate-edge QueryException, which never
    // fires on the no-duplicate GNN graphs this path targets (papers100M/cora).
    // cora bit-identical (0.8574939) gates this. Opt-in via
    // MDB_PROJECTION_PARALLEL_SCAN (ScanMode::PARALLEL); the default classic path
    // is unchanged.
    const bool edge_scan_profile_par =
        truthy_env_(std::getenv("MDB_PROJECTION_EDGE_SCAN_PROFILE"));
    const auto par_wall_t0 = std::chrono::steady_clock::now();
    uint64_t emitted_count = 0;
    std::mutex emit_mutex;

    for (const auto& type : types) {
        Orientation type_orientation = get_orientation_for_type(type);
        ObjectId type_id = type_id_map[type];

        scanner->scan_label_edge_endpoints_partitioned(
            type_id,
            /*num_partitions=*/0,  // auto-resolve via env knob; clamped <= 64
            [&](std::size_t /*part_idx*/, ObjectId edge_id,
                ObjectId from_node, ObjectId to_node) {
                // has_node + orientation run IN the worker (parallel, lock-free).
                if (!storage->has_node(from_node) || !storage->has_node(to_node)) {
                    return;  // endpoint(s) not in projection — drop
                }
                ProjectedEdge edge;
                switch (type_orientation) {
                    case Orientation::NATURAL: {
                        edge.from_node = from_node;
                        edge.to_node = to_node;
                        edge.edge_id = edge_id;
                        uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
                        edge.is_directed = (edge_type != ObjectId::MASK_UNDIRECTED_EDGE);
                        break;
                    }
                    case Orientation::REVERSE: {
                        edge.from_node = to_node;   // Swap endpoints
                        edge.to_node = from_node;   // Swap endpoints
                        edge.edge_id = edge_id;
                        uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
                        edge.is_directed = (edge_type != ObjectId::MASK_UNDIRECTED_EDGE);
                        break;
                    }
                    case Orientation::UNDIRECTED: {
                        if (from_node.id <= to_node.id) {
                            edge.from_node = from_node;
                            edge.to_node = to_node;
                        } else {
                            edge.from_node = to_node;
                            edge.to_node = from_node;
                        }
                        edge.edge_id = edge_id;
                        edge.is_directed = false;
                        break;
                    }
                }
                // Emit critical section: ALL shared-state mutation (edge_batch +
                // storage streaming buffers via flush_edges/add_edge_label) is
                // serialized here. has_node + orientation above ran in parallel.
                std::lock_guard<std::mutex> lk(emit_mutex);
                ++emitted_count;
                edge_batch.push_back(edge);
                storage->add_edge_label(edge.edge_id, type_id);
                if (edge_batch.size() >= BATCH_SIZE) {
                    flush_edges();
                }
            });
    }

    if (!edge_batch.empty()) {
        flush_edges();
    }

    if (edge_scan_profile_par) {
        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - par_wall_t0).count();
        std::cout << "[EDGE_SCAN_PROFILE] edges_to_consumer=" << emitted_count
                  << " edge_scan_wall_ms=" << static_cast<uint64_t>(wall_ms)
                  << " (PARALLEL mutex-emit: has_node+orient in TBB workers, emit serialized)"
                  << std::endl;
    }

    if (benchmark_timers_.enabled) {
        benchmark_timers_.edge_scan_ms += std::chrono::duration<double, std::milli>(
            ProjectionTimers::Clock::now() - bench_t0).count();
    }
}

NativeProjectionBuilder::Statistics NativeProjectionBuilder::finalize() {
    // total_ms is measured from `start_time`, set in the constructor's init
    // list, NOT from here. Anchoring it here excluded both the node scan and
    // the edge scan, which run before finalize() is ever called, so every phase
    // was divided by a total that did not contain it and the percentages summed
    // to more than 100. Using `start_time` also makes total_ms agree with
    // Statistics::duration_ms, which is measured over the same window.

    finalized_ = true;

    // Dispatch to the serialized orchestrator when MDB_PROJECTION_SERIAL_SCAN=1
    // and the public scan_*_by_* wrappers captured inputs
    // (scan_inputs_captured_ = true).
    //
    // finalize_serialized_() runs the full Phase A/B/C pipeline:
    //   Phase A — 5 node-index passes (scan + sort + build + reset each)
    //   Phase B — one edge-filter bitmap pass (no record emission)
    //   Phase C — 9 edge-index passes (scan + sort + build + reset each)
    //   Phase 4 — opens all BPlusTree readers via open_all_bplustree_readers_()
    //
    // After it returns the 14 .leaf/.dir index files exist on disk AND their
    // reader unique_ptrs are open inside ProjectionStorage.  The
    // storage->flush() call below will then find has_records == false (all
    // streaming buffers were drained by the per-pass build_one_index() calls)
    // and skip build_all_indexes_bulk() — preventing any double-sort.
    // save_catalog() inside flush() still runs, writing the catalog file.
    //
    // Under CLASSIC (default, SERIAL_SCAN unset or 0): scan_inputs_captured_
    // is false, so this block is a no-op.  The classic path continues below
    // unchanged: flush_nodes/flush_edges fill streaming buffers, then
    // storage->flush() → build_all_indexes_bulk() does sort+build+Phase 4.
    if (get_scan_mode() == ScanMode::SERIALIZED && scan_inputs_captured_) {
        auto bench_sort_start = benchmark_timers_.enabled
            ? ProjectionTimers::Clock::now()
            : ProjectionTimers::Clock::time_point{};
        // The serialized path runs the node and edge scans INSIDE
        // finalize_serialized_() — once per index pass, 5 for nodes and 9 for
        // edges — so their timers fire within the interval measured here. The
        // scan time has to be subtracted back out, otherwise the same
        // nanoseconds land in two buckets and every percentage is inflated.
        // This is what keeps the phase timers disjoint in all three scan modes,
        // which is what makes the residual meaningful.
        const double scan_ms_before =
            benchmark_timers_.node_scan_ms + benchmark_timers_.edge_scan_ms;
        finalize_serialized_();
        if (benchmark_timers_.enabled) {
            auto bench_sort_end = ProjectionTimers::Clock::now();
            const double wall_ms = std::chrono::duration<double, std::milli>(
                bench_sort_end - bench_sort_start).count();
            const double scan_ms_inside =
                (benchmark_timers_.node_scan_ms + benchmark_timers_.edge_scan_ms)
                - scan_ms_before;
            benchmark_timers_.sort_btree_ms += wall_ms - scan_ms_inside;
        }
    }

    // Final flush to ensure all data is written
    if (!node_batch.empty()) {
        flush_nodes();
    }
    if (!edge_batch.empty()) {
        flush_edges();
    }

    // Final flush to ProjectionStorage (commits all B+Tree writes, builds indexes).
    // Under SERIALIZED + captured, flush() is essentially just save_catalog()
    // (all streaming buffers were drained by finalize_serialized_()'s per-pass
    // build_one_index() calls).  Skip timer attribution in that case so the
    // save_catalog() wall time doesn't inflate sort_ms / btree_write_ms in the
    // benchmark CSVs — those buckets should reflect only sort +
    // index-build work, both of which are already captured by the dispatch block
    // above (finalize_serialized_() timing).
    const bool serialized_flush_is_noop =
        (get_scan_mode() == ScanMode::SERIALIZED && scan_inputs_captured_);
    if (serialized_flush_is_noop) {
        storage->flush();
    } else {
        auto bench_sort_start = benchmark_timers_.enabled
            ? ProjectionTimers::Clock::now()
            : ProjectionTimers::Clock::time_point{};
        storage->flush();
        if (benchmark_timers_.enabled) {
            auto bench_sort_end = ProjectionTimers::Clock::now();
            // One interval covers sort and index build together. It used to be
            // halved into two fields by a hardcoded 0.5, which reported a
            // constant as if it were a measurement; the two are reported
            // combined instead until something inside the sorter times them
            // apart.
            benchmark_timers_.sort_btree_ms += std::chrono::duration<double, std::milli>(
                bench_sort_end - bench_sort_start).count();
        }
    }

    // Calculate duration
    auto end_time = std::chrono::steady_clock::now();
    stats.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );

    // Get final statistics from storage
    stats.node_count = storage->get_node_count();
    stats.relationship_count = storage->get_edge_count();

#ifdef ENABLE_GNN
    // Write GNN metadata, labels, and splits to the projection directory.
    // These files are consumed by gnn_train and indexed by the same RowMapping
    // row order as the feature matrix.
    // Fail-loud contract guards: when a caller asks for GNN outputs
    // (features / labels / splits), refuse to silently produce a broken
    // projection. Each guard covers a misconfiguration that previously caused a
    // GNN sidecar (gnn_meta.bin / labels.bin / splits.bin) to be skipped without
    // any error, surfacing only much later as a confusing gnn_train failure.
    if (!include_features_.empty()) {
        namespace fs = std::filesystem;
        auto proj_dir = fs::path(db_folder) / "projections" / projection_name;

        // Guard 1 — features requested but the prerequisite row mapping is
        // absent. gnn_features/<feature>.rmap is created only by
        // `mdb import ... --with-tensors <features.npy>`, never by
        // graph_project; without it nothing GNN-related can be emitted.
        if (!gnn_row_mapping_) {
            throw std::runtime_error(
                "graph_project: includeFeatures='" + include_features_ +
                "' was requested but gnn_features/" + include_features_ +
                ".rmap does not exist. Node features must be imported first via "
                "`mdb import <data> <db> --with-tensors <features.npy>`; "
                "graph_project only reads that row mapping, it cannot create it.");
        }

        // Guard 2 — label property set but no node yielded an integer label
        // (property-name typo or non-integer values). Without this the labels
        // buffer stays all-unlabeled and gnn_train silently collapses to ~0 acc.
        if (!label_property_.empty() && unique_classes_.empty()) {
            throw std::runtime_error(
                "graph_project: labelProperty='" + label_property_ +
                "' was set but no node produced an integer label (every value "
                "was missing or non-integer). Check the property name and that "
                "labels are stored as integers.");
        }

        // Guard 3 — split property set but no node had a recognized split token
        // (expected train / val / valid / validation / test). Without this the
        // splits buffer is all-UNLABELED and usePredefinedSplits is unusable.
        if (!split_property_.empty()) {
            bool any_split = false;
            for (uint8_t s : splits_buffer_) {
                if (s != 255) { any_split = true; break; }
            }
            if (!any_split) {
                throw std::runtime_error(
                    "graph_project: splitProperty='" + split_property_ +
                    "' was set but no node had a recognized split value "
                    "(expected one of: train, val, valid, validation, test). "
                    "Check the property name and its values.");
            }
        }

        // Write gnn_meta.bin
        mdb::gnn::GnnMeta meta;
        meta.feature_name = include_features_;
        meta.feature_dim  = feature_dim_;
        meta.num_nodes    = gnn_row_mapping_->size();
        meta.num_classes  = unique_classes_.size();
        meta.has_labels   = !label_property_.empty();
        meta.has_splits   = !split_property_.empty();
        meta.write(proj_dir / "gnn_meta.bin");

        // Populate stats for YIELD
        stats.feature_dim = feature_dim_;
        stats.num_classes = unique_classes_.size();

        // Write labels.bin (format: magic(8) + version(4) + reserved(4) +
        //   num_nodes(8) + num_classes(8) + int64[N])
        if (!label_property_.empty() && !labels_buffer_.empty()) {
            auto labels_path = proj_dir / "labels.bin";
            std::ofstream lf(labels_path, std::ios::binary | std::ios::trunc);
            if (!lf.is_open()) {
                throw std::runtime_error("Cannot open " + labels_path.string() + " for writing");
            }
            const uint8_t label_magic[8] = {'G','N','N','L','\0','\0','\0','\0'};
            uint32_t label_version  = 1;
            uint32_t label_reserved = 0;
            uint64_t label_num_nodes   = labels_buffer_.size();
            uint64_t label_num_classes = unique_classes_.size();
            lf.write(reinterpret_cast<const char*>(label_magic), 8);
            lf.write(reinterpret_cast<const char*>(&label_version), 4);
            lf.write(reinterpret_cast<const char*>(&label_reserved), 4);
            lf.write(reinterpret_cast<const char*>(&label_num_nodes), 8);
            lf.write(reinterpret_cast<const char*>(&label_num_classes), 8);
            lf.write(reinterpret_cast<const char*>(labels_buffer_.data()),
                     static_cast<std::streamsize>(labels_buffer_.size() * sizeof(int64_t)));
            if (!lf) {
                throw std::runtime_error("I/O error writing " + labels_path.string());
            }
        }

        // Write splits.bin (format: magic(8) + version(4) + reserved(4) +
        //   num_nodes(8) + uint8[N])
        if (!split_property_.empty() && !splits_buffer_.empty()) {
            auto splits_path = proj_dir / "splits.bin";
            std::ofstream sf(splits_path, std::ios::binary | std::ios::trunc);
            if (!sf.is_open()) {
                throw std::runtime_error("Cannot open " + splits_path.string() + " for writing");
            }
            const uint8_t split_magic[8] = {'G','N','N','S','\0','\0','\0','\0'};
            uint32_t split_version  = 1;
            uint32_t split_reserved = 0;
            uint64_t split_num_nodes = splits_buffer_.size();
            sf.write(reinterpret_cast<const char*>(split_magic), 8);
            sf.write(reinterpret_cast<const char*>(&split_version), 4);
            sf.write(reinterpret_cast<const char*>(&split_reserved), 4);
            sf.write(reinterpret_cast<const char*>(&split_num_nodes), 8);
            sf.write(reinterpret_cast<const char*>(splits_buffer_.data()),
                     static_cast<std::streamsize>(splits_buffer_.size() * sizeof(uint8_t)));
            if (!sf) {
                throw std::runtime_error("I/O error writing " + splits_path.string());
            }
        }
    }
#endif

    // After the B+Tree .leaf / .dir files are fully written and fsync'd by
    // storage->flush(), emit the mmap-backed CSR topology sidecar files
    // (topology_fwd.csr / topology_rev.csr) when the opt-in flag is set.
    // build_topology_snapshots_() short-circuits when the flag is off and
    // never throws (failures are logged and swallowed to keep the projection
    // valid).
    build_topology_snapshots_();

    // Refresh projection cache so new projection is immediately visible
    auto bench_meta_start = benchmark_timers_.enabled ? ProjectionTimers::Clock::now()
                                                      : ProjectionTimers::Clock::time_point{};
    ProjectionManager::get_instance().scan_projections();
    if (benchmark_timers_.enabled) {
        benchmark_timers_.metadata_ms += std::chrono::duration<double, std::milli>(
            ProjectionTimers::Clock::now() - bench_meta_start).count();
    }

    if (benchmark_timers_.enabled) {
        benchmark_timers_.total_ms = std::chrono::duration<double, std::milli>(
            ProjectionTimers::Clock::now() - start_time).count();
        benchmark_timers_.print(projection_name, stats.relationship_count);
    }

    return stats;
}

void NativeProjectionBuilder::flush_nodes() {
    for (const auto& node : node_batch) {
        storage->add_node(node);
    }
    node_batch.clear();
}

void NativeProjectionBuilder::flush_edges() {
    // Edge deduplication is handled upstream by ParallelEdgeDetector::process_edge
    // (the windowed (from, to, type) pass in scan_edges_impl_classic_) and by the
    // exact record dedup at index-build time. The ProjectionStorage edge Bloom
    // filter keys on (from, to, EDGE_ID); because every projected edge carries a
    // distinct edge_id it can NEVER match a true duplicate, so its only effect is
    // the probabilistic false positive — which SILENTLY DROPS a legitimate unique
    // edge (and the build-time std::unique() cannot recover an edge that was never
    // inserted). At the configured ~1% terminal FPR the loss is invisible on small
    // graphs but compounds to ~0.17% (2.69M dropped edges, 78k fully-isolated
    // nodes) on papers100M-scale builds. Skip it, matching the streaming path.
    for (const auto& edge : edge_batch) {
        storage->add_edge(edge, /*skip_bloom_check=*/true);
    }
    edge_batch.clear();
}

void NativeProjectionBuilder::validate_label_exists(const std::string& label) {
    if (gql_model.catalog.node_labels2id.find(label) == gql_model.catalog.node_labels2id.end()) {
        throw QueryException(
            "Node label '" + label + "' does not exist in the database.\n"
            "Hint: Labels are case-sensitive."
        );
    }
}

void NativeProjectionBuilder::validate_type_exists(const std::string& type) {
    if (gql_model.catalog.edge_labels2id.find(type) == gql_model.catalog.edge_labels2id.end()) {
        throw QueryException(
            "Relationship type '" + type + "' does not exist in the database.\n"
            "Hint: Types are case-sensitive."
        );
    }
}

void NativeProjectionBuilder::try_extract_gnn_property(
    ObjectId node_id, const std::string& key_name, ObjectId value_id)
{
#ifdef ENABLE_GNN
    if (!gnn_row_mapping_) {
        return;
    }
    auto row_opt = gnn_row_mapping_->find(node_id);
    if (!row_opt.has_value()) {
        return;  // Node not in RowMapping (not part of the feature matrix)
    }
    uint64_t row = *row_opt;

    // Extract label value (integer property -> int64 label class)
    if (!label_property_.empty() && key_name == label_property_) {
        auto sub_type = GQL_OID::get_generic_sub_type(value_id);
        if (sub_type == GQL_OID::GenericSubType::INTEGER) {
            int64_t label_val = Common::Conversions::unpack_int(value_id);
            labels_buffer_[row] = label_val;
            unique_classes_.insert(label_val);
        }
    }

    // Extract split value (string property -> uint8 split enum)
    // Split encoding: 0=TRAIN, 1=VAL, 2=TEST, 255=UNLABELED
    if (!split_property_.empty() && key_name == split_property_) {
        auto generic_type = GQL_OID::get_generic_type(value_id);
        if (generic_type == GQL_OID::GenericType::STRING) {
            std::string split_str = Conversions::unpack_string(value_id);
            uint8_t split_val = 255;  // UNLABELED
            if (split_str == "train")           split_val = 0;
            else if (split_str == "val")        split_val = 1;
            else if (split_str == "valid")      split_val = 1;  // OGB convention
            else if (split_str == "validation") split_val = 1;
            else if (split_str == "test")       split_val = 2;
            splits_buffer_[row] = split_val;
        }
    }
#else
    (void)node_id;
    (void)key_name;
    (void)value_id;
#endif
}

void NativeProjectionBuilder::extract_node_properties(ObjectId node_id) {
    // Phase 3: Support property configurations with renaming and defaults

    // If we have property configs, use them for selective extraction with renaming/defaults
    if (!node_prop_configs.empty()) {
        bool interruption = false;
        auto& node_key_value = gql_model.get_node_key_value();

        // Build a set of extracted properties to track which defaults we need to apply
        std::unordered_set<std::string> found_properties;

        // Range scan: [node_id, 0, 0] to [node_id, MAX, MAX]
        auto it = node_key_value.get_range(
            &interruption,
            {node_id.id, 0, 0},
            {node_id.id, UINT64_MAX, UINT64_MAX}
        );

        auto record = it.next();
        while (record != nullptr) {
            ObjectId key_id((*record)[1]);
            ObjectId value_id((*record)[2]);

            // Look up the property name from inverse catalog map (O(1))
            std::string source_key_name;
            auto nk_it = node_key_id_to_name.find(key_id.id);
            if (nk_it != node_key_id_to_name.end()) {
                source_key_name = nk_it->second;
            }

            // GNN: extract label/split values indexed by RowMapping position
            try_extract_gnn_property(node_id, source_key_name, value_id);

            // Check if any property config uses this source property
            for (const auto& [projected_name, config] : node_prop_configs) {
                std::string config_source = config.source_property.empty()
                    ? projected_name : config.source_property;

                if (config_source == source_key_name) {
                    // Resolve the destination key. When renaming (source != projected)
                    // we need a key that represents the projected_name; allocate a
                    // synthetic id if the main catalog doesn't know it, so readers
                    // of n.<projected_name> find the value under the right key.
                    ObjectId projected_key_id = key_id;  // Default: reuse source key
                    if (!config.source_property.empty() &&
                        config.source_property != projected_name)
                    {
                        uint64_t new_key = ensure_projected_node_key(projected_name);
                        projected_key_id = ObjectId(new_key | ObjectId::MASK_NODE_KEY);
                    }

                    storage->add_node_property(node_id, projected_key_id, value_id);
                    found_properties.insert(projected_name);
                    break;
                }
            }

            record = it.next();
        }

        // Apply default values for missing properties
        for (const auto& [projected_name, config] : node_prop_configs) {
            if (found_properties.find(projected_name) == found_properties.end()
                && config.default_value.has_value()) {
                // Property absent on this node — materialize under the projected name.
                // ensure_projected_node_key guarantees the key exists in the projection,
                // allocating a synthetic id if needed. Previously the code silently
                // skipped the write when the name wasn't in the main catalog.
                uint64_t key_id_raw = ensure_projected_node_key(projected_name);
                ObjectId projected_key_id(key_id_raw | ObjectId::MASK_NODE_KEY);

                {
                    // Default materialization (always runs now that we always have a key).
                    // Use the persistent variant: pack_double falls back to tmp_manager
                    // whose ids don't survive across sessions — the projection is on
                    // disk, so the value must live in string_manager.
                    ObjectId default_value_id =
                        pack_double_persistent(config.default_value.value());
                    storage->add_node_property(node_id, projected_key_id, default_value_id);
                }
            }
        }
        return;
    }

    // Fallback: Original behavior - extract all properties or filtered list
    bool interruption = false;
    auto& node_key_value = gql_model.get_node_key_value();

    // Range scan: [node_id, 0, 0] to [node_id, MAX, MAX]
    auto it = node_key_value.get_range(
        &interruption,
        {node_id.id, 0, 0},
        {node_id.id, UINT64_MAX, UINT64_MAX}
    );

    auto record = it.next();
    while (record != nullptr) {
        ObjectId key_id((*record)[1]);
        ObjectId value_id((*record)[2]);

        // Resolve property name from inverse catalog map (O(1))
        std::string key_name;
        auto nk_it = node_key_id_to_name.find(key_id.id);
        if (nk_it != node_key_id_to_name.end()) {
            key_name = nk_it->second;
        }

        // GNN: extract label/split values indexed by RowMapping position
        try_extract_gnn_property(node_id, key_name, value_id);

        // If we have a filter list, check if this property is in it
        if (!node_property_keys.empty()) {
            if (std::find(node_property_keys.begin(), node_property_keys.end(),
              key_name) == node_property_keys.end()) {
                record = it.next();
                continue;  // Skip this property
            }
        }

        // Add property to projection storage
        storage->add_node_property(node_id, key_id, value_id);

        record = it.next();
    }
}

void NativeProjectionBuilder::extract_edge_properties(ObjectId edge_id) {
    // Phase 3: Support property configurations with renaming and defaults

    // If we have property configs, use them for selective extraction with renaming/defaults
    if (!edge_prop_configs.empty()) {
        bool interruption = false;
        auto& edge_key_value = gql_model.get_edge_key_value();

        // Build a set of extracted properties to track which defaults we need to apply
        std::unordered_set<std::string> found_properties;

        // Range scan: [edge_id, 0, 0] to [edge_id, MAX, MAX]
        auto it = edge_key_value.get_range(
            &interruption,
            {edge_id.id, 0, 0},
            {edge_id.id, UINT64_MAX, UINT64_MAX}
        );

        auto record = it.next();
        while (record != nullptr) {
            ObjectId key_id((*record)[1]);
            ObjectId value_id((*record)[2]);

            // Look up the property name from inverse catalog map (O(1))
            std::string source_key_name;
            auto ek_it = edge_key_id_to_name.find(key_id.id);
            if (ek_it != edge_key_id_to_name.end()) {
                source_key_name = ek_it->second;
            }

            // Check if any property config uses this source property
            for (const auto& [projected_name, config] : edge_prop_configs) {
                std::string config_source = config.source_property.empty()
                    ? projected_name : config.source_property;

                if (config_source == source_key_name) {
                    ObjectId projected_key_id = key_id;  // Default: reuse source key
                    if (!config.source_property.empty() &&
                        config.source_property != projected_name)
                    {
                        uint64_t new_key = ensure_projected_edge_key(projected_name);
                        projected_key_id = ObjectId(new_key | ObjectId::MASK_EDGE_KEY);
                    }

                    storage->add_edge_property(edge_id, projected_key_id, value_id);
                    found_properties.insert(projected_name);
                    break;
                }
            }

            record = it.next();
        }

        // Apply default values for missing properties
        for (const auto& [projected_name, config] : edge_prop_configs) {
            if (found_properties.find(projected_name) == found_properties.end()
                && config.default_value.has_value()) {
                uint64_t key_id_raw = ensure_projected_edge_key(projected_name);
                ObjectId projected_key_id(key_id_raw | ObjectId::MASK_EDGE_KEY);

                ObjectId default_value_id =
                    pack_double_persistent(config.default_value.value());
                storage->add_edge_property(edge_id, projected_key_id, default_value_id);
            }
        }
        return;
    }

    // Fallback: Original behavior - extract all properties or filtered list
    bool interruption = false;
    auto& edge_key_value = gql_model.get_edge_key_value();

    // Range scan: [edge_id, 0, 0] to [edge_id, MAX, MAX]
    auto it = edge_key_value.get_range(
        &interruption,
        {edge_id.id, 0, 0},
        {edge_id.id, UINT64_MAX, UINT64_MAX}
    );

    auto record = it.next();
    while (record != nullptr) {
        ObjectId key_id((*record)[1]);
        ObjectId value_id((*record)[2]);

        // If we have a filter list, check if this property is in it
        if (!edge_property_keys.empty()) {
            std::string key_name;
            auto ek_it = edge_key_id_to_name.find(key_id.id);
            if (ek_it != edge_key_id_to_name.end()) {
                key_name = ek_it->second;
            }

            if (std::find(edge_property_keys.begin(), edge_property_keys.end(),
              key_name) == edge_property_keys.end()) {
                record = it.next();
                continue;  // Skip this property
            }
        }

        // Add property to projection storage
        storage->add_edge_property(edge_id, key_id, value_id);

        record = it.next();
    }
}

void NativeProjectionBuilder::extract_edge_properties_excluding(
    ObjectId edge_id,
    const std::string& exclude_property
) {
    // This method is identical to extract_edge_properties() but skips the
    // exclude_property. Used for SUM/COUNT aggregation to prevent storing
    // the original value before the aggregated value is computed.

    // Phase 3: Support property configurations with renaming and defaults
    if (!edge_prop_configs.empty()) {
        bool interruption = false;
        auto& edge_key_value = gql_model.get_edge_key_value();

        std::unordered_set<std::string> found_properties;

        auto it = edge_key_value.get_range(
            &interruption,
            {edge_id.id, 0, 0},
            {edge_id.id, UINT64_MAX, UINT64_MAX}
        );

        auto record = it.next();
        while (record != nullptr) {
            ObjectId key_id((*record)[1]);
            ObjectId value_id((*record)[2]);

            // Look up the property name from inverse catalog map (O(1))
            std::string source_key_name;
            auto ek_it = edge_key_id_to_name.find(key_id.id);
            if (ek_it != edge_key_id_to_name.end()) {
                source_key_name = ek_it->second;
            }

            // EXCLUSION CHECK: Skip the aggregation property
            if (source_key_name == exclude_property) {
                record = it.next();
                continue;
            }

            // Check if any property config uses this source property
            for (const auto& [projected_name, config] : edge_prop_configs) {
                std::string config_source = config.source_property.empty()
                    ? projected_name : config.source_property;

                if (config_source == source_key_name) {
                    ObjectId projected_key_id = key_id;
                    if (!config.source_property.empty() &&
                        config.source_property != projected_name)
                    {
                        uint64_t new_key = ensure_projected_edge_key(projected_name);
                        projected_key_id = ObjectId(new_key | ObjectId::MASK_EDGE_KEY);
                    }

                    storage->add_edge_property(edge_id, projected_key_id, value_id);
                    found_properties.insert(projected_name);
                    break;
                }
            }

            record = it.next();
        }

        // Apply default values for missing properties (excluding the aggregation property)
        for (const auto& [projected_name, config] : edge_prop_configs) {
            // Skip the excluded property for defaults too
            std::string config_source = config.source_property.empty()
                    ? projected_name : config.source_property;
            if (config_source == exclude_property) {
                continue;
            }

            if (found_properties.find(projected_name) == found_properties.end()
                && config.default_value.has_value()) {
                uint64_t key_id_raw = ensure_projected_edge_key(projected_name);
                ObjectId projected_key_id(key_id_raw | ObjectId::MASK_EDGE_KEY);

                ObjectId default_value_id =
                    pack_double_persistent(config.default_value.value());
                storage->add_edge_property(edge_id, projected_key_id, default_value_id);
            }
        }
        return;
    }

    // Fallback: Original behavior - extract all properties or filtered list
    bool interruption = false;
    auto& edge_key_value = gql_model.get_edge_key_value();

    auto it = edge_key_value.get_range(
        &interruption,
        {edge_id.id, 0, 0},
        {edge_id.id, UINT64_MAX, UINT64_MAX}
    );

    auto record = it.next();
    while (record != nullptr) {
        ObjectId key_id((*record)[1]);
        ObjectId value_id((*record)[2]);

        // If we have a filter list, check if this property is in it
        if (!edge_property_keys.empty()) {
            std::string key_name;
            auto ek_it = edge_key_id_to_name.find(key_id.id);
            if (ek_it != edge_key_id_to_name.end()) {
                key_name = ek_it->second;
            }

            // EXCLUSION CHECK: Skip the aggregation property
            if (key_name == exclude_property) {
                record = it.next();
                continue;
            }

            if (std::find(edge_property_keys.begin(), edge_property_keys.end(),
              key_name) == edge_property_keys.end()) {
                record = it.next();
                continue;
            }
        } else {
            // No filter list - still need to check exclusion
            std::string key_name;
            auto ek_it = edge_key_id_to_name.find(key_id.id);
            if (ek_it != edge_key_id_to_name.end()) {
                key_name = ek_it->second;
            }

            // EXCLUSION CHECK: Skip the aggregation property
            if (key_name == exclude_property) {
                record = it.next();
                continue;
            }
        }

        // Add property to projection storage
        storage->add_edge_property(edge_id, key_id, value_id);

        record = it.next();
    }
}

std::optional<double> NativeProjectionBuilder::get_edge_property_value_for_aggregation(
    ObjectId edge_id,
    const std::string& property_key
) {
    // Convert property key string to ObjectId via catalog lookup
    auto it = gql_model.catalog.edge_keys2id.find(property_key);
    if (it == gql_model.catalog.edge_keys2id.end()) {
        // Property key doesn't exist in catalog
        return std::nullopt;
    }

    ObjectId key_id(it->second | ObjectId::MASK_EDGE_KEY);

    // Look up edge_key_value index for this edge and property key
    bool interruption = false;
    auto& edge_key_value = gql_model.get_edge_key_value();

    // Range scan: [edge_id, key_id, 0] to [edge_id, key_id, MAX]
    auto range_it = edge_key_value.get_range(
        &interruption,
        {edge_id.id, key_id.id, 0},
        {edge_id.id, key_id.id, UINT64_MAX}
    );

    auto record = range_it.next();
    if (record == nullptr) {
        // Edge doesn't have this property
        return std::nullopt;
    }

    // Extract value ObjectId
    ObjectId value_id((*record)[2]);

    // Convert to double using Common::Conversions
    try {
        // Try to convert to double (handles int, decimal, float, double)
        double double_val = Common::Conversions::to_double(value_id);
        return double_val;
    } catch (...) {
        // Property is not numeric - can't aggregate
        return std::nullopt;
    }
}

// ============================================================================
// Streaming Aggregation Implementation (Memory-Efficient COUNT/SUM)
// ============================================================================

void NativeProjectionBuilder::scan_edges_with_streaming_aggregation(
    const std::vector<std::string>& types,
    const std::unordered_map<std::string, ObjectId>& type_id_map
) {
    // Get projection directory for temporary files
    auto& manager = ProjectionManager::get_instance();
    std::string proj_dir = manager.get_projection_dir(projection_name);
    std::string temp_dir = proj_dir + "/tmp_sort";

    // Ensure temp directory exists
    std::filesystem::create_directories(temp_dir);

    // Phase 1: Collect all edges into EdgeAggregationBuffer
    // Memory: 64 MB buffer, spills to disk when full
    EdgeAggregationBuffer edge_buffer(temp_dir + "/edges", 64 * 1024 * 1024);

    std::cout << "[StreamingAggregation] Phase 1: Collecting edges..." << std::endl;

    // Optimization: Cache recent has_node() results to reduce hash lookups
    // Consecutive edges often share endpoints, so this can reduce lookup overhead by 30-50%
    // Cache size of 10K entries uses ~160KB memory and provides good hit rate
    //
    // With the node-id bitmap live, has_node() is a single L3 load (~1.6 ns)
    // while an unordered_map probe is 20-50 ns: the memo turns into a 12x to
    // 30x PESSIMIZATION. It is bypassed, not deleted, because in the fallback
    // world has_node() is a 292 ns binary search over 847 MB and the memo
    // genuinely pays there. The mode is read ONCE, outside the loop, so the
    // cost is one perfectly predicted branch per probe.
    const bool membership_is_o1 = storage->node_bitmap_bytes() > 0;

    constexpr size_t NODE_CACHE_SIZE = 10000;
    std::unordered_map<uint64_t, bool> node_in_projection_cache;
    node_in_projection_cache.reserve(NODE_CACHE_SIZE);

    // Helper lambda to check if node is in projection with caching
    auto cached_has_node =
        [this, &node_in_projection_cache, membership_is_o1](ObjectId node_id) -> bool {
        if (membership_is_o1) {
            return storage->has_node(node_id);
        }
        // Check cache first
        auto cache_it = node_in_projection_cache.find(node_id.id);
        if (cache_it != node_in_projection_cache.end()) {
            return cache_it->second;
        }

        // Cache miss - do actual lookup
        bool result = storage->has_node(node_id);

        // Add to cache, clear if too large (simple eviction strategy)
        if (node_in_projection_cache.size() >= NODE_CACHE_SIZE) {
            node_in_projection_cache.clear();
        }
        node_in_projection_cache[node_id.id] = result;

        return result;
    };

    for (const auto& type : types) {
        // Get per-type configuration
        Orientation type_orientation = get_orientation_for_type(type);
        Aggregation type_aggregation = get_aggregation_for_type(type);
        std::string type_agg_property = get_aggregation_property_for_type(type);

        // Safety check: Skip types that aren't COUNT/SUM (should not happen since
        // we only enter streaming mode when all types have COUNT/SUM aggregation,
        // but kept as defensive programming)
        if (type_aggregation != Aggregation::COUNT && type_aggregation != Aggregation::SUM) {
            std::cerr << "[StreamingAggregation] Warning: Type '" << type
                      << "' has non-COUNT/SUM aggregation - this should not happen. Skipping."
                      << std::endl;
            continue;
        }

        auto type_it = type_id_map.find(type);
        if (type_it == type_id_map.end()) {
            continue;
        }
        ObjectId type_id = type_it->second;

        // Scan all edges with this type
        scanner->scan_label_edge_with_endpoints(type_id,
            [this, &edge_buffer, type_id, type_orientation, &type_agg_property, &cached_has_node]
            (ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
                // Filter: only include if both endpoints are in projection
                // Use cached lookup for better performance on consecutive edges
                if (!cached_has_node(from_node) || !cached_has_node(to_node)) {
                    return;
                }

                // Extract property value for aggregation if needed
                double property_value = 0.0;
                if (!type_agg_property.empty()) {
                    auto prop_val = get_edge_property_value_for_aggregation(edge_id, type_agg_property);
                    if (prop_val.has_value()) {
                        property_value = prop_val.value();
                    }
                }

                // Apply orientation to determine storage order.
                // Zero-init silences a GCC -Wmaybe-uninitialized false positive:
                // the switch below is exhaustive over the Orientation enum class,
                // but GCC does not perform enum-exhaustiveness analysis.
                uint64_t store_from = 0, store_to = 0;
                switch (type_orientation) {
                    case Orientation::NATURAL:
                        store_from = from_node.id;
                        store_to = to_node.id;
                        break;
                    case Orientation::REVERSE:
                        store_from = to_node.id;
                        store_to = from_node.id;
                        break;
                    case Orientation::UNDIRECTED:
                        // Canonical ordering for undirected
                        if (from_node.id <= to_node.id) {
                            store_from = from_node.id;
                            store_to = to_node.id;
                        } else {
                            store_from = to_node.id;
                            store_to = from_node.id;
                        }
                        break;
                }

                // Create aggregation record
                EdgeAggregationRecord record{
                    store_from,
                    store_to,
                    type_id.id,
                    edge_id.id,
                    pack_double_to_bits(property_value)
                };

                edge_buffer.push_back(record);
            });
    }

    edge_buffer.finalize();

    std::cout << "[StreamingAggregation] Collected " << edge_buffer.size()
              << " edges, spilled: " << (edge_buffer.has_spilled() ? "yes" : "no") << std::endl;

    if (edge_buffer.size() == 0) {
        return;  // No edges to aggregate
    }

    // Phase 2: External sort by (from, to, type)
    std::cout << "[StreamingAggregation] Phase 2: External sort..." << std::endl;

    ExternalEdgeSort sorter(temp_dir);

    // Enable parallel I/O mode for large datasets (>1M edges)
    // This uses async I/O with prefetching for 2-4× speedup
    if (edge_buffer.size() > 1000000) {
        sorter.set_parallel_mode(4, 32);  // 4 threads, 32 queue depth
    }

    // Add spill files to sorter
    const auto& spill_paths = edge_buffer.get_spill_paths();
    const auto& spill_counts = edge_buffer.get_spill_record_counts();
    for (size_t i = 0; i < spill_paths.size(); ++i) {
        sorter.add_run(spill_paths[i], spill_counts[i]);
    }

    // Add any remaining memory records
    auto& mem_records = edge_buffer.get_memory_records();
    if (!mem_records.empty()) {
        sorter.add_memory_records(std::move(mem_records));
    }

    // Phase 3: Streaming aggregation with O(1) memory per group
    std::cout << "[StreamingAggregation] Phase 3: Streaming aggregation..." << std::endl;

    // Create _count property key for COUNT aggregation
    // Since _count is a projection-specific property (not in original graph),
    // we use a synthetic key ID (1 | MASK_EDGE_KEY) reserved for aggregation
    ObjectId count_key_id(COUNT_KEY_SYNTHETIC_ID | ObjectId::MASK_EDGE_KEY);
    std::cout << "[StreamingAggregation] Using synthetic _count key ID: "
              << std::hex << count_key_id.id << std::dec << std::endl;

    // Register the _count key in the projection catalog so queries can resolve it
    storage->register_edge_key("_count", COUNT_KEY_SYNTHETIC_ID);

    // Create streaming aggregator
    StreamingEdgeAggregator aggregator(
        Aggregation::COUNT,  // Use COUNT for the aggregation callback
        [this, count_key_id](uint64_t edge_id_raw, uint64_t count, double /*value*/) {
            ObjectId edge_id(edge_id_raw);

            // Store the count as a property on the representative edge
            if (count_key_id.id != 0) {
                ObjectId value_oid = Common::Conversions::pack_int(static_cast<int64_t>(count));
                storage->add_edge_property(edge_id, count_key_id, value_oid);
            }
        }
    );

    // Track the first edge of each group for adding to storage
    uint64_t prev_from = UINT64_MAX;
    uint64_t prev_to = UINT64_MAX;
    uint64_t prev_type = UINT64_MAX;
    bool first_of_group = true;

    // Progress tracking (low overhead: modulo check is ~1 CPU cycle)
    size_t progress_count = 0;
    size_t total_to_process = sorter.total_records();
    auto start_time = std::chrono::steady_clock::now();
    constexpr size_t PROGRESS_INTERVAL = 10000000;  // Every 10M records

    sorter.stream_sorted([&](const EdgeAggregationRecord& rec) {
        // Progress update (minimal overhead)
        if (++progress_count % PROGRESS_INTERVAL == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_sec = std::chrono::duration<double>(now - start_time).count();
            double pct = 100.0 * progress_count / total_to_process;
            double rate = progress_count / elapsed_sec / 1000000.0;  // M records/sec
            double eta_sec = (total_to_process - progress_count) / (progress_count / elapsed_sec);
            std::cout << "[Progress] " << progress_count / 1000000 << "M / "
                      << total_to_process / 1000000 << "M records ("
                      << std::fixed << std::setprecision(1) << pct << "%) - "
                      << std::setprecision(2) << rate << "M/s - ETA: "
                      << std::setprecision(0) << eta_sec << "s" << std::endl;
        }
        // Check if this is the first edge of a new group
        bool is_new_group = (rec.from_node != prev_from ||
                             rec.to_node != prev_to ||
                             rec.type_id != prev_type);

        if (is_new_group) {
            first_of_group = true;
            prev_from = rec.from_node;
            prev_to = rec.to_node;
            prev_type = rec.type_id;
        }

        // Add first edge of each group to projection storage
        if (first_of_group) {
            first_of_group = false;

            ObjectId edge_id(rec.edge_id);
            ObjectId from_node(rec.from_node);
            ObjectId to_node(rec.to_node);
            ObjectId type_id(rec.type_id);

            // Add edge to storage
            ProjectedEdge edge;
            edge.from_node = from_node;
            edge.to_node = to_node;
            edge.edge_id = edge_id;

            // Determine directedness from type configuration
            // For UNDIRECTED orientation, mark as undirected
            std::string type_name;
            for (const auto& [name, id] : gql_model.catalog.edge_labels2id) {
                if ((id | ObjectId::MASK_EDGE_LABEL) == type_id.id) {
                    type_name = name;
                    break;
                }
            }
            Orientation type_orientation = get_orientation_for_type(type_name);
            edge.is_directed = (type_orientation != Orientation::UNDIRECTED);

            // Skip Bloom filter check - streaming aggregation already guarantees uniqueness
            // via sorting by (from, to, type) and emitting only the first edge per group.
            // This eliminates false positives, achieving 100% edge accuracy.
            storage->add_edge(edge, true);  // skip_bloom_check = true

            // Extract properties for this edge
            if (!edge_property_keys.empty()) {
                extract_edge_properties(edge_id);
            }

            // Store edge label
            storage->add_edge_label(edge_id, type_id);
        }

        // Process for aggregation
        aggregator.process(rec);
    });

    // Finalize aggregation (emit last group)
    aggregator.finalize();

    std::cout << "[StreamingAggregation] Complete. Groups: " << aggregator.groups_emitted()
              << ", Records: " << aggregator.records_processed() << std::endl;

    // Cleanup temp directory
    std::filesystem::remove_all(temp_dir);
}

// ============================================================================
// Serialized scan implementations (MDB_PROJECTION_SERIAL_SCAN pipeline).
//
// Serialized equivalents of scan_nodes_impl_classic_ / scan_edges_impl_classic_
// that gate each storage emission on a target ProjectionIndex bitmask. These
// are invoked once per node-related index (5 passes) by finalize_serialized_,
// so the main scan loop runs five times over the same labels.
//
// Critical invariant: finalize_node_scan() MUST be called EXACTLY ONCE across
// all node-phase passes — specifically during the NODES pass — because it
// populates ProjectionStorage::collected_nodes_ which the edge filter's
// has_node() (Phase B) depends on. If we finalized on every pass, subsequent
// passes would mutate the node set the edge filter has already consulted.
//
// TODO(serialized-gnn-property-pass): The emit_properties gate here only
// consults the target_mask. In GNN-only configs (includeFeatures + labelProperty +
// splitProperty with no explicit nodeProperties), classic's
// extract_node_properties is called for its side effect of populating
// labels_buffer_ / splits_buffer_ via try_extract_gnn_property, even
// though no property records get emitted. For that to work under
// SERIALIZED mode, enabled_indexes_() MUST push NODE_KEY_VALUE /
// KEY_VALUE_NODE into the pass list when gnn_row_mapping_ != nullptr,
// regardless of features.include_node_properties. Otherwise GNN training
// with SERIAL_SCAN=1 trains on unlabeled data.
// ============================================================================

void NativeProjectionBuilder::scan_nodes_impl_serialized_(
    const std::vector<std::string>& labels,
    ProjectionIndex target_mask)
{
    auto bench_t0 = benchmark_timers_.enabled ? ProjectionTimers::Clock::now()
                                               : ProjectionTimers::Clock::time_point{};

    // NODES               -> emit to node_batch (main nodes index).
    // NODE_LABEL/LABEL_NODE -> emit to node<->label pairs via storage.
    // NODE_KEY_VALUE/KEY_VALUE_NODE -> extract node properties.
    const bool emit_nodes      = has_flag(target_mask, ProjectionIndex::NODES);
    const bool emit_node_label = has_flag(target_mask, ProjectionIndex::NODE_LABEL) ||
                                 has_flag(target_mask, ProjectionIndex::LABEL_NODE);
    const bool emit_properties = has_flag(target_mask, ProjectionIndex::NODE_KEY_VALUE) ||
                                 has_flag(target_mask, ProjectionIndex::KEY_VALUE_NODE);

    for (const auto& label : labels) {
        validate_label_exists(label);

        auto it = gql_model.catalog.node_labels2id.find(label);
        if (it == gql_model.catalog.node_labels2id.end()) {
            throw std::runtime_error(
                "Label '" + label + "' not found in catalog"
            );
        }
        ObjectId label_id(it->second | ObjectId::MASK_NODE_LABEL);

        scanner->scan_label_node(label_id,
            [this, label_id, emit_nodes, emit_node_label, emit_properties](ObjectId node_id) {
                if (emit_nodes) {
                    ProjectedNode node;
                    node.node_id = node_id;
                    node_batch.push_back(node);
                    if (node_batch.size() >= BATCH_SIZE) {
                        flush_nodes();
                    }
                }
                if (emit_properties) {
                    extract_node_properties(node_id);
                }
                if (emit_node_label) {
                    storage->add_node_label(node_id, label_id);
                }
            });
    }

    if (emit_nodes && !node_batch.empty()) {
        flush_nodes();
    }

    // Finalise the scan-phase node tracker ONLY on the NODES pass, so
    // collected_nodes_ is populated exactly once. Other passes
    // (NODE_LABEL, LABEL_NODE, NODE_KEY_VALUE, KEY_VALUE_NODE) read node
    // ids via scan_label_node without mutating collected_nodes_ — the
    // edge filter in Phase B/C depends on this single-finalise invariant
    // (has_node() binary-search contract for scan_edges_by_types).
    if (emit_nodes) {
        storage->finalize_node_scan();
    }

    if (benchmark_timers_.enabled) {
        benchmark_timers_.node_scan_ms += std::chrono::duration<double, std::milli>(
            ProjectionTimers::Clock::now() - bench_t0).count();
    }
}

// ============================================================================
// Serialized scan Phase B: precompute_edge_filter_
//
// Single full edge scan that evaluates has_node() on both endpoints for every
// edge of the requested types, recording the outcome bit-by-bit in an
// EdgeFilter (a composite of two EdgeKeepBitmap instances, directed +
// undirected, keyed by the 56-bit counter portion of the ObjectId). No record
// emission happens here — the filter is the sole output, consumed read-only
// by Phase C's 9 edge-index passes (scan_edges_impl_serialized_) so that
// has_node() work is paid once instead of 9×.
//
// Keying by counter (not the full tagged edge_id.id) is REQUIRED: GQL edge
// ObjectIds carry an 8-bit type prefix (MASK_DIRECTED_EDGE = 0xE0.., or
// MASK_UNDIRECTED_EDGE = 0xE4..), making the raw id ~1.6e19. Using that as
// a vector<bool> index would std::bad_alloc on the first edge. EdgeFilter
// strips the prefix via ObjectId::VALUE_MASK and routes the kept-bit into
// the correct per-orientation bitmap internally.
//
// Invariant I2: filter is finalized before return, so Phase C's consumers see
// an immutable snapshot and can read concurrently without synchronization if
// they later go parallel.
// ============================================================================
std::unique_ptr<EdgeFilter>
NativeProjectionBuilder::precompute_edge_filter_(const std::vector<std::string>& types)
{
    auto filter = std::make_unique<EdgeFilter>();

    // Pass 1 — resolve type_ids + estimate total edge count. Mirrors
    // scan_edges_impl_classic_'s catalog-lookup sequence to keep error
    // messages identical.
    uint64_t total_estimate = 0;
    std::unordered_map<std::string, ObjectId> type_id_map;
    for (const auto& type : types) {
        validate_type_exists(type);

        auto it = gql_model.catalog.edge_labels2id.find(type);
        if (it == gql_model.catalog.edge_labels2id.end()) {
            throw std::runtime_error("Type '" + type + "' not found in catalog");
        }
        ObjectId type_id(it->second | ObjectId::MASK_EDGE_LABEL);
        type_id_map[type] = type_id;
        total_estimate += scanner->count_edges_by_type(type_id);
    }

    // Resize ProjectionStorage's Bloom filter using the total estimate.
    // scan_edges_impl_classic_ does this on its own fast path (see classic
    // impl comment "Resize Bloom filter based on estimated edge count"),
    // and Phase C's per-index edge passes rely on has_edge() probes in the
    // property-emission branches of scan_edges_impl_serialized_, so we must
    // mirror the sizing here.
    if (total_estimate > 0) {
        storage->resize_bloom_filter(total_estimate);
    }

    // We do NOT pre-reserve per-orientation capacity in the filter: we
    // don't know the directed/undirected split without an extra scan, and
    // EdgeKeepBitmap::set_kept auto-grows cheaply (amortized O(1)) for both
    // bitmaps independently.

    // Pass 2 — scan all edges of each type, setting the bit for the ones
    // that survive the has_node() filter. No batch emission, no aggregation,
    // no property extraction: this phase is purely a filter pre-computation.
    //
    // The per-edge has_node()×2 work is GPU-friendly (parallel binary search
    // over the sorted collected_nodes_ array), so we route edges through
    // EdgeKeepBitmapGpuBatcher, which buffers (edge_id, from, to) triples
    // and dispatches them either to mdb::gpu::edge_keep_membership_gpu
    // (sister to the bitset filter in src/gpu/ops/gpu_filter.cu) or to a
    // CPU fallback that is bit-identical to the historic inline lambda.
    // Set MDB_PROJECTION_BITMAP_GPU=0 to force the CPU path for A/B
    // benchmarking. Tiny graphs (cora_gnn) are never sent to the GPU
    // regardless: the batcher's min_edges_for_gpu threshold dominates the
    // heuristic.
    //
    // NOTE: ParallelEdgeDetector is NOT run here.  A detector that is never
    // cleared grows to hold ALL kept edges (~138 bytes per entry), which
    // caused 25 GB RSS on papers100M (Run 7 PSI-abort).  Detection instead
    // runs in scan_edges_impl_serialized_ on the first Phase C pass
    // (FROM_TO_EDGE), mirroring the per-batch clear() pattern of the classic
    // path (scan_edges_impl_classic_, line 723).
    EdgeKeepBitmapGpuBatcher batcher(*filter, *storage);
    for (const auto& type : types) {
        ObjectId type_id = type_id_map[type];
        scanner->scan_label_edge_with_endpoints(
            type_id,
            [&batcher](ObjectId edge_id, ObjectId from_node, ObjectId to_node)
            {
                batcher.add(edge_id, from_node, to_node);
            });
    }
    batcher.flush();

    filter->finalize();
    return filter;
}

// ============================================================================
// Serialized scan Phase C: scan_edges_impl_serialized_
//
// Mirror of scan_edges_impl_classic_ for the SERIALIZED pipeline.
// finalize_serialized_() calls this once per edge-related index — typically
// with a single-bit target_mask — so each B+Tree pass does one sequential
// scan of the label_edge index without redundantly re-running
// ParallelEdgeDetector or has_node() lookups.
//
// The EdgeFilter produced by precompute_edge_filter_ (Phase B) is consumed
// here read-only: filter->is_kept(edge_id) is an O(1) bitmap probe that
// replaces the classic path's per-edge has_node(from) + has_node(to) pair.
//
// Aggregation modes (SUM/MIN/MAX/COUNT) are NOT handled here: graphs with any
// non-SINGLE aggregation fall back to the classic path (see
// has_non_single_aggregation_() + finalize_serialized_()). Only SINGLE mode
// reaches this function.
//
// Invariant I1: Phase B must complete before any Phase C call.
//   → Enforced by the nullptr guard below (finalize_serialized_() holds the
//     unique_ptr and passes a raw pointer here only after Phase B returns).
// Invariant I2: filter is immutable after finalize() — safe to read
//   concurrently if the per-index passes are later parallelised.
// ============================================================================
void NativeProjectionBuilder::scan_edges_impl_serialized_(
    const std::vector<std::string>& types,
    ProjectionIndex target_mask,
    const EdgeFilter* filter)
{
    auto bench_t0 = benchmark_timers_.enabled ? ProjectionTimers::Clock::now()
                                              : ProjectionTimers::Clock::time_point{};

    if (filter == nullptr) {
        throw std::logic_error(
            "scan_edges_impl_serialized_ requires non-null filter "
            "(precompute_edge_filter_ must run first)");
    }

    // Mask-gated emission booleans. In SERIALIZED mode each call receives
    // a single-bit target_mask, so typically only one emit_* is true —
    // except for the two label indexes (EDGE_LABEL / LABEL_EDGE) and the
    // two property indexes (EDGE_KEY_VALUE / KEY_VALUE_EDGE) which share
    // the underlying emission: when either single-bit is passed, both the
    // label buffer write (or property extraction) runs. The serialized
    // orchestrator dispatches each pass independently; build_one_index
    // then picks the correct B+Tree to sort-and-write.
    const bool emit_from_to         = has_flag(target_mask, ProjectionIndex::FROM_TO_EDGE);
    const bool emit_to_from         = has_flag(target_mask, ProjectionIndex::TO_FROM_EDGE);
    const bool emit_edge_direction  = has_flag(target_mask, ProjectionIndex::EDGE_DIRECTION);
    const bool emit_edge_from_to    = has_flag(target_mask, ProjectionIndex::EDGE_FROM_TO);
    const bool emit_edge_n1_n2      = has_flag(target_mask, ProjectionIndex::EDGE_N1_N2);
    const bool emit_edge_label      = has_flag(target_mask, ProjectionIndex::EDGE_LABEL) ||
                                      has_flag(target_mask, ProjectionIndex::LABEL_EDGE);
    const bool emit_edge_properties = has_flag(target_mask, ProjectionIndex::EDGE_KEY_VALUE) ||
                                      has_flag(target_mask, ProjectionIndex::KEY_VALUE_EDGE);

    // "any edge buffer" covers the 5 core edge-record buffers that share
    // the `edge_batch` flush lifecycle (FROM_TO_EDGE, TO_FROM_EDGE,
    // EDGE_DIRECTION, EDGE_FROM_TO, EDGE_N1_N2).
    const bool emit_any_edge_buffer = emit_from_to || emit_to_from || emit_edge_direction ||
                                      emit_edge_from_to || emit_edge_n1_n2;

    // SINGLE-mode parallel edge detection:
    // Run the ParallelEdgeDetector only on the first Phase C pass (FROM_TO_EDGE).
    // Gating on FROM_TO_EDGE avoids running detection 9× (once per edge index)
    // while still throwing the same QueryException before any B+Tree build begins.
    // The per-batch clear() mirrors classic's pattern in scan_edges_impl_classic_:
    // after each BATCH_SIZE flush the map is cleared, keeping peak RSS bounded
    // regardless of graph size.  This replaces the unbounded Phase B detector
    // that caused 25 GB RSS on papers100M (Run 7 PSI-abort).  The bound makes
    // detection windowed, not exact — parallels more than BATCH_SIZE apart in
    // scan order escape detection (see the ParallelEdgeDetector class doc).
    const bool run_detection = has_flag(target_mask, ProjectionIndex::FROM_TO_EDGE);

    std::unordered_map<std::string, ObjectId> type_id_map;
    for (const auto& type : types) {
        validate_type_exists(type);
        auto it = gql_model.catalog.edge_labels2id.find(type);
        if (it == gql_model.catalog.edge_labels2id.end()) {
            throw std::runtime_error("Type '" + type + "' not found in catalog");
        }
        type_id_map[type] = ObjectId(it->second | ObjectId::MASK_EDGE_LABEL);
    }

    for (const auto& type : types) {
        Orientation type_orientation = get_orientation_for_type(type);
        ObjectId type_id = type_id_map[type];

        // Create a per-type detector when running detection on this pass.
        // finalize_serialized_ only reaches Phase C when all types are SINGLE
        // (has_non_single_aggregation_ guard), so Aggregation::SINGLE is correct.
        std::unique_ptr<ParallelEdgeDetector> detector;
        if (run_detection) {
            detector = std::make_unique<ParallelEdgeDetector>(Aggregation::SINGLE);
        }

        scanner->scan_label_edge_with_endpoints(type_id,
            [this, filter, type_id, type_orientation,
             emit_any_edge_buffer, emit_edge_label, emit_edge_properties,
             &detector]
            (ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
            if (!filter->is_kept(edge_id)) return;  // O(1) bitmap lookup

            // Parallel edge detection for SINGLE mode: process_edge() throws
            // QueryException on the second occurrence of any (from, to, type)
            // triple — identical to classic path.
            if (detector) {
                detector->process_edge(
                    from_node.id, to_node.id, type_id.id,
                    edge_id, std::nullopt);
            }

            if (emit_any_edge_buffer) {
                switch (type_orientation) {
                    case Orientation::NATURAL: {
                        ProjectedEdge edge;
                        edge.from_node = from_node;
                        edge.to_node   = to_node;
                        edge.edge_id   = edge_id;
                        uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
                        edge.is_directed = (edge_type != ObjectId::MASK_UNDIRECTED_EDGE);
                        edge_batch.push_back(edge);
                        break;
                    }
                    case Orientation::REVERSE: {
                        ProjectedEdge edge;
                        edge.from_node = to_node;
                        edge.to_node   = from_node;
                        edge.edge_id   = edge_id;
                        uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
                        edge.is_directed = (edge_type != ObjectId::MASK_UNDIRECTED_EDGE);
                        edge_batch.push_back(edge);
                        break;
                    }
                    case Orientation::UNDIRECTED: {
                        ProjectedEdge edge;
                        if (from_node.id <= to_node.id) {
                            edge.from_node = from_node;
                            edge.to_node   = to_node;
                        } else {
                            edge.from_node = to_node;
                            edge.to_node   = from_node;
                        }
                        edge.edge_id     = edge_id;
                        edge.is_directed = false;
                        edge_batch.push_back(edge);
                        break;
                    }
                }
            }

            // Edges carry no GNN side-effects (labels/splits live on nodes — see the
            // TODO(serialized-gnn-property-pass) block in
            // scan_nodes_impl_serialized_), so the empty-key guard is a safe
            // short-circuit here that it is NOT for nodes.
            if (emit_edge_properties && !edge_property_keys.empty()) {
                extract_edge_properties(edge_id);
            }

            if (emit_edge_label) {
                storage->add_edge_label(edge_id, type_id);
            }

            // Auto-flush and clear detector when batch is full.
            // The clear() DOES drop SINGLE state — the seen-edge set is the
            // detector's only state — so duplicate detection is best-effort
            // within a BATCH_SIZE-edge window: two parallel edges more than
            // BATCH_SIZE apart in scan order are BOTH stored without the
            // QueryException. Accepted memory trade-off (an uncleared detector
            // holds every kept edge; 25 GB RSS on papers100M) matching
            // classic's per-batch clear in scan_edges_impl_classic_. See the
            // ParallelEdgeDetector class doc.
            if (emit_any_edge_buffer && edge_batch.size() >= BATCH_SIZE) {
                flush_edges();
                if (detector) {
                    detector->clear();
                }
            }
        });

        // Clear detector after each type to release memory promptly.
        if (detector) {
            detector->clear();
        }
    }

    if (emit_any_edge_buffer && !edge_batch.empty()) {
        flush_edges();
    }

    if (benchmark_timers_.enabled) {
        benchmark_timers_.edge_scan_ms += std::chrono::duration<double, std::milli>(
            ProjectionTimers::Clock::now() - bench_t0).count();
    }
}

// ============================================================================
// enabled_indexes_(), has_non_single_aggregation_(), finalize_serialized_()
// — serialized pipeline orchestrator.
//
// enabled_indexes_() computes the ordered list of single-bit ProjectionIndex
// values to iterate in Phase A (node indexes) and Phase C (edge indexes).
// The order within each phase is fixed:
//   Phase A: NODES → NODE_LABEL → LABEL_NODE → NODE_KEY_VALUE → KEY_VALUE_NODE
//   Phase C: FROM_TO_EDGE → TO_FROM_EDGE → EDGE_DIRECTION → EDGE_FROM_TO →
//             EDGE_N1_N2 → EDGE_LABEL → LABEL_EDGE → EDGE_KEY_VALUE →
//             KEY_VALUE_EDGE
// Only indexes that are actually configured (include_label_indexes_ /
// node_property_keys / edge_property_keys) are emitted — this is what bounds
// peak scratch disk to O(max single index) instead of O(sum all indexes).
//
// GNN disjunct (see TODO(serialized-gnn-property-pass)):
//   classic's extract_node_properties populates labels_buffer_ / splits_buffer_
//   via try_extract_gnn_property as a side-effect, even when node_property_keys
//   is empty. Under SERIALIZED, property extraction only fires during
//   NODE_KEY_VALUE / KEY_VALUE_NODE passes. So we MUST push those two indexes
//   whenever gnn_row_mapping_ != nullptr, regardless of whether node
//   properties were explicitly configured. Failure to do so causes Cora's
//   testAccuracy to drop from 0.7900 to random under SERIAL_SCAN=1.
//
// has_non_single_aggregation_(): aggregation state (COUNT/SUM/MIN/MAX maps)
//   would be too large to persist across 9 edge-index passes, so graphs with
//   any non-SINGLE aggregation fall back to the classic single-pass path with
//   a stderr warning.
//
// finalize_serialized_() orchestrates:
//   Phase A: node-index passes (scan_nodes_impl_serialized_ + build_one_index
//            + reset_sort_scratch_ + malloc_trim per index)
//   Phase B: single full edge scan → EdgeFilter bitmap
//            (precompute_edge_filter_)
//   Phase C: edge-index passes (scan_edges_impl_serialized_ + build_one_index
//            + reset_sort_scratch_ + malloc_trim per index)
// ============================================================================

std::vector<ProjectionIndex> NativeProjectionBuilder::enabled_indexes_() const {
    std::vector<ProjectionIndex> out;

    // Phase A: nodes first (CRITICAL — finalize_node_scan populates
    // collected_nodes_ that Phase B's has_node() depends on, spec §6 I1).
    // Label and property indexes follow in a fixed order.
    out.push_back(ProjectionIndex::NODES);

    if (include_label_indexes_) {
        out.push_back(ProjectionIndex::NODE_LABEL);
        out.push_back(ProjectionIndex::LABEL_NODE);
    }

    // Determine whether node-property passes are needed.
    // Uses the same conditions as scan_nodes_impl_classic_ (line 506):
    //   !node_property_keys.empty() || !node_prop_configs.empty()
    // PLUS the GNN disjunct: classic's extract_node_properties has a GNN
    // side-effect (populating labels_buffer_ / splits_buffer_ via
    // try_extract_gnn_property) that fires even when node_property_keys is
    // empty. Under SERIALIZED, property extraction only runs during
    // NODE_KEY_VALUE / KEY_VALUE_NODE passes — so we MUST include them
    // whenever GNN is active, regardless of features.include_node_properties.
    // Without this, Cora's testAccuracy drops from 0.7900 to random
    // (code-review finding, see TODO(serialized-gnn-property-pass) in
    // scan_nodes_impl_serialized_).
    bool needs_node_properties = !node_property_keys.empty() || !node_prop_configs.empty();
#ifdef ENABLE_GNN
    needs_node_properties = needs_node_properties || (gnn_row_mapping_ != nullptr);
#endif
    if (needs_node_properties) {
        out.push_back(ProjectionIndex::NODE_KEY_VALUE);
        out.push_back(ProjectionIndex::KEY_VALUE_NODE);
    }

    // Phase C: core 5 edge indexes (always required), then optional
    // label and property indexes.
    out.push_back(ProjectionIndex::FROM_TO_EDGE);
    out.push_back(ProjectionIndex::TO_FROM_EDGE);
    out.push_back(ProjectionIndex::EDGE_DIRECTION);
    out.push_back(ProjectionIndex::EDGE_FROM_TO);
    out.push_back(ProjectionIndex::EDGE_N1_N2);

    if (include_label_indexes_) {
        out.push_back(ProjectionIndex::EDGE_LABEL);
        out.push_back(ProjectionIndex::LABEL_EDGE);
    }

    // Determine whether edge-property passes are needed.
    // Mirrors the constructor's has_count_aggregation logic (lines 173-180):
    //   !edge_property_keys.empty()  ||  !edge_prop_configs.empty()
    //   || global COUNT aggregation
    //   || any per-type COUNT aggregation (synthetic _count key)
    bool needs_edge_properties = !edge_property_keys.empty() || !edge_prop_configs.empty();
    if (!needs_edge_properties && aggregation == Aggregation::COUNT) {
        needs_edge_properties = true;
    }
    if (!needs_edge_properties) {
        for (const auto& [t, agg] : per_type_aggregations) {
            if (agg == Aggregation::COUNT) {
                needs_edge_properties = true;
                break;
            }
        }
    }
    if (needs_edge_properties) {
        out.push_back(ProjectionIndex::EDGE_KEY_VALUE);
        out.push_back(ProjectionIndex::KEY_VALUE_EDGE);
    }

    return out;
}

bool NativeProjectionBuilder::has_non_single_aggregation_() const {
    // Global aggregation setting applies when stored_types_ is empty
    // (e.g., projection with no edge types but aggregation: COUNT set).
    if (aggregation != Aggregation::SINGLE) {
        return true;
    }
    for (const auto& type : stored_types_) {
        if (get_aggregation_for_type(type) != Aggregation::SINGLE) {
            return true;
        }
    }
    return false;
}

void NativeProjectionBuilder::finalize_serialized_() {
    if (!scan_inputs_captured_) {
        throw std::logic_error(
            "finalize_serialized_: no scan inputs captured - "
            "scan_nodes_by_labels + scan_edges_by_types must be called first");
    }

    // Aggregation modes (COUNT/SUM/MIN/MAX) require the classic path because
    // aggregation state would be too large to persist across 9 edge-index
    // passes. Warn and fall through to the classic impls.
    if (has_non_single_aggregation_()) {
        std::cerr << "[Projection] SERIAL_SCAN disabled for this projection: "
                     "aggregation mode (COUNT/SUM/MIN/MAX) requires the "
                     "classic path. Falling back to classic scan."
                  << std::endl;
        scan_nodes_impl_classic_(stored_labels_);
        scan_edges_impl_classic_(stored_types_);
        return;
    }

    auto all_idx = enabled_indexes_();

    // Compute the active IndexSet preset mask once per build so each iteration
    // of the Phase A / Phase C loops below can gate topology index
    // materialization with a cheap bitwise probe (has_flag). Property indexes
    // (NODE_KEY_VALUE, KEY_VALUE_NODE, EDGE_KEY_VALUE, KEY_VALUE_EDGE) are NOT
    // gated by IndexSet — their existing property-config gate (via
    // features.include_*_properties inside build_*_index_()) remains the sole
    // controller. The mask below is only consulted for the 10 topology / label
    // indexes.
    const ProjectionIndex active_mask = project_index_mask_for(index_set_);
    auto is_property_index = [](ProjectionIndex idx) {
        return idx == ProjectionIndex::NODE_KEY_VALUE
            || idx == ProjectionIndex::KEY_VALUE_NODE
            || idx == ProjectionIndex::EDGE_KEY_VALUE
            || idx == ProjectionIndex::KEY_VALUE_EDGE;
    };

    // Split into node phase (Phase A) + edge phase (Phase C) via bitmask.
    std::vector<ProjectionIndex> node_phase;
    std::vector<ProjectionIndex> edge_phase;
    for (auto idx : all_idx) {
        if (has_flag(ProjectionIndex::ALL_NODE, idx)) {
            node_phase.push_back(idx);
        } else {
            edge_phase.push_back(idx);
        }
    }

    // ---- Phase A: node indexes — one scan + build + reset per index ----
    // NODES pass runs first and calls finalize_node_scan() (spec §6 I1).
    // Subsequent label/property passes scan the same label_node index
    // but emit only to their target buffers.
    //
    // drain_pending_batches() flushes storage's internal BATCH_SIZE=250 node/edge
    // batches into the streaming record buffers before build_one_index() reads
    // them. Without it, up to 249 records can remain un-flushed when the index
    // build runs, causing those records to be omitted from the B+Tree and later
    // re-processed by the flush() call, which overwrites the correct index with
    // a partial dataset (correctness invariant: buffers must be fully flushed
    // before build_one_index() reads them — I4 golden compare guard).
    for (auto idx : node_phase) {
        // Skip topology index materialization when its bit is not set in the
        // active IndexSet preset mask. Property indexes bypass this gate (see
        // is_property_index lambda); they remain controlled solely by their
        // property-config gate inside build_one_index() → build_*_() helpers.
        // When the bit is masked out we skip the scan too — there is no buffer
        // to populate since no downstream consumer would read it, so the scan
        // work would be wasted I/O.
        if (!is_property_index(idx) && !has_flag(active_mask, idx)) {
            continue;
        }
        scan_nodes_impl_serialized_(stored_labels_, idx);
        storage->drain_pending_batches();
        storage->build_one_index(idx);
        storage->reset_sort_scratch_();
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
    }

    // ---- Phase B: precompute edge-keep filter (single full edge scan) ----
    // precompute_edge_filter_ walks every edge of stored_types_, evaluates
    // has_node() on both endpoints (O(log N) after Phase A's finalize_node_scan),
    // and records the outcome bit-by-bit. No record emission; the filter is
    // the sole output. Also resizes the Bloom filter to match the estimated
    // edge count (matching scan_edges_impl_classic_'s contract).
    auto filter = precompute_edge_filter_(stored_types_);

    // ---- Phase C: edge indexes — one scan + build + reset per index ----
    // Each pass calls scan_edges_impl_serialized_ with a single-bit mask;
    // the EdgeFilter (Phase B) replaces per-edge has_node() lookups with
    // O(1) bitmap probes (spec §6 I2: filter is immutable after finalize()).
    // drain_pending_batches() before each build ensures the streaming buffer
    // is fully populated (same correctness argument as Phase A above).
    for (auto idx : edge_phase) {
        // Skip topology index materialization when its bit is not set in the
        // active IndexSet preset mask. Property indexes bypass this gate (see
        // is_property_index lambda above). Skipping the scan too avoids wasted
        // per-pass I/O (filter probe, spill read) when no downstream buffer
        // would be populated.
        if (!is_property_index(idx) && !has_flag(active_mask, idx)) {
            continue;
        }
        // Arm the per-pass write mask BEFORE the scan so flush_edge_batch()
        // only populates the target buffer.  Also clears the bloom filter and
        // any stale spill files from previous passes (disk-bound fix for
        // papers100M ENOSPC — see ProjectionStorage::begin_serial_edge_pass_).
        storage->begin_serial_edge_pass_(idx);
        scan_edges_impl_serialized_(stored_types_, idx, filter.get());
        storage->drain_pending_batches();
        storage->build_one_index(idx);
        storage->end_serial_edge_pass_();
        storage->reset_sort_scratch_();
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
    }

    // filter unique_ptr releases EdgeFilter here; malloc_trim on the next
    // pass (or the caller's subsequent finalize() bookkeeping) returns the
    // heap pages to the kernel.

    // ---- Phase 4: open B+Tree readers ----
    // After all piecemeal build passes, open every .leaf/.dir reader so the
    // projection is queryable. Under CLASSIC, build_all_indexes_bulk() calls
    // open_all_bplustree_readers_() itself; under SERIALIZED we must do it
    // here because build_all_indexes_bulk() is bypassed (its backing buffers
    // are all empty after Phase A/B/C).
    storage->open_all_bplustree_readers_();
}

// ============================================================================
// CSR topology sidecar emission (topology_fwd.csr / topology_rev.csr)
// ============================================================================
//
// build_topology_snapshots_() orchestrates per-direction emission of
// mmap-backed CSR sidecar files gated by the active IndexSet mask. Called
// from finalize() after storage->flush() so the B+Tree `.leaf` / `.dir`
// files exist on disk and the BPT readers held by ProjectionStorage point
// at them.
//
// Per-direction failures are logged and swallowed: the projection stays
// valid, and the user can retry through the post-hoc gnn_build_topology_snapshot
// procedure.

void NativeProjectionBuilder::build_topology_snapshots_() {
    if (!build_topology_snapshot_) {
        return;
    }

    // Under CSR_HYBRID graph storage, the edge-index B+Tree leaves ARE the
    // CSR layout (FROM_TO_EDGE / TO_FROM_EDGE leaves embed the neighbor
    // list directly), which supersedes any separate topology sidecar file.
    // If the user also set buildTopologySnapshot=true we silently drop the
    // sidecar emission here; project_procedure.cc logs a single warning line
    // so callers notice the flag had no effect. The projection itself remains
    // valid and fully queryable via the CSR-aware BPT readers opened in
    // ProjectionStorage.
    if (graph_storage_ == BPT::GraphStorage::CSR_HYBRID) {
        return;
    }

    const ProjectionIndex active_mask = project_index_mask_for(index_set_);
    const bool fwd_ok = has_flag(active_mask, ProjectionIndex::FROM_TO_EDGE);
    const bool rev_ok = has_flag(active_mask, ProjectionIndex::TO_FROM_EDGE);

    if (!fwd_ok && !rev_ok) {
        std::cerr << "[Projection] buildTopologySnapshot requested but IndexSet "
                     "lacks both FROM_TO_EDGE and TO_FROM_EDGE — skipping sidecar "
                     "generation for projection '" << projection_name << "'."
                  << std::endl;
        return;
    }

    // Fast path: if the integrated build path (invoked inline by
    // ProjectionStorage::build_{from_to,to_from}_edge_index_) already
    // emitted both CSR sidecar files during the B+Tree build, we have
    // nothing to do. Fall through only for directions where the integrated
    // path was disabled or failed, so the legacy post-hoc BPT walker retries
    // them as a safety net. On a healthy build this branch is the common
    // case and the whole function returns with a single fast check.
    const bool fwd_done = storage->fwd_topology_snapshot_built();
    const bool rev_done = storage->rev_topology_snapshot_built();

    if (fwd_ok) {
        if (fwd_done) {
            // Integrated path already wrote topology_fwd.csr during the
            // FROM_TO_EDGE build. Nothing to do.
        } else {
            BPlusTree<3>* fwd_bpt = storage->get_from_to_edge_index();
            if (fwd_bpt == nullptr) {
                std::cerr << "[Projection] buildTopologySnapshot: from_to_edge "
                             "B+Tree not open for projection '" << projection_name
                          << "' — skipping topology_fwd.csr." << std::endl;
            } else {
                try {
                    build_one_topology_snapshot_(/*direction=*/0, fwd_bpt);
                } catch (const std::exception& e) {
                    std::cerr << "[Projection] failed to build topology_fwd.csr"
                              << " for '" << projection_name << "': " << e.what()
                              << std::endl;
                }
            }
        }
    } else {
        std::cerr << "[Projection] buildTopologySnapshot: FROM_TO_EDGE not in "
                     "active IndexSet — skipping topology_fwd.csr." << std::endl;
    }

    if (rev_ok) {
        if (rev_done) {
            // Integrated path already wrote topology_rev.csr.
        } else {
            BPlusTree<3>* rev_bpt = storage->get_to_from_edge_index();
            if (rev_bpt == nullptr) {
                std::cerr << "[Projection] buildTopologySnapshot: to_from_edge "
                             "B+Tree not open for projection '" << projection_name
                          << "' — skipping topology_rev.csr." << std::endl;
            } else {
                try {
                    build_one_topology_snapshot_(/*direction=*/1, rev_bpt);
                } catch (const std::exception& e) {
                    std::cerr << "[Projection] failed to build topology_rev.csr"
                              << " for '" << projection_name << "': " << e.what()
                              << std::endl;
                }
            }
        }
    } else {
        std::cerr << "[Projection] buildTopologySnapshot: TO_FROM_EDGE not in "
                     "active IndexSet — skipping topology_rev.csr." << std::endl;
    }
}

void NativeProjectionBuilder::build_one_topology_snapshot_(
    int   direction,
    void* edge_bpt_opaque)
{
    using Projection::TopologySnapshotWriter;

    auto* edge_bpt = static_cast<BPlusTree<3>*>(edge_bpt_opaque);
    if (edge_bpt == nullptr) {
        throw std::runtime_error("null B+Tree passed to build_one_topology_snapshot_");
    }

    const TopologySnapshotWriter::Direction dir =
        (direction == 0) ? TopologySnapshotWriter::Direction::FORWARD
                         : TopologySnapshotWriter::Direction::REVERSE;

    const uint64_t num_nodes = storage->get_node_count();
    const std::filesystem::path proj_dir = storage->get_projection_dir();

    // Pass 1: per-source degree histogram.
    // The B+Tree key layout for both from_to_edge (src, dst, edge_id) and
    // to_from_edge (dst, src, edge_id) places the node whose adjacency the
    // CSR is keyed by at index 0. Records store FULL ObjectIds (with the
    // 8-bit type tag), but the CSR's ROW_PTR is indexed by dense row id
    // — so we mask off the type tag via ObjectId::VALUE_MASK before using
    // the value as a degrees[] subscript. This assumes the projection's
    // node ObjectIds are a dense [0, N) range after stripping the type
    // tag, which holds for single-label projections (the thesis case:
    // cora_gnn, ogbn-*, papers100M). Non-dense multi-label projections
    // are a known limitation (skipped+warned in future work).
    std::vector<uint64_t> degrees(num_nodes, 0);
    bool interrupt = false;
    Record<3> min_rec = {0, 0, 0};
    Record<3> max_rec = {UINT64_MAX, UINT64_MAX, UINT64_MAX};

    {
        auto iter = edge_bpt->get_range(&interrupt, min_rec, max_rec);
        const Record<3>* rec = nullptr;
        while ((rec = iter.next()) != nullptr) {
            const uint64_t src_idx = (*rec)[0] & ObjectId::VALUE_MASK;
            if (src_idx < num_nodes) {
                ++degrees[src_idx];
            }
            // Out-of-range src is silently skipped: the B+Tree is the source
            // of truth, but defensive lower-bound avoids a buffer overrun if
            // node_count / tree are ever out of sync.
        }
    }

    // Pass 2: stream edges in src-monotonic order into the writer. BptIter
    // walks in key order, which for both directions starts with the key-0
    // component ascending — the exact monotonicity contract append_edge()
    // enforces. The writer uses src.id as the row subscript, so we pass
    // the stripped value; dst and edge_id retain their full ObjectId so
    // the reader's COL_IDX / EDGE_IDS slices match the B+Tree path.
    TopologySnapshotWriter writer(
        proj_dir,
        dir,
        num_nodes,
        std::move(degrees),
        /*include_edge_ids=*/true);

    {
        auto iter = edge_bpt->get_range(&interrupt, min_rec, max_rec);
        const Record<3>* rec = nullptr;
        while ((rec = iter.next()) != nullptr) {
            writer.append_edge(
                ObjectId{(*rec)[0] & ObjectId::VALUE_MASK},
                ObjectId{(*rec)[1]},
                ObjectId{(*rec)[2]});
        }
    }

    writer.finalize();
}

// ----------------------------------------------------------------------------
// Test-only helper for CSR topology sidecar emission — shares the BPT-scan
// + writer body of build_one_topology_snapshot_ so the gtest exercises the
// real code path. Deliberately surfaces exceptions (the production method
// swallows them to keep the projection valid).
// ----------------------------------------------------------------------------

namespace GQL {
namespace detail {

namespace {

void build_one_snapshot_for_test(
    ProjectionStorage& storage,
    Projection::TopologySnapshotWriter::Direction dir)
{
    using Projection::TopologySnapshotWriter;

    BPlusTree<3>* edge_bpt = (dir == TopologySnapshotWriter::Direction::FORWARD)
        ? storage.get_from_to_edge_index()
        : storage.get_to_from_edge_index();

    if (edge_bpt == nullptr) {
        throw std::runtime_error(
            "build_topology_snapshots_for_test: requested direction's "
            "B+Tree index is not open on this ProjectionStorage");
    }

    const uint64_t num_nodes = storage.get_node_count();
    const std::filesystem::path proj_dir = storage.get_projection_dir();

    // Mirror the production path's type-tag stripping so the test fixture
    // exercises the same code as the shipped builder.
    std::vector<uint64_t> degrees(num_nodes, 0);
    bool interrupt = false;
    Record<3> min_rec = {0, 0, 0};
    Record<3> max_rec = {UINT64_MAX, UINT64_MAX, UINT64_MAX};

    {
        auto iter = edge_bpt->get_range(&interrupt, min_rec, max_rec);
        const Record<3>* rec = nullptr;
        while ((rec = iter.next()) != nullptr) {
            const uint64_t src_idx = (*rec)[0] & ObjectId::VALUE_MASK;
            if (src_idx < num_nodes) {
                ++degrees[src_idx];
            }
        }
    }

    TopologySnapshotWriter writer(
        proj_dir,
        dir,
        num_nodes,
        std::move(degrees),
        /*include_edge_ids=*/true);

    {
        auto iter = edge_bpt->get_range(&interrupt, min_rec, max_rec);
        const Record<3>* rec = nullptr;
        while ((rec = iter.next()) != nullptr) {
            writer.append_edge(
                ObjectId{(*rec)[0] & ObjectId::VALUE_MASK},
                ObjectId{(*rec)[1]},
                ObjectId{(*rec)[2]});
        }
    }

    writer.finalize();
}

}  // namespace

void build_topology_snapshots_for_test(
    ProjectionStorage& storage,
    bool build_forward,
    bool build_reverse)
{
    using Projection::TopologySnapshotWriter;
    if (build_forward) {
        build_one_snapshot_for_test(storage, TopologySnapshotWriter::Direction::FORWARD);
    }
    if (build_reverse) {
        build_one_snapshot_for_test(storage, TopologySnapshotWriter::Direction::REVERSE);
    }
}

}  // namespace detail
}  // namespace GQL
