#include "native_projection_builder.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/edge_filter.h"
#include "graph_models/gql/projection/external_edge_sort.h"
#include "graph_models/gql/projection/native_scanner.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/streaming_aggregator.h"
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
// ScanMode selector (Spec #2): MDB_PROJECTION_SERIAL_SCAN env-var parsing.
//
// Parallel to MDB_PROJECTION_SORTER from Spec #1. The production path
// (get_scan_mode) caches the result for the process lifetime via a C++11
// magic static so downstream callers pay zero overhead after the first
// invocation.
//
// The detail::init_scan_mode_for_test helper exposes the same parse rules
// without the cache so unit tests can cover truthy / unknown / null inputs
// deterministically.
// ============================================================================
namespace {
NativeProjectionBuilder::ScanMode init_scan_mode() {
    const char* env = std::getenv("MDB_PROJECTION_SERIAL_SCAN");
    if (env == nullptr) return NativeProjectionBuilder::ScanMode::CLASSIC;
    std::string v(env);
    if (v == "1" || v == "true" || v == "yes") {
        return NativeProjectionBuilder::ScanMode::SERIALIZED;
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
    bool include_label_indexes
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
    features.include_node_properties = !node_property_keys.empty();
    // Enable edge properties if explicitly requested OR if COUNT aggregation will create _count
    bool has_count_aggregation = (aggregation == Aggregation::COUNT);
    for (const auto& [type, agg] : type_aggregations) {
        if (agg == Aggregation::COUNT) {
            has_count_aggregation = true;
            break;
        }
    }
    features.include_edge_properties = !edge_property_keys.empty() || has_count_aggregation;
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
            // they want to collapse parallels. Pick the strategy that matches
            // your intent:
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
        // Defer actual scan: store inputs for finalize_serialized_ (Task 10)
        // to replay via the 14-pass pipeline.
        stored_labels_ = labels;
        scan_inputs_captured_ = true;
        return;
    }
    scan_nodes_impl_classic_(labels);
}

void NativeProjectionBuilder::scan_edges_by_types(const std::vector<std::string>& types) {
    if (get_scan_mode() == ScanMode::SERIALIZED) {
        // Defer actual scan: store inputs for finalize_serialized_ (Task 10)
        // to replay via the 14-pass pipeline.
        stored_types_ = types;
        scan_inputs_captured_ = true;
        return;
    }
    scan_edges_impl_classic_(types);
}

void NativeProjectionBuilder::scan_nodes_impl_classic_(const std::vector<std::string>& labels) {
    auto bench_t0 = benchmark_timers_.enabled ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};

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
            std::chrono::high_resolution_clock::now() - bench_t0).count();
    }
}

void NativeProjectionBuilder::scan_edges_impl_classic_(const std::vector<std::string>& types) {
    auto bench_t0 = benchmark_timers_.enabled ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};

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

    for (const auto& type : types) {
        // Get per-type configuration (falls back to global defaults)
        Orientation type_orientation = get_orientation_for_type(type);
        Aggregation type_aggregation = get_aggregation_for_type(type);
        std::string type_agg_property = get_aggregation_property_for_type(type);

        // Get type_id from pre-built map
        ObjectId type_id = type_id_map[type];

        // Create detector for this type with type-specific aggregation
        auto detector = std::make_unique<ParallelEdgeDetector>(type_aggregation);

        // Scan all edges with this type (OPTIMIZED: get endpoints in single pass)
        scanner->scan_label_edge_with_endpoints(
            type_id,
            [this, &detector, type_id, type_orientation,
             type_aggregation, &type_agg_property]
            (ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
            // Filter: only include if both endpoints are in projection
            bool has_from = storage->has_node(from_node);
            bool has_to = storage->has_node(to_node);

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
            bool is_first_occurrence = detector->process_edge(
                from_node.id,
                to_node.id,
                type_id.id,
                edge_id,
                property_value
            );

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

            // Store aggregated value as property on each representative edge
            for (const auto& [edge_id_raw, agg_value] : aggregated_values) {
                ObjectId edge_id(edge_id_raw);
                ObjectId value_oid = Common::Conversions::pack_int(static_cast<int64_t>(agg_value));

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

    if (benchmark_timers_.enabled) {
        benchmark_timers_.edge_scan_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - bench_t0).count();
    }
}

NativeProjectionBuilder::Statistics NativeProjectionBuilder::finalize() {
    auto bench_total_start = benchmark_timers_.enabled ? std::chrono::high_resolution_clock::now()
                                                       : std::chrono::high_resolution_clock::time_point{};

    finalized_ = true;

    // Final flush to ensure all data is written
    if (!node_batch.empty()) {
        flush_nodes();
    }
    if (!edge_batch.empty()) {
        flush_edges();
    }

    // Final flush to ProjectionStorage (commits all B+Tree writes, builds indexes)
    auto bench_sort_start = benchmark_timers_.enabled ? std::chrono::high_resolution_clock::now()
                                                      : std::chrono::high_resolution_clock::time_point{};
    storage->flush();
    if (benchmark_timers_.enabled) {
        auto bench_sort_end = std::chrono::high_resolution_clock::now();
        double sort_btree_ms = std::chrono::duration<double, std::milli>(
            bench_sort_end - bench_sort_start).count();
        // Attribute combined sort+btree time; detailed breakdown deferred to future pass
        benchmark_timers_.sort_ms += sort_btree_ms * 0.5;
        benchmark_timers_.btree_write_ms += sort_btree_ms * 0.5;
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
    if (!include_features_.empty() && gnn_row_mapping_) {
        namespace fs = std::filesystem;
        auto proj_dir = fs::path(db_folder) / "projections" / projection_name;

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

    // Refresh projection cache so new projection is immediately visible
    auto bench_meta_start = benchmark_timers_.enabled ? std::chrono::high_resolution_clock::now()
                                                      : std::chrono::high_resolution_clock::time_point{};
    ProjectionManager::get_instance().scan_projections();
    if (benchmark_timers_.enabled) {
        benchmark_timers_.metadata_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - bench_meta_start).count();
    }

    if (benchmark_timers_.enabled) {
        auto bench_total_end = std::chrono::high_resolution_clock::now();
        benchmark_timers_.total_ms = std::chrono::duration<double, std::milli>(
            bench_total_end - bench_total_start).count();
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
    for (const auto& edge : edge_batch) {
        storage->add_edge(edge);
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
    constexpr size_t NODE_CACHE_SIZE = 10000;
    std::unordered_map<uint64_t, bool> node_in_projection_cache;
    node_in_projection_cache.reserve(NODE_CACHE_SIZE);

    // Helper lambda to check if node is in projection with caching
    auto cached_has_node = [this, &node_in_projection_cache](ObjectId node_id) -> bool {
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
// Serialized scan implementations (Spec #2).
//
// Serialized equivalents of scan_nodes_impl_classic_ / scan_edges_impl_classic_
// that gate each storage emission on a target ProjectionIndex bitmask. These
// are invoked once per node-related index (5 passes) by finalize_serialized_
// (Task 10), so the main scan loop runs five times over the same labels.
//
// Critical invariant (see spec §6, I1): finalize_node_scan() MUST be called
// EXACTLY ONCE across all node-phase passes — specifically during the NODES
// pass — because it populates ProjectionStorage::collected_nodes_ which the
// edge filter's has_node() (Phase B) depends on. If we finalized on every
// pass, subsequent passes would mutate the node set the edge filter has
// already consulted.
//
// TODO(task10-gnn): The emit_properties gate here only consults the
// target_mask. In GNN-only configs (includeFeatures + labelProperty +
// splitProperty with no explicit nodeProperties), classic's
// extract_node_properties is called for its side effect of populating
// labels_buffer_ / splits_buffer_ via try_extract_gnn_property, even
// though no property records get emitted. For that to work under
// SERIALIZED mode, Task 10's enabled_indexes_() MUST push
// NODE_KEY_VALUE / KEY_VALUE_NODE into the pass list when
// gnn_row_mapping_ != nullptr, regardless of
// features.include_node_properties. Otherwise GNN training with
// SERIAL_SCAN=1 trains on unlabeled data.
// ============================================================================

void NativeProjectionBuilder::scan_nodes_impl_serialized_(
    const std::vector<std::string>& labels,
    ProjectionIndex target_mask)
{
    auto bench_t0 = benchmark_timers_.enabled ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};

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
            std::chrono::high_resolution_clock::now() - bench_t0).count();
    }
}

// ============================================================================
// Serialized scan Phase B: precompute_edge_filter_ (Spec #2 §4 Phase B).
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
// Invariant I2 (spec §6): filter is finalized before return, so Phase C's
// consumers see an immutable snapshot and can read concurrently without
// synchronization if they later go parallel.
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
    for (const auto& type : types) {
        ObjectId type_id = type_id_map[type];
        scanner->scan_label_edge_with_endpoints(
            type_id,
            [this, &filter](ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
                const bool has_from = storage->has_node(from_node);
                const bool has_to   = storage->has_node(to_node);
                if (has_from && has_to) {
                    // Pass the FULL ObjectId; EdgeFilter routes by the
                    // top-byte type tag and keys by the 56-bit counter.
                    filter->set_kept(edge_id);
                }
            });
    }

    filter->finalize();
    return filter;
}

// ============================================================================
// Serialized scan Phase C: scan_edges_impl_serialized_ (Spec #2 §4 Phase C).
//
// Mirror of scan_edges_impl_classic_ for the SERIALIZED pipeline. Task 10's
// orchestrator (finalize_serialized_) calls this once per edge-related index
// — typically with a single-bit target_mask — so each B+Tree pass does one
// sequential scan of the label_edge index without redundantly re-running
// ParallelEdgeDetector or has_node() lookups.
//
// The EdgeFilter produced by precompute_edge_filter_ (Phase B) is consumed
// here read-only: filter->is_kept(edge_id) is an O(1) bitmap probe that
// replaces the classic path's per-edge has_node(from) + has_node(to) pair.
//
// Aggregation modes (SUM/MIN/MAX/COUNT) are NOT handled here: Spec §3 D8
// keeps classic as the gate-keeping path for aggregation; finalize_serialized_
// routes those graphs to finalize_classic_ before this code is ever reached.
// Only SINGLE mode reaches this function.
//
// Invariant I1 (spec §6): Phase B must complete before any Phase C call.
//   → Enforced by the nullptr guard below (Task 10's orchestrator holds the
//     unique_ptr and passes a raw pointer here only after Phase B returns).
// Invariant I2 (spec §6): filter is immutable after finalize() — safe to read
//   concurrently if Task 10 later parallelises the per-index passes.
// ============================================================================
void NativeProjectionBuilder::scan_edges_impl_serialized_(
    const std::vector<std::string>& types,
    ProjectionIndex target_mask,
    const EdgeFilter* filter)
{
    auto bench_t0 = benchmark_timers_.enabled ? std::chrono::high_resolution_clock::now()
                                              : std::chrono::high_resolution_clock::time_point{};

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
    // label buffer write (or property extraction) runs. Task 10's
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

    std::unordered_map<std::string, ObjectId> type_id_map;
    for (const auto& type : types) {
        auto it = gql_model.catalog.edge_labels2id.find(type);
        if (it == gql_model.catalog.edge_labels2id.end()) {
            throw std::runtime_error("Type '" + type + "' not found in catalog");
        }
        type_id_map[type] = ObjectId(it->second | ObjectId::MASK_EDGE_LABEL);
    }

    for (const auto& type : types) {
        Orientation type_orientation = get_orientation_for_type(type);
        ObjectId type_id = type_id_map[type];

        scanner->scan_label_edge_with_endpoints(type_id,
            [this, filter, type_id, type_orientation,
             emit_any_edge_buffer, emit_edge_label, emit_edge_properties]
            (ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
            if (!filter->is_kept(edge_id)) return;  // O(1) bitmap lookup

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

            if (emit_edge_properties && !edge_property_keys.empty()) {
                extract_edge_properties(edge_id);
            }

            if (emit_edge_label) {
                storage->add_edge_label(edge_id, type_id);
            }

            if (emit_any_edge_buffer && edge_batch.size() >= BATCH_SIZE) {
                flush_edges();
            }
        });
    }

    if (emit_any_edge_buffer && !edge_batch.empty()) {
        flush_edges();
    }

    if (benchmark_timers_.enabled) {
        benchmark_timers_.edge_scan_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - bench_t0).count();
    }
}
