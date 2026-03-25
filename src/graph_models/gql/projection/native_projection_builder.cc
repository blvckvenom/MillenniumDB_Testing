#include "native_projection_builder.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/external_edge_sort.h"
#include "graph_models/gql/projection/native_scanner.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/streaming_aggregator.h"
#include "storage/index/bplus_tree/bplus_tree.h"

using namespace GQL;

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
    const std::unordered_map<std::string, PropertyConfig>& edge_property_configs
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
    features.include_node_labels = true;  // Always include node labels (automatic, like Neo4j GDS)
    features.include_edge_labels = true;  // Always include edge labels (automatic, like Neo4j GDS)

    // Initialize projection storage with features
    storage = std::make_unique<ProjectionStorage>(proj_dir, db_folder, projection_name, features);
    storage->init();

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
            // Fail on duplicate edges (strict validation)
            throw std::runtime_error(
                "Parallel edges detected with aggregation strategy SINGLE. "
                "Use aggregation: 'MIN', 'MAX', 'SUM', or 'COUNT' to handle parallel edges."
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

            // Extract properties if configured
            if (!node_property_keys.empty()) {
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

    // NOTE: With bulk import, we don't flush here anymore.
    // has_node() now checks the inserted_nodes hash set first,
    // so it works during the collection phase before B+tree is built.
}

void NativeProjectionBuilder::scan_edges_by_types(const std::vector<std::string>& types) {
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
        scanner->scan_label_edge_with_endpoints(type_id, [this, &detector, type_id, type_orientation, type_aggregation, &type_agg_property](ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
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
                    std::cerr << "[Builder] Warning: Property key '" << property_key_name
                              << "' not found in catalog. Aggregated values may not be stored correctly." << std::endl;
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
}

NativeProjectionBuilder::Statistics NativeProjectionBuilder::finalize() {
    finalized_ = true;

    // Final flush to ensure all data is written
    if (!node_batch.empty()) {
        flush_nodes();
    }
    if (!edge_batch.empty()) {
        flush_edges();
    }

    // Final flush to ProjectionStorage (commits all B+Tree writes)
    storage->flush();

    // Calculate duration
    auto end_time = std::chrono::steady_clock::now();
    stats.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );

    // Get final statistics from storage
    stats.node_count = storage->get_node_count();
    stats.relationship_count = storage->get_edge_count();

    // Refresh projection cache so new projection is immediately visible
    ProjectionManager::get_instance().scan_projections();

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
        throw std::runtime_error(
            "Node label '" + label + "' does not exist in the database.\n"
            "Hint: Labels are case-sensitive."
        );
    }
}

void NativeProjectionBuilder::validate_type_exists(const std::string& type) {
    if (gql_model.catalog.edge_labels2id.find(type) == gql_model.catalog.edge_labels2id.end()) {
        throw std::runtime_error(
            "Relationship type '" + type + "' does not exist in the database.\n"
            "Hint: Types are case-sensitive."
        );
    }
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

            // Check if any property config uses this source property
            for (const auto& [projected_name, config] : node_prop_configs) {
                std::string config_source = config.source_property.empty() ? projected_name : config.source_property;

                if (config_source == source_key_name) {
                    // Get or create projected key ObjectId
                    ObjectId projected_key_id = key_id;  // Default: same as source

                    if (!config.source_property.empty() && config.source_property != projected_name) {
                        // Renaming: need to create/lookup new key
                        auto proj_key_it = gql_model.catalog.node_keys2id.find(projected_name);
                        if (proj_key_it != gql_model.catalog.node_keys2id.end()) {
                            projected_key_id = ObjectId(proj_key_it->second | ObjectId::MASK_NODE_KEY);
                        }
                        // If projected key doesn't exist in catalog, use source key
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
            if (found_properties.find(projected_name) == found_properties.end() && config.default_value.has_value()) {
                // Property not found - apply default value
                ObjectId projected_key_id(0);
                auto proj_key_it = gql_model.catalog.node_keys2id.find(projected_name);
                if (proj_key_it != gql_model.catalog.node_keys2id.end()) {
                    projected_key_id = ObjectId(proj_key_it->second | ObjectId::MASK_NODE_KEY);
                } else {
                    // Try source property name
                    std::string source_name = config.source_property.empty() ? projected_name : config.source_property;
                    auto src_key_it = gql_model.catalog.node_keys2id.find(source_name);
                    if (src_key_it != gql_model.catalog.node_keys2id.end()) {
                        projected_key_id = ObjectId(src_key_it->second | ObjectId::MASK_NODE_KEY);
                    }
                }

                if (projected_key_id.id != 0) {
                    // Create ObjectId from default value (double -> int64 for storage)
                    ObjectId default_value_id = Common::Conversions::pack_double(config.default_value.value());
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

        // If we have a filter list, check if this property is in it
        if (!node_property_keys.empty()) {
            std::string key_name;
            auto nk_it = node_key_id_to_name.find(key_id.id);
            if (nk_it != node_key_id_to_name.end()) {
                key_name = nk_it->second;
            }

            if (std::find(node_property_keys.begin(), node_property_keys.end(), key_name) == node_property_keys.end()) {
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
                std::string config_source = config.source_property.empty() ? projected_name : config.source_property;

                if (config_source == source_key_name) {
                    // Get or create projected key ObjectId
                    ObjectId projected_key_id = key_id;  // Default: same as source

                    if (!config.source_property.empty() && config.source_property != projected_name) {
                        // Renaming: need to create/lookup new key
                        auto proj_key_it = gql_model.catalog.edge_keys2id.find(projected_name);
                        if (proj_key_it != gql_model.catalog.edge_keys2id.end()) {
                            projected_key_id = ObjectId(proj_key_it->second | ObjectId::MASK_EDGE_KEY);
                        }
                        // If projected key doesn't exist in catalog, use source key
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
            if (found_properties.find(projected_name) == found_properties.end() && config.default_value.has_value()) {
                // Property not found - apply default value
                ObjectId projected_key_id(0);
                auto proj_key_it = gql_model.catalog.edge_keys2id.find(projected_name);
                if (proj_key_it != gql_model.catalog.edge_keys2id.end()) {
                    projected_key_id = ObjectId(proj_key_it->second | ObjectId::MASK_EDGE_KEY);
                } else {
                    // Try source property name
                    std::string source_name = config.source_property.empty() ? projected_name : config.source_property;
                    auto src_key_it = gql_model.catalog.edge_keys2id.find(source_name);
                    if (src_key_it != gql_model.catalog.edge_keys2id.end()) {
                        projected_key_id = ObjectId(src_key_it->second | ObjectId::MASK_EDGE_KEY);
                    }
                }

                if (projected_key_id.id != 0) {
                    // Create ObjectId from default value (double -> int64 for storage)
                    ObjectId default_value_id = Common::Conversions::pack_double(config.default_value.value());
                    storage->add_edge_property(edge_id, projected_key_id, default_value_id);
                }
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

            if (std::find(edge_property_keys.begin(), edge_property_keys.end(), key_name) == edge_property_keys.end()) {
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
                std::string config_source = config.source_property.empty() ? projected_name : config.source_property;

                if (config_source == source_key_name) {
                    ObjectId projected_key_id = key_id;

                    if (!config.source_property.empty() && config.source_property != projected_name) {
                        auto proj_key_it = gql_model.catalog.edge_keys2id.find(projected_name);
                        if (proj_key_it != gql_model.catalog.edge_keys2id.end()) {
                            projected_key_id = ObjectId(proj_key_it->second | ObjectId::MASK_EDGE_KEY);
                        }
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
            std::string config_source = config.source_property.empty() ? projected_name : config.source_property;
            if (config_source == exclude_property) {
                continue;
            }

            if (found_properties.find(projected_name) == found_properties.end() && config.default_value.has_value()) {
                ObjectId projected_key_id(0);
                auto proj_key_it = gql_model.catalog.edge_keys2id.find(projected_name);
                if (proj_key_it != gql_model.catalog.edge_keys2id.end()) {
                    projected_key_id = ObjectId(proj_key_it->second | ObjectId::MASK_EDGE_KEY);
                } else {
                    std::string source_name = config.source_property.empty() ? projected_name : config.source_property;
                    auto src_key_it = gql_model.catalog.edge_keys2id.find(source_name);
                    if (src_key_it != gql_model.catalog.edge_keys2id.end()) {
                        projected_key_id = ObjectId(src_key_it->second | ObjectId::MASK_EDGE_KEY);
                    }
                }

                if (projected_key_id.id != 0) {
                    ObjectId default_value_id = Common::Conversions::pack_double(config.default_value.value());
                    storage->add_edge_property(edge_id, projected_key_id, default_value_id);
                }
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

            if (std::find(edge_property_keys.begin(), edge_property_keys.end(), key_name) == edge_property_keys.end()) {
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

                // Apply orientation to determine storage order
                uint64_t store_from, store_to;
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

    ExternalEdgeSort sorter(temp_dir, 256 * 1024 * 1024);

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
