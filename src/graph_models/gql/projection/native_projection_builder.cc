#include "native_projection_builder.h"

#include <iostream>
#include <stdexcept>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/native_scanner.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "storage/index/bplus_tree/bplus_tree.h"

using namespace GQL;

NativeProjectionBuilder::NativeProjectionBuilder(
    const std::string& projection_name_,
    const std::string& db_folder_,
    const std::vector<std::string>& node_properties,
    const std::vector<std::string>& edge_properties
)
    : projection_name(projection_name_)
    , db_folder(db_folder_)
    , node_property_keys(node_properties)
    , edge_property_keys(edge_properties)
    , start_time(std::chrono::steady_clock::now())
{
    // Create projection directory
    auto& manager = ProjectionManager::get_instance();
    std::string proj_dir = manager.create_projection(projection_name);

    // Configure features based on property lists
    ProjectionStorage::Features features;
    features.include_node_properties = !node_property_keys.empty();
    features.include_edge_properties = !edge_property_keys.empty();

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
}

NativeProjectionBuilder::~NativeProjectionBuilder() {
    // RAII cleanup - unique_ptrs automatically destroy resources
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
        scanner->scan_label_node(label_id, [this](ObjectId node_id) {
            // Add to batch
            ProjectedNode node;
            node.node_id = node_id;
            node_batch.push_back(node);

            // Extract properties if configured
            if (!node_property_keys.empty()) {
                extract_node_properties(node_id);
            }

            // Auto-flush when batch is full
            if (node_batch.size() >= BATCH_SIZE) {
                flush_nodes();
            }
        });
    }

    // Flush any remaining nodes
    if (!node_batch.empty()) {
        flush_nodes();
    }

    // CRITICAL: Flush to B+Tree so has_node() works during edge scanning
    storage->flush();
    std::cerr << "[Builder] Flushed " << storage->get_node_count() << " nodes to B+Tree" << std::endl;
}

void NativeProjectionBuilder::scan_edges_by_types(const std::vector<std::string>& types) {
    for (const auto& type : types) {
        validate_type_exists(type);

        // Convert type string to ObjectId via catalog lookup
        auto it = gql_model.catalog.edge_labels2id.find(type);
        if (it == gql_model.catalog.edge_labels2id.end()) {
            throw std::runtime_error(
                "Type '" + type + "' not found in catalog"
            );
        }
        ObjectId type_id(it->second | ObjectId::MASK_EDGE_LABEL);

        // Scan all edges with this type (OPTIMIZED: get endpoints in single pass)
        scanner->scan_label_edge_with_endpoints(type_id, [this](ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
            // Filter: only include if both endpoints are in projection
            bool has_from = storage->has_node(from_node);
            bool has_to = storage->has_node(to_node);

            if (!has_from || !has_to) {
                std::cerr << "[Builder] Filtering out edge: from=0x" << std::hex << from_node.id
                          << " (has=" << has_from << ") to=0x" << to_node.id
                          << " (has=" << has_to << ")" << std::dec << std::endl;
                return; // Skip this edge
            }

            std::cerr << "[Builder] Including edge: from=0x" << std::hex << from_node.id
                      << " to=0x" << to_node.id << std::dec << std::endl;

            // Add to batch
            ProjectedEdge edge;
            edge.from_node = from_node;
            edge.to_node = to_node;
            edge.edge_id = edge_id;
            // Detect directionality from ObjectId type mask
            uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
            edge.is_directed = (edge_type != ObjectId::MASK_UNDIRECTED_EDGE);
            edge_batch.push_back(edge);

            // Extract properties if configured
            if (!edge_property_keys.empty()) {
                extract_edge_properties(edge_id);
            }

            // Auto-flush when batch is full
            if (edge_batch.size() >= BATCH_SIZE) {
                flush_edges();
            }
        });
    }

    // Flush any remaining edges
    if (!edge_batch.empty()) {
        flush_edges();
    }
}

NativeProjectionBuilder::Statistics NativeProjectionBuilder::finalize() {
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
    // Use proven pattern from AggProject (lines 367-388)
    // Scan all properties for this node from main graph's node_key_value index
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

        // Add property to projection storage
        storage->add_node_property(node_id, key_id, value_id);

        record = it.next();
    }
}

void NativeProjectionBuilder::extract_edge_properties(ObjectId edge_id) {
    // Use proven pattern from AggProject (lines 390-411)
    // Scan all properties for this edge from main graph's edge_key_value index
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

        // Add property to projection storage
        storage->add_edge_property(edge_id, key_id, value_id);

        record = it.next();
    }
}
