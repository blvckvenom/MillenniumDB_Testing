#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/gql_object_id.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "query/executor/binding_iter/aggregation/agg.h"
#include "query/executor/binding_iter/binding_expr/binding_expr_printer.h"
#include "query/parser/expr/gql/agg/expr_agg_project.h"  // For ProjectionOptions and DataConfig
#include "query/query_context.h"
#include "system/string_manager.h"

namespace GQL {
// Aggregate function that creates a graph projection from MATCH results
// Usage: MATCH (n)-[r]->(m) RETURN PROJECT('projection_name' [INCLUDE LABELS] [INCLUDE PROPERTIES])
class AggProject : public Agg {
public:
    AggProject(
        VarId var_id,
        std::unique_ptr<BindingExpr> projection_name_expr,
        ProjectionOptions options = ProjectionOptions(),
        DataConfig data_config = DataConfig()
    ) :
        Agg(var_id, std::move(projection_name_expr)),
        options(options),
        data_config(data_config)
    {
        // Validate mutually exclusive modes
        if (!data_config.is_empty() &&
            (options.include_labels || options.include_properties)) {
            throw std::runtime_error(
                "PROJECT: Cannot combine dataConfig with INCLUDE clauses. "
                "Use either:\n"
                "  Mode 1 (Implicit): PROJECT('name' INCLUDE LABELS INCLUDE PROPERTIES)\n"
                "  Mode 2 (Explicit): PROJECT('name', source, target, {sourceNodeProperties: ['age']})\n"
                "Do not mix both modes."
            );
        }
    }

    void begin() override
    {
        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject::begin() called" << std::endl;
        #endif

        // Reset state for new group
        projection_storage.reset();
        projection_name.clear();
        projection_name_oid = ObjectId(ObjectId::MASK_NULL);
        initialized = false;
    }

    void initialize_if_needed()
    {
        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject::initialize_if_needed() - initialized=" << initialized << std::endl;
        #endif

        if (initialized) {
            return;
        }

        // Evaluate the projection name expression to get the actual string
        // This should be a string literal in typical usage
        ObjectId name_oid = expr->eval(*binding);

        // Cache the ObjectId for later return
        projection_name_oid = name_oid;

        // DEBUG: Log the ObjectId for troubleshooting
        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject: Evaluated expression to ObjectId: 0x"
                  << std::hex << name_oid.id << std::dec << std::endl;
        #endif

        // Extract the string value using GQL's unpack_string function
        auto type = GQL_OID::get_type(name_oid);
        if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
            type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
            type == GQL_OID::Type::STRING_SIMPLE_TMP) {
            projection_name = Conversions::unpack_string(name_oid);
        } else {
            std::stringstream ss;
            ss << "PROJECT() requires a string literal as projection name. "
               << "Got type: " << static_cast<int>(type)
               << ", ObjectId: 0x" << std::hex << name_oid.id << std::dec;
            throw std::runtime_error(ss.str());
        }

        // Validate projection name is not empty
        if (projection_name.empty()) {
            std::stringstream ss;
            ss << "PROJECT() projection name cannot be empty. "
               << "ObjectId: 0x" << std::hex << name_oid.id << std::dec
               << " evaluated to empty string";
            throw std::runtime_error(ss.str());
        }

        // Create the projection directory
        auto& proj_manager = ProjectionManager::get_instance();
        std::string proj_dir = proj_manager.create_projection(projection_name);

        // Convert ProjectionOptions to ProjectionStorage::Features
        ProjectionStorage::Features features;
        features.include_node_labels = options.include_labels;
        features.include_edge_labels = options.include_labels;

        // For Mode 2 (explicit), enable property indexes if dataConfig specifies properties
        // For Mode 1 (implicit), use the include_properties option
        if (!data_config.is_empty()) {
            // Mode 2: Enable property indexes if any properties are specified
            features.include_node_properties =
                (data_config.source_node_properties && !data_config.source_node_properties->empty()) ||
                (data_config.target_node_properties && !data_config.target_node_properties->empty());
            features.include_edge_properties =
                (data_config.relationship_properties && !data_config.relationship_properties->empty());
        } else {
            // Mode 1: Use the option flags
            features.include_node_properties = options.include_properties;
            features.include_edge_properties = options.include_properties;
        }

        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject: Creating projection with features - labels: "
                  << options.include_labels << ", properties: " << options.include_properties << std::endl;
        #endif

        // Initialize the projection storage with projection name and features
        projection_storage = std::make_unique<ProjectionStorage>(
            proj_dir,
            proj_manager.get_db_folder(),
            projection_name,  // Pass projection name for catalog
            features          // Pass features for optional indexes
        );
        projection_storage->init();

        initialized = true;
    }

    void process() override
    {
        // Initialize on first process() call when binding has actual data
        initialize_if_needed();

        // Dispatch to appropriate processing mode
        if (data_config.is_empty()) {
            process_implicit();  // Backward compatible: implicit property detection
        } else {
            process_explicit();  // Selective properties: use DataConfig
        }
    }

    // Process with implicit property detection (original behavior)
    // Scans all variables and detects properties via naming convention (e.g., "n.age")
    void process_implicit()
    {

        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject::process() called, binding size: " << binding->size << std::endl;
        #endif

        if (!projection_storage) {
            throw std::runtime_error("ProjectionStorage not initialized in AggProject::process()");
        }

        // Maps to track nodes and edges with their properties
        std::map<ObjectId, std::unordered_map<std::string, ObjectId>> node_properties;
        std::map<ObjectId, std::unordered_map<std::string, ObjectId>> edge_properties;
        std::map<ObjectId, bool> edges_seen;
        std::map<ObjectId, std::pair<ObjectId, ObjectId>> edge_endpoints; // edge_id -> (from, to)
        std::vector<ObjectId> node_sequence; // Track order of nodes seen

        // First pass: collect nodes, edges, and identify property variables
        for (size_t i = 0; i < binding->size; i++) {
            VarId var_id(i);
            ObjectId oid = (*binding)[var_id];

            // Skip null/invalid values
            if (!oid.is_valid()) {
                continue;
            }

            auto var_name = get_query_ctx().get_var_name(var_id);
            auto type = GQL_OID::get_type(oid);

            // DEBUG: Print what we're processing
            std::cerr << "[DEBUG] Var " << i << " (" << var_name << "): OID=0x"
                      << std::hex << oid.id << std::dec
                      << " type=" << static_cast<int>(type) << std::endl;

            // Check if this is a property variable (contains '.')
            size_t dot_pos = var_name.find('.');
            if (dot_pos != std::string::npos) {
                // Extract parent variable and property name
                std::string parent_var = var_name.substr(0, dot_pos);
                std::string prop_name = var_name.substr(dot_pos + 1);

                // Find the parent variable's ObjectId
                bool found;
                VarId parent_var_id = get_query_ctx().get_var(parent_var, &found);
                if (found && parent_var_id.id < binding->size) {
                    ObjectId parent_oid = (*binding)[parent_var_id];
                    if (parent_oid.is_valid()) {
                        auto parent_type = GQL_OID::get_type(parent_oid);
                        if (parent_type == GQL_OID::Type::NODE) {
                            node_properties[parent_oid][prop_name] = oid;
                        } else if (parent_type == GQL_OID::Type::DIRECTED_EDGE ||
                                   parent_type == GQL_OID::Type::UNDIRECTED_EDGE) {
                            edge_properties[parent_oid][prop_name] = oid;
                        }
                    }
                }
            }
            // Handle nodes
            else if (type == GQL_OID::Type::NODE) {
                // Track node sequence for edge endpoint inference
                node_sequence.push_back(oid);

                // Mark that we've seen this node; will add with properties later
                if (node_properties.find(oid) == node_properties.end()) {
                    node_properties[oid] = std::unordered_map<std::string, ObjectId>();
                }
            }
            // Handle edges
            else if (type == GQL_OID::Type::DIRECTED_EDGE || type == GQL_OID::Type::UNDIRECTED_EDGE) {
                edges_seen[oid] = (type == GQL_OID::Type::DIRECTED_EDGE);
                if (edge_properties.find(oid) == edge_properties.end()) {
                    edge_properties[oid] = std::unordered_map<std::string, ObjectId>();
                }

                // Infer edge endpoints from adjacent nodes
                // Heuristic: if we have seen at least one node, use the last node as 'from'
                // and look ahead for the next node as 'to'
                ObjectId from_node = ObjectId(ObjectId::NULL_ID);
                ObjectId to_node = ObjectId(ObjectId::NULL_ID);

                if (!node_sequence.empty()) {
                    // Use the last node seen as the 'from' node
                    from_node = node_sequence.back();

                    // Look ahead for the next node
                    for (size_t j = i + 1; j < binding->size; j++) {
                        VarId next_var_id(j);
                        ObjectId next_oid = (*binding)[next_var_id];
                        if (next_oid.is_valid()) {
                            auto next_type = GQL_OID::get_type(next_oid);
                            if (next_type == GQL_OID::Type::NODE) {
                                to_node = next_oid;
                                break;
                            }
                        }
                    }
                }

                edge_endpoints[oid] = std::make_pair(from_node, to_node);
            }
        }

        // Second pass: add nodes with their properties
        for (const auto& [node_id, props] : node_properties) {
            ProjectedNode node;
            node.node_id = node_id;
            node.properties = props;
            projection_storage->add_node(node);
            #ifdef DEBUG_GQL_QUERY_VISITOR
            std::cerr << "  Added node: 0x" << std::hex << node_id.id << std::dec << std::endl;
            #endif
        }

        // Third pass: add edges with their properties
        std::cerr << "[AggProject] Adding " << edges_seen.size() << " edges to projection storage" << std::endl;
        for (const auto& [edge_id, is_directed] : edges_seen) {
            ProjectedEdge edge;
            edge.edge_id = edge_id;
            edge.is_directed = is_directed;

            // Add properties if any
            auto props_it = edge_properties.find(edge_id);
            if (props_it != edge_properties.end()) {
                edge.properties = props_it->second;
            }

            // Use inferred endpoints from adjacency heuristic
            auto endpoints_it = edge_endpoints.find(edge_id);
            if (endpoints_it != edge_endpoints.end()) {
                edge.from_node = endpoints_it->second.first;
                edge.to_node = endpoints_it->second.second;
            } else {
                // Fallback to NULL if no endpoints found
                edge.from_node = ObjectId(ObjectId::NULL_ID);
                edge.to_node = ObjectId(ObjectId::NULL_ID);
            }

            std::cerr << "[AggProject] Calling add_edge for edge 0x" << std::hex << edge_id.id << std::dec
                      << " (" << (is_directed ? "directed" : "undirected") << ")" << std::endl;
            projection_storage->add_edge(edge);
        }
        std::cerr << "[AggProject] Finished adding edges from MATCH results" << std::endl;

        // Fourth pass: extract labels from main graph if requested
        if (options.include_labels) {
            #ifdef DEBUG_GQL_QUERY_VISITOR
            std::cerr << "AggProject: Extracting labels from main graph" << std::endl;
            #endif

            // Extract node labels from main graph
            for (const auto& [node_id, props] : node_properties) {
                bool interruption = false;
                BptIter<2> it = gql_model.node_label
                                    ->get_range(&interruption, { node_id.id, 0 }, { node_id.id, UINT64_MAX });

                auto record = it.next();
                while (record != nullptr) {
                    ObjectId label_id((*record)[1]);
                    projection_storage->add_node_label(node_id, label_id);

                    #ifdef DEBUG_GQL_QUERY_VISITOR
                    std::cerr << "    Added node label: node=0x" << std::hex << node_id.id
                              << " label=0x" << label_id.id << std::dec << std::endl;
                    #endif

                    record = it.next();
                }
            }

            // Extract edge labels from main graph
            for (const auto& [edge_id, is_directed] : edges_seen) {
                bool interruption = false;
                BptIter<2> it = gql_model.edge_label
                                    ->get_range(&interruption, { edge_id.id, 0 }, { edge_id.id, UINT64_MAX });

                auto record = it.next();
                while (record != nullptr) {
                    ObjectId label_id((*record)[1]);
                    projection_storage->add_edge_label(edge_id, label_id);

                    #ifdef DEBUG_GQL_QUERY_VISITOR
                    std::cerr << "    Added edge label: edge=0x" << std::hex << edge_id.id
                              << " label=0x" << label_id.id << std::dec << std::endl;
                    #endif

                    record = it.next();
                }
            }

            #ifdef DEBUG_GQL_QUERY_VISITOR
            std::cerr << "AggProject: Label extraction complete" << std::endl;
            #endif
        }

        // Fifth pass: extract properties from main graph if requested
        if (options.include_properties) {
            std::cerr << "[AggProject] PROPERTY EXTRACTION STARTED - include_properties=" << options.include_properties << std::endl;
            std::cerr << "[AggProject] Number of nodes to extract properties for: " << node_properties.size() << std::endl;

            // Extract node properties from main graph
            for (const auto& [node_id, props] : node_properties) {
                std::cerr << "[AggProject] Extracting properties for node 0x" << std::hex << node_id.id << std::dec << std::endl;
                bool interruption = false;
                BptIter<3> it = gql_model.node_key_value
                                    ->get_range(&interruption, { node_id.id, 0, 0 }, { node_id.id, UINT64_MAX, UINT64_MAX });

                auto record = it.next();
                int prop_count = 0;
                while (record != nullptr) {
                    ObjectId key_id((*record)[1]);
                    ObjectId value_id((*record)[2]);
                    projection_storage->add_node_property(node_id, key_id, value_id);
                    prop_count++;

                    std::cerr << "[AggProject]     Added node property #" << prop_count << ": node=0x" << std::hex << node_id.id
                              << " key=0x" << key_id.id << " value=0x" << value_id.id << std::dec << std::endl;

                    record = it.next();
                }
                std::cerr << "[AggProject]   Total properties for node 0x" << std::hex << node_id.id << std::dec << ": " << prop_count << std::endl;
            }

            // Extract edge properties from main graph
            std::cerr << "[AggProject] Number of edges to extract properties for: " << edges_seen.size() << std::endl;
            for (const auto& [edge_id, is_directed] : edges_seen) {
                bool interruption = false;
                BptIter<3> it = gql_model.edge_key_value
                                    ->get_range(&interruption, { edge_id.id, 0, 0 }, { edge_id.id, UINT64_MAX, UINT64_MAX });

                auto record = it.next();
                int edge_prop_count = 0;
                while (record != nullptr) {
                    ObjectId key_id((*record)[1]);
                    ObjectId value_id((*record)[2]);
                    projection_storage->add_edge_property(edge_id, key_id, value_id);
                    edge_prop_count++;

                    std::cerr << "[AggProject]     Added edge property #" << edge_prop_count << ": edge=0x" << std::hex << edge_id.id
                              << " key=0x" << key_id.id << " value=0x" << value_id.id << std::dec << std::endl;

                    record = it.next();
                }
                std::cerr << "[AggProject]   Total properties for edge 0x" << std::hex << edge_id.id << std::dec << ": " << edge_prop_count << std::endl;
            }

            std::cerr << "[AggProject] PROPERTY EXTRACTION COMPLETE" << std::endl;
        }

        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject::process_implicit() complete - nodes: " << node_properties.size()
                  << ", edges: " << edges_seen.size() << std::endl;
        #endif
    }

    // Process with explicit selective property inclusion (new behavior)
    // Uses data_config to include only specified properties
    void process_explicit()
    {
        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject::process_explicit() called, binding size: " << binding->size << std::endl;
        #endif

        if (!projection_storage) {
            throw std::runtime_error("ProjectionStorage not initialized in AggProject::process_explicit()");
        }

        // Maps to track nodes and edges with their properties
        std::map<ObjectId, std::unordered_map<std::string, ObjectId>> node_properties;
        std::map<ObjectId, std::unordered_map<std::string, ObjectId>> edge_properties;
        std::map<ObjectId, bool> edges_seen;
        std::map<ObjectId, std::pair<ObjectId, ObjectId>> edge_endpoints;
        std::vector<ObjectId> node_sequence;

        // First pass: collect nodes and edges from bindings (same as implicit)
        for (size_t i = 0; i < binding->size; i++) {
            VarId var_id(i);
            ObjectId oid = (*binding)[var_id];

            if (!oid.is_valid()) {
                continue;
            }

            auto var_name = get_query_ctx().get_var_name(var_id);
            auto type = GQL_OID::get_type(oid);

            // Handle nodes
            if (type == GQL_OID::Type::NODE) {
                node_sequence.push_back(oid);
                if (node_properties.find(oid) == node_properties.end()) {
                    node_properties[oid] = std::unordered_map<std::string, ObjectId>();
                }
            }
            // Handle edges
            else if (type == GQL_OID::Type::DIRECTED_EDGE || type == GQL_OID::Type::UNDIRECTED_EDGE) {
                edges_seen[oid] = (type == GQL_OID::Type::DIRECTED_EDGE);
                if (edge_properties.find(oid) == edge_properties.end()) {
                    edge_properties[oid] = std::unordered_map<std::string, ObjectId>();
                }

                // Infer edge endpoints from adjacent nodes (same heuristic as implicit)
                ObjectId from_node = ObjectId(ObjectId::NULL_ID);
                ObjectId to_node = ObjectId(ObjectId::NULL_ID);

                if (!node_sequence.empty()) {
                    from_node = node_sequence.back();
                    for (size_t j = i + 1; j < binding->size; j++) {
                        VarId next_var_id(j);
                        ObjectId next_oid = (*binding)[next_var_id];
                        if (next_oid.is_valid()) {
                            auto next_type = GQL_OID::get_type(next_oid);
                            if (next_type == GQL_OID::Type::NODE) {
                                to_node = next_oid;
                                break;
                            }
                        }
                    }
                }
                edge_endpoints[oid] = std::make_pair(from_node, to_node);
            }
        }

        // Second pass: add nodes (without properties yet)
        for (const auto& [node_id, props] : node_properties) {
            ProjectedNode node;
            node.node_id = node_id;
            // Properties will be added in fourth pass
            projection_storage->add_node(node);
            #ifdef DEBUG_GQL_QUERY_VISITOR
            std::cerr << "  Added node: 0x" << std::hex << node_id.id << std::dec << std::endl;
            #endif
        }

        // Third pass: add edges (without properties yet)
        std::cerr << "[AggProject::explicit] Adding " << edges_seen.size() << " edges to projection storage" << std::endl;
        for (const auto& [edge_id, is_directed] : edges_seen) {
            ProjectedEdge edge;
            edge.edge_id = edge_id;
            edge.is_directed = is_directed;

            // Use inferred endpoints
            auto endpoints_it = edge_endpoints.find(edge_id);
            if (endpoints_it != edge_endpoints.end()) {
                edge.from_node = endpoints_it->second.first;
                edge.to_node = endpoints_it->second.second;
            } else {
                edge.from_node = ObjectId(ObjectId::NULL_ID);
                edge.to_node = ObjectId(ObjectId::NULL_ID);
            }

            projection_storage->add_edge(edge);
        }

        // Fourth pass: extract SELECTIVE properties from main graph
        // Combine source and target node properties (simplified approach)
        std::set<std::string> node_props_to_extract;
        if (data_config.source_node_properties) {
            for (const auto& prop : *data_config.source_node_properties) {
                node_props_to_extract.insert(prop);
            }
        }
        if (data_config.target_node_properties) {
            for (const auto& prop : *data_config.target_node_properties) {
                node_props_to_extract.insert(prop);
            }
        }

        // Extract specified node properties from main graph
        if (!node_props_to_extract.empty()) {
            std::cerr << "[AggProject::explicit] Extracting " << node_props_to_extract.size()
                      << " selective node properties from main graph" << std::endl;

            for (const auto& [node_id, props] : node_properties) {
                for (const std::string& prop_name : node_props_to_extract) {
                    // Convert property name to ObjectId
                    ObjectId key_id = Conversions::pack_node_property(prop_name);

                    // Query main graph for this specific property
                    // node_key_value is indexed: {node_id, key_id, value_id}
                    bool interruption = false;
                    BptIter<3> it = gql_model.node_key_value->get_range(
                        &interruption,
                        { node_id.id, key_id.id, 0 },
                        { node_id.id, key_id.id, UINT64_MAX }
                    );

                    auto record = it.next();
                    if (record != nullptr) {
                        ObjectId value_id((*record)[2]);
                        projection_storage->add_node_property(node_id, key_id, value_id);

                        std::cerr << "[AggProject::explicit] Added node property: node=0x" << std::hex
                                  << node_id.id << " key=" << prop_name << " value=0x"
                                  << value_id.id << std::dec << std::endl;
                    } else {
                        // Property not found - skip gracefully (no error)
                        std::cerr << "[AggProject::explicit] Property '" << prop_name
                                  << "' not found for node 0x" << std::hex << node_id.id
                                  << std::dec << " - skipping" << std::endl;
                    }
                }
            }
        }

        // Extract specified relationship properties from main graph
        if (data_config.relationship_properties) {
            std::cerr << "[AggProject::explicit] Extracting " << data_config.relationship_properties->size()
                      << " selective relationship properties from main graph" << std::endl;

            for (const auto& [edge_id, is_directed] : edges_seen) {
                for (const std::string& prop_name : *data_config.relationship_properties) {
                    // Convert property name to ObjectId
                    ObjectId key_id = Conversions::pack_edge_property(prop_name);

                    // Query main graph for this specific property
                    // edge_key_value is indexed: {edge_id, key_id, value_id}
                    bool interruption = false;
                    BptIter<3> it = gql_model.edge_key_value->get_range(
                        &interruption,
                        { edge_id.id, key_id.id, 0 },
                        { edge_id.id, key_id.id, UINT64_MAX }
                    );

                    auto record = it.next();
                    if (record != nullptr) {
                        ObjectId value_id((*record)[2]);
                        projection_storage->add_edge_property(edge_id, key_id, value_id);

                        std::cerr << "[AggProject::explicit] Added edge property: edge=0x" << std::hex
                                  << edge_id.id << " key=" << prop_name << " value=0x"
                                  << value_id.id << std::dec << std::endl;
                    } else {
                        // Property not found - skip gracefully (no error)
                        std::cerr << "[AggProject::explicit] Property '" << prop_name
                                  << "' not found for edge 0x" << std::hex << edge_id.id
                                  << std::dec << " - skipping" << std::endl;
                    }
                }
            }
        }

        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject::process_explicit() complete - nodes: " << node_properties.size()
                  << ", edges: " << edges_seen.size() << std::endl;
        #endif
    }

    // Called at the end of aggregation to get the result
    ObjectId get() override
    {
        // Initialize if needed (handles case of empty result set)
        initialize_if_needed();

        if (!projection_storage) {
            throw std::runtime_error("ProjectionStorage not initialized in AggProject::get()");
        }

        // Flush any pending writes
        projection_storage->flush();

        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject::get() - flushed projection storage" << std::endl;
        #endif

        // Refresh ProjectionManager cache so the new projection is immediately visible
        // This allows list-projections and USE queries to find it without server restart
        auto& proj_manager = ProjectionManager::get_instance();
        proj_manager.scan_projections();

        #ifdef DEBUG_GQL_QUERY_VISITOR
        std::cerr << "AggProject::get() - refreshed projection manager cache" << std::endl;
        #endif

        // Return the cached projection name ObjectId
        return projection_name_oid;
    }

    std::ostream& print(std::ostream& os) const override
    {
        os << "PROJECT(";
        BindingExprPrinter printer(os);
        printer.print(*expr);
        os << ")";
        return os;
    }

private:
    std::unique_ptr<ProjectionStorage> projection_storage;
    std::string projection_name;
    ObjectId projection_name_oid;
    bool initialized = false;
    ProjectionOptions options;  // Options for what to include in projection
    DataConfig data_config;     // Configuration for selective properties
};
} // namespace GQL
