#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "graph_models/gql/gql_object_id.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "query/executor/binding_iter/aggregation/agg.h"
#include "query/executor/binding_iter/binding_expr/binding_expr_printer.h"
#include "query/query_context.h"
#include "system/string_manager.h"

namespace GQL {
// Aggregate function that creates a graph projection from MATCH results
// Usage: MATCH (n)-[r]->(m) RETURN PROJECT('projection_name')
class AggProject : public Agg {
public:
    AggProject(VarId var_id, std::unique_ptr<BindingExpr> projection_name_expr) :
        Agg(var_id, std::move(projection_name_expr))
    { }

    void begin() override
    {
        // Reset state for new group
        projection_storage.reset();
        projection_name.clear();

        // Evaluate the projection name expression to get the actual string
        // This should be a string literal in typical usage
        ObjectId name_oid = expr->eval(*binding);

        // Extract the string value
        auto type = GQL_OID::get_type(name_oid);
        if (type == GQL_OID::Type::STRING_SIMPLE_INLINE) {
            // Extract inline string
            uint64_t value = name_oid.get_value();
            char str[8];
            int len = 0;
            for (int i = 6; i >= 0; i--) {
                char c = static_cast<char>((value >> (i * 8)) & 0xFF);
                if (c == '\0') break;
                str[len++] = c;
            }
            projection_name = std::string(str, len);
        } else if (type == GQL_OID::Type::STRING_SIMPLE_EXTERN || type == GQL_OID::Type::STRING_SIMPLE_TMP) {
            // Extract external string
            uint64_t external_id = name_oid.get_value();
            char buffer[StringManager::MAX_STRING_SIZE];
            string_manager.print_to_buffer(buffer, external_id);
            projection_name = std::string(buffer);
        } else {
            throw std::runtime_error("PROJECT() requires a string literal as projection name");
        }

        // Create the projection directory
        auto& proj_manager = ProjectionManager::get_instance();
        std::string proj_dir = proj_manager.create_projection(projection_name);

        // Initialize the projection storage
        projection_storage = std::make_unique<ProjectionStorage>(
            proj_dir,
            proj_manager.get_db_folder()
        );
        projection_storage->init();
    }

    void process() override
    {
        if (!projection_storage) {
            throw std::runtime_error("ProjectionStorage not initialized in AggProject::process()");
        }

        // Maps to track nodes and edges with their properties
        std::map<ObjectId, std::unordered_map<std::string, ObjectId>> node_properties;
        std::map<ObjectId, std::unordered_map<std::string, ObjectId>> edge_properties;
        std::map<ObjectId, bool> edges_seen;

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
                // Just mark that we've seen this node; will add with properties later
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
            }
        }

        // Second pass: add nodes with their properties
        for (const auto& [node_id, props] : node_properties) {
            ProjectedNode node;
            node.node_id = node_id;
            node.properties = props;
            projection_storage->add_node(node);
        }

        // Third pass: add edges with their properties
        for (const auto& [edge_id, is_directed] : edges_seen) {
            ProjectedEdge edge;
            edge.edge_id = edge_id;
            edge.is_directed = is_directed;

            // Add properties if any
            auto props_it = edge_properties.find(edge_id);
            if (props_it != edge_properties.end()) {
                edge.properties = props_it->second;
            }

            // TODO: Extract from/to nodes by analyzing adjacent bindings
            // For now, we store edges with placeholder from/to
            // Proper implementation requires pattern metadata passed during initialization
            edge.from_node = ObjectId(ObjectId::NULL_ID);
            edge.to_node = ObjectId(ObjectId::NULL_ID);

            projection_storage->add_edge(edge);
        }
    }

    // Called at the end of aggregation to get the result
    ObjectId get() override
    {
        if (!projection_storage) {
            throw std::runtime_error("ProjectionStorage not initialized in AggProject::get()");
        }

        // Flush any pending writes
        projection_storage->flush();

        // Return the projection name as a string ObjectId
        // This allows the query to return the projection name as the result
        uint64_t external_id = string_manager.get_str_id(projection_name);
        return ObjectId(ObjectId::MASK_STRING_SIMPLE_EXTERN | external_id);
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
};
} // namespace GQL
