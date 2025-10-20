// Tool to inspect projection contents
// Usage: projection_inspect <db_folder> <projection_name>

#include <iostream>
#include <string>
#include <iomanip>
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <db_folder> <projection_name>" << std::endl;
        std::cerr << "Example: " << argv[0] << " data/dbs/gql/posts all_papers" << std::endl;
        return 1;
    }

    std::string db_folder = argv[1];
    std::string projection_name = argv[2];

    try {
        // Initialize system
        System system(
            db_folder,
            256 * 1024 * 1024,   // str_static_size
            256 * 1024 * 1024,   // str_dynamic_size
            1024 * 1024 * 1024,  // shared_buffer_size
            256 * 1024 * 1024,   // private_buffer_size
            64 * 1024 * 1024,    // tensor_static_size
            64 * 1024 * 1024,    // tensor_dynamic_size
            1                    // workers
        );

        // Set up QueryContext
        QueryContext query_ctx;
        QueryContext::set_query_ctx(&query_ctx);

        // Get projection directory
        auto& proj_manager = GQL::ProjectionManager::get_instance();
        proj_manager.init(db_folder);

        std::string proj_dir = db_folder + "/projections/" + projection_name;

        std::cout << "==========================================" << std::endl;
        std::cout << "Projection Inspection Tool" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Database: " << db_folder << std::endl;
        std::cout << "Projection: " << projection_name << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << std::endl;

        // List all projections in the database
        std::cout << "Available projections:" << std::endl;
        auto projections = proj_manager.list_projections();
        for (size_t i = 0; i < projections.size(); i++) {
            if (projections[i].name == projection_name) {
                std::cout << "  [" << (i+1) << "] " << projections[i].name << " <-- (selected)" << std::endl;
            } else {
                std::cout << "  [" << (i+1) << "] " << projections[i].name << std::endl;
            }
        }
        std::cout << std::endl;

        // Check projection files
        std::cout << "Projection Structure:" << std::endl;
        std::cout << "  Directory: " << proj_dir << std::endl;
        std::cout << "  Files:" << std::endl;

        std::vector<std::string> expected_files = {
            "nodes.dir", "nodes.leaf",
            "from_to_edge.dir", "from_to_edge.leaf",
            "to_from_edge.dir", "to_from_edge.leaf",
            "edge_direction.dir", "edge_direction.leaf"
        };

        for (const auto& file : expected_files) {
            std::string full_path = proj_dir + "/" + file;
            std::ifstream f(full_path);
            if (f.good()) {
                f.seekg(0, std::ios::end);
                size_t size = f.tellg();
                std::cout << "    \u2713 " << std::setw(25) << std::left << file
                          << " (" << size << " bytes)" << std::endl;
            } else {
                std::cout << "    \u2717 " << file << " (missing)" << std::endl;
            }
        }
        std::cout << std::endl;

        // Read projection data
        std::cout << "Projection Data:" << std::endl;
        std::cout << "----------------" << std::endl;

        GQL::ProjectionStorage storage(proj_dir, db_folder);
        storage.open();  // Open existing projection, don't create new one!

        // Get all nodes
        auto node_ids = storage.get_all_node_ids();
        std::cout << "Nodes: " << node_ids.size() << std::endl;

        size_t node_limit = std::min(node_ids.size(), size_t(10));
        for (size_t i = 0; i < node_limit; i++) {
            std::cout << "  [" << (i+1) << "] Node: 0x" << std::hex << node_ids[i].id << std::dec;

            // Try to decode the node type
            auto type = GQL_OID::get_type(node_ids[i]);
            if (type == GQL_OID::Type::NODE) {
                std::cout << " (NODE)";
            }
            std::cout << std::endl;
        }
        if (node_ids.size() > node_limit) {
            std::cout << "  ... (" << (node_ids.size() - node_limit) << " more nodes)" << std::endl;
        }
        std::cout << std::endl;

        // Get all edges
        auto edges = storage.get_all_edges_info();
        std::cout << "Edges: " << edges.size() << std::endl;

        size_t edge_limit = std::min(edges.size(), size_t(10));
        for (size_t i = 0; i < edge_limit; i++) {
            auto [from, to, edge_id, is_directed] = edges[i];
            std::cout << "  [" << (i+1) << "] ";
            std::cout << "0x" << std::hex << from.id << std::dec;
            std::cout << (is_directed ? " -> " : " <-> ");
            std::cout << "0x" << std::hex << to.id << std::dec;
            std::cout << " (edge: 0x" << std::hex << edge_id.id << std::dec << ")";
            std::cout << std::endl;
        }
        if (edges.size() > edge_limit) {
            std::cout << "  ... (" << (edges.size() - edge_limit) << " more edges)" << std::endl;
        }
        std::cout << std::endl;

        std::cout << "==========================================" << std::endl;
        std::cout << "Summary:" << std::endl;
        std::cout << "  Total Nodes: " << node_ids.size() << std::endl;
        std::cout << "  Total Edges: " << edges.size() << std::endl;
        std::cout << "==========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
