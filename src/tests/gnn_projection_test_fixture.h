#pragma once

// Shared in-process ProjectionStorage fixture for the symmetric-topology GNN
// unit tests. Extracted verbatim from gnn_build_topology_snapshot_procedure_test.cc
// so the symmetric bake / parallel-edge / mode tests reuse one hermetic
// System + QueryContext + ProjectionManager bootstrap instead of copying the
// 130-line fixture into each file. Each test executable includes this header
// in exactly one TU, so the inline helpers never collide across binaries.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>

#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace gnn_test_fixture {

namespace fs = std::filesystem;

// Process-lifetime System + QueryContext + ProjectionManager.
class MdbFixture {
public:
    static MdbFixture& instance() {
        static MdbFixture f;
        return f;
    }

    const std::string& db_folder() const { return db_folder_; }

private:
    MdbFixture() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        db_folder_ = "test_db_gnn_symmetric_" + std::to_string(rng());
        fs::remove_all(db_folder_);

        system_.reset(new System(
            db_folder_,
            1024 * 1024,        // str_static_size
            1024 * 1024,        // str_dynamic_size
            64 * 1024 * 1024,   // shared_buffer_size
            32 * 1024 * 1024,   // private_buffer_size
            1024 * 1024,        // tensor_static_size
            1024 * 1024,        // tensor_dynamic_size
            1                   // workers
        ));

        query_ctx_.reset(new QueryContext());
        QueryContext::set_query_ctx(query_ctx_.get());

        auto& manager = GQL::ProjectionManager::get_instance();
        manager.init(db_folder_);
    }

    std::string                     db_folder_;
    std::unique_ptr<System>         system_;
    std::unique_ptr<QueryContext>   query_ctx_;
};

// Builds a 4-node / 4-directed-edge projection-storage on disk. Returns the
// absolute directory path. Topology: 0->1 (e=100), 0->2 (e=101), 1->2 (e=102),
// 2->3 (e=103). Undirected merge => 0:{1,2} 1:{2,0} 2:{3,0,1} 3:{2}.
inline std::string build_small_projection(const std::string& projection_name) {
    auto& manager = GQL::ProjectionManager::get_instance();
    std::string proj_dir = manager.create_projection(projection_name);

    GQL::ProjectionStorage storage(
        proj_dir,
        MdbFixture::instance().db_folder(),
        projection_name);
    storage.init();

    for (uint64_t i = 0; i < 4; ++i) {
        GQL::ProjectedNode node;
        node.node_id = ObjectId(i);
        storage.add_node(node);
    }

    auto make_edge = [](uint64_t from, uint64_t to, uint64_t eid) {
        GQL::ProjectedEdge edge;
        edge.from_node   = ObjectId(from);
        edge.to_node     = ObjectId(to);
        edge.edge_id     = ObjectId(eid);
        edge.is_directed = true;
        return edge;
    };

    storage.add_edge(make_edge(0, 1, 100));
    storage.add_edge(make_edge(0, 2, 101));
    storage.add_edge(make_edge(1, 2, 102));
    storage.add_edge(make_edge(2, 3, 103));

    storage.flush();  // builds B+Trees + opens readers
    return proj_dir;
}

// Open a fresh storage handle for an existing projection directory.
inline std::unique_ptr<GQL::ProjectionStorage> open_projection(const std::string& proj_dir) {
    auto storage = std::make_unique<GQL::ProjectionStorage>(
        proj_dir, MdbFixture::instance().db_folder());
    storage->open();
    return storage;
}

// Full byte content of a file.
inline std::string read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::string out((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    return out;
}

}  // namespace gnn_test_fixture
