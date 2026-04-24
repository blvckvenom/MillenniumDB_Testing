// native_projection_builder_topology_snapshot_test.cc
//
// Integration tests for Spec #4-B T4.6 — TopologySnapshotWriter wired into
// NativeProjectionBuilder's finalize path.
//
// Rather than spinning up the full GQL graph catalog (which would require
// loading a real dataset through the import pipeline), this test drives
// the exact BPT-scan + writer body via the `GQL::detail::build_topology_snapshots_for_test`
// hook the builder exposes. The hook shares source with
// `NativeProjectionBuilder::build_one_topology_snapshot_`, so what is
// validated here is byte-for-byte the code path finalize() runs.
//
// Scope:
//   1. Constructor default for `build_topology_snapshot` is false (no behavior
//      change for existing callers).
//   2. A projection populated with a small hand-authored graph produces
//      `topology_fwd.csr` / `topology_rev.csr` whose headers match the
//      writer contract and whose body passes TopologySnapshotReader's
//      `has_data()` validation.
//   3. Neighbor slices returned by the reader match the hand-authored
//      adjacency list exactly.
//
// Spec reference: docs/superpowers/specs/2026-04-25-topology-snapshot-design.md
//                 §3.7 (integration point), §5.1 / §5.2 (reader validation).

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace {

namespace fs = std::filesystem;

// A process-lifetime System + ProjectionManager singleton shared across
// tests. MDB's System constructor opens a buffer pool + file manager that
// can only be bound once per process.
class MdbFixture {
public:
    static MdbFixture& instance() {
        static MdbFixture f;
        return f;
    }

    const std::string& db_folder() const { return db_folder_; }

private:
    MdbFixture() {
        // Use a random suffix so repeat runs under the same working
        // directory never collide with a leftover test_db_* tree.
        std::random_device rd;
        std::mt19937_64 rng(rd());
        db_folder_ = "test_db_topo_snap_" + std::to_string(rng());
        fs::remove_all(db_folder_);

        system_.reset(new System(
            db_folder_,
            1024 * 1024,        // str_static_size (1MB)
            1024 * 1024,        // str_dynamic_size (1MB)
            64 * 1024 * 1024,   // shared_buffer_size (64MB)
            32 * 1024 * 1024,   // private_buffer_size (32MB)
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

// Builds a small projection-storage directly via ProjectionStorage::add_node /
// add_edge + flush. Returns the absolute projection directory so tests can
// locate the generated .csr sidecars.
//
// Topology:
//   nodes: 0, 1, 2, 3
//   directed edges: 0->1 (e=100), 0->2 (e=101), 1->2 (e=102), 2->3 (e=103)
// Four nodes is small enough to verify every row_ptr slice by hand.
std::string build_small_projection(const std::string& projection_name) {
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

    storage.flush();  // builds all 6 required B+Trees + opens readers

    // Drive the builder-level integration hook: this invokes the same code
    // that NativeProjectionBuilder::build_one_topology_snapshot_() does
    // from finalize() when build_topology_snapshot=true.
    GQL::detail::build_topology_snapshots_for_test(
        storage,
        /*build_forward=*/true,
        /*build_reverse=*/true);

    return proj_dir;
}

}  // namespace

// ---------------------------------------------------------------------------
// Compile-time plumbing check — referencing the 18-arg constructor + the
// new accessor ensures any drift in the header surface trips `-Werror=...`
// instead of silently breaking downstream callers. The function is never
// called at runtime (its body is gated on an impossible branch) so it
// does NOT require a live graph catalog; the NativeScanner dependency is
// only reached at construction time.
// ---------------------------------------------------------------------------

[[maybe_unused]] void compile_time_plumbing_check() {
    // Unreachable: forces the compiler to resolve the symbol without ever
    // invoking the constructor (which would need a live gql_model).
    if (volatile int guard = 0; guard != 0) {
        GQL::NativeProjectionBuilder b(
            "", "",
            std::vector<std::string>{}, std::vector<std::string>{},
            GQL::Orientation::NATURAL, GQL::Aggregation::SINGLE, "",
            std::unordered_map<std::string, GQL::Orientation>{},
            std::unordered_map<std::string, GQL::Aggregation>{},
            std::unordered_map<std::string, std::string>{},
            std::unordered_map<std::string, GQL::PropertyConfig>{},
            std::unordered_map<std::string, GQL::PropertyConfig>{},
            "", "", "", true, GQL::IndexSet::ALL, /*build_topology_snapshot=*/true);
        (void)b.get_build_topology_snapshot();
        (void)b.get_index_set();
    }
}

// ---------------------------------------------------------------------------
// Integration tests: .csr files land on disk, reader validates them.
//
// Direct runtime instantiation of the full builder is skipped because its
// NativeScanner dependency expects a live graph catalog — the end-to-end
// path exercised below (ProjectionStorage populated →
// detail::build_topology_snapshots_for_test → reader) covers the same
// sidecar emission code path byte-for-byte as the builder's
// `build_one_topology_snapshot_` method.
// ---------------------------------------------------------------------------

TEST(NativeProjectionBuilderTopologySnapshot, SidecarFilesExistAndValidate) {
    (void)MdbFixture::instance();

    const std::string proj_name = "topo_snap_valid_proj";
    const std::string proj_dir  = build_small_projection(proj_name);

    const fs::path fwd_path = fs::path(proj_dir) / "topology_fwd.csr";
    const fs::path rev_path = fs::path(proj_dir) / "topology_rev.csr";

    ASSERT_TRUE(fs::exists(fwd_path))
        << "topology_fwd.csr missing at " << fwd_path;
    ASSERT_TRUE(fs::exists(rev_path))
        << "topology_rev.csr missing at " << rev_path;

    // File size sanity: header (64) + row_ptr ((N+1)*8) + col_idx (M*8) +
    // edge_ids (M*8). Our fixture has N=4, M=4 so the minimum is:
    //   64 + 40 + 32 + 32 = 168 bytes.
    const std::uintmax_t fwd_sz = fs::file_size(fwd_path);
    const std::uintmax_t rev_sz = fs::file_size(rev_path);
    EXPECT_GE(fwd_sz, static_cast<std::uintmax_t>(168));
    EXPECT_GE(rev_sz, static_cast<std::uintmax_t>(168));
    // Upper bound: can't exceed a generous constant given the fixture.
    EXPECT_LT(fwd_sz, static_cast<std::uintmax_t>(1024));
    EXPECT_LT(rev_sz, static_cast<std::uintmax_t>(1024));

    // Reader open must succeed (has_data true) for both directions.
    using Dir = GQL::Projection::TopologySnapshotReader::Direction;
    auto fwd_reader = GQL::Projection::TopologySnapshotReader::open(proj_dir, Dir::FORWARD);
    auto rev_reader = GQL::Projection::TopologySnapshotReader::open(proj_dir, Dir::REVERSE);
    EXPECT_TRUE(fwd_reader.has_data());
    EXPECT_TRUE(rev_reader.has_data());

    // Header fields match fixture (N=4, M=4, has_edge_ids bit set).
    EXPECT_EQ(fwd_reader.num_nodes(), 4u);
    EXPECT_EQ(fwd_reader.num_edges(), 4u);
    EXPECT_TRUE(fwd_reader.has_edge_ids());
    EXPECT_EQ(rev_reader.num_nodes(), 4u);
    EXPECT_EQ(rev_reader.num_edges(), 4u);
    EXPECT_TRUE(rev_reader.has_edge_ids());
}

TEST(NativeProjectionBuilderTopologySnapshot, ForwardNeighborSlicesAreCorrect) {
    (void)MdbFixture::instance();

    const std::string proj_name = "topo_snap_fwd_slice_proj";
    const std::string proj_dir  = build_small_projection(proj_name);

    using Dir = GQL::Projection::TopologySnapshotReader::Direction;
    auto reader = GQL::Projection::TopologySnapshotReader::open(proj_dir, Dir::FORWARD);
    ASSERT_TRUE(reader.has_data());

    // Fixture forward adjacency:
    //   0 -> {1, 2}
    //   1 -> {2}
    //   2 -> {3}
    //   3 -> {}
    auto n0 = reader.neighbors(0);
    auto n1 = reader.neighbors(1);
    auto n2 = reader.neighbors(2);
    auto n3 = reader.neighbors(3);

    ASSERT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], 1u);
    EXPECT_EQ(n0[1], 2u);

    ASSERT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1[0], 2u);

    ASSERT_EQ(n2.size(), 1u);
    EXPECT_EQ(n2[0], 3u);

    EXPECT_EQ(n3.size(), 0u);  // isolated destination
}

TEST(NativeProjectionBuilderTopologySnapshot, ReverseNeighborSlicesAreCorrect) {
    (void)MdbFixture::instance();

    const std::string proj_name = "topo_snap_rev_slice_proj";
    const std::string proj_dir  = build_small_projection(proj_name);

    using Dir = GQL::Projection::TopologySnapshotReader::Direction;
    auto reader = GQL::Projection::TopologySnapshotReader::open(proj_dir, Dir::REVERSE);
    ASSERT_TRUE(reader.has_data());

    // Fixture reverse adjacency (to -> set of from):
    //   0 -> {}     (no one points at 0)
    //   1 -> {0}
    //   2 -> {0, 1}
    //   3 -> {2}
    auto n0 = reader.neighbors(0);
    auto n1 = reader.neighbors(1);
    auto n2 = reader.neighbors(2);
    auto n3 = reader.neighbors(3);

    EXPECT_EQ(n0.size(), 0u);

    ASSERT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1[0], 0u);

    ASSERT_EQ(n2.size(), 2u);
    EXPECT_EQ(n2[0], 0u);
    EXPECT_EQ(n2[1], 1u);

    ASSERT_EQ(n3.size(), 1u);
    EXPECT_EQ(n3[0], 2u);
}

// ---------------------------------------------------------------------------
// Spec #4-B T4.18 — golden compare: integrated path vs post-hoc path.
//
// Builds the same fixture two ways:
//   A. Integrated path: set_build_topology_snapshot(true) before flush(),
//      which triggers the mmap-over-.leaf builder inside
//      ProjectionStorage::build_{from_to,to_from}_edge_index_().
//   B. Post-hoc path: flush() without the flag, then
//      detail::build_topology_snapshots_for_test() drives the legacy
//      BPT-iterator walker (build_one_topology_snapshot_ body).
//
// The byte contents of `topology_fwd.csr` and `topology_rev.csr` must be
// identical under both paths — same header, same ROW_PTR, same COL_IDX,
// same EDGE_IDS, same source-.leaf SHA-256. This is the load-bearing
// invariant that makes the performance optimization safe to ship.
// ---------------------------------------------------------------------------

namespace {

// Read a file fully into a byte vector. Returns an empty vector if the
// file is missing (caller asserts existence separately).
std::vector<unsigned char> read_file_bytes(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<unsigned char> out(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());
    return out;
}

// Build a small projection that exercises both the dense row indices
// AND a couple of non-trivial degrees (src 0 has degree 2, src 1 has
// degree 1, src 2 has degree 1, src 3 has degree 0). Returns the proj
// directory. Unlike build_small_projection above, this one does NOT
// drive the post-hoc walker — callers choose the path.
std::string build_projection_without_snapshot(const std::string& proj_name,
                                              bool               set_flag) {
    auto& manager = GQL::ProjectionManager::get_instance();
    std::string proj_dir = manager.create_projection(proj_name);

    GQL::ProjectionStorage storage(
        proj_dir,
        MdbFixture::instance().db_folder(),
        proj_name);
    storage.init();

    // When set_flag == true, the two edge-index builders inside flush()
    // will emit topology_{fwd,rev}.csr via the integrated mmap path.
    storage.set_build_topology_snapshot(set_flag);

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

    storage.flush();

    if (!set_flag) {
        // Drive the legacy BPT-iterator walker. With set_flag == true,
        // the integrated path already produced the sidecars during
        // flush() — no post-hoc call is needed.
        GQL::detail::build_topology_snapshots_for_test(
            storage,
            /*build_forward=*/true,
            /*build_reverse=*/true);
    }

    return proj_dir;
}

}  // namespace

TEST(NativeProjectionBuilderTopologySnapshot,
     IntegratedPathProducesByteIdenticalOutputVsPostHoc) {
    (void)MdbFixture::instance();

    // Path A (integrated — set_flag=true triggers the mmap path inside
    // ProjectionStorage::build_from_to_edge_index_ / build_to_from_edge_index_).
    const std::string integrated_dir = build_projection_without_snapshot(
        "topo_snap_integrated_proj", /*set_flag=*/true);

    // Path B (post-hoc — the pre-T4.18 BPT-iterator walker driven via the
    // test detail hook).
    const std::string posthoc_dir = build_projection_without_snapshot(
        "topo_snap_posthoc_proj", /*set_flag=*/false);

    // Both sidecars must exist on both sides.
    for (const auto* basename : {"topology_fwd.csr", "topology_rev.csr"}) {
        const fs::path a = fs::path(integrated_dir) / basename;
        const fs::path b = fs::path(posthoc_dir)    / basename;
        ASSERT_TRUE(fs::exists(a)) << "integrated path missing " << basename;
        ASSERT_TRUE(fs::exists(b)) << "post-hoc path missing "   << basename;
    }

    // Byte compare: header (including source-.leaf SHA-256), ROW_PTR,
    // COL_IDX, EDGE_IDS. A single byte diff would indicate either a
    // reordering, a dedup gap, or a stale SHA digest — any of those
    // would silently break TopologyAccessor at read time.
    const auto a_fwd = read_file_bytes(fs::path(integrated_dir) / "topology_fwd.csr");
    const auto b_fwd = read_file_bytes(fs::path(posthoc_dir)    / "topology_fwd.csr");
    ASSERT_EQ(a_fwd.size(), b_fwd.size())
        << "topology_fwd.csr size mismatch: integrated=" << a_fwd.size()
        << " post-hoc=" << b_fwd.size();
    EXPECT_EQ(a_fwd, b_fwd)
        << "topology_fwd.csr bytes differ between integrated and post-hoc "
           "paths — integration broke byte-identical invariant";

    const auto a_rev = read_file_bytes(fs::path(integrated_dir) / "topology_rev.csr");
    const auto b_rev = read_file_bytes(fs::path(posthoc_dir)    / "topology_rev.csr");
    ASSERT_EQ(a_rev.size(), b_rev.size())
        << "topology_rev.csr size mismatch: integrated=" << a_rev.size()
        << " post-hoc=" << b_rev.size();
    EXPECT_EQ(a_rev, b_rev)
        << "topology_rev.csr bytes differ between integrated and post-hoc "
           "paths — integration broke byte-identical invariant";
}

// ---------------------------------------------------------------------------
// Zero-impact check: building with the flag OFF must produce byte-identical
// `.leaf` / `.dir` files as any prior baseline. We can't diff against a
// previously-checked-in baseline (we have no such fixture), but we can
// verify: (a) no `.csr` files land on disk when the flag is off, and
// (b) the builder's "emitted" accessors remain false.
// ---------------------------------------------------------------------------

TEST(NativeProjectionBuilderTopologySnapshot, FlagOffEmitsNoSidecars) {
    (void)MdbFixture::instance();

    auto& manager = GQL::ProjectionManager::get_instance();
    const std::string proj_name = "topo_snap_flag_off_proj";
    std::string proj_dir = manager.create_projection(proj_name);

    GQL::ProjectionStorage storage(
        proj_dir,
        MdbFixture::instance().db_folder(),
        proj_name);
    storage.init();
    // Flag defaults to false; not calling set_build_topology_snapshot.

    for (uint64_t i = 0; i < 3; ++i) {
        GQL::ProjectedNode node;
        node.node_id = ObjectId(i);
        storage.add_node(node);
    }
    GQL::ProjectedEdge e;
    e.from_node = ObjectId(0);
    e.to_node = ObjectId(1);
    e.edge_id = ObjectId(200);
    e.is_directed = true;
    storage.add_edge(e);

    storage.flush();

    EXPECT_FALSE(fs::exists(fs::path(proj_dir) / "topology_fwd.csr"))
        << "topology_fwd.csr leaked through a build with the flag OFF";
    EXPECT_FALSE(fs::exists(fs::path(proj_dir) / "topology_rev.csr"))
        << "topology_rev.csr leaked through a build with the flag OFF";
    EXPECT_FALSE(storage.fwd_topology_snapshot_built());
    EXPECT_FALSE(storage.rev_topology_snapshot_built());
    EXPECT_FALSE(storage.get_build_topology_snapshot());
}
