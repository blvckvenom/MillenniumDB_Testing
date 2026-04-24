// bench_topology_micro.cc
//
// Spec #4-B T4.17 — isolated micro-benchmark for
// `mdb::gnn::TopologyAccessor::sample_khop_neighbors`.
//
// The end-to-end bench in `scripts/bench_topology_snapshot.sh` goes through
// `gnn_offline_sample`, which spends ~96% of its wall-clock in non-topology
// work (PackedBatchStore writes, Torch tensor construction, unordered_map
// inserts, RNG). That crowds out the CSR fast-path's O(1) neighbor-lookup
// win, producing a near-1x speedup on ogbn-products in the pipeline-level
// numbers.
//
// This binary strips everything off except the TopologyAccessor call itself
// so the raw B+Tree-vs-CSR ratio is observable. It:
//
//   1. Opens an existing projection directory on disk (built externally via
//      `graph_project` — no server needed).
//   2. Collects N seeds via `NodeIterator` using a fixed RNG.
//   3. Warm-up pass to stabilize the page cache.
//   4. Times `TopologyAccessor::sample_khop_neighbors` on both paths:
//        - CSR present (topology_fwd.csr + topology_rev.csr on disk)
//        - CSR absent (files moved aside) → B+Tree fallback
//   5. Restores the CSR files.
//
// The benchmark ONLY measures the sampling call. No disk output, no Torch
// serialization, no procedure-layer overhead.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <torch/torch.h>

#include "gnn/projection/topology_accessor.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace fs = std::filesystem;

namespace {

// -----------------------------------------------------------------------------
// CLI usage
// -----------------------------------------------------------------------------
[[noreturn]] void die(const std::string& msg, int code = 2) {
    std::cerr << "ERROR: " << msg << std::endl;
    std::exit(code);
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <db_folder> <projection_name> "
        << "[--num-seeds N] [--fanouts a,b,...] [--warmup N] "
        << "[--orientation NATURAL|REVERSE|UNDIRECTED] [--dataset-label NAME]\n"
        << "\n"
        << "Example:\n"
        << "  " << argv0 << " data/dbs/gql/cora_gnn cora_proj \\\n"
        << "      --num-seeds 10000 --fanouts 15,10 --warmup 100\n";
}

struct Options {
    std::string  db_folder;
    std::string  proj_name;
    std::string  dataset_label;         // Cosmetic — defaults to proj_name.
    int64_t      num_seeds     = 10000;
    int64_t      warmup_seeds  = 100;
    std::vector<int64_t> fanouts { 15, 10 };
    mdb::gnn::EdgeOrientation orient = mdb::gnn::EdgeOrientation::UNDIRECTED;
};

std::vector<int64_t> parse_csv_int64(const std::string& csv) {
    std::vector<int64_t> out;
    std::string cur;
    for (char c : csv) {
        if (c == ',' || c == ' ' || c == '[' || c == ']') {
            if (!cur.empty()) {
                out.push_back(std::stoll(cur));
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(std::stoll(cur));
    }
    return out;
}

Options parse_args(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        std::exit(2);
    }
    Options o;
    o.db_folder = argv[1];
    o.proj_name = argv[2];
    o.dataset_label = o.proj_name;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) die(std::string("missing value for ") + flag);
            return std::string(argv[++i]);
        };
        if (a == "--num-seeds") {
            o.num_seeds = std::stoll(need("--num-seeds"));
        } else if (a == "--warmup") {
            o.warmup_seeds = std::stoll(need("--warmup"));
        } else if (a == "--fanouts") {
            o.fanouts = parse_csv_int64(need("--fanouts"));
            if (o.fanouts.empty()) die("--fanouts produced empty list");
        } else if (a == "--orientation") {
            std::string s = need("--orientation");
            if      (s == "NATURAL")    o.orient = mdb::gnn::EdgeOrientation::NATURAL;
            else if (s == "REVERSE")    o.orient = mdb::gnn::EdgeOrientation::REVERSE;
            else if (s == "UNDIRECTED") o.orient = mdb::gnn::EdgeOrientation::UNDIRECTED;
            else die("unknown orientation: " + s);
        } else if (a == "--dataset-label") {
            o.dataset_label = need("--dataset-label");
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            die("unknown argument: " + a);
        }
    }
    return o;
}

// -----------------------------------------------------------------------------
// Singleton MDB bootstrap — same pattern as the T4.11 fixture (System +
// QueryContext + ProjectionManager are process-lifetime singletons).
//
// Buffer pool sized for large datasets: 1 GB shared, 256 MB private. This
// must be small enough to leave RAM for the sampling workload and large
// enough that a 60M-edge projection's BPT root pages stay resident.
// -----------------------------------------------------------------------------
class MdbRuntime {
public:
    MdbRuntime(const std::string& db_folder) {
        system_ = std::make_unique<System>(
            db_folder,
            4 * 1024 * 1024,             // str_static_size
            4 * 1024 * 1024,             // str_dynamic_size
            1024ULL * 1024 * 1024,       // shared_buffer_size = 1 GB
            256ULL  * 1024 * 1024,       // private_buffer_size = 256 MB
            4 * 1024 * 1024,             // tensor_static_size
            4 * 1024 * 1024,             // tensor_dynamic_size
            1);                          // workers
        query_ctx_ = std::make_unique<QueryContext>();
        QueryContext::set_query_ctx(query_ctx_.get());
        auto& manager = GQL::ProjectionManager::get_instance();
        manager.init(db_folder);
    }

private:
    std::unique_ptr<System>         system_;
    std::unique_ptr<QueryContext>   query_ctx_;
};

// -----------------------------------------------------------------------------
// Seed generation.
//
// Pulls the first (num_seeds * 4) node IDs via NodeIterator, picks num_seeds
// of them uniformly via mt19937_64(42). Using a multiplicative oversample
// gives a reasonable seed spread on cora (2708 nodes, asks 10000 seeds) by
// capping at the total node count, and on ogbn-products (2.5M nodes, asks
// 10000 seeds) by sampling a stratified slice from the iterator output.
// -----------------------------------------------------------------------------
std::vector<ObjectId> collect_seeds(
    GQL::ProjectionStorage& storage,
    int64_t desired_count,
    uint64_t rng_seed)
{
    mdb::gnn::NodeIterator iter(storage);
    std::vector<ObjectId> all;

    while (true) {
        auto chunk = iter.next_batch(4096);
        if (!chunk.has_value()) break;
        for (auto id : *chunk) all.push_back(id);
    }

    if (all.empty()) die("projection has zero nodes");

    std::mt19937_64 rng(rng_seed);
    std::shuffle(all.begin(), all.end(), rng);
    int64_t keep =
        std::min<int64_t>(desired_count, static_cast<int64_t>(all.size()));
    all.resize(static_cast<std::size_t>(keep));
    return all;
}

// -----------------------------------------------------------------------------
// Timed k-hop sampling pass.
//
// Returns wall-clock seconds. Discards the sampled subgraph — we only care
// about the time it took to produce it.
// -----------------------------------------------------------------------------
double run_timed_pass(
    mdb::gnn::TopologyAccessor&          accessor,
    const std::vector<ObjectId>&         seeds,
    const std::vector<int64_t>&          fanouts,
    mdb::gnn::EdgeOrientation            orient,
    uint64_t                             rng_seed)
{
    accessor.set_random_seed(rng_seed);
    auto t0 = std::chrono::steady_clock::now();
    volatile std::size_t layer_count = 0;
    auto layers = accessor.sample_khop_neighbors(
        seeds, fanouts, mdb::gnn::SamplingStrategy::UNIFORM, orient);
    layer_count = layers.size();
    (void)layer_count;
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    return dt.count();
}

// -----------------------------------------------------------------------------
// CSR sidecar management.
//
// The production layer's `TopologyAccessor::Impl` ctor calls
// `TopologySnapshotReader::open(...)` once, which mmaps the file if present
// and sets an internal `has_data()` flag otherwise. Therefore: to switch
// paths we must construct a *new* TopologyAccessor after moving/restoring
// the files.
// -----------------------------------------------------------------------------
struct CsrSidecars {
    fs::path proj_dir;
    fs::path fwd_live;
    fs::path rev_live;
    fs::path fwd_park;
    fs::path rev_park;
    bool     fwd_present_initially = false;
    bool     rev_present_initially = false;

    explicit CsrSidecars(const fs::path& dir)
        : proj_dir(dir),
          fwd_live(dir / "topology_fwd.csr"),
          rev_live(dir / "topology_rev.csr"),
          fwd_park(dir / "topology_fwd.csr.bench_parked"),
          rev_park(dir / "topology_rev.csr.bench_parked")
    {}

    bool has_both() const {
        return fs::exists(fwd_live) && fs::exists(rev_live);
    }

    void note_initial_state() {
        fwd_present_initially = fs::exists(fwd_live);
        rev_present_initially = fs::exists(rev_live);
    }

    // Move the sidecars to a parking location (same filesystem) so the
    // TopologyAccessor's next construction sees `has_data() == false` and
    // drops to the BPT path.
    void park() {
        std::error_code ec;
        if (fs::exists(fwd_live)) fs::rename(fwd_live, fwd_park, ec);
        if (ec) die("failed to park topology_fwd.csr: " + ec.message());
        ec.clear();
        if (fs::exists(rev_live)) fs::rename(rev_live, rev_park, ec);
        if (ec) die("failed to park topology_rev.csr: " + ec.message());
    }

    void unpark() {
        std::error_code ec;
        if (fs::exists(fwd_park)) fs::rename(fwd_park, fwd_live, ec);
        if (ec) {
            std::cerr << "WARN: failed to restore topology_fwd.csr: "
                      << ec.message() << std::endl;
        }
        ec.clear();
        if (fs::exists(rev_park)) fs::rename(rev_park, rev_live, ec);
        if (ec) {
            std::cerr << "WARN: failed to restore topology_rev.csr: "
                      << ec.message() << std::endl;
        }
    }
};

std::string orient_to_string(mdb::gnn::EdgeOrientation o) {
    switch (o) {
        case mdb::gnn::EdgeOrientation::NATURAL:    return "NATURAL";
        case mdb::gnn::EdgeOrientation::REVERSE:    return "REVERSE";
        case mdb::gnn::EdgeOrientation::UNDIRECTED: return "UNDIRECTED";
    }
    return "?";
}

std::string fanouts_to_string(const std::vector<int64_t>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

}  // namespace

int main(int argc, char** argv) {
    Options opts = parse_args(argc, argv);

    if (!fs::is_directory(opts.db_folder)) {
        die("db_folder is not a directory: " + opts.db_folder);
    }

    MdbRuntime runtime(opts.db_folder);
    auto& manager = GQL::ProjectionManager::get_instance();

    if (!manager.projection_exists(opts.proj_name)) {
        die("projection not found: " + opts.proj_name +
            " (db_folder=" + opts.db_folder + ")");
    }

    std::string proj_dir = manager.get_projection_dir(opts.proj_name);
    CsrSidecars csr(proj_dir);
    csr.note_initial_state();

    // -------------------------------------------------------------------------
    // Storage open (once — the benchmark reuses the same ProjectionStorage
    // across both modes; only the TopologyAccessor instance is rebuilt).
    // -------------------------------------------------------------------------
    auto storage = std::make_unique<GQL::ProjectionStorage>(proj_dir, opts.db_folder);
    storage->open();

    const uint64_t node_count = storage->get_node_count();
    const uint64_t edge_count = storage->get_edge_count();
    if (node_count == 0) die("projection has zero nodes");

    // -------------------------------------------------------------------------
    // If the projection has no CSRs on disk, build them in-place via the
    // test-only hook. We never leave the projection in a worse state than
    // we found it (tracked in CsrSidecars::*_present_initially).
    // -------------------------------------------------------------------------
    if (!csr.has_both()) {
        std::cerr << "  [setup] topology_*.csr missing — building via "
                     "detail::build_topology_snapshots_for_test()...\n";
        GQL::detail::build_topology_snapshots_for_test(
            *storage,
            /*build_forward=*/true,
            /*build_reverse=*/true);
        if (!csr.has_both()) {
            die("CSR build did not produce both sidecar files");
        }
    }

    // -------------------------------------------------------------------------
    // Seed collection.
    // -------------------------------------------------------------------------
    auto seeds = collect_seeds(*storage, opts.num_seeds, /*rng_seed=*/42ULL);
    const int64_t actual_seeds = static_cast<int64_t>(seeds.size());

    std::vector<ObjectId> warm_seeds(
        seeds.begin(),
        seeds.begin() + std::min<int64_t>(opts.warmup_seeds, actual_seeds));

    std::cout << "=== bench_topology_micro ===\n";
    std::cout << "dataset:      " << opts.dataset_label << "\n";
    std::cout << "projection:   " << opts.proj_name << "  (dir=" << proj_dir << ")\n";
    std::cout << "node_count:   " << node_count << "\n";
    std::cout << "edge_count:   " << edge_count << "\n";
    std::cout << "num_seeds:    " << actual_seeds << "\n";
    std::cout << "warmup_seeds: " << warm_seeds.size() << "\n";
    std::cout << "fanouts:      " << fanouts_to_string(opts.fanouts) << "\n";
    std::cout << "orientation:  " << orient_to_string(opts.orient) << "\n";
    std::cout << std::flush;

    // Use CPU device for the edge-index tensors so we aren't measuring a
    // CUDA allocator warm-up on the first call.
    const torch::Device cpu_dev = torch::kCPU;

    // -------------------------------------------------------------------------
    // Pass 1 — CSR-path.
    // -------------------------------------------------------------------------
    double csr_sec = 0.0;
    {
        mdb::gnn::TopologyAccessor acc(*storage);
        acc.set_target_device(cpu_dev);
        // Warm up: populates page cache + torch allocator + RNG.
        (void)run_timed_pass(acc, warm_seeds, opts.fanouts, opts.orient, 42ULL);
        csr_sec = run_timed_pass(acc, seeds, opts.fanouts, opts.orient, 42ULL);
    }

    // -------------------------------------------------------------------------
    // Pass 2 — BPT-path (sidecars parked, fresh accessor).
    // -------------------------------------------------------------------------
    csr.park();

    double bpt_sec = 0.0;
    {
        mdb::gnn::TopologyAccessor acc(*storage);
        acc.set_target_device(cpu_dev);
        (void)run_timed_pass(acc, warm_seeds, opts.fanouts, opts.orient, 42ULL);
        bpt_sec = run_timed_pass(acc, seeds, opts.fanouts, opts.orient, 42ULL);
    }

    csr.unpark();

    // -------------------------------------------------------------------------
    // Report.
    // -------------------------------------------------------------------------
    auto sps = [&](double sec) {
        return (sec > 0.0) ? (static_cast<double>(actual_seeds) / sec) : 0.0;
    };
    double csr_sps = sps(csr_sec);
    double bpt_sps = sps(bpt_sec);
    double speedup = (bpt_sec > 0.0) ? (bpt_sec / csr_sec) : 0.0;

    std::cout << "\n";
    std::cout << "            wall_sec      seeds/sec\n";
    std::printf("BPT-path:   %10.4f   %12.0f\n", bpt_sec, bpt_sps);
    std::printf("CSR-path:   %10.4f   %12.0f\n", csr_sec, csr_sps);
    std::printf("speedup:    %10.2fx\n", speedup);
    std::cout << std::flush;

    // Machine-readable line for shell aggregation. Stable column order:
    // dataset, orient, fanouts, num_seeds, bpt_sec, csr_sec, bpt_sps, csr_sps, speedup
    std::cout << "\n";
    std::printf(
        "BENCH_CSV,%s,%s,%s,%lld,%.6f,%.6f,%.2f,%.2f,%.3f\n",
        opts.dataset_label.c_str(),
        orient_to_string(opts.orient).c_str(),
        fanouts_to_string(opts.fanouts).c_str(),
        static_cast<long long>(actual_seeds),
        bpt_sec, csr_sec, bpt_sps, csr_sps, speedup);

    return 0;
}
