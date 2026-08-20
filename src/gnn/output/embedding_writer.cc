// Include tensor_manager.h FIRST, before any header that transitively pulls
// in <linux/io_uring.h> (via liburing.h -> linux/fs.h) which #defines
// BLOCK_SIZE as a macro, conflicting with TensorManager::BLOCK_SIZE.
#include "system/tensor_manager.h"

#include "gnn/output/embedding_writer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "gnn/storage/feature_matrix.h"
#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "misc/available_ram.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace mdb::gnn {

// =============================================================================
// Construction
// =============================================================================

EmbeddingWriter::EmbeddingWriter(
    GraphSAGEModel&            model,
    BatchAssembler&            assembler,
    SampleStorage&             sample_storage,
    const RowMapping&          row_mapping,
    const SampleCatalog&       catalog,
    GQL::ProjectionStorage&    projection_storage,
    Config                     config
)
    : model_(model)
    , assembler_(assembler)
    , sample_storage_(sample_storage)
    , row_mapping_(row_mapping)
    , catalog_(catalog)
    , config_(std::move(config))
    , projection_storage_(projection_storage)
    , rng_(42)  // deterministic seed for reproducible inference
{
}

// =============================================================================
// Coverage::ALL affordability gate
//
// Full coverage holds one hidden-dim tensor per node in the RowMapping and,
// once Phase B starts, an in-memory adjacency map over the whole projection.
// Both peak simultaneously.  On papers100M (111 M nodes, 1.6 G edges scanned
// in both directions) that runs to hundreds of GB, so the run has to be
// refused BEFORE Phase A's full batch scan -- otherwise the failure surfaces
// hours in, or as the OOM killer.
// =============================================================================

uint64_t EmbeddingWriter::estimate_seed_write_bytes(uint64_t num_seeds,
                                                    uint64_t hidden_dim)
{
    // float32 payload + the map node and TensorImpl behind each entry.
    return num_seeds * (hidden_dim * 4ull + EMBEDDING_ENTRY_OVERHEAD_BYTES);
}

uint64_t EmbeddingWriter::estimate_all_write_bytes(uint64_t num_nodes,
                                                   uint64_t hidden_dim,
                                                   uint64_t adj_entries)
{
    return estimate_seed_write_bytes(num_nodes, hidden_dim)  // emb_map, all nodes
         + num_nodes * sizeof(uint64_t)                      // `missing` row indices
         + adj_entries * sizeof(AdjEntry)                    // adjacency payload
         + num_nodes * ADJACENCY_NODE_OVERHEAD_BYTES;        // one bucket per source
}

void EmbeddingWriter::check_all_coverage_fits_() const {
    // Only Coverage::ALL with a non-empty fanout list reaches Phase B; every
    // other combination writes at most the seeds Phase A already collected.
    if (config_.coverage != Coverage::ALL || config_.fanouts.empty()) {
        return;
    }

    const uint64_t mem_available = get_mem_available();
    if (mem_available == 0) {
        // /proc/meminfo unreadable (non-Linux, procfs-less container).  Never
        // refuse on a number we could not read.
        return;
    }

    const uint64_t num_nodes  = row_mapping_.size();
    const uint64_t hidden_dim = static_cast<uint64_t>(model_.config().hidden_dim);

    // build_adjacency_cache_ scans from_to_edge for NATURAL, to_from_edge for
    // REVERSE, and both for UNDIRECTED -- each holding one record per edge.
    const uint64_t directions =
        config_.orientation == EdgeOrientation::UNDIRECTED ? 2ull : 1ull;
    const uint64_t adj_entries =
        projection_storage_.get_edge_count() * directions;

    // The estimate is deliberately generous: it charges the Phase B adjacency
    // cache in full even though build_adjacency_cache_() only ever runs when
    // some node is missing an embedding, and it rounds every per-entry
    // overhead up.  Refusing on a bare `required > mem_available` would put a
    // products-scale write (about 5.1 GB estimated, of which ~2.2 GB is that
    // possibly-unbuilt cache) one memory dip away from failing a run that
    // completes today.  Refuse only past a 2x margin: papers100M still clears
    // the bar by roughly 5x (about 210 GB estimated), and nothing that fits
    // today starts refusing.
    const uint64_t required =
        estimate_all_write_bytes(num_nodes, hidden_dim, adj_entries);
    if (required <= mem_available * 2) {
        return;
    }

    // One decimal place without pulling <iomanip>/<sstream> into this TU.
    auto gb = [](uint64_t bytes) {
        const uint64_t tenths = (bytes * 10ull) / (1024ull * 1024 * 1024);
        return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10);
    };

    // The seed set is only known after Phase A, so size the hint from the
    // catalog's batch layout (an upper bound: the last batch may be short).
    const uint64_t seed_estimate =
        std::min(catalog_.total_batches * catalog_.batch_size, num_nodes);
    std::string seeds_hint;
    if (seed_estimate > 0) {
        seeds_hint = ", about "
                   + gb(estimate_seed_write_bytes(seed_estimate, hidden_dim))
                   + " GB for this sample's " + std::to_string(seed_estimate)
                   + " seeds";
    }

    throw std::runtime_error(
        "EmbeddingWriter: writing property '" + config_.property_name
        + "' over all " + std::to_string(num_nodes)
        + " projection nodes needs about " + gb(required)
        + " GB of RAM, but only " + gb(mem_available)
        + " GB is available on this host. Re-run with writeCoverage:'seeds'"
        + seeds_hint
        + ", which writes only the nodes that were sampling seeds -- with "
          "usePredefinedSplits that is the whole labelled train/val/test set "
          "-- or run the full write on a host with more RAM.");
}

// =============================================================================
// write_all -- full orchestrator (Phases A + B; C is a stub)
// =============================================================================

EmbeddingWriter::Result EmbeddingWriter::write_all() {
    Result result;

    // Refuse an unaffordable full-coverage run before anything expensive has
    // been paid for -- Phase A alone reads every batch in the sample.
    check_all_coverage_fits_();

    // --- Phase A: collect seed embeddings from pre-computed batches ----------
    auto seed_embs = collect_seed_embeddings();

    // Deduplicate: if a node was a seed in multiple batches, keep last
    std::unordered_map<uint64_t, torch::Tensor> emb_map;
    emb_map.reserve(seed_embs.size());
    for (auto& [idx, emb] : seed_embs) {
        emb_map[idx] = std::move(emb);
    }

    // Coverage::SEEDS writes exactly what Phase A collected, so neither the
    // missing-node scan nor Phase B runs.  Skipping the scan is not just a
    // shortcut: on papers100M the `missing` vector alone is ~876 MB of row
    // indices built only to be discarded.
    if (config_.coverage == Coverage::ALL) {
        // Identify missing indices (nodes in RowMapping without embeddings)
        std::vector<uint64_t> missing;
        missing.reserve(row_mapping_.size() > emb_map.size()
                            ? row_mapping_.size() - emb_map.size()
                            : 0);
        for (uint64_t i = 0; i < row_mapping_.size(); ++i) {
            if (emb_map.find(i) == emb_map.end()) {
                missing.push_back(i);
            }
        }

        // --- Phase B: infer non-seed nodes via on-the-fly k-hop sampling ----
        if (!missing.empty() && !config_.fanouts.empty()) {
            auto infer_start = std::chrono::steady_clock::now();
            auto inferred = infer_non_seed_embeddings(missing);
            auto infer_end = std::chrono::steady_clock::now();

            result.inference_ms = std::chrono::duration<double, std::milli>(
                infer_end - infer_start).count();
            result.nodes_inferred = inferred.size();

            // Merge inferred into map
            for (auto& [idx, emb] : inferred) {
                emb_map[idx] = std::move(emb);
            }
        }
    }

    // --- Phase C: write embeddings to projection as tensor properties -------
    {
        auto write_start = std::chrono::steady_clock::now();
        result.nodes_written = write_to_projection(emb_map);
        auto write_end = std::chrono::steady_clock::now();
        result.write_ms = std::chrono::duration<double, std::milli>(
            write_end - write_start).count();
    }

    return result;
}

// =============================================================================
// Phase A: collect_seed_embeddings
// =============================================================================

std::vector<std::pair<uint64_t, torch::Tensor>> EmbeddingWriter::collect_seed_embeddings() {
    // --- Put model in inference mode ----------------------------------------
    model_.eval();
    torch::NoGradGuard no_grad;

    // --- Detect model device ------------------------------------------------
    // Same pattern as TrainingLoop::evaluate() (training_loop.cc:201).
    torch::Device device(torch::kCPU);
    if (!model_.parameters().empty()) {
        device = model_.parameters().begin()->device();
    }

    // --- Iterate all batches ------------------------------------------------
    std::vector<std::pair<uint64_t, torch::Tensor>> result;

    for (uint64_t bid = 0; bid < catalog_.total_batches; ++bid) {
        // 1. Assemble MiniBatch (features + edge_indices + labels)
        MiniBatch mini = assembler_.assemble(bid);

        // 2. Read GraphSample to recover seed ObjectIds (nodes_per_layer[0])
        GraphSample sample = sample_storage_.read_sample(bid);

        const auto& seed_oids = sample.nodes_per_layer[0];
        const auto  num_seeds = static_cast<int64_t>(seed_oids.size());

        if (num_seeds == 0) {
            continue;
        }

        // 3. Move batch tensors to model device if CUDA
        if (!device.is_cpu()) {
            mini.features = mini.features.to(device);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(device);
            }
            for (auto& ai : mini.active_indices_per_layer) {
                ai = ai.to(device);
            }
        }

        // 4. Forward pass (hidden representation, NOT logits)
        auto emb = model_.get_embeddings(
            mini.features,
            mini.edge_indices,
            mini.active_sizes_per_layer
        );
        // emb shape: [num_seeds, hidden_dim]

        // 5. Move to CPU for storage
        emb = emb.cpu().contiguous();

        // 6. Map each seed embedding to its RowMapping index
        for (int64_t i = 0; i < num_seeds; ++i) {
            auto row_opt = row_mapping_.find(seed_oids[static_cast<size_t>(i)]);
            if (!row_opt) {
                // Defensive: skip seeds not found in RowMapping.
                // Should never happen with consistent data.
                continue;
            }
            result.emplace_back(*row_opt, emb[i].clone());
        }
    }

    return result;
}

// =============================================================================
// Phase B: infer_non_seed_embeddings
// =============================================================================

std::vector<std::pair<uint64_t, torch::Tensor>>
EmbeddingWriter::infer_non_seed_embeddings(const std::vector<uint64_t>& missing) {
    model_.eval();
    torch::NoGradGuard no_grad;

    // Detect model device
    torch::Device device(torch::kCPU);
    if (!model_.parameters().empty()) {
        device = model_.parameters().begin()->device();
    }

    std::vector<std::pair<uint64_t, torch::Tensor>> result;
    result.reserve(missing.size());

    // Open FeatureMatrix for node-level feature loading.
    // Cannot use the main assembler_ (FourLevelStore mode) because inference
    // batches have no pre-packed files — load_batch_features(batch_id) would fail.
    FeatureMatrix fm = FeatureMatrix::open(config_.feature_matrix_path);
    BatchAssembler fm_assembler(fm, sample_storage_, nullptr, nullptr, row_mapping_);
    auto inference_assembler = &fm_assembler;

    const uint64_t chunk_size = config_.batch_size > 0 ? config_.batch_size : 256;
    uint64_t batch_id_counter = catalog_.total_batches;  // avoid ID collision

    const uint64_t total_chunks =
        (missing.size() + chunk_size - 1) / chunk_size;
    uint64_t chunk_idx = 0;
    const auto progress_t0 = std::chrono::steady_clock::now();
    std::cerr << "[EmbeddingWriter] Phase B starting: " << missing.size()
              << " non-seed nodes in " << total_chunks
              << " chunks of " << chunk_size
              << " (device=" << (device.is_cpu() ? "cpu" : "cuda") << ")"
              << std::endl;

    // Optional per-chunk timing instrumentation (env MDB_EMBWRITER_TIMING=1).
    // Emits one line per chunk with wall-clock breakdown of each step so we
    // can diagnose O(N^2)-style growth without a profiler.
    const char* timing_env = std::getenv("MDB_EMBWRITER_TIMING");
    const bool  emit_timing = (timing_env != nullptr && timing_env[0] != '0'
                               && timing_env[0] != '\0');

    for (uint64_t start = 0; start < missing.size(); start += chunk_size) {
        uint64_t end = std::min(start + chunk_size,
                                static_cast<uint64_t>(missing.size()));

        auto t_step0 = std::chrono::steady_clock::now();

        // 1. Convert row indices to ObjectIds via RowMapping
        std::vector<ObjectId> seed_oids;
        std::vector<uint64_t> chunk_row_indices;
        seed_oids.reserve(end - start);
        chunk_row_indices.reserve(end - start);

        for (uint64_t i = start; i < end; ++i) {
            ObjectId oid = row_mapping_.get(missing[i]);
            seed_oids.push_back(oid);
            chunk_row_indices.push_back(missing[i]);
        }

        if (seed_oids.empty()) {
            continue;
        }

        auto t_step1 = std::chrono::steady_clock::now();

        // 2. Build GraphSample by k-hop sampling from the projection topology
        GraphSample sample = build_graph_sample(seed_oids, batch_id_counter++);

        auto t_step2 = std::chrono::steady_clock::now();

        // 3. Assemble MiniBatch using FeatureMatrix mode (not FourLevelStore).
        //    Inference batches have no pre-packed files in packed_slim, so
        //    FourLevelStore::load_batch_features(batch_id) would fail.
        //    Use a temporary BatchAssembler in FeatureMatrix fallback mode.
        MiniBatch mini = inference_assembler->assemble_from_sample(sample);

        auto t_step3 = std::chrono::steady_clock::now();

        // 4. Move batch tensors to model device
        if (!device.is_cpu()) {
            mini.features = mini.features.to(device);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(device);
            }
            for (auto& ai : mini.active_indices_per_layer) {
                ai = ai.to(device);
            }
        }

        auto t_step4 = std::chrono::steady_clock::now();

        // 5. Forward pass to get embeddings
        auto num_seeds = static_cast<int64_t>(seed_oids.size());
        auto emb = model_.get_embeddings(
            mini.features,
            mini.edge_indices,
            mini.active_sizes_per_layer
        );
        // emb shape: [num_seeds, hidden_dim]

        auto t_step5 = std::chrono::steady_clock::now();

        // 6. Move to CPU and collect
        emb = emb.cpu().contiguous();

        for (int64_t i = 0; i < num_seeds; ++i) {
            result.emplace_back(chunk_row_indices[static_cast<size_t>(i)],
                                emb[i].clone());
        }

        auto t_step6 = std::chrono::steady_clock::now();

        ++chunk_idx;

        if (emit_timing) {
            using ms = std::chrono::duration<double, std::milli>;
            const double t_convert   = ms(t_step1 - t_step0).count();
            const double t_build     = ms(t_step2 - t_step1).count();
            const double t_assemble  = ms(t_step3 - t_step2).count();
            const double t_to_device = ms(t_step4 - t_step3).count();
            const double t_forward   = ms(t_step5 - t_step4).count();
            const double t_to_cpu    = ms(t_step6 - t_step5).count();
            const double t_total     = ms(t_step6 - t_step0).count();
            const size_t nun         = sample.all_unique_nodes.size();
            size_t l0 = sample.nodes_per_layer.size() > 0 ? sample.nodes_per_layer[0].size() : 0;
            size_t l1 = sample.nodes_per_layer.size() > 1 ? sample.nodes_per_layer[1].size() : 0;
            size_t l2 = sample.nodes_per_layer.size() > 2 ? sample.nodes_per_layer[2].size() : 0;
            size_t e0 = sample.edges_per_layer.size() > 0 ? sample.edges_per_layer[0].size() : 0;
            size_t e1 = sample.edges_per_layer.size() > 1 ? sample.edges_per_layer[1].size() : 0;
            std::cerr << "[EmbWriter-timing] chunk=" << chunk_idx
                      << "/" << total_chunks
                      << " nseeds=" << num_seeds
                      << " nunique=" << nun
                      << " layers=" << l0 << "," << l1 << "," << l2
                      << " edges=" << e0 << "," << e1
                      << " t_convert=" << static_cast<int>(t_convert)
                      << " t_build=" << static_cast<int>(t_build)
                      << " t_assemble=" << static_cast<int>(t_assemble)
                      << " t_to_dev=" << static_cast<int>(t_to_device)
                      << " t_fwd=" << static_cast<int>(t_forward)
                      << " t_to_cpu=" << static_cast<int>(t_to_cpu)
                      << " t_total=" << static_cast<int>(t_total)
                      << "ms" << std::endl;
        }

        if (chunk_idx == 1 || chunk_idx == total_chunks
            || chunk_idx % std::max<uint64_t>(1, total_chunks / 20) == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed_s =
                std::chrono::duration<double>(now - progress_t0).count();
            const double pct = 100.0 * static_cast<double>(chunk_idx)
                                      / static_cast<double>(total_chunks);
            const double eta_s = chunk_idx > 0
                ? elapsed_s * (static_cast<double>(total_chunks - chunk_idx)
                               / static_cast<double>(chunk_idx))
                : 0.0;
            std::cerr << "[EmbeddingWriter] chunk " << chunk_idx
                      << "/" << total_chunks
                      << " (" << static_cast<int>(pct) << "%)"
                      << " elapsed=" << static_cast<int>(elapsed_s) << "s"
                      << " eta=" << static_cast<int>(eta_s) << "s"
                      << std::endl;
        }
    }

    return result;
}

// =============================================================================
// build_graph_sample -- on-the-fly k-hop sampling via TopologyAccessor
//
// Replicates the BasicKHopSampler::Impl::do_sample() algorithm using
// TopologyAccessor::get_neighbors() for per-node lookups.  This is the
// PER_NODE strategy (optimal for small inference batches).
// =============================================================================

GraphSample EmbeddingWriter::build_graph_sample(
    const std::vector<ObjectId>& seeds,
    uint64_t batch_id)
{
    GraphSample sample;
    sample.batch_id = batch_id;
    sample.split    = SplitType::TRAIN;  // irrelevant for inference

    if (seeds.empty() || config_.fanouts.empty()) {
        sample.nodes_per_layer.push_back(seeds);
        return sample;
    }

    // Build the adjacency cache (directions selected by config_.orientation)
    // on first use. Subsequent chunks re-use the cached hash map so the
    // O(|E|) scan is amortised across Phase B.
    if (!adj_cache_built_) {
        build_adjacency_cache_();
    }

    const size_t K = config_.fanouts.size();

    // nodes_per_layer[0] = seeds, nodes_per_layer[k] = k-hop neighbors
    sample.nodes_per_layer.resize(K + 1);
    sample.nodes_per_layer[0] = seeds;

    // Track sampled edges at each layer:
    // sampled_edges[k] = {dst_node_id -> [(neighbor_node, edge_id), ...]}
    std::vector<std::unordered_map<uint64_t, std::vector<std::pair<ObjectId, ObjectId>>>>
        sampled_edges(K);

    // Sample layer by layer
    for (size_t k = 0; k < K; ++k) {
        uint64_t fanout = config_.fanouts[k];
        std::unordered_set<uint64_t> next_layer_set;
        const auto& current_layer = sample.nodes_per_layer[k];

        for (const ObjectId& node_id : current_layer) {
            // Pull neighbors from the in-memory adjacency cache (Phase B
            // performance path — see EmbeddingWriter::build_adjacency_cache_
            // for rationale). This replaces the per-call B+Tree range query,
            // which is O(page_tuples) under CSR_HYBRID v3 storage and
            // dominates wall-clock time on arxiv-scale graphs.
            const auto& cached = get_neighbors_cached_(node_id.id);
            const size_t n = cached.size();
            if (n == 0) {
                continue;
            }

            const size_t f = std::min(static_cast<size_t>(fanout), n);

            std::vector<std::pair<ObjectId, ObjectId>> selected;
            selected.reserve(f);

            if (f == n) {
                // Take all neighbors
                for (size_t i = 0; i < n; ++i) {
                    selected.emplace_back(ObjectId(cached[i].node_id),
                                          ObjectId(cached[i].edge_id));
                }
            } else {
                // Fisher-Yates partial shuffle on indices
                std::vector<size_t> indices(n);
                std::iota(indices.begin(), indices.end(), 0);

                for (size_t i = 0; i < f; ++i) {
                    std::uniform_int_distribution<size_t> dist(i, n - 1);
                    size_t j = dist(rng_);
                    std::swap(indices[i], indices[j]);
                }

                for (size_t i = 0; i < f; ++i) {
                    size_t idx = indices[i];
                    selected.emplace_back(ObjectId(cached[idx].node_id),
                                          ObjectId(cached[idx].edge_id));
                }
            }

            // Record sampled edges and collect next-layer nodes
            if (!selected.empty()) {
                for (const auto& [neighbor_node, edge_id] : selected) {
                    next_layer_set.insert(neighbor_node.id);
                }
                sampled_edges[k][node_id.id] = std::move(selected);
            }
        }

        // Convert next-layer set to vector
        sample.nodes_per_layer[k + 1].reserve(next_layer_set.size());
        for (uint64_t nid : next_layer_set) {
            sample.nodes_per_layer[k + 1].emplace_back(nid);
        }
    }

    // Build edge indices (same algorithm as BasicKHopSampler::Impl::build_edges)
    size_t num_layers = sample.nodes_per_layer.size();
    sample.edges_per_layer.resize(num_layers - 1);

    // Build node_id -> local_index mapping for each layer
    std::vector<std::unordered_map<uint64_t, int32_t>> layer_mappings(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        const auto& nodes = sample.nodes_per_layer[layer];
        for (size_t i = 0; i < nodes.size(); ++i) {
            layer_mappings[layer][nodes[i].id] = static_cast<int32_t>(i);
        }
    }

    // Build edges for each layer connection
    // edges_per_layer[k] connects layer k+1 (src) to layer k (dst)
    for (size_t k = 0; k < num_layers - 1; ++k) {
        auto& edges = sample.edges_per_layer[k];

        // sampled_edges[k] maps dst_node -> [(neighbor_node, edge_id), ...]
        // dst is in layer k, src (neighbors) are in layer k+1
        for (const auto& [dst_id, neighbor_edges] : sampled_edges[k]) {
            auto dst_it = layer_mappings[k].find(dst_id);
            if (dst_it == layer_mappings[k].end()) continue;
            int32_t dst_idx = dst_it->second;

            for (const auto& [src_node, edge_id] : neighbor_edges) {
                auto src_it = layer_mappings[k + 1].find(src_node.id);
                if (src_it == layer_mappings[k + 1].end()) continue;
                int32_t src_idx = src_it->second;

                edges.src_indices.push_back(src_idx);
                edges.dst_indices.push_back(dst_idx);
                edges.edge_ids.push_back(edge_id);
            }
        }
    }

    // Build all_unique_nodes
    sample.rebuild_unique_nodes();

    return sample;
}

// =============================================================================
// Phase C: write_to_projection
//
// Persists all embeddings as tensor node properties in the projection.
//
// Strategy: Since the projection is already built (B+Trees finalized), we
// cannot use ProjectionStorage::add_node_property() which buffers for bulk
// build.  Instead, we insert directly into the B+Trees via
// BPlusTree<3>::insert().
//
// If the projection was built without node property indexes
// (include_node_properties=false), we create empty B+Trees first so that
// subsequent queries can find the embedding property.
// =============================================================================

uint64_t EmbeddingWriter::next_available_key_id(
    const std::unordered_map<std::string, uint64_t>& node_keys,
    const std::unordered_map<std::string, uint64_t>& edge_keys)
{
    // Start at the synthetic base (mirroring NativeProjectionBuilder's
    // RENAME_KEY_SYNTHETIC_START) and go one past every id already in use.
    // Both namespaces are scanned because the builder allocates node and
    // edge synthetic ids from a single shared counter.
    uint64_t next_id = EMBEDDING_KEY_SYNTHETIC_BASE;
    for (const auto& [name, id] : node_keys) {
        if (id >= next_id) {
            next_id = id + 1;
        }
    }
    for (const auto& [name, id] : edge_keys) {
        if (id >= next_id) {
            next_id = id + 1;
        }
    }
    return next_id;
}

uint64_t EmbeddingWriter::resolve_property_key_id(
    GQL::ProjectionStorage& storage,
    const std::string&      property_name)
{
    // ProjectionStorage::open() restores statistics from the on-disk catalog
    // but not the property key maps, so a storage opened from disk starts
    // with empty key maps even when earlier sessions registered keys.
    // Re-sync them before allocating: otherwise an existing property would
    // be re-allocated under a colliding id, and the save_catalog() at flush
    // time would drop every previously persisted key from the catalog.
    // register_node_key / register_edge_key keep the first binding for a
    // name, so in-memory registrations from this session always win.
    const std::filesystem::path catalog_file =
        std::filesystem::path(storage.get_projection_dir()) / "catalog.dat";
    if (std::filesystem::exists(catalog_file)) {
        GQL::ProjectionCatalog catalog(storage.get_projection_dir());
        catalog.load();
        for (const auto& [name, id] : catalog.node_keys2id) {
            storage.register_node_key(name, id);
        }
        for (const auto& [name, id] : catalog.edge_keys2id) {
            storage.register_edge_key(name, id);
        }
    }

    // Reuse the existing binding when the property was already registered.
    auto existing = storage.get_node_key_id(property_name);
    if (existing) {
        return *existing;
    }

    // Allocate a fresh id strictly above every id in use, refusing to bind
    // an id that already belongs to a different property (two names on one
    // key id would make their values indistinguishable at query time).
    const uint64_t key_id_raw = next_available_key_id(
        storage.get_node_keys(), storage.get_edge_keys());

    for (const auto& [name, id] : storage.get_node_keys()) {
        if (id == key_id_raw) {
            throw std::runtime_error(
                "EmbeddingWriter: key id " + std::to_string(key_id_raw)
                + " for property '" + property_name
                + "' is already bound to property '" + name + "'");
        }
    }

    storage.register_node_key(property_name, key_id_raw);

    // register_node_key silently ignores a name that is already present, so
    // verify the binding actually took the id we allocated.
    auto bound = storage.get_node_key_id(property_name);
    if (!bound || *bound != key_id_raw) {
        throw std::runtime_error(
            "EmbeddingWriter: failed to register property key '"
            + property_name + "' under id " + std::to_string(key_id_raw));
    }

    return key_id_raw;
}

uint64_t EmbeddingWriter::write_to_projection(
    const std::unordered_map<uint64_t, torch::Tensor>& emb_map)
{
    if (emb_map.empty()) {
        return 0;
    }

    // -------------------------------------------------------------------------
    // Step 1: Resolve (or allocate + register) the property key
    // -------------------------------------------------------------------------

    uint64_t key_id_raw = resolve_property_key_id(projection_storage_,
                                                  config_.property_name);

    ObjectId key_oid(key_id_raw | ObjectId::MASK_NODE_KEY);

    // -------------------------------------------------------------------------
    // Step 2: Ensure node property B+Tree indexes exist
    //
    // If the projection was built without include_node_properties, the
    // node_key_value and key_value_node indexes will be nullptr.  Create
    // empty B+Trees so we can insert into them.
    // -------------------------------------------------------------------------

    // Ensure node property B+Trees exist — creates empty indexes if the
    // projection was built without properties (e.g., STRING syntax).
    projection_storage_.ensure_node_property_indexes();

    auto* nkv_index = projection_storage_.get_node_key_value_index();
    auto* kvn_index = projection_storage_.get_key_value_node_index();

    // Persist the catalog (carrying the key registered above) BEFORE any
    // property record is inserted: pages dirtied by a partially-completed
    // loop can still reach disk via the buffer pool, and records under a
    // key id that never made it into catalog.dat would be unresolvable.
    // flush() is idempotent — it only rebuilds indexes when streaming
    // buffers have pending records (they don't; we use direct inserts).
    projection_storage_.flush();

    // -------------------------------------------------------------------------
    // Step 3: For each embedding, serialize -> TensorManager -> B+Tree insert
    // -------------------------------------------------------------------------

    uint64_t written = 0;

    try {
        for (const auto& [row_index, emb_tensor] : emb_map) {
            // 3a. Get node ObjectId from RowMapping
            ObjectId node_oid = row_mapping_.get(row_index);
            if (node_oid.is_null()) {
                // Defensive: skip invalid row indices
                continue;
            }

            // 3b. Serialize embedding to contiguous float bytes
            torch::Tensor emb_cpu = emb_tensor.cpu().contiguous().to(torch::kFloat32);
            const auto* bytes = reinterpret_cast<const char*>(emb_cpu.data_ptr<float>());
            size_t num_bytes = static_cast<size_t>(emb_cpu.numel()) * sizeof(float);

            // 3c. Store in TensorManager (deduplicates identical tensors)
            uint64_t tensor_id = tensor_manager.get_or_create_id(bytes, num_bytes);

            // 3d. Build tensor ObjectId (float extern)
            ObjectId tensor_oid(ObjectId::MASK_TENSOR_FLOAT_EXTERN | tensor_id);

            // 3e. Delete any existing records for this (node, key) BEFORE
            //     inserting.  get_node_property() returns the first record in
            //     the (node, key) range, so a leftover record from a previous
            //     write (smaller tensor id sorts first) would shadow the new
            //     value.  Property indexes are v1-mutable, so delete_record is
            //     supported.  Stale values are collected first so the range
            //     iterator is destroyed before the tree is mutated.
            std::vector<uint64_t> stale_value_ids;
            {
                Record<3> min_record = { node_oid.id, key_oid.id, 0 };
                Record<3> max_record = { node_oid.id, key_oid.id, UINT64_MAX };

                bool interruption_requested = false;
                auto iter = nkv_index->get_range(&interruption_requested,
                                                 min_record, max_record);
                const Record<3>* rec;
                while ((rec = iter.next()) != nullptr) {
                    stale_value_ids.push_back((*rec)[2]);
                }
            }
            //     A record carrying exactly the value we are about to write is
            //     not stale, it is already the target state.  Remember that so
            //     the inserts below can tell "already present" (legitimate on
            //     a re-run) apart from "refused", which is a defect.
            bool already_present = false;
            for (uint64_t stale_value_id : stale_value_ids) {
                if (stale_value_id == tensor_oid.id) {
                    already_present = true;
                    continue;  // identical record; the insert below is a no-op
                }
                Record<3> stale_nkv = { node_oid.id, key_oid.id, stale_value_id };
                nkv_index->delete_record(stale_nkv);

                Record<3> stale_kvn = { key_oid.id, stale_value_id, node_oid.id };
                kvn_index->delete_record(stale_kvn);
            }

            // 3f. Insert into both B+Tree property indexes, back-to-back so a
            //     failure leaves at most the FINAL (node, key) pair ragged
            //     (nkv inserted, kvn not).  The delete-before-insert pass in
            //     3e repairs such a pair on re-run.
            //     Primary: (node_id, key_id, value_id) -- for node property lookups
            Record<3> nkv_record;
            nkv_record[0] = node_oid.id;
            nkv_record[1] = key_oid.id;
            nkv_record[2] = tensor_oid.id;
            const bool nkv_inserted = nkv_index->insert(nkv_record);

            //     Auxiliary: (key_id, value_id, node_id) -- for property -> node lookups
            Record<3> kvn_record;
            kvn_record[0] = key_oid.id;
            kvn_record[1] = tensor_oid.id;
            kvn_record[2] = node_oid.id;
            const bool kvn_inserted = kvn_index->insert(kvn_record);

            //     BPlusTree::insert returns false only when the identical
            //     record was already in the tree.  After the pass above that
            //     can be true only for the record we deliberately kept, so any
            //     other refusal is a write that did NOT happen -- surface it
            //     instead of counting it.  `written` is the value reported as
            //     nodesWritten, and it must never exceed what is on disk.
            if ((!nkv_inserted || !kvn_inserted) && !already_present) {
                throw std::runtime_error(
                    "property record refused by the index for node "
                    + std::to_string(node_oid.id)
                    + " (node_key_value inserted="
                    + (nkv_inserted ? "true" : "false")
                    + ", key_value_node inserted="
                    + (kvn_inserted ? "true" : "false") + ")");
            }

            ++written;
        }
    } catch (const std::exception& e) {
        // Persist the records inserted before the failure so the on-disk
        // B+Trees match the already-saved catalog, then surface the partial
        // progress so the user knows a re-run is needed (re-runs are
        // self-healing via the 3e upsert).
        try {
            projection_storage_.flush();
        } catch (...) {
            // keep the original error — it names the actionable failure
        }
        throw std::runtime_error(
            "EmbeddingWriter::write_to_projection: failed after writing "
            + std::to_string(written) + " of "
            + std::to_string(emb_map.size()) + " embeddings: " + e.what());
    }

    // -------------------------------------------------------------------------
    // Step 4: Persist catalog with new key mapping
    //
    // save_catalog() requires projection_name to be non-empty.  When the
    // storage was opened via the read-only constructor, projection_name may
    // be empty.  In that case we skip catalog persistence — the key mapping
    // is already live in memory for the current session, and the B+Tree
    // records are persisted via the buffer pool.
    // -------------------------------------------------------------------------
    // Trigger catalog save by calling flush() which internally calls
    // save_catalog().  flush() is safe to call multiple times — it only
    // rebuilds indexes when streaming buffers have pending records (which
    // they won't, since we used direct insert).
    projection_storage_.flush();

    // -------------------------------------------------------------------------
    // Step 5: Structural verification of both property indexes
    //
    // Every record above went in through BPlusTree::insert one at a time, so
    // both trees took the directory-split path -- and key_value_node takes it
    // with keys in ASCENDING order (its leading columns are the property key
    // and the tensor id, and tensor ids are monotonically growing blob
    // offsets), which makes "the child that splits is the rightmost child" the
    // normal case rather than a rare one.
    //
    // check() verifies, for every directory entry in both trees,
    //
    //     greatest_left_record < separator <= smallest_right_record
    //
    // A separator that violates it leaves records physically present but
    // unreachable by descent: the procedure would report every node written
    // while `n.embedding IS NULL` holds for some of them, and a lookup landing
    // past the end of the left leaf would follow next_leaf and answer with
    // ANOTHER node's tensor.  Because that invariant is exactly what makes a
    // record findable, check() passing is what licenses `written` below to be
    // reported as nodesWritten.  Refuse to report a number we cannot stand
    // behind.
    // -------------------------------------------------------------------------
    {
        std::ostringstream nkv_report;
        std::ostringstream kvn_report;
        const bool nkv_sane = nkv_index->check(nkv_report);
        const bool kvn_sane = kvn_index->check(kvn_report);

        if (!nkv_sane || !kvn_sane) {
            throw std::runtime_error(
                "EmbeddingWriter::write_to_projection: property index corrupted "
                "after writing " + std::to_string(written) + " embeddings for '"
                + config_.property_name + "'. Records are on disk but not all of "
                "them are reachable by lookup, so the write must not be reported "
                "as complete; the projection's property indexes need to be "
                "rebuilt. node_key_value: "
                + (nkv_sane ? std::string("ok") : nkv_report.str())
                + " key_value_node: "
                + (kvn_sane ? std::string("ok") : kvn_report.str()));
        }
    }

    return written;
}

// =============================================================================
// Lazy TopologyAccessor
// =============================================================================

TopologyAccessor& EmbeddingWriter::get_topology_() {
    if (!topology_) {
        topology_ = std::make_unique<TopologyAccessor>(projection_storage_);
    }
    return *topology_;
}

// =============================================================================
// Adjacency cache (Phase B performance path)
// =============================================================================

void EmbeddingWriter::build_adjacency_cache_() {
    if (adj_cache_built_) {
        return;
    }

    const auto t_start = std::chrono::steady_clock::now();

    auto* fwd_index = projection_storage_.get_from_to_edge_index();
    auto* rev_index = projection_storage_.get_to_from_edge_index();

    // Reserve a generous bucket count up-front. We size against the node
    // count from RowMapping because every non-isolated node will become a
    // key in the adjacency map; load factor headroom avoids mid-scan
    // rehashes that would serialize against the streaming inserts below.
    adj_cache_.reserve(row_mapping_.size());

    auto scan_and_merge = [&](BPlusTree<3>* index) {
        if (!index) return;
        Record<3> min_record = {0, 0, 0};
        Record<3> max_record = {UINT64_MAX, UINT64_MAX, UINT64_MAX};
        bool interruption_requested = false;
        auto it = index->get_range(&interruption_requested,
                                   min_record, max_record);
        const Record<3>* rec;
        while ((rec = it.next()) != nullptr) {
            const uint64_t a = std::get<0>(*rec);
            const uint64_t b = std::get<1>(*rec);
            const uint64_t e = std::get<2>(*rec);
            adj_cache_[a].push_back({b, e});
        }
    };

    // from_to_edge stores (from, to, edge_id) and to_from_edge stores
    // (to, from, edge_id), both keyed on "this endpoint": the forward scan
    // contributes out-neighbors, the reverse scan in-neighbors.
    // config_.orientation selects which directions Phase B sampling may
    // traverse, so inference neighborhoods match the neighbor semantics the
    // model was trained on.
    if (config_.orientation == EdgeOrientation::NATURAL
        || config_.orientation == EdgeOrientation::UNDIRECTED)
    {
        scan_and_merge(fwd_index);
    }
    if (config_.orientation == EdgeOrientation::REVERSE
        || config_.orientation == EdgeOrientation::UNDIRECTED)
    {
        scan_and_merge(rev_index);
    }

    adj_cache_built_ = true;

    // Report cache size + build time so large-graph runs (products) can be
    // diagnosed without re-running under instrumentation. Emitted once per
    // Phase B.
    const auto t_end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();
    uint64_t total_edges = 0;
    for (const auto& [k, v] : adj_cache_) {
        total_edges += v.size();
    }
    const char* orient_name =
        config_.orientation == EdgeOrientation::NATURAL    ? "NATURAL"
        : config_.orientation == EdgeOrientation::REVERSE  ? "REVERSE"
                                                           : "UNDIRECTED";
    std::cerr << "[EmbeddingWriter] adjacency cache built ("
              << orient_name << "): "
              << adj_cache_.size() << " nodes, "
              << total_edges << " directed entries, "
              << static_cast<int>(ms) << " ms"
              << std::endl;
}

const std::vector<EmbeddingWriter::AdjEntry>&
EmbeddingWriter::get_neighbors_cached_(uint64_t node_id) const
{
    auto it = adj_cache_.find(node_id);
    if (it == adj_cache_.end()) {
        return adj_empty_sentinel_;
    }
    return it->second;
}

} // namespace mdb::gnn
