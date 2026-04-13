#include "gnn/output/embedding_writer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/torch.h>

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
    , topology_(projection_storage)
    , rng_(42)  // deterministic seed for reproducible inference
{
}

// =============================================================================
// write_all -- full orchestrator (Phases A + B; C is a stub)
// =============================================================================

EmbeddingWriter::Result EmbeddingWriter::write_all() {
    Result result;

    // --- Phase A: collect seed embeddings from pre-computed batches ----------
    auto seed_embs = collect_seed_embeddings();

    // Deduplicate: if a node was a seed in multiple batches, keep last
    std::unordered_map<uint64_t, torch::Tensor> emb_map;
    emb_map.reserve(seed_embs.size());
    for (auto& [idx, emb] : seed_embs) {
        emb_map[idx] = std::move(emb);
    }

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

    // --- Phase B: infer non-seed nodes via on-the-fly k-hop sampling --------
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

    // --- Phase C (Task 6 stub): write to projection -------------------------
    result.nodes_written = emb_map.size();
    // TODO: write_to_projection(emb_map) -- Task 6

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
        }

        // 4. Forward pass (hidden representation, NOT logits)
        auto emb = model_.get_embeddings(
            mini.features,
            mini.edge_indices,
            num_seeds
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

    const uint64_t chunk_size = config_.batch_size > 0 ? config_.batch_size : 256;
    uint64_t batch_id_counter = catalog_.total_batches;  // avoid ID collision

    for (uint64_t start = 0; start < missing.size(); start += chunk_size) {
        uint64_t end = std::min(start + chunk_size,
                                static_cast<uint64_t>(missing.size()));

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

        // 2. Build GraphSample by k-hop sampling from the projection topology
        GraphSample sample = build_graph_sample(seed_oids, batch_id_counter++);

        // 3. Assemble MiniBatch from the on-the-fly sample
        MiniBatch mini = assembler_.assemble_from_sample(sample);

        // 4. Move batch tensors to model device
        if (!device.is_cpu()) {
            mini.features = mini.features.to(device);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(device);
            }
        }

        // 5. Forward pass to get embeddings
        auto num_seeds = static_cast<int64_t>(seed_oids.size());
        auto emb = model_.get_embeddings(
            mini.features,
            mini.edge_indices,
            num_seeds
        );
        // emb shape: [num_seeds, hidden_dim]

        // 6. Move to CPU and collect
        emb = emb.cpu().contiguous();

        for (int64_t i = 0; i < num_seeds; ++i) {
            result.emplace_back(chunk_row_indices[static_cast<size_t>(i)],
                                emb[i].clone());
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
            // Get all neighbors with the configured orientation
            Neighbors all_neighbors = topology_.get_neighbors(
                node_id, config_.orientation);

            if (all_neighbors.node_ids.empty()) {
                continue;
            }

            // Uniform sampling via Fisher-Yates partial shuffle
            size_t n = all_neighbors.node_ids.size();
            size_t f = std::min(static_cast<size_t>(fanout), n);

            std::vector<std::pair<ObjectId, ObjectId>> selected;
            selected.reserve(f);

            if (f == n) {
                // Take all neighbors
                for (size_t i = 0; i < n; ++i) {
                    selected.emplace_back(all_neighbors.node_ids[i],
                                          all_neighbors.edge_ids[i]);
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
                    selected.emplace_back(all_neighbors.node_ids[idx],
                                          all_neighbors.edge_ids[idx]);
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

} // namespace mdb::gnn
