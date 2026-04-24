// Include tensor_manager.h FIRST, before any header that transitively pulls
// in <linux/io_uring.h> (via liburing.h -> linux/fs.h) which #defines
// BLOCK_SIZE as a macro, conflicting with TensorManager::BLOCK_SIZE.
#include "system/tensor_manager.h"

#include "gnn/output/embedding_writer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "gnn/storage/feature_matrix.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
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

        // 3. Assemble MiniBatch using FeatureMatrix mode (not FourLevelStore).
        //    Inference batches have no pre-packed files in packed_slim, so
        //    FourLevelStore::load_batch_features(batch_id) would fail.
        //    Use a temporary BatchAssembler in FeatureMatrix fallback mode.
        MiniBatch mini = inference_assembler->assemble_from_sample(sample);

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

        ++chunk_idx;
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

/// Synthetic key ID for GNN embedding properties.  Chosen to be well above
/// any realistic number of schema-defined node property keys (mirroring the
/// COUNT_KEY_SYNTHETIC_ID=1000 pattern from NativeProjectionBuilder) while
/// avoiding collision with it.
static constexpr uint64_t EMBEDDING_KEY_SYNTHETIC_ID = 2000;

uint64_t EmbeddingWriter::write_to_projection(
    const std::unordered_map<uint64_t, torch::Tensor>& emb_map)
{
    if (emb_map.empty()) {
        return 0;
    }

    // -------------------------------------------------------------------------
    // Step 1: Register the property key in the projection catalog
    // -------------------------------------------------------------------------

    // Check if the key already exists in the projection's node key map.
    uint64_t key_id_raw = EMBEDDING_KEY_SYNTHETIC_ID;
    auto existing = projection_storage_.get_node_key_id(config_.property_name);
    if (existing) {
        key_id_raw = *existing;
    } else {
        // Register a new synthetic key for this embedding property.
        projection_storage_.register_node_key(config_.property_name, key_id_raw);
    }

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

    // -------------------------------------------------------------------------
    // Step 3: For each embedding, serialize -> TensorManager -> B+Tree insert
    // -------------------------------------------------------------------------

    uint64_t written = 0;

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

        // 3e. Insert into both B+Tree property indexes
        //     Primary: (node_id, key_id, value_id) -- for node property lookups
        Record<3> nkv_record;
        nkv_record[0] = node_oid.id;
        nkv_record[1] = key_oid.id;
        nkv_record[2] = tensor_oid.id;
        nkv_index->insert(nkv_record);

        //     Auxiliary: (key_id, value_id, node_id) -- for property -> node lookups
        Record<3> kvn_record;
        kvn_record[0] = key_oid.id;
        kvn_record[1] = tensor_oid.id;
        kvn_record[2] = node_oid.id;
        kvn_index->insert(kvn_record);

        ++written;
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

    return written;
}

} // namespace mdb::gnn
