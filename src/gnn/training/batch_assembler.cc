#include "gnn/training/batch_assembler.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace mdb::gnn {

// =============================================================================
// Constructors
// =============================================================================

BatchAssembler::BatchAssembler(
    FourLevelStore& feature_store,
    SampleStorage&  samples,
    LabelStore*     labels,
    SplitStore*     splits,
    const RowMapping& row_mapping
)
    : feature_store_(&feature_store)
    , feature_matrix_(nullptr)
    , samples_(samples)
    , labels_(labels)
    , splits_(splits)
    , row_mapping_(row_mapping)
{}

BatchAssembler::BatchAssembler(
    const FeatureMatrix& feature_matrix,
    SampleStorage&       samples,
    LabelStore*          labels,
    SplitStore*          splits,
    const RowMapping&    row_mapping
)
    : feature_store_(nullptr)
    , feature_matrix_(&feature_matrix)
    , samples_(samples)
    , labels_(labels)
    , splits_(splits)
    , row_mapping_(row_mapping)
{}

// =============================================================================
// Public: assemble(batch_id)
// =============================================================================

MiniBatch BatchAssembler::assemble(uint64_t batch_id) {
    auto t0 = std::chrono::steady_clock::now();
    GraphSample sample = samples_.read_sample(batch_id);
    auto t1 = std::chrono::steady_clock::now();

    MiniBatch mini = assemble_from_sample(sample);
    mini.timing.sample_read_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    return mini;
}

// =============================================================================
// Public: assemble_from_sample(sample)
// =============================================================================

MiniBatch BatchAssembler::assemble_from_sample(const GraphSample& sample) {
    MiniBatch mini;
    mini.batch_id = sample.batch_id;
    mini.split    = sample.split;

    // Step 1: Build global index map (ObjectId -> position in all_unique_nodes)
    std::unordered_map<uint64_t, int64_t> oid_to_global;
    oid_to_global.reserve(sample.all_unique_nodes.size());
    for (int64_t i = 0; i < static_cast<int64_t>(sample.all_unique_nodes.size()); ++i) {
        oid_to_global[sample.all_unique_nodes[i].id] = i;
    }

    // Step 2: Build per-layer active-set indices for the active-set-shrinking
    // model refactor — see plan ~/Desktop/2026-05-14-graphsage-active-set-shrinking-plan.md.
    // Must precede build_edge_indices so the edge tensors can be remapped from
    // global positions into local positions within each active set.
    //
    // Round 2C (2026-05-15): build_active_indices now also emits a per-layer
    // ObjectId.id -> local-A_k-index hash table, which build_edge_indices
    // uses directly to halve the per-edge hash count (one lookup per
    // endpoint instead of two).
    auto t_active_start = std::chrono::steady_clock::now();
    auto active = build_active_indices(sample, oid_to_global);
    auto t_active_end = std::chrono::steady_clock::now();
    mini.active_indices_per_layer = std::move(active.indices_per_layer);
    mini.active_sizes_per_layer   = std::move(active.sizes_per_layer);
    mini.timing.active_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_active_end - t_active_start).count());

    // Step 3: Load features
    // Round 2B (2026-05-15): pass the already-deserialized sample so the
    // FourLevelStore path skips re-reading the same ~55 MB sample file.
    auto t_load_start = std::chrono::steady_clock::now();
    mini.features = load_features(sample);
    auto t_load_end = std::chrono::steady_clock::now();
    mini.timing.assembler_kernel_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_load_end - t_load_start).count());

    // STEP 6 (2026-05-31): capture the v2 addr-table dispatch result for THIS
    // batch right after the load, while feature_store_->last_used_addr_tables()
    // still refers to it. Carried on the MiniBatch so TrainingLoop reports
    // correct v2 telemetry on the async-prefetcher path (the worker stamps
    // here; the consumer reads after next()).
    if (feature_store_) {
        mini.timing.used_addr_tables = feature_store_->last_used_addr_tables();
        mini.timing.addr_load_ns     = feature_store_->last_addr_load_us() * 1000ULL;
    }

    // Step 4: Build edge indices per layer (LOCAL to active sets).
    auto t_edge_start = std::chrono::steady_clock::now();
    mini.edge_indices = build_edge_indices(sample, active.oid_to_local_per_layer);
    auto t_edge_end = std::chrono::steady_clock::now();
    mini.timing.edge_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_edge_end - t_edge_start).count());

    // Step 4: Gather labels for seed nodes (layer 0)
    int64_t num_seeds = static_cast<int64_t>(
        sample.nodes_per_layer.empty() ? 0 : sample.nodes_per_layer[0].size()
    );
    mini.num_seeds = static_cast<uint64_t>(num_seeds);
    mini.num_nodes = sample.all_unique_nodes.size();

    if (labels_ && num_seeds > 0) {
        std::vector<uint64_t> seed_row_indices;
        seed_row_indices.reserve(static_cast<size_t>(num_seeds));
        for (const auto& oid : sample.nodes_per_layer[0]) {
            auto row = row_mapping_.find(oid);
            if (row) {
                seed_row_indices.push_back(*row);
            } else {
                // Unknown node — use sentinel that LabelStore will treat as -1
                seed_row_indices.push_back(std::numeric_limits<uint64_t>::max());
            }
        }
        mini.labels     = labels_->gather(seed_row_indices);
        mini.label_mask = (mini.labels != -1);
    } else {
        mini.labels     = torch::zeros({num_seeds}, torch::kInt64);
        mini.label_mask = torch::zeros({num_seeds}, torch::kBool);
    }

    // Round 1E (2026-05-15): CPU-side count of labeled seeds. label_mask is
    // still on CPU at this point (device transfer happens in TrainingLoop),
    // so reading it costs only an O(num_seeds) memory scan and avoids the
    // per-batch `label_mask.any().item<bool>()` GPU→CPU sync.
    mini.num_labeled = 0;
    if (mini.label_mask.numel() > 0) {
        auto mask_acc = mini.label_mask.accessor<bool, 1>();
        for (int64_t i = 0; i < mini.label_mask.size(0); ++i) {
            if (mask_acc[i]) mini.num_labeled++;
        }
    }

    return mini;
}

// =============================================================================
// Private: build_active_indices
// =============================================================================

BatchAssembler::ActiveIndicesResult
BatchAssembler::build_active_indices(
    const GraphSample& sample,
    const std::unordered_map<uint64_t, int64_t>& oid_to_global)
{
    // K+1 active sets for K-layer sample: A_0 .. A_K.
    const size_t K = sample.edges_per_layer.size();
    ActiveIndicesResult out;
    out.indices_per_layer.reserve(K + 1);
    out.sizes_per_layer.reserve(K + 1);
    out.oid_to_local_per_layer.reserve(K + 1);

    // Accumulating active set. By the rebuild_unique_nodes ordering
    // invariant, layer-k nodes always have higher positions than
    // layer-(k-1) nodes (modulo cross-layer duplicates which are
    // already deduped during rebuild). So adding layer k's positions
    // in order produces a monotonically non-decreasing sequence —
    // no explicit sort needed for the prefix invariant to hold.
    std::vector<int64_t>         active_positions;
    std::unordered_set<int64_t>  active_set;
    active_positions.reserve(sample.all_unique_nodes.size());
    active_set.reserve(sample.all_unique_nodes.size());

    for (size_t k = 0; k <= K; ++k) {
        // Add nodes_per_layer[k] to the cumulative active set.
        if (k < sample.nodes_per_layer.size()) {
            for (const auto& oid : sample.nodes_per_layer[k]) {
                auto it = oid_to_global.find(oid.id);
                if (it == oid_to_global.end()) {
                    throw std::runtime_error(
                        "BatchAssembler::build_active_indices: node at layer " +
                        std::to_string(k) + " not in all_unique_nodes (oid=" +
                        std::to_string(oid.id) + ")"
                    );
                }
                if (active_set.insert(it->second).second) {
                    active_positions.push_back(it->second);
                }
            }
        }

        // Sort for deterministic order (matches the model's gather pattern;
        // also defensive in case future sampler changes break the
        // monotone-insertion property).
        std::sort(active_positions.begin(), active_positions.end());

        // Build tensor for A_k.
        auto t = torch::empty(
            {static_cast<int64_t>(active_positions.size())},
            torch::kInt64
        );
        if (!active_positions.empty()) {
            std::memcpy(t.data_ptr<int64_t>(),
                        active_positions.data(),
                        active_positions.size() * sizeof(int64_t));
        }
        out.sizes_per_layer.push_back(static_cast<int64_t>(active_positions.size()));
        out.indices_per_layer.push_back(std::move(t));

        // Round 2C (2026-05-15): build the ObjectId.id -> local-A_k-index
        // hash table for this layer. Walk active_positions (= sorted global
        // positions in A_k) and pull the ObjectId from
        // sample.all_unique_nodes[global_pos]. The resulting map lets
        // build_edge_indices skip the oid->global->local two-hop dance.
        std::unordered_map<uint64_t, int64_t> oid_to_local;
        oid_to_local.reserve(active_positions.size());
        for (int64_t local_idx = 0;
             local_idx < static_cast<int64_t>(active_positions.size());
             ++local_idx)
        {
            const int64_t global_pos = active_positions[static_cast<size_t>(local_idx)];
            oid_to_local[sample.all_unique_nodes[static_cast<size_t>(global_pos)].id] =
                local_idx;
        }
        out.oid_to_local_per_layer.push_back(std::move(oid_to_local));
    }

    return out;
}

// =============================================================================
// Private: build_edge_indices
// =============================================================================

std::vector<torch::Tensor> BatchAssembler::build_edge_indices(
    const GraphSample& sample,
    const std::vector<std::unordered_map<uint64_t, int64_t>>& oid_to_local_per_layer)
{
    std::vector<torch::Tensor> result;
    const size_t num_layers = sample.edges_per_layer.size();
    result.reserve(num_layers);

    // Round 2C (2026-05-15): oid_to_local_per_layer[k] is already built by
    // build_active_indices and maps each ObjectId.id present in A_k to its
    // local index. One hash lookup per edge endpoint instead of two —
    // half the hash work versus the prior oid->global->local two-hop path.
    for (size_t k = 0; k < num_layers; ++k) {
        const LayerEdges& edges = sample.edges_per_layer[k];
        const int64_t E = static_cast<int64_t>(edges.size());

        const auto& src_map = oid_to_local_per_layer[k + 1];  // A_{k+1}
        const auto& dst_map = oid_to_local_per_layer[k];      // A_k

        auto edge_index = torch::empty({2, E}, torch::kInt64);
        auto acc = edge_index.accessor<int64_t, 2>();

        for (int64_t i = 0; i < E; ++i) {
            // Resolve OIDs via nodes_per_layer.
            const ObjectId src_oid = sample.nodes_per_layer[k + 1][
                static_cast<size_t>(edges.src_indices[i])
            ];
            const ObjectId dst_oid = sample.nodes_per_layer[k][
                static_cast<size_t>(edges.dst_indices[i])
            ];

            // Single hash lookup per endpoint: ObjectId.id -> local index in A_k.
            auto src_it = src_map.find(src_oid.id);
            auto dst_it = dst_map.find(dst_oid.id);
            if (src_it == src_map.end()) {
                throw std::runtime_error(
                    "BatchAssembler::build_edge_indices: src node not in A_" +
                    std::to_string(k + 1) + " active set at layer " +
                    std::to_string(k) + ", edge " + std::to_string(i)
                );
            }
            if (dst_it == dst_map.end()) {
                throw std::runtime_error(
                    "BatchAssembler::build_edge_indices: dst node not in A_" +
                    std::to_string(k) + " active set at layer " +
                    std::to_string(k) + ", edge " + std::to_string(i)
                );
            }

            acc[0][i] = src_it->second;  // src local in A_{k+1}
            acc[1][i] = dst_it->second;  // dst local in A_k
        }
        result.push_back(std::move(edge_index));
    }

    return result;
}

// =============================================================================
// Private: load_features
// =============================================================================

torch::Tensor BatchAssembler::load_features(const GraphSample& sample) {
    if (feature_store_) {
        // Full mode: FourLevelStore handles all four tiers.
        // Round 2B (2026-05-15): pass the GraphSample directly so the store
        // does not re-read it from disk inside load_batch_features.
        return feature_store_->load_batch_features(sample);
    }

    // Fallback mode: FeatureMatrix + RowMapping
    const auto& unique_nodes = sample.all_unique_nodes;
    std::vector<uint64_t> row_ids;
    row_ids.reserve(unique_nodes.size());
    for (const auto& oid : unique_nodes) {
        auto row = row_mapping_.find(oid);
        if (!row) {
            throw std::runtime_error(
                "BatchAssembler: node not in RowMapping: " + std::to_string(oid.id)
            );
        }
        row_ids.push_back(*row);
    }

    const size_t N          = row_ids.size();
    const size_t D          = feature_matrix_->num_cols();
    const size_t elem_bytes = feature_matrix_->row_bytes();  // D * sizeof(element)

    std::vector<char> buffer(N * elem_bytes);
    feature_matrix_->extract_rows(row_ids, buffer.data());

    // Determine LibTorch scalar type from stored GnnDtype
    torch::ScalarType scalar_type = torch::kFloat32;
    switch (feature_matrix_->dtype()) {
        case GnnDtype::FLOAT32: scalar_type = torch::kFloat32; break;
        case GnnDtype::FLOAT64: scalar_type = torch::kFloat64; break;
        case GnnDtype::INT32:   scalar_type = torch::kInt32;   break;
        case GnnDtype::INT64:   scalar_type = torch::kInt64;   break;
        case GnnDtype::UINT8:   scalar_type = torch::kUInt8;   break;
        case GnnDtype::BOOL:    scalar_type = torch::kBool;    break;
        default:
            throw std::runtime_error(
                "BatchAssembler: unsupported GnnDtype in FeatureMatrix fallback"
            );
    }

    // .clone() is required because buffer lives on the stack
    return torch::from_blob(
        buffer.data(),
        {static_cast<int64_t>(N), static_cast<int64_t>(D)},
        scalar_type
    ).clone();
}

} // namespace mdb::gnn
