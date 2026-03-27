#include "gnn/training/batch_assembler.h"

#include <cstring>
#include <stdexcept>
#include <string>

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
    GraphSample sample = samples_.read_sample(batch_id);
    return assemble_from_sample(sample);
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

    // Step 2: Load features
    mini.features = load_features(sample.all_unique_nodes, sample.batch_id);

    // Step 3: Build edge indices per layer
    mini.edge_indices = build_edge_indices(sample, oid_to_global);

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

    return mini;
}

// =============================================================================
// Private: build_edge_indices
// =============================================================================

std::vector<torch::Tensor> BatchAssembler::build_edge_indices(
    const GraphSample& sample,
    const std::unordered_map<uint64_t, int64_t>& oid_to_global)
{
    std::vector<torch::Tensor> result;
    const size_t num_layers = sample.edges_per_layer.size();
    result.reserve(num_layers);

    for (size_t k = 0; k < num_layers; ++k) {
        const LayerEdges& edges = sample.edges_per_layer[k];
        const int64_t E = static_cast<int64_t>(edges.size());

        auto edge_index = torch::empty({2, E}, torch::kInt64);
        auto acc = edge_index.accessor<int64_t, 2>();

        for (int64_t i = 0; i < E; ++i) {
            // src_indices[i] indexes into nodes_per_layer[k+1]
            const ObjectId src_oid = sample.nodes_per_layer[k + 1][
                static_cast<size_t>(edges.src_indices[i])
            ];
            // dst_indices[i] indexes into nodes_per_layer[k]
            const ObjectId dst_oid = sample.nodes_per_layer[k][
                static_cast<size_t>(edges.dst_indices[i])
            ];

            auto src_it = oid_to_global.find(src_oid.id);
            if (src_it == oid_to_global.end()) {
                throw std::runtime_error(
                    "BatchAssembler: src node not in all_unique_nodes at layer " +
                    std::to_string(k) + ", edge " + std::to_string(i)
                );
            }
            auto dst_it = oid_to_global.find(dst_oid.id);
            if (dst_it == oid_to_global.end()) {
                throw std::runtime_error(
                    "BatchAssembler: dst node not in all_unique_nodes at layer " +
                    std::to_string(k) + ", edge " + std::to_string(i)
                );
            }

            acc[0][i] = src_it->second;  // row 0 = message source
            acc[1][i] = dst_it->second;  // row 1 = message destination
        }

        result.push_back(std::move(edge_index));
    }

    return result;
}

// =============================================================================
// Private: load_features
// =============================================================================

torch::Tensor BatchAssembler::load_features(
    const std::vector<ObjectId>& unique_nodes,
    uint64_t batch_id)
{
    if (feature_store_) {
        // Full mode: FourLevelStore handles all four tiers
        return feature_store_->load_batch_features(batch_id);
    }

    // Fallback mode: FeatureMatrix + RowMapping
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
