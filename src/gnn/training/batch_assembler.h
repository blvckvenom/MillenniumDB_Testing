#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "graph_models/object_id.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/four_level_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/label_store.h"
#include "gnn/training/mini_batch.h"
#include "gnn/training/split_store.h"

namespace mdb::gnn {

/**
 * @brief Assembles MiniBatch objects from four data sources for the training loop.
 *
 * Supports two feature-loading modes:
 *   - Full mode: FourLevelStore (L1 GPU, L2 CPU, L3 mmap, L4 disk)
 *   - Fallback mode: FeatureMatrix directly via RowMapping
 *
 * Labels and splits are optional (nullptr = unsupervised / no-split).
 *
 * Usage:
 * @code
 *   BatchAssembler assembler(feature_matrix, samples, &labels, &splits, row_mapping);
 *   MiniBatch batch = assembler.assemble(batch_id);
 *   // or equivalently:
 *   GraphSample sample = samples.read_sample(batch_id);
 *   MiniBatch batch = assembler.assemble_from_sample(sample);
 * @endcode
 */
class BatchAssembler {
public:
    /**
     * @brief Full mode: FourLevelStore for features.
     *
     * @param feature_store FourLevelStore that owns all four tiers
     * @param samples       Pre-computed GraphSamples storage
     * @param labels        Optional label store (nullptr = unsupervised)
     * @param splits        Optional split store (nullptr = no split)
     * @param row_mapping   ObjectId <-> row_index mapping (used for label lookup)
     */
    BatchAssembler(
        FourLevelStore& feature_store,
        SampleStorage&  samples,
        LabelStore*     labels,
        SplitStore*     splits,
        const RowMapping& row_mapping
    );

    /**
     * @brief Fallback mode: FeatureMatrix directly (no FourLevelStore).
     *
     * @param feature_matrix Flat [N, D] feature matrix opened from disk
     * @param samples        Pre-computed GraphSamples storage
     * @param labels         Optional label store (nullptr = unsupervised)
     * @param splits         Optional split store (nullptr = no split)
     * @param row_mapping    ObjectId <-> row_index mapping
     */
    BatchAssembler(
        const FeatureMatrix& feature_matrix,
        SampleStorage&       samples,
        LabelStore*          labels,
        SplitStore*          splits,
        const RowMapping&    row_mapping
    );

    /**
     * @brief Assemble a MiniBatch from a stored sample identified by batch_id.
     *
     * Reads the GraphSample from storage, then delegates to assemble_from_sample().
     *
     * @param batch_id  Batch identifier to read
     * @return MiniBatch ready for the training loop
     */
    MiniBatch assemble(uint64_t batch_id);

    /**
     * @brief Assemble a MiniBatch directly from a GraphSample.
     *
     * Public method (rather than private) to facilitate testing without
     * requiring a fully-initialized SampleStorage on disk.
     *
     * @param sample Pre-populated GraphSample
     * @return MiniBatch ready for the training loop
     */
    MiniBatch assemble_from_sample(const GraphSample& sample);

private:
    /**
     * @brief Remap layer-local indices to global subgraph indices per GNN layer.
     *
     * GraphSample stores src/dst as local indices (into nodes_per_layer[k+1]
     * and nodes_per_layer[k] respectively). This converts them to global indices
     * into all_unique_nodes.
     *
     * @param sample         Source GraphSample
     * @param oid_to_global  Map from ObjectId.id to global index in all_unique_nodes
     * @return One [2, E_k] int64 tensor per layer
     */
    std::vector<torch::Tensor> build_edge_indices(
        const GraphSample& sample,
        const std::unordered_map<uint64_t, int64_t>& oid_to_global
    );

    /**
     * @brief Load features for a set of unique nodes.
     *
     * Routes through FourLevelStore (if in full mode) or FeatureMatrix fallback.
     *
     * @param unique_nodes  Ordered list of ObjectIds (all_unique_nodes)
     * @param batch_id      Batch identifier (used only by FourLevelStore path)
     * @return [N, D] float32 tensor
     */
    torch::Tensor load_features(
        const std::vector<ObjectId>& unique_nodes,
        uint64_t batch_id
    );

    // Feature source — exactly one of the two is set.
    FourLevelStore*      feature_store_   = nullptr;  // full mode
    const FeatureMatrix* feature_matrix_  = nullptr;  // fallback mode

    SampleStorage&    samples_;
    LabelStore*       labels_;    // nullable
    SplitStore*       splits_;    // nullable
    const RowMapping& row_mapping_;
};

} // namespace mdb::gnn
