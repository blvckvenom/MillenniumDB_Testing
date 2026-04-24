#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "gnn/models/graphsage_model.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/batch_assembler.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

/**
 * @brief Collects seed embeddings from all training batches, infers
 *        embeddings for non-seed nodes, and (in future phases) writes
 *        them to the projection's tensor store.
 *
 * ## Overview
 *
 * After training, every node that appeared as a *seed* (layer 0) in any
 * batch already has a hidden representation that the model computed during
 * the forward pass.  Rather than re-running inference on the full graph,
 * we iterate over the pre-computed batches and collect those seed
 * embeddings, mapping each one back to its row_index in the RowMapping.
 *
 * Nodes that were never seeds still need embeddings.  Phase B performs
 * on-the-fly k-hop sampling via TopologyAccessor, builds a GraphSample,
 * assembles a MiniBatch, and runs a model forward pass to produce those
 * embeddings.
 *
 * ## Seed ObjectId recovery
 *
 * MiniBatch does not carry ObjectIds (it contains only tensors).  The
 * BatchAssembler's internal SampleStorage reference is private and not
 * exposed.  Therefore EmbeddingWriter receives its own SampleStorage
 * reference and reads GraphSamples directly to recover seed ObjectIds
 * from `nodes_per_layer[0]`.
 *
 * ## Phases (incremental implementation)
 *
 *   - Phase A: collect_seed_embeddings() iterates ALL batches,
 *     runs model.get_embeddings(), and maps results to (row_index, tensor).
 *   - Phase B: infer_non_seed_embeddings() for nodes never seen
 *     as seeds, using on-the-fly k-hop sampling from the projection topology.
 *   - Phase C (Task 6): write_to_projection() persists embeddings as a
 *     new tensor property in the GQL projection.
 *
 * ## Thread safety
 *
 * Not thread-safe.  Intended for single-threaded post-training use.
 */
class EmbeddingWriter {
public:
    // =========================================================================
    // Configuration
    // =========================================================================

    struct Config {
        std::string              property_name;     ///< Target property for embeddings
        uint64_t                 batch_size = 2048; ///< Inference batch size (Phase B)
        std::vector<uint64_t>    fanouts;           ///< Sampling fanouts   (Phase B)
        EdgeOrientation          orientation = EdgeOrientation::UNDIRECTED;
        std::filesystem::path    feature_matrix_path; ///< Path to .fmat for inference feature loading
    };

    // =========================================================================
    // Result
    // =========================================================================

    struct Result {
        uint64_t nodes_written   = 0;   ///< Nodes whose embeddings were collected/written
        uint64_t nodes_inferred  = 0;   ///< Non-seed nodes inferred in Phase B
        double   inference_ms    = 0.0; ///< Wall-clock time for inference (Phase B)
        double   write_ms        = 0.0; ///< Wall-clock time for writes (Phase C)
    };

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Construct an EmbeddingWriter.
     *
     * All reference parameters must outlive this object (non-owning).
     *
     * @param model              Trained GraphSAGE model (set to eval internally)
     * @param assembler          BatchAssembler for producing MiniBatches
     * @param sample_storage     SampleStorage for reading GraphSamples (seed ObjectIds)
     * @param row_mapping        ObjectId <-> row_index mapping
     * @param catalog            Batch counts per split (train/val/test)
     * @param projection_storage ProjectionStorage for topology access (Phase B inference)
     * @param config             Writer configuration
     */
    EmbeddingWriter(
        GraphSAGEModel&            model,
        BatchAssembler&            assembler,
        SampleStorage&             sample_storage,
        const RowMapping&          row_mapping,
        const SampleCatalog&       catalog,
        GQL::ProjectionStorage&    projection_storage,
        Config                     config
    );

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * @brief Run the full embedding write pipeline.
     *
     * Phase A: Collect seed embeddings from pre-computed batches.
     * Phase B: Infer embeddings for non-seed nodes via on-the-fly k-hop sampling.
     * Phase C (Task 6 stub): Write to projection tensor store.
     *
     * @return Summary with counts and timing.
     */
    Result write_all();

private:
    // =========================================================================
    // Phase A: Seed embedding collection
    // =========================================================================

    /**
     * @brief Iterate all batches, run model.get_embeddings(), and map seed
     *        embeddings back to RowMapping indices.
     *
     * For each batch_id in [0, catalog_.total_batches):
     *   1. Assemble the MiniBatch via BatchAssembler.
     *   2. Read the GraphSample via SampleStorage to recover seed ObjectIds.
     *   3. Run model.get_embeddings() on the MiniBatch.
     *   4. For each seed, look up its row_index via RowMapping::find().
     *   5. Store (row_index, embedding_tensor) in the result vector.
     *
     * Seeds whose ObjectId is not found in the RowMapping are silently
     * skipped (defensive: should never happen with consistent data).
     *
     * @return Vector of (row_index, embedding[1, hidden_dim]) pairs.
     */
    std::vector<std::pair<uint64_t, torch::Tensor>> collect_seed_embeddings();

    // =========================================================================
    // Phase B: Non-seed inference
    // =========================================================================

    /**
     * @brief Infer embeddings for nodes that were never seeds in any batch.
     *
     * For each chunk of config_.batch_size missing nodes:
     *   1. Build a GraphSample by k-hop sampling from the projection topology.
     *   2. Assemble a MiniBatch via assembler_.assemble_from_sample().
     *   3. Forward pass through model_.get_embeddings().
     *   4. Map each seed embedding back to its row_index.
     *
     * Uses TopologyAccessor for neighbor lookups, respecting config_.orientation
     * and config_.fanouts for the sampling parameters.
     *
     * @param missing  Row indices of nodes without embeddings from Phase A.
     * @return Vector of (row_index, embedding[hidden_dim]) pairs.
     */
    std::vector<std::pair<uint64_t, torch::Tensor>>
        infer_non_seed_embeddings(const std::vector<uint64_t>& missing);

    /**
     * @brief Build a GraphSample for a batch of seed nodes via k-hop sampling.
     *
     * Replicates the BasicKHopSampler algorithm (per-node uniform sampling)
     * using TopologyAccessor::get_neighbors() for neighbor lookups.
     *
     * @param seeds       Seed ObjectIds (become nodes_per_layer[0])
     * @param batch_id    Batch identifier for the GraphSample
     * @return GraphSample ready for BatchAssembler::assemble_from_sample()
     */
    GraphSample build_graph_sample(
        const std::vector<ObjectId>& seeds,
        uint64_t batch_id
    );

    // =========================================================================
    // Phase C: Write embeddings to projection as tensor properties
    // =========================================================================

    /**
     * @brief Persist all embeddings as tensor node properties in the projection.
     *
     * For each (row_index, embedding) pair:
     *   1. Look up the node ObjectId via RowMapping.
     *   2. Serialize the embedding to raw float bytes.
     *   3. Store in TensorManager via get_or_create_id().
     *   4. Build a tensor ObjectId (MASK_TENSOR_FLOAT_EXTERN | tensor_id).
     *   5. Insert (node_oid, key_oid, tensor_oid) into the B+Tree property
     *      indexes via BPlusTree::insert().
     *
     * The property key is registered in the projection catalog under
     * config_.property_name (e.g., "embedding"). If the projection was not
     * built with node property indexes, they are created as empty B+Trees
     * first.
     *
     * @param emb_map  Map of row_index -> embedding tensor.
     * @return Number of embeddings successfully written.
     */
    uint64_t write_to_projection(
        const std::unordered_map<uint64_t, torch::Tensor>& emb_map);

    // =========================================================================
    // Members (non-owning references)
    // =========================================================================

    GraphSAGEModel&            model_;
    BatchAssembler&            assembler_;
    SampleStorage&             sample_storage_;
    const RowMapping&          row_mapping_;
    const SampleCatalog&       catalog_;
    Config                     config_;

    /// ProjectionStorage for Phase C writes (non-owning reference).
    GQL::ProjectionStorage&    projection_storage_;

    /// TopologyAccessor for on-the-fly k-hop sampling (Phase B).
    /// Owned by this object; constructed from projection_storage in ctor.
    TopologyAccessor           topology_;

    /// RNG for uniform neighbor sampling (Phase B).
    std::mt19937_64            rng_;
};

} // namespace mdb::gnn
