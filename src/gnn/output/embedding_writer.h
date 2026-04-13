#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "gnn/models/graphsage_model.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/batch_assembler.h"

namespace mdb::gnn {

/**
 * @brief Collects seed embeddings from all training batches and (in future
 *        phases) infers embeddings for non-seed nodes and writes them to the
 *        projection's tensor store.
 *
 * ## Overview
 *
 * After training, every node that appeared as a *seed* (layer 0) in any
 * batch already has a hidden representation that the model computed during
 * the forward pass.  Rather than re-running inference on the full graph,
 * we iterate over the pre-computed batches and collect those seed
 * embeddings, mapping each one back to its row_index in the RowMapping.
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
 *   - Phase A (this task): collect_seed_embeddings() iterates ALL batches,
 *     runs model.get_embeddings(), and maps results to (row_index, tensor).
 *   - Phase B (Task 5): infer_non_seed_embeddings() for nodes never seen
 *     as seeds.
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
        uint64_t                 batch_size = 256;  ///< Inference batch size (Phase B)
        std::vector<uint64_t>    fanouts;           ///< Sampling fanouts   (Phase B)
    };

    // =========================================================================
    // Result
    // =========================================================================

    struct Result {
        uint64_t nodes_written   = 0;   ///< Nodes whose embeddings were collected/written
        uint64_t nodes_inferred  = 0;   ///< Non-seed nodes inferred in Phase B
        double   inference_ms    = 0.0; ///< Wall-clock time for inference
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
     * @param model          Trained GraphSAGE model (set to eval internally)
     * @param assembler      BatchAssembler for producing MiniBatches
     * @param sample_storage SampleStorage for reading GraphSamples (seed ObjectIds)
     * @param row_mapping    ObjectId <-> row_index mapping
     * @param catalog        Batch counts per split (train/val/test)
     * @param config         Writer configuration
     */
    EmbeddingWriter(
        GraphSAGEModel&         model,
        BatchAssembler&         assembler,
        SampleStorage&          sample_storage,
        const RowMapping&       row_mapping,
        const SampleCatalog&    catalog,
        Config                  config
    );

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * @brief Run the full embedding write pipeline.
     *
     * Currently only Phase A (seed collection).  Phases B and C will be
     * added in Tasks 5 and 6.
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

    // Phase B (Task 5):
    // std::vector<std::pair<uint64_t, torch::Tensor>>
    //     infer_non_seed_embeddings(const std::vector<std::pair<uint64_t, torch::Tensor>>& seeds);

    // Phase C (Task 6):
    // uint64_t write_to_projection(const std::vector<std::pair<uint64_t, torch::Tensor>>& embeddings);

    // =========================================================================
    // Members (non-owning references)
    // =========================================================================

    GraphSAGEModel&      model_;
    BatchAssembler&      assembler_;
    SampleStorage&       sample_storage_;
    const RowMapping&    row_mapping_;
    const SampleCatalog& catalog_;
    Config               config_;
};

} // namespace mdb::gnn
