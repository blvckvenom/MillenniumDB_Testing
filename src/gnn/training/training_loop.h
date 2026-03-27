#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "gnn/models/graphsage_model.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/training/batch_assembler.h"

namespace mdb::gnn {

/**
 * @brief Orchestrates GNN training: epoch iteration, forward/backward,
 *        validation, and early stopping.
 *
 * Consumes MiniBatches from a BatchAssembler and drives a GraphSAGEModel
 * through supervised cross-entropy training.
 *
 * ## Early-stopping policy
 *
 * Two distinct termination conditions are tracked:
 *   1. **Convergence** — two consecutive epochs differ in average loss by
 *      less than `config.tolerance`.  Sets `Result::converged = true`.
 *   2. **Patience** — validation accuracy fails to improve for
 *      `config.patience` consecutive epochs.  Sets `Result::converged = false`
 *      and `Result::ran_epochs` reflects how many epochs actually ran.
 *
 * The best model checkpoint is written to
 * `config.output_dir + "/checkpoint.pt"` whenever validation accuracy
 * improves (only if `output_dir` is non-empty).
 *
 * ## Batch layout contract (SampleCatalog)
 *
 * - Training batches:   batch_id in [0,               train_batches)
 * - Validation batches: batch_id in [train_batches,   train_batches + validation_batches)
 * - Test batches:       batch_id in [train_batches + validation_batches, total_batches)
 */
class TrainingLoop {
public:
    // =========================================================================
    // Configuration
    // =========================================================================

    struct Config {
        uint64_t    epochs        = 50;
        double      learning_rate = 0.01;
        double      weight_decay  = 0.0;
        double      tolerance     = 1e-4;
        uint64_t    patience      = 5;
        int64_t     random_seed   = -1;    ///< -1 = non-deterministic
        std::string output_dir;            ///< directory for checkpoint.pt
    };

    // =========================================================================
    // Result
    // =========================================================================

    struct Result {
        uint64_t             ran_epochs       = 0;
        bool                 converged        = false;
        double               best_val_accuracy= 0.0;
        std::vector<double>  epoch_losses;
        double               train_seconds    = 0.0;
    };

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @param model     GraphSAGE model (must outlive TrainingLoop)
     * @param assembler BatchAssembler (must outlive TrainingLoop)
     * @param catalog   SampleCatalog describing train/val/test split counts
     * @param config    Hyper-parameters and stopping criteria
     */
    TrainingLoop(
        GraphSAGEModel&      model,
        BatchAssembler&      assembler,
        const SampleCatalog& catalog,
        Config               config
    );

    // =========================================================================
    // Training
    // =========================================================================

    /**
     * @brief Run the full training loop.
     *
     * Iterates up to config.epochs epochs.  Each epoch:
     *   1. Forward/backward over all training batches (cross-entropy loss).
     *   2. Accuracy evaluation on validation batches.
     *   3. Early-stopping check (patience + convergence).
     *
     * @return Summary of the training run.
     */
    Result train();

    // =========================================================================
    // Evaluation
    // =========================================================================

    /**
     * @brief Evaluate classification accuracy over a contiguous range of batches.
     *
     * Runs under torch::NoGradGuard and temporarily sets the model to eval mode.
     * Restores training mode before returning.
     *
     * @param start_batch First batch id to evaluate.
     * @param count       Number of consecutive batches to evaluate.
     * @return Fraction of correctly classified labeled seed nodes in [0.0, 1.0].
     *         Returns 0.0 if no labeled seeds exist across all batches.
     */
    double evaluate(uint64_t start_batch, uint64_t count);

private:
    GraphSAGEModel&      model_;
    BatchAssembler&      assembler_;
    const SampleCatalog& catalog_;
    Config               config_;
};

} // namespace mdb::gnn
