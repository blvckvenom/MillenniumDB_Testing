#include "gnn/training/training_loop.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace mdb::gnn {

// =============================================================================
// Construction
// =============================================================================

TrainingLoop::TrainingLoop(
    GraphSAGEModel&      model,
    BatchAssembler&      assembler,
    const SampleCatalog& catalog,
    Config               config
)
    : model_(model)
    , assembler_(assembler)
    , catalog_(catalog)
    , config_(std::move(config))
{
}

// =============================================================================
// Training
// =============================================================================

TrainingLoop::Result TrainingLoop::train()
{
    Result result;

    // --- Reproducibility -------------------------------------------------
    if (config_.random_seed >= 0) {
        torch::manual_seed(static_cast<uint64_t>(config_.random_seed));
    }

    // --- Optimizer -------------------------------------------------------
    auto optimizer = torch::optim::Adam(
        model_.parameters(),
        torch::optim::AdamOptions(config_.learning_rate)
            .weight_decay(config_.weight_decay)
    );

    double   best_val_acc     = 0.0;
    uint64_t patience_counter = 0;

    const uint64_t train_batches = catalog_.train_batches;
    const uint64_t val_batches   = catalog_.validation_batches;

    auto wall_start = std::chrono::steady_clock::now();

    for (uint64_t epoch = 0; epoch < config_.epochs; ++epoch) {

        // === Training phase ===
        model_.train();

        double   total_loss       = 0.0;
        uint64_t num_train_batches = 0;

        for (uint64_t bid = 0; bid < train_batches; ++bid) {
            MiniBatch mini = assembler_.assemble(bid);

            optimizer.zero_grad();

            auto logits = model_.forward(
                mini.features,
                mini.edge_indices,
                static_cast<int64_t>(mini.num_seeds)
            );

            // Only back-prop if at least one labeled seed exists in this batch
            if (mini.label_mask.any().item<bool>()) {
                auto masked_logits = logits.index({mini.label_mask});
                auto masked_labels = mini.labels.index({mini.label_mask});

                auto loss = torch::nn::functional::cross_entropy(
                    masked_logits, masked_labels
                );

                loss.backward();
                optimizer.step();

                total_loss += loss.item<double>();
            }

            ++num_train_batches;
        }

        double avg_loss = (num_train_batches > 0)
            ? (total_loss / static_cast<double>(num_train_batches))
            : 0.0;
        result.epoch_losses.push_back(avg_loss);

        // === Validation phase ===
        double val_accuracy = evaluate(train_batches, val_batches);

        // === Checkpoint if improved ===
        if (val_accuracy > best_val_acc) {
            best_val_acc = val_accuracy;
            patience_counter = 0;
            if (!config_.output_dir.empty()) {
                torch::serialize::OutputArchive archive;
                model_.save(archive);
                archive.save_to(config_.output_dir + "/checkpoint.pt");
            }
        } else {
            ++patience_counter;
            if (patience_counter >= config_.patience) {
                // Stopped by patience — not a convergence stop
                result.converged = false;
                ++epoch;  // account for this epoch before break
                result.ran_epochs = epoch;
                result.best_val_accuracy = best_val_acc;

                auto wall_end = std::chrono::steady_clock::now();
                result.train_seconds = std::chrono::duration<double>(
                    wall_end - wall_start).count();
                return result;
            }
        }

        // === Convergence check ===
        const size_t n = result.epoch_losses.size();
        if (n >= 2) {
            double delta = std::abs(
                result.epoch_losses[n - 1] - result.epoch_losses[n - 2]
            );
            if (delta < config_.tolerance) {
                result.converged  = true;
                result.ran_epochs = epoch + 1;
                result.best_val_accuracy = best_val_acc;

                auto wall_end = std::chrono::steady_clock::now();
                result.train_seconds = std::chrono::duration<double>(
                    wall_end - wall_start).count();
                return result;
            }
        }
    }

    // Completed all epochs without early stopping
    result.ran_epochs        = config_.epochs;
    result.best_val_accuracy = best_val_acc;

    auto wall_end = std::chrono::steady_clock::now();
    result.train_seconds = std::chrono::duration<double>(
        wall_end - wall_start).count();
    return result;
}

// =============================================================================
// Evaluation
// =============================================================================

double TrainingLoop::evaluate(uint64_t start_batch, uint64_t count)
{
    if (count == 0) {
        return 0.0;
    }

    torch::NoGradGuard no_grad;
    model_.eval();

    int64_t correct = 0;
    int64_t total   = 0;

    for (uint64_t i = 0; i < count; ++i) {
        auto mini = assembler_.assemble(start_batch + i);
        auto logits = model_.forward(
            mini.features,
            mini.edge_indices,
            static_cast<int64_t>(mini.num_seeds)
        );

        if (mini.label_mask.any().item<bool>()) {
            auto masked_logits = logits.index({mini.label_mask});
            auto masked_labels = mini.labels.index({mini.label_mask});

            auto predicted = masked_logits.argmax(1);
            correct += (predicted == masked_labels).sum().item<int64_t>();
            total   += masked_labels.size(0);
        }
    }

    // Restore training mode for subsequent use
    model_.train();

    return (total > 0) ? (static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
}

} // namespace mdb::gnn
